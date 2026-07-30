"""
PyTorch Lightning training loop for the wildlife classifier.

This is a first cut — a real training loop wired end-to-end against
whatever shards src/data/ingest.py has finished so far (data/shards/*.done),
proving the pipeline connects: manifest -> shard images -> model -> MLflow.
It is deliberately scoped down from the full design:

  IMPLEMENTED here:
    - AdamW optimiser + CosineAnnealingLR scheduler
    - BCEWithLogitsLoss(pos_weight) computed from whatever train images
      are currently available, for basic class-imbalance correction
    - MLflow logging (params + per-epoch train/val loss), manual calls
      rather than Lightning's logger abstraction
    - mps/cuda/cpu auto-selected accelerator

  NOT YET implemented (follow-up work, not this pass):
    - WeightedRandomSampler and the full naive-vs-3-lever imbalance
      ablation described in the proposal — this pass only does
      pos_weight, on whatever slice of data is currently ingested
    - Optuna HPO sweeps
    - codecarbon EmissionsTracker
    - MLflow Model Registry versioning

Known simplification: ShardDataset resizes to an exact 300x300 square
(non-uniform stretch), since ingest.py stores images at native aspect
ratio with the longer side capped at 300px. A production preprocessing
step should crop/pad instead of stretching — fine for a wiring smoke
test, worth revisiting before reporting real accuracy numbers.

Entry point: `python -m src.training.train [model-config-path]`
             (reads configs/data/default.yaml, configs/training/default.yaml,
             and configs/model/efficientnet_b3.yaml by default -- pass a
             different model config path, e.g.
             configs/model/resnet50.yaml, to train that architecture
             instead. Not yet composed through Hydra's config groups,
             same pragmatic choice as src/data/build_manifest.py.
             Checkpoint is written to models/checkpoint_<architecture>.pt
             so runs for different architectures don't overwrite each
             other.)

HPC usage: ECS GPU servers are shared, cooperative-etiquette machines with
no job scheduler — run via `tmux`, pinned to a specific GPU, not submitted
to a queue.
"""
from __future__ import annotations

import os
from pathlib import Path

import mlflow
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src.models.classifier import WildlifeClassifier
from src.training.dataset import ShardDataset

DEFAULT_DATA_CONFIG = "configs/data/default.yaml"
DEFAULT_TRAINING_CONFIG = "configs/training/default.yaml"
DEFAULT_MODEL_CONFIG = "configs/model/efficientnet_b3.yaml"


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class WildlifeLightningModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        max_epochs: int,
        architecture: str = "efficientnet_b3",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = WildlifeClassifier(
            architecture=architecture, pretrained=pretrained, dropout_rate=dropout_rate
        )
        pw = torch.tensor(pos_weight) if pos_weight is not None else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        self._train_losses.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        self._val_losses.append(loss.item())
        return loss

    def on_train_epoch_end(self) -> None:
        if self._train_losses:
            mlflow.log_metric(
                "train_loss", sum(self._train_losses) / len(self._train_losses), step=self.current_epoch
            )
            self._train_losses.clear()

    def on_validation_epoch_end(self) -> None:
        if self._val_losses:
            mlflow.log_metric(
                "val_loss", sum(self._val_losses) / len(self._val_losses), step=self.current_epoch
            )
            self._val_losses.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def build_dataloaders(manifest_path: str, shards_dir: str, batch_size: int, num_workers: int = 0):
    manifest = pd.read_csv(manifest_path)
    train_ds = ShardDataset(manifest, shards_dir=shards_dir, split="train")
    val_ds = ShardDataset(manifest, shards_dir=shards_dir, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_ds, val_ds


def _resolve_accelerator() -> str:
    if torch.cuda.is_available():
        return "gpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_val_metrics(
    module: WildlifeLightningModule, val_loader: DataLoader, threshold: float = 0.5
) -> dict:
    """Precision/recall/F1/confusion matrix on a validation set -- accuracy
    alone doesn't tell you whether the model is actually catching mammals,
    same reasoning as src/training/baselines.py."""
    device = _resolve_torch_device()
    module.to(device)
    module.eval()

    all_true: list[int] = []
    all_pred: list[int] = []
    with torch.no_grad():
        for x, y in val_loader:
            probs = torch.sigmoid(module(x.to(device))).cpu()
            all_pred.extend((probs >= threshold).int().tolist())
            all_true.extend(y.int().tolist())

    return {
        "accuracy": accuracy_score(all_true, all_pred),
        "precision": precision_score(all_true, all_pred, zero_division=0),
        "recall": recall_score(all_true, all_pred, zero_division=0),
        "f1": f1_score(all_true, all_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(all_true, all_pred, labels=[0, 1]),
    }


def main(max_epochs: int | None = None, model_config_path: str = DEFAULT_MODEL_CONFIG) -> None:
    data_cfg = _load_yaml(DEFAULT_DATA_CONFIG)["data"]
    training_cfg = _load_yaml(DEFAULT_TRAINING_CONFIG)["training"]
    model_cfg = _load_yaml(model_config_path)["model"]
    architecture = model_cfg["architecture"]

    num_workers = int(os.environ.get("NUM_WORKERS", training_cfg.get("num_workers", 0)))

    train_loader, val_loader, train_ds, val_ds = build_dataloaders(
        manifest_path=data_cfg["manifest_path"],
        shards_dir=data_cfg["shards_dir"],
        batch_size=training_cfg["batch_size"],
        num_workers=num_workers,
    )

    print(f"images available in ingested shards so far: train={len(train_ds)}  val={len(val_ds)}  (num_workers={num_workers})")
    if len(train_ds) == 0:
        raise RuntimeError(
            f"No training images found in {data_cfg['shards_dir']} yet — has "
            "src.data.ingest produced any finished (.done) shards?"
        )

    train_labels = train_ds._rows["label"]
    n_pos = int((train_labels == 1).sum())
    n_neg = int((train_labels == 0).sum())
    pos_weight = (n_neg / n_pos) if n_pos else 1.0

    epochs = max_epochs or training_cfg["max_epochs"]

    module = WildlifeLightningModule(
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        max_epochs=epochs,
        architecture=architecture,
        pretrained=model_cfg["pretrained"],
        dropout_rate=model_cfg["dropout_rate"],
        pos_weight=pos_weight,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=_resolve_accelerator(),
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    with mlflow.start_run(run_name=f"{architecture}_full_run"):
        mlflow.log_params(
            {
                "architecture": architecture,
                "pretrained": model_cfg["pretrained"],
                "batch_size": training_cfg["batch_size"],
                "learning_rate": training_cfg["learning_rate"],
                "max_epochs": epochs,
                "train_images_available": len(train_ds),
                "val_images_available": len(val_ds),
                "pos_weight": pos_weight,
            }
        )
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        val_metrics = compute_val_metrics(module, val_loader)
        mlflow.log_metric("val_accuracy_final", val_metrics["accuracy"])
        mlflow.log_metric("val_precision_final", val_metrics["precision"])
        mlflow.log_metric("val_recall_final", val_metrics["recall"])
        mlflow.log_metric("val_f1_final", val_metrics["f1"])
        print(
            f"[val, final] accuracy={val_metrics['accuracy']:.4f} "
            f"precision={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f} "
            f"f1={val_metrics['f1']:.4f}"
        )
        print(f"  confusion matrix (rows=true, cols=pred) [bird, mammal]:\n{val_metrics['confusion_matrix']}")

    checkpoint_path = Path(f"models/checkpoint_{architecture}.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.model.state_dict(), checkpoint_path)
    print(f"wrote {checkpoint_path}")


if __name__ == "__main__":
    import sys

    model_config = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_CONFIG
    main(model_config_path=model_config)
