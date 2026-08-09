"""
Seasonal generalisation experiment -- the headline contribution of this
thesis per the AIML 589 proposal: train on Spring/Summer imagery, test on
Autumn/Winter, and quantify the gap against the standard site-split
baseline (src/training/train.py + evaluate.py).

Split is by season, not site -- manifest.csv's site-based `split` column
is untouched; this reads the `season` column instead. Images with no
timestamp (season=None, ~23k of them) are excluded entirely, same as
noted when the manifest was built -- they can't be assigned to either
side of this split.

A small slice of the warm (train) season is held out as a validation set
purely to monitor training (watch for overfitting) -- the real reported
result is always the held-out cool (autumn/winter) season, evaluated
separately via:
    python -m src.training.evaluate <architecture> seasonal

Honest caveat, worth stating in the report rather than hiding: the
seasonal train set (~27.7k images) is smaller than the site-split train
set (~57.8k), since it's restricted to two of four seasons. Any
performance drop here isn't *purely* a seasonal effect -- some of it may
be attributable to less training data. The proposal's own design accepts
this tradeoff; it's just worth naming explicitly.

This script covers comparison 2 of the proposal's three ("standard
split baseline" / "seasonal split" / "post-retraining recovery").
Comparison 3 (retraining recovery) is follow-up work, worth doing only
if this comparison shows a meaningful gap to recover from.

Run: python -m src.training.seasonal_experiment [model-config-path] [--crop-to-box]
     (--crop-to-box crops every image to its MegaDetector box first, same
     as boxcrop_experiment.py, to test whether that fix for background
     reliance -- see boxcrop_experiment.py's docstring -- also closes the
     seen-site vs. novel-site generalisation gap)
"""
from __future__ import annotations

import os
from pathlib import Path

import mlflow
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.training.dataset import ShardDataset
from src.training.train import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    WildlifeLightningModule,
    _load_yaml,
    _resolve_accelerator,
    compute_val_metrics,
)

WARM_SEASONS = ("spring", "summer")
COOL_SEASONS = ("autumn", "winter")


def seasonal_split(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Warm (train) / cool (test) split by season. Rows with no season are
    excluded from both -- they can't be assigned to either side."""
    warm = manifest[manifest["season"].isin(WARM_SEASONS)].reset_index(drop=True)
    cool = manifest[manifest["season"].isin(COOL_SEASONS)].reset_index(drop=True)
    return warm, cool


def build_seasonal_dataloaders(
    manifest_path: str,
    shards_dir: str,
    batch_size: int,
    num_workers: int = 0,
    val_fraction: float = 0.15,
    seed: int = 42,
    crop_to_box: bool = False,
    box_crop_padding: float = 0.15,
):
    manifest = pd.read_csv(manifest_path)
    warm, cool = seasonal_split(manifest)

    # Small held-out slice of the warm season for monitoring only -- the
    # real reported result is the full cool season, evaluated separately.
    warm_shuffled = warm.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = int(len(warm_shuffled) * val_fraction)
    warm_val = warm_shuffled.iloc[:n_val].reset_index(drop=True)
    warm_train = warm_shuffled.iloc[n_val:].reset_index(drop=True)

    train_ds = ShardDataset(
        warm_train, shards_dir=shards_dir, split=None, crop_to_box=crop_to_box, box_crop_padding=box_crop_padding
    )
    val_ds = ShardDataset(
        warm_val, shards_dir=shards_dir, split=None, crop_to_box=crop_to_box, box_crop_padding=box_crop_padding
    )
    test_ds = ShardDataset(
        cool, shards_dir=shards_dir, split=None, crop_to_box=crop_to_box, box_crop_padding=box_crop_padding
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


def main(
    max_epochs: int | None = None,
    model_config_path: str = DEFAULT_MODEL_CONFIG,
    crop_to_box: bool = False,
) -> None:
    data_cfg = _load_yaml(DEFAULT_DATA_CONFIG)["data"]
    training_cfg = _load_yaml(DEFAULT_TRAINING_CONFIG)["training"]
    model_cfg = _load_yaml(model_config_path)["model"]
    architecture = model_cfg["architecture"]

    num_workers = int(os.environ.get("NUM_WORKERS", training_cfg.get("num_workers", 0)))
    box_crop_padding = data_cfg.get("box_crop_padding", 0.15)
    tag_suffix = "_boxcrop" if crop_to_box else ""

    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = build_seasonal_dataloaders(
        manifest_path=data_cfg["manifest_path"],
        shards_dir=data_cfg["shards_dir"],
        batch_size=training_cfg["batch_size"],
        num_workers=num_workers,
        crop_to_box=crop_to_box,
        box_crop_padding=box_crop_padding,
    )

    print(
        f"seasonal split -- train(warm)={len(train_ds)}  val(warm holdout)={len(val_ds)}  "
        f"test(cool)={len(test_ds)}  (num_workers={num_workers})"
    )
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(
            "No images available for the seasonal split -- has ingest finished, "
            "and does the manifest have a 'season' column?"
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

    with mlflow.start_run(run_name=f"{architecture}{tag_suffix}_seasonal_experiment"):
        mlflow.log_params(
            {
                "architecture": architecture,
                "experiment": "seasonal_generalisation",
                "crop_to_box": crop_to_box,
                "train_seasons": str(WARM_SEASONS),
                "test_seasons": str(COOL_SEASONS),
                "batch_size": training_cfg["batch_size"],
                "learning_rate": training_cfg["learning_rate"],
                "max_epochs": epochs,
                "train_images": len(train_ds),
                "val_images": len(val_ds),
                "test_images_cool_season": len(test_ds),
                "pos_weight": pos_weight,
            }
        )
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Quick reference number logged here; the full predictions CSV +
        # plots for the real reported result come from running
        # `python -m src.training.evaluate <architecture> seasonal`
        # against the checkpoint this writes.
        cool_metrics = compute_val_metrics(module, test_loader)
        mlflow.log_metric("cool_season_accuracy", cool_metrics["accuracy"])
        mlflow.log_metric("cool_season_precision", cool_metrics["precision"])
        mlflow.log_metric("cool_season_recall", cool_metrics["recall"])
        mlflow.log_metric("cool_season_f1", cool_metrics["f1"])
        print(
            f"[cool season, autumn/winter] accuracy={cool_metrics['accuracy']:.4f} "
            f"precision={cool_metrics['precision']:.4f} recall={cool_metrics['recall']:.4f} "
            f"f1={cool_metrics['f1']:.4f}"
        )
        print(f"  confusion matrix (rows=true, cols=pred) [bird, mammal]:\n{cool_metrics['confusion_matrix']}")

    checkpoint_path = Path(f"models/checkpoint_{architecture}_seasonal{tag_suffix}.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.model.state_dict(), checkpoint_path)
    print(f"wrote {checkpoint_path}")
    eval_mode = "seasonal_boxcrop" if crop_to_box else "seasonal"
    print(
        f"for the full predictions CSV + plots, run: "
        f"python -m src.training.evaluate {architecture} {eval_mode}"
    )


if __name__ == "__main__":
    import sys

    crop_to_box = "--crop-to-box" in sys.argv
    sys.argv = [a for a in sys.argv if a != "--crop-to-box"]
    model_config = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_CONFIG
    main(model_config_path=model_config, crop_to_box=crop_to_box)
