"""
Box-crop experiment -- does cropping training images to their MegaDetector
box (instead of feeding the whole frame) make the model actually learn to
recognise the animal, rather than the background?

Triggered by src/xai/gradcam.py's attention audit: across the full test
set, correctly-classified BIRDS get essentially zero Grad-CAM attention
on the bird itself half the time (median pointing-game score 0.0,
n=8,444), while correctly-classified MAMMALS average 0.67-0.73 -- a model
that mostly recognises "mammal" by looking at the mammal, but recognises
"bird" as a residual/default answer driven by background cues. See
metrics/efficientnet_b3/xai/attention_audit/summary.json for the numbers
this is responding to.

Same site-based train/val/test split as train.py (manifest.csv's `split`
column is untouched, and this reuses train.py's own build_dataloaders) --
the only difference from the main baseline is that every image is
cropped to its MegaDetector box (padded by configs/data/default.yaml's
box_crop_padding) before being resized to 300x300. Images without a
trustworthy box fall back to the full, uncropped frame -- see
src.training.dataset.crop_to_bbox.

Run: python -m src.training.boxcrop_experiment [model-config-path]
"""
from __future__ import annotations

import os
from pathlib import Path

import mlflow
import pytorch_lightning as pl
import torch

from src.training.train import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    WildlifeLightningModule,
    _load_yaml,
    _resolve_accelerator,
    build_dataloaders,
    compute_val_metrics,
)


def main(max_epochs: int | None = None, model_config_path: str = DEFAULT_MODEL_CONFIG) -> None:
    data_cfg = _load_yaml(DEFAULT_DATA_CONFIG)["data"]
    training_cfg = _load_yaml(DEFAULT_TRAINING_CONFIG)["training"]
    model_cfg = _load_yaml(model_config_path)["model"]
    architecture = model_cfg["architecture"]

    num_workers = int(os.environ.get("NUM_WORKERS", training_cfg.get("num_workers", 0)))
    box_crop_padding = data_cfg.get("box_crop_padding", 0.15)

    train_loader, val_loader, train_ds, val_ds = build_dataloaders(
        manifest_path=data_cfg["manifest_path"],
        shards_dir=data_cfg["shards_dir"],
        batch_size=training_cfg["batch_size"],
        num_workers=num_workers,
        crop_to_box=True,
        box_crop_padding=box_crop_padding,
    )

    print(
        f"box-crop experiment -- train={len(train_ds)}  val={len(val_ds)}  "
        f"(num_workers={num_workers}, box_crop_padding={box_crop_padding})"
    )
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

    with mlflow.start_run(run_name=f"{architecture}_boxcrop_experiment"):
        mlflow.log_params(
            {
                "architecture": architecture,
                "experiment": "box_crop",
                "box_crop_padding": box_crop_padding,
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

    checkpoint_path = Path(f"models/checkpoint_{architecture}_boxcrop.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.model.state_dict(), checkpoint_path)
    print(f"wrote {checkpoint_path}")
    print(
        f"for the full predictions CSV + plots, run: "
        f"python -m src.training.evaluate {architecture} boxcrop"
    )


if __name__ == "__main__":
    import sys

    model_config = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_CONFIG
    main(model_config_path=model_config)
