"""
Evaluate a trained checkpoint against the external "birds_and_mammals"
held-out set (see build_external_manifest.py) -- a real, independently
sourced distribution shift (different reserves, cameras, and time period
than the LILA NZ trail-cam data everything was trained on), used to
measure how far real-world performance drops on data the model has
genuinely never seen in any form. This is the intended drift-detection
signal for the MLOps retraining trigger, not a training source itself.

Deliberately evaluates the DEPLOYED model as-is: the served checkpoint
(configs/inference/serving.yaml, currently the plain baseline) at its
live decision threshold (configs/inference/threshold_config.yaml),
full-frame input, no MegaDetector/box-crop involved -- this dataset has
no MegaDetector boxes at all, and the question this script answers is
"how does what's actually running in production do on new data", not a
box-crop-specific comparison. If box-crop ever becomes the served model,
point --checkpoint-path at it and re-run; the crop-to-box question can
be revisited then.

Reports metrics two ways, since burst-event reconstruction
(build_external_manifest.py) found 1,865 usable frames collapse to only
444 real encounters (mean 4.2 frames/event, one event as large as 113):
  - all frames, as-is
  - one frame per reconstructed event (first by timestamp) -- so no
    single heavily-photographed encounter can dominate the headline
    number the way it could in the frame-level view

Run: python -m src.external_data.evaluate_external [architecture] [tag] [--checkpoint-path PATH] [--split SPLIT]
     (--split restricts evaluation to one value of the manifest's split
     column, e.g. "test" -- REQUIRED when evaluating a checkpoint that
     was retrained on data/manifest_combined.csv, see
     src/external_data/combine_manifests.py and
     src/monitoring/drift.py's generate_retrain_runbook, or the reported
     numbers would partly measure memorisation of the retrain's own
     training images. Omitted/None evaluates every row -- correct for
     the original drift-detection use of this script, evaluating a
     model that never trained on any of this data.)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from src.models.classifier import load_checkpoint
from src.training.dataset import DEFAULT_TRANSFORM
from src.training.evaluate import DEFAULT_THRESHOLD_CONFIG, _load_yaml, _resolve_torch_device

DEFAULT_MANIFEST_PATH = "src/external_data/metadata/external_manifest.csv"
DEFAULT_IMAGES_ROOT = "src/external_data/birds_and_mammals"
DEFAULT_ARCHITECTURE = "efficientnet_b3"


class PlainImageDataset(Dataset):
    """Reads images straight off disk by manifest['file_name'] joined
    against `images_root` -- no tar shards, no ShardReader indexing pass,
    appropriate for a dataset this size (~1,900 images) that was
    extracted directly rather than shipped as shards."""

    def __init__(self, manifest: pd.DataFrame, images_root: str, transform=DEFAULT_TRANSFORM) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.images_root = Path(images_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        image = Image.open(self.images_root / row["file_name"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(float(row["label"]))
        return image, label


def _compute_metrics(y_true, y_pred, y_prob) -> dict:
    metrics = {
        "n_images": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_per_class": precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "recall_per_class": recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    # ROC-AUC/PR-AUC need both classes present -- a per-event or per-species
    # slice can end up single-class, so these are opportunistic, not assumed.
    if len(set(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    return metrics


def run_external_evaluation(
    architecture: str = DEFAULT_ARCHITECTURE,
    tag: str | None = None,
    checkpoint_path: str | None = None,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    images_root: str = DEFAULT_IMAGES_ROOT,
    threshold: float | None = None,
    batch_size: int = 32,
    split_filter: str | None = None,
) -> dict:
    """`split_filter` restricts evaluation to one value of the manifest's
    `split` column (e.g. "test") -- None (the default) evaluates every
    row, which is what the original drift-detection evaluation of the
    deployed model needed (it isn't the model that trained on any of this
    data, so there's no leakage to guard against). Evaluating a
    checkpoint that WAS retrained on combined data (including part of
    this external set -- see combine_manifests.py) must pass
    split_filter="test", or the reported numbers would partly measure
    memorisation of its own training images, not real generalisation.
    """
    tag = tag or architecture
    checkpoint_path = checkpoint_path or f"models/checkpoint_{tag}.pt"
    out_dir = Path("metrics/external_birds_and_mammals") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if threshold is None:
        threshold = _load_yaml(DEFAULT_THRESHOLD_CONFIG)["thresholds"]["default"]

    manifest = pd.read_csv(manifest_path)
    if split_filter is not None:
        if "split" not in manifest.columns:
            raise ValueError(f"split_filter={split_filter!r} given but {manifest_path} has no 'split' column")
        manifest = manifest[manifest["split"] == split_filter].reset_index(drop=True)
        if len(manifest) == 0:
            raise ValueError(f"split_filter={split_filter!r} matched 0 rows in {manifest_path}")
    dataset = PlainImageDataset(manifest, images_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = _resolve_torch_device()
    model = load_checkpoint(checkpoint_path, architecture=architecture, device=device)

    probabilities: list[float] = []
    with torch.no_grad():
        for x, _ in loader:
            probabilities.extend(torch.sigmoid(model(x.to(device))).cpu().tolist())

    predictions = manifest.reset_index(drop=True).copy()
    predictions["probability"] = probabilities
    predictions["confidence"] = predictions["probability"].apply(lambda p: max(p, 1 - p))
    predictions["pred_label"] = (predictions["probability"] >= threshold).astype(int)
    predictions_path = Path("data/predictions") / f"external_predictions_{tag}.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)

    frame_metrics = _compute_metrics(
        predictions["label"].to_numpy(), predictions["pred_label"].to_numpy(), predictions["probability"].to_numpy()
    )

    # One frame per reconstructed event, first by timestamp -- deterministic,
    # avoids a single heavily-photographed encounter dominating the result.
    event_level = predictions.sort_values("datetime").groupby("event_id", as_index=False).first()
    event_metrics = _compute_metrics(
        event_level["label"].to_numpy(), event_level["pred_label"].to_numpy(), event_level["probability"].to_numpy()
    )

    result = {
        "tag": tag,
        "architecture": architecture,
        "checkpoint_path": str(checkpoint_path),
        "threshold": threshold,
        "split_filter": split_filter,
        "n_frames": len(predictions),
        "n_events": int(predictions["event_id"].nunique()),
        "frame_level": frame_metrics,
        "event_level": event_metrics,
    }
    (out_dir / "eval.json").write_text(json.dumps(result, indent=2))

    print(f"[external, {tag}] threshold={threshold}  split_filter={split_filter}")
    print(
        f"  frame-level (n={frame_metrics['n_images']}):  accuracy={frame_metrics['accuracy']:.4f}  "
        f"precision={frame_metrics['precision']:.4f}  recall={frame_metrics['recall']:.4f}  f1={frame_metrics['f1']:.4f}"
    )
    print(f"    confusion matrix (rows=true, cols=pred) [bird, mammal]:\n{frame_metrics['confusion_matrix']}")
    print(
        f"  event-level (n={event_metrics['n_images']}, one frame/event):  accuracy={event_metrics['accuracy']:.4f}  "
        f"precision={event_metrics['precision']:.4f}  recall={event_metrics['recall']:.4f}  f1={event_metrics['f1']:.4f}"
    )
    print(f"    confusion matrix (rows=true, cols=pred) [bird, mammal]:\n{event_metrics['confusion_matrix']}")
    print(f"wrote {out_dir}/eval.json, {predictions_path}")
    return result


def main() -> None:
    import sys

    argv = sys.argv[1:]
    checkpoint_path = None
    split_filter = None
    for flag, is_checkpoint in (("--checkpoint-path", True), ("--split", False)):
        if flag in argv:
            i = argv.index(flag)
            value = argv[i + 1]
            argv = argv[:i] + argv[i + 2 :]
            if is_checkpoint:
                checkpoint_path = value
            else:
                split_filter = value

    architecture = argv[0] if len(argv) > 0 else DEFAULT_ARCHITECTURE
    tag = argv[1] if len(argv) > 1 else architecture
    run_external_evaluation(
        architecture=architecture, tag=tag, checkpoint_path=checkpoint_path, split_filter=split_filter
    )


if __name__ == "__main__":
    main()
