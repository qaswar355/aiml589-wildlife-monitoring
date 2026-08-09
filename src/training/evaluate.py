"""
Model evaluation: per-image predictions, aggregate metrics, and plots.

Loads a trained checkpoint and runs it over a test set (either the
manifest's site-based "test" split, or the seasonal experiment's
autumn/winter set -- see src/training/seasonal_experiment.py), producing,
namespaced by `tag` so multiple models/experiments' results sit side by
side rather than overwriting each other:
  - data/predictions/test_predictions_<tag>.csv -- one row per test
      image: image_id, site_id, species, season, true_label, pred_label,
      confidence, probability. DVC-tracked (see dvc.yaml), so it's
      reproducible and shareable without re-running training.
  - metrics/<tag>/eval.json -- precision/recall/F1/accuracy/ROC-AUC/PR-AUC
      (binary, i.e. mammal-class-only) PLUS macro- and micro-averaged
      precision/recall/F1, and confusion matrix in absolute counts.
      Accuracy and micro-average are intentionally not the headline
      numbers -- micro-average is mathematically identical to accuracy
      in binary classification, so it's reported for transparency, not
      as new information. Macro-average is the informative one for the
      imbalance story: it weights bird and mammal equally regardless of
      which has more test examples.
  - metrics/<tag>/roc_curve.csv, pr_curve.csv -- DVC-plottable curve
      points, plus rendered .png versions and a confusion matrix heatmap
      for direct use in the report

`probability` is the raw sigmoid output (P(mammal)); `confidence` is the
probability of whichever class was actually predicted (max(p, 1-p)) --
these differ whenever the model predicts bird (pred_label=0), which is
what lets a review queue flag "low confidence" regardless of which way
the model leaned.

Threshold sweep / per-zone thresholds (configs/inference/threshold_config.yaml)
is follow-up work, not this pass -- this evaluates a single trained
checkpoint at the default threshold.

Run: python -m src.training.evaluate [architecture] [test_split]
     (architecture defaults to efficientnet_b3; test_split defaults to
     "test" -- the site-based split. Pass "seasonal" to evaluate against
     the autumn/winter set instead, matching a checkpoint produced by
     seasonal_experiment.py; "boxcrop" for a checkpoint produced by
     boxcrop_experiment.py; "seasonal_boxcrop" for a checkpoint produced
     by seasonal_experiment.py --crop-to-box)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.models.classifier import load_checkpoint
from src.training.dataset import ShardDataset

DEFAULT_DATA_CONFIG = "configs/data/default.yaml"
DEFAULT_THRESHOLD_CONFIG = "configs/inference/threshold_config.yaml"
DEFAULT_ARCHITECTURE = "efficientnet_b3"
COOL_SEASONS = ("autumn", "winter")


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _save_plots(y_true, y_pred, fpr, tpr, roc_auc, prec, rec, pr_auc, out_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["bird", "mammal"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["bird", "mammal"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, label=f"PR AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=150)
    plt.close(fig)


def _site_familiarity_breakdown(predictions: pd.DataFrame, warm_sites: set[str]) -> dict:
    """Seasonal-experiment-only diagnostic: does performance differ between
    cool-season test images from sites also seen during warm-season training
    versus sites never seen at all? Distinguishes "the model generalises
    across seasons" from "the model has just seen this camera's background
    before" -- see the seasonal generalisation discussion in the report."""
    seen_mask = predictions["site_id"].isin(warm_sites)
    breakdown = {}
    for key, mask in (("seen_site", seen_mask), ("novel_site", ~seen_mask)):
        sub = predictions[mask]
        y_true, y_pred = sub["true_label"], sub["pred_label"]
        breakdown[key] = {
            "n": int(len(sub)),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
    return breakdown


def run_evaluation(
    architecture: str = DEFAULT_ARCHITECTURE,
    tag: str | None = None,
    test_split: str = "test",
    checkpoint_path: str | None = None,
    manifest_path: str | None = None,
    shards_dir: str | None = None,
    threshold: float | None = None,
    batch_size: int = 32,
    crop_to_box: bool = False,
) -> dict:
    tag = tag or architecture
    checkpoint_path = checkpoint_path or f"models/checkpoint_{tag}.pt"
    predictions_path = Path(f"data/predictions/test_predictions_{tag}.csv")
    metrics_dir = Path("metrics") / tag

    data_cfg = _load_yaml(DEFAULT_DATA_CONFIG)["data"]
    manifest_path = manifest_path or data_cfg["manifest_path"]
    shards_dir = shards_dir or data_cfg["shards_dir"]
    box_crop_padding = data_cfg.get("box_crop_padding", 0.15)

    if threshold is None:
        threshold = _load_yaml(DEFAULT_THRESHOLD_CONFIG)["thresholds"]["default"]

    manifest = pd.read_csv(manifest_path)
    if test_split == "seasonal":
        test_df = manifest[manifest["season"].isin(COOL_SEASONS)]
        test_ds = ShardDataset(
            test_df, shards_dir=shards_dir, split=None, crop_to_box=crop_to_box, box_crop_padding=box_crop_padding
        )
    else:
        test_ds = ShardDataset(
            manifest, shards_dir=shards_dir, split="test", crop_to_box=crop_to_box, box_crop_padding=box_crop_padding
        )
    if len(test_ds) == 0:
        raise RuntimeError(f"No test images found in {shards_dir} -- has ingest finished?")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = _resolve_torch_device()
    model = load_checkpoint(checkpoint_path, architecture=architecture, device=device)

    probabilities: list[float] = []
    with torch.no_grad():
        for x, _ in test_loader:
            probabilities.extend(torch.sigmoid(model(x.to(device))).cpu().tolist())

    rows = test_ds._rows.reset_index(drop=True).copy()
    rows["probability"] = probabilities
    rows["confidence"] = rows["probability"].apply(lambda p: max(p, 1 - p))
    rows["pred_label"] = (rows["probability"] >= threshold).astype(int)
    rows["true_label"] = rows["label"]
    rows["site_id"] = rows["location"]

    predictions = rows[
        ["image_id", "site_id", "species", "season", "true_label", "pred_label", "confidence", "probability"]
    ]
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)

    y_true = predictions["true_label"].to_numpy()
    y_pred = predictions["pred_label"].to_numpy()
    y_prob = predictions["probability"].to_numpy()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    # Binary (mammal-class-only) metrics -- the headline numbers used elsewhere.
    # Macro-average weights bird and mammal equally regardless of test-set size,
    # which is the informative complement to the imbalance story. Micro-average
    # is mathematically identical to accuracy in binary classification --
    # reported for transparency (per supervisor request), not as new signal.
    metrics = {
        "threshold": threshold,
        "n_test_images": len(predictions),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_per_class": precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "recall_per_class": recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

    if test_split == "seasonal":
        warm_sites = set(manifest[manifest["season"].isin(("spring", "summer"))]["location"])
        metrics["site_familiarity"] = _site_familiarity_breakdown(predictions, warm_sites)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "eval.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(metrics_dir / "roc_curve.csv", index=False)
    pd.DataFrame({"precision": prec, "recall": rec}).to_csv(metrics_dir / "pr_curve.csv", index=False)
    _save_plots(y_true, y_pred, fpr, tpr, roc_auc, prec, rec, pr_auc, metrics_dir)

    with mlflow.start_run(run_name=f"evaluate_{tag}"):
        mlflow.log_param("architecture", architecture)
        mlflow.log_param("test_split", test_split)
        mlflow.log_param("checkpoint_path", str(checkpoint_path))
        mlflow.log_param("threshold", threshold)
        scalar_keys = (
            "accuracy", "precision", "recall", "f1",
            "precision_macro", "recall_macro", "f1_macro",
            "precision_micro", "recall_micro", "f1_micro",
            "roc_auc", "pr_auc",
        )
        for key in scalar_keys:
            mlflow.log_metric(key, metrics[key])
        if test_split == "seasonal":
            for site_kind, site_metrics in metrics["site_familiarity"].items():
                for metric_key in ("accuracy", "precision", "recall", "f1"):
                    mlflow.log_metric(f"{site_kind}_{metric_key}", site_metrics[metric_key])
        mlflow.log_artifact(str(predictions_path))
        mlflow.log_artifact(str(metrics_dir / "eval.json"))
        mlflow.log_artifact(str(metrics_dir / "roc_curve.csv"))
        mlflow.log_artifact(str(metrics_dir / "pr_curve.csv"))
        mlflow.log_artifact(str(metrics_dir / "confusion_matrix.png"))
        mlflow.log_artifact(str(metrics_dir / "roc_curve.png"))
        mlflow.log_artifact(str(metrics_dir / "pr_curve.png"))

    print(
        f"[{test_split}, {tag}] accuracy={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
        f"roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f}"
    )
    print(
        f"  macro:  precision={metrics['precision_macro']:.4f} recall={metrics['recall_macro']:.4f} "
        f"f1={metrics['f1_macro']:.4f}"
    )
    print(
        f"  micro:  precision={metrics['precision_micro']:.4f} recall={metrics['recall_micro']:.4f} "
        f"f1={metrics['f1_micro']:.4f}  (== accuracy in binary classification)"
    )
    print(
        f"  per-class [bird, mammal]:  precision={metrics['precision_per_class']}  "
        f"recall={metrics['recall_per_class']}  f1={metrics['f1_per_class']}"
    )
    print(f"confusion matrix (rows=true, cols=pred) [bird, mammal]:\n{metrics['confusion_matrix']}")
    if test_split == "seasonal":
        fam = metrics["site_familiarity"]
        print("  site familiarity breakdown (does the model rely on having seen this camera before?):")
        for key in ("seen_site", "novel_site"):
            m = fam[key]
            print(
                f"    {key}: n={m['n']}  accuracy={m['accuracy']:.4f} precision={m['precision']:.4f} "
                f"recall={m['recall']:.4f} f1={m['f1']:.4f}"
            )
    print(f"wrote {predictions_path}, {metrics_dir}/eval.json, roc_curve.{{csv,png}}, pr_curve.{{csv,png}}, confusion_matrix.png")

    return metrics


def main() -> None:
    import sys

    architecture = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ARCHITECTURE
    test_split = sys.argv[2] if len(sys.argv) > 2 else "test"

    if test_split == "boxcrop":
        # Box-crop experiment: same site-based test rows as the main
        # baseline, just cropped to their MegaDetector box first, matching
        # how the box-crop checkpoint was trained.
        tag = f"{architecture}_boxcrop"
        run_evaluation(architecture=architecture, tag=tag, test_split="test", crop_to_box=True)
        return

    if test_split == "seasonal_boxcrop":
        # Seasonal generalisation, box-crop variant: same cool-season test
        # rows as the plain seasonal experiment, cropped to their
        # MegaDetector box first, matching how the seasonal_boxcrop
        # checkpoint was trained.
        tag = f"{architecture}_seasonal_boxcrop"
        run_evaluation(architecture=architecture, tag=tag, test_split="seasonal", crop_to_box=True)
        return

    tag = f"{architecture}_seasonal" if test_split == "seasonal" else architecture
    run_evaluation(architecture=architecture, tag=tag, test_split=test_split)


if __name__ == "__main__":
    main()
