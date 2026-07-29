"""
Model evaluation: per-image predictions, aggregate metrics, and plots.

Loads a trained checkpoint and runs it over the manifest's test split,
producing:
  - data/predictions/test_predictions.csv -- one row per test image:
      image_id, site_id, species, season, true_label, pred_label,
      confidence, probability. DVC-tracked (see dvc.yaml), so it's
      reproducible and shareable without re-running training.
  - metrics/eval.json -- precision/recall/F1/accuracy/ROC-AUC/PR-AUC +
      confusion matrix in absolute counts (accuracy intentionally not
      the headline number, same reasoning as baselines.py)
  - metrics/roc_curve.csv, metrics/pr_curve.csv -- DVC-plottable curve
      points, plus rendered .png versions and a confusion matrix heatmap
      for direct use in the report

`probability` is the raw sigmoid output (P(mammal)); `confidence` is the
probability of whichever class was actually predicted (max(p, 1-p)) --
these differ whenever the model predicts bird (pred_label=0), which is
what lets a review queue flag "low confidence" regardless of which way
the model leaned.

Threshold sweep / per-zone thresholds (configs/inference/threshold_config.yaml)
and the seasonal three-way ablation are follow-up work, not this pass --
this evaluates the single trained checkpoint at the default threshold.

Run: python -m src.training.evaluate
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

from src.models.efficientnet import WildlifeClassifier
from src.training.dataset import ShardDataset

DEFAULT_DATA_CONFIG = "configs/data/default.yaml"
DEFAULT_THRESHOLD_CONFIG = "configs/inference/threshold_config.yaml"
DEFAULT_CHECKPOINT = "models/checkpoint.pt"
PREDICTIONS_PATH = Path("data/predictions/test_predictions.csv")
METRICS_DIR = Path("metrics")


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


def run_evaluation(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    manifest_path: str | None = None,
    shards_dir: str | None = None,
    threshold: float | None = None,
    batch_size: int = 32,
) -> dict:
    data_cfg = _load_yaml(DEFAULT_DATA_CONFIG)["data"]
    manifest_path = manifest_path or data_cfg["manifest_path"]
    shards_dir = shards_dir or data_cfg["shards_dir"]

    if threshold is None:
        threshold = _load_yaml(DEFAULT_THRESHOLD_CONFIG)["thresholds"]["default"]

    manifest = pd.read_csv(manifest_path)
    test_ds = ShardDataset(manifest, shards_dir=shards_dir, split="test")
    if len(test_ds) == 0:
        raise RuntimeError(f"No test images found in {shards_dir} -- has ingest finished?")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = _resolve_torch_device()
    model = WildlifeClassifier()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

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
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    y_true = predictions["true_label"].to_numpy()
    y_pred = predictions["pred_label"].to_numpy()
    y_prob = predictions["probability"].to_numpy()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    metrics = {
        "threshold": threshold,
        "n_test_images": len(predictions),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "eval.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(METRICS_DIR / "roc_curve.csv", index=False)
    pd.DataFrame({"precision": prec, "recall": rec}).to_csv(METRICS_DIR / "pr_curve.csv", index=False)
    _save_plots(y_true, y_pred, fpr, tpr, roc_auc, prec, rec, pr_auc, METRICS_DIR)

    with mlflow.start_run(run_name="evaluate_test_set"):
        mlflow.log_param("checkpoint_path", str(checkpoint_path))
        mlflow.log_param("threshold", threshold)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            mlflow.log_metric(key, metrics[key])
        mlflow.log_artifact(str(PREDICTIONS_PATH))
        mlflow.log_artifact(str(METRICS_DIR / "eval.json"))
        mlflow.log_artifact(str(METRICS_DIR / "roc_curve.csv"))
        mlflow.log_artifact(str(METRICS_DIR / "pr_curve.csv"))
        mlflow.log_artifact(str(METRICS_DIR / "confusion_matrix.png"))
        mlflow.log_artifact(str(METRICS_DIR / "roc_curve.png"))
        mlflow.log_artifact(str(METRICS_DIR / "pr_curve.png"))

    print(
        f"[test] accuracy={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
        f"roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f}"
    )
    print(f"confusion matrix (rows=true, cols=pred) [bird, mammal]:\n{metrics['confusion_matrix']}")
    print(f"wrote {PREDICTIONS_PATH}, {METRICS_DIR}/eval.json, roc_curve.{{csv,png}}, pr_curve.{{csv,png}}, confusion_matrix.png")

    return metrics


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
