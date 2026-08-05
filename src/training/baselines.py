"""
Trivial baselines that need only manifest.csv labels — no image pixels,
no GPU, no dependency on the shard ingest finishing.

The majority-class classifier is the accuracy-is-misleading demonstration
called out in the 501 report and the AIML 589 proposal: predicting the
single most common label for every image can score deceptively high
"accuracy" while having zero recall on whichever class it never predicts.
This is the number every later model (ResNet-50, EfficientNet-B3) has to
actually beat on precision/recall/F1, not just accuracy.

Logs to MLflow using MLflow's own local file-store default (./mlruns) —
deliberately not going through Hydra/configs/training/default.yaml's
tracking_uri here, since that defaults to http://localhost:5000 and on
this machine port 5000 is already answering (macOS AirPlay Receiver, not
an MLflow server). Point MLflow's UI at ./mlruns locally:
  mlflow ui --backend-store-uri ./mlruns

Run: python -m src.training.baselines
"""
from __future__ import annotations

import pandas as pd
import mlflow
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

MANIFEST_PATH = "data/manifest.csv"
LABELS = [0, 1]  # 0 = bird, 1 = mammal


def _log_split_metrics(split_name: str, y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    rec_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    mlflow.log_metric(f"{split_name}_accuracy", acc)
    mlflow.log_metric(f"{split_name}_precision", prec)
    mlflow.log_metric(f"{split_name}_recall", rec)
    mlflow.log_metric(f"{split_name}_f1", f1)
    mlflow.log_metric(f"{split_name}_precision_macro", prec_macro)
    mlflow.log_metric(f"{split_name}_recall_macro", rec_macro)
    mlflow.log_metric(f"{split_name}_f1_macro", f1_macro)
    mlflow.log_metric(f"{split_name}_precision_micro", prec_micro)
    mlflow.log_metric(f"{split_name}_recall_micro", rec_micro)
    mlflow.log_metric(f"{split_name}_f1_micro", f1_micro)

    print(
        f"[{split_name}] accuracy={acc:.4f} precision={prec:.4f} "
        f"recall={rec:.4f} f1={f1:.4f}"
    )
    print(f"  macro:  precision={prec_macro:.4f} recall={rec_macro:.4f} f1={f1_macro:.4f}")
    print(
        f"  micro:  precision={prec_micro:.4f} recall={rec_micro:.4f} f1={f1_micro:.4f}  "
        "(== accuracy in binary classification)"
    )
    print(
        f"  per-class [bird, mammal]:  precision={prec_per_class.tolist()}  "
        f"recall={rec_per_class.tolist()}  f1={f1_per_class.tolist()}"
    )
    print(f"  confusion matrix (rows=true, cols=pred) [bird, mammal]:\n{cm}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_micro": prec_micro,
        "recall_micro": rec_micro,
        "f1_micro": f1_micro,
        "precision_per_class": prec_per_class.tolist(),
        "recall_per_class": rec_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def run_majority_class_baseline(manifest_path: str = MANIFEST_PATH) -> dict:
    df = pd.read_csv(manifest_path)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    majority_label = int(train["label"].mode().iloc[0])
    majority_name = "mammal" if majority_label == 1 else "bird"

    print(f"train rows: {len(train):,}  majority label: {majority_label} ({majority_name})")
    print(f"train label distribution:\n{train['label'].value_counts().to_string()}")

    results: dict = {}
    with mlflow.start_run(run_name="majority_class_baseline"):
        mlflow.log_param("model", "majority_class")
        mlflow.log_param("majority_label", majority_label)
        mlflow.log_param("majority_label_name", majority_name)
        mlflow.log_param("train_rows", len(train))
        mlflow.log_param("val_rows", len(val))
        mlflow.log_param("test_rows", len(test))

        for split_name, split_df in (("val", val), ("test", test)):
            y_true = split_df["label"].to_numpy()
            y_pred = [majority_label] * len(y_true)
            results[split_name] = _log_split_metrics(split_name, y_true, y_pred)

    return results


def main() -> None:
    run_majority_class_baseline()


if __name__ == "__main__":
    main()
