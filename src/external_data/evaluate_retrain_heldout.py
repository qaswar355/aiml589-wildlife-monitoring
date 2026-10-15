"""
Before-vs-after comparison of retraining on every external image the
retrained model never trained on.

The external data was split by site into train / val / test
(build_external_manifest.py). Retraining (boxcrop_experiment.py on
data/manifest_combined.csv) only used the val split to log validation
loss: there is no early stopping and no best-checkpoint saving, and the
final-epoch weights are kept. So val is as unseen as test, and both are
pooled here into one held-out set (1,074 images from 5 sites) -- the test
split alone is only 152 images and 26 birds.

Reads the saved, MegaDetector-cropped external predictions of both
checkpoints (evaluate_external_cropped.py), so nothing is re-run. Per
threshold: accuracy, false positives, false negatives, how many errors
retraining fixed vs introduced, and an exact McNemar test on those.

Run: python -m src.external_data.evaluate_retrain_heldout
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

BEFORE_PATH = Path("data/predictions/external_predictions_efficientnet_b3_boxcrop_cropped.csv")
AFTER_PATH = Path("data/predictions/external_predictions_efficientnet_b3_boxcrop_retrained_cropped.csv")
HELDOUT_SPLITS = {"heldout": ("val", "test"), "test_only": ("test",)}
THRESHOLDS = (0.05, 0.5, 0.95)
OUT_PATH = Path("metrics/comparisons/retrain_heldout.json")


def _counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "false_positives": int(((y_pred == 1) & (y_true == 0)).sum()),
        "false_negatives": int(((y_pred == 0) & (y_true == 1)).sum()),
    }


def compare_at_threshold(
    y_true: np.ndarray, before_prob: np.ndarray, after_prob: np.ndarray, threshold: float
) -> dict:
    before_pred = (before_prob >= threshold).astype(int)
    after_pred = (after_prob >= threshold).astype(int)
    before_wrong = before_pred != y_true
    after_wrong = after_pred != y_true
    fixed = int((before_wrong & ~after_wrong).sum())
    new = int((~before_wrong & after_wrong).sum())
    return {
        "threshold": threshold,
        "before": _counts(y_true, before_pred),
        "after": _counts(y_true, after_pred),
        "errors_fixed": fixed,
        "errors_new": new,
        "mcnemar_p": float(binomtest(fixed, fixed + new, 0.5).pvalue) if fixed + new else 1.0,
    }


def load_paired(before_path: Path = BEFORE_PATH, after_path: Path = AFTER_PATH) -> pd.DataFrame:
    before = pd.read_csv(before_path).set_index("image_id")
    after = pd.read_csv(after_path).set_index("image_id")
    if set(before.index) != set(after.index):
        raise ValueError("before/after prediction files cover different images")
    after = after.reindex(before.index)
    if not (before["label"].equals(after["label"]) and before["split"].equals(after["split"])):
        raise ValueError("before/after prediction files disagree on label or split")
    paired = before[["label", "split", "event_id"]].copy()
    paired["site"] = paired["event_id"].str.split("|").str[0]
    paired["before_prob"] = before["probability"]
    paired["after_prob"] = after["probability"]
    return paired


def run_retrain_heldout() -> dict:
    paired = load_paired()
    train_sites = set(paired.loc[paired["split"] == "train", "site"])
    result = {}
    for name, splits in HELDOUT_SPLITS.items():
        subset = paired[paired["split"].isin(splits)]
        overlap = train_sites & set(subset["site"])
        if overlap:
            raise ValueError(f"{name}: sites also in the retraining split: {sorted(overlap)}")
        y = subset["label"].to_numpy()
        result[name] = {
            "splits": list(splits),
            "n_images": len(subset),
            "n_birds": int((y == 0).sum()),
            "n_mammals": int((y == 1).sum()),
            "n_sites": int(subset["site"].nunique()),
            "thresholds": [
                compare_at_threshold(y, subset["before_prob"].to_numpy(), subset["after_prob"].to_numpy(), t)
                for t in THRESHOLDS
            ],
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    for name, r in result.items():
        print(f"{name}: {r['n_images']} images ({r['n_birds']} birds, {r['n_mammals']} mammals) from {r['n_sites']} sites")
        for t in r["thresholds"]:
            b, a = t["before"], t["after"]
            print(f"  {t['threshold']:<5} accuracy {b['accuracy']:.1%} -> {a['accuracy']:.1%}  "
                  f"FP {b['false_positives']} -> {a['false_positives']}  FN {b['false_negatives']} -> {a['false_negatives']}  "
                  f"fixed {t['errors_fixed']} / new {t['errors_new']}  McNemar p={t['mcnemar_p']:.2g}")
    print(f"wrote {OUT_PATH}")
    return result


if __name__ == "__main__":
    run_retrain_heldout()
