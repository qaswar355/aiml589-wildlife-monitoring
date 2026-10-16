"""
Is MC Dropout's uncertainty any better at spotting errors than the
model's own single-pass confidence?

The plain-confidence score is how close one deterministic prediction is
to 50/50: 0.5 - |p - 0.5|, so higher means less sure. It is compared
against the saved MC Dropout uncertainty
(metrics/<tag>/calibration/uncertainty_scores.csv, from
calibration.run_uncertainty_analysis) on the same validation images and
the same errors (counted at that run's decision threshold), so the only
difference between the two rankings is the score itself.

Only one deterministic forward pass is needed, so this is quick. The
validation loader is unshuffled, so rows line up with the saved MC
Dropout scores; the labels and predicted labels are checked to make
sure.

Writes, per tag:
  - metrics/<tag>/calibration/val_probabilities.csv (true_label, probability)
  - metrics/<tag>/calibration/confidence_baseline.json

Run: python -m src.inference.confidence_baseline
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.inference.calibration import _collect_val_logits
from src.training.evaluate import DEFAULT_DATA_CONFIG, _load_yaml

# (tag, crop_to_box) -- the standard model analysed in Chapter 4 and the deployed model.
RUNS = (("efficientnet_b3", False), ("efficientnet_b3_boxcrop_retrained", True))
FLAG_FRACTIONS = (0.10, 0.20)


def plain_confidence_score(probability: np.ndarray) -> np.ndarray:
    """Higher = less sure: 0 for a prediction of 0 or 1, 0.5 at 50/50."""
    return 0.5 - np.abs(probability - 0.5)


def errors_caught(score: np.ndarray, is_error: np.ndarray, flag_fraction: float) -> float:
    """Share of all errors among the `flag_fraction` highest-scoring predictions."""
    k = int(round(flag_fraction * len(score)))
    flagged = np.argsort(-score, kind="stable")[:k]
    return float(is_error[flagged].sum() / is_error.sum())


def compare(mc_uncertainty: np.ndarray, probability: np.ndarray, is_error: np.ndarray) -> dict:
    plain = plain_confidence_score(probability)
    out = {}
    for name, score in (("mc_dropout", mc_uncertainty), ("plain_confidence", plain)):
        out[name] = {
            "auc": float(roc_auc_score(is_error, score)),
            **{f"errors_caught_at_{int(f * 100)}pct": errors_caught(score, is_error, f) for f in FLAG_FRACTIONS},
        }
    return out


def run_confidence_baseline(tag: str, crop_to_box: bool) -> dict:
    data_cfg = _load_yaml(DEFAULT_DATA_CONFIG)["data"]
    calib_dir = Path("metrics") / tag / "calibration"
    mc = pd.read_csv(calib_dir / "uncertainty_scores.csv")
    threshold = json.loads((calib_dir / "uncertainty.json").read_text())["threshold_used"]

    logits, labels = _collect_val_logits(
        "efficientnet_b3", f"models/checkpoint_{tag}.pt", data_cfg["manifest_path"], data_cfg["shards_dir"],
        crop_to_box=crop_to_box, box_crop_padding=data_cfg.get("box_crop_padding", 0.15),
    )
    probability = torch.sigmoid(logits).numpy().astype(np.float64)
    labels = labels.int().numpy()
    if len(labels) != len(mc) or not np.array_equal(labels, mc["true_label"].to_numpy()):
        raise ValueError(f"{tag}: validation rows do not line up with uncertainty_scores.csv")
    if not np.array_equal((probability >= threshold).astype(int), mc["pred_label"].to_numpy()):
        raise ValueError(f"{tag}: predicted labels differ from uncertainty_scores.csv")

    pd.DataFrame({"true_label": labels, "probability": probability}).to_csv(
        calib_dir / "val_probabilities.csv", index=False
    )
    result = {"tag": tag, "threshold_used": threshold, "n_val_images": len(labels),
              "n_errors": int(mc["is_error"].sum()),
              **compare(mc["uncertainty"].to_numpy(), probability, mc["is_error"].to_numpy())}
    (calib_dir / "confidence_baseline.json").write_text(json.dumps(result, indent=2))
    for name in ("mc_dropout", "plain_confidence"):
        r = result[name]
        print(f"[{tag}] {name:16s} AUC {r['auc']:.3f}  caught {r['errors_caught_at_10pct']:.1%} @10%, "
              f"{r['errors_caught_at_20pct']:.1%} @20%")
    return result


if __name__ == "__main__":
    for tag, crop in RUNS:
        run_confidence_baseline(tag, crop)
