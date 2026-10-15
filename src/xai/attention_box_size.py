"""
Checks the attention audit (gradcam.run_attention_audit, predicted-class
heatmaps) for box size.

The pointing-game score is "what fraction of the heatmap falls inside the
MegaDetector box", so a bigger box catches more attention just by being
bigger. Bird boxes are much smaller than mammal boxes, so part of the
bird-vs-mammal gap could be box size rather than where the model looks.
This answers that three ways, all from saved audit scores (no model or
GPU needed):

  - box size per class: how much of the image the box covers
  - the audit grouped by box size, so birds and mammals are compared at
    the same box size
  - each score against its chance level -- what a heatmap spread evenly
    over the image would score. For the standard model that is the box's
    share of the frame; for the box-crop model it is the box's share of
    the padded crop, since compute_boxcrop_cam leaves everything outside
    the crop at zero (about 0.6 with 15% padding). "Above chance" is
    (score - chance) / (1 - chance): 0 = no better than even attention,
    1 = all attention inside the box. This is what makes the two models'
    scores comparable.

Chance levels are computed by scoring a uniform heatmap with the real
pointing_game_score / crop geometry, so they use exactly the same pixel
rounding as the audit itself.

Run: python -m src.xai.attention_box_size
     (needs both audits run first with --audit predicted, see gradcam.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.dataset import crop_bounds_from_bbox
from src.xai.gradcam import crop_canvas_bounds, pointing_game_score

CANVAS_SIZE = 300
BOX_CROP_PADDING = 0.15
STANDARD_TAG = "efficientnet_b3"
BOXCROP_TAG = "efficientnet_b3_boxcrop"
AUDIT_SUBDIR = "xai/attention_audit_predicted_class"
OUTCOMES = ["bird_correct", "mammal_correct", "false_positive", "false_negative"]
# Share of the image the box covers; (low, high] bins.
SIZE_BINS = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 1.0]
LOW_ATTENTION = 0.1
SMALL_BOX = 0.025
OUT_PATH = Path("metrics/comparisons/attention_box_size.json")


def full_frame_chance(bbox: list[float], canvas_size: int = CANVAS_SIZE) -> float:
    """Score of a heatmap spread evenly over the whole frame."""
    return pointing_game_score(np.ones((canvas_size, canvas_size), dtype=np.float32), bbox)


def boxcrop_chance(
    bbox: list[float], padding: float = BOX_CROP_PADDING, canvas_size: int = CANVAS_SIZE
) -> float:
    """Score of a heatmap spread evenly over the padded crop, placed on the
    original frame the same way compute_boxcrop_cam does."""
    bounds = crop_bounds_from_bbox(bbox, padding)
    if bounds is None:
        return full_frame_chance(bbox, canvas_size)
    cx0, cy0, cx1, cy1 = crop_canvas_bounds(bounds, canvas_size)
    cam = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    cam[cy0:cy1, cx0:cx1] = 1.0
    return pointing_game_score(cam, bbox)


def above_chance(score: pd.Series, chance: pd.Series) -> pd.Series:
    return (score - chance) / (1.0 - chance)


def load_audit(tag: str, test_boxes: pd.Series, chance_fn) -> pd.DataFrame:
    path = Path("metrics") / tag / AUDIT_SUBDIR / "attention_audit_scores.csv"
    df = pd.read_csv(path)
    bboxes = test_boxes.reindex(df["image_id"])
    if bboxes.isna().any():
        raise ValueError(f"{path}: {int(bboxes.isna().sum())} audited images have no box in the manifest")
    boxes = [json.loads(b) for b in bboxes]
    df["box_share"] = [b[2] * b[3] for b in boxes]
    df["chance"] = [chance_fn(b) for b in boxes]
    df["above_chance"] = above_chance(df["pointing_game_score"], df["chance"])
    return df


def outcome_summary(df: pd.DataFrame) -> dict:
    out = {}
    for outcome in OUTCOMES:
        g = df[df["outcome"] == outcome]
        out[outcome] = {
            "n": len(g),
            "score_median": float(g["pointing_game_score"].median()),
            "chance_median": float(g["chance"].median()),
            "above_chance_median": float(g["above_chance"].median()),
            "share_below_chance": float((g["pointing_game_score"] < g["chance"]).mean()),
        }
    return out


def by_box_size(df: pd.DataFrame) -> list[dict]:
    binned = pd.cut(df["box_share"], SIZE_BINS)
    rows = []
    for interval, g in df.groupby(binned, observed=False):
        row = {"box_share_low": float(interval.left), "box_share_high": float(interval.right)}
        for outcome in OUTCOMES:
            scores = g.loc[g["outcome"] == outcome, "pointing_game_score"]
            row[outcome] = {"n": len(scores), "median": float(scores.median()) if len(scores) else None}
        rows.append(row)
    return rows


def low_attention_birds(df: pd.DataFrame) -> dict:
    birds = df[df["outcome"] == "bird_correct"]
    low = birds[birds["pointing_game_score"] < LOW_ATTENTION]
    return {
        "n_correct_birds": len(birds),
        "n_below_0.1": len(low),
        "share_below_0.1": len(low) / len(birds),
        "share_of_those_with_box_under_2.5pct": float((low["box_share"] <= SMALL_BOX).mean()),
    }


def run_attention_box_size(manifest_path: str = "data/manifest.csv") -> dict:
    manifest = pd.read_csv(manifest_path, usecols=["image_id", "label", "bbox", "split"])
    test = manifest[(manifest["split"] == "test") & manifest["bbox"].notna()]
    test_boxes = test.set_index("image_id")["bbox"]

    box_share = test["bbox"].map(lambda s: (lambda b: b[2] * b[3])(json.loads(s)))
    box_size = {
        name: {
            "n": int((test["label"] == label).sum()),
            "median": float(box_share[test["label"] == label].median()),
            "mean": float(box_share[test["label"] == label].mean()),
        }
        for name, label in (("birds", 0), ("mammals", 1))
    }

    standard = load_audit(STANDARD_TAG, test_boxes, full_frame_chance)
    boxcrop = load_audit(BOXCROP_TAG, test_boxes, boxcrop_chance)

    result = {
        "box_size_by_class": box_size,
        "standard_by_box_size": by_box_size(standard),
        "standard_low_attention_birds": low_attention_birds(standard),
        "standard_by_outcome": outcome_summary(standard),
        "boxcrop_by_outcome": outcome_summary(boxcrop),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))

    print("box size (share of image):")
    for name, s in box_size.items():
        print(f"  {name:8} n={s['n']:5}  median={s['median']:.3f}  mean={s['mean']:.3f}")
    print("standard model, median score by box size:")
    for row in result["standard_by_box_size"]:
        cells = "  ".join(
            f"{o}={row[o]['median']:.2f} ({row[o]['n']})" if row[o]["n"] else f"{o}=-- (0)" for o in OUTCOMES[:3]
        )
        print(f"  ({row['box_share_low']}, {row['box_share_high']}]  {cells}")
    low = result["standard_low_attention_birds"]
    print(f"correct birds below 0.1: {low['n_below_0.1']} ({low['share_below_0.1']:.1%}), "
          f"{low['share_of_those_with_box_under_2.5pct']:.1%} of them with a box under 2.5% of the image")
    for name in ("standard", "boxcrop"):
        print(f"{name} by outcome (median score / chance / above chance):")
        for o, s in result[f"{name}_by_outcome"].items():
            print(f"  {o:15} n={s['n']:5}  {s['score_median']:.2f} / {s['chance_median']:.2f} / {s['above_chance_median']:.2f}")
    print(f"wrote {OUT_PATH}")
    return result


if __name__ == "__main__":
    run_attention_box_size()
