"""
Integrated Gradients attribution (captum) -- pixel-level cross-validation
against Grad-CAM++ (src/xai/gradcam.py).

Why a second method, and why this one specifically:
  Grad-CAM++ in this project always computes gradients toward the mammal
  logit (BinaryClassifierOutputTarget(1) -- see gradcam.py), even for
  images correctly classified as birds. That's a real asymmetry: a
  bird's heatmap shows "what looks mammal-like here", not a symmetric
  "why did the model call this a bird". Integrated Gradients also has a
  different theoretical grounding -- attributions sum exactly to
  F(input) - F(baseline) (the completeness axiom), rather than Grad-CAM's
  single-layer spatial approximation. Agreement between two methods this
  different in mechanism and question is much stronger evidence than
  either one alone.

This module computes IG two ways, kept deliberately separate so the two
comparisons don't get conflated:
  - ig_score_mammal_target: attributed toward the mammal logit, exactly
    matching Grad-CAM's own target -- the strict apples-to-apples
    cross-validation of "do two different methods agree on the same
    question".
  - ig_score: attributed toward whichever direction supports the
    model's ACTUAL prediction for that image (mammal logit for a
    predicted mammal, its negation for a predicted bird) -- a
    genuinely different, arguably more natural question ("what
    supports this prediction"), and a direct test of whether Grad-CAM's
    fixed-target asymmetry was hiding something.

Baseline for IG is an all-zero (post-normalisation) input -- "no
signal", the standard default for IG on natural images.

Cost note: IG needs n_steps forward+backward passes per image (default
20 here) versus Grad-CAM's single pass, so this audits a fixed-size
stratified SAMPLE per outcome (seeded, reproducible) rather than scoring
the entire test set the way gradcam.py's run_attention_audit does --
enough images to check statistical agreement between methods without an
hours-long run per checkpoint.

Run: python -m src.xai.integrated_gradients <tag> --audit [architecture] [--crop-to-box]
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from scipy.stats import pearsonr, spearmanr

from src.models.classifier import load_checkpoint
from src.training.dataset import DEFAULT_TRANSFORM, ShardReader, crop_bounds_from_bbox, safe_member_name
from src.training.evaluate import DEFAULT_DATA_CONFIG, _load_yaml, _resolve_torch_device
from src.xai.gradcam import OUTCOME_DEFINITIONS, TARGET_LAYER_BY_ARCHITECTURE, compute_boxcrop_cam, pointing_game_score

DEFAULT_ARCHITECTURE = "efficientnet_b3"
DEFAULT_N_STEPS = 20
DEFAULT_N_PER_OUTCOME = 200
N_EXAMPLE_HEATMAPS = 6


def reduce_ig_attribution(attr: np.ndarray, sign: float) -> np.ndarray:
    """Turns a raw (C, H, W) IG attribution tensor into a single-channel,
    non-negative (H, W) importance map: sum across colour channels, apply
    `sign` (see module docstring), then clip to non-negative.

    The clip matters for comparability with Grad-CAM: Grad-CAM's own
    final step also keeps only positive contributions toward its target
    (see pytorch_grad_cam's BaseCAM), so clipping here keeps both methods
    answering "where is there POSITIVE evidence for the target
    direction", not "where is there evidence in either direction". Pure
    numpy, no model/device involved, so this is unit-testable on its own
    (see tests/xai/test_integrated_gradients.py).
    """
    signed = sign * attr.sum(axis=0)
    return np.clip(signed, a_min=0.0, a_max=None)


def compute_ig_map(
    ig: IntegratedGradients,
    input_tensor: torch.Tensor,
    sign: float,
    n_steps: int = DEFAULT_N_STEPS,
) -> np.ndarray:
    """Runs Integrated Gradients on one already-transformed input tensor,
    returns a pointing-game-scoreable (H, W) map. IG attribution is
    linear in the function being attributed (it's an integral of a
    gradient along a straight-line path), so attributing toward
    `sign * model_output` is exactly `sign` times attributing toward
    `model_output` -- computing the raw attribution once and applying
    `sign` afterward (in reduce_ig_attribution) is mathematically
    identical to negating the model's output before attributing, and
    avoids a second attribute() call.
    """
    baseline = torch.zeros_like(input_tensor)
    attr = ig.attribute(input_tensor, baselines=baseline, target=None, n_steps=n_steps)
    attr = attr.squeeze(0).detach().cpu().numpy()  # (3, H, W)
    return reduce_ig_attribution(attr, sign)


def compute_boxcrop_ig_map(
    ig: IntegratedGradients,
    image: Image.Image,
    bbox: list[float],
    padding: float,
    sign: float,
    device: torch.device,
    n_steps: int = DEFAULT_N_STEPS,
    canvas_size: int = 300,
) -> np.ndarray:
    """Crop-aware counterpart to compute_ig_map, mirroring
    gradcam.compute_boxcrop_cam exactly: crops to the MegaDetector box
    the same way training did (crop_bounds_from_bbox), runs IG on that
    crop, then places the result back at its true location on a
    canvas_size x canvas_size canvas representing the original,
    uncropped frame -- zero everywhere outside the crop, since a
    box-crop-trained model never saw anything outside it. See
    compute_boxcrop_cam's docstring for why this remapping (rather than
    scoring directly on the crop) is what keeps scores comparable across
    checkpoints trained with and without cropping.
    """
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    bounds = crop_bounds_from_bbox(bbox, padding)
    if bounds is None:
        input_tensor = DEFAULT_TRANSFORM(image).unsqueeze(0).to(device)
        return compute_ig_map(ig, input_tensor, sign, n_steps)

    x0, y0, x1, y1 = bounds
    img_w, img_h = image.size
    left, top = int(x0 * img_w), int(y0 * img_h)
    right, bottom = int(x1 * img_w), int(y1 * img_h)
    cropped = image.crop((left, top, right, bottom))

    input_tensor = DEFAULT_TRANSFORM(cropped).unsqueeze(0).to(device)
    crop_map = compute_ig_map(ig, input_tensor, sign, n_steps)

    cx0, cy0 = int(x0 * canvas_size), int(y0 * canvas_size)
    cx1, cy1 = int(x1 * canvas_size), int(y1 * canvas_size)
    cx1, cy1 = max(cx1, cx0 + 1), max(cy1, cy0 + 1)
    resized = np.array(Image.fromarray(crop_map, mode="F").resize((cx1 - cx0, cy1 - cy0), Image.BILINEAR))
    canvas[cy0:cy1, cx0:cx1] = resized
    return canvas


def _save_side_by_side_heatmap(
    image: Image.Image,
    gradcam_map: np.ndarray,
    ig_map: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """One figure per example: the two methods' heatmaps side by side, so
    agreement/disagreement is visible at a glance rather than only as a
    number."""
    display_image = np.asarray(image.resize((300, 300)), dtype=np.float32) / 255.0
    gradcam_overlay = show_cam_on_image(display_image, gradcam_map, use_rgb=True)

    ig_norm = ig_map / ig_map.max() if ig_map.max() > 0 else ig_map
    ig_overlay = show_cam_on_image(display_image, ig_norm, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(gradcam_overlay)
    axes[0].set_title("Grad-CAM++", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(ig_overlay)
    axes[1].set_title("Integrated Gradients", fontsize=9)
    axes[1].axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _sample_per_outcome(joined: pd.DataFrame, n_per_outcome: int, seed: int) -> pd.DataFrame:
    parts = []
    for outcome, mask_fn in OUTCOME_DEFINITIONS.items():
        subset = joined[mask_fn(joined)].copy()
        subset["outcome"] = outcome
        n = min(n_per_outcome, len(subset))
        parts.append(subset.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def run_cross_validation_audit(
    tag: str,
    architecture: str = DEFAULT_ARCHITECTURE,
    checkpoint_path: str | None = None,
    manifest_path: str | None = None,
    shards_dir: str | None = None,
    predictions_path: str | None = None,
    n_per_outcome: int = DEFAULT_N_PER_OUTCOME,
    n_steps: int = DEFAULT_N_STEPS,
    crop_to_box: bool = False,
    box_crop_padding: float | None = None,
    n_examples_per_outcome: int = N_EXAMPLE_HEATMAPS,
    seed: int = 42,
) -> dict:
    """Scores a seeded, stratified sample of `n_per_outcome` images per
    outcome (see OUTCOME_DEFINITIONS) with BOTH Grad-CAM++ (mammal
    target, exactly matching gradcam.py's existing audit) and Integrated
    Gradients (both the same mammal-target and the predicted-class-target
    variant -- see module docstring), so the two methods can be compared
    image-for-image, not just via separately-computed aggregate means.

    `crop_to_box` must be set for any checkpoint trained by
    boxcrop_experiment.py / seasonal_experiment.py --crop-to-box -- see
    gradcam.run_attention_audit's docstring for why (distribution
    mismatch otherwise, and scoring directly on a crop would inflate the
    pointing-game score mechanically).
    """
    checkpoint_path = checkpoint_path or f"models/checkpoint_{tag}.pt"
    out_dir = Path("metrics") / tag / "xai" / "ig_cross_validation"
    heatmap_dir = out_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = _load_yaml(DEFAULT_DATA_CONFIG)["data"]
    manifest_path = manifest_path or data_cfg["manifest_path"]
    shards_dir = shards_dir or data_cfg["shards_dir"]
    if box_crop_padding is None:
        box_crop_padding = data_cfg.get("box_crop_padding", 0.15)

    manifest = pd.read_csv(manifest_path)
    predictions_path = predictions_path or f"data/predictions/test_predictions_{tag}.csv"
    predictions = pd.read_csv(predictions_path)

    boxes = manifest[["image_id", "bbox", "has_box", "project", "location"]]
    joined = predictions.merge(boxes, on="image_id", how="left")
    joined["bbox"] = joined["bbox"].apply(lambda b: json.loads(b) if isinstance(b, str) and b else None)
    joined = joined[joined["bbox"].notna()]

    sample = _sample_per_outcome(joined, n_per_outcome, seed)
    print(f"sampled {len(sample)} images ({n_per_outcome}/outcome, seed={seed}) for cross-validation")

    device = _resolve_torch_device()
    model = load_checkpoint(checkpoint_path, architecture=architecture, device=device)
    target_layer = TARGET_LAYER_BY_ARCHITECTURE[architecture](model)
    cam_builder = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    ig = IntegratedGradients(lambda x: model(x))
    reader = ShardReader(shards_dir)

    rows: list[dict] = []
    saved_per_outcome: dict[str, int] = {o: 0 for o in OUTCOME_DEFINITIONS}
    for _, row in sample.iterrows():
        member_name = safe_member_name(row["image_id"])
        if member_name not in reader:
            continue
        image = Image.open(io.BytesIO(reader.read(member_name))).convert("RGB")
        predicted_sign = 1.0 if row["pred_label"] == 1 else -1.0

        if crop_to_box and row.get("has_box", False):
            gradcam_map = compute_boxcrop_cam(cam_builder, image, row["bbox"], box_crop_padding)
            ig_mammal_map = compute_boxcrop_ig_map(ig, image, row["bbox"], box_crop_padding, 1.0, device, n_steps)
            ig_pred_map = compute_boxcrop_ig_map(
                ig, image, row["bbox"], box_crop_padding, predicted_sign, device, n_steps
            )
        else:
            input_tensor = DEFAULT_TRANSFORM(image).unsqueeze(0).to(device)
            gradcam_map = cam_builder(input_tensor=input_tensor, targets=[BinaryClassifierOutputTarget(1)])[0]
            ig_mammal_map = compute_ig_map(ig, input_tensor, 1.0, n_steps)
            ig_pred_map = compute_ig_map(ig, input_tensor, predicted_sign, n_steps)

        gradcam_score = pointing_game_score(gradcam_map, row["bbox"])
        ig_mammal_score = pointing_game_score(ig_mammal_map, row["bbox"])
        ig_pred_score = pointing_game_score(ig_pred_map, row["bbox"])

        rows.append(
            {
                "image_id": row["image_id"],
                "species": row["species"],
                "project": row["project"],
                "outcome": row["outcome"],
                "gradcam_score": gradcam_score,
                "ig_score_mammal_target": ig_mammal_score,
                "ig_score": ig_pred_score,
            }
        )

        outcome = row["outcome"]
        if saved_per_outcome[outcome] < n_examples_per_outcome:
            title = f"{row['species']} ({outcome}, gradcam={gradcam_score:.2f}, ig={ig_pred_score:.2f})"
            _save_side_by_side_heatmap(
                image, gradcam_map, ig_pred_map, heatmap_dir / f"{outcome}__{member_name}", title
            )
            saved_per_outcome[outcome] += 1

    scores_df = pd.DataFrame(rows)
    scores_df.to_csv(out_dir / "cross_validation_scores.csv", index=False)

    summary: dict = {"tag": tag, "n_per_outcome": n_per_outcome, "n_steps": n_steps, "crop_to_box": crop_to_box, "seed": seed}
    for outcome in OUTCOME_DEFINITIONS:
        subset = scores_df[scores_df["outcome"] == outcome]
        summary[outcome] = {
            "n_scored": len(subset),
            "gradcam_mean": float(subset["gradcam_score"].mean()) if len(subset) else None,
            "gradcam_median": float(subset["gradcam_score"].median()) if len(subset) else None,
            "ig_mammal_target_mean": float(subset["ig_score_mammal_target"].mean()) if len(subset) else None,
            "ig_mammal_target_median": float(subset["ig_score_mammal_target"].median()) if len(subset) else None,
            "ig_mean": float(subset["ig_score"].mean()) if len(subset) else None,
            "ig_median": float(subset["ig_score"].median()) if len(subset) else None,
        }
        print(
            f"{outcome}: n={summary[outcome]['n_scored']}  "
            f"gradcam mean={summary[outcome]['gradcam_mean']:.4f}  "
            f"ig(mammal-target) mean={summary[outcome]['ig_mammal_target_mean']:.4f}  "
            f"ig(predicted-class) mean={summary[outcome]['ig_mean']:.4f}"
        )

    # Strict apples-to-apples correlation: same target definition (mammal),
    # two different attribution mechanisms, same images.
    if len(scores_df) >= 2:
        pearson_r, pearson_p = pearsonr(scores_df["gradcam_score"], scores_df["ig_score_mammal_target"])
        spearman_r, spearman_p = spearmanr(scores_df["gradcam_score"], scores_df["ig_score_mammal_target"])
        summary["gradcam_vs_ig_mammal_target"] = {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
        }
        print(
            f"gradcam vs ig(mammal-target) correlation across all {len(scores_df)} sampled images: "
            f"pearson r={pearson_r:.3f} (p={pearson_p:.2e})  spearman rho={spearman_r:.3f} (p={spearman_p:.2e})"
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    with mlflow.start_run(run_name=f"ig_cross_validation_{tag}"):
        mlflow.log_param("architecture", architecture)
        mlflow.log_param("checkpoint_path", str(checkpoint_path))
        mlflow.log_param("crop_to_box", crop_to_box)
        mlflow.log_param("n_per_outcome", n_per_outcome)
        mlflow.log_param("n_steps", n_steps)
        for outcome in OUTCOME_DEFINITIONS:
            if summary[outcome]["gradcam_mean"] is not None:
                mlflow.log_metric(f"{outcome}_gradcam_mean", summary[outcome]["gradcam_mean"])
                mlflow.log_metric(f"{outcome}_ig_mammal_target_mean", summary[outcome]["ig_mammal_target_mean"])
                mlflow.log_metric(f"{outcome}_ig_mean", summary[outcome]["ig_mean"])
        if "gradcam_vs_ig_mammal_target" in summary:
            mlflow.log_metric("gradcam_vs_ig_pearson_r", summary["gradcam_vs_ig_mammal_target"]["pearson_r"])
            mlflow.log_metric("gradcam_vs_ig_spearman_r", summary["gradcam_vs_ig_mammal_target"]["spearman_r"])
        mlflow.log_artifact(str(out_dir / "summary.json"))
        mlflow.log_artifact(str(out_dir / "cross_validation_scores.csv"))

    print(f"wrote {out_dir}/summary.json, cross_validation_scores.csv, and heatmaps to {heatmap_dir}")
    return summary


def main() -> None:
    import sys

    argv = sys.argv[1:]
    crop_to_box = "--crop-to-box" in argv
    argv = [a for a in argv if a != "--crop-to-box"]

    if len(argv) < 2 or argv[1] != "--audit":
        raise SystemExit(
            "usage: python -m src.xai.integrated_gradients <tag> --audit [architecture] [--crop-to-box]"
        )
    tag = argv[0]
    architecture = argv[2] if len(argv) > 2 else DEFAULT_ARCHITECTURE
    run_cross_validation_audit(tag=tag, architecture=architecture, crop_to_box=crop_to_box)


if __name__ == "__main__":
    main()
