"""
Grad-CAM++ over the model's false positives -- birds it called mammals.

The question this answers: when the model gets it wrong in this specific
direction, is it actually looking at the animal, or has it latched onto
something in the background (foliage, lighting, the camera housing)? A
heatmap answers that visually; MegaDetector's bounding box (already in
the manifest for almost every image) lets us turn that into an actual
number instead of just eyeballing pictures -- the "pointing game" score
below is just "what fraction of the model's attention landed on the real
animal". 1.0 means all of it did, 0.0 means none of it did.

This is built to run against whatever checkpoint you point it at, the
same way evaluate.py does -- it reads that model's predictions CSV
(produced by `python -m src.training.evaluate <tag>`), so evaluate.py
has to run first. When the predictions came from the seasonal
experiment, this also breaks the score down by seen-site vs novel-site,
extending that finding with an actual explanation rather than just a
number.

Note: MegaDetector's boxes are model predictions, not human annotations
(there are no human-drawn boxes for this dataset -- see the EDA notes),
so the pointing-game score is measured against a pseudo-ground-truth,
not verified ground truth. Worth saying so in the report.

Also here: run_attention_audit(), a broader check triggered by a manual
live-demo test that found a correctly-classified kiwi whose heatmap
showed zero attention on the bird at all. run_gradcam_analysis() only
ever looked at false positives, so it couldn't say whether that was a
one-off or how this model generally solves the task, right answers
included. The audit scores the pointing game across all four prediction
outcomes (bird correctly classified, mammal correctly classified, and
both error directions), over the whole test set rather than a sample.

Run: python -m src.xai.gradcam <tag> [architecture] [test_split]
     e.g. python -m src.xai.gradcam efficientnet_b3_seasonal efficientnet_b3 seasonal
     python -m src.xai.gradcam <tag> --audit [architecture]
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
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from src.models.classifier import load_checkpoint
from src.training.dataset import DEFAULT_TRANSFORM, ShardReader, crop_bounds_from_bbox, safe_member_name
from src.training.evaluate import DEFAULT_DATA_CONFIG, _load_yaml, _resolve_torch_device

DEFAULT_ARCHITECTURE = "efficientnet_b3"
WARM_SEASONS = ("spring", "summer")
N_EXAMPLE_HEATMAPS = 16

# Grad-CAM needs a specific convolutional layer to read gradients from --
# by convention, the last one before the classifier head, since that's
# where spatial detail and learned features are both still present.
# Public (no leading underscore) because src.inference.predict reuses this
# same architecture -> layer mapping rather than keeping a second copy.
TARGET_LAYER_BY_ARCHITECTURE = {
    "efficientnet_b3": lambda model: model.backbone.features[-1],
    "resnet50": lambda model: model.backbone.layer4[-1],
}


def compute_and_save_heatmap(
    cam_builder: GradCAMPlusPlus,
    image: Image.Image,
    out_path: Path,
    title: str,
) -> Path:
    """Runs Grad-CAM++ over `image` and saves the image+heatmap overlay to
    `out_path`. Shared by the batch false-positive analysis below and by
    live single-image inference, so there's one implementation instead of
    two that could drift apart.

    Takes `cam_builder` rather than a separate device argument -- it
    already knows the model's device (pytorch_grad_cam's BaseCAM sets
    `.device` from the wrapped model's own parameters).
    """
    input_tensor = DEFAULT_TRANSFORM(image).unsqueeze(0).to(cam_builder.device)
    grayscale_cam = cam_builder(input_tensor=input_tensor, targets=[BinaryClassifierOutputTarget(1)])[0]

    display_image = np.asarray(image.resize((300, 300)), dtype=np.float32) / 255.0
    overlay = show_cam_on_image(display_image, grayscale_cam, use_rgb=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(overlay)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def pointing_game_score(cam: np.ndarray, bbox: list[float]) -> float:
    """Fraction of a heatmap's attention that falls inside a bounding box.

    `bbox` is MegaDetector's [x, y, width, height], normalised 0-1 against
    the original image -- that normalisation is what lets it line up with
    `cam` correctly even though `cam` is a fixed 300x300, regardless of
    the original photo's aspect ratio.
    """
    h, w = cam.shape
    x0 = max(0, int(bbox[0] * w))
    y0 = max(0, int(bbox[1] * h))
    x1 = min(w, int((bbox[0] + bbox[2]) * w))
    y1 = min(h, int((bbox[1] + bbox[3]) * h))

    total = cam.sum()
    if total <= 0 or x1 <= x0 or y1 <= y0:
        return 0.0
    return float(cam[y0:y1, x0:x1].sum() / total)


def compute_boxcrop_cam(
    cam_builder: GradCAMPlusPlus,
    image: Image.Image,
    bbox: list[float],
    padding: float,
    canvas_size: int = 300,
) -> np.ndarray:
    """Runs Grad-CAM++ on `image` cropped to its MegaDetector box the same
    way training did (crop_bounds_from_bbox), then places the resulting
    heatmap back at its true location on a canvas_size x canvas_size
    canvas representing the *original, uncropped* frame -- zero
    everywhere outside the crop, since a box-crop-trained model never saw
    anything outside it.

    This matters for comparing against a model evaluated on full frames:
    scoring the pointing game directly on the crop would inflate the
    score mechanically (the box now covers most of a much smaller frame)
    regardless of whether the model actually learned to attend to the
    animal. Keeping the canvas the size of the original frame keeps the
    box the same true fraction of the scoring area either way, so the two
    models' pointing-game scores stay comparable.
    """
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    bounds = crop_bounds_from_bbox(bbox, padding)
    if bounds is None:
        input_tensor = DEFAULT_TRANSFORM(image).unsqueeze(0).to(cam_builder.device)
        return cam_builder(input_tensor=input_tensor, targets=[BinaryClassifierOutputTarget(1)])[0]

    x0, y0, x1, y1 = bounds
    img_w, img_h = image.size
    left, top = int(x0 * img_w), int(y0 * img_h)
    right, bottom = int(x1 * img_w), int(y1 * img_h)
    cropped = image.crop((left, top, right, bottom))

    input_tensor = DEFAULT_TRANSFORM(cropped).unsqueeze(0).to(cam_builder.device)
    crop_cam = cam_builder(input_tensor=input_tensor, targets=[BinaryClassifierOutputTarget(1)])[0]

    cx0, cy0 = int(x0 * canvas_size), int(y0 * canvas_size)
    cx1, cy1 = int(x1 * canvas_size), int(y1 * canvas_size)
    cx1, cy1 = max(cx1, cx0 + 1), max(cy1, cy0 + 1)
    resized = np.array(Image.fromarray(crop_cam, mode="F").resize((cx1 - cx0, cy1 - cy0), Image.BILINEAR))
    canvas[cy0:cy1, cx0:cx1] = resized
    return canvas


def compute_and_save_boxcrop_heatmap(
    cam_builder: GradCAMPlusPlus,
    image: Image.Image,
    bbox: list[float],
    padding: float,
    out_path: Path,
    title: str,
) -> Path:
    """Saves a heatmap overlay on the crop the box-crop model actually saw
    (rather than the full frame, which would mostly show black padding
    from compute_boxcrop_cam's zeroed canvas) -- falls back to
    compute_and_save_heatmap on the full frame when there's no
    trustworthy box, matching training's own fallback."""
    bounds = crop_bounds_from_bbox(bbox, padding)
    if bounds is None:
        return compute_and_save_heatmap(cam_builder, image, out_path, title)
    x0, y0, x1, y1 = bounds
    img_w, img_h = image.size
    left, top = int(x0 * img_w), int(y0 * img_h)
    right, bottom = int(x1 * img_w), int(y1 * img_h)
    cropped = image.crop((left, top, right, bottom))
    return compute_and_save_heatmap(cam_builder, cropped, out_path, title)


def _load_false_positives(
    tag: str, manifest: pd.DataFrame, predictions_path: str | None = None
) -> pd.DataFrame:
    """The model's predictions joined back against the manifest for each
    image's MegaDetector box, filtered down to false positives (a bird
    the model called a mammal)."""
    predictions_path = predictions_path or f"data/predictions/test_predictions_{tag}.csv"
    predictions = pd.read_csv(predictions_path)
    false_positives = predictions[
        (predictions["true_label"] == 0) & (predictions["pred_label"] == 1)
    ].copy()

    boxes = manifest[["image_id", "bbox", "has_box"]]
    false_positives = false_positives.merge(boxes, on="image_id", how="left")
    false_positives["bbox"] = false_positives["bbox"].apply(
        lambda b: json.loads(b) if isinstance(b, str) and b else None
    )
    return false_positives


def _seen_site_mask(false_positives: pd.DataFrame, manifest: pd.DataFrame) -> pd.Series:
    warm_sites = set(manifest[manifest["season"].isin(WARM_SEASONS)]["location"])
    return false_positives["site_id"].isin(warm_sites)


# The four cells of the confusion matrix, named by what actually happened
# rather than by TP/FP/TN/FN -- easier to keep straight when reading the
# audit's output. Used by run_attention_audit() to check whether the
# background-reliance seen in false positives also shows up when the
# model gets the answer right.
OUTCOME_DEFINITIONS = {
    "bird_correct": lambda df: (df["true_label"] == 0) & (df["pred_label"] == 0),
    "mammal_correct": lambda df: (df["true_label"] == 1) & (df["pred_label"] == 1),
    "false_positive": lambda df: (df["true_label"] == 0) & (df["pred_label"] == 1),  # bird -> mammal
    "false_negative": lambda df: (df["true_label"] == 1) & (df["pred_label"] == 0),  # mammal -> bird
}


def run_gradcam_analysis(
    tag: str,
    architecture: str = DEFAULT_ARCHITECTURE,
    test_split: str = "test",
    checkpoint_path: str | None = None,
    manifest_path: str | None = None,
    shards_dir: str | None = None,
    predictions_path: str | None = None,
    n_examples: int = N_EXAMPLE_HEATMAPS,
) -> dict:
    checkpoint_path = checkpoint_path or f"models/checkpoint_{tag}.pt"
    out_dir = Path("metrics") / tag / "xai"
    heatmap_dir = out_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = _load_yaml(DEFAULT_DATA_CONFIG)["data"]
    manifest_path = manifest_path or data_cfg["manifest_path"]
    shards_dir = shards_dir or data_cfg["shards_dir"]

    manifest = pd.read_csv(manifest_path)
    false_positives = _load_false_positives(tag, manifest, predictions_path)
    if len(false_positives) == 0:
        raise RuntimeError(f"No false positives found for tag={tag!r} -- nothing to explain")

    if test_split == "seasonal":
        false_positives["seen_site"] = _seen_site_mask(false_positives, manifest)

    device = _resolve_torch_device()
    model = load_checkpoint(checkpoint_path, architecture=architecture, device=device)

    target_layer = TARGET_LAYER_BY_ARCHITECTURE[architecture](model)
    cam_builder = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    reader = ShardReader(shards_dir)

    # Most-confidently-wrong cases first -- both because they're the most
    # interesting ones to actually look at, and so the saved example
    # heatmaps aren't just whichever rows happened to sort first.
    false_positives = false_positives.sort_values("confidence", ascending=False).reset_index(drop=True)

    scores: list[dict] = []
    saved_examples = 0
    for _, row in false_positives.iterrows():
        member_name = safe_member_name(row["image_id"])
        if member_name not in reader:
            continue

        image = Image.open(io.BytesIO(reader.read(member_name))).convert("RGB")
        input_tensor = DEFAULT_TRANSFORM(image).unsqueeze(0).to(device)

        grayscale_cam = cam_builder(input_tensor=input_tensor, targets=[BinaryClassifierOutputTarget(1)])[0]

        score = None
        if row["bbox"] is not None:
            score = pointing_game_score(grayscale_cam, row["bbox"])

        record = {"image_id": row["image_id"], "site_id": row["site_id"], "pointing_game_score": score}
        if test_split == "seasonal":
            record["seen_site"] = bool(row["seen_site"])
        scores.append(record)

        if saved_examples < n_examples:
            title = f"{row['species']} -> mammal (conf={row['confidence']:.2f})"
            compute_and_save_heatmap(cam_builder, image, heatmap_dir / member_name, title)
            saved_examples += 1

    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(out_dir / "pointing_game_scores.csv", index=False)

    scored = scores_df.dropna(subset=["pointing_game_score"])
    summary = {
        "tag": tag,
        "n_false_positives": len(false_positives),
        "n_scored": len(scored),
        "n_skipped_no_box": len(false_positives) - len(scored),
        "pointing_game_mean": float(scored["pointing_game_score"].mean()) if len(scored) else None,
        "pointing_game_median": float(scored["pointing_game_score"].median()) if len(scored) else None,
    }

    if test_split == "seasonal":
        summary["by_site_familiarity"] = {}
        for seen, label in ((True, "seen_site"), (False, "novel_site")):
            sub = scored[scored["seen_site"] == seen]
            summary["by_site_familiarity"][label] = {
                "n_scored": len(sub),
                "pointing_game_mean": float(sub["pointing_game_score"].mean()) if len(sub) else None,
                "pointing_game_median": float(sub["pointing_game_score"].median()) if len(sub) else None,
            }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    with mlflow.start_run(run_name=f"gradcam_{tag}"):
        mlflow.log_param("architecture", architecture)
        mlflow.log_param("test_split", test_split)
        mlflow.log_param("checkpoint_path", str(checkpoint_path))
        mlflow.log_metric("n_false_positives", summary["n_false_positives"])
        if summary["pointing_game_mean"] is not None:
            mlflow.log_metric("pointing_game_mean", summary["pointing_game_mean"])
            mlflow.log_metric("pointing_game_median", summary["pointing_game_median"])
        if test_split == "seasonal":
            for label, stats in summary["by_site_familiarity"].items():
                if stats["pointing_game_mean"] is not None:
                    mlflow.log_metric(f"{label}_pointing_game_mean", stats["pointing_game_mean"])
        mlflow.log_artifact(str(out_dir / "summary.json"))
        mlflow.log_artifact(str(out_dir / "pointing_game_scores.csv"))

    print(f"false positives: {summary['n_false_positives']}  (scored against a box: {summary['n_scored']})")
    print(f"pointing-game score -- mean={summary['pointing_game_mean']:.4f}  median={summary['pointing_game_median']:.4f}")
    if test_split == "seasonal":
        for label, stats in summary["by_site_familiarity"].items():
            print(f"  {label}: n={stats['n_scored']}  mean={stats['pointing_game_mean']:.4f}  median={stats['pointing_game_median']:.4f}")
    print(f"wrote {out_dir}/summary.json, pointing_game_scores.csv, and {saved_examples} example heatmaps to {heatmap_dir}")

    return summary


def run_attention_audit(
    tag: str,
    architecture: str = DEFAULT_ARCHITECTURE,
    checkpoint_path: str | None = None,
    manifest_path: str | None = None,
    shards_dir: str | None = None,
    predictions_path: str | None = None,
    n_examples_per_outcome: int = 8,
    crop_to_box: bool = False,
    box_crop_padding: float | None = None,
) -> dict:
    """Pointing-game score across the whole test set, broken down by
    outcome (see OUTCOME_DEFINITIONS) rather than false positives alone.

    Scores every test image with a box, but only re-renders and saves a
    heatmap for the worst-scoring `n_examples_per_outcome` per outcome --
    those are the interesting ones: for the *_correct outcomes, a low
    score means the model got the right answer while looking at the wrong
    thing entirely.

    `crop_to_box` must be set for any checkpoint trained by
    boxcrop_experiment.py -- it crops each image to its MegaDetector box
    before running Grad-CAM (matching what the model actually saw during
    training) and remaps the heatmap back onto the original frame before
    scoring, via compute_boxcrop_cam. Feeding a box-crop-trained model the
    full, uncropped frame it never trained on would be a distribution
    mismatch; scoring the pointing game directly on the crop instead of
    remapping it back would inflate the score mechanically, since the box
    then covers most of a much smaller frame. Rows without a trustworthy
    box (has_box False) fall back to the full frame either way, matching
    crop_to_bbox's own training-time fallback.
    """
    checkpoint_path = checkpoint_path or f"models/checkpoint_{tag}.pt"
    out_dir = Path("metrics") / tag / "xai" / "attention_audit"
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

    device = _resolve_torch_device()
    model = load_checkpoint(checkpoint_path, architecture=architecture, device=device)
    target_layer = TARGET_LAYER_BY_ARCHITECTURE[architecture](model)
    cam_builder = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    reader = ShardReader(shards_dir)

    def score_row(row) -> float | None:
        member_name = safe_member_name(row["image_id"])
        if member_name not in reader:
            return None
        image = Image.open(io.BytesIO(reader.read(member_name))).convert("RGB")
        if crop_to_box and row.get("has_box", False):
            cam = compute_boxcrop_cam(cam_builder, image, row["bbox"], box_crop_padding)
        else:
            input_tensor = DEFAULT_TRANSFORM(image).unsqueeze(0).to(device)
            cam = cam_builder(input_tensor=input_tensor, targets=[BinaryClassifierOutputTarget(1)])[0]
        return pointing_game_score(cam, row["bbox"])

    all_scored = []
    summary: dict = {"tag": tag}
    for outcome, mask_fn in OUTCOME_DEFINITIONS.items():
        subset = joined[mask_fn(joined)].copy()
        subset["pointing_game_score"] = [score_row(row) for _, row in subset.iterrows()]
        subset = subset.dropna(subset=["pointing_game_score"])
        subset["outcome"] = outcome
        all_scored.append(subset[["image_id", "species", "project", "location", "confidence", "outcome", "pointing_game_score"]])

        summary[outcome] = {
            "n_scored": len(subset),
            "pointing_game_mean": float(subset["pointing_game_score"].mean()) if len(subset) else None,
            "pointing_game_median": float(subset["pointing_game_score"].median()) if len(subset) else None,
        }
        print(
            f"{outcome}: n={summary[outcome]['n_scored']}  "
            f"mean={summary[outcome]['pointing_game_mean']:.4f}  median={summary[outcome]['pointing_game_median']:.4f}"
        )

        worst = subset.sort_values("pointing_game_score").head(n_examples_per_outcome)
        for _, row in worst.iterrows():
            member_name = safe_member_name(row["image_id"])
            image = Image.open(io.BytesIO(reader.read(member_name))).convert("RGB")
            title = f"{row['species']} ({outcome}, score={row['pointing_game_score']:.2f})"
            out_path = heatmap_dir / f"{outcome}__{member_name}"
            if crop_to_box and row.get("has_box", False):
                compute_and_save_boxcrop_heatmap(cam_builder, image, row["bbox"], box_crop_padding, out_path, title)
            else:
                compute_and_save_heatmap(cam_builder, image, out_path, title)

    scores_df = pd.concat(all_scored, ignore_index=True)
    scores_df.to_csv(out_dir / "attention_audit_scores.csv", index=False)

    # Does the zero-attention pattern cluster by project/site? A concrete
    # check for whether a recurring background object (like the white
    # card found behind one myna during manual testing) is a site-level
    # dataset artifact rather than a one-off.
    zero_attention = scores_df[scores_df["pointing_game_score"] == 0.0]
    by_project = zero_attention["project"].value_counts()
    summary["zero_attention_count_by_project"] = by_project.head(15).to_dict()
    summary["crop_to_box"] = crop_to_box

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    with mlflow.start_run(run_name=f"attention_audit_{tag}"):
        mlflow.log_param("architecture", architecture)
        mlflow.log_param("checkpoint_path", str(checkpoint_path))
        mlflow.log_param("crop_to_box", crop_to_box)
        for outcome in OUTCOME_DEFINITIONS:
            if summary[outcome]["pointing_game_mean"] is not None:
                mlflow.log_metric(f"{outcome}_pointing_game_mean", summary[outcome]["pointing_game_mean"])
                mlflow.log_metric(f"{outcome}_pointing_game_median", summary[outcome]["pointing_game_median"])
        mlflow.log_artifact(str(out_dir / "summary.json"))
        mlflow.log_artifact(str(out_dir / "attention_audit_scores.csv"))

    print(f"wrote {out_dir}/summary.json, attention_audit_scores.csv, and heatmaps to {heatmap_dir}")
    print("zero-attention count by project (top 15):")
    print(by_project.head(15))

    return summary


def main() -> None:
    import sys

    argv = sys.argv[1:]
    crop_to_box = "--crop-to-box" in argv
    argv = [a for a in argv if a != "--crop-to-box"]

    if len(argv) < 1:
        raise SystemExit(
            "usage: python -m src.xai.gradcam <tag> [architecture] [test_split]\n"
            "       python -m src.xai.gradcam <tag> --audit [architecture] [--crop-to-box]"
        )
    tag = argv[0]
    if len(argv) > 1 and argv[1] == "--audit":
        architecture = argv[2] if len(argv) > 2 else DEFAULT_ARCHITECTURE
        run_attention_audit(tag=tag, architecture=architecture, crop_to_box=crop_to_box)
        return

    architecture = argv[1] if len(argv) > 1 else DEFAULT_ARCHITECTURE
    test_split = argv[2] if len(argv) > 2 else "test"
    run_gradcam_analysis(tag=tag, architecture=architecture, test_split=test_split)


if __name__ == "__main__":
    main()
