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

Run: python -m src.xai.gradcam <tag> [architecture] [test_split]
     e.g. python -m src.xai.gradcam efficientnet_b3_seasonal efficientnet_b3 seasonal
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
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from src.models.classifier import WildlifeClassifier
from src.training.dataset import DEFAULT_TRANSFORM, ShardReader, safe_member_name
from src.training.evaluate import DEFAULT_DATA_CONFIG, _load_yaml, _resolve_torch_device

DEFAULT_ARCHITECTURE = "efficientnet_b3"
WARM_SEASONS = ("spring", "summer")
N_EXAMPLE_HEATMAPS = 16

# Grad-CAM needs a specific convolutional layer to read gradients from --
# by convention, the last one before the classifier head, since that's
# where spatial detail and learned features are both still present.
_TARGET_LAYER_BY_ARCHITECTURE = {
    "efficientnet_b3": lambda model: model.backbone.features[-1],
    "resnet50": lambda model: model.backbone.layer4[-1],
}


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
    model = WildlifeClassifier(architecture=architecture)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    target_layer = _TARGET_LAYER_BY_ARCHITECTURE[architecture](model)
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
            display_image = np.asarray(image.resize((300, 300)), dtype=np.float32) / 255.0
            overlay = show_cam_on_image(display_image, grayscale_cam, use_rgb=True)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(overlay)
            ax.set_title(f"{row['species']} -> mammal (conf={row['confidence']:.2f})", fontsize=9)
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(heatmap_dir / f"{member_name}", dpi=120)
            plt.close(fig)
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


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m src.xai.gradcam <tag> [architecture] [test_split]")
    tag = sys.argv[1]
    architecture = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ARCHITECTURE
    test_split = sys.argv[3] if len(sys.argv) > 3 else "test"
    run_gradcam_analysis(tag=tag, architecture=architecture, test_split=test_split)


if __name__ == "__main__":
    main()
