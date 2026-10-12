"""
Evaluate the box-crop models on the external dataset the way they are
actually served: each image cropped to its MegaDetector box
(detect_external_boxes.py) exactly like training (crop_to_bbox, padding
0.15) and the live API, or the full frame when there is no box.

evaluate_external.py feeds full frames only, which is right for the
standard model but not for a box-crop model -- on full frames the
box-crop model's accuracy on the external set collapses to ~48%, a
mismatch artefact rather than real drift.

Writes:
  - data/predictions/external_predictions_<tag>_cropped.csv
  - metrics/external_birds_and_mammals/<tag>_cropped/eval.json
      (all 1,865 images and the 152-image held-out test split, at 0.5
      and 0.95; only the test split is a fair number for the retrained
      model, which trained on part of the rest)
  - metrics/drift/efficientnet_b3_boxcrop_cropped_vs_external/report.json
      (confidence, concept and XAI-centroid drift for the box-crop model,
      using src/monitoring/drift.py's own functions, cropped on both sides)

Run: python -m src.external_data.evaluate_external_cropped
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from torch.utils.data import DataLoader, Dataset

from src.external_data.detect_external_boxes import DEFAULT_BOXES_PATH
from src.external_data.evaluate_external import DEFAULT_IMAGES_ROOT, DEFAULT_MANIFEST_PATH
from src.models.classifier import load_checkpoint
from src.monitoring.drift import (
    compute_attention_centroid,
    compute_centroid_drift,
    compute_concept_drift,
    compute_confidence_drift,
)
from src.training.dataset import DEFAULT_TRANSFORM, ShardReader, crop_to_bbox, safe_member_name
from src.training.evaluate import _resolve_torch_device
from src.xai.gradcam import TARGET_LAYER_BY_ARCHITECTURE, compute_boxcrop_cam

BOX_CROP_PADDING = 0.15
THRESHOLDS = (0.5, 0.95)
TAGS = ("efficientnet_b3_boxcrop", "efficientnet_b3_boxcrop_retrained")
DRIFT_MODEL_TAG = "efficientnet_b3_boxcrop"
DRIFT_BASELINE_PREDICTIONS = "data/predictions/test_predictions_efficientnet_b3_boxcrop.csv"
DRIFT_TAG = "efficientnet_b3_boxcrop_cropped_vs_external"
N_CENTROID_SAMPLES = 100


def load_external_with_boxes(manifest_path: str, boxes_path: str) -> pd.DataFrame:
    """External manifest with its (empty) has_box/bbox columns replaced by
    the MegaDetector results. Raises if any image has no detection row,
    so a partial boxes file can never silently become full-frame input."""
    manifest = pd.read_csv(manifest_path).drop(columns=["has_box", "bbox"], errors="ignore")
    boxes = pd.read_csv(boxes_path, keep_default_na=False)
    boxes["has_box"] = boxes["has_box"].astype(str) == "True"
    merged = manifest.merge(boxes, on="image_id", how="left", validate="one_to_one")
    missing = merged["has_box"].isna().sum()
    if missing:
        raise ValueError(f"{missing} external images have no MegaDetector result in {boxes_path}")
    return merged


class CroppedImageDataset(Dataset):
    """Crops each image with training's own crop_to_bbox before the
    standard transform -- full frame when has_box is False."""

    def __init__(self, df: pd.DataFrame, images_root: str, padding: float = BOX_CROP_PADDING) -> None:
        self.df = df.reset_index(drop=True)
        self.images_root = Path(images_root)
        self.padding = padding

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.df.iloc[idx]
        image = Image.open(self.images_root / row["file_name"]).convert("RGB")
        return DEFAULT_TRANSFORM(crop_to_bbox(image, row, padding=self.padding))


def threshold_metrics(y_true, y_prob, threshold: float) -> dict:
    y_true = pd.Series(y_true).to_numpy()
    pred = (pd.Series(y_prob).to_numpy() >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return {
        "n_images": len(y_true),
        "accuracy": (tp + tn) / len(y_true),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "false_positives": fp,
        "false_negatives": fn,
    }


def predict_cropped(df: pd.DataFrame, tag: str, device: torch.device) -> tuple[pd.DataFrame, torch.nn.Module]:
    model = load_checkpoint(f"models/checkpoint_{tag}.pt", architecture="efficientnet_b3", device=device)
    loader = DataLoader(CroppedImageDataset(df, DEFAULT_IMAGES_ROOT), batch_size=32, shuffle=False, num_workers=0)
    probabilities: list[float] = []
    with torch.no_grad():
        for x in loader:
            probabilities.extend(torch.sigmoid(model(x.to(device))).cpu().flatten().tolist())
    predictions = df[["image_id", "file_name", "label", "split", "event_id", "has_box", "bbox"]].copy()
    predictions["probability"] = probabilities
    predictions["confidence"] = predictions["probability"].apply(lambda p: max(p, 1 - p))
    return predictions, model


def _centroid(cam_builder, image: Image.Image, has_box: bool, bbox_json, device) -> tuple[float, float]:
    """Attention centroid on the input the box-crop model actually sees:
    the crop, remapped onto the full frame (compute_boxcrop_cam), or the
    full frame itself when there is no box."""
    if has_box and isinstance(bbox_json, str) and bbox_json:
        cam = compute_boxcrop_cam(cam_builder, image, json.loads(bbox_json), BOX_CROP_PADDING)
    else:
        tensor = DEFAULT_TRANSFORM(image).unsqueeze(0).to(device)
        cam = cam_builder(input_tensor=tensor, targets=[BinaryClassifierOutputTarget(1)])[0]
    return compute_attention_centroid(cam)


def run_cropped_drift_report(current: pd.DataFrame, model: torch.nn.Module, device: torch.device) -> dict:
    baseline = pd.read_csv(DRIFT_BASELINE_PREDICTIONS)
    confidence_result = compute_confidence_drift(baseline["confidence"], current["confidence"])
    concept_result = compute_concept_drift(
        baseline["true_label"].to_numpy(),
        baseline["probability"].to_numpy(),
        current["label"].to_numpy(),
        current["probability"].to_numpy(),
        threshold=0.5,
    )

    cam_builder = GradCAMPlusPlus(model=model, target_layers=[TARGET_LAYER_BY_ARCHITECTURE["efficientnet_b3"](model)])
    main_manifest = pd.read_csv("data/manifest.csv").set_index("image_id")
    reader = ShardReader("data/shards")
    baseline_centroids = []
    for image_id in baseline.sample(n=N_CENTROID_SAMPLES, random_state=42)["image_id"]:
        member = safe_member_name(image_id)
        if member not in reader:
            continue
        image = Image.open(io.BytesIO(reader.read(member))).convert("RGB")
        row = main_manifest.loc[image_id]
        baseline_centroids.append(_centroid(cam_builder, image, bool(row["has_box"]), row["bbox"], device))
    current_centroids = []
    for row in current.sample(n=N_CENTROID_SAMPLES, random_state=42).itertuples(index=False):
        image = Image.open(Path(DEFAULT_IMAGES_ROOT) / row.file_name).convert("RGB")
        current_centroids.append(_centroid(cam_builder, image, bool(row.has_box), row.bbox, device))

    return {
        "tag": DRIFT_TAG,
        "model": DRIFT_MODEL_TAG,
        "external_input": f"MegaDetector-cropped (conf 0.2, padding {BOX_CROP_PADDING}), full frame when no box",
        "confidence_drift": confidence_result,
        "concept_drift": concept_result,
        "xai_centroid_drift": compute_centroid_drift(baseline_centroids, current_centroids),
        "retrain_recommended": bool(concept_result["drifted"]),
        "retrain_rationale": (
            "concept_drift.drifted is the deciding signal (labelled correctness actually degraded beyond "
            "tolerance); confidence_drift and xai_centroid_drift are reported as corroborating context."
        ),
    }


def main() -> None:
    device = _resolve_torch_device()
    df = load_external_with_boxes(DEFAULT_MANIFEST_PATH, DEFAULT_BOXES_PATH)
    print(f"MegaDetector coverage: {df.has_box.mean():.1%} ({int(df.has_box.sum())}/{len(df)})")

    models = {}
    for tag in TAGS:
        predictions, models[tag] = predict_cropped(df, tag, device)
        predictions_path = Path("data/predictions") / f"external_predictions_{tag}_cropped.csv"
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(predictions_path, index=False)

        result = {"tag": f"{tag}_cropped", "md_coverage": float(df.has_box.mean())}
        for subset, part in (("all", predictions), ("test", predictions[predictions.split == "test"])):
            for t in THRESHOLDS:
                result[f"{subset}@{t}"] = threshold_metrics(part["label"], part["probability"], t)
                m = result[f"{subset}@{t}"]
                print(f"{tag:36s} {subset:4s} t={t:<4} acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
                      f"rec={m['recall']:.4f} FP={m['false_positives']} FN={m['false_negatives']}")
        out_dir = Path("metrics/external_birds_and_mammals") / f"{tag}_cropped"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval.json").write_text(json.dumps(result, indent=2))

    report = run_cropped_drift_report(
        pd.read_csv(Path("data/predictions") / f"external_predictions_{DRIFT_MODEL_TAG}_cropped.csv"),
        models[DRIFT_MODEL_TAG],
        device,
    )
    out_dir = Path("metrics/drift") / DRIFT_TAG
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"wrote {out_dir}/report.json  retrain_recommended={report['retrain_recommended']}")


if __name__ == "__main__":
    main()
