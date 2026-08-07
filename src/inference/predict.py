"""
Core inference logic — model loading and single/batch prediction.

Design:
  - Model + configs are loaded once, at API startup, via
    build_serving_context() (called from src.inference.app's lifespan) --
    not reloaded per request.
  - image_path is any filesystem path readable by PIL on the machine the
    API process runs on (an extracted-from-shards path, or a freshly
    uploaded temp file) -- no ShardReader involvement, since ShardReader
    pays a full-shard-index-scan cost on every construction that a live,
    single-image request path can't afford to repeat per call.
  - The label is decided from a single deterministic forward pass (dropout
    off) -- exactly the computation evaluate.py's reported accuracy/
    precision/recall are measured against, so a live label always means
    what the formal evaluation numbers say it means. A *separate* MC
    Dropout pass (N forward passes, dropout active, N from
    configs/model/*.yaml via serving.yaml's model_config_path) estimates
    epistemic uncertainty only -- its mean probability is discarded, not
    used for the label. Earlier iterations of this endpoint mistakenly
    used that stochastic mean for the label too; on a classifier head
    that's just one Dropout -> Linear layer, dropping 30% of pooled
    features per pass can swing a single linear layer enough that
    averaging 20 passes flips images the deterministic pass called
    correctly and confidently. MC Dropout is still useful -- just as an
    uncertainty signal alongside the label, not as the label itself.
    Temperature scaling is explicitly out of scope for this pass -- this
    uses raw, uncalibrated MC Dropout std, a known limitation rather than
    a bug.
  - label = 1 if deterministic probability >= per-site threshold else 0
    (thresholds from configs/inference/threshold_config.yaml, keyed by
    site_id).
  - requires_review = std_prob > uncertainty_review_cutoff -- a distinct
    concept from the decision threshold above, pulled from its own config
    key on purpose.
  - Every /predict call gets logged twice: an MLflow run (params + a CSV
    artifact of per-image results), and an appended row per image into
    data/predictions/serving_log.csv, standing in for the not-yet-
    provisioned PostgreSQL prediction log the review dashboard will
    eventually read from.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus

from src.inference.schemas import ImageItem, PredictRequest, PredictResponse, PredictionResult
from src.models.classifier import WildlifeClassifier, load_checkpoint
from src.training.dataset import DEFAULT_TRANSFORM
from src.training.evaluate import _load_yaml, _resolve_torch_device
from src.xai.gradcam import TARGET_LAYER_BY_ARCHITECTURE, compute_and_save_heatmap

DEFAULT_SERVING_CONFIG = "configs/inference/serving.yaml"
DEFAULT_THRESHOLD_CONFIG = "configs/inference/threshold_config.yaml"


@dataclass
class ServingContext:
    """Everything a /predict or /model-info request needs, built once at
    API startup and handed to every request via app.state.ctx."""

    model: WildlifeClassifier
    device: torch.device
    architecture: str
    tag: str
    checkpoint_path: str
    dropout_rate: float
    mc_dropout_passes: int
    threshold_cfg: dict
    cam_builder: GradCAMPlusPlus
    target_layer: nn.Module
    heatmap_dir: str
    prediction_log_path: str


def build_serving_context(
    serving_config_path: str = DEFAULT_SERVING_CONFIG,
    threshold_config_path: str = DEFAULT_THRESHOLD_CONFIG,
) -> ServingContext:
    serving_cfg = _load_yaml(serving_config_path)["serving"]
    threshold_cfg = _load_yaml(threshold_config_path)
    model_cfg = _load_yaml(serving_cfg["model_config_path"])["model"]

    device = _resolve_torch_device()
    model = load_checkpoint(
        serving_cfg["checkpoint_path"],
        architecture=serving_cfg["architecture"],
        device=device,
        dropout_rate=model_cfg["dropout_rate"],
    )

    target_layer = TARGET_LAYER_BY_ARCHITECTURE[serving_cfg["architecture"]](model)
    cam_builder = GradCAMPlusPlus(model=model, target_layers=[target_layer])

    return ServingContext(
        model=model,
        device=device,
        architecture=serving_cfg["architecture"],
        tag=serving_cfg["tag"],
        checkpoint_path=serving_cfg["checkpoint_path"],
        dropout_rate=model_cfg["dropout_rate"],
        mc_dropout_passes=model_cfg["mc_dropout_passes"],
        threshold_cfg=threshold_cfg,
        cam_builder=cam_builder,
        target_layer=target_layer,
        heatmap_dir=serving_cfg.get("heatmap_dir", "data/predictions/heatmaps"),
        prediction_log_path=serving_cfg.get("prediction_log_path", "data/predictions/serving_log.csv"),
    )


def _resolve_threshold(site_id: str | None, threshold_cfg: dict) -> float:
    """Per-site decision threshold, falling back to thresholds.default when
    site_id is None or not a recognised zone key."""
    thresholds = threshold_cfg["thresholds"]
    if site_id is None:
        return thresholds["default"]
    return thresholds.get(site_id, thresholds["default"])


def _needs_review(uncertainty: float, cutoff: float) -> bool:
    """True when MC Dropout std strictly exceeds the review cutoff -- a
    distinct question from "which side of the decision threshold did this
    land on", deliberately not reusing that threshold here."""
    return uncertainty > cutoff


def predict_one(
    image_path: str,
    threshold: float,
    ctx: ServingContext,
    return_heatmap: bool = False,
) -> PredictionResult:
    image = Image.open(image_path).convert("RGB")
    input_tensor = DEFAULT_TRANSFORM(image).unsqueeze(0).to(ctx.device)

    # Deterministic pass (dropout off) decides the label -- this is the
    # exact computation evaluate.py's reported metrics are measured
    # against, so the live label always agrees with what those numbers
    # mean.
    deterministic_prob, _ = ctx.model.predict_proba(input_tensor, mc_dropout_passes=1)
    probability = float(deterministic_prob.item())

    # A separate MC Dropout pass estimates epistemic uncertainty only; its
    # mean is discarded on purpose -- see the module docstring for why it
    # doesn't get to decide the label.
    _, std_prob = ctx.model.predict_proba(input_tensor, mc_dropout_passes=ctx.mc_dropout_passes)
    uncertainty = float(std_prob.item())

    label = 1 if probability >= threshold else 0
    # confidence in the predicted label, not raw P(mammal) -- matches
    # evaluate.py's convention exactly (see its docstring) so a bird
    # prediction with probability=0.004 reports confidence=0.996, not
    # 0.004, which would read as "barely thinks it's a bird" when it's
    # actually the opposite.
    confidence = max(probability, 1 - probability)
    requires_review = _needs_review(uncertainty, ctx.threshold_cfg["uncertainty_review_cutoff"])

    heatmap_path: str | None = None
    if return_heatmap:
        Path(ctx.heatmap_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(ctx.heatmap_dir) / f"{Path(image_path).stem}_heatmap.png"
        title = f"{Path(image_path).name} -> {'mammal' if label else 'bird'} (p={probability:.2f})"
        heatmap_path = str(compute_and_save_heatmap(ctx.cam_builder, image, out_path, title))

    return PredictionResult(
        image_path=image_path,
        label=label,
        confidence=confidence,
        uncertainty=uncertainty,
        requires_review=requires_review,
        heatmap_path=heatmap_path,
    )


def predict_batch(
    request: PredictRequest,
    ctx: ServingContext,
    log_mlflow: bool = True,
    log_csv: bool = True,
) -> PredictResponse:
    results: list[PredictionResult] = []
    resolved_thresholds: list[float] = []
    for item in request.images:
        threshold = _resolve_threshold(item.site_id, ctx.threshold_cfg)
        resolved_thresholds.append(threshold)
        results.append(
            predict_one(
                image_path=item.image_path,
                threshold=threshold,
                ctx=ctx,
                return_heatmap=request.return_heatmaps,
            )
        )

    # PredictResponse.threshold_used is a single float for the whole batch,
    # but thresholds are resolved per image -- report the first image's
    # resolved threshold. A batch mixing multiple site_ids with different
    # thresholds is a known simplification of the existing schema; every
    # real caller in this slice (the Live Demo view) sends one image per
    # request, so it doesn't bite in practice yet.
    threshold_used = resolved_thresholds[0] if resolved_thresholds else ctx.threshold_cfg["thresholds"]["default"]

    if log_csv:
        _append_serving_log(request.images, results, threshold_used, ctx)
    if log_mlflow:
        _log_predict_run(results, threshold_used, ctx)

    return PredictResponse(model_version=ctx.tag, threshold_used=threshold_used, predictions=results)


def _append_serving_log(
    items: list[ImageItem],
    results: list[PredictionResult],
    threshold_used: float,
    ctx: ServingContext,
) -> None:
    """Appends one row per prediction to ctx.prediction_log_path (creating
    it with a header if it doesn't exist yet). Extra columns beyond the
    minimal prediction fields -- timestamp, site_id, threshold_used,
    model_tag -- are here because a future review queue will need to
    sort/filter this log by time and by which model version produced each
    row."""
    csv_log_path = Path(ctx.prediction_log_path)
    csv_log_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    rows = pd.DataFrame(
        [
            {
                "timestamp": now,
                "image_path": r.image_path,
                "site_id": item.site_id,
                "probability": r.confidence,
                "uncertainty": r.uncertainty,
                "label": r.label,
                "requires_review": r.requires_review,
                "threshold_used": threshold_used,
                "model_tag": ctx.tag,
            }
            for item, r in zip(items, results)
        ]
    )
    write_header = not csv_log_path.exists()
    rows.to_csv(csv_log_path, mode="a", header=write_header, index=False)


def _log_predict_run(results: list[PredictionResult], threshold_used: float, ctx: ServingContext) -> None:
    with mlflow.start_run(run_name=f"predict_{ctx.tag}"):
        mlflow.log_param("architecture", ctx.architecture)
        mlflow.log_param("checkpoint_path", ctx.checkpoint_path)
        mlflow.log_param("tag", ctx.tag)
        mlflow.log_param("mc_dropout_passes", ctx.mc_dropout_passes)
        mlflow.log_param("threshold_used", threshold_used)
        mlflow.log_metric("n_images", len(results))
        mlflow.log_metric("n_requires_review", sum(r.requires_review for r in results))
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "predict_results.csv"
            pd.DataFrame(
                [
                    {
                        "image_path": r.image_path,
                        "probability": r.confidence,
                        "uncertainty": r.uncertainty,
                        "label": r.label,
                        "requires_review": r.requires_review,
                    }
                    for r in results
                ]
            ).to_csv(artifact_path, index=False)
            mlflow.log_artifact(str(artifact_path))
