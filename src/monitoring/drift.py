"""
Drift detection comparing a reference (baseline) prediction distribution
against a current one -- built specifically to compare the original
LILA-derived test set against the external "birds_and_mammals" set
(src/external_data/), the real distribution shift this project has on
hand, rather than a simulated one.

Three signals, deliberately different in what they check:

  1. Confidence drift (compute_confidence_drift) -- does the raw
     distribution of predicted confidence shift, regardless of whether
     labels are even available? Uses evidently's DataDriftPreset
     (Wasserstein distance between the two confidence distributions).
     This is the cheapest signal and the only one of the three that
     doesn't need ground truth -- in a real deployment without labels
     yet, this is the one that could still run.

  2. Concept drift (compute_concept_drift) -- does the model's actual
     correctness degrade, and does the shape of its per-class score
     distribution change? Two deliberate design choices here, both
     driven by a real mistake this project already made and corrected:
       - Metrics are recomputed at a FIXED reference threshold (default
         0.5) for both reference and current data, not whatever
         threshold happens to be deployed. Comparing already-thresholded
         metrics computed at different thresholds conflates "the
         threshold doesn't transfer to a new distribution" with "the
         model's real signal got worse" -- exactly the confound found
         when the external set was first evaluated at the live 0.95
         threshold and looked far worse than a matched-threshold
         comparison showed it actually was.
       - scipy.stats.ks_2samp compares the probability distribution
         WITHIN each true class separately (P(mammal) among true
         mammals, and separately among true birds) rather than pooled --
         a pooled comparison would be dominated by any change in class
         balance between the two datasets rather than genuine concept
         drift.

  3. XAI-centroid drift (compute_centroid_drift / the sampling
     functions around it) -- does WHERE the model looks shift, using
     Grad-CAM++ attention centroids (see gradcam.py -- same
     BinaryClassifierOutputTarget(1)/mammal-target convention used
     throughout this project's XAI tooling, for consistency with the
     existing attention-audit results). Sampled rather than run over
     the full dataset (Grad-CAM is far more expensive than a forward
     pass) -- see sample_centroids_from_shards/_external.

Retrain recommendation: concept drift (the only signal grounded in
actual labelled correctness) is the deciding signal; confidence drift
and XAI-centroid drift are reported as corroborating/diagnostic context,
not tie-breakers -- matching how real drift-monitoring setups generally
treat unlabelled signals as early warning rather than the sole trigger
when labelled evaluation is available at all.

Run: python -m src.monitoring.drift <baseline_predictions_csv> <current_predictions_csv> [options]
     e.g. python -m src.monitoring.drift \
       data/predictions/test_predictions_efficientnet_b3.csv \
       data/predictions/external_predictions_efficientnet_b3.csv \
       --tag efficientnet_b3_vs_external --push-to-datadog
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.models.classifier import load_checkpoint
from src.training.dataset import DEFAULT_TRANSFORM, ShardReader, safe_member_name
from src.training.evaluate import _resolve_torch_device
from src.xai.gradcam import TARGET_LAYER_BY_ARCHITECTURE

DEFAULT_ARCHITECTURE = "efficientnet_b3"
DEFAULT_CONFIDENCE_DRIFT_THRESHOLD = 0.1
DEFAULT_CONCEPT_METRIC_TOLERANCE = 0.03
DEFAULT_ALPHA = 0.05
DEFAULT_N_CENTROID_SAMPLES = 100


# ---------------------------------------------------------------------------
# 1. Confidence drift
# ---------------------------------------------------------------------------


def compute_confidence_drift(
    baseline_confidence: pd.Series,
    current_confidence: pd.Series,
    drift_threshold: float = DEFAULT_CONFIDENCE_DRIFT_THRESHOLD,
) -> dict:
    ref_ds = Dataset.from_pandas(pd.DataFrame({"confidence": baseline_confidence.to_numpy()}))
    cur_ds = Dataset.from_pandas(pd.DataFrame({"confidence": current_confidence.to_numpy()}))

    # num_method is forced explicitly -- evidently's own default auto-selects
    # between several stat tests based on the data's sample size/shape
    # (verified directly: the exact same code picked "wasserstein" on one
    # dataset and silently switched to "K-S p_value" on another, which
    # has an OPPOSITE value-vs-threshold comparison direction, i.e. lower
    # is more drifted, not higher). Forcing one method keeps this
    # function's return value on a single, known scale.
    report = Report([DataDriftPreset(columns=["confidence"], num_method="wasserstein", threshold=drift_threshold)])
    snapshot = report.run(current_data=cur_ds, reference_data=ref_ds)
    result = snapshot.dict()

    value_drift = next(m for m in result["metrics"] if m["metric_name"].startswith("ValueDrift"))
    drifted_count = next(m for m in result["metrics"] if m["metric_name"].startswith("DriftedColumnsCount"))

    return {
        "drift_score": float(value_drift["value"]),
        "method": "wasserstein_normed",
        "threshold": drift_threshold,
        "drifted": bool(drifted_count["value"]["count"] >= 1),
    }


# ---------------------------------------------------------------------------
# 2. Concept drift
# ---------------------------------------------------------------------------


def compute_concept_drift(
    baseline_true: np.ndarray,
    baseline_prob: np.ndarray,
    current_true: np.ndarray,
    current_prob: np.ndarray,
    threshold: float = 0.5,
    metric_drop_tolerance: float = DEFAULT_CONCEPT_METRIC_TOLERANCE,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    def _metrics(y_true, y_prob) -> dict:
        y_pred = (y_prob >= threshold).astype(int)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

    baseline_metrics = _metrics(baseline_true, baseline_prob)
    current_metrics = _metrics(current_true, current_prob)
    deltas = {k: current_metrics[k] - baseline_metrics[k] for k in baseline_metrics}

    def _ks_for_class(label: int) -> dict:
        b = baseline_prob[baseline_true == label]
        c = current_prob[current_true == label]
        if len(b) < 2 or len(c) < 2:
            return {"statistic": None, "p_value": None, "n_baseline": len(b), "n_current": len(c)}
        stat, p = ks_2samp(b, c)
        return {"statistic": float(stat), "p_value": float(p), "n_baseline": len(b), "n_current": len(c)}

    score_shift = {"mammal": _ks_for_class(1), "bird": _ks_for_class(0)}

    drifted = (deltas["accuracy"] < -metric_drop_tolerance) or (deltas["f1"] < -metric_drop_tolerance)

    return {
        "threshold_used": threshold,
        "baseline_metrics": baseline_metrics,
        "current_metrics": current_metrics,
        "deltas": deltas,
        "score_distribution_shift": score_shift,
        "metric_drop_tolerance": metric_drop_tolerance,
        "drifted": drifted,
    }


# ---------------------------------------------------------------------------
# 3. XAI-centroid drift
# ---------------------------------------------------------------------------


def compute_attention_centroid(cam: np.ndarray) -> tuple[float, float]:
    """Weighted centroid of a Grad-CAM heatmap's attention mass, as
    (x, y) normalised 0-1 against the heatmap's own width/height. (0.5,
    0.5) -- the frame's centre -- is returned for an all-zero heatmap
    rather than dividing by zero; this only happens for the same
    zero-attention cases gradcam.py's audit already reports separately
    via pointing_game_score, so it isn't silently hidden here either.

    Pixel indices are offset by +0.5 before normalising, so an index
    represents the CENTRE of that pixel's cell, not its top-left corner
    -- without this, uniform attention across the whole frame centres
    at (0.45, 0.45) instead of (0.5, 0.5), a small but real systematic
    bias toward the top-left corner (caught by a unit test comparing a
    single top-left hotspot against its expected (0.05, 0.05), not the
    (0.0, 0.0) the unoffset version actually returned).
    """
    h, w = cam.shape
    total = cam.sum()
    if total <= 0:
        return (0.5, 0.5)
    ys, xs = np.indices((h, w))
    cx = float(((xs + 0.5) * cam).sum() / total / w)
    cy = float(((ys + 0.5) * cam).sum() / total / h)
    return cx, cy


def compute_centroid_drift(
    baseline_centroids: list[tuple[float, float]],
    current_centroids: list[tuple[float, float]],
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    bx, by = zip(*baseline_centroids)
    cx, cy = zip(*current_centroids)

    stat_x, p_x = ks_2samp(bx, cx)
    stat_y, p_y = ks_2samp(by, cy)

    return {
        "n_baseline": len(baseline_centroids),
        "n_current": len(current_centroids),
        "x_axis": {"statistic": float(stat_x), "p_value": float(p_x)},
        "y_axis": {"statistic": float(stat_y), "p_value": float(p_y)},
        "alpha": alpha,
        "drifted": bool(p_x < alpha or p_y < alpha),
    }


def _centroid_for_pil_image(
    image: Image.Image, cam_builder: GradCAMPlusPlus, device: torch.device
) -> tuple[float, float]:
    input_tensor = DEFAULT_TRANSFORM(image).unsqueeze(0).to(device)
    cam = cam_builder(input_tensor=input_tensor, targets=[BinaryClassifierOutputTarget(1)])[0]
    return compute_attention_centroid(cam)


def sample_centroids_from_shards(
    predictions_csv: str,
    shards_dir: str,
    cam_builder: GradCAMPlusPlus,
    device: torch.device,
    n_samples: int = DEFAULT_N_CENTROID_SAMPLES,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """Samples n_samples image_ids from a predictions CSV produced by
    evaluate.py, reads them from the tar shards (ShardReader, same as
    gradcam.py), and returns their Grad-CAM attention centroids."""
    predictions = pd.read_csv(predictions_csv)
    sample = predictions.sample(n=min(n_samples, len(predictions)), random_state=seed)
    reader = ShardReader(shards_dir)

    centroids = []
    for image_id in sample["image_id"]:
        member_name = safe_member_name(image_id)
        if member_name not in reader:
            continue
        image = Image.open(io.BytesIO(reader.read(member_name))).convert("RGB")
        centroids.append(_centroid_for_pil_image(image, cam_builder, device))
    return centroids


def sample_centroids_from_external(
    predictions_csv: str,
    images_root: str,
    cam_builder: GradCAMPlusPlus,
    device: torch.device,
    n_samples: int = DEFAULT_N_CENTROID_SAMPLES,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """Same as sample_centroids_from_shards, but for the external
    dataset's predictions CSV (evaluate_external.py's output), reading
    images directly off disk by file_name rather than from tar shards."""
    predictions = pd.read_csv(predictions_csv)
    sample = predictions.sample(n=min(n_samples, len(predictions)), random_state=seed)
    root = Path(images_root)

    centroids = []
    for file_name in sample["file_name"]:
        path = root / file_name
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        centroids.append(_centroid_for_pil_image(image, cam_builder, device))
    return centroids


# ---------------------------------------------------------------------------
# Retrain runbook
# ---------------------------------------------------------------------------

RETRAIN_MANIFEST_PATH = "data/manifest_combined.csv"


def generate_retrain_runbook(
    architecture: str,
    tag: str,
    manifest_path: str = RETRAIN_MANIFEST_PATH,
) -> str:
    """Ready-to-paste ECS commands for a human to run themselves -- see
    drift.py's module docstring and this project's repeated,
    deliberate design decision: the GPU-consuming retrain step stays a
    human action (checking which GPU is actually free on a shared,
    cooperative-etiquette cluster with no job scheduler -- see
    train.py's own HPC-usage note -- isn't something safe to automate),
    so this generates exact TEXT, not an executed action. crop_to_box is
    always on: retraining without it risks diluting the background-
    reliance fix the box-crop experiment already established, especially
    since the new data is IR-heavy from different camera equipment.

    retrained_tag matches boxcrop_experiment.py's own default naming
    (f"{architecture}_boxcrop_retrained") whenever --manifest-path is
    given -- kept in sync explicitly here (not re-derived by guessing)
    so the runbook's evaluate.py commands always name the checkpoint
    boxcrop_experiment.py will actually have written.
    """
    retrained_tag = f"{architecture}_boxcrop_retrained"
    checkpoint_path = f"models/checkpoint_{retrained_tag}.pt"

    return f"""\
# Drift detected on tag={tag!r} -- retrain recommended.
# Run this on cuda16. Check GPU availability first (this project's
# ECS boxes have no job scheduler -- pick whichever GPU has the most
# free memory, not just lowest utilisation):
#   nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

cd ~/aiml589-wildlife-monitoring
git pull
source .venv/bin/activate

# Replace <GPU_ID> with the GPU index chosen above.
CUDA_VISIBLE_DEVICES=<GPU_ID> python -m src.training.boxcrop_experiment \\
  configs/model/{architecture}.yaml \\
  --manifest-path {manifest_path} \\
  --tag {retrained_tag}

# After training finishes, evaluate the retrained checkpoint two ways:

# 1) Regression check -- did it hold up on the original site-based test set?
python -m src.training.evaluate {architecture} boxcrop \\
  --checkpoint-path {checkpoint_path} --tag {retrained_tag}

# 2) The real question -- did it close the external-distribution gap?
#    --split test restricts this to the held-out slice of external data
#    (see combine_manifests.py) -- WITHOUT it, this would partly
#    evaluate on images the retrain just trained on.
python -m src.external_data.evaluate_external {architecture} {retrained_tag} \\
  --checkpoint-path {checkpoint_path} --split test
"""


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_drift_report(
    baseline_predictions_csv: str,
    current_predictions_csv: str,
    tag: str,
    architecture: str = DEFAULT_ARCHITECTURE,
    checkpoint_path: str | None = None,
    baseline_shards_dir: str | None = None,
    current_images_root: str | None = None,
    threshold: float = 0.5,
    n_centroid_samples: int = DEFAULT_N_CENTROID_SAMPLES,
    run_centroid_drift: bool = True,
    push_to_datadog: bool = False,
    datadog_tags: list[str] | None = None,
) -> dict:
    checkpoint_path = checkpoint_path or f"models/checkpoint_{architecture}.pt"
    out_dir = Path("metrics/drift") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(baseline_predictions_csv)
    current = pd.read_csv(current_predictions_csv)

    confidence_result = compute_confidence_drift(baseline["confidence"], current["confidence"])
    concept_result = compute_concept_drift(
        baseline["true_label"].to_numpy(),
        baseline["probability"].to_numpy(),
        current["label"].to_numpy(),
        current["probability"].to_numpy(),
        threshold=threshold,
    )

    centroid_result = None
    if run_centroid_drift:
        device = _resolve_torch_device()
        model = load_checkpoint(checkpoint_path, architecture=architecture, device=device)
        target_layer = TARGET_LAYER_BY_ARCHITECTURE[architecture](model)
        cam_builder = GradCAMPlusPlus(model=model, target_layers=[target_layer])

        baseline_centroids = sample_centroids_from_shards(
            baseline_predictions_csv, baseline_shards_dir, cam_builder, device, n_centroid_samples
        )
        current_centroids = sample_centroids_from_external(
            current_predictions_csv, current_images_root, cam_builder, device, n_centroid_samples
        )
        centroid_result = compute_centroid_drift(baseline_centroids, current_centroids)

    retrain_recommended = concept_result["drifted"]

    report = {
        "tag": tag,
        "architecture": architecture,
        "checkpoint_path": str(checkpoint_path),
        "confidence_drift": confidence_result,
        "concept_drift": concept_result,
        "xai_centroid_drift": centroid_result,
        "retrain_recommended": retrain_recommended,
        "retrain_rationale": (
            "concept_drift.drifted is the deciding signal (labelled correctness actually "
            "degraded beyond tolerance); confidence_drift and xai_centroid_drift are "
            "reported as corroborating context, not tie-breakers."
        ),
    }

    runbook_path = None
    if retrain_recommended:
        runbook = generate_retrain_runbook(architecture, tag)
        runbook_path = out_dir / "retrain_runbook.sh"
        runbook_path.write_text(runbook)
        report["retrain_runbook_path"] = str(runbook_path)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print(f"[drift, {tag}]")
    print(
        f"  confidence drift: score={confidence_result['drift_score']:.4f} "
        f"(threshold={confidence_result['threshold']})  drifted={confidence_result['drifted']}"
    )
    print(
        f"  concept drift (threshold={threshold}): "
        f"accuracy {concept_result['baseline_metrics']['accuracy']:.4f} -> "
        f"{concept_result['current_metrics']['accuracy']:.4f}  "
        f"(delta={concept_result['deltas']['accuracy']:+.4f})  drifted={concept_result['drifted']}"
    )
    if centroid_result is not None:
        print(
            f"  xai-centroid drift: x_p={centroid_result['x_axis']['p_value']:.4f}  "
            f"y_p={centroid_result['y_axis']['p_value']:.4f}  drifted={centroid_result['drifted']}"
        )
    print(f"  RETRAIN RECOMMENDED: {retrain_recommended}")
    print(f"wrote {out_dir}/report.json")
    if runbook_path is not None:
        print(f"wrote {runbook_path} -- ready-to-paste ECS commands (fill in <GPU_ID> after checking nvidia-smi)")

    if push_to_datadog:
        from src.monitoring.datadog_client import push_metrics

        metrics_to_push = {
            f"aiml589.drift.{tag}.confidence_drift_score": confidence_result["drift_score"],
            f"aiml589.drift.{tag}.accuracy_delta": concept_result["deltas"]["accuracy"],
            f"aiml589.drift.{tag}.f1_delta": concept_result["deltas"]["f1"],
            f"aiml589.drift.{tag}.retrain_recommended": float(retrain_recommended),
        }
        if centroid_result is not None:
            metrics_to_push[f"aiml589.drift.{tag}.centroid_x_pvalue"] = centroid_result["x_axis"]["p_value"]
            metrics_to_push[f"aiml589.drift.{tag}.centroid_y_pvalue"] = centroid_result["y_axis"]["p_value"]
        response = push_metrics(metrics_to_push, tags=(datadog_tags or []) + [f"tag:{tag}"])
        print(f"pushed {len(metrics_to_push)} metrics to Datadog: {response}")

    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_predictions_csv")
    parser.add_argument("current_predictions_csv")
    parser.add_argument("--tag", default="drift_check")
    parser.add_argument("--architecture", default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--baseline-shards-dir", default="data/shards")
    parser.add_argument("--current-images-root", default="src/external_data/birds_and_mammals")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-centroid-samples", type=int, default=DEFAULT_N_CENTROID_SAMPLES)
    parser.add_argument("--no-centroid-drift", action="store_true")
    parser.add_argument("--push-to-datadog", action="store_true")
    args = parser.parse_args()

    if args.push_to_datadog:
        from dotenv import load_dotenv

        load_dotenv()

    run_drift_report(
        baseline_predictions_csv=args.baseline_predictions_csv,
        current_predictions_csv=args.current_predictions_csv,
        tag=args.tag,
        architecture=args.architecture,
        baseline_shards_dir=args.baseline_shards_dir,
        current_images_root=args.current_images_root,
        threshold=args.threshold,
        n_centroid_samples=args.n_centroid_samples,
        run_centroid_drift=not args.no_centroid_drift,
        push_to_datadog=args.push_to_datadog,
    )


if __name__ == "__main__":
    main()
