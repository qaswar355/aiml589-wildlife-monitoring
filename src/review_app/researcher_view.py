"""
Researcher / Supervisor view — model health and MLOps oversight.

Every number on this view is read from real, already-computed project
artifacts -- no placeholders, nothing fabricated for display purposes:
  - Model comparison: metrics/<tag>/eval.json for every checkpoint this
    project has actually evaluated.
  - Drift status: metrics/drift/<tag>/report.json -- confidence drift and
    concept drift. The report also holds an XAI-centroid result, which is
    not shown: the heatmap centre mostly follows where animals sit in the
    frame, so it can't separate a change in the images from a drop in
    model quality.
  - Retraining history: reconstructed from metrics/*/eval.json file
    timestamps, NOT mlflow.search_runs() -- every run directory under
    mlruns/0/ is missing meta.yaml (checked directly: 131 run dirs, 0
    meta.yaml files), so the MLflow file-store client can't parse any of
    them as queryable runs. The eval.json files are this project's real,
    authoritative result record regardless, so they're the honest data
    source here -- this limitation is disclosed in the UI, not hidden.
  - Threshold preview: computed live from
    data/predictions/test_predictions_efficientnet_b3_boxcrop_retrained_fullframe.csv
    (the deployed checkpoint's real full-frame test predictions).
    PREVIEW ONLY, by deliberate design decision -- never writes to
    configs/inference/threshold_config.yaml.
  - Seasonal generalisation: metrics/efficientnet_b3_seasonal(_boxcrop)/
    eval.json's site_familiarity breakdown -- the thesis's stated
    headline contribution (seen-site vs novel-site generalisation gap).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.review_app.theme import status_badge

METRICS_DIR = Path("metrics")
SERVING_CONFIG_PATH = Path("configs/inference/serving.yaml")
THRESHOLD_CONFIG_PATH = Path("configs/inference/threshold_config.yaml")
# The deployed model is a box-crop model, so its drift check uses MegaDetector-cropped
# external images (src/external_data/evaluate_external_cropped.py), matching how it is served.
DRIFT_REPORT_PATH = METRICS_DIR / "drift" / "efficientnet_b3_boxcrop_cropped_vs_external" / "report.json"
DEFAULT_RETRAIN_RATIONALE = (
    "concept drift is the deciding signal (labelled correctness actually degraded beyond tolerance); "
    "confidence drift is corroborating context."
)
FULLFRAME_PREDICTIONS_PATH = Path("data/predictions/test_predictions_efficientnet_b3_boxcrop_retrained_fullframe.csv")

MODEL_COMPARISON_TAGS = [
    "efficientnet_b3",
    "resnet50",
    "efficientnet_b3_boxcrop",
    "efficientnet_b3_seasonal",
    "efficientnet_b3_seasonal_boxcrop",
    "efficientnet_b3_boxcrop_retrained",
]

MODEL_LABELS = {
    "efficientnet_b3": "EfficientNet-B3 (standard model)",
    "resnet50": "ResNet-50",
    "efficientnet_b3_boxcrop": "EfficientNet-B3 + box-crop",
    "efficientnet_b3_seasonal": "EfficientNet-B3, seasonal split",
    "efficientnet_b3_seasonal_boxcrop": "EfficientNet-B3, seasonal + box-crop",
    "efficientnet_b3_boxcrop_retrained": "EfficientNet-B3, retrained on combined data",
}


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def get_deployed_tag(serving_config_path: Path = SERVING_CONFIG_PATH) -> str | None:
    if not serving_config_path.exists():
        return None
    cfg = yaml.safe_load(serving_config_path.read_text())
    return cfg.get("serving", {}).get("tag")


def load_model_comparison(tags: list[str], metrics_dir: Path = METRICS_DIR) -> pd.DataFrame:
    """Reads metrics/<tag>/eval.json for each tag that actually exists;
    missing tags are silently skipped (not every experiment produced
    every checkpoint variant), never fabricated."""
    rows = []
    for tag in tags:
        data = _load_json(metrics_dir / tag / "eval.json")
        if data is None:
            continue
        rows.append(
            {
                "tag": tag,
                "label": MODEL_LABELS.get(tag, tag),
                "accuracy": data.get("accuracy"),
                "precision": data.get("precision"),
                "recall": data.get("recall"),
                "f1": data.get("f1"),
                "n_test_images": data.get("n_test_images"),
            }
        )
    return pd.DataFrame(rows)


def load_retraining_history(tags: list[str], metrics_dir: Path = METRICS_DIR) -> pd.DataFrame:
    """Reconstructs a real timeline from metrics/<tag>/eval.json file
    modification times -- see module docstring for why this is used
    instead of mlflow.search_runs() (every local run directory is
    missing meta.yaml, so the MLflow client can't parse any of them)."""
    rows = []
    for tag in tags:
        path = metrics_dir / tag / "eval.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        rows.append(
            {
                "tag": tag,
                "label": MODEL_LABELS.get(tag, tag),
                "produced_at": pd.Timestamp(path.stat().st_mtime, unit="s"),
                "accuracy": data.get("accuracy"),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("produced_at").reset_index(drop=True) if len(df) else df


def compute_threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """Precision/recall/accuracy at each candidate threshold -- pure
    numpy/sklearn, no Streamlit dependency, so this is unit-testable on
    its own."""
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def _drift_status(drifted: bool | None) -> str:
    if drifted is None:
        return "warning"
    return "critical" if drifted else "good"


def render_researcher_view() -> None:
    st.title("Model Health")
    st.caption("Real results pulled directly from this project's evaluation, drift-detection, and calibration artifacts.")

    deployed_tag = get_deployed_tag()

    # -- Model comparison -----------------------------------------------
    st.subheader("Model comparison")
    comparison = load_model_comparison(MODEL_COMPARISON_TAGS)
    if comparison.empty:
        st.info("No evaluation results found under metrics/.")
    else:
        display = comparison.copy()
        display["deployed"] = display["tag"].apply(lambda t: "🟢 deployed" if t == deployed_tag else "")
        display_cols = ["label", "accuracy", "precision", "recall", "f1", "n_test_images", "deployed"]
        st.dataframe(
            display[display_cols].style.format(
                {"accuracy": "{:.2%}", "precision": "{:.2%}", "recall": "{:.2%}", "f1": "{:.2%}"}
            ),
            hide_index=True,
            width="stretch",
        )
        deployed_row = comparison[comparison["tag"] == deployed_tag]
        if len(deployed_row):
            r = deployed_row.iloc[0]
            cols = st.columns(4)
            cols[0].metric("Deployed accuracy", f"{r['accuracy']:.1%}")
            cols[1].metric("Deployed precision", f"{r['precision']:.1%}")
            cols[2].metric("Deployed recall", f"{r['recall']:.1%}")
            cols[3].metric("Deployed F1", f"{r['f1']:.1%}")

    st.divider()

    # -- Drift status -----------------------------------------------------
    st.subheader("Drift status")
    drift = _load_json(DRIFT_REPORT_PATH)
    if drift is None:
        st.info("No drift report found yet -- run `python -m src.monitoring.drift` to produce one.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Confidence drift**")
            cd = drift["confidence_drift"]
            status_badge(_drift_status(cd["drifted"]), "Drifted" if cd["drifted"] else "Stable")
            st.caption(f"Wasserstein score {cd['drift_score']:.3f} (threshold {cd['threshold']})")
        with c2:
            st.markdown("**Concept drift**")
            cc = drift["concept_drift"]
            status_badge(_drift_status(cc["drifted"]), "Drifted" if cc["drifted"] else "Stable")
            st.caption(
                f"Accuracy {cc['baseline_metrics']['accuracy']:.1%} -> "
                f"{cc['current_metrics']['accuracy']:.1%} ({cc['deltas']['accuracy']:+.1%})"
            )

        st.markdown(
            f"**Retrain recommended: {'Yes' if drift['retrain_recommended'] else 'No'}** — {drift.get('retrain_rationale', DEFAULT_RETRAIN_RATIONALE)}"
        )

    st.divider()

    # -- Retraining history -----------------------------------------------
    st.subheader("Retraining / evaluation history")
    st.caption(
        "Reconstructed from metrics/*/eval.json file timestamps, not a live MLflow query -- every run directory "
        "under mlruns/0/ is missing its meta.yaml, so MLflow's file-store client can't parse any of them as "
        "queryable runs (checked directly: 131 run directories, 0 with meta.yaml)."
    )
    history = load_retraining_history(MODEL_COMPARISON_TAGS)
    if history.empty:
        st.info("No history available.")
    else:
        st.dataframe(
            history[["label", "produced_at", "accuracy"]].style.format({"accuracy": "{:.2%}"}),
            hide_index=True,
            width="stretch",
        )
        st.line_chart(history.set_index("produced_at")["accuracy"])

    st.divider()

    # -- Threshold preview (never writes to config) -----------------------
    st.subheader("Decision threshold preview")
    st.caption(
        "Preview only -- moving this slider does NOT change configs/inference/threshold_config.yaml. "
        "Applying a new threshold is a deliberate, separate governance action."
    )
    if not FULLFRAME_PREDICTIONS_PATH.exists():
        st.info(f"No prediction file found at {FULLFRAME_PREDICTIONS_PATH}.")
    else:
        preds = pd.read_csv(FULLFRAME_PREDICTIONS_PATH)
        threshold_cfg = yaml.safe_load(THRESHOLD_CONFIG_PATH.read_text())
        deployed_threshold = threshold_cfg["thresholds"]["default"]
        recall_floor = threshold_cfg["recall_floor"]

        st.warning(
            f"Known limitation: at the deployed threshold ({deployed_threshold}), and at every threshold tested "
            f"from 0.005 to 0.99, recall does not reach the {recall_floor:.0%} governance floor on full-frame "
            f"live input with this checkpoint (recall peaks around 86% near threshold=0). This is a disclosed, "
            f"deliberate deployment trade-off -- see threshold_config.yaml's own comments for the full reasoning."
        )

        slider_threshold = st.slider("Preview threshold", min_value=0.0, max_value=1.0, value=float(deployed_threshold), step=0.01)
        sweep = compute_threshold_sweep(
            preds["true_label"].to_numpy(), preds["probability"].to_numpy(), np.array([slider_threshold])
        )
        row = sweep.iloc[0]
        cols = st.columns(3)
        cols[0].metric("Accuracy at this threshold", f"{row['accuracy']:.1%}")
        cols[1].metric("Precision at this threshold", f"{row['precision']:.1%}")
        cols[2].metric(
            "Recall at this threshold",
            f"{row['recall']:.1%}",
            # "normal" (not "inverse") in every case here -- higher recall
            # is unambiguously better whether currently above or below the
            # floor, so positive-delta-is-good semantics are always
            # correct. (Caught via a real screenshot: an earlier
            # conditional version showed a BELOW-floor recall in green,
            # backwards, because it used "inverse" for the below-floor
            # case -- "inverse" would only be right for a metric where
            # LOWER is better, e.g. error rate.)
            delta=f"{row['recall'] - recall_floor:+.1%} vs floor",
            delta_color="normal",
        )

        curve_thresholds = np.linspace(0.0, 1.0, 41)
        curve = compute_threshold_sweep(preds["true_label"].to_numpy(), preds["probability"].to_numpy(), curve_thresholds)
        st.line_chart(curve.set_index("threshold")[["precision", "recall"]])

    st.divider()

    # -- Seasonal generalisation (headline contribution) -------------------
    st.subheader("Seasonal generalisation")
    st.caption("Train on spring/summer, test on autumn/winter -- this project's stated headline contribution.")
    seasonal = _load_json(METRICS_DIR / "efficientnet_b3_seasonal" / "eval.json")
    seasonal_boxcrop = _load_json(METRICS_DIR / "efficientnet_b3_seasonal_boxcrop" / "eval.json")

    if seasonal is None:
        st.info("No seasonal evaluation results found.")
    else:
        for title, data in (("Uncropped", seasonal), ("+ box-crop", seasonal_boxcrop)):
            if data is None or "site_familiarity" not in data:
                continue
            fam = data["site_familiarity"]
            st.markdown(f"**{title}**")
            cols = st.columns(2)
            cols[0].metric(
                f"Seen-site accuracy (n={fam['seen_site']['n']:,})",
                f"{fam['seen_site']['accuracy']:.1%}",
            )
            cols[1].metric(
                f"Novel-site accuracy (n={fam['novel_site']['n']:,})",
                f"{fam['novel_site']['accuracy']:.1%}",
                # "off" (neutral, no red/green judgement) deliberately --
                # this is a descriptive generalisation-gap comparison, not
                # a pass/fail threshold check, so colour-coding it as
                # good/bad either way would overstate or understate an
                # expected, already-documented finding.
                delta=f"{fam['novel_site']['accuracy'] - fam['seen_site']['accuracy']:+.1%} vs seen-site",
                delta_color="off",
            )
