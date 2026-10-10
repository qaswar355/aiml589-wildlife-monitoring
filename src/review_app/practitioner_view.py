"""
Practitioner view — designed for DOC field rangers (non-technical users).

Displays a review queue of flagged predictions (requires_review == True
in data/predictions/serving_log.csv -- MC Dropout uncertainty above
configs/inference/threshold_config.yaml's uncertainty_review_cutoff at
the time each prediction was made), each with its Grad-CAM heatmap when
one was saved, and Confirm/Override actions.

Confirm/Override write real, persistent rows to
data/predictions/overrides.csv (mirrors src.inference.predict's own
_append_serving_log pattern: header-if-missing, one row per action) --
not ephemeral session state. There is no PostgreSQL in this project
(docker-compose.yml has no db service); this CSV is the same kind of
"real log, not a placeholder" this project has used everywhere else
(src.inference.predict's own docstring already frames serving_log.csv
this way -- "standing in for the not-yet-provisioned PostgreSQL log").
A future automated retraining pipeline can read this CSV the same way
serving_log.csv already gets read here.

Historical rows in serving_log.csv may have been logged under a
different deployed checkpoint (model_tag column) before today's
redeployment/recalibration -- their uncertainty values are on whatever
scale that earlier model's MC Dropout produced, not necessarily
comparable to the CURRENT uncertainty_review_cutoff. Rather than
re-score old rows against a cutoff that wasn't calibrated for them, the
queue just surfaces model_tag per row so a reviewer can see which model
version actually flagged it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from src.review_app.theme import status_badge

DEFAULT_SERVING_LOG_PATH = "data/predictions/serving_log.csv"
DEFAULT_HEATMAP_DIR = "data/predictions/heatmaps"
DEFAULT_OVERRIDES_PATH = "data/predictions/overrides.csv"
DEFAULT_SERVING_CONFIG_PATH = "configs/inference/serving.yaml"

OVERRIDES_COLUMNS = [
    "timestamp", "image_path", "original_label", "corrected_label", "action", "note",
]


def load_review_queue(log_path: str | Path = DEFAULT_SERVING_LOG_PATH) -> pd.DataFrame:
    """Rows requiring human review, most recent first. Empty (but
    correctly-columned) DataFrame if the log doesn't exist yet or has no
    flagged rows -- an empty queue is a normal, expected state, not an
    error."""
    log_path = Path(log_path)
    columns = [
        "timestamp", "image_path", "site_id", "probability", "uncertainty",
        "label", "requires_review", "threshold_used", "model_tag",
    ]
    if not log_path.exists():
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(log_path)
    flagged = df[df["requires_review"] == True].copy()  # noqa: E712
    flagged = flagged.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return flagged


def find_heatmap_path(image_path: str, heatmap_dir: str | Path = DEFAULT_HEATMAP_DIR) -> Path | None:
    """A saved heatmap already has the original image composited into it
    (see src.xai.gradcam.compute_and_save_heatmap), so it doubles as a
    visual for the row even when the original uploaded image itself is
    gone (live_demo_view.py uploads go through a TemporaryDirectory that
    gets cleaned up after the request -- see its own docstring)."""
    stem = Path(image_path).stem
    candidate = Path(heatmap_dir) / f"{stem}_heatmap.png"
    return candidate if candidate.exists() else None


def append_override(
    original_label: int,
    corrected_label: int,
    action: str,
    image_path: str,
    note: str = "",
    overrides_path: str | Path = DEFAULT_OVERRIDES_PATH,
) -> None:
    """Appends one real, persistent correction row. `action` is "confirm"
    (ranger agrees with the model; corrected_label == original_label) or
    "override" (ranger supplies the real answer)."""
    overrides_path = Path(overrides_path)
    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "image_path": image_path,
                "original_label": original_label,
                "corrected_label": corrected_label,
                "action": action,
                "note": note,
            }
        ]
    )
    write_header = not overrides_path.exists()
    row.to_csv(overrides_path, mode="a", header=write_header, index=False)


def _label_text(label: int) -> str:
    return "Mammal (invasive)" if label == 1 else "Bird (native)"


def _current_serving_tag(serving_config_path: str = DEFAULT_SERVING_CONFIG_PATH) -> str | None:
    path = Path(serving_config_path)
    if not path.exists():
        return None
    with path.open() as f:
        return yaml.safe_load(f)["serving"]["tag"]


def render_practitioner_view() -> None:
    st.title("Field Review Queue")
    st.caption(
        "Predictions the model flagged as uncertain enough to need a human decision, "
        "most recent first. Confirm agrees with the model; Override records the real answer."
    )

    queue = load_review_queue()
    current_tag = _current_serving_tag()

    if len(queue) == 0:
        st.info("Nothing in the review queue right now -- no flagged predictions logged.")
        return

    n_stale_model = int((queue["model_tag"] != current_tag).sum()) if current_tag else 0
    tiles_cols = st.columns(3)
    tiles_cols[0].metric("Flagged for review", len(queue))
    tiles_cols[1].metric("Mammal calls", int((queue["label"] == 1).sum()))
    tiles_cols[2].metric("Bird calls", int((queue["label"] == 0).sum()))
    if n_stale_model:
        st.caption(
            f"{n_stale_model} of these were logged by a model version other than the "
            f"currently deployed one ({current_tag}) -- their uncertainty values were "
            "measured on a different scale and aren't directly comparable to today's cutoff."
        )

    for _, row in queue.iterrows():
        with st.container(border=True):
            cols = st.columns([1, 2])
            heatmap_path = find_heatmap_path(row["image_path"])
            with cols[0]:
                if heatmap_path is not None:
                    st.image(str(heatmap_path), caption="Grad-CAM++ heatmap")
                elif Path(row["image_path"]).exists():
                    st.image(row["image_path"], caption="Original image")
                else:
                    st.caption("Image no longer available (temporary upload was cleaned up).")

            with cols[1]:
                st.markdown(f"**{_label_text(int(row['label']))}**")
                status_badge("warning", "Flagged for review")
                st.caption(
                    f"{row['timestamp']}  ·  confidence {row['probability']:.1%}  ·  "
                    f"uncertainty {row['uncertainty']:.4f}  ·  model `{row['model_tag']}`"
                )

                note = st.text_input("Reviewer note (optional)", key=f"note_{row.name}")
                button_cols = st.columns(2)
                if button_cols[0].button("Confirm", key=f"confirm_{row.name}"):
                    append_override(
                        original_label=int(row["label"]),
                        corrected_label=int(row["label"]),
                        action="confirm",
                        image_path=row["image_path"],
                        note=note,
                    )
                    st.success("Recorded.")
                if button_cols[1].button("Override", key=f"override_{row.name}"):
                    corrected = 0 if row["label"] == 1 else 1
                    append_override(
                        original_label=int(row["label"]),
                        corrected_label=corrected,
                        action="override",
                        image_path=row["image_path"],
                        note=note,
                    )
                    st.success(f"Recorded as {_label_text(corrected)}.")
