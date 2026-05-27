"""
Practitioner view — designed for DOC field rangers (non-technical users).

Displays:
  - Review queue: low-confidence predictions requiring human sign-off
  - Per-image Grad-CAM heatmap (visual explanation of model decision)
  - Intervention recommendation (confirm detection / override / flag)
  - Confirm / Override buttons that feed back into the retraining label pool

Override decisions are written to the PostgreSQL prediction log and picked
up by the next drift-triggered retraining cycle, closing the feedback loop.
"""
import streamlit as st


def render_practitioner_view() -> None:
    st.title("Field Review Queue")
    st.info("Implementation coming in Phase 4.")
    # TODO Phase 4: fetch low-confidence predictions from PostgreSQL
    # TODO Phase 4: render Grad-CAM heatmap per image
    # TODO Phase 4: Confirm / Override buttons → write correction to DB
