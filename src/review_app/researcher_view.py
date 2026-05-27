"""
Researcher / Supervisor view — model health and MLOps oversight.

Displays:
  - Recall / precision trends over retraining cycles
  - Drift status (confidence drift, concept drift, XAI centroid drift)
  - Retraining history and model version timeline
  - Threshold controls per deployment zone
  - Seasonal generalisation experiment results
"""
import streamlit as st


def render_researcher_view() -> None:
    st.title("Model Health & Research Overview")
    st.info("Implementation coming in Phase 4.")
    # TODO Phase 4: pull metrics from MLflow tracking server
    # TODO Phase 4: render drift reports from Evidently
    # TODO Phase 4: threshold slider → write to threshold_config.yaml
