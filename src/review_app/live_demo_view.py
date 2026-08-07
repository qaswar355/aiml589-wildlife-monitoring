"""
Live Demo view — upload one trail-camera image, get a prediction back from
the FastAPI inference service in real time.

This is the thin vertical slice made visible: an image goes in here, a
prediction (with MC Dropout uncertainty and an optional Grad-CAM heatmap)
comes back from POST /predict, and that same request is what produces the
MLflow run and data/predictions/serving_log.csv row that make it "logged".

API_BASE_URL is configurable via an environment variable, defaulting to
http://localhost:8000 for local dev (Streamlit and FastAPI as separate
processes on the same host).

Known limitation: image_path (see src.inference.schemas.ImageItem) is a
filesystem path, read by PIL on whatever machine the API process runs on.
That holds for local dev, but not across docker-compose's separate
containers unless a shared volume is added between the api and app
services for a scratch/upload directory -- docker-compose.yml doesn't
have one today. Under Compose, this view won't be able to reach the
uploaded file until that volume exists.
"""
import os
import tempfile
from pathlib import Path

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


def render_live_demo_view() -> None:
    st.title("Live Demo")
    st.caption(f"Calling inference API at {API_BASE_URL}")

    uploaded_file = st.file_uploader("Upload a trail-camera image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        return

    st.image(uploaded_file, caption="Uploaded image", width=300)

    if not st.button("Classify"):
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / uploaded_file.name
        tmp_path.write_bytes(uploaded_file.getvalue())

        with st.spinner("Calling inference API..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/predict",
                    json={"images": [{"image_path": str(tmp_path)}], "return_heatmaps": True},
                    timeout=60,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                st.error(f"Prediction request failed: {exc}")
                return

        result = response.json()["predictions"][0]

    label_text = "Mammal (invasive)" if result["label"] == 1 else "Bird (native)"
    st.subheader(label_text)
    col1, col2 = st.columns(2)
    col1.metric("Confidence", f"{result['confidence']:.2%}")
    col2.metric("Uncertainty (MC Dropout std)", f"{result['uncertainty']:.3f}")

    if result["requires_review"]:
        st.warning("Flagged for human review (uncertainty above cutoff).")
    else:
        st.success("Not flagged for review.")

    if result.get("heatmap_path"):
        st.image(result["heatmap_path"], caption="Grad-CAM++ heatmap")
