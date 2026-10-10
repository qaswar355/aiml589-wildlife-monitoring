"""
Streamlit review dashboard — entry point.

Three views via Streamlit's native multi-page navigation (st.navigation/
st.Page) rather than a hand-built sidebar radio button -- the idiomatic,
current pattern for a multi-view Streamlit app, and it reads as a real
app (top navigation, per-page titles/icons) rather than a single script
switching on a radio value.

  Live Demo     — the thin vertical slice made visible: upload an image,
                  get a real prediction
  Practitioner  — DOC field rangers: review queue for low-confidence/
                  flagged predictions, Confirm/Override
  Researcher    — supervisor/examiner: model health, drift status,
                  experiment results

Visual theme lives entirely in .streamlit/config.toml (see
src/review_app/theme.py's module docstring for why).

To run locally:   streamlit run src/review_app/app.py
To run in Docker: see docker-compose.yml (app service)
"""
import streamlit as st

from src.review_app.live_demo_view import render_live_demo_view
from src.review_app.practitioner_view import render_practitioner_view
from src.review_app.researcher_view import render_researcher_view

st.set_page_config(
    page_title="Wildlife Monitoring Dashboard",
    page_icon="🦜",
    layout="wide",
)

pages = [
    st.Page(render_live_demo_view, title="Live Demo", icon=":material/photo_camera:", default=True),
    st.Page(render_practitioner_view, title="Field Review", icon=":material/fact_check:"),
    st.Page(render_researcher_view, title="Model Health", icon=":material/monitoring:"),
]
nav = st.navigation(pages, position="top")
nav.run()
