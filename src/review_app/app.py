"""
Streamlit review dashboard — entry point.

Two views selected via sidebar radio button:
  Practitioner View  — DOC field rangers (non-technical users)
  Researcher View    — supervisor and thesis examiner

To run locally:   streamlit run src/review_app/app.py
To run in Docker: see docker-compose.yml (app service)
"""
import streamlit as st

from src.review_app.practitioner_view import render_practitioner_view
from src.review_app.researcher_view import render_researcher_view

st.set_page_config(
    page_title="Wildlife Monitoring Dashboard",
    page_icon="🦜",
    layout="wide",
)

st.sidebar.title("Wildlife Monitoring")
view = st.sidebar.radio("Select view", ["Practitioner", "Researcher / Supervisor"])

if view == "Practitioner":
    render_practitioner_view()
else:
    render_researcher_view()
