"""Smoke tests for the FastAPI inference application."""
from fastapi.testclient import TestClient

from src.inference.app import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_status():
    response = client.get("/health")
    assert response.json() == {"status": "ok"}


def test_openapi_schema_available():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "Wildlife Monitoring API"


def test_docs_available():
    response = client.get("/docs")
    assert response.status_code == 200
