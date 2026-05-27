"""
FastAPI inference application — entry point for the wildlife classifier API.

How FastAPI works (quick reference):
  @app.post("/predict")  — registers a route that accepts POST requests
  async def predict(req: PredictRequest) — Pydantic validates the body automatically
  return PredictResponse(...)  — FastAPI serialises this to JSON automatically

  To run locally:   uvicorn src.inference.app:app --reload
  To run in Docker: see Dockerfile.api + docker-compose.yml

Routes (to be implemented in Phase 3):
  POST /predict    — batch inference on a list of images
  GET  /health     — liveness check for the Docker health check
  GET  /model-info — current model version and threshold in use
"""
from fastapi import FastAPI

from src.inference.schemas import PredictRequest, PredictResponse

app = FastAPI(
    title="Wildlife Monitoring API",
    description="Camera trap image classifier — NZ invasive mammals vs native birds",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# TODO Phase 3: implement /predict batch endpoint
# TODO Phase 3: load model + threshold at startup via lifespan context
# TODO Phase 3: add Sentry middleware for error tracking
