"""
FastAPI inference application — entry point for the wildlife classifier API.

Routes:
  POST /predict    — batch inference on a list of images
  GET  /health     — liveness check for the Docker health check
  GET  /model-info — current model version, architecture, checkpoint, and
                      thresholds in use

Model + configs are loaded once at startup via a FastAPI lifespan context
into app.state.ctx (a ServingContext) -- not per request. create_app() is
a factory rather than a bare module-level app so tests can build an
isolated instance pointed at fake serving/threshold configs and a tiny
untrained checkpoint, without touching the real ones under configs/ and
models/.

  To run locally:   uvicorn src.inference.app:app --reload
  To run in Docker: see Dockerfile.api + docker-compose.yml
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.inference.predict import (
    DEFAULT_SERVING_CONFIG,
    DEFAULT_THRESHOLD_CONFIG,
    build_serving_context,
    predict_batch,
)
from src.inference.schemas import PredictRequest, PredictResponse


def create_app(
    serving_config_path: str = DEFAULT_SERVING_CONFIG,
    threshold_config_path: str = DEFAULT_THRESHOLD_CONFIG,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ctx = build_serving_context(serving_config_path, threshold_config_path)
        yield
        # Nothing to tear down -- the model and cam_builder are in-process
        # objects, garbage-collected with the process; no open connections
        # or file handles held past request lifetime.

    app = FastAPI(
        title="Wildlife Monitoring API",
        description="Camera trap image classifier — NZ invasive mammals vs native birds",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest) -> PredictResponse:
        if not request.images:
            raise HTTPException(status_code=422, detail="images must be a non-empty list")
        return predict_batch(request, app.state.ctx)

    @app.get("/model-info")
    async def model_info() -> dict:
        ctx = app.state.ctx
        return {
            "tag": ctx.tag,
            "architecture": ctx.architecture,
            "checkpoint_path": ctx.checkpoint_path,
            "mc_dropout_passes": ctx.mc_dropout_passes,
            "thresholds": ctx.threshold_cfg["thresholds"],
            "uncertainty_review_cutoff": ctx.threshold_cfg["uncertainty_review_cutoff"],
            "device": str(ctx.device),
        }

    return app


app = create_app()
