"""
Pydantic request and response schemas for the FastAPI inference API.

Understanding Pydantic in this project:
  Pydantic is a data validation library. Every piece of data that enters
  or leaves the API is described as a Python class with typed fields.
  FastAPI uses these classes to automatically validate requests and
  serialise responses — no manual checking needed.

  Example flow:
    POST /predict  →  FastAPI reads the request body
                   →  Pydantic validates it matches PredictRequest
                   →  your handler receives a clean, typed Python object
                   →  your handler returns a PredictResponse object
                   →  FastAPI serialises it to JSON automatically
"""
from pydantic import BaseModel


class ImageItem(BaseModel):
    """A single image to classify, referenced by its DVC-tracked path."""
    image_path: str
    site_id: str | None = None
    timestamp: str | None = None


class PredictRequest(BaseModel):
    """Batch prediction request — one POST call per field data dump."""
    images: list[ImageItem]
    return_heatmaps: bool = False


class PredictionResult(BaseModel):
    """Per-image prediction with uncertainty."""
    image_path: str
    label: int                  # 1 = mammal (invasive), 0 = bird (native)
    confidence: float           # probability of whichever label was predicted, max(p, 1-p) -- see src/training/evaluate.py
    uncertainty: float          # MC Dropout epistemic uncertainty
    requires_review: bool       # True when confidence < threshold
    heatmap_path: str | None = None


class PredictResponse(BaseModel):
    """Batch prediction response."""
    model_version: str
    threshold_used: float
    predictions: list[PredictionResult]
