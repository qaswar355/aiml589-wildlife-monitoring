"""Smoke tests for Pydantic request/response schemas."""
import pytest
from pydantic import ValidationError

from src.inference.schemas import (
    ImageItem,
    PredictRequest,
    PredictResponse,
    PredictionResult,
)


def test_image_item_required_field():
    item = ImageItem(image_path="data/processed/img_001.jpg")
    assert item.image_path == "data/processed/img_001.jpg"
    assert item.site_id is None
    assert item.timestamp is None


def test_image_item_optional_fields():
    item = ImageItem(
        image_path="img.jpg",
        site_id="site_northland_01",
        timestamp="2024-06-15T08:30:00",
    )
    assert item.site_id == "site_northland_01"


def test_image_item_missing_required_field_raises():
    with pytest.raises(ValidationError):
        ImageItem()


def test_predict_request_batch():
    req = PredictRequest(
        images=[
            ImageItem(image_path="img_001.jpg"),
            ImageItem(image_path="img_002.jpg"),
        ]
    )
    assert len(req.images) == 2
    assert req.return_heatmaps is False


def test_predict_request_heatmaps_flag():
    req = PredictRequest(images=[ImageItem(image_path="img.jpg")], return_heatmaps=True)
    assert req.return_heatmaps is True


def test_prediction_result_mammal():
    result = PredictionResult(
        image_path="img.jpg",
        label=1,
        confidence=0.92,
        uncertainty=0.03,
        requires_review=False,
    )
    assert result.label == 1
    assert result.heatmap_path is None


def test_prediction_result_requires_review():
    result = PredictionResult(
        image_path="img.jpg",
        label=0,
        confidence=0.55,
        uncertainty=0.18,
        requires_review=True,
    )
    assert result.requires_review is True


def test_predict_response_structure():
    response = PredictResponse(
        model_version="v1.0.0",
        threshold_used=0.5,
        predictions=[
            PredictionResult(
                image_path="img.jpg",
                label=1,
                confidence=0.88,
                uncertainty=0.05,
                requires_review=False,
            )
        ],
    )
    assert response.model_version == "v1.0.0"
    assert len(response.predictions) == 1
