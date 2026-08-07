"""Tests for the pure logic in predict.py -- threshold resolution, the
review-flag decision, and the local prediction log -- none of which need a
real model or GPU to check."""
import pandas as pd

from src.inference.predict import _append_serving_log, _needs_review, _resolve_threshold
from src.inference.schemas import ImageItem, PredictionResult


def test_resolve_threshold_uses_site_specific_value():
    cfg = {"thresholds": {"default": 0.5, "fiordland": 0.7}}
    assert _resolve_threshold("fiordland", cfg) == 0.7


def test_resolve_threshold_falls_back_to_default_for_none_site():
    cfg = {"thresholds": {"default": 0.5, "fiordland": 0.7}}
    assert _resolve_threshold(None, cfg) == 0.5


def test_resolve_threshold_falls_back_to_default_for_unrecognised_zone():
    cfg = {"thresholds": {"default": 0.5, "fiordland": 0.7}}
    assert _resolve_threshold("some_unknown_zone", cfg) == 0.5


def test_needs_review_true_strictly_above_cutoff():
    assert _needs_review(0.20, 0.15) is True


def test_needs_review_false_at_cutoff_boundary():
    assert _needs_review(0.15, 0.15) is False


def test_needs_review_false_below_cutoff():
    assert _needs_review(0.05, 0.15) is False


class _FakeServingContext:
    """Just enough of ServingContext for _append_serving_log -- it only
    reads .prediction_log_path and .tag."""

    def __init__(self, prediction_log_path, tag="test_tag"):
        self.prediction_log_path = prediction_log_path
        self.tag = tag


def _sample_item_and_result(image_path="img.jpg", site_id="fiordland"):
    item = ImageItem(image_path=image_path, site_id=site_id)
    result = PredictionResult(
        image_path=image_path, label=1, confidence=0.9, uncertainty=0.02, requires_review=False
    )
    return item, result


def test_append_serving_log_creates_file_with_header(tmp_path):
    log_path = tmp_path / "serving_log.csv"
    ctx = _FakeServingContext(str(log_path))
    item, result = _sample_item_and_result()

    _append_serving_log([item], [result], threshold_used=0.5, ctx=ctx)

    assert log_path.exists()
    df = pd.read_csv(log_path)
    assert len(df) == 1
    assert set(df.columns) == {
        "timestamp", "image_path", "site_id", "probability", "uncertainty",
        "label", "requires_review", "threshold_used", "model_tag",
    }
    assert df.iloc[0]["model_tag"] == "test_tag"


def test_append_serving_log_appends_without_duplicate_header(tmp_path):
    log_path = tmp_path / "serving_log.csv"
    ctx = _FakeServingContext(str(log_path))
    item, result = _sample_item_and_result()

    _append_serving_log([item], [result], threshold_used=0.5, ctx=ctx)
    _append_serving_log([item], [result], threshold_used=0.5, ctx=ctx)

    lines = log_path.read_text().splitlines()
    assert len(lines) == 3  # one header + two data rows
    df = pd.read_csv(log_path)
    assert len(df) == 2
