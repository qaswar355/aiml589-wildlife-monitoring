"""Tests for the pure logic in predict.py -- threshold resolution, the
review-flag decision, and the local prediction log -- none of which need a
real model or GPU to check."""
import pandas as pd
import pytest
import torch
from PIL import Image

from src.inference.predict import ServingContext, _append_serving_log, _needs_review, _resolve_threshold, predict_one
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


class _FixedLogitModel:
    """Stands in for WildlifeClassifier with a known, fixed logit -- lets
    the temperature-scaling effect be asserted precisely, rather than
    relying on whatever a randomly-initialised real model happens to
    output (which could be close enough to 0 that sigmoid(logit) and
    sigmoid(logit/T) are indistinguishable within floating-point
    tolerance)."""

    def __init__(self, logit_value: float):
        self._logit_value = logit_value

    def __call__(self, x):
        return torch.tensor([self._logit_value])

    def predict_proba(self, x, mc_dropout_passes: int = 1):
        prob = torch.sigmoid(torch.tensor([self._logit_value]))
        return prob, torch.tensor([0.01])


class _FakeDetectorNoDetections:
    """Stands in for a loaded MegaDetector PTDetector -- always reports no
    animal found, so predict_one() falls back to classifying the full
    frame, matching what these tests' assertions were already written
    against before live detection existed."""

    def generate_detections_one_image(self, image, image_id, detection_threshold):
        return {"file": image_id, "detections": []}


def _fake_ctx(logit_value: float, temperature: float, threshold_cfg=None) -> ServingContext:
    return ServingContext(
        model=_FixedLogitModel(logit_value),
        device=torch.device("cpu"),
        architecture="efficientnet_b3",
        tag="test",
        checkpoint_path="unused.pt",
        dropout_rate=0.3,
        mc_dropout_passes=2,
        threshold_cfg=threshold_cfg or {"uncertainty_review_cutoff": 0.15},
        cam_builder=None,
        target_layer=None,
        heatmap_dir="",
        prediction_log_path="",
        temperature=temperature,
        detector=_FakeDetectorNoDetections(),
        md_conf_threshold=0.2,
        box_crop_padding=0.15,
    )


def test_predict_one_applies_temperature_to_confidence_not_label(tmp_path):
    image_path = tmp_path / "img.jpg"
    Image.new("RGB", (20, 20), color=(50, 100, 150)).save(image_path)

    logit_value = 4.0  # sigmoid(4.0) ~= 0.982 -- confidently "mammal"
    raw_prob = torch.sigmoid(torch.tensor(logit_value)).item()

    ctx_uncalibrated = _fake_ctx(logit_value, temperature=1.0)
    ctx_calibrated = _fake_ctx(logit_value, temperature=5.0)

    result_uncalibrated = predict_one(str(image_path), threshold=0.5, ctx=ctx_uncalibrated)
    result_calibrated = predict_one(str(image_path), threshold=0.5, ctx=ctx_calibrated)

    # Same label either way -- the label decision uses the RAW
    # probability, which temperature never touches.
    assert result_uncalibrated.label == result_calibrated.label == 1

    # T=1.0 is a no-op: confidence matches the raw probability exactly.
    assert result_uncalibrated.confidence == pytest.approx(raw_prob, abs=1e-4)

    # T=5.0 softens an overconfident logit -- calibrated confidence must be
    # LOWER than the raw one, not equal to it.
    expected_calibrated = torch.sigmoid(torch.tensor(logit_value / 5.0)).item()
    assert result_calibrated.confidence == pytest.approx(expected_calibrated, abs=1e-4)
    assert result_calibrated.confidence < result_uncalibrated.confidence


def test_predict_one_label_uses_raw_probability_even_near_a_high_threshold(tmp_path):
    """Regression guard for a real deployment scenario: the live default
    threshold is ~0.95, not 0.5. A raw probability can sit above 0.95
    while its temperature-calibrated counterpart sits well below it --
    if calibration ever leaked into the label decision instead of only
    the displayed confidence, this exact case would silently flip the
    label and invalidate every threshold already calibrated against raw
    probabilities."""
    image_path = tmp_path / "img.jpg"
    Image.new("RGB", (20, 20), color=(50, 100, 150)).save(image_path)

    logit_value = 3.2  # sigmoid(3.2) ~= 0.961 -- above a 0.95 threshold
    threshold = 0.95
    calibrated_prob_at_t5 = torch.sigmoid(torch.tensor(logit_value / 5.0)).item()
    assert calibrated_prob_at_t5 < threshold  # confirms this scenario actually exercises the guard

    ctx_uncalibrated = _fake_ctx(logit_value, temperature=1.0)
    ctx_calibrated = _fake_ctx(logit_value, temperature=5.0)

    result_uncalibrated = predict_one(str(image_path), threshold=threshold, ctx=ctx_uncalibrated)
    result_calibrated = predict_one(str(image_path), threshold=threshold, ctx=ctx_calibrated)

    assert result_uncalibrated.label == 1
    assert result_calibrated.label == 1  # unchanged despite a much lower calibrated confidence
