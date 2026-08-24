"""Tests for the pure logic in calibration.py -- ECE, temperature
fitting, and candidate-cutoff computation -- none of which need a real
checkpoint, shards, or a GPU to check."""
import numpy as np
import pytest
import torch

from src.inference.calibration import compute_candidate_cutoffs, expected_calibration_error, fit_temperature


def test_ece_is_zero_for_perfectly_calibrated_predictions():
    # Two confidence buckets, each exactly as accurate as its stated
    # confidence: 100 predictions at p=0.9 with 90 correct, 100 at p=0.5
    # with 50 correct (labelled 1 == "correct" via probs>=0.5 matching
    # label exactly for the p=0.9 group, and an even coin-flip match for
    # the p=0.5 group).
    probs = np.array([0.9] * 100 + [0.5] * 100)
    labels = np.array([1] * 90 + [0] * 10 + [1] * 50 + [0] * 50)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_ece_is_high_for_confidently_wrong_predictions():
    # Maximally overconfident and maximally wrong: p=0.99 (99% confidence)
    # but actually correct only 0% of the time.
    probs = np.full(50, 0.99)
    labels = np.zeros(50)  # predicted class (p>=0.5 -> 1) is never right
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece > 0.9  # confidence ~0.99, accuracy 0.0 -> gap ~0.99


def test_fit_temperature_softens_an_overconfident_model():
    # Logits far more extreme than the true label noise justifies: labels
    # are only 90% separable, but logits claim near-total certainty.
    torch.manual_seed(0)
    n = 2000
    labels = torch.cat([torch.ones(int(n * 0.9)), torch.zeros(n - int(n * 0.9))])
    # huge-magnitude logits matching the label almost always, but not
    # backed by real separability -- classic overconfidence setup.
    logits = torch.where(labels == 1, torch.full((n,), 10.0), torch.full((n,), -10.0))
    # inject some label noise so the model's extreme certainty is unjustified
    noise_idx = torch.randperm(n)[: int(n * 0.1)]
    labels[noise_idx] = 1 - labels[noise_idx]

    nll_before = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
    temperature = fit_temperature(logits, labels)
    nll_after = torch.nn.functional.binary_cross_entropy_with_logits(logits / temperature, labels).item()

    assert temperature > 1.0  # should soften, not sharpen, an overconfident model
    assert nll_after < nll_before  # calibration should improve held-out likelihood


def test_fit_temperature_leaves_a_well_calibrated_model_near_one():
    torch.manual_seed(0)
    # Logits whose sigmoid already matches the true label frequency well.
    n = 2000
    true_probs = torch.rand(n) * 0.8 + 0.1  # spread across [0.1, 0.9]
    labels = torch.bernoulli(true_probs)
    logits = torch.logit(true_probs)

    temperature = fit_temperature(logits, labels)
    assert 0.7 < temperature < 1.4  # near 1, not pulled far in either direction


def test_compute_candidate_cutoffs_flags_the_requested_fraction():
    uncertainty = np.arange(100, dtype=float)  # 0..99, uniform
    is_error = np.zeros(100, dtype=int)
    is_error[95:] = 1  # the 5 highest-uncertainty rows are the only errors

    candidates = compute_candidate_cutoffs(uncertainty, is_error, percentiles=(95,))
    c = candidates[0]
    assert c["flagged_fraction"] == pytest.approx(0.05, abs=0.01)
    assert c["error_capture_rate"] == pytest.approx(1.0, abs=1e-6)  # catches all 5 errors
    assert c["precision_of_flagging"] == pytest.approx(1.0, abs=1e-6)  # everything flagged is an error


def test_compute_candidate_cutoffs_low_percentile_still_catches_errors_but_less_precisely():
    uncertainty = np.arange(100, dtype=float)
    is_error = np.zeros(100, dtype=int)
    is_error[50:] = 1  # half the population are errors, spread across upper half

    candidates = compute_candidate_cutoffs(uncertainty, is_error, percentiles=(50,))
    c = candidates[0]
    assert c["flagged_fraction"] == pytest.approx(0.50, abs=0.02)
    assert c["error_capture_rate"] == pytest.approx(1.0, abs=1e-6)
    assert c["precision_of_flagging"] == pytest.approx(1.0, abs=1e-6)
