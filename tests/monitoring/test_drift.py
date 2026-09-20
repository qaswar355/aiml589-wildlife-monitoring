"""Tests for the pure logic in drift.py -- confidence drift (real
evidently calls against synthetic data, no model/network needed),
concept drift, and XAI-centroid drift. sample_centroids_from_shards/
_external and the Datadog push are exercised by a real live run
instead, not unit tests (they need a real checkpoint/images/network)."""
import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    compute_attention_centroid,
    compute_centroid_drift,
    compute_concept_drift,
    compute_confidence_drift,
    generate_retrain_runbook,
)


def test_confidence_drift_detects_a_real_distribution_shift():
    rng = np.random.default_rng(0)
    # A realistic spread for the reference data (not an artificially
    # tight range) -- the "normed" Wasserstein distance divides by the
    # reference's own std, so an unrealistically low-variance baseline
    # inflates the score into a regime this test isn't meant to probe.
    baseline = pd.Series(rng.uniform(0.3, 0.9, size=500))
    current = pd.Series(rng.uniform(0.8, 1.0, size=500))

    result = compute_confidence_drift(baseline, current)
    assert result["drifted"] is True
    assert result["drift_score"] > result["threshold"]
    assert result["method"] == "wasserstein_normed"


def test_confidence_drift_does_not_flag_identical_distributions():
    rng = np.random.default_rng(0)
    baseline = pd.Series(rng.uniform(0.5, 0.9, size=500))
    current = pd.Series(rng.uniform(0.5, 0.9, size=500))  # same distribution, different draw

    result = compute_confidence_drift(baseline, current)
    assert result["drifted"] is False


def _make_probs(n, true_label, target_accuracy, seed):
    """n predictions all with true_label, whose probability is on the
    correct side of 0.5 target_accuracy of the time."""
    rng = np.random.default_rng(seed)
    n_correct = int(n * target_accuracy)
    if true_label == 1:
        correct = rng.uniform(0.6, 0.99, n_correct)
        wrong = rng.uniform(0.01, 0.4, n - n_correct)
    else:
        correct = rng.uniform(0.01, 0.4, n_correct)
        wrong = rng.uniform(0.6, 0.99, n - n_correct)
    return np.concatenate([correct, wrong])


def test_concept_drift_flags_a_real_accuracy_drop():
    baseline_true = np.array([1] * 500 + [0] * 500)
    baseline_prob = np.concatenate([_make_probs(500, 1, 0.95, seed=1), _make_probs(500, 0, 0.95, seed=2)])

    current_true = np.array([1] * 500 + [0] * 500)
    current_prob = np.concatenate([_make_probs(500, 1, 0.60, seed=3), _make_probs(500, 0, 0.60, seed=4)])

    result = compute_concept_drift(baseline_true, baseline_prob, current_true, current_prob)
    assert result["drifted"] is True
    assert result["deltas"]["accuracy"] < 0


def test_concept_drift_does_not_flag_a_small_accuracy_change():
    baseline_true = np.array([1] * 500 + [0] * 500)
    baseline_prob = np.concatenate([_make_probs(500, 1, 0.95, seed=1), _make_probs(500, 0, 0.95, seed=2)])

    current_true = np.array([1] * 500 + [0] * 500)
    current_prob = np.concatenate([_make_probs(500, 1, 0.94, seed=5), _make_probs(500, 0, 0.94, seed=6)])

    result = compute_concept_drift(baseline_true, baseline_prob, current_true, current_prob)
    assert result["drifted"] is False


def test_concept_drift_uses_fixed_threshold_not_a_moving_one():
    """Regression guard for the exact confound this project already hit:
    metrics must be recomputed at the SAME threshold for both sides, not
    whatever's passed as the live/deployed one, or a threshold change
    alone could look like concept drift."""
    baseline_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    baseline_prob = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
    # Identical underlying scores -- only "current" is checked at a
    # different nominal threshold elsewhere in the system, but this
    # function must use its own `threshold` arg consistently for both.
    current_true = baseline_true.copy()
    current_prob = baseline_prob.copy()

    result = compute_concept_drift(baseline_true, baseline_prob, current_true, current_prob, threshold=0.5)
    assert result["deltas"]["accuracy"] == pytest.approx(0.0)
    assert result["drifted"] is False


def test_compute_attention_centroid_finds_a_corner_hotspot():
    cam = np.zeros((10, 10), dtype=np.float32)
    cam[0, 0] = 1.0  # all attention in the top-left corner
    cx, cy = compute_attention_centroid(cam)
    assert cx == pytest.approx(0.05, abs=0.01)
    assert cy == pytest.approx(0.05, abs=0.01)


def test_compute_attention_centroid_finds_the_middle_for_uniform_attention():
    cam = np.ones((10, 10), dtype=np.float32)
    cx, cy = compute_attention_centroid(cam)
    assert cx == pytest.approx(0.5, abs=1e-6)
    assert cy == pytest.approx(0.5, abs=1e-6)


def test_compute_attention_centroid_zero_attention_falls_back_to_centre():
    cam = np.zeros((10, 10), dtype=np.float32)
    assert compute_attention_centroid(cam) == (0.5, 0.5)


def test_centroid_drift_flags_a_real_shift():
    rng = np.random.default_rng(0)
    baseline = [(x, y) for x, y in zip(rng.uniform(0.4, 0.6, 100), rng.uniform(0.4, 0.6, 100))]
    current = [(x, y) for x, y in zip(rng.uniform(0.0, 0.2, 100), rng.uniform(0.0, 0.2, 100))]

    result = compute_centroid_drift(baseline, current)
    assert result["drifted"] is True


def test_centroid_drift_does_not_flag_the_same_distribution():
    rng = np.random.default_rng(0)
    baseline = [(x, y) for x, y in zip(rng.uniform(0.4, 0.6, 200), rng.uniform(0.4, 0.6, 200))]
    current = [(x, y) for x, y in zip(rng.uniform(0.4, 0.6, 200), rng.uniform(0.4, 0.6, 200))]

    result = compute_centroid_drift(baseline, current)
    assert result["drifted"] is False


def test_generate_retrain_runbook_uses_the_combined_manifest_and_crop_to_box():
    runbook = generate_retrain_runbook("efficientnet_b3", tag="efficientnet_b3_vs_external")
    assert "data/manifest_combined.csv" in runbook
    assert "boxcrop_experiment" in runbook
    assert "--tag efficientnet_b3_boxcrop_retrained" in runbook


def test_generate_retrain_runbook_never_bakes_in_a_gpu_id():
    """GPU selection has to stay a live human decision on this shared,
    unscheduled cluster -- the runbook must leave a placeholder, never a
    guessed-at concrete GPU index."""
    runbook = generate_retrain_runbook("efficientnet_b3", tag="some_tag")
    assert "<GPU_ID>" in runbook
    assert "nvidia-smi" in runbook


def test_generate_retrain_runbook_second_eval_command_restricts_to_held_out_split():
    """Regression guard: evaluating a retrained checkpoint against the
    FULL external manifest (no --split test) would silently include the
    images it just trained on -- a real leakage bug caught once already
    while building this."""
    runbook = generate_retrain_runbook("efficientnet_b3", tag="some_tag")
    assert "--split test" in runbook


def test_generate_retrain_runbook_evaluate_commands_reference_the_retrained_checkpoint():
    runbook = generate_retrain_runbook("resnet50", tag="some_tag")
    assert "models/checkpoint_resnet50_boxcrop_retrained.pt" in runbook
    assert "--checkpoint-path models/checkpoint_resnet50_boxcrop_retrained.pt" in runbook
