"""Tests for the pure logic in integrated_gradients.py -- the attribution
reduction and per-outcome sampling, none of which need captum, a real
model, or a GPU to check."""
import numpy as np
import pandas as pd
import pytest

from src.xai.integrated_gradients import _sample_per_outcome, reduce_ig_attribution


def test_reduce_ig_attribution_sums_across_channels():
    attr = np.zeros((3, 4, 4), dtype=np.float32)
    attr[:, 1, 1] = [0.1, 0.2, 0.3]  # positive evidence at one pixel, all channels
    out = reduce_ig_attribution(attr, sign=1.0)
    assert out.shape == (4, 4)
    assert out[1, 1] == pytest.approx(0.6)


def test_reduce_ig_attribution_negative_sign_flips_sum_before_clip():
    attr = np.zeros((3, 4, 4), dtype=np.float32)
    attr[:, 0, 0] = [-0.1, -0.1, -0.1]  # negative evidence for the mammal direction
    # sign=-1 (predicted bird) turns "evidence against mammal" into positive
    # support for "evidence toward bird" -- exactly the predicted-class question.
    out = reduce_ig_attribution(attr, sign=-1.0)
    assert out[0, 0] == pytest.approx(0.3)


def test_reduce_ig_attribution_clips_negative_to_zero():
    attr = np.zeros((3, 4, 4), dtype=np.float32)
    attr[:, 2, 2] = [-1.0, -1.0, -1.0]  # evidence against the target direction
    out = reduce_ig_attribution(attr, sign=1.0)
    assert out[2, 2] == 0.0
    assert (out >= 0).all()


def test_sample_per_outcome_respects_n_and_is_reproducible():
    df = pd.DataFrame(
        {
            "true_label": [0] * 10 + [1] * 10,
            "pred_label": [0] * 10 + [1] * 10,  # all bird_correct / mammal_correct
        }
    )
    sample_a = _sample_per_outcome(df, n_per_outcome=3, seed=42)
    sample_b = _sample_per_outcome(df, n_per_outcome=3, seed=42)

    assert sample_a["outcome"].value_counts().to_dict() == {"bird_correct": 3, "mammal_correct": 3}
    assert sample_a["outcome"].isin(["false_positive", "false_negative"]).sum() == 0
    pd.testing.assert_frame_equal(sample_a, sample_b)


def test_sample_per_outcome_caps_at_available_rows():
    df = pd.DataFrame({"true_label": [0, 0], "pred_label": [0, 0]})  # only 2 bird_correct rows
    sample = _sample_per_outcome(df, n_per_outcome=200, seed=1)
    assert len(sample) == 2
