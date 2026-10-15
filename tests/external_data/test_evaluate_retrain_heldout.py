"""Tests for the before/after retraining comparison -- pure counting on
small hand-made prediction files."""
import numpy as np
import pandas as pd
import pytest

from src.external_data.evaluate_retrain_heldout import compare_at_threshold, load_paired


def test_compare_at_threshold_counts_fixed_and_new_errors():
    y = np.array([0, 0, 1, 1])
    before = np.array([0.9, 0.1, 0.2, 0.8])  # wrong, right, wrong, right
    after = np.array([0.1, 0.1, 0.9, 0.2])   # right, right, right, wrong
    r = compare_at_threshold(y, before, after, 0.5)
    assert r["before"] == {"accuracy": 0.5, "false_positives": 1, "false_negatives": 1}
    assert r["after"] == {"accuracy": 0.75, "false_positives": 0, "false_negatives": 1}
    assert (r["errors_fixed"], r["errors_new"]) == (2, 1)
    assert 0.0 < r["mcnemar_p"] <= 1.0


def test_compare_at_threshold_with_no_disagreements_has_p_of_one():
    y = np.array([0, 1])
    probs = np.array([0.1, 0.9])
    assert compare_at_threshold(y, probs, probs, 0.5)["mcnemar_p"] == 1.0


def _write(path, ids, probs, labels=(0, 1), splits=("test", "val")):
    pd.DataFrame({
        "image_id": ids, "label": list(labels), "split": list(splits),
        "event_id": ["A|kaka|evt1", "B|possum|evt1"], "probability": probs,
    }).to_csv(path, index=False)


def test_load_paired_aligns_rows_by_image_id(tmp_path):
    _write(tmp_path / "b.csv", ["x", "y"], [0.2, 0.7])
    _write(tmp_path / "a.csv", ["y", "x"], [0.9, 0.1], labels=(1, 0), splits=("val", "test"))
    paired = load_paired(tmp_path / "b.csv", tmp_path / "a.csv")
    assert paired.loc["x", "after_prob"] == pytest.approx(0.1)
    assert paired.loc["y", "site"] == "B"


def test_load_paired_rejects_mismatched_images(tmp_path):
    _write(tmp_path / "b.csv", ["x", "y"], [0.2, 0.7])
    _write(tmp_path / "a.csv", ["x", "z"], [0.2, 0.7])
    with pytest.raises(ValueError):
        load_paired(tmp_path / "b.csv", tmp_path / "a.csv")
