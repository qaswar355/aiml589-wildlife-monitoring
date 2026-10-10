"""Tests for the pure logic in researcher_view.py -- threshold sweep,
model comparison loading, retraining history, and drift status mapping.
None of these need Streamlit's runtime."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.review_app.researcher_view import (
    _drift_status,
    compute_threshold_sweep,
    get_deployed_tag,
    load_model_comparison,
    load_retraining_history,
)


def test_compute_threshold_sweep_matches_hand_counted_values():
    y_true = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.3, 0.6, 0.9])

    sweep = compute_threshold_sweep(y_true, y_prob, np.array([0.5]))
    row = sweep.iloc[0]
    # threshold=0.5: pred = [0,0,0,1,1] -> 2/3 true mammals caught, 0 false positives
    assert row["accuracy"] == pytest.approx(4 / 5)
    assert row["precision"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(2 / 3)


def test_compute_threshold_sweep_one_row_per_threshold():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.2, 0.8, 0.6, 0.4])
    sweep = compute_threshold_sweep(y_true, y_prob, np.array([0.1, 0.5, 0.9]))
    assert len(sweep) == 3
    assert list(sweep["threshold"]) == [0.1, 0.5, 0.9]


def test_compute_threshold_sweep_higher_threshold_never_increases_recall():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.uniform(0, 1, size=200)
    sweep = compute_threshold_sweep(y_true, y_prob, np.array([0.2, 0.5, 0.8]))
    assert sweep.iloc[0]["recall"] >= sweep.iloc[1]["recall"] >= sweep.iloc[2]["recall"]


def test_drift_status_maps_drifted_to_critical():
    assert _drift_status(True) == "critical"


def test_drift_status_maps_not_drifted_to_good():
    assert _drift_status(False) == "good"


def test_drift_status_maps_missing_signal_to_warning():
    assert _drift_status(None) == "warning"


def test_load_model_comparison_skips_missing_tags(tmp_path):
    metrics_dir = tmp_path / "metrics"
    (metrics_dir / "tag_a").mkdir(parents=True)
    (metrics_dir / "tag_a" / "eval.json").write_text(
        json.dumps({"accuracy": 0.9, "precision": 0.8, "recall": 0.85, "f1": 0.82, "n_test_images": 100})
    )
    # tag_b deliberately has no eval.json -- must be skipped, not crash.

    df = load_model_comparison(["tag_a", "tag_b"], metrics_dir=metrics_dir)
    assert len(df) == 1
    assert df.iloc[0]["tag"] == "tag_a"
    assert df.iloc[0]["accuracy"] == 0.9


def test_load_model_comparison_empty_when_nothing_exists(tmp_path):
    df = load_model_comparison(["nope"], metrics_dir=tmp_path / "metrics")
    assert df.empty


def test_load_retraining_history_sorted_by_time(tmp_path):
    metrics_dir = tmp_path / "metrics"
    for tag, acc in [("old_tag", 0.8), ("new_tag", 0.9)]:
        d = metrics_dir / tag
        d.mkdir(parents=True)
        (d / "eval.json").write_text(json.dumps({"accuracy": acc}))

    # Force a real, distinct mtime ordering (new_tag written after old_tag).
    import os
    import time

    old_path = metrics_dir / "old_tag" / "eval.json"
    new_path = metrics_dir / "new_tag" / "eval.json"
    now = time.time()
    os.utime(old_path, (now - 100, now - 100))
    os.utime(new_path, (now, now))

    history = load_retraining_history(["new_tag", "old_tag"], metrics_dir=metrics_dir)
    assert list(history["tag"]) == ["old_tag", "new_tag"]  # chronological, not input order


def test_get_deployed_tag_reads_real_serving_yaml(tmp_path):
    serving_path = tmp_path / "serving.yaml"
    serving_path.write_text(yaml.dump({"serving": {"tag": "some_tag"}}))
    assert get_deployed_tag(serving_path) == "some_tag"


def test_get_deployed_tag_returns_none_when_config_missing(tmp_path):
    assert get_deployed_tag(tmp_path / "does_not_exist.yaml") is None
