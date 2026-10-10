"""Tests for the pure logic in practitioner_view.py -- queue loading/
filtering, heatmap lookup, and the overrides log -- none of which need
Streamlit's runtime."""
import pandas as pd
import pytest
from PIL import Image

from src.review_app.practitioner_view import (
    OVERRIDES_COLUMNS,
    append_override,
    find_heatmap_path,
    load_review_queue,
)


def _write_log(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_review_queue_returns_empty_frame_when_file_missing(tmp_path):
    queue = load_review_queue(tmp_path / "does_not_exist.csv")
    assert len(queue) == 0
    assert "requires_review" in queue.columns


def test_load_review_queue_filters_to_flagged_rows_only(tmp_path):
    log_path = tmp_path / "serving_log.csv"
    _write_log(
        log_path,
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"],
            "image_path": ["a.jpg", "b.jpg", "c.jpg"],
            "site_id": [None, None, None],
            "probability": [0.9, 0.6, 0.8],
            "uncertainty": [0.01, 0.02, 0.03],
            "label": [1, 0, 1],
            "requires_review": [True, False, True],
            "threshold_used": [0.5, 0.5, 0.5],
            "model_tag": ["m1", "m1", "m1"],
        },
    )
    queue = load_review_queue(log_path)
    assert len(queue) == 2
    assert set(queue["image_path"]) == {"a.jpg", "c.jpg"}


def test_load_review_queue_sorts_most_recent_first(tmp_path):
    log_path = tmp_path / "serving_log.csv"
    _write_log(
        log_path,
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z", "2026-01-02T00:00:00Z"],
            "image_path": ["oldest.jpg", "newest.jpg", "middle.jpg"],
            "site_id": [None, None, None],
            "probability": [0.9, 0.9, 0.9],
            "uncertainty": [0.1, 0.1, 0.1],
            "label": [1, 1, 1],
            "requires_review": [True, True, True],
            "threshold_used": [0.5, 0.5, 0.5],
            "model_tag": ["m1", "m1", "m1"],
        },
    )
    queue = load_review_queue(log_path)
    assert queue["image_path"].tolist() == ["newest.jpg", "middle.jpg", "oldest.jpg"]


def test_load_review_queue_empty_when_nothing_flagged(tmp_path):
    log_path = tmp_path / "serving_log.csv"
    _write_log(
        log_path,
        {
            "timestamp": ["2026-01-01T00:00:00Z"],
            "image_path": ["a.jpg"],
            "site_id": [None],
            "probability": [0.9],
            "uncertainty": [0.01],
            "label": [1],
            "requires_review": [False],
            "threshold_used": [0.5],
            "model_tag": ["m1"],
        },
    )
    queue = load_review_queue(log_path)
    assert len(queue) == 0


def test_find_heatmap_path_returns_path_when_present(tmp_path):
    heatmap_dir = tmp_path / "heatmaps"
    heatmap_dir.mkdir()
    Image.new("RGB", (10, 10)).save(heatmap_dir / "abc123_heatmap.png")

    result = find_heatmap_path("/some/upload/dir/abc123.jpg", heatmap_dir)
    assert result == heatmap_dir / "abc123_heatmap.png"


def test_find_heatmap_path_returns_none_when_absent(tmp_path):
    heatmap_dir = tmp_path / "heatmaps"
    heatmap_dir.mkdir()
    assert find_heatmap_path("/some/upload/dir/missing.jpg", heatmap_dir) is None


def test_append_override_creates_file_with_header(tmp_path):
    overrides_path = tmp_path / "overrides.csv"
    append_override(
        original_label=1, corrected_label=1, action="confirm",
        image_path="a.jpg", note="looks right", overrides_path=overrides_path,
    )
    df = pd.read_csv(overrides_path)
    assert len(df) == 1
    assert list(df.columns) == OVERRIDES_COLUMNS
    assert df.iloc[0]["action"] == "confirm"
    assert df.iloc[0]["note"] == "looks right"


def test_append_override_records_a_real_correction(tmp_path):
    overrides_path = tmp_path / "overrides.csv"
    append_override(
        original_label=1, corrected_label=0, action="override",
        image_path="a.jpg", overrides_path=overrides_path,
    )
    df = pd.read_csv(overrides_path)
    assert df.iloc[0]["original_label"] == 1
    assert df.iloc[0]["corrected_label"] == 0
    assert df.iloc[0]["action"] == "override"


def test_append_override_appends_without_duplicate_header(tmp_path):
    overrides_path = tmp_path / "overrides.csv"
    append_override(original_label=1, corrected_label=1, action="confirm", image_path="a.jpg", overrides_path=overrides_path)
    append_override(original_label=0, corrected_label=1, action="override", image_path="b.jpg", overrides_path=overrides_path)

    lines = overrides_path.read_text().splitlines()
    assert len(lines) == 3  # header + 2 rows
    df = pd.read_csv(overrides_path)
    assert len(df) == 2
