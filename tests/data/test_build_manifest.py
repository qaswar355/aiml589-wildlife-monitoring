"""Tests for season assignment, site-aware splitting, and event-capped sampling."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.build_manifest import (
    MANIFEST_COLUMNS,
    assign_seasons,
    sample_events_per_species,
    site_split,
)

MANIFEST_PATH = Path("data/manifest.csv")


@pytest.mark.parametrize(
    ("dt", "expected_season"),
    [
        ("2023-06-15 10:00:00", "winter"),  # June -> winter 
        ("2023-12-01 10:00:00", "summer"),  # December -> summer
        ("2023-09-01 10:00:00", "spring"),
        ("2023-01-15 10:00:00", "summer"),
        ("2023-03-01 10:00:00", "autumn"),
        ("2023-08-31 23:59:59", "winter"),
    ],
)
def test_southern_hemisphere_season_mapping(dt, expected_season):
    result = assign_seasons(pd.Series([dt]))
    assert result.iloc[0] == expected_season


def test_missing_or_unparseable_timestamp_gives_no_season():
    result = assign_seasons(pd.Series([None, "not-a-date"]))
    assert result.isna().all()


def test_site_split_has_no_leakage():
    rng = np.random.default_rng(0)
    n = 2000
    locations = rng.integers(0, 50, size=n).astype(str)
    df = pd.DataFrame({"location": locations})
    split = site_split(df, group_col="location", test_size=0.2, val_size=0.1, seed=42)

    per_location_splits = pd.DataFrame({"location": locations, "split": split.values}).groupby(
        "location"
    )["split"].nunique()
    assert (per_location_splits == 1).all()

    props = split.value_counts(normalize=True)
    assert {"train", "val", "test"} <= set(props.index)
    assert 0.5 < props["train"] < 0.85


def test_site_split_unknown_location_is_a_single_group():
    locations = ["unknown"] * 100 + [f"L{i}" for i in range(400)]
    df = pd.DataFrame({"location": locations})
    split = site_split(df, group_col="location", test_size=0.2, val_size=0.1, seed=1)
    unknown_splits = split[df["location"] == "unknown"]
    assert unknown_splits.nunique() == 1


def test_sample_events_per_species_respects_cap():
    rows = [
        {"species": "mouse", "event_id": f"evt{i}", "image_id": f"m{i}", "_select_conf": 1.0 - i * 0.01}
        for i in range(10)
    ] + [
        {"species": "kiwi", "event_id": f"kevt{i}", "image_id": f"k{i}", "_select_conf": 0.9}
        for i in range(3)
    ]
    df = pd.DataFrame(rows)

    out = sample_events_per_species(df, max_images_per_species=5, frames_per_event=1)
    counts = out["species"].value_counts()
    assert counts["mouse"] == 5  # capped
    assert counts["kiwi"] == 3  # under the cap — kept in full, same rule applied


def test_sample_events_per_species_prefers_highest_confidence():
    rows = [
        {"species": "mouse", "event_id": "e1", "image_id": "a", "_select_conf": 0.9},
        {"species": "mouse", "event_id": "e2", "image_id": "b", "_select_conf": 0.1},
    ]
    df = pd.DataFrame(rows)
    out = sample_events_per_species(df, max_images_per_species=1, frames_per_event=1)
    assert out["image_id"].tolist() == ["a"]


def test_sample_events_per_species_takes_top_n_frames_per_event():
    rows = [
        {"species": "mouse", "event_id": "e1", "image_id": "a", "_select_conf": 0.9},
        {"species": "mouse", "event_id": "e1", "image_id": "b", "_select_conf": 0.5},
        {"species": "mouse", "event_id": "e1", "image_id": "c", "_select_conf": 0.1},
    ]
    df = pd.DataFrame(rows)
    out = sample_events_per_species(df, max_images_per_species=10, frames_per_event=2)
    assert sorted(out["image_id"].tolist()) == ["a", "b"]


@pytest.mark.skipif(
    not MANIFEST_PATH.exists(),
    reason="requires a built data/manifest.csv (run python -m src.data.build_manifest)",
)
def test_real_manifest_has_no_site_leakage():
    m = pd.read_csv(MANIFEST_PATH)
    per_location = m.groupby("location")["split"].nunique()
    assert (per_location == 1).all()


@pytest.mark.skipif(
    not MANIFEST_PATH.exists(),
    reason="requires a built data/manifest.csv (run python -m src.data.build_manifest)",
)
def test_real_manifest_has_no_event_leakage():
    m = pd.read_csv(MANIFEST_PATH)
    per_event = m.groupby("event_id")["split"].nunique()
    assert (per_event == 1).all()


@pytest.mark.skipif(
    not MANIFEST_PATH.exists(),
    reason="requires a built data/manifest.csv (run python -m src.data.build_manifest)",
)
def test_real_manifest_schema():
    m = pd.read_csv(MANIFEST_PATH, nrows=5)
    assert list(m.columns) == MANIFEST_COLUMNS
