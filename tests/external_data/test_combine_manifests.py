"""Tests for combine_manifests.py's pure merging/reconciliation logic,
using small synthetic manifests rather than the real ~60k-row files."""
import pandas as pd
import pytest

from src.external_data.combine_manifests import COMBINED_COLUMNS, combine_manifests


def _write(tmp_path, name, df):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def _original_manifest(image_ids):
    return pd.DataFrame(
        {
            "image_id": image_ids,
            "file_name": image_ids,
            "species": ["rat"] * len(image_ids),
            "taxon": ["mammal"] * len(image_ids),
            "label": [1] * len(image_ids),
            "native": [False] * len(image_ids),
            "wild": [True] * len(image_ids),
            "project": ["ACC"] * len(image_ids),
            "location": ["site1"] * len(image_ids),
            "datetime": ["2026-01-01"] * len(image_ids),
            "season": ["summer"] * len(image_ids),
            "event_id": ["evt1"] * len(image_ids),
            "has_box": [True] * len(image_ids),
            "box_conf": [0.9] * len(image_ids),
            "bbox": ["[0.1,0.1,0.2,0.2]"] * len(image_ids),
            "split": ["train"] * len(image_ids),
        }
    )


def _external_manifest(image_ids):
    return pd.DataFrame(
        {
            "image_id": image_ids,
            "file_name": image_ids,
            "species": ["kaka"] * len(image_ids),
            "orig_species": ["Kaka"] * len(image_ids),
            "taxon": ["bird"] * len(image_ids),
            "label": [0] * len(image_ids),
            "native": [True] * len(image_ids),
            "wild": [True] * len(image_ids),
            "colour": ["rgb"] * len(image_ids),
            "reserve": ["POW"] * len(image_ids),
            "bait_station": ["B6"] * len(image_ids),
            "location": ["POW_B6"] * len(image_ids),
            "deployment": ["dep1"] * len(image_ids),
            "datetime": ["2026-08-01"] * len(image_ids),
            "season": ["winter"] * len(image_ids),
            "event_id": ["POW_B6|kaka|evt1"] * len(image_ids),
            "has_timestamp": [True] * len(image_ids),
            "size_kb": [500.0] * len(image_ids),
            "orig_filename": ["a.jpg"] * len(image_ids),
            "split": ["test"] * len(image_ids),
            "source": ["external_birds_and_mammals"] * len(image_ids),
            "has_box": [False] * len(image_ids),
            "bbox": [""] * len(image_ids),
        }
    )


def test_combine_manifests_concatenates_both_sources(tmp_path):
    original_path = _write(tmp_path, "orig.csv", _original_manifest(["a.jpg", "b.jpg"]))
    external_path = _write(tmp_path, "ext.csv", _external_manifest(["kaka/rgb/c.jpg"]))

    combined, stats = combine_manifests(str(original_path), str(external_path))
    assert stats["n_original"] == 2
    assert stats["n_external"] == 1
    assert stats["n_combined"] == 3
    assert set(combined.columns) == set(COMBINED_COLUMNS)


def test_combine_manifests_maps_source_to_project_for_external_rows(tmp_path):
    original_path = _write(tmp_path, "orig.csv", _original_manifest(["a.jpg"]))
    external_path = _write(tmp_path, "ext.csv", _external_manifest(["kaka/rgb/c.jpg"]))

    combined, _ = combine_manifests(str(original_path), str(external_path))
    external_row = combined[combined["image_id"] == "kaka/rgb/c.jpg"].iloc[0]
    assert external_row["project"] == "external_birds_and_mammals"
    assert external_row["box_conf"] == 0.0


def test_combine_manifests_preserves_original_rows_unchanged(tmp_path):
    original_path = _write(tmp_path, "orig.csv", _original_manifest(["a.jpg"]))
    external_path = _write(tmp_path, "ext.csv", _external_manifest(["kaka/rgb/c.jpg"]))

    combined, _ = combine_manifests(str(original_path), str(external_path))
    original_row = combined[combined["image_id"] == "a.jpg"].iloc[0]
    assert original_row["project"] == "ACC"
    assert original_row["box_conf"] == 0.9
    assert original_row["has_box"] == True  # noqa: E712


def test_combine_manifests_raises_on_overlapping_image_ids(tmp_path):
    original_path = _write(tmp_path, "orig.csv", _original_manifest(["dup.jpg"]))
    external_path = _write(tmp_path, "ext.csv", _external_manifest(["dup.jpg"]))

    with pytest.raises(ValueError, match="both manifests|BOTH manifests"):
        combine_manifests(str(original_path), str(external_path))
