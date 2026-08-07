"""Tests for the pure logic in gradcam.py -- the pointing-game metric and
the false-positive filtering/joining, none of which need a real model or
GPU to check."""
import numpy as np
import pandas as pd
import pytest

from src.xai.gradcam import OUTCOME_DEFINITIONS, _load_false_positives, _seen_site_mask, pointing_game_score


def test_pointing_game_score_all_attention_inside_box():
    cam = np.zeros((10, 10), dtype=np.float32)
    cam[4:6, 4:6] = 1.0  # a hot spot right in the middle
    bbox = [0.3, 0.3, 0.4, 0.4]  # covers roughly the middle of the image
    assert pointing_game_score(cam, bbox) == pytest.approx(1.0)


def test_pointing_game_score_all_attention_outside_box():
    cam = np.zeros((10, 10), dtype=np.float32)
    cam[0, 0] = 1.0  # top-left corner
    bbox = [0.7, 0.7, 0.2, 0.2]  # bottom-right corner, nowhere near it
    assert pointing_game_score(cam, bbox) == pytest.approx(0.0)


def test_pointing_game_score_partial_overlap():
    cam = np.ones((10, 10), dtype=np.float32)  # attention spread evenly everywhere
    bbox = [0.0, 0.0, 0.5, 0.5]  # box covers exactly a quarter of the image
    assert pointing_game_score(cam, bbox) == pytest.approx(0.25, abs=0.02)


def test_pointing_game_score_zero_attention_is_zero_not_a_crash():
    cam = np.zeros((10, 10), dtype=np.float32)
    bbox = [0.0, 0.0, 1.0, 1.0]
    assert pointing_game_score(cam, bbox) == 0.0


def test_load_false_positives_filters_correctly(tmp_path):
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "image_id": ["a", "b", "c", "d"],
            "site_id": ["S1", "S1", "S2", "S2"],
            "species": ["robin", "mouse", "robin", "stoat"],
            "true_label": [0, 1, 0, 1],
            "pred_label": [1, 1, 0, 0],  # a: false positive, b: correct, c: correct, d: false negative
            "confidence": [0.9, 0.8, 0.7, 0.6],
            "probability": [0.9, 0.8, 0.3, 0.4],
        }
    ).to_csv(predictions_path, index=False)

    manifest = pd.DataFrame(
        {
            "image_id": ["a", "b", "c", "d"],
            "bbox": ['[0.1, 0.1, 0.2, 0.2]', "", '[0.5, 0.5, 0.1, 0.1]', ""],
            "has_box": [True, False, True, False],
        }
    )

    false_positives = _load_false_positives("dummy_tag", manifest, predictions_path=str(predictions_path))

    assert list(false_positives["image_id"]) == ["a"]
    assert false_positives.iloc[0]["bbox"] == [0.1, 0.1, 0.2, 0.2]


def test_seen_site_mask_matches_warm_season_locations():
    false_positives = pd.DataFrame({"site_id": ["S1", "S2", "S3"]})
    manifest = pd.DataFrame(
        {
            "location": ["S1", "S1", "S4"],
            "season": ["spring", "autumn", "summer"],  # S1 seen in warm season, S4 is warm but not in our FPs
        }
    )
    mask = _seen_site_mask(false_positives, manifest)
    assert mask.tolist() == [True, False, False]


def _sample_outcomes_df():
    # a: bird correctly classified, b: mammal correctly classified,
    # c: bird misclassified as mammal, d: mammal misclassified as bird
    return pd.DataFrame(
        {
            "image_id": ["a", "b", "c", "d"],
            "true_label": [0, 1, 0, 1],
            "pred_label": [0, 1, 1, 0],
        }
    )


def test_outcome_definitions_bird_correct():
    df = _sample_outcomes_df()
    assert df[OUTCOME_DEFINITIONS["bird_correct"](df)]["image_id"].tolist() == ["a"]


def test_outcome_definitions_mammal_correct():
    df = _sample_outcomes_df()
    assert df[OUTCOME_DEFINITIONS["mammal_correct"](df)]["image_id"].tolist() == ["b"]


def test_outcome_definitions_false_positive_is_bird_called_mammal():
    df = _sample_outcomes_df()
    assert df[OUTCOME_DEFINITIONS["false_positive"](df)]["image_id"].tolist() == ["c"]


def test_outcome_definitions_false_negative_is_mammal_called_bird():
    df = _sample_outcomes_df()
    assert df[OUTCOME_DEFINITIONS["false_negative"](df)]["image_id"].tolist() == ["d"]


def test_outcome_definitions_partition_the_whole_dataframe():
    """Every row belongs to exactly one outcome -- the four masks should
    never overlap and should never miss a row."""
    df = _sample_outcomes_df()
    masks = [mask_fn(df) for mask_fn in OUTCOME_DEFINITIONS.values()]
    coverage = sum(m.astype(int) for m in masks)
    assert coverage.tolist() == [1, 1, 1, 1]
