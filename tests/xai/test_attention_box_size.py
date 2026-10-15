"""Tests for the chance-level and grouping logic in attention_box_size.py --
pure arithmetic on boxes and scores, no model or GPU needed."""
import pandas as pd
import pytest

from src.xai.attention_box_size import (
    above_chance,
    boxcrop_chance,
    by_box_size,
    full_frame_chance,
    low_attention_birds,
)


def test_full_frame_chance_is_the_box_share_of_the_frame():
    assert full_frame_chance([0.0, 0.0, 0.5, 0.5]) == pytest.approx(0.25, abs=0.01)


def test_boxcrop_chance_with_padding_room_on_every_side():
    # 15% padding each side: the box is 1/1.3 of the crop in each direction.
    assert boxcrop_chance([0.4, 0.4, 0.2, 0.2]) == pytest.approx(1 / 1.3**2, abs=0.02)


def test_boxcrop_chance_is_higher_when_the_crop_is_clamped_at_the_edge():
    # Box touching the top-left corner: no padding fits on two sides.
    assert boxcrop_chance([0.0, 0.0, 0.2, 0.2]) > boxcrop_chance([0.4, 0.4, 0.2, 0.2])


def test_above_chance_is_zero_at_chance_and_one_when_all_inside():
    score = pd.Series([0.6, 1.0, 0.3])
    chance = pd.Series([0.6, 0.6, 0.6])
    assert above_chance(score, chance).tolist() == pytest.approx([0.0, 1.0, -0.75])


def test_by_box_size_groups_outcomes_into_size_bins():
    df = pd.DataFrame({
        "box_share": [0.005, 0.005, 0.3],
        "outcome": ["bird_correct", "mammal_correct", "bird_correct"],
        "pointing_game_score": [0.1, 0.2, 0.9],
    })
    rows = by_box_size(df)
    smallest, largest_used = rows[0], rows[5]
    assert smallest["bird_correct"] == {"n": 1, "median": pytest.approx(0.1)}
    assert smallest["mammal_correct"]["n"] == 1
    assert largest_used["bird_correct"]["median"] == pytest.approx(0.9)
    assert rows[1]["bird_correct"] == {"n": 0, "median": None}


def test_low_attention_birds_counts_small_boxes():
    df = pd.DataFrame({
        "outcome": ["bird_correct"] * 4 + ["mammal_correct"],
        "pointing_game_score": [0.05, 0.05, 0.5, 0.9, 0.01],
        "box_share": [0.01, 0.2, 0.01, 0.01, 0.01],
    })
    r = low_attention_birds(df)
    assert r["n_correct_birds"] == 4
    assert r["n_below_0.1"] == 2
    assert r["share_of_those_with_box_under_2.5pct"] == pytest.approx(0.5)
