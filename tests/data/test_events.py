"""Tests for burst-event reconstruction."""
import pandas as pd

from src.data.events import assign_events


def _df(rows):
    return pd.DataFrame(rows, columns=["image_id", "location", "species", "datetime"])


def test_frames_within_window_merge_into_one_event():
    df = _df(
        [
            ("a", "L1", "mouse", "2023-01-01 10:00:00"),
            ("b", "L1", "mouse", "2023-01-01 10:00:45"),
        ]
    )
    out = assign_events(df, burst_window_seconds=60).set_index("image_id")
    assert out.loc["a", "event_id"] == out.loc["b", "event_id"]


def test_frames_beyond_window_split_into_new_event():
    df = _df(
        [
            ("a", "L1", "mouse", "2023-01-01 10:00:00"),
            ("b", "L1", "mouse", "2023-01-01 10:05:00"),
        ]
    )
    out = assign_events(df, burst_window_seconds=60).set_index("image_id")
    assert out.loc["a", "event_id"] != out.loc["b", "event_id"]


def test_different_species_same_location_same_time_are_different_events():
    df = _df(
        [
            ("a", "L1", "mouse", "2023-01-01 10:00:00"),
            ("b", "L1", "bird1", "2023-01-01 10:00:00"),
        ]
    )
    out = assign_events(df, burst_window_seconds=60).set_index("image_id")
    assert out.loc["a", "event_id"] != out.loc["b", "event_id"]


def test_different_location_same_species_same_time_are_different_events():
    df = _df(
        [
            ("a", "L1", "mouse", "2023-01-01 10:00:00"),
            ("b", "L2", "mouse", "2023-01-01 10:00:00"),
        ]
    )
    out = assign_events(df, burst_window_seconds=60).set_index("image_id")
    assert out.loc["a", "event_id"] != out.loc["b", "event_id"]


def test_missing_timestamp_gets_singleton_event_and_flag():
    df = _df(
        [
            ("a", "L1", "mouse", None),
            ("b", "L1", "mouse", None),
        ]
    )
    out = assign_events(df, burst_window_seconds=60).set_index("image_id")
    assert out.loc["a", "has_timestamp"] == False  # noqa: E712
    assert out.loc["b", "has_timestamp"] == False  # noqa: E712
    # never silently merged just because both are missing a timestamp
    assert out.loc["a", "event_id"] != out.loc["b", "event_id"]


def test_burst_window_is_configurable():
    df = _df(
        [
            ("a", "L1", "mouse", "2023-01-01 10:00:00"),
            ("b", "L1", "mouse", "2023-01-01 10:01:30"),  # 90s gap
        ]
    )
    same_event = assign_events(df, burst_window_seconds=120).set_index("image_id")
    assert same_event.loc["a", "event_id"] == same_event.loc["b", "event_id"]

    diff_event = assign_events(df, burst_window_seconds=60).set_index("image_id")
    assert diff_event.loc["a", "event_id"] != diff_event.loc["b", "event_id"]
