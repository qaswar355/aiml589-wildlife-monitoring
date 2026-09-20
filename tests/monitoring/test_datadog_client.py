"""Tests for the pure payload-construction logic in datadog_client.py.
push_metrics itself (a real network call) is exercised by a real live
submission instead of a unit test -- see the drift-report run notes."""
from datadog_api_client.v2.model.metric_intake_type import MetricIntakeType

from src.monitoring.datadog_client import build_metric_series


def test_build_metric_series_one_series_per_metric():
    series = build_metric_series({"a.b": 1.0, "a.c": 2.5}, tags=["env:test"], timestamp=1000)
    assert len(series) == 2
    names = {s.metric for s in series}
    assert names == {"a.b", "a.c"}


def test_build_metric_series_uses_gauge_type_and_given_timestamp():
    series = build_metric_series({"a.b": 1.0}, tags=[], timestamp=1234)
    s = series[0]
    assert s.type == MetricIntakeType.GAUGE
    assert len(s.points) == 1
    assert s.points[0].timestamp == 1234
    assert s.points[0].value == 1.0


def test_build_metric_series_applies_the_same_tags_to_every_series():
    series = build_metric_series({"a.b": 1.0, "a.c": 2.0}, tags=["env:test", "tag:x"], timestamp=1000)
    for s in series:
        assert list(s.tags) == ["env:test", "tag:x"]


def test_build_metric_series_defaults_timestamp_to_now_when_omitted():
    import time

    before = int(time.time())
    series = build_metric_series({"a.b": 1.0}, tags=[])
    after = int(time.time())
    assert before <= series[0].points[0].timestamp <= after
