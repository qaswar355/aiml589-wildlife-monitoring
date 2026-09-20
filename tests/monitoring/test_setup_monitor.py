"""Tests for the pure Monitor-payload construction in setup_monitor.py.
create_retrain_monitor itself (a real network call) is exercised by a
real live creation instead of a unit test."""
from datadog_api_client.v1.model.monitor_type import MonitorType

from src.monitoring.setup_monitor import build_monitor


def test_build_monitor_watches_the_right_metric_and_tag():
    monitor = build_monitor("efficientnet_b3_vs_external", "someone@example.com")
    assert monitor.type == MonitorType.METRIC_ALERT
    assert "aiml589.drift.efficientnet_b3_vs_external.retrain_recommended" in monitor.query
    assert "tag:efficientnet_b3_vs_external" in monitor.tags


def test_build_monitor_message_at_mentions_the_given_email():
    monitor = build_monitor("some_tag", "student@vuw.ac.nz")
    assert "@student@vuw.ac.nz" in monitor.message


def test_build_monitor_uses_the_given_threshold_in_query_and_options():
    monitor = build_monitor("some_tag", "a@b.com", threshold=0.7)
    assert ">= 0.7" in monitor.query
    assert monitor.options.thresholds.critical == 0.7
