"""Tests for the plain-confidence vs MC Dropout comparison -- pure numpy,
no model needed."""
import numpy as np
import pytest

from src.inference.confidence_baseline import compare, errors_caught, plain_confidence_score


def test_plain_confidence_score_is_highest_at_fifty_fifty():
    assert plain_confidence_score(np.array([0.0, 0.5, 1.0, 0.8])) == pytest.approx([0.0, 0.5, 0.0, 0.2])


def test_errors_caught_counts_errors_among_the_top_scores():
    score = np.array([0.9, 0.8, 0.1, 0.05])
    is_error = np.array([1, 0, 1, 0])
    assert errors_caught(score, is_error, 0.5) == pytest.approx(0.5)  # top 2 hold 1 of 2 errors
    assert errors_caught(score, is_error, 1.0) == pytest.approx(1.0)


def test_compare_gives_both_methods_the_same_errors():
    probability = np.array([0.99, 0.6, 0.02, 0.45])
    is_error = np.array([0, 1, 0, 1])
    mc = np.array([0.001, 0.2, 0.0, 0.3])
    r = compare(mc, probability, is_error)
    assert r["mc_dropout"]["auc"] == pytest.approx(1.0)
    assert r["plain_confidence"]["auc"] == pytest.approx(1.0)
    assert set(r["plain_confidence"]) == {"auc", "errors_caught_at_10pct", "errors_caught_at_20pct"}
