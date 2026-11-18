"""Tests for classification and route metrics."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import log_loss as sklearn_log_loss

from pathsnap.metrics import (
    Timer,
    alignment_coverage,
    candidate_log_loss,
    candidate_macro_f1,
    candidate_precision_recall,
    observation_top1_accuracy,
    route_continuity,
    route_mismatch_fraction,
    route_precision_recall,
)

ARCS = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
ARC_LENGTHS = np.array([10.0, 10.0, 10.0, 10.0])


def test_perfect_predictions_score_one_and_near_zero_log_loss() -> None:
    y_true = np.array([1, 0, 1, 0])
    y_pred = y_true.copy()
    probabilities = np.array([1.0, 0.0, 1.0, 0.0])

    assert observation_top1_accuracy(y_true, y_pred) == 1.0
    assert candidate_macro_f1(y_true, y_pred) == 1.0
    assert candidate_precision_recall(y_true, y_pred) == (1.0, 1.0)
    assert candidate_log_loss(y_true, probabilities) == pytest.approx(0.0, abs=1e-9)


def test_wrong_route_scores_lower() -> None:
    reference = np.array([0, 1, 2])
    predicted = np.array([0, 3, 3])
    arc_lengths = np.array([10.0, 10.0, 10.0, 10.0])

    precision, recall = route_precision_recall(predicted, reference)
    mismatch = route_mismatch_fraction(predicted, reference, arc_lengths)

    assert precision < 1.0
    assert recall < 1.0
    assert mismatch > 0.0
    assert route_continuity(predicted, ARCS) < 1.0


def test_continuity_is_one_for_contiguous_and_zero_for_broken() -> None:
    assert route_continuity(np.array([0, 1, 2]), ARCS) == 1.0
    assert route_continuity(np.array([0, 2]), ARCS) == 0.0
    assert route_continuity(np.array([0]), ARCS) == 1.0


def test_repeated_arcs_are_scored_by_sequence_position() -> None:
    reference = np.array([1, 2, 1])
    predicted = np.array([1, 1, 2])
    arc_lengths = np.ones(4)

    precision, recall = route_precision_recall(predicted, reference)
    mismatch = route_mismatch_fraction(predicted, reference, arc_lengths)

    assert precision < 1.0
    assert recall < 1.0
    assert mismatch > 0.0

    perfect_reference = np.array([1, 2, 1])
    assert route_precision_recall(perfect_reference, perfect_reference) == (1.0, 1.0)
    assert route_mismatch_fraction(
        perfect_reference, perfect_reference, arc_lengths
    ) == pytest.approx(0.0)


def test_log_loss_matches_sklearn_fixture() -> None:
    y_true = np.array([1, 0, 1, 1, 0])
    probabilities = np.array([0.9, 0.2, 0.6, 0.4, 0.1])

    ours = candidate_log_loss(y_true, probabilities)
    theirs = sklearn_log_loss(
        y_true, np.column_stack([1.0 - probabilities, probabilities]), labels=[0, 1]
    )

    assert ours == pytest.approx(theirs)


def test_observation_accuracy_uses_only_alignable_rows() -> None:
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, -1, 2])
    alignable = np.array([True, False, True])

    assert observation_top1_accuracy(y_true, y_pred, alignable) == 1.0
    assert observation_top1_accuracy(np.array([]), np.array([])) == 0.0


def test_alignment_coverage_is_zero_safe() -> None:
    assert alignment_coverage(np.array([], dtype=bool)) == 0.0
    assert alignment_coverage(np.array([True, False, True, True])) == 0.75


def test_timer_reports_positive_elapsed_time() -> None:
    with Timer() as timer:
        sum(range(1000))
    assert timer.elapsed_ms >= 0.0
