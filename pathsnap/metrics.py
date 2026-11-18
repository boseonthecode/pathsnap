"""Classification and route-level metrics.

Observation metrics are computed on alignable observations only. Route metrics
use sequence alignment over route positions, so repeated arcs (out-and-back
routes) are compared by occurrence rather than by arc-id set membership.
"""

from __future__ import annotations

import time
from difflib import SequenceMatcher

import numpy as np
from sklearn.metrics import (
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)


class Timer:
    """Context-manager wall-clock timer reporting milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


def observation_top1_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alignable: np.ndarray | None = None,
) -> float:
    """Fraction of alignable observations whose top-1 arc matches the label."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if alignable is not None:
        mask = np.asarray(alignable, dtype=bool)
        y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def candidate_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro F1 for binary candidate classification (labels ``0`` and ``1``)."""
    return float(
        f1_score(
            np.asarray(y_true),
            np.asarray(y_pred),
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
    )


def candidate_precision_recall(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float]:
    """Precision/recall for the positive class (candidate is the true arc)."""
    precision, recall, _, _ = precision_recall_fscore_support(
        np.asarray(y_true),
        np.asarray(y_pred),
        labels=[0, 1],
        pos_label=1,
        average="binary",
        zero_division=0,
    )
    return float(precision), float(recall)


def candidate_log_loss(y_true: np.ndarray, probability_true: np.ndarray) -> float:
    """Binary log loss matching scikit-learn's definition for labels [0, 1]."""
    y_true = np.asarray(y_true)
    probability_true = np.asarray(probability_true, dtype=np.float64)
    if y_true.size == 0:
        return 0.0
    probabilities = np.column_stack([1.0 - probability_true, probability_true])
    return float(log_loss(y_true, probabilities, labels=[0, 1]))


def alignment_coverage(alignable: np.ndarray) -> float:
    """Fraction of observations that received an alignment label (zero-safe)."""
    alignable = np.asarray(alignable, dtype=bool)
    if alignable.size == 0:
        return 0.0
    return float(np.mean(alignable))


def route_continuity(route: np.ndarray, arcs: np.ndarray) -> float:
    """Fraction of adjacent predicted arcs that share a node (zero-safe)."""
    route = np.asarray(route)
    if route.size < 2:
        return 1.0
    route_pairs = arcs[route]
    connected = int(np.count_nonzero(route_pairs[:-1, 1] == route_pairs[1:, 0]))
    return float(connected / (route.size - 1))


def _matching_blocks(predicted: np.ndarray, reference: np.ndarray):
    matcher = SequenceMatcher(
        a=np.asarray(predicted).astype(np.int64).tolist(),
        b=np.asarray(reference).astype(np.int64).tolist(),
        autojunk=False,
    )
    return matcher.get_matching_blocks()


def route_precision_recall(
    predicted: np.ndarray, reference: np.ndarray
) -> tuple[float, float]:
    """Sequence-aligned precision/recall over route positions."""
    predicted = np.asarray(predicted)
    reference = np.asarray(reference)
    if predicted.size == 0 and reference.size == 0:
        return 1.0, 1.0
    if predicted.size == 0:
        return 0.0, 0.0
    if reference.size == 0:
        return 0.0, 0.0

    matched = sum(block.size for block in _matching_blocks(predicted, reference))
    return float(matched / predicted.size), float(matched / reference.size)


def route_mismatch_fraction(
    predicted: np.ndarray,
    reference: np.ndarray,
    arc_lengths_m: np.ndarray,
) -> float:
    """Wrongly added plus missed route length divided by reference route length.

    Matched arcs come from a sequence alignment over route positions, so
    repeated arcs are matched occurrence-by-occurrence.
    """
    predicted = np.asarray(predicted)
    reference = np.asarray(reference)
    arc_lengths_m = np.asarray(arc_lengths_m, dtype=np.float64)

    reference_length = float(arc_lengths_m[reference].sum()) if reference.size else 0.0
    if reference_length == 0.0:
        return 0.0 if predicted.size == 0 else 1.0

    matched_reference = np.zeros(reference.size, dtype=bool)
    matched_predicted = np.zeros(predicted.size, dtype=bool)
    for block in _matching_blocks(predicted, reference):
        if block.size == 0:
            continue
        matched_predicted[block.a : block.a + block.size] = True
        matched_reference[block.b : block.b + block.size] = True

    missing_length = float(arc_lengths_m[reference[~matched_reference]].sum())
    extra_length = float(arc_lengths_m[predicted[~matched_predicted]].sum())
    return float((missing_length + extra_length) / reference_length)
