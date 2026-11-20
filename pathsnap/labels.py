"""Monotonic alignment of GPS observations to reference-route positions.

The reference route is a chronological arc sequence that may revisit arcs and
overlap itself. Labels are therefore assigned to a *route position* (index into
the route array), never to a bare arc id. Alignment is a dynamic program that
enforces non-decreasing positions and emits labels only for observations whose
distance to the route is within a configurable threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pathsnap import config
from pathsnap.geometry import (
    point_to_segment_distance_matrix,
    projection_for_trip,
)
from pathsnap.io import Trip


@dataclass(frozen=True)
class AlignmentResult:
    """Per-observation route alignment for one trip.

    ``positions`` indexes ``trip.route`` and is ``-1`` for unalignable
    observations. ``arc_ids`` mirrors ``trip.route[positions]``.
    """

    trip_id: str
    positions: np.ndarray
    arc_ids: np.ndarray
    distances_m: np.ndarray
    alignable: np.ndarray
    coverage: float
    backward_steps: int
    low_coverage: bool


def _monotonic_align(cost: np.ndarray) -> np.ndarray:
    """Viterbi-style DP for non-decreasing route positions.

    ``cost`` is ``(n_observations, n_route_positions)`` distances. The
    recurrence is ``dp[t, r] = cost[t, r] + min_{r' <= r} dp[t-1, r']``; the
    prefix minimum plus an arg-min index array keeps it O(T*R).
    """
    n_obs, n_pos = cost.shape
    if n_obs == 0 or n_pos == 0:
        return np.full(n_obs, -1, dtype=np.int32)

    backpointer_dtype = np.int32
    backpointers = np.full((n_obs, n_pos), -1, dtype=backpointer_dtype)
    dp = cost[0].astype(np.float64, copy=True)
    positions_range = np.arange(n_pos)

    for t in range(1, n_obs):
        previous = dp
        prefix_min = np.minimum.accumulate(previous)

        # Index of the first position attaining each prefix minimum.
        is_new_min = np.empty(n_pos, dtype=bool)
        is_new_min[0] = True
        np.less(previous[1:], prefix_min[:-1], out=is_new_min[1:])
        prefix_argmin = np.maximum.accumulate(
            np.where(is_new_min, positions_range, 0)
        )

        dp = cost[t] + prefix_min
        backpointers[t] = prefix_argmin

    positions = np.empty(n_obs, dtype=np.int32)
    position = int(np.argmin(dp))
    for t in range(n_obs - 1, -1, -1):
        positions[t] = position
        if t > 0:
            position = int(backpointers[t, position])
    return positions


def align_observations(
    trip: Trip,
    max_distance_m: float | None = None,
) -> AlignmentResult:
    """Align every observation to a monotonic reference-route position."""
    max_distance_m = (
        config.ALIGN_MAX_DISTANCE_M if max_distance_m is None else max_distance_m
    )
    n_obs = trip.n_points
    if trip.route.size == 0 or n_obs == 0:
        empty_positions = np.full(n_obs, -1, dtype=np.int32)
        return AlignmentResult(
            trip_id=trip.trip_id,
            positions=empty_positions,
            arc_ids=empty_positions.copy(),
            distances_m=np.full(n_obs, np.inf, dtype=np.float64),
            alignable=np.zeros(n_obs, dtype=bool),
            coverage=0.0,
            backward_steps=0,
            low_coverage=True,
        )

    projection = projection_for_trip(trip)
    node_xy = projection.to_xy(trip.nodes)
    route_nodes = trip.arcs[trip.route]
    distances = point_to_segment_distance_matrix(
        projection.to_xy(trip.track[:, :2]),
        node_xy[route_nodes[:, 0]],
        node_xy[route_nodes[:, 1]],
    )

    min_distances = distances.min(axis=1)
    alignable = min_distances <= max_distance_m
    positions = np.full(n_obs, -1, dtype=np.int32)

    alignable_indices = np.flatnonzero(alignable)
    if alignable_indices.size:
        aligned = _monotonic_align(distances[alignable_indices])
        positions[alignable_indices] = aligned

    arc_ids = np.full(n_obs, -1, dtype=np.int32)
    aligned_positions = positions[alignable]
    if aligned_positions.size and not (
        0 <= aligned_positions.min() and aligned_positions.max() < trip.route.size
    ):
        raise ValueError("Aligned route positions fall outside the reference route")
    arc_ids[alignable] = trip.route[aligned_positions]

    backward_steps = int(np.count_nonzero(np.diff(positions[alignable]) < 0))
    coverage = float(np.mean(alignable)) if n_obs else 0.0

    return AlignmentResult(
        trip_id=trip.trip_id,
        positions=positions,
        arc_ids=arc_ids,
        distances_m=min_distances,
        alignable=alignable,
        coverage=coverage,
        backward_steps=backward_steps,
        low_coverage=coverage < config.LOW_COVERAGE_THRESHOLD,
    )


def align_to_inferred_route(
    trip: Trip,
    arc_ids: Sequence[int],
    observation_indices: Sequence[int],
) -> np.ndarray:
    """Monotonically align observations to an *inferred* arc sequence.

    Used to refine point predictions once the decoded route direction is known.
    It never consults the reference route, only the inferred arcs and the GPS
    track, and returns one arc id per supplied observation.
    """
    route = np.asarray(arc_ids, dtype=np.int32)
    indices = np.asarray(observation_indices, dtype=np.int64)
    if route.size == 0 or indices.size == 0:
        return np.empty(0, dtype=np.int32)

    projection = projection_for_trip(trip)
    node_xy = projection.to_xy(trip.nodes)
    endpoints = trip.arcs[route]
    costs = point_to_segment_distance_matrix(
        projection.to_xy(trip.track[indices, :2]),
        node_xy[endpoints[:, 0]],
        node_xy[endpoints[:, 1]],
    )
    return route[_monotonic_align(costs)]
