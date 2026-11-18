"""Canonical ``(observation, candidate)`` feature extraction.

The feature order is part of the model artifact contract: it is stored in
``artifacts/config.json`` and must not change without invalidating artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pathsnap.candidates import CandidateSet
from pathsnap.geometry import projection_for_trip
from pathsnap.io import Trip

FEATURE_NAMES: tuple[str, ...] = (
    "distance_m",
    "log_distance",
    "heading_diff_deg",
    "speed_consistency",
    "candidate_rank",
    "along_fraction",
    "arc_length_m",
    "time_gap_s",
    "from_degree",
    "to_degree",
)

MIN_TIME_GAP_S: float = 1e-6


@dataclass(frozen=True)
class FeatureMatrix:
    """Feature rows plus the identifiers needed to attach labels later."""

    trip_id: str
    feature_names: tuple[str, ...]
    observation_indices: np.ndarray
    arc_ids: np.ndarray
    ranks: np.ndarray
    features: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])


def _observation_motion(
    trip: Trip,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-observation step distance (m) and time gap (s).

    The first observation uses its forward step, the rest use the backward
    step, so every row has a local motion context even at track ends.
    """
    n_obs = trip.n_points
    step_distance_m = np.zeros(n_obs, dtype=np.float64)
    time_gap_s = np.ones(n_obs, dtype=np.float64)
    if n_obs < 2:
        return step_distance_m, time_gap_s

    projection = projection_for_trip(trip)
    track_xy = projection.to_xy(trip.track[:, :2])
    deltas = np.linalg.norm(np.diff(track_xy, axis=0), axis=1)
    gaps = np.abs(np.diff(trip.track[:, 2]))

    step_distance_m[1:] = deltas
    step_distance_m[0] = deltas[0]
    time_gap_s[1:] = gaps
    time_gap_s[0] = gaps[0]
    return step_distance_m, np.maximum(time_gap_s, MIN_TIME_GAP_S)


def _node_degrees(trip: Trip) -> np.ndarray:
    out_degree = np.bincount(trip.arcs[:, 0], minlength=trip.n_nodes)
    in_degree = np.bincount(trip.arcs[:, 1], minlength=trip.n_nodes)
    return (out_degree + in_degree).astype(np.float64)


def extract_features(trip: Trip, candidate_set: CandidateSet) -> FeatureMatrix:
    """Build the deterministic feature matrix for all candidates of a trip."""
    n_rows = len(candidate_set.candidates)
    observation_indices = np.empty(n_rows, dtype=np.int32)
    arc_ids = np.empty(n_rows, dtype=np.int32)
    ranks = np.empty(n_rows, dtype=np.int32)
    features = np.empty((n_rows, len(FEATURE_NAMES)), dtype=np.float64)

    if n_rows == 0:
        return FeatureMatrix(
            trip_id=trip.trip_id,
            feature_names=FEATURE_NAMES,
            observation_indices=observation_indices,
            arc_ids=arc_ids,
            ranks=ranks,
            features=features,
        )

    step_distance_m, time_gap_s = _observation_motion(trip)
    total_degree = _node_degrees(trip)

    for row, candidate in enumerate(candidate_set.candidates):
        obs = candidate.obs_index
        gap = time_gap_s[obs]
        gps_speed = step_distance_m[obs] / gap
        candidate_speed = candidate.arc_length_m / gap
        from_node = int(trip.arcs[candidate.arc_id, 0])
        to_node = int(trip.arcs[candidate.arc_id, 1])

        observation_indices[row] = obs
        arc_ids[row] = candidate.arc_id
        ranks[row] = candidate.rank
        features[row] = (
            candidate.distance_m,
            np.log1p(candidate.distance_m),
            candidate.heading_diff_deg,
            abs(gps_speed - candidate_speed),
            float(candidate.rank),
            candidate.along_fraction,
            candidate.arc_length_m,
            gap,
            total_degree[from_node],
            total_degree[to_node],
        )

    return FeatureMatrix(
        trip_id=trip.trip_id,
        feature_names=FEATURE_NAMES,
        observation_indices=observation_indices,
        arc_ids=arc_ids,
        ranks=ranks,
        features=features,
    )
