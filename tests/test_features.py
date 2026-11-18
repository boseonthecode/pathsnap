"""Tests for deterministic feature extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pathsnap.candidates import generate_candidates
from pathsnap.features import FEATURE_NAMES, extract_features
from pathsnap.io import load_trip
from conftest import write_trip

EXPECTED_FEATURES = (
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


def test_feature_order_is_documented_and_stable() -> None:
    assert FEATURE_NAMES == EXPECTED_FEATURES


def test_feature_matrix_shape_and_finiteness(tiny_trip_dir: Path) -> None:
    trip = load_trip(tiny_trip_dir)
    candidates = generate_candidates(trip)

    matrix = extract_features(trip, candidates)

    assert matrix.n_rows == len(candidates.candidates)
    assert matrix.n_features == len(EXPECTED_FEATURES)
    assert matrix.features.shape == (matrix.n_rows, len(EXPECTED_FEATURES))
    assert np.all(np.isfinite(matrix.features))
    assert matrix.observation_indices.shape == (matrix.n_rows,)
    assert matrix.arc_ids.shape == (matrix.n_rows,)
    assert np.all(matrix.ranks >= 0)


def test_feature_values_match_candidate_metadata(tiny_trip_dir: Path) -> None:
    trip = load_trip(tiny_trip_dir)
    candidates = generate_candidates(trip, top_k=3)
    matrix = extract_features(trip, candidates)

    for row, candidate in enumerate(candidates.candidates):
        assert matrix.features[row, 0] == candidate.distance_m
        assert matrix.features[row, 4] == candidate.rank
        assert matrix.features[row, 5] == candidate.along_fraction
        assert matrix.features[row, 6] == candidate.arc_length_m
        assert matrix.features[row, 7] > 0.0
        assert matrix.arc_ids[row] == candidate.arc_id


def test_features_are_identical_across_runs(tiny_trip_dir: Path) -> None:
    trip = load_trip(tiny_trip_dir)

    first = extract_features(trip, generate_candidates(trip))
    second = extract_features(trip, generate_candidates(trip))

    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.observation_indices, second.observation_indices)
    np.testing.assert_array_equal(first.arc_ids, second.arc_ids)


def test_empty_candidate_set_yields_empty_matrix(tmp_path: Path) -> None:
    far_track = np.array([[0.01, 0.01, 1.0], [0.02, 0.02, 2.0]])
    trip = load_trip(write_trip(tmp_path / "train", "00000084", track=far_track))
    candidates = generate_candidates(trip)

    matrix = extract_features(trip, candidates)

    assert matrix.features.shape == (0, len(EXPECTED_FEATURES))
    assert matrix.observation_indices.size == 0
