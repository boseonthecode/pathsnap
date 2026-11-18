"""Tests for dataset assembly, grouped folds, and scaler isolation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathsnap.cache import build_trip_artifacts
from pathsnap.dataset import (
    assemble_dataset,
    build_trip_dataset,
    fit_feature_scaler,
    trip_grouped_folds,
)
from pathsnap.io import load_trip
from conftest import write_trip

OUT_AND_BACK_NODES = np.array(
    [[0.0000, 0.0000], [0.0000, 0.0010], [0.0000, 0.0020], [0.0000, 0.0030]]
)
OUT_AND_BACK_ARCS = np.array([[0, 1], [1, 2], [2, 1], [1, 0]])
OUT_AND_BACK_TRACK = np.array(
    [
        [0.00005, 0.0002, 1.0],
        [0.00005, 0.0008, 2.0],
        [0.00005, 0.0018, 3.0],
        [0.00005, 0.0008, 4.0],
        [0.00005, 0.0002, 5.0],
    ]
)


def _trip_and_artifacts(tmp_path: Path, trip_id: str, partition: str = "train", **kwargs):
    trip = load_trip(write_trip(tmp_path / partition, trip_id, **kwargs), partition)
    return trip, build_trip_artifacts(trip)


def test_candidate_labels_match_aligned_arc(tmp_path: Path) -> None:
    trip, artifacts = _trip_and_artifacts(tmp_path, "00000080")

    dataset = build_trip_dataset(trip, artifacts)

    assert dataset.features.shape[0] == dataset.labels.shape[0]
    assert set(np.unique(dataset.labels)).issubset({0, 1})
    expected = (
        dataset.route_positions >= 0
    ) & (
        dataset.arc_ids
        == artifacts.alignment.arc_ids[dataset.observation_indices]
    )
    np.testing.assert_array_equal(dataset.labels, expected.astype(np.int8))


def test_unalignable_observations_are_excluded_and_counted(tmp_path: Path) -> None:
    track = np.array(
        [
            [0.00005, 0.0002, 1.0],
            [0.01000, 0.0100, 2.0],
            [0.00005, 0.0008, 3.0],
        ]
    )
    trip, artifacts = _trip_and_artifacts(tmp_path, "00000081", track=track)

    dataset = build_trip_dataset(trip, artifacts)

    assert dataset.n_unalignable_observations == 1
    assert dataset.n_missing_candidate_observations == 0
    assert 1 not in set(dataset.observation_indices.tolist())
    assert np.all(dataset.route_positions >= 0)


def test_low_coverage_training_trip_is_excluded(tmp_path: Path) -> None:
    far_track = np.array([[0.01, 0.01, 1.0], [0.02, 0.02, 2.0]])
    trip, artifacts = _trip_and_artifacts(tmp_path, "00000033", track=far_track)

    dataset = build_trip_dataset(trip, artifacts)

    assert dataset.low_coverage is True
    assert dataset.coverage < 0.9
    assert dataset.included_in_training is False


def test_student_test_partition_is_never_included_in_training(tmp_path: Path) -> None:
    trip, artifacts = _trip_and_artifacts(tmp_path, "00000082", partition="test")

    dataset = build_trip_dataset(trip, artifacts)

    assert dataset.included_in_training is False
    assert dataset.partition == "test"


def test_trip_grouped_folds_are_disjoint_and_exhaustive() -> None:
    trip_ids = ["a", "b", "c", "d", "e"]

    folds = trip_grouped_folds(trip_ids, n_splits=3, seed=7)

    assert len(folds) == 3
    seen_validation: list[str] = []
    for training, validation in folds:
        assert set(training).isdisjoint(validation)
        assert set(training) | set(validation) == set(trip_ids)
        seen_validation.extend(validation)
    assert sorted(seen_validation) == trip_ids


def test_grouped_folds_are_deterministic() -> None:
    assert trip_grouped_folds(["a", "b", "c", "d"], seed=1) == trip_grouped_folds(
        ["a", "b", "c", "d"], seed=1
    )


def test_scaler_is_fitted_on_training_trips_only(tmp_path: Path) -> None:
    train_trip, train_artifacts = _trip_and_artifacts(tmp_path, "00000083")
    test_trip, test_artifacts = _trip_and_artifacts(
        tmp_path, "00000084", partition="test"
    )
    bundle = assemble_dataset([train_trip, test_trip], [train_artifacts, test_artifacts])

    scaler = fit_feature_scaler(bundle, [train_trip.trip_id])
    train_features, _ = bundle.stacked([train_trip.trip_id])

    np.testing.assert_allclose(scaler.mean_, train_features.mean(axis=0))
    assert scaler.n_samples_seen_ == train_features.shape[0]
    assert train_features.shape[0] < len(bundle.stacked()[0])


def test_repeated_arc_labels_use_the_aligned_route_occurrence(tmp_path: Path) -> None:
    trip, artifacts = _trip_and_artifacts(
        tmp_path,
        "00000085",
        nodes=OUT_AND_BACK_NODES,
        arcs=OUT_AND_BACK_ARCS,
        track=OUT_AND_BACK_TRACK,
        route=np.array([0, 1, 2, 3]),
    )

    dataset = build_trip_dataset(trip, artifacts)
    last_obs = 4
    rows = dataset.observation_indices == last_obs

    assert artifacts.alignment.positions[last_obs] == 3
    assert np.all(dataset.route_positions[rows] == 3)
    assert np.any((dataset.arc_ids[rows] == 3) & (dataset.labels[rows] == 1))
    assert np.any((dataset.arc_ids[rows] == 0) & (dataset.labels[rows] == 0))
