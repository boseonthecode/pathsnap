"""Tests for STRtree candidate generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathsnap import config
from pathsnap.candidates import generate_candidates
from pathsnap.io import iter_trips, load_trip
from pathsnap.labels import align_observations
from conftest import write_trip

FAR_FROM_TINY_TRACK = np.array(
    [[0.0100, 0.0100, 1.0], [0.0101, 0.0101, 2.0], [0.0102, 0.0102, 3.0]]
)


def test_known_point_finds_true_arc_in_top_k(tiny_trip_dir: Path) -> None:
    trip = load_trip(tiny_trip_dir)

    candidate_set = generate_candidates(trip, top_k=4)
    grouped = candidate_set.by_observation()

    assert not candidate_set.miss_count
    assert grouped[0][0].arc_id == 0
    assert {candidate.arc_id for candidate in grouped[0]} == {0, 3}


def test_candidate_fields_are_valid(tiny_trip_dir: Path) -> None:
    candidate_set = generate_candidates(load_trip(tiny_trip_dir), top_k=8)

    for candidate in candidate_set.candidates:
        assert candidate.distance_m >= 0.0
        assert 0.0 <= candidate.along_fraction <= 1.0
        assert 0.0 <= candidate.heading_diff_deg <= 180.0
        assert candidate.arc_length_m > 0.0
        assert candidate.rank >= 0

    grouped = candidate_set.by_observation()
    for candidates in grouped.values():
        assert [c.rank for c in candidates] == list(range(len(candidates)))
        assert [c.distance_m for c in candidates] == sorted(
            c.distance_m for c in candidates
        )


def test_top_k_caps_candidates_per_observation(tiny_trip_dir: Path) -> None:
    candidate_set = generate_candidates(load_trip(tiny_trip_dir), top_k=2)

    for candidates in candidate_set.by_observation().values():
        assert len(candidates) <= 2


def test_empty_candidate_set_is_reported_as_miss(tmp_path: Path) -> None:
    trip_dir = write_trip(tmp_path / "train", "00000085", track=FAR_FROM_TINY_TRACK)

    candidate_set = generate_candidates(load_trip(trip_dir))

    assert candidate_set.candidates == ()
    assert candidate_set.miss_count == 3
    assert candidate_set.miss_rate == pytest.approx(1.0)


def test_miss_rate_only_counts_alignable_observations(tiny_trip_dir: Path) -> None:
    trip = load_trip(tiny_trip_dir)
    alignable = np.array([True, False, True])

    candidate_set = generate_candidates(trip, alignable=alignable)

    assert candidate_set.n_alignable == 2
    assert candidate_set.miss_count == 0
    assert candidate_set.miss_rate == pytest.approx(0.0)


@pytest.mark.slow
def test_student_test_candidate_miss_rate_below_one_percent() -> None:
    for trip in iter_trips(config.TEST_DATA_DIR, "test"):
        alignment = align_observations(trip)
        candidate_set = generate_candidates(trip, alignable=alignment.alignable)
        grouped = candidate_set.by_observation()

        assert candidate_set.miss_rate <= 0.01, trip.trip_id
        for obs_index in np.flatnonzero(alignment.alignable):
            assert int(obs_index) in grouped, (trip.trip_id, int(obs_index))


@pytest.mark.slow
def test_anomalous_training_trips_report_candidates_without_failing() -> None:
    for trip_id in ("00000033", "00000054"):
        trip = load_trip(config.TRAIN_DATA_DIR / trip_id, partition="train")
        alignment = align_observations(trip)
        candidate_set = generate_candidates(trip, alignable=alignment.alignable)

        assert candidate_set.n_alignable > 0, trip_id
        assert 0.0 <= candidate_set.miss_rate <= 1.0, trip_id
