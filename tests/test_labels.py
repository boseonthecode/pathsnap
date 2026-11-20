"""Tests for monotonic reference-route label alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathsnap import config
from pathsnap.io import iter_trips, load_trip
from pathsnap.labels import align_observations
from conftest import write_trip

OUT_AND_BACK_NODES = np.array(
    [[0.0000, 0.0000], [0.0000, 0.0010], [0.0000, 0.0020], [0.0000, 0.0030]]
)
# 0->1 outbound, 1->2 outbound, 2->1 return, 1->0 return.
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


def test_alignment_positions_are_non_decreasing_and_complete(tiny_trip_dir: Path) -> None:
    result = align_observations(load_trip(tiny_trip_dir))

    assert result.alignable.all()
    assert result.coverage == pytest.approx(1.0)
    assert result.backward_steps == 0
    assert np.all(np.diff(result.positions) >= 0)
    assert np.all(result.arc_ids >= 0)


def test_out_and_back_labels_track_route_position_not_first_occurrence(
    tmp_path: Path,
) -> None:
    trip_dir = write_trip(
        tmp_path / "train",
        "00000087",
        nodes=OUT_AND_BACK_NODES,
        arcs=OUT_AND_BACK_ARCS,
        track=OUT_AND_BACK_TRACK,
        route=np.array([0, 1, 2, 3]),
    )
    trip = load_trip(trip_dir)

    result = align_observations(trip)

    assert result.alignable.all()
    assert np.all(np.diff(result.positions) >= 0)
    assert result.positions[0] == 0
    assert result.positions[-1] == 3
    assert result.arc_ids[0] == 0
    assert result.arc_ids[-1] == 3
    assert result.backward_steps == 0


def test_unalignable_observations_get_no_label_and_are_counted(tmp_path: Path) -> None:
    track = np.array(
        [
            [0.00005, 0.0002, 1.0],
            [0.01000, 0.0100, 2.0],
            [0.00005, 0.0008, 3.0],
        ]
    )
    trip_dir = write_trip(tmp_path / "train", "00000086", track=track)

    result = align_observations(load_trip(trip_dir))

    assert result.positions[1] == -1
    assert result.arc_ids[1] == -1
    assert result.alignable.tolist() == [True, False, True]
    assert result.coverage == pytest.approx(2.0 / 3.0)
    assert result.backward_steps == 0


@pytest.mark.slow
def test_all_student_test_trips_reach_99_percent_coverage() -> None:
    for trip in iter_trips(config.TEST_DATA_DIR, "test"):
        result = align_observations(trip)
        alignable_positions = result.positions[result.alignable]

        assert result.coverage >= 0.99, trip.trip_id
        assert not result.low_coverage, trip.trip_id
        assert np.all(np.diff(alignable_positions) >= 0), trip.trip_id
        assert np.all(result.arc_ids[result.alignable] >= 0), trip.trip_id
        assert np.all(result.arc_ids[result.alignable] < trip.n_arcs), trip.trip_id


@pytest.mark.slow
def test_known_anomalous_training_trips_are_flagged_low_coverage() -> None:
    for trip_id in ("00000033", "00000054"):
        trip = load_trip(config.TRAIN_DATA_DIR / trip_id, partition="train")
        result = align_observations(trip)

        assert result.low_coverage, trip_id
        assert result.coverage < 0.5, trip_id


def test_candidate_labels_flags_matching_arcs() -> None:
    from pathsnap.labels import AlignmentResult, candidate_labels

    alignment = AlignmentResult(
        trip_id="t",
        positions=np.array([0, 1, -1], dtype=np.int32),
        arc_ids=np.array([4, 5, -1], dtype=np.int32),
        distances_m=np.zeros(3),
        alignable=np.array([True, True, False]),
        coverage=2 / 3,
        backward_steps=0,
        low_coverage=True,
    )
    labels = candidate_labels(alignment, np.array([0, 1, 2]), np.array([4, 9, 7]))
    assert labels.tolist() == [1, 0, 0]
    assert labels.dtype == np.int8


def test_candidate_labels_rejects_out_of_range_observations() -> None:
    from pathsnap.labels import AlignmentResult, candidate_labels

    alignment = AlignmentResult(
        trip_id="t",
        positions=np.array([0], dtype=np.int32),
        arc_ids=np.array([1], dtype=np.int32),
        distances_m=np.zeros(1),
        alignable=np.array([True]),
        coverage=1.0,
        backward_steps=0,
        low_coverage=False,
    )
    with pytest.raises(ValueError, match="observation index"):
        candidate_labels(alignment, np.array([5]), np.array([1]))
