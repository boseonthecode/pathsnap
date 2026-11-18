"""Tests for trip loading, validation, and discovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathsnap import config
from pathsnap.io import (
    TripValidationError,
    discover_trip_ids,
    iter_trips,
    load_trip,
    route_diagnostics,
)
from conftest import TINY_ARCS, write_trip


def test_load_trip_returns_expected_shapes_and_bbox(tiny_trip_dir: Path) -> None:
    trip = load_trip(tiny_trip_dir, partition="train")

    assert trip.trip_id == "00000099"
    assert trip.partition == "train"
    assert trip.track.shape == (3, 3)
    assert trip.nodes.shape == (4, 2)
    assert trip.arcs.shape == (4, 2)
    assert trip.route.shape == (3,)
    assert trip.n_points == 3 and trip.n_nodes == 4 and trip.n_arcs == 4
    np.testing.assert_allclose(trip.bbox, [0.0001, 0.0002, 0.0010, 0.0009])


def test_route_discontinuity_is_reported_not_fatal(tmp_path: Path) -> None:
    trip_dir = write_trip(tmp_path / "train", "00000098", route=np.array([0, 2]))

    trip = load_trip(trip_dir)
    diagnostics = route_diagnostics(trip)

    assert diagnostics.discontinuities == 1
    assert diagnostics.repeated_arcs == 0
    assert diagnostics.is_node_connected is False


def test_repeated_arcs_are_counted(tmp_path: Path) -> None:
    trip_dir = write_trip(
        tmp_path / "train", "00000097", route=np.array([0, 1, 2, 3, 0])
    )

    diagnostics = route_diagnostics(load_trip(trip_dir))

    assert diagnostics.repeated_arcs == 1
    assert diagnostics.discontinuities == 0


def test_out_of_range_route_arc_id_is_rejected(tmp_path: Path) -> None:
    trip_dir = write_trip(tmp_path / "train", "00000096")
    np.savetxt(trip_dir / "00000096.route", np.array([0, 99]), fmt="%d")

    with pytest.raises(TripValidationError, match="arc ids must be within"):
        load_trip(trip_dir)


def test_out_of_range_arc_node_id_is_rejected(tmp_path: Path) -> None:
    bad_arcs = TINY_ARCS.copy()
    bad_arcs[0, 1] = 99
    trip_dir = write_trip(tmp_path / "train", "00000095", arcs=bad_arcs)

    with pytest.raises(TripValidationError, match="node ids must be within"):
        load_trip(trip_dir)


def test_non_monotonic_timestamps_are_rejected(tmp_path: Path) -> None:
    bad_track = np.array([[0.0, 0.0, 5.0], [0.0, 0.001, 2.0]])
    trip_dir = write_trip(tmp_path / "train", "00000094", track=bad_track)

    with pytest.raises(TripValidationError, match="strictly increasing"):
        load_trip(trip_dir)


def test_non_finite_coordinates_are_rejected(tmp_path: Path) -> None:
    bad_track = np.array([[0.0, 0.0, 1.0], [0.0, np.nan, 2.0]])
    trip_dir = write_trip(tmp_path / "train", "00000093", track=bad_track)

    with pytest.raises(TripValidationError, match="non-finite"):
        load_trip(trip_dir)


def test_missing_file_raises_clear_error(tmp_path: Path) -> None:
    trip_dir = write_trip(tmp_path / "train", "00000092")
    (trip_dir / "00000092.route").unlink()

    with pytest.raises(TripValidationError, match="Missing required trip file"):
        load_trip(trip_dir)


def test_discovery_is_sorted_and_infers_partition(tmp_path: Path) -> None:
    write_trip(tmp_path / "train", "00000002")
    write_trip(tmp_path / "train", "00000001")

    assert discover_trip_ids(tmp_path / "train") == ["00000001", "00000002"]
    trips = list(iter_trips(tmp_path / "train"))
    assert [trip.trip_id for trip in trips] == ["00000001", "00000002"]
    assert all(trip.partition == "train" for trip in trips)


@pytest.mark.slow
def test_all_90_provided_trips_load_with_expected_diagnostics() -> None:
    train_trips = list(iter_trips(config.TRAIN_DATA_DIR))
    test_trips = list(iter_trips(config.TEST_DATA_DIR))

    assert len(train_trips) == 80
    assert len(test_trips) == 10

    all_trips = train_trips + test_trips
    for trip in all_trips:
        assert trip.track.shape[1] == 3
        assert trip.nodes.shape[1] == 2
        assert trip.arcs.shape[1] == 2
        assert trip.route.ndim == 1

    discontinued = [t for t in all_trips if t.diagnostics().discontinuities > 0]
    repeated = [t for t in all_trips if t.diagnostics().repeated_arcs > 0]

    assert len(discontinued) == 11
    assert len(repeated) == 10
