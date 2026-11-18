"""Tests for projection, bearing, metric distance, and route polylines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathsnap.geometry import (
    arc_lengths_m,
    bearing_deg,
    build_route_polyline,
    distance_m,
    point_to_segment_distances,
    project_point_to_segment,
    projection_for_trip,
    smallest_angle_diff_deg,
)
from pathsnap.io import load_trip
from conftest import TINY_NODES, write_trip


def test_bearing_east_is_90_degrees() -> None:
    assert bearing_deg(0.0, 0.0, 0.001, 0.0) == pytest.approx(90.0, abs=1e-6)


def test_bearing_north_is_zero_degrees() -> None:
    assert bearing_deg(0.0, 0.0, 0.0, 0.001) == pytest.approx(0.0, abs=1e-6)


def test_smallest_angle_difference_wraps() -> None:
    assert smallest_angle_diff_deg(350.0, 10.0) == pytest.approx(20.0)
    assert smallest_angle_diff_deg(10.0, 350.0) == pytest.approx(20.0)


def test_geodesic_distance_one_degree_latitude() -> None:
    assert distance_m(0.0, 0.0, 0.0, 1.0) == pytest.approx(110_574.0, abs=200.0)


def test_projection_round_trip_under_one_centimetre(tiny_trip_dir: Path) -> None:
    trip = load_trip(tiny_trip_dir)
    projection = projection_for_trip(trip)
    lonlat = np.array([[8.595203, 49.844053]])

    round_trip = projection.to_lonlat(projection.to_xy(lonlat))
    error_m = distance_m(
        lonlat[0, 0], lonlat[0, 1], round_trip[0, 0], round_trip[0, 1]
    )

    assert error_m < 0.01


def test_projected_point_to_segment_distance_is_metric(tiny_trip_dir: Path) -> None:
    projection = projection_for_trip(load_trip(tiny_trip_dir))
    start = projection.to_xy(np.array([[0.0, 0.0]]))[0]
    end = projection.to_xy(np.array([[0.0, 0.001]]))[0]
    point = projection.to_xy(np.array([[0.0005, 0.0005]]))[0]

    distance, t, projected = project_point_to_segment(point, start, end)
    expected = distance_m(0.0005, 0.0005, 0.0, 0.0005)

    assert distance == pytest.approx(expected, abs=0.5)
    assert 0.0 <= t <= 1.0
    assert projected[0] == pytest.approx(start[0], abs=0.01)


def test_point_to_segment_distances_picks_nearest() -> None:
    points = np.array([[0.0, 1.0], [10.0, 1.0]])
    starts = np.array([[0.0, 0.0], [10.0, 0.0]])
    ends = np.array([[0.0, 2.0], [10.0, 2.0]])

    distances = point_to_segment_distances(points, starts, ends)

    np.testing.assert_allclose(distances, [0.0, 0.0])


def test_route_polyline_vertex_order_matches_arc_order(tiny_trip_dir: Path) -> None:
    trip = load_trip(tiny_trip_dir)

    polyline = build_route_polyline(trip)

    assert polyline.n_breaks == 0
    assert len(polyline.parts) == 1
    np.testing.assert_allclose(polyline.coords, TINY_NODES[[0, 1, 2, 3]])


def test_repeated_arc_route_preserves_both_occurrences(tmp_path: Path) -> None:
    trip_dir = write_trip(
        tmp_path / "train", "00000091", route=np.array([0, 1, 2, 3, 0])
    )
    trip = load_trip(trip_dir)

    polyline = build_route_polyline(trip)

    assert polyline.n_breaks == 0
    np.testing.assert_allclose(polyline.coords, TINY_NODES[[0, 1, 2, 3, 0, 1]])


def test_discontinuity_produces_explicit_break(tmp_path: Path) -> None:
    trip_dir = write_trip(tmp_path / "train", "00000090", route=np.array([0, 2]))
    trip = load_trip(trip_dir)

    polyline = build_route_polyline(trip, bridge=False)

    assert len(polyline.parts) == 2
    assert polyline.n_breaks == 1
    assert polyline.n_bridged == 0
    assert polyline.is_connected is False


def test_discontinuity_is_bridged_with_directed_shortest_path(tmp_path: Path) -> None:
    trip_dir = write_trip(tmp_path / "train", "00000089", route=np.array([0, 2]))
    trip = load_trip(trip_dir)

    polyline = build_route_polyline(trip, bridge=True)

    assert len(polyline.parts) == 1
    assert polyline.n_breaks == 0
    assert polyline.n_bridged == 1
    np.testing.assert_allclose(polyline.coords, TINY_NODES[[0, 1, 2, 3]])


def test_unbridgeable_discontinuity_falls_back_to_break(tmp_path: Path) -> None:
    nodes = np.array(
        [
            [0.0000, 0.0000],
            [0.0000, 0.0010],
            [0.0010, 0.0000],
            [0.0010, 0.0010],
        ]
    )
    arcs = np.array([[0, 1], [1, 0], [2, 3], [3, 2]])
    track = np.array([[0.0001, 0.0001, 1.0], [0.0009, 0.0009, 2.0]])
    trip_dir = write_trip(
        tmp_path / "train",
        "00000088",
        nodes=nodes,
        arcs=arcs,
        track=track,
        route=np.array([0, 2]),
    )
    trip = load_trip(trip_dir)

    polyline = build_route_polyline(trip, bridge=True)

    assert len(polyline.parts) == 2
    assert polyline.n_breaks == 1
    assert polyline.n_bridged == 0


def test_arc_lengths_are_metric_and_ordered(tiny_trip_dir: Path) -> None:
    trip = load_trip(tiny_trip_dir)

    lengths = arc_lengths_m(trip)

    assert lengths.shape == (4,)
    assert np.all(lengths > 0)
    assert lengths[0] == pytest.approx(lengths[1], rel=0.01)
