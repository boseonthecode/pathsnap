"""Trip data loading, validation, and discovery.

Each trip directory holds four whitespace-separated, headerless files whose row
indices are the IDs used throughout the project:

``<trip_id>.track``   ``lon lat timestamp_s`` (T, 3)
``<trip_id>.nodes``   ``lon lat`` (N, 2), row index = node_id
``<trip_id>.arcs``    ``from_node_id to_node_id`` (M, 2), row index = arc_id
``<trip_id>.route``   ``arc_id`` (R,), chronological reference arc sequence

Reference routes are labels, not verified ground truth. They may contain
repeated arcs and node discontinuities, so those conditions are reported as
diagnostics instead of rejecting an otherwise usable trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np

Partition = Literal["train", "test", "eval", "unknown"]
TRACK_COLUMNS = 3
NODE_COLUMNS = 2
ARC_COLUMNS = 2
ROUTE_COLUMNS = 1


class TripValidationError(ValueError):
    """Raised when a trip file fails structural or range validation."""


@dataclass(frozen=True)
class RouteDiagnostics:
    """Quality summary for a reference route (labels are not guaranteed clean)."""

    n_route_arcs: int
    n_unique_arcs: int
    repeated_arcs: int
    discontinuities: int

    @property
    def is_node_connected(self) -> bool:
        return self.discontinuities == 0


@dataclass(frozen=True)
class Trip:
    """Validated per-trip dataset.

    Arrays are shaped ``nodes (N, 2)`` lon/lat, ``arcs (M, 2)`` int node ids,
    ``track (T, 3)`` lon/lat/timestamp_s, ``route (R,)`` int arc ids.
    """

    trip_id: str
    partition: str
    path: Path
    nodes: np.ndarray
    arcs: np.ndarray
    track: np.ndarray
    route: np.ndarray
    bbox: np.ndarray

    @property
    def n_points(self) -> int:
        return int(self.track.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_arcs(self) -> int:
        return int(self.arcs.shape[0])

    @property
    def n_route_arcs(self) -> int:
        return int(self.route.shape[0])

    def diagnostics(self) -> RouteDiagnostics:
        return route_diagnostics(self)


def route_diagnostics(trip: Trip) -> RouteDiagnostics:
    """Count repeated arcs and node discontinuities along the reference route."""
    route = trip.route
    if route.size == 0:
        return RouteDiagnostics(0, 0, 0, 0)

    route_arcs = trip.arcs[route]
    discontinuities = int(np.count_nonzero(route_arcs[:-1, 1] != route_arcs[1:, 0]))
    n_unique = int(np.unique(route).size)
    return RouteDiagnostics(
        n_route_arcs=int(route.size),
        n_unique_arcs=n_unique,
        repeated_arcs=int(route.size - n_unique),
        discontinuities=discontinuities,
    )


def _read_numeric(path: Path, columns: int) -> np.ndarray:
    """Load a headerless whitespace-separated numeric file as float64 ``(n, columns)``."""
    if not path.is_file():
        raise TripValidationError(f"Missing required trip file: {path}")

    try:
        data = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except ValueError as exc:  # non-numeric or ragged rows
        raise TripValidationError(f"{path.name}: not a valid {columns}-column numeric file") from exc

    if data.size == 0:
        return np.empty((0, columns), dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != columns:
        raise TripValidationError(
            f"{path.name}: expected {columns} columns, found shape {data.shape}"
        )
    if not np.all(np.isfinite(data)):
        raise TripValidationError(f"{path.name}: contains non-finite values")
    return data


def _as_int_array(data: np.ndarray, name: str) -> np.ndarray:
    """Return an int64 array, rejecting non-integral values before casting."""
    if data.size and not np.all(data == np.floor(data)):
        raise TripValidationError(f"{name}: IDs must be integers")
    return data.astype(np.int64)


def _validate_coordinates(coords: np.ndarray, name: str) -> None:
    if coords.size == 0:
        return
    lon, lat = coords[:, 0], coords[:, 1]
    if np.any(lon < -180.0) or np.any(lon > 180.0):
        raise TripValidationError(f"{name}: longitude outside [-180, 180]")
    if np.any(lat < -90.0) or np.any(lat > 90.0):
        raise TripValidationError(f"{name}: latitude outside [-90, 90]")


def load_track(path: Path) -> np.ndarray:
    track = _read_numeric(path, TRACK_COLUMNS)
    _validate_coordinates(track[:, :2], path.name)
    if track.shape[0] > 1 and not np.all(np.diff(track[:, 2]) > 0.0):
        raise TripValidationError(f"{path.name}: timestamps must be strictly increasing")
    return track


def load_nodes(path: Path) -> np.ndarray:
    nodes = _read_numeric(path, NODE_COLUMNS)
    _validate_coordinates(nodes, path.name)
    return nodes


def load_arcs(path: Path) -> np.ndarray:
    arcs = _as_int_array(_read_numeric(path, ARC_COLUMNS), path.name)
    return arcs


def load_route(path: Path) -> np.ndarray:
    return _as_int_array(_read_numeric(path, ROUTE_COLUMNS), path.name)[:, 0]


def load_trip(trip_dir: str | Path, partition: str = "unknown") -> Trip:
    """Load and validate one trip directory, raising ``TripValidationError`` on failure."""
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name

    track = load_track(trip_dir / f"{trip_id}.track")
    nodes = load_nodes(trip_dir / f"{trip_id}.nodes")
    arcs = load_arcs(trip_dir / f"{trip_id}.arcs")
    route = load_route(trip_dir / f"{trip_id}.route")

    n_nodes = nodes.shape[0]
    if arcs.size and (arcs.min() < 0 or arcs.max() >= n_nodes):
        raise TripValidationError(
            f"{trip_id}.arcs: node ids must be within [0, {n_nodes})"
        )
    n_arcs = arcs.shape[0]
    if route.size and (route.min() < 0 or route.max() >= n_arcs):
        raise TripValidationError(
            f"{trip_id}.route: arc ids must be within [0, {n_arcs})"
        )

    if track.shape[0]:
        bbox = np.array(
            [
                track[:, 0].min(),
                track[:, 1].min(),
                track[:, 0].max(),
                track[:, 1].max(),
            ],
            dtype=np.float64,
        )
    elif nodes.shape[0]:
        bbox = np.array(
            [
                nodes[:, 0].min(),
                nodes[:, 1].min(),
                nodes[:, 0].max(),
                nodes[:, 1].max(),
            ],
            dtype=np.float64,
        )
    else:
        bbox = np.zeros(4, dtype=np.float64)

    return Trip(
        trip_id=trip_id,
        partition=partition,
        path=trip_dir,
        nodes=nodes,
        arcs=arcs,
        track=track,
        route=route,
        bbox=bbox,
    )


def discover_trip_dirs(partition_dir: str | Path) -> list[Path]:
    """Return trip directories in a partition, sorted by trip id."""
    partition_dir = Path(partition_dir)
    if not partition_dir.is_dir():
        raise FileNotFoundError(f"Trip partition directory not found: {partition_dir}")
    return sorted(
        path
        for path in partition_dir.iterdir()
        if path.is_dir() and (path / f"{path.name}.track").is_file()
    )


def discover_trip_ids(partition_dir: str | Path) -> list[str]:
    return [path.name for path in discover_trip_dirs(partition_dir)]


def _infer_partition(trip_dir: Path) -> str:
    parent = trip_dir.resolve().parent.name
    return parent if parent in {"train", "test", "eval"} else "unknown"


def iter_trips(partition_dir: str | Path, partition: str | None = None) -> Iterator[Trip]:
    """Yield validated trips from a partition directory one at a time."""
    for trip_dir in discover_trip_dirs(partition_dir):
        yield load_trip(trip_dir, partition or _infer_partition(trip_dir))
