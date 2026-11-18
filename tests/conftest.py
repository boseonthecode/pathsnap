"""Shared pytest fixtures and tiny synthetic trip builders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

TINY_NODES = np.array(
    [[0.0000, 0.0000], [0.0000, 0.0010], [0.0010, 0.0010], [0.0010, 0.0000]],
    dtype=np.float64,
)
TINY_ARCS = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
TINY_TRACK = np.array(
    [[0.0001, 0.0002, 1.0], [0.0005, 0.0009, 2.0], [0.0010, 0.0008, 3.0]],
    dtype=np.float64,
)
TINY_ROUTE = np.array([0, 1, 2], dtype=np.int64)


def write_trip(
    root: Path,
    trip_id: str,
    *,
    nodes: np.ndarray = TINY_NODES,
    arcs: np.ndarray = TINY_ARCS,
    track: np.ndarray = TINY_TRACK,
    route: np.ndarray = TINY_ROUTE,
    track_format: str = "%.6f",
) -> Path:
    """Write a valid four-file trip directory and return its path."""
    trip_dir = root / trip_id
    trip_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(trip_dir / f"{trip_id}.track", track, fmt=track_format)
    np.savetxt(trip_dir / f"{trip_id}.nodes", nodes, fmt="%.6f")
    np.savetxt(trip_dir / f"{trip_id}.arcs", arcs, fmt="%d")
    np.savetxt(trip_dir / f"{trip_id}.route", route, fmt="%d")
    return trip_dir


@pytest.fixture
def tiny_trip_dir(tmp_path: Path) -> Path:
    return write_trip(tmp_path / "train", "00000099")
