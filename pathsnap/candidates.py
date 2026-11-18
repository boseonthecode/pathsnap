"""Spatial candidate generation per trip.

An ``STRtree`` over projected arc geometries returns nearby arcs for each GPS
observation. Candidates are ranked by metric distance and carry the projection
point, along-arc fraction, heading difference, and arc metadata needed by the
feature and HMM stages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely import Point, STRtree, linestrings

from pathsnap import config
from pathsnap.geometry import (
    Projection,
    project_point_to_segment,
    projection_for_trip,
    smallest_angle_diff_deg,
)
from pathsnap.io import Trip

NO_HEADING_DIFF_DEG: float = 90.0


@dataclass(frozen=True)
class Candidate:
    """One ``(observation, arc)`` candidate in projected metric space."""

    trip_id: str
    obs_index: int
    arc_id: int
    distance_m: float
    proj_lon: float
    proj_lat: float
    along_fraction: float
    heading_diff_deg: float
    rank: int
    arc_length_m: float


@dataclass(frozen=True)
class CandidateSet:
    """All candidates for a trip, with miss-rate diagnostics."""

    trip_id: str
    candidates: tuple[Candidate, ...]
    n_observations: int
    n_alignable: int
    miss_count: int
    miss_rate: float

    def by_observation(self) -> dict[int, list[Candidate]]:
        grouped: dict[int, list[Candidate]] = {}
        for candidate in self.candidates:
            grouped.setdefault(candidate.obs_index, []).append(candidate)
        return grouped


def _observation_headings(track_xy: np.ndarray) -> np.ndarray:
    n_obs = track_xy.shape[0]
    if n_obs == 0:
        return np.empty(0, dtype=np.float64)
    if n_obs == 1:
        return np.array([np.nan], dtype=np.float64)
    deltas = np.diff(track_xy, axis=0)
    segment_headings = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0])) % 360.0
    headings = np.empty(n_obs, dtype=np.float64)
    headings[:-1] = segment_headings
    headings[-1] = segment_headings[-1]
    return headings


def _arc_headings(arc_coords: np.ndarray) -> np.ndarray:
    deltas = arc_coords[:, 1, :] - arc_coords[:, 0, :]
    lengths = np.linalg.norm(deltas, axis=1)
    headings = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0])) % 360.0
    headings[lengths == 0.0] = np.nan
    return headings


def _search_radii(base_radius_m: float, max_radius_m: float) -> tuple[float, ...]:
    candidates = sorted({base_radius_m, max(base_radius_m, 100.0), max_radius_m})
    return tuple(radius for radius in candidates if radius >= base_radius_m)


def generate_candidates(
    trip: Trip,
    *,
    top_k: int | None = None,
    base_radius_m: float | None = None,
    max_radius_m: float | None = None,
    alignable: np.ndarray | None = None,
) -> CandidateSet:
    """Generate top-k candidates for every observation.

    The base radius is tried first; only observations with no candidate fall
    back to wider radii (100 m, then ``max_radius_m``). Miss rate is measured
    over alignable observations when a mask is supplied.
    """
    top_k = config.CANDIDATE_TOP_K if top_k is None else top_k
    base_radius_m = config.CANDIDATE_RADIUS_M if base_radius_m is None else base_radius_m
    max_radius_m = (
        config.CANDIDATE_MAX_RADIUS_M if max_radius_m is None else max_radius_m
    )

    projection = projection_for_trip(trip)
    node_xy = projection.to_xy(trip.nodes)
    track_xy = projection.to_xy(trip.track[:, :2])

    if trip.n_arcs and track_xy.shape[0]:
        arc_coords = node_xy[trip.arcs]
        geometries = linestrings(arc_coords)
        tree: STRtree | None = STRtree(geometries)
        arc_headings = _arc_headings(arc_coords)
        arc_lengths = np.linalg.norm(
            arc_coords[:, 1, :] - arc_coords[:, 0, :], axis=1
        )
    else:
        tree, arc_coords, arc_headings, arc_lengths = None, None, None, None

    observation_headings = _observation_headings(track_xy)
    radii = _search_radii(base_radius_m, max_radius_m)

    collected: list[Candidate] = []
    miss_count = 0
    evaluated = 0

    for obs_index in range(track_xy.shape[0]):
        if alignable is not None and not alignable[obs_index]:
            continue
        evaluated += 1

        is_miss = True
        if tree is not None:
            query_point = Point(
                float(track_xy[obs_index, 0]), float(track_xy[obs_index, 1])
            )
            for radius in radii:
                nearby = tree.query(query_point, predicate="dwithin", distance=radius)
                if len(nearby) == 0:
                    continue
                is_miss = False
                collected.extend(
                    _build_candidates(
                        trip.trip_id,
                        obs_index,
                        nearby,
                        track_xy[obs_index],
                        track_xy,
                        observation_headings,
                        arc_coords,
                        arc_headings,
                        arc_lengths,
                        projection,
                        top_k,
                    )
                )
                break
        if is_miss:
            miss_count += 1

    n_alignable = int(evaluated)
    miss_rate = float(miss_count / n_alignable) if n_alignable else 0.0
    return CandidateSet(
        trip_id=trip.trip_id,
        candidates=tuple(collected),
        n_observations=trip.n_points,
        n_alignable=n_alignable,
        miss_count=miss_count,
        miss_rate=miss_rate,
    )


def _build_candidates(
    trip_id: str,
    obs_index: int,
    nearby: np.ndarray,
    point_xy: np.ndarray,
    track_xy: np.ndarray,
    observation_headings: np.ndarray,
    arc_coords: np.ndarray,
    arc_headings: np.ndarray,
    arc_lengths: np.ndarray,
    projection: Projection,
    top_k: int,
) -> list[Candidate]:
    scored: list[tuple[float, int, float, float, float]] = []
    for raw_arc_id in nearby:
        arc_id = int(raw_arc_id)
        start, end = arc_coords[arc_id]
        distance, along_fraction, projected = project_point_to_segment(
            point_xy, start, end
        )
        scored.append((distance, arc_id, along_fraction, projected[0], projected[1]))

    scored.sort(key=lambda item: (item[0], item[1]))
    selected = scored[:top_k]
    projected_xy = np.array([[item[3], item[4]] for item in selected], dtype=np.float64)
    projected_lonlat = projection.to_lonlat(projected_xy) if selected else np.empty((0, 2))

    observation_heading = observation_headings[obs_index]
    candidates: list[Candidate] = []
    for rank, (distance, arc_id, along_fraction, _, _) in enumerate(selected):
        arc_heading = arc_headings[arc_id]
        if np.isnan(observation_heading) or np.isnan(arc_heading):
            heading_diff = NO_HEADING_DIFF_DEG
        else:
            heading_diff = smallest_angle_diff_deg(
                float(observation_heading), float(arc_heading)
            )
        candidates.append(
            Candidate(
                trip_id=trip_id,
                obs_index=obs_index,
                arc_id=arc_id,
                distance_m=float(distance),
                proj_lon=float(projected_lonlat[rank, 0]),
                proj_lat=float(projected_lonlat[rank, 1]),
                along_fraction=float(along_fraction),
                heading_diff_deg=float(heading_diff),
                rank=rank,
                arc_length_m=float(arc_lengths[arc_id]),
            )
        )
    return candidates
