"""Dataset inventory and data-quality computations for EDA.

This module turns validated :class:`~pathsnap.io.Trip` objects into one summary
row per trip and renders a small interactive Folium overview map. Distance
calculations use a per-trip UTM projection so every threshold is metric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import folium
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.geometry import (
    Projection,
    point_to_segment_distances,
    polyline_length_m,
    projection_for_trip,
)
from pathsnap.io import Trip

COVERAGE_THRESHOLD_M: float = 50.0
SEVERE_MISMATCH_COVERAGE: float = 0.50

EDA_COLUMNS: tuple[str, ...] = (
    "partition",
    "trip_id",
    "n_points",
    "n_nodes",
    "n_arcs",
    "n_route_arcs",
    "duration_s",
    "median_dt_s",
    "max_gap_s",
    "nonunit_dt_frac",
    "bbox_min_lon",
    "bbox_min_lat",
    "bbox_max_lon",
    "bbox_max_lat",
    "route_len_m",
    "track_len_m",
    "route_track_ratio",
    "route_gps_within50_frac",
    "median_route_dist_m",
    "route_discontinuities",
    "repeated_route_arcs",
    "n_weak_components",
    "n_strong_components",
    "n_isolated_nodes",
    "severe_mismatch",
)


def route_segments_xy(
    trip: Trip, projection: Projection
) -> tuple[np.ndarray, np.ndarray]:
    """Return projected start/end points for every arc in the reference route."""
    if trip.route.size == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    node_pairs = trip.arcs[trip.route]
    return (
        projection.to_xy(trip.nodes[node_pairs[:, 0]]),
        projection.to_xy(trip.nodes[node_pairs[:, 1]]),
    )


def graph_component_summary(arcs: np.ndarray, n_nodes: int) -> tuple[int, int, int]:
    """Return (weak components, strong components, isolated nodes)."""
    if n_nodes == 0:
        return 0, 0, 0

    from_nodes, to_nodes = arcs[:, 0], arcs[:, 1]
    weights = np.ones(arcs.shape[0], dtype=np.int8)
    directed = csr_matrix((weights, (from_nodes, to_nodes)), shape=(n_nodes, n_nodes))
    undirected = directed + directed.T

    degrees = np.asarray(undirected.sum(axis=1)).ravel()
    n_isolated = int(np.count_nonzero(degrees == 0))
    n_weak, _ = connected_components(undirected, directed=False)
    n_strong, _ = connected_components(directed, directed=True, connection="strong")
    return int(n_weak), int(n_strong), n_isolated


def trip_summary(trip: Trip) -> dict[str, Any]:
    """Compute one inventory/quality row for a trip."""
    projection = projection_for_trip(trip)
    track_xy = projection.to_xy(trip.track[:, :2])
    seg_start, seg_end = route_segments_xy(trip, projection)

    route_len_m = float(np.sum(np.linalg.norm(seg_end - seg_start, axis=1))) if seg_start.size else 0.0
    track_len_m = polyline_length_m(track_xy)
    distances = point_to_segment_distances(track_xy, seg_start, seg_end)

    if trip.track.shape[0] > 1:
        dt = np.diff(trip.track[:, 2])
        median_dt_s = float(np.median(dt))
        max_gap_s = float(dt.max())
        nonunit_dt_frac = float(np.mean(np.abs(dt - 1.0) > 1e-6))
        duration_s = float(trip.track[-1, 2] - trip.track[0, 2])
    else:
        median_dt_s = max_gap_s = nonunit_dt_frac = duration_s = 0.0

    diagnostics = trip.diagnostics()
    n_weak, n_strong, n_isolated = graph_component_summary(trip.arcs, trip.n_nodes)
    coverage = (
        float(np.mean(distances <= COVERAGE_THRESHOLD_M)) if distances.size else 0.0
    )

    return {
        "partition": trip.partition,
        "trip_id": trip.trip_id,
        "n_points": trip.n_points,
        "n_nodes": trip.n_nodes,
        "n_arcs": trip.n_arcs,
        "n_route_arcs": trip.n_route_arcs,
        "duration_s": duration_s,
        "median_dt_s": median_dt_s,
        "max_gap_s": max_gap_s,
        "nonunit_dt_frac": nonunit_dt_frac,
        "bbox_min_lon": float(trip.bbox[0]),
        "bbox_min_lat": float(trip.bbox[1]),
        "bbox_max_lon": float(trip.bbox[2]),
        "bbox_max_lat": float(trip.bbox[3]),
        "route_len_m": route_len_m,
        "track_len_m": track_len_m,
        "route_track_ratio": route_len_m / track_len_m if track_len_m > 0 else np.nan,
        "route_gps_within50_frac": coverage,
        "median_route_dist_m": float(np.median(distances)) if distances.size else np.nan,
        "route_discontinuities": diagnostics.discontinuities,
        "repeated_route_arcs": diagnostics.repeated_arcs,
        "n_weak_components": n_weak,
        "n_strong_components": n_strong,
        "n_isolated_nodes": n_isolated,
        "severe_mismatch": coverage < SEVERE_MISMATCH_COVERAGE,
    }


def build_eda_table(trips: Iterable[Trip]) -> pd.DataFrame:
    """Build the one-row-per-trip EDA table with a stable column order."""
    rows = [trip_summary(trip) for trip in trips]
    frame = pd.DataFrame(rows, columns=list(EDA_COLUMNS))
    return frame


def write_eda_report(frame: pd.DataFrame, out_path: Path) -> None:
    """Write a short markdown data-quality report from the summary table."""
    discontinued = frame[frame["route_discontinuities"] > 0]
    repeated = frame[frame["repeated_route_arcs"] > 0]
    severe = frame[frame["severe_mismatch"]]

    lines = [
        "# PathSnap — EDA data-quality report",
        "",
        f"Trips inspected: **{len(frame)}** "
        f"(train={int((frame['partition'] == 'train').sum())}, "
        f"test={int((frame['partition'] == 'test').sum())}).",
        "",
        "## Inventory",
        f"- Observations: {int(frame['n_points'].sum()):,}",
        f"- Road-network arcs: {int(frame['n_arcs'].sum()):,}",
        f"- Median sampling interval: {frame['median_dt_s'].median():.3f} s",
        f"- Largest GPS gap: {frame['max_gap_s'].max():.2f} s",
        "",
        "## Reference-route quality",
        f"- Trips with node discontinuities: **{len(discontinued)}** "
        f"({', '.join(discontinued['trip_id'].head(20))})",
        f"- Trips with repeated route arcs: **{len(repeated)}** "
        f"({', '.join(repeated['trip_id'].head(20))})",
        f"- Trips with severe GPS-to-route mismatch "
        f"(coverage < {SEVERE_MISMATCH_COVERAGE:.0%}): **{len(severe)}** "
        f"({', '.join(severe['trip_id'])})",
        "",
        "Coverage is the fraction of GPS observations within "
        f"{COVERAGE_THRESHOLD_M:.0f} m of the reference route geometry. "
        "Reference routes are labels, not verified ground truth; discontinuous "
        "and repeated-arc routes are reported rather than repaired.",
        "",
        "## Assumptions",
        "- Distances use a per-trip UTM projection chosen from the trip centroid.",
        "- Reference routes may be non-contiguous; route geometry is evaluated "
        "per arc, so jumps are never bridged here.",
        "- Severe-mismatch trips are flagged for low-coverage handling in label "
        "alignment and training, not silently treated as clean labels.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_eda_map(trip: Trip, out_path: Path, max_network_arcs: int = 1500) -> Path:
    """Render a Folium map with GPS, road-network, and reference-route layers."""
    center = [
        float((trip.bbox[1] + trip.bbox[3]) / 2.0),
        float((trip.bbox[0] + trip.bbox[2]) / 2.0),
    ]
    fmap = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    gps_layer = folium.FeatureGroup(name="GPS observations")
    for lon, lat, _ in trip.track:
        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=2,
            color="#d62728",
            fill=True,
            fill_opacity=0.8,
        ).add_to(gps_layer)
    gps_layer.add_to(fmap)

    network_layer = folium.FeatureGroup(name="Road network")
    if trip.n_arcs:
        indices = np.linspace(
            0, trip.n_arcs - 1, num=min(max_network_arcs, trip.n_arcs), dtype=int
        )
        for arc_id in np.unique(indices):
            from_node, to_node = trip.arcs[arc_id]
            coords = [
                [float(trip.nodes[from_node, 1]), float(trip.nodes[from_node, 0])],
                [float(trip.nodes[to_node, 1]), float(trip.nodes[to_node, 0])],
            ]
            folium.PolyLine(coords, color="#7f7f7f", weight=1, opacity=0.6).add_to(
                network_layer
            )
    network_layer.add_to(fmap)

    reference_layer = folium.FeatureGroup(name="Reference route")
    for start, end in _route_polylines(trip):
        folium.PolyLine(
            [[float(lat), float(lon)] for lon, lat in zip(start, end)],
            color="#2ca02c",
            weight=4,
            opacity=0.9,
        ).add_to(reference_layer)
    reference_layer.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    return out_path


def _route_polylines(trip: Trip) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a route into contiguous lon/lat vertex chains at discontinuities."""
    if trip.route.size == 0:
        return []
    node_pairs = trip.arcs[trip.route]
    chains: list[list[int]] = []
    current: list[int] = [int(node_pairs[0, 0]), int(node_pairs[0, 1])]
    for from_node, to_node in node_pairs[1:]:
        if int(from_node) == current[-1]:
            current.append(int(to_node))
        else:
            chains.append(current)
            current = [int(from_node), int(to_node)]
    chains.append(current)

    polylines: list[tuple[np.ndarray, np.ndarray]] = []
    for chain in chains:
        lonlat = trip.nodes[np.asarray(chain, dtype=np.int64)]
        polylines.append((lonlat[:, 0], lonlat[:, 1]))
    return polylines
