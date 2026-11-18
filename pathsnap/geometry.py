"""Projection, bearing, distance, and route-polyline geometry.

All distance and threshold math is metric: coordinates are projected per trip
to the UTM zone of the trip centroid (or the configured override). Bearings and
geodesic distances are computed in WGS84.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from pyproj import CRS, Geod, Transformer

from pathsnap import config
from pathsnap.io import Trip

_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class Projection:
    """Forward/inverse WGS84 ↔ projected-metres transform for one trip."""

    crs: CRS
    forward: Transformer
    inverse: Transformer

    def to_xy(self, lonlat: np.ndarray) -> np.ndarray:
        """Project ``(n, 2)`` lon/lat to ``(n, 2)`` metres."""
        if lonlat.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        x, y = self.forward.transform(lonlat[:, 0], lonlat[:, 1])
        return np.column_stack([x, y]).astype(np.float64)

    def to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        """Unproject ``(n, 2)`` metres back to ``(n, 2)`` lon/lat."""
        if xy.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        lon, lat = self.inverse.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([lon, lat]).astype(np.float64)


@dataclass(frozen=True)
class RoutePolyline:
    """Reference-route geometry split at unbridged discontinuities.

    ``parts`` holds one lon/lat chain per contiguous run. ``n_breaks`` counts
    discontinuities left as explicit breaks and ``n_bridged`` counts ones
    repaired with a directed shortest path.
    """

    parts: tuple[np.ndarray, ...]
    n_breaks: int
    n_bridged: int

    @property
    def coords(self) -> np.ndarray:
        if not self.parts:
            return np.empty((0, 2), dtype=np.float64)
        return np.vstack(self.parts)

    @property
    def is_connected(self) -> bool:
        return self.n_breaks == 0


def utm_crs_for_bbox(bbox: np.ndarray) -> CRS:
    """Return the UTM CRS for a trip bbox, using the centroid zone by default."""
    if config.UTM_ZONE_OVERRIDE is not None:
        zone = config.UTM_ZONE_OVERRIDE
    else:
        centroid_lon = float((bbox[0] + bbox[2]) / 2.0)
        zone = int((centroid_lon + 180.0) // 6.0) + 1
    centroid_lat = float((bbox[1] + bbox[3]) / 2.0)
    epsg = (32600 if centroid_lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def projection_for_trip(trip: Trip) -> Projection:
    crs = utm_crs_for_bbox(trip.bbox)
    return Projection(
        crs=crs,
        forward=Transformer.from_crs("EPSG:4326", crs, always_xy=True),
        inverse=Transformer.from_crs(crs, "EPSG:4326", always_xy=True),
    )


def bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Initial WGS84 bearing from point 1 to point 2 in ``[0, 360)``."""
    lon1_r, lat1_r, lon2_r, lat2_r = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2_r - lon1_r
    x = np.sin(dlon) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
    return float((np.degrees(np.arctan2(x, y)) + 360.0) % 360.0)


def distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Geodesic distance in metres between two WGS84 points."""
    _, _, distance = _GEOD.inv(lon1, lat1, lon2, lat2)
    return float(distance)


def smallest_angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """Smallest absolute difference between two bearings, in ``[0, 180]``."""
    diff = abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)
    return float(diff)


def project_point_to_segment(
    point_xy: np.ndarray, start_xy: np.ndarray, end_xy: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """Project a point onto a segment in projected metres.

    Returns ``(distance_m, t, projected_xy)`` where ``t`` is clipped to
    ``[0, 1]`` and zero-length segments are handled as point distances.
    """
    ab = end_xy - start_xy
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        t = 0.0
    else:
        t = float(np.clip(np.dot(point_xy - start_xy, ab) / denom, 0.0, 1.0))
    projected = start_xy + t * ab
    return float(np.linalg.norm(point_xy - projected)), t, projected


def point_to_segment_distances(
    points_xy: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Minimum projected distance from each point to a set of segments."""
    n_points = points_xy.shape[0]
    if n_points == 0 or seg_start.shape[0] == 0:
        return np.full(n_points, np.inf, dtype=np.float64)

    ab = seg_end - seg_start
    ab_sq = np.sum(ab * ab, axis=1)
    ab_sq = np.where(ab_sq == 0.0, 1.0, ab_sq)
    out = np.empty(n_points, dtype=np.float64)

    for start in range(0, n_points, chunk_size):
        stop = min(start + chunk_size, n_points)
        pts = points_xy[start:stop]
        ap = pts[:, None, :] - seg_start[None, :, :]
        t = np.clip(np.sum(ap * ab[None, :, :], axis=2) / ab_sq[None, :], 0.0, 1.0)
        closest = seg_start[None, :, :] + t[:, :, None] * ab[None, :, :]
        delta = pts[:, None, :] - closest
        out[start:stop] = np.sqrt(np.sum(delta * delta, axis=2)).min(axis=1)

    return out


def polyline_length_m(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points_xy, axis=0), axis=1)))


def arc_lengths_m(trip: Trip, projection: Projection | None = None) -> np.ndarray:
    """Metric length of every arc in file order."""
    projection = projection or projection_for_trip(trip)
    node_xy = projection.to_xy(trip.nodes)
    if trip.n_arcs == 0:
        return np.empty(0, dtype=np.float64)
    return np.linalg.norm(node_xy[trip.arcs[:, 1]] - node_xy[trip.arcs[:, 0]], axis=1)


def build_directed_graph(trip: Trip, projection: Projection | None = None) -> nx.DiGraph:
    """Directed node graph with metric edge weights; parallel arcs keep the shortest."""
    projection = projection or projection_for_trip(trip)
    node_xy = projection.to_xy(trip.nodes)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(trip.n_nodes))
    for from_node, to_node in trip.arcs:
        length = float(np.linalg.norm(node_xy[to_node] - node_xy[from_node]))
        if graph.has_edge(int(from_node), int(to_node)):
            if graph[int(from_node)][int(to_node)]["length_m"] <= length:
                continue
        graph.add_edge(int(from_node), int(to_node), length_m=length)
    return graph


def build_route_polyline(
    trip: Trip,
    projection: Projection | None = None,
    bridge: bool = False,
    graph: nx.DiGraph | None = None,
) -> RoutePolyline:
    """Build lon/lat geometry for the reference route in arc order.

    Consecutive non-connected arcs either get an explicit break (``bridge=False``)
    or are joined with a directed shortest path when one exists. Repeated arcs
    are preserved because iteration follows route positions, not unique arc ids.
    """
    if trip.route.size == 0:
        return RoutePolyline(parts=(), n_breaks=0, n_bridged=0)

    route_nodes = trip.arcs[trip.route]
    parts: list[list[int]] = []
    current: list[int] = [int(route_nodes[0, 0]), int(route_nodes[0, 1])]
    last_node = int(route_nodes[0, 1])
    n_breaks = 0
    n_bridged = 0

    graph_needed = bridge and bool(np.any(route_nodes[:-1, 1] != route_nodes[1:, 0]))
    if graph_needed and graph is None:
        graph = build_directed_graph(trip, projection)

    for from_node, to_node in route_nodes[1:]:
        from_node, to_node = int(from_node), int(to_node)
        if from_node == last_node:
            current.append(to_node)
        else:
            path: list[int] | None = None
            if graph is not None:
                try:
                    path = nx.shortest_path(graph, last_node, from_node, weight="length_m")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    path = None
            if path:
                current.extend(path[1:])
                current.append(to_node)
                n_bridged += 1
            else:
                parts.append(current)
                current = [from_node, to_node]
                n_breaks += 1
        last_node = to_node

    parts.append(current)
    lonlat_parts = tuple(trip.nodes[np.asarray(part, dtype=np.int64)] for part in parts)
    return RoutePolyline(parts=lonlat_parts, n_breaks=n_breaks, n_bridged=n_bridged)
