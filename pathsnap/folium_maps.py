"""Folium and static map exports for development/report use only.

These maps may show the reference route because they are generated offline for
student-testing trips; they are never served by the evaluation API.
"""

from __future__ import annotations

from pathlib import Path

import folium
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from pathsnap.artifacts import ModelArtifact
from pathsnap.geometry import build_route_polyline
from pathsnap.hmm import HMMConfig
from pathsnap.io import Trip
from pathsnap.pipeline import PredictionResult, predict_trip


def _network_segments(
    trip: Trip, max_arcs: int = 1500
) -> list[list[tuple[float, float]]]:
    if trip.n_arcs == 0:
        return []
    indices = np.linspace(
        0, trip.n_arcs - 1, num=min(max_arcs, trip.n_arcs), dtype=int
    )
    segments = []
    for arc_id in np.unique(indices):
        from_node, to_node = trip.arcs[arc_id]
        segments.append(
            [
                (float(trip.nodes[from_node, 0]), float(trip.nodes[from_node, 1])),
                (float(trip.nodes[to_node, 0]), float(trip.nodes[to_node, 1])),
            ]
        )
    return segments


def _reference_parts(trip: Trip) -> list[np.ndarray]:
    polyline = build_route_polyline(trip, bridge=True)
    return list(polyline.parts)


def build_prediction_map(
    trip: Trip,
    result: PredictionResult,
    out_path: Path,
    max_network_arcs: int = 1500,
) -> Path:
    center = [
        float((trip.bbox[1] + trip.bbox[3]) / 2.0),
        float((trip.bbox[0] + trip.bbox[2]) / 2.0),
    ]
    fmap = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    network_layer = folium.FeatureGroup(name="Road network")
    for segment in _network_segments(trip, max_network_arcs):
        folium.PolyLine(
            [(lat, lon) for lon, lat in segment],
            color="#7f7f7f",
            weight=1,
            opacity=0.5,
        ).add_to(network_layer)
    network_layer.add_to(fmap)

    gps_layer = folium.FeatureGroup(name="GPS observations")
    for lon, lat, _ in result.effective_track:
        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=2,
            color="#d62728",
            fill=True,
            fill_opacity=0.85,
        ).add_to(gps_layer)
    gps_layer.add_to(fmap)

    reference_layer = folium.FeatureGroup(name="Reference route")
    for part in _reference_parts(trip):
        folium.PolyLine(
            [[float(lat), float(lon)] for lon, lat in part],
            color="#1f77b4",
            weight=3,
            opacity=0.8,
            dash_array="6",
        ).add_to(reference_layer)
    reference_layer.add_to(fmap)

    predicted_layer = folium.FeatureGroup(name="Predicted route")
    folium.PolyLine(
        [[float(lat), float(lon)] for lon, lat in result.route_coords],
        color="#2ca02c",
        weight=5,
        opacity=0.95,
    ).add_to(predicted_layer)
    predicted_layer.add_to(fmap)

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 14px; border-radius: 6px;
                font-size: 13px; box-shadow: 0 1px 4px rgba(0,0,0,.3);">
      <b>Layers</b><br>
      <span style="color:#d62728;">&#9679;</span> GPS observations<br>
      <span style="color:#7f7f7f;">&#9473;</span> Road network<br>
      <span style="color:#1f77b4;">&#9473;</span> Reference route (dev)<br>
      <span style="color:#2ca02c;">&#9473;</span> Predicted route
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(fmap)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    return out_path


def build_static_preview(
    trip: Trip,
    result: PredictionResult,
    out_path: Path,
    max_network_arcs: int = 1500,
) -> Path:
    network = _network_segments(trip, max_network_arcs)
    network_segments = [
        [(lon, lat) for lon, lat in segment] for segment in network
    ]

    fig, axis = plt.subplots(figsize=(8, 8))
    if network_segments:
        axis.add_collection(
            LineCollection(network_segments, colors="#9a9a9a", linewidths=0.5, alpha=0.5)
        )
    for part in _reference_parts(trip):
        axis.plot(part[:, 0], part[:, 1], color="#1f77b4", linestyle="--", linewidth=2, label="Reference route")
    track = result.effective_track
    axis.scatter(track[:, 0], track[:, 1], s=8, color="#d62728", label="GPS observations")
    if result.route_coords.size:
        axis.plot(
            result.route_coords[:, 0],
            result.route_coords[:, 1],
            color="#2ca02c",
            linewidth=3,
            label="Predicted route",
        )

    axis.set_title(f"PathSnap prediction — trip {trip.trip_id}")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.2)
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def predict_for_map(
    trip: Trip,
    artifact: ModelArtifact,
    hmm_config: HMMConfig,
    method: str = "hmm",
) -> PredictionResult:
    return predict_trip(trip, artifact, method=method, hmm_config=hmm_config)
