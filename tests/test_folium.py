"""Tests for Folium and static prediction map exports."""

from __future__ import annotations

from pathlib import Path

from pathsnap.artifacts import build_model_artifact
from pathsnap.cache import build_trip_artifacts
from pathsnap.dataset import build_trip_dataset
from pathsnap.folium_maps import (
    build_prediction_map,
    build_static_preview,
    predict_for_map,
)
from pathsnap.hmm import HMMConfig
from pathsnap.io import load_trip
from pathsnap.models import fit_model
from conftest import write_trip

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _trip_and_artifact(tmp_path: Path):
    trip = load_trip(write_trip(tmp_path / "train", "00000043"), "train")
    dataset = build_trip_dataset(trip, build_trip_artifacts(trip))
    trained = fit_model("extra_trees", dataset.features, dataset.labels)
    return trip, build_model_artifact(trained)


def test_folium_map_contains_expected_layers(tmp_path: Path) -> None:
    trip, artifact = _trip_and_artifact(tmp_path)
    result = predict_for_map(trip, artifact, HMMConfig())

    path = build_prediction_map(trip, result, tmp_path / "map.html")

    html = path.read_text(encoding="utf-8")
    assert path.stat().st_size > 1000
    for layer in (
        "GPS observations",
        "Road network",
        "Reference route",
        "Predicted route",
    ):
        assert layer in html


def test_static_preview_is_a_non_empty_png(tmp_path: Path) -> None:
    trip, artifact = _trip_and_artifact(tmp_path)
    result = predict_for_map(trip, artifact, HMMConfig())

    path = build_static_preview(trip, result, tmp_path / "preview.png")

    assert path.is_file()
    assert path.stat().st_size > 1000
    assert path.read_bytes()[:8] == PNG_SIGNATURE
