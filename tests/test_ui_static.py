"""Static smoke tests for the build-free Leaflet UI."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from pathsnap import config

UI_DIR = config.PROJECT_ROOT / "ui"


def _read(name: str) -> str:
    return (UI_DIR / name).read_text(encoding="utf-8")


def test_index_contains_required_controls_and_map() -> None:
    html = _read("index.html")

    for element_id in (
        "map",
        "trip-select",
        "method-select",
        "noise-slider",
        "sample-slider",
        "metrics",
        "rerun-button",
        "health",
    ):
        assert f'id="{element_id}"' in html
    assert "app.js" in html and "config.js" in html and "leaflet" in html.lower()


def test_ui_config_exposes_api_base_url() -> None:
    config_js = _read("config.js")
    assert "apiBaseUrl" in config_js
    assert "http://localhost:8000" in config_js


def test_app_sends_expected_prediction_parameters_and_guards_stale_responses() -> None:
    app_js = _read("app.js")

    for token in (
        "/trips",
        "/predict",
        "trip_id",
        "method",
        "noise_sigma_m",
        "sample_every_s",
        "seed",
        "requestSeq",
        "fitBounds",
        "clearLayers",
    ):
        assert token in app_js
    assert "debounce" in app_js


def test_ui_contains_no_reference_route_or_label_payload() -> None:
    combined = "\n".join(
        _read(name) for name in ("index.html", "app.js", "config.js")
    ).lower()
    for forbidden in ("reference_route", "route_arcs", "ground_truth", "labels"):
        assert forbidden not in combined


def test_styles_define_metric_and_legend_classes() -> None:
    css = _read("style.css")
    assert "metrics__row" in css
    assert "legend__swatch" in css
    assert "panel__section" in css


def test_javascript_syntax_is_valid_when_node_is_available() -> None:
    node = shutil.which("node")
    if node is None:
        return
    completed = subprocess.run(
        [node, "--check", str(UI_DIR / "app.js")],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
