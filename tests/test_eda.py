"""Tests for the EDA summary, report, and Folium map."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pathsnap.eda import (
    EDA_COLUMNS,
    build_eda_map,
    build_eda_table,
    trip_summary,
    write_eda_report,
)
from pathsnap.io import load_trip
from conftest import write_trip

FAR_TRACK = np.array(
    [[0.0100, 0.0100, 1.0], [0.0101, 0.0101, 2.0], [0.0102, 0.0102, 3.0]],
    dtype=np.float64,
)


def test_summary_has_stable_columns_and_no_nulls(tiny_trip_dir: Path) -> None:
    frame = build_eda_table([load_trip(tiny_trip_dir, partition="train")])

    assert tuple(frame.columns) == EDA_COLUMNS
    assert len(frame) == 1
    assert frame.loc[0, "trip_id"] == "00000099"
    assert frame.loc[0, "route_gps_within50_frac"] == pytest.approx(1.0)
    assert not frame.loc[0, "severe_mismatch"]
    assert pd.notna(frame.loc[0, "route_track_ratio"])


def test_far_track_is_flagged_as_severe_mismatch(tmp_path: Path) -> None:
    trip_dir = write_trip(tmp_path / "train", "00000033", track=FAR_TRACK)

    summary = trip_summary(load_trip(trip_dir, partition="train"))

    assert summary["route_gps_within50_frac"] < 0.5
    assert summary["severe_mismatch"] is True


def test_report_flags_known_anomalies(tmp_path: Path) -> None:
    trips = [
        load_trip(write_trip(tmp_path / "train", "00000033", track=FAR_TRACK)),
        load_trip(write_trip(tmp_path / "train", "00000054", track=FAR_TRACK)),
        load_trip(write_trip(tmp_path / "train", "00000001")),
    ]
    frame = build_eda_table(trips)
    report_path = tmp_path / "eda_report.md"
    write_eda_report(frame, report_path)

    text = report_path.read_text(encoding="utf-8")
    assert "00000033" in text
    assert "00000054" in text
    assert "severe GPS-to-route mismatch" in text


def test_folium_map_contains_required_layers(tiny_trip_dir: Path, tmp_path: Path) -> None:
    out = build_eda_map(load_trip(tiny_trip_dir), tmp_path / "eda.html")

    html = out.read_text(encoding="utf-8")
    assert "GPS observations" in html
    assert "Road network" in html
    assert "Reference route" in html


@pytest.mark.slow
def test_eda_over_provided_dataset_has_90_rows() -> None:
    from itertools import chain

    from pathsnap import config
    from pathsnap.io import iter_trips

    frame = build_eda_table(
        chain(
            iter_trips(config.TRAIN_DATA_DIR, "train"),
            iter_trips(config.TEST_DATA_DIR, "test"),
        )
    )

    assert len(frame) == 90
    required = [
        "trip_id",
        "n_points",
        "n_nodes",
        "n_arcs",
        "n_route_arcs",
        "route_gps_within50_frac",
        "route_discontinuities",
        "repeated_route_arcs",
    ]
    assert frame[required].notna().all().all()
    assert int((frame["route_discontinuities"] > 0).sum()) == 11
    assert int((frame["repeated_route_arcs"] > 0).sum()) == 10
    assert set(frame.loc[frame["severe_mismatch"], "trip_id"]) >= {
        "00000033",
        "00000054",
    }
