"""Tests for the serving pipeline: track perturbation and per-trip prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathsnap.artifacts import build_model_artifact, save_model_artifact
from pathsnap.cache import build_trip_artifacts
from pathsnap.dataset import build_trip_dataset
from pathsnap.hmm import HMMConfig
from pathsnap.io import load_trip
from pathsnap.models import fit_model
from pathsnap.pipeline import (
    METRIC_LABELS,
    SUPPORTED_METHODS,
    effective_track,
    predict_directory,
    predict_trip,
    prediction_summary,
    write_prediction_outputs,
)
from conftest import write_trip


@pytest.fixture
def trip(tmp_path: Path):
    return load_trip(write_trip(tmp_path / "eval", "00000041"), "eval")


@pytest.fixture
def artifact(tmp_path: Path, trip):
    dataset = build_trip_dataset(trip, build_trip_artifacts(trip))
    trained = fit_model("logistic_regression", dataset.features, dataset.labels)
    trained_artifact = build_model_artifact(trained)
    save_model_artifact(trained_artifact, tmp_path / "artifacts" / "model.joblib")
    return trained_artifact


def test_effective_track_is_identity_when_no_noise_and_no_sampling(trip) -> None:
    track = effective_track(trip, noise_sigma_m=0.0, sample_every_s=0.0)
    np.testing.assert_allclose(track, trip.track)


def test_effective_track_downsamples_in_time_order(trip) -> None:
    track = effective_track(trip, noise_sigma_m=0.0, sample_every_s=1.5)
    assert 1 <= track.shape[0] <= trip.n_points
    assert np.all(np.diff(track[:, 2]) > 0)


def test_effective_track_noise_is_seeded(trip) -> None:
    first = effective_track(trip, noise_sigma_m=5.0, sample_every_s=0.0, seed=7)
    again = effective_track(trip, noise_sigma_m=5.0, sample_every_s=0.0, seed=7)
    other = effective_track(trip, noise_sigma_m=5.0, sample_every_s=0.0, seed=8)
    np.testing.assert_allclose(first, again)
    assert not np.allclose(first, other)


def test_predict_trip_returns_metrics_for_every_method(trip, artifact) -> None:
    expected_keys = {key for _, key in METRIC_LABELS} | {"log_loss", "rmf"}
    for method in SUPPORTED_METHODS:
        result = predict_trip(trip, artifact, method=method)
        assert result.method == method
        assert expected_keys <= set(result.metrics)
        assert result.point_arc_ids.shape == (trip.n_points,)
        assert result.route_coords.shape[1] == 2
        assert result.inference_ms >= 0.0


def test_predict_trip_rejects_unknown_method(trip, artifact) -> None:
    with pytest.raises(ValueError, match="Unsupported method"):
        predict_trip(trip, artifact, method="teleport")


def test_candidate_layer_is_capped(trip, artifact) -> None:
    from pathsnap import config

    result = predict_trip(trip, artifact)
    assert len(result.candidate_layer) <= config.MAX_CANDIDATE_LAYER_POINTS


def test_prediction_summary_counts_match_metrics(trip, artifact) -> None:
    result = predict_trip(trip, artifact)
    summary = prediction_summary(result)
    assert summary["trip_id"] == trip.trip_id
    assert summary["method"] == "hmm"
    assert summary["scored_points"] >= summary["correct_points"] >= 0
    assert summary["scored_points"] <= len(result.effective_track)


def test_write_prediction_outputs_writes_route_and_json(trip, artifact, tmp_path: Path) -> None:
    result = predict_trip(trip, artifact)
    summary = write_prediction_outputs(result, tmp_path / "out")

    route_path = tmp_path / "out" / f"{trip.trip_id}.route"
    json_path = tmp_path / "out" / f"{trip.trip_id}.json"
    assert route_path.is_file() and json_path.is_file()

    written = np.loadtxt(route_path, dtype=np.int64, ndmin=1)
    np.testing.assert_array_equal(written, result.predicted_arcs)
    assert summary["trip_id"] == trip.trip_id


def test_predict_directory_runs_every_trip(tmp_path: Path, artifact) -> None:
    data_dir = tmp_path / "eval"
    write_trip(data_dir, "00000041")
    write_trip(data_dir, "00000042")
    results = predict_directory(data_dir, artifact, method="nearest")
    assert [item.trip_id for item in results] == ["00000041", "00000042"]
    assert all(item.method == "nearest" for item in results)


def test_hmm_config_is_honoured(trip, artifact) -> None:
    strict = predict_trip(trip, artifact, hmm_config=HMMConfig(beta_m=2.0, refine_points=True))
    assert strict.metrics["route_continuity"] >= 0.0
