"""Tests for decoder comparison, beta tuning, and failure analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pathsnap.artifacts import build_model_artifact, load_model_artifact
from pathsnap.cache import build_trip_artifacts
from pathsnap.dataset import build_trip_dataset
from pathsnap.evaluate import (
    DECODER_COLUMNS,
    DECODER_TUNING_COLUMNS,
    evaluate_decoders,
    prepare_trip_inference,
    tune_decoder_betas,
    write_decoder_failure_analysis,
)
from pathsnap.hmm import HMMConfig
from pathsnap.io import iter_trips, load_trip
from pathsnap.models import fit_model
from conftest import write_trip


def _artifact_and_trips(tmp_path: Path):
    train_trip = load_trip(write_trip(tmp_path / "train", "00000044"), "train")
    dataset = build_trip_dataset(train_trip, build_trip_artifacts(train_trip))
    trained = fit_model("extra_trees", dataset.features, dataset.labels)
    artifact = build_model_artifact(trained)
    second_trip = load_trip(write_trip(tmp_path / "test", "00000045"), "test")
    return artifact, train_trip, second_trip


def test_decoder_comparison_returns_three_methods(tmp_path: Path) -> None:
    artifact, trip, _ = _artifact_and_trips(tmp_path)
    inference = prepare_trip_inference(trip, artifact)

    frame = evaluate_decoders(inference, HMMConfig(emission="classifier"))

    assert list(frame.columns) == list(DECODER_COLUMNS)
    assert set(frame["method"]) == {"nearest", "classifier", "hmm"}
    assert frame["point_accuracy"].between(0, 1).all()
    assert frame["route_continuity"].between(0, 1).all()
    assert (frame["alignment_coverage"] > 0).all()
    assert (frame["inference_ms"] >= 0).all()


def test_beta_tuning_uses_only_train_and_test_partitions(tmp_path: Path) -> None:
    artifact, train_trip, test_trip = _artifact_and_trips(tmp_path)
    inferences = [
        prepare_trip_inference(train_trip, artifact),
        prepare_trip_inference(test_trip, artifact),
    ]

    frame, best_beta = tune_decoder_betas(
        inferences, betas=(5.0, 10.0), base_config=HMMConfig()
    )

    assert list(frame.columns) == list(DECODER_TUNING_COLUMNS)
    assert len(frame) == 4
    assert set(frame["beta_m"]) == {5.0, 10.0}
    assert set(frame["partition"]).issubset({"train", "test"})
    assert best_beta in (5.0, 10.0)


def test_failure_analysis_names_trips_and_cases(tmp_path: Path) -> None:
    artifact, train_trip, test_trip = _artifact_and_trips(tmp_path)
    frames = [
        evaluate_decoders(prepare_trip_inference(trip, artifact), HMMConfig())
        for trip in (train_trip, test_trip)
    ]
    comparison = pd.concat(frames, ignore_index=True)
    out_path = tmp_path / "failure.md"

    write_decoder_failure_analysis(comparison, out_path)

    text = out_path.read_text(encoding="utf-8")
    assert train_trip.trip_id in text
    assert test_trip.trip_id in text
    assert "Case study: HMM helps" in text
    assert "Case study: HMM does not help" in text


@pytest.mark.slow
def test_all_student_test_trips_produce_complete_decoder_rows() -> None:
    from pathsnap import config

    artifact = load_model_artifact(config.PROJECT_ROOT / "artifacts" / "model.joblib")
    for trip in iter_trips(config.TEST_DATA_DIR, "test"):
        inference = prepare_trip_inference(trip, artifact)
        frame = evaluate_decoders(inference, HMMConfig(emission="classifier"))

        assert len(frame) == 3, trip.trip_id
        assert set(frame["method"]) == {"nearest", "classifier", "hmm"}
        assert frame["alignment_coverage"].min() >= 0.99, trip.trip_id
        assert frame["point_accuracy"].notna().all(), trip.trip_id
        assert frame["n_missing_points"].max() <= trip.n_points
