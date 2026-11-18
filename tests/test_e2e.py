"""Tests for end-to-end API verification."""

from __future__ import annotations

from pathlib import Path

import pytest

from api.e2e import E2E_COLUMNS, run_end_to_end
from api.service import PredictionService
from pathsnap.artifacts import (
    build_model_artifact,
    save_model_artifact,
    write_config_json,
    write_json,
)
from pathsnap.cache import build_trip_artifacts
from pathsnap.dataset import build_trip_dataset
from pathsnap.hmm import HMMConfig
from pathsnap.io import load_trip
from pathsnap.models import fit_model
from conftest import write_trip


def _service(tmp_path: Path) -> PredictionService:
    eval_dir = tmp_path / "eval"
    trip_dir = write_trip(eval_dir, "00000042")
    training_trip = load_trip(trip_dir, "train")
    dataset = build_trip_dataset(training_trip, build_trip_artifacts(training_trip))
    trained = fit_model("logistic_regression", dataset.features, dataset.labels)
    artifact = build_model_artifact(trained)

    artifact_dir = tmp_path / "artifacts"
    save_model_artifact(artifact, artifact_dir / "model.joblib")
    write_config_json(
        artifact_dir / "config.json",
        artifact,
        candidate_config={},
        alignment_config={},
        hmm_config=HMMConfig().to_dict(),
        seed=42,
    )
    write_json(artifact_dir / "hmm.json", HMMConfig().to_dict())
    return PredictionService(eval_dir, artifact_dir)


def test_end_to_end_summary_has_complete_rows(tmp_path: Path) -> None:
    summary = run_end_to_end(_service(tmp_path))

    assert list(summary.columns) == list(E2E_COLUMNS)
    assert len(summary) == 1
    assert summary.loc[0, "status"] == 200
    assert summary.loc[0, "n_predicted_arcs"] > 0
    assert summary.drop(columns=["trip_id", "status", "method"]).notna().all().all()


@pytest.mark.slow
def test_real_artifact_returns_http_200_for_all_test_trips() -> None:
    from pathsnap import config

    service = PredictionService(config.TEST_DATA_DIR, config.PROJECT_ROOT / "artifacts")
    summary = run_end_to_end(service, method="hmm")

    assert len(summary) == 10
    assert (summary["status"] == 200).all()
    assert summary.drop(columns=["trip_id", "status", "method"]).notna().all().all()
    assert summary["route_continuity"].min() >= 0.9
    assert summary["alignment_coverage"].min() >= 0.99
