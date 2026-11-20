"""FastAPI integration tests for health, trips, observations, and /predict."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
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

FORBIDDEN_KEYS = {
    "route_arcs",
    "reference_route",
    "reference_arcs",
    "reference_route_arcs",
    "ground_truth",
    "ground_truth_arcs",
    "label",
    "labels",
}


def _collect_keys(payload) -> set[str]:
    keys: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.add(key)
            keys |= _collect_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            keys |= _collect_keys(item)
    return keys


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    eval_dir = tmp_path / "eval"
    trip_dir = write_trip(eval_dir, "00000046")
    training_trip = load_trip(trip_dir, "train")
    training_dataset = build_trip_dataset(
        training_trip, build_trip_artifacts(training_trip)
    )
    trained = fit_model(
        "logistic_regression", training_dataset.features, training_dataset.labels
    )
    artifact = build_model_artifact(trained)

    artifact_dir = tmp_path / "artifacts"
    save_model_artifact(artifact, artifact_dir / "model.joblib")
    write_config_json(
        artifact_dir / "config.json",
        artifact,
        candidate_config={"top_k": 8},
        alignment_config={"max_distance_m": 50.0},
        hmm_config=HMMConfig().to_dict(),
        seed=42,
    )
    write_json(artifact_dir / "hmm.json", HMMConfig().to_dict())

    service = PredictionService(eval_dir, artifact_dir)
    app = create_app(service)
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_loaded_model(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["model_name"] == "logistic_regression"
    assert body["version"]


def test_trip_listing_contains_no_reference_fields(client: TestClient) -> None:
    response = client.get("/trips")

    assert response.status_code == 200
    body = response.json()
    assert [trip["trip_id"] for trip in body["trips"]] == ["00000046"]
    assert not (_collect_keys(body) & FORBIDDEN_KEYS)
    assert set(body["trips"][0]) == {"trip_id", "point_count", "bbox", "duration_s"}


def test_observations_return_raw_gps_only(client: TestClient) -> None:
    response = client.get("/trips/00000046/observations")

    assert response.status_code == 200
    body = response.json()
    assert body["trip_id"] == "00000046"
    assert len(body["points"]) == 3
    assert set(body["points"][0]) == {"lon", "lat", "timestamp"}
    assert not (_collect_keys(body) & FORBIDDEN_KEYS)


@pytest.mark.parametrize("method", ["nearest", "classifier", "hmm"])
def test_predict_returns_route_metrics_and_no_ground_truth(
    client: TestClient, method: str
) -> None:
    response = client.post(
        "/predict", json={"trip_id": "00000046", "method": method}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["trip_id"] == "00000046"
    assert body["method"] == method
    assert body["predicted_arcs"]
    assert body["route_coords"]
    assert len(body["point_predictions"]) == len(body["layers"]["gps_points"])
    assert body["metrics"]["top1_accuracy"] >= 0.0
    assert body["inference_ms"] >= 0.0
    assert not (_collect_keys(body) & FORBIDDEN_KEYS)


def test_unknown_trip_returns_404(client: TestClient) -> None:
    assert client.get("/trips/99999999/observations").status_code == 404
    assert (
        client.post("/predict", json={"trip_id": "99999999"}).status_code == 404
    )


def test_invalid_parameters_return_422(client: TestClient) -> None:
    assert (
        client.post(
            "/predict", json={"trip_id": "00000046", "method": "magic"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/predict", json={"trip_id": "00000046", "noise_sigma_m": 500}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/predict", json={"trip_id": "00000046", "sample_every_s": 0}
        ).status_code
        == 422
    )


def test_cors_preflight_allows_ui_origin(client: TestClient) -> None:
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:5500",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code in (200, 204)
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5500"


def test_cors_origins_environment_override(monkeypatch) -> None:
    from api import deps

    monkeypatch.setenv("CORS_ORIGINS", "http://a:5500, http://b:5500")
    assert deps.cors_origins() == ["http://a:5500", "http://b:5500"]


def test_prediction_logs_contain_no_ground_truth(client: TestClient, caplog) -> None:
    caplog.set_level("INFO")
    assert client.post("/predict", json={"trip_id": "00000046"}).status_code == 200

    text = "\n".join(record.getMessage() for record in caplog.records).lower()
    assert "reference" not in text
    assert "ground_truth" not in text
    assert "00000046" in text


def test_missing_artifacts_raise_a_clear_startup_error(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, OSError)):
        PredictionService(tmp_path / "eval", tmp_path / "missing-artifacts")


def test_prediction_service_rejects_feature_contract_mismatch(tmp_path: Path) -> None:
    import json

    eval_dir = tmp_path / "eval"
    trip_dir = write_trip(eval_dir, "00000048")
    training_trip = load_trip(trip_dir, "train")
    training_dataset = build_trip_dataset(
        training_trip, build_trip_artifacts(training_trip)
    )
    trained = fit_model(
        "logistic_regression", training_dataset.features, training_dataset.labels
    )
    artifact = build_model_artifact(trained)

    artifact_dir = tmp_path / "artifacts"
    save_model_artifact(artifact, artifact_dir / "model.joblib")
    config_path = write_config_json(
        artifact_dir / "config.json",
        artifact,
        candidate_config={"top_k": 8},
        alignment_config={"max_distance_m": 50.0},
        hmm_config=HMMConfig().to_dict(),
        seed=42,
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["feature_names"] = list(reversed(payload["feature_names"]))
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="feature order"):
        PredictionService(eval_dir, artifact_dir)
