"""In-process end-to-end API verification across evaluation trips."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from api.service import PredictionService

E2E_COLUMNS: tuple[str, ...] = (
    "trip_id",
    "status",
    "method",
    "n_predicted_arcs",
    "top1_accuracy",
    "route_continuity",
    "route_precision",
    "route_recall",
    "rmf",
    "alignment_coverage",
    "candidate_miss_rate",
    "inference_ms",
)


def run_end_to_end(
    service: PredictionService, method: str = "hmm"
) -> pd.DataFrame:
    """Call /health, /trips, observations, and /predict for every trip."""
    app = create_app(service)
    rows: list[dict[str, Any]] = []
    with TestClient(app) as client:
        health = client.get("/health")
        health.raise_for_status()
        trips = client.get("/trips")
        trips.raise_for_status()
        trip_payloads = trips.json()["trips"]

        for trip in trip_payloads:
            trip_id = trip["trip_id"]
            observations = client.get(f"/trips/{trip_id}/observations")
            observations.raise_for_status()
            response = client.post(
                "/predict", json={"trip_id": trip_id, "method": method}
            )
            if response.status_code != 200:
                rows.append(
                    {
                        "trip_id": trip_id,
                        "status": response.status_code,
                        "method": method,
                    }
                )
                continue
            body = response.json()
            metrics = body["metrics"]
            rows.append(
                {
                    "trip_id": trip_id,
                    "status": 200,
                    "method": body["method"],
                    "n_predicted_arcs": len(body["predicted_arcs"]),
                    "top1_accuracy": metrics["top1_accuracy"],
                    "route_continuity": metrics["route_continuity"],
                    "route_precision": metrics["route_precision"],
                    "route_recall": metrics["route_recall"],
                    "rmf": metrics["rmf"],
                    "alignment_coverage": metrics["alignment_coverage"],
                    "candidate_miss_rate": metrics["candidate_miss_rate"],
                    "inference_ms": body["inference_ms"],
                }
            )
    return pd.DataFrame(rows, columns=list(E2E_COLUMNS))
