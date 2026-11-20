"""Artifact loading and prediction orchestration for the API.

Trips, reference routes, and scoring stay inside this service. Callers receive
only predictions, layers, and aggregate metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from api.schemas import (
    CandidatePoint,
    ObservationPoint,
    PointPrediction,
    PredictionLayers,
    PredictionMetrics,
    PredictResponse,
    TripSummary,
)
from pathsnap import config
from pathsnap.artifacts import ModelArtifact, load_hmm_config, load_model_artifact, read_json
from pathsnap.hmm import HMMConfig
from pathsnap.io import Trip, discover_trip_dirs, iter_trips
from pathsnap.pipeline import PredictionResult, predict_trip


class PredictionService:
    """Loads artifacts and evaluation trips once, then serves predictions."""

    def __init__(self, eval_data_dir: Path, artifact_dir: Path) -> None:
        self.eval_data_dir = Path(eval_data_dir)
        self.artifact_dir = Path(artifact_dir)
        self.artifact: ModelArtifact = load_model_artifact(
            self.artifact_dir / "model.joblib"
        )
        self._validate_feature_contract()
        self.hmm_config = self._load_hmm_config()
        self._trips: dict[str, Trip] = {
            trip.trip_id: trip for trip in iter_trips(self.eval_data_dir, "eval")
        }
        if not self._trips:
            raise RuntimeError(
                f"No evaluator trips found in {self.eval_data_dir}"
            )

    def _validate_feature_contract(self) -> None:
        config_path = self.artifact_dir / "config.json"
        if not config_path.is_file():
            return
        payload = read_json(config_path)
        feature_names = tuple(payload.get("feature_names", ()))
        if feature_names and feature_names != self.artifact.feature_names:
            raise ValueError(
                "Artifact feature order does not match config.json; retrain or "
                "regenerate artifacts."
            )

    def _load_hmm_config(self) -> HMMConfig:
        return load_hmm_config(
            self.artifact_dir / "hmm.json", fallback=self.artifact_dir / "config.json"
        )

    @property
    def model_loaded(self) -> bool:
        return self.artifact is not None

    @property
    def model_name(self) -> str:
        return self.artifact.model_name

    def list_trips(self) -> list[TripSummary]:
        summaries = []
        for trip in sorted(self._trips.values(), key=lambda item: item.trip_id):
            duration = (
                float(trip.track[-1, 2] - trip.track[0, 2])
                if trip.n_points > 1
                else 0.0
            )
            summaries.append(
                TripSummary(
                    trip_id=trip.trip_id,
                    point_count=trip.n_points,
                    bbox=[float(value) for value in trip.bbox],
                    duration_s=duration,
                )
            )
        return summaries

    def get_trip(self, trip_id: str) -> Trip:
        if trip_id not in self._trips:
            raise KeyError(trip_id)
        return self._trips[trip_id]

    def observations(self, trip_id: str) -> list[ObservationPoint]:
        trip = self.get_trip(trip_id)
        return [
            ObservationPoint(
                lon=float(lon), lat=float(lat), timestamp=float(timestamp)
            )
            for lon, lat, timestamp in trip.track
        ]

    def predict(
        self,
        trip_id: str,
        method: str = "hmm",
        noise_sigma_m: float = 0.0,
        sample_every_s: float = 1.0,
        seed: int = config.PERTURBATION_SEED,
    ) -> PredictResponse:
        trip = self.get_trip(trip_id)
        result = predict_trip(
            trip,
            self.artifact,
            method=method,
            noise_sigma_m=noise_sigma_m,
            sample_every_s=sample_every_s,
            seed=seed,
            hmm_config=self.hmm_config,
        )
        return self._to_response(result)

    def _to_response(self, result: PredictionResult) -> PredictResponse:
        gps_points = [
            ObservationPoint(lon=float(lon), lat=float(lat), timestamp=float(timestamp))
            for lon, lat, timestamp in result.effective_track
        ]
        candidate_points = [
            CandidatePoint(**row) for row in result.candidate_layer
        ]
        point_predictions = [
            PointPrediction(
                obs_index=int(index),
                arc_id=int(arc_id),
                confidence=float(confidence),
            )
            for index, (arc_id, confidence) in enumerate(
                zip(result.point_arc_ids, result.point_confidences)
            )
        ]
        return PredictResponse(
            trip_id=result.trip_id,
            method=result.method,
            predicted_arcs=[int(arc) for arc in result.predicted_arcs],
            route_coords=[
                [float(lon), float(lat)] for lon, lat in result.route_coords
            ],
            point_predictions=point_predictions,
            metrics=PredictionMetrics(**result.metrics),
            layers=PredictionLayers(
                gps_points=gps_points, candidate_points=candidate_points
            ),
            inference_ms=float(result.inference_ms),
        )


def discover_eval_trip_ids(eval_data_dir: Path) -> list[str]:
    return [path.name for path in discover_trip_dirs(eval_data_dir)]
