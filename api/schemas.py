"""Typed API request/response models. No reference-route fields exist here."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    model_name: str | None = None


class TripSummary(BaseModel):
    trip_id: str
    point_count: int
    bbox: list[float]
    duration_s: float


class TripsResponse(BaseModel):
    trips: list[TripSummary]


class ObservationPoint(BaseModel):
    lon: float
    lat: float
    timestamp: float


class ObservationsResponse(BaseModel):
    trip_id: str
    points: list[ObservationPoint]


class PredictRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=64)
    method: Literal["nearest", "classifier", "hmm"] = "hmm"
    noise_sigma_m: float = Field(default=0.0, ge=0.0, le=100.0)
    sample_every_s: float = Field(default=1.0, ge=1.0, le=60.0)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)


class PointPrediction(BaseModel):
    obs_index: int
    arc_id: int
    confidence: float


class CandidatePoint(BaseModel):
    obs_index: int
    arc_id: int
    lon: float
    lat: float


class PredictionMetrics(BaseModel):
    top1_accuracy: float
    macro_f1: float
    decoder_macro_f1: float = 0.0
    route_f1: float = 0.0
    precision: float
    recall: float
    log_loss: float
    candidate_miss_rate: float
    route_continuity: float
    route_precision: float
    route_recall: float
    rmf: float
    alignment_coverage: float


class PredictionLayers(BaseModel):
    gps_points: list[ObservationPoint]
    candidate_points: list[CandidatePoint]


class PredictResponse(BaseModel):
    trip_id: str
    method: str
    predicted_arcs: list[int]
    route_coords: list[list[float]]
    point_predictions: list[PointPrediction]
    metrics: PredictionMetrics
    layers: PredictionLayers
    inference_ms: float
