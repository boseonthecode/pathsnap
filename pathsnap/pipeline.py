"""End-to-end single-trip prediction shared by the API and scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from pathsnap import config
from pathsnap.artifacts import ModelArtifact
from pathsnap.candidates import CandidateSet, generate_candidates
from pathsnap.features import extract_features
from pathsnap.geometry import arc_lengths_m, projection_for_trip
from pathsnap.hmm import (
    DecodeResult,
    HMMConfig,
    decode_classifier,
    decode_hmm,
    decode_nearest,
)
from pathsnap.io import Trip, discover_trip_dirs, load_trip
from pathsnap.labels import align_observations, candidate_labels
from pathsnap.metrics import (
    Timer,
    candidate_log_loss,
    candidate_macro_f1,
    candidate_precision_recall,
    observation_top1_accuracy,
    route_continuity,
    route_f1,
    route_mismatch_fraction,
    route_precision_recall,
)

SUPPORTED_METHODS = ("nearest", "classifier", "hmm")
MAX_CANDIDATE_LAYER_POINTS = config.MAX_CANDIDATE_LAYER_POINTS


@dataclass(frozen=True)
class PredictionResult:
    """Prediction plus the layers and server-side metrics safe to expose."""

    trip_id: str
    method: str
    predicted_arcs: np.ndarray
    route_coords: np.ndarray
    point_arc_ids: np.ndarray
    point_confidences: np.ndarray
    effective_track: np.ndarray
    candidate_layer: list[dict[str, Any]]
    metrics: dict[str, float]
    inference_ms: float


def effective_track(
    trip: Trip,
    noise_sigma_m: float = 0.0,
    sample_every_s: float = 1.0,
    seed: int = config.PERTURBATION_SEED,
) -> np.ndarray:
    """Deterministically downsample and perturb the GPS track in metric space."""
    track = np.array(trip.track, dtype=np.float64, copy=True)
    if track.shape[0] == 0:
        return track

    if sample_every_s > 0 and track.shape[0] > 1:
        indices = [0]
        next_time = track[0, 2] + sample_every_s
        for index in range(1, track.shape[0]):
            if track[index, 2] >= next_time:
                indices.append(index)
                next_time = track[index, 2] + sample_every_s
        track = track[indices]

    if noise_sigma_m > 0 and track.shape[0] > 0:
        projection = projection_for_trip(trip)
        xy = projection.to_xy(track[:, :2])
        rng = np.random.default_rng(seed)
        xy = xy + rng.normal(0.0, noise_sigma_m, size=xy.shape)
        track[:, :2] = projection.to_lonlat(xy)

    return track


def route_coordinates(trip: Trip, arcs: np.ndarray) -> np.ndarray:
    """Build a lon/lat polyline from an ordered arc sequence."""
    coords: list[list[float]] = []
    for arc_id in arcs:
        from_node, to_node = trip.arcs[int(arc_id)]
        for node in (int(from_node), int(to_node)):
            lon, lat = float(trip.nodes[node, 0]), float(trip.nodes[node, 1])
            if not coords or coords[-1] != [lon, lat]:
                coords.append([lon, lat])
    return np.asarray(coords, dtype=np.float64).reshape(-1, 2)


def _decode(
    method: str,
    runtime_trip: Trip,
    candidates: CandidateSet,
    probabilities: np.ndarray,
    hmm_config: HMMConfig,
) -> DecodeResult:
    if method == "nearest":
        return decode_nearest(runtime_trip, candidates)
    if method == "classifier":
        return decode_classifier(runtime_trip, candidates, probabilities)
    if method == "hmm":
        return decode_hmm(
            runtime_trip,
            candidates,
            probabilities=probabilities,
            config=replace(hmm_config, emission="classifier"),
        )
    raise ValueError(f"Unsupported method {method!r}")


def _candidate_layer(candidates: CandidateSet) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates.candidates[:MAX_CANDIDATE_LAYER_POINTS]:
        rows.append(
            {
                "obs_index": candidate.obs_index,
                "arc_id": candidate.arc_id,
                "lon": candidate.proj_lon,
                "lat": candidate.proj_lat,
            }
        )
    return rows


def _score(
    runtime_trip: Trip,
    candidates: CandidateSet,
    probabilities: np.ndarray,
    result: DecodeResult,
) -> dict[str, float]:
    alignment = align_observations(runtime_trip)
    observation_indices = np.array(
        [candidate.obs_index for candidate in candidates.candidates], dtype=np.int32
    )
    candidate_arcs = np.array(
        [candidate.arc_id for candidate in candidates.candidates], dtype=np.int32
    )
    mask = alignment.alignable[observation_indices] if observation_indices.size else np.empty(0, bool)
    if observation_indices.size:
        labels = candidate_labels(alignment, observation_indices, candidate_arcs) & mask
        masked_probabilities = probabilities[mask]
        masked_labels = labels[mask]
        predictions = (masked_probabilities >= config.CANDIDATE_THRESHOLD).astype(np.int8)
        macro_f1 = candidate_macro_f1(masked_labels, predictions)
        decoded_predictions = (
            candidate_arcs[mask] == result.point_arc_ids[observation_indices[mask]]
        ).astype(np.int8)
        decoder_macro_f1 = candidate_macro_f1(masked_labels, decoded_predictions)
        log_loss_value = candidate_log_loss(masked_labels, masked_probabilities)
        precision, recall_candidate = candidate_precision_recall(
            masked_labels, predictions
        )
    else:
        macro_f1 = decoder_macro_f1 = log_loss_value = precision = recall_candidate = 0.0

    precision_route, recall_route = route_precision_recall(
        result.predicted_arcs, runtime_trip.route
    )
    lengths = arc_lengths_m(runtime_trip)
    return {
        "top1_accuracy": observation_top1_accuracy(
            alignment.arc_ids, result.point_arc_ids, alignment.alignable
        ),
        "macro_f1": float(macro_f1),
        "decoder_macro_f1": float(decoder_macro_f1),
        "route_f1": route_f1(precision_route, recall_route),
        "precision": float(precision),
        "recall": float(recall_candidate),
        "log_loss": float(log_loss_value),
        "candidate_miss_rate": float(candidates.miss_rate),
        "route_continuity": route_continuity(result.predicted_arcs, runtime_trip.arcs),
        "route_precision": float(precision_route),
        "route_recall": float(recall_route),
        "rmf": route_mismatch_fraction(result.predicted_arcs, runtime_trip.route, lengths),
        "alignment_coverage": float(alignment.coverage),
    }


def predict_trip(
    trip: Trip,
    artifact: ModelArtifact,
    method: str = "hmm",
    noise_sigma_m: float = 0.0,
    sample_every_s: float = 1.0,
    seed: int = config.PERTURBATION_SEED,
    hmm_config: HMMConfig | None = None,
) -> PredictionResult:
    """Run the full map-matching pipeline for one trip without exposing labels."""
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method {method!r}")
    hmm_config = replace(hmm_config or HMMConfig(), emission="classifier")

    with Timer() as timer:
        track = effective_track(trip, noise_sigma_m, sample_every_s, seed)
        runtime_trip = replace(trip, track=track)
        candidates = generate_candidates(runtime_trip)
        features = extract_features(runtime_trip, candidates)
        probabilities = artifact.positive_probability(features.features)
        result = _decode(method, runtime_trip, candidates, probabilities, hmm_config)
        metrics = _score(runtime_trip, candidates, probabilities, result)

    return PredictionResult(
        trip_id=trip.trip_id,
        method=method,
        predicted_arcs=result.predicted_arcs,
        route_coords=route_coordinates(runtime_trip, result.predicted_arcs),
        point_arc_ids=result.point_arc_ids,
        point_confidences=result.point_confidences,
        effective_track=track,
        candidate_layer=_candidate_layer(candidates),
        metrics=metrics,
        inference_ms=float(timer.elapsed_ms),
    )


METRIC_LABELS: tuple[tuple[str, str], ...] = (
    ("Point Top-1", "top1_accuracy"),
    ("Classifier Macro F1", "macro_f1"),
    ("Decoder Macro F1", "decoder_macro_f1"),
    ("Route F1", "route_f1"),
    ("Route precision", "route_precision"),
    ("Route recall", "route_recall"),
    ("Route continuity", "route_continuity"),
    ("Alignment coverage", "alignment_coverage"),
)


def prediction_summary(result: PredictionResult) -> dict[str, Any]:
    """One row describing a prediction: its metrics plus scored/correct counts."""
    metrics = result.metrics
    scored_points = round(len(result.effective_track) * metrics["alignment_coverage"])
    return {
        "trip_id": result.trip_id,
        "method": result.method,
        **metrics,
        "inference_ms": result.inference_ms,
        "scored_points": scored_points,
        "correct_points": round(scored_points * metrics["top1_accuracy"]),
    }


def write_prediction_outputs(
    result: PredictionResult, out_dir: Path
) -> dict[str, Any]:
    """Write ``<trip>.route`` and ``<trip>.json`` and return the summary row."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = prediction_summary(result)
    np.savetxt(out_dir / f"{result.trip_id}.route", result.predicted_arcs, fmt="%d")
    (out_dir / f"{result.trip_id}.json").write_text(
        json.dumps(
            {
                **summary,
                "predicted_arcs": result.predicted_arcs.tolist(),
                "point_arc_ids": result.point_arc_ids.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def predict_directory(
    data_dir: Path,
    artifact: ModelArtifact,
    *,
    partition: str | None = None,
    method: str = "hmm",
    noise_sigma_m: float = 0.0,
    sample_every_s: float = 1.0,
    seed: int = config.PERTURBATION_SEED,
    hmm_config: HMMConfig | None = None,
) -> list[PredictionResult]:
    """Run the pipeline over every trip directory in a data partition."""
    results = []
    for directory in discover_trip_dirs(data_dir):
        trip = load_trip(directory, partition) if partition else load_trip(directory)
        results.append(
            predict_trip(
                trip,
                artifact,
                method=method,
                noise_sigma_m=noise_sigma_m,
                sample_every_s=sample_every_s,
                seed=seed,
                hmm_config=hmm_config,
            )
        )
    return results
