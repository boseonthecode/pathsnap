"""Decoder optimisation helpers: preparation, sampling, scoring, and search grids.

These are the pieces ``scripts/10_optimize_decoder.py`` and
``scripts/12_metrics_dashboard.py`` share. Training, tuning, and evaluation all
go through the same preparation path so the numbers stay comparable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from pathsnap import config
from pathsnap.artifacts import ModelArtifact
from pathsnap.cache import compute_cache_version
from pathsnap.candidates import CandidateSet, generate_candidates
from pathsnap.evaluate import TripInference, _decode_row
from pathsnap.features import extract_features
from pathsnap.geometry import arc_lengths_m
from pathsnap.hmm import HMMConfig, NetworkDistance, decode_hmm
from pathsnap.io import Trip
from pathsnap.labels import AlignmentResult, align_observations, candidate_labels
from pathsnap.metrics import Timer, route_f1


@dataclass(frozen=True)
class DecoderTrip:
    """Candidates, features, and alignment for one trip, with metric arc lengths."""

    trip: Trip
    candidates: CandidateSet
    features: np.ndarray
    alignment: AlignmentResult
    arc_lengths: np.ndarray


def _trip_digest(trip: Trip) -> str:
    digest = hashlib.sha256(compute_cache_version().encode())
    for values in (trip.track, trip.nodes, trip.arcs, trip.route):
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()[:16]


def prepare_decoder_trip(trip: Trip, cache_dir: Path | None = None) -> DecoderTrip:
    """Build (or load) the decoder inputs for one trip, keyed by content hash."""
    cache_dir = config.get_paths().cache_dir if cache_dir is None else Path(cache_dir)
    cache = cache_dir / "decoder_optimization" / (
        f"{trip.partition}_{trip.trip_id}_{_trip_digest(trip)}.joblib"
    )
    if cache.exists():
        return joblib.load(cache)

    candidates = generate_candidates(trip)
    prepared = DecoderTrip(
        trip=trip,
        candidates=candidates,
        features=extract_features(trip, candidates).features,
        alignment=align_observations(trip),
        arc_lengths=arc_lengths_m(trip),
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(prepared, cache, compress=3)
    return prepared


def training_arrays(
    prepared: Mapping[str, DecoderTrip], trip_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Stack sampled feature rows and labels for the given trips."""
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for trip_id in trip_ids:
        item = prepared[trip_id]
        if item.alignment.coverage < config.LOW_COVERAGE_THRESHOLD:
            continue
        observation_indices = np.array(
            [candidate.obs_index for candidate in item.candidates.candidates]
        )
        arc_ids = np.array([candidate.arc_id for candidate in item.candidates.candidates])
        valid_observations = np.flatnonzero(item.alignment.alignable)
        if not valid_observations.size:
            continue
        cap = min(config.TRAIN_MAX_OBSERVATIONS_PER_TRIP, len(valid_observations))
        sampled = valid_observations[
            np.linspace(0, len(valid_observations) - 1, cap, dtype=int)
        ]
        mask = np.isin(observation_indices, sampled)
        features.append(item.features[mask])
        labels.append(
            candidate_labels(item.alignment, observation_indices[mask], arc_ids[mask])
        )
    if not features:
        return np.empty((0, 0)), np.empty(0, dtype=np.int8)
    return np.vstack(features), np.concatenate(labels)


def to_inference(prepared: DecoderTrip, artifact: ModelArtifact) -> TripInference:
    """Score a prepared trip with an artifact, producing decoder-ready inputs."""
    return TripInference(
        trip=prepared.trip,
        candidates=prepared.candidates,
        probabilities=artifact.positive_probability(prepared.features),
        alignment=prepared.alignment,
        arc_lengths=prepared.arc_lengths,
    )


def score_decoder(
    inference: TripInference, settings: HMMConfig, network: NetworkDistance
) -> dict[str, Any]:
    """Decode one trip and return its metric row, including route F1."""
    with Timer() as timer:
        decoded = decode_hmm(
            inference.trip,
            inference.candidates,
            inference.probabilities,
            settings,
            network,
        )
    row = _decode_row(inference, decoded, timer.elapsed_ms)
    row["route_f1"] = route_f1(row["route_precision"], row["route_recall"])
    return row


def search_grid(base: HMMConfig) -> list[HMMConfig]:
    """The baseline plus every backward-tolerance/beta/distance-weight combination."""
    grid = [base]
    for tolerance in (10.0, 30.0, 60.0):
        for beta in (2.0, 10.0, 50.0):
            for weight in (0.0, 1.0):
                grid.append(
                    HMMConfig(
                        **{
                            **base.to_dict(),
                            "backward_tolerance_m": tolerance,
                            "beta_m": beta,
                            "distance_weight": weight,
                        }
                    )
                )
    return grid
