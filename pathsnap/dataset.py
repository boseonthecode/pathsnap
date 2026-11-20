"""Assemble model-ready datasets from cached trip artifacts.

Candidate labels are binary (``1`` when the candidate arc equals the aligned
reference arc). Unalignable observations and alignable observations without
candidates are excluded and counted. Low-coverage training trips are kept in the
bundle for reporting but excluded from fitting by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from pathsnap import config
from pathsnap.cache import TripArtifacts, load_or_build_trip_artifacts
from pathsnap.features import FEATURE_NAMES
from pathsnap.io import Trip
from pathsnap.labels import candidate_labels


@dataclass(frozen=True)
class TripDataset:
    """Candidate rows for one trip with labels and diagnostics."""

    trip_id: str
    partition: str
    feature_names: tuple[str, ...]
    features: np.ndarray
    labels: np.ndarray
    observation_indices: np.ndarray
    arc_ids: np.ndarray
    route_positions: np.ndarray
    coverage: float
    low_coverage: bool
    n_unalignable_observations: int
    n_missing_candidate_observations: int
    included_in_training: bool

    @property
    def n_rows(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class DatasetBundle:
    """All trip datasets plus helpers for stacking and grouped folds."""

    trips: tuple[TripDataset, ...]
    feature_names: tuple[str, ...]

    def by_trip_id(self) -> dict[str, TripDataset]:
        return {trip.trip_id: trip for trip in self.trips}

    def training_trips(self) -> list[TripDataset]:
        return [trip for trip in self.trips if trip.included_in_training]

    def test_trips(self) -> list[TripDataset]:
        return [trip for trip in self.trips if trip.partition == "test"]

    def excluded_training_trips(self) -> list[TripDataset]:
        return [
            trip
            for trip in self.trips
            if trip.partition == "train" and not trip.included_in_training
        ]

    def stacked(
        self, trip_ids: Sequence[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate features and labels for the requested trips."""
        lookup = self.by_trip_id()
        selected = (
            [lookup[trip_id] for trip_id in trip_ids]
            if trip_ids is not None
            else list(self.trips)
        )
        selected = [trip for trip in selected if trip.n_rows > 0]
        if not selected:
            return (
                np.empty((0, len(self.feature_names)), dtype=np.float64),
                np.empty(0, dtype=np.int8),
            )
        features = np.vstack([trip.features for trip in selected])
        labels = np.concatenate([trip.labels for trip in selected])
        return features, labels


def build_trip_dataset(
    trip: Trip,
    artifacts: TripArtifacts,
    min_coverage: float = config.LOW_COVERAGE_THRESHOLD,
) -> TripDataset:
    """Convert one trip's candidates/features/alignment into labeled rows."""
    alignment = artifacts.alignment
    features = artifacts.features
    observation_indices = features.observation_indices
    arc_ids = features.arc_ids

    n_rows = features.n_rows
    labels = np.zeros(n_rows, dtype=np.int8)
    route_positions = np.full(n_rows, -1, dtype=np.int32)

    if n_rows:
        row_positions = alignment.positions[observation_indices]
        route_positions = row_positions.astype(np.int32)
        labeled = row_positions >= 0
        labels[labeled] = candidate_labels(
            alignment, observation_indices[labeled], arc_ids[labeled]
        )

    alignable_observations = set(np.flatnonzero(alignment.alignable).tolist())
    observations_with_candidates = set(
        np.unique(observation_indices).tolist()
    ) if n_rows else set()
    missing_candidate_observations = len(alignable_observations - observations_with_candidates)
    n_unalignable = int(np.count_nonzero(~alignment.alignable))

    included_in_training = (
        trip.partition == "train" and alignment.coverage >= min_coverage
    )

    return TripDataset(
        trip_id=trip.trip_id,
        partition=trip.partition,
        feature_names=FEATURE_NAMES,
        features=features.features,
        labels=labels,
        observation_indices=observation_indices.astype(np.int32),
        arc_ids=arc_ids.astype(np.int32),
        route_positions=route_positions,
        coverage=alignment.coverage,
        low_coverage=alignment.low_coverage,
        n_unalignable_observations=n_unalignable,
        n_missing_candidate_observations=missing_candidate_observations,
        included_in_training=included_in_training,
    )


def assemble_dataset(
    trips: Iterable[Trip],
    artifacts: Sequence[TripArtifacts] | None = None,
    min_coverage: float = config.LOW_COVERAGE_THRESHOLD,
    use_cache: bool = True,
    load_candidates: bool = False,
) -> DatasetBundle:
    """Build a bundle from trips, loading cached artifacts when possible."""
    trip_list = list(trips)
    if artifacts is None:
        artifacts = [
            load_or_build_trip_artifacts(
                trip, use_cache=use_cache, load_candidates=load_candidates
            )
            for trip in trip_list
        ]
    if len(artifacts) != len(trip_list):
        raise ValueError("trips and artifacts must have the same length")

    datasets = tuple(
        build_trip_dataset(trip, trip_artifacts, min_coverage=min_coverage)
        for trip, trip_artifacts in zip(trip_list, artifacts)
    )
    return DatasetBundle(trips=datasets, feature_names=FEATURE_NAMES)


def trip_grouped_folds(
    trip_ids: Sequence[str],
    n_splits: int = config.N_SPLITS,
    seed: int = config.CV_SEED,
) -> list[tuple[list[str], list[str]]]:
    """Deterministic trip-disjoint folds: every trip appears in exactly one validation fold."""
    unique_trip_ids = sorted(set(trip_ids))
    if not unique_trip_ids:
        return []
    n_splits = max(1, min(n_splits, len(unique_trip_ids)))

    shuffled = np.array(unique_trip_ids, dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)

    validation_folds: list[list[str]] = [[] for _ in range(n_splits)]
    for index, trip_id in enumerate(shuffled):
        validation_folds[index % n_splits].append(str(trip_id))

    folds: list[tuple[list[str], list[str]]] = []
    for validation in validation_folds:
        validation_set = set(validation)
        training = [trip_id for trip_id in unique_trip_ids if trip_id not in validation_set]
        folds.append((training, validation))
    return folds


def fit_feature_scaler(
    bundle: DatasetBundle, training_trip_ids: Sequence[str]
) -> StandardScaler:
    """Fit a scaler on the requested (training-fold) trips only."""
    features, _ = bundle.stacked(training_trip_ids)
    scaler = StandardScaler()
    if features.shape[0] > 0:
        scaler.fit(features)
    return scaler
