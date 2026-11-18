"""Trip-grouped cross-validation, model comparison, and selection helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.artifacts import ModelArtifact
from pathsnap.candidates import CandidateSet, generate_candidates
from pathsnap.dataset import DatasetBundle, TripDataset, trip_grouped_folds
from pathsnap.features import extract_features
from pathsnap.geometry import arc_lengths_m
from pathsnap.hmm import (
    DecodeResult,
    HMMConfig,
    decode_classifier,
    decode_hmm,
    decode_nearest,
)
from pathsnap.io import Trip
from pathsnap.labels import AlignmentResult, align_observations
from pathsnap.metrics import (
    Timer,
    observation_top1_accuracy,
    route_continuity,
    route_mismatch_fraction,
    route_precision_recall,
)
from pathsnap.models import MODEL_SPECS, ModelResult, evaluate_model, fit_model

DECODER_COLUMNS: tuple[str, ...] = (
    "trip_id",
    "partition",
    "method",
    "point_accuracy",
    "route_continuity",
    "route_precision",
    "route_recall",
    "rmf",
    "alignment_coverage",
    "inference_ms",
    "n_points",
    "n_missing_points",
    "n_predicted_arcs",
    "n_unbridged_transitions",
)

COMPARISON_COLUMNS: tuple[str, ...] = (
    "model",
    "family",
    "config",
    "scope",
    "fold",
    "n_rows",
    "n_observations",
    "top1_accuracy",
    "macro_f1",
    "precision",
    "recall",
    "log_loss",
    "train_time_s",
    "inference_ms",
    "tn",
    "fp",
    "fn",
    "tp",
)


def _empty_features(bundle: DatasetBundle) -> np.ndarray:
    return np.empty((0, len(bundle.feature_names)), dtype=np.float64)


def sample_trip_rows(
    dataset: TripDataset, max_observations: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Take all candidates for up to ``max_observations`` evenly spaced observations."""
    if dataset.n_rows == 0:
        return (
            np.empty((0, len(dataset.feature_names)), dtype=np.float64),
            np.empty(0, dtype=np.int8),
            np.empty(0, dtype=np.int32),
        )

    observations = dataset.observation_indices
    unique = np.unique(observations)
    if max_observations is None or unique.size <= max_observations:
        mask = np.ones(observations.shape[0], dtype=bool)
    else:
        chosen = unique[
            np.linspace(0, unique.size - 1, num=max_observations, dtype=int)
        ]
        mask = np.isin(observations, chosen)

    return dataset.features[mask], dataset.labels[mask], observations[mask]


def stack_sampled(
    bundle: DatasetBundle,
    trip_ids: Sequence[str],
    max_observations_per_trip: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate sampled rows from several trips with globally unique observation ids."""
    lookup = bundle.by_trip_id()
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    group_blocks: list[np.ndarray] = []
    offset = 0

    for trip_id in trip_ids:
        dataset = lookup[trip_id]
        features, labels, observations = sample_trip_rows(
            dataset, max_observations_per_trip
        )
        if features.shape[0] == 0:
            continue
        groups = observations.astype(np.int64) + offset
        offset = int(groups.max()) + 1
        feature_blocks.append(features)
        label_blocks.append(labels)
        group_blocks.append(groups)

    if not feature_blocks:
        return (
            _empty_features(bundle),
            np.empty(0, dtype=np.int8),
            np.empty(0, dtype=np.int64),
        )
    return (
        np.vstack(feature_blocks),
        np.concatenate(label_blocks),
        np.concatenate(group_blocks),
    )


def _result_row(
    result: ModelResult, scope: str, fold: int
) -> dict[str, object]:
    row = result.to_row()
    row["config"] = json.dumps(row["config"], sort_keys=True)
    row["scope"] = scope
    row["fold"] = fold
    return row


def _aggregate_row(model_name: str, results: Sequence[ModelResult]) -> dict[str, object]:
    first = results[0]

    def mean(attribute: str) -> float:
        return float(np.mean([getattr(result, attribute) for result in results]))

    return {
        "model": model_name,
        "family": first.family,
        "config": json.dumps(first.config, sort_keys=True),
        "scope": "cv_aggregate",
        "fold": -1,
        "n_rows": int(sum(result.n_rows for result in results)),
        "n_observations": int(sum(result.n_observations for result in results)),
        "top1_accuracy": mean("top1_accuracy"),
        "macro_f1": mean("macro_f1"),
        "precision": mean("precision"),
        "recall": mean("recall"),
        "log_loss": mean("log_loss"),
        "train_time_s": mean("train_time_s"),
        "inference_ms": mean("inference_ms"),
        "tn": int(sum(result.tn for result in results)),
        "fp": int(sum(result.fp for result in results)),
        "fn": int(sum(result.fn for result in results)),
        "tp": int(sum(result.tp for result in results)),
    }


def cross_validate(
    bundle: DatasetBundle,
    model_names: Sequence[str] | None = None,
    n_splits: int = config.N_SPLITS,
    seed: int = config.CV_SEED,
    max_observations_per_trip: int | None = config.CV_MAX_OBSERVATIONS_PER_TRIP,
) -> pd.DataFrame:
    """Run trip-grouped CV and return fold-level plus aggregate rows."""
    model_names = list(MODEL_SPECS) if model_names is None else list(model_names)
    training_trip_ids = [trip.trip_id for trip in bundle.training_trips()]
    folds = trip_grouped_folds(training_trip_ids, n_splits=n_splits, seed=seed)
    if not folds:
        raise ValueError("No training trips available for cross-validation")

    rows: list[dict[str, object]] = []
    for model_name in model_names:
        fold_results: list[ModelResult] = []
        for fold_index, (train_ids, validation_ids) in enumerate(folds):
            train_features, train_labels, _ = stack_sampled(
                bundle, train_ids, max_observations_per_trip
            )
            validation_features, validation_labels, validation_groups = stack_sampled(
                bundle, validation_ids, max_observations_per_trip
            )
            trained = fit_model(model_name, train_features, train_labels, seed=seed)
            result = evaluate_model(
                trained,
                validation_features,
                validation_labels,
                validation_groups,
            )
            rows.append(_result_row(result, scope="fold", fold=fold_index))
            fold_results.append(result)
        rows.append(_aggregate_row(model_name, fold_results))

    return pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS))


def evaluate_on_test(
    bundle: DatasetBundle,
    model_names: Sequence[str] | None = None,
    seed: int = config.CV_SEED,
    train_max_observations_per_trip: int | None = config.CV_MAX_OBSERVATIONS_PER_TRIP,
    test_max_observations_per_trip: int | None = config.TEST_MAX_OBSERVATIONS_PER_TRIP,
) -> pd.DataFrame:
    """Fit each model on all training trips and score it on student-test trips."""
    model_names = list(MODEL_SPECS) if model_names is None else list(model_names)
    training_trip_ids = [trip.trip_id for trip in bundle.training_trips()]
    test_trip_ids = [trip.trip_id for trip in bundle.test_trips()]
    if not test_trip_ids:
        raise ValueError("No student-test trips available")

    train_features, train_labels, _ = stack_sampled(
        bundle, training_trip_ids, train_max_observations_per_trip
    )
    test_features, test_labels, test_groups = stack_sampled(
        bundle, test_trip_ids, test_max_observations_per_trip
    )

    rows: list[dict[str, object]] = []
    for model_name in model_names:
        trained = fit_model(model_name, train_features, train_labels, seed=seed)
        result = evaluate_model(trained, test_features, test_labels, test_groups)
        rows.append(_result_row(result, scope="student_test", fold=-1))
    return pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS))


def select_best_model(comparison: pd.DataFrame) -> str:
    """Select the highest CV Macro F1 model, breaking ties on Top-1 accuracy."""
    aggregates = comparison[comparison["scope"] == "cv_aggregate"].copy()
    if aggregates.empty:
        raise ValueError("Comparison table has no cv_aggregate rows")
    aggregates = aggregates.sort_values(
        ["macro_f1", "top1_accuracy"], ascending=[False, False]
    )
    return str(aggregates.iloc[0]["model"])


def write_selection_note(
    comparison: pd.DataFrame, chosen_model: str, out_path: Path
) -> None:
    aggregates = comparison[comparison["scope"] == "cv_aggregate"].sort_values(
        "macro_f1", ascending=False
    )
    lines = [
        "# Model selection note",
        "",
        "Selection metric: mean trip-grouped CV Macro F1, with CV Top-1 accuracy "
        "as the tie-breaker. Training accuracy was not used.",
        "",
        f"**Selected model:** `{chosen_model}`",
        "",
        "| Model | Family | CV Macro F1 | CV Top-1 | CV Log Loss |",
        "| :--- | :--- | ---: | ---: | ---: |",
    ]
    for _, row in aggregates.iterrows():
        lines.append(
            f"| {row['model']} | {row['family']} | {row['macro_f1']:.4f} | "
            f"{row['top1_accuracy']:.4f} | {row['log_loss']:.4f} |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Decoder comparison (nearest-snap / classifier-only / classifier + HMM)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TripInference:
    """Prepared candidates, probabilities, alignment, and arc lengths for one trip."""

    trip: Trip
    candidates: CandidateSet
    probabilities: np.ndarray
    alignment: AlignmentResult
    arc_lengths: np.ndarray


def prepare_trip_inference(trip: Trip, artifact: ModelArtifact) -> TripInference:
    """Generate all candidates, score them with the artifact, and align labels."""
    alignment = align_observations(trip)
    candidates = generate_candidates(trip)
    features = extract_features(trip, candidates)
    probabilities = artifact.positive_probability(features.features)
    return TripInference(
        trip=trip,
        candidates=candidates,
        probabilities=probabilities,
        alignment=alignment,
        arc_lengths=arc_lengths_m(trip),
    )


def _decode_row(
    inference: TripInference, result: DecodeResult, inference_ms: float
) -> dict[str, Any]:
    alignment = inference.alignment
    point_accuracy = observation_top1_accuracy(
        alignment.arc_ids, result.point_arc_ids, alignment.alignable
    )
    precision, recall = route_precision_recall(result.predicted_arcs, inference.trip.route)
    mismatch = route_mismatch_fraction(
        result.predicted_arcs, inference.trip.route, inference.arc_lengths
    )
    return {
        "trip_id": inference.trip.trip_id,
        "partition": inference.trip.partition,
        "method": result.method,
        "point_accuracy": point_accuracy,
        "route_continuity": route_continuity(result.predicted_arcs, inference.trip.arcs),
        "route_precision": precision,
        "route_recall": recall,
        "rmf": mismatch,
        "alignment_coverage": alignment.coverage,
        "inference_ms": inference_ms,
        "n_points": inference.trip.n_points,
        "n_missing_points": int(np.count_nonzero(result.point_arc_ids < 0)),
        "n_predicted_arcs": int(result.predicted_arcs.size),
        "n_unbridged_transitions": result.n_unbridged_transitions,
    }


def evaluate_decoders(
    inference: TripInference,
    hmm_config: HMMConfig | None = None,
) -> pd.DataFrame:
    """Run nearest-snap, classifier-only, and classifier+HMM on one prepared trip."""
    hmm_config = hmm_config or HMMConfig()

    with Timer() as nearest_timer:
        nearest = decode_nearest(inference.trip, inference.candidates)
    with Timer() as classifier_timer:
        classifier = decode_classifier(
            inference.trip, inference.candidates, inference.probabilities
        )
    with Timer() as hmm_timer:
        hmm = decode_hmm(
            inference.trip,
            inference.candidates,
            probabilities=inference.probabilities,
            config=replace(hmm_config, emission="classifier"),
        )

    rows = [
        _decode_row(inference, nearest, nearest_timer.elapsed_ms),
        _decode_row(inference, classifier, classifier_timer.elapsed_ms),
        _decode_row(inference, hmm, hmm_timer.elapsed_ms),
    ]
    return pd.DataFrame(rows, columns=list(DECODER_COLUMNS))


DECODER_TUNING_COLUMNS: tuple[str, ...] = DECODER_COLUMNS + ("beta_m",)


def tune_decoder_betas(
    inferences: Sequence[TripInference],
    betas: Sequence[float] = (5.0, 10.0, 20.0, 50.0),
    base_config: HMMConfig | None = None,
) -> tuple[pd.DataFrame, float]:
    """Grid-search beta on training/student-test trips only.

    Returns the per-trial table and the beta with the best mean point accuracy
    (route continuity breaks ties).
    """
    base_config = base_config or HMMConfig()
    rows: list[dict[str, Any]] = []

    for beta in betas:
        config_for_beta = replace(base_config, beta_m=float(beta), emission="classifier")
        for inference in inferences:
            with Timer() as timer:
                result = decode_hmm(
                    inference.trip,
                    inference.candidates,
                    probabilities=inference.probabilities,
                    config=config_for_beta,
                )
            row = _decode_row(inference, result, timer.elapsed_ms)
            row["beta_m"] = float(beta)
            rows.append(row)

    frame = pd.DataFrame(rows, columns=list(DECODER_TUNING_COLUMNS))
    if frame.empty:
        return frame, float(base_config.beta_m)
    summary = (
        frame.groupby("beta_m")[["point_accuracy", "route_continuity"]]
        .mean()
        .sort_values(["point_accuracy", "route_continuity"], ascending=[False, False])
    )
    return frame, float(summary.index[0])


def write_decoder_failure_analysis(frame: pd.DataFrame, out_path: Path) -> None:
    """Compare classifier-only vs HMM per trip and document improvement/regression."""
    accuracy = frame.pivot_table(
        index="trip_id", columns="method", values="point_accuracy"
    )
    continuity = frame.pivot_table(
        index="trip_id", columns="method", values="route_continuity"
    )
    delta_accuracy = accuracy["hmm"] - accuracy["classifier"]
    delta_continuity = continuity["hmm"] - continuity["classifier"]

    best_trip = str(delta_accuracy.idxmax())
    worst_trip = str(delta_accuracy.idxmin())

    lines = [
        "# Decoder failure analysis",
        "",
        "Comparison of classifier-only predictions with classifier+HMM decoding. "
        "Positive deltas mean HMM improved the metric.",
        "",
        "| Trip | Classifier accuracy | HMM accuracy | Δ accuracy | "
        "Classifier continuity | HMM continuity | Δ continuity |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for trip_id in accuracy.index:
        lines.append(
            f"| {trip_id} | {accuracy.loc[trip_id, 'classifier']:.4f} | "
            f"{accuracy.loc[trip_id, 'hmm']:.4f} | {delta_accuracy.loc[trip_id]:+.4f} | "
            f"{continuity.loc[trip_id, 'classifier']:.4f} | "
            f"{continuity.loc[trip_id, 'hmm']:.4f} | "
            f"{delta_continuity.loc[trip_id]:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Case study: HMM helps",
            "",
            f"Trip `{best_trip}` gained {delta_accuracy.loc[best_trip]:+.4f} point "
            f"accuracy and {delta_continuity.loc[best_trip]:+.4f} route continuity. "
            "The transition term suppresses locally plausible but topologically "
            "isolated candidates, so the decoded sequence follows connected arcs "
            "instead of independent per-point snap decisions.",
            "",
            "## Case study: HMM does not help",
            "",
        ]
    )
    worst_delta = float(delta_accuracy.loc[worst_trip])
    if worst_delta < 0:
        lines.append(
            f"Trip `{worst_trip}` lost {abs(worst_delta):.4f} point accuracy "
            f"(continuity delta {delta_continuity.loc[worst_trip]:+.4f}). When the "
            "GPS noise or a parallel-road ambiguity makes several candidates nearly "
            "equally likely, the transition penalty can prefer a connected but "
            "locally wrong arc, overriding the classifier's top choice."
        )
    else:
        lines.append(
            f"Trip `{worst_trip}` showed the smallest gain "
            f"({worst_delta:+.4f} point accuracy, continuity delta "
            f"{delta_continuity.loc[worst_trip]:+.4f}). No student-test trip "
            "regressed in point accuracy; this trip is the weakest positive case, "
            "where connectivity constraints had little room to correct errors."
        )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")

