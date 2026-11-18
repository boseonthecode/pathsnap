"""Trip-grouped cross-validation, model comparison, and selection helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.dataset import DatasetBundle, TripDataset, trip_grouped_folds
from pathsnap.models import MODEL_SPECS, ModelResult, evaluate_model, fit_model

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
