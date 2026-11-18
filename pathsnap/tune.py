"""Bayesian hyperparameter optimisation and tuned-vs-untuned comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import optuna
import pandas as pd

from pathsnap import config
from pathsnap.dataset import DatasetBundle, trip_grouped_folds
from pathsnap.evaluate import stack_sampled
from pathsnap.models import ModelResult, evaluate_model, fit_model

TUNED_COMPARISON_COLUMNS: tuple[str, ...] = (
    "variant",
    "model",
    "cv_macro_f1",
    "cv_top1_accuracy",
    "cv_log_loss",
    "cv_inference_ms",
    "student_macro_f1",
    "student_top1_accuracy",
    "student_log_loss",
    "student_inference_ms",
    "train_time_s",
)

TRIAL_COLUMNS: tuple[str, ...] = (
    "trial",
    "state",
    "params_json",
    "cv_macro_f1",
    "duration_s",
)


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Size-bounded search space for the ExtraTrees candidate scorer.

    Leaves and depth are capped so the exported artifact stays within a
    repository-friendly size while still exploring meaningful capacity.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 200, step=50),
        "max_depth": trial.suggest_int("max_depth", 8, 14, step=2),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 30, step=10),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.5]
        ),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "class_weight": trial.suggest_categorical(
            "class_weight", ["balanced_subsample", "balanced"]
        ),
    }


@dataclass(frozen=True)
class TuningResult:
    model_name: str
    best_params: dict[str, Any]
    best_value: float
    n_completed: int
    trials: pd.DataFrame


def _fold_data(
    bundle: DatasetBundle,
    n_splits: int,
    seed: int,
    max_observations_per_trip: int | None,
) -> list[
    tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ]
]:
    training_trip_ids = [trip.trip_id for trip in bundle.training_trips()]
    folds = trip_grouped_folds(training_trip_ids, n_splits=n_splits, seed=seed)
    return [
        (
            stack_sampled(bundle, train_ids, max_observations_per_trip),
            stack_sampled(bundle, validation_ids, max_observations_per_trip),
        )
        for train_ids, validation_ids in folds
    ]


def _trials_frame(study: optuna.Study) -> pd.DataFrame:
    records = []
    for trial in study.trials:
        records.append(
            {
                "trial": trial.number,
                "state": trial.state.name,
                "params_json": json.dumps(trial.params, sort_keys=True),
                "cv_macro_f1": trial.value if trial.value is not None else np.nan,
                "duration_s": (
                    trial.duration.total_seconds() if trial.duration else 0.0
                ),
            }
        )
    return pd.DataFrame(records, columns=list(TRIAL_COLUMNS))


def run_optimisation(
    bundle: DatasetBundle,
    model_name: str = "extra_trees",
    n_trials: int = 15,
    timeout_s: float | None = 600.0,
    n_splits: int = config.N_SPLITS,
    seed: int = config.DEFAULT_SEED,
    max_observations_per_trip: int | None = config.CV_MAX_OBSERVATIONS_PER_TRIP,
) -> TuningResult:
    """Run TPE search with a trip-grouped CV Macro F1 objective."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fold_data = _fold_data(bundle, n_splits, seed, max_observations_per_trip)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        scores: list[float] = []
        for (train_features, train_labels, _), (
            validation_features,
            validation_labels,
            validation_groups,
        ) in fold_data:
            trained = fit_model(
                model_name, train_features, train_labels, seed=seed, params=params
            )
            result = evaluate_model(
                trained, validation_features, validation_labels, validation_groups
            )
            scores.append(result.macro_f1)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s)
    trials = _trials_frame(study)
    completed = int((trials["state"] == "COMPLETE").sum())
    best_value = float(study.best_value)
    return TuningResult(
        model_name=model_name,
        best_params=dict(study.best_params),
        best_value=best_value,
        n_completed=completed,
        trials=trials,
    )


def _mean_metric(results: Sequence[ModelResult], attribute: str) -> float:
    return float(np.mean([getattr(result, attribute) for result in results]))


def _compare_variant(
    variant: str,
    model_name: str,
    params: dict[str, Any] | None,
    fold_data,
    train_all,
    test_all,
    seed: int,
) -> dict[str, Any]:
    fold_results: list[ModelResult] = []
    for (train_features, train_labels, _), (
        validation_features,
        validation_labels,
        validation_groups,
    ) in fold_data:
        trained = fit_model(
            model_name, train_features, train_labels, seed=seed, params=params
        )
        fold_results.append(
            evaluate_model(
                trained, validation_features, validation_labels, validation_groups
            )
        )

    train_features, train_labels, _ = train_all
    test_features, test_labels, test_groups = test_all
    trained = fit_model(model_name, train_features, train_labels, seed=seed, params=params)
    test_result = evaluate_model(trained, test_features, test_labels, test_groups)

    return {
        "variant": variant,
        "model": model_name,
        "cv_macro_f1": _mean_metric(fold_results, "macro_f1"),
        "cv_top1_accuracy": _mean_metric(fold_results, "top1_accuracy"),
        "cv_log_loss": _mean_metric(fold_results, "log_loss"),
        "cv_inference_ms": _mean_metric(fold_results, "inference_ms"),
        "student_macro_f1": test_result.macro_f1,
        "student_top1_accuracy": test_result.top1_accuracy,
        "student_log_loss": test_result.log_loss,
        "student_inference_ms": test_result.inference_ms,
        "train_time_s": test_result.train_time_s,
    }


def compare_tuned_vs_untuned(
    bundle: DatasetBundle,
    best_params: dict[str, Any],
    model_name: str = "extra_trees",
    n_splits: int = config.N_SPLITS,
    seed: int = config.DEFAULT_SEED,
    max_observations_per_trip: int | None = config.CV_MAX_OBSERVATIONS_PER_TRIP,
    test_max_observations_per_trip: int | None = config.TEST_MAX_OBSERVATIONS_PER_TRIP,
) -> pd.DataFrame:
    """Evaluate untuned and tuned configurations on identical folds and test data."""
    fold_data = _fold_data(bundle, n_splits, seed, max_observations_per_trip)
    training_trip_ids = [trip.trip_id for trip in bundle.training_trips()]
    test_trip_ids = [trip.trip_id for trip in bundle.test_trips()]
    train_all = stack_sampled(bundle, training_trip_ids, max_observations_per_trip)
    test_all = stack_sampled(bundle, test_trip_ids, test_max_observations_per_trip)

    rows = [
        _compare_variant(
            "untuned", model_name, None, fold_data, train_all, test_all, seed
        ),
        _compare_variant(
            "tuned", model_name, best_params, fold_data, train_all, test_all, seed
        ),
    ]
    return pd.DataFrame(rows, columns=list(TUNED_COMPARISON_COLUMNS))


def fit_final_model(
    bundle: DatasetBundle,
    model_name: str,
    params: dict[str, Any] | None,
    seed: int = config.DEFAULT_SEED,
    max_observations_per_trip: int | None = None,
):
    """Fit the exported model on all available training trips."""
    training_trip_ids = [trip.trip_id for trip in bundle.training_trips()]
    features, labels, _ = stack_sampled(
        bundle, training_trip_ids, max_observations_per_trip
    )
    return fit_model(model_name, features, labels, seed=seed, params=params)
