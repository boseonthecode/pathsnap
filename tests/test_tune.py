"""Tests for Optuna tuning and tuned-vs-untuned comparison."""

from __future__ import annotations

from pathlib import Path

from pathsnap.cache import build_trip_artifacts
from pathsnap.dataset import assemble_dataset
from pathsnap.io import load_trip
from pathsnap.models import fit_model
from pathsnap.tune import (
    TUNED_COMPARISON_COLUMNS,
    TRIAL_COLUMNS,
    compare_tuned_vs_untuned,
    fit_final_model,
    run_optimisation,
)
from conftest import write_trip


def _bundle(tmp_path: Path):
    trips = []
    artifacts = []
    for index in range(4):
        trip = load_trip(write_trip(tmp_path / "train", f"0000006{index}"), "train")
        trips.append(trip)
        artifacts.append(build_trip_artifacts(trip))
    test_trip = load_trip(write_trip(tmp_path / "test", "00000001"), "test")
    trips.append(test_trip)
    artifacts.append(build_trip_artifacts(test_trip))
    return assemble_dataset(trips, artifacts)


def test_optuna_records_at_least_ten_trials_and_best_refits(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    tuning = run_optimisation(
        bundle,
        model_name="extra_trees",
        n_trials=10,
        timeout_s=None,
        n_splits=2,
        max_observations_per_trip=None,
    )

    assert tuning.n_completed >= 10
    assert list(tuning.trials.columns) == list(TRIAL_COLUMNS)
    assert tuning.trials["params_json"].notna().all()
    assert tuning.trials["cv_macro_f1"].notna().all()
    assert 0.0 <= tuning.best_value <= 1.0
    assert tuning.best_params

    features, labels = bundle.stacked(
        [trip.trip_id for trip in bundle.training_trips()]
    )
    trained = fit_model(
        "extra_trees", features, labels, params=tuning.best_params
    )
    probabilities = trained.positive_probability(features)
    assert probabilities.shape == (features.shape[0],)


def test_tuned_vs_untuned_reports_both_variants(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    best_params = {
        "n_estimators": 120,
        "max_depth": 8,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "criterion": "gini",
        "class_weight": "balanced",
    }

    comparison = compare_tuned_vs_untuned(
        bundle,
        best_params,
        model_name="extra_trees",
        n_splits=2,
        max_observations_per_trip=None,
        test_max_observations_per_trip=None,
    )

    assert list(comparison.columns) == list(TUNED_COMPARISON_COLUMNS)
    assert set(comparison["variant"]) == {"untuned", "tuned"}
    assert comparison["cv_macro_f1"].between(0, 1).all()
    assert comparison["student_macro_f1"].between(0, 1).all()
    assert (comparison["student_inference_ms"] >= 0).all()


def test_final_model_trains_on_all_training_trips(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    best_params = {
        "n_estimators": 100,
        "max_depth": 8,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "criterion": "gini",
        "class_weight": "balanced",
    }

    trained = fit_final_model(bundle, "extra_trees", best_params)

    assert trained.spec.name == "extra_trees"
    assert trained.spec.params["n_estimators"] == 100
    assert trained.train_time_s >= 0.0


def test_suggest_params_stays_inside_the_configured_ranges() -> None:
    import optuna

    from pathsnap.tune import suggest_params

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trial = optuna.create_study(direction="maximize").ask()
    params = suggest_params(trial)

    assert 100 <= params["n_estimators"] <= 200
    assert 8 <= params["max_depth"] <= 14
    assert 10 <= params["min_samples_leaf"] <= 30
    assert params["max_features"] in {"sqrt", "log2", 0.5}
    assert params["criterion"] in {"gini", "entropy"}
    assert params["class_weight"] in {"balanced_subsample", "balanced"}
