"""Steps 16/17 — Optuna tuning, tuned-vs-untuned comparison, artifact export."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from pathsnap import config
from pathsnap.artifacts import (
    build_model_artifact,
    save_model_artifact,
    write_config_json,
    write_json,
)
from pathsnap.dataset import assemble_dataset
from pathsnap.hmm import HMMConfig
from pathsnap.io import iter_trips
from pathsnap.tune import (
    compare_tuned_vs_untuned,
    fit_final_model,
    run_optimisation,
)


def _hmm_config() -> dict[str, object]:
    """Persist the full decoder configuration, not a hand-picked subset."""
    return HMMConfig().to_dict()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune and export the PathSnap model")
    parser.add_argument("--model", type=str, default="extra_trees")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--folds", type=int, default=config.N_SPLITS)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument(
        "--final-max-observations-per-trip",
        type=int,
        default=300,
        help="Cap the final fit observations per trip (default: 300, matching CV)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = config.get_paths()
    paths.validate_data()
    config.ensure_output_directories(paths)
    results_dir = args.results_dir or paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    trips = list(iter_trips(paths.train_data_dir, "train")) + list(
        iter_trips(paths.test_data_dir, "test")
    )
    bundle = assemble_dataset(trips, load_candidates=False)
    print(f"Loaded {len(bundle.training_trips())} training and {len(bundle.test_trips())} test trips")

    tuning = run_optimisation(
        bundle,
        model_name=args.model,
        n_trials=args.trials,
        timeout_s=args.timeout,
        n_splits=args.folds,
    )
    trials_path = results_dir / "optuna_trials.csv"
    tuning.trials.to_csv(trials_path, index=False)
    best_params_path = results_dir / "best_params.json"
    write_json(
        best_params_path,
        {
            "model": tuning.model_name,
            "best_params": tuning.best_params,
            "best_cv_macro_f1": tuning.best_value,
            "completed_trials": tuning.n_completed,
        },
    )
    print(
        f"Optuna: {tuning.n_completed} completed trials, "
        f"best CV Macro F1={tuning.best_value:.4f}"
    )
    print(f"Best params: {tuning.best_params}")

    comparison = compare_tuned_vs_untuned(
        bundle, tuning.best_params, model_name=args.model, n_splits=args.folds
    )
    comparison_path = results_dir / "tuned_vs_untuned.csv"
    comparison.to_csv(comparison_path, index=False)
    print(comparison.to_string(index=False))

    final_model = fit_final_model(
        bundle,
        args.model,
        tuning.best_params,
        max_observations_per_trip=args.final_max_observations_per_trip,
    )
    artifact = build_model_artifact(
        final_model,
        metadata={
            "selection": "optuna",
            "cv_macro_f1": tuning.best_value,
            "best_params": tuning.best_params,
            "completed_trials": tuning.n_completed,
        },
    )
    model_path = save_model_artifact(artifact, paths.artifact_dir / "model.joblib")
    write_config_json(
        paths.artifact_dir / "config.json",
        artifact,
        candidate_config={
            "top_k": config.CANDIDATE_TOP_K,
            "radius_m": config.CANDIDATE_RADIUS_M,
            "max_radius_m": config.CANDIDATE_MAX_RADIUS_M,
        },
        alignment_config={
            "max_distance_m": config.ALIGN_MAX_DISTANCE_M,
            "low_coverage_threshold": config.LOW_COVERAGE_THRESHOLD,
        },
        hmm_config=_hmm_config(),
        seed=config.DEFAULT_SEED,
    )
    write_json(paths.artifact_dir / "hmm.json", _hmm_config())

    print(f"Wrote {trials_path}")
    print(f"Wrote {best_params_path}")
    print(f"Wrote {comparison_path}")
    print(f"Wrote {model_path}")
    print(f"Total runtime: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
