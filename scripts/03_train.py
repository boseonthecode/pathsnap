"""Step 14/15 — trip-grouped CV, model comparison, selection, and plots."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from pathsnap import config
from pathsnap.dataset import assemble_dataset
from pathsnap.evaluate import (
    cross_validate,
    evaluate_on_test,
    select_best_model,
    write_selection_note,
)
from pathsnap.io import iter_trips
from pathsnap.models import MODEL_SPECS
from pathsnap.viz import generate_all_plots


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare PathSnap models")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=config.N_SPLITS)
    parser.add_argument(
        "--max-observations-per-trip",
        type=int,
        default=config.CV_MAX_OBSERVATIONS_PER_TRIP,
    )
    parser.add_argument(
        "--test-max-observations-per-trip",
        type=int,
        default=config.TEST_MAX_OBSERVATIONS_PER_TRIP,
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=f"Subset of {sorted(MODEL_SPECS)} (default: all)",
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
    train_trips = list(iter_trips(paths.train_data_dir, "train"))
    test_trips = list(iter_trips(paths.test_data_dir, "test"))
    bundle = assemble_dataset(train_trips + test_trips, load_candidates=False)
    print(
        f"Dataset: {len(bundle.training_trips())} training trips, "
        f"{len(bundle.test_trips())} student-test trips, "
        f"{len(bundle.excluded_training_trips())} low-coverage training trips excluded."
    )

    comparison = cross_validate(
        bundle,
        model_names=args.models,
        n_splits=args.folds,
        max_observations_per_trip=args.max_observations_per_trip,
    )
    student_test = evaluate_on_test(
        bundle,
        model_names=args.models,
        test_max_observations_per_trip=args.test_max_observations_per_trip,
    )

    comparison_path = results_dir / "model_comparison.csv"
    student_path = results_dir / "student_test_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    student_test.to_csv(student_path, index=False)

    chosen_model = select_best_model(comparison)
    write_selection_note(comparison, chosen_model, results_dir / "model_selection.md")

    combined = pd.concat(
        [pd.read_csv(comparison_path), pd.read_csv(student_path)], ignore_index=True
    )
    plot_paths = generate_all_plots(combined, results_dir / "figures")

    elapsed = time.perf_counter() - start
    print(f"Selected model: {chosen_model}")
    print(f"Wrote {comparison_path} ({len(comparison)} rows)")
    print(f"Wrote {student_path} ({len(student_test)} rows)")
    print(f"Wrote {len(plot_paths)} figures to {results_dir / 'figures'}")
    print(f"Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
