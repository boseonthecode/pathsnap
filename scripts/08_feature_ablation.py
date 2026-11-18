"""Feature selection and ablation on trip-grouped training folds."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from pathsnap import config
from pathsnap.ablation import (
    run_feature_ablation,
    select_features,
    write_feature_selection_note,
)
from pathsnap.artifacts import write_json
from pathsnap.dataset import assemble_dataset
from pathsnap.io import iter_trips
from pathsnap.viz import plot_feature_ablation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature selection and ablation")
    parser.add_argument("--model", type=str, default="extra_trees")
    parser.add_argument("--folds", type=int, default=config.N_SPLITS)
    parser.add_argument("--tolerance", type=float, default=0.002)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def _tuned_params(results_dir: Path) -> dict:
    path = results_dir / "best_params.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return dict(payload.get("best_params", {}))
    return {}


def main() -> None:
    args = _parse_args()
    paths = config.get_paths()
    paths.validate_data()
    config.ensure_output_directories(paths)
    results_dir = args.results_dir or paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    params = _tuned_params(results_dir)

    start = time.perf_counter()
    trips = list(iter_trips(paths.train_data_dir, "train")) + list(
        iter_trips(paths.test_data_dir, "test")
    )
    bundle = assemble_dataset(trips, load_candidates=False)
    print(
        f"Loaded {len(bundle.training_trips())} training trips; "
        f"feature selection uses training folds only."
    )

    ablation = run_feature_ablation(
        bundle,
        model_name=args.model,
        params=params,
        n_splits=args.folds,
    )
    ablation_path = results_dir / "feature_ablation.csv"
    ablation.to_csv(ablation_path, index=False)

    selected, selected_row = select_features(
        ablation,
        bundle,
        model_name=args.model,
        params=params,
        tolerance=args.tolerance,
        n_splits=args.folds,
    )
    write_feature_selection_note(
        ablation, selected, selected_row, results_dir / "feature_selection.md"
    )
    write_json(
        results_dir / "selected_features.json",
        {
            "model": args.model,
            "params": params,
            "tolerance": args.tolerance,
            "selected_features": selected,
            "selected_cv_macro_f1": selected_row["cv_macro_f1"],
            "full_cv_macro_f1": float(
                ablation.loc[ablation["config"] == "full", "cv_macro_f1"].iloc[0]
            ),
        },
    )
    figure_path = plot_feature_ablation(
        ablation, results_dir / "figures" / "feature_ablation.png"
    )

    full = ablation[ablation["config"] == "full"].iloc[0]
    print(
        f"Full ({int(full['n_features'])} features): CV Macro F1="
        f"{full['cv_macro_f1']:.4f}"
    )
    print(
        f"Selected ({len(selected)} features): CV Macro F1="
        f"{selected_row['cv_macro_f1']:.4f}"
    )
    print(f"Selected features: {selected}")
    print(f"Wrote {ablation_path}")
    print(f"Wrote {figure_path}")
    print(f"Total runtime: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
