"""Step 20/21 — decoder comparison, beta tuning, plots, and failure analysis."""

from __future__ import annotations

import argparse
import time
from dataclasses import fields, replace
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from pathsnap import config
from pathsnap.artifacts import load_model_artifact, read_json, write_json
from pathsnap.evaluate import (
    evaluate_decoders,
    prepare_trip_inference,
    tune_decoder_betas,
    write_decoder_failure_analysis,
)
from pathsnap.hmm import HMMConfig
from pathsnap.io import iter_trips
from pathsnap.viz import generate_decoder_plots


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare map-matching decoders")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--betas", nargs="*", type=float, default=[5.0, 10.0, 20.0, 50.0])
    parser.add_argument("--tune-test-trips", type=int, default=3)
    parser.add_argument("--tune-train-trips", type=int, default=2)
    return parser.parse_args()


def _base_hmm_config(path: Path) -> HMMConfig:
    if not path.is_file():
        return HMMConfig()
    payload = read_json(path)
    allowed = {field.name for field in fields(HMMConfig)}
    return HMMConfig(**{key: value for key, value in payload.items() if key in allowed})


def main() -> None:
    args = _parse_args()
    paths = config.get_paths()
    paths.validate_data()
    config.ensure_output_directories(paths)
    results_dir = args.results_dir or paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    artifact = load_model_artifact(paths.artifact_dir / "model.joblib")
    base_config = _base_hmm_config(paths.artifact_dir / "hmm.json")

    train_trips = list(iter_trips(paths.train_data_dir, "train"))
    test_trips = list(iter_trips(paths.test_data_dir, "test"))
    tuning_trips = test_trips[: args.tune_test_trips] + train_trips[: args.tune_train_trips]
    print(
        f"Tuning beta on {len(tuning_trips)} trips "
        f"({min(args.tune_train_trips, len(train_trips))} train, "
        f"{min(args.tune_test_trips, len(test_trips))} student-test)"
    )

    inferences = {}
    for trip in tuning_trips:
        inferences[trip.trip_id] = prepare_trip_inference(trip, artifact)
    tuning_frame, best_beta = tune_decoder_betas(
        list(inferences.values()), betas=args.betas, base_config=base_config
    )
    tuning_path = results_dir / "decoder_tuning.csv"
    tuning_frame.to_csv(tuning_path, index=False)

    best_config = replace(base_config, beta_m=best_beta, emission="classifier")
    write_json(results_dir / "best_decoder_params.json", best_config.to_dict())
    write_json(paths.artifact_dir / "hmm.json", best_config.to_dict())
    config_path = paths.artifact_dir / "config.json"
    if config_path.is_file():
        config_payload = read_json(config_path)
        config_payload["hmm"] = best_config.to_dict()
        write_json(config_path, config_payload)
    print(f"Best beta: {best_beta:g}")

    rows = []
    for trip in test_trips:
        inference = inferences.get(trip.trip_id)
        if inference is None:
            inference = prepare_trip_inference(trip, artifact)
        rows.append(evaluate_decoders(inference, best_config))
    comparison = pd.concat(rows, ignore_index=True)
    comparison_path = results_dir / "hmm_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    stored = pd.read_csv(comparison_path, dtype={"trip_id": str})
    plot_paths = generate_decoder_plots(stored, results_dir / "figures")
    write_decoder_failure_analysis(stored, results_dir / "hmm_failure_analysis.md")

    summary = stored.groupby("method")[["point_accuracy", "route_continuity"]].mean()
    print(summary.to_string())
    print(f"Wrote {tuning_path}")
    print(f"Wrote {comparison_path}")
    print(f"Wrote {len(plot_paths)} decoder figures")
    print(f"Total runtime: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
