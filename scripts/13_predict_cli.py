"""Run the saved PathSnap model live from the terminal, without an API server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from pathsnap import config
from pathsnap.artifacts import load_hmm_config, load_model_artifact
from pathsnap.io import discover_trip_dirs, load_trip
from pathsnap.pipeline import (
    METRIC_LABELS,
    predict_trip,
    write_prediction_outputs,
)

EVALUATOR_DIR = config.PROJECT_ROOT / "data" / "evaluator_test"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=EVALUATOR_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--trip-id", help="Run one trip, for example 00000023")
    selection.add_argument("--all", action="store_true", help="Run every trip in the partition")
    selection.add_argument("--list", action="store_true", help="List available trip IDs")
    parser.add_argument("--method", choices=["hmm", "classifier", "nearest"], default="hmm")
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Additional GPS noise standard deviation in metres",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=1.0,
        help="Sampling interval; 0 keeps every input observation",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _print_metrics(result: object) -> None:
    metrics = result.metrics
    for label, key in METRIC_LABELS:
        print(f"  {label:<24} {metrics[key] * 100:8.2f}%")
    print(f"  {'Classifier log loss':<24} {metrics['log_loss']:9.4f}")
    print(f"  {'Route mismatch fraction':<24} {metrics['rmf']:9.4f}")
    print(f"  {'Inference time':<24} {result.inference_ms / 1000:8.3f} s")
    print(f"  {'Predicted road segments':<24} {len(result.predicted_arcs):8d}")


def main() -> None:
    args = _parse_args()
    if args.noise < 0 or args.sample_seconds < 0:
        raise SystemExit("--noise and --sample-seconds must be nonnegative")

    paths = config.get_paths()
    artifact_dir = args.artifact_dir or paths.artifact_dir
    output_dir = args.output_dir or (paths.results_dir / "live_demo")
    partition = args.data_dir.name

    trips = discover_trip_dirs(args.data_dir)
    if not trips:
        raise SystemExit(f"No trips found in {args.data_dir}")
    if args.list:
        print("\n".join(path.name for path in trips))
        return
    if args.trip_id:
        trips = [path for path in trips if path.name == args.trip_id]
        if not trips:
            raise SystemExit("Unknown trip ID. Use --list to see available trips.")
    elif not args.all:
        raise SystemExit("Choose --trip-id ID, --all, or --list")

    artifact = load_model_artifact(artifact_dir / "model.joblib")
    settings = load_hmm_config(artifact_dir / "hmm.json")
    print(
        f"Model: {artifact.model_name} | trees={artifact.params.get('n_estimators')} "
        f"| depth={artifact.params.get('max_depth')} "
        f"| features={len(artifact.feature_names)}"
    )
    print(
        f"Method: {args.method} | noise={args.noise:g} m "
        f"| sampling={args.sample_seconds:g} s"
    )
    print(f"Data: {args.data_dir.resolve()}")
    print("Inference runs now; no training or cached predictions are used.\n", flush=True)

    summaries = []
    for directory in trips:
        print(f"Running trip {directory.name}...", flush=True)
        trip = load_trip(directory, partition)
        result = predict_trip(
            trip,
            artifact,
            method=args.method,
            noise_sigma_m=args.noise,
            sample_every_s=args.sample_seconds,
            hmm_config=settings,
        )
        _print_metrics(result)
        summaries.append(write_prediction_outputs(result, output_dir))
        print("", flush=True)

    scored = sum(row["scored_points"] for row in summaries)
    correct = sum(row["correct_points"] for row in summaries)
    print(f"Completed {len(summaries)} trip(s).")
    if scored:
        print(f"Pooled point accuracy: {100 * correct / scored:.2f}%")
    else:
        print("No alignable points to score.")
    mean_f1 = np.mean([row["decoder_macro_f1"] for row in summaries])
    print(f"Mean-trip decoder Macro F1: {100 * mean_f1:.2f}%")
    print(f"Predictions and metrics saved in: {output_dir.resolve()}")
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
