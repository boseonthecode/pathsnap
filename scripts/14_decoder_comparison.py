"""Regenerate the three-decoder comparison at the deployed HMM settings.

Uses ``artifacts/hmm.json`` exactly as served by the API: no beta grid search,
no artifact writes. Rewrites ``results/hmm_comparison.csv``, its figures, and
the failure analysis so the report describes the shipped configuration.

It writes the same three outputs as ``scripts/05_hmm.py``; run one or the other,
not both, or the later run wins.

Run from the repository root: python scripts/14_decoder_comparison.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from pathsnap import config
from pathsnap.artifacts import load_hmm_config, load_model_artifact
from pathsnap.evaluate import (
    evaluate_decoders,
    prepare_trip_inference,
    write_decoder_failure_analysis,
)
from pathsnap.io import iter_trips
from pathsnap.viz import generate_decoder_plots


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    start = time.perf_counter()
    paths = config.get_paths()
    if args.data_dir is not None:
        paths.validate_data()
    data_dir = args.data_dir or paths.test_data_dir
    results_dir = args.results_dir or paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)

    artifact = load_model_artifact(paths.artifact_dir / "model.joblib")
    settings = load_hmm_config(paths.artifact_dir / "hmm.json")
    print(
        f"Deployed config: beta_m={settings.beta_m:g} "
        f"refine_points={settings.refine_points} emission={settings.emission}"
    )

    rows = []
    for trip in iter_trips(data_dir, "test"):
        inference = prepare_trip_inference(trip, artifact)
        rows.append(evaluate_decoders(inference, settings))
        print(f"  {trip.trip_id} decoded", flush=True)

    comparison = pd.concat(rows, ignore_index=True)
    comparison_path = results_dir / "hmm_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    generate_decoder_plots(comparison, results_dir / "figures")
    write_decoder_failure_analysis(comparison, results_dir / "hmm_failure_analysis.md")

    summary = comparison.groupby("method")[
        ["point_accuracy", "route_continuity", "route_precision", "route_recall"]
    ].mean()
    print(summary.to_string())
    print(f"Wrote {comparison_path}")
    print(f"Runtime: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
