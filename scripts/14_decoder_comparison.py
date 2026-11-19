"""Regenerate the three-decoder comparison at the deployed HMM settings.

Uses ``artifacts/hmm.json`` exactly as served by the API: no beta grid search,
no artifact writes. Rewrites ``results/hmm_comparison.csv``, its figures, and
the failure analysis so the report describes the shipped configuration.

Run from the repository root: python scripts/14_decoder_comparison.py
"""

from __future__ import annotations

import time
from dataclasses import fields
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from pathsnap import config
from pathsnap.artifacts import load_model_artifact, read_json
from pathsnap.evaluate import evaluate_decoders, prepare_trip_inference
from pathsnap.hmm import HMMConfig
from pathsnap.io import iter_trips
from pathsnap.viz import generate_decoder_plots


def deployed_hmm_config(path: Path) -> HMMConfig:
    payload = read_json(path)
    allowed = {field.name for field in fields(HMMConfig)}
    return HMMConfig(**{k: v for k, v in payload.items() if k in allowed})


def main() -> None:
    start = time.perf_counter()
    paths = config.get_paths()
    paths.validate_data()
    config.ensure_output_directories(paths)

    artifact = load_model_artifact(paths.artifact_dir / "model.joblib")
    settings = deployed_hmm_config(paths.artifact_dir / "hmm.json")
    print(
        f"Deployed config: beta_m={settings.beta_m:g} "
        f"refine_points={settings.refine_points} emission={settings.emission}"
    )

    rows = []
    for trip in iter_trips(paths.test_data_dir, "test"):
        inference = prepare_trip_inference(trip, artifact)
        rows.append(evaluate_decoders(inference, settings))
        print(f"  {trip.trip_id} decoded", flush=True)

    comparison = pd.concat(rows, ignore_index=True)
    comparison_path = paths.results_dir / "hmm_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    stored = pd.read_csv(comparison_path, dtype={"trip_id": str})
    generate_decoder_plots(stored, paths.results_dir / "figures")

    from pathsnap.evaluate import write_decoder_failure_analysis

    write_decoder_failure_analysis(
        stored, paths.results_dir / "hmm_failure_analysis.md"
    )

    summary = stored.groupby("method")[
        ["point_accuracy", "route_continuity", "route_precision", "route_recall"]
    ].mean()
    print(summary.to_string())
    print(f"Wrote {comparison_path}")
    print(f"Runtime: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
