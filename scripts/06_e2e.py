"""Step 29 — end-to-end API verification across all evaluation trips."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from api.e2e import run_end_to_end
from api.service import PredictionService
from pathsnap import config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PathSnap end-to-end checks")
    parser.add_argument("--eval-data-dir", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--method", type=str, default="hmm")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = config.get_paths()
    config.ensure_output_directories(paths)
    eval_dir = args.eval_data_dir or paths.eval_data_dir
    artifact_dir = args.artifact_dir or paths.artifact_dir
    results_dir = args.results_dir or paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    service = PredictionService(eval_dir, artifact_dir)
    summary = run_end_to_end(service, method=args.method)
    out_path = results_dir / "e2e_summary.csv"
    summary.to_csv(out_path, index=False)

    failed = summary[summary["status"] != 200]
    print(f"Trips checked: {len(summary)}; failures: {len(failed)}")
    if not failed.empty:
        print(failed[["trip_id", "status"]].to_string(index=False))
    print(summary.drop(columns=["status"]).describe().to_string())
    print(f"Wrote {out_path}")
    print(f"Total runtime: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
