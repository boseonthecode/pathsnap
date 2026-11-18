"""Build and cache candidate/feature artifacts for every provided trip."""

from __future__ import annotations

import argparse
import time
from itertools import chain
from pathlib import Path

import _bootstrap  # noqa: F401

from pathsnap import config
from pathsnap.cache import compute_cache_version, load_or_build_trip_artifacts
from pathsnap.io import iter_trips


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PathSnap feature caches")
    parser.add_argument("--max-trips", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = config.get_paths()
    paths.validate_data()
    config.ensure_output_directories(paths)
    version = compute_cache_version()
    print(f"Cache version: {version}")

    start = time.perf_counter()
    total_rows = 0
    trips = 0
    for trip in chain(
        iter_trips(paths.train_data_dir, "train"),
        iter_trips(paths.test_data_dir, "test"),
    ):
        if args.max_trips is not None and trips >= args.max_trips:
            break
        artifacts = load_or_build_trip_artifacts(trip)
        total_rows += artifacts.features.n_rows
        trips += 1

    print(
        f"Cached {trips} trips, {total_rows} candidate rows in "
        f"{time.perf_counter() - start:.1f}s under {paths.cache_dir / version}"
    )


if __name__ == "__main__":
    main()
