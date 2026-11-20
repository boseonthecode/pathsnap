"""Step 04 — dataset inventory and data-quality EDA.

Thin orchestrator: loads every provided trip, writes the summary CSV, the
markdown data-quality report, and one Folium overview map.
"""

from __future__ import annotations

import argparse
from itertools import chain
from pathlib import Path

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)

from pathsnap import config
from pathsnap.eda import build_eda_map, build_eda_table, write_eda_report
from pathsnap.io import discover_trip_dirs, iter_trips, load_trip


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PathSnap dataset EDA")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--map-trip", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = config.get_paths()
    paths.validate_data()
    config.ensure_output_directories(paths)
    results_dir = args.results_dir or paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    trips = chain(
        iter_trips(paths.train_data_dir, "train"),
        iter_trips(paths.test_data_dir, "test"),
    )
    frame = build_eda_table(trips)

    csv_path = results_dir / "eda_summary.csv"
    frame.to_csv(csv_path, index=False)
    report_path = results_dir / "eda_report.md"
    write_eda_report(frame, report_path)

    train_trip_ids = [path.name for path in discover_trip_dirs(paths.train_data_dir)]
    if not train_trip_ids:
        raise SystemExit(f"No training trips found in {paths.train_data_dir}")
    map_trip_id = args.map_trip or train_trip_ids[0]
    if args.map_trip and args.map_trip not in train_trip_ids:
        raise SystemExit(
            f"Unknown --map-trip {args.map_trip!r}; available: "
            f"{', '.join(train_trip_ids[:5])}..."
        )
    map_trip = load_trip(paths.train_data_dir / map_trip_id, partition="train")
    map_path = build_eda_map(map_trip, results_dir / "folium" / f"eda_{map_trip_id}.html")

    print(f"Wrote {len(frame)} trip rows to {csv_path}")
    print(f"Wrote data-quality report to {report_path}")
    print(f"Wrote EDA map to {map_path}")


if __name__ == "__main__":
    main()
