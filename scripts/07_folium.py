"""Generate Folium maps and static previews for representative test trips."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from pathsnap import config
from pathsnap.artifacts import load_hmm_config, load_model_artifact
from pathsnap.folium_maps import (
    build_prediction_map,
    build_static_preview,
    predict_for_map,
)
from pathsnap.io import load_trip


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Folium prediction maps")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--method", type=str, default="hmm")
    return parser.parse_args()


def _select_trips(comparison_path: Path, fallback: list[str]) -> list[str]:
    if not comparison_path.is_file():
        return fallback[:1]
    frame = pd.read_csv(comparison_path, dtype={"trip_id": str})
    pivot = frame.pivot_table(
        index="trip_id", columns="method", values="point_accuracy"
    )
    if "hmm" not in pivot or "classifier" not in pivot:
        return fallback[:1]
    delta = pivot["hmm"] - pivot["classifier"]
    selected = [str(delta.idxmax()), str(delta.idxmin()), str(pivot.index[0])]
    return list(dict.fromkeys(selected))


def main() -> None:
    args = _parse_args()
    paths = config.get_paths()
    paths.validate_data()
    config.ensure_output_directories(paths)
    results_dir = args.results_dir or paths.results_dir

    artifact = load_model_artifact(paths.artifact_dir / "model.joblib")
    hmm_config = load_hmm_config(paths.artifact_dir / "hmm.json")
    trip_ids = sorted(path.name for path in paths.test_data_dir.iterdir() if path.is_dir())
    chosen = _select_trips(results_dir / "hmm_comparison.csv", trip_ids)
    print(f"Generating maps for: {chosen}")

    for trip_id in chosen:
        trip = load_trip(paths.test_data_dir / trip_id, "test")
        result = predict_for_map(trip, artifact, hmm_config, method=args.method)
        map_path = build_prediction_map(
            trip, result, results_dir / "folium" / f"{trip_id}_{args.method}.html"
        )
        preview_path = build_static_preview(
            trip, result, results_dir / "figures" / f"folium_{trip_id}_preview.png"
        )
        print(f"{trip_id}: {map_path}")
        print(f"{trip_id}: {preview_path}")


if __name__ == "__main__":
    main()
