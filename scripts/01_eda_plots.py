"""Generate static EDA plots from the stored trip summary table."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from pathsnap import config
from pathsnap.eda_viz import generate_eda_plots


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PathSnap EDA summaries")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = config.get_paths()
    summary_path = args.summary or (paths.results_dir / "eda_summary.csv")
    figures_dir = args.figures_dir or (paths.results_dir / "figures")
    frame = pd.read_csv(summary_path, dtype={"trip_id": str})

    outputs = generate_eda_plots(frame, figures_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
