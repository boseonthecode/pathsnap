"""Static EDA plots built from ``results/eda_summary.csv``.

All figures are derived from the stored trip summary table, not recomputed
numbers, so they can be regenerated deterministically from the EDA outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TRAIN_COLOR = "#1f77b4"
TEST_COLOR = "#ff7f0e"
SEVERE_MISMATCH_COVERAGE = 0.50
LOW_COVERAGE_THRESHOLD = 0.90


def _partition_colors(frame: pd.DataFrame) -> list[str]:
    return [
        TRAIN_COLOR if partition == "train" else TEST_COLOR
        for partition in frame["partition"]
    ]


def plot_inventory_and_network(frame: pd.DataFrame, out_path: Path) -> Path:
    train = frame[frame["partition"] == "train"]
    test = frame[frame["partition"] == "test"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(
        0, float(frame["n_points"].max()) * 1.02, num=30
    )
    left.hist(train["n_points"], bins=bins, color=TRAIN_COLOR, alpha=0.7, label="train")
    left.hist(test["n_points"], bins=bins, color=TEST_COLOR, alpha=0.8, label="test")
    left.set_title("GPS observations per trip")
    left.set_xlabel("Observations")
    left.set_ylabel("Trips")
    left.legend()
    left.grid(axis="y", alpha=0.3)

    for partition, color, label in (
        ("train", TRAIN_COLOR, "train"),
        ("test", TEST_COLOR, "test"),
    ):
        subset = frame[frame["partition"] == partition]
        right.scatter(
            subset["n_nodes"],
            subset["n_arcs"],
            s=22,
            alpha=0.7,
            color=color,
            label=label,
        )
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_title("Road-network size per trip")
    right.set_xlabel("Nodes")
    right.set_ylabel("Arcs")
    right.legend()
    right.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_sampling_and_gaps(frame: pd.DataFrame, out_path: Path) -> Path:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    left.hist(frame["median_dt_s"], bins=25, color="#2ca02c")
    left.set_title("Median GPS sampling interval")
    left.set_xlabel("Seconds between samples")
    left.set_ylabel("Trips")
    left.grid(axis="y", alpha=0.3)

    positive_gaps = frame.loc[frame["max_gap_s"] > 0, "max_gap_s"]
    right.hist(np.log10(positive_gaps), bins=25, color="#9467bd")
    right.set_title("Largest GPS gap per trip")
    right.set_xlabel("log10(max gap seconds)")
    right.set_ylabel("Trips")
    right.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_route_coverage(frame: pd.DataFrame, out_path: Path) -> Path:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    left.hist(
        frame["route_gps_within50_frac"],
        bins=25,
        color="#1f77b4",
        edgecolor="white",
    )
    left.axvline(
        SEVERE_MISMATCH_COVERAGE,
        color="#d62728",
        linestyle="--",
        label="severe mismatch (<50%)",
    )
    left.axvline(
        LOW_COVERAGE_THRESHOLD,
        color="#7f7f7f",
        linestyle=":",
        label="low-coverage flag (<90%)",
    )
    left.set_title("GPS-to-reference-route coverage")
    left.set_xlabel("Fraction of points within 50 m")
    left.set_ylabel("Trips")
    left.legend(fontsize=8)
    left.grid(axis="y", alpha=0.3)

    right.scatter(
        frame["median_route_dist_m"],
        frame["route_gps_within50_frac"],
        c=_partition_colors(frame),
        s=24,
        alpha=0.75,
    )
    for _, row in frame[frame["severe_mismatch"]].iterrows():
        right.annotate(
            str(row["trip_id"]),
            (row["median_route_dist_m"], row["route_gps_within50_frac"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color="#d62728",
        )
    right.axhline(SEVERE_MISMATCH_COVERAGE, color="#d62728", linestyle="--")
    right.set_title("Coverage vs median route distance")
    right.set_xlabel("Median distance to route (m)")
    right.set_ylabel("Coverage within 50 m")
    right.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_route_anomalies(frame: pd.DataFrame, out_path: Path) -> Path:
    affected = frame[
        (frame["route_discontinuities"] > 0) | (frame["repeated_route_arcs"] > 0)
    ].sort_values("trip_id")
    positions = np.arange(affected.shape[0])
    width = 0.4

    fig, axis = plt.subplots(figsize=(13, 4.8))
    axis.bar(
        positions - width / 2,
        affected["route_discontinuities"],
        width,
        label="node discontinuities",
        color="#d62728",
    )
    axis.bar(
        positions + width / 2,
        affected["repeated_route_arcs"],
        width,
        label="repeated route arcs",
        color="#9467bd",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(affected["trip_id"], rotation=60, ha="right", fontsize=8)
    axis.set_title("Reference-route anomalies by trip")
    axis.set_xlabel("Trip ID")
    axis.set_ylabel("Count")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_route_track_ratio(frame: pd.DataFrame, out_path: Path) -> Path:
    ratios = frame["route_track_ratio"].replace([np.inf, -np.inf], np.nan).dropna()

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(ratios, bins=30, color="#17becf", edgecolor="white")
    axis.axvline(1.0, color="#d62728", linestyle="--", label="route = track length")
    axis.set_title("Reference-route length / GPS track length")
    axis.set_xlabel("Route length ÷ track length")
    axis.set_ylabel("Trips")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_trip_extents(frame: pd.DataFrame, out_path: Path) -> Path:
    fig, axis = plt.subplots(figsize=(8.5, 6))
    for partition, color, label in (
        ("train", TRAIN_COLOR, "train"),
        ("test", TEST_COLOR, "test"),
    ):
        subset = frame[frame["partition"] == partition]
        axis.scatter(
            (subset["bbox_min_lon"] + subset["bbox_max_lon"]) / 2.0,
            (subset["bbox_min_lat"] + subset["bbox_max_lat"]) / 2.0,
            s=26,
            alpha=0.7,
            color=color,
            label=label,
        )
    axis.set_title("Trip geographic centres")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_eda_plots(frame: pd.DataFrame, figures_dir: Path) -> dict[str, Path]:
    """Write the full static EDA figure set."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {
        "eda_inventory_network.png": plot_inventory_and_network(
            frame, figures_dir / "eda_inventory_network.png"
        ),
        "eda_sampling_gaps.png": plot_sampling_and_gaps(
            frame, figures_dir / "eda_sampling_gaps.png"
        ),
        "eda_route_coverage.png": plot_route_coverage(
            frame, figures_dir / "eda_route_coverage.png"
        ),
        "eda_route_anomalies.png": plot_route_anomalies(
            frame, figures_dir / "eda_route_anomalies.png"
        ),
        "eda_route_track_ratio.png": plot_route_track_ratio(
            frame, figures_dir / "eda_route_track_ratio.png"
        ),
        "eda_trip_extents.png": plot_trip_extents(
            frame, figures_dir / "eda_trip_extents.png"
        ),
    }
