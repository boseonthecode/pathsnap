"""Tests for comparison plot generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pathsnap.evaluate import COMPARISON_COLUMNS
from pathsnap.viz import FIGURE_NAMES, generate_all_plots, plot_feature_ablation

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _row(model: str, family: str, scope: str) -> dict[str, object]:
    return {
        "model": model,
        "family": family,
        "config": "{}",
        "scope": scope,
        "fold": -1 if scope != "fold" else 0,
        "n_rows": 100,
        "n_observations": 20,
        "top1_accuracy": 0.8,
        "macro_f1": 0.75,
        "precision": 0.7,
        "recall": 0.8,
        "log_loss": 0.3,
        "train_time_s": 1.2,
        "inference_ms": 3.4,
        "tn": 60,
        "fp": 10,
        "fn": 5,
        "tp": 25,
    }


def test_all_required_plots_are_written_and_non_empty(tmp_path: Path) -> None:
    rows = [
        _row("logistic_regression", "linear", "cv_aggregate"),
        _row("extra_trees", "ensemble", "cv_aggregate"),
        _row("logistic_regression", "linear", "student_test"),
        _row("extra_trees", "ensemble", "student_test"),
    ]
    comparison = pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS))

    paths = generate_all_plots(comparison, tmp_path / "figures")

    assert set(paths) == set(FIGURE_NAMES)
    for path in paths.values():
        assert path.is_file()
        assert path.stat().st_size > 1000
        assert path.read_bytes()[:8] == PNG_SIGNATURE


def test_feature_ablation_plot_is_written(tmp_path: Path) -> None:
    from pathsnap.ablation import ABLATION_COLUMNS

    rows = [
        {
            "config": "full",
            "features": "distance_m",
            "n_features": 1,
            "cv_macro_f1": 0.8,
            "cv_top1_accuracy": 0.8,
            "cv_log_loss": 0.3,
            "cv_precision": 0.8,
            "cv_recall": 0.8,
            "delta_macro_f1_vs_full": 0.0,
            "train_time_s": 1.0,
            "inference_ms": 2.0,
        },
        {
            "config": "drop_distance_m",
            "features": "log_distance",
            "n_features": 1,
            "cv_macro_f1": 0.7,
            "cv_top1_accuracy": 0.7,
            "cv_log_loss": 0.4,
            "cv_precision": 0.7,
            "cv_recall": 0.7,
            "delta_macro_f1_vs_full": 0.1,
            "train_time_s": 1.0,
            "inference_ms": 2.0,
        },
        {
            "config": "cumulative_through_distance",
            "features": "distance_m,log_distance",
            "n_features": 2,
            "cv_macro_f1": 0.8,
            "cv_top1_accuracy": 0.8,
            "cv_log_loss": 0.3,
            "cv_precision": 0.8,
            "cv_recall": 0.8,
            "delta_macro_f1_vs_full": 0.0,
            "train_time_s": 1.0,
            "inference_ms": 2.0,
        },
    ]
    ablation = pd.DataFrame(rows, columns=list(ABLATION_COLUMNS))

    path = plot_feature_ablation(ablation, tmp_path / "feature_ablation.png")

    assert path.is_file()
    assert path.stat().st_size > 1000
    assert path.read_bytes()[:8] == PNG_SIGNATURE
