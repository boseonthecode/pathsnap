"""Matplotlib comparison plots driven entirely by stored result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIGURE_NAMES: tuple[str, ...] = (
    "cv_macro_f1_comparison.png",
    "student_test_top1_macro_f1.png",
    "log_loss_comparison.png",
    "confusion_matrices.png",
    "timing_comparison.png",
)


def _cv_aggregate(comparison: pd.DataFrame) -> pd.DataFrame:
    frame = comparison[comparison["scope"] == "cv_aggregate"].copy()
    return frame.sort_values("macro_f1", ascending=False)


def _student_test(comparison: pd.DataFrame) -> pd.DataFrame:
    frame = comparison[comparison["scope"] == "student_test"].copy()
    return frame.sort_values("macro_f1", ascending=False)


def plot_cv_macro_f1(comparison: pd.DataFrame, out_path: Path) -> Path:
    frame = _cv_aggregate(comparison)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(frame["model"], frame["macro_f1"], color="#1f77b4")
    axis.set_title("Trip-grouped CV Macro F1 by model")
    axis.set_xlabel("Model")
    axis.set_ylabel("Macro F1")
    axis.set_ylim(0, 1.0)
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_student_test_top1_macro_f1(comparison: pd.DataFrame, out_path: Path) -> Path:
    frame = _student_test(comparison)
    indices = np.arange(frame.shape[0])
    width = 0.38
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(indices - width / 2, frame["top1_accuracy"], width, label="Top-1 accuracy")
    axis.bar(indices + width / 2, frame["macro_f1"], width, label="Macro F1")
    axis.set_xticks(indices)
    axis.set_xticklabels(frame["model"], rotation=20, ha="right")
    axis.set_title("Student-test Top-1 accuracy and Macro F1")
    axis.set_xlabel("Model")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.0)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_log_loss(comparison: pd.DataFrame, out_path: Path) -> Path:
    frame = _student_test(comparison)
    if frame.empty:
        frame = _cv_aggregate(comparison)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(frame["model"], frame["log_loss"], color="#d62728")
    axis.set_title("Log loss by model")
    axis.set_xlabel("Model")
    axis.set_ylabel("Log loss")
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_confusion_matrices(comparison: pd.DataFrame, out_path: Path) -> Path:
    frame = _student_test(comparison)
    if frame.empty:
        frame = _cv_aggregate(comparison)
    n_models = frame.shape[0]
    columns = 3
    rows = int(np.ceil(n_models / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.6 * rows))
    axes_array = np.atleast_1d(axes).ravel()

    for axis, (_, row) in zip(axes_array, frame.iterrows()):
        matrix = np.array(
            [[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=np.float64
        )
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(str(row["model"]))
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])
        axis.set_xticklabels(["negative", "positive"])
        axis.set_yticklabels(["negative", "positive"])
        for i in range(2):
            for j in range(2):
                axis.text(
                    j,
                    i,
                    f"{int(matrix[i, j])}",
                    ha="center",
                    va="center",
                    color="black" if matrix[i, j] < matrix.max() * 0.6 else "white",
                )
    for axis in axes_array[n_models:]:
        axis.axis("off")
    fig.suptitle("Candidate confusion matrices (student-test)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_timing(comparison: pd.DataFrame, out_path: Path) -> Path:
    cv_frame = _cv_aggregate(comparison)
    test_frame = _student_test(comparison)
    inference_frame = test_frame if not test_frame.empty else cv_frame

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    left.bar(cv_frame["model"], cv_frame["train_time_s"], color="#2ca02c")
    left.set_title("Mean CV training time")
    left.set_xlabel("Model")
    left.set_ylabel("Seconds")
    left.tick_params(axis="x", rotation=20)
    left.grid(axis="y", alpha=0.3)

    right.bar(inference_frame["model"], inference_frame["inference_ms"], color="#9467bd")
    right.set_title("Inference time")
    right.set_xlabel("Model")
    right.set_ylabel("Milliseconds")
    right.tick_params(axis="x", rotation=20)
    right.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_all_plots(
    comparison: pd.DataFrame, figures_dir: Path
) -> dict[str, Path]:
    """Write every required comparison figure from the results table."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "cv_macro_f1_comparison.png": plot_cv_macro_f1(
            comparison, figures_dir / "cv_macro_f1_comparison.png"
        ),
        "student_test_top1_macro_f1.png": plot_student_test_top1_macro_f1(
            comparison, figures_dir / "student_test_top1_macro_f1.png"
        ),
        "log_loss_comparison.png": plot_log_loss(
            comparison, figures_dir / "log_loss_comparison.png"
        ),
        "confusion_matrices.png": plot_confusion_matrices(
            comparison, figures_dir / "confusion_matrices.png"
        ),
        "timing_comparison.png": plot_timing(
            comparison, figures_dir / "timing_comparison.png"
        ),
    }
    return paths
