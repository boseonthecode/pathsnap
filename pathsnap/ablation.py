"""Feature ablation and selection on trip-grouped training folds.

The full candidate feature set is compared against (a) leave-one-feature-out
models and (b) cumulative feature groups. Selection is made using training
folds only; student-test data is never used to choose features.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.dataset import DatasetBundle, trip_grouped_folds
from pathsnap.evaluate import stack_sampled
from pathsnap.features import FEATURE_NAMES
from pathsnap.models import evaluate_model, fit_model

ABLATION_COLUMNS: tuple[str, ...] = (
    "config",
    "features",
    "n_features",
    "cv_macro_f1",
    "cv_top1_accuracy",
    "cv_log_loss",
    "cv_precision",
    "cv_recall",
    "delta_macro_f1_vs_full",
    "train_time_s",
    "inference_ms",
)

# Ordered cumulative groups; the union must equal FEATURE_NAMES.
FEATURE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("distance", ("distance_m", "log_distance")),
    ("heading", ("heading_diff_deg",)),
    ("speed", ("speed_consistency",)),
    ("candidate_context", ("candidate_rank", "along_fraction", "arc_length_m")),
    ("temporal", ("time_gap_s",)),
    ("topology", ("from_degree", "to_degree")),
)


def _validate_groups() -> None:
    grouped = [name for _, names in FEATURE_GROUPS for name in names]
    if sorted(grouped) != sorted(FEATURE_NAMES):
        raise RuntimeError("FEATURE_GROUPS must partition FEATURE_NAMES")


_validate_groups()


FoldData = list[
    tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ]
]


def _fold_data(
    bundle: DatasetBundle,
    n_splits: int,
    seed: int,
    max_observations_per_trip: int | None,
) -> FoldData:
    training_trip_ids = [trip.trip_id for trip in bundle.training_trips()]
    folds = trip_grouped_folds(training_trip_ids, n_splits=n_splits, seed=seed)
    return [
        (
            stack_sampled(bundle, train_ids, max_observations_per_trip),
            stack_sampled(bundle, validation_ids, max_observations_per_trip),
        )
        for train_ids, validation_ids in folds
    ]


def evaluate_feature_subset(
    fold_data: FoldData,
    feature_names: Sequence[str],
    model_name: str = "extra_trees",
    params: dict[str, Any] | None = None,
    seed: int = config.DEFAULT_SEED,
    label: str = "subset",
) -> dict[str, Any]:
    """Cross-validate one feature subset across the precomputed folds."""
    all_features = list(FEATURE_NAMES)
    indices = [all_features.index(name) for name in feature_names]

    results = []
    for (train_features, train_labels, _), (
        validation_features,
        validation_labels,
        validation_groups,
    ) in fold_data:
        trained = fit_model(
            model_name,
            train_features[:, indices],
            train_labels,
            seed=seed,
            params=params,
            feature_names=feature_names,
        )
        results.append(
            evaluate_model(
                trained,
                validation_features[:, indices],
                validation_labels,
                validation_groups,
            )
        )

    def mean(attribute: str) -> float:
        return float(np.mean([getattr(result, attribute) for result in results]))

    return {
        "config": label,
        "features": ",".join(feature_names),
        "n_features": len(feature_names),
        "cv_macro_f1": mean("macro_f1"),
        "cv_top1_accuracy": mean("top1_accuracy"),
        "cv_log_loss": mean("log_loss"),
        "cv_precision": mean("precision"),
        "cv_recall": mean("recall"),
        "delta_macro_f1_vs_full": np.nan,
        "train_time_s": mean("train_time_s"),
        "inference_ms": mean("inference_ms"),
    }


def run_feature_ablation(
    bundle: DatasetBundle,
    model_name: str = "extra_trees",
    params: dict[str, Any] | None = None,
    n_splits: int = config.N_SPLITS,
    seed: int = config.DEFAULT_SEED,
    max_observations_per_trip: int | None = config.CV_MAX_OBSERVATIONS_PER_TRIP,
) -> pd.DataFrame:
    """Full, leave-one-out, and cumulative-group ablation table."""
    fold_data = _fold_data(bundle, n_splits, seed, max_observations_per_trip)
    rows = [
        evaluate_feature_subset(
            fold_data, list(FEATURE_NAMES), model_name, params, seed, label="full"
        )
    ]
    full_score = rows[0]["cv_macro_f1"]

    for feature in FEATURE_NAMES:
        subset = [name for name in FEATURE_NAMES if name != feature]
        rows.append(
            evaluate_feature_subset(
                fold_data,
                subset,
                model_name,
                params,
                seed,
                label=f"drop_{feature}",
            )
        )

    cumulative: list[str] = []
    for group_name, group_features in FEATURE_GROUPS:
        cumulative.extend(group_features)
        rows.append(
            evaluate_feature_subset(
                fold_data,
                list(cumulative),
                model_name,
                params,
                seed,
                label=f"cumulative_through_{group_name}",
            )
        )

    frame = pd.DataFrame(rows, columns=list(ABLATION_COLUMNS))
    frame["delta_macro_f1_vs_full"] = full_score - frame["cv_macro_f1"]
    return frame


def select_features(
    ablation: pd.DataFrame,
    bundle: DatasetBundle,
    model_name: str = "extra_trees",
    params: dict[str, Any] | None = None,
    tolerance: float = 0.002,
    n_splits: int = config.N_SPLITS,
    seed: int = config.DEFAULT_SEED,
    max_observations_per_trip: int | None = config.CV_MAX_OBSERVATIONS_PER_TRIP,
) -> tuple[list[str], dict[str, Any]]:
    """Drop features whose removal costs at most ``tolerance`` CV Macro F1.

    The candidate reduced set is re-evaluated; if it underperforms the full set
    by more than ``tolerance``, the full set is retained.
    """
    full_row = ablation[ablation["config"] == "full"].iloc[0]
    drop_rows = ablation[ablation["config"].str.startswith("drop_")]

    removable = []
    for _, row in drop_rows.iterrows():
        feature = str(row["config"]).removeprefix("drop_")
        if float(row["delta_macro_f1_vs_full"]) <= tolerance and feature != "distance_m":
            removable.append(feature)

    candidate = [name for name in FEATURE_NAMES if name not in removable]
    if not candidate:
        candidate = ["distance_m", "log_distance"]

    fold_data = _fold_data(bundle, n_splits, seed, max_observations_per_trip)
    candidate_row = evaluate_feature_subset(
        fold_data, candidate, model_name, params, seed, label="selected_candidate"
    )

    selected_full = (
        float(candidate_row["cv_macro_f1"]) >= float(full_row["cv_macro_f1"]) - tolerance
    )
    if not selected_full:
        candidate = list(FEATURE_NAMES)
        candidate_row = evaluate_feature_subset(
            fold_data, candidate, model_name, params, seed, label="selected_full"
        )

    return candidate, candidate_row


def write_feature_selection_note(
    ablation: pd.DataFrame,
    selected_features: Sequence[str],
    selected_row: dict[str, Any],
    out_path,
) -> None:
    full_row = ablation[ablation["config"] == "full"].iloc[0]
    drop_rows = ablation[ablation["config"].str.startswith("drop_")].copy()
    drop_rows["removed_feature"] = drop_rows["config"].str.removeprefix("drop_")
    drop_rows = drop_rows.sort_values("delta_macro_f1_vs_full", ascending=False)

    lines = [
        "# Feature selection and ablation",
        "",
        "Selection used trip-grouped CV Macro F1 on the training trips only. "
        "Student-test data was not used to choose features.",
        "",
        f"- Full feature set: {len(FEATURE_NAMES)} features, "
        f"CV Macro F1 = {full_row['cv_macro_f1']:.4f}",
        f"- Selected feature set: {len(selected_features)} features, "
        f"CV Macro F1 = {selected_row['cv_macro_f1']:.4f}",
        f"- Selected features: `{', '.join(selected_features)}`",
        "",
        "## Leave-one-feature-out importance",
        "",
        "Positive delta means removing the feature hurt CV Macro F1.",
        "",
        "| Removed feature | CV Macro F1 without it | Delta vs full |",
        "| :--- | ---: | ---: |",
    ]
    for _, row in drop_rows.iterrows():
        lines.append(
            f"| {row['removed_feature']} | {row['cv_macro_f1']:.4f} | "
            f"{row['delta_macro_f1_vs_full']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Cumulative feature groups",
            "",
            "| Feature group reached | n features | CV Macro F1 |",
            "| :--- | ---: | ---: |",
        ]
    )
    cumulative = ablation[ablation["config"].str.startswith("cumulative_")]
    for _, row in cumulative.iterrows():
        group = str(row["config"]).removeprefix("cumulative_through_")
        lines.append(f"| {group} | {int(row['n_features'])} | {row['cv_macro_f1']:.4f} |")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
