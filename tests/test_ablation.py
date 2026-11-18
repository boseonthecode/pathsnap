"""Tests for feature ablation and selection."""

from __future__ import annotations

from pathlib import Path

from pathsnap.ablation import (
    ABLATION_COLUMNS,
    FEATURE_GROUPS,
    run_feature_ablation,
    select_features,
    write_feature_selection_note,
)
from pathsnap.cache import build_trip_artifacts
from pathsnap.dataset import assemble_dataset
from pathsnap.features import FEATURE_NAMES
from pathsnap.io import load_trip
from conftest import write_trip

PARAMS = {
    "n_estimators": 100,
    "max_depth": 8,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "criterion": "gini",
    "class_weight": "balanced",
}


def _bundle(tmp_path: Path):
    trips = []
    artifacts = []
    for index in range(4):
        trip = load_trip(write_trip(tmp_path / "train", f"0000004{index}"), "train")
        trips.append(trip)
        artifacts.append(build_trip_artifacts(trip))
    test_trip = load_trip(write_trip(tmp_path / "test", "00000003"), "test")
    trips.append(test_trip)
    artifacts.append(build_trip_artifacts(test_trip))
    return assemble_dataset(trips, artifacts)


def test_feature_groups_partition_the_feature_set() -> None:
    grouped = [name for _, names in FEATURE_GROUPS for name in names]
    assert sorted(grouped) == sorted(FEATURE_NAMES)


def test_ablation_contains_full_leave_one_out_and_cumulative_rows(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)

    ablation = run_feature_ablation(
        bundle, params=PARAMS, n_splits=2, max_observations_per_trip=None
    )

    assert list(ablation.columns) == list(ABLATION_COLUMNS)
    assert (ablation["config"] == "full").sum() == 1
    assert ablation["config"].str.startswith("drop_").sum() == len(FEATURE_NAMES)
    assert ablation["config"].str.startswith("cumulative_").sum() == len(
        FEATURE_GROUPS
    )
    assert ablation["cv_macro_f1"].between(0, 1).all()
    full_row = ablation[ablation["config"] == "full"].iloc[0]
    assert full_row["delta_macro_f1_vs_full"] == 0.0


def test_selection_returns_non_empty_subset_and_writes_note(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    ablation = run_feature_ablation(
        bundle, params=PARAMS, n_splits=2, max_observations_per_trip=None
    )

    selected, selected_row = select_features(
        ablation,
        bundle,
        params=PARAMS,
        tolerance=0.002,
        n_splits=2,
        max_observations_per_trip=None,
    )
    note_path = tmp_path / "feature_selection.md"
    write_feature_selection_note(ablation, selected, selected_row, note_path)

    assert selected
    assert "distance_m" in selected
    assert set(selected).issubset(set(FEATURE_NAMES))
    assert selected_row["n_features"] == len(selected)
    assert 0.0 <= selected_row["cv_macro_f1"] <= 1.0
    text = note_path.read_text(encoding="utf-8")
    assert "Feature selection and ablation" in text
    for feature in selected:
        assert feature in text
