"""Tests for cross-validation, comparison, selection, and sampling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pathsnap.cache import build_trip_artifacts
from pathsnap.dataset import assemble_dataset, build_trip_dataset
from pathsnap.evaluate import (
    COMPARISON_COLUMNS,
    cross_validate,
    evaluate_on_test,
    sample_trip_rows,
    select_best_model,
    stack_sampled,
    write_selection_note,
)
from pathsnap.io import load_trip
from conftest import write_trip


def _bundle(tmp_path: Path):
    trips = []
    artifacts = []
    for index in range(4):
        trip = load_trip(write_trip(tmp_path / "train", f"0000007{index}"), "train")
        trips.append(trip)
        artifacts.append(build_trip_artifacts(trip))
    test_trip = load_trip(write_trip(tmp_path / "test", "00000005"), "test")
    trips.append(test_trip)
    artifacts.append(build_trip_artifacts(test_trip))
    return assemble_dataset(trips, artifacts)


MODELS = ["logistic_regression", "extra_trees"]


def test_cross_validate_has_fold_and_aggregate_rows(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    comparison = cross_validate(bundle, model_names=MODELS, n_splits=2, max_observations_per_trip=None)

    assert list(comparison.columns) == list(COMPARISON_COLUMNS)
    assert set(comparison["scope"]) == {"fold", "cv_aggregate"}
    for model in MODELS:
        model_rows = comparison[comparison["model"] == model]
        assert (model_rows["scope"] == "fold").sum() == 2
        assert (model_rows["scope"] == "cv_aggregate").sum() == 1
    assert comparison["n_rows"].min() > 0


def test_cross_validate_is_reproducible_with_same_seed(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    first = cross_validate(bundle, model_names=MODELS, n_splits=2, seed=11)
    second = cross_validate(bundle, model_names=MODELS, n_splits=2, seed=11)

    metric_columns = [
        column
        for column in COMPARISON_COLUMNS
        if column not in {"train_time_s", "inference_ms"}
    ]
    pd.testing.assert_frame_equal(first[metric_columns], second[metric_columns])


def test_selection_and_note_name_the_best_cv_model(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    comparison = cross_validate(bundle, model_names=MODELS, n_splits=2)

    chosen = select_best_model(comparison)
    note_path = tmp_path / "selection.md"
    write_selection_note(comparison, chosen, note_path)

    assert chosen in MODELS
    text = note_path.read_text(encoding="utf-8")
    assert chosen in text
    assert "CV Macro F1" in text


def test_student_test_evaluation_returns_one_row_per_model(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    student_test = evaluate_on_test(bundle, model_names=MODELS)

    assert len(student_test) == len(MODELS)
    assert set(student_test["scope"]) == {"student_test"}
    assert student_test["n_rows"].min() > 0


def test_sample_trip_rows_caps_observations(tmp_path: Path) -> None:
    trip = load_trip(write_trip(tmp_path / "train", "00000078"), "train")
    dataset = build_trip_dataset(trip, build_trip_artifacts(trip))
    total_observations = len(set(dataset.observation_indices.tolist()))

    features, labels, observations = sample_trip_rows(dataset, max_observations=2)

    assert 0 < features.shape[0] <= dataset.n_rows
    assert labels.shape[0] == features.shape[0]
    assert len(set(observations.tolist())) <= min(2, total_observations)


def test_stack_sampled_creates_unique_observation_groups(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    train_ids = [trip.trip_id for trip in bundle.training_trips()]

    _, _, groups = stack_sampled(bundle, train_ids, max_observations_per_trip=3)

    assert groups.size > 0
    assert len(set(groups.tolist())) <= 3 * len(train_ids)
