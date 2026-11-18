"""Tests for the first three model families."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from pathsnap.cache import build_trip_artifacts
from pathsnap.dataset import build_trip_dataset
from pathsnap.io import load_trip
from pathsnap.models import (
    MODEL_SPECS,
    evaluate_model,
    fit_model,
    model_configs,
    observation_top1_accuracy,
)
from conftest import write_trip

ALL_MODELS = [
    "logistic_regression",
    "knn",
    "extra_trees",
    "lightgbm",
    "mlp",
]


def _training_rows(tmp_path: Path):
    trip = load_trip(write_trip(tmp_path / "train", "00000079"))
    dataset = build_trip_dataset(trip, build_trip_artifacts(trip))
    return dataset.features, dataset.labels, dataset.observation_indices


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_model_trains_and_produces_valid_probabilities(
    model_name: str, tmp_path: Path
) -> None:
    features, labels, observation_indices = _training_rows(tmp_path)
    assert set(np.unique(labels)) == {0, 1}

    trained = fit_model(model_name, features, labels)
    probabilities = trained.positive_probability(features)
    result = evaluate_model(trained, features, labels, observation_indices)

    assert probabilities.shape == (features.shape[0],)
    assert np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0)
    assert result.model == model_name
    assert 0.0 <= result.macro_f1 <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert result.log_loss >= 0.0
    assert result.train_time_s >= 0.0
    assert result.inference_ms >= 0.0
    assert result.top1_accuracy is not None


def test_model_configs_are_serialisable_and_at_least_five_families(
    tmp_path: Path,
) -> None:
    configs = model_configs()

    assert set(ALL_MODELS) <= set(configs)
    families = {MODEL_SPECS[name].family for name in ALL_MODELS}
    assert len(families) >= 5
    for config in configs.values():
        assert isinstance(config["family"], str)
        assert isinstance(config["params"], dict)


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_model_reloads_from_saved_artifact_and_predicts(
    model_name: str, tmp_path: Path
) -> None:
    features, labels, _ = _training_rows(tmp_path)
    trained = fit_model(model_name, features, labels)
    artifact_path = tmp_path / f"{model_name}.joblib"

    joblib.dump(trained.estimator, artifact_path)
    reloaded = joblib.load(artifact_path)
    probabilities = reloaded.predict_proba(features)[:, 1]

    assert probabilities.shape == (features.shape[0],)
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0)


def test_observation_top1_accuracy_picks_best_candidate_per_observation() -> None:
    probabilities = np.array([0.9, 0.1, 0.2, 0.8])
    labels = np.array([1, 0, 0, 1])
    observation_indices = np.array([0, 0, 1, 1])

    assert observation_top1_accuracy(probabilities, labels, observation_indices) == 1.0

    flipped = np.array([0.1, 0.9, 0.2, 0.8])
    assert observation_top1_accuracy(flipped, labels, observation_indices) == 0.5
    assert observation_top1_accuracy(np.array([]), np.array([]), np.array([])) == 0.0
