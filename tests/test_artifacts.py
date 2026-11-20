"""Tests for model/config artifact save, load, validation, and fresh-process parity."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pathsnap import config
from pathsnap.artifacts import (
    build_model_artifact,
    load_model_artifact,
    read_json,
    save_model_artifact,
    write_config_json,
    write_json,
)
from pathsnap.io import load_trip
from pathsnap.models import fit_model
from pathsnap.cache import build_trip_artifacts
from pathsnap.dataset import build_trip_dataset
from conftest import write_trip


def _trained_and_features(tmp_path: Path):
    trip = load_trip(write_trip(tmp_path / "train", "00000055"), "train")
    dataset = build_trip_dataset(trip, build_trip_artifacts(trip))
    trained = fit_model("logistic_regression", dataset.features, dataset.labels)
    return trained, dataset.features


def test_model_artifact_round_trip_preserves_predictions(tmp_path: Path) -> None:
    trained, features = _trained_and_features(tmp_path)
    artifact = build_model_artifact(trained)
    path = save_model_artifact(artifact, tmp_path / "model.joblib")

    loaded = load_model_artifact(path)

    assert loaded.model_name == "logistic_regression"
    assert loaded.feature_names == trained.feature_names
    np.testing.assert_allclose(
        loaded.positive_probability(features),
        trained.positive_probability(features),
    )


def test_artifact_rejects_feature_spec_mismatch(tmp_path: Path) -> None:
    trained, features = _trained_and_features(tmp_path)
    artifact = build_model_artifact(trained)

    with pytest.raises(ValueError, match="Expected .* features"):
        artifact.positive_probability(features[:, :3])


def test_config_json_records_contract_and_hmm_defaults(tmp_path: Path) -> None:
    trained, _ = _trained_and_features(tmp_path)
    artifact = build_model_artifact(trained)
    config_path = write_config_json(
        tmp_path / "config.json",
        artifact,
        candidate_config={"top_k": 8},
        alignment_config={"max_distance_m": 50.0},
        hmm_config={"beta_m": 10.0},
        seed=42,
    )
    payload = read_json(config_path)

    assert payload["feature_names"] == list(trained.feature_names)
    assert payload["candidate"]["top_k"] == 8
    assert payload["hmm"]["beta_m"] == 10.0
    assert payload["seed"] == 42
    assert payload["model_name"] == "logistic_regression"


def test_json_write_and_read_round_trip(tmp_path: Path) -> None:
    path = write_json(tmp_path / "data.json", {"b": 2, "a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert read_json(path) == {"a": 1, "b": 2}


def test_fresh_process_loads_artifact_and_predicts_identically(tmp_path: Path) -> None:
    trained, features = _trained_and_features(tmp_path)
    artifact = build_model_artifact(trained)
    artifact_path = save_model_artifact(artifact, tmp_path / "model.joblib")
    features_path = tmp_path / "features.npy"
    np.save(features_path, features)

    script = f"""
import json
import numpy as np
from pathsnap.artifacts import load_model_artifact

artifact = load_model_artifact({str(artifact_path)!r})
features = np.load({str(features_path)!r})
print(json.dumps(artifact.positive_probability(features).tolist()))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(config.PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    fresh_probabilities = np.array(json.loads(completed.stdout))

    np.testing.assert_allclose(
        fresh_probabilities,
        artifact.positive_probability(features),
        rtol=1e-12,
        atol=1e-12,
    )


def test_load_hmm_config_drops_unknown_keys(tmp_path: Path) -> None:
    from pathsnap.artifacts import load_hmm_config

    payload = {"beta_m": 2.0, "refine_points": True, "unknown_future_key": 1}
    path = write_json(tmp_path / "hmm.json", payload)
    settings = load_hmm_config(path)
    assert settings.beta_m == 2.0
    assert settings.refine_points is True


def test_load_hmm_config_falls_back_to_config_json(tmp_path: Path) -> None:
    from pathsnap.artifacts import load_hmm_config

    write_json(tmp_path / "config.json", {"hmm": {"beta_m": 7.5}})
    settings = load_hmm_config(tmp_path / "missing.json", fallback=tmp_path / "config.json")
    assert settings.beta_m == 7.5


def test_load_hmm_config_uses_defaults_when_nothing_exists(tmp_path: Path) -> None:
    from pathsnap.artifacts import load_hmm_config
    from pathsnap.hmm import HMMConfig

    assert load_hmm_config(tmp_path / "missing.json") == HMMConfig()
