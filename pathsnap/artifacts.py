"""Model and configuration artifact persistence.

Artifacts are plain joblib/JSON files. Loading validates the feature order so a
stale model can never silently mispredict against a newer feature spec.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

import pathsnap
from pathsnap.hmm import HMMConfig
from pathsnap.models import TrainedModel, positive_class_probability


@dataclass(frozen=True)
class ModelArtifact:
    """A fitted estimator plus the contract it was trained against."""

    estimator: Any
    feature_names: tuple[str, ...]
    model_name: str
    family: str
    params: dict[str, Any]
    metadata: dict[str, Any]

    def positive_probability(self, features: np.ndarray) -> np.ndarray:
        """Positive-class probability, rejecting a feature-order mismatch."""
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, "
                f"got shape {features.shape}"
            )
        if features.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        return positive_class_probability(self.estimator, features)


def version_info() -> dict[str, str]:
    import lightgbm
    import sklearn

    return {
        "pathsnap": pathsnap.__version__,
        "sklearn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "numpy": np.__version__,
    }


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            )
            .strip()
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_model_artifact(
    trained: TrainedModel, metadata: dict[str, Any] | None = None
) -> ModelArtifact:
    payload = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "versions": version_info(),
    }
    if metadata:
        payload.update(metadata)
    return ModelArtifact(
        estimator=trained.estimator,
        feature_names=tuple(trained.feature_names),
        model_name=trained.spec.name,
        family=trained.spec.family,
        params=dict(trained.spec.params),
        metadata=payload,
    )


def save_model_artifact(artifact: ModelArtifact, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "estimator": artifact.estimator,
            "feature_names": list(artifact.feature_names),
            "model_name": artifact.model_name,
            "family": artifact.family,
            "params": artifact.params,
            "metadata": artifact.metadata,
        },
        path,
        compress=3,
    )
    return path


def load_model_artifact(path: Path) -> ModelArtifact:
    payload = joblib.load(path)
    required = {"estimator", "feature_names", "model_name", "params"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Model artifact missing keys: {sorted(missing)}")
    return ModelArtifact(
        estimator=payload["estimator"],
        feature_names=tuple(payload["feature_names"]),
        model_name=str(payload["model_name"]),
        family=str(payload.get("family", "unknown")),
        params=dict(payload["params"]),
        metadata=dict(payload.get("metadata", {})),
    )


def write_config_json(
    path: Path,
    artifact: ModelArtifact,
    *,
    candidate_config: dict[str, Any],
    alignment_config: dict[str, Any],
    hmm_config: dict[str, Any],
    seed: int,
) -> Path:
    config_payload = {
        "model_name": artifact.model_name,
        "family": artifact.family,
        "params": artifact.params,
        "feature_names": list(artifact.feature_names),
        "candidate": candidate_config,
        "alignment": alignment_config,
        "hmm": hmm_config,
        "seed": seed,
        "metadata": artifact.metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_hmm_config(path: Path, fallback: Path | None = None) -> HMMConfig:
    """Load decoder settings, falling back to ``config.json["hmm"]``.

    Unknown keys are dropped so an artifact written by a newer build still loads
    against this ``HMMConfig`` definition.
    """
    payload: dict[str, Any] = {}
    if path.is_file():
        payload.update(read_json(path))
    elif fallback is not None and fallback.is_file():
        payload.update(read_json(fallback).get("hmm", {}))
    allowed = {field.name for field in fields(HMMConfig)}
    return HMMConfig(**{key: value for key, value in payload.items() if key in allowed})
