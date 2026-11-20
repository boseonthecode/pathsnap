"""Model registry, fitting, probability prediction, and evaluation rows.

Each family exposes ``predict_proba`` so its positive-class probability can be
used directly as an HMM emission score. Configurations are serialisable for the
model comparison artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier

from pathsnap import config
from pathsnap.features import FEATURE_NAMES
from pathsnap.metrics import (
    Timer,
    candidate_log_loss,
    candidate_macro_f1,
    candidate_precision_recall,
)


@dataclass(frozen=True)
class ModelSpec:
    """Constructive description of one model family."""

    name: str
    family: str
    params: dict[str, Any]
    factory: Callable[[int], BaseEstimator]

    def config(self) -> dict[str, Any]:
        return {"family": self.family, "params": dict(self.params)}


def _logistic_regression(seed: int) -> BaseEstimator:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def _distance_weighted_knn(seed: int) -> BaseEstimator:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=15,
                    weights="distance",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _extra_trees(seed: int) -> BaseEstimator:
    return ExtraTreesClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )


def _lightgbm(seed: int) -> BaseEstimator:
    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def _shallow_mlp(seed: int) -> BaseEstimator:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=1e-3,
                    batch_size=1024,
                    learning_rate_init=1e-3,
                    max_iter=150,
                    early_stopping=False,
                    random_state=seed,
                ),
            ),
        ]
    )


def _hist_gradient_boosting(seed: int) -> BaseEstimator:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1e-3,
        class_weight="balanced",
        early_stopping=False,
        random_state=seed,
    )


def _random_forest(seed: int) -> BaseEstimator:
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )


def _decision_tree(seed: int) -> BaseEstimator:
    return DecisionTreeClassifier(
        max_depth=14,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=seed,
    )


def _gaussian_nb(seed: int) -> BaseEstimator:
    return GaussianNB()


def _sgd_logistic(seed: int) -> BaseEstimator:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-4,
                    max_iter=1500,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def _adaboost(seed: int) -> BaseEstimator:
    return AdaBoostClassifier(
        n_estimators=150,
        learning_rate=0.05,
        random_state=seed,
    )


MODEL_SPECS: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        name="logistic_regression",
        family="linear",
        params={
            "max_iter": 1000,
            "class_weight": "balanced",
            "scaler": "standard",
        },
        factory=_logistic_regression,
    ),
    "knn": ModelSpec(
        name="knn",
        family="instance_based",
        params={
            "n_neighbors": 15,
            "weights": "distance",
            "scaler": "standard",
        },
        factory=_distance_weighted_knn,
    ),
    "extra_trees": ModelSpec(
        name="extra_trees",
        family="ensemble",
        params={
            "n_estimators": 200,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
        },
        factory=_extra_trees,
    ),
    "lightgbm": ModelSpec(
        name="lightgbm",
        family="boosting",
        params={
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
        },
        factory=_lightgbm,
    ),
    "mlp": ModelSpec(
        name="mlp",
        family="neural",
        params={
            "hidden_layer_sizes": (64, 32),
            "alpha": 1e-3,
            "batch_size": 1024,
            "max_iter": 150,
            "scaler": "standard",
        },
        factory=_shallow_mlp,
    ),
    "hist_gradient_boosting": ModelSpec(
        name="hist_gradient_boosting",
        family="boosting_hist",
        params={
            "max_iter": 300,
            "learning_rate": 0.08,
            "max_leaf_nodes": 31,
            "l2_regularization": 1e-3,
            "class_weight": "balanced",
        },
        factory=_hist_gradient_boosting,
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        family="ensemble_forest",
        params={
            "n_estimators": 200,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
        },
        factory=_random_forest,
    ),
    "decision_tree": ModelSpec(
        name="decision_tree",
        family="tree",
        params={
            "max_depth": 14,
            "min_samples_leaf": 10,
            "class_weight": "balanced",
        },
        factory=_decision_tree,
    ),
    "gaussian_nb": ModelSpec(
        name="gaussian_nb",
        family="probabilistic",
        params={},
        factory=_gaussian_nb,
    ),
    "sgd_logistic": ModelSpec(
        name="sgd_logistic",
        family="linear_sgd",
        params={
            "loss": "log_loss",
            "alpha": 1e-4,
            "max_iter": 1500,
            "class_weight": "balanced",
            "scaler": "standard",
        },
        factory=_sgd_logistic,
    ),
    "adaboost": ModelSpec(
        name="adaboost",
        family="boosting_adaptive",
        params={
            "n_estimators": 150,
            "learning_rate": 0.05,
        },
        factory=_adaboost,
    ),
}


def positive_class_probability(estimator: Any, features: np.ndarray) -> np.ndarray:
    """Probability of the positive (true-arc) class for each row.

    Shared by :class:`TrainedModel` and :class:`pathsnap.artifacts.ModelArtifact`
    so a reloaded estimator scores identically to the freshly fitted one.
    """
    probabilities = estimator.predict_proba(features)
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    return np.zeros(probabilities.shape[0], dtype=np.float64)


@dataclass
class TrainedModel:
    """A fitted estimator plus provenance and timing."""

    spec: ModelSpec
    estimator: BaseEstimator
    feature_names: tuple[str, ...] = FEATURE_NAMES
    train_time_s: float = 0.0

    def positive_probability(self, features: np.ndarray) -> np.ndarray:
        """Probability of the positive (true-arc) class for each row."""
        if features.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        if features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {features.shape[1]}"
            )
        return positive_class_probability(self.estimator, features)


@dataclass(frozen=True)
class ModelResult:
    """Metrics row for one trained model on one evaluation slice."""

    model: str
    family: str
    config: dict[str, Any]
    n_rows: int
    n_observations: int
    macro_f1: float
    precision: float
    recall: float
    log_loss: float
    top1_accuracy: float | None
    train_time_s: float
    inference_ms: float
    tn: int
    fp: int
    fn: int
    tp: int

    def to_row(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "family": self.family,
            "config": self.config,
            "n_rows": self.n_rows,
            "n_observations": self.n_observations,
            "macro_f1": self.macro_f1,
            "precision": self.precision,
            "recall": self.recall,
            "log_loss": self.log_loss,
            "top1_accuracy": self.top1_accuracy,
            "train_time_s": self.train_time_s,
            "inference_ms": self.inference_ms,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
        }


def fit_model(
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = config.DEFAULT_SEED,
    params: dict[str, Any] | None = None,
    feature_names: Sequence[str] | None = None,
) -> TrainedModel:
    """Fit a registered model family, optionally overriding parameters or feature subset."""
    if name not in MODEL_SPECS:
        raise KeyError(f"Unknown model {name!r}; available: {sorted(MODEL_SPECS)}")
    spec = MODEL_SPECS[name]
    estimator = spec.factory(seed)
    if params:
        estimator.set_params(**params)

    # Instance-based models cannot request more neighbours than training rows.
    estimator_params = estimator.get_params()
    if "classifier__n_neighbors" in estimator_params:
        estimator.set_params(
            classifier__n_neighbors=min(
                int(estimator_params["classifier__n_neighbors"]),
                max(1, int(features.shape[0])),
            )
        )

    effective_params = {**spec.params, **(params or {})}
    effective_spec = ModelSpec(
        name=spec.name,
        family=spec.family,
        params=effective_params,
        factory=spec.factory,
    )

    with Timer() as timer:
        estimator.fit(features, labels)
    return TrainedModel(
        spec=effective_spec,
        estimator=estimator,
        feature_names=tuple(feature_names) if feature_names is not None else FEATURE_NAMES,
        train_time_s=timer.elapsed_ms / 1000.0,
    )


def candidate_top1_accuracy(
    probabilities: np.ndarray,
    labels: np.ndarray,
    observation_indices: np.ndarray,
) -> float:
    """Top-1 accuracy after picking the highest-probability candidate per observation.

    Distinct from :func:`pathsnap.metrics.observation_top1_accuracy`, which
    compares predicted arc ids against aligned labels.
    """
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    observation_indices = np.asarray(observation_indices)
    if probabilities.size == 0:
        return 0.0

    order = np.argsort(observation_indices, kind="stable")
    sorted_obs = observation_indices[order]
    sorted_prob = probabilities[order]
    sorted_labels = labels[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_obs[1:] != sorted_obs[:-1]])
    ends = np.r_[boundaries[1:], sorted_obs.size]

    correct = 0
    total = 0
    for start, end in zip(boundaries, ends):
        best = start + int(np.argmax(sorted_prob[start:end]))
        correct += int(sorted_labels[best] == 1)
        total += 1
    return float(correct / total) if total else 0.0


def evaluate_model(
    trained: TrainedModel,
    features: np.ndarray,
    labels: np.ndarray,
    observation_indices: np.ndarray | None = None,
) -> ModelResult:
    """Score a fitted model, recording quality metrics and inference time."""
    with Timer() as timer:
        probabilities = trained.positive_probability(features)
    predictions = (
        (probabilities >= config.CANDIDATE_THRESHOLD).astype(np.int8)
        if probabilities.size
        else probabilities
    )
    precision, recall = candidate_precision_recall(labels, predictions)
    top1 = (
        candidate_top1_accuracy(probabilities, labels, observation_indices)
        if observation_indices is not None
        else None
    )

    labels_int = np.asarray(labels).astype(np.int8)
    tn = int(np.count_nonzero((labels_int == 0) & (predictions == 0)))
    fp = int(np.count_nonzero((labels_int == 0) & (predictions == 1)))
    fn = int(np.count_nonzero((labels_int == 1) & (predictions == 0)))
    tp = int(np.count_nonzero((labels_int == 1) & (predictions == 1)))
    n_observations = (
        int(np.unique(observation_indices).size) if observation_indices is not None else 0
    )

    return ModelResult(
        model=trained.spec.name,
        family=trained.spec.family,
        config=trained.spec.config(),
        n_rows=int(features.shape[0]),
        n_observations=n_observations,
        macro_f1=candidate_macro_f1(labels, predictions),
        precision=precision,
        recall=recall,
        log_loss=candidate_log_loss(labels, probabilities),
        top1_accuracy=top1,
        train_time_s=trained.train_time_s,
        inference_ms=timer.elapsed_ms,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )


def model_configs() -> dict[str, dict[str, Any]]:
    """Serialisable configurations for every registered model."""
    return {name: spec.config() for name, spec in MODEL_SPECS.items()}
