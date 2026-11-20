"""Recompute classifier CV and frozen test metrics; export aggregate UI data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

from pathsnap import config
from pathsnap.artifacts import load_hmm_config, load_model_artifact
from pathsnap.config import Paths
from pathsnap.dataset import trip_grouped_folds
from pathsnap.decoder_tuning import (
    DecoderTrip,
    prepare_decoder_trip,
    to_inference,
    training_arrays,
)
from pathsnap.hmm import decode_hmm
from pathsnap.io import iter_trips
from pathsnap.labels import candidate_labels
from pathsnap.metrics import Timer

EVALUATOR_DIR = config.PROJECT_ROOT / "data" / "evaluator_test"


def classify(prepared: DecoderTrip, estimator: Any) -> tuple:
    """Classifier-only per-candidate labels, probabilities, and top-1 counts."""
    trip = prepared.trip
    candidates = prepared.candidates
    alignment = prepared.alignment
    probability = estimator.predict_proba(prepared.features)[:, list(estimator.classes_).index(1)]
    observation_indices = np.array([c.obs_index for c in candidates.candidates], dtype=int)
    arc_ids = np.array([c.arc_id for c in candidates.candidates], dtype=int)
    mask = alignment.alignable[observation_indices]
    labels = candidate_labels(
        alignment, observation_indices[mask], arc_ids[mask]
    ).astype(int)

    selected = np.full(trip.n_points, -1, dtype=int)
    starts = np.r_[0, np.flatnonzero(np.diff(observation_indices)) + 1]
    ends = np.r_[starts[1:], len(observation_indices)]
    for start, end in zip(starts, ends):
        best = start + int(np.argmax(probability[start:end]))
        selected[observation_indices[start]] = arc_ids[best]

    correct = int(
        np.count_nonzero(
            selected[alignment.alignable] == alignment.arc_ids[alignment.alignable]
        )
    )
    return (
        labels,
        probability[mask],
        correct,
        int(alignment.alignable.sum()),
        observation_indices,
        arc_ids,
        mask,
    )


def aggregate(items: list[tuple]) -> dict[str, float]:
    """Pooled candidate metrics over several classified trips."""
    labels = np.concatenate([item[0] for item in items])
    probabilities = np.concatenate([item[1] for item in items])
    predicted = probabilities >= config.CANDIDATE_THRESHOLD
    scored = sum(item[3] for item in items)
    return {
        "macro_f1": float(
            f1_score(labels, predicted, average="macro", labels=[0, 1], zero_division=0)
        ),
        "top1_accuracy": sum(item[2] for item in items) / scored,
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "scored_points": scored,
        "candidate_rows": len(labels),
    }


def _decoder_rows(artifact: Any, settings: Any, directory: Path, partition: str) -> dict[str, float]:
    fold_f1: list[float] = []
    true_rows: list[np.ndarray] = []
    selected_rows: list[np.ndarray] = []
    correct = 0
    total = 0
    for trip in iter_trips(directory, partition):
        prepared = prepare_decoder_trip(trip)
        inference = to_inference(prepared, artifact)
        result = decode_hmm(
            inference.trip, inference.candidates, inference.probabilities, settings
        )
        observation_indices = np.array(
            [c.obs_index for c in inference.candidates.candidates]
        )
        arc_ids = np.array([c.arc_id for c in inference.candidates.candidates])
        mask = inference.alignment.alignable[observation_indices]
        truth = arc_ids[mask] == inference.alignment.arc_ids[observation_indices[mask]]
        predicted = arc_ids[mask] == result.point_arc_ids[observation_indices[mask]]
        fold_f1.append(
            float(f1_score(truth, predicted, average="macro", labels=[0, 1], zero_division=0))
        )
        true_rows.append(truth)
        selected_rows.append(predicted)
        correct += int(
            np.count_nonzero(
                result.point_arc_ids[inference.alignment.alignable]
                == inference.alignment.arc_ids[inference.alignment.alignable]
            )
        )
        total += int(inference.alignment.alignable.sum())
    return {
        "decoded_pooled_top1": correct / total,
        "decoded_mean_trip_macro_f1": float(np.mean(fold_f1)),
        "decoded_pooled_macro_f1": float(
            f1_score(
                np.concatenate(true_rows),
                np.concatenate(selected_rows),
                average="macro",
                labels=[0, 1],
                zero_division=0,
            )
        ),
    }


def complete_decoder_metrics(
    payload: dict[str, Any], artifact: Any, paths: Paths
) -> dict[str, Any]:
    """Add decoded top-1 and macro-F1 to the existing classifier report."""
    settings = load_hmm_config(paths.artifact_dir / "hmm.json")
    for partition, directory in (
        ("student_test", paths.test_data_dir),
        ("evaluator_test", EVALUATOR_DIR),
    ):
        payload["partitions"][partition].update(
            _decoder_rows(
                artifact,
                settings,
                directory,
                "test" if partition == "student_test" else "eval",
            )
        )
        print(f"Completed decoder metrics for {partition}", flush=True)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decoder-only",
        action="store_true",
        help="Complete decoder metrics in the existing classifier/CV report",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = config.get_paths()
    out = paths.results_dir / "model_metrics"
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = paths.project_root / "ui" / "metrics.json"
    artifact = load_model_artifact(paths.artifact_dir / "model.joblib")

    if args.decoder_only:
        payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        if payload["params"] != artifact.params:
            raise ValueError("Model parameters changed; regenerate the full report")
        payload = complete_decoder_metrics(payload, artifact, paths)
        (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    prepared = {
        trip.trip_id: prepare_decoder_trip(trip)
        for trip in iter_trips(paths.train_data_dir, "train")
    }
    folds = []
    split_records = []
    splits = trip_grouped_folds(
        sorted(prepared), n_splits=config.N_SPLITS, seed=config.CV_SEED
    )
    for fold, (train_ids, validation_ids) in enumerate(splits):
        print(f"Fitting CV fold {fold + 1}/{len(splits)}", flush=True)
        features, labels = training_arrays(prepared, train_ids)
        model = clone(artifact.estimator).set_params(n_jobs=4).fit(features, labels)
        row = aggregate([classify(prepared[trip_id], model) for trip_id in validation_ids])
        row["fold"] = fold + 1
        folds.append(row)
        split_records.append(
            dict(fold=fold + 1, training=train_ids, validation=validation_ids)
        )
        pd.DataFrame(folds).to_csv(out / "cv_folds.csv", index=False)
        print(row, flush=True)

    cv = {
        key: float(np.mean([fold[key] for fold in folds]))
        for key in ("macro_f1", "top1_accuracy", "log_loss", "precision", "recall")
    }
    cv["macro_f1_std"] = float(np.std([fold["macro_f1"] for fold in folds], ddof=1))
    cv["folds"] = len(splits)

    partitions: dict[str, dict[str, Any]] = {}
    for partition, directory, result_dir in (
        ("student_test", paths.test_data_dir, paths.results_dir / "decoder_optimization"),
        ("evaluator_test", EVALUATOR_DIR, paths.results_dir / "evaluator_test"),
    ):
        items = []
        trip_rows = []
        for trip in iter_trips(directory, "test" if partition == "student_test" else "eval"):
            item = classify(prepare_decoder_trip(trip), artifact.estimator)
            items.append(item)
            trip_rows.append(dict(trip_id=trip.trip_id, **aggregate([item])))
        metrics = aggregate(items)
        metrics["trip_count"] = len(items)
        metrics["mean_trip_classifier_macro_f1"] = float(
            np.mean([row["macro_f1"] for row in trip_rows])
        )
        metrics["mean_trip_classifier_top1"] = float(
            np.mean([row["top1_accuracy"] for row in trip_rows])
        )
        name = (
            "per_trip_metrics.csv"
            if partition == "evaluator_test"
            else "test_comparison.csv"
        )
        decoder = pd.read_csv(result_dir / name)
        decoder = decoder[decoder.variant == "optimized"]
        for key in (
            "route_f1",
            "route_precision",
            "route_recall",
            "route_continuity",
            "rmf",
            "alignment_coverage",
        ):
            metrics[key] = float(decoder[key].mean())
        metrics["decoded_mean_trip_top1"] = float(decoder.point_accuracy.mean())
        if partition == "evaluator_test":
            metrics["decoded_pooled_top1"] = float(
                decoder.n_correct_points.sum() / decoder.n_scored_points.sum()
            )
            metrics["decoded_mean_trip_macro_f1"] = float(
                decoder.decoded_candidate_macro_f1.mean()
            )
        pd.DataFrame(trip_rows).to_csv(out / f"{partition}_classifier.csv", index=False)
        partitions[partition] = metrics

    payload = dict(
        generated_at=datetime.now(timezone.utc).isoformat(),
        model=type(artifact.estimator).__name__,
        decoder="HMM / Viterbi + monotonic point refinement",
        params=artifact.params,
        feature_count=len(artifact.feature_names),
        training_trips=len(prepared),
        max_observations_per_trip=config.TRAIN_MAX_OBSERVATIONS_PER_TRIP,
        cv=cv,
        partitions=partitions,
        protocol=(
            f"{len(splits)} trip-disjoint folds, seed {config.CV_SEED}. Classifier test "
            "metrics pool candidate rows / observations. Decoder and route metrics are "
            "mean-trip unless explicitly marked pooled. Evaluator data used only for "
            "frozen scoring."
        ),
        definitions={
            "macro_f1": "Mean binary candidate F1 across match/non-match classes, "
            f"classifier threshold {config.CANDIDATE_THRESHOLD}.",
            "decoder_macro_f1": "Binary candidate Macro F1 after the decoder selects "
            "roads; mean across evaluator trips.",
            "route_f1": "Harmonic mean of sequence-aligned route precision and recall; "
            "mean across trips.",
            "log_loss": "Binary candidate probability log loss before decoding. Lower "
            "is better.",
            "cv": "Mean of the classifier validation folds. Decoder settings were "
            "selected separately; this is not nested CV.",
        },
    )
    payload = complete_decoder_metrics(payload, artifact, paths)
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "cv_splits.json").write_text(json.dumps(split_records, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
