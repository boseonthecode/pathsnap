"""Tune the decoder on trip-disjoint training validation; evaluate test once.

Results and artifacts are isolated under ``results/decoder_optimization`` until
reviewed. Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
from sklearn.base import clone

from pathsnap import config
from pathsnap.artifacts import (
    load_hmm_config,
    load_model_artifact,
    save_model_artifact,
    version_info,
)
from pathsnap.dataset import trip_grouped_folds
from pathsnap.decoder_tuning import (
    prepare_decoder_trip,
    score_decoder,
    search_grid,
    to_inference,
    training_arrays,
)
from pathsnap.hmm import HMMConfig, NetworkDistance
from pathsnap.io import iter_trips


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refine-best",
        action="store_true",
        help="Validate point refinement for the preceding search winner",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="Directory holding the baseline model.joblib and hmm.json",
    )
    return parser.parse_args()


def _mean(rows: list[dict], key: str) -> float:
    return float(sum(row[key] for row in rows) / len(rows))


def main() -> None:
    args = _parse_args()
    paths = config.get_paths()
    baseline_dir = args.baseline_dir or paths.artifact_dir
    out = paths.results_dir / "decoder_optimization"
    out.mkdir(parents=True, exist_ok=True)

    original = load_model_artifact(baseline_dir / "model.joblib")
    base = load_hmm_config(baseline_dir / "hmm.json")

    prepared = {}
    for index, trip in enumerate(iter_trips(paths.train_data_dir, "train")):
        prepared[trip.trip_id] = prepare_decoder_trip(trip)
        print(f"Prepared training {index + 1}: {trip.trip_id}", flush=True)

    train_ids, validation_ids = trip_grouped_folds(
        list(prepared), n_splits=config.N_SPLITS, seed=config.CV_SEED
    )[0]
    features, labels = training_arrays(prepared, train_ids)
    model = clone(original.estimator).set_params(n_jobs=4).fit(features, labels)
    validation_model = replace(original, estimator=model)

    inferences = [to_inference(prepared[trip_id], validation_model) for trip_id in validation_ids]
    networks = [NetworkDistance(item.trip) for item in inferences]

    candidates: list[HMMConfig] = search_grid(base)
    prefix = "validation"
    if args.refine_best:
        selected = load_hmm_config(out / "hmm.json")
        candidates = [
            replace(selected, refine_points=False),
            replace(selected, refine_points=True),
        ]
        prefix = "refinement_validation"

    rows: list[dict] = []
    for index, settings in enumerate(candidates):
        current = []
        for inference, network in zip(inferences, networks):
            row = score_decoder(inference, settings, network)
            row.update(trial=index, **settings.to_dict())
            current.append(row)
        rows.extend(current)
        frame = pd.DataFrame(rows)
        frame.to_csv(out / f"{prefix}_trials.csv", index=False)
        print(
            f"Trial {index}: accuracy={_mean(current, 'point_accuracy'):.5f}, "
            f"route_f1={_mean(current, 'route_f1'):.5f}",
            flush=True,
        )

    summary = frame.groupby("trial")[["point_accuracy", "route_f1", "rmf"]].mean()
    # Primary target: observation top-1 accuracy; route F1 breaks ties.
    best = int(summary.sort_values(["point_accuracy", "route_f1"], ascending=False).index[0])
    settings = candidates[best]
    summary.to_csv(out / f"{prefix}_summary.csv")
    (out / "hmm.json").write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
    (out / f"{prefix}_split.json").write_text(
        json.dumps(
            {
                "fit_trip_ids": train_ids,
                "validation_trip_ids": validation_ids,
                "seed": config.CV_SEED,
                "selection_metric": "mean trip point_accuracy; route_f1 tie-break",
                "best_trial": best,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Selected {best}; refitting on all training trips", flush=True)
    features, labels = training_arrays(prepared, sorted(prepared))
    model = clone(original.estimator).set_params(n_jobs=4).fit(features, labels)
    final = replace(
        original,
        estimator=model,
        metadata={
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "versions": version_info(),
            "max_observations_per_trip": config.TRAIN_MAX_OBSERVATIONS_PER_TRIP,
            "training_trip_ids": sorted(prepared),
            "decoder_validation_trip_ids": validation_ids,
            "selection": (
                "ExtraTrees parameters inherited; decoder selected by training "
                "holdout point accuracy"
            ),
            "seed": config.CV_SEED,
        },
    )
    save_model_artifact(final, out / "model.joblib")

    test_rows: list[dict] = []
    for trip in iter_trips(paths.test_data_dir, "test"):
        prepared_trip = prepare_decoder_trip(trip)
        network = NetworkDistance(trip)
        for name, artifact, settings_for_variant in (
            ("baseline", original, base),
            ("optimized", final, settings),
        ):
            row = score_decoder(to_inference(prepared_trip, artifact), settings_for_variant, network)
            row["variant"] = name
            test_rows.append(row)
        pd.DataFrame(test_rows).to_csv(out / "test_comparison.csv", index=False)
        print(f"Test {trip.trip_id}: {test_rows[-1]['point_accuracy']:.5f}", flush=True)

    print(
        pd.DataFrame(test_rows)
        .groupby("variant")[["point_accuracy", "route_f1", "rmf"]]
        .mean()
        .to_string(),
        flush=True,
    )


if __name__ == "__main__":
    main()
