"""Tune the decoder on trip-disjoint training validation; evaluate test once.

Run from the repository: python scripts/10_optimize_decoder.py
Results/artifacts are isolated under results/decoder_optimization until reviewed.
"""
from __future__ import annotations

import json
import hashlib
import argparse
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from pathsnap import config
from pathsnap.artifacts import load_model_artifact, save_model_artifact, version_info
from pathsnap.cache import compute_cache_version
from pathsnap.candidates import generate_candidates
from pathsnap.dataset import trip_grouped_folds
from pathsnap.evaluate import TripInference, _decode_row
from pathsnap.features import extract_features
from pathsnap.geometry import arc_lengths_m
from pathsnap.hmm import HMMConfig, NetworkDistance, decode_hmm
from pathsnap.io import iter_trips
from pathsnap.labels import align_observations
from pathsnap.metrics import Timer


def prepare(trip):
    digest = hashlib.sha256(compute_cache_version().encode())
    for values in (trip.track, trip.nodes, trip.arcs, trip.route):
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    cache = config.CACHE_DIR / 'decoder_optimization' / f'{trip.partition}_{trip.trip_id}_{digest.hexdigest()[:16]}.joblib'
    if cache.exists():
        return joblib.load(cache)
    candidates = generate_candidates(trip)
    features = extract_features(trip, candidates).features
    alignment = align_observations(trip)
    payload = (trip, candidates, features, alignment, arc_lengths_m(trip))
    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, cache, compress=3)
    return payload


def training_arrays(prepared, ids):
    xs, ys = [], []
    for trip_id in ids:
        trip, candidates, features, alignment, _ = prepared[trip_id]
        if alignment.coverage < config.LOW_COVERAGE_THRESHOLD:
            continue
        obs = np.array([c.obs_index for c in candidates.candidates])
        arcs = np.array([c.arc_id for c in candidates.candidates])
        valid_obs = np.flatnonzero(alignment.alignable)
        if not valid_obs.size:
            continue
        sampled = valid_obs[np.linspace(0, len(valid_obs) - 1, min(1000, len(valid_obs)), dtype=int)]
        mask = np.isin(obs, sampled)
        xs.append(features[mask])
        ys.append((arcs[mask] == alignment.arc_ids[obs[mask]]).astype(np.int8))
    return np.vstack(xs), np.concatenate(ys)


def inference(payload, artifact):
    trip, candidates, features, alignment, lengths = payload
    return TripInference(trip, candidates, artifact.positive_probability(features), alignment, lengths)


def score(inf, settings, network):
    with Timer() as timer:
        decoded = decode_hmm(inf.trip, inf.candidates, inf.probabilities, settings, network)
    row = _decode_row(inf, decoded, timer.elapsed_ms)
    p, r = row['route_precision'], row['route_recall']
    row['route_f1'] = 2 * p * r / (p + r) if p + r else 0.0
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refine-best', action='store_true', help='Validate point refinement for the preceding search winner')
    parser.add_argument('--baseline-dir', type=Path, default=Path('artifacts'))
    args = parser.parse_args()
    paths = config.get_paths()
    out = paths.results_dir / 'decoder_optimization'
    out.mkdir(parents=True, exist_ok=True)
    original = load_model_artifact(args.baseline_dir / 'model.joblib')
    prepared = {}
    for i, trip in enumerate(iter_trips(paths.train_data_dir, 'train')):
        prepared[trip.trip_id] = prepare(trip)
        print(f'Prepared training {i + 1}: {trip.trip_id}', flush=True)
    train_ids, val_ids = trip_grouped_folds(list(prepared), n_splits=5, seed=42)[0]
    x, y = training_arrays(prepared, train_ids)
    model = clone(original.estimator).set_params(n_jobs=4)
    model.fit(x, y)
    validation_model = replace(original, estimator=model)
    inferences = [inference(prepared[t], validation_model) for t in val_ids]
    networks = [NetworkDistance(i.trip) for i in inferences]
    base = HMMConfig(**json.loads((args.baseline_dir / 'hmm.json').read_text()))
    candidates = [base]
    for tolerance in (10., 30., 60.):
        for beta in (2., 10., 50.):
            for weight in (0., 1.):
                candidates.append(replace(base, backward_tolerance_m=tolerance, beta_m=beta, distance_weight=weight))
    prefix = 'validation'
    if args.refine_best:
        selected = HMMConfig(**json.loads((out / 'hmm.json').read_text()))
        candidates = [replace(selected, refine_points=False), replace(selected, refine_points=True)]
        prefix = 'refinement_validation'
    rows = []
    for index, settings in enumerate(candidates):
        current = []
        for inf, network in zip(inferences, networks):
            row = score(inf, settings, network)
            row.update(trial=index, **settings.to_dict())
            current.append(row)
        rows.extend(current)
        frame = pd.DataFrame(rows)
        frame.to_csv(out / f'{prefix}_trials.csv', index=False)
        print(f'Trial {index}: accuracy={np.mean([r["point_accuracy"] for r in current]):.5f}, route_f1={np.mean([r["route_f1"] for r in current]):.5f}', flush=True)
    summary = frame.groupby('trial')[['point_accuracy', 'route_f1', 'rmf']].mean()
    # Primary target: observation top-1 accuracy; route F1 breaks ties.
    best = int(summary.sort_values(['point_accuracy', 'route_f1'], ascending=False).index[0])
    settings = candidates[best]
    summary.to_csv(out / f'{prefix}_summary.csv')
    (out / 'hmm.json').write_text(json.dumps(settings.to_dict(), indent=2))
    (out / f'{prefix}_split.json').write_text(json.dumps({'fit_trip_ids': train_ids, 'validation_trip_ids': val_ids, 'seed': 42, 'selection_metric': 'mean trip point_accuracy; route_f1 tie-break', 'best_trial': best}, indent=2))
    print(f'Selected {best}; refitting on all training trips', flush=True)
    x, y = training_arrays(prepared, sorted(prepared))
    model = clone(original.estimator).set_params(n_jobs=4).fit(x, y)
    final = replace(original, estimator=model, metadata={
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'versions': version_info(),
        'max_observations_per_trip': 1000,
        'training_trip_ids': sorted(prepared),
        'decoder_validation_trip_ids': val_ids,
        'selection': 'ExtraTrees parameters inherited; decoder selected by training holdout point accuracy',
        'seed': 42,
    })
    save_model_artifact(final, out / 'model.joblib')
    test_rows = []
    for trip in iter_trips(paths.test_data_dir, 'test'):
        payload = prepare(trip)
        network = NetworkDistance(trip)
        for name, artifact, cfg in [('baseline', original, base), ('optimized', final, settings)]:
            row = score(inference(payload, artifact), cfg, network)
            row['variant'] = name
            test_rows.append(row)
        pd.DataFrame(test_rows).to_csv(out / 'test_comparison.csv', index=False)
        print(f'Test {trip.trip_id}: {test_rows[-1]["point_accuracy"]:.5f}', flush=True)
    print(pd.DataFrame(test_rows).groupby('variant')[['point_accuracy', 'route_f1', 'rmf']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
