"""Evaluate frozen artifacts on evaluator trips; never fit or tune anything."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.artifacts import load_model_artifact
from pathsnap.evaluate import prepare_trip_inference, _decode_row
from pathsnap.hmm import HMMConfig, NetworkDistance, decode_hmm
from pathsnap.io import iter_trips
from pathsnap.metrics import Timer, candidate_macro_f1


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, default=Path('artifacts'))
    parser.add_argument('--candidate-dir', type=Path, default=Path('results/decoder_optimization'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/evaluator_test'))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    variants = {'baseline': args.baseline_dir, 'optimized': args.candidate_dir}
    manifest = {
        'protocol': 'Frozen before evaluator scoring; no training or parameter search in this script',
        'data_dir': str(args.data_dir.resolve()),
        'artifacts': {name: {f: sha256(directory / f) for f in ('model.joblib', 'hmm.json')}
                      for name, directory in variants.items()},
    }
    (out / 'frozen_manifest.json').write_text(json.dumps(manifest, indent=2))
    models = {name: load_model_artifact(directory / 'model.joblib') for name, directory in variants.items()}
    settings = {name: HMMConfig(**json.loads((directory / 'hmm.json').read_text())) for name, directory in variants.items()}
    training_ids = {p.name for p in config.get_paths().train_data_dir.iterdir() if p.is_dir()}
    rows = []
    for trip in iter_trips(args.data_dir, 'eval'):
        if trip.trip_id in training_ids:
            raise ValueError(f'Evaluator trip {trip.trip_id} overlaps training IDs')
        network = NetworkDistance(trip)
        for name in variants:
            inf = prepare_trip_inference(trip, models[name])
            with Timer() as timer:
                result = decode_hmm(trip, inf.candidates, inf.probabilities, settings[name], network)
            row = _decode_row(inf, result, timer.elapsed_ms)
            row['variant'] = name
            p, r = row['route_precision'], row['route_recall']
            row['route_f1'] = 2*p*r/(p+r) if p+r else 0.
            obs = np.array([c.obs_index for c in inf.candidates.candidates])
            arcs = np.array([c.arc_id for c in inf.candidates.candidates])
            valid = inf.alignment.alignable[obs]
            labels = arcs[valid] == inf.alignment.arc_ids[obs[valid]]
            row['classifier_candidate_macro_f1'] = candidate_macro_f1(labels, inf.probabilities[valid] >= .5)
            row['decoded_candidate_macro_f1'] = candidate_macro_f1(labels, arcs[valid] == result.point_arc_ids[obs[valid]])
            row['n_scored_points'] = int(inf.alignment.alignable.sum())
            row['n_correct_points'] = int(np.count_nonzero(result.point_arc_ids[inf.alignment.alignable] == inf.alignment.arc_ids[inf.alignment.alignable]))
            rows.append(row)
            prediction_dir = out / 'predictions' / name
            prediction_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(prediction_dir / f'{trip.trip_id}.route', result.predicted_arcs, fmt='%d')
            np.savetxt(prediction_dir / f'{trip.trip_id}.point_arcs', result.point_arc_ids, fmt='%d')
            print(f'{name} {trip.trip_id}: accuracy={row["point_accuracy"]:.5f}, route_f1={row["route_f1"]:.5f}', flush=True)
        pd.DataFrame(rows).to_csv(out / 'per_trip_metrics.csv', index=False)
    frame = pd.DataFrame(rows)
    metrics = ['point_accuracy', 'route_precision', 'route_recall', 'route_f1', 'rmf', 'route_continuity', 'alignment_coverage', 'classifier_candidate_macro_f1', 'decoded_candidate_macro_f1']
    summary = frame.groupby('variant')[metrics].mean()
    counts = frame.groupby('variant')[['n_correct_points', 'n_scored_points']].sum()
    summary['pooled_point_accuracy'] = counts.n_correct_points / counts.n_scored_points
    summary.to_csv(out / 'summary.csv')
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
