"""Run the saved PathSnap model live from the terminal, without an API server."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from pathsnap import config
from pathsnap.artifacts import load_hmm_config, load_model_artifact
from pathsnap.io import discover_trip_dirs, load_trip
from pathsnap.pipeline import predict_trip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=config.PROJECT_ROOT/'data/evaluator_test')
    parser.add_argument('--artifact-dir', type=Path, default=config.PROJECT_ROOT/'artifacts')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--trip-id', help='Run one trip, for example 00000023')
    selection.add_argument('--all', action='store_true', help='Run every trip in the chosen partition')
    selection.add_argument('--list', action='store_true', help='List available trip IDs')
    parser.add_argument('--method', choices=['hmm', 'classifier', 'nearest'], default='hmm')
    parser.add_argument('--noise', type=float, default=0, help='Additional GPS noise standard deviation in metres')
    parser.add_argument('--sample-seconds', type=float, default=1, help='Sampling interval; use 0 to preserve every input observation')
    parser.add_argument('--output-dir', type=Path, default=config.PROJECT_ROOT/'results/live_demo')
    args = parser.parse_args()
    if args.noise < 0 or args.sample_seconds < 0:
        parser.error('--noise and --sample-seconds must be nonnegative')
    trips = discover_trip_dirs(args.data_dir)
    if not trips:
        parser.error(f'No trips found in {args.data_dir}')
    if args.list:
        print('\n'.join(p.name for p in trips))
        return
    if args.trip_id:
        trips = [p for p in trips if p.name == args.trip_id]
        if not trips:
            parser.error('Unknown trip ID. Use --list to see available trips.')
    elif not args.all:
        parser.error('Choose --trip-id ID, --all, or --list')

    print('Loading saved ExtraTreesClassifier...', flush=True)
    artifact = load_model_artifact(args.artifact_dir/'model.joblib')
    settings = load_hmm_config(args.artifact_dir / 'hmm.json')
    print(f'Model: {artifact.model_name} | trees={artifact.params.get("n_estimators")} | depth={artifact.params.get("max_depth")} | features={len(artifact.feature_names)}')
    print(f'Method: {args.method} | noise={args.noise:g} m | sampling={args.sample_seconds:g} s')
    print(f'Data: {args.data_dir.resolve()}')
    print('Inference runs now; no training or cached predictions are used.\n', flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for directory in trips:
        print(f'Running trip {directory.name}...', flush=True)
        trip = load_trip(directory, 'eval')
        result = predict_trip(trip, artifact, method=args.method,
                              noise_sigma_m=args.noise, sample_every_s=args.sample_seconds,
                              hmm_config=settings)
        m = result.metrics
        for name, key in [('Point Top-1', 'top1_accuracy'), ('Classifier Macro F1', 'macro_f1'),
                          ('Decoder Macro F1', 'decoder_macro_f1'), ('Route F1', 'route_f1'),
                          ('Route precision', 'route_precision'), ('Route recall', 'route_recall'),
                          ('Route continuity', 'route_continuity'), ('Alignment coverage', 'alignment_coverage')]:
            print(f'  {name:<24} {m[key]*100:8.2f}%')
        print(f'  Classifier log loss      {m["log_loss"]:9.4f}')
        print(f'  Route mismatch fraction {m["rmf"]:9.4f}')
        print(f'  Inference time           {result.inference_ms/1000:8.3f} s')
        print(f'  Predicted road segments  {len(result.predicted_arcs):8d}')
        scored = round(len(result.effective_track)*m['alignment_coverage'])
        summary = dict(trip_id=trip.trip_id, method=args.method, **m,
                       inference_ms=result.inference_ms, scored_points=scored,
                       correct_points=round(scored*m['top1_accuracy']))
        summaries.append(summary)
        np.savetxt(args.output_dir/f'{trip.trip_id}.route', result.predicted_arcs, fmt='%d')
        (args.output_dir/f'{trip.trip_id}.json').write_text(json.dumps({**summary,
            'predicted_arcs':result.predicted_arcs.tolist(),
            'point_arc_ids':result.point_arc_ids.tolist()}, indent=2))
        print('', flush=True)
    total = sum(row['scored_points'] for row in summaries)
    correct = sum(row['correct_points'] for row in summaries)
    print(f'Completed {len(summaries)} trip(s).')
    print(f'Pooled point accuracy: {100*correct/total:.2f}%' if total else 'No alignable points to score.')
    print(f'Mean-trip decoder Macro F1: {100*np.mean([r["decoder_macro_f1"] for r in summaries]):.2f}%')
    print(f'Predictions and metrics saved in: {args.output_dir.resolve()}')
    (args.output_dir/'summary.json').write_text(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
