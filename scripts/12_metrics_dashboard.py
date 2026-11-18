"""Recompute classifier CV and frozen test metrics; export aggregate UI data."""
import importlib
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

from pathsnap import config
from pathsnap.artifacts import load_model_artifact
from pathsnap.dataset import trip_grouped_folds
from pathsnap.io import iter_trips
from pathsnap.hmm import HMMConfig, decode_hmm

helpers = importlib.import_module('10_optimize_decoder')


def classify(payload, model):
    trip, candidates, features, alignment, _ = payload
    probability = model.predict_proba(features)[:, list(model.classes_).index(1)]
    obs = np.array([c.obs_index for c in candidates.candidates], dtype=int)
    arcs = np.array([c.arc_id for c in candidates.candidates], dtype=int)
    mask = alignment.alignable[obs]
    y = (arcs[mask] == alignment.arc_ids[obs[mask]]).astype(int)
    selected = np.full(trip.n_points, -1, dtype=int)
    starts = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]
    ends = np.r_[starts[1:], len(obs)]
    for start, end in zip(starts, ends):
        best = start + np.argmax(probability[start:end])
        selected[obs[start]] = arcs[best]
    correct = int(np.count_nonzero(selected[alignment.alignable] == alignment.arc_ids[alignment.alignable]))
    return y, probability[mask], correct, int(alignment.alignable.sum()), obs, arcs, mask


def aggregate(items):
    y = np.concatenate([i[0] for i in items])
    p = np.concatenate([i[1] for i in items])
    return dict(macro_f1=float(f1_score(y, p >= .5, average='macro', labels=[0, 1], zero_division=0)),
                top1_accuracy=sum(i[2] for i in items)/sum(i[3] for i in items),
                log_loss=float(log_loss(y, p, labels=[0, 1])),
                precision=float(precision_score(y, p >= .5, zero_division=0)),
                recall=float(recall_score(y, p >= .5, zero_division=0)),
                scored_points=sum(i[3] for i in items), candidate_rows=len(y))


def complete_decoder_metrics(payload, artifact, paths):
    settings = HMMConfig(**json.loads((paths.artifact_dir/'hmm.json').read_text()))
    for partition, directory in [('student_test', paths.test_data_dir), ('evaluator_test', paths.project_root/'data/evaluator_test')]:
        f1s, true_rows, selected_rows, correct, total = [], [], [], 0, 0
        for trip in iter_trips(directory, 'test' if partition == 'student_test' else 'eval'):
            prepared = helpers.prepare(trip)
            _, candidates, features, alignment, _ = prepared
            result = decode_hmm(trip, candidates, artifact.positive_probability(features), settings)
            obs = np.array([c.obs_index for c in candidates.candidates])
            arcs = np.array([c.arc_id for c in candidates.candidates])
            mask = alignment.alignable[obs]
            y = arcs[mask] == alignment.arc_ids[obs[mask]]
            selected = arcs[mask] == result.point_arc_ids[obs[mask]]
            f1s.append(float(f1_score(y,selected,average='macro',labels=[0,1],zero_division=0)))
            true_rows.append(y); selected_rows.append(selected)
            correct += int(np.count_nonzero(result.point_arc_ids[alignment.alignable] == alignment.arc_ids[alignment.alignable]))
            total += int(alignment.alignable.sum())
        payload['partitions'][partition].update(
            decoded_pooled_top1=correct/total,
            decoded_mean_trip_macro_f1=float(np.mean(f1s)),
            decoded_pooled_macro_f1=float(f1_score(np.concatenate(true_rows),np.concatenate(selected_rows),average='macro',labels=[0,1],zero_division=0)))
        print(f'Completed decoder metrics for {partition}',flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decoder-only', action='store_true', help='Complete decoder metrics in the existing classifier/CV report')
    args = parser.parse_args()
    paths = config.get_paths()
    out = paths.results_dir/'model_metrics'
    out.mkdir(parents=True, exist_ok=True)
    artifact = load_model_artifact(paths.artifact_dir/'model.joblib')
    if args.decoder_only:
        payload = json.loads((out/'summary.json').read_text())
        if payload['params'] != artifact.params:
            raise ValueError('Model parameters changed; regenerate the full report')
        payload = complete_decoder_metrics(payload, artifact, paths)
        (out/'summary.json').write_text(json.dumps(payload,indent=2))
        (paths.project_root/'ui/metrics.json').write_text(json.dumps(payload,indent=2))
        return
    prepared = {t.trip_id: helpers.prepare(t) for t in iter_trips(paths.train_data_dir, 'train')}
    folds = []
    split_records = []
    for fold, (train_ids, val_ids) in enumerate(trip_grouped_folds(sorted(prepared), 5, 42)):
        print(f'Fitting CV fold {fold+1}/5', flush=True)
        x, y = helpers.training_arrays(prepared, train_ids)
        model = clone(artifact.estimator).set_params(n_jobs=4).fit(x, y)
        row = aggregate([classify(prepared[t], model) for t in val_ids])
        row['fold'] = fold+1
        folds.append(row)
        split_records.append(dict(fold=fold+1, training=train_ids, validation=val_ids))
        pd.DataFrame(folds).to_csv(out/'cv_folds.csv', index=False)
        print(row, flush=True)
    cv = {key: float(np.mean([f[key] for f in folds])) for key in ('macro_f1','top1_accuracy','log_loss','precision','recall')}
    cv['macro_f1_std'] = float(np.std([f['macro_f1'] for f in folds], ddof=1))
    cv['folds'] = 5
    partitions = {}
    for partition, directory, result_dir in [
        ('student_test', paths.test_data_dir, paths.results_dir/'decoder_optimization'),
        ('evaluator_test', paths.project_root/'data/evaluator_test', paths.results_dir/'evaluator_test')]:
        items, trip_rows = [], []
        for trip in iter_trips(directory, 'test' if partition == 'student_test' else 'eval'):
            item = classify(helpers.prepare(trip), artifact.estimator)
            items.append(item)
            trip_rows.append(dict(trip_id=trip.trip_id, **aggregate([item])))
        metrics = aggregate(items)
        metrics['trip_count'] = len(items)
        metrics['mean_trip_classifier_macro_f1'] = float(np.mean([r['macro_f1'] for r in trip_rows]))
        metrics['mean_trip_classifier_top1'] = float(np.mean([r['top1_accuracy'] for r in trip_rows]))
        name = 'per_trip_metrics.csv' if partition == 'evaluator_test' else 'test_comparison.csv'
        decoder = pd.read_csv(result_dir/name)
        decoder = decoder[decoder.variant == 'optimized']
        for key in ('route_f1','route_precision','route_recall','route_continuity','rmf','alignment_coverage'):
            metrics[key] = float(decoder[key].mean())
        metrics['decoded_mean_trip_top1'] = float(decoder.point_accuracy.mean())
        if partition == 'evaluator_test':
            metrics['decoded_pooled_top1'] = float(decoder.n_correct_points.sum()/decoder.n_scored_points.sum())
            metrics['decoded_mean_trip_macro_f1'] = float(decoder.decoded_candidate_macro_f1.mean())
        pd.DataFrame(trip_rows).to_csv(out/f'{partition}_classifier.csv',index=False)
        partitions[partition] = metrics
    payload = dict(generated_at=datetime.now(timezone.utc).isoformat(), model='ExtraTreesClassifier',
        decoder='HMM / Viterbi + monotonic point refinement', params=artifact.params,
        feature_count=len(artifact.feature_names), training_trips=80, max_observations_per_trip=1000,
        cv=cv, partitions=partitions,
        protocol='Five trip-disjoint folds, seed 42. Classifier test metrics pool candidate rows / observations. Decoder and route metrics are mean-trip unless explicitly marked pooled. Evaluator data used only for frozen scoring.',
        definitions={'macro_f1':'Mean binary candidate F1 across match/non-match classes, classifier threshold 0.5.',
          'decoder_macro_f1':'Binary candidate Macro F1 after the decoder selects roads; mean across evaluator trips.',
          'route_f1':'Harmonic mean of sequence-aligned route precision and recall; mean across trips.',
          'log_loss':'Binary candidate probability log loss before decoding. Lower is better.',
          'cv':'Mean of five classifier validation folds. Decoder settings were selected separately; this is not nested CV.'})
    payload = complete_decoder_metrics(payload, artifact, paths)
    (out/'summary.json').write_text(json.dumps(payload,indent=2))
    (out/'cv_splits.json').write_text(json.dumps(split_records,indent=2))
    (paths.project_root/'ui/metrics.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2),flush=True)


if __name__ == '__main__':
    main()
