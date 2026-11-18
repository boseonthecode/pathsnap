# Frozen evaluator results

Evaluated all 10 trips from `evaluator_test.zip`. No evaluator trips were used for fitting, parameter selection, or refinement selection. Artifact hashes were written before scoring in `frozen_manifest.json` and verified unchanged afterward.

| Metric | Original | Improved |
|---|---:|---:|
| Mean trip point accuracy | 85.19% | 96.84% |
| Pooled point accuracy | 75.62% | 96.72% |
| Mean route sequence F1 | 67.03% | 85.94% |
| Mean route precision | 56.23% | 78.82% |
| Mean route recall | 99.21% | 98.16% |
| Classifier candidate Macro F1 | 88.10% | 87.88% |
| Decoded candidate Macro F1 | 91.30% | 98.30% |
| Mean alignment coverage | 99.97% | 99.97% |

The 95% target is exceeded for point accuracy. Route F1 and classifier-only Macro F1 remain below 95%. Decoded candidate Macro F1 scores one selected candidate per observation; it is distinct from the classifier threshold metric shown by the existing UI.

Total scored points: 19,653; correct: 19,008. Point accuracy uses the existing monotonic reference-route alignment and excludes unalignable points. Mean-trip and pooled accuracy weight trips differently.

| Evaluator trip | Point accuracy | Route F1 |
|---|---:|---:|
| 00000015 | 94.70% | 49.17% |
| 00000017 | 94.73% | 81.69% |
| 00000023 | 95.41% | 73.89% |
| 00000030 | 89.45% | 90.16% |
| 00000049 | 100.00% | 97.73% |
| 00000059 | 98.45% | 93.24% |
| 00000067 | 100.00% | 95.29% |
| 00000073 | 99.56% | 96.36% |
| 00000087 | 98.71% | 97.70% |
| 00000094 | 97.43% | 84.18% |

## Changes and selection

- ExtraTrees retains the existing feature contract and tree hyperparameters; training sampling increases from 300 to at most 1,000 aligned observations per trip.
- Model fitting uses 64 training trips during decoder selection, with 16 disjoint training trips for validation (seed 42). The final classifier is fitted on all 80 training trips.
- Nineteen decoder configurations were compared on that validation split, followed by a two-way refinement comparison. No evaluator score influenced these choices.
- Selected settings: beta 2 m, same-road backward projection tolerance 60 m, distance-emission weight 0, monotonic alignment to the inferred route enabled.
- Point refinement reads only the predicted route. A regression test changes reference labels and verifies predictions remain identical.
- Missing-observation/time-gap segments are stitched separately, avoiding invented connecting routes across gaps.

## Artifacts and reproduction

The improved default artifacts are in `artifacts/`. The original artifacts are preserved locally in `.cache/baseline_artifacts/`. The improved compressed model is 21.91 MB.

Run from the repository root:

```powershell
.\.venv\Scripts\python scripts_optimize_decoder.py
.\.venv\Scripts\python scripts_optimize_decoder.py --refine-best
.\.venv\Scripts\python scripts	_evaluate_frozen.py --data-dir data\evaluator_test --baseline-dir .cacheaseline_artifacts
```

Selection logs, split IDs, per-trip results, and installed package versions are retained under `results/decoder_optimization/`. Evaluator predictions and metrics are under `results/evaluator_test/`. The supplied raw datasets and supporting PDFs are now tracked at the owner's request; docs/provided_files_manifest.json records their hashes.

Verification: 43 focused tests passed after artifact promotion. The prediction service loaded the promoted artifacts and reproduced 100% point accuracy on evaluator trip 00000049 (2,642 points); see service_smoke.json. The HTTP TestClient suite stalled in this environment and was interrupted, so no full HTTP integration pass is claimed.
