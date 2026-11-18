# Current model evaluation

Model: **ExtraTreesClassifier**, 200 trees, maximum depth 12, minimum leaf size 10, 10 features. Sequential inference: HMM/Viterbi with backward GPS jitter tolerance and monotonic point refinement along the inferred route.

| Metric | 5-fold CV | Student test | Evaluator test |
|---|---:|---:|---:|
| Classifier Macro F1 | 85.46% | 86.29% | 83.56% |
| Classifier Top-1 | 78.86% | 80.93% | 74.31% |
| Classifier log loss (lower is better) | 0.2545 | 0.2382 | 0.2707 |
| Decoded Top-1 (pooled) | N/A | 95.62% | 96.72% |
| Decoder Macro F1 (pooled) | N/A | 97.44% | 98.14% |
| Decoder Macro F1 (mean trip) | N/A | 98.03% | 98.30% |
| Route F1 (mean trip) | N/A | 89.68% | 85.94% |
| Route precision | N/A | 82.73% | 78.82% |
| Route recall | N/A | 99.28% | 98.16% |
| Route continuity | N/A | 99.86% | 99.80% |
| Route mismatch fraction | N/A | 0.1117 | 0.1063 |

Five trip-disjoint folds, seed 42. Classifier test metrics pool candidate rows / observations. Decoder and route metrics are mean-trip unless explicitly marked pooled. Evaluator data used only for frozen scoring.

Classifier test Macro F1 and log loss pool all candidate rows; classifier Top-1 pools aligned observations. Mean-trip metrics give equal weight to every trip. For example, evaluator classifier Macro F1 is 83.56% pooled and 87.88% averaged per trip. These are different aggregations of the same predictions.

CV Macro F1 standard deviation: 2.51 percentage points. CV covers the classifier; the decoder was tuned separately. These are not nested-CV estimates. Test datasets were never used for fitting this model or selecting its new decoder settings.

Recompute with `python scripts/12_metrics_dashboard.py`. Per-fold and per-trip classifier results are saved in this directory. The UI reads the same aggregate JSON from `ui/metrics.json`.

Verification: 27 map/metric/static checks and 19 API/end-to-end/static checks passed (overlapping tests included). The live map and results panel were inspected in the browser. Screenshots are in `results/ui/`.