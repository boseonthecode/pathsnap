# PathSnap — Progress Tracker

> Update this file after every completed implementation step.
> Move steps from **To Do** to **In Progress** to **Completed**; include the completion date and a one-line verification note.
> **Last updated:** 2026-11-18
> **Data audit:** 2026-11-18 — verified dataset against plan; findings and doc corrections recorded in `architecture.md` under `## Updates`.

---

## ⏭️ To Do

- [ ] Step 20 — Decoder comparison
- [ ] Step 21 — Decoder visualisation and failure analysis
- [ ] Step 22 — Artifact loading service
- [ ] Step 23 — FastAPI core endpoints
- [ ] Step 24 — API safeguards, configuration, and logging
- [ ] Step 25 — Static Leaflet UI shell
- [ ] Step 26 — Interactive noise and sampling controls
- [ ] Step 27 — Two-laptop deployment and CORS verification
- [ ] Step 28 — Folium map generation
- [ ] Step 29 — End-to-end verification across all ten test trips
- [ ] Step 30 — README and deployment instructions
- [ ] Step 31 — Technical report
- [ ] Step 32 — Leakage, security, and repository QA

---

## 🔄 In Progress

_None._

---

## ✅ Completed

- [x] Step 01 — Repository scaffolding (2026-11-18) — folder skeleton, `requirements.txt`, `.gitignore`, placeholder README, package import smoke test; `pytest tests/test_package.py` passed and all dependencies installed.
- [x] Step 02 — Configuration module (2026-11-18) — `pathsnap/config.py` with seeds, candidate/alignment/HMM defaults, UTM override, and `TRAIN_DATA_DIR`/`TEST_DATA_DIR`/`EVAL_DATA_DIR`/`ARTIFACT_DIR`/`RESULTS_DIR` env overrides; missing data dirs raise `FileNotFoundError`; `pytest tests/test_config.py` 5 passed.
- [x] Step 03 — Trip loader and validator (2026-11-18) — `pathsnap/io.py` loads the four raw files with shape/dtype/finite/monotonic/bounds validation; route discontinuities and repeated arcs are diagnostics not fatal; all 80 train + 10 test trips load, corrupt fixtures are rejected, and the audit counts (11 discontinuities, 10 repeated) reproduce; `pytest tests/test_io.py` 10 passed.
- [x] Step 04 — Dataset inventory / EDA script (2026-11-18) — `scripts/01_eda.py` + `pathsnap/eda.py` write `results/eda_summary.csv` (90 rows, 25 columns, no nulls), `results/eda_report.md`, and `results/folium/eda_00000001.html` with GPS/network/reference layers. Report flags 11 discontinued routes, 10 repeated-arc routes, and severe mismatch trips `00000033`/`00000054`; `pytest tests/test_eda.py` 5 passed.
- [x] Step 05 — Geometry and projection utilities (2026-11-18) — `pathsnap/geometry.py` adds per-trip UTM `Projection` (round-trip <1 cm), bearings, geodesic and projected distances, point-to-segment projection, arc lengths, directed graph, and `build_route_polyline` with explicit breaks or directed shortest-path bridging. EDA now reuses these utilities. `pytest tests/test_geometry.py` 13 passed.
- [x] Step 06 — Reference label alignment (2026-11-18) — `pathsnap/labels.py` implements monotonic DP alignment over route positions (`_monotonic_align`), emits labels only for observations within 50 m, returns coverage/backward-step/low-coverage diagnostics, and preserves repeated arcs by route position. All ten student-test trips align at ≥99% and `00000033`/`00000054` are flagged low-coverage; `pytest tests/test_labels.py` 5 passed.
- [x] Step 07 — Candidate generation with spatial index (2026-11-18) — `pathsnap/candidates.py` builds an STRtree over projected arcs, returns top-k candidates with distance/projection/along-fraction/heading metadata, falls back 50→100→200 m, and reports miss rate over alignable observations only. Student-test miss rate ≤1% with every alignable observation covered; `pytest tests/test_candidates.py` 7 passed.
- [x] Step 08 — Feature extraction (2026-11-18) — `pathsnap/features.py` defines the stable 10-feature order and extracts distance, log-distance, heading difference, speed consistency, rank, along-fraction, arc length, time gap, and node-degree context; empty candidate sets yield a typed empty matrix. Output is finite and bit-identical across runs; `pytest tests/test_features.py` 5 passed.
- [x] Step 09 — Feature cache and versioning (2026-11-18) — `pathsnap/cache.py` caches alignment/candidates/features per trip under a SHA-256 version key derived from the feature spec, candidate/alignment config, UTM override, and package version; mismatched/corrupt entries are rebuilt. `pytest tests/test_cache.py` 5 passed (reuse, feature-list invalidation, numerical identity, corruption recovery).
- [x] Step 10 — Metrics module (2026-11-18) — `pathsnap/metrics.py` implements observation Top-1, macro F1, precision/recall, sklearn-parity log loss, alignable-only coverage, route continuity, sequence-aligned route precision/recall, and length-weighted RMF plus a `Timer`. Repeated arcs score by position; perfect predictions give 1.0/zero loss; `pytest tests/test_metrics.py` 8 passed.
- [x] Step 11 — Training dataset assembly (2026-11-18) — `pathsnap/dataset.py` builds binary-labeled candidate rows with per-row route positions, excludes unalignable observations and low-coverage training trips (<90%), keeps student-test trips out of training, provides deterministic trip-grouped folds, and fits scalers on requested training trips only. `pytest tests/test_dataset.py` 8 passed (disjoint folds, excluded counts, repeated-arc occurrence, scaler isolation).
- [x] Step 12 — First three model families (2026-11-18) — `pathsnap/models.py` registers Logistic Regression, distance-weighted KNN, and ExtraTrees with serialisable configs, positive-class probabilities, top-1-by-observation helper, and metrics rows including train/inference timing. KNN neighbourhood adapts to the training-set size; `pytest tests/test_models.py` 5 passed.
- [x] Step 13 — Remaining model families (2026-11-18) — added LightGBM (boosting) and a shallow MLP (neural), bringing the registry to five distinct families. Every model trains on a subset, emits probabilities in [0,1], round-trips through joblib, and predicts after reload; `pytest tests/test_models.py` 12 passed.
- [x] Step 14 — Cross-validation and model comparison (2026-11-18) — `pathsnap/evaluate.py` + `scripts/03_train.py` run deterministic trip-grouped CV (300 evenly spaced observations/trip) and student-test evaluation for all five models, writing 30-row `results/model_comparison.csv`, `results/student_test_comparison.csv`, and `results/model_selection.md`. Selected `extra_trees` by CV Macro F1 (0.8858; student-test 0.8878). `pytest tests/test_evaluate.py` 6 passed.
- [x] Step 15 — Model comparison plots (2026-11-18) — `pathsnap/viz.py` generates CV Macro F1, student-test Top-1/Macro F1, log loss, confusion matrices, and timing figures from the stored CSVs into `results/figures/`. `pytest tests/test_viz.py` passed.
- [x] Step 16 — Bayesian hyperparameter optimisation (2026-11-18) — `pathsnap/tune.py` + `scripts/04_tune.py` run TPE with a trip-grouped CV Macro F1 objective and a size-bounded ExtraTrees space. 15/15 trials completed; `results/optuna_trials.csv` and `results/best_params.json` record params/objective; best CV Macro F1 0.8671. `pytest tests/test_tune.py` 3 passed.
- [x] Step 17 — Tuned vs. untuned comparison and export (2026-11-18) — `results/tuned_vs_untuned.csv` compares both variants on identical folds and student-test data: tuned 0.8671 vs untuned 0.8858 CV Macro F1, but tuned inference 69 ms vs 150 ms. Exported a 15 MB compressed `artifacts/model.joblib` plus `config.json` (feature order, params, thresholds, versions) and `hmm.json`. Fresh-process prediction parity verified; `pytest tests/test_artifacts.py` 5 passed.
- [x] Step 18 — HMM emission and transition models (2026-11-18) — `pathsnap/hmm.py` implements classifier and Gaussian emissions normalised in log space, directed network-distance transitions with same-arc fast paths, a cached `NetworkDistance`, and an explicit invalid-transition penalty. Toy tests confirm connected > disconnected, one-way violations are invalid, and emissions normalise; `pytest tests/test_hmm.py -m "not slow"` 5 passed.
- [x] Step 19 — Viterbi decoding and route stitching (2026-11-18) — added log-space `viterbi_log`, segment splitting on missing candidates/time gaps, `decode_hmm`/`decode_nearest`/`decode_classifier`, and `stitch_arc_sequence` using directed shortest paths. Toy graphs match hand-computed optima; a real test trip decodes a contiguous route (continuity ≥0.9) covering ≥95% of observations; total HMM tests 6 passed.
- [x] Extra — Static EDA plots (2026-11-18) — `pathsnap/eda_viz.py` + `scripts/01_eda_plots.py` generate six PNGs from `results/eda_summary.csv`: inventory/network size, sampling/gaps, route coverage with anomaly annotations, route anomalies, route/track ratio, and trip extents.
- [x] Extra — Feature selection and ablation (2026-11-18) — `pathsnap/ablation.py` + `scripts/08_feature_ablation.py` run full, leave-one-feature-out, and cumulative-group CV on training folds only. `results/feature_ablation.csv`, `results/feature_selection.md`, `results/selected_features.json`, and `results/figures/feature_ablation.png` record the 6-feature subset (CV Macro F1 0.8666 vs 0.8671 full).
