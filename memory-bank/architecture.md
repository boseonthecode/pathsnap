# PathSnap — Architecture

> Initial architecture for the MVP. This document describes planned structure; files are created progressively per `implementation-plan.md`.
> **Last updated:** 2026-11-18

---

## 1. System Overview

PathSnap is a local, two-tier system:

```
┌──────────────────────────── Laptop 2 ────────────────────────────┐
│  Static UI (HTML + CSS + JS + Leaflet)                            │
│  - trip list • map layers • sliders • metrics panel               │
└───────────────────────────────┬───────────────────────────────────┘
                                │ HTTP (JSON)
┌───────────────────────────────▼──────────────────────────── Laptop 1 ──┐
│  FastAPI backend                                                       │
│  - artifact loader • prediction service • server-side scorer           │
│  - no ground truth leaves this process                                 │
└───────────────────────────────┬────────────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────────────┐
│  ML pipeline (offline scripts, shared `pathsnap` package)              │
│  data ingestion → candidates → features → models → tuning → HMM/Viterbi│
└───────────────────────────────┬────────────────────────────────────────┘
                                │
                    File-based data + artifacts
        data/{train,test}/, artifacts/, results/
```

The same `pathsnap` package powers training scripts, evaluation, and the API so that inference matches experiments exactly.

---

## 2. Planned Folder Structure

```
ml_hackathon_26/
├── data/
│   ├── train/                     # 80 labelled training trips
│   └── test/                      # 10 labelled student-testing trips
├── docs/
│   ├── idea.md
│   └── problem/                   # Provided hackathon briefs
├── memory-bank/
│   ├── prd.md
│   ├── tech-stack.md
│   ├── implementation-plan.md
│   ├── architecture.md
│   └── progress.md
├── .claude/
│   └── rules/
│       ├── code-style.md
│       ├── testing.md
│       └── security.md
├── pathsnap/                      # Core Python package (shared by scripts/API)
│   ├── __init__.py
│   ├── config.py                  # Paths, env overrides, defaults, seeds
│   ├── io.py                      # Trip loading + validation + discovery
│   ├── geometry.py                # Projection, bearings, projections, polylines
│   ├── labels.py                  # Observation → reference arc alignment
│   ├── candidates.py              # STRtree candidate generation
│   ├── features.py                # (observation, candidate) feature extraction
│   ├── cache.py                   # Versioned feature/candidate caches
│   ├── metrics.py                 # Classification/route metrics
│   ├── models.py                  # Model registry, fit/predict wrappers
│   ├── tune.py                    # Optuna search + grouped CV objective
│   ├── hmm.py                     # Emission/transition, Viterbi, stitching
│   ├── pipeline.py                # End-to-end prediction for one trip
│   ├── artifacts.py               # Save/load model + config + HMM metadata
│   ├── evaluate.py                # Method comparison utilities
│   └── viz.py                     # matplotlib plots and report helpers
├── scripts/
│   ├── 01_eda.py                  # Inventory, data quality, EDA outputs
│   ├── 02_features.py             # Build/cache candidates + features
│   ├── 03_train.py                # Train + compare ≥ 5 models (grouped CV)
│   ├── 04_tune.py                 # Bayesian optimisation + export
│   ├── 05_hmm.py                  # Nearest-snap vs clf vs clf+HMM comparison
│   ├── 06_e2e.py                  # End-to-end API verification over test trips
│   └── 07_folium.py               # Interactive report maps
├── api/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app, startup/shutdown, CORS
│   ├── schemas.py                 # Pydantic request/response models
│   ├── service.py                 # Artifact loading + prediction orchestration
│   └── deps.py                    # Shared dependencies/config access
├── ui/
│   ├── index.html                 # Single-page evaluator/demo interface
│   ├── style.css                  # Layout, panels, legend, responsive styles
│   ├── app.js                     # Trip list, map layers, sliders, metrics
│   └── config.js                  # API base URL + UI defaults
├── artifacts/
│   ├── model.joblib               # Final tuned model
│   ├── config.json                # Feature order, thresholds, versions
│   └── hmm.json                   # Decoder + candidate configuration
├── results/
│   ├── eda_summary.csv
│   ├── eda_report.md
│   ├── model_comparison.csv
│   ├── optuna_trials.csv
│   ├── hmm_comparison.csv
│   ├── e2e_summary.csv
│   ├── figures/                   # PNG plots for the report
│   └── folium/                    # Interactive HTML maps
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_io.py
│   ├── test_geometry.py
│   ├── test_labels.py
│   ├── test_candidates.py
│   ├── test_features.py
│   ├── test_metrics.py
│   ├── test_models.py
│   ├── test_hmm.py
│   ├── test_artifacts.py
│   └── test_api.py
├── report/
│   ├── report.md                  # Report source content
│   ├── report.pdf                 # Final PDF deliverable
│   └── assets/                    # Screenshots and exported figures
├── requirements.txt
├── README.md
└── .gitignore
```

### Directory responsibilities
| Path | Responsibility |
| :--- | :--- |
| `data/` | Immutable raw trips. Never modified by code. |
| `docs/` | Provided briefs and original idea. Read-only context. |
| `memory-bank/` | Product/architecture/plan/progress memory. Updated after each step. |
| `pathsnap/` | All reusable data-processing, ML, and matching logic. |
| `scripts/` | Thin, numbered entry points for experiments and report generation. |
| `api/` | FastAPI application and server-side evaluation. |
| `ui/` | Build-free browser interface served separately from the API. |
| `artifacts/` | Inference assets required at runtime. |
| `results/` | Generated evidence (CSVs, plots, maps) for the report. |
| `tests/` | Automated verification of core modules and API. |
| `report/` | Technical report source, assets, and final PDF. |

---

## 3. Key File Map

| File | Planned responsibility |
| :--- | :--- |
| `pathsnap/config.py` | Central defaults, env overrides, path validation, random seeds. |
| `pathsnap/io.py` | `Trip` dataclass, loaders for four file types, validation, route-quality diagnostics, discovery. |
| `pathsnap/geometry.py` | UTM projection, inverse projection, bearing, point projection, route polyline with bridge/break handling for non-contiguous references. |
| `pathsnap/labels.py` | Robust monotonic sequence alignment to route positions; repeated-arc and unalignable-point handling; coverage diagnostics. |
| `pathsnap/candidates.py` | Per-trip STRtree, adaptive-radius top-k candidates, fallback behavior. |
| `pathsnap/features.py` | Canonical feature definitions and vectorised extraction. |
| `pathsnap/cache.py` | Content/version-keyed candidate and feature caches. |
| `pathsnap/metrics.py` | Top-1, Macro F1, log loss, miss rate, continuity, route P/R, RMF, alignment coverage. |
| `pathsnap/models.py` | Model registry, grouped CV orchestration, probability prediction. |
| `pathsnap/tune.py` | Optuna search space, objective, trial export. |
| `pathsnap/hmm.py` | Emissions, transitions, Viterbi, gap handling, shortest-path stitching. |
| `pathsnap/pipeline.py` | Single-trip prediction used by API and scripts. |
| `pathsnap/artifacts.py` | Save/load/verify model, feature spec, HMM config, metadata. |
| `pathsnap/evaluate.py` | Nearest-snap/classifier/HMM comparison helpers. |
| `pathsnap/viz.py` | Reusable matplotlib plot functions driven by results tables. |
| `api/main.py` | App factory, lifespan artifact loading, CORS, routers. |
| `api/schemas.py` | Typed request/response contracts; no reference-route fields. |
| `api/service.py` | Load artifacts once, retrieve trips, run pipeline, server-side scoring. |
| `ui/app.js` | Fetch trips/predictions, Leaflet layers, slider debounce, metric rendering. |
| `ui/config.js` | Runtime API base URL for single-machine or two-laptop deployment. |

---

## 4. Data Contracts ("Database Schema")

No database is used. The following are on-disk contracts and in-memory record shapes.

### 4.1 Raw trip files
Each trip directory is named by `trip_id` and contains four files:

| File | Columns | Semantics |
| :--- | :--- | :--- |
| `<trip_id>.track` | `longitude: float`, `latitude: float`, `timestamp_s: float` | GPS observations; nominally 1 Hz. |
| `<trip_id>.nodes` | `longitude: float`, `latitude: float` | Node coordinates; row index = `node_id`. |
| `<trip_id>.arcs` | `from_node_id: int`, `to_node_id: int` | Directed arc endpoints; row index = `arc_id`. |
| `<trip_id>.route` | `arc_id: int` | Chronological reference arc sequence; may contain repeated arcs and may skip connector arcs. |

**Constraints**
- `0 ≤ node_id < len(nodes)`; `0 ≤ arc_id < len(arcs)`.
- Track timestamps strictly increasing after preprocessing.
- Arc IDs are file row indices, not node IDs.
- Route connectivity is **expected but not guaranteed**: the data audit found 1–3 node discontinuities in 11/90 trips and repeated arcs in 10/90 trips. Validation must report these, not reject the trip.
- Because arcs can repeat, labels and scoring must track route position/sequence, never a bare `arc_id` lookup.

### 4.2 Conceptual entities (in memory / derived)

**Trip**
| Field | Type | Description |
| :--- | :--- | :--- |
| `trip_id` | string | Folder name, zero-padded. |
| `partition` | enum(`train`,`test`,`eval`) | Source partition. |
| `nodes` | `float64[N,2]` | Lon/lat. |
| `arcs` | `int32[M,2]` | From/to node IDs. |
| `track` | `float64[T,3]` | Lon/lat/time. |
| `route` | `int32[R]` | Reference arc sequence (never sent to UI). |
| `bbox` | `float64[4]` | Min/max lon/lat for map extent. |

**Candidate** (per observation)
| Field | Type | Description |
| :--- | :--- | :--- |
| `trip_id`, `obs_index` | string, int | Keys. |
| `arc_id` | int | Candidate arc. |
| `distance_m` | float | Perpendicular distance from observation. |
| `proj_lon`, `proj_lat` | float | Projection on arc. |
| `along_fraction` | float `[0,1]` | Normalised position along arc. |
| `heading_diff_deg` | float | Smallest angular difference. |
| `rank` | int | Spatial rank. |

**FeatureRow** (per candidate)
Stable ordered columns: `distance_m`, `log_distance`, `heading_diff_deg`, `speed_consistency`, `candidate_rank`, `along_fraction`, `arc_length_m`, `time_gap_s`, `from_degree`, `to_degree`. Exact order is stored in `artifacts/config.json`.

**ModelArtifact**
| Field | Type | Description |
| :--- | :--- | :--- |
| `model` | joblib object | Fitted classifier with `predict_proba`. |
| `feature_names` | string[] | Canonical order. |
| `model_name` | string | Selected family. |
| `trained_at` | ISO timestamp | Provenance. |
| `git_commit` | string | Reproducibility. |
| `sklearn_version`, `lightgbm_version` | string | Compatibility. |

**HMMConfig**
| Field | Type | Description |
| :--- | :--- | :--- |
| `top_k` | int | Candidate cap. |
| `radius_m` | float | Base candidate radius. |
| `max_radius_m` | float | Adaptive fallback. |
| `beta_m` | float | Transition scale. |
| `invalid_penalty` | float | Disconnected/one-way penalty. |
| `gap_split_s` | float | Split track on large time gaps. |
| `emission` | enum(`classifier`,`gaussian`) | Emission source. |

### 4.3 Result tables (CSV)
| File | Columns (planned) |
| :--- | :--- |
| `results/eda_summary.csv` | `partition, trip_id, n_points, n_nodes, n_arcs, n_route_arcs, duration_s, median_dt_s, max_gap_s, nonunit_dt_frac, bbox_*, route_len_m, track_len_m, route_track_ratio, route_gps_within50_frac, route_discontinuities, repeated_route_arcs, candidate_miss_rate, alignment_coverage` |
| `results/model_comparison.csv` | `model, family, config, fold, top1_accuracy, macro_f1, precision, recall, log_loss, candidate_miss_rate, route_continuity, alignment_coverage, train_time_s, inference_ms` |
| `results/optuna_trials.csv` | `trial, state, params_json, cv_macro_f1, duration_s` |
| `results/hmm_comparison.csv` | `trip_id, method, top1_accuracy, macro_f1, log_loss, route_continuity, route_precision, route_recall, rmf, alignment_coverage, inference_ms` |
| `results/e2e_summary.csv` | `trip_id, status, method, n_predicted_arcs, top1_accuracy, route_continuity, alignment_coverage, inference_ms` |

---

## 5. API Route Plan

Base URL: `http://<laptop-1>:8000`. All responses are JSON. No endpoint exposes reference routes, labels, or evaluator file contents.

| Method | Path | Purpose | Auth | Response highlights |
| :--- | :--- | :--- | :--- | :--- |
| GET | `/health` | Liveness + artifact readiness | None | `status`, `model_loaded`, `version` |
| GET | `/trips` | List evaluation trips | None | `trip_id`, `point_count`, `bbox`, `duration_s` |
| GET | `/trips/{trip_id}/observations` | Raw GPS for display | None | `trip_id`, `points[{lon,lat,timestamp}]` |
| POST | `/predict` | Run full matching pipeline | None | predicted arcs, route geometry, per-point confidence, aggregate metrics |
| GET | `/docs` | Auto OpenAPI UI | None | Swagger/Redoc |

### `POST /predict` request
| Field | Type | Default | Notes |
| :--- | :--- | :--- | :--- |
| `trip_id` | string | required | Must exist in configured eval directory. |
| `method` | enum(`nearest`,`classifier`,`hmm`) | `hmm` | Selects decoder. |
| `noise_sigma_m` | float `[0,100]` | `0` | Server-side deterministic perturbation. |
| `sample_every_s` | float `[1,60]` | `1` | Downsample observations. |
| `seed` | int | `42` | Reproducible perturbation. |

### `POST /predict` response
| Field | Type | Notes |
| :--- | :--- | :--- |
| `trip_id`, `method` | string | Echoed request context. |
| `predicted_arcs` | int[] | Final decoded/stiched arc sequence. |
| `route_coords` | `[lon,lat][]` | Polyline for the map. |
| `point_predictions` | `[{obs_index, arc_id, confidence}]` | Per-observation output. |
| `metrics` | object | `top1_accuracy`, `macro_f1`, `log_loss`, `candidate_miss_rate`, `route_continuity`, `route_precision`, `route_recall`, `rmf`, `alignment_coverage`, `inference_ms`. |
| `layers` | object | `gps_points` (effective perturbed/downsampled points used for inference), optional `candidate_points`; never reference geometry. |

### Error contracts
- `404` trip not found.
- `422` invalid method/parameter/trip id.
- `503` artifacts not loaded.
- Error body: `{ "detail": "..." }` without internal paths or labels.

---

## 6. Key Architectural Decisions

| Decision | Reasoning |
| :--- | :--- |
| **Per-trip UTM projection** | All distance/threshold math must be metric; UTM chosen from trip centroid keeps distortion negligible at city scale. |
| **Arc ID = row index, node ID = row index** | Matches provided file format exactly and avoids fragile external ID maps. |
| **STRtree candidate generation** | Fast nearest-geometry queries on graphs up to ~307k arcs; returns candidates for all observations without pairwise brute force. |
| **Monotonic sequence alignment for labels** | The reference route is chronological, but naive nearest-segment projection is not reliably monotonic on self-overlapping/out-and-back tracks (audit: median ~98.5%, as low as 67%). Use dynamic-programming / constrained-progress alignment over route positions; track route position rather than arc ID because arcs can repeat. |
| **Reference-quality diagnostics and unalignable-point handling** | The audit found non-contiguous routes (11/90), repeated arcs (10/90), and severe GPS–route mismatch in 2/80 training trips. Labels are emitted only for alignable observations; coverage is reported, and anomalous trips are flagged rather than silently mislabelled. |
| **Binary candidate classifier** | Converts map matching into supervised `(observation, candidate)` classification, enabling ≥5 model families and calibrated probabilities for HMM emissions. |
| **Classifier emission + network transition HMM** | Learned plausibility handles GPS noise/parallel roads; graph transitions enforce physical connectivity and one-way rules. |
| **Viterbi in log space with cached shortest paths** | Prevents underflow and keeps decoding tractable on large graphs; cutoff and top-k bound the search. |
| **Shortest-path stitching** | Ensures the returned route is contiguous even when observations skip short arcs between 1 Hz samples. |
| **Trip-grouped CV** | Prevents within-trip leakage; matches the requirement that no trip spans partitions. |
| **No database** | File artifacts and CSVs are inspectable, reproducible, and require zero schema work in a 7-hour build. |
| **No authentication** | Trusted local LAN demo; security effort is redirected to ground-truth isolation and input validation. |
| **Build-free static UI** | Removes Node/build/deploy complexity; Leaflet via CDN; UI can be served with any static server. |
| **Server-side scoring + no GT in responses** | Mandatory evaluation integrity: reference routes never enter the prediction/UI path. |
| **Configurable eval directory** | The ten evaluator trips can be mounted at runtime without code changes or hardcoding. |
| **Single package shared by scripts and API** | Guarantees inference parity with experiments and avoids duplicated preprocessing. |
| **Deterministic seeds and versioned caches** | Required for reproducible metrics, trials, and report figures. |

---

## 7. Request Flow (Prediction)

1. UI calls `GET /trips`; API lists trip metadata from `EVAL_DATA_DIR` without labels.
2. Evaluator selects a trip; UI calls `GET /trips/{id}/observations` for display.
3. UI calls `POST /predict` with method and optional perturbation parameters.
4. API validates the request, loads the trip (including reference route internally), and applies deterministic perturbation/downsampling.
5. Candidate generation and feature extraction run with the persisted feature specification.
6. The selected model scores candidates.
7. Classifier-only returns top candidates; HMM runs Viterbi with network transitions, then stitches.
8. Server-side scorer compares predictions to the reference route and computes metrics.
9. API returns only predicted route geometry, per-point predictions, and aggregate metrics.
10. UI renders layers, legend, and metrics; reference geometry never reaches the browser.

---

## Updates

> Append dated notes here as implementation decisions change, files are added/renamed, or schema/API contracts evolve.

- 2026-11-18 — Initial architecture created from the PRD, tech stack, and 7-hour implementation plan. No code yet.
- 2026-11-18 — Extra: expanded `MODEL_SPECS` to 11 models (added `hist_gradient_boosting`/boosting_hist, `random_forest`/ensemble_forest, `decision_tree`/tree, `gaussian_nb`/probabilistic, `sgd_logistic`/linear_sgd, `adaboost`/boosting_adaptive). `scripts/03_train.py` reran CV (66 rows), overwriting `model_comparison.csv`, `student_test_comparison.csv`, `model_selection.md`, and comparison figures; RandomForest leads CV Macro F1 0.8894. `scripts/09_report.py` now generates the table and per-model analysis for all 11 models. HMM emission checks on student-test: HistGB 0.8603, RF 0.8583, deployed tuned ExtraTrees 0.8562.
- 2026-11-18 — Steps 31–32: added `scripts/09_report.py` (builds `report/report.md` and `report/report.pdf` from CSVs, images, and captured screenshots) plus `tests/test_report.py` and `tests/test_no_leakage.py`. QA verifies `data/` is untracked, runtime code has no hardcoded trip IDs or credential assignments, the artifact feature contract matches `FEATURE_NAMES`, no tracked file exceeds 50 MB, required result assets exist, and reference anomalies remain documented. Report assets: `report/assets/ui_screenshot.png` (real Leaflet UI) and `api_response_terminal.png` (real `/predict` response).
- 2026-11-18 — Step 30: rewrote `README.md` (install, data layout, all pipeline commands, API/UI, endpoints, verification, results, artifacts, integrity, troubleshooting) and added `scripts/02_features.py` to build/refresh the versioned feature cache.
- 2026-11-18 — Steps 28–29: added `pathsnap/folium_maps.py` (`build_prediction_map`, `build_static_preview`, `predict_for_map`) and `scripts/07_folium.py` (selects best-gain/regression/first trips from `hmm_comparison.csv`, avoiding hardcoded IDs). Added `api/e2e.py` (`run_end_to_end`, `E2E_COLUMNS`) and `scripts/06_e2e.py`; `results/e2e_summary.csv` shows 10/10 HTTP 200, finite metrics, max inference 6.50 s. Maps write to `results/folium/` and previews to `results/figures/`.
- 2026-11-18 — Steps 25–27: added the build-free UI (`ui/index.html`, `style.css`, `app.js`, `config.js`): trip list, health badge, Leaflet map with GPS/candidate/predicted-route layers, legend, metrics panel, method selector, debounced noise/sampling sliders, and stale-response protection. Added `docs/deployment.md` for single-machine and two-laptop LAN operation (CORS, `EVAL_DATA_DIR`, `apiBaseUrl`). UI tests are static/Node-syntax smoke tests.
- 2026-11-18 — Steps 22–24: added `pathsnap/pipeline.py` (`PredictionResult`, `effective_track`, `route_coordinates`, `predict_trip`) and the API package: `api/schemas.py` (typed request/response, no reference fields), `api/service.py` (`PredictionService` artifact/trip loading, contract validation, metrics), `api/deps.py` (CORS/log env), `api/main.py` (lifespan, routes, request logging, error handling). Evaluation integrity: reference routes are used only inside `predict_trip` scoring; responses contain `predicted_arcs`, `route_coords`, `point_predictions`, `metrics`, and display `layers`.
- 2026-11-18 — Steps 20–21: added decoder evaluation to `pathsnap/evaluate.py` (`TripInference`, `prepare_trip_inference`, `evaluate_decoders`, `tune_decoder_betas`, `write_decoder_failure_analysis`), decoder plots to `pathsnap/viz.py`, and `scripts/05_hmm.py`. `NetworkDistance` now caches one Dijkstra per source node (worst trip 44 s → 5.9 s). Outputs: `results/decoder_tuning.csv`, `hmm_comparison.csv`, `hmm_failure_analysis.md`, `best_decoder_params.json`, two figures; tuned β=50 persisted to `artifacts/hmm.json` and `config.json`.
- 2026-11-18 — Extra: added `pathsnap/eda_viz.py` (+ `scripts/01_eda_plots.py`) for six static EDA figures from `results/eda_summary.csv`, and `pathsnap/ablation.py` (+ `scripts/08_feature_ablation.py`) for full/leave-one-out/cumulative feature ablation on training folds. `fit_model` accepts a `feature_names` subset and `viz.py` gains `plot_feature_ablation`. Outputs: `results/feature_ablation.csv`, `feature_selection.md`, `selected_features.json`, `figures/feature_ablation.png`, `figures/eda_*.png`.
- 2026-11-18 — Steps 18–19: added `pathsnap/hmm.py` (`HMMConfig`, `DecodeResult`, `NetworkDistance`, `normalize_log_scores`, `gaussian_log_emissions`, `classifier_log_emissions`, `transition_log_probability`, `transition_log_matrix`, `viterbi_log`, `decode_hmm`, `decode_nearest`, `decode_classifier`, `stitch_arc_sequence`). Emissions are normalised per observation; transitions use directed shortest-path distance with a cutoff and explicit invalid penalty; decoding splits at missing candidates/`gap_split_s`, runs log-space Viterbi per segment, and stitches with shortest paths. Added `HMM_GAUSSIAN_SIGMA_M` and `HMM_MAX_PATH_M` config defaults.
- 2026-11-18 — Steps 16–17: added `pathsnap/tune.py` (`suggest_params`, `run_optimisation`, `compare_tuned_vs_untuned`, `fit_final_model`) and `pathsnap/artifacts.py` (`ModelArtifact`, `build_model_artifact`, `save_model_artifact` with joblib compression, `load_model_artifact`, `write_config_json`, `write_json`, `read_json`). `scripts/04_tune.py` writes `optuna_trials.csv`, `best_params.json`, `tuned_vs_untuned.csv`, and exports `artifacts/model.joblib` (15 MB), `config.json`, `hmm.json`. Search space is deliberately size-bounded (depth 8–14, leaves 10–30, trees 100–200); final fit uses the deterministic 300-observations/trip sample. `fit_model` now accepts estimator `params` overrides and records the effective spec.
- 2026-11-18 — Steps 14–15: added `pathsnap/evaluate.py` (`sample_trip_rows`, `stack_sampled`, `cross_validate`, `evaluate_on_test`, `select_best_model`, `write_selection_note`) and `pathsnap/viz.py` (`generate_all_plots`). `scripts/03_train.py` writes `results/model_comparison.csv` (fold + `cv_aggregate` scopes), `results/student_test_comparison.csv`, `results/model_selection.md`, and five figures. Added deterministic per-trip observation caps (`CV_MAX_OBSERVATIONS_PER_TRIP=300`, `TEST_MAX_OBSERVATIONS_PER_TRIP=800`) and a `load_candidates=False` cache path to avoid rebuilding Python candidate objects for training. `ModelResult` now carries `n_observations` and confusion counts.
- 2026-11-18 — Step 13: extended `MODEL_SPECS` with `lightgbm` (family `boosting`, 200 trees, balanced class weights) and `mlp` (family `neural`, standard-scaled 64×32 ReLU, max_iter 150). Registry now covers five distinct families: linear, instance_based, ensemble, boosting, neural.
- 2026-11-18 — Step 12: added `pathsnap/models.py` with `ModelSpec`, `MODEL_SPECS` (logistic_regression/linear, knn/instance_based, extra_trees/ensemble), `TrainedModel`, `ModelResult`, `fit_model`, `evaluate_model`, `observation_top1_accuracy`, and `model_configs`. Pipelines standard-scale linear/KNN inputs; KNN neighbours clamp to training rows; positive-class probabilities feed HMM emissions later.
- 2026-11-18 — Step 11: added `pathsnap/dataset.py` (new module, now part of the file map): `TripDataset`, `DatasetBundle`, `build_trip_dataset`, `assemble_dataset`, `trip_grouped_folds`, `fit_feature_scaler`. Candidate labels are binary by aligned arc; each row carries its route position. Training excludes `partition=test` and trips with coverage below `LOW_COVERAGE_THRESHOLD`; excluded/unalignable/missing-candidate counts are reported. Folds are deterministic and trip-disjoint (seeded shuffle).
- 2026-11-18 — Step 10: added `pathsnap/metrics.py` (`Timer`, `observation_top1_accuracy`, `candidate_macro_f1`, `candidate_precision_recall`, `candidate_log_loss`, `alignment_coverage`, `route_continuity`, `route_precision_recall`, `route_mismatch_fraction`). Route P/R and RMF use `difflib.SequenceMatcher` alignment over route positions and arc lengths; observation metrics accept an alignable mask; empty inputs are zero-safe.
- 2026-11-18 — Step 09: added `pathsnap/cache.py` (`TripArtifacts`, `compute_cache_version`, `build_trip_artifacts`, `save_trip_artifacts`, `load_trip_artifacts`, `load_or_build_trip_artifacts`). Cache layout: `<CACHE_DIR>/<version>/<trip_id>.npz` + `.json` manifest; version = SHA-256 of package version, feature names, candidate radius/top-k, alignment threshold, and UTM override. Added `CACHE_DIR`/`cache_dir` to config (default `.cache/pathsnap`, ignored by git).
- 2026-11-18 — Step 08: added `pathsnap/features.py` with the fixed `FEATURE_NAMES` contract (`distance_m`, `log_distance`, `heading_diff_deg`, `speed_consistency`, `candidate_rank`, `along_fraction`, `arc_length_m`, `time_gap_s`, `from_degree`, `to_degree`) and `FeatureMatrix` carrying observation/arc/rank ids for label attachment. Features are deterministic and finite.
- 2026-11-18 — Step 07: added `pathsnap/candidates.py` (`Candidate`, `CandidateSet`, `generate_candidates`). Per-trip STRtree over projected arc linestrings; base 50 m query, 100 m then 200 m fallback for misses; top-k by metric distance; outputs distance, projection lon/lat, along-fraction, heading difference, rank, and arc length. Miss rate is computed over the supplied alignable mask only.
- 2026-11-18 — Step 06: added `pathsnap/labels.py` (`AlignmentResult`, `align_observations`, `_monotonic_align`). Labels are route positions, not arc ids, so repeated arcs and out-and-back routes resolve correctly; alignment is a non-decreasing DP over the per-observation/per-route-position distance matrix, with unalignable observations marked `-1` and excluded. Added `geometry.point_to_segment_distance_matrix`. Diagnostics: coverage, backward steps, `low_coverage` (<90%).
- 2026-11-18 — Step 05: added `pathsnap/geometry.py` (`Projection`, `utm_crs_for_bbox`, `projection_for_trip`, `bearing_deg`, `distance_m`, `smallest_angle_diff_deg`, `project_point_to_segment`, `point_to_segment_distances`, `polyline_length_m`, `arc_lengths_m`, `build_directed_graph`, `RoutePolyline`, `build_route_polyline`). Route polylines preserve route-position order/repeats and either insert an explicit break or bridge discontinuities via directed shortest path. `pathsnap/eda.py` now imports geometry utilities instead of duplicating them.
- 2026-11-18 — Step 04: added `pathsnap/eda.py` (per-trip UTM projection, point-to-segment distances, component summaries, `trip_summary`, `build_eda_table`, `write_eda_report`, `build_eda_map`) and `scripts/01_eda.py`. EDA table columns are fixed by `EDA_COLUMNS` (25 columns). Generated `results/eda_summary.csv`, `results/eda_report.md`, `results/folium/eda_00000001.html`. Added `scripts/_bootstrap.py` so direct `python scripts/...` runs import the package.
- 2026-11-18 — Step 03: added `pathsnap/io.py` with the `Trip`/`RouteDiagnostics` dataclasses, four-file loaders, bounds/timestamp/coordinate validation (`TripValidationError`), `route_diagnostics`, and sorted trip discovery (`discover_trip_dirs`, `discover_trip_ids`, `iter_trips`). Added `tests/conftest.py` synthetic-trip fixtures. Audit counts reproduced from the real data: 80 train, 10 test, 11 discontinued routes, 10 repeated-arc routes.
- 2026-11-18 — Step 02: added `pathsnap/config.py` as the single source for paths (`Paths`, `get_paths`, `ensure_output_directories`), seeds, candidate/alignment/HMM defaults, and env overrides. `pytest.ini` defines the `slow` marker and `tests/` testpath.
- 2026-11-18 — Step 01 scaffolding: created `pathsnap/`, `api/`, `scripts/`, `ui/`, `tests/`, `artifacts/`, `results/{figures,folium}/`, `report/assets/`; added `requirements.txt`, `.gitignore` (excludes `data/` and local tool state), and placeholder `README.md`. Python 3.14.8 with FastAPI/Starlette, Pydantic v2, Optuna, and Folium installed.
- 2026-11-18 — Data audit against `data/` and `docs/problem/`: 80 train + 10 test trips, disjoint IDs, all four files present, all route/node bounds valid, all timestamps finite and strictly increasing. Found 11/90 routes with 1–3 node discontinuities, 10/90 with repeated arcs (up to 255 repeats), and 2/80 training trips with severe GPS–route mismatch (`00000033`, `00000054`); all 10 student-test trips have ≥99% of points within 50 m of the reference route. Updated label alignment, EDA columns, metrics, and API contracts to report alignment coverage and tolerate anomalous references.
