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
- 2026-11-18 — Step 02: added `pathsnap/config.py` as the single source for paths (`Paths`, `get_paths`, `ensure_output_directories`), seeds, candidate/alignment/HMM defaults, and env overrides. `pytest.ini` defines the `slow` marker and `tests/` testpath.
- 2026-11-18 — Step 01 scaffolding: created `pathsnap/`, `api/`, `scripts/`, `ui/`, `tests/`, `artifacts/`, `results/{figures,folium}/`, `report/assets/`; added `requirements.txt`, `.gitignore` (excludes `data/` and local tool state), and placeholder `README.md`. Python 3.14.8 with FastAPI/Starlette, Pydantic v2, Optuna, and Folium installed.
- 2026-11-18 — Data audit against `data/` and `docs/problem/`: 80 train + 10 test trips, disjoint IDs, all four files present, all route/node bounds valid, all timestamps finite and strictly increasing. Found 11/90 routes with 1–3 node discontinuities, 10/90 with repeated arcs (up to 255 repeats), and 2/80 training trips with severe GPS–route mismatch (`00000033`, `00000054`); all 10 student-test trips have ≥99% of points within 50 m of the reference route. Updated label alignment, EDA columns, metrics, and API contracts to report alignment coverage and tolerate anomalous references.
