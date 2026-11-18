# PathSnap

Map-matching and route reconstruction for noisy GPS trajectories on provided
per-trip road networks. PathSnap trains a learned candidate scorer, decodes a
connected route with an HMM/Viterbi model, and serves predictions through a
FastAPI backend plus a build-free Leaflet UI.

## What is included

- **Data pipeline** — validated per-trip loaders, monotonic route-position label
  alignment, STRtree candidate generation, and deterministic feature caching.
- **Five model families** — Logistic Regression, distance-weighted KNN,
  ExtraTrees, LightGBM, and a shallow MLP, compared with trip-grouped CV.
- **Feature ablation** — leave-one-feature-out and cumulative-group experiments.
- **Bayesian tuning** — Optuna TPE search with recorded trials and a
  tuned-vs-untuned comparison.
- **Route reconstruction** — network-aware emissions/transitions, log-space
  Viterbi, gap splitting, and shortest-path stitching.
- **FastAPI backend** — health, trip listing, observations, and prediction with
  server-side scoring; reference routes never leave the server.
- **Static Leaflet UI** — trip picker, map layers, method selector, noise and
  sampling sliders, and live metrics.
- **Report assets** — EDA figures, model comparison plots, decoder plots,
  Folium maps, and CSV result tables.

## Repository layout

```
pathsnap/     reusable data, ML, geometry, matching, and artifact logic
scripts/      thin numbered experiment entry points
api/          FastAPI app, schemas, service layer, e2e helper
ui/           plain HTML/CSS/JS + Leaflet (no build step)
artifacts/    model.joblib, config.json, hmm.json
results/      CSV tables, figures, Folium maps, analysis notes
tests/        pytest suite
docs/         problem briefs and deployment guide
memory-bank/  PRD, architecture, implementation plan, progress
```

## Data layout

The provided data is **not committed**. Each partition directory contains one
folder per trip:

```
data/train/<trip_id>/<trip_id>.track|.nodes|.arcs|.route
data/test/<trip_id>/...
```

The API reads evaluation trips from `EVAL_DATA_DIR` (default `data/test`).
Evaluator-only trips can be mounted at runtime; no code changes are required.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Offline pipeline

```bash
python scripts/01_eda.py            # inventory, quality report, Folium EDA map
python scripts/01_eda_plots.py      # static EDA figures
python scripts/02_features.py       # build/refresh versioned feature caches
python scripts/03_train.py          # five-model CV comparison + plots
python scripts/04_tune.py           # Optuna tuning + artifacts/model.joblib
python scripts/05_hmm.py            # nearest vs classifier vs classifier+HMM
python scripts/08_feature_ablation.py  # feature selection/ablation
python scripts/07_folium.py         # prediction maps + static previews
```

`scripts/03_train.py` takes a few minutes because it runs trip-grouped CV for
five models. `scripts/04_tune.py` performs 15 Optuna trials and exports the
final artifacts. All randomness uses seeds in `pathsnap/config.py`.

## API + UI

Terminal 1 — API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Terminal 2 — UI:

```bash
python -m http.server 5500 --directory ui
```

Open `http://localhost:5500`. For a two-laptop LAN demo, set
`ui/config.js` to Laptop 1's URL and `CORS_ORIGINS` on the API. Full
instructions are in [`docs/deployment.md`](docs/deployment.md).

### Endpoints

| Method | Path | Purpose |
| :--- | :--- | :--- |
| GET | `/health` | Liveness and model status |
| GET | `/trips` | Evaluation trip metadata only |
| GET | `/trips/{trip_id}/observations` | Raw GPS points for display |
| POST | `/predict` | Predicted route, point predictions, metrics |

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"trip_id":"<trip-id>","method":"hmm","noise_sigma_m":0,"sample_every_s":1}'
```

## Verification

```bash
pytest
pytest --cov=pathsnap --cov=api
python scripts/06_e2e.py       # HTTP checks for every evaluation trip
```

`scripts/06_e2e.py` writes `results/e2e_summary.csv`.

## Results

| File | Contents |
| :--- | :--- |
| `results/eda_summary.csv` | One row per trip, including route anomalies and coverage |
| `results/feature_ablation.csv` | Full / leave-one-out / cumulative feature ablation |
| `results/model_comparison.csv` | Fold-level and aggregate CV metrics for five models |
| `results/student_test_comparison.csv` | Student-test metrics per model |
| `results/optuna_trials.csv` | Bayesian optimisation trials |
| `results/tuned_vs_untuned.csv` | Tuned vs untuned comparison |
| `results/hmm_comparison.csv` | Nearest vs classifier vs classifier+HMM |
| `results/e2e_summary.csv` | End-to-end API checks for all evaluation trips |

Figures live in `results/figures/`; interactive maps live in `results/folium/`.

## Artifacts

`artifacts/model.joblib` is a compressed ExtraTrees classifier trained on the
deterministic 300-observations-per-trip sample. `artifacts/config.json` records
the feature order, model parameters, candidate/alignment settings, HMM settings,
versions, and git commit. `artifacts/hmm.json` records the tuned decoder
parameters (β = 50).

To regenerate:

```bash
python scripts/04_tune.py
python scripts/05_hmm.py
```

## Evaluation integrity

- Evaluator-testing data is never used for training, tuning, feature selection,
  or threshold selection.
- Reference routes and labels stay inside the server-side data/artifact path.
- `/predict` and `/trips` responses contain no reference geometry or label data.
- Trip IDs are resolved through a discovered-trip allowlist; no raw path joining.
- Logs contain request ID, trip ID, method, status, latency, and aggregate
  metrics only.

## Troubleshooting

- **`Required data directory not found`** — place the trip folders under
  `data/` or set `EVAL_DATA_DIR`/`TRAIN_DATA_DIR`.
- **`No evaluator trips found`** — confirm `EVAL_DATA_DIR` points at a directory
  containing `<trip_id>/<trip_id>.track` folders.
- **Artifact feature mismatch** — delete `artifacts/` and rerun
  `scripts/04_tune.py` so the model and `config.json` share one feature spec.
- **Stale feature cache** — remove `.cache/pathsnap/` or change the feature list;
  the cache key includes the feature specification.
- **Slow first prediction** — the HMM builds a directed graph per trip; after
  candidate generation, decoding is typically a few seconds. The worst
  student-test trip decodes in about 6 s on the development machine.
