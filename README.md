# PathSnap

Map-matching and route reconstruction for noisy GPS trajectories on provided
per-trip road networks. PathSnap trains a learned candidate scorer, decodes a
connected route with an HMM/Viterbi model, and serves predictions through a
FastAPI backend plus a build-free Leaflet UI.

## What is included

- **Data pipeline** — validated per-trip loaders, monotonic route-position label
  alignment, STRtree candidate generation, and deterministic feature caching.
- **Eleven model families** — Logistic Regression, SGD logistic, distance-
  weighted KNN, Gaussian Naive Bayes, ExtraTrees, Random Forest, Decision Tree,
  LightGBM, HistGradientBoosting, AdaBoost, and a shallow MLP, compared with
  trip-grouped CV.
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
pathsnap/     reusable data, ML, geometry, matching, artifact and report logic
scripts/      thin numbered experiment entry points
api/          FastAPI app, schemas, service layer, e2e helper
ui/           plain HTML/CSS/JS + Leaflet (no build step)
report/       report.md / report.pdf, report_latex.tex / .pdf, assets
artifacts/    model.joblib, config.json, hmm.json
results/      CSV/JSON tables, figures, Folium maps, analysis notes,
              plus model_metrics/, evaluator_test/, evaluator/,
              decoder_optimization/, live_demo/ and ui/ subdirectories
tests/        pytest suite
docs/         problem briefs and deployment guide
memory-bank/  PRD, architecture, implementation plan, progress
CLAUDE.md     repository conventions for contributors and agents
pytest.ini    test configuration and the `slow` marker
start_ui.ps1  Windows launcher for the API + bundled UI
```

## Data layout

The three supplied extracted folders are included at the owner's request.
Training, student-test, and evaluator trips are under `data/train`, `data/test`,
and `data/evaluator_test`. Supporting PDFs are under `docs/student_materials`.
`docs/provided_files_manifest.json` records all 405 source files and SHA-256 hashes.
Each partition directory contains one folder per trip:

```
data/train/<trip_id>/<trip_id>.track|.nodes|.arcs|.route
data/test/<trip_id>/...
data/evaluator_test/<trip_id>/...
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
python scripts/03_train.py          # 11-model CV comparison + plots
python scripts/04_tune.py           # Optuna tuning + artifacts/model.joblib
python scripts/05_hmm.py            # nearest vs classifier vs classifier+HMM
python scripts/06_e2e.py            # HTTP checks for every evaluation trip
python scripts/07_folium.py         # prediction maps + static previews
python scripts/08_feature_ablation.py  # feature selection/ablation
python scripts/09_report.py         # technical report (Markdown/PDF/LaTeX)
python scripts/10_optimize_decoder.py  # decoder grid search + artifact export
python scripts/11_evaluate_frozen.py   # frozen-artifact evaluator scoring
python scripts/12_metrics_dashboard.py # model metrics for the UI dashboard
python scripts/13_predict_cli.py    # live command-line inference demo
python scripts/14_decoder_comparison.py # decoder comparison at deployed settings
```

`scripts/03_train.py` takes a few minutes because it runs trip-grouped CV for
eleven models. `scripts/04_tune.py` performs 15 Optuna trials and exports the
final artifacts. All randomness uses seeds in `pathsnap/config.py`.

## API + UI

### Live command-line demonstration (PowerShell)

These commands load the saved model and run inference live; no API server or
retraining is needed. Run from the repository root:

```powershell
# List the ten evaluator trips
.\.venv\Scripts\python scripts\13_predict_cli.py --list

# Predict a single trip and print classifier, decoder, and route scores
.\.venv\Scripts\python scripts\13_predict_cli.py --trip-id 00000023

# Run all ten evaluator trips and show aggregate accuracy
.\.venv\Scripts\python scripts\13_predict_cli.py --all

# Demonstrate the effect of GPS noise and sparse observations
.\.venv\Scripts\python scripts\13_predict_cli.py --trip-id 00000023 --noise 10 --sample-seconds 5

# Compare with the classifier before HMM decoding
.\.venv\Scripts\python scripts\13_predict_cli.py --trip-id 00000023 --method classifier
```

Predicted `.route` files and JSON metrics are written to `results/live_demo/`.
Use `--sample-seconds 0` to preserve every input observation. The program prints
the actual score for each selected trip; individual trips can score below the
overall evaluator average.

### Local evaluator dashboard (Windows)

Run `./start_ui.ps1` from PowerShell, then open
`http://127.0.0.1:8000/ui/`. The launcher uses `data/evaluator_test` and serves
the UI and API together. The map has route/GPS/candidate switches, start/end
markers, and a fit-journey control. Map tiles are OpenStreetMap.

The **Model results** tab separates five-fold classifier CV, student-test,
and evaluator-test metrics, including classifier/decoder Macro F1, Top-1,
log loss, and route F1. Regenerate the current model's report with
`python scripts/12_metrics_dashboard.py`; see
[`results/model_metrics/REPORT.md`](results/model_metrics/REPORT.md).

The separate-server setup below remains available for a two-laptop demo.

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
| `results/model_comparison.csv` | Fold-level and aggregate CV metrics for 11 models |
| `results/student_test_comparison.csv` | Student-test metrics per model |
| `results/optuna_trials.csv` | Bayesian optimisation trials |
| `results/tuned_vs_untuned.csv` | Tuned vs untuned comparison |
| `results/hmm_comparison.csv` | Nearest vs classifier vs classifier+HMM |
| `results/e2e_summary.csv` | End-to-end API checks for all evaluation trips |
| `results/best_params.json` | Winning Optuna trial for the classifier |
| `results/best_decoder_params.json` | Baseline decoder settings (beta = 50) |
| `results/decoder_tuning.csv` | Decoder beta grid search |
| `results/eda_report.md` | Data-quality report |
| `results/feature_selection.md` | Feature selection notes |
| `results/model_selection.md` | Model comparison notes |
| `results/hmm_failure_analysis.md` | Failure analysis for the decoder |
| `results/selected_features.json` | Features kept for the final model |
| `results/model_metrics/REPORT.md` | Deployed-model metrics dashboard |
| `results/evaluator_test/REPORT.md` | Frozen-artifact evaluator report |
| `report/report_latex.tex`, `.pdf` | LaTeX source and PDF build |

Figures live in `results/figures/`; interactive maps live in `results/folium/`.

## Artifacts

`artifacts/model.joblib` is a compressed ExtraTrees classifier trained on up to
1,000 aligned observations per training trip. `artifacts/config.json` records
the feature order, model parameters, candidate/alignment settings, HMM settings,
versions, and git commit. `artifacts/hmm.json` records the tuned decoder
parameters (beta = 2 m, backward projection tolerance = 60 m, inferred-route
point refinement enabled).

`scripts/04_tune.py` + `scripts/05_hmm.py` reproduce the **beta = 50 baseline**
(the settings in `results/best_decoder_params.json`), not the committed
artifacts. The promoted artifacts are produced by:

```bash
python scripts/10_optimize_decoder.py
python scripts/11_evaluate_frozen.py
```

Regenerate the report from the stored result tables with
`python scripts/09_report.py` (add `--out-dir DIR` to write elsewhere).

### Optimized decoder and evaluator results

The frozen optimized model achieved **96.84% mean-trip point accuracy** and
**96.72% pooled point accuracy** on all ten trips from `evaluator_test.zip`.
Mean route sequence F1 is **85.94%**. These metrics are different from the
classifier-only candidate Macro F1, which is **87.88%**.

Decoder selection used a 64/16 split of the 80 training trips. Evaluator data
was scored only after model/settings selection. See
[`results/evaluator_test/REPORT.md`](results/evaluator_test/REPORT.md) for the
full comparison, per-trip results, limitations, and reproduction commands.

To serve the evaluator archive after extracting it under `data/evaluator_test`,
set `EVAL_DATA_DIR=data/evaluator_test` before starting the API.

## AI assistance

Team members used AI coding assistants for repository scaffolding, refactoring,
test generation, and report drafting. All experiments, model training and
tuning, decoder selection, and result verification were run by the team on the
provided data; every reported number is produced by a script in `scripts/` and
stored under `results/`. Team members can explain and defend each claim in the
report.

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
