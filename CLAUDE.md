# CLAUDE.md — AI Agent Instructions for PathSnap

## Project

**PathSnap** — map-matching and route reconstruction for noisy GPS trajectories on provided per-trip road networks. The system trains a learned candidate scorer, decodes a connected route with HMM/Viterbi, and exposes a FastAPI backend plus a build-free Leaflet UI for a two-laptop evaluation demo.

Primary reference documents (read these first every session):
1. `memory-bank/prd.md`
2. `memory-bank/tech-stack.md`
3. `memory-bank/implementation-plan.md`
4. `memory-bank/architecture.md`
5. `memory-bank/progress.md`

---

## Tech Stack

| Layer | Choice |
| :--- | :--- |
| Language | Python ≥ 3.11 (3.14 available) |
| Backend | FastAPI + Uvicorn + Pydantic v2 |
| ML | scikit-learn, LightGBM |
| Tuning | Optuna |
| Geospatial | pyproj, shapely (`STRtree`), NetworkX |
| Data | NumPy, pandas, SciPy |
| Artifacts | joblib + JSON + CSV (no database) |
| Frontend | Plain HTML + CSS + JavaScript + Leaflet (CDN, no build step) |
| Maps/report | Folium, matplotlib |
| Tests | pytest, pytest-cov, FastAPI `TestClient` |
| Deployment | Single machine or two laptops on a LAN; no Docker required |
| Auth | None (trusted local demo) |

---

## File Naming Conventions

- Python modules: `snake_case.py` (e.g., `candidate_generation.py`, `hmm.py`).
- Test files: `tests/test_<module>.py`; test functions: `test_<behavior>`.
- Scripts: numbered phase prefixes, e.g., `scripts/03_train.py`.
- Data files: preserve provided naming — `<trip_id>/<trip_id>.track|.nodes|.arcs|.route`.
- Artifacts: `model.joblib`, `config.json`, `hmm.json`.
- Result tables: `<topic>.csv` in `results/` (e.g., `model_comparison.csv`).
- Figures: descriptive snake_case PNGs under `results/figures/`.
- Frontend: `index.html`, `app.js`, `style.css`, `config.js`.
- Memory/docs: lowercase kebab or fixed names in `memory-bank/`.

---

## Code Style Rules

- Follow PEP 8; use type hints on all public functions and dataclasses.
- Keep modules focused; no file should mix data loading, modelling, and presentation.
- Prefer small pure functions in `pathsnap/`; scripts must stay thin and only orchestrate.
- Use dataclasses (or Pydantic models at API boundaries) for structured records.
- Use `pathlib.Path` for filesystem paths.
- Use explicit random seeds from `pathsnap/config.py`; never rely on global randomness.
- Use log-space math for HMM probabilities; guard against `log(0)`.
- Validate array shapes and ID bounds at module boundaries.
- Docstrings on public modules/functions; comments only where logic is non-obvious (especially HMM recurrences and geometry formulas).
- No hardcoded trip IDs, routes, metrics, or predicted answers.
- Python imports order: standard library → third-party → local package.
- JavaScript: ES modules or one IIFE; `const`/`let` only; no global mutable state beyond the documented config.
- CSS/BEM-like class names (e.g., `metrics__row`); avoid inline styles except dynamic colours.

---

## What to Always Do

1. **Read the memory bank first.** Start every session with `memory-bank/` and this file.
2. Check `memory-bank/progress.md` and continue with the next incomplete step.
3. Orient in `memory-bank/architecture.md` before creating or moving files.
4. Keep training, evaluation, and API inference on the same `pathsnap` package.
5. Use only training data for model fitting and student-testing data for development comparisons; never touch evaluator data.
6. Treat reference routes as labels that may be non-contiguous, incomplete, or repeated; use monotonic alignment, report alignment coverage, and never key labels solely by `arc_id`.
7. Keep reference routes and evaluator labels strictly server-side.
8. Add or update a test for every behavioral change.
9. Run the relevant tests before marking a step complete.
10. Update `memory-bank/progress.md` and `memory-bank/architecture.md` after every completed step.
11. Record experiment results in `results/` as CSV/JSON, not only in prose.
12. Use deterministic seeds and version metadata for reproducibility.

---

## What to Never Do

1. Never use `any`-style untyped blobs at API boundaries; always define typed Pydantic schemas.
2. Never create monolithic files that combine multiple responsibilities.
3. Never expose secrets, tokens, credentials, or environment files.
4. Never log ground-truth routes, evaluator labels, or full raw data rows.
5. Never hardcode predictions or special-case trip IDs for evaluation.
6. Never train, tune, or select features using evaluator-testing data.
7. Never send reference-route geometry to the UI or include it in `/predict` responses.
8. Never commit datasets, large artifacts, virtual environments, or caches unless explicitly required.
9. Never rewrite or "fix" the provided raw data in place.
10. Never bypass the API with a precomputed route in the UI.
11. Never use degree-space Euclidean distance for metric thresholds.
12. Never add a database, auth provider, or build system without an approved architecture change.
13. Never assume reference routes are connected, complete, or unique per arc; use route-position alignment and report coverage.
14. Never train on observations that are unalignable to the reference route as if they were valid labels.

---

## How to Run the Project Locally

### Install
- Create and activate a virtual environment.
- Install dependencies: `pip install -r requirements.txt`.

### Train / evaluate (offline)
- Data preparation: `python scripts/02_features.py`
- Model comparison: `python scripts/03_train.py`
- Bayesian tuning and export: `python scripts/04_tune.py`
- HMM comparison: `python scripts/05_hmm.py`

### Run the API (Laptop 1)
- `uvicorn api.main:app --host 0.0.0.0 --port 8000`
- API docs: `http://<laptop-1-ip>:8000/docs`
- Set `EVAL_DATA_DIR` to the evaluator trip directory when required.

### Run the UI (Laptop 2 or same machine)
- Serve statically: `python -m http.server 5500 --directory ui`
- Open `http://<laptop-2-ip>:5500`
- Set `apiBaseUrl` in `ui/config.js` to Laptop 1's LAN URL.

---

## How to Run Tests

- Full suite: `pytest`
- With coverage: `pytest --cov=pathsnap --cov=api`
- Fast subset while iterating: `pytest tests/test_geometry.py tests/test_metrics.py`
- API tests: `pytest tests/test_api.py`
- Any step is only complete when its plan test passes and the suite remains green.

---

## Step Completion Protocol

After completing any implementation step:

1. Run the step's verification test and the relevant test subset.
2. Update `memory-bank/progress.md`:
   - Move the step to **Completed** with the date and a one-line verification note.
   - Keep exactly one step in **In Progress** if work continues.
3. Update `memory-bank/architecture.md`:
   - Add a dated entry under `## Updates` for new modules, schema changes, API changes, or decisions.
4. Commit only when asked; never commit secrets or evaluator data.
