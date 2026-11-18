# PathSnap — Technology Stack

> **Principle:** choose the simplest stack that is robust, reproducible, and demo-able within a 7-hour hackathon.
> **Last updated:** 2026-11-18

---

## 1. Recommended Stack at a Glance

| Layer | Choice | Version guidance |
| :--- | :--- | :--- |
| Language | Python | 3.14 available; target ≥ 3.11 |
| Backend / API | FastAPI + Uvicorn | FastAPI ≥ 0.115, Uvicorn ≥ 0.30 |
| Request/response models | Pydantic | v2 |
| ML | scikit-learn, LightGBM | sklearn ≥ 1.5, LightGBM ≥ 4.5 |
| Hyperparameter search | Optuna | ≥ 4.0 |
| Geometry / spatial index | pyproj, shapely | pyproj ≥ 3.7, shapely ≥ 2.1 |
| Graph algorithms | NetworkX | ≥ 3.4 |
| Data handling | NumPy, pandas | current installed versions |
| Artifacts | joblib + JSON + CSV | filesystem only |
| Frontend | Plain HTML + CSS + JavaScript + Leaflet | Leaflet 1.9 via CDN |
| Maps for report | Folium, matplotlib | latest compatible |
| Testing | pytest, pytest-cov | ≥ 8 |
| Deployment | Local machine(s), no containers required | LAN + `uvicorn --host 0.0.0.0` |
| Auth | None (trusted local demo) | — |
| Database | None (file-based artifacts) | — |

---

## 2. Justification by Layer

### 2.1 Language — Python
The dataset tooling, geospatial libraries, ML ecosystem, and the required Folium/API stack are all Python-native. Python 3.14 is already present in the environment.

### 2.2 Backend — FastAPI + Uvicorn
- Native async support and automatic OpenAPI docs (`/docs`) make the API easy to demonstrate and inspect.
- Pydantic v2 request/response models provide typed validation and reject malformed trip IDs/parameters automatically.
- Lightweight enough to run on Laptop 1 with the model loaded in-process; no separate model server needed.
- FastAPI's dependency injection makes it clean to load artifacts once at startup and expose them to route handlers.

### 2.3 ML — scikit-learn + LightGBM
- scikit-learn supplies the required model diversity with consistent APIs: Logistic Regression, KNN, ExtraTrees/Random Forest, MLP, and optional RBF SVM.
- LightGBM (already installed) is the expected accuracy/speed winner for tabular `(observation, candidate)` features and integrates with `predict_proba` for HMM emissions.
- Both expose `n_jobs`, random seeds, and serialisable pipelines, satisfying reproducibility requirements.

### 2.4 Hyperparameter optimisation — Optuna
- Implements Bayesian (TPE) search with pruning, trial history, and visualisations.
- Simple integration with scikit-learn/LightGBM and trip-grouped CV.
- Trials can be exported to CSV/plots for the report, which is a mandatory deliverable.

### 2.5 Geospatial — pyproj + shapely + NetworkX
- `pyproj` provides per-trip UTM projections so distances are in metres, avoiding degree-space errors.
- `shapely.STRtree` is the fastest pure-Python spatial index for candidate arc lookup and projection.
- `networkx` supports directed shortest paths for transition costs and route stitching, including one-way handling.

### 2.6 Frontend — Plain HTML/CSS/JS + Leaflet
- No build step, no Node dependency, and trivial to serve on Laptop 2 with `python -m http.server`.
- Leaflet is a mature mapping library with simple polyline/marker layers, controls, and legends.
- A single-page app with a small JS module is enough for the trip picker, map layers, and metric panel.
- Rejected framework overhead keeps the 7-hour budget focused on the matching pipeline.

### 2.7 Maps for report — Folium + matplotlib
- Folium produces interactive HTML maps required by the brief (GPS observations, road network, reference route, predicted route).
- matplotlib is used for CV F1, Top-1, log-loss, confusion matrices, optimisation history, and timing plots.

### 2.8 Persistence — filesystem only
- Model: `joblib` artifact.
- Configuration: JSON.
- Experiment results: CSV.
- Figures/maps: PNG/HTML.
- Trips are read directly from the data directory.
- No database is needed for a single-user local demo; avoiding SQLite/ORM removes schema and migration work.

### 2.9 Auth — none by design
- The demo runs on a trusted LAN with no user accounts.
- Security still applies: no secrets in the repo, no ground-truth leakage, strict input validation, and constrained CORS.

### 2.10 Deployment — single machine or two laptops on the same LAN
- Laptop 1: `uvicorn api.main:app --host 0.0.0.0 --port 8000`.
- Laptop 2: serve `ui/` statically and set `ui/config.js` to Laptop 1's LAN URL.
- Single-machine fallback: both run on localhost.
- No Docker required; requirements are pinned in `requirements.txt`.

### 2.11 Testing — pytest
- Unit tests for geometry, labels, candidates, metrics, HMM transitions, and artifact loading.
- FastAPI integration tests with `TestClient`.
- Lightweight UI smoke tests via HTTP and DOM/JS checks.
- Frontend logic kept small and deterministic to keep test coverage practical.

---

## 3. Key Libraries and Responsibilities

| Library | Responsibility |
| :--- | :--- |
| `numpy`, `pandas` | Array/table processing, summary statistics |
| `scipy` | Distances, interpolation, statistical helpers |
| `scikit-learn` | Models, CV splitters, preprocessing, metrics |
| `lightgbm` | Gradient-boosted tree candidate scorer |
| `optuna` | Bayesian hyperparameter search |
| `shapely` | Arc linestrings, projections, spatial index |
| `pyproj` | WGS84 → UTM transforms and geodesic distances |
| `networkx` | Directed graph, shortest paths, connectivity |
| `joblib` | Model/pipeline serialisation |
| `fastapi`, `uvicorn`, `pydantic` | API layer and validation |
| `folium` | Interactive report maps |
| `matplotlib` | Static result/EDA plots |
| `pytest`, `pytest-cov` | Automated verification |

---

## 4. Considered and Rejected

| Option | Why rejected |
| :--- | :--- |
| React + Vite + Leaflet | Build tooling and dependency management are unnecessary for a single-page demo under time pressure. |
| Next.js / Nuxt | Server rendering and routing are irrelevant for a local two-laptop demo. |
| Flask | Works, but weaker typed validation and no built-in OpenAPI docs. |
| Django | Far too heavy; brings ORM/admin/auth we explicitly do not need. |
| PostgreSQL / SQLite + ORM | Adds schema/migration overhead; file artifacts are sufficient and easier to inspect. |
| Auth0 / OAuth / JWT accounts | No user accounts are in scope for the local demo. |
| OSMnx live network downloads | Dataset already provides per-trip networks; live downloads risk unreproducibility and network failure. |
| Synthetic trajectory generator from `docs/idea.md` | The official brief provides labelled trips; synthetic data would add scope and diverge from the required student/evaluator partitions. |
| PyTorch GRU/LSTM sequence model | High training/setup cost for marginal benefit in 7 hours; revisit only as a stretch. |
| XGBoost / CatBoost | LightGBM is installed, fast, and sufficient; adding more libraries increases setup risk. |
| Docker Compose | Not needed for two laptops; adds image-build latency and port/config complexity. |
| Valhalla / OSRM map-matching services | External services add deployment risk; the brief expects a custom implementation. |
| Real-time WebSocket streaming | Explicitly out of scope; offline per-trip requests are sufficient. |

---

## 5. Environment and Configuration

### Required environment variables
| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `EVAL_DATA_DIR` | `data/test` | Directory holding trip folders used by the API for evaluation. |
| `ARTIFACT_DIR` | `artifacts` | Location of model/config/HMM artifacts. |
| `RESULTS_DIR` | `results` | Location of experiment outputs. |
| `LOG_LEVEL` | `INFO` | API log verbosity. |
| `CORS_ORIGINS` | `http://localhost:5500` | Comma-separated allowed origins; add Laptop 2's LAN origin for the two-laptop demo. |

### UI configuration
- `ui/config.js` exposes `window.PATHSNAP_CONFIG = { apiBaseUrl: "http://<laptop-1-ip>:8000" }`.
- No secrets or labels are stored in the UI.

### Reproducibility
- Pin dependencies in `requirements.txt`.
- Fix random seeds for splits, models, perturbation, and optimisation.
- Record package versions and git commit in `artifacts/config.json`.
- Store optimisation trials and experiment metrics as CSV.

### Data-quality constraints (from the audit)
- Reference routes are labels, **not** guaranteed node-contiguous or complete: 11/90 trips have 1–3 discontinuities, 10/90 contain repeated arcs, and 2/80 training trips (`00000033`, `00000054`) are severely offset from the GPS track.
- Use monotonic sequence alignment (dynamic programming / constrained progress) for labels; naive nearest-segment projection is not reliably monotonic on self-overlapping routes.
- Track route position, not just `arc_id`, because arcs can repeat.
- Emit labels only for alignable observations and report alignment coverage; do not silently train on mismatched points.
- All ten student-test trips are clean for evaluation (≥ 99% of points within 50 m of the reference route).

---

## 6. Performance Notes

- Per-trip graphs can reach ~307k arcs; build the spatial index once per trip and cache it in-process.
- Cap candidates to top-k (default 5–10) and cap shortest-path search with a distance cutoff.
- Cache directed shortest-path results for repeated candidate pairs during Viterbi.
- Use LightGBM with modest tree counts for fast inference; avoid expensive SVM/KNN on the full dataset.
- Return only route geometry and aggregate metrics from the API; never stream raw reference data.
