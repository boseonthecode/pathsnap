# PathSnap Deployment

PathSnap runs as two independent processes: a FastAPI backend and a static
Leaflet UI. No database, auth provider, or build step is required.

## 1. Prerequisites

- Python ≥ 3.11 with `pip`
- The provided per-trip data folders (`data/train`, `data/test`, or a mounted
  evaluator directory)
- Trained artifacts in `artifacts/` (`model.joblib`, `config.json`, `hmm.json`)

```bash
pip install -r requirements.txt
```

If artifacts are missing, regenerate them:

```bash
python scripts/04_tune.py
```

## 2. Single-machine demo

Terminal 1 — API:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Terminal 2 — UI:

```bash
python -m http.server 5500 --directory ui
```

Open `http://localhost:5500`. `ui/config.js` defaults to
`http://localhost:8000`.

## 3. Two-laptop LAN demo

Laptop 1 (API):

```bash
export EVAL_DATA_DIR=/path/to/evaluator/trips
export CORS_ORIGINS=http://<laptop-2-ip>:5500
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Laptop 2 (UI):

```bash
python -m http.server 5500 --directory ui
```

Edit `ui/config.js`:

```js
window.PATHSNAP_CONFIG = {
  apiBaseUrl: "http://<laptop-1-ip>:8000",
  defaultZoom: 14,
};
```

Open `http://<laptop-2-ip>:5500`.

## 4. Environment variables

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `EVAL_DATA_DIR` | `data/test` | Directory of trip folders served by the API |
| `ARTIFACT_DIR` | `artifacts` | Model/config/HMM artifacts |
| `RESULTS_DIR` | `results` | Experiment outputs |
| `CACHE_DIR` | `.cache/pathsnap` | Versioned feature caches |
| `LOG_LEVEL` | `INFO` | API log verbosity |
| `CORS_ORIGINS` | `http://localhost:5500` | Comma-separated allowed UI origins |

## 5. Verify

```bash
curl http://localhost:8000/health
curl http://localhost:8000/trips
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"trip_id":"<trip-id>","method":"hmm"}'
```

The response contains `predicted_arcs`, `route_coords`, `point_predictions`,
server-side `metrics`, and display `layers`. Reference routes and labels are
never returned.

## 6. Security notes for the demo

- Bind the API to `0.0.0.0` only on the trusted LAN.
- Keep `CORS_ORIGINS` restricted to the UI origin.
- Serve only `ui/` over HTTP; never serve `data/`, `artifacts/`, or the repo
  root from the static server.
- The API resolves `trip_id` through a discovered-trip allowlist, never by
  joining user input into a filesystem path.
- Logs contain request ID, trip ID, method, status, latency, and aggregate
  metrics only — never reference routes or labels.
