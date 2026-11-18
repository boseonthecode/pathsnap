# PathSnap

Map-matching and route reconstruction for noisy GPS trajectories on provided
per-trip road networks.

> Placeholder README created during repository scaffolding. Full installation,
> training, API, UI, and two-laptop deployment instructions are added in Step 30.

## Layout

- `pathsnap/` — reusable data-processing, ML, and map-matching logic
- `scripts/` — thin numbered experiment entry points
- `api/` — FastAPI backend
- `ui/` — build-free Leaflet interface
- `tests/` — pytest suite
- `artifacts/`, `results/`, `report/` — generated assets and deliverables

## Quick start

```bash
pip install -r requirements.txt
pytest
```
