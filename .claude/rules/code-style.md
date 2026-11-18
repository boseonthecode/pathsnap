# Code Style Guide — PathSnap

> Applies to Python (`pathsnap/`, `scripts/`, `api/`, `tests/`) and the build-free frontend (`ui/`).

---

## 1. General Principles

- Optimise for clarity and reviewability over cleverness.
- One module = one responsibility. If a file starts mixing I/O, modelling, and presentation, split it.
- Prefer small pure functions; push side effects to script/API boundaries.
- Keep functions short enough to test in isolation.
- Be explicit about units: use `_m`, `_s`, `_deg`, `_kmh` suffixes where ambiguity is possible.
- Determinism is a feature: no unseeded randomness, no time-dependent outputs in core logic.

---

## 2. Naming Conventions

### Python
| Element | Convention | Example |
| :--- | :--- | :--- |
| Modules | `snake_case` | `candidate_generation.py` |
| Packages | `snake_case` | `pathsnap` |
| Functions | `snake_case` verbs | `generate_candidates`, `project_to_utm` |
| Variables | `snake_case` nouns | `candidate_frame`, `route_arcs` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_RADIUS_M` |
| Classes / dataclasses | `PascalCase` | `Trip`, `HMMConfig` |
| Private helpers | leading underscore | `_score_transition` |
| Test files | `test_<module>.py` | `test_geometry.py` |
| Test functions | `test_<behavior>` | `test_projection_is_monotonic` |
| Boolean | `is_`, `has_`, `can_` prefix | `has_candidates`, `is_connected` |

### JavaScript
| Element | Convention | Example |
| :--- | :--- | :--- |
| Variables/functions | `camelCase` | `loadTrips`, `noiseSigma` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_ZOOM` |
| Classes | `PascalCase` | `MapController` |
| DOM IDs | `kebab-case` | `trip-select` |
| CSS classes | BEM-like | `metrics__row`, `metrics__row--warning` |
| Files | lowercase | `app.js`, `config.js` |

### Files and folders
- Scripts use phase prefixes: `01_eda.py`, `02_features.py`, etc.
- Generated results use topic names: `model_comparison.csv`, `hmm_comparison.csv`.
- Figures: `cv_macro_f1_comparison.png`, `confusion_lightgbm.png`.

---

## 3. File Organization

- `pathsnap/` contains reusable logic only. No CLI parsing, no `if __name__ == "__main__"` experiment code.
- `scripts/` are thin orchestrators: parse arguments, call `pathsnap`, write results, print a short summary.
- `api/` contains transport concerns: schemas, dependency wiring, routing, logging. Core logic lives in `pathsnap/`.
- `ui/` contains presentation only. It must never reimplement matching logic or embed reference routes.
- `tests/` mirrors the package structure: `tests/test_<module>.py`.
- Keep `__init__.py` files free of heavy imports or side effects.

### Python module layout
1. Module docstring.
2. `from __future__ import annotations` if needed.
3. Standard-library imports.
4. Third-party imports.
5. Local package imports.
6. Constants.
7. Dataclasses/types.
8. Public functions.
9. Private helpers.

### JavaScript module layout
1. File-level comment.
2. Constants/config.
3. Pure helpers.
4. API client functions.
5. Rendering functions.
6. Event wiring/initialisation.

---

## 4. Component Patterns

### Python
- Use dataclasses for validated records (`Trip`, `Candidate`, `HMMConfig`).
- Use Pydantic models only at API boundaries; do not leak them into core ML code.
- Return typed containers (`list[Candidate]`, `np.ndarray`, dataclass) — never unstructured dicts from `pathsnap`.
- Raise specific exceptions (`ValueError`, custom `TripValidationError`) with actionable messages; do not silently coerce invalid data.
- Validate array shapes and ID bounds at every public boundary.
- Accept `pathlib.Path` for filesystem inputs; never build paths via string concatenation.
- Avoid mutable default arguments; use `field(default_factory=...)`.
- Use `numpy` vectorisation in candidate/feature/metric loops when practical, but keep correctness-first.

### API
- Separate request/response schemas in `api/schemas.py`.
- Never return ORM/file objects directly; map explicitly to response models.
- Use dependency injection for loaded artifacts and configuration.
- Keep route handlers thin; delegate to `api/service.py`.

### JavaScript
- Use an IIFE or ES module; do not leak globals except the documented config.
- `const` by default; `let` only when reassigned; never `var`.
- Use `async/await` for network calls; centralise fetch/error handling.
- Debounce slider-driven requests and guard against out-of-order responses.
- Render functions receive data and return DOM updates; avoid hidden shared state.
- Escape or validate any data inserted into the DOM.
- Use Leaflet layer groups and controls; no inline styles except dynamic colours.

### CSS
- BEM-like naming.
- No inline styles except dynamic values (route colours, confidence gradients).
- Keep layout responsive for laptop screens; ensure legend and metric panels remain readable.

---

## 5. Import Order

1. Standard library (`dataclasses`, `pathlib`, `typing`, `json`).
2. Third-party (`numpy`, `pandas`, `sklearn`, `shapely`, `pyproj`, `networkx`, `fastapi`).
3. Local package (`pathsnap.*`, `api.*`).

- One import per line; no wildcard imports.
- Avoid circular imports by keeping shared types in leaf modules (`config.py`, small dataclasses).
- Scripts import from `pathsnap`, never the reverse.

---

## 6. Type Hints and Validation

- Type hints required on public functions, dataclass fields, and API models.
- Use `Sequence`, `Iterable`, and `Mapping` for inputs; concrete types for outputs.
- Use `Literal`/`Enum` for constrained options (e.g., method names).
- Validate external input (API params, file contents) before use.
- Annotate NumPy array shapes in docstrings where relevant, e.g., `(T, 3)`.

---

## 7. Comment Standards

- Write module and public-function docstrings explaining purpose, inputs, outputs, and assumptions.
- Comment non-obvious logic only: HMM recurrences, transition formulas, projection math, monotonic alignment, cache invalidation.
- Explain **why**, not **what**; avoid restating code.
- Keep comments up to date when behaviour changes.
- Do not leave commented-out code or TODO notes without an owner/decision.
- No decorative banners in code.

---

## 8. Formatting and Linting

- Follow PEP 8; 4-space indentation; maximum line length ~100 characters.
- Use blank lines to separate logical blocks; avoid dense nested expressions.
- Name magic numbers via constants in `pathsnap/config.py` (e.g., radii, cutoffs, penalties).
- Keep functions and files free of unused imports and variables.
- Run the project's formatter/linter if present; otherwise keep formatting consistent with surrounding code.
