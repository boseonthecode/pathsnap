# Testing Standards — PathSnap

> Framework: **pytest** (+ `pytest-cov`, FastAPI `TestClient`).
> Goal: every behavioural change has a fast, deterministic test that verifies the step's acceptance criterion.

---

## 1. Test Framework and Layout

- Use `pytest` for all Python tests.
- Mirror the package structure:
  - `tests/test_config.py`
  - `tests/test_io.py`
  - `tests/test_geometry.py`
  - `tests/test_labels.py`
  - `tests/test_candidates.py`
  - `tests/test_features.py`
  - `tests/test_metrics.py`
  - `tests/test_models.py`
  - `tests/test_hmm.py`
  - `tests/test_artifacts.py`
  - `tests/test_api.py`
- Shared fixtures live in `tests/conftest.py`.
- Frontend checks are lightweight HTTP/DOM smoke tests; do not introduce a heavy E2E framework for the MVP.

---

## 2. Test Naming

- File: `test_<module>.py`.
- Function: `test_<behavior>` in plain language.
- Use names that describe the assertion, not the implementation:
  - `test_loader_rejects_out_of_range_arc_ids`
  - `test_alignment_positions_are_non_decreasing`
  - `test_repeated_arc_labels_use_route_position`
  - `test_unalignable_observations_are_excluded_and_counted`
  - `test_disconnected_transition_gets_penalty`
  - `test_predict_response_has_no_reference_route`
- Group related cases with `pytest.mark.parametrize`.
- Keep one clear reason to fail per test where practical.

---

## 3. What Must Be Tested

### Critical correctness
- File loaders: shapes, dtypes, bounds, malformed input rejection.
- Geometry: projection round-trip, known bearings/distances, point-to-segment projection.
- Labels: monotonic route positions, valid arc IDs, no backward jumps, repeated-arc/self-overlap handling, unalignable-point exclusion, and reported alignment coverage.
- Data quality: EDA detects route discontinuities, repeated route arcs, and severe GPS–route mismatch without failing; known anomalous training trips are flagged.
- Candidates: top-k contents, radius behaviour, empty-candidate handling, miss-rate computation.
- Features: stable column order, no NaN/inf, deterministic output.
- Metrics: perfect predictions, known wrong cases, parity with scikit-learn for log loss.
- HMM: toy-graph Viterbi optimum, emission normalisation, invalid/one-way penalties, gap handling, route contiguity.
- Artifacts: save/load round-trip, feature-spec mismatch rejection, fresh-process prediction parity.
- API: success schemas, 404/422 paths, and **absence of ground-truth fields**.
- Determinism: same seed produces identical splits, features, predictions, and optimisation results.

### Data integrity and leakage
- No trip appears in both train and validation folds.
- Student-testing data is never used for fitting.
- Evaluator directory can be mounted without code changes.
- `/predict` and `/trips` responses contain no reference route, labels, or raw evaluator file paths.
- No hardcoded trip IDs or expected routes exist in the codebase (audit test/script).

### Frontend
- Page loads and populates the trip list from a mocked/real API.
- Selecting a trip triggers observation/prediction calls and renders route layers.
- Slider changes produce requests with the expected parameters.
- Errors (API down, 404, 503) render a user-visible message.
- No reference-route geometry is present in the UI or its network payloads.

---

## 4. Test Data and Fixtures

- Use tiny synthetic fixtures for unit tests (2–5 nodes/arcs, 3–5 observations) so failures are easy to reason about.
- Use at least one real trip from `data/test` for integration-style tests where appropriate.
- Never copy or embed evaluator data in tests.
- Cache expensive artifacts per session (scope fixtures) but keep tests independent in behaviour.
- Mark slow tests with `@pytest.mark.slow`; keep the default suite fast (target: under ~1–2 minutes for the core subset).
- Do not rely on network access in tests.

---

## 5. Coverage Expectations

- Core modules (`io`, `geometry`, `labels`, `candidates`, `features`, `metrics`, `hmm`, `artifacts`) should have meaningful branch coverage of public behaviour.
- API route handlers must have happy-path and error-path tests.
- Aim for ≥ 80% line coverage across `pathsnap/` and `api/`; do not chase coverage with trivial assertions.
- Every implementation-plan step's verification test must exist and pass before the step is marked complete.

---

## 6. Running Tests

- Full suite: `pytest`
- With coverage: `pytest --cov=pathsnap --cov=api`
- Fast subset while iterating: `pytest tests/test_geometry.py tests/test_metrics.py`
- API tests: `pytest tests/test_api.py`
- Slow tests only: `pytest -m slow`
- Exclude slow tests: `pytest -m "not slow"`

---

## 7. Assertion and Hygiene Standards

- Assert on behaviour/values, not implementation details.
- Use approximate comparisons for floating point (`pytest.approx` or explicit tolerances).
- Prefer specific exceptions and messages in `pytest.raises`.
- Clean up temporary files/directories with pytest fixtures (`tmp_path`); never write into `data/`.
- Tests must not depend on execution order or shared mutable state.
- No sleeps; use deterministic stubs/mocks for time or network.
- If a test reveals a bug, fix the code — do not loosen the assertion without justification.
