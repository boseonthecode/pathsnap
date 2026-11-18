# Security Rules — PathSnap

> PathSnap is a trusted local demo with **no authentication**, but security still matters: protect ground truth, validate all input, avoid secrets, and keep the API surface minimal.

---

## 1. Threat Model (MVP)

| Concern | Relevance | Mitigation |
| :--- | :--- | :--- |
| Evaluator ground-truth leakage | Critical | Keep reference routes server-side; never include them in API responses or UI state. |
| Hardcoded predictions | Critical | No trip-specific routes, metrics, or special cases; audit the repo. |
| Path traversal via `trip_id` | High | Validate `trip_id` against discovered trip directories; never join raw user input into file paths. |
| Malformed/hostile request payloads | Medium | Pydantic validation; constrained enums and numeric ranges; reject unknown/oversized inputs. |
| Secret leakage | Medium | No secrets in code or repo; environment variables only. |
| Overly permissive CORS | Medium | Limit to the UI origin when possible; widen only for the trusted LAN demo. |
| Log leakage | Medium | Never log ground-truth routes, raw evaluator rows, tokens, or absolute evaluator paths. |
| Dependency risk | Low–Medium | Pin versions; avoid unnecessary packages. |

Authentication and user accounts are explicitly out of scope.

---

## 2. Secrets and Environment Variables

- Never commit `.env`, credentials, tokens, API keys, SSH keys, or private certificates.
- Keep `.env` and local overrides in `.gitignore`.
- Use environment variables for paths and runtime configuration:
  - `EVAL_DATA_DIR`, `ARTIFACT_DIR`, `RESULTS_DIR`, `LOG_LEVEL`, `CORS_ORIGINS`.
- Access environment variables centrally in `pathsnap/config.py` or `api/deps.py`; do not scatter `os.getenv` calls.
- Provide safe defaults for local development; never provide default credentials or secret keys.
- Do not print environment values at startup. If configuration must be logged, log only non-sensitive keys and validated paths.
- The UI must not contain secrets; `ui/config.js` contains only the API base URL.

---

## 3. Ground-Truth Isolation (Mandatory)

- Reference routes (`.route`) are read **only** by training/evaluation code and the server-side scorer.
- `/predict` may compute accuracy against reference data internally, but **must not** return reference geometry, reference arc IDs, or label file contents.
- `/trips` returns identifiers/metadata only; `/trips/{id}/observations` returns raw GPS only.
- Never send reference routes to the UI, not even hidden layers or debug fields.
- Do not expose evaluator data through `/docs`, static file serving, logs, stack traces, or error messages.
- Development/report maps may show reference routes for student-testing trips, but those maps must not be served by the evaluation API.
- Prediction code must not accept a reference route as input.

---

## 4. Input Validation

- Validate all API inputs with Pydantic schemas:
  - `trip_id` must match an existing discovered trip (allowlist), not an arbitrary path.
  - `method` must be one of `nearest`, `classifier`, `hmm`.
  - `noise_sigma_m` must be finite and within `[0, 100]`.
  - `sample_every_s` must be finite and within `[1, 60]`.
  - `seed` must fit a bounded integer range.
- Reject unknown trip IDs with `404`; reject invalid parameters with `422`.
- Validate file contents at load time (shape, dtype, bounds, monotonic timestamps); report route continuity/repeated-arc diagnostics without rejecting otherwise usable trips.
- Never construct filesystem paths from raw request strings; resolve via discovered trip metadata and confirm the resolved path stays inside the configured data directory.
- Cap request sizes and response sizes to prevent accidental resource exhaustion.
- Do not trust client-supplied metrics, routes, or labels.

---

## 5. Authentication and Authorization

- No authentication is implemented for the local demo, by design.
- Do not add user accounts, sessions, cookies, or JWT handling without an approved architecture change.
- If the API is exposed beyond the trusted LAN, add a shared secret or reverse-proxy auth **and update the architecture doc**.
- Do not rely on "no auth" as permission to expose sensitive data; ground-truth isolation still applies.

---

## 6. Logging Rules

### Always log
- Request ID, endpoint, method, trip ID, selected method, HTTP status, latency.
- Artifact load status at startup (model name/version, not file contents).
- Validation failures with a safe reason (no raw payload dumps).

### Never log
- Reference routes, ground-truth arc sequences, or evaluator labels.
- Full raw data file rows.
- Absolute evaluator filesystem paths.
- Secrets, tokens, credentials, or environment dumps.
- Stack traces with ground-truth values in production/demo responses (log server-side, return a generic message).

Use structured logs where practical. Keep `LOG_LEVEL` configurable and default to `INFO`.

---

## 7. CORS and Network Exposure

- Bind the API to `0.0.0.0` only for the LAN demo; bind to `127.0.0.1` for single-machine use.
- Restrict `CORS_ORIGINS` to the UI origin when known (e.g., `http://<laptop-2-ip>:5500`); use `*` only for the trusted local demo and document why.
- Do not enable credentials with wildcard origins.
- Serve only `ui/` statically; never serve `data/`, `artifacts/`, `results/`, `memory-bank/`, or the repo root from the UI server.
- Do not expose the FastAPI `/docs` or API beyond the LAN.
- Ensure error responses do not reveal internal directory structures.

---

## 8. File and Artifact Safety

- Treat all provided data as read-only; never modify files in `data/`.
- Write generated outputs only to `results/`, `artifacts/`, and temporary directories.
- Validate artifact integrity on load (expected keys, feature count/order, model type, version compatibility).
- Reject artifact/feature-spec mismatches with a clear startup error rather than silent misprediction.
- Do not commit datasets, large artifacts, virtual environments, caches, or secrets unless explicitly required and approved.
- Use `.gitignore` to exclude environments, caches, logs, and large generated files as appropriate.

---

## 9. Dependency and Code Hygiene

- Pin dependencies in `requirements.txt`; avoid unnecessary packages.
- Avoid `eval`, `exec`, `pickle` on untrusted input, and shell execution with user-controlled strings.
- Do not deserialize untrusted joblib/pickle artifacts; only load artifacts produced by this project.
- Escape any dynamic content rendered into the DOM in the UI.
- Reject HTML/script injection attempts in trip metadata or error messages.
- Run `pytest` after security-relevant changes; add regression tests for leakage and validation fixes.

---

## 10. Security Checklist Before Submission

- [ ] No `.env`, keys, tokens, or credentials in `git status` or history.
- [ ] No reference-route fields in any API response or UI payload.
- [ ] No hardcoded trip IDs, routes, or metrics.
- [ ] `trip_id` resolved via an allowlist, not raw path joining.
- [ ] CORS configured for the deployment mode.
- [ ] Error responses contain no internal paths or labels.
- [ ] Evaluator data directory untouched and untracked.
- [ ] Tests cover leakage prevention and invalid-input handling.
