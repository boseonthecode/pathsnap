"""Repository QA: leakage, secrets, artifact contract, and tracked-file checks."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import joblib
import pytest

from pathsnap import config
from pathsnap.artifacts import read_json
from pathsnap.features import FEATURE_NAMES

RUNTIME_DIRS = ("pathsnap", "api", "ui", "scripts")
ALLOWED_TRIP_IDS: set[str] = set()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=config.PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
    )


def test_data_directory_is_not_tracked() -> None:
    tracked = _git("ls-files", "data").strip()
    assert tracked == ""


def test_gitignore_excludes_data_and_caches() -> None:
    text = (config.PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/" in text
    assert ".cache/" in text
    assert ".env" in text


def test_runtime_code_contains_no_hardcoded_trip_ids() -> None:
    known_ids = {
        "00000000",
        "00000002",
        "00000033",
        "00000052",
        "00000054",
        "00000056",
        "00000065",
        "00000076",
    } - ALLOWED_TRIP_IDS
    offenders: list[str] = []
    for directory in RUNTIME_DIRS:
        for path in (config.PROJECT_ROOT / directory).rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".js", ".html", ".css"}:
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            for trip_id in known_ids:
                if trip_id in content:
                    offenders.append(f"{path.relative_to(config.PROJECT_ROOT)}: {trip_id}")
    assert offenders == []


def test_no_private_keys_or_credential_assignments() -> None:
    patterns = [
        re.compile(r"BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY"),
        re.compile(r"(api[_-]?key|password|secret[_-]?token)\s*=\s*['\"][^'\"]+['\"]", re.I),
    ]
    offenders: list[str] = []
    for directory in RUNTIME_DIRS:
        for path in (config.PROJECT_ROOT / directory).rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".js", ".html", ".css", ".json"}:
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            if any(pattern.search(content) for pattern in patterns):
                offenders.append(str(path.relative_to(config.PROJECT_ROOT)))
    assert offenders == []


def test_artifact_feature_contract_matches_feature_names() -> None:
    payload = read_json(config.PROJECT_ROOT / "artifacts" / "config.json")
    assert tuple(payload["feature_names"]) == FEATURE_NAMES
    bundle = joblib.load(config.PROJECT_ROOT / "artifacts" / "model.joblib")
    assert tuple(bundle["feature_names"]) == FEATURE_NAMES


def test_api_schemas_expose_no_reference_fields() -> None:
    from api import schemas

    forbidden = {
        "route_arcs",
        "reference_route",
        "reference_arcs",
        "ground_truth",
        "labels",
    }

    def collect_fields(model) -> set[str]:
        fields: set[str] = set()
        for name, field in model.model_fields.items():
            fields.add(name)
            annotation = field.annotation
            for candidate in getattr(annotation, "__args__", ()):
                if hasattr(candidate, "model_fields"):
                    fields |= collect_fields(candidate)
        return fields

    assert not (collect_fields(schemas.PredictResponse) & forbidden)
    assert not (collect_fields(schemas.TripsResponse) & forbidden)


def test_no_tracked_file_exceeds_repository_size_limit() -> None:
    files = _git("ls-files").splitlines()
    oversized = []
    for relative in files:
        path = config.PROJECT_ROOT / relative
        if path.is_file() and path.stat().st_size > 50 * 1024 * 1024:
            oversized.append((relative, path.stat().st_size))
    assert oversized == []


def test_required_result_assets_exist() -> None:
    required = [
        "results/eda_summary.csv",
        "results/feature_ablation.csv",
        "results/model_comparison.csv",
        "results/student_test_comparison.csv",
        "results/optuna_trials.csv",
        "results/tuned_vs_untuned.csv",
        "results/hmm_comparison.csv",
        "results/e2e_summary.csv",
        "results/figures/confusion_matrices.png",
        "results/figures/decoder_accuracy_continuity.png",
        "results/folium/00000052_hmm.html",
        "report/report.pdf",
        "artifacts/model.joblib",
        "artifacts/config.json",
        "artifacts/hmm.json",
    ]
    missing = [name for name in required if not (config.PROJECT_ROOT / name).is_file()]
    assert missing == []


def test_known_reference_anomalies_are_documented() -> None:
    report = (config.PROJECT_ROOT / "results" / "eda_report.md").read_text(encoding="utf-8")
    assert "00000033" in report
    assert "00000054" in report
    assert "node discontinuities" in report
    assert "repeated route arcs" in report
