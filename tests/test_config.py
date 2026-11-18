"""Tests for the central configuration module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pathsnap import config


def test_defaults_load_with_project_relative_paths() -> None:
    paths = config.get_paths()

    assert paths.project_root == config.PROJECT_ROOT
    assert paths.train_data_dir == config.PROJECT_ROOT / "data" / "train"
    assert paths.test_data_dir == config.PROJECT_ROOT / "data" / "test"
    assert paths.eval_data_dir == paths.test_data_dir
    assert paths.artifact_dir == config.PROJECT_ROOT / "artifacts"
    assert paths.results_dir == config.PROJECT_ROOT / "results"


def test_environment_overrides_take_effect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    eval_dir = tmp_path / "eval_trips"
    artifact_dir = tmp_path / "artifacts"
    results_dir = tmp_path / "results"
    monkeypatch.setenv("EVAL_DATA_DIR", str(eval_dir))
    monkeypatch.setenv("ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("RESULTS_DIR", str(results_dir))

    paths = config.get_paths()

    assert paths.eval_data_dir == eval_dir
    assert paths.artifact_dir == artifact_dir
    assert paths.results_dir == results_dir


def test_relative_environment_override_resolves_against_project_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVAL_DATA_DIR", "custom/eval")
    paths = config.get_paths()
    assert paths.eval_data_dir == (config.PROJECT_ROOT / "custom" / "eval").resolve()


def test_missing_data_directory_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EVAL_DATA_DIR", str(tmp_path / "does-not-exist"))
    paths = config.get_paths()

    with pytest.raises(FileNotFoundError, match="Required data director"):
        paths.validate_data()


def test_ensure_output_directories_creates_only_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact_dir = tmp_path / "artifacts"
    results_dir = tmp_path / "results"
    monkeypatch.setenv("ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("RESULTS_DIR", str(results_dir))

    config.ensure_output_directories(config.get_paths())

    assert artifact_dir.is_dir()
    assert results_dir.is_dir()
