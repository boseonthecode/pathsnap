"""Central configuration for PathSnap.

All runtime paths, random seeds, and algorithm defaults live here. Environment
variables are read only in this module so callers never scatter ``os.getenv``
calls across the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data partitions (the raw data is provided read-only and never committed)
# ---------------------------------------------------------------------------
TRAIN_DATA_DIR: Path = PROJECT_ROOT / "data" / "train"
TEST_DATA_DIR: Path = PROJECT_ROOT / "data" / "test"
CACHE_DIR: Path = PROJECT_ROOT / ".cache" / "pathsnap"

# ---------------------------------------------------------------------------
# Randomness — every stochastic component draws from these seeds
# ---------------------------------------------------------------------------
DEFAULT_SEED: int = 42
CV_SEED: int = 42
PERTURBATION_SEED: int = 42
N_SPLITS: int = 5
CV_MAX_OBSERVATIONS_PER_TRIP: int = 300
TRAIN_MAX_OBSERVATIONS_PER_TRIP: int = 1000
TEST_MAX_OBSERVATIONS_PER_TRIP: int = 800

# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------
CANDIDATE_RADIUS_M: float = 50.0
CANDIDATE_MAX_RADIUS_M: float = 200.0
CANDIDATE_TOP_K: int = 8
CANDIDATE_THRESHOLD: float = 0.5
MAX_CANDIDATE_LAYER_POINTS: int = 3000

# ---------------------------------------------------------------------------
# Label alignment
# ---------------------------------------------------------------------------
ALIGN_MAX_DISTANCE_M: float = 50.0
LOW_COVERAGE_THRESHOLD: float = 0.90

# ---------------------------------------------------------------------------
# HMM / Viterbi defaults (the per-run config is materialised in ``hmm.py``)
# ---------------------------------------------------------------------------
HMM_TOP_K: int = CANDIDATE_TOP_K
HMM_RADIUS_M: float = CANDIDATE_RADIUS_M
HMM_MAX_RADIUS_M: float = CANDIDATE_MAX_RADIUS_M
HMM_BETA_M: float = 10.0
HMM_INVALID_PENALTY: float = 50.0
HMM_GAP_SPLIT_S: float = 30.0
HMM_EMISSION: str = "classifier"
HMM_GAUSSIAN_SIGMA_M: float = 20.0
HMM_MAX_PATH_M: float = 2000.0
HMM_BETA_GRID: tuple[float, ...] = (5.0, 10.0, 20.0, 50.0)

# ---------------------------------------------------------------------------
# UTM selection
# ---------------------------------------------------------------------------
# ``None`` means "choose the UTM zone from the trip centroid". Set this only to
# override projection for debugging; production code always projects per trip.
UTM_ZONE_OVERRIDE: int | None = None

# ---------------------------------------------------------------------------
# API and logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = "INFO"
DEFAULT_CORS_ORIGINS: str = "http://localhost:5500"


def _env_path(name: str, default: Path) -> Path:
    """Return an environment-overridden path, resolved against the project root."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


@dataclass(frozen=True)
class Paths:
    """Resolved filesystem layout used by scripts, tests, and the API."""

    project_root: Path
    train_data_dir: Path
    test_data_dir: Path
    eval_data_dir: Path
    artifact_dir: Path
    results_dir: Path
    cache_dir: Path

    def validate_data(self) -> None:
        """Raise a clear error if a required dataset partition is missing."""
        missing = [
            path
            for path in (self.train_data_dir, self.test_data_dir, self.eval_data_dir)
            if not path.is_dir()
        ]
        if missing:
            joined = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"Required data director{'ies' if len(missing) > 1 else 'y'} "
                f"not found: {joined}. Set EVAL_DATA_DIR/TRAIN_DATA_DIR or place "
                "the provided trip folders under data/."
            )


def cors_origins() -> list[str]:
    """CORS allowlist parsed from ``CORS_ORIGINS`` (comma-separated)."""
    raw = os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def log_level() -> str:
    """Logging level name from ``LOG_LEVEL``."""
    return os.getenv("LOG_LEVEL", LOG_LEVEL).upper()


def get_paths() -> Paths:
    """Build the runtime path layout from defaults plus environment overrides.

    Supported environment variables:
    ``TRAIN_DATA_DIR``, ``TEST_DATA_DIR``, ``EVAL_DATA_DIR``,
    ``ARTIFACT_DIR``, ``RESULTS_DIR``.
    """
    return Paths(
        project_root=PROJECT_ROOT,
        train_data_dir=_env_path("TRAIN_DATA_DIR", TRAIN_DATA_DIR),
        test_data_dir=_env_path("TEST_DATA_DIR", TEST_DATA_DIR),
        eval_data_dir=_env_path("EVAL_DATA_DIR", TEST_DATA_DIR),
        artifact_dir=_env_path("ARTIFACT_DIR", PROJECT_ROOT / "artifacts"),
        results_dir=_env_path("RESULTS_DIR", PROJECT_ROOT / "results"),
        cache_dir=_env_path("CACHE_DIR", CACHE_DIR),
    )


def ensure_output_directories(paths: Paths) -> None:
    """Create generated-output directories (never the read-only data partitions)."""
    for path in (paths.artifact_dir, paths.results_dir, paths.cache_dir):
        path.mkdir(parents=True, exist_ok=True)
