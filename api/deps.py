"""Shared API configuration helpers (environment read centrally)."""

from __future__ import annotations

import os

from pathsnap import config


def cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", config.DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def log_level() -> str:
    return os.getenv("LOG_LEVEL", config.LOG_LEVEL).upper()
