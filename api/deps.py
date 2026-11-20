"""Backwards-compatible aliases for the settings resolved in ``pathsnap.config``."""

from __future__ import annotations

from pathsnap.config import cors_origins, log_level

__all__ = ["cors_origins", "log_level"]
