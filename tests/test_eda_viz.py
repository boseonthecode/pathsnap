"""Smoke tests for the static EDA figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pathsnap import config
from pathsnap.eda_viz import generate_eda_plots

EDA_SUMMARY = config.PROJECT_ROOT / "results" / "eda_summary.csv"


@pytest.mark.skipif(not EDA_SUMMARY.is_file(), reason="results/eda_summary.csv not present")
def test_generate_eda_plots_writes_every_figure(tmp_path: Path) -> None:
    frame = pd.read_csv(EDA_SUMMARY, dtype={"trip_id": str})
    written = generate_eda_plots(frame, tmp_path)
    assert written
    for name, path in written.items():
        assert path.is_file(), name
        assert path.stat().st_size > 1_000, name
