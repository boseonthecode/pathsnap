"""Verification of the generated technical report and screenshots."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from pathsnap import config

REPORT_DIR = config.PROJECT_ROOT / "report"
REQUIRED_SECTIONS = (
    "Dataset understanding",
    "Model experimentation",
    "Bayesian hyperparameter",
    "HMM and Viterbi",
    "Geographical visualisation",
    "API, UI, and deployment",
    "Evaluation integrity",
)


def test_markdown_report_contains_required_sections() -> None:
    text = (REPORT_DIR / "report.md").read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section.lower() in text.lower()


def test_pdf_exists_and_contains_required_sections() -> None:
    pdf_path = REPORT_DIR / "report.pdf"
    assert pdf_path.is_file()
    assert pdf_path.stat().st_size > 500_000

    reader = PdfReader(str(pdf_path))
    assert len(reader.pages) >= 10
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    for section in REQUIRED_SECTIONS:
        assert section.lower() in text.lower()


def test_report_assets_exist() -> None:
    for name in (
        config.PROJECT_ROOT / "results" / "ui" / "route-explorer.jpg",
        config.PROJECT_ROOT / "results" / "ui" / "model-results.jpg",
        REPORT_DIR / "assets" / "api_response_terminal.png",
    ):
        assert name.is_file(), name
        assert name.stat().st_size > 10_000


def test_report_figures_referenced_by_markdown_exist() -> None:
    text = (REPORT_DIR / "report.md").read_text(encoding="utf-8")
    referenced = [
        line.split("](", 1)[1].rstrip(")")
        for line in text.splitlines()
        if line.startswith("![")
    ]
    assert referenced
    for relative in referenced:
        resolved = (REPORT_DIR / relative).resolve()
        assert resolved.is_file(), relative
