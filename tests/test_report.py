"""Verification of the report document model, renderers, and committed output."""

from __future__ import annotations

from pathlib import Path

import pytest

from pathsnap import config
from pathsnap.report import build_document, render_latex, render_markdown
from pathsnap.report.blocks import Authors, Figure, Heading, Paragraph, Table

REPORT_DIR = config.PROJECT_ROOT / "report"
RESULTS_DIR = config.PROJECT_ROOT / "results"

REQUIRED_SECTIONS = (
    "Dataset understanding",
    "Model experimentation",
    "Bayesian hyperparameter",
    "HMM and Viterbi",
    "Geographical visualisation",
    "API, UI, and deployment",
    "Evaluation integrity",
)

REQUIRED_INPUTS = (
    RESULTS_DIR / "model_comparison.csv",
    RESULTS_DIR / "student_test_comparison.csv",
    RESULTS_DIR / "tuned_vs_untuned.csv",
    RESULTS_DIR / "hmm_comparison.csv",
    RESULTS_DIR / "e2e_summary.csv",
    RESULTS_DIR / "eda_summary.csv",
    RESULTS_DIR / "feature_ablation.csv",
    RESULTS_DIR / "best_params.json",
    RESULTS_DIR / "selected_features.json",
    config.PROJECT_ROOT / "artifacts" / "config.json",
    config.PROJECT_ROOT / "artifacts" / "hmm.json",
    config.PROJECT_ROOT / "artifacts" / "model.joblib",
)

requires_results = pytest.mark.skipif(
    any(not path.is_file() for path in REQUIRED_INPUTS),
    reason="report inputs under results/ and artifacts/ are not present",
)


@requires_results
def test_document_has_the_expected_block_structure(tmp_path: Path) -> None:
    document = build_document(config.get_paths(), tmp_path)

    assert document.blocks, "report document is empty"
    assert isinstance(document.blocks[0], Heading)
    assert document.blocks[0].level == 1
    assert sum(isinstance(block, Authors) for block in document.blocks) == 1
    assert sum(isinstance(block, Table) for block in document.blocks) >= 5
    assert sum(isinstance(block, Figure) for block in document.blocks) >= 5
    assert any(
        isinstance(block, Paragraph) and len(block.text) > 200
        for block in document.blocks
    )


@requires_results
def test_document_contains_every_required_section(tmp_path: Path) -> None:
    document = build_document(config.get_paths(), tmp_path)
    headings = [
        block.text.lower() for block in document.blocks if isinstance(block, Heading)
    ]
    joined = "\n".join(headings)
    for section in REQUIRED_SECTIONS:
        assert section.lower() in joined, section


@requires_results
def test_document_is_reproducible(tmp_path: Path) -> None:
    first = build_document(config.get_paths(), tmp_path)
    second = build_document(config.get_paths(), tmp_path)
    assert [type(block) for block in first.blocks] == [
        type(block) for block in second.blocks
    ]
    assert len(first.blocks) == len(second.blocks)


@requires_results
def test_every_figure_resolves_inside_the_project(tmp_path: Path) -> None:
    document = build_document(config.get_paths(), tmp_path)
    figures = [block for block in document.blocks if isinstance(block, Figure)]
    assert figures
    for figure in figures:
        assert (config.PROJECT_ROOT / figure.path).is_file(), figure.path


@requires_results
def test_markdown_renderer_writes_all_captions(tmp_path: Path) -> None:
    document = build_document(config.get_paths(), tmp_path)
    out = render_markdown(document, tmp_path / "report.md", project_root=config.PROJECT_ROOT)
    text = out.read_text(encoding="utf-8")

    for block in document.blocks:
        if isinstance(block, Figure):
            assert block.caption in text, block.caption
        elif isinstance(block, Heading):
            assert f"{'#' * block.level} {block.text}" in text


@requires_results
def test_latex_renderer_produces_a_standalone_document(tmp_path: Path) -> None:
    document = build_document(config.get_paths(), tmp_path)
    out = render_latex(document, tmp_path / "report_latex.tex", project_root=config.PROJECT_ROOT)
    text = out.read_text(encoding="utf-8")

    assert text.startswith("\\documentclass")
    assert text.rstrip().endswith("\\end{document}")
    assert "\\includegraphics" in text


def test_markdown_renderer_rejects_a_missing_figure(tmp_path: Path) -> None:
    from pathsnap.report.blocks import Document

    document = Document()
    document.image("results/figures/does_not_exist.png", "Missing figure")

    with pytest.raises(FileNotFoundError, match="missing"):
        render_markdown(document, tmp_path / "report.md", project_root=config.PROJECT_ROOT)


@pytest.mark.skipif(
    not (REPORT_DIR / "report.md").is_file(), reason="report/report.md is not present"
)
def test_committed_markdown_lists_required_sections() -> None:
    text = (REPORT_DIR / "report.md").read_text(encoding="utf-8")
    headings = [
        line.lstrip("#").strip().lower()
        for line in text.splitlines()
        if line.startswith("#")
    ]
    for section in REQUIRED_SECTIONS:
        assert any(section.lower() in heading for heading in headings), section


@pytest.mark.skipif(
    not (REPORT_DIR / "report.pdf").is_file(), reason="report/report.pdf is not present"
)
def test_committed_pdf_is_non_trivial() -> None:
    from pypdf import PdfReader

    pdf_path = REPORT_DIR / "report.pdf"
    assert pdf_path.stat().st_size > 100_000

    reader = PdfReader(str(pdf_path))
    assert len(reader.pages) >= 5
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    for section in REQUIRED_SECTIONS:
        assert section.lower() in text.lower(), section


def test_committed_report_assets_exist() -> None:
    for path in (
        config.PROJECT_ROOT / "results" / "ui" / "route-explorer.jpg",
        config.PROJECT_ROOT / "results" / "ui" / "model-results.jpg",
        REPORT_DIR / "assets" / "api_response_terminal.png",
    ):
        if not path.is_file():
            pytest.skip(f"missing report asset: {path}")
        assert path.stat().st_size > 1_000, path
