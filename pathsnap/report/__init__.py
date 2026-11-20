"""Report assembly and rendering.

``build_document`` turns the stored result tables into a :class:`Document`;
the renderers write it as Markdown, PDF, or LaTeX.
"""

from __future__ import annotations

from pathsnap.report.blocks import (
    Authors,
    Block,
    Document,
    Figure,
    Heading,
    Paragraph,
    Table,
)
from pathsnap.report.content import build_document
from pathsnap.report.latex import compile_latex, render_latex
from pathsnap.report.markdown import render_markdown
from pathsnap.report.pdf import render_pdf

__all__ = [
    "Authors",
    "Block",
    "Document",
    "Figure",
    "Heading",
    "Paragraph",
    "Table",
    "build_document",
    "compile_latex",
    "render_latex",
    "render_markdown",
    "render_pdf",
]
