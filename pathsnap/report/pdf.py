"""ReportLab renderer for :class:`pathsnap.report.blocks.Document`."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from pathsnap.report.blocks import Authors, Document, Figure, Heading, Paragraph as Text, Table as Rows


def _styles() -> Any:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCenter",
            parent=styles["Title"],
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AuthorLine",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=11,
            alignment=TA_CENTER,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["Italic"],
            fontSize=8.5,
            textColor=colors.HexColor("#444444"),
            alignment=TA_CENTER,
            spaceAfter=10,
        )
    )
    return styles


def _footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(7.8 * inch, 0.45 * inch, f"Page {document.page}")
    canvas.drawString(0.7 * inch, 0.45 * inch, "PathSnap technical report")
    canvas.restoreState()


def _pdf_table(block: Rows, styles: Any) -> Table:
    rows = [
        [Paragraph(escape(str(cell)), styles["BodyText"]) for cell in row]
        for row in block.rows
    ]
    if block.widths:
        total = sum(block.widths)
        scale = min(1.0, 7.0 / total)
        col_widths = [width * scale * inch for width in block.widths]
    else:
        col_widths = None
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde5ee")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9aa8b5")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def render_pdf(document: Document, path: Path, *, project_root: Path) -> Path:
    """Write the report as a PDF."""
    styles = _styles()
    authors = ", ".join(
        name.split(" - ")[0]
        for block in document.blocks
        if isinstance(block, Authors)
        for name in block.names
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    template = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="PathSnap Technical Report",
        author=authors,
    )

    story: list[Any] = []
    for block in document.blocks:
        if isinstance(block, Heading):
            style = {1: "TitleCenter", 2: "Heading1", 3: "Heading2"}[block.level]
            story.append(Paragraph(escape(block.text), styles[style]))
            story.append(Spacer(1, {1: 8, 2: 4, 3: 3}[block.level]))
        elif isinstance(block, Text):
            story.append(Paragraph(escape(block.text), styles["BodyText"]))
            story.append(Spacer(1, 6))
        elif isinstance(block, Authors):
            for name in block.names:
                story.append(Paragraph(escape(name), styles["AuthorLine"]))
            story.append(Spacer(1, 14))
        elif isinstance(block, Rows):
            story.append(_pdf_table(block, styles))
            story.append(Spacer(1, 10))
        elif isinstance(block, Figure):
            image_path = project_root / block.path
            if not image_path.is_file():
                continue
            with PILImage.open(image_path) as pil_image:
                width, height = pil_image.size
            width_in = block.width_in
            height_in = width_in * height / width
            if height_in > 7.2:
                height_in = 7.2
                width_in = height_in * width / height
            story.append(
                Image(str(image_path), width=width_in * inch, height=height_in * inch)
            )
            story.append(Paragraph(escape(block.caption), styles["Caption"]))
    template.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return path
