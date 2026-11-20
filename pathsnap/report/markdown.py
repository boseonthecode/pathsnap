"""Markdown renderer for :class:`pathsnap.report.blocks.Document`."""

from __future__ import annotations

from pathlib import Path

from pathsnap.report.blocks import Authors, Document, Figure, Heading, Paragraph, Table


def render_markdown(document: Document, path: Path, *, project_root: Path) -> Path:
    """Write the Markdown report, refusing to emit broken figure links."""
    lines: list[str] = []
    for block in document.blocks:
        if isinstance(block, Heading):
            lines.extend(["#" * block.level + f" {block.text}", ""])
        elif isinstance(block, Paragraph):
            lines.extend([block.text, ""])
        elif isinstance(block, Authors):
            lines.extend(f'<p align="center"><b>{name}</b></p>' for name in block.names)
            lines.append("")
        elif isinstance(block, Table):
            header = block.rows[0]
            lines.append("| " + " | ".join(str(cell) for cell in header) + " |")
            lines.append("| " + " | ".join("---" for _ in header) + " |")
            for row in block.rows[1:]:
                lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
            lines.append("")
        elif isinstance(block, Figure):
            if not (project_root / block.path).is_file():
                raise FileNotFoundError(
                    f"Report figure is missing: {project_root / block.path}"
                )
            lines.extend(
                [
                    f"![{block.caption}](../{block.path})",
                    "",
                    f"*{block.caption}*",
                    "",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
