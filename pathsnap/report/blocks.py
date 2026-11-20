"""Typed document model shared by the Markdown, PDF, and LaTeX renderers.

The report is built once into a :class:`Document`; renderers only read it. There
is no module-level state, so building twice cannot duplicate output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass(frozen=True)
class Heading:
    """A section heading at ``level`` 1-3."""

    level: int
    text: str


@dataclass(frozen=True)
class Paragraph:
    """A body paragraph."""

    text: str


@dataclass(frozen=True)
class Authors:
    """The centred author block shown beneath the title."""

    names: tuple[str, ...]


@dataclass(frozen=True)
class Table:
    """A table with an explicit header row."""

    rows: tuple[tuple[str, ...], ...]
    widths: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Figure:
    """A figure resolved relative to the project root."""

    path: str
    caption: str
    width_in: float = 6.4


Block = Union[Heading, Paragraph, Authors, Table, Figure]


@dataclass
class Document:
    """An ordered sequence of report blocks."""

    blocks: list[Block] = field(default_factory=list)

    def add(self, block: Block) -> None:
        """Append a block."""
        self.blocks.append(block)

    def h1(self, text: str) -> None:
        """Append a top-level heading."""
        self.add(Heading(1, text))

    def h2(self, text: str) -> None:
        """Append a second-level heading."""
        self.add(Heading(2, text))

    def h3(self, text: str) -> None:
        """Append a third-level heading."""
        self.add(Heading(3, text))

    def para(self, text: str) -> None:
        """Append a paragraph."""
        self.add(Paragraph(text))

    def authors(self, names: list[str]) -> None:
        """Append the author block."""
        self.add(Authors(tuple(names)))

    def table(self, rows: list[list[str]], widths: list[float] | None = None) -> None:
        """Append a table; ``widths`` are relative column weights."""
        self.add(
            Table(
                rows=tuple(tuple(str(cell) for cell in row) for row in rows),
                widths=tuple(widths) if widths else None,
            )
        )

    def image(self, path: str, caption: str, width_in: float = 6.4) -> None:
        """Append a figure, referenced relative to the project root."""
        self.add(Figure(path=path, caption=caption, width_in=width_in))
