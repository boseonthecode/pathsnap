"""LaTeX renderer and tectonic driver for the technical report."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from pathsnap.report.blocks import Authors, Document, Figure, Heading, Paragraph, Table

LATEX_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=2.2cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage{array}
\usepackage{float}
\usepackage{caption}
\usepackage{enumitem}
\usepackage{needspace}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\usepackage{titlesec}

\captionsetup{font=small,labelfont=bf,skip=5pt}
\setlength{\parskip}{0.45em}
\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.10}
\setlist{nosep,leftmargin=1.2em}
\raggedbottom

\titleformat{\section}{\Large\bfseries}{\thesection}{0.7em}{}
\titleformat{\subsection}{\large\bfseries}{\thesubsection}{0.6em}{}

\setcounter{secnumdepth}{0}

\newcommand{\code}[1]{\texttt{\small #1}}

\begin{document}
"""

CODE_SPAN = re.compile(r"`([^`]+)`")

REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "|": r"\textbar{}",
}

BREAK_TOKENS = (r"\_", "/", ".", "-", ",")


def latex_escape(text: str) -> str:
    r"""Escape LaTeX specials, then render inline `code` spans as \code{}."""
    placeholders: list[str] = []

    def stash(match: re.Match[str]) -> str:
        placeholders.append(match.group(1))
        return f"\x00{len(placeholders) - 1}\x00"

    text = CODE_SPAN.sub(stash, text)
    text = "".join(REPLACEMENTS.get(ch, ch) for ch in text)
    for index, span in enumerate(placeholders):
        inner = "".join(REPLACEMENTS.get(ch, ch) for ch in span)
        # Allow long identifiers and paths to wrap at punctuation instead of
        # running into the margin.
        for token in BREAK_TOKENS:
            inner = inner.replace(token, token + r"\allowbreak{}")
        text = text.replace(f"\x00{index}\x00", rf"\code{{{inner}}}")
    return text


def _latex_cell(value: str) -> str:
    """Escape a table cell and allow weak breaks inside long identifiers."""
    text = latex_escape(str(value))
    for token in BREAK_TOKENS:
        text = text.replace(token, token + r"\allowbreak{}")
    return text


def _latex_table(rows: tuple[tuple[str, ...], ...], widths: tuple[float, ...] | None) -> str:
    if not rows:
        raise ValueError("Cannot render an empty table")
    n_cols = len(rows[0])
    if widths:
        total = sum(widths)
        # A4 with 2.2cm margins leaves 16.6cm of text. Each column also spends
        # 2\tabcolsep (12pt) of padding, which must come out of the content.
        usable = 16.6 - n_cols * 12 / 72.4 * 2.54
        spec = "".join(
            rf">{{\raggedright\arraybackslash}}"
            rf"p{{{w / total * usable:.2f}cm}}"
            for w in widths[:n_cols]
        )
    else:
        spec = "".join(r">{\raggedright\arraybackslash}X" for _ in range(n_cols))
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        rf"\begin{{tabularx}}{{\textwidth}}{{{spec}}}",
        r"\toprule",
        " & ".join(rf"\textbf{{{_latex_cell(c)}}}" for c in rows[0]) + r" \\",
        r"\midrule",
    ]
    for row in rows[1:]:
        lines.append(" & ".join(_latex_cell(c) for c in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    return "\n".join(lines)


def _paragraph_latex(text: str) -> str:
    """Escape a paragraph and bold its structural lead-ins."""
    escaped = latex_escape(text)
    if escaped.startswith("Conclusion."):
        return (
            r"\needspace{5\baselineskip}\textbf{Conclusion.}"
            + escaped[len("Conclusion."):]
        )
    for label in ("Abstract.", "Reproducibility."):
        if escaped.startswith(label):
            return rf"\textbf{{{label}}}" + escaped[len(label):]
    # Lead-in labels such as "A. Model understanding." get a hanging indent so
    # each required answer reads as a distinct block.
    match = re.match(r"^(A|B|C|D)\.\s+(.+?\.)\s", text)
    if match:
        lead = latex_escape(f"{match.group(1)}. {match.group(2)}")
        rest = latex_escape(text[match.end():])
        return rf"\noindent\textbf{{{lead}}} {rest}"
    return escaped


def render_latex(document: Document, path: Path, *, project_root: Path) -> Path:
    """Emit a standalone LaTeX source mirroring the Markdown/PDF content."""
    body: list[str] = []
    first_heading_done = False

    for block in document.blocks:
        if isinstance(block, Heading):
            if block.level == 1 and not first_heading_done:
                body.extend(
                    [
                        r"\begin{center}",
                        rf"{{\LARGE\bfseries {latex_escape(block.text)}}}\\[4pt]",
                        r"\end{center}",
                    ]
                )
                first_heading_done = True
            elif block.level == 1:
                body.extend([rf"\section{{{latex_escape(block.text)}}}", ""])
            elif block.level == 2:
                body.extend([rf"\subsection{{{latex_escape(block.text)}}}", ""])
            else:
                body.extend([rf"\subsubsection{{{latex_escape(block.text)}}}", ""])
        elif isinstance(block, Authors):
            body.append(r"\begin{center}")
            for name in block.names:
                body.append(rf"\textbf{{{latex_escape(name)}}}\\[2pt]")
            body.extend([r"\end{center}", r"\vspace{0.4em}", ""])
        elif isinstance(block, Paragraph):
            body.extend([_paragraph_latex(block.text), ""])
        elif isinstance(block, Table):
            body.append(_latex_table(block.rows, block.widths))
        elif isinstance(block, Figure):
            image_path = (project_root / block.path).resolve()
            if not image_path.is_file():
                continue
            caption = re.sub(r"^Figure\s+\d+\.\s*", "", block.caption)
            # Relative to the report directory so the .tex file stays portable.
            rel = Path(os.path.relpath(image_path, path.parent)).as_posix()
            body.extend(
                [
                    r"\begin{figure}[H]",
                    r"\centering",
                    rf"\includegraphics[width={block.width_in * 2.54:.1f}cm]"
                    rf"{{{rel}}}",
                    rf"\caption{{{latex_escape(caption)}}}",
                    r"\end{figure}",
                    "",
                ]
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        LATEX_PREAMBLE + "\n".join(body) + "\n\\end{document}\n", encoding="utf-8"
    )
    return path


def compile_latex(tex_path: Path, out_name: str = "report_latex") -> Path | None:
    """Compile the LaTeX source with tectonic if it is available."""
    engine = shutil.which("tectonic")
    if engine is None:
        print("tectonic not found; skipping LaTeX PDF build")
        return None
    result = subprocess.run(
        [engine, "--outdir", str(tex_path.parent), str(tex_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise SystemExit("LaTeX compilation failed")
    produced = tex_path.with_suffix(".pdf")
    target = tex_path.parent / f"{out_name}.pdf"
    produced.replace(target)
    return target
