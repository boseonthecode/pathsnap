"""Build the technical report (Markdown, PDF, LaTeX) from stored result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from pathsnap import config
from pathsnap.report import (
    build_document,
    compile_latex,
    render_latex,
    render_markdown,
    render_pdf,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Destination directory (default: <project>/report)",
    )
    parser.add_argument("--no-pdf", action="store_true", help="Skip the ReportLab PDF")
    parser.add_argument("--no-latex", action="store_true", help="Skip the LaTeX output")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = config.get_paths()
    out_dir = args.out_dir or (paths.project_root / "report")
    out_dir.mkdir(parents=True, exist_ok=True)

    document = build_document(paths, out_dir)

    print(
        "Wrote "
        f"{render_markdown(document, out_dir / 'report.md', project_root=paths.project_root)}"
    )
    if not args.no_pdf:
        print(
            "Wrote "
            f"{render_pdf(document, out_dir / 'report.pdf', project_root=paths.project_root)}"
        )
    if not args.no_latex:
        tex_path = render_latex(
            document, out_dir / "report_latex.tex", project_root=paths.project_root
        )
        print(f"Wrote {tex_path}")
        latex_pdf = compile_latex(tex_path)
        if latex_pdf is not None:
            print(f"Wrote {latex_pdf}")


if __name__ == "__main__":
    main()
