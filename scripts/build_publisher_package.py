#!/usr/bin/env python3
"""Build the macro-free source package requested by the journal.

The canonical manuscript remains paper/sudoku_padic_regression.tex.  This
script externalises the TikZ and sudoku-package drawings, removes their package
dependencies and the manuscript's custom commands, and compiles the resulting
publisher-facing source as a verification step.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "sudoku_padic_regression.tex"
OUTPUT_ROOT = ROOT / "output" / "submission"
PACKAGE_DIR = OUTPUT_ROOT / "publisher-ready"
FIGURE_DIR = PACKAGE_DIR / "figures"
BUILD_DIR = ROOT / "tmp" / "publisher-package"
ARCHIVE = OUTPUT_ROOT / "sudoku_padic_regression-publisher-source.zip"


def run(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def compile_figure(name: str, document: str) -> Path:
    source_dir = BUILD_DIR / name
    source_dir.mkdir(parents=True, exist_ok=True)
    tex_path = source_dir / f"{name}.tex"
    tex_path.write_text(document, encoding="utf-8")
    run(
        "latexmk",
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        tex_path.name,
        cwd=source_dir,
    )
    return source_dir / f"{name}.pdf"


def standalone_tikz(picture: str, *, patterns: bool = False) -> str:
    library = "\\usetikzlibrary{patterns.meta}\n" if patterns else ""
    return (
        "\\documentclass[tikz,border=2pt]{standalone}\n"
        "\\usepackage{tikz}\n"
        f"{library}"
        "\\begin{document}\n"
        f"{picture}\n"
        "\\end{document}\n"
    )


def standalone_sudoku(block: str) -> str:
    return (
        "\\documentclass[border=2pt]{standalone}\n"
        "\\usepackage{sudoku}\n"
        "\\begin{document}\n"
        "\\setlength{\\sudokusize}{6.2cm}\n"
        "\\setlength{\\sudokuthinline}{0.8pt}\n"
        "\\setlength{\\sudokuthickline}{2.8pt}\n"
        "\\renewcommand*\\sudokuformat[1]{\\Large\\sffamily #1}\n"
        f"{block}\n"
        "\\end{document}\n"
    )


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {description}; found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    canonical = SOURCE.read_text(encoding="utf-8")

    tikz_pictures = re.findall(
        r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
        canonical,
        flags=re.DOTALL,
    )
    if len(tikz_pictures) != 2:
        raise RuntimeError(f"Expected two TikZ pictures; found {len(tikz_pictures)}")

    sudoku_groups = re.findall(
        r"\{\\papersudokustyle\s*(\\begin\{sudoku-block\}.*?\\end\{sudoku-block\})\}",
        canonical,
        flags=re.DOTALL,
    )
    if len(sudoku_groups) != 2:
        raise RuntimeError(f"Expected two Sudoku drawings; found {len(sudoku_groups)}")

    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    BUILD_DIR.mkdir(parents=True)
    FIGURE_DIR.mkdir(parents=True)

    generated = {
        "list_colouring_example": compile_figure(
            "list_colouring_example", standalone_tikz(tikz_pictures[0])
        ),
        "sudoku_peer_graph": compile_figure(
            "sudoku_peer_graph", standalone_tikz(tikz_pictures[1], patterns=True)
        ),
        "standard_sudoku_puzzle": compile_figure(
            "standard_sudoku_puzzle", standalone_sudoku(sudoku_groups[0])
        ),
        "seed_zero_initialisation": compile_figure(
            "seed_zero_initialisation", standalone_sudoku(sudoku_groups[1])
        ),
    }
    for name, pdf in generated.items():
        shutil.copy2(pdf, FIGURE_DIR / f"{name}.pdf")

    publisher = canonical
    publisher = replace_once(
        publisher,
        "\\usepackage{sudoku}\n\\usepackage{tikz}\n\\usetikzlibrary{patterns.meta}\n",
        "",
        "TikZ/Sudoku package block",
    )
    publisher = replace_once(
        publisher,
        "\\newcommand{\\sudokublank}{\\phantom{0}}\n"
        "\\newcommand{\\papersudokustyle}{%\n"
        "    \\setlength{\\sudokusize}{6.2cm}%\n"
        "    \\setlength{\\sudokuthinline}{0.8pt}%\n"
        "    \\setlength{\\sudokuthickline}{2.8pt}%\n"
        "    \\renewcommand*\\sudokuformat[1]{\\Large\\sffamily ##1}%\n"
        "}\n",
        "",
        "custom Sudoku command block",
    )
    publisher = replace_once(
        publisher,
        "\\renewcommand{\\arraystretch}{0.96}\n",
        "",
        "array-stretch redefinition",
    )

    figure_replacements = (
        "\\includegraphics[width=7cm]{figures/list_colouring_example.pdf}",
        "\\includegraphics[width=0.80\\linewidth]{figures/sudoku_peer_graph.pdf}",
    )
    for picture, replacement in zip(tikz_pictures, figure_replacements):
        publisher = replace_once(publisher, picture, replacement, "inline TikZ picture")

    sudoku_replacements = (
        "\\includegraphics[width=6.2cm]{figures/standard_sudoku_puzzle.pdf}",
        "\\includegraphics[width=6.2cm]{figures/seed_zero_initialisation.pdf}",
    )
    for group, replacement in zip(sudoku_groups, sudoku_replacements):
        wrapped_group = "{\\papersudokustyle\n" + group + "}"
        publisher = replace_once(
            publisher, wrapped_group, replacement, "inline Sudoku drawing"
        )

    forbidden = re.findall(
        r"\\(?:def|gdef|edef|xdef|newcommand|renewcommand|providecommand|DeclareMathOperator)\b",
        publisher,
    )
    if forbidden:
        raise RuntimeError(f"Publisher source still contains macro definitions: {forbidden}")

    expected_packages = {
        "geometry",
        "amsmath",
        "amssymb",
        "amsthm",
        "graphicx",
        "booktabs",
        "algorithm",
        "algpseudocode",
        "hyperref",
    }
    package_lines = re.findall(r"\\usepackage(?:\[[^]]*\])?\{([^}]*)\}", publisher)
    actual_packages = {
        package.strip()
        for package_line in package_lines
        for package in package_line.split(",")
    }
    if actual_packages != expected_packages:
        raise RuntimeError(
            "Unexpected publisher package set: "
            f"expected {sorted(expected_packages)}, found {sorted(actual_packages)}"
        )

    tex_path = PACKAGE_DIR / "sudoku_padic_regression.tex"
    tex_path.write_text(publisher, encoding="utf-8")
    shutil.copy2(ROOT / "paper" / "loss_curve.pdf", PACKAGE_DIR / "loss_curve.pdf")
    shutil.copy2(
        ROOT / "paper" / "figures" / "padic_logic_sudoku_solution.png",
        FIGURE_DIR / "padic_logic_sudoku_solution.png",
    )

    run(
        "latexmk",
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        tex_path.name,
        cwd=PACKAGE_DIR,
    )

    archive_files = [
        tex_path,
        PACKAGE_DIR / "loss_curve.pdf",
        FIGURE_DIR / "list_colouring_example.pdf",
        FIGURE_DIR / "sudoku_peer_graph.pdf",
        FIGURE_DIR / "standard_sudoku_puzzle.pdf",
        FIGURE_DIR / "seed_zero_initialisation.pdf",
        FIGURE_DIR / "padic_logic_sudoku_solution.png",
    ]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in archive_files:
            bundle.write(path, path.relative_to(PACKAGE_DIR))

    with zipfile.ZipFile(ARCHIVE) as bundle:
        bad_member = bundle.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad_member}")

    print(tex_path)
    print(PACKAGE_DIR / "sudoku_padic_regression.pdf")
    print(ARCHIVE)


if __name__ == "__main__":
    main()
