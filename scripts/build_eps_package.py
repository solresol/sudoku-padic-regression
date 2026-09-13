#!/usr/bin/env python3
"""Derive EPS figures and source from the existing publisher-source ZIP.

The original ZIP and canonical manuscript are read-only inputs. Requires TeX
Live, Poppler's pdftops, and Ghostscript. Run from any working directory.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "submission"
ORIGINAL = OUTPUT / "sudoku_padic_regression-publisher-source.zip"
WORK = ROOT / "tmp" / "publisher-eps"
PACKAGE = WORK / "publisher-eps"
FIGURES = PACKAGE / "figures"
TEX_NAME = "sudoku_padic_regression.tex"
NAMES = [
    "list_colouring_example", "synthetic_dataset", "sudoku_peer_graph",
    "padic_logic_sudoku_solution", "standard_sudoku_puzzle",
    "seed_zero_initialisation", "loss_curve",
]


def run(*args: str, cwd: Path = WORK) -> None:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{' '.join(args)}\n{result.stdout}\n{result.stderr}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def standalone(name: str, body: str, *, table: bool = False) -> Path:
    # Match the 11pt A4 manuscript's text width before evaluating the table's
    # p{0.27\linewidth} column. A narrow standalone default would change it.
    preamble = (
        "\\documentclass[11pt,border=2pt]{standalone}\n"
        "\\usepackage{amsmath,amssymb,graphicx,booktabs}\n"
        "\\begin{document}\n"
    )
    if table:
        preamble += (
            "\\setlength{\\textwidth}{\\dimexpr210mm-2in\\relax}\n"
            "\\setlength{\\linewidth}{\\textwidth}\n\\small\n"
        )
    path = WORK / f"{name}.tex"
    path.write_text(preamble + body + "\n\\end{document}\n")
    run("pdflatex", "-interaction=nonstopmode", "-halt-on-error", path.name)
    return path.with_suffix(".pdf")


def make_zip(path: Path, files: list[Path], base: Path) -> None:
    temporary = path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as bundle:
        for item in files:
            bundle.write(item, item.relative_to(base))
    with zipfile.ZipFile(temporary) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError(f"ZIP verification failed: {temporary}")
    temporary.replace(path)


def main() -> None:
    for program in ("pdflatex", "latexmk", "epstopdf", "pdftops", "gs"):
        if shutil.which(program) is None:
            raise RuntimeError(f"Missing required program: {program}")
    WORK.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    sources = WORK / "original"
    sources.mkdir(exist_ok=True)
    with zipfile.ZipFile(ORIGINAL) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Original publisher ZIP failed verification")
        for name in bundle.namelist():
            dest = sources / name
            if not dest.resolve().is_relative_to(sources.resolve()):
                raise RuntimeError(f"Unsafe archive path: {name}")
            if not name.endswith("/"):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(bundle.read(name))
    original = (sources / TEX_NAME).read_text()
    blocks = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", original, re.S)
    if len(blocks) != len(NAMES):
        raise RuntimeError(f"Expected seven figures, found {len(blocks)}")
    publisher = original
    records = []
    for number, (name, block) in enumerate(zip(NAMES, blocks), 1):
        eps_name = f"figure_{number}_{name}.eps"
        eps = FIGURES / eps_name
        graphic = re.search(r"\\includegraphics(\[[^]]*\])?\{([^}]+)\}", block)
        if graphic:
            asset_name = graphic.group(2)
            asset = sources / asset_name
            if asset.suffix == ".png":
                # TeX embeds the PNG at its original pixel resolution.
                pdf = standalone(name, f"\\includegraphics[width=\\dimexpr210mm-2in\\relax]{{{asset.as_posix()}}}")
            elif asset.suffix == ".pdf":
                pdf = asset
            else:
                raise RuntimeError(f"Unexpected figure input: {asset}")
            updated = block.replace(asset_name, f"figures/{eps_name}")
        else:
            if number != 2:
                raise RuntimeError(f"Unexpected inline figure: {number}")
            table = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", block, re.S)
            if table is None:
                raise RuntimeError("Figure 2 tabular environment is missing")
            asset_name = f"{TEX_NAME}#fig:list-colouring-regression"
            pdf = standalone(name, table.group(), table=True)
            updated = block.replace(table.group(), f"\\includegraphics{{figures/{eps_name}}}")
        run("pdftops", "-eps", "-level3", "-rasterize", "never", str(pdf), str(eps))
        header = eps.read_bytes()[:4096].decode("latin-1")
        if not header.startswith("%!PS-Adobe-3.0 EPSF-3.0"):
            raise RuntimeError(f"Invalid EPS header: {eps}")
        bbox = re.search(r"^%%BoundingBox: (\d+) (\d+) (\d+) (\d+)$", header, re.M)
        if bbox is None or int(bbox[3]) <= int(bbox[1]) or int(bbox[4]) <= int(bbox[2]):
            raise RuntimeError(f"Invalid EPS bounding box: {eps}")
        # Actually interpret every EPS rather than validating just its header.
        roundtrip = WORK / f"figure_{number}-eps-check.pdf"
        run("gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-dEPSCrop",
            "-dAutoFilterColorImages=false", "-dColorImageFilter=/FlateEncode",
            "-dAutoFilterGrayImages=false", "-dGrayImageFilter=/FlateEncode",
            "-dDownsampleColorImages=false", "-dDownsampleGrayImages=false",
            "-sDEVICE=pdfwrite", f"-sOutputFile={roundtrip}", str(eps))
        publisher = publisher.replace(block, updated, 1)
        records.append({
            "figure": number, "file": eps_name, "source": asset_name,
            "sha256": sha256(eps), "source_pdf": str(pdf.relative_to(ROOT)),
            "check_pdf": str(roundtrip.relative_to(ROOT)),
        })
        print(f"Converted and interpreted Figure {number}: {eps_name}", flush=True)

    references = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", publisher)
    if len(references) != 7 or any(not ref.endswith(".eps") for ref in references):
        raise RuntimeError("Publisher source must reference exactly seven EPS figures")
    (PACKAGE / TEX_NAME).write_text(publisher)
    # TeX Live converts the supplied EPS figures automatically for pdfLaTeX.
    # This preserves the manuscript's existing handling of long hyperlinks.
    run("latexmk", "-pdf", "-g", "-interaction=nonstopmode", "-halt-on-error", TEX_NAME, cwd=PACKAGE)
    log = (PACKAGE / "sudoku_padic_regression.log").read_text()
    if re.search(r"LaTeX Warning: (?:Reference .* undefined|There were undefined references)", log):
        raise RuntimeError("Unresolved references in the EPS manuscript")

    readme = [
        "Signed p-adic Residual Encodings - EPS figures", "",
        "All seven numbered figures, in manuscript order. Captions remain in the manuscript.",
        "Figure 2 is also provided as EPS although it was originally typeset as a table.",
        "Figures 1, 2, 3, 5, 6 and 7 retain vector artwork and embedded fonts.",
        "Figure 4 embeds the original 1865 x 554 pixel screenshot; it is inherently raster.",
        "", "Figure file mapping:",
    ]
    readme.extend(f"  Figure {r['figure']}: {r['file']}" for r in records)
    readme.extend([
        "", f"Derived from: {ORIGINAL.name}", f"Original ZIP SHA-256: {sha256(ORIGINAL)}",
        "", "The companion publisher-eps-source ZIP contains the manuscript with EPS references.",
        "Build with TeX Live: latexmk -pdf sudoku_padic_regression.tex",
        "TeX Live automatically converts the supplied EPS files for pdfLaTeX.",
        "The original publisher ZIP and canonical manuscript have not been modified.", "",
    ])
    (PACKAGE / "README.txt").write_text("\n".join(readme))
    manifest = {"original_zip_sha256": sha256(ORIGINAL), "figures": records}
    (WORK / "verification.json").write_text(json.dumps(manifest, indent=2) + "\n")
    figure_files = [FIGURES / r["file"] for r in records]
    (FIGURES / "README.txt").write_text("\n".join(readme))
    figures_zip = OUTPUT / "sudoku_padic_regression-eps-figures.zip"
    source_zip = OUTPUT / "sudoku_padic_regression-publisher-eps-source.zip"
    make_zip(figures_zip, figure_files + [FIGURES / "README.txt"], FIGURES)
    make_zip(source_zip, [PACKAGE / TEX_NAME, PACKAGE / "README.txt"] + figure_files, PACKAGE)
    final = OUTPUT / "publisher-eps"
    final.mkdir(exist_ok=True)
    for item in figure_files + [PACKAGE / TEX_NAME, PACKAGE / "README.txt", PACKAGE / "sudoku_padic_regression.pdf"]:
        dest = final / item.relative_to(PACKAGE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, dest)
    shutil.copy2(WORK / "verification.json", final / "verification.json")
    print(figures_zip)
    print(source_zip)


if __name__ == "__main__":
    main()
