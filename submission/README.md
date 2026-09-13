# Submission materials

This directory contains the human-readable material used to prepare and check a
paper submission. It is not a second copy of the live paper or implementation.

- `reviewer-feedback/`: original referee reports received after submission,
  organized by report date.
- `source_package_README.md`: map and reproduction instructions for the source
  archive.
- `huggingface_dataset_README.md`: documentation bundled with the experiment
  dataset.

Generated archives are written under `output/submission/` and
`output/huggingface/`; those packages are ignored because they can be rebuilt
from the tracked sources and experiment records.

## Publisher source and EPS figures

Run `make publisher-package` to build the publisher source ZIP from the canonical
manuscript. It externalises the TikZ and Sudoku drawings, removes custom macros,
and verifies the resulting manuscript with pdfLaTeX. TeX Live, including
`latexmk`, `standalone`, TikZ and the `sudoku` package, is required.

Run `make publisher-eps-package` to convert all seven numbered figures in that
existing ZIP to EPS. This also creates a companion manuscript source ZIP with
EPS figure references. It requires Poppler's `pdftops` and Ghostscript in
addition to TeX Live. Each EPS is interpreted with Ghostscript, and the companion
manuscript is compiled before the ZIP files are written. The six drawn figures
retain vector artwork; the website screenshot retains its original pixels.

The EPS builder reads the existing publisher ZIP without changing it. Keep that
ZIP if you need to reproduce figures for the version already sent to the
publisher. Running `make publisher-package` again regenerates it from the current
canonical source. The EPS output records the input ZIP's SHA-256 and a numbered
figure map in its README.

Generated packages belong in `output/submission/`, outside Git. Commit the
manuscript, original assets and build scripts so the packages can be rebuilt.
