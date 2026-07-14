# Sudoku via integer p-adic regression

This repository contains the paper, reference Python experiments, and interactive
browser implementation for encoding finite-domain constraint problems as signed
integer p-adic regression data.

## Repository map

- `paper/`: LaTeX source, figures, and compiled paper PDF.
- `code/`: the 81-coefficient Sudoku solver, unique-puzzle generator, and
  experiment driver.
- `padic-logic.symmachus.org/`: the TypeScript/React CSP and Sudoku compiler at
  <https://padic-logic.symmachus.org> (Sudoku: `#sudoku`).
- `tests/`: Python regression tests.
- `submission/`: cover letter, reviewer notes, and source-package documentation.
- `archive/`: frozen earlier experiments and their separate environment.
- `outputs/`: ignored working directories produced by experiment runs.
- `output/`: packaged deliverables such as the Kindle PDF and submission archives.

The singular and plural output directories are deliberately separate: `outputs/`
contains reproducible intermediate data, while `output/` contains files prepared
for distribution.

## Python setup and tests

The live Python code targets Python 3.11 and uses `uv` to lock the plotting and
test dependencies.

```sh
uv sync
uv run pytest
```

The solver itself uses only the standard library. Matplotlib is needed when the
experiment driver writes the loss-curve figures.

## Solve or generate a puzzle

Puzzle strings contain 81 digits, with `0` or `.` for blanks.

```sh
uv run python code/padic_sudoku_regression.py solve \
  --puzzle 530070000600195000098000060800060003400803001700020006060000280000419005000080079 \
  --seed 0 --max-steps 60000 --restarts 15
```

Available methods are `stepwise`, `greedy`, `zubarev`, `local-best`,
`local-first`, and `local-zubarev`. The Zubarev methods also accept `--beta0`,
`--beta1`, and `--beta-schedule`. Add `--moves 8` to print the first few moves.

Generate a unique-solution puzzle with:

```sh
uv run python code/padic_sudoku_regression.py generate --clues 30 --seed 42
```

Swap-based methods preserve row permutations and optimise column-plus-box
conflicts. The three `local-*` methods allow arbitrary non-clue digit edits and
therefore optimise row-plus-column-plus-box conflicts.

## Reproduce experiments

The quick experiment command uses fast clue carving rather than uniqueness
checking:

```sh
uv run python code/run_experiments.py \
  --outdir outputs/local --seed 123 --n 3 --clues 36,30,26
```

The exact commands and directories used for the paper are recorded in
`submission/source_package_README.md`. The archived historical experiments have
their own instructions and lockfile in `archive/`.

## Build the paper

The root Makefile is the canonical build entry point and requires `latexmk` plus
the LaTeX packages used by the manuscript.

```sh
make paper     # paper/sudoku_padic_regression.pdf
make site      # copy the current PDF into site/
make kindle    # compact PDF in tmp/pdfs/
```

## Test and build the browser app

```sh
cd padic-logic.symmachus.org
npm ci
npm test
npm run build
```

The GitHub Actions workflow runs the browser tests and production build, then
deploys pushes to `main`. It also runs the root Python test suite when the Python
implementation, fixtures, or dependency metadata changes.

The related neighbour-hyperplane landscape code is in the sibling repository
`../padic-landscapes`.
