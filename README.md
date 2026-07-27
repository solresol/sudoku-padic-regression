# Sudoku via integer p-adic regression

This public repository is the reproducibility archive for the paper
*Signed p-adic Residual Encodings of Finite-Domain All-Different Systems with a
Sudoku Case Study*. It contains the manuscript, the reference Python
experiments, the exact archived result sets, and the submission documentation.

## Repository map

- `paper/`: LaTeX source, figures, and compiled paper PDF.
- `code/`: the 81-coefficient Sudoku solver, unique-puzzle generator, CNF
  comparison algorithms, and experiment driver used by the paper.
- `fixtures/`: stable cross-language objective fixtures.
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
`local-first`, `local-zubarev`, and `mihara`. The Zubarev methods also accept
`--beta0`, `--beta1`, and `--beta-schedule`. Add `--moves 8` to print the first
few moves. `mihara` is deliberately a failed comparison: it treats the signed
Sudoku dataframe as samples from one hidden equality, then reports equality
inliers, off-domain coefficients, clue violations, and peer conflicts rather
than claiming that the result is a Sudoku solution.

Generate a unique-solution puzzle with:

```sh
uv run python code/padic_sudoku_regression.py generate --clues 30 --seed 42
```

Swap-based methods preserve row permutations and optimise column-plus-box
conflicts. The three `local-*` methods allow arbitrary non-clue digit edits and
therefore optimise row-plus-column-plus-box conflicts.

## Compare algorithms on CNF

The CNF comparison script accepts DIMACS or uses a small built-in example. Its
Zubarev walk applies Boltzmann-weighted single-bit moves to the actual signed
regression loss. Its Mihara path implements the digitwise outer loop with a
RANSAC-style modulo-p equality fit, then shows why that statistical model is
wrong for forbidden CNF hyperplanes.

```sh
uv run python code/padic_comparison_algorithms.py --seed 0
uv run python code/padic_comparison_algorithms.py --dimacs problem.cnf
```

The separate browser implementation exposes the corresponding exhaustive,
Zubarev, and Mihara comparison paths at
<https://github.com/solresol/padic-logic>.

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

## Companion repositories

- [`solresol/padic-logic`](https://github.com/solresol/padic-logic): public
  TypeScript/React browser application deployed at
  <https://padic-logic.symmachus.org>.
- `solresol/degree-window-feasibility`: private follow-up paper programme for
  polynomial degree-window systems and their tropical shadow.
- `solresol/constraint-zeta`: private exploratory graph-zeta and
  partition-polynomial programme.

The related neighbour-hyperplane landscape code is in the sibling repository
`../padic-landscapes`.
