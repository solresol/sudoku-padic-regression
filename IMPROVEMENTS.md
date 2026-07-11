# IMPROVEMENTS

*Analysis as of 2026-07-11.*

This repo explores solving Sudoku as integer p-adic regression (81 coefficients): a LaTeX paper (`paper/sudoku_padic_regression.tex`), a Python solver/experiment harness (`code/padic_sudoku_regression.py`, `code/run_experiments.py`), and an interactive TypeScript/Vite playground deployed as `padic-logic.symmachus.org` (with its own CI workflow in `.github/workflows/padic-logic.yml`). The paper has been through pre-submission polish ("Version I'll send to Uri", Zp-necessity fixes) and the playground recently gained a Sudoku/all-different mode. Current state: healthy git history, but a substantial uncommitted diff (~658 lines across 6 playground files plus two brand-new untracked files `src/lib/lossHistory.ts` / `lossHistory.test.ts`) is sitting in the working tree.

## Bugs & Fixes / Unfinished Work

1. **Commit or discard the dirty playground work.** `padic-logic.symmachus.org/src/{App.tsx,SudokuMode.tsx,sudoku.ts,styles.css}` plus tests are modified, and `lossHistory.ts(.test.ts)` are untracked. This is the "Improve browser p-adic CSP solver" follow-on that never landed. Run the vitest suite, then commit (your own global rule: commit and push after every moderate chunk of work).
2. **Stale `.pyc` files in `paper/`**: `make_experiment_report.cpython-311.pyc` and `padic_linear_regression.cpython-311.pyc` reference source files that no longer exist anywhere in the repo — either the sources were deleted intentionally (delete the .pyc) or they were lost (recover from history/archive).
3. **`output/` vs `outputs/`**: two parallel results directories exist. Consolidate to one (README only documents `outputs/`).

## Housekeeping / Modernization (highest leverage)

- **No Python packaging at all** — no `pyproject.toml`, no `uv.lock` (and thankfully no `requirements.txt`). Initialize with `uv init` in the repo root, `uv add matplotlib` (whatever `run_experiments.py` imports), and check in `pyproject.toml` + `uv.lock`. Then all README commands become `uv run code/padic_sudoku_regression.py ...` instead of `python3 ...`. The stray `.venv/` at root (Jan 2026 vintage) should be removed once uv manages the environment; ensure `.gitignore` covers it.
- **Remove `__pycache__/` from `code/` and `.pyc` files from `paper/`** and gitignore them if tracked.
- **`padic-logic.symmachus.org/node_modules` and `dist/` present in tree** — confirm they are gitignored (the 5 KB .gitignore probably covers this, but `dist/` is easy to miss and the CI deploy workflow should build it fresh).
- **`tmp/` directories** at root and inside the playground: delete or gitignore.
- **`README.txt` → `README.md`**: convert to Markdown so GitHub renders it; the current file is good content in the wrong format. Also mention the playground and paper build (`paper/Makefile`, root `Makefile`) in it.
- **`archive/` (11 entries) and `submission/`**: add a one-line README in each saying what they are frozen snapshots of, so future-you doesn't have to diff them against `code/`.

## Improvements

- **Split `code/padic_sudoku_regression.py` (1494 lines)**: it holds the solver, five search methods (zubarev, greedy, local-best, local-first, local-zubarev), the generator, and the CLI. Break into `solver.py`, `search_methods.py`, `generator.py`, `cli.py` under a small package — this also makes the pieces importable by `run_experiments.py` without path hacks.
- **Share the loss function between Python and TypeScript.** The playground reimplements the p-adic loss in `src/lib/sudoku.ts`; a small cross-language golden-value test fixture (JSON of board → expected loss) would catch drift between the paper's Python results and the demo.
- **The new `lossHistory.ts` suggests loss-curve plotting in the browser** — once landed, link the playground's loss curve display from the paper (commit a7cc3ce already references the interactive demo).

## Testing

- **Python has zero tests.** Priorities: (a) loss = 0 iff valid solved grid, (b) the known puzzle in the README solves under each `--method` with a fixed seed and bounded steps, (c) `generate --clues 30 --seed 42` is deterministic and unique-solution. Put them in `tests/` and run via `uv run pytest` in CI.
- **CI only covers the playground deploy** (`padic-logic.yml`). Add a workflow (or a job) that runs `uv run pytest` and `npm test` (vitest) on push — the vitest suite exists (`App.test.tsx`, `sudoku.test.ts`) but nothing runs it automatically.
- Land the uncommitted `sudoku.test.ts` / `lossHistory.test.ts` additions (see Bugs #1).

## Documentation

- README doesn't mention: the playground, how to deploy it (that lives in `padic-logic.symmachus.org/DEPLOYMENT.md`), the two Makefiles, or the `submission/` contents. Add a short repo map section.
- Document the relationship between `outputs/experiment_results.csv` and the figures actually used in the paper (which script + seed produced each), for reproducibility claims in the paper.

## Security

- No secrets or credentials spotted in the repo root, code, or workflow (deploy workflow should be double-checked that it uses GitHub secrets, not inline tokens — filename only was inspected here).

## Quick Wins

1. `git add` + commit the playground diff and new lossHistory files (after `npm test`).
2. `uv init && uv add <deps>`; commit `pyproject.toml` + `uv.lock`; update README commands to `uv run`.
3. Delete `code/__pycache__/`, stray `paper/*.pyc`, root `.venv/`, and `tmp/`.
4. Rename `README.txt` → `README.md`.
5. Merge `output/` into `outputs/`.
