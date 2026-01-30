# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A research playground exploring Sudoku solving via p-adic losses, modular constraints, and Hensel-ish lifting heuristics. The core question: for real Sudoku puzzles, how often can we "Hensel lift" successfully, and do particular primes work best?

**Key finding:** Naive local search (greedy, simulated annealing) with p-adic objectives fails completely - 0% success rate across all primes tested. See RESULTS.md for full analysis.

## Development Commands

Run experiments:
```bash
uv run scripts/run_experiments.py
```

Add dependencies:
```bash
uv add <package>
```

## Project Structure

```
src/padic_sudoku/
  encoding.py    - Power-of-2 digit encoding, constraint group definitions
  padic.py       - p-adic valuations, norms, residuals, loss functions
  puzzle.py      - Puzzle parsing, embedded Euler 50 puzzles
  heuristics.py  - Greedy cell-swap, simulated annealing
  experiment.py  - Experiment runner, result collection
scripts/
  run_experiments.py  - Main entry point for experiments
results/              - CSV output from experiments
figures/              - Generated matplotlib plots
```

## Mathematical Context

- Digits 1-9 encoded as powers of 2: {1, 2, 4, ..., 256}
- Valid row/column/box sums to 511 (= 2^9 - 1)
- 27 constraints: 9 rows + 9 columns + 9 boxes
- p-adic valuation v_p(residual) measures divisibility by prime p
- Required lift level k where p^k > 2304 implies integer solution
- Primes 7 and 73 divide 511, giving them structural properties

## Key Research Questions

1. Which primes maximize success rate / lift level / speed?
2. Why do certain primes fail (hallucinations, local minima, encoding conflicts)?
3. Are best primes stable across heuristics, puzzle difficulties, and initializations?
4. Does combining primes (sequential restart or combined objectives) help?
