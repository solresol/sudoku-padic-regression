# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A research playground exploring Sudoku solving via p-adic losses, modular constraints, and Hensel-ish lifting heuristics. The core question: for real Sudoku puzzles, how often can we "Hensel lift" successfully, and do particular primes work best?

Two components:
- **Python backend**: puzzle generation/solving, p-adic objectives, lifting heuristics, experiments
- **JavaScript frontend**: interactive visualization of lifting attempts, prime comparison

## Development Commands

Run Python scripts with `uv run`:
```bash
uv run script.py
```

Add dependencies with:
```bash
uv add <package>
```

## Mathematical Context

- Digits 1-9 are encoded as powers of 2: {1, 2, 4, ..., 256}
- A valid row/column/box sums to 511 (= 2^9 - 1)
- This reduces 729 one-hot variables to 81 cell variables
- 27 constraints: 9 rows + 9 columns + 9 boxes, each summing to 511
- The p-adic objective measures how divisible residuals are by a chosen prime p
- "Hensel lifting" iteratively satisfies constraints mod p^k for increasing k
- Once p^k > 2304 (max possible row sum = 9 * 256), mod-p^k satisfaction implies integer solution

## Key Research Questions

1. Which primes maximize success rate / lift level / speed?
2. Why do certain primes fail (hallucinations, local minima, encoding conflicts)?
3. Are best primes stable across heuristics, puzzle difficulties, and initializations?
4. Does combining primes (sequential restart or combined objectives) help?

## Referenced Libraries

Python: `py-sudoku`, `dokusan`, `exact-cover`, `sudoku-smt-solvers`
JavaScript: `sudoku.js` (robatron), `sudoku-puzzle` (npm)
Datasets: Kaggle 9M puzzles, Norvig top95, Project Euler Problem 96, tdoku benchmarks
