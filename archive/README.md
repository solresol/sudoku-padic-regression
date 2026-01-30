# sudoku-padic-regression
Solving Sudoku the hard way with p-adic linear regression

# p-adic Sudoku Lab 🧮🧪
**A research playground for “solving Sudoku the hard way” via p-adic losses, modular constraints, and Hensel-ish lifting heuristics.**

This repo has two halves:

1) **Python**: generate/solve Sudoku puzzles, implement the p-adic/regression-style objectives, run large experiments, and answer the core empirical question:

> **For real Sudoku puzzles, how often can we “Hensel lift” successfully, and do particular primes work best?**

2) **JavaScript (web)**: a small website where people can pick a puzzle + prime(s), watch the lifting attempts step-by-step, and compare primes.

This is not trying to beat world-class Sudoku solvers. It’s trying to measure, visualise, and understand the *weirdness*.

---

## Table of contents
- [Core idea](#core-idea)
- [Mathematical encoding](#mathematical-encoding)
- [What “Hensel lifting” means here](#what-hensel-lifting-means-here)
- [Research questions](#research-questions)
- [Repository layout](#repository-layout)
- [Quickstart](#quickstart)
- [Python: puzzle generation + baseline solvers](#python-puzzle-generation--baseline-solvers)
- [Python: p-adic objectives + lifting heuristics](#python-p-adic-objectives--lifting-heuristics)
- [Running experiments](#running-experiments)
- [Website](#website)
- [API (optional backend)](#api-optional-backend)
- [Data formats](#data-formats)
- [Reproducibility](#reproducibility)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [Licence](#licence)

---

## Core idea

We encode Sudoku digits using **powers of two**, so that a row/column/box containing each digit exactly once has a unique sum.

### Why powers of 2?
For a standard 9×9 Sudoku, map digits to:
\[
\{1,2,4,8,16,32,64,128,256\} = \{2^0,\dots,2^8\}.
\]

Then the “perfect set” sum is:
\[
1 + 2 + \cdots + 256 = 2^9 - 1 = 511.
\]

If every cell in a row is constrained to be one of these powers of 2, then **“row sum = 511” implies the row contains each digit exactly once** (no duplicates, no missing digits). Same for columns and 3×3 boxes.

That collapses classic 729 one-hot variables down to **81 cell variables**.

---

## Mathematical encoding

Let each cell be a coefficient:
\[
\beta_{r,c} \in \{1,2,4,8,16,32,64,128,256\}.
\]

### Structural constraints (27 total)
- **Rows**: for each row \(r\),
  \[
  \sum_{c=1}^9 \beta_{r,c} = 511
  \]
- **Columns**: for each column \(c\),
  \[
  \sum_{r=1}^9 \beta_{r,c} = 511
  \]
- **Boxes**: for each 3×3 box \(b\),
  \[
  \sum_{(r,c)\in b} \beta_{r,c} = 511
  \]

### Clues
If the puzzle says \((r,c)=d\), we force:
\[
\beta_{r,c} = 2^{d-1}.
\]

---

## p-adic “regression” viewpoint

For each constraint \(g\) (a row/col/box), define a residual:
\[
r_g(\beta) = 511 - \sum_{(r,c)\in g}\beta_{r,c}.
\]

For a prime \(p\), define:
- \(v_p(n)\): the exponent of \(p\) in \(n\) (with \(v_p(0)=+\infty\))
- \(|n|_p = p^{-v_p(n)}\) (and \(|0|_p = 0\))

A simple p-adic objective:
\[
L_p(\beta) = \sum_{g\in\text{groups}} |r_g(\beta)|_p
\]
(optionally weighted, or with a tunable base \(q\) as \(|n|_{p,q}=q^{-v_p(n)}\)).

### Binding cells to allowed values (“regularisation”)
We’ll support two modes:

**Hard mode (recommended for clean experiments):**
\[
\beta_{r,c} \in A \;\;(\text{enforced by the search space itself})
\]
where \(A=\{1,2,4,\dots,256\}\).

**Soft mode (for more “p-adic regression” flavour):**
Add a penalty that encourages \(\beta_{r,c}\) to sit on allowed digits:
\[
R_p(\beta) = \lambda \sum_{r,c}\;\; \min_{a\in A} |\beta_{r,c}-a|_p
\]
or (more literal but more expensive):
\[
R_p(\beta) = \lambda \sum_{r,c} \sum_{a\in A} |\beta_{r,c}-a|_p.
\]

---

## What “Hensel lifting” means here

Classical Hensel lifting is about taking a solution mod \(p\) and lifting it to mod \(p^2, p^3,\dots\).

Here, we’re investigating a *computational analogue*:

- Define “constraint \(g\) is satisfied at lift level \(k\)” iff
  \[
  r_g(\beta) \equiv 0 \pmod{p^k}
  \quad\Leftrightarrow\quad
  v_p(r_g(\beta)) \ge k.
  \]

- A “lift attempt” tries to move from level \(k\) to \(k+1\) by modifying \(\beta\) while preserving:
  - clue constraints
  - (optionally) allowed-digit constraints

- Eventually, for fixed \(p\), once \(p^k > 2304\) (since row sums are at most \(9\cdot 256=2304\)), satisfying mod \(p^k\) implies satisfying over the integers — i.e. an actual Sudoku solution.

### Why primes smaller than 256 get weird (and interesting)
With small primes, many *wrong* sums can still be divisible by high powers of \(p\), making them look “close” in the p-adic metric. This creates:
- abundant modular “hallucinations” at low \(k\)
- a genuine question about whether lifting heuristics can climb out of them

We’re explicitly measuring how this plays out in practice.

---

## Research questions

We’ll treat these as measurable, not philosophical:

1) **Prime ranking**  
   For a given puzzle distribution, which primes \(p\) maximise:
   - success rate (find a true solution within budget)
   - median/max lift level reached
   - speed to reach a full integer solution (if reached)

2) **Prime failure modes**  
   For primes that perform badly, is it because:
   - too many mod-\(p\) solutions that don’t lift
   - the landscape has strong local minima
   - certain primes interact badly with the powers-of-two encoding

3) **Heuristic dependence**  
   Are “best primes” stable across:
   - different lift heuristics
   - different puzzle difficulties (clue counts, human difficulty ratings)
   - different initialisation strategies

4) **Multi-prime strategies**  
   Does combining primes help?
   - sequential restart: try \(p_1\), if stuck switch to \(p_2\)
   - combined objective: minimise \(\sum_j L_{p_j}(\beta)\)


## Notes on referenced libraries & datasets (for the “hopefully there are libraries” part)

If you want ready-made building blocks rather than writing everything from scratch, these exist and look useful:

- **Python Sudoku generator/solver**: `py-sudoku` exists on PyPI and advertises generation + solving.  [oai_citation:0‡PyPI](https://pypi.org/project/py-sudoku/?utm_source=chatgpt.com)  
- **Python “human-style” and other Sudoku tooling**: `dokusan` provides Sudoku objects and solvers.  [oai_citation:1‡PyPI](https://pypi.org/project/dokusan/?utm_source=chatgpt.com)  
- **Exact cover / DLX helper**: the `exact-cover` package implements Algorithm X / dancing links (good backbone for a baseline solver).  [oai_citation:2‡PyPI](https://pypi.org/project/exact-cover/?utm_source=chatgpt.com)  
- **SMT-based Sudoku tools**: `sudoku-smt-solvers` exists as a package with solver + benchmarking focus.  [oai_citation:3‡PyPI](https://pypi.org/project/sudoku-smt-solvers/?utm_source=chatgpt.com)  
- **JavaScript generator/solver**: `sudoku.js` (robatron) is a generator/solver library with a demo.  [oai_citation:4‡GitHub](https://github.com/robatron/sudoku.js/?utm_source=chatgpt.com)  
- **JS/NPM option**: `sudoku-puzzle` is an npm package that advertises generation/solving/validation.  [oai_citation:5‡jsDelivr](https://www.jsdelivr.com/package/npm/sudoku-puzzle?utm_source=chatgpt.com)  

For datasets you can benchmark on:

- Kaggle has a large “9 million Sudoku puzzles and solutions” dataset.  [oai_citation:6‡kaggle.com](https://www.kaggle.com/datasets/rohanrao/sudoku?utm_source=chatgpt.com)  
- Peter Norvig’s Sudoku essay is a classic source for solver ideas and benchmark puzzles (e.g., “top95”).  [oai_citation:7‡norvig.com](https://norvig.com/sudoku.html?utm_source=chatgpt.com)  
- Project Euler Problem 96 provides a file of 50 Sudoku puzzles with unique solutions (small, handy, widely used).  [oai_citation:8‡projecteuler.net](https://projecteuler.net/problem%3D96?utm_source=chatgpt.com)  
- The `tdoku` repo describes benchmark datasets including a million sampled minimal puzzles.  [oai_citation:9‡GitHub](https://github.com/t-dillon/tdoku/blob/master/benchmarks/README.md?utm_source=chatgpt.com)
