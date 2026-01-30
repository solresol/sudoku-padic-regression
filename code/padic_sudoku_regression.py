
#!/usr/bin/env python3
"""
padic_sudoku_regression.py

Sudoku as an integer-valued p-adic regression problem with positive "digit snapping"
regularisation and negative "all-different" regularisation.

This is intentionally not the standard 729-variable one-hot CSP encoding. Each cell is a
single integer coefficient x_{rc} ∈ {1,…,9}.

Mathematically (see paper/), we work over Z_p with a prime p > 9 (default p=11), so that for
digits a,b∈{1,…,9} we have:
    |a-b|_p = 0  iff a=b
    |a-b|_p = 1  iff a≠b
because 0 < |a-b| < p implies p ∤ (a-b).

That makes the p-adic norm of a difference behave like an indicator of inequality.

The solver is a heuristic "stepwise regression" / coordinate descent:
- initialise each row as a permutation of 1..9 consistent with clues (strong row snapping),
- repeatedly swap two non-clue entries within a row to reduce the regression loss
  (equivalently, reduce column+box conflict pairs).

This file also contains a generator for random *unique-solution* puzzles (moderate clue counts),
using a simple backtracking uniqueness checker.

No external libraries required.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Dict

Digits = List[int]
Grid = List[int]  # length 81, row-major, 0 = blank


# -------------------------
# p-adic utilities
# -------------------------

def v_p(n: int, p: int) -> int:
    """p-adic valuation v_p(n) for integer n (v_p(0)=+inf is not returned here)."""
    n = abs(n)
    if n == 0:
        raise ValueError("v_p(0) is infinite; call p_adic_norm instead.")
    k = 0
    while n % p == 0:
        n //= p
        k += 1
    return k


def p_adic_norm(n: int, p: int) -> float:
    """p-adic absolute value |n|_p on integers (|0|_p=0)."""
    if n == 0:
        return 0.0
    return p ** (-v_p(n, p))


def digit_snapping_penalty(x: int, p: int, allowed: Iterable[int] = range(1, 10)) -> float:
    """
    R_p(x) = Σ_{k in allowed} |x-k|_p
    For p>max|k-k'| and x in Z, this is minimised by x equal to one of the allowed digits.
    """
    return sum(p_adic_norm(x - k, p) for k in allowed)


# -------------------------
# Sudoku indexing helpers
# -------------------------

def rc_to_i(r: int, c: int) -> int:
    return 9 * r + c


def i_to_rc(i: int) -> Tuple[int, int]:
    return divmod(i, 9)


def box_index(r: int, c: int) -> int:
    return (r // 3) * 3 + (c // 3)


def unit_cells_rows() -> List[List[int]]:
    return [[rc_to_i(r, c) for c in range(9)] for r in range(9)]


def unit_cells_cols() -> List[List[int]]:
    return [[rc_to_i(r, c) for r in range(9)] for c in range(9)]


def unit_cells_boxes() -> List[List[int]]:
    units = []
    for br in range(3):
        for bc in range(3):
            cells = []
            for dr in range(3):
                for dc in range(3):
                    r = 3 * br + dr
                    c = 3 * bc + dc
                    cells.append(rc_to_i(r, c))
            units.append(cells)
    return units


ROWS = unit_cells_rows()
COLS = unit_cells_cols()
BOXS = unit_cells_boxes()


# -------------------------
# Validation / printing
# -------------------------

def parse_puzzle(s: str) -> Grid:
    s = s.strip().replace("\n", "").replace(" ", "")
    if len(s) != 81:
        raise ValueError("Puzzle must be 81 characters (digits, . or 0).")
    grid = []
    for ch in s:
        if ch in ".0":
            grid.append(0)
        elif ch.isdigit():
            d = int(ch)
            if not (0 <= d <= 9):
                raise ValueError("Digits must be 0..9.")
            grid.append(d)
        else:
            raise ValueError(f"Invalid character: {ch}")
    return grid


def grid_to_string(grid: Grid) -> str:
    return "".join(str(d) if d != 0 else "." for d in grid)


def pretty(grid: Grid) -> str:
    lines = []
    for r in range(9):
        row = []
        for c in range(9):
            v = grid[rc_to_i(r, c)]
            row.append(str(v) if v else ".")
            if c in (2, 5):
                row.append("|")
        lines.append(" ".join(row))
        if r in (2, 5):
            lines.append("-" * 21)
    return "\n".join(lines)


def is_valid_complete(grid: Grid) -> bool:
    """Check if grid is a complete valid Sudoku solution."""
    if any(v not in range(1, 10) for v in grid):
        return False
    for unit in ROWS + COLS + BOXS:
        vals = [grid[i] for i in unit]
        if sorted(vals) != list(range(1, 10)):
            return False
    return True


def respects_clues(solution: Grid, puzzle: Grid) -> bool:
    for i, g in enumerate(puzzle):
        if g != 0 and solution[i] != g:
            return False
    return True


# -------------------------
# Puzzle generation
# -------------------------

def random_solved_grid(rng: random.Random) -> Grid:
    """Generate a random solved Sudoku grid by shuffling a canonical pattern."""
    base = 3
    side = base * base

    def pattern(r: int, c: int) -> int:
        return (base * (r % base) + r // base + c) % side

    r_base = list(range(base))
    rows = [g * base + r for g in rng.sample(r_base, base) for r in rng.sample(r_base, base)]
    cols = [g * base + c for g in rng.sample(r_base, base) for c in rng.sample(r_base, base)]
    nums = rng.sample(list(range(1, 10)), 9)

    grid = [0] * 81
    for r in range(9):
        for c in range(9):
            grid[rc_to_i(r, c)] = nums[pattern(rows[r], cols[c])]
    return grid


def count_solutions(grid: Grid, limit: int = 2) -> int:
    """
    Count solutions up to 'limit' using backtracking with MRV (minimum remaining values).
    Assumes 0=blank, digits 1..9.
    """
    # Prepare candidate sets
    rows_missing = [set(range(1, 10)) for _ in range(9)]
    cols_missing = [set(range(1, 10)) for _ in range(9)]
    box_missing = [set(range(1, 10)) for _ in range(9)]

    for i, v in enumerate(grid):
        if v == 0:
            continue
        r, c = i_to_rc(i)
        b = box_index(r, c)
        if v not in rows_missing[r] or v not in cols_missing[c] or v not in box_missing[b]:
            return 0  # already inconsistent
        rows_missing[r].remove(v)
        cols_missing[c].remove(v)
        box_missing[b].remove(v)

    # list of blanks
    blanks = [i for i, v in enumerate(grid) if v == 0]

    # For speed: work on a mutable copy
    g = grid[:]
    solutions = 0

    def recurse() -> None:
        nonlocal solutions
        if solutions >= limit:
            return
        # find next blank with MRV
        best_i = -1
        best_cands = None
        best_len = 10
        for i in blanks:
            if g[i] != 0:
                continue
            r, c = i_to_rc(i)
            b = box_index(r, c)
            cands = rows_missing[r] & cols_missing[c] & box_missing[b]
            l = len(cands)
            if l == 0:
                return
            if l < best_len:
                best_len = l
                best_i = i
                best_cands = list(cands)
                if l == 1:
                    break
        if best_i == -1:
            solutions += 1
            return
        r, c = i_to_rc(best_i)
        b = box_index(r, c)
        # try candidates in arbitrary order
        for v in best_cands:
            g[best_i] = v
            rows_missing[r].remove(v)
            cols_missing[c].remove(v)
            box_missing[b].remove(v)
            recurse()
            box_missing[b].add(v)
            cols_missing[c].add(v)
            rows_missing[r].add(v)
            g[best_i] = 0
            if solutions >= limit:
                return

    recurse()
    return solutions


def generate_unique_puzzle(clues: int, seed: int = 0, max_attempts: int = 200) -> Grid:
    """
    Generate a random puzzle with (approximately) the requested number of clues,
    preserving uniqueness (exactly 1 solution).
    """
    if not (17 <= clues <= 81):
        raise ValueError("clues should be in [17, 81].")
    rng = random.Random(seed)

    for attempt in range(max_attempts):
        solved = random_solved_grid(rng)
        puzzle = solved[:]
        positions = list(range(81))
        rng.shuffle(positions)

        # remove while unique
        for pos in positions:
            if puzzle[pos] == 0:
                continue
            backup = puzzle[pos]
            puzzle[pos] = 0
            if count_solutions(puzzle, limit=2) != 1:
                puzzle[pos] = backup
            if sum(1 for v in puzzle if v != 0) <= clues:
                break

        if count_solutions(puzzle, limit=2) == 1 and sum(1 for v in puzzle if v != 0) == clues:
            return puzzle

    raise RuntimeError(f"Failed to generate a unique puzzle with {clues} clues after {max_attempts} attempts.")


# -------------------------
# Regression-style objective (conflict count)
# -------------------------

def unit_conflict_pairs(values: List[int]) -> int:
    """
    Number of violating pairs in a unit: sum_d C(count[d],2), ignoring zeros.
    """
    counts = [0] * 10
    for v in values:
        if v == 0:
            continue
        counts[v] += 1
    return sum(c * (c - 1) // 2 for c in counts)


def conflicts_cols_boxes(grid: Grid) -> int:
    """
    Conflict pairs counted in columns and boxes only (rows assumed to be permutations).
    """
    total = 0
    for unit in COLS + BOXS:
        total += unit_conflict_pairs([grid[i] for i in unit])
    return total


# -------------------------
# Stepwise / local search solver
# -------------------------

@dataclass
class SolveResult:
    solved: bool
    grid: Grid
    steps: int
    restarts: int
    seconds: float
    final_conflicts: int
    trace: Optional[List[int]] = None  # conflicts over time (optional)


def initialise_by_rows(puzzle: Grid, rng: random.Random) -> Tuple[Grid, List[bool]]:
    """
    Fill each row with a random permutation of its missing digits, respecting clues.
    Returns (grid, fixed_mask).
    """
    grid = puzzle[:]
    fixed = [v != 0 for v in puzzle]
    for r in range(9):
        row_cells = ROWS[r]
        present = {grid[i] for i in row_cells if grid[i] != 0}
        missing = [d for d in range(1, 10) if d not in present]
        rng.shuffle(missing)
        mi = 0
        for i in row_cells:
            if grid[i] == 0:
                grid[i] = missing[mi]
                mi += 1
    return grid, fixed


def solve_stepwise_swap(
    puzzle: Grid,
    seed: int = 0,
    max_steps: int = 200_000,
    restarts: int = 30,
    record_trace: bool = False,
    trace_every: int = 200,
) -> SolveResult:
    """
    Local search using within-row swaps (keeps each row a permutation of 1..9).
    Loss = number of conflicting pairs in columns and boxes.
    """
    rng = random.Random(seed)
    t0 = time.time()
    best_grid = None
    best_conf = 10**9

    trace = [] if record_trace else None

    for restart in range(restarts):
        grid, fixed = initialise_by_rows(puzzle, rng)
        conf = conflicts_cols_boxes(grid)
        if record_trace:
            trace.append(conf)

        if conf == 0 and is_valid_complete(grid) and respects_clues(grid, puzzle):
            return SolveResult(True, grid, 0, restart, time.time() - t0, 0, trace)

        # Precompute non-fixed positions per row for swap candidates
        row_free = []
        for r in range(9):
            free = [i for i in ROWS[r] if not fixed[i]]
            row_free.append(free)

        for step in range(1, max_steps + 1):
            if conf == 0 and is_valid_complete(grid) and respects_clues(grid, puzzle):
                return SolveResult(True, grid, step, restart, time.time() - t0, 0, trace)

            # pick a row that is "involved" in conflicts (heuristic):
            # sample rows until we find one whose free cells participate in a conflicting col/box
            candidate_rows = list(range(9))
            rng.shuffle(candidate_rows)
            chosen_r = None
            for r in candidate_rows:
                free = row_free[r]
                if len(free) < 2:
                    continue
                # quick test: if any cell is in a conflicted column or box
                for i in free:
                    rr, cc = i_to_rc(i)
                    b = box_index(rr, cc)
                    col_vals = [grid[j] for j in COLS[cc]]
                    box_vals = [grid[j] for j in BOXS[b]]
                    v = grid[i]
                    if col_vals.count(v) > 1 or box_vals.count(v) > 1:
                        chosen_r = r
                        break
                if chosen_r is not None:
                    break

            if chosen_r is None:
                chosen_r = rng.randrange(9)

            free = row_free[chosen_r]
            if len(free) < 2:
                continue

            # Evaluate all swaps in that row (36 max). Choose the best delta.
            best_delta = 0
            best_pair = None

            # Precompute affected unit indices for each cell quickly
            # (only columns and boxes matter due to row-permutation invariant)
            for a_idx in range(len(free) - 1):
                i1 = free[a_idx]
                r1, c1 = i_to_rc(i1)
                b1 = box_index(r1, c1)
                v1 = grid[i1]
                for b_idx in range(a_idx + 1, len(free)):
                    i2 = free[b_idx]
                    r2, c2 = i_to_rc(i2)
                    b2 = box_index(r2, c2)
                    v2 = grid[i2]
                    if v1 == v2:
                        # swapping equal values changes nothing
                        continue

                    # compute delta by recomputing conflicts in affected cols/boxes only
                    affected_units = []
                    affected_units.append(("C", c1))
                    affected_units.append(("C", c2))
                    affected_units.append(("B", b1))
                    affected_units.append(("B", b2))
                    # dedupe
                    seen = set()
                    affected_units2 = []
                    for typ, idxu in affected_units:
                        key = (typ, idxu)
                        if key not in seen:
                            seen.add(key)
                            affected_units2.append((typ, idxu))

                    old = 0
                    for typ, idxu in affected_units2:
                        unit = (COLS[idxu] if typ == "C" else BOXS[idxu])
                        old += unit_conflict_pairs([grid[j] for j in unit])

                    # perform virtual swap
                    grid[i1], grid[i2] = v2, v1
                    new = 0
                    for typ, idxu in affected_units2:
                        unit = (COLS[idxu] if typ == "C" else BOXS[idxu])
                        new += unit_conflict_pairs([grid[j] for j in unit])
                    # swap back
                    grid[i1], grid[i2] = v1, v2

                    delta = new - old
                    if delta < best_delta:
                        best_delta = delta
                        best_pair = (i1, i2)

            # If no improving swap, do a random swap to escape local minima.
            if best_pair is None:
                i1, i2 = rng.sample(free, 2)
            else:
                i1, i2 = best_pair

            # apply swap
            grid[i1], grid[i2] = grid[i2], grid[i1]
            conf = conflicts_cols_boxes(grid)

            if record_trace and step % trace_every == 0:
                trace.append(conf)

            if conf < best_conf:
                best_conf = conf
                best_grid = grid[:]

        # end of steps for this restart

    # finished restarts
    final_grid = best_grid if best_grid is not None else puzzle[:]
    return SolveResult(False, final_grid, max_steps, restarts, time.time() - t0, best_conf, trace)


# -------------------------
# CLI
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Sudoku via integer p-adic regression (81 variables, no one-hot).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_gen = sub.add_parser("generate", help="Generate a random unique-solution puzzle.")
    ap_gen.add_argument("--clues", type=int, default=30)
    ap_gen.add_argument("--seed", type=int, default=0)

    ap_solve = sub.add_parser("solve", help="Solve a puzzle given as an 81-char string with 0 or . for blanks.")
    ap_solve.add_argument("--puzzle", type=str, required=True)
    ap_solve.add_argument("--seed", type=int, default=0)
    ap_solve.add_argument("--max-steps", type=int, default=200_000)
    ap_solve.add_argument("--restarts", type=int, default=30)
    ap_solve.add_argument("--trace", action="store_true")

    args = ap.parse_args()

    if args.cmd == "generate":
        puzzle = generate_unique_puzzle(args.clues, seed=args.seed)
        print(grid_to_string(puzzle))
        print()
        print(pretty(puzzle))
        return

    if args.cmd == "solve":
        puzzle = parse_puzzle(args.puzzle)
        res = solve_stepwise_swap(
            puzzle,
            seed=args.seed,
            max_steps=args.max_steps,
            restarts=args.restarts,
            record_trace=args.trace,
            trace_every=200,
        )
        print("Solved:", res.solved)
        print("Steps:", res.steps, "Restarts:", res.restarts, "Seconds:", f"{res.seconds:.3f}")
        print("Final conflicts (cols+boxes):", res.final_conflicts)
        print()
        print(pretty(res.grid))
        return


if __name__ == "__main__":
    main()
