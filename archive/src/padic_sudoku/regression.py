"""
Sudoku as a p-adic linear regression problem.

Data points:
1. Forcing points: For each cell j and target k ∈ {1,...,9}, add (e_j, k) with weight M
   This pushes β_j toward one of {1,...,9}

2. Sum constraints: For each row/col/box g, add (indicator of g, 45) with weight W_sum
   This pushes each group to sum to 45

3. Clue constraints: For each clue (j, value), add (e_j, value) with very high weight
   This fixes the known cells

4. Inequality constraints: For each pair (A, B) in same group, add (e_A - e_B, 0) with weight -λ
   This pushes cells in same group to be different (NEGATIVE weight)

Loss function:
L(β) = Σ w_i |y_i - x_i^T β|_p

where some w_i are negative (the inequality constraints).
"""

from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable
import random

from .encoding import CONSTRAINT_GROUPS, get_cell_groups


def v_p(x: Fraction | int, p: int) -> int | float:
    """p-adic valuation of a rational number."""
    if isinstance(x, int):
        x = Fraction(x)
    if x == 0:
        return float("inf")

    num = abs(x.numerator)
    den = x.denominator

    v = 0
    while num % p == 0:
        num //= p
        v += 1
    while den % p == 0:
        den //= p
        v -= 1

    return v


def padic_abs(x: Fraction | int, p: int) -> Fraction:
    """p-adic absolute value."""
    if isinstance(x, int):
        x = Fraction(x)
    val = v_p(x, p)
    if val == float("inf"):
        return Fraction(0)
    return Fraction(1, p ** val) if val >= 0 else Fraction(p ** (-val), 1)


@dataclass
class DataPoint:
    """A data point in the regression problem."""
    x: list[Fraction]  # 81-dimensional vector
    y: Fraction
    weight: Fraction  # Can be negative for inequality constraints
    description: str = ""


def make_unit_vector(j: int, n: int = 81) -> list[Fraction]:
    """Create a unit vector e_j."""
    vec = [Fraction(0)] * n
    vec[j] = Fraction(1)
    return vec


def make_difference_vector(a: int, b: int, n: int = 81) -> list[Fraction]:
    """Create a difference vector e_a - e_b."""
    vec = [Fraction(0)] * n
    vec[a] = Fraction(1)
    vec[b] = Fraction(-1)
    return vec


def make_group_indicator(group: list[tuple[int, int]], n: int = 81) -> list[Fraction]:
    """Create indicator vector for a constraint group."""
    vec = [Fraction(0)] * n
    for r, c in group:
        idx = r * 9 + c
        vec[idx] = Fraction(1)
    return vec


def cell_index(r: int, c: int) -> int:
    """Convert (row, col) to linear index."""
    return r * 9 + c


def index_to_cell(idx: int) -> tuple[int, int]:
    """Convert linear index to (row, col)."""
    return divmod(idx, 9)


def build_regression_problem(
    clues: dict[tuple[int, int], int],  # (row, col) -> digit
    M_forcing: Fraction = Fraction(10),
    W_sum: Fraction = Fraction(1),
    W_clue: Fraction = Fraction(1000),
    lambda_ineq: Fraction = Fraction(1, 10),  # Negative weight magnitude
    targets: list[int] = list(range(1, 10)),
) -> list[DataPoint]:
    """
    Build the p-adic regression problem for a Sudoku puzzle.

    Args:
        clues: Dictionary mapping (row, col) to known digit values
        M_forcing: Weight for forcing points (pushes toward target values)
        W_sum: Weight for sum constraints
        W_clue: Weight for clue constraints (should be very high)
        lambda_ineq: Magnitude of negative weight for inequality constraints

    Returns:
        List of DataPoint objects defining the regression problem
    """
    data_points = []

    # 1. Forcing points: for each cell j and target k, add (e_j, k)
    for j in range(81):
        for k in targets:
            dp = DataPoint(
                x=make_unit_vector(j),
                y=Fraction(k),
                weight=M_forcing,
                description=f"forcing: cell {j} toward {k}",
            )
            data_points.append(dp)

    # 2. Sum constraints: for each group, add (indicator, 45)
    target_sum = sum(targets)  # 45 for {1,...,9}
    for g_idx, group in enumerate(CONSTRAINT_GROUPS):
        dp = DataPoint(
            x=make_group_indicator(group),
            y=Fraction(target_sum),
            weight=W_sum,
            description=f"sum: group {g_idx} -> {target_sum}",
        )
        data_points.append(dp)

    # 3. Clue constraints: for each clue, add (e_j, value) with high weight
    for (r, c), value in clues.items():
        j = cell_index(r, c)
        dp = DataPoint(
            x=make_unit_vector(j),
            y=Fraction(value),
            weight=W_clue,
            description=f"clue: cell ({r},{c}) = {value}",
        )
        data_points.append(dp)

    # 4. Inequality constraints: for each pair in same group, add (e_A - e_B, 0) with NEGATIVE weight
    pairs_added = set()
    for group in CONSTRAINT_GROUPS:
        cells = [cell_index(r, c) for r, c in group]
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                a, b = cells[i], cells[j]
                if a > b:
                    a, b = b, a
                if (a, b) not in pairs_added:
                    pairs_added.add((a, b))
                    dp = DataPoint(
                        x=make_difference_vector(a, b),
                        y=Fraction(0),
                        weight=-lambda_ineq,  # NEGATIVE!
                        description=f"ineq: cell {a} != cell {b}",
                    )
                    data_points.append(dp)

    return data_points


def evaluate_residual(dp: DataPoint, beta: list[Fraction]) -> Fraction:
    """Compute y - x^T β for a data point."""
    xb = sum(dp.x[i] * beta[i] for i in range(len(beta)))
    return dp.y - xb


def evaluate_loss(
    data_points: list[DataPoint],
    beta: list[Fraction],
    p: int,
) -> Fraction:
    """
    Compute total weighted p-adic loss.

    L(β) = Σ w_i |y_i - x_i^T β|_p

    Note: Some weights may be negative (inequality constraints).
    """
    total = Fraction(0)
    for dp in data_points:
        residual = evaluate_residual(dp, beta)
        padic_val = padic_abs(residual, p)
        total += dp.weight * padic_val
    return total


def evaluate_loss_components(
    data_points: list[DataPoint],
    beta: list[Fraction],
    p: int,
) -> dict[str, Fraction]:
    """Evaluate loss broken down by component type."""
    components = {
        "forcing": Fraction(0),
        "sum": Fraction(0),
        "clue": Fraction(0),
        "ineq": Fraction(0),
    }

    for dp in data_points:
        residual = evaluate_residual(dp, beta)
        padic_val = padic_abs(residual, p)
        contribution = dp.weight * padic_val

        if dp.description.startswith("forcing"):
            components["forcing"] += contribution
        elif dp.description.startswith("sum"):
            components["sum"] += contribution
        elif dp.description.startswith("clue"):
            components["clue"] += contribution
        elif dp.description.startswith("ineq"):
            components["ineq"] += contribution

    return components


def random_beta(targets: list[int] = list(range(1, 10)), rng: random.Random = None) -> list[Fraction]:
    """Generate a random initial β with values from targets."""
    if rng is None:
        rng = random.Random()
    return [Fraction(rng.choice(targets)) for _ in range(81)]


def check_solution(beta: list[Fraction], clues: dict[tuple[int, int], int]) -> dict:
    """
    Check if β represents a valid Sudoku solution.

    Returns dict with:
    - is_integer: all β values are integers
    - is_valid_digits: all β values are in {1,...,9}
    - clues_satisfied: all clues match
    - sums_correct: all row/col/box sums are 45
    - all_different: no duplicates in any group
    - is_solution: all of the above
    """
    result = {
        "is_integer": all(b.denominator == 1 for b in beta),
        "is_valid_digits": all(b in [Fraction(k) for k in range(1, 10)] for b in beta),
        "clues_satisfied": True,
        "sums_correct": True,
        "all_different": True,
    }

    # Check clues
    for (r, c), value in clues.items():
        j = cell_index(r, c)
        if beta[j] != Fraction(value):
            result["clues_satisfied"] = False
            break

    # Check sums
    for group in CONSTRAINT_GROUPS:
        group_sum = sum(beta[cell_index(r, c)] for r, c in group)
        if group_sum != Fraction(45):
            result["sums_correct"] = False
            break

    # Check all different
    for group in CONSTRAINT_GROUPS:
        values = [beta[cell_index(r, c)] for r, c in group]
        if len(set(values)) != len(values):
            result["all_different"] = False
            break

    result["is_solution"] = all([
        result["is_integer"],
        result["is_valid_digits"],
        result["clues_satisfied"],
        result["sums_correct"],
        result["all_different"],
    ])

    return result


def beta_to_grid(beta: list[Fraction]) -> list[list[int]]:
    """Convert β vector to 9x9 grid (rounding to nearest integer)."""
    grid = []
    for r in range(9):
        row = []
        for c in range(9):
            val = beta[cell_index(r, c)]
            row.append(round(float(val)))
        grid.append(row)
    return grid


def grid_to_beta(grid: list[list[int]]) -> list[Fraction]:
    """Convert 9x9 grid to β vector."""
    beta = []
    for r in range(9):
        for c in range(9):
            beta.append(Fraction(grid[r][c]))
    return beta
