"""p-adic arithmetic: valuations, norms, and objective functions."""

from __future__ import annotations
import math
from .encoding import TARGET_SUM, CONSTRAINT_GROUPS


def v_p(n: int, p: int) -> int | float:
    """
    Compute the p-adic valuation of n.

    v_p(n) is the largest exponent k such that p^k divides n.
    Returns float('inf') if n == 0.
    """
    if n == 0:
        return float("inf")
    n = abs(n)
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


def padic_norm(n: int, p: int) -> float:
    """
    Compute the p-adic norm |n|_p = p^{-v_p(n)}.

    Returns 0 if n == 0.
    """
    val = v_p(n, p)
    if val == float("inf"):
        return 0.0
    return p ** (-val)


def residual(grid: list[list[int]], group: list[tuple[int, int]]) -> int:
    """
    Compute the residual for a constraint group.

    residual = TARGET_SUM - sum of cells in group
    A residual of 0 means the constraint is satisfied.
    """
    return TARGET_SUM - sum(grid[r][c] for r, c in group)


def compute_all_residuals(grid: list[list[int]]) -> list[int]:
    """Compute residuals for all 27 constraint groups."""
    return [residual(grid, g) for g in CONSTRAINT_GROUPS]


def compute_all_valuations(grid: list[list[int]], p: int) -> list[int | float]:
    """Compute p-adic valuations of all 27 residuals."""
    return [v_p(residual(grid, g), p) for g in CONSTRAINT_GROUPS]


def min_valuation(grid: list[list[int]], p: int) -> int | float:
    """
    Compute the minimum p-adic valuation across all 27 constraints.

    This is the current 'lift level'. Higher is better.
    When this reaches required_lift_level(p), we have a valid solution.
    """
    return min(v_p(residual(grid, g), p) for g in CONSTRAINT_GROUPS)


def total_padic_loss(grid: list[list[int]], p: int) -> float:
    """
    Compute total p-adic loss: sum of |r_g|_p across all groups.

    Lower is better. A loss of 0 means all constraints are satisfied.
    """
    return sum(padic_norm(residual(grid, g), p) for g in CONSTRAINT_GROUPS)


def required_lift_level(p: int) -> int:
    """
    Compute the smallest k such that p^k > 2304.

    The maximum possible row sum is 9 * 256 = 2304.
    Once all valuations >= k, satisfaction mod p^k implies integer solution.
    """
    return math.ceil(math.log(2305) / math.log(p))


def is_solved(grid: list[list[int]], p: int) -> bool:
    """Check if the grid is solved (all residuals are 0)."""
    return min_valuation(grid, p) == float("inf")


def is_valid_sudoku(grid: list[list[int]]) -> bool:
    """Check if the grid is a valid Sudoku solution (all residuals are 0)."""
    return all(residual(grid, g) == 0 for g in CONSTRAINT_GROUPS)
