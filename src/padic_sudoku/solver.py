"""
Solvers for p-adic linear regression on Sudoku.

From the paper: optimal solutions pass through at least n+1 points.
For Sudoku (n=81), we need 82 points. With ~1600 data points,
exhaustive enumeration of C(1600, 82) is impossible.

This module implements heuristic solvers that sample subsets of points.
"""

from __future__ import annotations
from fractions import Fraction
from dataclasses import dataclass
import random
import numpy as np
from typing import Callable

from .regression import (
    DataPoint,
    evaluate_loss,
    evaluate_loss_components,
    check_solution,
    beta_to_grid,
    grid_to_beta,
    random_beta,
)


@dataclass
class SolverResult:
    """Result from a solver run."""
    beta: list[Fraction]
    loss: Fraction
    loss_components: dict[str, Fraction]
    solution_check: dict
    iterations: int
    method: str


def solve_linear_system(
    points: list[DataPoint],
    n_dims: int = 81,
) -> list[Fraction] | None:
    """
    Solve the linear system defined by n_dims points.

    Given points (x_i, y_i), solve X @ β = y where X has rows x_i.

    Returns None if system is singular or has wrong dimensions.
    """
    if len(points) != n_dims:
        return None

    # Build matrix X and vector y
    X = np.array([[float(x) for x in dp.x] for dp in points], dtype=np.float64)
    y = np.array([float(dp.y) for dp in points], dtype=np.float64)

    try:
        # Solve X @ β = y
        beta_float = np.linalg.solve(X, y)

        # Convert back to Fractions (with rounding for numerical stability)
        # We round to nearest 1/1000 to handle floating point errors
        beta = []
        for b in beta_float:
            # Round to nearest integer if close
            rounded = round(b)
            if abs(b - rounded) < 1e-6:
                beta.append(Fraction(rounded))
            else:
                # Keep as fraction approximation
                beta.append(Fraction(b).limit_denominator(1000))
        return beta

    except np.linalg.LinAlgError:
        return None


def sample_and_solve(
    data_points: list[DataPoint],
    p: int,
    n_samples: int = 1000,
    n_dims: int = 81,
    prioritize_clues: bool = True,
    rng: random.Random = None,
) -> SolverResult:
    """
    Sample subsets of n_dims points, solve each, keep best.

    This is a randomized algorithm based on the theorem that
    optimal solutions pass through n+1 points. We sample n_dims points
    (not n_dims+1) because we want to uniquely determine β.

    Args:
        data_points: Full list of data points
        p: Prime for p-adic evaluation
        n_samples: Number of subsets to try
        n_dims: Number of dimensions (default 81 for Sudoku)
        prioritize_clues: If True, always include clue constraints
        rng: Random number generator
    """
    if rng is None:
        rng = random.Random()

    # Separate data points by type
    clue_points = [dp for dp in data_points if dp.description.startswith("clue")]
    other_points = [dp for dp in data_points if not dp.description.startswith("clue")]

    best_beta = None
    best_loss = None

    for _ in range(n_samples):
        # Select points for this sample
        if prioritize_clues and len(clue_points) <= n_dims:
            # Include all clues, sample rest from other points
            n_needed = n_dims - len(clue_points)
            if n_needed > len(other_points):
                # Not enough points
                continue
            sample = clue_points + rng.sample(other_points, n_needed)
        else:
            # Random sample from all points
            if len(data_points) < n_dims:
                continue
            sample = rng.sample(data_points, n_dims)

        # Solve the linear system
        beta = solve_linear_system(sample, n_dims)
        if beta is None:
            continue

        # Evaluate loss on ALL data points
        loss = evaluate_loss(data_points, beta, p)

        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_beta = beta

    if best_beta is None:
        # Fall back to random initialization
        best_beta = random_beta(rng=rng)
        best_loss = evaluate_loss(data_points, best_beta, p)

    return SolverResult(
        beta=best_beta,
        loss=best_loss,
        loss_components=evaluate_loss_components(data_points, best_beta, p),
        solution_check=check_solution(best_beta, {}),  # No clues for check
        iterations=n_samples,
        method="sample_and_solve",
    )


def greedy_coordinate_descent(
    data_points: list[DataPoint],
    p: int,
    initial_beta: list[Fraction] = None,
    targets: list[int] = list(range(1, 10)),
    max_iterations: int = 1000,
    rng: random.Random = None,
) -> SolverResult:
    """
    Greedy coordinate descent over target values.

    At each step, find the cell and target value that most improves the loss.
    Only considers integer target values.
    """
    if rng is None:
        rng = random.Random()

    if initial_beta is None:
        beta = random_beta(targets, rng)
    else:
        beta = initial_beta.copy()

    current_loss = evaluate_loss(data_points, beta, p)

    for iteration in range(max_iterations):
        improved = False

        # Try each cell
        for j in range(81):
            old_val = beta[j]

            # Try each target value
            for k in targets:
                new_val = Fraction(k)
                if new_val == old_val:
                    continue

                beta[j] = new_val
                new_loss = evaluate_loss(data_points, beta, p)

                if new_loss < current_loss:
                    current_loss = new_loss
                    improved = True
                    break  # Accept first improvement
                else:
                    beta[j] = old_val  # Revert

            if improved:
                break

        if not improved:
            break

    return SolverResult(
        beta=beta,
        loss=current_loss,
        loss_components=evaluate_loss_components(data_points, beta, p),
        solution_check=check_solution(beta, {}),
        iterations=iteration + 1,
        method="greedy_coordinate_descent",
    )


def gradient_descent_continuous(
    data_points: list[DataPoint],
    p: int,
    initial_beta: list[Fraction] = None,
    learning_rate: float = 0.1,
    max_iterations: int = 1000,
    targets: list[int] = list(range(1, 10)),
    rng: random.Random = None,
) -> SolverResult:
    """
    Continuous gradient descent, then round to targets.

    This relaxes the integer constraint, optimizes in continuous space,
    then rounds to the nearest target value.

    Note: The p-adic absolute value is not differentiable in the usual sense,
    so we use a smooth approximation or numerical gradients.
    """
    if rng is None:
        rng = random.Random()

    if initial_beta is None:
        # Initialize at center of target range
        center = sum(targets) / len(targets)
        beta = [center] * 81
    else:
        beta = [float(b) for b in initial_beta]

    def loss_float(beta_float):
        beta_frac = [Fraction(b).limit_denominator(1000) for b in beta_float]
        return float(evaluate_loss(data_points, beta_frac, p))

    current_loss = loss_float(beta)

    for iteration in range(max_iterations):
        # Numerical gradient
        grad = [0.0] * 81
        eps = 0.01

        for j in range(81):
            beta[j] += eps
            loss_plus = loss_float(beta)
            beta[j] -= 2 * eps
            loss_minus = loss_float(beta)
            beta[j] += eps  # Restore

            grad[j] = (loss_plus - loss_minus) / (2 * eps)

        # Update
        new_beta = [beta[j] - learning_rate * grad[j] for j in range(81)]
        new_loss = loss_float(new_beta)

        if new_loss < current_loss:
            beta = new_beta
            current_loss = new_loss
        else:
            learning_rate *= 0.9

        if learning_rate < 1e-6:
            break

    # Round to nearest targets
    beta_rounded = []
    for b in beta:
        closest = min(targets, key=lambda k: abs(k - b))
        beta_rounded.append(Fraction(closest))

    return SolverResult(
        beta=beta_rounded,
        loss=evaluate_loss(data_points, beta_rounded, p),
        loss_components=evaluate_loss_components(data_points, beta_rounded, p),
        solution_check=check_solution(beta_rounded, {}),
        iterations=iteration + 1,
        method="gradient_descent_continuous",
    )


def hybrid_solver(
    data_points: list[DataPoint],
    clues: dict[tuple[int, int], int],
    p: int,
    n_restarts: int = 10,
    samples_per_restart: int = 100,
    greedy_iterations: int = 500,
    rng: random.Random = None,
) -> SolverResult:
    """
    Hybrid solver combining sampling and coordinate descent.

    1. Try sample_and_solve to get initial solutions
    2. Refine best solutions with greedy coordinate descent
    3. Return best overall
    """
    if rng is None:
        rng = random.Random()

    best_result = None

    for restart in range(n_restarts):
        # Phase 1: Sample and solve
        result = sample_and_solve(
            data_points, p,
            n_samples=samples_per_restart,
            prioritize_clues=True,
            rng=rng,
        )

        # Phase 2: Refine with coordinate descent
        refined = greedy_coordinate_descent(
            data_points, p,
            initial_beta=result.beta,
            max_iterations=greedy_iterations,
            rng=rng,
        )

        if best_result is None or refined.loss < best_result.loss:
            best_result = refined

    best_result.method = "hybrid_solver"
    return best_result
