#!/usr/bin/env python3
"""
Test the p-adic regression formulation of Sudoku.
"""

from __future__ import annotations
import sys
from pathlib import Path
from fractions import Fraction

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from padic_sudoku.puzzle import get_euler_50, parse_81_string
from padic_sudoku.regression import (
    build_regression_problem,
    evaluate_loss,
    evaluate_loss_components,
    check_solution,
    beta_to_grid,
    grid_to_beta,
    cell_index,
)
from padic_sudoku.solver import (
    sample_and_solve,
    greedy_coordinate_descent,
    hybrid_solver,
)

import random


def print_grid(grid: list[list[int]]):
    """Pretty print a Sudoku grid."""
    for i, row in enumerate(grid):
        if i % 3 == 0 and i > 0:
            print("-" * 21)
        row_str = ""
        for j, val in enumerate(row):
            if j % 3 == 0 and j > 0:
                row_str += "| "
            row_str += f"{val} "
        print(row_str)


def puzzle_to_clues(puzzle_str: str) -> dict[tuple[int, int], int]:
    """Extract clues from puzzle string."""
    clues = {}
    puzzle_str = puzzle_str.replace(".", "0")
    for i, ch in enumerate(puzzle_str):
        if ch != "0":
            r, c = divmod(i, 9)
            clues[(r, c)] = int(ch)
    return clues


def test_single_puzzle(puzzle_str: str, solution_str: str, p: int = 19):
    """Test the regression formulation on a single puzzle."""
    print("\n" + "=" * 60)
    print(f"Testing with prime p = {p}")
    print("=" * 60)

    clues = puzzle_to_clues(puzzle_str)
    print(f"\nPuzzle has {len(clues)} clues")

    # Build the regression problem
    print("\nBuilding regression problem...")
    data_points = build_regression_problem(
        clues,
        M_forcing=Fraction(10),
        W_sum=Fraction(1),
        W_clue=Fraction(1000),
        lambda_ineq=Fraction(1, 10),
    )

    # Count data points by type
    forcing = sum(1 for dp in data_points if dp.description.startswith("forcing"))
    sums = sum(1 for dp in data_points if dp.description.startswith("sum"))
    clue_pts = sum(1 for dp in data_points if dp.description.startswith("clue"))
    ineq = sum(1 for dp in data_points if dp.description.startswith("ineq"))

    print(f"Data points: {len(data_points)} total")
    print(f"  Forcing: {forcing}")
    print(f"  Sum constraints: {sums}")
    print(f"  Clue constraints: {clue_pts}")
    print(f"  Inequality constraints: {ineq}")

    # Evaluate loss at the known solution
    solution = parse_81_string(solution_str)
    solution_beta = grid_to_beta(solution.grid)

    solution_loss = evaluate_loss(data_points, solution_beta, p)
    solution_components = evaluate_loss_components(data_points, solution_beta, p)

    print(f"\nLoss at known solution: {float(solution_loss):.4f}")
    print("  Components:")
    for k, v in solution_components.items():
        print(f"    {k}: {float(v):.4f}")

    # Check the solution
    solution_check = check_solution(solution_beta, clues)
    print(f"\nSolution check: {solution_check}")

    # Try random initialization
    rng = random.Random(42)
    print("\n" + "-" * 40)
    print("Testing solvers...")

    # Test greedy coordinate descent from random start
    print("\n1. Greedy coordinate descent (random start):")
    result = greedy_coordinate_descent(
        data_points, p,
        initial_beta=None,
        max_iterations=500,
        rng=rng,
    )
    print(f"   Loss: {float(result.loss):.4f}")
    print(f"   Iterations: {result.iterations}")
    print(f"   Solution check: {result.solution_check}")

    # Test greedy from near-solution
    print("\n2. Greedy coordinate descent (from perturbed solution):")
    perturbed = solution_beta.copy()
    # Perturb a few non-clue cells
    non_clue_cells = [j for j in range(81) if (j // 9, j % 9) not in clues]
    for j in rng.sample(non_clue_cells, min(10, len(non_clue_cells))):
        perturbed[j] = Fraction(rng.randint(1, 9))

    result2 = greedy_coordinate_descent(
        data_points, p,
        initial_beta=perturbed,
        max_iterations=500,
        rng=rng,
    )
    print(f"   Loss: {float(result2.loss):.4f}")
    print(f"   Iterations: {result2.iterations}")
    print(f"   Solution check: {result2.solution_check}")

    # Test hybrid solver
    print("\n3. Hybrid solver:")
    result3 = hybrid_solver(
        data_points, clues, p,
        n_restarts=5,
        samples_per_restart=50,
        greedy_iterations=200,
        rng=rng,
    )
    print(f"   Loss: {float(result3.loss):.4f}")
    print(f"   Solution check: {result3.solution_check}")

    # Show the best result grid
    print("\nBest result grid:")
    best_grid = beta_to_grid(result3.beta)
    print_grid(best_grid)

    return result3


def analyze_inequality_constraints(p: int = 19):
    """
    Analyze what happens with the inequality (negative weight) constraints.
    """
    print("\n" + "=" * 60)
    print("Analysis of Negative Weight Constraints")
    print("=" * 60)

    # Simple test: two cells that should be different
    from padic_sudoku.regression import (
        DataPoint, make_unit_vector, make_difference_vector,
        evaluate_loss, padic_abs,
    )

    # Two cells: β_0 and β_1
    # Forcing: push both toward {1,...,9}
    # Inequality: push β_0 != β_1

    data_points = []

    # Forcing for cell 0
    for k in range(1, 10):
        data_points.append(DataPoint(
            x=[Fraction(1), Fraction(0)],
            y=Fraction(k),
            weight=Fraction(10),
        ))

    # Forcing for cell 1
    for k in range(1, 10):
        data_points.append(DataPoint(
            x=[Fraction(0), Fraction(1)],
            y=Fraction(k),
            weight=Fraction(10),
        ))

    # Inequality: β_0 - β_1 = 0 with NEGATIVE weight
    data_points.append(DataPoint(
        x=[Fraction(1), Fraction(-1)],
        y=Fraction(0),
        weight=Fraction(-1),  # Negative!
    ))

    print(f"\nSimple 2-cell test with p={p}")
    print("Forcing both cells toward {1,...,9}")
    print("Inequality pushes cells to be different (negative weight)")

    # Test some configurations
    test_cases = [
        ([5, 5], "same value"),
        ([5, 6], "different (adjacent)"),
        ([1, 9], "different (far)"),
        ([3, 7], "different (medium)"),
    ]

    print(f"\n{'Config':<20} {'Loss':>10} {'Forcing':>10} {'Ineq':>10}")
    print("-" * 52)

    for beta_vals, desc in test_cases:
        beta = [Fraction(b) for b in beta_vals]

        # Calculate components
        forcing_loss = Fraction(0)
        ineq_loss = Fraction(0)

        for dp in data_points:
            residual = dp.y - sum(dp.x[i] * beta[i] for i in range(2))
            pval = padic_abs(residual, p)
            contrib = dp.weight * pval

            if dp.weight > 0:
                forcing_loss += contrib
            else:
                ineq_loss += contrib

        total = forcing_loss + ineq_loss
        print(f"{str(beta_vals):<20} {float(total):>10.4f} {float(forcing_loss):>10.4f} {float(ineq_loss):>10.4f}")

    print("\nNote: Negative inequality contribution means the loss is REDUCED")
    print("when cells are different (residual β_0 - β_1 is non-zero).")


def main():
    # First, analyze the inequality constraints
    analyze_inequality_constraints(p=19)

    # Get a puzzle
    puzzles = get_euler_50()
    puzzle = puzzles[0]

    puzzle_str = "003020600900305001001806400008102900700000008006708200002609500800203009005010300"
    solution_str = "483921657967345821251876493548132976729564138136798245372689514814253769695417382"

    # Test with p=19 (best gap from forcing analysis)
    test_single_puzzle(puzzle_str, solution_str, p=19)

    # Also try p=2
    print("\n\n" + "#" * 60)
    print("Comparison with p=2")
    print("#" * 60)
    test_single_puzzle(puzzle_str, solution_str, p=2)


if __name__ == "__main__":
    main()
