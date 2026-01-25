#!/usr/bin/env python3
"""
Analyze the p-adic regression formulation without running slow solvers.
"""

from __future__ import annotations
import sys
from pathlib import Path
from fractions import Fraction

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from padic_sudoku.regression import (
    build_regression_problem,
    evaluate_loss,
    evaluate_loss_components,
    check_solution,
    beta_to_grid,
    grid_to_beta,
    DataPoint,
    padic_abs,
)
from padic_sudoku.puzzle import parse_81_string


def puzzle_to_clues(puzzle_str: str) -> dict[tuple[int, int], int]:
    """Extract clues from puzzle string."""
    clues = {}
    puzzle_str = puzzle_str.replace(".", "0")
    for i, ch in enumerate(puzzle_str):
        if ch != "0":
            r, c = divmod(i, 9)
            clues[(r, c)] = int(ch)
    return clues


def analyze_loss_landscape():
    """
    Analyze the loss at correct solution vs perturbed solutions.

    Key question: Does the correct solution have lower loss than incorrect ones?
    """
    print("=" * 60)
    print("Analyzing p-adic Regression Loss Landscape")
    print("=" * 60)

    puzzle_str = "003020600900305001001806400008102900700000008006708200002609500800203009005010300"
    solution_str = "483921657967345821251876493548132976729564138136798245372689514814253769695417382"

    clues = puzzle_to_clues(puzzle_str)
    print(f"\nPuzzle has {len(clues)} clues")

    # Build regression problem with different negative weights
    for lambda_val in [Fraction(0), Fraction(1, 100), Fraction(1, 10), Fraction(1, 2), Fraction(1)]:
        print(f"\n{'=' * 60}")
        print(f"lambda_ineq = {lambda_val}")
        print("=" * 60)

        data_points = build_regression_problem(
            clues,
            M_forcing=Fraction(10),
            W_sum=Fraction(1),
            W_clue=Fraction(1000),
            lambda_ineq=lambda_val,
        )

        # Parse solution
        solution = parse_81_string(solution_str)
        solution_beta = grid_to_beta(solution.grid)

        for p in [2, 19]:
            print(f"\n  Prime p = {p}:")

            # Loss at correct solution
            correct_loss = evaluate_loss(data_points, solution_beta, p)
            correct_components = evaluate_loss_components(data_points, solution_beta, p)

            print(f"    Correct solution loss: {float(correct_loss):.4f}")
            for k, v in correct_components.items():
                print(f"      {k}: {float(v):.4f}")

            # Loss at solution with one wrong cell
            wrong_beta = solution_beta.copy()
            # Find a non-clue cell and change it
            for j in range(81):
                r, c = divmod(j, 9)
                if (r, c) not in clues:
                    # Change this cell to a wrong value
                    old_val = wrong_beta[j]
                    for new_val in range(1, 10):
                        if Fraction(new_val) != old_val:
                            wrong_beta[j] = Fraction(new_val)
                            break
                    break

            wrong_loss = evaluate_loss(data_points, wrong_beta, p)
            wrong_components = evaluate_loss_components(data_points, wrong_beta, p)

            print(f"    One-error solution loss: {float(wrong_loss):.4f}")
            for k, v in wrong_components.items():
                print(f"      {k}: {float(v):.4f}")

            diff = wrong_loss - correct_loss
            print(f"    Difference (wrong - correct): {float(diff):.4f}")

            if diff > 0:
                print("    --> Correct solution has LOWER loss (good!)")
            elif diff < 0:
                print("    --> Correct solution has HIGHER loss (bad!)")
            else:
                print("    --> Same loss (neutral)")


def analyze_inequality_effect():
    """
    Understand exactly what the negative weight inequality constraints do.
    """
    print("\n" + "=" * 60)
    print("Detailed Analysis of Inequality Constraints")
    print("=" * 60)

    # Simplest case: just two cells that must be different
    # β_0 and β_1 in the same row

    p = 19
    print(f"\nUsing prime p = {p}")

    # The inequality constraint is: (e_0 - e_1, 0) with weight -λ
    # Residual = 0 - (β_0 - β_1) = β_1 - β_0
    # Contribution = -λ × |β_1 - β_0|_p

    print("\nFor two cells in the same group:")
    print("Inequality data point: (e_0 - e_1, 0) with weight -λ")
    print("Residual = 0 - β_0 + β_1 = β_1 - β_0")
    print("Contribution = -λ × |β_1 - β_0|_p")

    print(f"\nIf β_0 = β_1 (SAME value):")
    print(f"  Residual = 0")
    print(f"  |0|_p = 0")
    print(f"  Contribution = -λ × 0 = 0")

    print(f"\nIf β_0 ≠ β_1 (DIFFERENT values):")
    print(f"  Residual = β_1 - β_0 ≠ 0")
    print(f"  |β_1 - β_0|_p ≥ p^(-v) > 0 for some valuation v")
    print(f"  Contribution = -λ × |β_1 - β_0|_p < 0")

    print("\n--> When cells are DIFFERENT, the loss DECREASES (negative contribution)")
    print("--> When cells are SAME, contribution is 0")
    print("--> So negative weight ENCOURAGES different values")

    # Show specific examples
    print("\nSpecific examples with p=19:")
    for diff in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        pval = padic_abs(Fraction(diff), p)
        print(f"  |{diff}|_19 = {float(pval):.4f}")


def analyze_stability():
    """
    Analyze when negative weights might cause instability.
    """
    print("\n" + "=" * 60)
    print("Stability Analysis")
    print("=" * 60)

    print("""
The concern with negative weights is that they could make the loss unbounded below.

For p-adic regression, the loss is:
  L(β) = Σ w_i |y_i - x_i^T β|_p

With some w_i < 0 (inequality constraints).

The loss is bounded below if the positive terms dominate.

For Sudoku:
- Positive terms: forcing (81×9 = 729 points with weight M each)
                  sum (27 points with weight W)
                  clues (~30 points with weight W_clue)

- Negative terms: inequality (~810 pairs with weight -λ each)

Total positive weight: 729×M + 27×W + 30×W_clue
Total negative weight: -810×λ

For stability, we want:
  729×M + 27×W + 30×W_clue >> 810×λ

With M=10, W=1, W_clue=1000, λ=0.1:
  Positive: 729×10 + 27 + 30×1000 = 7290 + 27 + 30000 = 37317
  Negative: 810×0.1 = 81

Ratio: 37317 / 81 ≈ 460

So the positive terms dominate by a factor of ~460, which should be stable.
""")


def main():
    analyze_inequality_effect()
    analyze_stability()
    analyze_loss_landscape()


if __name__ == "__main__":
    main()
