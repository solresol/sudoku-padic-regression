from collections.abc import Callable
import json
from pathlib import Path

import pytest

from padic_sudoku_regression import (
    Grid,
    SolveResult,
    conflicts_all_units,
    conflicts_cols_boxes,
    count_solutions,
    deduped_peer_conflicts,
    generate_unique_puzzle,
    grid_to_string,
    is_valid_complete,
    p_adic_norm,
    respects_clues,
    solve_greedy_descent_swap,
    solve_greedy_local_edit_best,
    solve_greedy_local_edit_first,
    solve_mihara_digitwise_attempt,
    solve_stepwise_swap,
    solve_zubarev_local_edit,
    solve_zubarev_walk,
)

SOLVED_GRID = [
    int(digit)
    for digit in (
        "534678912"
        "672195348"
        "198342567"
        "859761423"
        "426853791"
        "713924856"
        "961537284"
        "287419635"
        "345286179"
    )
]

EXPECTED_SEED_42_PUZZLE = (
    ".....96.8"
    ".798642.."
    "6.....5.."
    "..3...4.."
    ".2....1.6"
    "45.61..7."
    "71..8..6."
    "3....2..9"
    ".9....7.."
)

Solver = Callable[..., SolveResult]


@pytest.fixture(scope="module")
def solver_smoke_puzzle() -> Grid:
    return generate_unique_puzzle(72, seed=7)


def test_p_adic_norm_separates_equal_and_unequal_digits() -> None:
    for left in range(1, 10):
        for right in range(1, 10):
            assert p_adic_norm(left - right, 11) == (0 if left == right else 1)


def test_conflict_counts_on_known_extremes() -> None:
    assert conflicts_cols_boxes(SOLVED_GRID) == 0
    assert conflicts_all_units(SOLVED_GRID) == 0

    all_ones = [1] * 81
    assert conflicts_cols_boxes(all_ones) == 648
    assert conflicts_all_units(all_ones) == 972


def test_shared_objective_fixture() -> None:
    fixture_path = Path(__file__).parents[1] / "fixtures" / "sudoku_objective_golden.json"
    cases = json.loads(fixture_path.read_text())["cases"]

    for case in cases:
        grid = [int(digit) for digit in case["grid"]]
        assert deduped_peer_conflicts(grid) == case["deduped_peer_conflicts"], case["name"]
        assert conflicts_cols_boxes(grid) == case["column_box_conflicts"], case["name"]
        assert conflicts_all_units(grid) == case["all_unit_conflicts"], case["name"]


def test_unique_generator_is_deterministic() -> None:
    generated_puzzle = generate_unique_puzzle(30, seed=42)

    assert grid_to_string(generated_puzzle) == EXPECTED_SEED_42_PUZZLE
    assert generate_unique_puzzle(30, seed=42) == generated_puzzle
    assert count_solutions(generated_puzzle, limit=2) == 1


def test_mihara_equality_fit_is_exposed_as_a_failed_sudoku_attempt() -> None:
    puzzle = [0 if index % 4 else value for index, value in enumerate(SOLVED_GRID)]
    result = solve_mihara_digitwise_attempt(puzzle, seed=3, trials=4)

    assert not result.solved
    assert len(result.raw_coefficients) == 81
    assert result.fit.inliers > 0
    assert result.fit.total_observations > result.fit.inliers
    assert "inequality rewards" in result.explanation


@pytest.mark.parametrize(
    "solver",
    [
        solve_stepwise_swap,
        solve_greedy_descent_swap,
        solve_zubarev_walk,
        solve_greedy_local_edit_best,
        solve_greedy_local_edit_first,
        solve_zubarev_local_edit,
    ],
)
def test_fixed_seed_solver_smoke_test(solver: Solver, solver_smoke_puzzle: Grid) -> None:
    result = solver(solver_smoke_puzzle, seed=11, max_steps=2_000, restarts=3)

    assert result.solved
    assert result.final_conflicts == 0
    assert is_valid_complete(result.grid)
    assert respects_clues(result.grid, solver_smoke_puzzle)
