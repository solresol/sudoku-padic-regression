"""Lifting heuristics for p-adic Sudoku solving."""

from __future__ import annotations
import random
import math
from dataclasses import dataclass, field
from typing import Callable

from .encoding import ALLOWED_VALUES, CONSTRAINT_GROUPS, get_cell_groups
from .padic import (
    min_valuation,
    total_padic_loss,
    required_lift_level,
    v_p,
    residual,
    is_valid_sudoku,
)
from .puzzle import Puzzle


@dataclass
class LiftResult:
    """Result of a lifting attempt."""

    solved: bool
    final_min_valuation: int | float
    required_valuation: int
    steps_taken: int
    final_grid: list[list[int]]
    valuation_trajectory: list[int | float] = field(default_factory=list)
    loss_trajectory: list[float] = field(default_factory=list)


def initialize_random(puzzle: Puzzle, rng: random.Random) -> list[list[int]]:
    """
    Initialize non-clue cells randomly from ALLOWED_VALUES.

    Returns a copy of the grid with unfilled cells randomly assigned.
    """
    grid = puzzle.copy_grid()
    for r in range(9):
        for c in range(9):
            if (r, c) not in puzzle.clues:
                grid[r][c] = rng.choice(ALLOWED_VALUES)
    return grid


def greedy_best_swap(
    puzzle: Puzzle,
    p: int,
    max_steps: int = 2000,
    rng: random.Random | None = None,
    track_every: int = 10,
) -> LiftResult:
    """
    Greedy single-cell swap heuristic.

    At each step, find the single non-clue cell change that most improves
    the p-adic objective (maximize min_valuation, break ties by minimizing loss).
    """
    if rng is None:
        rng = random.Random()

    grid = initialize_random(puzzle, rng)
    req_level = required_lift_level(p)

    val_trajectory = []
    loss_trajectory = []

    non_clue_cells = [(r, c) for r in range(9) for c in range(9) if (r, c) not in puzzle.clues]

    for step in range(max_steps):
        current_min_val = min_valuation(grid, p)
        current_loss = total_padic_loss(grid, p)

        # Track trajectory
        if step % track_every == 0:
            val_trajectory.append(current_min_val)
            loss_trajectory.append(current_loss)

        # Check if solved
        if current_min_val == float("inf"):
            return LiftResult(
                solved=True,
                final_min_valuation=current_min_val,
                required_valuation=req_level,
                steps_taken=step,
                final_grid=grid,
                valuation_trajectory=val_trajectory,
                loss_trajectory=loss_trajectory,
            )

        # Find the best single-cell change
        best_improvement = None
        best_cell = None
        best_value = None

        for r, c in non_clue_cells:
            old_val = grid[r][c]
            for new_val in ALLOWED_VALUES:
                if new_val == old_val:
                    continue

                # Temporarily make the change
                grid[r][c] = new_val
                new_min_val = min_valuation(grid, p)
                new_loss = total_padic_loss(grid, p)
                grid[r][c] = old_val  # Restore

                # Compare: prefer higher min_val, then lower loss
                improvement = (new_min_val, -new_loss)
                current = (current_min_val, -current_loss)

                if improvement > current:
                    if best_improvement is None or improvement > best_improvement:
                        best_improvement = improvement
                        best_cell = (r, c)
                        best_value = new_val

        # Apply the best change (if any improvement found)
        if best_cell is not None:
            grid[best_cell[0]][best_cell[1]] = best_value
        else:
            # No improvement possible - stuck
            break

    final_min_val = min_valuation(grid, p)
    val_trajectory.append(final_min_val)
    loss_trajectory.append(total_padic_loss(grid, p))

    return LiftResult(
        solved=final_min_val == float("inf"),
        final_min_valuation=final_min_val,
        required_valuation=req_level,
        steps_taken=step + 1 if "step" in dir() else 0,
        final_grid=grid,
        valuation_trajectory=val_trajectory,
        loss_trajectory=loss_trajectory,
    )


def simulated_annealing(
    puzzle: Puzzle,
    p: int,
    max_steps: int = 5000,
    T_start: float = 2.0,
    T_end: float = 0.01,
    cooling_rate: float = 0.9997,
    rng: random.Random | None = None,
    track_every: int = 20,
) -> LiftResult:
    """
    Simulated annealing with p-adic energy function.

    Energy = -min_valuation + alpha * total_loss
    Lower energy is better (we maximize valuation and minimize loss).
    """
    if rng is None:
        rng = random.Random()

    grid = initialize_random(puzzle, rng)
    req_level = required_lift_level(p)
    alpha = 0.1  # Weight for total_loss component

    val_trajectory = []
    loss_trajectory = []

    non_clue_cells = [(r, c) for r in range(9) for c in range(9) if (r, c) not in puzzle.clues]
    T = T_start

    def energy(g: list[list[int]]) -> float:
        mv = min_valuation(g, p)
        if mv == float("inf"):
            return float("-inf")  # Solved - lowest possible energy
        tl = total_padic_loss(g, p)
        return -mv + alpha * tl

    current_energy = energy(grid)
    best_grid = [row[:] for row in grid]
    best_min_val = min_valuation(grid, p)

    for step in range(max_steps):
        # Track trajectory
        if step % track_every == 0:
            val_trajectory.append(min_valuation(grid, p))
            loss_trajectory.append(total_padic_loss(grid, p))

        # Check if solved
        if current_energy == float("-inf"):
            return LiftResult(
                solved=True,
                final_min_valuation=float("inf"),
                required_valuation=req_level,
                steps_taken=step,
                final_grid=grid,
                valuation_trajectory=val_trajectory,
                loss_trajectory=loss_trajectory,
            )

        # Propose a random change
        r, c = rng.choice(non_clue_cells)
        old_val = grid[r][c]
        new_val = rng.choice([v for v in ALLOWED_VALUES if v != old_val])

        # Compute new energy
        grid[r][c] = new_val
        new_energy = energy(grid)

        # Accept or reject
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_energy = new_energy
            # Track best seen
            mv = min_valuation(grid, p)
            if mv > best_min_val:
                best_min_val = mv
                best_grid = [row[:] for row in grid]
        else:
            grid[r][c] = old_val  # Reject - restore

        # Cool down
        T = max(T * cooling_rate, T_end)

    # Return best found
    final_min_val = min_valuation(best_grid, p)
    val_trajectory.append(final_min_val)
    loss_trajectory.append(total_padic_loss(best_grid, p))

    return LiftResult(
        solved=final_min_val == float("inf"),
        final_min_valuation=final_min_val,
        required_valuation=req_level,
        steps_taken=max_steps,
        final_grid=best_grid,
        valuation_trajectory=val_trajectory,
        loss_trajectory=loss_trajectory,
    )


def multi_prime_sequential(
    puzzle: Puzzle,
    primes: list[int],
    heuristic: Callable[..., LiftResult],
    steps_per_prime: int = 1000,
    rng: random.Random | None = None,
) -> LiftResult:
    """
    Try each prime in sequence.

    If one prime gets stuck, switch to the next using the current grid state.
    """
    if rng is None:
        rng = random.Random()

    # Start with first prime
    grid = initialize_random(puzzle, rng)
    total_steps = 0
    val_trajectory = []
    loss_trajectory = []

    for p in primes:
        # Create a temporary puzzle with current grid as starting point
        temp_puzzle = Puzzle(
            grid=grid,
            clues=puzzle.clues,
            solution=puzzle.solution,
            puzzle_id=puzzle.puzzle_id,
        )

        result = heuristic(temp_puzzle, p, max_steps=steps_per_prime, rng=rng)
        grid = result.final_grid
        total_steps += result.steps_taken
        val_trajectory.extend(result.valuation_trajectory)
        loss_trajectory.extend(result.loss_trajectory)

        if result.solved:
            return LiftResult(
                solved=True,
                final_min_valuation=float("inf"),
                required_valuation=required_lift_level(primes[0]),
                steps_taken=total_steps,
                final_grid=grid,
                valuation_trajectory=val_trajectory,
                loss_trajectory=loss_trajectory,
            )

    # Return final state after all primes
    final_p = primes[0]  # Report valuation w.r.t. first prime
    final_min_val = min_valuation(grid, final_p)

    return LiftResult(
        solved=is_valid_sudoku(grid),
        final_min_valuation=final_min_val,
        required_valuation=required_lift_level(final_p),
        steps_taken=total_steps,
        final_grid=grid,
        valuation_trajectory=val_trajectory,
        loss_trajectory=loss_trajectory,
    )
