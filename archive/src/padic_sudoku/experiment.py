"""Experiment runner for p-adic Sudoku lifting."""

from __future__ import annotations
import random
import time
from dataclasses import dataclass, asdict
from typing import Callable
import csv
import json
from pathlib import Path

from .puzzle import Puzzle
from .heuristics import LiftResult, greedy_best_swap, simulated_annealing
from .padic import required_lift_level


@dataclass
class RunResult:
    """Complete result of a single experiment run."""

    puzzle_id: str
    prime: int
    heuristic: str
    init_seed: int

    solved: bool
    final_min_valuation: float  # int or inf
    required_valuation: int
    lift_fraction: float  # final / required (capped at 1.0 for solved)
    steps_taken: int
    wall_time_seconds: float


def run_single(
    puzzle: Puzzle,
    prime: int,
    heuristic_name: str,
    heuristic_fn: Callable[..., LiftResult],
    seed: int,
    **heuristic_kwargs,
) -> RunResult:
    """Run a single (puzzle, prime, heuristic, seed) experiment."""
    rng = random.Random(seed)
    start_time = time.time()

    result = heuristic_fn(puzzle, prime, rng=rng, **heuristic_kwargs)

    wall_time = time.time() - start_time
    req = required_lift_level(prime)

    # Compute lift fraction
    if result.final_min_valuation == float("inf"):
        lift_frac = 1.0
    else:
        lift_frac = min(result.final_min_valuation / req, 1.0)

    return RunResult(
        puzzle_id=puzzle.puzzle_id,
        prime=prime,
        heuristic=heuristic_name,
        init_seed=seed,
        solved=result.solved,
        final_min_valuation=result.final_min_valuation if result.final_min_valuation != float("inf") else 999,
        required_valuation=req,
        lift_fraction=lift_frac,
        steps_taken=result.steps_taken,
        wall_time_seconds=wall_time,
    )


def run_experiment_matrix(
    puzzles: list[Puzzle],
    primes: list[int],
    heuristics: dict[str, tuple[Callable[..., LiftResult], dict]],
    num_inits: int = 3,
    base_seed: int = 42,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[RunResult]:
    """
    Run the full experiment matrix.

    Args:
        puzzles: List of puzzles to test
        primes: List of primes to test
        heuristics: Dict mapping heuristic name to (function, kwargs)
        num_inits: Number of random initializations per (puzzle, prime, heuristic)
        base_seed: Base random seed for reproducibility
        progress_callback: Optional callback(completed, total) for progress updates

    Returns:
        List of RunResult objects
    """
    results = []
    total = len(puzzles) * len(primes) * len(heuristics) * num_inits
    completed = 0

    for puzzle in puzzles:
        for prime in primes:
            for heuristic_name, (heuristic_fn, kwargs) in heuristics.items():
                for init_idx in range(num_inits):
                    seed = base_seed + hash((puzzle.puzzle_id, prime, heuristic_name, init_idx)) % (2**31)
                    result = run_single(puzzle, prime, heuristic_name, heuristic_fn, seed, **kwargs)
                    results.append(result)

                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)

    return results


def save_results_csv(results: list[RunResult], path: Path) -> None:
    """Save results to a CSV file."""
    if not results:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def load_results_csv(path: Path) -> list[RunResult]:
    """Load results from a CSV file."""
    results = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(
                RunResult(
                    puzzle_id=row["puzzle_id"],
                    prime=int(row["prime"]),
                    heuristic=row["heuristic"],
                    init_seed=int(row["init_seed"]),
                    solved=row["solved"] == "True",
                    final_min_valuation=float(row["final_min_valuation"]),
                    required_valuation=int(row["required_valuation"]),
                    lift_fraction=float(row["lift_fraction"]),
                    steps_taken=int(row["steps_taken"]),
                    wall_time_seconds=float(row["wall_time_seconds"]),
                )
            )
    return results
