
#!/usr/bin/env python3
"""
run_experiments.py

Reproducible experiments for the "Sudoku the Hard Way" p-adic regression solver.

Default behaviour is *fast*: puzzles are produced by carving random solved grids down to a
given clue count (no uniqueness guarantee). This keeps runtimes sane.

Use --unique to enforce unique solutions (slower, but closer to "real" Sudoku).
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
import random
import statistics

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover
    plt = None

from padic_sudoku_regression import (
    generate_unique_puzzle,
    random_solved_grid,
    grid_to_string,
    parse_puzzle,
    solve_stepwise_swap,
    solve_greedy_descent_swap,
    solve_greedy_local_edit_best,
    solve_greedy_local_edit_first,
    solve_zubarev_local_edit,
    solve_zubarev_walk,
    pretty,
)

def carve_fast(solved, clues: int, rng: random.Random):
    puzzle = solved[:]
    positions = list(range(81))
    rng.shuffle(positions)
    to_remove = 81 - clues
    for pos in positions[:to_remove]:
        puzzle[pos] = 0
    return puzzle

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default=str(Path(__file__).resolve().parent.parent / "outputs"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=12, help="puzzles per clue count")
    ap.add_argument("--clues", type=str, default="36,30,26", help="comma-separated clue counts")
    ap.add_argument("--max-steps", type=int, default=200000)
    ap.add_argument("--restarts", type=int, default=30)
    ap.add_argument(
        "--method",
        type=str,
        default="stepwise",
        choices=["stepwise", "greedy", "zubarev", "local-best", "local-first", "local-zubarev"],
    )
    ap.add_argument("--beta0", type=float, default=0.5, help="Zubarev walk: initial beta (inverse temperature).")
    ap.add_argument("--beta1", type=float, default=6.0, help="Zubarev walk: final beta (ignored if schedule=constant).")
    ap.add_argument("--beta-schedule", type=str, default="linear", choices=["constant", "linear", "exp"])
    ap.add_argument("--unique", action="store_true", help="enforce uniqueness (slower)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    clue_counts = [int(x.strip()) for x in args.clues.split(",") if x.strip()]
    rng = random.Random(args.seed)

    results_rows = []

    # We'll capture a trace for the first puzzle in the middle clue count (if available).
    trace_target = clue_counts[len(clue_counts)//2]
    trace_saved = False

    t0 = time.time()

    for clues in clue_counts:
        for j in range(args.n):
            seed = rng.randrange(1_000_000_000)

            if args.unique:
                puzzle = generate_unique_puzzle(clues=clues, seed=seed)
            else:
                solved = random_solved_grid(random.Random(seed))
                puzzle = carve_fast(solved, clues=clues, rng=random.Random(seed ^ 0xDEADBEEF))

            puzzle_str = grid_to_string(puzzle)

            record_trace = (not trace_saved) and (clues == trace_target) and (j == 0)
            if args.method == "stepwise":
                res = solve_stepwise_swap(
                    puzzle,
                    seed=seed ^ 0xA5A5A5A5,
                    max_steps=args.max_steps,
                    restarts=args.restarts,
                    record_trace=record_trace,
                    trace_every=200,
                )
            elif args.method == "greedy":
                res = solve_greedy_descent_swap(
                    puzzle,
                    seed=seed ^ 0xA5A5A5A5,
                    max_steps=args.max_steps,
                    restarts=args.restarts,
                    record_trace=record_trace,
                    trace_every=200,
                )
            elif args.method == "local-best":
                res = solve_greedy_local_edit_best(
                    puzzle,
                    seed=seed ^ 0xA5A5A5A5,
                    max_steps=args.max_steps,
                    restarts=args.restarts,
                    record_trace=record_trace,
                    trace_every=200,
                )
            elif args.method == "local-first":
                res = solve_greedy_local_edit_first(
                    puzzle,
                    seed=seed ^ 0xA5A5A5A5,
                    max_steps=args.max_steps,
                    restarts=args.restarts,
                    record_trace=record_trace,
                    trace_every=200,
                )
            elif args.method == "local-zubarev":
                res = solve_zubarev_local_edit(
                    puzzle,
                    seed=seed ^ 0xA5A5A5A5,
                    max_steps=args.max_steps,
                    restarts=args.restarts,
                    beta0=args.beta0,
                    beta1=args.beta1,
                    beta_schedule=args.beta_schedule,
                    record_trace=record_trace,
                    trace_every=200,
                )
            else:
                res = solve_zubarev_walk(
                    puzzle,
                    seed=seed ^ 0xA5A5A5A5,
                    max_steps=args.max_steps,
                    restarts=args.restarts,
                    beta0=args.beta0,
                    beta1=args.beta1,
                    beta_schedule=args.beta_schedule,
                    record_trace=record_trace,
                    trace_every=200,
                )

            results_rows.append({
                "clues": clues,
                "puzzle_seed": seed,
                "solve_seed": seed ^ 0xA5A5A5A5,
                "unique_enforced": int(args.unique),
                "method": args.method,
                "objective": res.objective_label,
                "puzzle": puzzle_str,
                "solved": int(res.solved),
                "steps": res.steps,
                "restarts_used": res.restarts,
                "seconds": res.seconds,
                "final_conflicts": res.final_conflicts,
            })

            if record_trace and res.trace is not None:
                # Save trace plot
                if plt is None:
                    print("Note: matplotlib not installed; skipping loss_curve plot.")
                else:
                    fig = plt.figure()
                    x_values = res.trace_steps if res.trace_steps is not None else [200 * k for k in range(len(res.trace))]
                    plt.plot(x_values, res.trace)
                    plt.xlabel("Iteration (approx.)")
                    plt.ylabel(f"{res.objective_label} conflict pairs")
                    plt.title(f"Loss trajectory (clues={clues}, seed={seed})")
                    fig.tight_layout()
                    png_path = outdir / "loss_curve.png"
                    pdf_path = outdir / "loss_curve.pdf"
                    fig.savefig(png_path, dpi=200)
                    fig.savefig(pdf_path)
                    plt.close(fig)

                # also save the puzzle and (if solved) its solution
                (outdir / "trace_puzzle.txt").write_text(pretty(parse_puzzle(puzzle_str)))
                if res.solved:
                    (outdir / "trace_solution.txt").write_text(pretty(res.grid))
                trace_saved = True

    # Write CSV
    csv_path = outdir / "experiment_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
        w.writeheader()
        for row in results_rows:
            w.writerow(row)

    # Summaries
    summary_lines = []
    summary_lines.append(f"Total runs: {len(results_rows)}")
    summary_lines.append(f"Unique enforced: {args.unique}")
    for clues in clue_counts:
        subset = [r for r in results_rows if r["clues"] == clues]
        solved = [r for r in subset if r["solved"] == 1]
        summary_lines.append(f"\nClues={clues}: solved {len(solved)}/{len(subset)}")
        if solved:
            steps = [r["steps"] for r in solved]
            secs = [r["seconds"] for r in solved]
            summary_lines.append(f"  median steps: {int(statistics.median(steps))}")
            summary_lines.append(f"  median seconds: {statistics.median(secs):.3f}")
    (outdir / "summary.txt").write_text("\n".join(summary_lines))

    dt = time.time() - t0
    print("Wrote", csv_path)
    print("Wall time:", f"{dt:.2f}s")
    print((outdir / "summary.txt").read_text())

if __name__ == "__main__":
    main()
