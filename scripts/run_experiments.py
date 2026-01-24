#!/usr/bin/env python3
"""
Run p-adic Sudoku lifting experiments.

This script runs the main experiments and generates analysis figures.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from padic_sudoku.puzzle import get_euler_50
from padic_sudoku.heuristics import greedy_best_swap, simulated_annealing
from padic_sudoku.experiment import run_experiment_matrix, save_results_csv, RunResult
from padic_sudoku.padic import required_lift_level

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# Configuration
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

# Primes to test (selected for interesting properties)
EXPERIMENT_PRIMES = [2, 3, 5, 7, 11, 13, 17, 23, 47, 73, 97, 127, 257, 521, 2311]

# Heuristic configurations
HEURISTICS = {
    "greedy": (greedy_best_swap, {"max_steps": 2000}),
    "sa": (simulated_annealing, {"max_steps": 5000, "T_start": 2.0, "cooling_rate": 0.9997}),
}


def run_prime_sweep_experiment() -> pd.DataFrame:
    """
    Experiment E1: Prime sweep with greedy heuristic.

    Test all primes across all puzzles to find which primes work best.
    """
    print("\n" + "=" * 60)
    print("Experiment E1: Prime Sweep (Greedy Heuristic)")
    print("=" * 60)

    puzzles = get_euler_50()
    primes = EXPERIMENT_PRIMES

    print(f"Testing {len(primes)} primes on {len(puzzles)} puzzles")
    print(f"Primes: {primes}")

    heuristics = {"greedy": HEURISTICS["greedy"]}

    pbar = tqdm(total=len(puzzles) * len(primes) * 3, desc="E1 Progress")

    def progress(done, total):
        pbar.update(1)

    results = run_experiment_matrix(
        puzzles=puzzles,
        primes=primes,
        heuristics=heuristics,
        num_inits=3,
        progress_callback=progress,
    )
    pbar.close()

    # Save results
    csv_path = RESULTS_DIR / "e1_prime_sweep.csv"
    save_results_csv(results, csv_path)
    print(f"Saved {len(results)} results to {csv_path}")

    return pd.DataFrame([r.__dict__ for r in results])


def run_heuristic_comparison_experiment(top_primes: list[int]) -> pd.DataFrame:
    """
    Experiment E2: Compare heuristics on top-performing primes.
    """
    print("\n" + "=" * 60)
    print("Experiment E2: Heuristic Comparison")
    print("=" * 60)

    puzzles = get_euler_50()
    print(f"Comparing heuristics on {len(puzzles)} puzzles with primes {top_primes}")

    pbar = tqdm(
        total=len(puzzles) * len(top_primes) * len(HEURISTICS) * 3,
        desc="E2 Progress",
    )

    def progress(done, total):
        pbar.update(1)

    results = run_experiment_matrix(
        puzzles=puzzles,
        primes=top_primes,
        heuristics=HEURISTICS,
        num_inits=3,
        progress_callback=progress,
    )
    pbar.close()

    csv_path = RESULTS_DIR / "e2_heuristic_comparison.csv"
    save_results_csv(results, csv_path)
    print(f"Saved {len(results)} results to {csv_path}")

    return pd.DataFrame([r.__dict__ for r in results])


def analyze_prime_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze prime performance from experiment results."""
    stats = df.groupby("prime").agg(
        success_rate=("solved", "mean"),
        mean_lift_fraction=("lift_fraction", "mean"),
        median_lift_fraction=("lift_fraction", "median"),
        mean_steps=("steps_taken", "mean"),
        mean_time=("wall_time_seconds", "mean"),
        num_runs=("solved", "count"),
    ).reset_index()

    stats["required_level"] = stats["prime"].apply(required_lift_level)
    stats = stats.sort_values("success_rate", ascending=False)

    return stats


def generate_figures(e1_df: pd.DataFrame, e2_df: pd.DataFrame | None = None):
    """Generate analysis figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: Prime success rate bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    stats = analyze_prime_performance(e1_df)

    colors = []
    for p in stats["prime"]:
        if p in [7, 73]:  # Divides 511
            colors.append("orange")
        elif p == 2:  # Interacts with encoding
            colors.append("red")
        elif p > 500:  # Large primes
            colors.append("green")
        else:
            colors.append("steelblue")

    bars = ax.bar(range(len(stats)), stats["success_rate"], color=colors)
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(stats["prime"], rotation=45, ha="right")
    ax.set_xlabel("Prime p")
    ax.set_ylabel("Success Rate")
    ax.set_title("Experiment E1: Prime Success Rate (Greedy Heuristic)")
    ax.set_ylim(0, 1.05)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", label="Regular primes"),
        Patch(facecolor="orange", label="Divides 511 (p=7, 73)"),
        Patch(facecolor="red", label="p=2 (encoding interaction)"),
        Patch(facecolor="green", label="Large primes (p>500)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "e1_prime_success_rate.png", dpi=150)
    plt.close()
    print(f"Saved figure: e1_prime_success_rate.png")

    # Figure 2: Prime vs lift fraction heatmap
    pivot = e1_df.pivot_table(
        values="lift_fraction",
        index="puzzle_id",
        columns="prime",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(0, len(pivot.index), 5))
    ax.set_yticklabels(pivot.index[::5])
    ax.set_xlabel("Prime p")
    ax.set_ylabel("Puzzle")
    ax.set_title("Experiment E1: Lift Fraction by Prime and Puzzle")
    plt.colorbar(im, ax=ax, label="Lift Fraction")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "e1_prime_puzzle_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved figure: e1_prime_puzzle_heatmap.png")

    # Figure 3: Required lift level vs success rate
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(stats["required_level"], stats["success_rate"], s=100, c=colors)
    for i, row in stats.iterrows():
        ax.annotate(
            str(row["prime"]),
            (row["required_level"], row["success_rate"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_xlabel("Required Lift Level k (where p^k > 2304)")
    ax.set_ylabel("Success Rate")
    ax.set_title("Required Lift Level vs Success Rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "e1_lift_level_vs_success.png", dpi=150)
    plt.close()
    print(f"Saved figure: e1_lift_level_vs_success.png")

    # Figure 4: Heuristic comparison (if E2 data available)
    if e2_df is not None and len(e2_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        e2_stats = e2_df.groupby(["prime", "heuristic"])["solved"].mean().unstack()
        e2_stats.plot(kind="bar", ax=ax)
        ax.set_xlabel("Prime p")
        ax.set_ylabel("Success Rate")
        ax.set_title("Experiment E2: Heuristic Comparison")
        ax.legend(title="Heuristic")
        ax.set_xticklabels([str(p) for p in e2_stats.index], rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "e2_heuristic_comparison.png", dpi=150)
        plt.close()
        print(f"Saved figure: e2_heuristic_comparison.png")


def generate_report(e1_df: pd.DataFrame, e2_df: pd.DataFrame | None = None) -> str:
    """Generate a markdown report of the experiment results."""
    stats = analyze_prime_performance(e1_df)

    report = []
    report.append("# p-adic Sudoku Lifting Experiment Results\n")

    report.append("## Experiment E1: Prime Sweep\n")
    report.append("Tested 15 primes on 50 Euler Project puzzles using greedy cell-swap heuristic.\n")
    report.append("Each (puzzle, prime) combination was run 3 times with different random initializations.\n")

    report.append("### Prime Performance Summary\n")
    report.append("| Prime | Success Rate | Mean Lift Frac | Required Level | Mean Steps |")
    report.append("|-------|--------------|----------------|----------------|------------|")
    for _, row in stats.head(15).iterrows():
        report.append(
            f"| {row['prime']:>5} | {row['success_rate']:>12.1%} | "
            f"{row['mean_lift_fraction']:>14.3f} | {row['required_level']:>14} | "
            f"{row['mean_steps']:>10.0f} |"
        )
    report.append("")

    # Key findings
    report.append("### Key Findings\n")

    top_primes = stats.head(5)["prime"].tolist()
    report.append(f"**Top 5 primes by success rate:** {top_primes}\n")

    worst_primes = stats.tail(3)["prime"].tolist()
    report.append(f"**Worst 3 primes:** {worst_primes}\n")

    # Analysis of special primes
    report.append("### Special Prime Analysis\n")

    for p in [2, 7, 73]:
        p_stats = stats[stats["prime"] == p].iloc[0]
        report.append(f"**p = {p}:**")
        if p == 2:
            report.append("- Interacts with power-of-2 encoding (all allowed values except 1 are even)")
        elif p == 7:
            report.append("- Divides 511 (the target sum), so v_7(511) = 1")
        elif p == 73:
            report.append("- Divides 511 (511 = 7 * 73), so v_73(511) = 1")
        report.append(f"- Success rate: {p_stats['success_rate']:.1%}")
        report.append(f"- Mean lift fraction: {p_stats['mean_lift_fraction']:.3f}")
        report.append(f"- Required lift level: {p_stats['required_level']}")
        report.append("")

    # Heuristic comparison
    if e2_df is not None and len(e2_df) > 0:
        report.append("## Experiment E2: Heuristic Comparison\n")
        e2_stats = e2_df.groupby("heuristic").agg(
            success_rate=("solved", "mean"),
            mean_lift_fraction=("lift_fraction", "mean"),
            mean_time=("wall_time_seconds", "mean"),
        )
        report.append("| Heuristic | Success Rate | Mean Lift Frac | Mean Time (s) |")
        report.append("|-----------|--------------|----------------|---------------|")
        for heur, row in e2_stats.iterrows():
            report.append(
                f"| {heur:>9} | {row['success_rate']:>12.1%} | "
                f"{row['mean_lift_fraction']:>14.3f} | {row['mean_time']:>13.2f} |"
            )
        report.append("")

    return "\n".join(report)


def main():
    """Main entry point."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("p-adic Sudoku Lifting Experiments")
    print("=" * 60)

    # Run E1: Prime sweep
    e1_df = run_prime_sweep_experiment()
    stats = analyze_prime_performance(e1_df)
    print("\nPrime Performance (sorted by success rate):")
    print(stats.to_string(index=False))

    # Get top 5 primes for E2
    top_primes = stats.head(5)["prime"].tolist()
    print(f"\nTop 5 primes: {top_primes}")

    # Run E2: Heuristic comparison
    e2_df = run_heuristic_comparison_experiment(top_primes)

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(e1_df, e2_df)

    # Generate report
    print("\nGenerating report...")
    report = generate_report(e1_df, e2_df)

    report_path = Path(__file__).parent.parent / "RESULTS.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report to {report_path}")

    print("\n" + "=" * 60)
    print("Experiments complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
