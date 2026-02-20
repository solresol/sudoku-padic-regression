#!/usr/bin/env python3
"""
make_experiment_report.py

Generate a lightweight, self-contained experiment report (Markdown + plots) from CSV outputs.

This script is intentionally dependency-light: it uses only the stdlib + matplotlib (already
used elsewhere in this repo).

Expected directory layout (default):
  outputs/<YYYY-MM-DD>/
    padic_lr_v3/mc_*.csv   (preferred; deduped-hyperplane, multi-policy runs)
    padic_lr_v2/mc_*.csv   (older multi-policy format)
    padic_lr/mc_*.csv      (older single-policy format)
    sudoku/*/experiment_results.csv
    analysis/              (generated)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt  # type: ignore


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return statistics.stdev(xs)


def _stderr(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return _stdev(xs) / math.sqrt(len(xs))


def _maybe_int(s: str) -> Optional[int]:
    s = (s or "").strip()
    if not s:
        return None
    return int(s)


def _maybe_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    return float(s)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


@dataclass(frozen=True)
class PadicKey:
    p: int
    noise_model: str
    noise_k0: int
    policy: str
    temperature: Optional[float]


@dataclass
class PadicSummary:
    key: PadicKey
    trials: int
    global_hit_mean: float
    global_hit_stderr: float
    avg_steps_mean: float
    local_minima_mean: float
    bad_local_minima_mean: float
    all_local_minima_global_rate: float
    true_global_rate: float
    seconds_mean: float


def summarize_padic(mc_paths: Iterable[Path]) -> Tuple[List[PadicSummary], List[Dict[str, Any]]]:
    rows_all: List[Dict[str, Any]] = []
    by_key: Dict[PadicKey, List[Dict[str, Any]]] = {}

    for path in sorted(mc_paths):
        rows = _read_csv(path)
        for row in rows:
            p = int(row["p"])
            noise_model = row["noise_model"]
            noise_k0 = int(row["noise_k0"])
            policy = row.get("policy", "steepest") or "steepest"
            temperature = _maybe_float(row.get("temperature", ""))
            key = PadicKey(p=p, noise_model=noise_model, noise_k0=noise_k0, policy=policy, temperature=temperature)

            local_minima = _maybe_int(row.get("local_minima", ""))
            bad_local_minima = _maybe_int(row.get("bad_local_minima", ""))
            all_local_minima_global = _maybe_int(row.get("all_local_minima_global", ""))
            global_hit_prob = _maybe_float(row.get("global_hit_prob", row.get("basin_fraction_global", "")))
            avg_steps = _maybe_float(row.get("avg_steps", row.get("avg_greedy_steps", "")))
            true_is_global = _maybe_int(row.get("true_is_global", ""))
            seconds = _maybe_float(row.get("seconds", ""))

            d2: Dict[str, Any] = {
                **row,
                "path": str(path),
                "p": p,
                "noise_model": noise_model,
                "noise_k0": noise_k0,
                "policy": policy,
                "temperature": temperature,
                "local_minima": local_minima,
                "bad_local_minima": bad_local_minima,
                "all_local_minima_global": all_local_minima_global,
                "global_hit_prob": global_hit_prob,
                "avg_steps": avg_steps,
                "true_is_global": true_is_global,
                "seconds": seconds,
            }
            rows_all.append(d2)
            by_key.setdefault(key, []).append(d2)

    summaries: List[PadicSummary] = []
    for key, xs in sorted(by_key.items(), key=lambda kv: (kv[0].p, kv[0].noise_model, kv[0].noise_k0, kv[0].policy, kv[0].temperature or -1.0)):
        hits = [x["global_hit_prob"] for x in xs if x["global_hit_prob"] is not None]
        steps = [x["avg_steps"] for x in xs if x["avg_steps"] is not None]
        locals_ = [x["local_minima"] for x in xs if x["local_minima"] is not None]
        bad_locals = [x["bad_local_minima"] for x in xs if x["bad_local_minima"] is not None]
        all_globals = [x["all_local_minima_global"] for x in xs if x["all_local_minima_global"] is not None]
        trues = [x["true_is_global"] for x in xs if x["true_is_global"] is not None]
        secs = [x["seconds"] for x in xs if x["seconds"] is not None]

        summaries.append(
            PadicSummary(
                key=key,
                trials=len(xs),
                global_hit_mean=_mean(hits),
                global_hit_stderr=_stderr(hits),
                avg_steps_mean=_mean(steps),
                local_minima_mean=_mean([float(v) for v in locals_]) if locals_ else float("nan"),
                bad_local_minima_mean=_mean([float(v) for v in bad_locals]) if bad_locals else float("nan"),
                all_local_minima_global_rate=_mean([float(v) for v in all_globals]) if all_globals else float("nan"),
                true_global_rate=_mean([float(t) for t in trues]) if trues else float("nan"),
                seconds_mean=_mean(secs),
            )
        )

    return summaries, rows_all


@dataclass(frozen=True)
class SudokuKey:
    label: str  # e.g. "fast_greedy_r1"
    method: str
    restarts: int


def _infer_sudoku_key_from_dir(dir_name: str) -> Optional[SudokuKey]:
    # Expected patterns used in this repo: fast_<method>_r<restarts>
    parts = dir_name.split("_")
    if len(parts) < 3:
        return None
    if parts[0] != "fast":
        return None
    method = parts[1]
    rpart = parts[2]
    if not rpart.startswith("r"):
        return None
    try:
        restarts = int(rpart[1:])
    except ValueError:
        return None
    return SudokuKey(label=dir_name, method=method, restarts=restarts)


def summarize_sudoku(results_paths: Iterable[Path]) -> Dict[SudokuKey, Dict[int, Dict[str, float]]]:
    """
    Returns:
      summaries[key][clues] = {
        "n": ...,
        "solved_rate": ...,
        "median_steps_solved": ... (nan if none),
        "median_seconds_solved": ...,
        "median_final_conflicts_unsolved": ...
      }
    """
    out: Dict[SudokuKey, Dict[int, Dict[str, float]]] = {}
    for path in sorted(results_paths):
        key = _infer_sudoku_key_from_dir(path.parent.name)
        if key is None:
            continue
        rows = _read_csv(path)
        by_clues: Dict[int, List[Dict[str, str]]] = {}
        for row in rows:
            c = int(row["clues"])
            by_clues.setdefault(c, []).append(row)

        summary_by_c: Dict[int, Dict[str, float]] = {}
        for c, xs in sorted(by_clues.items()):
            n = len(xs)
            solved = [r for r in xs if int(r["solved"]) == 1]
            unsolved = [r for r in xs if int(r["solved"]) == 0]
            solved_rate = len(solved) / n if n else float("nan")

            def _median_float(field: str, rows2: List[Dict[str, str]]) -> float:
                if not rows2:
                    return float("nan")
                vs = [float(r[field]) for r in rows2]
                return float(statistics.median(vs))

            def _median_int(field: str, rows2: List[Dict[str, str]]) -> float:
                if not rows2:
                    return float("nan")
                vs = [int(r[field]) for r in rows2]
                return float(statistics.median(vs))

            summary_by_c[c] = {
                "n": float(n),
                "solved_rate": solved_rate,
                "median_steps_solved": _median_int("steps", solved),
                "median_seconds_solved": _median_float("seconds", solved),
                "median_final_conflicts_unsolved": _median_int("final_conflicts", unsolved),
            }

        out[key] = summary_by_c
    return out


def _git_rev(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception:
        return "unknown"


def _write_metadata(path: Path, repo_root: Path) -> None:
    meta = {
        "git_head": _git_rev(repo_root),
        "python": sys.version,
        "argv": sys.argv,
    }
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def plot_padic_basin(summaries: List[PadicSummary], outdir: Path) -> Path:
    fig = plt.figure()
    ax = fig.gca()

    # For multi-policy runs, match the old plot by using the steepest-descent policy.
    summaries = [s for s in summaries if s.key.policy == "steepest"]

    # Group by (p, model)
    groups: Dict[Tuple[int, str], List[PadicSummary]] = {}
    for s in summaries:
        groups.setdefault((s.key.p, s.key.noise_model), []).append(s)
    for (p, model), xs in sorted(groups.items()):
        xs2 = sorted(xs, key=lambda t: t.key.noise_k0)
        k0s = [t.key.noise_k0 for t in xs2]
        means = [t.global_hit_mean for t in xs2]
        errs = [t.global_hit_stderr for t in xs2]
        label = f"p={p}, {model}"
        ax.errorbar(k0s, means, yerr=errs, marker="o", capsize=3, label=label)

    ax.set_xlabel("noise_k0 (valuation shift; larger = p-adically smaller noise)")
    ax.set_ylabel("mean global-hit probability (steepest descent)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    png = outdir / "padic_lr_basin_fraction.png"
    pdf = outdir / "padic_lr_basin_fraction.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    return png


def plot_padic_true_global(summaries: List[PadicSummary], outdir: Path) -> Path:
    fig = plt.figure()
    ax = fig.gca()

    summaries = [s for s in summaries if s.key.policy == "steepest"]

    groups: Dict[Tuple[int, str], List[PadicSummary]] = {}
    for s in summaries:
        groups.setdefault((s.key.p, s.key.noise_model), []).append(s)
    for (p, model), xs in sorted(groups.items()):
        xs2 = sorted(xs, key=lambda t: t.key.noise_k0)
        k0s = [t.key.noise_k0 for t in xs2]
        means = [t.true_global_rate for t in xs2]
        label = f"p={p}, {model}"
        ax.plot(k0s, means, marker="o", label=label)

    ax.set_xlabel("noise_k0")
    ax.set_ylabel("P(true beta is globally optimal)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    png = outdir / "padic_lr_true_is_global.png"
    pdf = outdir / "padic_lr_true_is_global.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    return png


def plot_padic_policy_noiseless_vs_noisy(
    summaries: List[PadicSummary],
    outdir: Path,
    *,
    noisy_model: str = "haar",
    noisy_k0: int = 0,
    softmax_temperature: float = 1.0,
) -> Optional[Path]:
    """
    Bar chart: compare global-hit probability in the noiseless setting (noise_model=none)
    vs a baseline noisy setting (default: haar, k0=0), for each policy and p.
    """
    summ_by_key: Dict[PadicKey, PadicSummary] = {s.key: s for s in summaries}
    ps = sorted({s.key.p for s in summaries})
    if not ps:
        return None

    policies = ["steepest", "uniform", "proportional", "softmax"]

    fig, axes = plt.subplots(1, len(ps), figsize=(5.0 * len(ps), 3.4), squeeze=False)
    axes = axes[0]

    for ax, p in zip(axes, ps):
        xs = list(range(len(policies)))
        w = 0.38

        noiseless_means: List[float] = []
        noiseless_errs: List[float] = []
        noisy_means: List[float] = []
        noisy_errs: List[float] = []

        for pol in policies:
            t = softmax_temperature if pol == "softmax" else None
            k_none = PadicKey(p=p, noise_model="none", noise_k0=0, policy=pol, temperature=t)
            k_noisy = PadicKey(p=p, noise_model=noisy_model, noise_k0=noisy_k0, policy=pol, temperature=t)
            s_none = summ_by_key.get(k_none)
            s_noisy = summ_by_key.get(k_noisy)
            if s_none is None or s_noisy is None:
                # Missing data; use NaNs so the plot is still generated.
                noiseless_means.append(float("nan"))
                noiseless_errs.append(0.0)
                noisy_means.append(float("nan"))
                noisy_errs.append(0.0)
                continue
            noiseless_means.append(s_none.global_hit_mean)
            noiseless_errs.append(s_none.global_hit_stderr)
            noisy_means.append(s_noisy.global_hit_mean)
            noisy_errs.append(s_noisy.global_hit_stderr)

        ax.bar([x - w / 2 for x in xs], noiseless_means, width=w, yerr=noiseless_errs, capsize=3, label="no noise")
        ax.bar([x + w / 2 for x in xs], noisy_means, width=w, yerr=noisy_errs, capsize=3, label=f"{noisy_model}, k0={noisy_k0}")
        ax.set_title(f"p={p}")
        ax.set_xticks(xs)
        ax.set_xticklabels(policies, rotation=15, ha="right")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylabel("mean global-hit probability")
        ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()

    png = outdir / "padic_lr_policy_noiseless_vs_noisy.png"
    pdf = outdir / "padic_lr_policy_noiseless_vs_noisy.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    return png


def plot_sudoku_solve_rate(summaries: Dict[SudokuKey, Dict[int, Dict[str, float]]], outdir: Path) -> Optional[Path]:
    if not summaries:
        return None
    fig = plt.figure()
    ax = fig.gca()

    for key in sorted(summaries.keys(), key=lambda k: (k.method, k.restarts)):
        by_c = summaries[key]
        clues = sorted(by_c.keys(), reverse=True)
        rates = [by_c[c]["solved_rate"] for c in clues]
        ax.plot(clues, rates, marker="o", label=f"{key.method} (r={key.restarts})")

    ax.set_xlabel("clues (higher usually easier)")
    ax.set_ylabel("solve rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    png = outdir / "sudoku_solve_rate.png"
    pdf = outdir / "sudoku_solve_rate.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    return png


def write_markdown_report(
    out_path: Path,
    *,
    base_dir: Path,
    base_label: str,
    padic_source_dir: str,
    padic_summaries: List[PadicSummary],
    padic_rows: List[Dict[str, Any]],
    sudoku_summaries: Dict[SudokuKey, Dict[int, Dict[str, float]]],
) -> None:
    rel = lambda p: str(p.relative_to(base_dir)) if p.is_absolute() is False else str(p)

    lines: List[str] = []
    lines.append(f"# Experiment results ({base_dir.name})")
    lines.append("")
    lines.append(f"This report is generated by `code/make_experiment_report.py` from CSV outputs in `{base_label}/`.")
    lines.append("")

    # p-adic regression summary
    lines.append("## p-adic linear regression: neighbour-hyperplane descent policies")
    lines.append("")
    lines.append("Setup:")
    lines.append("- Synthetic linear model: `y = beta0 + beta·x + noise` (integers).")
    lines.append("- Ground-truth coefficients are sampled as $p$-powers (default in `padic_linear_regression.py`: `--coef-model p_power`).")
    lines.append("- Enumerate all hyperplanes through `d+1` points (here: `d=3`, so 4 points), then **deduplicate** identical hyperplanes (same coefficient vector).")
    lines.append("- Loss: `L1` sum of `p`-adic norms of residuals (exact rationals).")
    lines.append("- Neighbour edges are induced by swapping exactly one defining point (one index in a `(d+1)`-subset).")
    lines.append("- Descent process repeats an improving neighbour move until a local minimum (no improving neighbours).")
    lines.append("")

    if padic_summaries:
        # Policy comparison plot (noiseless vs baseline noisy)
        lines.append("Policy comparison (noiseless vs baseline noisy):")
        lines.append("")
        lines.append("![policy comparison](padic_lr_policy_noiseless_vs_noisy.png)")
        lines.append("")

        # Pivot table: for each p and noise config, show mean±stderr global-hit probability by policy.
        summ_by_key: Dict[PadicKey, PadicSummary] = {s.key: s for s in padic_summaries}
        ps = sorted({s.key.p for s in padic_summaries})
        policies = ["steepest", "uniform", "proportional", "softmax"]
        softmax_t = 1.0
        if any(s.key.policy == "softmax" and s.key.temperature is not None for s in padic_summaries):
            softmax_t = sorted({s.key.temperature for s in padic_summaries if s.key.policy == "softmax" and s.key.temperature is not None})[0]  # type: ignore[arg-type]

        lines.append("Global-hit probability by policy (mean ± stderr over trials):")
        lines.append("")
        for p in ps:
            lines.append(f"### p={p}")
            lines.append("")
            header = "| noise model | noise_k0 | " + " | ".join(
                ["steepest", "uniform", "proportional", f"softmax(T={softmax_t:g})"]
            ) + " |"
            sep = "|---|---:|" + "|".join(["---:"] * 4) + "|"
            lines.append(header)
            lines.append(sep)

            # Collect all (noise_model, k0) pairs for this p (stable order).
            configs = sorted({(s.key.noise_model, s.key.noise_k0) for s in padic_summaries if s.key.p == p},
                             key=lambda t: (t[0] != "none", t[0], t[1]))
            for noise_model, k0 in configs:
                cells: List[str] = []
                for pol in policies:
                    t = softmax_t if pol == "softmax" else None
                    s = summ_by_key.get(PadicKey(p=p, noise_model=noise_model, noise_k0=k0, policy=pol, temperature=t))
                    if s is None or math.isnan(s.global_hit_mean):
                        cells.append("")
                    else:
                        cells.append(f"{s.global_hit_mean:.3f} ± {s.global_hit_stderr:.3f}")
                lines.append(f"| {noise_model} | {k0} | " + " | ".join(cells) + " |")
            lines.append("")

        # Simple statistical comparison: noiseless vs haar(k0=0) baseline.
        def _vals(p: int, noise_model: str, noise_k0: int, policy: str, temperature: Optional[float]) -> List[float]:
            out: List[float] = []
            for r in padic_rows:
                if r.get("p") != p:
                    continue
                if r.get("noise_model") != noise_model:
                    continue
                if r.get("noise_k0") != noise_k0:
                    continue
                if r.get("policy") != policy:
                    continue
                if policy == "softmax":
                    if r.get("temperature") != temperature:
                        continue
                v = r.get("global_hit_prob")
                if isinstance(v, (int, float)) and math.isfinite(v):
                    out.append(float(v))
            return out

        def _bootstrap_diff_mean(xs: List[float], ys: List[float], n_boot: int = 5000, seed: int = 0) -> Tuple[float, float, float]:
            rng = random.Random(seed)
            if not xs or not ys:
                return float("nan"), float("nan"), float("nan")
            diffs: List[float] = []
            nx = len(xs)
            ny = len(ys)
            for _ in range(n_boot):
                mx = sum(rng.choice(xs) for _ in range(nx)) / nx
                my = sum(rng.choice(ys) for _ in range(ny)) / ny
                diffs.append(mx - my)
            diffs.sort()
            lo = diffs[int(0.025 * n_boot)]
            hi = diffs[int(0.975 * n_boot)]
            return (_mean(xs) - _mean(ys), lo, hi)

        lines.append("Noiseless vs noisy baseline (bootstrap 95% CI for mean difference in global-hit probability):")
        lines.append("")
        lines.append("| p | policy | mean(no noise) | mean(haar,k0=0) | diff | 95% CI |")
        lines.append("|---:|---|---:|---:|---:|---:|")
        for p in ps:
            for pol in policies:
                t = softmax_t if pol == "softmax" else None
                xs = _vals(p, "none", 0, pol, t)
                ys = _vals(p, "haar", 0, pol, t)
                diff, lo, hi = _bootstrap_diff_mean(xs, ys, n_boot=5000, seed=1 + p)
                lines.append(f"| {p} | {pol} | {_mean(xs):.3f} | {_mean(ys):.3f} | {diff:.3f} | [{lo:.3f}, {hi:.3f}] |")
        lines.append("")

        lines.append("Additional plots (steepest-descent baseline):")
        lines.append("")
        lines.append(f"- ![steepest global hit vs k0](padic_lr_basin_fraction.png)")
        lines.append(f"- ![true beta global vs k0](padic_lr_true_is_global.png)")
        lines.append("")
        lines.append(f"Raw CSVs: `{padic_source_dir}/mc_*.csv`.")
        lines.append("")
    else:
        lines.append("No p-adic regression CSVs found.")
        lines.append("")

    # Sudoku summary
    lines.append("## Sudoku: greedy descent on swap graph")
    lines.append("")
    lines.append("Setup:")
    lines.append("- Puzzle generation: fast carve (no uniqueness guarantee).")
    lines.append("- State space: row-wise permutations consistent with clues; move = within-row swap of two non-clue cells.")
    lines.append("- Loss: column+box conflict pairs (0 iff solved).")
    lines.append("- Greedy descent: at each step, choose the swap with the largest decrease; stop at local minimum.")
    lines.append("")

    if sudoku_summaries:
        lines.append("Solve-rate plot:")
        lines.append("")
        lines.append("![solve rate](sudoku_solve_rate.png)")
        lines.append("")
        lines.append("Per-method table (solve rate by clue count):")
        lines.append("")
        for key in sorted(sudoku_summaries.keys(), key=lambda k: (k.method, k.restarts)):
            lines.append(f"### {key.method} (restarts={key.restarts})")
            lines.append("")
            lines.append("| clues | n | solved rate | median final conflicts (unsolved) |")
            lines.append("|---:|---:|---:|---:|")
            by_c = sudoku_summaries[key]
            for c in sorted(by_c.keys(), reverse=True):
                row = by_c[c]
                lines.append(
                    f"| {c} | {int(row['n'])} | {row['solved_rate']:.3f} | {row['median_final_conflicts_unsolved']:.1f} |"
                )
            lines.append("")
        lines.append("Raw CSVs live under `sudoku/*/experiment_results.csv`.")
        lines.append("")
    else:
        lines.append("No Sudoku experiment CSVs found under `outputs/2026-02-19/sudoku/`.")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="outputs/2026-02-19", help="Base experiment output directory.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    base_dir = (repo_root / args.base).resolve()
    analysis_dir = base_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Write metadata
    _write_metadata(analysis_dir / "metadata.json", repo_root)

    # p-adic regression
    padic_dir = base_dir / "padic_lr_v3"
    if not padic_dir.exists():
        padic_dir = base_dir / "padic_lr_v2"
    if not padic_dir.exists():
        padic_dir = base_dir / "padic_lr"
    padic_mc_paths = list(padic_dir.glob("mc_*.csv"))
    padic_summaries, padic_rows = summarize_padic(padic_mc_paths)
    if padic_summaries:
        plot_padic_basin(padic_summaries, analysis_dir)
        plot_padic_true_global(padic_summaries, analysis_dir)
        plot_padic_policy_noiseless_vs_noisy(padic_summaries, analysis_dir)

    # Sudoku
    sudoku_paths = list((base_dir / "sudoku").glob("*/experiment_results.csv"))
    sudoku_summaries = summarize_sudoku(sudoku_paths)
    if sudoku_summaries:
        plot_sudoku_solve_rate(sudoku_summaries, analysis_dir)

    # Markdown report
    report_path = base_dir / "analysis" / "experiment_report.md"
    write_markdown_report(
        report_path,
        base_dir=base_dir,
        base_label=args.base,
        padic_source_dir=padic_dir.name,
        padic_summaries=padic_summaries,
        padic_rows=padic_rows,
        sudoku_summaries=sudoku_summaries,
    )
    print("Wrote", report_path)


if __name__ == "__main__":
    main()
