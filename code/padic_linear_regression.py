#!/usr/bin/env python3
"""
padic_linear_regression.py

Toy p-adic / ultrametric linear regression utilities:

- Generate synthetic linear data with "p-adic noise" (integer-valued approximations).
- Enumerate all candidate best-fit hyperplanes that interpolate (d+1) data points.
- Build the neighbour relation between hyperplanes (swap 1 defining point).
- Run steepest-descent ("greedy walk") on that hyperplane graph and measure when local minima are global.

This is intended as a small, dependency-free sandbox to explore the phenomenon described in:
  "best ultrametric fit passes through data points (d+1 points determine a hyperplane)".

Notes on "p-adic noise"
-----------------------
There isn't a single canonical definition of "noise" on Q_p in the way there is for R.
The simplest, standard baseline on Z_p is Haar-uniform noise. A convenient discrete approximation is:

  ε ≡ U (mod p^M), with U uniform on {0,1,...,p^M-1}.

Equivalently, v_p(ε) has a geometric tail:
  P(v_p(ε)=k) = (p-1)/p^{k+1}  for 0<=k<M, and P(ε=0)=1/p^M.

We support this ("haar") and an explicit "valuation+unit" model ("valuation") which matches the
"choose an exponent K then add a multiple of p^K" intuition.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import itertools
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover
    plt = None

Vector = List[int]
Point = Tuple[Vector, int]  # (x, y) with x in Z^d, y in Z


# -------------------------
# p-adic utilities on Q
# -------------------------


def v_p_int(n: int, p: int) -> int:
    """p-adic valuation v_p(n) on integers; n!=0 required."""
    if p <= 1:
        raise ValueError("p must be >= 2")
    n = abs(n)
    if n == 0:
        raise ValueError("v_p(0) is +inf; handle separately.")
    k = 0
    while n % p == 0:
        n //= p
        k += 1
    return k


def v_p(x: Fraction, p: int) -> int | float:
    """p-adic valuation v_p(x) for rational x (v_p(0)=+inf)."""
    if x == 0:
        return float("inf")
    num = abs(x.numerator)
    den = x.denominator
    v = 0
    while num % p == 0:
        num //= p
        v += 1
    while den % p == 0:
        den //= p
        v -= 1
    return v


def padic_abs(x: Fraction, p: int) -> Fraction:
    """p-adic absolute value |x|_p for rational x."""
    val = v_p(x, p)
    if val == float("inf"):
        return Fraction(0)
    # |x|_p = p^{-v_p(x)}
    if val >= 0:
        return Fraction(1, p**val)
    return Fraction(p ** (-val), 1)


# -------------------------
# Noise models (integer-valued)
# -------------------------


def _rand_sign(rng: random.Random) -> int:
    return -1 if rng.randrange(2) == 0 else 1


def sample_noise(
    rng: random.Random,
    *,
    p: int,
    k0: int = 0,
    kmax: int = 8,
    model: str = "haar",
    unit_bound: int = 10,
) -> int:
    """
    Sample an integer-valued "p-adic noise" term.

    Parameters
    ----------
    p:
        The prime base for the valuation.
    k0:
        Baseline valuation shift. Returned noise is always divisible by p**k0.
        Larger k0 = smaller noise in the p-adic norm.
    kmax:
        Controls truncation/precision (and runtime). For the 'haar' model this is M.
        For the 'valuation' model this caps the sampled valuation at k0+kmax.
    model:
        - 'none': always return 0 (noise-free / exactly on the generating hyperplane).
        - 'haar': sample U uniform mod p**kmax, then return p**k0 * (signed representative of U).
        - 'valuation': sample K in [k0, k0+kmax] (geometric tail), choose a unit-ish integer a,
          return a * p**K.
    unit_bound:
        Only used for 'valuation'. Chooses a uniformly from [-unit_bound, unit_bound] excluding 0 and
        excluding multiples of p.
    """
    if k0 < 0 or kmax < 0:
        raise ValueError("k0 and kmax must be >= 0")
    if p <= 1:
        raise ValueError("p must be >= 2")

    if model == "none":
        return 0

    if model == "haar":
        # Discrete Haar approximation on p^k0 Z_p / p^(k0+kmax) Z_p.
        if kmax == 0:
            return 0
        mod = p**kmax
        u = rng.randrange(mod)  # 0..mod-1
        # Choose a symmetric integer representative to avoid huge archimedean magnitudes.
        if u > mod // 2:
            u -= mod
        return (p**k0) * u

    if model == "valuation":
        # Sample valuation: P(K=k0+j) ∝ p^{-j} for j=0..kmax, with remaining mass at j=kmax.
        # We implement by sampling a uniform residue mod p**(kmax+1) and taking its valuation.
        if kmax == 0:
            k = k0
        else:
            u = rng.randrange(p ** (kmax + 1))
            if u == 0:
                k = k0 + kmax
            else:
                k = k0 + min(v_p_int(u, p), kmax)

        # Choose a small coefficient not divisible by p (a "unit" in Z_p).
        choices = [a for a in range(-unit_bound, unit_bound + 1) if a != 0 and a % p != 0]
        if not choices:
            raise ValueError("unit_bound too small to choose a non-zero coefficient not divisible by p")
        a = rng.choice(choices)
        return a * (p**k)

    raise ValueError(f"Unknown noise model: {model!r}")


# -------------------------
# Synthetic data generation
# -------------------------


def generate_linear_data(
    rng: random.Random,
    *,
    n_points: int,
    n_features: int,
    coef_model: str = "uniform",
    coef_bound: int = 5,
    coef_exp_min: int = 0,
    coef_exp_max: int = 6,
    coef_unit_bound: int = 1,
    x_bound: int = 5,
    p: int = 3,
    noise_model: str = "haar",
    noise_k0: int = 1,
    noise_kmax: int = 6,
) -> Tuple[List[Point], List[int]]:
    """
    Generate points (x, y) with y = beta0 + beta·x + noise, all integer-valued.
    Returns (data, true_beta) where true_beta has length (n_features+1).
    """
    if n_points <= 0:
        raise ValueError("n_points must be > 0")
    if n_features <= 0:
        raise ValueError("n_features must be > 0")
    if coef_bound < 0 or x_bound < 0:
        raise ValueError("coef_bound and x_bound must be >= 0")

    if coef_model not in {"uniform", "p_power", "p_power_unit"}:
        raise ValueError("coef_model must be one of: uniform, p_power, p_power_unit")

    def _sample_coef_uniform() -> int:
        return rng.randint(-coef_bound, coef_bound)

    def _sample_coef_p_power(*, unit: bool) -> int:
        if coef_exp_min < 0 or coef_exp_max < 0:
            raise ValueError("coef_exp_min/coef_exp_max must be >= 0 for integer p-powers")
        if coef_exp_min > coef_exp_max:
            raise ValueError("coef_exp_min must be <= coef_exp_max")
        k = rng.randint(coef_exp_min, coef_exp_max)
        if unit:
            if coef_unit_bound <= 0:
                raise ValueError("coef_unit_bound must be > 0 for p_power_unit")
            units = [a for a in range(-coef_unit_bound, coef_unit_bound + 1) if a != 0 and a % p != 0]
            if not units:
                raise ValueError("coef_unit_bound too small to choose a unit not divisible by p")
            u = rng.choice(units)
        else:
            u = _rand_sign(rng)
        return u * (p**k)

    if coef_model == "uniform":
        beta = [_sample_coef_uniform() for _ in range(n_features + 1)]
        if all(b == 0 for b in beta):
            beta[0] = 1
    elif coef_model == "p_power":
        beta = [_sample_coef_p_power(unit=False) for _ in range(n_features + 1)]
    else:  # p_power_unit
        beta = [_sample_coef_p_power(unit=True) for _ in range(n_features + 1)]

    data: List[Point] = []
    for _ in range(n_points):
        x = [rng.randint(-x_bound, x_bound) for _ in range(n_features)]
        y0 = beta[0] + sum(beta[j + 1] * x[j] for j in range(n_features))
        eps = sample_noise(rng, p=p, k0=noise_k0, kmax=noise_kmax, model=noise_model)
        y = y0 + eps
        data.append((x, y))

    return data, beta


# -------------------------
# Hyperplanes through (d+1) points
# -------------------------


def _solve_linear_system(A: List[List[Fraction]], b: List[Fraction]) -> Optional[List[Fraction]]:
    """
    Solve A x = b over Q (Fractions) using Gauss-Jordan elimination.
    Returns x or None if singular.
    """
    n = len(A)
    if n == 0 or any(len(row) != n for row in A) or len(b) != n:
        raise ValueError("A must be square and match b")

    # Augment matrix
    M = [row[:] + [b_i] for row, b_i in zip(A, b)]

    for col in range(n):
        # Find pivot
        pivot = None
        for r in range(col, n):
            if M[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            return None
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]

        # Normalize pivot row
        piv = M[col][col]
        for c in range(col, n + 1):
            M[col][c] /= piv

        # Eliminate other rows
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if factor == 0:
                continue
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]

    return [M[i][n] for i in range(n)]


def fit_hyperplane_through_points(data: Sequence[Point], idxs: Sequence[int]) -> Optional[List[Fraction]]:
    """
    Fit y = b0 + b1 x1 + ... + bd xd through the selected points.
    Returns beta as Fractions length (d+1), or None if the system is singular.
    """
    if not idxs:
        raise ValueError("idxs must be non-empty")
    d = len(data[0][0])
    m = d + 1
    if len(idxs) != m:
        raise ValueError(f"Need exactly d+1={m} points to determine a hyperplane (got {len(idxs)})")

    A: List[List[Fraction]] = []
    bvec: List[Fraction] = []
    for i in idxs:
        x, y = data[i]
        if len(x) != d:
            raise ValueError("Inconsistent feature dimension in data")
        A.append([Fraction(1)] + [Fraction(xx) for xx in x])
        bvec.append(Fraction(y))
    return _solve_linear_system(A, bvec)


def residual(beta: Sequence[Fraction], x: Sequence[int], y: int) -> Fraction:
    """Compute y - (beta0 + beta·x)."""
    yhat = beta[0]
    for j, xx in enumerate(x):
        yhat += beta[j + 1] * xx
    return Fraction(y) - yhat


def loss_ultrametric(
    beta: Sequence[Fraction],
    data: Sequence[Point],
    *,
    p: int,
    loss: str = "l1",
) -> Fraction:
    """
    Compute an ultrametric-style loss based on p-adic norms of residuals.

    loss:
      - 'l1': sum_i |r_i|_p
      - 'linf': max_i |r_i|_p
    """
    if loss not in {"l1", "linf"}:
        raise ValueError("loss must be 'l1' or 'linf'")
    if loss == "l1":
        total = Fraction(0)
        for x, y in data:
            total += padic_abs(residual(beta, x, y), p)
        return total
    # linf
    m = Fraction(0)
    for x, y in data:
        m = max(m, padic_abs(residual(beta, x, y), p))
    return m


def loss_for_integer_beta(
    beta_int: Sequence[int],
    data: Sequence[Point],
    *,
    p: int,
    loss: str = "l1",
) -> Fraction:
    """Convenience wrapper: compute loss for an integer coefficient vector."""
    beta_frac = [Fraction(b) for b in beta_int]
    return loss_ultrametric(beta_frac, data, p=p, loss=loss)


# -------------------------
# Hyperplane graph + greedy walk
# -------------------------


@dataclass(frozen=True)
class HyperplaneNode:
    beta: Tuple[Fraction, ...]             # unique coefficients
    loss: Fraction
    supports: List[Tuple[int, ...]]        # (d+1)-subsets that determine this beta


def enumerate_unique_hyperplanes(
    data: Sequence[Point],
    *,
    p: int,
    loss: str = "l1",
) -> Tuple[List[HyperplaneNode], dict[Tuple[int, ...], int], int]:
    """
    Enumerate all hyperplanes through (d+1) points, but deduplicate by coefficient vector.

    Returns:
      (hyperplanes, subset_to_hyperplane_id, n_singular_skipped)

    In noiseless data, many (d+1)-subsets define the *same* hyperplane; this routine collapses
    those to a single node, matching the "finite set of hyperplanes" interpretation.
    """
    if not data:
        raise ValueError("data must be non-empty")
    d = len(data[0][0])
    m = d + 1
    n = len(data)
    if n < m:
        raise ValueError(f"Need at least d+1={m} points (got n={n})")

    hyperplanes: List[HyperplaneNode] = []
    subset_to_id: dict[Tuple[int, ...], int] = {}
    beta_to_id: dict[Tuple[Fraction, ...], int] = {}
    skipped = 0

    for idxs in itertools.combinations(range(n), m):
        beta = fit_hyperplane_through_points(data, idxs)
        if beta is None:
            skipped += 1
            continue
        beta_t = tuple(beta)
        hid = beta_to_id.get(beta_t)
        if hid is None:
            L = loss_ultrametric(beta, data, p=p, loss=loss)
            hid = len(hyperplanes)
            beta_to_id[beta_t] = hid
            hyperplanes.append(HyperplaneNode(beta=beta_t, loss=L, supports=[tuple(idxs)]))
        else:
            hyperplanes[hid].supports.append(tuple(idxs))
        subset_to_id[tuple(idxs)] = hid

    return hyperplanes, subset_to_id, skipped


def neighbours_of(idxs: Tuple[int, ...], *, n_points: int) -> Iterator[Tuple[int, ...]]:
    """
    Neighbours by swapping exactly one defining point (keep subset size fixed).
    """
    m = len(idxs)
    in_set = set(idxs)
    for pos in range(m):
        for new in range(n_points):
            if new in in_set:
                continue
            replaced = list(idxs)
            replaced[pos] = new
            replaced.sort()
            yield tuple(replaced)


def build_hyperplane_adjacency(
    subset_to_id: dict[Tuple[int, ...], int],
    *,
    n_points: int,
) -> List[List[int]]:
    """
    Build the neighbour graph on *unique hyperplanes*.

    We add an (undirected) edge between hyperplane IDs u and v if there exist defining
    (d+1)-subsets S and S' that differ by one index (a single-point swap) with
    id(S)=u and id(S')=v.
    """
    if not subset_to_id:
        return []
    n_nodes = max(subset_to_id.values()) + 1
    adj: List[set[int]] = [set() for _ in range(n_nodes)]
    for idxs, u in subset_to_id.items():
        for nb in neighbours_of(idxs, n_points=n_points):
            v = subset_to_id.get(nb)
            if v is None or v == u:
                continue
            adj[u].add(v)
            adj[v].add(u)
    return [sorted(xs) for xs in adj]


def precompute_improving_moves(
    nodes: Sequence[HyperplaneNode],
    adjacency: Sequence[Sequence[int]],
) -> List[List[Tuple[int, Fraction]]]:
    """For each node ID i, list (j, loss_i - loss_j) over strictly improving neighbours j."""
    improving: List[List[Tuple[int, Fraction]]] = []
    for i, node in enumerate(nodes):
        cur_loss = node.loss
        moves: List[Tuple[int, Fraction]] = []
        for nb in adjacency[i]:
            nb_loss = nodes[nb].loss
            if nb_loss < cur_loss:
                moves.append((nb, cur_loss - nb_loss))
        improving.append(moves)
    return improving


def analyze_improving_descent(
    nodes: Sequence[HyperplaneNode],
    adjacency: Sequence[Sequence[int]],
    *,
    policy: str,
    temperature: float = 1.0,
    tie_break: str = "lex",
    _improving: Optional[List[List[Tuple[int, Fraction]]]] = None,
) -> dict[str, object]:
    """
    Analyse an improvement-only descent process on the neighbour graph.

    The process stops at a local minimum (a node with no strictly improving neighbours).
    If the move selection is random among improving neighbours, the success probability
    is computed exactly by dynamic programming because loss strictly decreases (a DAG).

    policy:
      - 'steepest': deterministic steepest descent (largest improvement); ties by `tie_break`.
      - 'uniform': choose uniformly among all improving neighbours.
      - 'proportional': choose with probability proportional to improvement size.
      - 'softmax': choose with probability proportional to exp(improvement / temperature).
    """
    if not nodes:
        raise ValueError("nodes must be non-empty")
    if policy not in {"steepest", "uniform", "proportional", "softmax"}:
        raise ValueError(f"Unknown policy: {policy!r}")
    if tie_break not in {"lex", "random"}:
        raise ValueError("tie_break must be 'lex' or 'random'")
    if policy == "softmax" and temperature <= 0:
        raise ValueError("temperature must be > 0 for softmax policy")

    min_loss = min(n.loss for n in nodes)
    global_minima = {i for i, n in enumerate(nodes) if n.loss == min_loss}

    # Precompute improving neighbours (strict loss decrease only).
    if _improving is None:
        improving = precompute_improving_moves(nodes, adjacency)
    else:
        improving = _improving

    local_minima = {i for i, moves in enumerate(improving) if not moves}
    bad_local_minima = local_minima - global_minima

    # Dynamic programming in increasing loss order (all improving neighbours have lower loss).
    order = sorted(range(len(nodes)), key=lambda i: (nodes[i].loss, nodes[i].beta))
    success = [0.0 for _ in nodes]
    exp_steps = [0.0 for _ in nodes]

    for cur in order:
        moves = improving[cur]
        if not moves:
            success[cur] = 1.0 if cur in global_minima else 0.0
            exp_steps[cur] = 0.0
            continue

        if policy == "steepest":
            best_impr = max(impr for _nb, impr in moves)
            best_nbs = [nb for nb, impr in moves if impr == best_impr]
            if tie_break == "lex":
                nxt = min(best_nbs, key=lambda j: (nodes[j].beta, j))
                success[cur] = success[nxt]
                exp_steps[cur] = 1.0 + exp_steps[nxt]
            else:
                # Random tie-break among best neighbours: compute exactly by averaging.
                k = len(best_nbs)
                success[cur] = sum(success[nb] for nb in best_nbs) / k
                exp_steps[cur] = 1.0 + sum(exp_steps[nb] for nb in best_nbs) / k
            continue

        nbs = [nb for nb, _impr in moves]
        imprs = [impr for _nb, impr in moves]

        if policy == "uniform":
            k = len(nbs)
            probs = [1.0 / k] * k
        elif policy == "proportional":
            tot = sum(imprs, start=Fraction(0))
            probs = [float(impr / tot) for impr in imprs]
        else:  # softmax
            # weights ∝ exp(improvement / temperature)
            args = [float(impr) / temperature for impr in imprs]
            m = max(args)
            ws = [math.exp(a - m) for a in args]
            tot = sum(ws)
            probs = [w / tot for w in ws]

        s = 0.0
        e = 0.0
        for nb, p in zip(nbs, probs):
            s += p * success[nb]
            e += p * exp_steps[nb]
        success[cur] = s
        exp_steps[cur] = 1.0 + e

    global_hit_prob = sum(success) / len(success)
    avg_steps = sum(exp_steps) / len(exp_steps)

    return {
        "n_nodes": len(nodes),
        "min_loss": min_loss,
        "min_loss_float": float(min_loss),
        "n_global_minima": len(global_minima),
        "n_local_minima": len(local_minima),
        "n_bad_local_minima": len(bad_local_minima),
        "all_local_minima_global": int(len(bad_local_minima) == 0),
        "global_hit_prob": global_hit_prob,
        "avg_steps": avg_steps,
    }


# -------------------------
# Graph / landscape analyses
# -------------------------


def steepest_descent_map(
    nodes: Sequence[HyperplaneNode],
    improving: Sequence[Sequence[Tuple[int, Fraction]]],
    *,
    tie_break: str = "lex",
) -> Tuple[List[int], List[int], List[Optional[int]]]:
    """
    Deterministic steepest-descent basins on the unique-hyperplane graph.

    Returns:
      (sink_of[node], steps_to_sink[node], next[node])
    where next[node] is the chosen steepest-improving neighbour (or None at a sink).
    """
    if tie_break not in {"lex"}:
        raise ValueError("Only tie_break='lex' is supported for deterministic basins")

    n = len(nodes)
    nxt: List[Optional[int]] = [None] * n
    for i, moves in enumerate(improving):
        if not moves:
            continue
        best_impr = max(impr for _nb, impr in moves)
        best_nbs = [nb for nb, impr in moves if impr == best_impr]
        nxt[i] = min(best_nbs, key=lambda j: (nodes[j].beta, j))

    order = sorted(range(n), key=lambda i: (nodes[i].loss, nodes[i].beta))
    sink_of = [-1] * n
    steps = [0] * n
    for i in order:
        j = nxt[i]
        if j is None:
            sink_of[i] = i
            steps[i] = 0
        else:
            sink_of[i] = sink_of[j]
            steps[i] = 1 + steps[j]
    return sink_of, steps, nxt


def bottleneck_distance_to_global_minima(
    nodes: Sequence[HyperplaneNode],
    adjacency: Sequence[Sequence[int]],
    global_minima: Sequence[int],
) -> List[Fraction]:
    """
    For each node v, compute the minimised maximum loss along any path from v to a global minimum:
      dist[v] = min_{path v->g} max_{u in path} loss[u].
    """
    n = len(nodes)
    if n == 0:
        return []
    if not global_minima:
        raise ValueError("global_minima must be non-empty")

    dist: List[Optional[Fraction]] = [None] * n
    heap: List[Tuple[Fraction, int]] = []
    for g in global_minima:
        d0 = nodes[g].loss
        if dist[g] is None or d0 < dist[g]:
            dist[g] = d0
            heapq.heappush(heap, (d0, g))

    while heap:
        d, u = heapq.heappop(heap)
        if dist[u] is None or d != dist[u]:
            continue
        for v in adjacency[u]:
            cand = d if d >= nodes[v].loss else nodes[v].loss
            if dist[v] is None or cand < dist[v]:
                dist[v] = cand
                heapq.heappush(heap, (cand, v))

    # graph should be connected in practice, but fall back to self-loss if not.
    return [dv if dv is not None else nodes[i].loss for i, dv in enumerate(dist)]


def plateau_component_sizes(
    nodes: Sequence[HyperplaneNode],
    adjacency: Sequence[Sequence[int]],
) -> Tuple[List[int], int]:
    """
    Return (component_sizes, equal_loss_edge_count) for the neutral (equal-loss) subgraph.
    """
    n = len(nodes)
    seen = [False] * n
    sizes: List[int] = []
    equal_edges = 0
    for u in range(n):
        for v in adjacency[u]:
            if v > u and nodes[v].loss == nodes[u].loss:
                equal_edges += 1

    for i in range(n):
        if seen[i]:
            continue
        L = nodes[i].loss
        stack = [i]
        seen[i] = True
        sz = 0
        while stack:
            u = stack.pop()
            sz += 1
            for v in adjacency[u]:
                if seen[v] or nodes[v].loss != L:
                    continue
                seen[v] = True
                stack.append(v)
        sizes.append(sz)
    return sizes, equal_edges


def local_optima_network_edges(
    sink_of: Sequence[int],
    adjacency: Sequence[Sequence[int]],
) -> dict[Tuple[int, int], int]:
    """
    Build a simple "local optima network" (LON) on sinks by counting basin boundary edges.

    Returns:
      edge_counts[(sink_u, sink_v)] = number of undirected hyperplane edges that cross from basin(sink_u) to basin(sink_v),
    recorded directionally for convenience (i.e. counts are symmetric).
    """
    n = len(sink_of)
    counts: dict[Tuple[int, int], int] = {}
    for u in range(n):
        su = sink_of[u]
        for v in adjacency[u]:
            if v <= u:
                continue
            sv = sink_of[v]
            if su == sv:
                continue
            counts[(su, sv)] = counts.get((su, sv), 0) + 1
            counts[(sv, su)] = counts.get((sv, su), 0) + 1
    return counts


def support_point_frequencies(
    nodes: Sequence[HyperplaneNode],
    node_ids: Iterable[int],
    *,
    n_points: int,
) -> Tuple[List[int], int]:
    """
    Count how often each data point index appears in the supports of the given hyperplane nodes.

    Returns (counts_by_point, total_supports_counted).
    """
    counts = [0] * n_points
    total = 0
    for hid in node_ids:
        for supp in nodes[hid].supports:
            total += 1
            for i in supp:
                counts[i] += 1
    return counts, total


# -------------------------
# CLI helpers
# -------------------------


def _format_beta(beta: Sequence[int | Fraction]) -> str:
    return "[" + ", ".join(str(b) for b in beta) + "]"


def _write_dataset_csv(path: Path, data: Sequence[Point]) -> None:
    if not data:
        raise ValueError("data empty")
    d = len(data[0][0])
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"x{j+1}" for j in range(d)] + ["y"])
        for x, y in data:
            w.writerow(list(x) + [y])


def main() -> None:
    ap = argparse.ArgumentParser(description="Toy p-adic linear regression + hyperplane graph walks.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_gen = sub.add_parser("gen", help="Generate a synthetic dataset.")
    ap_gen.add_argument("--out", type=str, default="", help="CSV output path (default: print to stdout)")
    ap_gen.add_argument("--seed", type=int, default=0)
    ap_gen.add_argument("--n", type=int, default=20, help="number of points")
    ap_gen.add_argument("--d", type=int, default=3, help="number of features (x-dimension)")
    ap_gen.add_argument("--p", type=int, default=3)
    ap_gen.add_argument("--coef-model", type=str, default="p_power", choices=["uniform", "p_power", "p_power_unit"])
    ap_gen.add_argument("--coef-bound", type=int, default=5, help="uniform model: coefficients in [-coef-bound, coef-bound]")
    ap_gen.add_argument("--coef-exp-min", type=int, default=0, help="p_power models: min exponent for p^k")
    ap_gen.add_argument("--coef-exp-max", type=int, default=6, help="p_power models: max exponent for p^k")
    ap_gen.add_argument("--coef-unit-bound", type=int, default=1, help="p_power_unit: choose unit a in [-B,B] not divisible by p")
    ap_gen.add_argument("--x-bound", type=int, default=5)
    ap_gen.add_argument("--noise-model", type=str, default="haar", choices=["none", "haar", "valuation"])
    ap_gen.add_argument("--noise-k0", type=int, default=1)
    ap_gen.add_argument("--noise-kmax", type=int, default=6)

    ap_one = sub.add_parser("analyze", help="Enumerate hyperplanes + analyse greedy basins for one synthetic dataset.")
    ap_one.add_argument("--seed", type=int, default=0)
    ap_one.add_argument("--n", type=int, default=20)
    ap_one.add_argument("--d", type=int, default=3)
    ap_one.add_argument("--p", type=int, default=3)
    ap_one.add_argument("--coef-model", type=str, default="p_power", choices=["uniform", "p_power", "p_power_unit"])
    ap_one.add_argument("--coef-bound", type=int, default=5, help="uniform model: coefficients in [-coef-bound, coef-bound]")
    ap_one.add_argument("--coef-exp-min", type=int, default=0)
    ap_one.add_argument("--coef-exp-max", type=int, default=6)
    ap_one.add_argument("--coef-unit-bound", type=int, default=1)
    ap_one.add_argument("--x-bound", type=int, default=5)
    ap_one.add_argument("--noise-model", type=str, default="haar", choices=["none", "haar", "valuation"])
    ap_one.add_argument("--noise-k0", type=int, default=1)
    ap_one.add_argument("--noise-kmax", type=int, default=6)
    ap_one.add_argument("--loss", type=str, default="l1", choices=["l1", "linf"])
    ap_one.add_argument("--policy", type=str, default="steepest", choices=["steepest", "uniform", "proportional", "softmax"])
    ap_one.add_argument("--temperature", type=float, default=1.0, help="softmax policy: temperature (>0). Smaller = greedier.")
    ap_one.add_argument("--tie-break", type=str, default="lex", choices=["lex", "random"])
    ap_one.add_argument("--show-best", action="store_true", help="print the best hyperplane coefficients")

    ap_mc = sub.add_parser("mc", help="Monte Carlo: how often is greedy local=min global?")
    ap_mc.add_argument("--out", type=str, default="",
                       help="CSV output path (default: outputs/padic_linear_regression_mc.csv)")
    ap_mc.add_argument("--seed", type=int, default=0)
    ap_mc.add_argument("--trials", type=int, default=50)
    ap_mc.add_argument("--n", type=int, default=20)
    ap_mc.add_argument("--d", type=int, default=3)
    ap_mc.add_argument("--p", type=int, default=3)
    ap_mc.add_argument("--coef-model", type=str, default="p_power", choices=["uniform", "p_power", "p_power_unit"])
    ap_mc.add_argument("--coef-bound", type=int, default=5, help="uniform model: coefficients in [-coef-bound, coef-bound]")
    ap_mc.add_argument("--coef-exp-min", type=int, default=0)
    ap_mc.add_argument("--coef-exp-max", type=int, default=6)
    ap_mc.add_argument("--coef-unit-bound", type=int, default=1)
    ap_mc.add_argument("--x-bound", type=int, default=5)
    ap_mc.add_argument("--noise-model", type=str, default="haar", choices=["none", "haar", "valuation"])
    ap_mc.add_argument("--noise-k0", type=int, default=1)
    ap_mc.add_argument("--noise-kmax", type=int, default=6)
    ap_mc.add_argument("--loss", type=str, default="l1", choices=["l1", "linf"])
    ap_mc.add_argument("--policies", type=str, default="steepest",
                       help="comma-separated policies: steepest,uniform,proportional,softmax")
    ap_mc.add_argument("--temperature", type=float, default=1.0, help="softmax policy temperature (>0)")
    ap_mc.add_argument("--tie-break", type=str, default="lex", choices=["lex", "random"])

    ap_land = sub.add_parser("landscape", help="Landscape analysis on the unique-hyperplane neighbour graph.")
    ap_land.add_argument("--outdir", type=str, default="",
                         help="Output directory (default: outputs/<YYYY-MM-DD>/padic_lr_landscape/run_<seed>_<timestamp>/)")
    ap_land.add_argument("--seed", type=int, default=0)
    ap_land.add_argument("--n", type=int, default=20)
    ap_land.add_argument("--d", type=int, default=3)
    ap_land.add_argument("--p", type=int, default=3)
    ap_land.add_argument("--coef-model", type=str, default="p_power", choices=["uniform", "p_power", "p_power_unit"])
    ap_land.add_argument("--coef-bound", type=int, default=5)
    ap_land.add_argument("--coef-exp-min", type=int, default=0)
    ap_land.add_argument("--coef-exp-max", type=int, default=6)
    ap_land.add_argument("--coef-unit-bound", type=int, default=1)
    ap_land.add_argument("--x-bound", type=int, default=5)
    ap_land.add_argument("--noise-model", type=str, default="haar", choices=["none", "haar", "valuation"])
    ap_land.add_argument("--noise-k0", type=int, default=1)
    ap_land.add_argument("--noise-kmax", type=int, default=6)
    ap_land.add_argument("--loss", type=str, default="l1", choices=["l1", "linf"])
    ap_land.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots.")

    args = ap.parse_args()

    if args.cmd == "gen":
        rng = random.Random(args.seed)
        data, beta_true = generate_linear_data(
            rng,
            n_points=args.n,
            n_features=args.d,
            coef_model=args.coef_model,
            coef_bound=args.coef_bound,
            coef_exp_min=args.coef_exp_min,
            coef_exp_max=args.coef_exp_max,
            coef_unit_bound=args.coef_unit_bound,
            x_bound=args.x_bound,
            p=args.p,
            noise_model=args.noise_model,
            noise_k0=args.noise_k0,
            noise_kmax=args.noise_kmax,
        )

        header = [f"x{j+1}" for j in range(args.d)] + ["y"]
        if args.out:
            path = Path(args.out)
            _write_dataset_csv(path, data)
            print("Wrote", path)
            print("True beta:", _format_beta(beta_true))
        else:
            print("# True beta:", _format_beta(beta_true))
            w = csv.writer(_Stdout())
            w.writerow(header)
            for x, y in data:
                w.writerow(list(x) + [y])
        return

    if args.cmd == "analyze":
        rng = random.Random(args.seed)
        data, beta_true = generate_linear_data(
            rng,
            n_points=args.n,
            n_features=args.d,
            coef_model=args.coef_model,
            coef_bound=args.coef_bound,
            coef_exp_min=args.coef_exp_min,
            coef_exp_max=args.coef_exp_max,
            coef_unit_bound=args.coef_unit_bound,
            x_bound=args.x_bound,
            p=args.p,
            noise_model=args.noise_model,
            noise_k0=args.noise_k0,
            noise_kmax=args.noise_kmax,
        )

        t0 = time.time()
        nodes, subset_to_id, skipped = enumerate_unique_hyperplanes(data, p=args.p, loss=args.loss)
        if not nodes:
            raise SystemExit("No non-singular hyperplanes found.")
        adjacency = build_hyperplane_adjacency(subset_to_id, n_points=args.n)
        stats = analyze_improving_descent(
            nodes,
            adjacency,
            policy=args.policy,
            temperature=args.temperature,
            tie_break=args.tie_break,
        )
        loss_true = loss_for_integer_beta(beta_true, data, p=args.p, loss=args.loss)
        dt = time.time() - t0

        print("True beta:", _format_beta(beta_true))
        print("Loss(true beta):", loss_true, f"({float(loss_true):.6f})")
        print("Enumerated unique hyperplanes:", stats["n_nodes"], "Non-singular subsets:", len(subset_to_id), "Skipped singular:", skipped, f"({dt:.3f}s)")
        print("Loss type:", args.loss, "p:", args.p, "noise:", f"{args.noise_model}(k0={args.noise_k0},kmax={args.noise_kmax})")
        print("Min loss:", stats["min_loss"], f"({stats['min_loss_float']:.6f})")
        print("True is global min:", bool(loss_true == stats["min_loss"]))
        print("Global minima:", stats["n_global_minima"], "Local minima:", stats["n_local_minima"], "Bad local minima:", stats["n_bad_local_minima"])
        print("All local minima are global:", bool(stats["all_local_minima_global"]))
        print("Descent policy:", args.policy, ("T=" + str(args.temperature) if args.policy == "softmax" else ""))
        print("Global hit probability:", f"{stats['global_hit_prob']:.3f}")
        print("Avg steps to local min:", f"{stats['avg_steps']:.2f}")

        if args.show_best:
            min_loss = stats["min_loss"]
            best = [n for n in nodes if n.loss == min_loss]
            # Print one representative.
            n0 = best[0]
            print("Best beta:", _format_beta(n0.beta))
            if n0.supports:
                print("One defining subset:", n0.supports[0], f"(supports={len(n0.supports)})")
        return

    if args.cmd == "mc":
        out = Path(args.out) if args.out else (Path(__file__).resolve().parent.parent / "outputs" / "padic_linear_regression_mc.csv")
        out.parent.mkdir(parents=True, exist_ok=True)

        policies = [p.strip() for p in args.policies.split(",") if p.strip()]
        for pol in policies:
            if pol not in {"steepest", "uniform", "proportional", "softmax"}:
                raise SystemExit(f"Unknown policy in --policies: {pol!r}")

        fieldnames = [
            "trial",
            "seed",
            "n",
            "d",
            "p",
            "coef_model",
            "coef_bound",
            "coef_exp_min",
            "coef_exp_max",
            "coef_unit_bound",
            "loss",
            "noise_model",
            "noise_k0",
            "noise_kmax",
            "policy",
            "temperature",
            "beta_true",
            "loss_true",
            "loss_true_float",
            "true_is_global",
            "true_beta_appears",
            "nodes",
            "subsets_nonsingular",
            "singular_skipped",
            "min_loss",
            "min_loss_float",
            "global_minima",
            "local_minima",
            "bad_local_minima",
            "all_local_minima_global",
            "global_hit_prob",
            "avg_steps",
            "seconds",
        ]
        rng0 = random.Random(args.seed)
        t_start = time.time()
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for trial in range(args.trials):
                seed = rng0.randrange(1_000_000_000)
                rng = random.Random(seed)

                data, beta_true = generate_linear_data(
                    rng,
                    n_points=args.n,
                    n_features=args.d,
                    coef_model=args.coef_model,
                    coef_bound=args.coef_bound,
                    coef_exp_min=args.coef_exp_min,
                    coef_exp_max=args.coef_exp_max,
                    coef_unit_bound=args.coef_unit_bound,
                    x_bound=args.x_bound,
                    p=args.p,
                    noise_model=args.noise_model,
                    noise_k0=args.noise_k0,
                    noise_kmax=args.noise_kmax,
                )

                t0 = time.time()
                nodes, subset_to_id, skipped = enumerate_unique_hyperplanes(data, p=args.p, loss=args.loss)
                loss_true = loss_for_integer_beta(beta_true, data, p=args.p, loss=args.loss)
                if not nodes:
                    # extremely degenerate; record and continue
                    for pol in policies:
                        w.writerow({
                            "trial": trial,
                            "seed": seed,
                            "n": args.n,
                            "d": args.d,
                            "p": args.p,
                            "coef_model": args.coef_model,
                            "coef_bound": args.coef_bound,
                            "coef_exp_min": args.coef_exp_min,
                            "coef_exp_max": args.coef_exp_max,
                            "coef_unit_bound": args.coef_unit_bound,
                            "loss": args.loss,
                            "noise_model": args.noise_model,
                            "noise_k0": args.noise_k0,
                            "noise_kmax": args.noise_kmax,
                            "policy": pol,
                            "temperature": args.temperature if pol == "softmax" else "",
                            "beta_true": _format_beta(beta_true),
                            "loss_true": str(loss_true),
                            "loss_true_float": f"{float(loss_true):.12g}",
                            "true_is_global": "",
                            "true_beta_appears": "",
                            "nodes": 0,
                            "subsets_nonsingular": len(subset_to_id),
                            "singular_skipped": skipped,
                            "min_loss": "",
                            "min_loss_float": "",
                            "global_minima": 0,
                            "local_minima": 0,
                            "bad_local_minima": 0,
                            "all_local_minima_global": "",
                            "global_hit_prob": "",
                            "avg_steps": "",
                            "seconds": time.time() - t0,
                        })
                    continue

                adjacency = build_hyperplane_adjacency(subset_to_id, n_points=args.n)
                improving = precompute_improving_moves(nodes, adjacency)

                min_loss = min(n.loss for n in nodes)
                global_minima = {i for i, n in enumerate(nodes) if n.loss == min_loss}
                local_minima = {i for i, moves in enumerate(improving) if not moves}
                bad_local_minima = local_minima - global_minima

                true_is_global = int(bool(loss_true == min_loss))
                # Because the data is noisy, beta_true is usually not an interpolating hyperplane.
                # When it is, it must appear among the enumerated nodes.
                beta_true_frac = tuple(Fraction(b) for b in beta_true)
                true_beta_appears = int(any(n.beta == beta_true_frac for n in nodes))

                for pol in policies:
                    stats_pol = analyze_improving_descent(
                        nodes,
                        adjacency,
                        policy=pol,
                        temperature=args.temperature,
                        tie_break=args.tie_break,
                        _improving=improving,
                    )
                    w.writerow({
                        "trial": trial,
                        "seed": seed,
                        "n": args.n,
                        "d": args.d,
                        "p": args.p,
                        "coef_model": args.coef_model,
                        "coef_bound": args.coef_bound,
                        "coef_exp_min": args.coef_exp_min,
                        "coef_exp_max": args.coef_exp_max,
                        "coef_unit_bound": args.coef_unit_bound,
                        "loss": args.loss,
                        "noise_model": args.noise_model,
                        "noise_k0": args.noise_k0,
                        "noise_kmax": args.noise_kmax,
                        "policy": pol,
                        "temperature": args.temperature if pol == "softmax" else "",
                        "beta_true": _format_beta(beta_true),
                        "loss_true": str(loss_true),
                        "loss_true_float": f"{float(loss_true):.12g}",
                        "true_is_global": true_is_global,
                        "true_beta_appears": true_beta_appears,
                        "nodes": stats_pol["n_nodes"],
                        "subsets_nonsingular": len(subset_to_id),
                        "singular_skipped": skipped,
                        "min_loss": min_loss,
                        "min_loss_float": f"{float(min_loss):.12g}",
                        "global_minima": len(global_minima),
                        "local_minima": len(local_minima),
                        "bad_local_minima": len(bad_local_minima),
                        "all_local_minima_global": stats_pol["all_local_minima_global"],
                        "global_hit_prob": f"{stats_pol['global_hit_prob']:.6f}",
                        "avg_steps": f"{stats_pol['avg_steps']:.6f}",
                        "seconds": time.time() - t0,
                    })

        dt = time.time() - t_start
        print("Wrote", out)
        print("Wall time:", f"{dt:.2f}s")
        return

    if args.cmd == "landscape":
        repo_root = Path(__file__).resolve().parent.parent
        if args.outdir:
            run_dir = Path(args.outdir)
        else:
            day = time.strftime("%Y-%m-%d")
            stamp = time.strftime("%H%M%S")
            run_dir = repo_root / "outputs" / day / "padic_lr_landscape" / f"run_seed{args.seed}_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(args.seed)
        data, beta_true = generate_linear_data(
            rng,
            n_points=args.n,
            n_features=args.d,
            coef_model=args.coef_model,
            coef_bound=args.coef_bound,
            coef_exp_min=args.coef_exp_min,
            coef_exp_max=args.coef_exp_max,
            coef_unit_bound=args.coef_unit_bound,
            x_bound=args.x_bound,
            p=args.p,
            noise_model=args.noise_model,
            noise_k0=args.noise_k0,
            noise_kmax=args.noise_kmax,
        )
        _write_dataset_csv(run_dir / "dataset.csv", data)

        nodes, subset_to_id, skipped = enumerate_unique_hyperplanes(data, p=args.p, loss=args.loss)
        if not nodes:
            raise SystemExit("No non-singular hyperplanes found.")
        adjacency = build_hyperplane_adjacency(subset_to_id, n_points=args.n)
        improving = precompute_improving_moves(nodes, adjacency)

        min_loss = min(n.loss for n in nodes)
        global_minima = [i for i, n in enumerate(nodes) if n.loss == min_loss]
        local_minima = [i for i, moves in enumerate(improving) if not moves]
        bad_local_minima = [i for i in local_minima if i not in set(global_minima)]

        sink_of, steps_to_sink, nxt = steepest_descent_map(nodes, improving, tie_break="lex")
        basin_sizes: dict[int, int] = {}
        for s in sink_of:
            basin_sizes[s] = basin_sizes.get(s, 0) + 1
        sinks = sorted({s for s in basin_sizes.keys()})
        global_sinks = sorted(set(global_minima))

        basin_fraction_global = sum(basin_sizes[s] for s in global_sinks) / len(nodes)
        basin_probs = [basin_sizes[s] / len(nodes) for s in sinks]
        basin_entropy = -sum(p * math.log(p) for p in basin_probs if p > 0)

        dist_to_global = bottleneck_distance_to_global_minima(nodes, adjacency, global_sinks)
        barrier_to_global = [dist_to_global[i] - nodes[i].loss for i in range(len(nodes))]

        plateau_sizes, equal_loss_edges = plateau_component_sizes(nodes, adjacency)
        lon_edges = local_optima_network_edges(sink_of, adjacency)

        # Support-point frequencies for global minima supports.
        support_counts, support_total = support_point_frequencies(nodes, global_sinks, n_points=args.n)
        support_rows = [
            {"point": i, "count": c, "fraction": (c / support_total) if support_total else 0.0}
            for i, c in enumerate(support_counts)
        ]
        support_rows.sort(key=lambda r: (-r["count"], r["point"]))

        # Write CSVs
        with (run_dir / "nodes.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["node", "loss", "loss_float", "degree", "supports", "is_global_min", "sink", "steps_to_sink", "barrier_to_global"])
            for i, n in enumerate(nodes):
                w.writerow([
                    i,
                    str(n.loss),
                    f"{float(n.loss):.12g}",
                    len(adjacency[i]),
                    len(n.supports),
                    int(i in set(global_sinks)),
                    sink_of[i],
                    steps_to_sink[i],
                    str(barrier_to_global[i]),
                ])

        with (run_dir / "sinks.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sink", "loss", "loss_float", "basin_size", "is_global_min", "degree", "supports", "barrier_to_global"])
            for s in sorted(basin_sizes.keys(), key=lambda i: (nodes[i].loss, nodes[i].beta)):
                w.writerow([
                    s,
                    str(nodes[s].loss),
                    f"{float(nodes[s].loss):.12g}",
                    basin_sizes[s],
                    int(s in set(global_sinks)),
                    len(adjacency[s]),
                    len(nodes[s].supports),
                    str(barrier_to_global[s]),
                ])

        with (run_dir / "lon_edges.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sink_u", "sink_v", "boundary_edges"])
            for (u, v), c in sorted(lon_edges.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])):
                w.writerow([u, v, c])

        with (run_dir / "support_points_global_minima.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["point", "count", "fraction"])
            for r in support_rows:
                w.writerow([r["point"], r["count"], f"{r['fraction']:.6f}"])

        # Summary JSON
        degrees = [len(adjacency[i]) for i in range(len(nodes))]
        summary = {
            "seed": args.seed,
            "n_points": args.n,
            "d_features": args.d,
            "p": args.p,
            "coef_model": args.coef_model,
            "coef_bound": args.coef_bound,
            "coef_exp_min": args.coef_exp_min,
            "coef_exp_max": args.coef_exp_max,
            "coef_unit_bound": args.coef_unit_bound,
            "x_bound": args.x_bound,
            "noise_model": args.noise_model,
            "noise_k0": args.noise_k0,
            "noise_kmax": args.noise_kmax,
            "loss": args.loss,
            "beta_true": beta_true,
            "loss_true": str(loss_for_integer_beta(beta_true, data, p=args.p, loss=args.loss)),
            "unique_hyperplanes": len(nodes),
            "non_singular_subsets": len(subset_to_id),
            "singular_subsets_skipped": skipped,
            "undirected_edges": sum(len(xs) for xs in adjacency) // 2,
            "degree_mean": statistics.mean(degrees) if degrees else 0.0,
            "degree_median": statistics.median(degrees) if degrees else 0.0,
            "min_loss": str(min_loss),
            "min_loss_float": float(min_loss),
            "global_minima": len(global_sinks),
            "local_minima": len(local_minima),
            "bad_local_minima": len(bad_local_minima),
            "basin_fraction_global": basin_fraction_global,
            "basin_entropy": basin_entropy,
            "avg_steps_to_sink": statistics.mean(steps_to_sink) if steps_to_sink else 0.0,
            "plateau_components": len(plateau_sizes),
            "plateau_max_size": max(plateau_sizes) if plateau_sizes else 0,
            "equal_loss_edges": equal_loss_edges,
            "barrier_sink_median_float": float(statistics.median([float(barrier_to_global[s]) for s in sinks])) if sinks else 0.0,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

        if (plt is not None) and (not args.no_plots):
            # Loss histogram
            fig = plt.figure()
            plt.hist([float(n.loss) for n in nodes], bins=30)
            plt.xlabel("loss (float)")
            plt.ylabel("count")
            plt.title("Hyperplane loss distribution")
            fig.tight_layout()
            fig.savefig(run_dir / "loss_hist.png", dpi=200)
            plt.close(fig)

            # Basin sizes histogram (sinks only)
            fig = plt.figure()
            plt.hist(list(basin_sizes.values()), bins=30)
            plt.xlabel("basin size (# start nodes)")
            plt.ylabel("count of sinks")
            plt.title("Basin size distribution (steepest descent)")
            fig.tight_layout()
            fig.savefig(run_dir / "basin_sizes.png", dpi=200)
            plt.close(fig)

            # Barrier histogram (sinks only)
            fig = plt.figure()
            plt.hist([float(barrier_to_global[s]) for s in sinks], bins=30)
            plt.xlabel("barrier to global (float)")
            plt.ylabel("count of sinks")
            plt.title("Energy barriers from sinks to a global minimum")
            fig.tight_layout()
            fig.savefig(run_dir / "barrier_sinks.png", dpi=200)
            plt.close(fig)

            # Plateau sizes
            fig = plt.figure()
            plt.hist(plateau_sizes, bins=30)
            plt.xlabel("plateau component size")
            plt.ylabel("count")
            plt.title("Neutral components (equal-loss subgraph)")
            fig.tight_layout()
            fig.savefig(run_dir / "plateau_sizes.png", dpi=200)
            plt.close(fig)

            # Sink scatter: loss vs basin size
            fig = plt.figure()
            xs = [float(nodes[s].loss) for s in sinks]
            ys = [basin_sizes[s] for s in sinks]
            cs = ["tab:red" if s in set(global_sinks) else "tab:blue" for s in sinks]
            plt.scatter(xs, ys, s=18, alpha=0.75, c=cs)
            plt.xlabel("sink loss (float)")
            plt.ylabel("basin size")
            plt.title("Local minima: loss vs basin size")
            fig.tight_layout()
            fig.savefig(run_dir / "sinks_loss_vs_basin.png", dpi=200)
            plt.close(fig)

        print("Wrote landscape outputs to", run_dir)
        return


class _Stdout:
    # Tiny adapter so csv.writer can write to print()-like output.
    def write(self, s: str) -> int:
        print(s, end="")
        return len(s)


if __name__ == "__main__":
    main()
