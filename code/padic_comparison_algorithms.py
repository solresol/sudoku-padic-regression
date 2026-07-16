"""Comparison algorithms for the signed p-adic CSP/Sudoku constructions.

The Zubarev routine below optimises the actual finite-domain signed loss.  The
Mihara routine intentionally applies an equality-regression algorithm to the
same synthetic rows, even when a row was constructed as a forbidden equality
or an inequality reward.  That second routine is a diagnostic counterexample:
it implements the digitwise outer loop and a RANSAC-style modulo-p fit, but the
statistical model it assumes is absent from CSP/CNF and Sudoku data.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class AffineObservation:
    """One equality sample ``features . coefficients = target``."""

    features: tuple[int, ...]
    target: int
    source: str = ""


@dataclass(frozen=True)
class MiharaDigitFit:
    coefficients: tuple[int, ...]
    inliers: int
    total_observations: int
    precision: int
    active_counts: tuple[int, ...]
    successful_trials: int
    singular_trials: int


@dataclass(frozen=True)
class CnfProblem:
    """CNF with signed, one-based literals (``1`` is z1, ``-1`` is not z1)."""

    variable_names: tuple[str, ...]
    clauses: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.variable_names:
            raise ValueError("A CNF problem needs at least one variable.")
        if not self.clauses:
            raise ValueError("A CNF problem needs at least one clause.")
        dimension = len(self.variable_names)
        for clause in self.clauses:
            if not clause:
                raise ValueError("Empty clauses are not supported by this demonstration.")
            if any(literal == 0 or abs(literal) > dimension for literal in clause):
                raise ValueError(f"Literal outside 1..{dimension}: {clause}")


@dataclass(frozen=True)
class CnfSearchResult:
    assignment: tuple[bool, ...]
    violated_clauses: int
    steps: int
    solved: bool
    method: str
    beta: float | None = None


@dataclass(frozen=True)
class MiharaCnfResult:
    fit: MiharaDigitFit
    assignment: tuple[bool, ...] | None
    violated_clauses: int | None
    domain_violations: int


def _mod(value: int, p: int) -> int:
    return value % p


def _inverse_mod(value: int, p: int) -> int:
    value %= p
    if value == 0:
        raise ZeroDivisionError("zero has no inverse modulo p")
    return pow(value, -1, p)


def solve_square_system_mod_p(
    matrix: Sequence[Sequence[int]],
    targets: Sequence[int],
    p: int,
) -> tuple[int, ...] | None:
    """Solve a square system over F_p, returning ``None`` when singular."""

    dimension = len(matrix)
    if dimension == 0 or len(targets) != dimension:
        raise ValueError("Expected a non-empty square system.")
    if any(len(row) != dimension for row in matrix):
        raise ValueError("Expected a square matrix.")

    augmented = [
        [_mod(value, p) for value in row] + [_mod(target, p)]
        for row, target in zip(matrix, targets, strict=True)
    ]

    for column in range(dimension):
        pivot = next(
            (row for row in range(column, dimension) if augmented[row][column] % p),
            None,
        )
        if pivot is None:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        inverse = _inverse_mod(augmented[column][column], p)
        augmented[column] = [(_mod(value * inverse, p)) for value in augmented[column]]

        for row in range(dimension):
            if row == column:
                continue
            factor = augmented[row][column] % p
            if factor == 0:
                continue
            augmented[row] = [
                _mod(left - factor * right, p)
                for left, right in zip(augmented[row], augmented[column], strict=True)
            ]

    return tuple(augmented[row][-1] % p for row in range(dimension))


def _residual(observation: AffineObservation, coefficients: Sequence[int]) -> int:
    return observation.target - sum(
        feature * coefficient
        for feature, coefficient in zip(observation.features, coefficients, strict=True)
    )


def _fit_last_digit(
    observations: Sequence[AffineObservation],
    p: int,
    rng: random.Random,
    trials: int,
) -> tuple[tuple[int, ...], int, int, int]:
    dimension = len(observations[0].features)
    if len(observations) < dimension:
        raise ValueError(
            f"Need at least {dimension} observations to recover {dimension} coefficients."
        )

    best: tuple[int, ...] | None = None
    best_inliers = -1
    successful_trials = 0
    singular_trials = 0

    for _ in range(trials):
        sample = rng.sample(observations, dimension)
        candidate = solve_square_system_mod_p(
            [observation.features for observation in sample],
            [observation.target for observation in sample],
            p,
        )
        if candidate is None:
            singular_trials += 1
            continue
        successful_trials += 1
        inliers = sum(_residual(observation, candidate) % p == 0 for observation in observations)
        if inliers > best_inliers or (inliers == best_inliers and (best is None or candidate < best)):
            best = candidate
            best_inliers = inliers

    if best is None:
        # Deterministically try every cyclic window before admitting that the
        # synthetic dataframe contains no full-rank equality sample.
        for start in range(len(observations)):
            sample = [observations[(start + offset) % len(observations)] for offset in range(dimension)]
            candidate = solve_square_system_mod_p(
                [observation.features for observation in sample],
                [observation.target for observation in sample],
                p,
            )
            if candidate is None:
                singular_trials += 1
                continue
            successful_trials += 1
            inliers = sum(
                _residual(observation, candidate) % p == 0 for observation in observations
            )
            if inliers > best_inliers or (
                inliers == best_inliers and (best is None or candidate < best)
            ):
                best = candidate
                best_inliers = inliers

    if best is None:
        return tuple(0 for _ in range(dimension)), 0, successful_trials, singular_trials
    return best, best_inliers, successful_trials, singular_trials


def mihara_digitwise_regression(
    observations: Sequence[AffineObservation],
    *,
    p: int,
    precision: int = 1,
    seed: int = 0,
    trials: int = 96,
) -> MiharaDigitFit:
    """Attempt Mihara's digitwise equality recovery on integer observations.

    This is the fixed-zero-intercept specialisation used by the repository's
    residual dataframes.  Each digit is estimated by a RANSAC-style modulo-p
    solver, the coefficient is lifted, and only exact rows modulo the next power
    of p remain active.  The caller is responsible for checking whether treating
    its rows as samples from one hidden equality is mathematically appropriate.
    """

    if not observations:
        raise ValueError("At least one observation is required.")
    if p < 2:
        raise ValueError("p must be prime and at least 2.")
    if precision < 1:
        raise ValueError("precision must be at least 1.")
    if trials < 1:
        raise ValueError("trials must be at least 1.")

    dimension = len(observations[0].features)
    if dimension == 0 or any(len(observation.features) != dimension for observation in observations):
        raise ValueError("All observations must have the same positive dimension.")

    rng = random.Random(seed)
    coefficients = [0] * dimension
    active = list(observations)
    active_counts: list[int] = []
    successful_trials = 0
    singular_trials = 0

    for exponent in range(precision):
        scale = p**exponent
        digit_observations = [
            AffineObservation(
                tuple(feature % p for feature in observation.features),
                (_residual(observation, coefficients) // scale) % p,
                observation.source,
            )
            for observation in active
        ]
        if len(digit_observations) < dimension:
            break
        digit, _, successes, singular = _fit_last_digit(
            digit_observations,
            p,
            rng,
            trials,
        )
        successful_trials += successes
        singular_trials += singular
        coefficients = [
            coefficient + scale * digit_value
            for coefficient, digit_value in zip(coefficients, digit, strict=True)
        ]
        modulus = p ** (exponent + 1)
        active = [
            observation
            for observation in active
            if _residual(observation, coefficients) % modulus == 0
        ]
        active_counts.append(len(active))

    final_modulus = p**precision
    inliers = sum(
        _residual(observation, coefficients) % final_modulus == 0
        for observation in observations
    )
    return MiharaDigitFit(
        coefficients=tuple(coefficient % final_modulus for coefficient in coefficients),
        inliers=inliers,
        total_observations=len(observations),
        precision=precision,
        active_counts=tuple(active_counts),
        successful_trials=successful_trials,
        singular_trials=singular_trials,
    )


def parse_dimacs_cnf(source: str) -> CnfProblem:
    """Parse a small DIMACS CNF string for the command-line comparison demo."""

    declared_variables = 0
    clauses: list[tuple[int, ...]] = []
    pending: list[int] = []
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("p"):
            fields = line.split()
            if len(fields) != 4 or fields[1].lower() != "cnf":
                raise ValueError("Expected a DIMACS header such as 'p cnf 3 2'.")
            declared_variables = int(fields[2])
            continue
        for field in line.split():
            literal = int(field)
            if literal == 0:
                if not pending:
                    raise ValueError("Empty DIMACS clauses are not supported.")
                clauses.append(tuple(pending))
                pending = []
            else:
                pending.append(literal)
    if pending:
        raise ValueError("The final DIMACS clause must end with 0.")
    inferred_variables = max((abs(literal) for clause in clauses for literal in clause), default=0)
    variable_count = declared_variables or inferred_variables
    if variable_count < inferred_variables:
        raise ValueError("A literal exceeds the variable count in the DIMACS header.")
    return CnfProblem(
        tuple(f"z{index}" for index in range(1, variable_count + 1)),
        tuple(clauses),
    )


def clause_is_satisfied(clause: Sequence[int], assignment: Sequence[bool]) -> bool:
    return any(
        assignment[abs(literal) - 1] if literal > 0 else not assignment[abs(literal) - 1]
        for literal in clause
    )


def count_violated_clauses(problem: CnfProblem, assignment: Sequence[bool]) -> int:
    return sum(not clause_is_satisfied(clause, assignment) for clause in problem.clauses)


def cnf_regression_observations(problem: CnfProblem) -> tuple[AffineObservation, ...]:
    """Return the signed-CNF rows after deliberately forgetting ``!=`` semantics."""

    dimension = len(problem.variable_names)
    observations: list[AffineObservation] = []
    for index, clause in enumerate(problem.clauses, start=1):
        features = [0] * dimension
        target = 0
        for literal in clause:
            variable = abs(literal) - 1
            if literal > 0:
                features[variable] += 1
                target += 1
            else:
                features[variable] -= 1
        observations.append(
            AffineObservation(
                tuple(features),
                target,
                f"C{index}: forbidden hyperplane treated as an equality",
            )
        )

    for variable in range(dimension):
        features = tuple(1 if index == variable else 0 for index in range(dimension))
        observations.append(AffineObservation(features, 0, f"{problem.variable_names[variable]} = 0"))
        observations.append(AffineObservation(features, 1, f"{problem.variable_names[variable]} = 1"))
    return tuple(observations)


def solve_cnf_mihara_attempt(
    problem: CnfProblem,
    *,
    p: int = 17,
    seed: int = 0,
    trials: int = 96,
) -> MiharaCnfResult:
    fit = mihara_digitwise_regression(
        cnf_regression_observations(problem),
        p=p,
        precision=1,
        seed=seed,
        trials=trials,
    )
    domain_violations = sum(coefficient not in (0, 1) for coefficient in fit.coefficients)
    if domain_violations:
        return MiharaCnfResult(fit, None, None, domain_violations)
    # The paper's Boolean encoding is x_i=0 for true and x_i=1 for false.
    assignment = tuple(coefficient == 0 for coefficient in fit.coefficients)
    return MiharaCnfResult(
        fit,
        assignment,
        count_violated_clauses(problem, assignment),
        0,
    )


def _sample_log_weights(log_weights: Sequence[float], rng: random.Random) -> int:
    maximum = max(log_weights)
    weights = [math.exp(weight - maximum) for weight in log_weights]
    threshold = rng.random() * sum(weights)
    for index, weight in enumerate(weights):
        threshold -= weight
        if threshold <= 0:
            return index
    return len(weights) - 1


def solve_cnf_zubarev(
    problem: CnfProblem,
    *,
    seed: int = 0,
    max_steps: int = 20_000,
    restarts: int = 8,
    beta0: float = 0.5,
    beta1: float = 6.0,
) -> CnfSearchResult:
    """Discrete Zubarev walk over single-bit Boolean moves."""

    if max_steps < 1 or restarts < 1:
        raise ValueError("max_steps and restarts must be positive.")
    if beta0 < 0 or beta1 < 0:
        raise ValueError("beta values must be non-negative.")
    rng = random.Random(seed)
    dimension = len(problem.variable_names)
    best_assignment: tuple[bool, ...] | None = None
    best_violations = len(problem.clauses) + 1
    total_steps = 0
    last_beta = beta0

    for _ in range(restarts):
        assignment = [rng.choice((False, True)) for _ in range(dimension)]
        violations = count_violated_clauses(problem, assignment)
        if violations < best_violations:
            best_assignment = tuple(assignment)
            best_violations = violations
        if violations == 0:
            return CnfSearchResult(tuple(assignment), 0, total_steps, True, "zubarev", beta0)

        for step in range(max_steps):
            fraction = step / max(max_steps - 1, 1)
            last_beta = beta0 * (1 - fraction) + beta1 * fraction
            candidates: list[tuple[int, int]] = []
            log_weights: list[float] = []
            for variable in range(dimension):
                assignment[variable] = not assignment[variable]
                candidate_violations = count_violated_clauses(problem, assignment)
                assignment[variable] = not assignment[variable]
                delta = candidate_violations - violations
                candidates.append((variable, candidate_violations))
                log_weights.append(-last_beta * delta)

            variable, candidate_violations = candidates[_sample_log_weights(log_weights, rng)]
            assignment[variable] = not assignment[variable]
            violations = candidate_violations
            total_steps += 1
            if violations < best_violations:
                best_assignment = tuple(assignment)
                best_violations = violations
            if violations == 0:
                return CnfSearchResult(
                    tuple(assignment),
                    0,
                    total_steps,
                    True,
                    "zubarev",
                    last_beta,
                )

    assert best_assignment is not None
    return CnfSearchResult(
        best_assignment,
        best_violations,
        total_steps,
        False,
        "zubarev",
        last_beta,
    )


def _render_assignment(problem: CnfProblem, assignment: Iterable[bool]) -> str:
    return ", ".join(
        f"{name}={'true' if value else 'false'}"
        for name, value in zip(problem.variable_names, assignment, strict=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a Zubarev Boolean walk with a Mihara digitwise misapplication."
    )
    parser.add_argument(
        "--dimacs",
        help="Path to a DIMACS CNF file. Defaults to a small built-in satisfiable example.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--mihara-trials", type=int, default=96)
    args = parser.parse_args()

    if args.dimacs:
        with open(args.dimacs, encoding="utf-8") as handle:
            problem = parse_dimacs_cnf(handle.read())
    else:
        problem = CnfProblem(
            ("A", "B", "C"),
            ((1, 2), (-1, 3), (-2, -3)),
        )

    zubarev = solve_cnf_zubarev(
        problem,
        seed=args.seed,
        max_steps=args.steps,
        restarts=args.restarts,
    )
    print("Zubarev walk")
    print("  solved:", zubarev.solved)
    print("  violated clauses:", zubarev.violated_clauses)
    print("  assignment:", _render_assignment(problem, zubarev.assignment))

    mihara = solve_cnf_mihara_attempt(
        problem,
        seed=args.seed,
        trials=args.mihara_trials,
    )
    print("\nMihara digitwise equality fit (deliberate model mismatch)")
    print(f"  equality inliers: {mihara.fit.inliers}/{mihara.fit.total_observations}")
    print(
        "  RANSAC samples:",
        f"{mihara.fit.successful_trials} full-rank, {mihara.fit.singular_trials} singular",
    )
    print("  raw coefficients mod 17:", mihara.fit.coefficients)
    print("  non-Boolean coefficients:", mihara.domain_violations)
    if mihara.assignment is None:
        print("  result: cannot decode as a Boolean assignment")
    else:
        print("  assignment:", _render_assignment(problem, mihara.assignment))
        print("  violated clauses:", mihara.violated_clauses)
    print(
        "  diagnosis: clause rows encode forbidden equalities, so equality recovery rewards the wrong event."
    )


if __name__ == "__main__":
    main()
