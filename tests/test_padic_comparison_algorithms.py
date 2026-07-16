from padic_comparison_algorithms import (
    AffineObservation,
    CnfProblem,
    count_violated_clauses,
    mihara_digitwise_regression,
    parse_dimacs_cnf,
    solve_cnf_mihara_attempt,
    solve_cnf_zubarev,
)


def test_mihara_digitwise_fit_recovers_one_clean_mod_p_digit() -> None:
    observations = (
        AffineObservation((1, 0), 3),
        AffineObservation((0, 1), 5),
        AffineObservation((1, 1), 8),
        AffineObservation((2, 1), 11),
    )

    fit = mihara_digitwise_regression(observations, p=17, seed=2, trials=8)

    assert fit.coefficients == (3, 5)
    assert fit.inliers == len(observations)
    assert fit.active_counts == (len(observations),)


def test_zubarev_boolean_walk_optimises_the_actual_cnf_loss() -> None:
    problem = CnfProblem(("A", "B", "C"), ((1, 2), (-1, 3), (-2, -3)))

    result = solve_cnf_zubarev(problem, seed=7, max_steps=200, restarts=2)

    assert result.solved
    assert result.violated_clauses == 0
    assert count_violated_clauses(problem, result.assignment) == 0


def test_mihara_cnf_attempt_reports_the_equality_model_mismatch() -> None:
    problem = CnfProblem(("A", "B"), ((1,), (2,), (-1, -2)))

    result = solve_cnf_mihara_attempt(problem, seed=1, trials=24)

    assert result.fit.inliers < result.fit.total_observations
    assert result.domain_violations > 0 or (result.violated_clauses or 0) > 0


def test_dimacs_parser_accepts_clauses_split_across_lines() -> None:
    problem = parse_dimacs_cnf("""
        c demo
        p cnf 3 2
        1 -2
        3 0
        -1 0
    """)

    assert problem.variable_names == ("z1", "z2", "z3")
    assert problem.clauses == ((1, -2, 3), (-1,))
