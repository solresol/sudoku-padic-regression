"""
Investigate the generalization of the forcing lemma to {1,...,9}.

For {0,1}, the paper proves: |a|_2 + |a-1|_2 >= 1, with equality iff a in {0,1}.

For {1,...,9}, we want to understand:
  f_p(a) = sum_{k=1}^{9} |a - k|_p

Questions:
1. For which p is f_p minimized at integers 1-9?
2. What is the gap between f_p(integer) and f_p(non-integer)?
3. Which p gives the best separation?
"""

from fractions import Fraction
from typing import Callable
import math


def v_p(x: Fraction, p: int) -> int | float:
    """p-adic valuation of a rational number."""
    if x == 0:
        return float("inf")

    # Factor out p from numerator and denominator
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
    """p-adic absolute value of a rational number."""
    val = v_p(x, p)
    if val == float("inf"):
        return Fraction(0)
    return Fraction(1, p ** val) if val >= 0 else Fraction(p ** (-val), 1)


def forcing_sum(a: Fraction, p: int, targets: list[int]) -> Fraction:
    """
    Compute sum_{k in targets} |a - k|_p

    This is the "forcing" term that should push a toward one of the targets.
    """
    return sum(padic_abs(a - k, p) for k in targets)


def analyze_forcing_for_prime(p: int, targets: list[int] = list(range(1, 10))):
    """Analyze the forcing function for a given prime."""
    print(f"\n{'='*60}")
    print(f"Prime p = {p}")
    print(f"Targets: {targets}")
    print(f"{'='*60}")

    # Compute f_p at each target
    print("\nValues at integer targets:")
    target_values = []
    for k in targets:
        val = forcing_sum(Fraction(k), p, targets)
        target_values.append(float(val))
        print(f"  f_{p}({k}) = {val} = {float(val):.4f}")

    min_at_targets = min(target_values)
    max_at_targets = max(target_values)
    print(f"\nRange at targets: [{min_at_targets:.4f}, {max_at_targets:.4f}]")

    # Test some non-integer values
    print("\nValues at non-integers:")
    test_values = [
        Fraction(1, 2),      # 0.5
        Fraction(3, 2),      # 1.5
        Fraction(5, 2),      # 2.5
        Fraction(7, 2),      # 3.5
        Fraction(9, 2),      # 4.5
        Fraction(11, 2),     # 5.5
        Fraction(1, 3),      # 0.333...
        Fraction(4, 3),      # 1.333...
        Fraction(7, 3),      # 2.333...
        Fraction(10, 3),     # 3.333...
        Fraction(1, p),      # 1/p
        Fraction(1, p**2),   # 1/p^2
    ]

    non_integer_min = float("inf")
    for a in test_values:
        val = forcing_sum(a, p, targets)
        float_val = float(val)
        non_integer_min = min(non_integer_min, float_val)
        comparison = "< min_target!" if float_val < min_at_targets else ""
        print(f"  f_{p}({a} = {float(a):.4f}) = {float_val:.4f} {comparison}")

    # Check the gap
    print(f"\nMinimum at non-integers (sampled): {non_integer_min:.4f}")
    print(f"Gap (non-integer min - target min): {non_integer_min - min_at_targets:.4f}")

    if non_integer_min < min_at_targets:
        print("WARNING: Non-integer achieves lower value than targets!")
    elif non_integer_min > min_at_targets:
        print("GOOD: Targets achieve strictly lower values than tested non-integers.")
    else:
        print("TIE: Some non-integer matches target minimum.")

    return min_at_targets, non_integer_min


def find_best_prime():
    """Find which prime gives the best separation."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    targets = list(range(1, 10))

    print("\n" + "="*60)
    print("SUMMARY: Comparing primes for forcing {1,...,9}")
    print("="*60)

    results = []
    for p in primes:
        min_target, min_non_int = analyze_forcing_for_prime(p, targets)
        gap = min_non_int - min_target
        results.append((p, min_target, min_non_int, gap))

    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    print(f"{'Prime':>6} {'Min@Targets':>12} {'Min@NonInt':>12} {'Gap':>10}")
    print("-"*44)
    for p, mt, mni, gap in results:
        status = "OK" if gap > 0 else "BAD"
        print(f"{p:>6} {mt:>12.4f} {mni:>12.4f} {gap:>10.4f}  {status}")


if __name__ == "__main__":
    find_best_prime()
