# p-adic Sudoku Lifting Experiment Results

## Executive Summary

We tested whether p-adic norms can guide local search heuristics to solve Sudoku puzzles. The answer is **no** - at least not with naive greedy or simulated annealing approaches.

**Key finding:** All primes tested achieved 0% success rate with both heuristics. The algorithms get stuck at local minima within 3-29 steps, unable to make progress toward valid solutions.

This is a meaningful negative result: the p-adic objective landscape is highly non-convex with abundant local minima that trap local search.

---

## Experiment E1: Prime Sweep

Tested 15 primes on 50 Euler Project puzzles using greedy cell-swap heuristic. Each (puzzle, prime) combination was run 3 times with different random initializations.

### Prime Performance Summary

| Prime | Success Rate | Mean Lift Frac | Required Level | Mean Steps |
|-------|--------------|----------------|----------------|------------|
|     2 |         0.0% |          0.036 |             12 |         29 |
|     3 |         0.0% |          0.002 |              8 |         19 |
|     5 |         0.0% |          0.008 |              5 |         17 |
|     7 |         0.0% |          0.000 |              4 |         18 |
|    11 |         0.0% |          0.000 |              4 |         14 |
|    13 |         0.0% |          0.000 |              4 |         14 |
|    17 |         0.0% |          0.000 |              3 |         14 |
|    23 |         0.0% |          0.000 |              3 |         14 |
|    47 |         0.0% |          0.000 |              3 |         12 |
|    73 |         0.0% |          0.000 |              2 |         11 |
|    97 |         0.0% |          0.000 |              2 |          8 |
|   127 |         0.0% |          0.000 |              2 |          7 |
|   257 |         0.0% |          0.000 |              2 |          4 |
|   521 |         0.0% |          0.000 |              2 |          3 |
|  2311 |         0.0% |          0.000 |              1 |          3 |

### Analysis

**Rapid convergence to local minima:** The mean steps column reveals that all primes get stuck almost immediately. Larger primes (which require fewer lift levels) get stuck faster - p=2311 averages only 3 steps before no improvement is possible.

**p=2 achieves the highest lift fraction (0.036):** This is because p=2 requires 12 lift levels, and even a single level of progress registers as ~0.08 progress. Other primes with lower required levels show 0.000 lift fraction, meaning they couldn't even complete a single lift level.

**Why the greedy heuristic fails:** With 27 constraints and 81 cells, there are abundant configurations where changing any single cell makes at least one constraint worse. The greedy approach cannot escape these local minima.

### Special Prime Analysis

**p = 2:**
- Interacts with power-of-2 encoding: all allowed values except 1 are even
- Highest lift fraction (0.036) but still fails completely
- Required 12 lift levels is the hardest target

**p = 7:**
- Divides 511 (the target sum), giving v_7(511) = 1
- This means correct sums automatically have v_7 >= 1
- Despite this "free first lift", cannot progress further

**p = 73:**
- Also divides 511 (511 = 7 * 73), so v_73(511) = 1
- Same structural advantage as p=7, but still fails
- Only needs 2 lift levels, yet cannot achieve even one

---

## Experiment E2: Heuristic Comparison

Compared greedy cell-swap vs simulated annealing on the 5 primes with highest E1 performance.

| Heuristic | Success Rate | Mean Lift Frac | Mean Time (s) |
|-----------|--------------|----------------|---------------|
|    greedy |         0.0% |          0.009 |          0.31 |
|        sa |         0.0% |          0.000 |          0.27 |

### Analysis

**Simulated annealing performs worse than greedy:** Despite its ability to accept uphill moves and escape local minima, SA achieves lower lift fractions. This suggests the p-adic landscape is not just locally rough but globally pathological for these search methods.

**SA's random moves destroy progress:** The greedy algorithm at least makes some progress before getting stuck. SA's random perturbations appear to undo any gains made.

---

## Conclusions

### Why P-adic Lifting Fails for Local Search

1. **Combinatorial explosion of constraints:** With 27 overlapping constraints, improving one often hurts others. The p-adic norm provides no gradient-like signal to navigate this.

2. **Discrete search space:** Unlike continuous optimization where gradient descent can follow infinitesimal improvements, we can only make discrete jumps between allowed values.

3. **No structure preservation:** Random or greedy cell changes don't respect Sudoku structure. A change that improves one row may catastrophically break its column and box.

### What Might Work Instead

1. **Constraint propagation:** Use standard Sudoku inference (naked singles, hidden singles, etc.) to reduce the search space before applying p-adic techniques.

2. **Row/column permutation moves:** Instead of changing individual cell values, swap entire rows or columns within a band/stack to preserve more structure.

3. **Population-based methods:** Genetic algorithms or ant colony optimization might explore multiple regions of the search space simultaneously.

4. **Hybrid approach:** Use p-adic norms to guide which cells to modify, but use backtracking to actually make valid assignments.

5. **Different encoding:** The power-of-2 encoding creates a very sparse valid solution set. Alternative encodings might create smoother landscapes.

---

## Experimental Details

- **Puzzles:** 50 Euler Project Problem 96 puzzles
- **Primes tested:** 2, 3, 5, 7, 11, 13, 17, 23, 47, 73, 97, 127, 257, 521, 2311
- **Random initializations:** 3 per (puzzle, prime, heuristic) combination
- **Greedy parameters:** max 2000 steps
- **SA parameters:** max 5000 steps, T_start=2.0, cooling_rate=0.9997
- **Total runtime:** ~8 minutes

---

## Files Generated

- `results/e1_prime_sweep.csv` - Raw data from prime sweep experiment
- `results/e2_heuristic_comparison.csv` - Raw data from heuristic comparison
- `figures/e1_prime_success_rate.png` - Bar chart of success rates by prime
- `figures/e1_prime_puzzle_heatmap.png` - Heatmap of lift fractions by prime and puzzle
- `figures/e1_lift_level_vs_success.png` - Scatter plot of required lift level vs success
- `figures/e2_heuristic_comparison.png` - Comparison of heuristics across primes
