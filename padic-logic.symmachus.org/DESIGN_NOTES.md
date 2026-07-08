# Design Notes

## Revision 2 Direction

- Do not include a p-adic space preview, Euclidean coordinate plot, scatterplot, or 3D hyperplane visualization. The dimensionality is high, and the p-adic structure should not be depicted as ordinary Euclidean geometry.
- The first implementation should center on brute-force search over truth assignments.
- The compiled CSP clause form should use only conjunction of disjunctive clauses: `^` between clauses and `v` inside clauses. `xor` is not a target primitive; if accepted in input, it must be expanded into ordinary CNF clauses before display, scoring, or evaluator generation.
- Search can be modeled as integer masks: assign each boolean variable to a bit position, split contiguous integer ranges across workers, and evaluate constraints by extracting bits.
- The UI may mention random search as a possible alternate strategy, but the primary strategy is exhaustive brute force.
- The implementation does not need to calculate explicit hyperplanes or perform rational arithmetic for the first version.
- The solver can still present the approach conceptually as p-adic regression, but the execution path can be a generated JavaScript evaluator for close-to-the-metal browser performance.
- Report p-adic loss over time, including the theoretical minimum possible loss. The minimum is reached when all constraints that are not unit wells score zero.
- The worker screen should show assignment counts, masks tested, masks per second, worker ranges, current best assignment, current best loss, and progress toward the theoretical floor.
- The browser-local language model feature, where available in Chrome or Edge, should produce an editable CSP form from natural language before the same compile-and-run flow.

## Concept Set

- `concepts/v2-01-csp-entry-bit-search.png`: manual CSP entry, ternary reduction, generated JavaScript evaluator, and brute-force search plan.
- `concepts/v2-02-language-model-bit-search.png`: Chrome/Edge language-model intake, generated CSP review, and compile/search handoff.
- `concepts/v2-03-worker-bit-search.png`: active worker-thread exhaustive search with integer masks and p-adic loss over time.
