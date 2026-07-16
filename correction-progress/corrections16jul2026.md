# Signed manuscript corrections — 16 July 2026

## Metadata

- Generated at: `2026-07-16T08:11:44.876962+00:00`
- Annotated PDF: `/Users/gregb/Downloads/sudoku_padic_regression-corrections16jul2026.pdf`
- Annotated PDF SHA-256: `e1e76b68c26c7c10025574bfb8269795d5261b6748ac97c0b7424f07da16a2c1`
- Annotated PDF pages: `25`
- Annotated PDF file size: `1267157 bytes`
- Base PDF SHA-256: `75d32996079eaa07ccf402d46b67d269df820ebeebe5cc7be71ba7ec72675261`
- Base PDF pages: `25`
- Base PDF file size: `810581 bytes`
- Preserved base: `/Users/gregb/.codex/document-correction-bases/sudoku-padic-regression/corrections16jul2026/sudoku_padic_regression.pdf`
- Render workspace: `/Users/gregb/Documents/devel/sudoku-padic-regression/scratch/corrections16jul2026/`
- Source file: `paper/sudoku_padic_regression.tex`
- Progress branch: `codex/corrections16jul2026-progress`
- Progress PR: [#33](https://github.com/solresol/sudoku-padic-regression/pull/33)

## Review method

The annotated and base PDFs have identical extracted text. Both PDFs were rendered at 200 dpi and compared page by page. The initial blue-pixel detector was not useful for this export because the handwriting is red and the manuscript itself contains a blue chart. Base-versus-annotated image differencing identified changes on pages 1–5, 10–11, and 13–18; every rendered page was also inspected visually. The small page-12 raster difference is in existing printed text and contains no added ink.

The review found 31 coherent correction units on 13 pages. Each unit was implemented on an independent branch from the same manuscript baseline, built with `make -B paper`, committed, pushed, and opened as its own PR. Generated PDFs were deliberately not committed in the individual correction PRs, avoiding a binary conflict between otherwise independent source changes.

## Correction-to-PR ledger

| ID | PDF page(s) | Intended correction | Branch | PR | Source | Verification |
|---|---:|---|---|---|---|---|
| C01 | 1 | Explain sum-of-squares, weighted variants, and the weighted p-adic objective. | `codex/corrections16jul2026-c01-regression-background` | [#1](https://github.com/solresol/sudoku-padic-regression/pull/1) | introduction | build passed |
| C02 | 1–2 | State the negative-weight novelty and the mechanical CSP dataset/solution transformation. | `codex/corrections16jul2026-c02-negative-weight-contribution` | [#2](https://github.com/solresol/sudoku-padic-regression/pull/2) | introduction | build passed |
| C03 | 2 | Place the dataframe immediately below Figure 1. | `codex/corrections16jul2026-c03-move-dataframe-table` | [#3](https://github.com/solresol/sudoku-padic-regression/pull/3) | worked all-different example | build passed |
| C04 | 3 | Remove the candidate-hyperplane enumeration. | `codex/corrections16jul2026-c04-remove-candidate-enumeration` | [#4](https://github.com/solresol/sudoku-padic-regression/pull/4) | worked all-different example | build passed |
| C05 | 3 | Move the minimum-loss list-colouring result from the caption into the main text. | `codex/corrections16jul2026-c05-move-list-result-to-text` | [#5](https://github.com/solresol/sudoku-padic-regression/pull/5) | Table 2 | build passed |
| C06 | 3 | Present the CNF example as a question about the two Boolean variables. | `codex/corrections16jul2026-c06-rewrite-cnf-prompt` | [#6](https://github.com/solresol/sudoku-padic-regression/pull/6) | worked CNF example | build passed |
| C07 | 4 | Move the unique CNF minimum from the caption into the main text. | `codex/corrections16jul2026-c07-move-cnf-result-to-text` | [#7](https://github.com/solresol/sudoku-padic-regression/pull/7) | Table 4 | build passed |
| C08 | 4 | Remove the operational-form summary paragraph. | `codex/corrections16jul2026-c08-remove-operational-summary` | [#8](https://github.com/solresol/sudoku-padic-regression/pull/8) | worked CNF example | build passed |
| C09 | 4 | Use specific false-label polynomials and explain the degree-based ultrametric. | `codex/corrections16jul2026-c09-specific-polynomial-ultrametric` | [#9](https://github.com/solresol/sudoku-padic-regression/pull/9) | polynomial-valued false labels | build passed |
| C10 | 5 | Remove the repeated p-adic valuation and norm definition. | `codex/corrections16jul2026-c10-remove-repeated-norm-definition` | [#10](https://github.com/solresol/sudoku-padic-regression/pull/10) | general template opening | build passed |
| C11 | 5 | State the distinct-value norm condition directly. | `codex/corrections16jul2026-c11-clarify-distinct-values` | [#11](https://github.com/solresol/sudoku-padic-regression/pull/11) | general template opening | build passed |
| C12 | 10 | Show that NP-hardness also holds for the polynomial degree metric. | `codex/corrections16jul2026-c12-polynomial-degree-hardness` | [#12](https://github.com/solresol/sudoku-padic-regression/pull/12) | CNF compiler theorem | build passed |
| C13 | 10 | Remove the hedge about positive encodings being impossible. | `codex/corrections16jul2026-c13-remove-positive-encoding-hedge` | [#13](https://github.com/solresol/sudoku-padic-regression/pull/13) | complement encodings | build passed |
| C14 | 11 | Add weighted best-effort constraints and the corresponding pinning-weight adjustment. | `codex/corrections16jul2026-c14-weighted-best-effort` | [#14](https://github.com/solresol/sudoku-padic-regression/pull/14) | unsatisfiable instances | build passed |
| C15 | 11 | State the p-adic machine-learning experimental aim for Sudoku-devil instances. | `codex/corrections16jul2026-c15-sudoku-devil-experiments` | [#15](https://github.com/solresol/sudoku-padic-regression/pull/15) | Sudoku introduction | build passed |
| C16 | 13 | Remove the marked lead-in and duplicated-objective qualification from the bound paragraph. | `codex/corrections16jul2026-c16-tighten-sudoku-bound` | [#16](https://github.com/solresol/sudoku-padic-regression/pull/16) | Sudoku bound | build passed |
| C17 | 13 | Remove the ultrametric-residual aside. | `codex/corrections16jul2026-c17-remove-ultrametric-aside` | [#17](https://github.com/solresol/sudoku-padic-regression/pull/17) | Sudoku special case | build passed |
| C18 | 13 | Rename the computations section and remove the illustrative-only disclaimer. | `codex/corrections16jul2026-c18-rename-computations` | [#19](https://github.com/solresol/sudoku-padic-regression/pull/19) | computations section | build passed |
| C19 | 13–14 | Move the heuristic loss definition and explanation to Appendix A. | `codex/corrections16jul2026-c19-move-heuristic-loss-to-appendix` | [#20](https://github.com/solresol/sudoku-padic-regression/pull/20) | computations / Appendix A | build passed |
| C20 | 14 | Explain why row-swap search is a p-adic regression technique and relate it to graph walks. | `codex/corrections16jul2026-c20-explain-row-swap-technique` | [#21](https://github.com/solresol/sudoku-padic-regression/pull/21) | computations section | build passed |
| C21 | 14 | Move representational parsimony before computations and remove the marked heuristic sentence. | `codex/corrections16jul2026-c21-move-parsimony-before-computations` | [#22](https://github.com/solresol/sudoku-padic-regression/pull/22) | Sections 5–6 | build passed |
| C22 | 14 | Remove the positive-only affine residual discussion. | `codex/corrections16jul2026-c22-remove-positive-only-discussion` | [#23](https://github.com/solresol/sudoku-padic-regression/pull/23) | representational parsimony | build passed |
| C23 | 15 | Remove the interactive-section heading while preserving the URL and figure. | `codex/corrections16jul2026-c23-remove-interactive-heading` | [#24](https://github.com/solresol/sudoku-padic-regression/pull/24) | companion-site figure | build passed |
| C24 | 15 | Remove the marked earlier-regression paragraph and contribution claim. | `codex/corrections16jul2026-c24-prune-earlier-regression` | [#25](https://github.com/solresol/sudoku-padic-regression/pull/25) | related work | build passed |
| C25 | 16 | Refocus the Mihara section on why his regression does not apply and remove marked scaffolding. | `codex/corrections16jul2026-c25-refocus-mihara-section` | [#26](https://github.com/solresol/sudoku-padic-regression/pull/26) | Mihara subsection | build passed |
| C26 | 16 | Rephrase the introduction to Mihara's digitwise algorithm. | `codex/corrections16jul2026-c26-describe-digitwise-algorithm` | [#27](https://github.com/solresol/sudoku-padic-regression/pull/27) | Mihara subsection | build passed |
| C27 | 17 | Remove the equal-characteristic aside. | `codex/corrections16jul2026-c27-remove-equal-characteristic-aside` | [#28](https://github.com/solresol/sudoku-padic-regression/pull/28) | Mihara subsection | build passed |
| C28 | 17 | Remove the word “exactly” from the finite-domain comparison. | `codex/corrections16jul2026-c28-remove-exactly-claim` | [#29](https://github.com/solresol/sudoku-padic-regression/pull/29) | Mihara subsection | build passed |
| C29 | 17 | Replace the drop-in-replacement sentence with the NP-hardness explanation. | `codex/corrections16jul2026-c29-explain-mihara-limitation` | [#30](https://github.com/solresol/sudoku-padic-regression/pull/30) | Mihara subsection | build passed |
| C30 | 17 | Change the companion-site cross-reference from the removed section to the figure. | `codex/corrections16jul2026-c30-reference-companion-figure` | [#31](https://github.com/solresol/sudoku-padic-regression/pull/31) | conclusion | build passed |
| C31 | 17–18 | Remove the appendix NP-hardness comparison and its dangling “however”. | `codex/corrections16jul2026-c31-remove-appendix-nphard-intro` | [#32](https://github.com/solresol/sudoku-padic-regression/pull/32) | Appendix A | build passed |

## Page ledger

| Page | Review status | Correction IDs / notes |
|---:|---|---|
| 001 | annotated; fully mapped | C01, C02 |
| 002 | annotated; fully mapped | C02, C03 |
| 003 | annotated; fully mapped | C04, C05, C06 |
| 004 | annotated; fully mapped | C07, C08, C09 |
| 005 | annotated; fully mapped | C10, C11 |
| 006 | reviewed; no annotations | no added ink |
| 007 | reviewed; no annotations | pixel-identical to base |
| 008 | reviewed; no annotations | pixel-identical to base |
| 009 | reviewed; no annotations | pixel-identical to base |
| 010 | annotated; fully mapped | C12, C13 |
| 011 | annotated; fully mapped | C14, C15 |
| 012 | reviewed; no annotations | small printed-text raster difference only |
| 013 | annotated; fully mapped | C16, C17, C18, C19 |
| 014 | annotated; fully mapped | C19, C20, C21, C22 |
| 015 | annotated; fully mapped | C23, C24 |
| 016 | annotated; fully mapped | C25, C26 |
| 017 | annotated; fully mapped | C27, C28, C29, C30, C31 |
| 018 | annotated; fully mapped | C31 |
| 019 | reviewed; no annotations | pixel-identical to base |
| 020 | reviewed; no annotations | pixel-identical to base |
| 021 | reviewed; no annotations | pixel-identical to base |
| 022 | reviewed; no annotations | pixel-identical to base |
| 023 | reviewed; no annotations | pixel-identical to base; blue chart is part of base |
| 024 | reviewed; no annotations | pixel-identical to base |
| 025 | reviewed; no annotations | pixel-identical to base |

## Completeness audit before merge

- Annotated PDF pages reviewed: **25 / 25**.
- Pages containing added annotation ink: **13**.
- Coherent correction units found: **31**.
- Correction units with a dedicated PR: **31 / 31**.
- Correction branches present in the live GitHub open-PR list: **31 / 31**.
- Unmapped annotations: **0**.
- Ambiguous or deferred corrections: **0**.
- Individual branch build verification: **31 / 31 passed** with `make -B paper`.

This ledger confirms that every change marked in `sudoku_padic_regression-corrections16jul2026.pdf` appears in a dedicated pull request before merging begins.
