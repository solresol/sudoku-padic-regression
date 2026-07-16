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

## Merge results

- Correction PRs merged: **31 / 31**.
- Correction PRs remaining open: **0**.
- Final generated-PDF PR: [#34](https://github.com/solresol/sudoku-padic-regression/pull/34), merged as [`a975a1c`](https://github.com/solresol/sudoku-padic-regression/commit/a975a1cd6aa3643ba6455bf5c93d8b0b8a20605d).
- Final manuscript: **25 A4 pages**, with `paper/sudoku_padic_regression.pdf` byte-identical to `site/sudoku_padic_regression.pdf` before publication.
- Final verification: clean LaTeX rebuild with no undefined references or citations; all 25 rendered pages visually reviewed; Python tests **10 / 10 passed**; site tests **67 / 67 passed**; production site build passed.

| ID | PR | Merge commit |
|---|---:|---|
| C01 | [#1](https://github.com/solresol/sudoku-padic-regression/pull/1) | [`15a29d2`](https://github.com/solresol/sudoku-padic-regression/commit/15a29d2aab9ea3652ee56c36e79b12764512e076) |
| C02 | [#2](https://github.com/solresol/sudoku-padic-regression/pull/2) | [`98c3657`](https://github.com/solresol/sudoku-padic-regression/commit/98c3657f95eaf815d341b39cef7871423998bb03) |
| C03 | [#3](https://github.com/solresol/sudoku-padic-regression/pull/3) | [`cce7fe2`](https://github.com/solresol/sudoku-padic-regression/commit/cce7fe23ab1e4b8e73f274a68f5354f3b10c25f0) |
| C04 | [#4](https://github.com/solresol/sudoku-padic-regression/pull/4) | [`60c6bc4`](https://github.com/solresol/sudoku-padic-regression/commit/60c6bc4a047687ff761172f42276b4d6b06b293a) |
| C05 | [#5](https://github.com/solresol/sudoku-padic-regression/pull/5) | [`e2b268d`](https://github.com/solresol/sudoku-padic-regression/commit/e2b268d27c19ae1325f0a9c6d00f6127ff7e2dea) |
| C06 | [#6](https://github.com/solresol/sudoku-padic-regression/pull/6) | [`8f49646`](https://github.com/solresol/sudoku-padic-regression/commit/8f4964695b2ecef04c35c7ddc8937de0eca58d03) |
| C07 | [#7](https://github.com/solresol/sudoku-padic-regression/pull/7) | [`dc86749`](https://github.com/solresol/sudoku-padic-regression/commit/dc86749d88d9281acb7be74877e24e2729c7bee6) |
| C08 | [#8](https://github.com/solresol/sudoku-padic-regression/pull/8) | [`84d5199`](https://github.com/solresol/sudoku-padic-regression/commit/84d519996f97e91dea229671ee0b8fac9d0e04da) |
| C09 | [#9](https://github.com/solresol/sudoku-padic-regression/pull/9) | [`fa9af66`](https://github.com/solresol/sudoku-padic-regression/commit/fa9af66b7966a9986a38d8be49ea1922a2007551) |
| C10 | [#10](https://github.com/solresol/sudoku-padic-regression/pull/10) | [`92db1fa`](https://github.com/solresol/sudoku-padic-regression/commit/92db1fa5b3fa1e2a12c9259abf3e9604ca14140e) |
| C11 | [#11](https://github.com/solresol/sudoku-padic-regression/pull/11) | [`c13e4e8`](https://github.com/solresol/sudoku-padic-regression/commit/c13e4e8b257f61bb4e6bfcf51421402def3c24a2) |
| C12 | [#12](https://github.com/solresol/sudoku-padic-regression/pull/12) | [`879fa51`](https://github.com/solresol/sudoku-padic-regression/commit/879fa5177b847fc1cd2cf49d803d9dee2244bfea) |
| C13 | [#13](https://github.com/solresol/sudoku-padic-regression/pull/13) | [`932a107`](https://github.com/solresol/sudoku-padic-regression/commit/932a107afa1b04cf8f88ab069bd6fade80bf478e) |
| C14 | [#14](https://github.com/solresol/sudoku-padic-regression/pull/14) | [`f43b1c8`](https://github.com/solresol/sudoku-padic-regression/commit/f43b1c80bd5a2993cd97c364f90ec14f2a645935) |
| C15 | [#15](https://github.com/solresol/sudoku-padic-regression/pull/15) | [`dba2568`](https://github.com/solresol/sudoku-padic-regression/commit/dba256874262984034b0ae13d00e3b07eb6a6219) |
| C16 | [#16](https://github.com/solresol/sudoku-padic-regression/pull/16) | [`e3ab11e`](https://github.com/solresol/sudoku-padic-regression/commit/e3ab11e05746f0a18a991161eb95653d0d7376af) |
| C17 | [#17](https://github.com/solresol/sudoku-padic-regression/pull/17) | [`6576595`](https://github.com/solresol/sudoku-padic-regression/commit/65765952cab950d1477629f4cb7ca9a31a9e916e) |
| C18 | [#19](https://github.com/solresol/sudoku-padic-regression/pull/19) | [`2b36ece`](https://github.com/solresol/sudoku-padic-regression/commit/2b36ece623cfca45c399eb9872e14473a5800905) |
| C19 | [#20](https://github.com/solresol/sudoku-padic-regression/pull/20) | [`25d64df`](https://github.com/solresol/sudoku-padic-regression/commit/25d64df2e5e23647478698d1617b65ff8672641b) |
| C20 | [#21](https://github.com/solresol/sudoku-padic-regression/pull/21) | [`9374ac6`](https://github.com/solresol/sudoku-padic-regression/commit/9374ac6ee64b636d1c16027f0be90670786bf6e4) |
| C21 | [#22](https://github.com/solresol/sudoku-padic-regression/pull/22) | [`6fa19d6`](https://github.com/solresol/sudoku-padic-regression/commit/6fa19d6ad84649610902ae3bd54f83a19eb2316a) |
| C22 | [#23](https://github.com/solresol/sudoku-padic-regression/pull/23) | [`89781cb`](https://github.com/solresol/sudoku-padic-regression/commit/89781cb6736fdd7724e1494dfb12f934a6290d38) |
| C23 | [#24](https://github.com/solresol/sudoku-padic-regression/pull/24) | [`bed4e23`](https://github.com/solresol/sudoku-padic-regression/commit/bed4e238d5cc3ff35ab4398ad304f40ae0528cfd) |
| C24 | [#25](https://github.com/solresol/sudoku-padic-regression/pull/25) | [`cec4bd1`](https://github.com/solresol/sudoku-padic-regression/commit/cec4bd13d19a493832794ea6f1d6a989028036fa) |
| C25 | [#26](https://github.com/solresol/sudoku-padic-regression/pull/26) | [`a1d7532`](https://github.com/solresol/sudoku-padic-regression/commit/a1d753266a56f20f9871fa2af9452bf767fac145) |
| C26 | [#27](https://github.com/solresol/sudoku-padic-regression/pull/27) | [`da07bf9`](https://github.com/solresol/sudoku-padic-regression/commit/da07bf95cec47c14dabdc7b855d9d16b2ef4649a) |
| C27 | [#28](https://github.com/solresol/sudoku-padic-regression/pull/28) | [`f64e354`](https://github.com/solresol/sudoku-padic-regression/commit/f64e35423be2ec0cd791ea438b93f41543a61e99) |
| C28 | [#29](https://github.com/solresol/sudoku-padic-regression/pull/29) | [`809d5ef`](https://github.com/solresol/sudoku-padic-regression/commit/809d5ef2e8d8ce54384d420f3fc67ae1b1834bd1) |
| C29 | [#30](https://github.com/solresol/sudoku-padic-regression/pull/30) | [`7e40b00`](https://github.com/solresol/sudoku-padic-regression/commit/7e40b00f4462df8469514d137387b3e284b59789) |
| C30 | [#31](https://github.com/solresol/sudoku-padic-regression/pull/31) | [`b1182c5`](https://github.com/solresol/sudoku-padic-regression/commit/b1182c5980873a91072decd4d0d930537213042c) |
| C31 | [#32](https://github.com/solresol/sudoku-padic-regression/pull/32) | [`21c682a`](https://github.com/solresol/sudoku-padic-regression/commit/21c682aff946344badeaa26e8c2f1e02f4d51c70) |
