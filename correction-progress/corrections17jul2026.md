# Signed manuscript corrections — 17 July 2026

## Metadata

- Annotated PDF: `/Users/gregb/Downloads/sudoku_padic_regression-corrections17jul2026.pdf`
- Annotated PDF SHA-256: `10cf28d4986e4d5c190e9ac73f3be1e450d2d4cf48c442592964357f7538f1ed`
- Annotated PDF pages: `28`
- Annotated PDF file size: `1098292 bytes`
- Base PDF SHA-256: `9d1c39e1cc68fdc21c589beb50e1a537e7785f34b750365eaa4cd12556b3c7c3`
- Base PDF pages: `28`
- Base PDF file size: `821858 bytes`
- Final PDF SHA-256: `52a6176051c66bd0e24e8e9405c77d5003e21167f21ae04ea063218a538ec026`
- Final PDF pages: `28`
- Final PDF file size: `819483 bytes`
- Render workspace: `/Users/gregb/Documents/devel/sudoku-padic-regression/tmp/pdfs/corrections17jul2026/`
- Source file: `paper/sudoku_padic_regression.tex`

## Review method

The annotated and base PDFs have identical extracted text. Both were rendered and compared page by page, including image differencing to distinguish added red ink from exporter and glyph-rendering changes. Every page was visually inspected. Substantive annotations occur on pages 1–6 and 17; small differences on pages 8, 10, 14, 16, 20, and 25 contain no added correction ink.

The handwriting was independently transcribed with `gemini-3-flash-preview` and checked against the rendered pages and LaTeX source. The retired `gemini-3-pro-preview` alias returned a 404, so it was not used for the completed extraction. The bracketed instruction on page 5 was resolved by taking its explicit option to reproduce Table 3 with polynomial targets.

## Correction ledger

| ID | PDF page | Intended correction | Source area | Result |
|---|---:|---|---|---|
| C01 | 1 | Remove the threshold-hardness and complement-expansion detail from the abstract. | abstract | applied |
| C02 | 1 | Remove solver-benchmark and Mihara-diagnostic detail from the abstract. | abstract | applied |
| C03 | 1 | Replace the expressivity lead-in with “This makes p-adic linear regression surprisingly expressive.” | introduction | applied |
| C04 | 2 | Replace the long caveat discussion with a short NP-hardness paragraph. | introduction | applied |
| C05 | 2 | Generalise the Sudoku contribution bullet to a variety of p-adic linear-regression techniques. | contributions | applied |
| C06 | 3 | Identify ordinary weighted least squares as Euclidean. | regression background | applied |
| C07 | 3 | State directly that the Euclidean boundedness issue does not occur for the p-adic objective. | regression background | applied |
| C08 | 3 | Preserve the boxed finite-domain paragraph in sequence with the preceding discussion. | regression background | reviewed; no textual change required |
| C09 | 4 | Put Figure 1, Table 1, and Figure 2 on a page of their own, with Figure 2 directly after Table 1. | worked example layout | applied |
| C10 | 4 | Add the requested cross-reference from the domain-respecting candidates sentence. | worked example | applied |
| C11 | 5 | Replace the product construction with a version of Table 3 using distinct polynomial targets. | polynomial-valued false labels | applied |
| C12 | 6 | Warn that naive false-label choices for more than two variables can collide, using the marked four-variable example. | polynomial-valued false labels | applied |
| C13 | 6 | Remove the claim that factorisation recovers false-clause labels. | polynomial-valued false labels | applied |
| C14 | 6 | Remove the logarithmic-objective equivalence paragraph. | polynomial-valued false labels | applied |
| C15 | 6 | Remove the claim that degree exactly counts false clauses and retains their identities. | polynomial-valued false labels | applied |
| C16 | 6 | State that no useful application of the construction has been found, including over finite fields. | polynomial-valued false labels | applied |
| C17 | 6 | Change “record” to “include” in the future-direction sentence. | polynomial-valued false labels | applied |
| C18 | 17 | Move Mihara’s digitwise-regression subsection from Section 8 into Section 7. | related work and scope | applied |

## Dependent consistency edits

- Removed the degree-metric NP-hardness corollary because its proof depended on the deleted product construction.
- Removed the now-unused De Loera bibliography entry.
- Updated the roadmap and conclusion to match the shortened abstract, polynomial-target construction, and new Mihara subsection location.
- Kept the Mihara comparison table with its subsection so it cannot float above the Section 7.1 heading.

## Page ledger

| Page(s) | Review status | Correction IDs / notes |
|---:|---|---|
| 001 | annotated; fully mapped | C01–C03 |
| 002 | annotated; fully mapped | C04–C05 |
| 003 | annotated; fully mapped | C06–C08 |
| 004 | annotated; fully mapped | C09–C10 |
| 005 | annotated; fully mapped | C11 |
| 006 | annotated; fully mapped | C12–C17 |
| 007–016 | reviewed; no annotations | exporter/glyph differences on 8, 10, 14, and 16 contain no added ink |
| 017 | annotated; fully mapped | C18 |
| 018–028 | reviewed; no annotations | exporter/glyph differences on 20 and a one-pixel difference on 25 contain no added ink |

## Verification

- Annotated PDF pages reviewed: **28 / 28**.
- Coherent correction units found: **18**.
- Correction units resolved: **18 / 18**.
- Unmapped annotations: **0**.
- Ambiguous or deferred corrections: **0**.
- `make -B paper`: passed; no undefined references, undefined citations, overfull boxes, or fatal LaTeX errors.
- Final rendered pages visually reviewed: **28 / 28**; no clipping or overlap found.
- `paper/sudoku_padic_regression.pdf` and `site/sudoku_padic_regression.pdf`: byte-identical.
- Python tests: **15 / 15 passed**.
- Site tests: **82 / 82 passed** across **9 / 9** test files.
- Production site build: passed.
- `git diff --check`: passed.
