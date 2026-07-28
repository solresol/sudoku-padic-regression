# Manuscript corrections — 28 July 2026

## Metadata

- Completed at: `2026-07-28T17:47:04+10:00`
- Annotated PDF: `/Users/gregb/Downloads/sudoku_padic_regression-corrections-28july2026.pdf`
- Annotated PDF SHA-256: `57ac78f3dccdb2b342aca3d1ffbdef98cdc2a8a96d3f8194b403aa7cf50fd5e0`
- Annotated PDF pages: `32`
- Annotated PDF file size: `1207024 bytes`
- Base PDF SHA-256: `6f49f1e677916eb59c01efab93b315ce629faa0e07153ffe0b1ebf75e2d49478`
- Final PDF SHA-256: `0458ca6a52199bd1f1fe5de8126c9d65f718e8a9cad3edbf0ee83df87216af6c`
- Final PDF pages: `29`
- Final PDF file size: `835623 bytes`
- Canonical source: `paper/sudoku_padic_regression.tex`
- Distribution copy: `site/sudoku_padic_regression.pdf`

## Review method

The annotated PDF was compared with the repository PDF page by page. Its annotation export contains a broken transparency-state reference (`FXE2`), which hides red ink on some pages in standard PDF renderers. A temporary review copy under `tmp/pdfs/corrections28jul2026/` aliases that state to the working red-ink state so the marks can be inspected; neither the annotated PDF nor the temporary copy is versioned.

Substantive annotation ink occurs on original PDF pages 4–6, 13, 15–17, 19, 24, 26–31. Apparent machine-extracted changes on clean pages 1, 11, and 22 were rejected after direct image comparison. Every original page and every rebuilt page was visually reviewed.

## Correction ledger

| Original PDF page(s) | Correction | Result |
|---:|---|---|
| 4–5 | Move the compact observation table and accompanying explanation earlier; reflow the worked example; add a page reference to the general theorem. | Figure 1 and Table 1 now precede the worked-example subsection on page 3. Its compact-tuple lead-in remains with the example, Figure 2 and Table 2 follow at the top of page 4, and the theorem reference names page 6. |
| 5 | Add the explanation of the two Boolean wells and their domination of clause rewards. | Applied before the CNF dataframe. |
| 6 | Remove the marked false-label derivation and Davenport-width sentence; link “Boolean-CSP page” to the requested URL. | Applied. |
| 13 | Delete the Potts-energies subsection. | Applied; dependent headings and roadmap text were reconciled. |
| 15 | Replace “this encoding” with “that encoding”. | Applied. |
| 16 | Remove the marked Mihara qualifications and unsuccessful diagnostic trace; state why the solver is excluded from the 36 runs; label the solver URL. | Applied. |
| 17 | Make the Mihara digitwise-regression material an appendix. | Moved to Appendix A; later appendices renumber automatically. |
| 19 | Tighten the conclusion, replace the figure reference with the companion-site URL, and remove the marked Mihara-view sentence. | Applied. |
| 24 | Add the JavaScript implementation, provide the GitHub URL, and remove italics from “carving”. | Applied. The circled word “not” was retained because deleting it would incorrectly claim that random carving enforces uniqueness. |
| 26 | Delete the comparison paragraph after the powers-of-two results and qualify the forbidden-target summation with membership in the clause. | Applied. |
| 27–28 | Remove the Davenport-width claim; replace the (R2) transition with “causes obvious problems”; delete the syndrome, value-label, sign-choice, and limits material. | Applied. |
| 29 | Change “matters” to “does matter”. | Applied. |
| 30 | Delete the marked Davenport-constant and falsifying-assignment detail and the marked provenance phrase. | Applied. |
| 31 | Delete the final sentence under the interactive demonstration. | Applied. |

## Annotation decisions

- The isolated handwritten `at` above “solvers: a greedy row-swap search” was not inserted. All literal placements produce an ungrammatical sentence, and there is no accompanying deletion or replacement mark.
- The circled `not` in “does not enforce uniqueness” was treated as a review mark rather than a deletion because removing it reverses the documented procedure’s meaning.
- The page-placement request was implemented as readable reflow: Figure 1 and Table 1 precede the worked-example subsection on page 3, while Figure 2 and Table 2 follow together on page 4.

## Verification

- LaTeX rebuild completed: `29` A4 pages.
- No undefined references, undefined citations, multiply defined labels, or overfull boxes.
- Rebuilt pages visually reviewed: `29 / 29`.
- Python tests under the repository’s Python 3.12 environment: `15 / 15 passed`.
- `git diff --check`: passed.
- `paper/sudoku_padic_regression.pdf` and `site/sudoku_padic_regression.pdf`: byte-identical.
