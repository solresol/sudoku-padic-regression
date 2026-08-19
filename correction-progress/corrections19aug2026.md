# Manuscript corrections — 19 August 2026

## Metadata

- Completed at: `2026-08-19T18:28:44+10:00`
- Annotated PDF: `/Users/gregb/Downloads/Signed p-adic Residual Encodings - Corrections 2026-08-190.pdf`
- Annotated PDF SHA-256: `e513f4f963074e7edd8f66e1c46c61678c47506e8baae523db120a31810023b6`
- Annotated PDF pages: `31`
- Annotated PDF file size: `1058562 bytes`
- Base PDF: `paper/sudoku_padic_regression.pdf` before this pass
- Base PDF SHA-256: `8fb4f18118323b948998bdabdc1713f5cef1ff14071523569aaa557bf0834717`
- Base PDF pages: `31`
- Base PDF file size: `881678 bytes`
- Preserved base copy: `/Users/gregb/Documents/correction-bases/sudoku_padic_regression.pdf`
- Canonical source: `paper/sudoku_padic_regression.tex`
- Final source SHA-256: `fe930668d6999ae36fe47daa4264f968ef607ec7698c8814e0fc5ac6ceba455a`
- Final PDF SHA-256: `e5340ed4e256e59eb8e2b7909ee39e0613e481e1abdfc98fd8cd715ce2c44cea`
- Final PDF pages: `31`
- Final PDF file size: `880153 bytes`
- Distribution copy: `site/sudoku_padic_regression.pdf`
- Rendered work directory: `/Users/gregb/Documents/devel/sudoku-padic-regression/tmp/pdfs/corrections19aug2026`

## Review method

The annotated PDF and base PDF have identical extracted text, confirming that the corrections are annotation overlays rather than revised manuscript text. The export refers to a missing transparency state (`FXE1`) on some pages. An untracked repair copy aliases that state to the supplied `FXE2` state so all red ink can be rendered; the downloaded original is unchanged.

The annotated and base page content streams and page renders were compared. Added ink occurs on pages 2–4, 15–16, 18, 20–21, and 27–28. All 31 annotated pages and all 31 rebuilt pages were visually reviewed. No external OCR service was used.

## Correction ledger

| PDF page(s) | Correction | Source location | Result |
|---:|---|---|---|
| 2 | Remove the contrast with “fitted slopes from observed data” from the coefficient-vector terminology entry. | terminology table | Applied. |
| 3 | Replace roadmap verbs as marked, remove the doctoral-thesis aside, describe Section 9 as related work, and split the two preliminary consequences into paragraphs. | introduction roadmap; preliminaries | Applied. |
| 4 | Shorten the novelty sentence and add the requested page reference for the compiler theorem. | preliminaries; Archimedean comparison | Applied. |
| 15 | Replace “share a unit” with “share an exclusion constraint.” | Sudoku minimal objective | Applied with singular agreement. |
| 16 | Repair the peer-grid line geometry so all cell boundaries render evenly and none disappear. | Figure 3 TikZ source | Applied by drawing every cell boundary explicitly, then overlaying the heavier 3-by-3 boundaries. |
| 18 | Shorten “Related work and scope” to “Related work” and remove the redundant subsection heading. | Section 9 heading | Applied. |
| 20–21 | Remove the marked browser-implementation-detail paragraph and replace “companion browser” with “companion website.” | Appendix A | Applied. |
| 27 | Clarify that the experiments used the 50 puzzles from Project Euler Problem 96; tighten the topological and plateau descriptions. | Appendix D | Applied. |
| 28 | Replace “inversion” with “problem” twice and remove the two marked lead-in phrases. | Appendix D | Applied. |

## Page ledger

| Page | Annotation status | Source location / change | Branch / PR | Verification |
|---:|---|---|---|---|
| 001 | Reviewed; no added ink | — | Not created | Final page visually reviewed. |
| 002 | Annotated; resolved | Terminology table | Not created | Text and final page checked. |
| 003 | Annotated; resolved | Introduction; preliminaries | Not created | Text, paragraph breaks, and final page checked. |
| 004 | Annotated; resolved | Preliminaries | Not created | Page reference resolved; final page checked. |
| 005 | Reviewed; no added ink | — | Not created | Final page visually reviewed. |
| 006 | Reviewed; no added ink | — | Not created | Final page visually reviewed. |
| 007 | Reviewed; no added ink | — | Not created | Final page visually reviewed. |
| 008 | Reviewed; no added ink | — | Not created | Final page visually reviewed. |
| 009 | Reviewed; no added ink | — | Not created | Final page visually reviewed. |
| 010 | Reviewed; no added ink | — | Not created | Final page visually reviewed. |
| 011 | Reviewed; no added ink | — | Not created | Final page visually reviewed. |
| 012 | Reviewed; no added ink | — | Not created | Final page visually reviewed. |
| 013 | Reviewed; no added ink | — | Not created | Final page visually reviewed. |
| 014 | Reviewed; no added ink | — | Not created | Final page visually reviewed. |
| 015 | Annotated; resolved | Sudoku minimal objective | Not created | Text and final page checked. |
| 016 | Annotated; resolved | Figure 3 | Not created | Close visual review confirms complete, even grid lines. |
| 017 | Reviewed; no added ink | — | Not created | Final page visually reviewed. |
| 018 | Annotated; resolved | Related-work heading | Not created | Heading hierarchy and final page checked. |
| 019 | Reviewed; no added ink | — | Not created | Final page visually reviewed. |
| 020 | Annotated; resolved | Appendix A paragraph deletion | Not created | Reflow and final page checked. |
| 021 | Annotated; resolved | Appendix A wording | Not created | Reflow and final page checked. |
| 022 | Reviewed; no added ink | — | Not created | Final page visually reviewed. |
| 023 | Reviewed; no added ink | — | Not created | Final page visually reviewed. |
| 024 | Reviewed; no added ink | — | Not created | Final page visually reviewed. |
| 025 | Reviewed; no added ink | — | Not created | Final page visually reviewed. |
| 026 | Reviewed; no added ink | — | Not created | Final page visually reviewed. |
| 027 | Annotated; resolved | Appendix D experiment framing | Not created | Text, table adjacency, and final page checked. |
| 028 | Annotated; resolved | Appendix D discussion | Not created | Text and final page checked. |
| 029 | Reviewed; no added ink | — | Not created | Final page visually reviewed. |
| 030 | Reviewed; no added ink | — | Not created | Final page visually reviewed. |
| 031 | Reviewed; no added ink | — | Not created | Final page visually reviewed. |

## Verification

- Annotation pages resolved: `10 / 10`.
- Annotated PDF pages reviewed: `31 / 31`.
- Rebuilt PDF pages visually reviewed: `31 / 31`.
- LaTeX rebuild: passed (`31` A4 pages).
- Undefined references, undefined citations, multiply defined labels, and overfull boxes: none.
- Python tests: `15 / 15 passed`.
- `git diff --check`: passed.
- `paper/sudoku_padic_regression.pdf` and `site/sudoku_padic_regression.pdf`: byte-identical.

## Repository status

- Working branch: `main`.
- Commit: not created.
- Push: not performed.
- Pull request: not created.
