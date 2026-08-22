# Manuscript corrections — 22 August 2026

## Metadata

- Completed at: `2026-08-22T19:05:06+10:00`
- Annotated PDF: `/Users/gregb/Downloads/sudoku_padic_regression-corrections-22aug2026.pdf`
- Annotated PDF SHA-256: `99a74c8c61f76cbaf46bad80204727ff3b2a60597e009541798ee37221a79aa4`
- Annotated PDF pages: `31`
- Annotated PDF file size: `1036318 bytes`
- Base PDF: `paper/sudoku_padic_regression.pdf` before this pass
- Base PDF SHA-256: `b1f5d0cad7695b99614f5795d89bc3b8d136ad33965784647ba8ef3222423747`
- Base PDF pages: `31`
- Base PDF file size: `882372 bytes`
- Preserved base copy: `/Users/gregb/Documents/correction-bases/sudoku_padic_regression-base-22aug2026.pdf`
- Canonical source: `paper/sudoku_padic_regression.tex`
- Final source SHA-256: `45c89efaa2d6301dc2b618068303a37ada93e5a626172f08d7f075a74be2295c`
- Final PDF SHA-256: `32be41330892fed73b28580de68d77ca5a5e6c8965d3b501ec6d84355874f26c`
- Final PDF pages: `31`
- Final PDF file size: `881804 bytes`
- Distribution copy: `site/sudoku_padic_regression.pdf`
- Rendered work directory: `/Users/gregb/Documents/devel/sudoku-padic-regression/tmp/pdfs/corrections22aug2026`

## Review method

The annotated PDF and base PDF have identical extracted text, confirming that the corrections are annotation overlays rather than revised manuscript text. The export refers to a missing transparency state (`FXE2`) on some marked content. An untracked review copy aliases that state to the supplied `FXE1` state so all red ink can be rendered; the downloaded original is unchanged.

Page-content comparison identified added ink on pages 4, 5, 7, 13, 14, 17, 18, 19, and 28. The annotations were extracted into a structured ledger with `gemini-3.1-pro-preview`, independently cross-checked on the dense pages with `gemini-3-flash-preview`, and then checked directly against the page images and source anchors. The two grammatically or structurally ambiguous clusters on pages 5 and 13–14 were applied only after author confirmation.

## Correction ledger

| PDF page(s) | Correction | Source location | Result |
|---:|---|---|---|
| 4 | Replace the early Archimedean comparison with “we can't do this” and delete the deferred-comparison sentence. | preliminaries | Applied. |
| 5 | Remove “displayed,” change “while” to “and that,” remove “below,” merge the positive/negative row description, make Table 2 and Figure 2 the subject of “show,” and remove “exactly” from the minimum-loss sentence. | worked example | Applied; the dense sentence restructuring was author-confirmed. |
| 7 | Remove “rather than correctness” and “displayed” from the false-label discussion. | general false labels | Applied. |
| 13 | Describe the signed construction as “more compact” and recast the Archimedean comparison as an attempted transplant “by scoring” the same data. | positive-complement comparison; Archimedean comparison | Applied. |
| 13–14 | Replace the large-weight Boolean-well claim with the requested statement that increasing `lambda` keeps the fitted value closer to one-half; remove the marked complexity comparison; begin the local-flatness paragraph “Unlike R”. | Archimedean comparison | Applied after author confirmation. |
| 14 | Remove the “Minimum conflicts” wrapper, promote “Unsatisfiable instances” to Section 5, and begin “Consider what happens”. | minimum-conflict section | Applied after author confirmation; the existing section label was preserved. |
| 17 | Shorten the design-choice heading to “clues as data.” | Sudoku section | Applied. |
| 18–19 | Add the Figure 4 screenshot cross-reference, identify the figure as a screenshot of the companion site, and remove the redundant caption ending. | computations; Figure 4 | Applied. |
| 19 | Remove “exactly” from the conclusion's minimum-conflict sentence. | conclusion | Applied. |
| 28 | Delete the sentence declaring the failed local-search result “topological.” | Appendix D | Applied. |

## Page ledger

| Page(s) | Annotation status | Verification |
|---:|---|---|
| 4, 5, 7, 13, 14, 17, 18, 19, 28 | Annotated; resolved | Each marked page was inspected at full-page resolution after source mapping; the corresponding rebuilt pages were inspected again after compilation. |
| 1–3, 6, 8–12, 15–16, 20–27, 29–31 | No added annotation content | Page-content comparison found no added ink; the rebuilt pages were included in the full-document visual review. |

## Verification

- Annotation items resolved: `28 / 28`.
- Annotated pages visually reviewed: `9 / 9`.
- Rebuilt PDF pages visually reviewed: `31 / 31`.
- LaTeX rebuild: passed (`31` A4 pages).
- Undefined references, undefined citations, multiply defined labels, destination warnings, and overfull boxes: none.
- Python tests: `15 / 15 passed`.
- `git diff --check`: passed.
- Source manifest: all entries passed SHA-256 verification.
- `paper/sudoku_padic_regression.pdf` and `site/sudoku_padic_regression.pdf`: byte-identical.
