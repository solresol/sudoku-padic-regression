# Signed manuscript corrections — 15 July 2026

## Metadata

- Annotated PDF: `/Users/gregb/Downloads/signed-p-adic-residual-encodings-corrections-15jul2026.pdf`
- Annotated PDF SHA-256: `43b87e55623ca1cc204d75a3411388f743b91f005662a0ffd680dbc5e27b5e95`
- Annotated PDF pages: `26`
- Matching base PDF SHA-256: `a6a5d6192b5dfecc2c9ffcd9944a754ee3f4a41d9463a82addaf004c1e298d9e`
- Preserved base: `/Users/gregb/.codex/document-correction-bases/sudoku-padic-regression/signed-15jul2026/sudoku_padic_regression.pdf`
- Source edited: `paper/sudoku_padic_regression.tex`
- Render workspace: `scratch/signed-15jul2026/`
- Final PDF: `paper/sudoku_padic_regression.pdf`
- Final PDF SHA-256: `75d32996079eaa07ccf402d46b67d269df820ebeebe5cc7be71ba7ec72675261`
- Final PDF pages: `25`

## Review method

The blue-ink detector did not reliably identify the handwriting in this export. Page 23 was its only candidate because of a blue chart already present in the manuscript. I therefore compared the annotated PDF with its matching base, inspected every page at full-page resolution, and checked all 25 pages of the rebuilt PDF after applying the corrections.

No branch, commit, push, or pull request was created.

## Page ledger

### Page 001 — applied

- Removed the title footnote.
- Added Section 1, “p-adic linear regression”, with the requested background.
- Made the worked all-different material subsection 1.1, “An example”, and adjusted its opening sentence.
- Verified on rebuilt pages 1–2.

### Page 002 — applied

- Put the dataframe table before the compact tuple figure.
- Split “Signed weight” and “Role” into separate columns.
- Removed the struck discussion of the earlier basis theorem.
- Verified on rebuilt pages 2–3.

### Page 003 — applied

- Stated that the general theorem guarantees a global minimum among the finite candidates.
- Made the CNF example subsection 1.2 with the requested title.
- Removed the struck “mechanical construction” sentence in the example.
- Split the CNF table’s signed-weight and role columns.
- Verified on rebuilt pages 3–4.

### Page 004 — applied

- Made polynomial-valued false labels subsection 1.3.
- Added the preceding CNF example in polynomial form and showed that degree counts false clauses for unit-degree labels.
- Removed the struck comparison with global all-different propagation.
- Verified on rebuilt pages 4–5.

### Pages 005–008 — reviewed; no corrections marked

- Inspected the annotated and rebuilt pages.

### Page 009 — applied

- Created a separate “Minimum conflicts and Potts energies” section.
- Put unsatisfiable instances and Potts energies in their own subsections.
- Removed the struck lead-in sentence.
- Verified on rebuilt page 11.

### Pages 010–012 — reviewed; no corrections marked

- Inspected the annotated and rebuilt pages.

### Page 013 — applied

- Replaced the smoke-test description with the evidence-scoped claim that general p-adic linear-regression heuristics solved most sampled Sudoku instances in practice, while retaining the non-benchmark caveat.
- Verified on rebuilt page 14.

### Page 014 — applied

- Removed the struck opening compiler prose and the Boolean-CSP route material.
- Verified on rebuilt page 15.

### Page 015 — applied

- Removed the trust-boundary figure, natural-language/Prompt API discussion, detailed route prose, and struck post-figure summary.
- Retained the site URL and Sudoku screenshot, and kept the screenshot with the interactive section.
- Verified on rebuilt page 15.

### Page 016 — applied

- Folded the Mihara material into Related work.
- Added subsection 8.2 and subsubsections 8.2.1 and 8.2.2. The numbers shifted from the handwritten 9.2 labels because the earlier marked restructuring reduces the main-section count.
- Verified on rebuilt page 16.

### Page 017 — applied

- Replaced the struck transition with the concise statement that Mihara’s method does not apply to the CNF construction.
- Reduced the final comparison to the requested “not a drop-in replacement” conclusion.
- Verified on rebuilt page 17.

### Page 018 — applied

- Removed the struck solver disclaimer and compiler-theorem sentence.
- Tightened the final site paragraph to say that the site lets readers trace or compile the examples.
- Verified on rebuilt page 17.

### Pages 019–026 — reviewed; no corrections marked

- Inspected the annotated pages and the corresponding rebuilt pages 18–25.

## Verification

- `make paper` — passed; two LaTeX runs completed.
- Final output — 25 A4 pages, 810581 bytes.
- All final pages rendered and inspected; no clipping, overlap, missing figures, or broken layout found.
- No undefined citations or references and no overfull boxes. Remaining diagnostics are benign underfull-box notices in narrow table and bibliography cells.
- `make site` — passed; `paper/sudoku_padic_regression.pdf` and `site/sudoku_padic_regression.pdf` are byte-identical with the SHA-256 above.
