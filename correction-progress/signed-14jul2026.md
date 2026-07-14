# Signed 14Jul2026 correction pass

## Metadata

- Annotated PDF: `/Users/gregb/Downloads/signed-p-adic-residual-encodings-corrections-14jul2026.pdf`
- Annotated PDF SHA-256: `f5631717376ff79b4e1e4e20484e11ae255643a2d1dcbd04ba92e115fc524eaa`
- Annotated PDF pages: `28`
- Base PDF: `/Users/gregb/Documents/devel/sudoku-padic-regression/paper/sudoku_padic_regression.pdf`
- Base PDF SHA-256: `1c08b874088ec574d6df3b15f28bf4bb7107ebc34efa50af09f5e67b074d19fe`
- Base PDF pages: `28`
- Preserved base: `/Users/gregb/.codex/document-correction-bases/sudoku-padic-regression/signed-14jul2026/sudoku_padic_regression.pdf`
- Rendered annotation work: `/Users/gregb/Documents/devel/sudoku-padic-regression/scratch/signed-14jul2026`
- Implementation: batched into the current working tree at the user's request; no branch, commit, push, or PR was requested.
- Detection note: the automatic blue-ink scan did not reliably identify the handwritten ink in this signed PDF. Full contact sheets and the candidate pages from image differencing were therefore inspected visually.

## Page ledger

### Page 001

- Status: Applied.
- Source: `paper/sudoku_padic_regression.tex`, abstract and opening section.
- Changes: made the signed weights explicit; stated the mechanical conversion from all-different/CNF inputs to regression data; removed the struck-out abstract material; rewrote the opening to begin from the finite graph and its data rather than from general mathematics.
- Verification: source compiles; revised page inspected in the final rendering.

### Page 002

- Status: Applied.
- Source: `paper/sudoku_padic_regression.tex`, opening sections and later `Related work and scope` section.
- Changes: moved related work behind the worked examples and formal construction; removed the struck negative-ridge tangent and shortened the surrounding comparison prose.
- Verification: source compiles; revised pages inspected in the final rendering.

### Page 003

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 004

- Status: Applied.
- Source: `paper/sudoku_padic_regression.tex`, `From an all-different graph to a regression dataframe`.
- Changes: added a dataframe version of the synthetic observations; removed the struck explanatory paragraph; added the finite three-point-support search and its decoded list-colouring result.
- Verification: source compiles; dataframe and candidate table inspected in the final rendering.

### Page 005

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 006

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 007

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 008

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 009

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 010

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 011

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 012

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 013

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 014

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 015

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 016

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 017

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 018

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 019

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 020

- Status: No handwritten corrections.
- Verification: inspected at full resolution; an image-difference candidate was caused by rendering variation rather than ink.

### Page 021

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 022

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 023

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 024

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 025

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 026

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 027

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

### Page 028

- Status: No handwritten corrections.
- Verification: annotated page inspected visually.

## Structural changes requested with this pass

- Removed the constraint-locator, repair-probe, and finite-field-fingerprint material from the paper and interactive site.
- Added the all-different graph, tuple view, dataframe view, exhaustive candidate-hyperplane calculation, and decoded answer before the formal theory.
- Added an analogous two-variable CNF dataframe and exhaustive regression calculation.
- Retained polynomial-valued false labels only as a short observation and possible future direction, with no claimed algorithmic benefit.
- Moved the former general construction to follow those worked examples and moved related work later.

## Final verification

- `make paper` completed successfully and produced a 26-page PDF with no overfull boxes, undefined citations, or undefined references.
- `make site` copied the rebuilt manuscript to the static site tree.
- The browser app test suite passed: 8 test files and 64 tests.
- The production TypeScript/Vite build completed successfully.
- `git diff --check` passed.
- All 26 revised pages were rendered and inspected; the first five pages were also checked at full resolution to verify figure/table order, dataframe legibility, and the transition into the formal theorem.
- The locator, repair-probe, fingerprint, and discarded negative-ridge material has no remaining reference in the manuscript, app source, repository README, or source-package manifest.
