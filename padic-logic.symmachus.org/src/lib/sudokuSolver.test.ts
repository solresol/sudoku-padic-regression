import { describe, expect, it } from "vitest";
import { isValidComplete, parsePuzzle, respectsClues } from "./sudoku";
import { DEFAULT_SOLVER_OPTIONS, SudokuSolver } from "./sudokuSolver";

const EXAMPLE_PUZZLE = [
  "53..7....",
  "6..195...",
  ".98....6.",
  "8...6...3",
  "4..8.3..1",
  "7...2...6",
  ".6....28.",
  "...419..5",
  "....8..79"
].join("");

const SOLVED = [
  "534678912",
  "672195348",
  "198342567",
  "859761423",
  "426853791",
  "713924856",
  "961537284",
  "287419635",
  "345286179"
].join("");

describe("SudokuSolver", () => {
  it("solves the standard example puzzle with stepwise swaps", () => {
    const solver = new SudokuSolver(parsePuzzle(EXAMPLE_PUZZLE), {
      ...DEFAULT_SOLVER_OPTIONS,
      method: "stepwise",
      seed: 1
    });
    const snap = solver.runToCompletion();
    expect(snap.solved).toBe(true);
    expect(isValidComplete(snap.grid)).toBe(true);
    expect(respectsClues(snap.grid, parsePuzzle(EXAMPLE_PUZZLE))).toBe(true);
    expect(snap.conflicts).toBe(0);
  });

  it("solves the standard example puzzle with the Zubarev walk", () => {
    const solver = new SudokuSolver(parsePuzzle(EXAMPLE_PUZZLE), {
      ...DEFAULT_SOLVER_OPTIONS,
      method: "zubarev",
      seed: 1
    });
    const snap = solver.runToCompletion();
    expect(snap.solved).toBe(true);
    expect(isValidComplete(snap.grid)).toBe(true);
    expect(respectsClues(snap.grid, parsePuzzle(EXAMPLE_PUZZLE))).toBe(true);
  });

  it("keeps clue cells fixed throughout the search", () => {
    const puzzle = parsePuzzle(EXAMPLE_PUZZLE);
    const solver = new SudokuSolver(puzzle, { ...DEFAULT_SOLVER_OPTIONS, seed: 3 });
    for (let i = 0; i < 500; i += 1) {
      const snap = solver.advance();
      for (let cell = 0; cell < 81; cell += 1) {
        if (puzzle[cell] !== 0) {
          expect(snap.grid[cell]).toBe(puzzle[cell]);
        }
      }
      if (snap.solved || snap.done) break;
    }
  });

  it("recognises an already-solved grid immediately", () => {
    const solver = new SudokuSolver(parsePuzzle(SOLVED), DEFAULT_SOLVER_OPTIONS);
    const snap = solver.snapshot();
    expect(snap.solved).toBe(true);
    expect(snap.conflicts).toBe(0);
  });

  it("runs Mihara on the weighted positive-complement Sudoku dataframe", () => {
    const solver = new SudokuSolver(parsePuzzle(EXAMPLE_PUZZLE), {
      ...DEFAULT_SOLVER_OPTIONS,
      method: "mihara",
      maxSteps: 1,
      restarts: 1
    });

    const snap = solver.advance();

    expect(snap.done).toBe(true);
    expect(snap.solved).toBe(false);
    expect(snap.miharaPrime).toBe(19);
    expect(snap.miharaObservationCount).toBe(13_449);
    expect(snap.miharaTotal).toBe(23_229);
    expect(snap.miharaFloor).toBe(20_718);
    expect(snap.miharaCoefficients).toHaveLength(81);
    expect(snap.miharaInliers).not.toBeNull();
    expect(snap.miharaLoss).toBe((snap.miharaTotal ?? 0) - (snap.miharaInliers ?? 0));
    expect(snap.miharaLoss).toBeGreaterThanOrEqual(snap.miharaFloor ?? 0);
    expect(snap.miharaSignedLoss).not.toBeNull();
    expect(
      (snap.miharaNegativeUnitValuations ?? 0) +
      (snap.miharaNegativeInfiniteValuations ?? 0)
    ).toBe(810);
    expect(snap.miharaSuccessfulTrials).toBe(1);
    expect(snap.miharaSingularTrials).toBe(0);
  });

  it("retries Mihara with alternate full-rank starts and retains only a non-increasing loss", () => {
    const solver = new SudokuSolver(parsePuzzle(EXAMPLE_PUZZLE), {
      ...DEFAULT_SOLVER_OPTIONS,
      method: "mihara",
      seed: 0,
      maxSteps: 8,
      restarts: 1
    });

    const losses: number[] = [];
    for (let index = 0; index < 8; index += 1) {
      const snapshot = solver.advance();
      if (snapshot.miharaLoss != null) losses.push(snapshot.miharaLoss);
    }
    const final = solver.snapshot();

    expect(final.totalSteps).toBe(8);
    expect(final.miharaSuccessfulTrials).toBe(8);
    expect(final.miharaSingularTrials).toBe(0);
    expect(final.done).toBe(true);
    expect(losses).toHaveLength(8);
    expect(losses.every((loss, index) => index === 0 || loss <= losses[index - 1])).toBe(true);
  });
});
