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

  it("runs the Mihara equality fit as a diagnostic rather than claiming Sudoku success", () => {
    const solver = new SudokuSolver(parsePuzzle(EXAMPLE_PUZZLE), {
      ...DEFAULT_SOLVER_OPTIONS,
      method: "mihara",
      maxSteps: 1,
      restarts: 1
    });

    const snap = solver.advance();

    expect(snap.done).toBe(true);
    expect(snap.solved).toBe(false);
    expect(snap.miharaTotal).toBeGreaterThan(1_000);
    expect(snap.miharaCoefficients === null || snap.miharaInliers !== null).toBe(true);
    expect((snap.miharaSuccessfulTrials ?? 0) + (snap.miharaSingularTrials ?? 0)).toBe(1);
  });
});
