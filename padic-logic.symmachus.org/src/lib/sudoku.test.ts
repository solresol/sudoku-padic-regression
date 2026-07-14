import { describe, expect, it } from "vitest";
import {
  ALPHA_DEFAULT,
  MAX_DEGREE,
  PEER_COUNT,
  PEERS,
  UNITS,
  buildPuzzleModel,
  buildSudokuRegressionDataFrame,
  conflictsAllUnits,
  conflictsColsBoxes,
  dedupedPeerConflicts,
  digitSnappingPenalty,
  evaluateObjective,
  isValidComplete,
  pAdicNorm,
  parsePuzzle,
  respectsClues
} from "./sudoku";

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

const EXAMPLE_SOLUTION = [
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

describe("sudoku graph structure", () => {
  it("has 27 units, 810 deduped peer pairs, degree 20", () => {
    expect(UNITS).toHaveLength(27);
    expect(PEER_COUNT).toBe(810);
    expect(MAX_DEGREE).toBe(20);
    expect(PEERS.every((peers) => peers.length === 20)).toBe(true);
  });
});

describe("genuine p-adic norms (p = 11)", () => {
  it("is 0 on equal digits and 1 on unequal digits", () => {
    expect(pAdicNorm(0, 11)).toBe(0);
    for (let a = 1; a <= 9; a += 1) {
      for (let b = 1; b <= 9; b += 1) {
        expect(pAdicNorm(a - b, 11)).toBe(a === b ? 0 : 1);
      }
    }
  });

  it("drops below 1 only on multiples of p", () => {
    expect(pAdicNorm(11, 11)).toBeCloseTo(1 / 11); // 11 = 11^1
    expect(pAdicNorm(22, 11)).toBeCloseTo(1 / 11); // 22 = 2 * 11
    expect(pAdicNorm(121, 11)).toBeCloseTo(1 / 121); // 121 = 11^2
    expect(pAdicNorm(5, 11)).toBe(1);
  });

  it("digit-snapping penalty is minimised (value 8) exactly on digits", () => {
    for (let t = 1; t <= 9; t += 1) {
      expect(digitSnappingPenalty(t, 11)).toBe(8);
    }
    expect(digitSnappingPenalty(0, 11)).toBe(9); // off-domain: no zero term
    expect(digitSnappingPenalty(12, 11)).toBeCloseTo(8 + 1 / 11); // 12 = 1 mod 11
  });
});

describe("conflict counting", () => {
  it("a valid completion has zero conflicts of every kind", () => {
    const grid = parsePuzzle(EXAMPLE_SOLUTION);
    expect(isValidComplete(grid)).toBe(true);
    expect(respectsClues(grid, parsePuzzle(EXAMPLE_PUZZLE))).toBe(true);
    expect(conflictsAllUnits(grid)).toBe(0);
    expect(conflictsColsBoxes(grid)).toBe(0);
    expect(dedupedPeerConflicts(grid)).toBe(0);
  });

  it("an all-ones grid maximises conflicts", () => {
    const grid = new Array(81).fill(1);
    expect(dedupedPeerConflicts(grid)).toBe(810); // every peer pair equal
    expect(conflictsColsBoxes(grid)).toBe(648); // 9*C(9,2) cols + 9*C(9,2) boxes
    expect(conflictsAllUnits(grid)).toBe(972); // + rows
    expect(isValidComplete(grid)).toBe(false);
  });
});

describe("signed p-adic residual objective", () => {
  it("summarises the construction for the 30-clue example puzzle", () => {
    const model = buildPuzzleModel(parsePuzzle(EXAMPLE_PUZZLE));
    expect(model.clueCount).toBe(30);
    expect(model.negativeObservations).toBe(810);
    expect(model.positiveObservationsMinimal).toBe(729 - 8 * 30);
    expect(model.unaryConstant).toBe(8 * (81 - 30));
    expect(model.theoreticalFloor).toBe(ALPHA_DEFAULT * 408 - 810); // 7758
  });

  it("builds the sparse p-adic regression dataframe defined by the puzzle", () => {
    const model = buildPuzzleModel(parsePuzzle(EXAMPLE_PUZZLE));
    const dataframe = buildSudokuRegressionDataFrame(model);

    expect(dataframe.variables).toHaveLength(81);
    expect(dataframe.variables.slice(0, 3).map((variable) => variable.name)).toEqual([
      "x_r1c1",
      "x_r1c2",
      "x_r1c3"
    ]);
    expect(dataframe.pinningRowCount).toBe(489);
    expect(dataframe.peerRewardRowCount).toBe(810);
    expect(dataframe.rows).toHaveLength(1_299);
    expect(dataframe.rows[0]).toMatchObject({
      kind: "pinning",
      label: "P1",
      coefficients: { x_r1c1: 1 },
      relation: "=",
      target: 5,
      sign: 1,
      weight: ALPHA_DEFAULT
    });
    expect(dataframe.rows.slice(2, 11).map((row) => row.target)).toEqual([
      1, 2, 3, 4, 5, 6, 7, 8, 9
    ]);
    expect(dataframe.rows[489]).toMatchObject({
      kind: "peer-reward",
      label: "E1",
      coefficients: { x_r1c1: 1, x_r1c2: -1 },
      relation: "≠",
      target: 0,
      sign: -1,
      weight: 1
    });
  });

  it("hits the theoretical floor exactly on a valid completion", () => {
    const model = buildPuzzleModel(parsePuzzle(EXAMPLE_PUZZLE));
    const result = evaluateObjective(parsePuzzle(EXAMPLE_SOLUTION), model);
    expect(result.domainRespecting).toBe(true);
    expect(result.edgeRewardTotal).toBe(810); // all 810 edges unequal
    expect(result.loss).toBe(model.theoreticalFloor);
    expect(result.dedupedConflicts).toBe(0);
  });

  it("loss - floor equals the deduped conflict count on domain-respecting states", () => {
    const model = buildPuzzleModel(new Array(81).fill(0)); // no clues
    const grid = new Array(81).fill(1);
    const result = evaluateObjective(grid, model);
    expect(result.domainRespecting).toBe(true);
    expect(result.edgeRewardTotal).toBe(0);
    expect(result.dedupedConflicts).toBe(dedupedPeerConflicts(grid));
    expect(result.dedupedConflicts).toBe(810);
  });
});
