// Local-search solvers for the signed p-adic residual Sudoku objective.
//
// Ported from padic_sudoku_regression.py (the solvers used in the paper). Both
// work on the digit-valued, clue-consistent row-permutation state space and
// minimise the duplicated unit-scope column/box conflict count H_cb. Each row is
// kept as a permutation of 1..9, so a move is a within-row swap of two non-clue
// cells and only the two affected columns and boxes change.
//
// The solver is written as a resumable stepper so the UI can animate it a few
// frames per second, matching the paper's "updates a few times per second" story.

import {
  type Grid,
  BOXES,
  COLS,
  MIHARA_SUDOKU_PRIME,
  PEER_COUNT,
  ROWS,
  buildMiharaPositiveSudokuDataFrame,
  buildPuzzleModel,
  boxIndex,
  conflictsAllUnits,
  dedupedPeerConflicts,
  evaluateObjective,
  indexToRc,
  isValidComplete,
  respectsClues,
  unitConflictPairs
} from "./sudoku";
import {
  fitMiharaLastDigit,
  type ModPObservation
} from "./mihara";

export type SudokuMethod = "stepwise" | "zubarev" | "mihara";

export interface SolverOptions {
  method: SudokuMethod;
  seed: number;
  maxSteps: number; // per restart
  restarts: number;
  beta0: number; // Zubarev: initial inverse temperature
  beta1: number; // Zubarev: final inverse temperature
}

export const DEFAULT_SOLVER_OPTIONS: SolverOptions = {
  method: "stepwise",
  seed: 0,
  maxSteps: 60_000,
  restarts: 15,
  beta0: 0.5,
  beta1: 6.0
};

export interface SolverMove {
  row: number; // 0-indexed
  col1: number;
  col2: number;
  delta: number;
  kind: "best" | "random" | "zubarev";
}

export interface SolverSnapshot {
  grid: Grid;
  fixed: boolean[];
  step: number; // step within the current restart
  totalSteps: number; // cumulative across restarts
  restart: number; // 0-indexed
  conflicts: number; // current H_cb (columns + boxes)
  initialConflicts: number; // H_cb at the current restart's initialisation
  bestConflicts: number;
  bestGrid: Grid;
  solved: boolean;
  done: boolean; // search exhausted, or a valid solution was recovered
  lastMove: SolverMove | null;
  beta: number | null; // active inverse temperature (Zubarev only)
  miharaCoefficients: number[] | null;
  miharaInliers: number | null;
  miharaTotal: number | null;
  miharaObservationCount: number | null;
  miharaLoss: number | null;
  miharaFloor: number | null;
  miharaSignedLoss: number | null;
  miharaPrime: number | null;
  miharaNegativeUnitValuations: number | null;
  miharaNegativeInfiniteValuations: number | null;
  miharaDomainViolations: number | null;
  miharaClueViolations: number | null;
  miharaSuccessfulTrials: number | null;
  miharaSingularTrials: number | null;
}

// ---- deterministic RNG (mulberry32) so a seed is reproducible ----
function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randInt(rng: () => number, n: number): number {
  return Math.floor(rng() * n);
}

function shuffle<T>(rng: () => number, array: T[]): T[] {
  for (let i = array.length - 1; i > 0; i -= 1) {
    const j = randInt(rng, i + 1);
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

// H_cb over columns and boxes only (rows are permutations -> contribute 0).
function conflictsColsBoxes(grid: Grid): number {
  let total = 0;
  for (const unit of COLS) total += unitConflictPairs(unit.map((i) => grid[i]));
  for (const unit of BOXES) total += unitConflictPairs(unit.map((i) => grid[i]));
  return total;
}

// Delta in H_cb for swapping the values of cells i1 and i2 (columns+boxes only).
function deltaSwap(grid: Grid, i1: number, i2: number): number {
  const v1 = grid[i1];
  const v2 = grid[i2];
  if (v1 === v2) return 0;
  const [r1, c1] = indexToRc(i1);
  const [r2, c2] = indexToRc(i2);
  const units = dedupeUnits([
    ["C", c1],
    ["C", c2],
    ["B", boxIndex(r1, c1)],
    ["B", boxIndex(r2, c2)]
  ]);
  let before = 0;
  for (const [type, idx] of units) {
    const cells = type === "C" ? COLS[idx] : BOXES[idx];
    before += unitConflictPairs(cells.map((i) => grid[i]));
  }
  grid[i1] = v2;
  grid[i2] = v1;
  let after = 0;
  for (const [type, idx] of units) {
    const cells = type === "C" ? COLS[idx] : BOXES[idx];
    after += unitConflictPairs(cells.map((i) => grid[i]));
  }
  grid[i1] = v1;
  grid[i2] = v2;
  return after - before;
}

type UnitRef = ["C" | "B", number];

function dedupeUnits(refs: UnitRef[]): UnitRef[] {
  const seen = new Set<string>();
  const out: UnitRef[] = [];
  for (const ref of refs) {
    const key = `${ref[0]}${ref[1]}`;
    if (!seen.has(key)) {
      seen.add(key);
      out.push(ref);
    }
  }
  return out;
}

function cellInConflictColBox(grid: Grid, i: number): boolean {
  const [r, c] = indexToRc(i);
  const b = boxIndex(r, c);
  const v = grid[i];
  let colCount = 0;
  for (const j of COLS[c]) if (grid[j] === v) colCount += 1;
  if (colCount > 1) return true;
  let boxCount = 0;
  for (const j of BOXES[b]) if (grid[j] === v) boxCount += 1;
  return boxCount > 1;
}

function betaAt(step: number, maxSteps: number, beta0: number, beta1: number): number {
  if (maxSteps <= 1) return beta1;
  const t = (step - 1) / (maxSteps - 1);
  return beta0 * (1 - t) + beta1 * t;
}

// Sample an index proportional to exp(logWeights[i]), numerically stabilised.
function sampleFromLogWeights(rng: () => number, logWeights: number[]): number {
  const m = Math.max(...logWeights);
  const weights = logWeights.map((lw) => Math.exp(lw - m));
  const total = weights.reduce((a, b) => a + b, 0);
  if (!(total > 0) || !Number.isFinite(total)) {
    let best = 0;
    for (let i = 1; i < logWeights.length; i += 1) {
      if (logWeights[i] > logWeights[best]) best = i;
    }
    return best;
  }
  let r = rng() * total;
  for (let i = 0; i < weights.length; i += 1) {
    r -= weights[i];
    if (r <= 0) return i;
  }
  return weights.length - 1;
}

export class SudokuSolver {
  private readonly puzzle: Grid;
  private readonly options: SolverOptions;
  private readonly rng: () => number;

  private grid: Grid = [];
  private fixed: boolean[] = [];
  private rowFree: number[][] = [];
  private conf = 0;
  private initialConf = 0;

  private step = 0;
  private restart = 0;
  private totalSteps = 0;

  private bestConf = Number.POSITIVE_INFINITY;
  private bestGrid: Grid = [];

  private solved = false;
  private done = false;
  private lastMove: SolverMove | null = null;
  private beta: number | null = null;
  private miharaObservations: ModPObservation[] = [];
  private miharaTotalWeight = 0;
  private miharaFloor = 0;
  private miharaCoefficients: number[] | null = null;
  private miharaInliers: number | null = null;
  private miharaSignedLoss: number | null = null;
  private miharaNegativeUnitValuations: number | null = null;
  private miharaNegativeInfiniteValuations: number | null = null;
  private miharaDomainViolations: number | null = null;
  private miharaClueViolations: number | null = null;
  private miharaSuccessfulTrials = 0;
  private miharaSingularTrials = 0;

  constructor(puzzle: Grid, options: SolverOptions) {
    this.puzzle = puzzle.slice();
    this.options = options;
    this.rng = makeRng(options.seed >>> 0);
    if (options.method === "mihara") {
      const dataframe = buildMiharaPositiveSudokuDataFrame(buildPuzzleModel(puzzle, {
        p: MIHARA_SUDOKU_PRIME
      }));
      this.miharaObservations = dataframe.rows.map((row) => ({
        coefficients: dataframe.variables.map((variable) =>
          row.coefficients[variable.name] ?? 0
        ),
        target: row.target,
        weight: row.weight,
        samplingGroup: row.kind === "pinning"
          ? `pin-${Object.keys(row.coefficients)[0]}`
          : row.id.replace(/-(?:n|p)\d+$/u, ""),
        source: row.source
      }));
      this.miharaTotalWeight = dataframe.totalWeight;
      this.miharaFloor = dataframe.satisfiableFloor;
    }
    this.initRestart();
  }

  private initRestart(): void {
    if (this.options.method === "mihara") {
      this.grid = this.puzzle.slice();
      this.fixed = this.puzzle.map((value) => value !== 0);
      this.rowFree = [];
      this.conf = conflictsAllUnits(this.grid);
      this.initialConf = this.conf;
      this.step = 0;
      this.beta = null;
      this.lastMove = null;
      this.miharaCoefficients = null;
      this.miharaInliers = null;
      this.miharaSignedLoss = null;
      this.miharaNegativeUnitValuations = null;
      this.miharaNegativeInfiniteValuations = null;
      this.miharaDomainViolations = null;
      this.miharaClueViolations = null;
      this.miharaSuccessfulTrials = 0;
      this.miharaSingularTrials = 0;
      this.bestConf = this.conf;
      this.bestGrid = this.grid.slice();
      return;
    }
    const grid = this.puzzle.slice();
    const fixed = this.puzzle.map((v) => v !== 0);
    for (let r = 0; r < 9; r += 1) {
      const cells = ROWS[r];
      const present = new Set(cells.map((i) => grid[i]).filter((v) => v !== 0));
      const missing = shuffle(
        this.rng,
        [1, 2, 3, 4, 5, 6, 7, 8, 9].filter((d) => !present.has(d))
      );
      let mi = 0;
      for (const i of cells) {
        if (grid[i] === 0) {
          grid[i] = missing[mi];
          mi += 1;
        }
      }
    }
    this.grid = grid;
    this.fixed = fixed;
    this.rowFree = ROWS.map((cells) => cells.filter((i) => !fixed[i]));
    this.conf = conflictsColsBoxes(grid);
    this.initialConf = this.conf;
    this.step = 0;
    this.beta = null;
    this.lastMove = null;
    if (this.conf < this.bestConf) {
      this.bestConf = this.conf;
      this.bestGrid = grid.slice();
    }
    if (this.isSolvedState()) {
      this.solved = true;
    }
  }

  private isSolvedState(): boolean {
    return (
      this.conf === 0 &&
      isValidComplete(this.grid) &&
      respectsClues(this.grid, this.puzzle)
    );
  }

  private chooseConflictedRow(): number {
    for (const r of shuffle(this.rng, [0, 1, 2, 3, 4, 5, 6, 7, 8])) {
      const free = this.rowFree[r];
      if (free.length < 2) continue;
      if (free.some((i) => cellInConflictColBox(this.grid, i))) {
        return r;
      }
    }
    return randInt(this.rng, 9);
  }

  // Advance one move. Handles restart rollover and terminal states.
  advance(): SolverSnapshot {
    if (this.solved || this.done) {
      return this.snapshot();
    }
    if (this.options.method === "mihara") {
      return this.advanceMiharaAttempt();
    }
    if (this.step >= this.options.maxSteps) {
      if (this.restart + 1 >= this.options.restarts) {
        this.done = true;
        return this.snapshot();
      }
      this.restart += 1;
      this.initRestart();
      if (this.solved) return this.snapshot();
    }

    const row = this.chooseConflictedRow();
    const free = this.rowFree[row];
    if (free.length < 2) {
      // Nothing to swap here; count the step and move on.
      this.step += 1;
      this.totalSteps += 1;
      return this.snapshot();
    }

    if (this.options.method === "zubarev") {
      this.beta = betaAt(this.step + 1, this.options.maxSteps, this.options.beta0, this.options.beta1);
      const moves: Array<{ i1: number; i2: number; delta: number }> = [];
      const logWeights: number[] = [];
      for (let a = 0; a < free.length - 1; a += 1) {
        for (let b = a + 1; b < free.length; b += 1) {
          const delta = deltaSwap(this.grid, free[a], free[b]);
          moves.push({ i1: free[a], i2: free[b], delta });
          logWeights.push(-this.beta * delta);
        }
      }
      const pick = moves[sampleFromLogWeights(this.rng, logWeights)];
      this.applySwap(pick.i1, pick.i2, pick.delta, "zubarev");
    } else {
      let bestDelta = 0;
      let bestPair: [number, number] | null = null;
      for (let a = 0; a < free.length - 1; a += 1) {
        for (let b = a + 1; b < free.length; b += 1) {
          const delta = deltaSwap(this.grid, free[a], free[b]);
          if (delta < bestDelta) {
            bestDelta = delta;
            bestPair = [free[a], free[b]];
          }
        }
      }
      if (bestPair) {
        this.applySwap(bestPair[0], bestPair[1], bestDelta, "best");
      } else {
        // No improving swap: random swap to escape the local minimum.
        const a = randInt(this.rng, free.length);
        let b = randInt(this.rng, free.length);
        if (b === a) b = (b + 1) % free.length;
        const delta = deltaSwap(this.grid, free[a], free[b]);
        this.applySwap(free[a], free[b], delta, "random");
      }
    }

    this.step += 1;
    this.totalSteps += 1;
    if (this.conf < this.bestConf) {
      this.bestConf = this.conf;
      this.bestGrid = this.grid.slice();
    }
    if (this.isSolvedState()) {
      this.solved = true;
    }
    return this.snapshot();
  }

  private advanceMiharaAttempt(): SolverSnapshot {
    const trialLimit = this.options.maxSteps;
    if (this.step >= trialLimit) {
      this.done = true;
      return this.snapshot();
    }

    const fit = fitMiharaLastDigit(
      this.miharaObservations,
      MIHARA_SUDOKU_PRIME,
      this.options.seed ^ Math.imul(this.step + 1, 0x9e37_79b9),
      1
    );
    this.step += 1;
    this.totalSteps += 1;
    this.miharaSuccessfulTrials += fit.successfulTrials;
    this.miharaSingularTrials += fit.singularTrials;
    if (fit.coefficients) {
      const candidateGrid = fit.coefficients.map((coefficient) =>
        coefficient >= 1 && coefficient <= 9 ? coefficient : 0
      );
      const candidateConflicts = conflictsAllUnits(candidateGrid);
      const candidateSolved = candidateConflicts === 0 &&
        isValidComplete(candidateGrid) && respectsClues(candidateGrid, this.puzzle);
      const improvesDiagnostic = this.miharaInliers == null || fit.inliers > this.miharaInliers;
      if (!candidateSolved && !improvesDiagnostic) {
        if (this.step >= trialLimit) this.done = true;
        return this.snapshot();
      }

      this.miharaCoefficients = fit.coefficients;
      this.miharaInliers = fit.inliers;
      const originalPeerConflicts = dedupedPeerConflicts(fit.coefficients);
      this.miharaSignedLoss = evaluateObjective(
        fit.coefficients,
        buildPuzzleModel(this.puzzle, { p: MIHARA_SUDOKU_PRIME })
      ).loss;
      this.miharaNegativeUnitValuations = PEER_COUNT - originalPeerConflicts;
      this.miharaNegativeInfiniteValuations = originalPeerConflicts;
      this.miharaDomainViolations = fit.coefficients.filter(
        (coefficient) => coefficient < 1 || coefficient > 9
      ).length;
      this.miharaClueViolations = this.puzzle.reduce(
        (count, clue, index) => count + (
          clue !== 0 && fit.coefficients?.[index] !== clue ? 1 : 0
        ),
        0
      );
      this.grid = candidateGrid;
      this.conf = candidateConflicts;
      this.bestConf = this.conf;
      this.bestGrid = this.grid.slice();
      this.solved = candidateSolved;
    }
    if (this.step >= trialLimit && !this.solved) this.done = true;
    return this.snapshot();
  }

  private applySwap(i1: number, i2: number, delta: number, kind: SolverMove["kind"]): void {
    [this.grid[i1], this.grid[i2]] = [this.grid[i2], this.grid[i1]];
    this.conf += delta;
    const [r1, c1] = indexToRc(i1);
    const [, c2] = indexToRc(i2);
    this.lastMove = { row: r1, col1: c1, col2: c2, delta, kind };
  }

  // Run (up to `budget` moves) until solved or exhausted. Returns the snapshot.
  runToCompletion(budget = 5_000_000): SolverSnapshot {
    let used = 0;
    while (!this.solved && !this.done && used < budget) {
      this.advance();
      used += 1;
    }
    return this.snapshot();
  }

  snapshot(): SolverSnapshot {
    return {
      grid: this.grid.slice(),
      fixed: this.fixed.slice(),
      step: this.step,
      totalSteps: this.totalSteps,
      restart: this.restart,
      conflicts: this.conf,
      initialConflicts: this.initialConf,
      bestConflicts: this.bestConf === Number.POSITIVE_INFINITY ? this.conf : this.bestConf,
      bestGrid: (this.bestGrid.length ? this.bestGrid : this.grid).slice(),
      solved: this.solved,
      done: this.done,
      lastMove: this.lastMove,
      beta: this.beta,
      miharaCoefficients: this.miharaCoefficients?.slice() ?? null,
      miharaInliers: this.miharaInliers,
      miharaTotal: this.options.method === "mihara" ? this.miharaTotalWeight : null,
      miharaObservationCount: this.options.method === "mihara"
        ? this.miharaObservations.length
        : null,
      miharaLoss: this.miharaInliers == null
        ? null
        : this.miharaTotalWeight - this.miharaInliers,
      miharaFloor: this.options.method === "mihara" ? this.miharaFloor : null,
      miharaSignedLoss: this.miharaSignedLoss,
      miharaPrime: this.options.method === "mihara" ? MIHARA_SUDOKU_PRIME : null,
      miharaNegativeUnitValuations: this.miharaNegativeUnitValuations,
      miharaNegativeInfiniteValuations: this.miharaNegativeInfiniteValuations,
      miharaDomainViolations: this.miharaDomainViolations,
      miharaClueViolations: this.miharaClueViolations,
      miharaSuccessfulTrials: this.options.method === "mihara"
        ? this.miharaSuccessfulTrials
        : null,
      miharaSingularTrials: this.options.method === "mihara"
        ? this.miharaSingularTrials
        : null
    };
  }
}
