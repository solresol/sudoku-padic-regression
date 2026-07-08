// Sudoku as a signed p-adic residual objective.
//
// This is the "all-different / list-colouring" side of the paper
// (Signed p-adic Residual Encodings of Finite-Domain All-Different Systems).
// One integer coefficient per cell (81 variables), NOT a 729-variable one-hot lift.
//
// Over Z_p with a prime p > 9 (default p = 11), for digits a, b in {1,...,9}:
//     |a - b|_p = 0  iff a = b
//     |a - b|_p = 1  iff a != b
// because 0 < |a - b| < p forces p to not divide (a - b). So a p-adic residual
// on a difference is an exact inequality indicator on the digit alphabet. The
// norms below are computed genuinely (via the p-adic valuation), not faked.

export const P_DEFAULT = 11;
// Deduped peer degree is 20, so the pinning weight alpha must exceed 20.
export const ALPHA_DEFAULT = 21;
export const DIGITS = [1, 2, 3, 4, 5, 6, 7, 8, 9] as const;

export type Grid = number[]; // length 81, row-major, 0 = blank

// -------------------------
// Indexing
// -------------------------

export function rcToIndex(r: number, c: number): number {
  return 9 * r + c;
}

export function indexToRc(i: number): [number, number] {
  return [Math.floor(i / 9), i % 9];
}

export function boxIndex(r: number, c: number): number {
  return Math.floor(r / 3) * 3 + Math.floor(c / 3);
}

function buildRows(): number[][] {
  return Array.from({ length: 9 }, (_, r) =>
    Array.from({ length: 9 }, (_, c) => rcToIndex(r, c))
  );
}

function buildCols(): number[][] {
  return Array.from({ length: 9 }, (_, c) =>
    Array.from({ length: 9 }, (_, r) => rcToIndex(r, c))
  );
}

function buildBoxes(): number[][] {
  const boxes: number[][] = [];
  for (let br = 0; br < 3; br += 1) {
    for (let bc = 0; bc < 3; bc += 1) {
      const cells: number[] = [];
      for (let dr = 0; dr < 3; dr += 1) {
        for (let dc = 0; dc < 3; dc += 1) {
          cells.push(rcToIndex(3 * br + dr, 3 * bc + dc));
        }
      }
      boxes.push(cells);
    }
  }
  return boxes;
}

export const ROWS = buildRows();
export const COLS = buildCols();
export const BOXES = buildBoxes();
export const UNITS = [...ROWS, ...COLS, ...BOXES];

// Deduplicated peer pairs: the set P of unordered cell pairs sharing a row,
// column, or box. There are exactly 810 of them (27 * C(9,2) - 162 overlaps).
function buildPeerPairs(): Array<[number, number]> {
  const seen = new Set<number>();
  const pairs: Array<[number, number]> = [];
  for (const unit of UNITS) {
    for (let a = 0; a < unit.length - 1; a += 1) {
      for (let b = a + 1; b < unit.length; b += 1) {
        const i = Math.min(unit[a], unit[b]);
        const j = Math.max(unit[a], unit[b]);
        const key = i * 81 + j;
        if (!seen.has(key)) {
          seen.add(key);
          pairs.push([i, j]);
        }
      }
    }
  }
  return pairs;
}

export const PEER_PAIRS = buildPeerPairs();
export const PEER_COUNT = PEER_PAIRS.length; // 810

// Per-cell peer list (20 peers each), for fast local conflict checks.
function buildPeers(): number[][] {
  const peers: Set<number>[] = Array.from({ length: 81 }, () => new Set<number>());
  for (const unit of UNITS) {
    for (const i of unit) {
      for (const j of unit) {
        if (i !== j) {
          peers[i].add(j);
        }
      }
    }
  }
  return peers.map((s) => Array.from(s).sort((a, b) => a - b));
}

export const PEERS = buildPeers();
export const MAX_DEGREE = PEERS[0].length; // 20

// -------------------------
// Parsing / formatting
// -------------------------

export function parsePuzzle(source: string): Grid {
  const s = source.replace(/\s+/gu, "");
  if (s.length !== 81) {
    throw new Error(`Puzzle must be 81 characters (digits, 0 or . for blanks); got ${s.length}.`);
  }
  const grid: Grid = [];
  for (const ch of s) {
    if (ch === "." || ch === "0") {
      grid.push(0);
    } else if (ch >= "1" && ch <= "9") {
      grid.push(Number(ch));
    } else {
      throw new Error(`Invalid character in puzzle: "${ch}".`);
    }
  }
  return grid;
}

export function gridToString(grid: Grid): string {
  return grid.map((d) => (d === 0 ? "." : String(d))).join("");
}

// -------------------------
// Conflict counting (matches padic_sudoku_regression.py)
// -------------------------

export function unitConflictPairs(values: number[]): number {
  const counts = new Array(10).fill(0);
  for (const v of values) {
    if (v !== 0) {
      counts[v] += 1;
    }
  }
  let total = 0;
  for (const c of counts) {
    total += (c * (c - 1)) / 2;
  }
  return total;
}

// H_cb: duplicated unit-scope column + box conflict count (paper eq. for the
// row-permutation state space; rows are permutations so contribute 0).
export function conflictsColsBoxes(grid: Grid): number {
  let total = 0;
  for (const unit of COLS) total += unitConflictPairs(unit.map((i) => grid[i]));
  for (const unit of BOXES) total += unitConflictPairs(unit.map((i) => grid[i]));
  return total;
}

export function conflictsAllUnits(grid: Grid): number {
  let total = 0;
  for (const unit of UNITS) total += unitConflictPairs(unit.map((i) => grid[i]));
  return total;
}

// Deduplicated peer-conflict count: the objective the theorem characterises.
export function dedupedPeerConflicts(grid: Grid): number {
  let total = 0;
  for (const [i, j] of PEER_PAIRS) {
    if (grid[i] !== 0 && grid[i] === grid[j]) {
      total += 1;
    }
  }
  return total;
}

// -------------------------
// Validity
// -------------------------

export function isValidComplete(grid: Grid): boolean {
  if (grid.some((v) => v < 1 || v > 9)) {
    return false;
  }
  for (const unit of UNITS) {
    const seen = new Array(10).fill(false);
    for (const i of unit) {
      if (seen[grid[i]]) {
        return false;
      }
      seen[grid[i]] = true;
    }
  }
  return true;
}

export function respectsClues(grid: Grid, puzzle: Grid): boolean {
  for (let i = 0; i < 81; i += 1) {
    if (puzzle[i] !== 0 && grid[i] !== puzzle[i]) {
      return false;
    }
  }
  return true;
}

// -------------------------
// Genuine p-adic norms
// -------------------------

export function vP(n: number, p: number): number {
  let m = Math.abs(n);
  if (m === 0) {
    return Number.POSITIVE_INFINITY;
  }
  let k = 0;
  while (m % p === 0) {
    m /= p;
    k += 1;
  }
  return k;
}

// |n|_p, with |0|_p = 0.
export function pAdicNorm(n: number, p: number): number {
  if (n === 0) {
    return 0;
  }
  return p ** -vP(n, p);
}

// U_i(t) = sum_{a in allowed} |t - a|_p. Minimised (value |allowed| - 1) exactly
// on t in allowed, once p exceeds the spread of the allowed set.
export function digitSnappingPenalty(
  t: number,
  p: number,
  allowed: readonly number[] = DIGITS
): number {
  let sum = 0;
  for (const a of allowed) {
    sum += pAdicNorm(t - a, p);
  }
  return sum;
}

// -------------------------
// Construction summary (for display)
// -------------------------

export interface PuzzleModel {
  puzzle: Grid;
  clues: number[]; // clue cell indices
  clueCount: number;
  p: number;
  alpha: number;
  variables: number; // 81
  positiveObservationsMinimal: number; // 729 - 8|C|
  positiveObservationsExpanded: number; // 729 + |C|
  negativeObservations: number; // |P| = 810
  peerCount: number; // 810
  maxDegree: number; // 20
  unaryConstant: number; // sum_i (|D_i| - 1) = 8(81 - |C|)
  theoreticalFloor: number; // alpha * unaryConstant - |P|
}

export function buildPuzzleModel(
  puzzle: Grid,
  options: { p?: number; alpha?: number } = {}
): PuzzleModel {
  const p = options.p ?? P_DEFAULT;
  const alpha = options.alpha ?? ALPHA_DEFAULT;
  const clues: number[] = [];
  for (let i = 0; i < 81; i += 1) {
    if (puzzle[i] !== 0) {
      clues.push(i);
    }
  }
  const clueCount = clues.length;
  const unaryConstant = 8 * (81 - clueCount); // unclued cells contribute |D_i| - 1 = 8
  return {
    puzzle,
    clues,
    clueCount,
    p,
    alpha,
    variables: 81,
    positiveObservationsMinimal: 729 - 8 * clueCount,
    positiveObservationsExpanded: 729 + clueCount,
    negativeObservations: PEER_COUNT,
    peerCount: PEER_COUNT,
    maxDegree: MAX_DEGREE,
    unaryConstant,
    theoreticalFloor: alpha * unaryConstant - PEER_COUNT
  };
}

// -------------------------
// Puzzle generation (for the "new random puzzle" button)
// -------------------------

// A random complete valid grid, by shuffling a canonical Latin-square pattern.
// The caller supplies an rng (e.g. Math.random or a seeded generator).
export function randomSolvedGrid(rng: () => number): Grid {
  const base = 3;
  const side = 9;
  const pattern = (r: number, c: number) =>
    (base * (r % base) + Math.floor(r / base) + c) % side;

  const shuffle = <T,>(arr: T[]): T[] => {
    for (let i = arr.length - 1; i > 0; i -= 1) {
      const j = Math.floor(rng() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  };

  const spread = (): number[] => {
    const groups = shuffle([0, 1, 2]);
    const out: number[] = [];
    for (const g of groups) {
      for (const x of shuffle([0, 1, 2])) {
        out.push(g * base + x);
      }
    }
    return out;
  };

  const rowOrder = spread();
  const colOrder = spread();
  const nums = shuffle([1, 2, 3, 4, 5, 6, 7, 8, 9]);

  const grid: Grid = new Array(81).fill(0);
  for (let r = 0; r < 9; r += 1) {
    for (let c = 0; c < 9; c += 1) {
      grid[rcToIndex(r, c)] = nums[pattern(rowOrder[r], colOrder[c])];
    }
  }
  return grid;
}

// Remove entries uniformly at random down to `clueCount` (not uniqueness-checked,
// matching the paper's fast-carving procedure).
export function carve(solution: Grid, clueCount: number, rng: () => number): Grid {
  const puzzle = solution.slice();
  const positions = Array.from({ length: 81 }, (_, i) => i);
  for (let i = positions.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [positions[i], positions[j]] = [positions[j], positions[i]];
  }
  const toRemove = Math.max(0, 81 - clueCount);
  for (let k = 0; k < toRemove; k += 1) {
    puzzle[positions[k]] = 0;
  }
  return puzzle;
}

// -------------------------
// Genuine objective evaluation
// -------------------------

export interface ObjectiveBreakdown {
  unaryTotal: number; // sum_i U_i(x_i), computed p-adically
  edgeRewardTotal: number; // sum over peer pairs of |x_i - x_j|_p
  loss: number; // alpha * unaryTotal - edgeRewardTotal  (= L_min)
  theoreticalFloor: number;
  dedupedConflicts: number; // = loss - floor on domain-respecting states
  hcbConflicts: number; // duplicated column/box surrogate H_cb
  domainRespecting: boolean;
}

// Evaluates the minimal signed p-adic residual objective
//     L_min(x) = alpha * sum_i U_i(x_i)  -  sum_{peer pairs} |x_i - x_j|_p
// genuinely (every term is a real p-adic norm). On domain-respecting digit
// states this equals the theoretical floor plus the deduped peer-conflict count.
export function evaluateObjective(grid: Grid, model: PuzzleModel): ObjectiveBreakdown {
  const { p, alpha, puzzle } = model;
  let unaryTotal = 0;
  let domainRespecting = true;
  for (let i = 0; i < 81; i += 1) {
    const allowed = puzzle[i] !== 0 ? [puzzle[i]] : DIGITS;
    unaryTotal += digitSnappingPenalty(grid[i], p, allowed);
    if (!(allowed as readonly number[]).includes(grid[i])) {
      domainRespecting = false;
    }
  }
  let edgeRewardTotal = 0;
  for (const [i, j] of PEER_PAIRS) {
    edgeRewardTotal += pAdicNorm(grid[i] - grid[j], p);
  }
  const loss = alpha * unaryTotal - edgeRewardTotal;
  return {
    unaryTotal,
    edgeRewardTotal,
    loss,
    theoreticalFloor: model.theoreticalFloor,
    dedupedConflicts: loss - model.theoreticalFloor,
    hcbConflicts: conflictsColsBoxes(grid),
    domainRespecting
  };
}
