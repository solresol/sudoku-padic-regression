import type { CompiledProblem } from "./csp";

export type SearchStrategy = "ordered" | "random" | "zubarev" | "mihara";

export interface AssignmentPermutation {
  multiplier: number;
  offset: number;
}

export interface AssignmentRange {
  workerId: number;
  start: number;
  endExclusive: number;
}

export interface SearchPlan {
  strategy: SearchStrategy;
  assignmentCount: number;
  workUnits: number;
  workLabel: "hyperplanes" | "walk steps" | "RANSAC trials";
  lossFloor: number;
  ranges: AssignmentRange[];
}

export interface RangeSearchResult {
  bestLoss: number;
  bestMask: number | null;
  tested: number;
}

export function splitAssignmentRanges(
  variableCount: number,
  workerCount: number
): AssignmentRange[] {
  const safeWorkerCount = Math.max(1, Math.floor(workerCount));
  const assignmentCount = 2 ** variableCount;
  const base = Math.floor(assignmentCount / safeWorkerCount);
  const remainder = assignmentCount % safeWorkerCount;
  const ranges: AssignmentRange[] = [];
  let start = 0;

  for (let workerIndex = 0; workerIndex < safeWorkerCount; workerIndex += 1) {
    const width = base + (workerIndex < remainder ? 1 : 0);
    ranges.push({
      workerId: workerIndex + 1,
      start,
      endExclusive: start + width
    });
    start += width;
  }

  return ranges;
}

export function splitWorkRanges(workUnits: number, workerCount: number): AssignmentRange[] {
  const safeWorkerCount = Math.max(1, Math.floor(workerCount));
  const safeWorkUnits = Math.max(0, Math.floor(workUnits));
  const base = Math.floor(safeWorkUnits / safeWorkerCount);
  const remainder = safeWorkUnits % safeWorkerCount;
  const ranges: AssignmentRange[] = [];
  let start = 0;
  for (let workerIndex = 0; workerIndex < safeWorkerCount; workerIndex += 1) {
    const width = base + (workerIndex < remainder ? 1 : 0);
    ranges.push({
      workerId: workerIndex + 1,
      start,
      endExclusive: start + width
    });
    start += width;
  }
  return ranges;
}

export function createSearchPlan(
  compiled: CompiledProblem,
  workerCount: number,
  strategy: SearchStrategy = "ordered"
): SearchPlan {
  const assignmentCount = compiled.assignmentCount;
  const workUnits = strategy === "mihara"
    ? Math.max(1, Math.floor(workerCount)) * 48
    : strategy === "zubarev"
      ? Math.max(512, Math.min(100_000, assignmentCount * 2))
      : assignmentCount;
  return {
    strategy,
    assignmentCount,
    workUnits,
    workLabel: strategy === "mihara"
      ? "RANSAC trials"
      : strategy === "zubarev"
        ? "walk steps"
        : "hyperplanes",
    lossFloor: compiled.scoring.theoreticalFloor,
    ranges: strategy === "ordered" || strategy === "random"
      ? splitAssignmentRanges(compiled.variables.length, workerCount)
      : splitWorkRanges(workUnits, workerCount)
  };
}

export function createSearchPermutation(
  assignmentCount: number,
  seed: number
): AssignmentPermutation {
  if (assignmentCount <= 1) {
    return { multiplier: 1, offset: 0 };
  }

  const normalizedSeed = seed >>> 0;
  if (assignmentCount > 2 ** 31) {
    return {
      multiplier: 1,
      offset: normalizedSeed % assignmentCount
    };
  }

  const mixedSeed = Math.imul(normalizedSeed ^ 0x9e37_79b9, 0x85eb_ca6b) >>> 0;
  return {
    multiplier: (mixedSeed | 1) >>> 0,
    offset: (normalizedSeed ^ (normalizedSeed >>> 16)) % assignmentCount
  };
}

export function permuteAssignmentIndex(
  index: number,
  assignmentCount: number,
  permutation: AssignmentPermutation
): number {
  if (assignmentCount <= 1) {
    return 0;
  }
  if (permutation.multiplier === 1 || assignmentCount > 2 ** 31) {
    return (index + permutation.offset) % assignmentCount;
  }
  return (
    (Math.imul(index >>> 0, permutation.multiplier >>> 0) + permutation.offset) >>> 0
  ) % assignmentCount;
}

export function exhaustiveSearchRange(
  compiled: CompiledProblem,
  start: number,
  endExclusive: number
): RangeSearchResult {
  const evaluateMask = buildEvaluator(compiled.evaluatorSource);
  let bestLoss = Number.POSITIVE_INFINITY;
  let bestMask: number | null = null;
  let tested = 0;

  for (let mask = start; mask < endExclusive; mask += 1) {
    const loss = evaluateMask(mask);
    tested += 1;
    if (loss < bestLoss) {
      bestLoss = loss;
      bestMask = mask;
    }
  }

  return { bestLoss, bestMask, tested };
}

export function maskToAssignment(
  compiled: Pick<CompiledProblem, "variables">,
  mask: number
): Record<string, boolean> {
  return Object.fromEntries(
    compiled.variables.map((variable) => [
      variable.name,
      Math.floor(mask / 2 ** variable.index) % 2 === 1
    ])
  );
}

export function formatAssignmentCount(variableCount: number): string {
  return new Intl.NumberFormat("en-US").format(2 ** variableCount);
}

export function buildEvaluator(source: string): (mask: number) => number {
  return new Function(`${source}; return evaluateMask;`)() as (
    mask: number
  ) => number;
}
