export interface ModPObservation {
  coefficients: number[];
  target: number;
  source?: string;
}

export interface MiharaLastDigitFit {
  coefficients: number[] | null;
  inliers: number;
  total: number;
  successfulTrials: number;
  singularTrials: number;
}

function mod(value: number, p: number): number {
  return ((value % p) + p) % p;
}

function inverseMod(value: number, p: number): number {
  let base = mod(value, p);
  let exponent = p - 2;
  let result = 1;
  while (exponent > 0) {
    if (exponent & 1) result = mod(result * base, p);
    base = mod(base * base, p);
    exponent = Math.floor(exponent / 2);
  }
  return result;
}

export function solveSquareSystemModP(
  matrix: number[][],
  targets: number[],
  p: number
): number[] | null {
  const dimension = matrix.length;
  if (dimension === 0 || targets.length !== dimension) {
    throw new Error("Expected a non-empty square system.");
  }
  if (matrix.some((row) => row.length !== dimension)) {
    throw new Error("Expected a square matrix.");
  }
  const augmented = matrix.map((row, index) => [
    ...row.map((value) => mod(value, p)),
    mod(targets[index], p)
  ]);

  for (let column = 0; column < dimension; column += 1) {
    let pivot = column;
    while (pivot < dimension && augmented[pivot][column] === 0) pivot += 1;
    if (pivot === dimension) return null;
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    const inverse = inverseMod(augmented[column][column], p);
    augmented[column] = augmented[column].map((value) => mod(value * inverse, p));

    for (let row = 0; row < dimension; row += 1) {
      if (row === column) continue;
      const factor = augmented[row][column];
      if (factor === 0) continue;
      augmented[row] = augmented[row].map((value, index) =>
        mod(value - factor * augmented[column][index], p)
      );
    }
  }
  return augmented.map((row) => row[dimension]);
}

export function makeDeterministicRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function sampleWithoutReplacement(
  length: number,
  count: number,
  rng: () => number
): number[] {
  const indices = Array.from({ length }, (_, index) => index);
  for (let index = 0; index < count; index += 1) {
    const selected = index + Math.floor(rng() * (length - index));
    [indices[index], indices[selected]] = [indices[selected], indices[index]];
  }
  return indices.slice(0, count);
}

export function countModPInliers(
  observations: ModPObservation[],
  coefficients: number[],
  p: number
): number {
  return observations.reduce((count, observation) => {
    const affineValue = observation.coefficients.reduce(
      (sum, coefficient, index) => sum + coefficient * coefficients[index],
      0
    );
    return count + (mod(affineValue - observation.target, p) === 0 ? 1 : 0);
  }, 0);
}

function lexicographicallyBefore(left: number[], right: number[]): boolean {
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) return left[index] < right[index];
  }
  return false;
}

export function fitMiharaLastDigit(
  observations: ModPObservation[],
  p: number,
  seed: number,
  trials: number
): MiharaLastDigitFit {
  if (!observations.length) throw new Error("At least one observation is required.");
  const dimension = observations[0].coefficients.length;
  if (dimension === 0 || observations.some((row) => row.coefficients.length !== dimension)) {
    throw new Error("All observations must have the same positive dimension.");
  }
  if (observations.length < dimension) {
    throw new Error(`Need at least ${dimension} observations for ${dimension} coefficients.`);
  }
  const rng = makeDeterministicRng(seed);
  let best: number[] | null = null;
  let bestInliers = -1;
  let successfulTrials = 0;
  let singularTrials = 0;

  for (let trial = 0; trial < trials; trial += 1) {
    const sample = sampleWithoutReplacement(observations.length, dimension, rng)
      .map((index) => observations[index]);
    const candidate = solveSquareSystemModP(
      sample.map((row) => row.coefficients),
      sample.map((row) => row.target),
      p
    );
    if (!candidate) {
      singularTrials += 1;
      continue;
    }
    successfulTrials += 1;
    const inliers = countModPInliers(observations, candidate, p);
    if (
      inliers > bestInliers ||
      (inliers === bestInliers && best !== null && lexicographicallyBefore(candidate, best))
    ) {
      best = candidate;
      bestInliers = inliers;
    }
  }

  return {
    coefficients: best,
    inliers: Math.max(bestInliers, 0),
    total: observations.length,
    successfulTrials,
    singularTrials
  };
}

// The CSP compiler uses x_i=0 for true and x_i=1 for false.
export function booleanCoordinatesToMask(coefficients: number[]): number | null {
  if (coefficients.some((coefficient) => coefficient !== 0 && coefficient !== 1)) {
    return null;
  }
  return coefficients.reduce(
    (mask, coordinate, index) => mask + (coordinate === 0 ? 2 ** index : 0),
    0
  );
}
