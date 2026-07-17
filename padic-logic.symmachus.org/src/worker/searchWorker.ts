import {
  booleanCoordinatesToMask,
  fitMiharaLastDigit,
  makeDeterministicRng,
  type ModPObservation
} from "../lib/mihara";
import {
  permuteAssignmentIndex,
  type AssignmentPermutation,
  type SearchStrategy
} from "../lib/search";

interface StartMessage {
  type: "start";
  workerId: number;
  evaluatorSource: string;
  lossFloor: number;
  assignmentCount: number;
  variableCount: number;
  strategy: SearchStrategy;
  permutation: AssignmentPermutation;
  miharaObservations: ModPObservation[];
  prime: number;
  start: number;
  endExclusive: number;
  unbounded: boolean;
  updateEveryMs: number;
}

interface StopMessage {
  type: "stop";
}

type IncomingMessage = StartMessage | StopMessage;

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<IncomingMessage>) => void) | null;
  postMessage: (message: unknown) => void;
};
let stopped = false;

ctx.onmessage = (event: MessageEvent<IncomingMessage>) => {
  if (event.data.type === "stop") {
    stopped = true;
    return;
  }

  stopped = false;
  if (event.data.strategy === "zubarev") {
    runZubarevWalk(event.data);
  } else if (event.data.strategy === "mihara") {
    runMiharaAttempt(event.data);
  } else {
    runExhaustiveSearch(event.data);
  }
};

function evaluatorFor(message: StartMessage): (mask: number) => number {
  return new Function(
    `${message.evaluatorSource}; return evaluateMask;`
  )() as (mask: number) => number;
}

function runExhaustiveSearch(message: StartMessage): void {
  const evaluateMask = evaluatorFor(message);
  let cursor = message.start;
  let tested = 0;
  let solutions = 0;
  let bestLoss = Number.POSITIVE_INFINITY;
  let bestMask: number | null = null;
  let currentMask = message.start;
  let lastUpdate = performance.now();
  const startedAt = performance.now();
  const chunkSize = 16_384;

  const emitProgress = (type: "progress" | "done", done: boolean, now: number) => {
    ctx.postMessage({
      type,
      workerId: message.workerId,
      tested,
      currentMask,
      bestLoss: Number.isFinite(bestLoss) ? bestLoss : null,
      bestMask,
      bestCoordinates: null,
      algorithmScore: null,
      algorithmTotal: null,
      algorithmLoss: null,
      algorithmSuccessfulTrials: null,
      algorithmSingularTrials: null,
      solutions,
      speed: tested / Math.max((now - startedAt) / 1000, 0.001),
      done
    });
  };

  const step = () => {
    const chunkEnd = Math.min(cursor + chunkSize, message.endExclusive);
    for (; cursor < chunkEnd; cursor += 1) {
      currentMask = message.strategy === "random"
        ? permuteAssignmentIndex(cursor, message.assignmentCount, message.permutation)
        : cursor;
      const loss = evaluateMask(currentMask);
      tested += 1;
      if (loss === message.lossFloor) solutions += 1;
      if (loss < bestLoss) {
        bestLoss = loss;
        bestMask = currentMask;
        const now = performance.now();
        emitProgress("progress", false, now);
        lastUpdate = now;
      }
    }
    finishOrContinue(step, emitProgress, cursor, message.endExclusive, message.updateEveryMs, lastUpdate);
  };

  step();
}

function runZubarevWalk(message: StartMessage): void {
  const evaluateMask = evaluatorFor(message);
  const rng = makeDeterministicRng(
    message.permutation.offset ^ Math.imul(message.workerId, 0x9e37_79b9)
  );
  let cursor = message.start;
  let tested = 0;
  let solutions = 0;
  let currentMask = Math.floor(rng() * message.assignmentCount);
  let currentLoss = evaluateMask(currentMask);
  let bestLoss = currentLoss;
  let bestMask = currentMask;
  let lastUpdate = performance.now();
  const startedAt = performance.now();
  const chunkSize = 256;

  const emitProgress = (type: "progress" | "done", done: boolean, now: number) => {
    ctx.postMessage({
      type,
      workerId: message.workerId,
      tested,
      currentMask,
      bestLoss,
      bestMask,
      bestCoordinates: null,
      algorithmScore: null,
      algorithmTotal: null,
      algorithmLoss: null,
      algorithmSuccessfulTrials: null,
      algorithmSingularTrials: null,
      solutions,
      speed: tested / Math.max((now - startedAt) / 1000, 0.001),
      done
    });
  };

  const step = () => {
    const chunkEnd = Math.min(cursor + chunkSize, message.endExclusive);
    for (; cursor < chunkEnd; cursor += 1) {
      const progress = (cursor - message.start) /
        Math.max(message.endExclusive - message.start - 1, 1);
      const beta = 0.5 * (1 - progress) + 6 * progress;
      const moves: Array<{ mask: number; loss: number }> = [];
      const logWeights: number[] = [];
      for (let variable = 0; variable < message.variableCount; variable += 1) {
        const bit = 2 ** variable;
        const nextMask = Math.floor(currentMask / bit) % 2 === 1
          ? currentMask - bit
          : currentMask + bit;
        const loss = evaluateMask(nextMask);
        moves.push({ mask: nextMask, loss });
        logWeights.push(-beta * (loss - currentLoss));
      }
      const selected = moves[sampleLogWeights(logWeights, rng)];
      currentMask = selected.mask;
      currentLoss = selected.loss;
      tested += 1;
      if (currentLoss === message.lossFloor) solutions += 1;
      if (currentLoss < bestLoss) {
        bestLoss = currentLoss;
        bestMask = currentMask;
        const now = performance.now();
        emitProgress("progress", false, now);
        lastUpdate = now;
      }
    }
    finishOrContinue(step, emitProgress, cursor, message.endExclusive, message.updateEveryMs, lastUpdate);
  };

  emitProgress("progress", false, startedAt);
  step();
}

function runMiharaAttempt(message: StartMessage): void {
  const evaluateMask = evaluatorFor(message);
  let cursor = message.start;
  let tested = 0;
  let bestCoordinates: number[] | null = null;
  let bestInliers = -1;
  let bestMask: number | null = null;
  let bestLoss: number | null = null;
  let successfulTrials = 0;
  let singularTrials = 0;
  const positiveTotalWeight = message.miharaObservations.reduce(
    (sum, observation) => sum + (observation.weight ?? 1),
    0
  );
  const startedAt = performance.now();
  let lastUpdate = startedAt;
  const chunkSize = 4;
  const endExclusive = message.unbounded ? Number.MAX_SAFE_INTEGER : message.endExclusive;

  const emitProgress = (type: "progress" | "done", done: boolean, now: number) => {
    ctx.postMessage({
      type,
      workerId: message.workerId,
      tested,
      currentMask: bestMask ?? 0,
      bestLoss,
      bestMask,
      bestCoordinates,
      algorithmScore: bestInliers < 0 ? null : bestInliers,
      algorithmTotal: positiveTotalWeight,
      algorithmLoss: bestInliers < 0 ? null : positiveTotalWeight - bestInliers,
      algorithmSuccessfulTrials: successfulTrials,
      algorithmSingularTrials: singularTrials,
      solutions: bestLoss === message.lossFloor ? 1 : 0,
      speed: tested / Math.max((now - startedAt) / 1000, 0.001),
      done
    });
  };

  const step = () => {
    const trials = Math.min(chunkSize, endExclusive - cursor);
    if (trials > 0) {
      const fit = fitMiharaLastDigit(
        message.miharaObservations,
        message.prime,
        message.permutation.offset ^ Math.imul(message.workerId, 0x85eb_ca6b) ^ cursor,
        trials
      );
      tested += trials;
      cursor += trials;
      successfulTrials += fit.successfulTrials;
      singularTrials += fit.singularTrials;
      if (fit.coefficients) {
        const candidateMask = booleanCoordinatesToMask(fit.coefficients);
        const candidateLoss = candidateMask == null ? null : evaluateMask(candidateMask);
        if (candidateLoss === message.lossFloor) {
          bestCoordinates = fit.coefficients;
          bestInliers = fit.inliers;
          bestMask = candidateMask;
          bestLoss = candidateLoss;
          emitProgress("done", true, performance.now());
          return;
        }
      }
      if (fit.coefficients && fit.inliers > bestInliers) {
        bestCoordinates = fit.coefficients;
        bestInliers = fit.inliers;
        bestMask = booleanCoordinatesToMask(fit.coefficients);
        bestLoss = bestMask == null ? null : evaluateMask(bestMask);
        const now = performance.now();
        emitProgress("progress", false, now);
        lastUpdate = now;
      }
    }
    finishOrContinue(step, emitProgress, cursor, endExclusive, message.updateEveryMs, lastUpdate);
  };

  step();
}

function sampleLogWeights(logWeights: number[], rng: () => number): number {
  const maximum = Math.max(...logWeights);
  const weights = logWeights.map((weight) => Math.exp(weight - maximum));
  let threshold = rng() * weights.reduce((sum, weight) => sum + weight, 0);
  for (let index = 0; index < weights.length; index += 1) {
    threshold -= weights[index];
    if (threshold <= 0) return index;
  }
  return weights.length - 1;
}

function finishOrContinue(
  step: () => void,
  emitProgress: (type: "progress" | "done", done: boolean, now: number) => void,
  cursor: number,
  endExclusive: number,
  updateEveryMs: number,
  lastUpdate: number
): void {
  const now = performance.now();
  const done = cursor >= endExclusive || stopped;
  if (now - lastUpdate >= updateEveryMs || done) {
    emitProgress(done ? "done" : "progress", done, now);
  }
  if (!done) setTimeout(step, 0);
}

export {};
