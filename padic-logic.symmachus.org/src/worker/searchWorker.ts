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
  strategy: SearchStrategy;
  permutation: AssignmentPermutation;
  start: number;
  endExclusive: number;
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
  runSearch(event.data);
};

function runSearch(message: StartMessage): void {
  const evaluateMask = new Function(
    `${message.evaluatorSource}; return evaluateMask;`
  )() as (mask: number) => number;

  let cursor = message.start;
  let tested = 0;
  let solutions = 0;
  let bestLoss = Number.POSITIVE_INFINITY;
  let bestMask: number | null = null;
  let currentMask = message.start;
  let lastUpdate = performance.now();
  const startedAt = performance.now();
  const chunkSize = 16_384;

  const emitProgress = (
    type: "progress" | "done",
    done: boolean,
    now: number
  ) => {
    ctx.postMessage({
      type,
      workerId: message.workerId,
      tested,
      currentMask,
      bestLoss: Number.isFinite(bestLoss) ? bestLoss : null,
      bestMask,
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
      if (loss === message.lossFloor) {
        solutions += 1;
      }
      if (loss < bestLoss) {
        bestLoss = loss;
        bestMask = currentMask;
        const now = performance.now();
        emitProgress("progress", false, now);
        lastUpdate = now;
      }
    }

    const now = performance.now();
    const shouldUpdate =
      now - lastUpdate >= message.updateEveryMs ||
      cursor >= message.endExclusive ||
      stopped;

    if (shouldUpdate) {
      lastUpdate = now;
      const done = cursor >= message.endExclusive || stopped;
      emitProgress(done ? "done" : "progress", done, now);
    }

    if (cursor < message.endExclusive && !stopped) {
      setTimeout(step, 0);
    }
  };

  step();
}

export {};
