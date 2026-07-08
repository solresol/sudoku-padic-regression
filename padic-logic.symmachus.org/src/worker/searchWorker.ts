interface StartMessage {
  type: "start";
  workerId: number;
  evaluatorSource: string;
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
  let lastUpdate = performance.now();
  const startedAt = performance.now();
  const chunkSize = 16_384;

  const step = () => {
    const chunkEnd = Math.min(cursor + chunkSize, message.endExclusive);
    for (; cursor < chunkEnd; cursor += 1) {
      const loss = evaluateMask(cursor);
      tested += 1;
      if (loss === 0) {
        solutions += 1;
      }
      if (loss < bestLoss) {
        bestLoss = loss;
        bestMask = cursor;
      }
    }

    const now = performance.now();
    const shouldUpdate =
      now - lastUpdate >= message.updateEveryMs ||
      cursor >= message.endExclusive ||
      stopped;

    if (shouldUpdate) {
      lastUpdate = now;
      ctx.postMessage({
        type: cursor >= message.endExclusive || stopped ? "done" : "progress",
        workerId: message.workerId,
        tested,
        currentMask: Math.max(message.start, cursor - 1),
        bestLoss: Number.isFinite(bestLoss) ? bestLoss : null,
        bestMask,
        solutions,
        speed: tested / Math.max((now - startedAt) / 1000, 0.001),
        done: cursor >= message.endExclusive || stopped
      });
    }

    if (cursor < message.endExclusive && !stopped) {
      setTimeout(step, 0);
    }
  };

  step();
}

export {};
