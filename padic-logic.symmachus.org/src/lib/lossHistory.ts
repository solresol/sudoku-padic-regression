export interface LossHistoryPoint {
  step: number;
  loss: number;
}

export const MAX_LOSS_HISTORY_POINTS = 600;

export function appendLossHistory(
  history: LossHistoryPoint[],
  point: LossHistoryPoint
): LossHistoryPoint[] {
  const next = [...history, point];
  if (next.length <= MAX_LOSS_HISTORY_POINTS) {
    return next;
  }

  const compacted: LossHistoryPoint[] = [next[0]];
  const finalIndex = next.length - 1;

  for (let start = 1; start < finalIndex; start += 4) {
    const bucket = next.slice(start, Math.min(start + 4, finalIndex));
    if (bucket.length === 0) {
      continue;
    }
    let minimum = bucket[0];
    let maximum = bucket[0];
    for (const candidate of bucket.slice(1)) {
      if (candidate.loss < minimum.loss) {
        minimum = candidate;
      }
      if (candidate.loss > maximum.loss) {
        maximum = candidate;
      }
    }
    const extrema = minimum.step <= maximum.step
      ? [minimum, maximum]
      : [maximum, minimum];
    for (const candidate of extrema) {
      if (compacted[compacted.length - 1].step !== candidate.step) {
        compacted.push(candidate);
      }
    }
  }

  if (compacted[compacted.length - 1].step !== point.step) {
    compacted.push(point);
  }
  return compacted;
}
