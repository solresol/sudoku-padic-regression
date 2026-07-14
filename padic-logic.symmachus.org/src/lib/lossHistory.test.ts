import { describe, expect, it } from "vitest";
import {
  MAX_LOSS_HISTORY_POINTS,
  appendLossHistory,
  type LossHistoryPoint
} from "./lossHistory";

describe("loss history downsampling", () => {
  it("keeps a bounded trace spanning the full search", () => {
    let history: LossHistoryPoint[] = [];
    for (let step = 0; step < 20_000; step += 1) {
      history = appendLossHistory(history, {
        step,
        loss: 8_000 + Math.round(30 * Math.sin(step / 17))
      });
    }

    expect(history.length).toBeLessThanOrEqual(MAX_LOSS_HISTORY_POINTS);
    expect(history[0].step).toBe(0);
    expect(history[history.length - 1].step).toBe(19_999);
    expect(history.every((point, index) => index === 0 || point.step > history[index - 1].step))
      .toBe(true);
    expect(Math.min(...history.map((point) => point.loss))).toBeLessThanOrEqual(7_971);
    expect(Math.max(...history.map((point) => point.loss))).toBeGreaterThanOrEqual(8_029);
  });

  it("always retains a newly reached floor as the final sample", () => {
    let history: LossHistoryPoint[] = [];
    for (let step = 0; step <= MAX_LOSS_HISTORY_POINTS; step += 1) {
      history = appendLossHistory(history, { step, loss: 100 - (step % 5) });
    }
    history = appendLossHistory(history, { step: 10_000, loss: 0 });

    expect(history[0].step).toBe(0);
    expect(history[history.length - 1]).toEqual({ step: 10_000, loss: 0 });
  });
});
