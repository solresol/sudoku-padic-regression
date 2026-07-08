import { describe, expect, it } from "vitest";
import { compileProblem } from "./csp";
import {
  createSearchPlan,
  exhaustiveSearchRange,
  splitAssignmentRanges
} from "./search";

describe("bit-mask search", () => {
  it("splits assignment ranges across workers without gaps", () => {
    const ranges = splitAssignmentRanges(20, 8);

    expect(ranges).toHaveLength(8);
    expect(ranges[0]).toMatchObject({ start: 0, endExclusive: 131_072 });
    expect(ranges[7]).toMatchObject({
      start: 917_504,
      endExclusive: 1_048_576
    });
  });

  it("builds a brute-force plan with a zero-loss floor", () => {
    const plan = createSearchPlan(compileProblem("A or B\nB xor not C"), 4);

    expect(plan.strategy).toBe("brute-force");
    expect(plan.assignmentCount).toBe(8);
    expect(plan.lossFloor).toBe(0);
    expect(plan.ranges).toHaveLength(4);
  });

  it("finds a satisfying assignment by exhaustive range search", () => {
    const compiled = compileProblem("A or B\nB xor not C");
    const result = exhaustiveSearchRange(compiled, 0, 8);

    expect(result.bestLoss).toBe(0);
    expect(result.bestMask).not.toBeNull();
    expect(result.tested).toBe(8);
  });
});
