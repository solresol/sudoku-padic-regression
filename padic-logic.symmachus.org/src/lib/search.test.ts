import { describe, expect, it } from "vitest";
import { compileProblem } from "./csp";
import {
  createSearchPermutation,
  createSearchPlan,
  exhaustiveSearchRange,
  permuteAssignmentIndex,
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

  it("builds an ordered exhaustive plan with the unit-well floor", () => {
    const compiled = compileProblem("A or B\nB xor not C");
    const plan = createSearchPlan(compiled, 4);

    expect(plan.strategy).toBe("ordered");
    expect(plan.assignmentCount).toBe(8);
    expect(plan.lossFloor).toBe(compiled.scoring.theoreticalFloor);
    expect(plan.lossFloor).toBe(9);
    expect(plan.ranges).toHaveLength(4);
  });

  it("permutes every hyperplane exactly once for random search", () => {
    const assignmentCount = 256;
    const permutation = createSearchPermutation(assignmentCount, 0x1234_5678);
    const order = Array.from(
      { length: assignmentCount },
      (_, index) => permuteAssignmentIndex(index, assignmentCount, permutation)
    );

    expect(new Set(order).size).toBe(assignmentCount);
    expect(order).not.toEqual(Array.from({ length: assignmentCount }, (_, index) => index));
    expect(Math.min(...order)).toBe(0);
    expect(Math.max(...order)).toBe(assignmentCount - 1);
  });

  it("finds a satisfying assignment by exhaustive range search", () => {
    const compiled = compileProblem("A or B\nB xor not C");
    const result = exhaustiveSearchRange(compiled, 0, 8);

    expect(result.bestLoss).toBe(compiled.scoring.theoreticalFloor);
    expect(result.bestMask).not.toBeNull();
    expect(result.tested).toBe(8);
  });
});
