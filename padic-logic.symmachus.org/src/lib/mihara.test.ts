import { describe, expect, it } from "vitest";
import {
  booleanCoordinatesToMask,
  countModPInliers,
  fitMiharaLastDigit,
  solveSquareSystemModP
} from "./mihara";

describe("Mihara digitwise comparison helpers", () => {
  it("solves a full-rank system modulo p", () => {
    expect(solveSquareSystemModP([[1, 1], [2, 1]], [8, 11], 17)).toEqual([3, 5]);
  });

  it("recovers a clean last digit by random full-rank samples", () => {
    const observations = [
      { coefficients: [1, 0], target: 3 },
      { coefficients: [0, 1], target: 5 },
      { coefficients: [1, 1], target: 8 },
      { coefficients: [2, 1], target: 11 }
    ];

    const fit = fitMiharaLastDigit(observations, 17, 2, 8);

    expect(fit.coefficients).toEqual([3, 5]);
    expect(fit.inliers).toBe(4);
    expect(countModPInliers(observations, [3, 5], 17)).toBe(4);
  });

  it("only decodes genuine Boolean coordinates", () => {
    expect(booleanCoordinatesToMask([0, 1, 0])).toBe(5);
    expect(booleanCoordinatesToMask([0, 16, 1])).toBeNull();
  });
});
