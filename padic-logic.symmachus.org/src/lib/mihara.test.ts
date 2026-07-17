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

  it("counts positive observation weights in the consensus score", () => {
    const observations = [
      { coefficients: [1], target: 0, weight: 5 },
      { coefficients: [1], target: 1, weight: 2 }
    ];

    expect(countModPInliers(observations, [0], 17)).toBe(5);
    expect(fitMiharaLastDigit(observations, 17, 3, 2).total).toBe(7);
  });

  it("samples a full-rank basis across equivalent-target groups", () => {
    const observations = [
      { coefficients: [1, 0], target: 0, samplingGroup: "x" },
      { coefficients: [1, 0], target: 1, samplingGroup: "x" },
      { coefficients: [0, 1], target: 0, samplingGroup: "y" },
      { coefficients: [0, 1], target: 1, samplingGroup: "y" },
      { coefficients: [1, 1], target: 1, samplingGroup: "sum" }
    ];

    const fit = fitMiharaLastDigit(observations, 17, 9, 1);

    expect(fit.successfulTrials).toBe(1);
    expect(fit.singularTrials).toBe(0);
    expect(fit.coefficients).toHaveLength(2);
  });
});
