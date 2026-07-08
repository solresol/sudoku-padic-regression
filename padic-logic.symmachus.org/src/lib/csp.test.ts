import { describe, expect, it } from "vitest";
import {
  compileProblem,
  evaluateAssignment,
  parseProblem,
  renderClause
} from "./csp";

const sample = `
# Example CSP
A or B or C
B xor C xor not D
not A or D
`;

describe("CSP compiler", () => {
  it("parses constraints and preserves variable order", () => {
    const problem = parseProblem(sample);

    expect(problem.constraints).toHaveLength(3);
    expect(problem.variables.map((variable) => variable.name)).toEqual([
      "A",
      "B",
      "C",
      "D"
    ]);
  });

  it("renders only OR clauses after ternary/CNF compilation", () => {
    const compiled = compileProblem(sample);

    expect(compiled.ternaryClauses.map(renderClause)).toEqual([
      "(A v B v C)",
      "(~B v C v D)",
      "(B v ~C v D)",
      "(B v C v ~D)",
      "(~B v ~C v ~D)",
      "(~A v D)"
    ]);
    expect(compiled.ternaryClauses.map(renderClause).join(" ")).not.toMatch(/xor/i);
    expect(compiled.validation.maxClauseWidth).toBe(3);
  });

  it("treats ^ as conjunction, not XOR", () => {
    const compiled = compileProblem("(A v B) ^ (~C)");

    expect(compiled.ternaryClauses.map(renderClause)).toEqual([
      "(A v B)",
      "(~C)"
    ]);
  });

  it("scores satisfying assignments at the theoretical floor", () => {
    const compiled = compileProblem(sample);
    const assignment = { A: true, B: true, C: false, D: true };

    expect(evaluateAssignment(compiled, assignment)).toMatchObject({
      loss: 0,
      nonUnitSatisfied: 6,
      theoreticalFloor: 0,
      unitWellViolations: 0
    });
  });

  it("generates an executable bit-mask evaluator", () => {
    const compiled = compileProblem(sample);
    const evaluatorFactory = new Function(
      `${compiled.evaluatorSource}; return evaluateMask;`
    ) as () => (mask: number) => number;
    const evaluateMask = evaluatorFactory();

    // mask 0b1011 => A=true, B=true, C=false, D=true
    expect(evaluateMask(0b1011)).toBe(0);
    expect(compiled.evaluatorSource).not.toMatch(/xor/i);
    expect(evaluateMask(0)).toBeGreaterThan(0);
  });
});
