import { describe, expect, it } from "vitest";
import { compileProblem, evaluateAssignment } from "./csp";
import { CSP_SAMPLES } from "./defaultProblems";

describe("CSP samples", () => {
  it("provides distinct, compilable sample problems", () => {
    expect(new Set(CSP_SAMPLES.map((sample) => sample.id)).size).toBe(CSP_SAMPLES.length);

    for (const sample of CSP_SAMPLES) {
      expect(() => compileProblem(sample.cnf)).not.toThrow();
      expect(sample.description.length).toBeGreaterThan(40);
    }
  });

  it.each([
    ["assignment", 16, 60, 65_536],
    ["blackout-restoration", 24, 47, 16_777_216],
    ["museum-security", 23, 23, 8_388_608],
    ["overbooked-festival", 20, 75, 1_048_576]
  ])("builds %s with the intended search size", (id, variables, clauses, assignments) => {
    const sample = CSP_SAMPLES.find((candidate) => candidate.id === id);
    const compiled = compileProblem(sample?.cnf ?? "");

    expect(compiled.variables).toHaveLength(variables);
    expect(compiled.ternaryClauses).toHaveLength(clauses);
    expect(compiled.assignmentCount).toBe(assignments);
  });

  it.each(["museum-security"])(
    "gives %s one known floor assignment",
    (id) => {
      const sample = CSP_SAMPLES.find((candidate) => candidate.id === id);
      const compiled = compileProblem(sample?.cnf ?? "");
      const assignment = Object.fromEntries(
        compiled.variables.map((variable) => [variable.name, !variable.name.includes("active") && !variable.name.includes("open") && !variable.name.includes("override")])
      );
      const evaluation = evaluateAssignment(compiled, assignment);

      expect(evaluation.nonUnitSatisfied).toBe(compiled.ternaryClauses.length);
      expect(evaluation.loss).toBe(compiled.scoring.theoreticalFloor);
    }
  );

  it("gives the blackout network one planted interacting solution", () => {
    const sample = CSP_SAMPLES.find((candidate) => candidate.id === "blackout-restoration");
    const compiled = compileProblem(sample?.cnf ?? "");
    const assignment = Object.fromEntries(compiled.variables.map((variable) => {
      const districtIndex = Math.floor(variable.index / 2);
      const usesBackup = districtIndex % 2 === 0;
      return [
        variable.name,
        variable.name.endsWith("_backup") ? usesBackup : !usesBackup
      ];
    }));
    const oppositeOrientation = Object.fromEntries(
      Object.entries(assignment).map(([name, value]) => [name, !value])
    );

    expect(evaluateAssignment(compiled, assignment).loss)
      .toBe(compiled.scoring.theoreticalFloor);
    expect(evaluateAssignment(compiled, oppositeOrientation).loss)
      .toBe(compiled.scoring.theoreticalFloor + 1);
  });
});
