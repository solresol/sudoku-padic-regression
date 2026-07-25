import { describe, expect, it } from "vitest";
import {
  buildMiharaPositiveRegressionDataFrame,
  buildRegressionDataFrame,
  compileProblem,
  decodeClauseResidual,
  evaluateAssignment,
  evaluateExpression,
  evaluateRegressionDataFrame,
  falseLabels,
  INFORMATIVE_VARIABLE_LIMIT,
  parseProblem,
  renderClause
} from "./csp";
import { DEFAULT_ASSIGNMENT_CSP } from "./defaultProblems";

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

  it("accepts both implication spellings and compiles them to CNF", () => {
    const compiled = compileProblem("A implies B\nC -> D");

    expect(compiled.variables.map((variable) => variable.name)).toEqual([
      "A",
      "B",
      "C",
      "D"
    ]);
    expect(compiled.ternaryClauses.map(renderClause)).toEqual([
      "(~A v B)",
      "(~C v D)"
    ]);
  });

  it("makes implication right-associative with lower precedence than or", () => {
    const chained = compileProblem("A -> B -> C");
    const mixed = compileProblem("A or B -> C");

    expect(chained.ternaryClauses.map(renderClause)).toEqual([
      "(~A v ~B v C)"
    ]);
    expect(evaluateAssignment(mixed, { A: true, B: true, C: false }).loss)
      .toBe(mixed.scoring.theoreticalFloor + 1);
    expect(evaluateAssignment(mixed, { A: true, B: true, C: true }).loss)
      .toBe(mixed.scoring.theoreticalFloor);
  });

  it("scores only true implies false as a violated constraint", () => {
    const compiled = compileProblem("A -> B");
    const floor = compiled.scoring.theoreticalFloor;

    expect(evaluateAssignment(compiled, { A: false, B: false }).loss).toBe(floor);
    expect(evaluateAssignment(compiled, { A: false, B: true }).loss).toBe(floor);
    expect(evaluateAssignment(compiled, { A: true, B: true }).loss).toBe(floor);
    expect(evaluateAssignment(compiled, { A: true, B: false }).loss).toBe(floor + 1);
  });

  it("scores satisfying assignments at the theoretical floor", () => {
    const compiled = compileProblem(sample);
    const assignment = { A: true, B: true, C: false, D: true };

    expect(evaluateAssignment(compiled, assignment)).toMatchObject({
      loss: 22,
      nonUnitSatisfied: 6,
      theoreticalFloor: 22,
      unitWellViolations: 0
    });
  });

  it("builds the p-adic regression dataframe with unit wells", () => {
    const compiled = compileProblem("A or not B");
    const rows = buildRegressionDataFrame(compiled);

    expect(rows).toHaveLength(5);
    expect(rows[0]).toMatchObject({
      kind: "constraint",
      coefficients: { A: 1, B: -1 },
      relation: "≠",
      target: 1,
      sign: -1,
      weight: 1
    });
    expect(rows.slice(1).map((row) => row.kind)).toEqual([
      "unit-well",
      "unit-well",
      "unit-well",
      "unit-well"
    ]);
    expect(rows.slice(1).map((row) => row.target)).toEqual([0, 1, 0, 1]);
    expect(rows.slice(1).map((row) => row.weight)).toEqual([2, 2, 2, 2]);
    expect(compiled.scoring).toMatchObject({
      unitWellWeight: 2,
      theoreticalFloor: 3
    });
  });

  it("uses a forbidden target and conservative well weight for the assignment CSP", () => {
    const oneOfFour = compileProblem(
      "Ava_test or Ava_design or Ava_documentation or Ava_development"
    );
    const row = buildRegressionDataFrame(oneOfFour)[0];

    expect(row).toMatchObject({
      coefficients: {
        Ava_test: 1,
        Ava_design: 1,
        Ava_documentation: 1,
        Ava_development: 1
      },
      relation: "≠",
      target: 4,
      sign: -1,
      weight: 1
    });

    const fullAssignment = compileProblem(DEFAULT_ASSIGNMENT_CSP);
    expect(fullAssignment.variables).toHaveLength(16);
    expect(fullAssignment.ternaryClauses).toHaveLength(60);
    expect(fullAssignment.scoring).toMatchObject({
      unitWellWeight: 61,
      theoreticalFloor: 916
    });
  });

  it("audits every dataframe row and sums to the assignment loss", () => {
    const compiled = compileProblem("A or not B");
    const satisfying = { A: true, B: false };
    const satisfiedFrame = evaluateRegressionDataFrame(compiled, satisfying);

    expect(satisfiedFrame.rows).toHaveLength(5);
    expect(satisfiedFrame.rows[0]).toMatchObject({
      affineValue: -1,
      residual: -2,
      pAdicValuation: 0,
      pAdicNorm: 1,
      signedWeight: -1,
      contribution: -1,
      status: "satisfied"
    });
    expect(satisfiedFrame.rows.map((row) => row.contribution)).toEqual([-1, 0, 2, 2, 0]);
    expect(satisfiedFrame.totalLoss).toBe(evaluateAssignment(compiled, satisfying).loss);

    const violated = { A: false, B: true };
    const violatedFrame = evaluateRegressionDataFrame(compiled, violated);
    expect(violatedFrame.rows[0]).toMatchObject({
      residual: 0,
      pAdicValuation: null,
      pAdicNorm: 0,
      contribution: 0,
      status: "violated"
    });
    expect(violatedFrame.totalLoss).toBe(compiled.scoring.theoreticalFloor + 1);
  });

  it("expands each negative clause row into its positive affine complement", () => {
    const compiled = compileProblem("A or not B");
    const frame = buildMiharaPositiveRegressionDataFrame(compiled);
    const clauseRows = frame.rows.filter((row) => row.kind === "constraint");

    expect(clauseRows.map((row) => row.target)).toEqual([-1, 0]);
    expect(clauseRows.every((row) => row.sign === 1 && row.relation === "="))
      .toBe(true);
    expect(clauseRows.map((row) => row.weight)).toEqual([1, 1]);
    expect(frame.totalWeight).toBe(10);
    expect(frame.satisfiableFloor).toBe(5);
  });

  it("generates an executable bit-mask evaluator", () => {
    const compiled = compileProblem(sample);
    const evaluatorFactory = new Function(
      `${compiled.evaluatorSource}; return evaluateMask;`
    ) as () => (mask: number) => number;
    const evaluateMask = evaluatorFactory();

    // mask 0b1011 => A=true, B=true, C=false, D=true
    expect(evaluateMask(0b1011)).toBe(22);
    expect(compiled.evaluatorSource).not.toMatch(/xor/i);
    expect(evaluateMask(0)).toBeGreaterThan(22);
  });
});

describe("informative false labels", () => {
  const everyAssignment = (names: string[]): Array<Record<string, boolean>> =>
    Array.from({ length: 2 ** names.length }, (_, mask) =>
      Object.fromEntries(names.map((name, index) => [name, ((mask >> index) & 1) === 1]))
    );

  it("labels false as 1 + p·2^i in wells and clause targets", () => {
    const compiled = compileProblem("A or not B");
    const rows = buildRegressionDataFrame(compiled, "informative");

    // b_A = 1 + 17·1 = 18, b_B = 1 + 17·2 = 35; target = sum over positive literals.
    expect(rows[0]).toMatchObject({
      kind: "constraint",
      coefficients: { A: 1, B: -1 },
      relation: "≠",
      target: 18
    });
    expect(rows.slice(1).map((row) => row.target)).toEqual([0, 18, 0, 35]);
  });

  it("produces the identical loss to the uniform labelling at every assignment", () => {
    const compiled = compileProblem(sample);
    const names = compiled.variables.map((variable) => variable.name);

    for (const assignment of everyAssignment(names)) {
      const uniform = evaluateRegressionDataFrame(compiled, assignment, "uniform");
      const informative = evaluateRegressionDataFrame(compiled, assignment, "informative");
      expect(informative.totalLoss).toBe(uniform.totalLoss);
      expect(informative.rows.map((row) => row.pAdicNorm))
        .toEqual(uniform.rows.map((row) => row.pAdicNorm));
    }
  });

  it("decodes every clause residual to the exact satisfied literals", () => {
    const compiled = compileProblem(sample);
    const names = compiled.variables.map((variable) => variable.name);
    const prime = compiled.scoring.prime;

    for (const assignment of everyAssignment(names)) {
      const frame = evaluateRegressionDataFrame(compiled, assignment, "informative");
      const clauseRows = frame.rows.filter((row) => row.kind === "constraint");
      expect(clauseRows).toHaveLength(compiled.ternaryClauses.length);

      clauseRows.forEach((row, clauseIndex) => {
        const clause = compiled.ternaryClauses[clauseIndex];
        const expected = clause.terms
          .filter((term) => evaluateExpression(term, (name) => assignment[name] ?? false))
          .map((term) =>
            term.type === "literal"
              ? term.name
              : term.type === "not" && term.expr.type === "literal"
                ? term.expr.name
                : ""
          )
          .sort();

        const decoded = decodeClauseResidual(row.residual, prime);
        expect(decoded).not.toBeNull();
        const decodedNames = decoded!.satisfiedVariableIndices
          .map((index) => compiled.variables[index].name)
          .sort();
        expect(decodedNames).toEqual(expected);
        expect(decoded!.satisfiedCount).toBe(expected.length);
      });
    }
  });

  it("keeps the count decodable but not the identity under uniform labels", () => {
    const compiled = compileProblem("A or not B");
    const frame = evaluateRegressionDataFrame(
      compiled,
      { A: true, B: false },
      "uniform"
    );
    const decoded = decodeClauseResidual(frame.rows[0].residual, compiled.scoring.prime);
    expect(decoded).toEqual({ satisfiedCount: 2, satisfiedVariableIndices: [] });
  });

  it("deduplicates repeated literals so targets match coefficients", () => {
    // (A v A) previously kept coefficient {A: 1} but target 2·b_A, so the
    // falsifying assignment missed the forbidden hyperplane entirely.
    const compiled = compileProblem("A or A");
    expect(compiled.ternaryClauses).toHaveLength(1);
    expect(compiled.ternaryClauses[0].terms).toHaveLength(1);
    expect(buildRegressionDataFrame(compiled)[0]).toMatchObject({
      coefficients: { A: 1 },
      target: 1
    });

    const falsified = evaluateRegressionDataFrame(compiled, { A: false });
    expect(falsified.rows[0].status).toBe("violated");
    expect(falsified.totalLoss).toBe(evaluateAssignment(compiled, { A: false }).loss);

    const informative = evaluateRegressionDataFrame(compiled, { A: true }, "informative");
    expect(decodeClauseResidual(informative.rows[0].residual, compiled.scoring.prime))
      .toEqual({ satisfiedCount: 1, satisfiedVariableIndices: [0] });
  });

  it("drops tautological clauses", () => {
    const compiled = compileProblem("A or not A");
    expect(compiled.ternaryClauses).toHaveLength(0);
  });

  it("keeps informative labels exactly representable up to the variable limit", () => {
    const atLimit = Array.from({ length: INFORMATIVE_VARIABLE_LIMIT }, (_, index) => ({
      name: `v${index}`,
      index
    }));
    const labels = falseLabels(atLimit, "informative", 17);
    expect(Object.values(labels).every((label) => Number.isSafeInteger(label))).toBe(true);

    // The reviewer's counterexample: at index 49 the label rounds and its
    // residue mod 17 collapses to 0.
    expect(Number.isSafeInteger(1 + 17 * 2 ** 49)).toBe(false);

    const overLimit = [...atLimit, { name: "v45", index: INFORMATIVE_VARIABLE_LIMIT }];
    expect(() => falseLabels(overLimit, "informative", 17)).toThrow(/45/);
    expect(() => falseLabels(overLimit, "uniform", 17)).not.toThrow();
  });

  it("expands informative Mihara complements over the labelled affine values", () => {
    const compiled = compileProblem("A or not B");
    const frame = buildMiharaPositiveRegressionDataFrame(compiled, "informative");
    const clauseRows = frame.rows.filter((row) => row.kind === "constraint");

    // attainable values of A - B with A in {0,18}, B in {0,35}: -35, -17, 0, 18;
    // the forbidden target 18 is dropped.
    expect(clauseRows.map((row) => row.target)).toEqual([-35, -17, 0]);
  });
});
