export const DEFAULT_ASSIGNMENT_PROBLEM = [
  "Ava, Ben, Cara, and Devina need to be assigned test, design, documentation and development. Everyone needs exactly one job. Every job needs exactly one person assigned to it.",
  "Rules: Ava cannot test. Neither Ben nor Cara can do  documentation. If Devina documents, then Ava designs."
].join("\n");

const PEOPLE = ["Ava", "Ben", "Cara", "Devina"] as const;
const JOBS = ["test", "design", "documentation", "development"] as const;

export interface DefaultCnfStep {
  cnf: string;
  purpose: string;
}

function variable(person: string, job: string): string {
  return `${person}_${job}`;
}

function exactlyOne(
  steps: DefaultCnfStep[],
  variables: string[],
  atLeastOnePurpose: string,
  pairPurpose: (left: string, right: string) => string
): void {
  steps.push({
    cnf: variables.join(" or "),
    purpose: atLeastOnePurpose
  });
  for (let left = 0; left < variables.length; left += 1) {
    for (let right = left + 1; right < variables.length; right += 1) {
      steps.push({
        cnf: `not ${variables[left]} or not ${variables[right]}`,
        purpose: pairPurpose(variables[left], variables[right])
      });
    }
  }
}

function buildDefaultAssignmentSteps(): DefaultCnfStep[] {
  const steps: DefaultCnfStep[] = [];

  for (const person of PEOPLE) {
    exactlyOne(
      steps,
      JOBS.map((job) => variable(person, job)),
      `${person} must be assigned at least one job.`,
      (left, right) => `${left} and ${right} cannot both be true.`
    );
  }

  for (const job of JOBS) {
    exactlyOne(
      steps,
      PEOPLE.map((person) => variable(person, job)),
      `The ${job} job must be assigned to at least one person.`,
      (left, right) => `${left} and ${right} cannot both be true.`
    );
  }

  steps.push({ cnf: "not Ava_test", purpose: "Ava cannot test." });
  steps.push({
    cnf: "not Ben_documentation",
    purpose: "Ben cannot do documentation."
  });
  steps.push({
    cnf: "not Cara_documentation",
    purpose: "Cara cannot do documentation."
  });
  steps.push({
    cnf: "not Devina_documentation or Ava_design",
    purpose: "If Devina documents, then Ava designs."
  });

  return steps;
}

export const DEFAULT_ASSIGNMENT_STEPS = buildDefaultAssignmentSteps();
export const DEFAULT_ASSIGNMENT_CSP = DEFAULT_ASSIGNMENT_STEPS
  .map((step) => step.cnf)
  .join("\n");

export function countCspConstraints(source: string): number {
  return source
    .split(/\r?\n/u)
    .map((line) => line.replace(/#.*/u, "").trim())
    .filter(Boolean).length;
}
