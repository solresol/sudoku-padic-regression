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

export interface CspSample {
  id: string;
  title: string;
  description: string;
  cnf: string;
  expected: "SAT" | "UNSAT";
}

const POWER_DISTRICTS = [
  "Harbour",
  "University",
  "Observatory",
  "Hospital",
  "Old_Town",
  "Airport",
  "Riverside",
  "Stadium",
  "Market",
  "Depot",
  "Hill",
  "Data_Centre"
] as const;

const MUSEUM_BEAMS = Array.from(
  { length: 20 },
  (_, index) => `gallery_${String(index + 1).padStart(2, "0")}_beam_armed`
);
const MUSEUM_OFF = [
  "west_service_door_open",
  "alarm_bypass_active",
  "night_guard_override"
] as const;

function fixedStateCnf(on: readonly string[], off: readonly string[]): string {
  return [...on, ...off.map((variableName) => `not ${variableName}`)].join("\n");
}

function blackoutRestorationCnf(): string {
  const constraints = POWER_DISTRICTS.map(
    (district) => `${district}_primary xor ${district}_backup`
  );

  for (let index = 0; index < POWER_DISTRICTS.length - 1; index += 1) {
    constraints.push(
      `${POWER_DISTRICTS[index]}_backup xor ${POWER_DISTRICTS[index + 1]}_backup`
    );
  }

  constraints.push("Harbour_backup or Observatory_backup or Old_Town_backup");
  return constraints.join("\n");
}

function overbookedFestivalCnf(): string {
  const speakers = ["Ada", "Benoit", "Chandra", "Diego", "Elena"];
  const stages = ["atrium", "forum", "library", "rooftop"];
  const clauses: string[] = [];

  for (const speaker of speakers) {
    clauses.push(...exactlyOneClauses(stages.map((stage) => `${speaker}_${stage}`)));
  }
  for (const stage of stages) {
    const stageSpeakers = speakers.map((speaker) => `${speaker}_${stage}`);
    for (let left = 0; left < stageSpeakers.length; left += 1) {
      for (let right = left + 1; right < stageSpeakers.length; right += 1) {
        clauses.push(`not ${stageSpeakers[left]} or not ${stageSpeakers[right]}`);
      }
    }
  }
  return clauses.join("\n");
}

function exactlyOneClauses(variables: string[]): string[] {
  const clauses = [variables.join(" or ")];
  for (let left = 0; left < variables.length; left += 1) {
    for (let right = left + 1; right < variables.length; right += 1) {
      clauses.push(`not ${variables[left]} or not ${variables[right]}`);
    }
  }
  return clauses;
}

export const CSP_SAMPLES: CspSample[] = [
  {
    id: "assignment",
    title: "Four-person job assignment",
    description: DEFAULT_ASSIGNMENT_PROBLEM,
    cnf: DEFAULT_ASSIGNMENT_CSP,
    expected: "SAT"
  },
  {
    id: "blackout-restoration",
    title: "City blackout restoration",
    description: [
      "After a city-wide blackout, twelve districts must each be connected to exactly one of two feeders: primary or backup. The districts, ordered north to south, are Harbour, University, Observatory, Hospital, Old Town, Airport, Riverside, Stadium, Market, Depot, Hill and Data Centre.",
      "To avoid overloading the shared transformers, every neighbouring pair of districts must use opposite feeder types. At least one of Harbour, Observatory and Old Town must use its backup feeder. Find the unique restoration plan."
    ].join("\n"),
    cnf: blackoutRestorationCnf(),
    expected: "SAT"
  },
  {
    id: "museum-security",
    title: "Midnight museum security",
    description: [
      "At midnight the curator must arm all 20 gallery beam zones before the diamond can be left in the building.",
      "The west service door, alarm bypass and night-guard override must all remain off. Find the unique secure configuration."
    ].join("\n"),
    cnf: fixedStateCnf(MUSEUM_BEAMS, MUSEUM_OFF),
    expected: "SAT"
  },
  {
    id: "overbooked-festival",
    title: "Five speakers, four stages",
    description: [
      "Ada, Benoit, Chandra, Diego and Elena must each give exactly one keynote.",
      "There are only four simultaneous stages: atrium, forum, library and rooftop. No stage can host two speakers. Can the timetable be built?"
    ].join("\n"),
    cnf: overbookedFestivalCnf(),
    expected: "UNSAT"
  }
];

export function countCspConstraints(source: string): number {
  return source
    .split(/\r?\n/u)
    .map((line) => line.replace(/#.*/u, "").trim())
    .filter(Boolean).length;
}
