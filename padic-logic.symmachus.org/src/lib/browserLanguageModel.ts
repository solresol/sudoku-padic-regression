import { type CompiledProblem, compileProblem } from "./csp";
import {
  DEFAULT_ASSIGNMENT_PROBLEM,
  DEFAULT_ASSIGNMENT_STEPS
} from "./defaultProblems";

export type LanguageModelAvailability =
  | "unavailable"
  | "downloadable"
  | "downloading"
  | "available";

export interface CnfGenerationStep {
  cnf: string;
  purpose: string;
}

export interface CnfReview {
  status: "complete" | "needs_changes";
  wrongClauses: string[];
  missingConstraints: string[];
  correctedClauses: CnfGenerationStep[];
}

export interface CnfConversationEntry {
  role: "system" | "user" | "assistant" | "assistant-recorded" | "browser" | "cache";
  content: string;
}

export type CnfGenerationEvent =
  | { type: "conversation"; entry: CnfConversationEntry }
  | { type: "variables"; variables: string[] }
  | { type: "clause"; step: CnfGenerationStep; source: string }
  | { type: "review"; review: CnfReview; source: string };

export interface CspGenerationResult {
  source: string;
  usedLocalModel: boolean;
  clauses: CnfGenerationStep[];
  review: CnfReview | null;
}

interface GenerateCspOptions {
  onProgress?: (event: CnfGenerationEvent) => void;
  maxClauses?: number;
  maxReviewCycles?: number;
  signal?: AbortSignal;
  useCache?: boolean;
}

interface CachedCnfGeneration {
  clauses: CnfGenerationStep[];
}

interface AssignmentLiteralSchema {
  person: string;
  job: string;
}

interface AssignmentForbiddenSchema extends AssignmentLiteralSchema {
  source?: string;
}

interface AssignmentImplicationSchema {
  if: AssignmentLiteralSchema;
  then: AssignmentLiteralSchema;
  source?: string;
}

interface AssignmentCspSchema {
  kind: "assignment";
  people: string[];
  jobs: string[];
  personExactlyOneJob: boolean;
  jobExactlyOnePerson: boolean;
  forbidden: AssignmentForbiddenSchema[];
  implications: AssignmentImplicationSchema[];
}

const LANGUAGE_OPTIONS: LanguageModelCreateOptions = {
  expectedInputs: [{ type: "text", languages: ["en"] }],
  expectedOutputs: [{ type: "text", languages: ["en"] }]
};

const CLAUSE_SCHEMA =
  '{"cnf":"one valid CNF OR-clause only","purpose":"the source sentence or constraint this captures"}';
const VARIABLE_STAGE_DISPLAY_MS = 600;

export const CNF_GENERATION_CACHE_ENABLED = false;

const cnfGenerationCache = new Map<string, CachedCnfGeneration>([
  [
    normalizeDescription(DEFAULT_ASSIGNMENT_PROBLEM),
    { clauses: DEFAULT_ASSIGNMENT_STEPS.map((step) => ({ ...step })) }
  ]
]);

export function isCnfGenerationCacheEnabled(): boolean {
  return CNF_GENERATION_CACHE_ENABLED;
}

export function getCachedCnfGeneration(description: string): CachedCnfGeneration | null {
  const cached = cnfGenerationCache.get(normalizeDescription(description));
  return cached
    ? { clauses: cached.clauses.map((clause) => ({ ...clause })) }
    : null;
}

export async function detectLanguageModel(): Promise<LanguageModelAvailability> {
  const languageModel = getLanguageModelFactory();
  if (!languageModel) {
    return "unavailable";
  }

  return languageModel.availability(LANGUAGE_OPTIONS);
}

export async function generateCspFromDescription(
  description: string,
  options: GenerateCspOptions = {}
): Promise<CspGenerationResult> {
  if (!description.trim()) {
    throw new Error("Enter a problem statement before generating CSP clauses.");
  }
  throwIfAborted(options.signal);

  const cached = getCachedCnfGeneration(description);
  if ((options.useCache ?? CNF_GENERATION_CACHE_ENABLED) && cached) {
    return generateFromCache(cached, options.onProgress, options.signal);
  }

  const languageModel = getLanguageModelFactory();
  if (!languageModel) {
    throw new Error("This browser does not expose window.languageModel.");
  }

  const availability = await languageModel.availability(LANGUAGE_OPTIONS);
  throwIfAborted(options.signal);
  emitConversation(options.onProgress, "browser", `LanguageModel.availability(): ${availability}`);
  if (availability === "unavailable") {
    throw new Error("This browser does not have a usable local language model.");
  }

  const schemaSession = await createLanguageModelSession(
    languageModel,
    "schema extraction",
    options.onProgress,
    options.signal
  );
  try {
    return await generateFromSchemaSession(
      schemaSession,
      description,
      options.onProgress,
      options.signal
    );
  } finally {
    schemaSession.destroy?.();
  }
}

async function createLanguageModelSession(
  languageModel: LanguageModelFactory,
  purpose: string,
  onProgress?: (event: CnfGenerationEvent) => void,
  signal?: AbortSignal
): Promise<LanguageModelSession> {
  emitConversation(onProgress, "browser", `Creating local language model session for ${purpose}.`);
  const session = await languageModel.create({
    ...LANGUAGE_OPTIONS,
    signal,
    monitor(monitor) {
      monitor.addEventListener("downloadprogress", (event) => {
        const progress = event as ProgressEvent;
        const detail = progress.lengthComputable && progress.total
          ? ` ${Math.round((progress.loaded / progress.total) * 100)}%`
          : "";
        emitConversation(onProgress, "browser", `Language model download progress${detail}.`);
      });
    }
  });
  throwIfAborted(signal);
  emitConversation(onProgress, "browser", `Local language model session ready for ${purpose}.`);
  return session;
}

async function generateFromSchemaSession(
  session: LanguageModelSession,
  description: string,
  onProgress?: (event: CnfGenerationEvent) => void,
  signal?: AbortSignal
): Promise<CspGenerationResult> {
  const history: LanguageModelPrompt[] = [];
  const response = await promptWithHistory(
    session,
    history,
    [
      "Extract a typed JSON schema for this assignment CSP. Output JSON only.",
      "Do not output CNF clauses. Do not explain. Do not use Markdown.",
      "Schema shape:",
      '{"kind":"assignment","people":["Ava"],"jobs":["test"],"person_exactly_one_job":true,"job_exactly_one_person":true,"forbidden":[{"person":"Ava","job":"test","source":"Ava cannot test."}],"implications":[{"if":{"person":"Devina","job":"documentation"},"then":{"person":"Ava","job":"design"},"source":"If Devina documents, then Ava designs."}]}',
      "Rules:",
      "- people and jobs must contain only names stated by the problem.",
      "- For `Neither Ben nor Cara can do documentation`, output two forbidden entries, one for Ben and one for Cara.",
      "- For `If X does job1, then Y does job2`, output one implication with `if` and `then` literals.",
      "- Set both exactly-one booleans only when the problem says everyone gets exactly one job and every job gets exactly one person.",
      "--- Problem ---",
      description
    ].join("\n"),
    onProgress,
    signal
  );
  throwIfAborted(signal);
  const schema = parseAssignmentSchemaResponse(response);
  const correctedJson = JSON.stringify(assignmentSchemaToJson(schema));
  appendAssistantJson(history, correctedJson, onProgress);
  const variables = assignmentSchemaVariables(schema);
  if (onProgress && variables.length) {
    onProgress({ type: "variables", variables });
    await waitForIntermediateStage(VARIABLE_STAGE_DISPLAY_MS, signal);
  }

  const clauses = assignmentSchemaToCnfSteps(schema);
  const accepted: CnfGenerationStep[] = [];
  for (const step of clauses) {
    throwIfAborted(signal);
    validateGeneratedStep(step, accepted);
    accepted.push(step);
    onProgress?.({
      type: "clause",
      step,
      source: stepsToSource(accepted)
    });
  }

  const review: CnfReview = {
    status: "complete",
    wrongClauses: [],
    missingConstraints: [],
    correctedClauses: []
  };
  onProgress?.({ type: "review", review, source: stepsToSource(accepted) });

  return {
    source: stepsToSource(accepted),
    usedLocalModel: true,
    clauses: accepted,
    review
  };
}

async function collectVariablesWithSession(
  session: LanguageModelSession,
  description: string,
  onProgress?: (event: CnfGenerationEvent) => void
): Promise<string> {
  const history: LanguageModelPrompt[] = [];
  const variables = await promptWithHistory(
    session,
    history,
    [
      "I am going to turn the following problem into CNF form.",
      "Name all boolean variables that we will use. Use identifiers with underscores.",
      "---",
      description
    ].join("\n"),
    onProgress
  );
  emitConversation(onProgress, "assistant-recorded", variables);
  return variables;
}

function parseAssignmentSchemaResponse(response: string): AssignmentCspSchema {
  const value = parseJsonObject(response);
  if (value.kind !== "assignment") {
    throw new Error("Expected assignment schema JSON with `kind: \"assignment\"`.");
  }

  const people = uniqueCleanStrings(value.people);
  const jobs = uniqueCleanStrings(value.jobs);
  if (!people.length || !jobs.length) {
    throw new Error("Assignment schema must include non-empty `people` and `jobs` arrays.");
  }

  const forbidden = Array.isArray(value.forbidden)
    ? value.forbidden.flatMap((entry) => parseForbiddenEntry(entry))
    : [];
  const implications = Array.isArray(value.implications)
    ? value.implications.flatMap((entry) => parseImplicationEntry(entry))
    : [];

  return {
    kind: "assignment",
    people,
    jobs,
    personExactlyOneJob: value.person_exactly_one_job === true,
    jobExactlyOnePerson: value.job_exactly_one_person === true,
    forbidden,
    implications
  };
}

function assignmentSchemaToJson(schema: AssignmentCspSchema): Record<string, unknown> {
  return {
    kind: schema.kind,
    people: schema.people,
    jobs: schema.jobs,
    person_exactly_one_job: schema.personExactlyOneJob,
    job_exactly_one_person: schema.jobExactlyOnePerson,
    forbidden: schema.forbidden,
    implications: schema.implications
  };
}

function assignmentSchemaVariables(schema: AssignmentCspSchema): string[] {
  const variables = new Set<string>();
  if (schema.personExactlyOneJob || schema.jobExactlyOnePerson) {
    for (const person of schema.people) {
      for (const job of schema.jobs) {
        variables.add(assignmentVariable(person, job));
      }
    }
  }
  for (const entry of schema.forbidden) {
    variables.add(assignmentVariable(entry.person, entry.job));
  }
  for (const entry of schema.implications) {
    variables.add(assignmentVariable(entry.if.person, entry.if.job));
    variables.add(assignmentVariable(entry.then.person, entry.then.job));
  }
  return [...variables];
}

function assignmentSchemaToCnfSteps(schema: AssignmentCspSchema): CnfGenerationStep[] {
  const steps: CnfGenerationStep[] = [];

  if (schema.personExactlyOneJob) {
    for (const person of schema.people) {
      addExactlyOneSteps(
        steps,
        schema.jobs.map((job) => assignmentVariable(person, job)),
        `${person} must be assigned at least one job.`,
        (left, right) => `${left} and ${right} cannot both be true.`
      );
    }
  }

  if (schema.jobExactlyOnePerson) {
    for (const job of schema.jobs) {
      addExactlyOneSteps(
        steps,
        schema.people.map((person) => assignmentVariable(person, job)),
        `The ${job} job must be assigned to at least one person.`,
        (left, right) => `${left} and ${right} cannot both be true.`
      );
    }
  }

  for (const entry of schema.forbidden) {
    steps.push({
      cnf: `not ${assignmentVariable(entry.person, entry.job)}`,
      purpose: entry.source?.trim() || `${entry.person} cannot do ${entry.job}.`
    });
  }

  for (const entry of schema.implications) {
    steps.push({
      cnf: `not ${assignmentVariable(entry.if.person, entry.if.job)} or ${assignmentVariable(entry.then.person, entry.then.job)}`,
      purpose:
        entry.source?.trim() ||
        `If ${entry.if.person} does ${entry.if.job}, then ${entry.then.person} does ${entry.then.job}.`
    });
  }

  return steps;
}

function addExactlyOneSteps(
  steps: CnfGenerationStep[],
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

function parseForbiddenEntry(entry: unknown): AssignmentForbiddenSchema[] {
  if (typeof entry !== "object" || entry == null) {
    return [];
  }
  const record = entry as Record<string, unknown>;
  if (typeof record.person !== "string" || typeof record.job !== "string") {
    return [];
  }
  return [
    {
      person: record.person.trim(),
      job: record.job.trim(),
      source: typeof record.source === "string" ? record.source.trim() : undefined
    }
  ];
}

function parseImplicationEntry(entry: unknown): AssignmentImplicationSchema[] {
  if (typeof entry !== "object" || entry == null) {
    return [];
  }
  const record = entry as Record<string, unknown>;
  const ifLiteral = parseLiteralEntry(record.if);
  const thenLiteral = parseLiteralEntry(record.then);
  if (!ifLiteral || !thenLiteral) {
    return [];
  }
  return [
    {
      if: ifLiteral,
      then: thenLiteral,
      source: typeof record.source === "string" ? record.source.trim() : undefined
    }
  ];
}

function parseLiteralEntry(entry: unknown): AssignmentLiteralSchema | null {
  if (typeof entry !== "object" || entry == null) {
    return null;
  }
  const record = entry as Record<string, unknown>;
  if (typeof record.person !== "string" || typeof record.job !== "string") {
    return null;
  }
  return {
    person: record.person.trim(),
    job: record.job.trim()
  };
}

function uniqueCleanStrings(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const seen = new Set<string>();
  const result: string[] = [];
  for (const entry of value) {
    if (typeof entry !== "string") {
      continue;
    }
    const cleaned = entry.trim();
    const normalized = cleaned.toLowerCase();
    if (cleaned && !seen.has(normalized)) {
      seen.add(normalized);
      result.push(cleaned);
    }
  }
  return result;
}

function assignmentVariable(person: string, job: string): string {
  return `${identifierPart(person)}_${identifierPart(job).toLowerCase()}`;
}

function identifierPart(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/gu, "")
    .replace(/[^a-z0-9]+/giu, "_")
    .replace(/^_+|_+$/gu, "")
    || "unknown";
}

async function generateFromCache(
  cached: CachedCnfGeneration,
  onProgress?: (event: CnfGenerationEvent) => void,
  signal?: AbortSignal
): Promise<CspGenerationResult> {
  const clauses: CnfGenerationStep[] = [];
  emitConversation(
    onProgress,
    "cache",
    `Cache hit: replaying ${cached.clauses.length} known CNF clauses.`
  );

  for (const step of cached.clauses) {
    throwIfAborted(signal);
    await Promise.resolve();
    validateGeneratedStep(step, clauses);
    clauses.push({ ...step });
    onProgress?.({
      type: "clause",
      step,
      source: stepsToSource(clauses)
    });
  }

  const review: CnfReview = {
    status: "complete",
    wrongClauses: [],
    missingConstraints: [],
    correctedClauses: []
  };
  onProgress?.({ type: "review", review, source: stepsToSource(clauses) });

  return {
    source: stepsToSource(clauses),
    usedLocalModel: false,
    clauses,
    review
  };
}

async function generateWithSession(
  session: LanguageModelSession,
  description: string,
  variables: string,
  options: GenerateCspOptions
): Promise<CspGenerationResult> {
  const maxClauses = options.maxClauses ?? 140;
  const maxReviewCycles = options.maxReviewCycles ?? 3;
  const clauses: CnfGenerationStep[] = [];
  const history: LanguageModelPrompt[] = [];
  let latestReview: CnfReview | null = null;

  for (let reviewCycle = 0; reviewCycle < maxReviewCycles; reviewCycle += 1) {
    await collectClauseBatch({
      session,
      history,
      description,
      variables,
      clauses,
      maxClauses,
      onProgress: options.onProgress,
      firstClause: clauses.length === 0,
      review: latestReview
    });

    latestReview = await reviewClauses(
      session,
      history,
      description,
      clauses,
      options.onProgress
    );
    applyReviewChanges(clauses, latestReview);
    options.onProgress?.({
      type: "review",
      review: latestReview,
      source: stepsToSource(clauses)
    });

    if (latestReview.status === "complete") {
      break;
    }
  }

  if (!clauses.length) {
    throw new Error("The browser model did not produce any CSP clauses.");
  }
  if (latestReview?.status !== "complete") {
    throw new Error("The browser model did not confirm complete CSP coverage.");
  }

  const source = stepsToSource(clauses);
  compileProblem(source);
  return {
    source,
    usedLocalModel: true,
    clauses,
    review: latestReview
  };
}

async function collectClauseBatch({
  session,
  history,
  description,
  variables,
  clauses,
  maxClauses,
  onProgress,
  firstClause,
  review
}: {
  session: LanguageModelSession;
  history: LanguageModelPrompt[];
  description: string;
  variables: string;
  clauses: CnfGenerationStep[];
  maxClauses: number;
  onProgress?: (event: CnfGenerationEvent) => void;
  firstClause: boolean;
  review: CnfReview | null;
}): Promise<void> {
  let nextPrompt = firstClause
    ? [
        "We are converting this problem into CNF:",
        description,
        "",
        "Use these variables exactly; do not invent other variable names:",
        variables,
        "",
        "Now go through the problem statement, and make the first CNF clause.",
        "Return one clause only. Put `or` between alternatives. Do not use `and` in the `cnf` value.",
        `Output JSON only in this exact format: ${CLAUSE_SCHEMA}.`,
      ].join("\n")
    : [
        "Go through the problem statement, and make the next CNF clause.",
        "Return one clause only. Put `or` between alternatives. Do not use `and` in the `cnf` value.",
        `Output JSON only in this exact format, or {} if you have addressed everything: ${CLAUSE_SCHEMA}.`
      ].join("\n");

  if (review && review.status !== "complete") {
    nextPrompt = [
      "The review found missing or wrong CSP coverage.",
      `Missing constraints: ${JSON.stringify(review.missingConstraints)}`,
      `Wrong clauses: ${JSON.stringify(review.wrongClauses)}`,
      "Continue from the corrected clause list and make the next needed CNF clause.",
      "Return one clause only. Put `or` between alternatives. Do not use `and` in the `cnf` value.",
      `Output JSON only in this exact format, or {} if everything is now addressed: ${CLAUSE_SCHEMA}.`
    ].join("\n");
  }

  while (clauses.length < maxClauses) {
    const response = await promptWithHistory(
      session,
      history,
      nextPrompt,
      onProgress
    );
    const parsed = parseCnfJsonResponse(response);
    appendAssistantJson(history, parsed.correctedJson, onProgress);

    if (!parsed.step) {
      return;
    }

    const step = await validateStep(session, history, parsed.step, clauses, onProgress);
    addClause(clauses, step);
    onProgress?.({
      type: "clause",
      step,
      source: stepsToSource(clauses)
    });

    nextPrompt = [
      "Go through the problem statement, and make the next CNF clause.",
      "Return one clause only. Put `or` between alternatives. Do not use `and` in the `cnf` value.",
      `Output JSON only in this exact format, or {} if you have addressed everything: ${CLAUSE_SCHEMA}.`
    ].join("\n");
  }

  throw new Error(`The browser model produced more than ${maxClauses} clauses without finishing.`);
}

async function validateStep(
  session: LanguageModelSession,
  history: LanguageModelPrompt[],
  step: CnfGenerationStep,
  acceptedClauses: CnfGenerationStep[],
  onProgress?: (event: CnfGenerationEvent) => void
): Promise<CnfGenerationStep> {
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      validateGeneratedStep(step, acceptedClauses);
      return step;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const response = await promptWithHistory(
        session,
        history,
        [
          "The previous JSON was parsed, but the `cnf` value is not an acceptable single CNF clause.",
          `Compiler or validation error: ${message}`,
          "If you wrote `x and not y`, that is not a CNF clause. Use an OR clause such as `not x or not y`, `not x or y`, or another clause matching the original rule.",
          "Return exactly one corrected CNF OR-clause. Put `or` between alternatives. Do not use `and` in the `cnf` value.",
          `Return corrected JSON only for the same purpose in this exact format: ${CLAUSE_SCHEMA}.`
        ].join("\n"),
        onProgress
      );
      const parsed = parseCnfJsonResponse(response);
      appendAssistantJson(history, parsed.correctedJson, onProgress);
      if (!parsed.step) {
        throw new Error("The browser model returned {} while correcting an invalid clause.");
      }
      step = parsed.step;
    }
  }

  validateGeneratedStep(step, acceptedClauses);
  return step;
}

async function reviewClauses(
  session: LanguageModelSession,
  history: LanguageModelPrompt[],
  description: string,
  clauses: CnfGenerationStep[],
  onProgress?: (event: CnfGenerationEvent) => void
): Promise<CnfReview> {
  const response = await promptWithHistory(
    session,
    history,
    [
      "Review whether these CSP clauses completely capture the problem.",
      "Report clauses that are wrong and any problem sentences or constraints that are missing CSP clauses.",
      "If clauses should be added immediately, include them in corrected_clauses.",
      "Every corrected clause must be one CNF OR-clause only; do not use `and` in a corrected clause.",
      "Output JSON only in this exact shape:",
      '{"status":"complete","wrong_clauses":[],"missing_constraints":[],"corrected_clauses":[]}',
      "--- Problem ---",
      description,
      "--- Current CSP clauses ---",
      JSON.stringify(clauses)
    ].join("\n"),
    onProgress
  );
  const parsed = parseReviewResponse(response);
  appendAssistantJson(history, parsed.correctedJson, onProgress);
  return parsed.review;
}

async function promptWithHistory(
  session: LanguageModelSession,
  history: LanguageModelPrompt[],
  userContent: string,
  onProgress?: (event: CnfGenerationEvent) => void,
  signal?: AbortSignal
): Promise<string> {
  throwIfAborted(signal);
  emitConversation(onProgress, "user", userContent);
  const userPrompt: LanguageModelPrompt = { role: "user", content: userContent };
  const response = await session.prompt([...history, userPrompt], { signal });
  throwIfAborted(signal);
  history.push(userPrompt);
  emitConversation(onProgress, "assistant", response);
  return response;
}

function appendAssistantJson(
  history: LanguageModelPrompt[],
  content: string,
  onProgress?: (event: CnfGenerationEvent) => void
): void {
  history.push({ role: "assistant", content });
  emitConversation(onProgress, "assistant-recorded", content);
}

export function parseCnfJsonResponse(response: string): {
  step: CnfGenerationStep | null;
  correctedJson: string;
} {
  const value = parseJsonObject(response);
  if (Object.keys(value).length === 0) {
    return { step: null, correctedJson: "{}" };
  }

  const cnf = typeof value.cnf === "string" ? normalizeCnf(value.cnf) : "";
  const purpose = typeof value.purpose === "string" ? value.purpose.trim() : "";
  if (!cnf || !purpose) {
    throw new Error("Expected JSON with string `cnf` and `purpose` fields.");
  }

  const step = { cnf, purpose };
  return {
    step,
    correctedJson: JSON.stringify(step)
  };
}

function parseReviewResponse(response: string): {
  review: CnfReview;
  correctedJson: string;
} {
  const value = parseJsonObject(response);
  const correctedClauses = Array.isArray(value.corrected_clauses)
    ? value.corrected_clauses.flatMap((candidate) => {
        if (
          typeof candidate === "object" &&
          candidate != null &&
          "cnf" in candidate &&
          "purpose" in candidate &&
          typeof candidate.cnf === "string" &&
          typeof candidate.purpose === "string"
        ) {
          return [
            {
              cnf: normalizeCnf(candidate.cnf),
              purpose: candidate.purpose.trim()
            }
          ];
        }
        return [];
      })
    : [];

  const review: CnfReview = {
    status: value.status === "complete" ? "complete" : "needs_changes",
    wrongClauses: stringArray(value.wrong_clauses),
    missingConstraints: stringArray(value.missing_constraints),
    correctedClauses
  };

  return {
    review,
    correctedJson: JSON.stringify({
      status: review.status,
      wrong_clauses: review.wrongClauses,
      missing_constraints: review.missingConstraints,
      corrected_clauses: review.correctedClauses
    })
  };
}

function applyReviewChanges(clauses: CnfGenerationStep[], review: CnfReview): void {
  if (review.wrongClauses.length) {
    const wrong = review.wrongClauses.map((entry) => entry.toLowerCase());
    for (let index = clauses.length - 1; index >= 0; index -= 1) {
      const cnf = clauses[index].cnf.toLowerCase();
      const purpose = clauses[index].purpose.toLowerCase();
      if (wrong.some((entry) => cnf === entry || purpose.includes(entry))) {
        clauses.splice(index, 1);
      }
    }
  }

  for (const clause of review.correctedClauses) {
    validateGeneratedStep(clause, clauses);
    addClause(clauses, clause);
  }
}

function addClause(clauses: CnfGenerationStep[], step: CnfGenerationStep): void {
  const normalized = normalizeForDedup(step.cnf);
  if (!clauses.some((clause) => normalizeForDedup(clause.cnf) === normalized)) {
    clauses.push(step);
  }
}

function validateGeneratedStep(
  step: CnfGenerationStep,
  acceptedClauses: CnfGenerationStep[] = []
): CompiledProblem {
  const textualContradiction = findTextualAndContradiction(step.cnf);
  if (textualContradiction) {
    throw new Error(
      `The expression requires ${textualContradiction} to be true and false at the same time. That is always false, not a CNF clause for the original rule.`
    );
  }

  if (isBundledNeitherNorProhibition(step)) {
    throw new Error(
      "A `neither X nor Y can ...` prohibition must be returned as separate unit clauses. Return one unit clause now, such as `not Ben_documentation`, and return the other on the next prompt."
    );
  }

  if (/\band\b/iu.test(step.cnf)) {
    throw new Error(
      "The `cnf` field contains `and`. Return one CNF clause at a time with `or` between alternatives."
    );
  }

  const compiled = compileProblem(
    acceptedClauses.length
      ? stepsToSource([...acceptedClauses, step])
      : step.cnf
  );
  const contradiction = findUnitContradiction(compiled);
  if (contradiction) {
    throw new Error(`The clause set contains both ${contradiction} and not ${contradiction}.`);
  }
  return compiled;
}

function isBundledNeitherNorProhibition(step: CnfGenerationStep): boolean {
  return (
    /\bneither\b/iu.test(step.purpose) &&
    /\bnor\b/iu.test(step.purpose) &&
    /\b(can|cannot|can't)\b/iu.test(step.purpose) &&
    /\bor\b/iu.test(step.cnf)
  );
}

function findTextualAndContradiction(cnf: string): string | null {
  const normalized = cnf.toLowerCase().replace(/[()]/gu, " ");
  const parts = normalized
    .split(/\band\b/iu)
    .map((part) => part.trim().replace(/\s+/gu, " "))
    .filter(Boolean);
  const positives = new Set<string>();
  const negatives = new Set<string>();

  for (const part of parts) {
    const negative = /^not\s+([a-z_][a-z0-9_]*)$/iu.exec(part);
    if (negative) {
      negatives.add(negative[1]);
      continue;
    }
    if (/^[a-z_][a-z0-9_]*$/iu.test(part)) {
      positives.add(part);
    }
  }

  for (const name of positives) {
    if (negatives.has(name)) {
      return name;
    }
  }

  return null;
}

function findUnitContradiction(compiled: CompiledProblem): string | null {
  const unitPolarity = new Map<string, Set<boolean>>();

  for (const clause of compiled.ternaryClauses) {
    if (clause.terms.length !== 1) {
      continue;
    }
    const term = clause.terms[0];
    const unit =
      term.type === "literal"
        ? { name: term.name, positive: true }
        : term.type === "not" && term.expr.type === "literal"
          ? { name: term.expr.name, positive: false }
          : null;
    if (!unit) {
      continue;
    }

    const polarities = unitPolarity.get(unit.name) ?? new Set<boolean>();
    polarities.add(unit.positive);
    unitPolarity.set(unit.name, polarities);
    if (polarities.size > 1) {
      return unit.name;
    }
  }

  return null;
}

function stepsToSource(clauses: CnfGenerationStep[]): string {
  return clauses.map((clause) => clause.cnf).join("\n");
}

function parseJsonObject(response: string): Record<string, unknown> {
  const objectText = extractFirstJsonObject(response);
  if (!objectText) {
    throw new Error("The browser model did not return a JSON object.");
  }

  try {
    return JSON.parse(objectText) as Record<string, unknown>;
  } catch {
    return JSON.parse(normalizeJsonishObject(objectText)) as Record<string, unknown>;
  }
}

function extractFirstJsonObject(response: string): string | null {
  const cleaned = response.replace(/```(?:json)?/giu, "").replace(/```/gu, "");
  const start = cleaned.indexOf("{");
  if (start < 0) {
    return null;
  }

  let depth = 0;
  let inString = false;
  let quote = "";
  let escaped = false;

  for (let index = start; index < cleaned.length; index += 1) {
    const char = cleaned[index];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === quote) {
        inString = false;
      }
      continue;
    }

    if (char === '"' || char === "'") {
      inString = true;
      quote = char;
      continue;
    }

    if (char === "{") {
      depth += 1;
    } else if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        return cleaned.slice(start, index + 1);
      }
    }
  }

  return null;
}

function normalizeJsonishObject(value: string): string {
  return value
    .replace(/([{,]\s*)'([^']+)'\s*:/gu, '$1"$2":')
    .replace(/:\s*'([^'\\]*(?:\\.[^'\\]*)*)'/gu, (_match, content: string) => {
      return `:${JSON.stringify(content.replace(/\\'/gu, "'"))}`;
    });
}

function normalizeCnf(cnf: string): string {
  return cnf
    .replace(/```(?:json)?/giu, "")
    .replace(/```/gu, "")
    .replace(/[¬~]/gu, " not ")
    .replace(/[∨|]+/gu, " or ")
    .replace(/[∧&]+/gu, " and ")
    .replace(/\s+/gu, " ")
    .trim()
    .replace(/^\((.*)\)$/u, "$1");
}

function normalizeForDedup(cnf: string): string {
  return cnf.toLowerCase().replace(/\s+/gu, " ").trim();
}

function normalizeDescription(description: string): string {
  return description.replace(/\s+/gu, " ").trim();
}

function throwIfAborted(signal?: AbortSignal): void {
  if (!signal?.aborted) {
    return;
  }

  throw new DOMException("CSP generation cancelled.", "AbortError");
}

async function waitForIntermediateStage(
  durationMs: number,
  signal?: AbortSignal
): Promise<void> {
  throwIfAborted(signal);

  await new Promise<void>((resolve, reject) => {
    const handleAbort = () => {
      clearTimeout(timeoutId);
      reject(new DOMException("CSP generation cancelled.", "AbortError"));
    };
    const timeoutId = globalThis.setTimeout(() => {
      signal?.removeEventListener("abort", handleAbort);
      resolve();
    }, durationMs);

    signal?.addEventListener("abort", handleAbort, { once: true });
  });
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((entry): entry is string => typeof entry === "string")
    : [];
}

function emitConversation(
  onProgress: ((event: CnfGenerationEvent) => void) | undefined,
  role: CnfConversationEntry["role"],
  content: string
): void {
  onProgress?.({
    type: "conversation",
    entry: { role, content }
  });
}

function getLanguageModelFactory(): LanguageModelFactory | undefined {
  return globalThis.LanguageModel ?? globalThis.languageModel;
}
