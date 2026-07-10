import { afterEach, describe, expect, it, vi } from "vitest";
import {
  detectLanguageModel,
  generateCspFromDescription,
  getCachedCnfGeneration,
  isCnfGenerationCacheEnabled,
  parseCnfJsonResponse
} from "./browserLanguageModel";
import { DEFAULT_ASSIGNMENT_PROBLEM } from "./defaultProblems";

function clearLanguageModel(): void {
  globalThis.LanguageModel = undefined;
  globalThis.languageModel = undefined;
}

function installLanguageModel(responses: string[]) {
  const pending = [...responses];
  const sessions: LanguageModelSession[] = [];
  const prompt = vi.fn(async () => {
    const response = pending.shift();
    if (response == null) {
      throw new Error("No mock languageModel response left.");
    }
    return response;
  });
  const factory: LanguageModelFactory = {
    availability: vi.fn(async () => "available"),
    create: vi.fn(async () => {
      const session: LanguageModelSession = {
        prompt,
        destroy: vi.fn()
      };
      sessions.push(session);
      return session;
    })
  };

  globalThis.languageModel = factory;
  return { factory, prompt, sessions };
}

describe("browser languageModel CSP generation", () => {
  afterEach(() => {
    clearLanguageModel();
    vi.restoreAllMocks();
  });

  it("detects the lowercase window.languageModel API", async () => {
    const { factory } = installLanguageModel(["unused"]);

    await expect(detectLanguageModel()).resolves.toBe("available");
    expect(factory.availability).toHaveBeenCalled();
  });

  it("extracts and normalizes JSON-only clause responses from noisy model output", () => {
    const parsed = parseCnfJsonResponse(
      "```json\n{'cnf':'¬Ava_test','purpose':'Ava cannot test.'}\n```\nExplanation."
    );

    expect(parsed.step).toEqual({
      cnf: "not Ava_test",
      purpose: "Ava cannot test."
    });
    expect(parsed.correctedJson).toBe(
      JSON.stringify({
        cnf: "not Ava_test",
        purpose: "Ava cannot test."
      })
    );
  });

  it("keeps the known default CNF cached but disabled by default", async () => {
    const cached = getCachedCnfGeneration(DEFAULT_ASSIGNMENT_PROBLEM);

    expect(isCnfGenerationCacheEnabled()).toBe(false);
    expect(cached?.clauses.length).toBe(60);
    expect(cached?.clauses.some((clause) => clause.cnf === "not Ava_test")).toBe(true);
    expect(cached?.clauses.some((clause) => clause.cnf.includes("development"))).toBe(true);
    expect(
      cached?.clauses.some(
        (clause) =>
          clause.purpose === "Ava cannot do documentation and development at the same time."
      )
    ).toBe(false);

    const events: string[] = [];
    const result = await generateCspFromDescription(DEFAULT_ASSIGNMENT_PROBLEM, {
      useCache: true,
      onProgress: (event) => {
        if (event.type === "clause") {
          events.push(event.step.cnf);
        }
      }
    });

    expect(result.usedLocalModel).toBe(false);
    expect(result.source).toContain("not Devina_documentation or Ava_design");
    expect(events).toHaveLength(60);
  });

  it("extracts an assignment schema and expands it deterministically", async () => {
    const events: string[] = [];
    const conversationRoles: string[] = [];
    const eventOrder: string[] = [];
    const variableSets: string[][] = [];
    const { factory, prompt, sessions } = installLanguageModel([
      JSON.stringify({
        kind: "assignment",
        people: ["Ava", "Ben"],
        jobs: ["test", "design"],
        person_exactly_one_job: true,
        job_exactly_one_person: true,
        forbidden: [{ person: "Ava", job: "test", source: "Ava cannot test." }],
        implications: []
      })
    ]);

    const result = await generateCspFromDescription(
      "Ava and Ben need exactly one of test and design. Every job needs exactly one person. Ava cannot test.",
      {
      onProgress: (event) => {
        if (event.type === "conversation") {
          conversationRoles.push(event.entry.role);
          eventOrder.push(`conversation:${event.entry.role}`);
        } else {
          eventOrder.push(event.type);
        }
        if (event.type === "variables") {
          variableSets.push(event.variables);
        }
        if (event.type === "clause") {
          events.push(event.step.purpose);
        }
      }
      }
    );

    expect(result.usedLocalModel).toBe(true);
    expect(result.source).toBe(
      [
        "Ava_test or Ava_design",
        "not Ava_test or not Ava_design",
        "Ben_test or Ben_design",
        "not Ben_test or not Ben_design",
        "Ava_test or Ben_test",
        "not Ava_test or not Ben_test",
        "Ava_design or Ben_design",
        "not Ava_design or not Ben_design",
        "not Ava_test"
      ].join("\n")
    );
    expect(events).toContain("Ava cannot test.");
    expect(conversationRoles).toContain("assistant");
    expect(conversationRoles).toContain("assistant-recorded");
    expect(conversationRoles).toContain("browser");
    expect(conversationRoles).not.toContain("system");
    expect(variableSets).toEqual([
      ["Ava_test", "Ava_design", "Ben_test", "Ben_design"]
    ]);
    expect(eventOrder.indexOf("conversation:assistant-recorded"))
      .toBeLessThan(eventOrder.indexOf("variables"));
    expect(eventOrder.indexOf("variables")).toBeLessThan(eventOrder.indexOf("clause"));
    expect(eventOrder.indexOf("clause")).toBeLessThan(eventOrder.indexOf("review"));

    expect(factory.create).toHaveBeenCalledTimes(1);
    expect(typeof vi.mocked(factory.create).mock.calls[0][0]?.monitor).toBe("function");
    const calls = prompt.mock.calls;
    expect(calls).toHaveLength(1);
    expect(calls[0][0]).toContainEqual(
      expect.objectContaining({
        role: "user",
        content: expect.stringContaining("Extract a typed JSON schema")
      })
    );
    expect(sessions.every((session) => vi.mocked(session.destroy).mock.calls.length > 0)).toBe(true);
  });

  it("expands the default assignment schema to the expected 60 clauses", async () => {
    installLanguageModel([
      JSON.stringify({
        kind: "assignment",
        people: ["Ava", "Ben", "Cara", "Devina"],
        jobs: ["test", "design", "documentation", "development"],
        person_exactly_one_job: true,
        job_exactly_one_person: true,
        forbidden: [
          { person: "Ava", job: "test", source: "Ava cannot test." },
          { person: "Ben", job: "documentation", source: "Ben cannot do documentation." },
          { person: "Cara", job: "documentation", source: "Cara cannot do documentation." }
        ],
        implications: [
          {
            if: { person: "Devina", job: "documentation" },
            then: { person: "Ava", job: "design" },
            source: "If Devina documents, then Ava designs."
          }
        ]
      })
    ]);

    const result = await generateCspFromDescription(DEFAULT_ASSIGNMENT_PROBLEM);

    expect(result.clauses).toHaveLength(60);
    expect(result.source).toContain("not Ava_test");
    expect(result.source).toContain("not Ben_documentation");
    expect(result.source).toContain("not Cara_documentation");
    expect(result.source).toContain("not Devina_documentation or Ava_design");
  });

  it("represents neither/nor prohibitions as separate unit clauses through the schema", async () => {
    installLanguageModel([
      JSON.stringify({
        kind: "assignment",
        people: ["Ben", "Cara"],
        jobs: ["documentation"],
        person_exactly_one_job: false,
        job_exactly_one_person: false,
        forbidden: [
          { person: "Ben", job: "documentation", source: "Ben cannot do documentation." },
          { person: "Cara", job: "documentation", source: "Cara cannot do documentation." }
        ],
        implications: []
      })
    ]);

    const result = await generateCspFromDescription("Neither Ben nor Cara can do documentation.");

    expect(result.source).toBe("not Ben_documentation\nnot Cara_documentation");
  });

  it("fails when the extracted schema creates contradictory unit clauses", async () => {
    installLanguageModel([
      JSON.stringify({
        kind: "assignment",
        people: ["Ava"],
        jobs: ["test"],
        person_exactly_one_job: true,
        job_exactly_one_person: false,
        forbidden: [{ person: "Ava", job: "test", source: "Ava cannot test." }],
        implications: []
      })
    ]);

    await expect(generateCspFromDescription("Ava has exactly one test job. Ava cannot test."))
      .rejects.toThrow("both Ava_test and not Ava_test");
  });
});
