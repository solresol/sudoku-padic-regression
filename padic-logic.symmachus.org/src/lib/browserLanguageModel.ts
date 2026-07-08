import { compileProblem } from "./csp";

export type LanguageModelAvailability =
  | "unavailable"
  | "downloadable"
  | "downloading"
  | "available";

const LANGUAGE_OPTIONS: LanguageModelCreateOptions = {
  expectedInputs: [{ type: "text", languages: ["en"] }],
  expectedOutputs: [{ type: "text", languages: ["en"] }]
};

const FALLBACK_CSP = [
  "A or B or C",
  "B or C or not D",
  "not A or D",
  "A or not C or D"
].join("\n");

export async function detectLanguageModel(): Promise<LanguageModelAvailability> {
  if (!("LanguageModel" in globalThis) || !globalThis.LanguageModel) {
    return "unavailable";
  }

  return globalThis.LanguageModel.availability(LANGUAGE_OPTIONS);
}

export async function generateCspFromDescription(description: string): Promise<{
  source: string;
  usedLocalModel: boolean;
}> {
  if (!description.trim()) {
    return { source: FALLBACK_CSP, usedLocalModel: false };
  }

  if (!("LanguageModel" in globalThis) || !globalThis.LanguageModel) {
    return { source: heuristicCsp(description), usedLocalModel: false };
  }

  const availability = await globalThis.LanguageModel.availability(LANGUAGE_OPTIONS);
  if (availability === "unavailable") {
    return { source: heuristicCsp(description), usedLocalModel: false };
  }

  const session = await globalThis.LanguageModel.create({
    ...LANGUAGE_OPTIONS,
    initialPrompts: [
      {
        role: "system",
        content: [
          "Convert a natural-language constraint problem to one CSP line per constraint.",
          "Use only boolean identifiers, not, v/or within clauses, and ^/and between clauses.",
          "Do not use xor; if parity is needed, expand it into ordinary CNF clauses.",
          "Return only the CSP lines. Do not explain."
        ].join(" ")
      }
    ]
  });

  try {
    const response = await session.prompt(description);
    const source = cleanModelResponse(response);
    compileProblem(source);
    return { source, usedLocalModel: true };
  } catch {
    return { source: heuristicCsp(description), usedLocalModel: false };
  } finally {
    session.destroy?.();
  }
}

function cleanModelResponse(response: string): string {
  return response
    .replace(/```[a-z]*\n?/giu, "")
    .replace(/```/gu, "")
    .split(/\r?\n/u)
    .map((line) => line.trim().replace(/^\d+\.\s*/u, ""))
    .filter((line) => line && !line.startsWith("#"))
    .join("\n");
}

function heuristicCsp(description: string): string {
  const words = description.match(/[A-Z][A-Za-z0-9_]*/gu) ?? [];
  const names = Array.from(new Set(words)).slice(0, 4);
  const [a = "A", b = "B", c = "C", d = "D"] = names;

  return [
    `${a} or ${b} or ${c}`,
    `${b} or ${c} or not ${d}`,
    `not ${a} or ${d}`,
    `${a} or not ${c} or ${d}`
  ].join("\n");
}
