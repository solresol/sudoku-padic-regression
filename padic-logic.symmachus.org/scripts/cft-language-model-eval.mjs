#!/usr/bin/env node

import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const chromePath =
  process.env.CHROME_FOR_TESTING_PATH ??
  "/Applications/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing";
const port = Number(process.env.CFT_DEBUG_PORT ?? 9223);
const profile = process.env.CFT_PROFILE_DIR ?? "/tmp/padic-logic-cft-profile";
const targetUrl =
  process.env.LANGUAGE_MODEL_TEST_URL ?? "https://padic-logic.symmachus.org/";
const waitMs = Number(process.env.LANGUAGE_MODEL_WAIT_MS ?? 1_800_000);
const apiWaitMs = Number(process.env.LANGUAGE_MODEL_API_WAIT_MS ?? 30_000);
const pollMs = Number(process.env.LANGUAGE_MODEL_POLL_MS ?? 5_000);
const progressLogMs = Number(process.env.LANGUAGE_MODEL_PROGRESS_LOG_MS ?? 30_000);
const outPath =
  process.env.LANGUAGE_MODEL_RESULT_JSON ??
  resolve(
    dirname(fileURLToPath(import.meta.url)),
    "../tmp/language-model-cft-result.json"
  );

const problems = [
  {
    id: "default-assignment",
    description: [
      "Ava, Ben, Cara, and Devina need to be assigned test, design, documentation and development. Everyone needs exactly one job. Every job needs exactly one person assigned to it.",
      "Rules: Ava cannot test. Neither Ben nor Cara can do  documentation. If Devina documents, then Ava designs."
    ].join("\n"),
    mustContain: [
      "not Ava_test",
      "not Ben_documentation",
      "not Cara_documentation",
      "not Devina_documentation or Ava_design"
    ],
    mustNotContain: ["and not"],
    expectedLineCount: 60
  },
  {
    id: "tiny-assignment",
    description: [
      "Ava and Ben need to be assigned test and design. Everyone needs exactly one job. Every job needs exactly one person assigned to it.",
      "Rule: Ava cannot test."
    ].join("\n"),
    mustContain: [
      "Ava_test or Ava_design",
      "Ben_test or Ben_design",
      "Ava_test or Ben_test",
      "Ava_design or Ben_design",
      "not Ava_test"
    ],
    mustNotContain: ["and not"]
  },
  {
    id: "single-implication",
    description: "If Devina documents, then Ava designs.",
    mustContain: ["not Devina_documentation or Ava_design"],
    mustNotContain: ["Devina_documentation and not Ava_design"]
  }
];

async function main() {
  const chrome = spawn(chromePath, [
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${profile}`,
    "--no-first-run",
    "--no-default-browser-check",
    targetUrl
  ], {
    stdio: ["ignore", "pipe", "pipe"]
  });

  let stderr = "";
  chrome.stderr.on("data", (chunk) => {
    stderr += chunk.toString();
  });

  try {
    await waitForCdp(port, 30_000);
    const tab = await getPageTab(port);
    const client = await CdpClient.connect(tab.webSocketDebuggerUrl);
    await client.send("Runtime.enable");
    await client.send("Page.enable");
    await client.send("Page.navigate", { url: targetUrl });
    await client.wait(2_000);

    const availability = await waitForLanguageModelApi(client, Math.min(waitMs, apiWaitMs));
    if (availability === "missing" || availability === "unavailable") {
      await writeResult({
        status: "blocked",
        reason: `LanguageModel.availability() returned ${availability}.`,
        targetUrl,
        chromePath,
        profile,
        stderr: stderr.slice(-4_000)
      });
      process.exitCode = 2;
    } else {
      const results = [];
      for (const problem of problems) {
        console.log(`${new Date().toISOString()} Running ${problem.id}`);
        const result = await runProblem(client, problem, waitMs);
        results.push(result);
        if (result.blocked) {
          break;
        }
      }

      const failed = results.filter((result) => !result.pass);
      const blocked = results.some((result) => result.blocked);
      await writeResult({
        status: blocked ? "blocked" : failed.length ? "failed" : "passed",
        targetUrl,
        chromePath,
        profile,
        initialAvailability: availability,
        results
      });
      process.exitCode = blocked ? 2 : failed.length ? 1 : 0;
    }
  } finally {
    chrome.kill("SIGTERM");
  }
}

async function runProblem(client, problem, timeoutMs) {
  await client.send("Page.navigate", { url: `${targetUrl}?case=${problem.id}` });
  await client.wait(1_000);
  const setup = await client.evaluate(
    async ({ description }) => {
      const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
      const deadline = Date.now() + 10_000;
      let descriptionEl = null;
      while (Date.now() < deadline) {
        descriptionEl = document.querySelector(
          'textarea[aria-label="Natural language problem"]'
        );
        if (descriptionEl) {
          break;
        }
        await sleep(250);
      }

      if (!descriptionEl) {
        return {
          ready: false,
          error: "Natural language problem textarea not found.",
          hasLanguageModel: Boolean(window.LanguageModel ?? window.languageModel)
        };
      }

      const languageModel = window.LanguageModel ?? window.languageModel;
      const availability = await languageModel.availability({
        expectedInputs: [{ type: "text", languages: ["en"] }],
        expectedOutputs: [{ type: "text", languages: ["en"] }]
      });
      const setter = Object.getOwnPropertyDescriptor(
        HTMLTextAreaElement.prototype,
        "value"
      )?.set;
      if (setter) {
        setter.call(descriptionEl, description);
      } else {
        descriptionEl.value = description;
      }
      descriptionEl.dispatchEvent(new Event("input", { bubbles: true }));

      const button = Array.from(document.querySelectorAll("button")).find((candidate) =>
        /Generate CSP/.test(candidate.textContent ?? "")
      );
      if (!button) {
        return { ready: false, error: "Generate CSP button not found.", availability };
      }

      button.scrollIntoView({ block: "center", inline: "center" });
      const rect = button.getBoundingClientRect();
      return {
        ready: true,
        availability,
        buttonCenter: {
          x: rect.left + rect.width / 2,
          y: rect.top + rect.height / 2
        }
      };
    },
    { description: problem.description }
  );

  if (!setup.ready) {
    return { id: problem.id, pass: false, ...setup };
  }

  await clickAt(client, setup.buttonCenter);

  const startedAt = Date.now();
  const deadline = startedAt + timeoutMs;
  let lastProgressLog = 0;
  let state = await readGenerationState(client, problem);
  while (Date.now() < deadline) {
    if (state.done) {
      return {
        id: problem.id,
        initialAvailability: setup.availability,
        elapsedMs: Date.now() - startedAt,
        ...state.result
      };
    }

    if (Date.now() - lastProgressLog >= progressLogMs) {
      const lastTranscript = state.transcript.at(-1);
      const lastTranscriptText = lastTranscript
        ? `${lastTranscript.role}: ${oneLine(lastTranscript.content).slice(0, 160)}`
        : "no transcript yet";
      console.log(
        [
          new Date().toISOString(),
          `Waiting for ${problem.id}`,
          `elapsed=${Math.round((Date.now() - startedAt) / 1000)}s`,
          `generating=${state.generating}`,
          `sourceLines=${state.sourceLines}`,
          lastTranscriptText
        ].join(" ")
      );
      lastProgressLog = Date.now();
    }

    await client.wait(pollMs);
    state = await readGenerationState(client, problem);
  }

  return {
    id: problem.id,
    initialAvailability: setup.availability,
    elapsedMs: Date.now() - startedAt,
    pass: false,
    blocked: !state.source.trim() && !state.errorText,
    error: `Timed out after ${timeoutMs}ms waiting for generation.`,
    source: state.source,
    missing: problem.mustContain.filter((clause) => !state.source.includes(clause)),
    forbidden: problem.mustNotContain.filter((clause) => state.source.includes(clause)),
    wrongLineCount: state.wrongLineCount,
    errorText: state.errorText,
    generating: state.generating,
    transcript: state.transcript
  };
}

async function readGenerationState(client, problem) {
  const state = await client.evaluate(
    ({ mustContain, mustNotContain, expectedLineCount }) => {
      const transcript = Array.from(
        document.querySelectorAll(".lm-message")
      ).map((entry) => ({
        role: entry.querySelector("span")?.textContent ?? "",
        content: entry.querySelector("pre")?.textContent ?? ""
      }));
      const errorText =
        document.querySelector(".error-box")?.textContent?.trim() ?? "";
      const progressText =
        document.querySelector(".generation-progress")?.textContent?.trim() ?? "";
      const source =
        document.querySelector('textarea[aria-label="CSP source"]')?.value ?? "";
      const generating = Array.from(document.querySelectorAll("button")).some(
        (candidate) => /Generating/.test(candidate.textContent ?? "")
      );
      const lines = source.trim() ? source.trim().split(/\r?\n/u) : [];
      const missing = mustContain.filter((clause) => !source.includes(clause));
      const forbidden = mustNotContain.filter((clause) => source.includes(clause));
      const wrongLineCount =
        typeof expectedLineCount === "number" && lines.length !== expectedLineCount
          ? { expected: expectedLineCount, actual: lines.length }
          : null;

      return {
        done: !generating && (Boolean(source.trim()) || Boolean(errorText)),
        generating,
        source,
        sourceLines: lines.length,
        missing,
        forbidden,
        wrongLineCount,
        errorText,
        progressText,
        transcript,
        pass: !errorText && missing.length === 0 && forbidden.length === 0 && !wrongLineCount
      };
    },
    {
      mustContain: problem.mustContain,
      mustNotContain: problem.mustNotContain,
      expectedLineCount: problem.expectedLineCount
    }
  );

  return {
    ...state,
    result: {
      pass: state.pass,
      source: state.source,
      missing: state.missing,
      forbidden: state.forbidden,
      wrongLineCount: state.wrongLineCount,
      errorText: state.errorText,
      generating: state.generating,
      transcript: state.transcript
    }
  };
}

function oneLine(value) {
  return value.replace(/\s+/gu, " ").trim();
}

async function clickAt(client, { x, y }) {
  await client.send("Input.dispatchMouseEvent", {
    type: "mouseMoved",
    x,
    y,
    button: "none"
  });
  await client.send("Input.dispatchMouseEvent", {
    type: "mousePressed",
    x,
    y,
    button: "left",
    clickCount: 1
  });
  await client.send("Input.dispatchMouseEvent", {
    type: "mouseReleased",
    x,
    y,
    button: "left",
    clickCount: 1
  });
}

async function waitForLanguageModelApi(client, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let availability = "missing";
  while (Date.now() < deadline) {
    availability = await client.evaluate(async () => {
      const languageModel = window.LanguageModel ?? window.languageModel;
      if (!languageModel) {
        return "missing";
      }
      return languageModel.availability({
        expectedInputs: [{ type: "text", languages: ["en"] }],
        expectedOutputs: [{ type: "text", languages: ["en"] }]
      });
    });
    console.log(`${new Date().toISOString()} LanguageModel=${availability}`);
    if (availability !== "missing") {
      return availability;
    }
    await client.wait(5_000);
  }
  return availability;
}

async function waitForCdp(debugPort, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      await fetch(`http://127.0.0.1:${debugPort}/json/version`).then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
      });
      return;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  }
  throw new Error(`Chrome for Testing did not open CDP port ${debugPort}.`);
}

async function getPageTab(debugPort) {
  const tabs = await fetch(`http://127.0.0.1:${debugPort}/json/list`).then((response) =>
    response.json()
  );
  const tab = tabs.find((candidate) => candidate.type === "page") ?? tabs[0];
  if (!tab?.webSocketDebuggerUrl) {
    throw new Error("No debuggable Chrome page found.");
  }
  return tab;
}

async function writeResult(result) {
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, `${JSON.stringify(result, null, 2)}\n`);
  console.log(`Wrote ${outPath}`);
}

class CdpClient {
  constructor(socket) {
    this.socket = socket;
    this.nextId = 1;
    this.pending = new Map();
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.id && this.pending.has(message.id)) {
        this.pending.get(message.id)(message);
        this.pending.delete(message.id);
      }
    };
  }

  static async connect(url) {
    const socket = new WebSocket(url);
    await new Promise((resolve, reject) => {
      socket.onopen = resolve;
      socket.onerror = reject;
    });
    return new CdpClient(socket);
  }

  send(method, params = {}) {
    const id = this.nextId;
    this.nextId += 1;
    this.socket.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve) => this.pending.set(id, resolve));
  }

  async evaluate(expressionOrFunction, arg) {
    const expression =
      typeof expressionOrFunction === "function"
        ? `(${expressionOrFunction.toString()})(${JSON.stringify(arg)})`
        : expressionOrFunction;
    const result = await this.send("Runtime.evaluate", {
      expression,
      returnByValue: true,
      awaitPromise: true
    });
    if (result.result.exceptionDetails) {
      throw new Error(
        result.result.exceptionDetails.exception?.description ??
          result.result.exceptionDetails.text
      );
    }
    return result.result.result.value;
  }

  wait(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

await main();
