import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
  type RenderResult
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { DEFAULT_ASSIGNMENT_PROBLEM } from "./lib/defaultProblems";

class MockWorker {
  static instances: MockWorker[] = [];

  onmessage: ((event: MessageEvent) => void) | null = null;
  postMessage = vi.fn();
  terminate = vi.fn();

  constructor() {
    MockWorker.instances.push(this);
  }
}

function clearLanguageModel(): void {
  globalThis.LanguageModel = undefined;
  globalThis.languageModel = undefined;
}

async function renderApp(): Promise<RenderResult> {
  let result: RenderResult | undefined;
  await act(async () => {
    result = render(<App />);
  });
  return result as RenderResult;
}

describe("p-adic logic app", () => {
  beforeEach(() => {
    clearLanguageModel();
    MockWorker.instances = [];
    globalThis.Worker = MockWorker as unknown as typeof Worker;
    window.history.replaceState(null, "", "/");
  });

  afterEach(() => {
    clearLanguageModel();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("puts the mode buttons at the top and removes the old workflow controls", async () => {
    await renderApp();

    const header = screen.getByRole("banner");
    expect(header).toContainElement(screen.getByRole("tab", { name: /Boolean CSP\/SAT/i }));
    expect(header).toContainElement(screen.getByRole("tab", { name: /^Sudoku/i }));
    expect(screen.queryByRole("link", { name: /Problem/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /Clauses/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /Search/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Settings/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Compile evaluator/i })).not.toBeInTheDocument();
    expect(screen.queryByText(/Local browser model available/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Generated with the deterministic fallback template/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/No problem text leaves this browser/i)).not.toBeInTheDocument();
  });

  it("opens Sudoku directly from the URL hash and keeps tab changes linkable", async () => {
    const user = userEvent.setup();
    window.history.replaceState(null, "", "/#sudoku");

    await renderApp();

    expect(screen.getByRole("tab", { name: /^Sudoku/i })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText(/Signed p-adic objective/i)).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: /Boolean CSP\/SAT/i }));
    expect(window.location.hash).toBe("#csp");
    expect(screen.getByText("CSP clauses")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: /^Sudoku/i }));
    expect(window.location.hash).toBe("#sudoku");
    expect(screen.getByText(/All-different instance/i)).toBeInTheDocument();
  });

  it("shows manual syntax hints unless a browser language model is exposed", async () => {
    const user = userEvent.setup();
    const { unmount } = await renderApp();
    expect(screen.queryByText("Enter CSP")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Default problem/i })).not.toBeInTheDocument();
    const manualEditor = screen.getByLabelText("CSP source");
    expect(manualEditor.getAttribute("placeholder")).toContain("One CSP clause per line");
    expect(manualEditor.getAttribute("placeholder")).toContain("not A or C");
    expect(manualEditor.getAttribute("placeholder")).toContain("A implies B");
    expect(manualEditor.getAttribute("placeholder")).toContain("C -> D");
    await user.click(screen.getByRole("button", { name: "CSP syntax help" }));
    const syntaxHelp = screen.getByRole("note", { name: "CSP clause syntax" });
    expect(syntaxHelp).toHaveTextContent("one Boolean constraint per line");
    expect(within(syntaxHelp).getByText("B xor C")).toBeInTheDocument();
    expect(within(syntaxHelp).getByText("A implies B")).toBeInTheDocument();
    expect(within(syntaxHelp).getByText("C -> D")).toBeInTheDocument();
    unmount();

    const availability = vi.fn().mockResolvedValue("available");
    globalThis.languageModel = {
      availability,
      create: vi.fn(async () => ({ prompt: vi.fn(), destroy: vi.fn() }))
    };

    await renderApp();
    await waitFor(() => expect(screen.getByText("Enter CSP")).toBeInTheDocument());
    expect(availability).toHaveBeenCalled();
    expect(screen.getByLabelText("Natural language problem")).toHaveValue(DEFAULT_ASSIGNMENT_PROBLEM);
    expect(screen.queryByRole("button", { name: "CSP syntax help" })).not.toBeInTheDocument();
    expect(screen.getByLabelText("CSP source")).not.toHaveAttribute("placeholder");
  });

  it("starts with a blank CSP editor", async () => {
    const { container } = await renderApp();

    expect(screen.getByRole("heading", {
      name: "p-adic linear regression",
      exact: true
    })).toBeInTheDocument();
    expect(screen.queryByRole("slider")).not.toBeInTheDocument();
    expect(screen.getByRole("separator", { name: /Resize CSP and CNF columns/i })).toBeInTheDocument();
    expect(screen.getByRole("separator", { name: /Resize CNF and data columns/i })).toBeInTheDocument();
    expect(screen.getByLabelText("CSP source")).toHaveValue("");
    expect(screen.getByText(/Constraints: 0/i)).toBeInTheDocument();
    const readyBand = container.querySelector<HTMLElement>(".ready-band");
    const setupGrid = container.querySelector<HTMLElement>(".setup-grid");
    const header = screen.getByRole("banner");
    expect(readyBand).toBeInTheDocument();
    expect(readyBand).toHaveAttribute("aria-hidden", "true");
    expect(readyBand).not.toHaveClass("is-visible");
    expect(readyBand?.querySelector(".run-button")).toBeDisabled();
    expect(header.compareDocumentPosition(readyBand as Node) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(readyBand?.compareDocumentPosition(setupGrid as Node) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Solve p-adic linear regression/i })).not.toBeInTheDocument();
    expect(screen.getByText(/No compiled problem yet/i)).toBeInTheDocument();
  });

  it("resizes adjacent setup columns from the dividers", async () => {
    const { container } = await renderApp();
    const rect = (width: number) => ({
      bottom: 0,
      height: 0,
      left: 0,
      right: width,
      top: 0,
      width,
      x: 0,
      y: 0,
      toJSON: () => ({})
    }) as DOMRect;
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function () {
      if (this.classList.contains("problem-panel")) return rect(400);
      if (this.classList.contains("cnf-panel")) return rect(360);
      if (this.classList.contains("regression-column")) return rect(420);
      return rect(0);
    });

    fireEvent.keyDown(
      screen.getByRole("separator", { name: /Resize CSP and CNF columns/i }),
      { key: "ArrowRight" }
    );

    const grid = container.querySelector<HTMLElement>(".setup-grid");
    expect(grid?.style.getPropertyValue("--problem-column")).toBe("424fr");
    expect(grid?.style.getPropertyValue("--cnf-column")).toBe("336fr");
    expect(grid?.style.getPropertyValue("--regression-column")).toBe("420fr");
  });

  it("shows the CNF-derived regression dataframe and unit-well floor", async () => {
    await renderApp();

    fireEvent.change(screen.getByLabelText("CSP source"), {
      target: { value: "A or not B" }
    });

    expect(screen.getByText("p-adic linear regression dataframe")).toBeInTheDocument();
    expect(screen.getByText("C1")).toBeInTheDocument();
    expect(screen.getByText("A = 0")).toBeInTheDocument();
    expect(screen.getByText("A = 1")).toBeInTheDocument();
    expect(screen.getByText("CNF constraints")).toBeInTheDocument();
    expect(screen.getByText("Unit wells")).toBeInTheDocument();
    expect(screen.getByText("Negative clause reward")).toBeInTheDocument();
    expect(screen.getByText("Unit-well weight")).toBeInTheDocument();
    expect(screen.getByText("α = 2 = clauses + 1")).toBeInTheDocument();
    expect(screen.getByText("Satisfiable floor")).toBeInTheDocument();
    expect(screen.getByText("3 = αn − clauses")).toBeInTheDocument();
    expect(screen.getByText("≠ 1")).toBeInTheDocument();
    expect(screen.getByText("−1")).toBeInTheDocument();
    expect(document.querySelector(".ready-band")).toHaveClass("is-visible");
    expect(document.querySelector(".ready-band")).toHaveAttribute("aria-hidden", "false");
    expect(screen.getByText("Ready to solve.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Solve p-adic linear regression/i })).toBeEnabled();
  });

  it("loads a story sample with its known CNF", async () => {
    const user = userEvent.setup();
    globalThis.languageModel = {
      availability: vi.fn().mockResolvedValue("available"),
      create: vi.fn(async () => ({ prompt: vi.fn(), destroy: vi.fn() }))
    };
    await renderApp();
    await waitFor(() => expect(screen.getByText("Enter CSP")).toBeInTheDocument());

    await user.selectOptions(screen.getByLabelText("Sample problem"), "blackout-restoration");

    expect((screen.getByLabelText("Natural language problem") as HTMLTextAreaElement).value)
      .toContain("city-wide blackout");
    expect((screen.getByLabelText("CSP source") as HTMLTextAreaElement).value)
      .toContain("Harbour_primary xor Harbour_backup");
    expect(screen.getByText(/Constraints: 24/i)).toBeInTheDocument();
    expect(screen.getByText("47 clauses")).toBeInTheDocument();
    expect(screen.getAllByText(/16,777,216/).length).toBeGreaterThan(0);
  });

  it("shows sample text as non-form content without a language model and fades it on clause edits", async () => {
    const { container } = await renderApp();

    fireEvent.change(screen.getByLabelText("Sample problem"), {
      target: { value: "blackout-restoration" }
    });

    const statement = screen.getByRole("note", { name: "Sample problem statement" });
    expect(statement.tagName).toBe("DIV");
    expect(statement).toHaveTextContent("city-wide blackout");

    vi.useFakeTimers();
    const source = screen.getByLabelText("CSP source") as HTMLTextAreaElement;
    fireEvent.change(source, { target: { value: `${source.value}\nnot Harbour_primary` } });

    expect(container.querySelector(".sample-description-box")).toHaveClass("is-fading");
    act(() => vi.advanceTimersByTime(180));
    expect(screen.queryByRole("note", { name: "Sample problem statement" })).not.toBeInTheDocument();
  });

  it("fades the language-model problem text on the first clause edit", async () => {
    globalThis.languageModel = {
      availability: vi.fn().mockResolvedValue("available"),
      create: vi.fn(async () => ({ prompt: vi.fn(), destroy: vi.fn() }))
    };
    const { container } = await renderApp();
    await waitFor(() => expect(screen.getByText("Enter CSP")).toBeInTheDocument());
    vi.useFakeTimers();

    fireEvent.change(screen.getByLabelText("CSP source"), {
      target: { value: "A or B" }
    });

    expect(container.querySelector(".description-box")).toHaveClass("is-fading");
    act(() => vi.advanceTimersByTime(180));
    expect(screen.queryByLabelText("Natural language problem")).not.toBeInTheDocument();
  });

  it("uses one scrolling textarea and keeps its line-number gutter in sync", async () => {
    const { container } = await renderApp();
    const editor = screen.getByLabelText("CSP source") as HTMLTextAreaElement;
    const gutter = container.querySelector<HTMLElement>(".line-numbers");

    expect(editor).toHaveAttribute("wrap", "off");
    expect(gutter).not.toBeNull();
    editor.scrollTop = 84;
    fireEvent.scroll(editor);
    expect(gutter?.scrollTop).toBe(84);
  });

  it("shows variables beside manually edited CSP clauses", async () => {
    await renderApp();
    const editor = screen.getByLabelText("CSP source");

    fireEvent.change(editor, {
      target: { value: "Alpha or Beta\nGamma xor not Alpha" }
    });

    let variables = screen.getByRole("list", { name: "CSP variables" });
    expect(within(variables).getAllByRole("listitem").map((item) => item.textContent))
      .toEqual(["Alpha", "Beta", "Gamma"]);
    expect(screen.getByText("Variables: 3")).toBeInTheDocument();

    fireEvent.change(editor, {
      target: { value: "Alpha or Beta\nDelta or" }
    });

    variables = screen.getByRole("list", { name: "CSP variables" });
    expect(within(variables).getAllByRole("listitem").map((item) => item.textContent))
      .toEqual(["Alpha", "Beta"]);
  });

  it("sends a complete random permutation to every search worker", async () => {
    const user = userEvent.setup();
    await renderApp();
    fireEvent.change(screen.getByLabelText("CSP source"), {
      target: { value: "A or B" }
    });

    await user.selectOptions(screen.getByLabelText("Search algorithm"), "random");
    expect(screen.getByText("Random-permutation exhaustive search")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Solve p-adic linear regression/i }));

    expect(MockWorker.instances).toHaveLength(2);
    for (const worker of MockWorker.instances) {
      expect(worker.postMessage).toHaveBeenCalledWith(expect.objectContaining({
        assignmentCount: 4,
        strategy: "random",
        permutation: expect.objectContaining({
          multiplier: expect.any(Number),
          offset: expect.any(Number)
        })
      }));
    }
  });

  it("offers Zubarev and Mihara algorithms for Boolean CSP", async () => {
    const user = userEvent.setup();
    await renderApp();
    fireEvent.change(screen.getByLabelText("CSP source"), {
      target: { value: "A or B" }
    });

    const algorithm = screen.getByLabelText("Search algorithm");
    expect(within(algorithm).getByRole("option", { name: "Zubarev walk" })).toBeInTheDocument();
    expect(within(algorithm).getByRole("option", { name: "Mihara digitwise attempt" }))
      .toBeInTheDocument();

    await user.selectOptions(algorithm, "zubarev");
    expect(screen.getByText("Zubarev Boltzmann walk on Boolean bit flips"))
      .toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Solve p-adic linear regression/i }));

    expect(MockWorker.instances).toHaveLength(2);
    expect(MockWorker.instances[0].postMessage).toHaveBeenCalledWith(expect.objectContaining({
      strategy: "zubarev",
      variableCount: 2,
      miharaObservations: expect.any(Array),
      prime: 17
    }));
  });

  it("keeps the Mihara attempt running past its initial trial batch", async () => {
    const user = userEvent.setup();
    await renderApp();
    await user.selectOptions(screen.getByLabelText("Sample problem"), "assignment");
    await user.selectOptions(screen.getByLabelText("Search algorithm"), "mihara");
    await user.click(screen.getByRole("button", { name: /Solve p-adic linear regression/i }));

    expect(MockWorker.instances).toHaveLength(2);
    for (const [index, worker] of MockWorker.instances.entries()) {
      expect(worker.postMessage).toHaveBeenCalledWith(expect.objectContaining({
        strategy: "mihara",
        unbounded: true
      }));
      const startMessage = worker.postMessage.mock.calls[0][0] as {
        miharaObservations: Array<{ weight?: number; source?: string }>;
      };
      expect(startMessage.miharaObservations.some((row) => row.weight === 61)).toBe(true);
      expect(startMessage.miharaObservations.some((row) =>
        row.source?.startsWith("Positive complement of C")
      )).toBe(true);
      act(() => worker.onmessage?.({
        data: {
          type: "progress",
          workerId: index + 1,
          tested: 52,
          currentMask: 0,
          speed: 52,
          bestLoss: null,
          bestMask: null,
          bestCoordinates: index === 0 ? Array(16).fill(2) : null,
          algorithmScore: index === 0 ? 21 : null,
          algorithmTotal: 148,
          algorithmLoss: index === 0 ? 127 : null,
          algorithmSuccessfulTrials: index === 0 ? 1 : 0,
          algorithmSingularTrials: index === 0 ? 51 : 52,
          solutions: 0,
          done: false
        }
      } as MessageEvent));
    }

    expect(screen.getByText("1 / 104")).toBeInTheDocument();
    expect(screen.getByText(/not Boolean, so there is no CSP assignment to validate/i))
      .toBeInTheDocument();
    expect(screen.getByText(/Mihara receives a positive-only complement expansion/i))
      .toBeInTheDocument();
    expect(screen.queryByText(/wrong statistical model on purpose/i)).not.toBeInTheDocument();
    expect(screen.getByRole("img", { name: "p-adic loss over time" }))
      .toHaveAttribute("data-points", "1");
    expect(screen.getByRole("status")).toHaveTextContent(/Retrying fresh starts/i);
    expect(screen.getByRole("button", { name: "Stop" })).toBeEnabled();
    expect(MockWorker.instances.every((worker) => worker.terminate.mock.calls.length === 0)).toBe(true);
    expect(screen.queryByText("Attempt complete")).not.toBeInTheDocument();
  });

  it("shows generation progress without the debug transcript", async () => {
    const user = userEvent.setup();
    const prompt = vi.fn().mockResolvedValueOnce(
      JSON.stringify({
        kind: "assignment",
        people: ["Ava"],
        jobs: ["test"],
        person_exactly_one_job: false,
        job_exactly_one_person: false,
        forbidden: [{ person: "Ava", job: "test", source: "Ava cannot test." }],
        implications: []
      })
    );
    globalThis.languageModel = {
      availability: vi.fn().mockResolvedValue("available"),
      create: vi.fn(async () => ({ prompt, destroy: vi.fn() }))
    };
    await renderApp();
    await waitFor(() => expect(screen.getByText("Enter CSP")).toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: /Generate CSP/i }));

    await waitFor(() => expect(screen.getByText(/1 Boolean variable found/i)).toBeInTheDocument());
    expect(screen.getByText("Ava_test")).toBeInTheDocument();
    expect(screen.getByLabelText("CSP source")).toHaveValue("");
    await waitFor(() => expect(screen.getByLabelText("CSP source")).toHaveValue("not Ava_test"));
    expect(screen.queryByText("Language model conversation")).not.toBeInTheDocument();
    expect(screen.queryByText("assistant-recorded")).not.toBeInTheDocument();
    expect(screen.queryByText(/Boolean variable found/i)).not.toBeInTheDocument();
    expect(screen.getByText(/Conversion complete/i)).toBeInTheDocument();
    expect(screen.getAllByText(/not Ava_test/).length).toBeGreaterThan(0);
  });

  it("moves a small reading highlight while the local model is thinking", async () => {
    const prompt = vi.fn((_input, options?: { signal?: AbortSignal }) => {
      return new Promise<string>((_resolve, reject) => {
        options?.signal?.addEventListener("abort", () => {
          reject(new DOMException("Aborted", "AbortError"));
        });
      });
    });
    globalThis.languageModel = {
      availability: vi.fn().mockResolvedValue("available"),
      create: vi.fn(async () => ({ prompt, destroy: vi.fn() }))
    };
    const { container } = await renderApp();
    await waitFor(() => expect(screen.getByText("Enter CSP")).toBeInTheDocument());
    vi.useFakeTimers();

    fireEvent.click(screen.getByRole("button", { name: /Generate CSP/i }));
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(prompt).toHaveBeenCalled();
    const highlightedWords = () => Array.from(
      container.querySelectorAll<HTMLElement>(".generation-reading-token.is-focused")
    ).map((token) => token.textContent).join(" ");
    const firstHighlight = highlightedWords();
    expect(firstHighlight.split(" ")).toHaveLength(2);

    act(() => vi.advanceTimersByTime(340));

    const nextHighlight = highlightedWords();
    expect(nextHighlight).not.toBe(firstHighlight);
    expect(nextHighlight.split(" ").length).toBeGreaterThanOrEqual(1);
    expect(nextHighlight.split(" ").length).toBeLessThanOrEqual(3);

    fireEvent.click(screen.getByRole("button", { name: /Cancel/i }));
    await act(async () => {
      await Promise.resolve();
    });
    expect(container.querySelector(".generation-reading-text")).not.toBeInTheDocument();
  });

  it("lets the user cancel CSP generation", async () => {
    const user = userEvent.setup();
    const prompt = vi.fn((_input, options?: { signal?: AbortSignal }) => {
      return new Promise<string>((_resolve, reject) => {
        options?.signal?.addEventListener("abort", () => {
          reject(new DOMException("Aborted", "AbortError"));
        });
      });
    });
    globalThis.languageModel = {
      availability: vi.fn().mockResolvedValue("available"),
      create: vi.fn(async () => ({ prompt, destroy: vi.fn() }))
    };
    await renderApp();
    await waitFor(() => expect(screen.getByText("Enter CSP")).toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: /Generate CSP/i }));
    await waitFor(() => expect(prompt).toHaveBeenCalled());
    expect(screen.getByText(/Reading the problem and extracting a typed structure/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Cancel/i }));

    await waitFor(() => expect(screen.getByText(/Generation cancelled/i)).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: /Cancel/i })).not.toBeInTheDocument();
    expect(document.querySelector(".generation-reading-text")).not.toBeInTheDocument();
    expect((prompt.mock.calls[0]?.[1] as { signal?: AbortSignal }).signal?.aborted).toBe(true);
  });

  it("keeps the CSP, CNF, and dataframe above the results when search starts", async () => {
    const user = userEvent.setup();
    const scrollIntoView = vi.fn();
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
      configurable: true,
      value: scrollIntoView,
      writable: true
    });
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) => {
      callback(0);
      return 1;
    });
    const { container } = await renderApp();

    const editor = screen.getByLabelText("CSP source");
    fireEvent.change(editor, { target: { value: "Zed" } });

    await user.click(screen.getByRole("button", { name: /Solve p-adic linear regression/i }));

    expect(container.querySelector(".ready-band")).not.toHaveClass("is-visible");
    expect(container.querySelector(".ready-band")).toHaveAttribute("aria-hidden", "true");
    expect(editor).toBeInTheDocument();
    expect(editor).toHaveValue("Zed");
    expect(screen.getByRole("heading", { name: "CNF view" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "p-adic linear regression dataframe" })).toBeInTheDocument();
    expect(screen.queryByText(/Regression problem/i)).not.toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Search results" })).toBeInTheDocument();
    const setupGrid = container.querySelector(".setup-grid");
    const runGrid = container.querySelector(".run-grid");
    expect(setupGrid).not.toBeNull();
    expect(runGrid).not.toBeNull();
    expect(setupGrid?.compareDocumentPosition(runGrid as Node) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(MockWorker.instances).toHaveLength(2);
    expect(MockWorker.instances[0].postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        evaluatorSource: expect.stringContaining("// Zed")
      })
    );
    await waitFor(() => expect(scrollIntoView).toHaveBeenCalledWith({
      behavior: "smooth",
      block: "start"
    }));
  });

  it("replaces active search controls when exhaustive search completes", async () => {
    const user = userEvent.setup();
    await renderApp();

    fireEvent.change(screen.getByLabelText("CSP source"), {
      target: { value: "Zed" }
    });
    await user.click(screen.getByRole("button", { name: /Solve p-adic linear regression/i }));
    expect(screen.getByRole("button", { name: /^Pause$/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^Stop$/i })).toBeInTheDocument();

    await act(async () => {
      for (const worker of MockWorker.instances) {
        const startMessage = worker.postMessage.mock.calls[0]?.[0] as {
          workerId: number;
          start: number;
          endExclusive: number;
        };
        const width = startMessage.endExclusive - startMessage.start;
        const containsSatisfyingAssignment =
          startMessage.start <= 1 && startMessage.endExclusive > 1;
        worker.onmessage?.({
          data: {
            type: "done",
            workerId: startMessage.workerId,
            tested: width,
            currentMask: Math.max(startMessage.start, startMessage.endExclusive - 1),
            speed: 1,
            bestLoss: width > 0 ? containsSatisfyingAssignment ? 1 : 2 : null,
            bestMask: width > 0
              ? containsSatisfyingAssignment ? 1 : startMessage.start
              : null,
            solutions: containsSatisfyingAssignment ? 1 : 0,
            done: true
          }
        } as MessageEvent);
      }
    });

    await waitFor(() => expect(screen.getByText("Search complete")).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: /^Pause$/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Stop$/i })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Clear results/i })).toBeInTheDocument();
    expect(screen.getByText(/Best p-adic regression solution/i)).toBeInTheDocument();
    expect(screen.getByText(/y = /i)).toBeInTheDocument();
    expect(screen.queryByText(/Regression problem/i)).not.toBeInTheDocument();
    expect(document.querySelector(".compiled-summary")).not.toBeInTheDocument();
    expect(document.querySelector(".assignment-panel")).not.toBeInTheDocument();
    const lossChart = document.querySelector(".loss-chart");
    const solutionSection = document.querySelector(".solution-section");
    expect(lossChart).not.toBeNull();
    expect(solutionSection).not.toBeNull();
    expect(
      lossChart?.compareDocumentPosition(solutionSection as Node) & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy();
    const contributionTable = screen.getByRole("table", { name: "Regression row contributions" });
    expect(contributionTable.querySelectorAll("tbody tr")).toHaveLength(3);
    expect(contributionTable).toHaveTextContent("v17(r)");
    expect(screen.getByText("Total L = 1")).toBeInTheDocument();
    expect(
      Array.from(document.querySelectorAll(".loss-chart .y-axis-label")).map((node) => node.textContent)
    ).toEqual(["1", "2"]);
    expect(document.querySelector(".loss-chart svg")).toHaveAttribute("data-points", "2");
    expect(document.querySelector(".loss-chart svg")).toHaveAttribute("data-last-tested", "2");
    expect(document.querySelector(".loss-chart svg")).toHaveAttribute("data-hit-floor", "true");
    expect(screen.getAllByText("Satisfiable floor").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("1").length).toBeGreaterThan(0);
    expect(screen.getAllByText(/candidate hyperplanes/i).length).toBeGreaterThan(0);
    expect(screen.queryByText(/mask/i)).not.toBeInTheDocument();

  });

  it("switches to Sudoku mode and shows the signed p-adic objective", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("tab", { name: /^Sudoku/i }));

    expect(screen.getByText(/Signed p-adic objective/i)).toBeInTheDocument();
    expect(screen.getByText(/All-different instance/i)).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "p-adic loss over time" }))
      .not.toBeInTheDocument();
    expect(screen.queryByText("Conflict count over time")).not.toBeInTheDocument();
    expect(screen.queryByRole("img", { name: "Sudoku p-adic loss over time" }))
      .not.toBeInTheDocument();
    const dataframe = screen.getByRole("region", {
      name: "Sudoku p-adic linear regression dataframe"
    });
    expect(within(dataframe).getByRole("table", { name: "Sudoku regression observations" }))
      .toBeInTheDocument();
    expect(dataframe).toHaveTextContent("1,299 rows · 81 coefficients");
    expect(dataframe).toHaveTextContent("Digit-pinning wells (+21)");
    expect(dataframe).toHaveTextContent("Peer inequality rewards (−1)");
    expect(screen.getByRole("button", { name: /Start search/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /Mihara attempt/i })).toBeEnabled();
  });

  it("explains the positive-complement Mihara Sudoku fit", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("tab", { name: /^Sudoku/i }));
    await user.click(screen.getByRole("button", { name: /Mihara attempt/i }));

    expect(screen.getByText(/Mihara receives a positive-only complement expansion/i))
      .toBeInTheDocument();
    expect(screen.getByText(/weighted consensus now rewards unequal peers/i)).toBeInTheDocument();
    expect(screen.getByText("Retry policy")).toBeInTheDocument();
    expect(screen.getByText("Until Sudoku solution or stopped")).toBeInTheDocument();
    expect(screen.getByText("Mihara prime").closest(".metric-row"))
      .toHaveTextContent("p = 19 (> 16)");
  });

  it("charts Mihara's selected positive loss separately from the original signed audit", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("tab", { name: /^Sudoku/i }));
    await user.click(screen.getByRole("button", { name: /Mihara attempt/i }));
    await user.click(screen.getByRole("button", { name: /Start search/i }));

    const positiveChart = screen.getByRole("img", {
      name: "Mihara positive-complement p-adic loss over time"
    });
    const signedChart = screen.getByRole("img", {
      name: "Original signed Sudoku p-adic loss over time"
    });
    await waitFor(() => {
      expect(Number(positiveChart.getAttribute("data-points"))).toBeGreaterThan(0);
      expect(Number(signedChart.getAttribute("data-points"))).toBeGreaterThan(0);
    }, { timeout: 5_000 });

    expect(screen.getByText("Positive-complement last-digit loss")).toBeInTheDocument();
    expect(screen.getByText("Original signed objective L(x)")).toBeInTheDocument();
    expect(screen.getByText(/this is the score Mihara selects/i)).toBeInTheDocument();
    expect(screen.getByText(/this does not select Mihara fits/i)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Pause/i }));
  });

  it("edits Sudoku clues on the board and locks cells after search starts", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("tab", { name: /^Sudoku/i }));

    const firstCell = screen.getByRole("textbox", { name: "Row 1, column 1" });
    const blankCell = screen.getByRole("textbox", { name: "Row 1, column 3" });
    expect(firstCell).toHaveValue("5");
    expect(blankCell).toHaveValue("");
    expect(firstCell).toBeEnabled();

    await user.clear(firstCell);
    await user.type(blankCell, "4");

    expect(firstCell).toHaveValue("");
    expect(blankCell).toHaveValue("4");
    const clueMetric = screen.getByText("Clues (singleton domains)").closest(".metric-row");
    expect(clueMetric).not.toBeNull();
    expect(within(clueMetric as HTMLElement).getByText("30")).toBeInTheDocument();
    expect((screen.getByLabelText(/81-character puzzle/i) as HTMLTextAreaElement).value)
      .toMatch(/^\.34/u);

    await user.click(screen.getByRole("button", { name: /Start search/i }));
    expect(firstCell).toBeDisabled();
    expect(blankCell).toBeDisabled();
    await waitFor(() => {
      expect(Number(
        screen.getByRole("img", { name: "Sudoku p-adic loss over time" })
          .getAttribute("data-points")
      )).toBeGreaterThan(0);
    });
    const lossPanel = screen.getByRole("heading", { name: "p-adic loss over time" })
      .closest("section");
    const dataframe = screen.getByRole("region", {
      name: "Sudoku p-adic linear regression dataframe"
    });
    expect(lossPanel).not.toBeNull();
    expect(
      lossPanel?.compareDocumentPosition(dataframe) & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy();
  });
});
