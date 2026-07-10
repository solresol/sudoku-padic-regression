import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  type RenderResult
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

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
  });

  afterEach(() => {
    clearLanguageModel();
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

  it("hides natural-language CSP entry unless a browser language model is exposed", async () => {
    const { unmount } = await renderApp();
    expect(screen.queryByText("Enter CSP")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Default problem/i })).toBeInTheDocument();
    unmount();

    const availability = vi.fn().mockResolvedValue("available");
    globalThis.languageModel = {
      availability,
      create: vi.fn(async () => ({ prompt: vi.fn(), destroy: vi.fn() }))
    };

    await renderApp();
    await waitFor(() => expect(screen.getByText("Enter CSP")).toBeInTheDocument());
    expect(availability).toHaveBeenCalled();
  });

  it("starts with a blank CSP editor", async () => {
    await renderApp();

    expect(screen.getByRole("heading", { name: /p-adic linear regression/i })).toBeInTheDocument();
    expect(screen.queryByRole("slider")).not.toBeInTheDocument();
    expect(screen.getByRole("separator", { name: /Resize CSP and CNF columns/i })).toBeInTheDocument();
    expect(screen.getByRole("separator", { name: /Resize CNF and data columns/i })).toBeInTheDocument();
    expect(screen.getByLabelText("CSP source")).toHaveValue("");
    expect(screen.getByText(/Constraints: 0/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Solve p-adic regression/i })).toBeDisabled();
    expect(screen.getByText(/No compiled problem yet/i)).toBeInTheDocument();
    expect(screen.getByText("Threads")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
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

    expect(screen.getByText("CNF regression dataframe")).toBeInTheDocument();
    expect(screen.getByText("C1")).toBeInTheDocument();
    expect(screen.getByText("A = 0")).toBeInTheDocument();
    expect(screen.getByText("A = 1")).toBeInTheDocument();
    expect(screen.getByText("CNF constraints")).toBeInTheDocument();
    expect(screen.getByText("Unit wells")).toBeInTheDocument();
    expect(screen.getByText("Loss floor")).toBeInTheDocument();
    expect(screen.getByText("2 (one unit well per coefficient)")).toBeInTheDocument();
  });

  it("resets the default natural-language problem without preloading CNF", async () => {
    const user = userEvent.setup();
    globalThis.languageModel = {
      availability: vi.fn().mockResolvedValue("available"),
      create: vi.fn(async () => ({ prompt: vi.fn(), destroy: vi.fn() }))
    };
    await renderApp();
    await waitFor(() => expect(screen.getByText("Enter CSP")).toBeInTheDocument());

    const editor = screen.getByLabelText("CSP source");
    const problem = screen.getByLabelText("Natural language problem");
    fireEvent.change(editor, { target: { value: "Zed" } });
    fireEvent.change(problem, { target: { value: "Custom problem" } });

    await user.click(screen.getByRole("button", { name: /Default problem/i }));

    expect(editor).toHaveValue("");
    expect((problem as HTMLTextAreaElement).value).toContain("Ava, Ben, Cara, and Devina");
    expect(screen.getByText(/Constraints: 0/i)).toBeInTheDocument();
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

    await waitFor(() => expect(screen.getByLabelText("CSP source")).toHaveValue("not Ava_test"));
    expect(screen.queryByText("Language model conversation")).not.toBeInTheDocument();
    expect(screen.queryByText("assistant-recorded")).not.toBeInTheDocument();
    expect(screen.getByText(/Review complete/i)).toBeInTheDocument();
    expect(screen.getAllByText(/not Ava_test/).length).toBeGreaterThan(0);
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
    expect(screen.getByText(/Decoding terms and finding variables/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Cancel/i }));

    await waitFor(() => expect(screen.getByText(/Generation cancelled/i)).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: /Cancel/i })).not.toBeInTheDocument();
    expect((prompt.mock.calls[0]?.[1] as { signal?: AbortSignal }).signal?.aborted).toBe(true);
  });

  it("compiles the current editor contents when exhaustive search starts", async () => {
    const user = userEvent.setup();
    await renderApp();

    const editor = screen.getByLabelText("CSP source");
    fireEvent.change(editor, { target: { value: "Zed" } });

    await user.click(screen.getByRole("button", { name: /Solve p-adic regression/i }));

    expect(screen.getByText(/Regression problem/i)).toBeInTheDocument();
    expect(MockWorker.instances).toHaveLength(2);
    expect(MockWorker.instances[0].postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        evaluatorSource: expect.stringContaining("// Zed")
      })
    );
  });

  it("replaces active search controls when exhaustive search completes", async () => {
    const user = userEvent.setup();
    await renderApp();

    fireEvent.change(screen.getByLabelText("CSP source"), {
      target: { value: "Zed" }
    });
    await user.click(screen.getByRole("button", { name: /Solve p-adic regression/i }));
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
        worker.onmessage?.({
          data: {
            type: "done",
            workerId: startMessage.workerId,
            tested: width,
            currentMask: Math.max(startMessage.start, startMessage.endExclusive - 1),
            speed: 1,
            bestLoss: width > 0 ? 0 : null,
            bestMask: width > 0 ? startMessage.start : null,
            solutions: width > 0 ? 1 : 0,
            done: true
          }
        } as MessageEvent);
      }
    });

    await waitFor(() => expect(screen.getByText("Search complete")).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: /^Pause$/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Stop$/i })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Back to setup/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Export proof/i })).toBeInTheDocument();
    expect(screen.getByText(/Best p-adic regression solution/i)).toBeInTheDocument();
    expect(screen.getByText(/y = /i)).toBeInTheDocument();
    expect(screen.getByText("Unit-well floor")).toBeInTheDocument();
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
    expect(screen.getByRole("button", { name: /Start search/i })).toBeEnabled();
  });
});
