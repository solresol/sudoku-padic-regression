import {
  Check,
  CircleHelp,
  CirclePause,
  Code2,
  Download,
  FileText,
  FlaskConical,
  Grid3x3,
  Play,
  RotateCcw,
  Search,
  ShieldCheck,
  Square,
  Triangle,
  WandSparkles,
  X
} from "lucide-react";
import {
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";
import {
  buildRegressionDataFrame,
  type CompiledProblem,
  compileProblem,
  evaluateAssignment,
  evaluateRegressionDataFrame,
  parseProblem,
  renderClause,
  renderClauseAffine
} from "./lib/csp";
import {
  type CnfGenerationStep,
  type LanguageModelAvailability,
  detectLanguageModel,
  generateCspFromDescription
} from "./lib/browserLanguageModel";
import {
  CSP_SAMPLES,
  DEFAULT_ASSIGNMENT_PROBLEM,
  countCspConstraints,
  type CspSample
} from "./lib/defaultProblems";
import {
  createSearchPlan,
  formatAssignmentCount,
  type SearchStrategy
} from "./lib/search";
import { useSearchController } from "./hooks/useSearchController";
import SudokuMode from "./SudokuMode";

type Mode = "csp" | "sudoku";
type GenerationPhase = "preparing" | "reading" | "variables" | "clauses";
type DescriptionVisibility = "visible" | "fading" | "hidden";
type ColumnWidths = {
  problem: number;
  cnf: number;
  regression: number;
};
type ColumnDivider = "problem-cnf" | "cnf-regression";
type ColumnResize = {
  cleanup: () => void;
  divider: ColumnDivider;
  pointerId: number;
  startX: number;
  widths: ColumnWidths;
};

const DEFAULT_COLUMN_WIDTHS: ColumnWidths = {
  problem: 1.12,
  cnf: 0.82,
  regression: 1.08
};
const MIN_COLUMN_WIDTHS: ColumnWidths = {
  problem: 320,
  cnf: 280,
  regression: 340
};
const DESCRIPTION_FADE_MS = 180;

function App() {
  const [mode, setMode] = useState<Mode>("csp");
  const [columnWidths, setColumnWidths] = useState<ColumnWidths>(DEFAULT_COLUMN_WIDTHS);
  const [source, setSource] = useState("");
  const [description, setDescription] = useState(DEFAULT_ASSIGNMENT_PROBLEM);
  const [descriptionVisibility, setDescriptionVisibility] =
    useState<DescriptionVisibility>("visible");
  const [selectedSampleId, setSelectedSampleId] = useState("");
  const [compiled, setCompiled] = useState<CompiledProblem | null>(null);
  const [compileError, setCompileError] = useState<string | null>(null);
  const [workerCount, setWorkerCount] = useState(2);
  const [searchStrategy, setSearchStrategy] = useState<SearchStrategy>("ordered");
  const [modelStatus, setModelStatus] =
    useState<LanguageModelAvailability>("unavailable");
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationSteps, setGenerationSteps] = useState<CnfGenerationStep[]>([]);
  const [generationStatus, setGenerationStatus] = useState<string | null>(null);
  const [generationReview, setGenerationReview] = useState<string | null>(null);
  const [generationPhase, setGenerationPhase] = useState<GenerationPhase | null>(null);
  const [generationVariables, setGenerationVariables] = useState<string[]>([]);
  const [submittedDescription, setSubmittedDescription] = useState("");
  const controller = useSearchController(compiled);
  const generationAbortRef = useRef<AbortController | null>(null);
  const descriptionFadeTimerRef = useRef<number | null>(null);
  const setupGridRef = useRef<HTMLElement | null>(null);
  const resultsRef = useRef<HTMLElement | null>(null);
  const previousSearchStatusRef = useRef(controller.snapshot.status);
  const columnResizeRef = useRef<ColumnResize | null>(null);

  useEffect(() => {
    let alive = true;
    detectLanguageModel()
      .then((availability) => {
        if (!alive) {
          return;
        }
        setModelStatus(availability);
      })
      .catch(() => {
        setModelStatus("unavailable");
      });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    const previousStatus = previousSearchStatusRef.current;
    const currentStatus = controller.snapshot.status;
    previousSearchStatusRef.current = currentStatus;

    if (currentStatus === "running" && previousStatus !== "running") {
      window.requestAnimationFrame(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    }
  }, [controller.snapshot.status]);

  useEffect(() => {
    return () => {
      columnResizeRef.current?.cleanup();
      columnResizeRef.current = null;
      if (descriptionFadeTimerRef.current != null) {
        window.clearTimeout(descriptionFadeTimerRef.current);
      }
      document.body.classList.remove("is-resizing-columns");
    };
  }, []);

  const searchPlan = useMemo(
    () => (compiled ? createSearchPlan(compiled, workerCount, searchStrategy) : null),
    [compiled, searchStrategy, workerCount]
  );
  const sourceVariables = useMemo(
    () => inferSourceVariableNames(source, compiled),
    [compiled, source]
  );

  const revealDescription = () => {
    if (descriptionFadeTimerRef.current != null) {
      window.clearTimeout(descriptionFadeTimerRef.current);
      descriptionFadeTimerRef.current = null;
    }
    setDescriptionVisibility("visible");
  };

  const fadeDescription = () => {
    if (descriptionVisibility !== "visible") {
      return;
    }
    setDescriptionVisibility("fading");
    descriptionFadeTimerRef.current = window.setTimeout(() => {
      setDescriptionVisibility("hidden");
      descriptionFadeTimerRef.current = null;
    }, DESCRIPTION_FADE_MS);
  };

  const handleSourceChange = (value: string) => {
    fadeDescription();
    setSelectedSampleId("");
    setSource(value);
    setGenerationReview(null);
    setGenerationVariables([]);
    try {
      setCompiled(compileCspSource(value));
      setCompileError(null);
      controller.reset();
    } catch (error) {
      setCompiled(null);
      setCompileError(error instanceof Error ? error.message : String(error));
      controller.reset();
    }
  };

  const handleLoadSample = (sampleId: string) => {
    const sample = CSP_SAMPLES.find((candidate) => candidate.id === sampleId);
    if (!sample) {
      return;
    }

    generationAbortRef.current?.abort();
    revealDescription();
    setIsGenerating(false);
    setSelectedSampleId(sample.id);
    setDescription(sample.description);
    setSource(sample.cnf);
    setGenerationSteps([]);
    setGenerationStatus(null);
    setGenerationReview(null);
    setGenerationPhase(null);
    setGenerationVariables([]);
    setSubmittedDescription("");
    try {
      setCompiled(compileCspSource(sample.cnf));
      setCompileError(null);
    } catch (error) {
      setCompiled(null);
      setCompileError(error instanceof Error ? error.message : String(error));
    }
    controller.reset();
  };

  const handleGenerateCsp = async () => {
    generationAbortRef.current?.abort();
    const abortController = new AbortController();
    generationAbortRef.current = abortController;
    const nextDescription = description;
    setIsGenerating(true);
    setSource("");
    setCompiled(null);
    setGenerationSteps([]);
    setGenerationStatus("Decoding terms and finding variables");
    setGenerationReview(null);
    setGenerationPhase("preparing");
    setGenerationVariables([]);
    setSubmittedDescription(nextDescription);
    try {
      const result = await generateCspFromDescription(nextDescription, {
        signal: abortController.signal,
        onProgress: (event) => {
          if (abortController.signal.aborted) {
            return;
          }

          if (event.type === "conversation") {
            if (event.entry.role === "user") {
              setGenerationPhase("reading");
              setGenerationStatus("Reading the problem and extracting a typed structure");
            } else if (event.entry.role === "browser") {
              setGenerationStatus(statusFromGenerationMessage(event.entry.content));
            }
            return;
          }

          if (event.type === "variables") {
            setGenerationPhase("variables");
            setGenerationVariables(event.variables);
            setGenerationStatus("Step 1 complete — expanding variables into CNF clauses");
            return;
          }

          setSource(event.source);
          try {
            setCompiled(compileCspSource(event.source));
            setCompileError(null);
          } catch {
            setCompiled(null);
          }

          if (event.type === "clause") {
            setGenerationPhase("clauses");
            setGenerationStatus("Expanding typed terms into CNF clauses");
            setGenerationSteps((steps) => [...steps, event.step]);
          } else if (event.review.status === "complete") {
            setGenerationPhase(null);
            setGenerationVariables([]);
            setGenerationStatus(null);
            setGenerationReview("Conversion complete: the extracted structure produced a consistent clause set.");
          } else {
            const missing = event.review.missingConstraints.length
              ? ` Missing: ${event.review.missingConstraints.join("; ")}.`
              : "";
            const wrong = event.review.wrongClauses.length
              ? ` Wrong: ${event.review.wrongClauses.join("; ")}.`
              : "";
            setGenerationReview(`Review requested changes.${missing}${wrong}`);
            if (event.review.correctedClauses.length) {
              setGenerationSteps((steps) => [
                ...steps,
                ...event.review.correctedClauses
              ]);
            }
          }
        }
      });
      if (abortController.signal.aborted) {
        return;
      }
      setSource(result.source);
      setGenerationSteps(result.clauses);
      setCompiled(compileCspSource(result.source));
      setCompileError(null);
      setGenerationPhase(null);
      setGenerationVariables([]);
    } catch (error) {
      if (isAbortError(error)) {
        setCompileError(null);
        setGenerationStatus(null);
        setGenerationReview("Generation cancelled.");
      } else {
        setCompileError(error instanceof Error ? error.message : String(error));
      }
      setGenerationPhase(null);
      setGenerationVariables([]);
    } finally {
      if (generationAbortRef.current === abortController) {
        generationAbortRef.current = null;
        setIsGenerating(false);
      }
    }
  };

  const handleCancelGenerate = () => {
    generationAbortRef.current?.abort();
    generationAbortRef.current = null;
    setIsGenerating(false);
    setGenerationStatus(null);
    setGenerationReview("Generation cancelled.");
    setGenerationPhase(null);
    setGenerationVariables([]);
  };

  const handleStart = () => {
    try {
      const nextCompiled = compileCspSource(source);
      if (!nextCompiled) {
        throw new Error("Enter CSP clauses before starting search.");
      }
      setCompiled(nextCompiled);
      setCompileError(null);
      controller.start(workerCount, nextCompiled, searchStrategy);
    } catch (error) {
      setCompiled(null);
      setCompileError(error instanceof Error ? error.message : String(error));
      controller.reset();
    }
  };

  const handleColumnPointerDown = (
    divider: ColumnDivider,
    event: ReactPointerEvent<HTMLDivElement>
  ) => {
    if (event.pointerType === "mouse" && event.button !== 0) {
      return;
    }

    const widths = measureColumnWidths(setupGridRef.current);
    if (!widths) {
      return;
    }

    event.preventDefault();
    columnResizeRef.current?.cleanup();

    const ownerWindow = event.currentTarget.ownerDocument.defaultView ?? window;
    const resize: ColumnResize = {
      cleanup: () => undefined,
      divider,
      pointerId: event.pointerId,
      startX: event.clientX,
      widths
    };
    const handlePointerMove = (moveEvent: PointerEvent) => {
      if (moveEvent.pointerId !== resize.pointerId) {
        return;
      }
      setColumnWidths(
        resizeAdjacentColumns(
          resize.widths,
          resize.divider,
          moveEvent.clientX - resize.startX
        )
      );
    };
    const finishResize = (endEvent?: PointerEvent) => {
      if (endEvent && endEvent.pointerId !== resize.pointerId) {
        return;
      }
      resize.cleanup();
      if (columnResizeRef.current === resize) {
        columnResizeRef.current = null;
      }
      document.body.classList.remove("is-resizing-columns");
    };
    const finishResizeOnBlur = () => finishResize();
    resize.cleanup = () => {
      ownerWindow.removeEventListener("pointermove", handlePointerMove);
      ownerWindow.removeEventListener("pointerup", finishResize);
      ownerWindow.removeEventListener("pointercancel", finishResize);
      ownerWindow.removeEventListener("blur", finishResizeOnBlur);
    };
    ownerWindow.addEventListener("pointermove", handlePointerMove);
    ownerWindow.addEventListener("pointerup", finishResize);
    ownerWindow.addEventListener("pointercancel", finishResize);
    ownerWindow.addEventListener("blur", finishResizeOnBlur);
    columnResizeRef.current = resize;
    document.body.classList.add("is-resizing-columns");
  };

  const handleColumnResizeKeyDown = (
    divider: ColumnDivider,
    event: ReactKeyboardEvent<HTMLDivElement>
  ) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
      return;
    }

    const widths = measureColumnWidths(setupGridRef.current);
    if (!widths) {
      return;
    }

    event.preventDefault();
    setColumnWidths(
      resizeAdjacentColumns(widths, divider, event.key === "ArrowLeft" ? -24 : 24)
    );
  };

  const isRunning =
    controller.snapshot.status === "running" ||
    controller.snapshot.status === "complete" ||
    controller.snapshot.status === "paused";
  const hasLanguageModel = modelStatus !== "unavailable";

  return (
    <div className="app-shell">
      <header className="app-title-row">
        <div className="app-title-copy">
          <h1>p-adic linear regression</h1>
          <p>Boolean CNF as a local regression dataframe over candidate hyperplanes.</p>
        </div>
        <ModeSwitch mode={mode} onChange={setMode} />
      </header>

      {mode === "sudoku" ? (
        <SudokuMode />
      ) : (
        <>
          <ReadyBand
            compiled={compiled}
            plan={searchPlan}
            searchStrategy={searchStrategy}
            visible={!isRunning && Boolean(compiled) && !isGenerating}
            workerCount={workerCount}
            onSearchStrategyChange={setSearchStrategy}
            onStart={handleStart}
            onWorkerCountChange={setWorkerCount}
          />
          <main
            className="setup-grid"
            ref={setupGridRef}
            style={{
              "--problem-column": `${columnWidths.problem}fr`,
              "--cnf-column": `${columnWidths.cnf}fr`,
              "--regression-column": `${columnWidths.regression}fr`
            } as CSSProperties}
          >
            <ProblemPanel
              source={source}
              sourceVariables={sourceVariables}
              description={description}
              descriptionVisibility={descriptionVisibility}
              generationPhase={generationPhase}
              generationReview={generationReview}
              generationStatus={generationStatus}
              generationSteps={generationSteps}
              generationVariables={generationVariables}
              hasLanguageModel={hasLanguageModel}
              isGenerating={isGenerating}
              onCancelGenerate={handleCancelGenerate}
              onDescriptionChange={setDescription}
              onGenerateCsp={handleGenerateCsp}
              onLoadSample={handleLoadSample}
              onSourceChange={handleSourceChange}
              selectedSampleId={selectedSampleId}
              submittedDescription={submittedDescription}
            />
            <ColumnResizeHandle
              label="Resize CSP and CNF columns"
              value={columnDividerValue(columnWidths, "problem-cnf")}
              onKeyDown={(event) => handleColumnResizeKeyDown("problem-cnf", event)}
              onPointerDown={(event) => handleColumnPointerDown("problem-cnf", event)}
            />
            <TernaryPanel compiled={compiled} error={compileError} />
            <ColumnResizeHandle
              label="Resize CNF and data columns"
              value={columnDividerValue(columnWidths, "cnf-regression")}
              onKeyDown={(event) => handleColumnResizeKeyDown("cnf-regression", event)}
              onPointerDown={(event) => handleColumnPointerDown("cnf-regression", event)}
            />
            <div className="regression-column">
              <RegressionDataFramePanel compiled={compiled} />
              <SearchPlanPanel
                compiled={compiled}
                searchStrategy={searchStrategy}
                workerCount={workerCount}
              />
            </div>
          </main>

          {isRunning && compiled && (
            <RunDashboard
              compiled={compiled}
              resultsRef={resultsRef}
              searchStrategy={searchStrategy}
              workerCount={workerCount}
              controller={controller}
            />
          )}
        </>
      )}

      <Footer compiled={compiled} />
    </div>
  );
}

function ModeSwitch({
  mode,
  onChange
}: {
  mode: Mode;
  onChange: (mode: Mode) => void;
}) {
  return (
    <div className="mode-switch" role="tablist" aria-label="Problem family">
      <button
        type="button"
        role="tab"
        aria-selected={mode === "csp"}
        className={mode === "csp" ? "mode-tab on" : "mode-tab"}
        onClick={() => onChange("csp")}
      >
        <FileText size={18} />
        <span>
          <strong>Boolean CSP/SAT</strong>
          <small>exhaustive search</small>
        </span>
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={mode === "sudoku"}
        className={mode === "sudoku" ? "mode-tab on" : "mode-tab"}
        onClick={() => onChange("sudoku")}
      >
        <Grid3x3 size={18} />
        <span>
          <strong>Sudoku</strong>
          <small>all-different search</small>
        </span>
      </button>
    </div>
  );
}

function ColumnResizeHandle({
  label,
  onKeyDown,
  onPointerDown,
  value
}: {
  label: string;
  onKeyDown: (event: ReactKeyboardEvent<HTMLDivElement>) => void;
  onPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => void;
  value: number;
}) {
  return (
    <div
      aria-label={label}
      aria-orientation="vertical"
      aria-valuemax={100}
      aria-valuemin={0}
      aria-valuenow={value}
      className="column-resize-handle"
      onKeyDown={onKeyDown}
      onPointerDown={onPointerDown}
      role="separator"
      tabIndex={0}
      title="Drag to resize columns"
    />
  );
}

function ProblemPanel({
  source,
  sourceVariables,
  description,
  descriptionVisibility,
  generationPhase,
  generationReview,
  generationStatus,
  generationSteps,
  generationVariables,
  hasLanguageModel,
  isGenerating,
  onCancelGenerate,
  onDescriptionChange,
  onGenerateCsp,
  onLoadSample,
  onSourceChange,
  selectedSampleId,
  submittedDescription
}: {
  source: string;
  sourceVariables: string[];
  description: string;
  descriptionVisibility: DescriptionVisibility;
  generationPhase: GenerationPhase | null;
  generationReview: string | null;
  generationStatus: string | null;
  generationSteps: CnfGenerationStep[];
  generationVariables: string[];
  hasLanguageModel: boolean;
  isGenerating: boolean;
  onCancelGenerate: () => void;
  onDescriptionChange: (value: string) => void;
  onGenerateCsp: () => void;
  onLoadSample: (sampleId: string) => void;
  onSourceChange: (value: string) => void;
  selectedSampleId: string;
  submittedDescription: string;
}) {
  const lineNumbersRef = useRef<HTMLDivElement | null>(null);
  const [showSyntaxHelp, setShowSyntaxHelp] = useState(false);
  const showDescription = descriptionVisibility !== "hidden" && (
    hasLanguageModel || selectedSampleId !== "" || descriptionVisibility === "fading"
  );
  const descriptionClassName = descriptionVisibility === "fading"
    ? "description-box is-fading"
    : "description-box";

  return (
    <section className="panel problem-panel" id="problem">
      <div className="panel-header panel-header-actions">
        <div className="panel-title">
          <FileText size={21} />
          <h2>CSP clauses</h2>
          {!hasLanguageModel && (
            <button
              aria-controls="csp-syntax-help"
              aria-expanded={showSyntaxHelp}
              aria-label="CSP syntax help"
              className="syntax-help-button"
              onClick={() => setShowSyntaxHelp((visible) => !visible)}
              title="CSP syntax help"
              type="button"
            >
              <CircleHelp size={17} />
            </button>
          )}
        </div>
        <div className="problem-header-actions">
          <label className="sample-select">
            <span>Sample</span>
            <select
              aria-label="Sample problem"
              value={selectedSampleId}
              disabled={isGenerating}
              onChange={(event) => onLoadSample(event.target.value)}
            >
              <option value="">Choose a sample</option>
              {CSP_SAMPLES.map((sample: CspSample) => (
                <option key={sample.id} value={sample.id}>
                  {sample.title}{sample.expected === "UNSAT" ? " (UNSAT)" : ""}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      {!hasLanguageModel && showSyntaxHelp && (
        <aside className="syntax-help" id="csp-syntax-help" aria-label="CSP clause syntax" role="note">
          <p>Enter one Boolean constraint per line. Use <code>not</code>, <code>and</code>, <code>or</code>, <code>xor</code>, <code>implies</code> or <code>-&gt;</code>, and parentheses.</p>
          <div className="syntax-examples">
            <code>A or B</code>
            <code>not A or C</code>
            <code>B xor C</code>
            <code>A implies B</code>
            <code>C -&gt; D</code>
          </div>
        </aside>
      )}

      {showDescription && (
        hasLanguageModel ? (
          <div className={descriptionClassName}>
            <div>
              <h3>Enter CSP</h3>
              <p>Ask the local browser model to decode the terms and find the variables.</p>
            </div>
            <div className="generation-actions">
              <button
                className="secondary-button"
                type="button"
                onClick={onGenerateCsp}
                disabled={isGenerating}
              >
                <WandSparkles size={16} />
                {isGenerating ? "Generating..." : "Generate CSP"}
              </button>
              {isGenerating && (
                <button
                  className="secondary-button cancel-generation-button"
                  type="button"
                  onClick={onCancelGenerate}
                >
                  <X size={16} />
                  Cancel
                </button>
              )}
            </div>
            <div
              className={generationPhase === "reading"
                ? "description-input-shell is-reading"
                : "description-input-shell"}
            >
              <textarea
                aria-label="Natural language problem"
                value={description}
                disabled={isGenerating}
                onChange={(event) => onDescriptionChange(event.target.value)}
              />
              {generationPhase === "reading" && (
                <ReadingHighlight description={submittedDescription} />
              )}
            </div>
            <GenerationProgress
              isGenerating={isGenerating}
              review={generationReview}
              status={generationStatus}
              steps={generationSteps}
              variables={generationVariables}
            />
          </div>
        ) : (
          <div className={`${descriptionClassName} sample-description-box`}>
            <div
              aria-label="Sample problem statement"
              className="sample-problem-statement"
              role="note"
            >
              {description}
            </div>
          </div>
        )
      )}

      <div className="editor-shell">
        <div className="line-numbers" aria-hidden="true" ref={lineNumbersRef}>
          {source.split(/\r?\n/u).map((_, index) => (
            <span key={index}>{index + 1}</span>
          ))}
        </div>
        <textarea
          aria-label="CSP source"
          className="code-editor"
          placeholder={hasLanguageModel
            ? undefined
            : "One CSP clause per line\nA or B\nnot A or C\nA implies B\nC -> D"}
          spellCheck={false}
          value={source}
          wrap="off"
          onChange={(event) => onSourceChange(event.target.value)}
          onScroll={(event) => {
            if (lineNumbersRef.current) {
              lineNumbersRef.current.scrollTop = event.currentTarget.scrollTop;
            }
          }}
        />
      </div>
      <div className="panel-foot">
        <span>Lines: {source.split(/\r?\n/u).length}</span>
        <span>Constraints: {countCspConstraints(source)}</span>
        <span>Variables: {sourceVariables.length}</span>
      </div>
      {sourceVariables.length > 0 && (
        <section className="source-variables" aria-labelledby="source-variables-heading">
          <div className="source-variables-heading" id="source-variables-heading">
            <strong>Variables</strong>
            <span>{sourceVariables.length}</span>
          </div>
          <div className="source-variable-list" role="list" aria-label="CSP variables">
            {sourceVariables.map((variable) => (
              <code key={variable} role="listitem">{variable}</code>
            ))}
          </div>
        </section>
      )}
    </section>
  );
}

function GenerationProgress({
  isGenerating,
  review,
  status,
  steps,
  variables
}: {
  isGenerating: boolean;
  review: string | null;
  status: string | null;
  steps: CnfGenerationStep[];
  variables: string[];
}) {
  if (!steps.length && !review && !isGenerating) {
    return null;
  }

  return (
    <div className="generation-progress" aria-label="CNF generation progress">
      {isGenerating && (
        <div
          className="generation-active"
          role="status"
          aria-live="polite"
          aria-atomic="true"
        >
          <div className="generation-swoosh" aria-hidden="true">
            <span />
          </div>
          <strong>{status ?? "Decoding terms and finding variables"}</strong>
        </div>
      )}
      {variables.length > 0 && (
        <div
          className="generation-variables"
          role="status"
          aria-live="polite"
          aria-atomic="true"
        >
          <div className="generation-variables-heading">
            <span><Check size={14} /> Step 1 complete</span>
            <strong>
              {variables.length} Boolean variable{variables.length === 1 ? "" : "s"} found
            </strong>
          </div>
          <div className="generation-variable-list">
            {variables.slice(0, 32).map((variable) => (
              <code key={variable}>{variable}</code>
            ))}
            {variables.length > 32 && <span>+{variables.length - 32} more</span>}
          </div>
        </div>
      )}
      {steps.map((step, index) => (
        <div className="generation-step" key={`${step.cnf}-${index}`}>
          <span>{index + 1}</span>
          <strong>{step.purpose}</strong>
          <code>{step.cnf}</code>
        </div>
      ))}
      {review && <div className="generation-review">{review}</div>}
    </div>
  );
}

const READING_GROUP_SIZES = [2, 1, 3, 2] as const;

function ReadingHighlight({ description }: { description: string }) {
  const tokens = useMemo(() => tokenizeReadingText(description), [description]);
  const wordCount = tokens.reduce(
    (count, token) => count + (token.wordIndex == null ? 0 : 1),
    0
  );
  const [scan, setScan] = useState({ start: 0, turn: 0 });

  useEffect(() => {
    setScan({ start: 0, turn: 0 });
    if (wordCount < 2 || prefersReducedMotion()) {
      return;
    }

    const intervalId = window.setInterval(() => {
      setScan((current) => {
        const currentSize = READING_GROUP_SIZES[current.turn % READING_GROUP_SIZES.length];
        return {
          start: (current.start + currentSize) % wordCount,
          turn: current.turn + 1
        };
      });
    }, 340);

    return () => window.clearInterval(intervalId);
  }, [wordCount]);

  const groupSize = Math.min(
    READING_GROUP_SIZES[scan.turn % READING_GROUP_SIZES.length],
    wordCount
  );

  return (
    <div className="generation-reading-text" aria-hidden="true">
      {tokens.map((token, index) => {
        if (token.wordIndex == null) {
          return token.text;
        }
        const distance = (token.wordIndex - scan.start + wordCount) % wordCount;
        return (
          <span
            className={distance < groupSize
              ? "generation-reading-token is-focused"
              : "generation-reading-token"}
            key={`${token.text}-${index}`}
          >
            {token.text}
          </span>
        );
      })}
    </div>
  );
}

function tokenizeReadingText(description: string): Array<{
  text: string;
  wordIndex: number | null;
}> {
  let wordIndex = 0;
  return description.split(/(\s+)/u).filter(Boolean).map((text) => {
    if (/^\s+$/u.test(text)) {
      return { text, wordIndex: null };
    }
    const token = { text, wordIndex };
    wordIndex += 1;
    return token;
  });
}

function prefersReducedMotion(): boolean {
  return typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function TernaryPanel({
  compiled,
  error
}: {
  compiled: CompiledProblem | null;
  error: string | null;
}) {
  return (
    <section className="panel cnf-panel" id="clauses">
      <PanelHeader icon={<Triangle size={21} />} title="CNF view" />
      {error ? (
        <div className="error-box">{error}</div>
      ) : compiled ? (
        <>
          <div className="panel-toolbar">
            <strong>CNF clauses</strong>
            <span>{compiled.ternaryClauses.length} clauses</span>
          </div>
          <div className="clause-list">
            {compiled.ternaryClauses.map((clause) => (
              <div className="clause-row" key={clause.id}>
                <span>{clause.id}</span>
                <div className="clause-forms">
                  <code>{renderClause(clause)}</code>
                  <code className="affine" title="p-adic clause residual">
                    {renderClauseAffine(clause)}
                  </code>
                </div>
              </div>
            ))}
          </div>
          <ValidationList compiled={compiled} />
        </>
      ) : (
        <EmptyState text="Enter valid CSP clauses to see the compiled form." />
      )}
    </section>
  );
}

function RegressionDataFramePanel({
  compiled
}: {
  compiled: CompiledProblem | null;
}) {
  const rows = useMemo(
    () => (compiled ? buildRegressionDataFrame(compiled) : []),
    [compiled]
  );

  return (
    <section className="panel dataframe-panel" aria-label="p-adic linear regression dataframe">
      <PanelHeader icon={<FlaskConical size={21} />} title="p-adic linear regression dataframe" />
      {compiled ? (
        <>
          <div className="dataframe-scroll">
            <table className="dataframe-table">
              <thead>
                <tr>
                  <th className="row-label">row</th>
                  {compiled.variables.map((variable) => (
                    <th className="feature-heading" key={variable.name}>
                      <span>{variable.name}</span>
                    </th>
                  ))}
                  <th className="target-heading">target</th>
                  <th className="weight-heading">signed<br />weight</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr className={`dataframe-row row-${row.kind}`} key={row.id}>
                    <th title={row.source}>{row.label}</th>
                    {compiled.variables.map((variable) => {
                      const value = row.coefficients[variable.name] ?? 0;
                      return (
                        <td className={value < 0 ? "negative-cell" : undefined} key={variable.name}>
                          {value}
                        </td>
                      );
                    })}
                    <td className="target-cell">{row.relation} {row.target}</td>
                    <td
                      className={row.sign < 0
                        ? "signed-weight-cell negative-weight-cell"
                        : "signed-weight-cell positive-weight-cell"}
                    >
                      {row.sign < 0 ? "−" : "+"}{row.weight}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="dataframe-legend">
            <span><i className="legend-constraint" /> CNF constraints</span>
            <span><i className="legend-unit" /> Unit wells</span>
            <span><i className="legend-negative" /> Negative clause reward</span>
          </div>
        </>
      ) : (
        <EmptyState text="Enter valid CSP clauses to see the regression dataframe." />
      )}
    </section>
  );
}

function SearchPlanPanel({
  compiled,
  searchStrategy,
  workerCount
}: {
  compiled: CompiledProblem | null;
  searchStrategy: SearchStrategy;
  workerCount: number;
}) {
  const plan = compiled ? createSearchPlan(compiled, workerCount, searchStrategy) : null;

  return (
    <section className="panel search-plan" id="search">
      <PanelHeader icon={<Search size={21} />} title="Search plan" />
      {compiled && plan ? (
        <>
          <MetricRow label="Coefficients" value={`n = ${compiled.variables.length}`} />
          <MetricRow
            label="Candidate hyperplanes"
            value={`2^${compiled.variables.length} = ${formatAssignmentCount(compiled.variables.length)}`}
          />
          <MetricRow
            label="Strategy"
            value={searchStrategy === "random"
              ? "Random-permutation exhaustive search"
              : "Ordered exhaustive search"}
            accent
          />
          <MetricRow label="Thread split" value="Disjoint hyperplane batches" />
          <MetricRow label="Clause reward" value="|u·x − t|ₚ  (p = 17)" />
          <MetricRow
            label="Unit-well weight"
            value={`α = ${compiled.scoring.unitWellWeight} = clauses + 1`}
          />
          <MetricRow label="Regression loss" value="αn − satisfied clauses" />
          <MetricRow
            label="Satisfiable floor"
            value={`${compiled.scoring.theoreticalFloor} = αn − clauses`}
          />
          <MetricRow label="Success criterion" value="loss reaches the satisfiable floor" />
        </>
      ) : (
        <EmptyState text="No compiled problem yet." />
      )}
    </section>
  );
}

function ReadyBand({
  compiled,
  plan,
  searchStrategy,
  visible,
  workerCount,
  onSearchStrategyChange,
  onStart,
  onWorkerCountChange
}: {
  compiled: CompiledProblem | null;
  plan: ReturnType<typeof createSearchPlan> | null;
  searchStrategy: SearchStrategy;
  visible: boolean;
  workerCount: number;
  onSearchStrategyChange: (value: SearchStrategy) => void;
  onStart: () => void;
  onWorkerCountChange: (value: number) => void;
}) {
  return (
    <section
      aria-hidden={!visible}
      aria-label="Browser search controls"
      className={visible ? "ready-band is-visible" : "ready-band"}
    >
      <div className="ready-copy">
        <FlaskConical size={30} />
        <div>
          <h2>Ready to solve.</h2>
        </div>
      </div>
      <label className="strategy-select">
        <span>Search order</span>
        <select
          aria-label="Search strategy"
          disabled={!visible}
          value={searchStrategy}
          onChange={(event) => onSearchStrategyChange(event.target.value as SearchStrategy)}
        >
          <option value="ordered">Ordered</option>
          <option value="random">Random permutation</option>
        </select>
      </label>
      <div className="stepper">
        <span>Threads</span>
        <button
          type="button"
          disabled={!visible}
          onClick={() => onWorkerCountChange(Math.max(1, workerCount - 1))}
        >
          -
        </button>
        <strong>{workerCount}</strong>
        <button
          type="button"
          disabled={!visible}
          onClick={() => onWorkerCountChange(Math.min(12, workerCount + 1))}
        >
          +
        </button>
      </div>
      <div className="runtime-card">
        <span>Hyperplanes</span>
        <strong>{plan ? plan.assignmentCount.toLocaleString() : "-"}</strong>
        <small>split across {workerCount} threads</small>
      </div>
      <button
        className="run-button"
        type="button"
        onClick={onStart}
        disabled={!compiled || !visible}
      >
        <Play size={18} />
        Solve p-adic linear regression
      </button>
    </section>
  );
}

function RunDashboard({
  compiled,
  resultsRef,
  searchStrategy,
  workerCount,
  controller
}: {
  compiled: CompiledProblem;
  resultsRef: RefObject<HTMLElement | null>;
  searchStrategy: SearchStrategy;
  workerCount: number;
  controller: ReturnType<typeof useSearchController>;
}) {
  const { snapshot, bestAssignment } = controller;
  const assignmentCount = compiled.assignmentCount;
  const progress = assignmentCount
    ? Math.min(snapshot.totalTested / assignmentCount, 1)
    : 0;
  const validation = bestAssignment
    ? evaluateAssignment(compiled, bestAssignment)
    : null;
  const regressionEvaluation = bestAssignment
    ? evaluateRegressionDataFrame(compiled, bestAssignment)
    : null;
  const isComplete = snapshot.status === "complete";
  const isPaused = snapshot.status === "paused";
  const solutionEquation = bestAssignment
    ? formatRegressionEquation(compiled, bestAssignment)
    : null;
  const regressionFloor = compiled.scoring.theoreticalFloor;

  return (
    <section className="run-grid" aria-label="Search results" ref={resultsRef}>
      <section className="panel worker-dashboard">
        <div className="run-stats">
          <Stat label="Hyperplanes tested" value={`${snapshot.totalTested.toLocaleString()} / ${assignmentCount.toLocaleString()}`} />
          <Stat label="Total speed" value={`${formatRate(snapshot.totalSpeed)} hyperplanes/s`} />
          <Stat label="Best loss" value={`${snapshot.bestLoss ?? "-"}`} />
          <Stat label="Floor hyperplanes" value={`${snapshot.solutions}`} />
        </div>
        <div className="global-progress">
          <span style={{ width: `${progress * 100}%` }} />
        </div>
        <WorkerTable compiled={compiled} lanes={snapshot.lanes} />
        <LossChart floor={regressionFloor} history={snapshot.history} />
        <section className="solution-section" aria-label="Best p-adic regression solution">
          <PanelHeader icon={<ShieldCheck size={20} />} title="Best p-adic regression solution" />
          {bestAssignment ? (
            <>
              {solutionEquation && (
                <div className="solution-equation">
                  <span>Boolean assignment (1 = selected)</span>
                  <code>{solutionEquation}</code>
                </div>
              )}
              <div className="solution-list">
                {compiled.variables.map((variable) => (
                  <div className="solution-row" key={variable.name}>
                    <strong>{variable.name}</strong>
                    <span>{bestAssignment[variable.name] ? 1 : 0}</span>
                  </div>
                ))}
              </div>
              <div className="constraint-check">
                <MetricRow
                  label="Satisfiable floor"
                  value={`${validation?.theoreticalFloor ?? regressionFloor}`}
                />
                <MetricRow
                  label="Linear constraints satisfied"
                  value={`${validation?.nonUnitSatisfied ?? 0} / ${compiled.constraints.length}`}
                />
                <MetricRow
                  label="Regression loss"
                  value={`${regressionEvaluation?.totalLoss ?? validation?.loss ?? "-"}`}
                />
              </div>
              {regressionEvaluation && (
                <RegressionEvaluationTable
                  evaluation={regressionEvaluation}
                  prime={compiled.scoring.prime}
                />
              )}
            </>
          ) : (
            <EmptyState text="Best coefficient vector will appear after the first thread update." />
          )}
          <div
            className={isComplete ? "run-actions complete-actions" : "run-actions"}
            aria-label="Search controls"
          >
            {isComplete ? (
              <>
                <span className="run-status complete">
                  <Check size={17} /> Search complete
                </span>
                <button className="secondary-button" type="button" onClick={controller.reset}>
                  <RotateCcw size={17} /> Clear results
                </button>
              </>
            ) : isPaused ? (
              <>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => controller.start(workerCount, compiled, searchStrategy)}
                >
                  <Play size={17} /> Restart search
                </button>
                <button className="danger-button" type="button" onClick={controller.reset}>
                  <Square size={15} /> Stop
                </button>
              </>
            ) : (
              <>
                <button className="secondary-button" type="button" onClick={controller.pause}>
                  <CirclePause size={17} /> Pause
                </button>
                <button className="danger-button" type="button" onClick={controller.reset}>
                  <Square size={15} /> Stop
                </button>
              </>
            )}
            <button className="proof-button" type="button">
              <Download size={17} /> Export proof (JSON)
            </button>
          </div>
        </section>
      </section>

      <section className="panel log-panel">
        <PanelHeader icon={<Code2 size={20} />} title="Search log" />
        <pre>
          {snapshot.logs.length
            ? snapshot.logs.map((entry) => entry.text).join("\n")
            : "[sys] waiting for thread updates..."}
        </pre>
      </section>

      <section className="panel split-panel">
        <PanelHeader icon={<RotateCcw size={20} />} title="Hyperplane batches" />
        <p>Each candidate hyperplane is a coefficient vector. Threads scan disjoint hyperplane batches.</p>
        <div className="basis-diagram">
          {compiled.variables.slice(0, 8).map((variable) => (
            <span key={variable.name}>
              coordinate {variable.index + 1}
              <strong>{variable.name}</strong>
            </span>
          ))}
        </div>
        <div className="hyperplane-table">
          {snapshot.lanes.map((lane) => (
            <div key={lane.workerId}>
              <strong>T{lane.workerId}</strong>
              <span>H{lane.start.toLocaleString()}</span>
              <span>H{(lane.endExclusive - 1).toLocaleString()}</span>
            </div>
          ))}
        </div>
      </section>
    </section>
  );
}

function WorkerTable({
  compiled,
  lanes
}: {
  compiled: CompiledProblem;
  lanes: ReturnType<typeof useSearchController>["snapshot"]["lanes"];
}) {
  return (
    <div className="worker-table" role="table" aria-label="Thread hyperplane batches">
      <div className="worker-row worker-head" role="row">
        <span>Thread</span>
        <span>Hyperplane batch</span>
        <span>Current hyperplane</span>
        <span>Hyperplanes/sec</span>
        <span>Progress</span>
        <span>Best loss</span>
      </div>
      {lanes.map((lane) => {
        const progress =
          (lane.currentMask - lane.start) / Math.max(lane.endExclusive - lane.start, 1);
        return (
          <div className="worker-row" key={lane.workerId} role="row">
            <span className="worker-id">T{lane.workerId}</span>
            <span>
              H{lane.start.toLocaleString()} - H{(lane.endExclusive - 1).toLocaleString()}
            </span>
            <span>H{lane.currentMask.toLocaleString()}</span>
            <span>{formatRate(lane.speed)}</span>
            <span className="lane-progress">
              <i style={{ width: `${Math.max(0, Math.min(progress, 1)) * 100}%` }} />
            </span>
            <span>{lane.bestLoss ?? "-"}</span>
          </div>
        );
      })}
      {!lanes.length && (
        <div className="worker-row">
          <span>Waiting</span>
          <span>H0 - H{compiled.assignmentCount - 1}</span>
          <span>-</span>
          <span>-</span>
          <span className="lane-progress" />
          <span>-</span>
        </div>
      )}
    </div>
  );
}

function RegressionEvaluationTable({
  evaluation,
  prime
}: {
  evaluation: ReturnType<typeof evaluateRegressionDataFrame>;
  prime: number;
}) {
  return (
    <section className="evaluation-section" aria-label="Regression row evaluation">
      <div className="evaluation-heading">
        <div>
          <h3>Regression row evaluation</h3>
          <p>Affine coordinates use x = 0 for true CSP variables and x = 1 for false.</p>
        </div>
        <strong>Total L = {evaluation.totalLoss}</strong>
      </div>
      <div className="evaluation-scroll">
        <table className="evaluation-table" aria-label="Regression row contributions">
          <thead>
            <tr>
              <th>row</th>
              <th>observation</th>
              <th>u·x</th>
              <th>target</th>
              <th>residual</th>
              <th>|r|<sub>{prime}</sub></th>
              <th>signed weight</th>
              <th>contribution to L</th>
              <th>result</th>
            </tr>
          </thead>
          <tbody>
            {evaluation.rows.map((row) => (
              <tr className={`evaluation-row row-${row.kind}`} key={row.id}>
                <th>{row.label}</th>
                <td className="evaluation-observation" title={row.source}>{row.source}</td>
                <td>{row.affineValue}</td>
                <td>{row.relation} {row.target}</td>
                <td>{formatSignedNumber(row.residual)}</td>
                <td>{row.pAdicNorm}</td>
                <td className={row.signedWeight < 0 ? "negative-evaluation" : undefined}>
                  {formatSignedNumber(row.signedWeight)}
                </td>
                <td className={row.contribution < 0 ? "negative-evaluation" : undefined}>
                  {formatSignedNumber(row.contribution)}
                </td>
                <td><span className={`evaluation-status status-${row.status}`}>{formatEvaluationStatus(row.status)}</span></td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <th colSpan={7}>Sum of signed row contributions</th>
              <td>{evaluation.totalLoss}</td>
              <td>total loss</td>
            </tr>
          </tfoot>
        </table>
      </div>
    </section>
  );
}

function LossChart({
  floor,
  history
}: {
  floor: number;
  history: Array<{ tested: number; loss: number }>;
}) {
  const floorIndex = history.findIndex((point) => point.loss <= floor);
  const visibleHistory = floorIndex >= 0 ? history.slice(0, floorIndex + 1) : history;
  const points = visibleHistory.length
    ? visibleHistory
    : [{ tested: 0, loss: Math.max(floor, 1) }];
  const maxLoss = Math.max(floor + 1, ...points.map((point) => point.loss));
  const width = 760;
  const height = 200;
  const plot = { left: 62, right: 18, top: 18, bottom: 40 };
  const plotWidth = width - plot.left - plot.right;
  const plotHeight = height - plot.top - plot.bottom;
  const lossRange = maxLoss - floor;
  const testedMax = Math.max(1, points[points.length - 1].tested);
  const yForLoss = (loss: number) =>
    plot.top + ((maxLoss - loss) / lossRange) * plotHeight;
  const tickStep = Math.max(1, Math.ceil(lossRange / 4));
  const yTicks: number[] = [];
  for (let value = floor; value <= maxLoss; value += tickStep) {
    yTicks.push(value);
  }
  if (yTicks[yTicks.length - 1] !== maxLoss) {
    yTicks.push(maxLoss);
  }
  const path = points
    .map((point, index) => {
      const x = plot.left + (point.tested / testedMax) * plotWidth;
      const y = yForLoss(Math.max(floor, Math.min(point.loss, maxLoss)));
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <div className="loss-chart">
      <div className="chart-title">
        <strong>p-adic loss over time</strong>
        <span>lower is better</span>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="p-adic loss over time"
        data-points={points.length}
        data-last-tested={points[points.length - 1].tested}
        data-hit-floor={floorIndex >= 0 ? "true" : "false"}
      >
        {yTicks.map((value) => {
          const y = yForLoss(value);
          return (
            <g key={value}>
              <line
                x1={plot.left}
                x2={width - plot.right}
                y1={y}
                y2={y}
                className={value === floor ? "floor-line" : "chart-grid-line"}
              />
              <text className="axis-label y-axis-label" x={plot.left - 10} y={y + 4} textAnchor="end">
                {value}
              </text>
            </g>
          );
        })}
        <line
          className="chart-axis"
          x1={plot.left}
          x2={plot.left}
          y1={plot.top}
          y2={height - plot.bottom}
        />
        <line
          className="chart-axis"
          x1={plot.left}
          x2={width - plot.right}
          y1={height - plot.bottom}
          y2={height - plot.bottom}
        />
        <path d={path} className="loss-line" />
        <text className="floor-label" x={plot.left + 10} y={height - plot.bottom - 9}>
          satisfiable floor = {floor}
        </text>
        <text className="axis-label x-axis-label" x={plot.left} y={height - 10} textAnchor="start">
          0 hyperplanes
        </text>
        <text className="axis-label x-axis-label" x={width - plot.right} y={height - 10} textAnchor="end">
          {points[points.length - 1].tested.toLocaleString()} tested
        </text>
      </svg>
    </div>
  );
}

function formatSignedNumber(value: number): string {
  if (value < 0) {
    return `−${Math.abs(value)}`;
  }
  if (value > 0) {
    return `+${value}`;
  }
  return "0";
}

function formatEvaluationStatus(
  status: ReturnType<typeof evaluateRegressionDataFrame>["rows"][number]["status"]
): string {
  switch (status) {
    case "satisfied":
      return "satisfied";
    case "violated":
      return "violated";
    case "at-target":
      return "at target";
    case "away-from-target":
      return "unit residual";
  }
}

function ValidationList({ compiled }: { compiled: CompiledProblem }) {
  return (
    <div className="validation-list">
      <strong>Validation</strong>
      <span>
        <Check size={15} /> Source clauses preserved
      </span>
      <span>
        <Check size={15} /> Each clause length {"<="} {compiled.validation.maxClauseWidth}
      </span>
      <span>
        <Check size={15} /> Ready for p-adic loss evaluation
      </span>
    </div>
  );
}

function PanelHeader({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <div className="panel-header">
      {icon}
      <h2>{title}</h2>
    </div>
  );
}

function MetricRow({
  label,
  value,
  accent = false
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div className="metric-row">
      <span>{label}</span>
      <strong className={accent ? "accent" : undefined}>{value}</strong>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return <div className="empty-state">{text}</div>;
}

function Footer({ compiled }: { compiled: CompiledProblem | null }) {
  return (
    <footer className="footer">
      <span className="footer-status">
        <span className="green-dot" /> All systems ready
      </span>
      <span>CSP</span>
      <span>Ternary</span>
      <span>p-adic loss</span>
      {compiled && <span>2^{compiled.variables.length} candidate hyperplanes</span>}
      <span className="footer-lock">All computation stays in your browser.</span>
    </footer>
  );
}

function statusFromGenerationMessage(message: string): string {
  if (/download/i.test(message)) {
    return "Preparing the local model";
  }
  if (/session ready/i.test(message)) {
    return "Decoding terms and finding variables";
  }
  if (/availability|creating/i.test(message)) {
    return "Checking the local model";
  }
  if (/typed JSON schema|assignment CSP/i.test(message)) {
    return "Decoding terms and finding variables";
  }
  if (/^\s*\{/.test(message) || /```json/i.test(message)) {
    return "Compiling typed terms into CNF";
  }

  return "Decoding terms and finding variables";
}

function measureColumnWidths(grid: HTMLElement | null): ColumnWidths | null {
  if (!grid) {
    return null;
  }

  const problem = grid.querySelector<HTMLElement>(".problem-panel");
  const cnf = grid.querySelector<HTMLElement>(".cnf-panel");
  const regression = grid.querySelector<HTMLElement>(".regression-column");
  if (!problem || !cnf || !regression) {
    return null;
  }

  const widths = {
    problem: problem.getBoundingClientRect().width,
    cnf: cnf.getBoundingClientRect().width,
    regression: regression.getBoundingClientRect().width
  };

  return Object.values(widths).every((width) => width > 0) ? widths : null;
}

function resizeAdjacentColumns(
  widths: ColumnWidths,
  divider: ColumnDivider,
  delta: number
): ColumnWidths {
  if (divider === "problem-cnf") {
    const pairWidth = widths.problem + widths.cnf;
    const problem = clamp(
      widths.problem + delta,
      MIN_COLUMN_WIDTHS.problem,
      pairWidth - MIN_COLUMN_WIDTHS.cnf
    );
    return {
      problem,
      cnf: pairWidth - problem,
      regression: widths.regression
    };
  }

  const pairWidth = widths.cnf + widths.regression;
  const cnf = clamp(
    widths.cnf + delta,
    MIN_COLUMN_WIDTHS.cnf,
    pairWidth - MIN_COLUMN_WIDTHS.regression
  );
  return {
    problem: widths.problem,
    cnf,
    regression: pairWidth - cnf
  };
}

function columnDividerValue(widths: ColumnWidths, divider: ColumnDivider): number {
  const leading = divider === "problem-cnf" ? widths.problem : widths.cnf;
  const trailing = divider === "problem-cnf" ? widths.cnf : widths.regression;
  return Math.round((leading / (leading + trailing)) * 100);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum);
}

function isAbortError(error: unknown): boolean {
  return (
    error instanceof Error &&
    (error.name === "AbortError" || /cancelled|aborted/iu.test(error.message))
  );
}

function formatRegressionEquation(
  compiled: CompiledProblem,
  coefficients: Record<string, boolean>
): string {
  const terms = compiled.variables.map((variable) => {
    const coefficient = coefficients[variable.name] ? 1 : 0;
    return `${coefficient} * ${variable.name}`;
  });
  return `y = ${terms.join(" + ")}`;
}

function formatRate(value: number): string {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(2)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K`;
  }
  return value.toFixed(0);
}

function inferSourceVariableNames(
  source: string,
  compiled: CompiledProblem | null
): string[] {
  if (compiled) {
    return compiled.variables.map((variable) => variable.name);
  }

  const names = new Set<string>();
  for (const line of source.split(/\r?\n/u)) {
    if (!line.replace(/#.*/u, "").trim()) {
      continue;
    }
    try {
      for (const variable of parseProblem(line).variables) {
        names.add(variable.name);
      }
    } catch {
      // Keep variables from the other valid lines during an incomplete edit.
    }
  }
  return Array.from(names);
}

function compileCspSource(source: string): CompiledProblem | null {
  return source.trim() ? compileProblem(source) : null;
}

export default App;
