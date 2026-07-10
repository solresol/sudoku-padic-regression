import {
  Check,
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
  DEFAULT_ASSIGNMENT_PROBLEM,
  countCspConstraints
} from "./lib/defaultProblems";
import { createSearchPlan, formatAssignmentCount } from "./lib/search";
import { useSearchController } from "./hooks/useSearchController";
import SudokuMode from "./SudokuMode";

type Mode = "csp" | "sudoku";
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

function App() {
  const [mode, setMode] = useState<Mode>("csp");
  const [columnWidths, setColumnWidths] = useState<ColumnWidths>(DEFAULT_COLUMN_WIDTHS);
  const [source, setSource] = useState("");
  const [description, setDescription] = useState(DEFAULT_ASSIGNMENT_PROBLEM);
  const [compiled, setCompiled] = useState<CompiledProblem | null>(null);
  const [compileError, setCompileError] = useState<string | null>(null);
  const [workerCount, setWorkerCount] = useState(2);
  const [modelStatus, setModelStatus] =
    useState<LanguageModelAvailability>("unavailable");
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationSteps, setGenerationSteps] = useState<CnfGenerationStep[]>([]);
  const [generationStatus, setGenerationStatus] = useState<string | null>(null);
  const [generationReview, setGenerationReview] = useState<string | null>(null);
  const generationAbortRef = useRef<AbortController | null>(null);
  const setupGridRef = useRef<HTMLElement | null>(null);
  const columnResizeRef = useRef<ColumnResize | null>(null);
  const controller = useSearchController(compiled);

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
    return () => {
      columnResizeRef.current?.cleanup();
      columnResizeRef.current = null;
      document.body.classList.remove("is-resizing-columns");
    };
  }, []);

  const searchPlan = useMemo(
    () => (compiled ? createSearchPlan(compiled, workerCount) : null),
    [compiled, workerCount]
  );

  const handleSourceChange = (value: string) => {
    setSource(value);
    setGenerationReview(null);
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

  const handleLoadDefaultProblem = () => {
    setDescription(DEFAULT_ASSIGNMENT_PROBLEM);
    setSource("");
    setGenerationSteps([]);
    setGenerationStatus(null);
    setGenerationReview(null);
    setCompiled(null);
    setCompileError(null);
    controller.reset();
  };

  const handleGenerateCsp = async () => {
    generationAbortRef.current?.abort();
    const abortController = new AbortController();
    generationAbortRef.current = abortController;
    setIsGenerating(true);
    setSource("");
    setCompiled(null);
    setGenerationSteps([]);
    setGenerationStatus("Decoding terms and finding variables");
    setGenerationReview(null);
    try {
      const result = await generateCspFromDescription(description, {
        signal: abortController.signal,
        onProgress: (event) => {
          if (abortController.signal.aborted) {
            return;
          }

          if (event.type === "conversation") {
            setGenerationStatus(statusFromGenerationMessage(event.entry.content));
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
            setGenerationStatus("Expanding typed terms into CNF clauses");
            setGenerationSteps((steps) => [...steps, event.step]);
          } else if (event.review.status === "complete") {
            setGenerationStatus(null);
            setGenerationReview("Review complete: the model found no missing or wrong clauses.");
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
    } catch (error) {
      if (isAbortError(error)) {
        setCompileError(null);
        setGenerationStatus(null);
        setGenerationReview("Generation cancelled.");
      } else {
        setCompileError(error instanceof Error ? error.message : String(error));
      }
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
  };

  const handleStart = () => {
    try {
      const nextCompiled = compileCspSource(source);
      if (!nextCompiled) {
        throw new Error("Enter CSP clauses before starting search.");
      }
      setCompiled(nextCompiled);
      setCompileError(null);
      controller.start(workerCount, nextCompiled);
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
          {isRunning && compiled ? (
            <RunDashboard
              compiled={compiled}
              workerCount={workerCount}
              controller={controller}
            />
          ) : (
            <>
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
                  description={description}
                  generationReview={generationReview}
                  generationStatus={generationStatus}
                  generationSteps={generationSteps}
                  hasLanguageModel={hasLanguageModel}
                  isGenerating={isGenerating}
                  onCancelGenerate={handleCancelGenerate}
                  onDescriptionChange={setDescription}
                  onGenerateCsp={handleGenerateCsp}
                  onLoadDefaultProblem={handleLoadDefaultProblem}
                  onSourceChange={handleSourceChange}
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
                  <SearchPlanPanel compiled={compiled} workerCount={workerCount} />
                </div>
              </main>

              <ReadyBand
                compiled={compiled}
                plan={searchPlan}
                workerCount={workerCount}
                onStart={handleStart}
                onWorkerCountChange={setWorkerCount}
              />
            </>
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
  description,
  generationReview,
  generationStatus,
  generationSteps,
  hasLanguageModel,
  isGenerating,
  onCancelGenerate,
  onDescriptionChange,
  onGenerateCsp,
  onLoadDefaultProblem,
  onSourceChange
}: {
  source: string;
  description: string;
  generationReview: string | null;
  generationStatus: string | null;
  generationSteps: CnfGenerationStep[];
  hasLanguageModel: boolean;
  isGenerating: boolean;
  onCancelGenerate: () => void;
  onDescriptionChange: (value: string) => void;
  onGenerateCsp: () => void;
  onLoadDefaultProblem: () => void;
  onSourceChange: (value: string) => void;
}) {
  return (
    <section className="panel problem-panel" id="problem">
      <div className="panel-header panel-header-actions">
        <div className="panel-title">
          <FileText size={21} />
          <h2>CSP clauses</h2>
        </div>
        <button
          className="secondary-button"
          type="button"
          onClick={onLoadDefaultProblem}
        >
          <RotateCcw size={16} />
          Default problem
        </button>
      </div>

      {hasLanguageModel && (
        <div className="description-box">
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
          <textarea
            aria-label="Natural language problem"
            value={description}
            onChange={(event) => onDescriptionChange(event.target.value)}
          />
          <GenerationProgress
            isGenerating={isGenerating}
            review={generationReview}
            status={generationStatus}
            steps={generationSteps}
          />
        </div>
      )}

      <div className="editor-shell">
        <div className="line-numbers" aria-hidden="true">
          {source.split(/\r?\n/u).map((_, index) => (
            <span key={index}>{index + 1}</span>
          ))}
        </div>
        <textarea
          aria-label="CSP source"
          className="code-editor"
          spellCheck={false}
          value={source}
          onChange={(event) => onSourceChange(event.target.value)}
        />
      </div>
      <div className="panel-foot">
        <span>Lines: {source.split(/\r?\n/u).length}</span>
        <span>Constraints: {countCspConstraints(source)}</span>
        <span>Variables are inferred</span>
      </div>
    </section>
  );
}

function GenerationProgress({
  isGenerating,
  review,
  status,
  steps
}: {
  isGenerating: boolean;
  review: string | null;
  status: string | null;
  steps: CnfGenerationStep[];
}) {
  if (!steps.length && !review && !isGenerating) {
    return null;
  }

  return (
    <div className="generation-progress" aria-label="CNF generation progress">
      {isGenerating && (
        <div className="generation-active">
          <div className="generation-swoosh" aria-hidden="true">
            <span />
          </div>
          <strong>{status ?? "Decoding terms and finding variables"}</strong>
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
    <section className="panel dataframe-panel" aria-label="CNF regression dataframe">
      <PanelHeader icon={<FlaskConical size={21} />} title="CNF regression dataframe" />
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
                    <td className="target-cell">{row.target}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="dataframe-legend">
            <span><i className="legend-constraint" /> CNF constraints</span>
            <span><i className="legend-unit" /> Unit wells</span>
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
  workerCount
}: {
  compiled: CompiledProblem | null;
  workerCount: number;
}) {
  const plan = compiled ? createSearchPlan(compiled, workerCount) : null;

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
          <MetricRow label="Strategy" value="Exhaustive p-adic hyperplane search" accent />
          <MetricRow label="Thread split" value="Disjoint hyperplane batches" />
          <MetricRow label="Clause reward" value="|u·x − t|ₚ  (p = 17)" />
          <MetricRow label="Regression loss" value="# constraints with reward 0" />
          <MetricRow
            label="Loss floor"
            value={`${compiled.scoring.theoreticalFloor} (one unit well per coefficient)`}
          />
          <MetricRow label="Success criterion" value="loss reaches the unit-well floor" />
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
  workerCount,
  onStart,
  onWorkerCountChange
}: {
  compiled: CompiledProblem | null;
  plan: ReturnType<typeof createSearchPlan> | null;
  workerCount: number;
  onStart: () => void;
  onWorkerCountChange: (value: number) => void;
}) {
  return (
    <section className="ready-band">
      <div className="ready-copy">
        <FlaskConical size={30} />
        <div>
          <h2>Ready to solve in this browser</h2>
          <p>Threads evaluate disjoint batches of candidate p-adic hyperplanes locally.</p>
        </div>
      </div>
      <div className="stepper">
        <span>Threads</span>
        <button
          type="button"
          onClick={() => onWorkerCountChange(Math.max(1, workerCount - 1))}
        >
          -
        </button>
        <strong>{workerCount}</strong>
        <button
          type="button"
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
        disabled={!compiled}
      >
        <Play size={18} />
        Solve p-adic regression
      </button>
    </section>
  );
}

function RunDashboard({
  compiled,
  workerCount,
  controller
}: {
  compiled: CompiledProblem;
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
  const isComplete = snapshot.status === "complete";
  const isPaused = snapshot.status === "paused";
  const solutionEquation = bestAssignment
    ? formatRegressionEquation(compiled, bestAssignment)
    : null;
  const regressionFloor = compiled.scoring.theoreticalFloor;
  const bestRegressionLoss =
    snapshot.bestLoss == null ? null : snapshot.bestLoss + regressionFloor;
  const regressionHistory = snapshot.history.map((point) => ({
    ...point,
    loss: point.loss + regressionFloor
  }));

  return (
    <main className="run-grid">
      <section className="panel compiled-summary">
        <PanelHeader icon={<FileText size={20} />} title="Regression problem" />
        <MetricRow label="Coefficients" value={`${compiled.variables.length}`} />
        <MetricRow label="Linear constraints" value={`${compiled.constraints.length}`} />
        <MetricRow label="Candidate hyperplanes" value={assignmentCount.toLocaleString()} />
        <MetricRow label="Hyperplane index span" value={`H0 ... H${assignmentCount - 1}`} />
        <MetricRow label="Threads" value={`${workerCount}`} />
        <div className="scoring-card">
          <strong>p-adic linear objective</strong>
          <code>L(H) = #&#123;i : |u_i · H − t_i|ₚ = 0&#125;</code>
          <span>The unit wells force a floor of {regressionFloor}; CNF violations add above that.</span>
        </div>
      </section>

      <section className="panel worker-dashboard">
        <div className="run-stats">
          <Stat label="Hyperplanes tested" value={`${snapshot.totalTested.toLocaleString()} / ${assignmentCount.toLocaleString()}`} />
          <Stat label="Total speed" value={`${formatRate(snapshot.totalSpeed)} hyperplanes/s`} />
          <Stat label="Best loss" value={`${bestRegressionLoss ?? "-"}`} />
          <Stat label="Floor hyperplanes" value={`${snapshot.solutions}`} />
        </div>
        <div className="global-progress">
          <span style={{ width: `${progress * 100}%` }} />
        </div>
        <WorkerTable compiled={compiled} lanes={snapshot.lanes} />
        <LossChart floor={regressionFloor} history={regressionHistory} />
      </section>

      <section className="panel assignment-panel">
        <PanelHeader icon={<ShieldCheck size={20} />} title="Best p-adic regression solution" />
        {bestAssignment ? (
          <>
            {solutionEquation && (
              <div className="solution-equation">
                <code>{solutionEquation}</code>
              </div>
            )}
            <div className="solution-list">
              {compiled.variables.slice(0, 10).map((variable) => (
                <div className="solution-row" key={variable.name}>
                  <strong>{variable.name}</strong>
                  <span>{bestAssignment[variable.name] ? 1 : 0}</span>
                </div>
              ))}
            </div>
            <div className="constraint-check">
              <MetricRow
                label="Unit-well floor"
                value={`${validation?.theoreticalFloor ?? regressionFloor}`}
              />
              <MetricRow
                label="Linear constraints satisfied"
                value={`${validation?.nonUnitSatisfied ?? 0} / ${compiled.constraints.length}`}
              />
              <MetricRow
                label="Regression loss"
                value={`${validation?.loss ?? "-"}`}
              />
            </div>
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
                <RotateCcw size={17} /> Back to setup
              </button>
            </>
          ) : isPaused ? (
            <>
              <button
                className="secondary-button"
                type="button"
                onClick={() => controller.start(workerCount, compiled)}
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
    </main>
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

function LossChart({
  floor,
  history
}: {
  floor: number;
  history: Array<{ second: number; loss: number }>;
}) {
  const points = history.length ? history : [{ second: 0, loss: Math.max(floor, 1) }];
  const maxLoss = Math.max(floor + 1, ...points.map((point) => point.loss));
  const width = 640;
  const height = 158;
  const floorY = height - (floor / maxLoss) * (height - 28) - 14;
  const path = points
    .map((point, index) => {
      const x = points.length === 1 ? 0 : (index / (points.length - 1)) * width;
      const y = height - (point.loss / maxLoss) * (height - 28) - 14;
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <div className="loss-chart">
      <div className="chart-title">
        <strong>p-adic loss over time</strong>
        <span>lower is better</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="p-adic loss over time">
        <line x1="0" x2={width} y1={floorY} y2={floorY} className="floor-line" />
        <path d={path} className="loss-line" />
        <text x="14" y={height - 20}>
          unit-well floor = {floor}
        </text>
      </svg>
    </div>
  );
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

function compileCspSource(source: string): CompiledProblem | null {
  return source.trim() ? compileProblem(source) : null;
}

export default App;
