import {
  BookOpen,
  Check,
  ChevronRight,
  CirclePause,
  Code2,
  Download,
  FileText,
  FlaskConical,
  Moon,
  Play,
  RotateCcw,
  Search,
  Send,
  Settings,
  ShieldCheck,
  Square,
  Triangle,
  WandSparkles
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  type CompiledProblem,
  compileProblem,
  evaluateAssignment,
  renderClause
} from "./lib/csp";
import {
  type LanguageModelAvailability,
  detectLanguageModel,
  generateCspFromDescription
} from "./lib/browserLanguageModel";
import { createSearchPlan, formatAssignmentCount } from "./lib/search";
import { useSearchController } from "./hooks/useSearchController";

const DEFAULT_CSP = [
  "# Example CSP (boolean variables, constraints)",
  "A or B or C",
  "B or C or not D",
  "not A or D or E",
  "A or not E or F",
  "not B or not C or G",
  "D or E or F",
  "not F or G or H",
  "A or not H or I",
  "J or K or not I",
  "not J or K or L",
  "L or M or N",
  "M or N or not O",
  "not O or P or Q",
  "Q or R or S",
  "not S or T or U",
  "V or W or X",
  "not V or X"
].join("\n");

const DEFAULT_DESCRIPTION = [
  "Four friends - Ava (A), Ben (B), Cara (C), and Dev (D) - need to be assigned.",
  "Rules: A cannot test. B and C cannot work on the same task. If D documents, then A designs."
].join("\n");

function App() {
  const [source, setSource] = useState(DEFAULT_CSP);
  const [description, setDescription] = useState(DEFAULT_DESCRIPTION);
  const [compiled, setCompiled] = useState<CompiledProblem | null>(() =>
    compileProblem(DEFAULT_CSP)
  );
  const [compileError, setCompileError] = useState<string | null>(null);
  const [workerCount, setWorkerCount] = useState(() =>
    Math.min(Math.max(navigator.hardwareConcurrency ?? 4, 2), 8)
  );
  const [modelStatus, setModelStatus] =
    useState<LanguageModelAvailability>("unavailable");
  const [modelNote, setModelNote] = useState("Checking local browser model...");
  const [isGenerating, setIsGenerating] = useState(false);
  const controller = useSearchController(compiled);

  useEffect(() => {
    let alive = true;
    detectLanguageModel()
      .then((availability) => {
        if (!alive) {
          return;
        }
        setModelStatus(availability);
        setModelNote(
          availability === "available"
            ? "Chrome/Edge languageModel is ready."
            : availability === "unavailable"
              ? "Manual CSP entry is available in every browser."
              : `Model is ${availability}; browser may download it on first use.`
        );
      })
      .catch(() => {
        setModelStatus("unavailable");
        setModelNote("Manual CSP entry is available in every browser.");
      });
    return () => {
      alive = false;
    };
  }, []);

  const searchPlan = useMemo(
    () => (compiled ? createSearchPlan(compiled, workerCount) : null),
    [compiled, workerCount]
  );

  const handleCompile = () => {
    try {
      const nextCompiled = compileProblem(source);
      setCompiled(nextCompiled);
      setCompileError(null);
      controller.reset();
    } catch (error) {
      setCompileError(error instanceof Error ? error.message : String(error));
    }
  };

  const handleGenerateCsp = async () => {
    setIsGenerating(true);
    try {
      const result = await generateCspFromDescription(description);
      setSource(result.source);
      setCompiled(compileProblem(result.source));
      setCompileError(null);
      setModelNote(
        result.usedLocalModel
          ? "Generated locally with browser languageModel."
          : "Generated with the deterministic fallback template."
      );
    } catch (error) {
      setCompileError(error instanceof Error ? error.message : String(error));
    } finally {
      setIsGenerating(false);
    }
  };

  const isRunning =
    controller.snapshot.status === "running" ||
    controller.snapshot.status === "complete" ||
    controller.snapshot.status === "paused";

  return (
    <div className="app-shell">
      <Header
        activeStep={isRunning ? "Run" : "Problem"}
        status={isRunning ? "Exhaustive search in this browser" : "Ready"}
      />
      <CapabilityStrip status={modelStatus} note={modelNote} />

      {isRunning && compiled ? (
        <RunDashboard
          compiled={compiled}
          workerCount={workerCount}
          controller={controller}
        />
      ) : (
        <>
          <main className="setup-grid">
            <ProblemPanel
              source={source}
              description={description}
              isGenerating={isGenerating}
              onDescriptionChange={setDescription}
              onGenerateCsp={handleGenerateCsp}
              onSourceChange={setSource}
            />
            <TernaryPanel compiled={compiled} error={compileError} />
            <SearchPlanPanel compiled={compiled} workerCount={workerCount} />
          </main>

          <ReadyBand
            compiled={compiled}
            plan={searchPlan}
            workerCount={workerCount}
            onCompile={handleCompile}
            onStart={() => compiled && controller.start(workerCount)}
            onWorkerCountChange={setWorkerCount}
          />
        </>
      )}

      <Footer compiled={compiled} />
    </div>
  );
}

function Header({
  activeStep,
  status
}: {
  activeStep: "Problem" | "Run";
  status: string;
}) {
  const tabs = [
    { label: "Problem", href: "#problem", icon: <FileText size={18} /> },
    { label: "Clauses", href: "#clauses", icon: <Triangle size={18} /> },
    { label: "Search", href: "#search", icon: <Search size={18} /> },
    { label: "Run", href: "#run", icon: <Play size={18} /> },
    { label: "Proof", href: "#proof", icon: <ShieldCheck size={18} /> }
  ];

  return (
    <header className="topbar">
      <div className="brand">
        <Mascot />
        <div>
          <h1>p-adic logic</h1>
          <p>padic-logic.symmachus.org</p>
        </div>
      </div>
      <nav className="nav-tabs" aria-label="Workflow">
        {tabs.map((tab) => (
          <a
            className={tab.label === activeStep ? "nav-tab is-active" : "nav-tab"}
            href={tab.href}
            key={tab.label}
          >
            {tab.icon}
            {tab.label}
          </a>
        ))}
      </nav>
      <div className="top-actions">
        <span className="status-pill">{status}</span>
        <button className="icon-button" type="button" aria-label="Settings">
          <Settings size={19} />
        </button>
        <button className="icon-button" type="button" aria-label="Docs">
          <BookOpen size={19} />
        </button>
        <button className="icon-button" type="button" aria-label="Theme">
          <Moon size={18} />
        </button>
      </div>
    </header>
  );
}

function CapabilityStrip({
  status,
  note
}: {
  status: LanguageModelAvailability;
  note: string;
}) {
  return (
    <section className="capability-strip" aria-label="Browser capability">
      <span className={`dot dot-${status}`} />
      <strong>
        {status === "available"
          ? "Local browser model available"
          : "Manual CSP mode ready"}
      </strong>
      <span className="capability-chip">Chrome/Edge languageModel</span>
      <span>{note}</span>
      <span className="privacy-note">No problem text leaves this browser.</span>
    </section>
  );
}

function ProblemPanel({
  source,
  description,
  isGenerating,
  onDescriptionChange,
  onGenerateCsp,
  onSourceChange
}: {
  source: string;
  description: string;
  isGenerating: boolean;
  onDescriptionChange: (value: string) => void;
  onGenerateCsp: () => void;
  onSourceChange: (value: string) => void;
}) {
  return (
    <section className="panel problem-panel" id="problem">
      <PanelHeader icon={<FileText size={21} />} title="Enter CSP" />
      <div className="description-box">
        <div>
          <h3>Describe the problem</h3>
          <p>Use Chrome or Edge to ask the local browser model for CSP lines.</p>
        </div>
        <textarea
          aria-label="Natural language problem"
          value={description}
          onChange={(event) => onDescriptionChange(event.target.value)}
        />
        <button
          className="secondary-button"
          type="button"
          onClick={onGenerateCsp}
          disabled={isGenerating}
        >
          <WandSparkles size={16} />
          {isGenerating ? "Generating..." : "Generate CSP"}
        </button>
      </div>
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
        <span>Constraints: {source.split(/\r?\n/u).filter(Boolean).length - 1}</span>
        <span>Variables are inferred</span>
      </div>
    </section>
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
    <section className="panel" id="clauses">
      <PanelHeader icon={<Triangle size={21} />} title="Ternary reduction" />
      {error ? (
        <div className="error-box">{error}</div>
      ) : compiled ? (
        <>
          <div className="panel-toolbar">
            <strong>Ternary CNF</strong>
            <span>{compiled.ternaryClauses.length} clauses</span>
          </div>
          <div className="clause-list">
            {compiled.ternaryClauses.map((clause) => (
              <div className="clause-row" key={clause.id}>
                <span>{clause.id}</span>
                <code>{renderClause(clause)}</code>
              </div>
            ))}
          </div>
          <ValidationList compiled={compiled} />
        </>
      ) : (
        <EmptyState text="Compile the evaluator to see ternary clauses." />
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
          <MetricRow label="Variables" value={`n = ${compiled.variables.length}`} />
          <MetricRow
            label="Assignments"
            value={`2^${compiled.variables.length} = ${formatAssignmentCount(compiled.variables.length)}`}
          />
          <MetricRow label="Strategy" value="Brute force (exhaustive)" accent />
          <MetricRow label="Worker split" value="Integer ranges" />
          <MetricRow label="Evaluator" value="Optimised p-adic linear regression solver" />
          <MetricRow label="Execution" value="Generated worker code hidden" />
          <MetricRow label="Loss floor" value="0 unit-well violations" />
          <MetricRow label="Success criterion" value="All non-unit constraints score zero" />
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
  onCompile,
  onStart,
  onWorkerCountChange
}: {
  compiled: CompiledProblem | null;
  plan: ReturnType<typeof createSearchPlan> | null;
  workerCount: number;
  onCompile: () => void;
  onStart: () => void;
  onWorkerCountChange: (value: number) => void;
}) {
  return (
    <section className="ready-band">
      <div className="ready-copy">
        <FlaskConical size={30} />
        <div>
          <h2>Ready to run in this browser</h2>
          <p>Workers read disjoint integer ranges and evaluate assignments locally.</p>
        </div>
      </div>
      <div className="stepper">
        <span>Workers</span>
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
        <span>Assignments</span>
        <strong>{plan ? plan.assignmentCount.toLocaleString() : "-"}</strong>
        <small>range split across {workerCount} workers</small>
      </div>
      <button className="primary-button" type="button" onClick={onCompile}>
        <Code2 size={18} />
        Compile evaluator
      </button>
      <button
        className="run-button"
        type="button"
        onClick={onStart}
        disabled={!compiled}
      >
        <Play size={18} />
        Start exhaustive search
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

  return (
    <main className="run-grid">
      <section className="panel compiled-summary">
        <PanelHeader icon={<FileText size={20} />} title="Compiled problem" />
        <MetricRow label="Variables" value={`${compiled.variables.length}`} />
        <MetricRow label="Clauses" value={`${compiled.constraints.length}`} />
        <MetricRow label="Assignment count" value={assignmentCount.toLocaleString()} />
        <MetricRow label="Mask range" value={`0 ... ${assignmentCount - 1}`} />
        <MetricRow label="Workers" value={`${workerCount}`} />
        <div className="scoring-card">
          <strong>Scoring model</strong>
          <code>loss(mask) = unit penalties + p-adic residual score</code>
          <span>Minimum possible loss: 0</span>
        </div>
      </section>

      <section className="panel worker-dashboard">
        <div className="run-stats">
          <Stat label="Total tested" value={`${snapshot.totalTested.toLocaleString()} / ${assignmentCount.toLocaleString()}`} />
          <Stat label="Total speed" value={`${formatRate(snapshot.totalSpeed)} masks/s`} />
          <Stat label="Best loss" value={`${snapshot.bestLoss ?? "-"}`} />
          <Stat label="Solutions found" value={`${snapshot.solutions}`} />
        </div>
        <div className="global-progress">
          <span style={{ width: `${progress * 100}%` }} />
        </div>
        <WorkerTable compiled={compiled} lanes={snapshot.lanes} />
        <LossChart history={snapshot.history} />
      </section>

      <section className="panel assignment-panel">
        <PanelHeader icon={<ShieldCheck size={20} />} title="Best assignment" />
        {bestAssignment ? (
          <>
            <div className="assignment-list">
              {compiled.variables.slice(0, 10).map((variable) => (
                <div className="assignment-row" key={variable.name}>
                  <strong>{variable.name}</strong>
                  <span>{bestAssignment[variable.name] ? "True" : "False"}</span>
                  <span className={bestAssignment[variable.name] ? "toggle on" : "toggle"} />
                </div>
              ))}
            </div>
            <div className="constraint-check">
              <MetricRow
                label="Unit-well violations"
                value={`${validation?.unitWellViolations ?? 0} / 0`}
              />
              <MetricRow
                label="Non-unit constraints zero"
                value={`${validation?.nonUnitSatisfied ?? 0} / ${compiled.constraints.length}`}
              />
              <MetricRow
                label="p-adic residual"
                value={`${validation?.loss ?? "-"} loss`}
              />
            </div>
          </>
        ) : (
          <EmptyState text="Best assignment will appear after the first worker update." />
        )}
        <div className="run-actions">
          <button className="secondary-button" type="button" onClick={controller.pause}>
            <CirclePause size={17} /> Pause
          </button>
          <button className="danger-button" type="button" onClick={controller.reset}>
            <Square size={15} /> Stop
          </button>
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
            : "[sys] waiting for worker updates..."}
        </pre>
      </section>

      <section className="panel split-panel">
        <PanelHeader icon={<RotateCcw size={20} />} title="How the search is split" />
        <p>Each integer mask encodes one truth assignment. Workers scan contiguous ranges.</p>
        <div className="bit-diagram">
          {compiled.variables.slice(0, 8).map((variable) => (
            <span key={variable.name}>
              bit {variable.index}
              <strong>{variable.name}</strong>
            </span>
          ))}
        </div>
        <div className="range-table">
          {snapshot.lanes.map((lane) => (
            <div key={lane.workerId}>
              <strong>W{lane.workerId}</strong>
              <span>{lane.start.toLocaleString()}</span>
              <span>{(lane.endExclusive - 1).toLocaleString()}</span>
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
    <div className="worker-table" role="table" aria-label="Worker lanes">
      <div className="worker-row worker-head" role="row">
        <span>Worker</span>
        <span>Assigned range</span>
        <span>Current mask</span>
        <span>Masks/sec</span>
        <span>Progress</span>
        <span>Best loss</span>
      </div>
      {lanes.map((lane) => {
        const progress =
          (lane.currentMask - lane.start) / Math.max(lane.endExclusive - lane.start, 1);
        return (
          <div className="worker-row" key={lane.workerId} role="row">
            <span className="worker-id">W{lane.workerId}</span>
            <span>
              {lane.start.toLocaleString()} - {(lane.endExclusive - 1).toLocaleString()}
            </span>
            <span>{lane.currentMask.toLocaleString()}</span>
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
          <span>0 - {compiled.assignmentCount - 1}</span>
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
  history
}: {
  history: Array<{ second: number; loss: number }>;
}) {
  const points = history.length ? history : [{ second: 0, loss: 4 }];
  const maxLoss = Math.max(4, ...points.map((point) => point.loss));
  const width = 640;
  const height = 158;
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
        <line x1="0" x2={width} y1={height - 14} y2={height - 14} className="floor-line" />
        <path d={path} className="loss-line" />
        <text x="14" y={height - 20}>
          minimum possible loss / theoretical floor
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
      {compiled && <span>2^{compiled.variables.length} assignments</span>}
      <span className="footer-lock">All computation stays in your browser.</span>
    </footer>
  );
}

function Mascot() {
  return (
    <svg className="mascot" viewBox="0 0 64 64" aria-hidden="true">
      <path d="M17 25 12 10l13 7h14l13-7-5 15" fill="#fff" stroke="#111827" strokeWidth="3" />
      <circle cx="32" cy="35" r="20" fill="#fff" stroke="#111827" strokeWidth="3" />
      <circle cx="24" cy="34" r="3" fill="#0f8f8f" />
      <circle cx="40" cy="34" r="3" fill="#0f8f8f" />
      <path d="M29 40h6M25 45c4 4 10 4 14 0" stroke="#111827" strokeWidth="3" strokeLinecap="round" />
      <path d="M20 18h24l-12-7-12 7Z" fill="#16a3a3" stroke="#111827" strokeWidth="3" />
      <circle cx="49" cy="49" r="8" fill="#f59e0b" stroke="#111827" strokeWidth="3" />
      <text x="46" y="53" fontSize="10" fontWeight="700" fill="#111827">
        p
      </text>
    </svg>
  );
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

export default App;
