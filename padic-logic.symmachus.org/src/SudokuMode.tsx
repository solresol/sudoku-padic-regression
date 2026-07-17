import {
  CirclePause,
  FlaskConical,
  Grid3x3,
  Play,
  RotateCcw,
  ShieldCheck,
  Shuffle,
  Sigma,
  Square,
  Table2
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type Grid,
  type PuzzleModel,
  type SudokuRegressionRow,
  ALPHA_DEFAULT,
  MIHARA_SUDOKU_PRIME,
  P_DEFAULT,
  PEERS,
  buildPuzzleModel,
  buildSudokuRegressionDataFrame,
  carve,
  dedupedPeerConflicts,
  evaluateObjective,
  gridToString,
  parsePuzzle,
  randomSolvedGrid
} from "./lib/sudoku";
import {
  type SolverSnapshot,
  type SudokuMethod,
  SudokuSolver
} from "./lib/sudokuSolver";
import {
  appendLossHistory,
  type LossHistoryPoint
} from "./lib/lossHistory";

const STANDARD_PUZZLE = [
  "53..7....",
  "6..195...",
  ".98....6.",
  "8...6...3",
  "4..8.3..1",
  "7...2...6",
  ".6....28.",
  "...419..5",
  "....8..79"
].join("");

const EMPTY_PUZZLE = ".".repeat(81);

type Phase = "setup" | "running" | "paused" | "solved" | "settled";

const STEPS_PER_FRAME: Record<string, number> = {
  slow: 6,
  normal: 40,
  fast: 400
};

function SudokuMode() {
  const [puzzleText, setPuzzleText] = useState(STANDARD_PUZZLE);
  const [method, setMethod] = useState<SudokuMethod>("stepwise");
  const [maxSteps, setMaxSteps] = useState(60_000);
  const [restarts, setRestarts] = useState(15);
  const [speed, setSpeed] = useState<keyof typeof STEPS_PER_FRAME>("normal");

  const [snap, setSnap] = useState<SolverSnapshot | null>(null);
  const [history, setHistory] = useState<LossHistoryPoint[]>([]);
  const [miharaHistory, setMiharaHistory] = useState<LossHistoryPoint[]>([]);
  const [phase, setPhase] = useState<Phase>("setup");
  const [parseError, setParseError] = useState<string | null>(null);

  const solverRef = useRef<SudokuSolver | null>(null);
  const rafRef = useRef<number | null>(null);
  const runningRef = useRef(false);

  const puzzle = useMemo<Grid | null>(() => {
    try {
      const grid = parsePuzzle(puzzleText);
      return grid;
    } catch {
      return null;
    }
  }, [puzzleText]);

  const model = useMemo<PuzzleModel | null>(
    () => (puzzle ? buildPuzzleModel(puzzle, { p: P_DEFAULT, alpha: ALPHA_DEFAULT }) : null),
    [puzzle]
  );

  useEffect(() => {
    try {
      parsePuzzle(puzzleText);
      setParseError(null);
    } catch (error) {
      setParseError(error instanceof Error ? error.message : String(error));
    }
  }, [puzzleText]);

  const stopRaf = useCallback(() => {
    runningRef.current = false;
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  useEffect(() => stopRaf, [stopRaf]);

  const frame = useCallback(() => {
    const solver = solverRef.current;
    if (!solver || !model || !runningRef.current) {
      return;
    }
    const steps = method === "mihara" ? 1 : STEPS_PER_FRAME[speed];
    let latest = solver.snapshot();
    for (let i = 0; i < steps; i += 1) {
      latest = solver.advance();
      if (latest.solved || latest.done) {
        break;
      }
    }
    setSnap(latest);
    const latestLoss = method === "mihara"
      ? latest.miharaSignedLoss
      : evaluateObjective(latest.grid, model).loss;
    if (latestLoss != null) {
      setHistory((previous) => appendLossHistory(previous, {
        step: latest.totalSteps,
        loss: latestLoss
      }));
    }
    if (method === "mihara" && latest.miharaLoss != null) {
      setMiharaHistory((previous) => appendLossHistory(previous, {
        step: latest.totalSteps,
        loss: latest.miharaLoss as number
      }));
    }

    if (latest.solved) {
      setPhase("solved");
      stopRaf();
      return;
    }
    if (latest.done) {
      setPhase("settled");
      stopRaf();
      return;
    }
    rafRef.current = requestAnimationFrame(frame);
  }, [method, model, speed, stopRaf]);

  const handleStart = useCallback(() => {
    if (!puzzle || !model) {
      return;
    }
    solverRef.current = new SudokuSolver(puzzle, {
      method,
      seed: 0,
      maxSteps: method === "mihara" ? Number.MAX_SAFE_INTEGER : maxSteps,
      restarts: method === "mihara" ? 1 : restarts,
      beta0: 0.5,
      beta1: 6.0
    });
    const initial = solverRef.current.snapshot();
    setSnap(initial);
    setHistory(method === "mihara"
      ? []
      : [{ step: 0, loss: evaluateObjective(initial.grid, model).loss }]);
    setMiharaHistory([]);
    if (initial.solved) {
      setPhase("solved");
      return;
    }
    setPhase("running");
    runningRef.current = true;
    rafRef.current = requestAnimationFrame(frame);
  }, [puzzle, model, method, maxSteps, restarts, frame]);

  const handlePause = useCallback(() => {
    stopRaf();
    setPhase("paused");
  }, [stopRaf]);

  const handleResume = useCallback(() => {
    if (!solverRef.current) {
      return;
    }
    setPhase("running");
    runningRef.current = true;
    rafRef.current = requestAnimationFrame(frame);
  }, [frame]);

  const handleReset = useCallback(() => {
    stopRaf();
    solverRef.current = null;
    setSnap(null);
    setHistory([]);
    setMiharaHistory([]);
    setPhase("setup");
  }, [stopRaf]);

  const setSample = useCallback(
    (text: string) => {
      handleReset();
      setPuzzleText(text);
    },
    [handleReset]
  );

  const newRandomPuzzle = useCallback(() => {
    const solution = randomSolvedGrid(Math.random);
    const carved = carve(solution, 30, Math.random);
    setSample(gridToString(carved));
  }, [setSample]);

  const updatePuzzleCell = useCallback((index: number, value: number) => {
    if (phase !== "setup") {
      return;
    }
    setPuzzleText((currentText) => {
      try {
        const nextPuzzle = parsePuzzle(currentText);
        nextPuzzle[index] = value;
        return gridToString(nextPuzzle);
      } catch {
        return currentText;
      }
    });
  }, [phase]);

  const displayGrid: Grid = snap ? snap.grid : puzzle ?? new Array(81).fill(0);
  const conflictedCells = useMemo(() => computeConflicts(displayGrid), [displayGrid]);
  const objective = useMemo(
    () => (model ? evaluateObjective(displayGrid, model) : null),
    [displayGrid, model]
  );
  const dedupConf = useMemo(() => dedupedPeerConflicts(displayGrid), [displayGrid]);

  const isRunning = phase === "running";
  const canEdit = phase === "setup";
  const hasLossHistory = history.length > 0 || miharaHistory.length > 0 || (
    method === "mihara" && phase !== "setup"
  );

  return (
    <div className={hasLossHistory ? "sudoku-mode has-loss-history" : "sudoku-mode"}>
      <section className="panel sudoku-board-panel">
        <div className="panel-header">
          <Grid3x3 size={21} />
          <h2>All-different instance</h2>
        </div>
        <SudokuGrid
          grid={displayGrid}
          puzzle={puzzle ?? new Array(81).fill(0)}
          conflicted={conflictedCells}
          lastMove={isRunning ? snap?.lastMove ?? null : null}
          editable={canEdit && Boolean(puzzle)}
          onCellChange={updatePuzzleCell}
        />
        <div className="board-legend">
          <span><i className="swatch clue" /> clue (singleton domain)</span>
          <span><i className="swatch fill" /> search variable</span>
          <span><i className="swatch conflict" /> peer conflict</span>
        </div>
        {canEdit ? (
          <>
            <div className="sample-row">
              <button className="secondary-button" type="button" onClick={() => setSample(STANDARD_PUZZLE)}>
                Standard
              </button>
              <button className="secondary-button" type="button" onClick={() => setSample(EMPTY_PUZZLE)}>
                Empty
              </button>
              <button className="secondary-button" type="button" onClick={newRandomPuzzle}>
                <Shuffle size={15} /> New random
              </button>
            </div>
            <label className="puzzle-input-label" htmlFor="puzzle-input">
              81-character puzzle (digits, 0 or . for blanks)
            </label>
            <textarea
              id="puzzle-input"
              className="puzzle-input"
              spellCheck={false}
              value={puzzleText}
              onChange={(event) => setPuzzleText(event.target.value.replace(/\s+/gu, ""))}
            />
            {parseError && <div className="error-box">{parseError}</div>}
          </>
        ) : (
          <div className="board-status">
            <StatusBanner phase={phase} snap={snap} />
          </div>
        )}
      </section>

      <section className="panel sudoku-objective-panel">
        <div className="panel-header">
          <Sigma size={21} />
          <h2>Signed p-adic objective</h2>
        </div>
        {model ? (
          <>
            <MetricRow label="Variables (cells)" value={`${model.variables}`} />
            <MetricRow label="Clues (singleton domains)" value={`${model.clueCount}`} />
            <MetricRow label="Prime" value={`p = ${model.p}  (> 9)`} />
            <MetricRow label="Pinning weight" value={`α = ${model.alpha}  (> 20)`} accent />
            <MetricRow
              label="Positive pinning observations"
              value={model.positiveObservationsMinimal.toLocaleString()}
            />
            <MetricRow
              label="Negative reward observations"
              value={`${model.negativeObservations.toLocaleString()}  (peer pairs)`}
            />
            <div className="scoring-card">
              <strong>Minimal objective</strong>
              <code>
                L(x) = α · Σᵢ Σₐ |xᵢ − a|ₚ − Σ |xᵢ − xⱼ|ₚ
              </code>
              <span>
                Every term is a genuine p-adic norm. For p &gt; 9 each edge norm is 0 (equal) or 1
                (unequal), so on digit states L = floor + conflicts.
              </span>
            </div>
            {objective && (
              <div className="constraint-check">
                <MetricRow label="Theoretical floor" value={model.theoreticalFloor.toLocaleString()} />
                <MetricRow
                  label={method === "mihara" ? "Rendered-grid signed loss" : "Current loss L(x)"}
                  value={objective.loss.toLocaleString()}
                />
                <MetricRow
                  label={objective.domainRespecting
                    ? "Peer conflicts (loss − floor)"
                    : "Loss − floor (off domain)"}
                  value={`${objective.dedupedConflicts}`}
                  accent={objective.dedupedConflicts > 0}
                />
              </div>
            )}
          </>
        ) : (
          <div className="empty-state">Enter a valid 81-character puzzle to compile the objective.</div>
        )}
      </section>

      <section className="panel sudoku-search-panel">
        <div className="panel-header">
          <FlaskConical size={21} />
          <h2>Algorithm comparison</h2>
        </div>
        <div className="control-block">
          <span className="control-label">Algorithm</span>
          <div className="segmented small">
            <button
              type="button"
              className={method === "stepwise" ? "seg on" : "seg"}
              onClick={() => canEdit && setMethod("stepwise")}
              disabled={!canEdit}
            >
              Row-swap
            </button>
            <button
              type="button"
              className={method === "zubarev" ? "seg on" : "seg"}
              onClick={() => canEdit && setMethod("zubarev")}
              disabled={!canEdit}
            >
              Zubarev walk
            </button>
            <button
              type="button"
              className={method === "mihara" ? "seg on" : "seg"}
              onClick={() => canEdit && setMethod("mihara")}
              disabled={!canEdit}
            >
              Mihara attempt
            </button>
          </div>
        </div>
        {method !== "mihara" && <div className="control-block">
          <span className="control-label">Speed</span>
          <div className="segmented small">
            {(["slow", "normal", "fast"] as const).map((s) => (
              <button
                key={s}
                type="button"
                className={speed === s ? "seg on" : "seg"}
                onClick={() => setSpeed(s)}
              >
                {s}
              </button>
            ))}
          </div>
        </div>}
        <MetricRow
          label={method === "mihara" ? "Retry policy" : "Max steps / restart"}
          value={method === "mihara" ? "Until Sudoku solution or stopped" : maxSteps.toLocaleString()}
        />
        {method !== "mihara" && <MetricRow label="Restarts" value={`${restarts}`} />}

        {method === "mihara" && (
          <div className="algorithm-diagnostic mihara-diagnostic" role="note">
            <strong>Mihara receives a positive-only complement expansion.</strong>
            <p>
              Each negative peer row is replaced by positive equality rows for every allowed nonzero
              digit difference. Weighted consensus now rewards unequal peers. The best diagnostic fit
              is displayed even when it is off-domain, while fresh starts continue until a valid Sudoku
              is recovered or you pause or reset. Off-domain residues are blanked in the grid; the
              original signed audit below evaluates the recovered modulo-{MIHARA_SUDOKU_PRIME} vector.
            </p>
          </div>
        )}

        {method === "mihara" && (
          <>
            <MetricRow label="Mihara prime" value={`p = ${MIHARA_SUDOKU_PRIME}  (> 16)`} />
            <MetricRow
              label="Mihara input"
              value={`${snap?.miharaObservationCount?.toLocaleString() ?? "–"} positive rows · weighted total ${snap?.miharaTotal?.toLocaleString() ?? "–"}`}
            />
            <MetricRow
              label="Positive last-digit loss"
              value={`${snap?.miharaLoss ?? "–"}`}
              accent={snap?.miharaLoss != null && snap.miharaLoss > (snap.miharaFloor ?? 0)}
            />
            <MetricRow label="Positive satisfiable floor" value={`${snap?.miharaFloor ?? "–"}`} />
            <MetricRow label="Original signed loss (audit)" value={`${snap?.miharaSignedLoss ?? "–"}`} />
          </>
        )}

        <div className="run-stats sudoku-stats">
          <Stat
            label={method === "mihara" ? "Rendered conflicts H_cb" : "Conflicts H_cb"}
            value={`${snap?.conflicts ?? objective?.hcbConflicts ?? 0}`}
          />
          <Stat label={method === "mihara" ? "Rendered peer conflicts" : "Peer conflicts"} value={`${dedupConf}`} />
          <Stat
            label={method === "mihara" ? "Weighted positive inliers" : "Best"}
            value={method === "mihara"
              ? `${snap?.miharaInliers ?? "–"} / ${snap?.miharaTotal ?? "–"}`
              : `${snap?.bestConflicts ?? "–"}`}
          />
          <Stat label={method === "mihara" ? "Trials" : "Steps"} value={`${snap?.totalSteps.toLocaleString() ?? 0}`} />
        </div>
        {method === "zubarev" && snap?.beta != null && (
          <MetricRow label="Inverse temperature β" value={snap.beta.toFixed(2)} />
        )}
        {method === "mihara" && snap && (
          <MetricRow
            label="Invertible samples"
            value={`${snap.miharaSuccessfulTrials ?? 0} / ${snap.totalSteps}`}
            accent={snap.done && (snap.miharaSuccessfulTrials ?? 0) === 0}
          />
        )}
        {method === "mihara" && snap?.miharaCoefficients && (
          <div className="mihara-result-summary">
            <MetricRow
              label="Off-domain coefficients"
              value={`${snap.miharaDomainViolations ?? 0} / 81`}
              accent={(snap.miharaDomainViolations ?? 0) > 0}
            />
            <MetricRow
              label="Clue violations"
              value={`${snap.miharaClueViolations ?? 0}`}
              accent={(snap.miharaClueViolations ?? 0) > 0}
            />
            <MetricRow
              label={`v₁₉(r) = 0 peer rows`}
              value={`${snap.miharaNegativeUnitValuations ?? 0} / 810`}
            />
            <MetricRow
              label={`v₁₉(r) = ∞ peer rows`}
              value={`${snap.miharaNegativeInfiniteValuations ?? 0} / 810`}
              accent={(snap.miharaNegativeInfiniteValuations ?? 0) > 0}
            />
            <div className="solution-equation">
              <span>Recovered cell coefficients modulo {MIHARA_SUDOKU_PRIME}</span>
              <code>{snap.miharaCoefficients.join(", ")}</code>
            </div>
          </div>
        )}
        {method !== "mihara" && (
          <MetricRow label="Restart" value={snap ? `${snap.restart + 1} / ${restarts}` : `–`} />
        )}

        <div className="run-actions">
          {phase === "setup" && (
            <button className="run-button" type="button" onClick={handleStart} disabled={!puzzle}>
              <Play size={17} /> Start search
            </button>
          )}
          {phase === "running" && (
            <button className="secondary-button" type="button" onClick={handlePause}>
              <CirclePause size={17} /> Pause
            </button>
          )}
          {phase === "paused" && (
            <button className="run-button" type="button" onClick={handleResume}>
              <Play size={17} /> Resume
            </button>
          )}
          {(phase === "solved" || phase === "settled") && (
            <div className="result-pill">
              {phase === "solved" ? (
                <>
                  <ShieldCheck size={17} /> Solved in {snap?.totalSteps.toLocaleString()} steps
                </>
              ) : (
                method === "mihara" ? (
                  <>Retry limit reached: no Sudoku solution</>
                ) : (
                  <>Settled at {snap?.bestConflicts} conflicts (minimum-conflict)</>
                )
              )}
            </div>
          )}
          <button className="danger-button" type="button" onClick={handleReset}>
            <RotateCcw size={15} /> Reset
          </button>
        </div>
      </section>

      {hasLossHistory && (
        <section className="panel sudoku-chart-panel">
          <div className="panel-header">
            <Square size={18} />
            <h2>p-adic loss over time</h2>
          </div>
          {method === "mihara" ? (
            <>
              <SudokuLossChart
                ariaLabel="Mihara positive-complement p-adic loss over time"
                floor={snap?.miharaFloor ?? 0}
                history={miharaHistory}
                subtitle="lower is better; this is the score Mihara selects"
                title="Positive-complement last-digit loss"
              />
              <SudokuLossChart
                ariaLabel="Original signed Sudoku p-adic loss over time"
                floor={model?.theoreticalFloor ?? 0}
                history={history}
                subtitle="audit only; this does not select Mihara fits"
                title="Original signed objective L(x)"
              />
            </>
          ) : (
            <SudokuLossChart floor={model?.theoreticalFloor ?? 0} history={history} />
          )}
        </section>
      )}

      <SudokuRegressionDataFramePanel model={model} />
    </div>
  );
}

function computeConflicts(grid: Grid): Set<number> {
  const conflicts = new Set<number>();
  for (let i = 0; i < 81; i += 1) {
    const v = grid[i];
    if (v === 0) continue;
    for (const j of PEERS[i]) {
      if (grid[j] === v) {
        conflicts.add(i);
        break;
      }
    }
  }
  return conflicts;
}

function SudokuGrid({
  grid,
  puzzle,
  conflicted,
  lastMove,
  editable,
  onCellChange
}: {
  grid: Grid;
  puzzle: Grid;
  conflicted: Set<number>;
  lastMove: { row: number; col1: number; col2: number } | null;
  editable: boolean;
  onCellChange: (index: number, value: number) => void;
}) {
  const movedCells = new Set<number>();
  if (lastMove) {
    movedCells.add(lastMove.row * 9 + lastMove.col1);
    movedCells.add(lastMove.row * 9 + lastMove.col2);
  }
  return (
    <div
      className={editable ? "sudoku-grid is-editable" : "sudoku-grid"}
      role="grid"
      aria-label="Sudoku grid"
      aria-readonly={!editable}
    >
      {grid.map((value, i) => {
        const isClue = puzzle[i] !== 0;
        const row = Math.floor(i / 9);
        const column = i % 9;
        const classes = ["sudoku-cell"];
        if (i % 3 === 2 && i % 9 !== 8) classes.push("box-right");
        if (Math.floor(i / 9) % 3 === 2 && Math.floor(i / 9) !== 8) classes.push("box-bottom");
        if (isClue) classes.push("clue");
        if (conflicted.has(i)) classes.push("conflict");
        if (movedCells.has(i)) classes.push("moved");
        return (
          <div
            aria-selected={isClue}
            className={classes.join(" ")}
            key={i}
            role="gridcell"
          >
            <input
              aria-label={`Row ${row + 1}, column ${column + 1}`}
              disabled={!editable}
              inputMode="numeric"
              maxLength={1}
              pattern="[1-9]"
              value={value === 0 ? "" : String(value)}
              onChange={(event) => {
                const nextValue = parseSudokuCellValue(event.target.value);
                if (nextValue !== null) {
                  onCellChange(i, nextValue);
                }
              }}
              onFocus={(event) => event.currentTarget.select()}
            />
          </div>
        );
      })}
    </div>
  );
}

function parseSudokuCellValue(value: string): number | null {
  const trimmed = value.trim();
  if (trimmed === "" || trimmed === "." || trimmed === "0") {
    return 0;
  }
  return /^[1-9]$/u.test(trimmed) ? Number(trimmed) : null;
}

const DATAFRAME_HEADER_HEIGHT = 164;
const DATAFRAME_ROW_HEIGHT = 32;
const DATAFRAME_WINDOW_ROWS = 40;
const DATAFRAME_OVERSCAN_ROWS = 8;

function SudokuRegressionDataFramePanel({ model }: { model: PuzzleModel | null }) {
  const dataframe = useMemo(
    () => (model ? buildSudokuRegressionDataFrame(model) : null),
    [model]
  );
  const [scrollTop, setScrollTop] = useState(0);
  const dataframeScrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setScrollTop(0);
    if (dataframeScrollRef.current) {
      dataframeScrollRef.current.scrollTop = 0;
    }
  }, [dataframe]);

  if (!dataframe || !model) {
    return (
      <section
        className="panel sudoku-dataframe-panel"
        aria-label="Sudoku p-adic linear regression dataframe"
      >
        <div className="panel-header">
          <Table2 size={20} />
          <h2>p-adic linear regression dataframe</h2>
        </div>
        <div className="empty-state">Enter a valid puzzle to build the regression dataframe.</div>
      </section>
    );
  }

  const firstVisibleRow = Math.max(
    0,
    Math.floor(Math.max(0, scrollTop - DATAFRAME_HEADER_HEIGHT) / DATAFRAME_ROW_HEIGHT) -
      DATAFRAME_OVERSCAN_ROWS
  );
  const lastVisibleRow = Math.min(
    dataframe.rows.length,
    firstVisibleRow + DATAFRAME_WINDOW_ROWS
  );
  const visibleRows = dataframe.rows.slice(firstVisibleRow, lastVisibleRow);
  const columnCount = dataframe.variables.length + 3;
  const variableNames = dataframe.variables.map((variable) => variable.name);

  return (
    <section
      className="panel sudoku-dataframe-panel dataframe-panel"
      aria-label="Sudoku p-adic linear regression dataframe"
    >
      <div className="panel-header">
        <Table2 size={20} />
        <h2>p-adic linear regression dataframe</h2>
      </div>
      <div className="panel-toolbar">
        <strong>Regression observations</strong>
        <span>
          {dataframe.rows.length.toLocaleString()} rows · {dataframe.variables.length} coefficients
        </span>
      </div>
      <div
        className="dataframe-scroll sudoku-dataframe-scroll"
        onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
        ref={dataframeScrollRef}
      >
        <table
          className="dataframe-table"
          aria-label="Sudoku regression observations"
        >
          <thead>
            <tr>
              <th className="row-label">row</th>
              {dataframe.variables.map((variable) => (
                <th className="feature-heading" key={variable.name}>
                  <span>{variable.name}</span>
                </th>
              ))}
              <th className="target-heading">target</th>
              <th className="weight-heading">signed<br />weight</th>
            </tr>
          </thead>
          <tbody>
            {firstVisibleRow > 0 && (
              <VirtualDataFrameSpacer
                columnCount={columnCount}
                height={firstVisibleRow * DATAFRAME_ROW_HEIGHT}
              />
            )}
            {visibleRows.map((row) => (
              <SudokuDataFrameRow
                key={row.id}
                row={row}
                variableNames={variableNames}
              />
            ))}
            {lastVisibleRow < dataframe.rows.length && (
              <VirtualDataFrameSpacer
                columnCount={columnCount}
                height={(dataframe.rows.length - lastVisibleRow) * DATAFRAME_ROW_HEIGHT}
              />
            )}
          </tbody>
        </table>
      </div>
      <div className="dataframe-window-status" aria-live="polite">
        Showing rows {(firstVisibleRow + 1).toLocaleString()}–{lastVisibleRow.toLocaleString()} of {dataframe.rows.length.toLocaleString()}
      </div>
      <div className="dataframe-legend">
        <span><i className="legend-unit" /> Digit-pinning wells (+{model.alpha})</span>
        <span><i className="legend-negative" /> Peer inequality rewards (−1)</span>
      </div>
    </section>
  );
}

function SudokuDataFrameRow({
  row,
  variableNames
}: {
  row: SudokuRegressionRow;
  variableNames: string[];
}) {
  return (
    <tr
      className={`dataframe-row ${row.kind === "pinning" ? "row-unit-well" : "row-constraint"}`}
    >
      <th title={row.source}>{row.label}</th>
      {variableNames.map((variableName) => {
        const value = row.coefficients[variableName] ?? 0;
        return (
          <td className={value < 0 ? "negative-cell" : undefined} key={variableName}>
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
  );
}

function VirtualDataFrameSpacer({
  columnCount,
  height
}: {
  columnCount: number;
  height: number;
}) {
  return (
    <tr className="dataframe-virtual-spacer" aria-hidden="true">
      <td colSpan={columnCount} style={{ height }} />
    </tr>
  );
}

function StatusBanner({ phase, snap }: { phase: Phase; snap: SolverSnapshot | null }) {
  if (phase === "solved") {
    return (
      <span className="status-good">
        <ShieldCheck size={16} /> Valid completion — all 810 peer edges unequal.
      </span>
    );
  }
  if (phase === "settled") {
    return (
      <span className="status-warn">
        Minimum-conflict state: {snap?.bestConflicts} conflicting pairs remain (likely unsatisfiable
        clues). This is the Max-CSP half of the theorem.
      </span>
    );
  }
  return <span className="status-run">Searching… conflicts {snap?.conflicts}</span>;
}

function SudokuLossChart({
  ariaLabel = "Sudoku p-adic loss over time",
  floor,
  history,
  subtitle = "lower is better",
  title = "Signed objective L(x)"
}: {
  ariaLabel?: string;
  floor: number;
  history: LossHistoryPoint[];
  subtitle?: string;
  title?: string;
}) {
  const floorIndex = history.findIndex((point) => point.loss <= floor);
  const points = floorIndex >= 0 ? history.slice(0, floorIndex + 1) : history;
  const maxLoss = Math.max(floor + 1, ...points.map((point) => point.loss));
  const width = 900;
  const height = 200;
  const plot = { left: 72, right: 18, top: 18, bottom: 40 };
  const plotWidth = width - plot.left - plot.right;
  const plotHeight = height - plot.top - plot.bottom;
  const lossRange = maxLoss - floor;
  const firstStep = points[0]?.step ?? 0;
  const lastStep = points[points.length - 1]?.step ?? firstStep;
  const stepRange = Math.max(1, lastStep - firstStep);
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
      const x = plot.left + ((point.step - firstStep) / stepRange) * plotWidth;
      const y = yForLoss(Math.max(floor, Math.min(point.loss, maxLoss)));
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <div className="loss-chart">
      <div className="chart-title">
        <strong>{title}</strong>
        <span>{subtitle}</span>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={ariaLabel}
        data-points={points.length}
        data-last-step={points[points.length - 1]?.step ?? 0}
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
              <text
                className="axis-label y-axis-label"
                x={plot.left - 10}
                y={y + 4}
                textAnchor="end"
              >
                {value.toLocaleString()}
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
          satisfiable floor = {floor.toLocaleString()}
        </text>
        <text className="axis-label x-axis-label" x={plot.left} y={height - 10} textAnchor="start">
          {firstStep.toLocaleString()} steps
        </text>
        <text
          className="axis-label x-axis-label"
          x={width - plot.right}
          y={height - 10}
          textAnchor="end"
        >
          {(points[points.length - 1]?.step ?? 0).toLocaleString()} steps
        </text>
      </svg>
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

export default SudokuMode;
