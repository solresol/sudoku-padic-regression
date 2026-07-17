import { useCallback, useMemo, useRef, useState } from "react";
import {
  buildMiharaPositiveRegressionDataFrame,
  type CompiledProblem
} from "../lib/csp";
import {
  createSearchPlan,
  createSearchPermutation,
  maskToAssignment,
  type SearchStrategy
} from "../lib/search";

export interface WorkerLane {
  workerId: number;
  start: number;
  endExclusive: number;
  tested: number;
  currentMask: number;
  speed: number;
  bestLoss: number | null;
  bestMask: number | null;
  bestCoordinates: number[] | null;
  algorithmScore: number | null;
  algorithmTotal: number | null;
  algorithmLoss: number | null;
  algorithmSuccessfulTrials: number | null;
  algorithmSingularTrials: number | null;
  solutions: number;
  done: boolean;
}

export interface SearchLogEntry {
  id: number;
  text: string;
}

export interface SearchSnapshot {
  status: "idle" | "running" | "paused" | "complete";
  startedAt: number | null;
  strategy: SearchStrategy;
  workUnits: number;
  unbounded: boolean;
  workLabel: "hyperplanes" | "walk steps" | "RANSAC trials";
  chartFloor: number;
  lanes: WorkerLane[];
  bestLoss: number | null;
  bestMask: number | null;
  bestCoordinates: number[] | null;
  algorithmScore: number | null;
  algorithmTotal: number | null;
  algorithmLoss: number | null;
  algorithmSuccessfulTrials: number;
  algorithmSingularTrials: number;
  totalTested: number;
  totalSpeed: number;
  solutions: number;
  history: Array<{ tested: number; loss: number }>;
  logs: SearchLogEntry[];
}

const INITIAL_SNAPSHOT: SearchSnapshot = {
  status: "idle",
  startedAt: null,
  strategy: "ordered",
  workUnits: 0,
  unbounded: false,
  workLabel: "hyperplanes",
  chartFloor: 0,
  lanes: [],
  bestLoss: null,
  bestMask: null,
  bestCoordinates: null,
  algorithmScore: null,
  algorithmTotal: null,
  algorithmLoss: null,
  algorithmSuccessfulTrials: 0,
  algorithmSingularTrials: 0,
  totalTested: 0,
  totalSpeed: 0,
  solutions: 0,
  history: [],
  logs: []
};

interface WorkerProgress {
  type: "progress" | "done";
  workerId: number;
  tested: number;
  currentMask: number;
  speed: number;
  bestLoss: number | null;
  bestMask: number | null;
  bestCoordinates: number[] | null;
  algorithmScore: number | null;
  algorithmTotal: number | null;
  algorithmLoss: number | null;
  algorithmSuccessfulTrials: number | null;
  algorithmSingularTrials: number | null;
  solutions: number;
  done: boolean;
}

export function useSearchController(compiled: CompiledProblem | null) {
  const workersRef = useRef<Worker[]>([]);
  const logCounterRef = useRef(0);

  const stopWorkers = useCallback(() => {
    for (const worker of workersRef.current) {
      worker.postMessage({ type: "stop" });
      worker.terminate();
    }
    workersRef.current = [];
  }, []);

  const [snapshot, setSnapshot] = useState<SearchSnapshot>(INITIAL_SNAPSHOT);

  const start = useCallback(
    (
      workerCount: number,
      compiledOverride?: CompiledProblem,
      strategy: SearchStrategy = "ordered"
    ) => {
      const problem = compiledOverride ?? compiled;
      if (!problem) {
        return;
      }

      stopWorkers();
      const plan = createSearchPlan(problem, workerCount, strategy);
      const ranges = plan.ranges;
      const startedAt = Date.now();
      const permutation = createSearchPermutation(
        problem.assignmentCount,
        startedAt ^ logCounterRef.current
      );
      const miharaFrame = buildMiharaPositiveRegressionDataFrame(problem);
      const miharaObservations = miharaFrame.rows.map((row) => ({
        coefficients: problem.variables.map((variable) => row.coefficients[variable.name] ?? 0),
        target: row.target,
        weight: row.weight,
        source: row.source
      }));
      const lanes = ranges.map((range) => ({
        ...range,
        tested: 0,
        currentMask: range.start,
        speed: 0,
        bestLoss: null,
        bestMask: null,
        bestCoordinates: null,
        algorithmScore: null,
        algorithmTotal: null,
        algorithmLoss: null,
        algorithmSuccessfulTrials: null,
        algorithmSingularTrials: null,
        solutions: 0,
        done: false
      }));

      setSnapshot({
        ...INITIAL_SNAPSHOT,
        status: "running",
        startedAt,
        strategy,
        workUnits: plan.workUnits,
        unbounded: plan.unbounded,
        workLabel: plan.workLabel,
        chartFloor: strategy === "mihara"
          ? miharaFrame.satisfiableFloor
          : problem.scoring.theoreticalFloor,
        lanes,
        logs: [
          {
            id: ++logCounterRef.current,
            text: plan.unbounded
              ? `[sys] ${strategyDescription(strategy)}: fresh starts continue across ${ranges.length} threads until a decoded fit satisfies the CSP or the attempt is stopped`
              : `[sys] ${strategyDescription(strategy)}: ${plan.workUnits.toLocaleString()} ${plan.workLabel} across ${ranges.length} threads`
          }
        ]
      });

      const nextWorkers = ranges.map((range) => {
        const worker = new Worker(
          new URL("../worker/searchWorker.ts", import.meta.url),
          { type: "module" }
        );

        worker.onmessage = (event: MessageEvent<WorkerProgress>) => {
          setSnapshot((previous) => applyProgress(previous, event.data));
          if (
            strategy === "mihara" &&
            event.data.done &&
            event.data.solutions > 0
          ) {
            stopWorkers();
          }
        };

        worker.postMessage({
          type: "start",
          workerId: range.workerId,
          evaluatorSource: problem.evaluatorSource,
          lossFloor: problem.scoring.theoreticalFloor,
          assignmentCount: problem.assignmentCount,
          variableCount: problem.variables.length,
          strategy,
          permutation,
          miharaObservations,
          prime: problem.scoring.prime,
          start: range.start,
          endExclusive: range.endExclusive,
          unbounded: plan.unbounded,
          updateEveryMs: 220
        });

        return worker;
      });

      workersRef.current = nextWorkers;
    },
    [compiled, stopWorkers]
  );

  const pause = useCallback(() => {
    stopWorkers();
    setSnapshot((previous) => ({ ...previous, status: "paused" }));
  }, [stopWorkers]);

  const reset = useCallback(() => {
    stopWorkers();
    setSnapshot(INITIAL_SNAPSHOT);
  }, [stopWorkers]);

  const bestAssignment = useMemo(() => {
    if (!compiled || snapshot.bestMask == null) {
      return null;
    }
    return maskToAssignment(compiled, snapshot.bestMask);
  }, [compiled, snapshot.bestMask]);

  return {
    snapshot,
    bestAssignment,
    start,
    pause,
    reset
  };
}

function applyProgress(
  previous: SearchSnapshot,
  progress: WorkerProgress
): SearchSnapshot {
  const recoveredMiharaSolution =
    previous.strategy === "mihara" && progress.done && progress.solutions > 0;
  const lanes = previous.lanes.map((lane) =>
    lane.workerId === progress.workerId
      ? {
          ...lane,
          tested: progress.tested,
          currentMask: progress.currentMask,
          speed: progress.speed,
          bestLoss: progress.bestLoss,
          bestMask: progress.bestMask,
          bestCoordinates: progress.bestCoordinates,
          algorithmScore: progress.algorithmScore,
          algorithmTotal: progress.algorithmTotal,
          algorithmLoss: progress.algorithmLoss,
          algorithmSuccessfulTrials: progress.algorithmSuccessfulTrials,
          algorithmSingularTrials: progress.algorithmSingularTrials,
          solutions: progress.solutions,
          done: progress.done
        }
      : lane
  ).map((lane) => recoveredMiharaSolution ? { ...lane, done: true } : lane);

  const laneBest = previous.strategy === "mihara"
    ? lanes
      .filter((lane) => lane.algorithmLoss != null)
      .sort((a, b) => (a.algorithmLoss ?? Infinity) - (b.algorithmLoss ?? Infinity))[0]
    : lanes
      .filter((lane) => lane.bestLoss != null)
      .sort((a, b) => (a.bestLoss ?? Infinity) - (b.bestLoss ?? Infinity))[0];
  const totalTested = lanes.reduce((sum, lane) => sum + lane.tested, 0);
  const totalSpeed = lanes.reduce((sum, lane) => sum + lane.speed, 0);
  const solutions = lanes.reduce((sum, lane) => sum + lane.solutions, 0);
  const bestLoss = laneBest?.bestLoss ?? previous.bestLoss;
  const bestMask = laneBest?.bestMask ?? previous.bestMask;
  const bestCoordinates = laneBest?.bestCoordinates ?? previous.bestCoordinates;
  const algorithmScore = laneBest?.algorithmScore ?? previous.algorithmScore;
  const algorithmTotal = laneBest?.algorithmTotal ?? previous.algorithmTotal;
  const algorithmLoss = laneBest?.algorithmLoss ?? previous.algorithmLoss;
  const algorithmSuccessfulTrials = lanes.reduce(
    (sum, lane) => sum + (lane.algorithmSuccessfulTrials ?? 0),
    0
  );
  const algorithmSingularTrials = lanes.reduce(
    (sum, lane) => sum + (lane.algorithmSingularTrials ?? 0),
    0
  );
  const done = lanes.length > 0 && lanes.every((lane) => lane.done);
  const changedBest = previous.strategy === "mihara"
    ? algorithmScore != null && algorithmScore !== previous.algorithmScore
    : bestLoss != null && bestLoss !== previous.bestLoss;
  const trackedLoss = previous.strategy === "mihara" ? algorithmLoss : bestLoss;
  const previousTrackedLoss = previous.strategy === "mihara"
    ? previous.algorithmLoss
    : previous.bestLoss;
  const changedTrackedLoss = trackedLoss != null && trackedLoss !== previousTrackedLoss;
  const lastHistoryPoint = previous.history[previous.history.length - 1];
  const appendCompletionPlateau =
    done && trackedLoss != null && lastHistoryPoint?.tested !== totalTested;
  const completedWithoutMiharaFit =
    done && previous.status !== "complete" && previous.strategy === "mihara" && bestCoordinates == null;

  const nextLogs = changedBest
    ? [
        ...previous.logs,
        {
          id: previous.logs.length + 1,
          text: previous.strategy === "mihara"
            ? `[t${progress.workerId}] positive consensus ${algorithmScore}/${algorithmTotal}; loss ${algorithmLoss}; ${bestMask == null ? "not Boolean" : `candidate H${bestMask}`}`
            : `[t${progress.workerId}] new best p-adic loss ${bestLoss} at hyperplane ${
              progress.bestMask == null ? "-" : `H${progress.bestMask}`
            }`
        }
      ]
    : completedWithoutMiharaFit
      ? [
          ...previous.logs,
          {
            id: previous.logs.length + 1,
            text: `[sys] attempt complete: no full-rank equality sample in ${algorithmSingularTrials} RANSAC trials`
          }
        ]
      : previous.logs;

  return {
    ...previous,
    status: done ? "complete" : previous.status,
    lanes,
    totalTested,
    totalSpeed,
    solutions,
    bestLoss,
    bestMask,
    bestCoordinates,
    algorithmScore,
    algorithmTotal,
    algorithmLoss,
    algorithmSuccessfulTrials,
    algorithmSingularTrials,
    history:
      trackedLoss == null || (!changedTrackedLoss && !appendCompletionPlateau)
        ? previous.history
        : [...previous.history, { tested: totalTested, loss: trackedLoss }].slice(-160),
    logs: nextLogs.slice(-12)
  };
}

function strategyDescription(strategy: SearchStrategy): string {
  if (strategy === "random") return "randomly permuted exhaustive scan";
  if (strategy === "zubarev") return "Zubarev single-bit walk";
  if (strategy === "mihara") return "Mihara positive-complement digitwise fit";
  return "ordered exhaustive scan";
}
