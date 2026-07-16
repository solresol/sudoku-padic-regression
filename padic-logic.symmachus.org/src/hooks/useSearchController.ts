import { useCallback, useMemo, useRef, useState } from "react";
import { buildRegressionDataFrame, type CompiledProblem } from "../lib/csp";
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
  workLabel: "hyperplanes" | "walk steps" | "RANSAC trials";
  lanes: WorkerLane[];
  bestLoss: number | null;
  bestMask: number | null;
  bestCoordinates: number[] | null;
  algorithmScore: number | null;
  algorithmTotal: number | null;
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
  workLabel: "hyperplanes",
  lanes: [],
  bestLoss: null,
  bestMask: null,
  bestCoordinates: null,
  algorithmScore: null,
  algorithmTotal: null,
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
        solutions: 0,
        done: false
      }));

      setSnapshot({
        ...INITIAL_SNAPSHOT,
        status: "running",
        startedAt,
        strategy,
        workUnits: plan.workUnits,
        workLabel: plan.workLabel,
        lanes,
        logs: [
          {
            id: ++logCounterRef.current,
            text: `[sys] ${strategyDescription(strategy)}: ${plan.workUnits.toLocaleString()} ${plan.workLabel} across ${ranges.length} threads`
          }
        ]
      });

      const miharaObservations = buildRegressionDataFrame(problem).map((row) => ({
        coefficients: problem.variables.map((variable) => row.coefficients[variable.name] ?? 0),
        target: row.target,
        source: row.source
      }));

      const nextWorkers = ranges.map((range) => {
        const worker = new Worker(
          new URL("../worker/searchWorker.ts", import.meta.url),
          { type: "module" }
        );

        worker.onmessage = (event: MessageEvent<WorkerProgress>) => {
          setSnapshot((previous) => applyProgress(previous, event.data));
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
          solutions: progress.solutions,
          done: progress.done
        }
      : lane
  );

  const laneBest = previous.strategy === "mihara"
    ? lanes
      .filter((lane) => lane.algorithmScore != null)
      .sort((a, b) => (b.algorithmScore ?? -Infinity) - (a.algorithmScore ?? -Infinity))[0]
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
  const done = lanes.length > 0 && lanes.every((lane) => lane.done);
  const changedBest = previous.strategy === "mihara"
    ? algorithmScore != null && algorithmScore !== previous.algorithmScore
    : bestLoss != null && bestLoss !== previous.bestLoss;
  const lastHistoryPoint = previous.history[previous.history.length - 1];
  const appendCompletionPlateau =
    done && bestLoss != null && lastHistoryPoint?.tested !== totalTested;

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
    history:
      bestLoss == null || (!changedBest && !appendCompletionPlateau)
        ? previous.history
        : [...previous.history, { tested: totalTested, loss: bestLoss }].slice(-160),
    logs: changedBest
      ? [
          ...previous.logs,
          {
            id: previous.logs.length + 1,
            text: previous.strategy === "mihara"
              ? `[t${progress.workerId}] equality consensus ${algorithmScore}/${algorithmTotal}; ${bestMask == null ? "not Boolean" : `candidate H${bestMask}`}`
              : `[t${progress.workerId}] new best p-adic loss ${bestLoss} at hyperplane ${
                progress.bestMask == null ? "-" : `H${progress.bestMask}`
              }`
          }
        ].slice(-12)
      : previous.logs
  };
}

function strategyDescription(strategy: SearchStrategy): string {
  if (strategy === "random") return "randomly permuted exhaustive scan";
  if (strategy === "zubarev") return "Zubarev single-bit walk";
  if (strategy === "mihara") return "Mihara digitwise equality fit";
  return "ordered exhaustive scan";
}
