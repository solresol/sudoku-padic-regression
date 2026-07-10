import { useCallback, useMemo, useRef, useState } from "react";
import type { CompiledProblem } from "../lib/csp";
import {
  createSearchPermutation,
  maskToAssignment,
  splitAssignmentRanges,
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
  lanes: WorkerLane[];
  bestLoss: number | null;
  bestMask: number | null;
  totalTested: number;
  totalSpeed: number;
  solutions: number;
  history: Array<{ tested: number; loss: number }>;
  logs: SearchLogEntry[];
}

const INITIAL_SNAPSHOT: SearchSnapshot = {
  status: "idle",
  startedAt: null,
  lanes: [],
  bestLoss: null,
  bestMask: null,
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
      const ranges = splitAssignmentRanges(problem.variables.length, workerCount);
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
        solutions: 0,
        done: false
      }));

      setSnapshot({
        ...INITIAL_SNAPSHOT,
        status: "running",
        startedAt,
        lanes,
        logs: [
          {
            id: ++logCounterRef.current,
            text: `[sys] ${strategy === "random" ? "randomly permuted" : "ordered"} scan of ${problem.assignmentCount.toLocaleString()} candidate hyperplanes across ${ranges.length} threads`
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
        };

        worker.postMessage({
          type: "start",
          workerId: range.workerId,
          evaluatorSource: problem.evaluatorSource,
          lossFloor: problem.scoring.theoreticalFloor,
          assignmentCount: problem.assignmentCount,
          strategy,
          permutation,
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
          solutions: progress.solutions,
          done: progress.done
        }
      : lane
  );

  const laneBest = lanes
    .filter((lane) => lane.bestLoss != null)
    .sort((a, b) => (a.bestLoss ?? Infinity) - (b.bestLoss ?? Infinity))[0];
  const totalTested = lanes.reduce((sum, lane) => sum + lane.tested, 0);
  const totalSpeed = lanes.reduce((sum, lane) => sum + lane.speed, 0);
  const solutions = lanes.reduce((sum, lane) => sum + lane.solutions, 0);
  const bestLoss = laneBest?.bestLoss ?? previous.bestLoss;
  const bestMask = laneBest?.bestMask ?? previous.bestMask;
  const done = lanes.length > 0 && lanes.every((lane) => lane.done);
  const changedBest = bestLoss != null && bestLoss !== previous.bestLoss;
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
    history:
      bestLoss == null || (!changedBest && !appendCompletionPlateau)
        ? previous.history
        : [...previous.history, { tested: totalTested, loss: bestLoss }].slice(-160),
    logs: changedBest
      ? [
          ...previous.logs,
          {
            id: previous.logs.length + 1,
            text: `[t${progress.workerId}] new best p-adic loss ${bestLoss} at hyperplane ${
              progress.bestMask == null ? "-" : `H${progress.bestMask}`
            }`
          }
        ].slice(-12)
      : previous.logs
  };
}
