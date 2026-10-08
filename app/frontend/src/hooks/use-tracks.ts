"use client";

import { useEffect, useState } from "react";

import { api } from "@/lib/api";

/** tracks.json, streamed and indexed in a worker.
 *
 *  The file is tens to hundreds of MB. `JSON.parse` on that needs the text AND
 *  the object graph resident at once, on the main thread, with no progress and
 *  no way back if it OOMs — so /public/tracks-worker.js streams it into typed
 *  arrays instead and this hook only carries the result across.
 */

export interface TrackRow {
  id: number;
  team: string | null;
  jersey: number | null;
  n: number;
  firstF: number;
  lastF: number;
  /** Row indices into the columns below, in frame order. */
  idx: Uint32Array;
}

export interface TracksIndex {
  fps: number;
  n: number;
  fmin: number;
  fmax: number;
  span: number;
  F: Int32Array;
  ID: Int32Array;
  X: Float32Array;
  Y: Float32Array;
  /** Rows for frame `fmin + k` are `start[k] .. start[k + 1]`. */
  start: Int32Array;
  /** Last frame at or before k that has any detection. Detections are routinely
   *  sparse, so a bare per-frame lookup would blank the overlay 4 frames in 5. */
  prev: Int32Array;
  tracks: TrackRow[];
  bytes: number;
  ms: number;
  /** Set when the stream ended before the detections array closed. */
  truncated: string | null;
}

export interface TracksProgress {
  bytes: number;
  total: number;
  n: number;
}

type WorkerMsg =
  | ({ type: "progress" } & TracksProgress)
  | { type: "done"; empty: true; truncated: string | null }
  | ({ type: "done"; empty?: undefined } & TracksIndex)
  | { type: "error"; message: string };

interface State {
  /** Which match this result belongs to. Everything is derived against the id
   *  being asked for, so switching match shows nothing rather than the previous
   *  match's rings — and no state has to be reset from inside the effect. */
  id: string;
  data: TracksIndex | null;
  progress: TracksProgress | null;
  error: string | null;
}

const EMPTY = { data: null, progress: null, error: null };

export function useTracks(matchId: string | null) {
  const [state, setState] = useState<State | null>(null);

  useEffect(() => {
    if (!matchId) return;

    const worker = new Worker("/tracks-worker.js");
    const put = (s: Omit<State, "id">) => setState({ id: matchId, ...s });
    worker.onmessage = (e: MessageEvent) => {
      const m = e.data as WorkerMsg;
      if (m.type === "progress") return put({ ...EMPTY, progress: m });
      if (m.type === "error") return put({ ...EMPTY, error: m.message });
      if (m.empty) {
        return put({ ...EMPTY, error: m.truncated ?? "the tracks file holds no detections" });
      }
      put({ ...EMPTY, data: m });
    };
    worker.onerror = () => put({ ...EMPTY, error: "the tracks worker could not start" });
    // Deliberately no calibUrl: this overlay draws image pixels and needs no
    // metres, and asking for them makes the worker derive a speed series for
    // every track before the page can draw a single ring.
    worker.postMessage({ tracksUrl: api.tracksUrl(matchId) });

    return () => worker.terminate();
  }, [matchId]);

  return state && state.id === matchId ? state : EMPTY;
}
