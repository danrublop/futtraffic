/** Playhead dragging, essentially
 *  unchanged. The rAF coalescing is the reason it is worth porting rather than
 *  writing again: a pointermove fires faster than the browser paints, and
 *  seeking a video on every one of them is the thing that makes a scrub feel
 *  like it is fighting you.
 *
 *  `sceneBoundaries` became `snapTimes` — here the things worth landing on are
 *  moments and clip edges, not cuts. */
import { useCallback, useRef, useState } from "react";

import { SNAP_THRESHOLD } from "./constants";

interface Options {
  pps: number;
  scrollX: number;
  containerRef: React.RefObject<HTMLDivElement | null>;
  snapTimes: number[];
  onSeek: (time: number) => void;
  /** Pixel offset from the container's left edge to where time = 0 begins. */
  leftOffset?: number;
  onScrubStart?: () => void;
  onScrubEnd?: () => void;
}

export function usePlayheadDrag({
  pps,
  scrollX,
  containerRef,
  snapTimes,
  onSeek,
  leftOffset = 0,
  onScrubStart,
  onScrubEnd,
}: Options) {
  const [isDragging, setIsDragging] = useState(false);
  const draggingRef = useRef(false);

  const computeTime = useCallback(
    (clientX: number) => {
      const el = containerRef.current;
      if (!el) return 0;
      const rect = el.getBoundingClientRect();
      const cursorPx = clientX - rect.left - leftOffset;
      const time = (scrollX + cursorPx) / pps;
      // Snap to the nearest moment within SNAP_THRESHOLD px.
      for (const s of snapTimes) {
        if (Math.abs(s * pps - scrollX - cursorPx) <= SNAP_THRESHOLD) return s;
      }
      return Math.max(0, time);
    },
    [pps, scrollX, containerRef, snapTimes, leftOffset],
  );

  const onPointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      draggingRef.current = true;
      setIsDragging(true);
      onScrubStart?.();
      onSeek(computeTime(e.clientX));

      // rAF-coalesce scrubbing so onSeek fires at most once per frame.
      let pendingX: number | null = null;
      let rafId: number | null = null;
      const applyMove = () => {
        rafId = null;
        if (!draggingRef.current || pendingX == null) return;
        const t = computeTime(pendingX);
        pendingX = null;
        onSeek(t);
      };
      const onMove = (ev: PointerEvent) => {
        if (!draggingRef.current) return;
        pendingX = ev.clientX;
        if (rafId !== null) return;
        rafId = requestAnimationFrame(applyMove);
      };
      const onEnd = () => {
        if (!draggingRef.current) return;
        if (rafId !== null) {
          cancelAnimationFrame(rafId);
          rafId = null;
        }
        if (pendingX != null) applyMove();
        draggingRef.current = false;
        setIsDragging(false);
        onScrubEnd?.();
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onEnd);
        window.removeEventListener("pointercancel", onEnd);
      };
      // On window, not the element: a scrub that leaves the timeline still has
      // to keep scrubbing, and pointer capture on the element does not survive
      // the pointer crossing into another one.
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onEnd);
      window.addEventListener("pointercancel", onEnd);
    },
    [computeTime, onSeek, onScrubStart, onScrubEnd],
  );

  return { isDragging, onPointerDown };
}
