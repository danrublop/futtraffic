/** Timeline zoom.
 *
 *  Same model — pixels-per-second plus a scroll offset, zoom anchored on the
 *  cursor, wheel bindings, ResizeObserver in a layout effect so the first
 *  paint is not at width 1. The one change is where the state lives: the original
 *  keeps zoom and scroll in its Zustand store because its agents write to
 *  them; nothing here does, so they are local. */
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

import { MAX_PPS, MIN_PPS } from "./constants";

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function useTimelineZoom(
  containerRef: React.RefObject<HTMLDivElement | null>,
  duration: number,
  /** Pixels reserved for non-scrollable UI (the lane-name gutter). */
  reservedWidth = 0,
) {
  const [zoom, setZoom] = useState(0);
  const [scrollXState, setScrollX] = useState(0);
  /** Set when the user pans by hand: the playhead stops dragging the view
   *  around under them until playback restarts. */
  const [followPaused, setFollowPaused] = useState(false);
  const [containerWidth, setContainerWidth] = useState(1);

  // useLayoutEffect so the width lands before the browser paints — measured in
  // a plain effect, every open flashes one frame of a fully-collapsed timeline.
  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) setContainerWidth(e.contentRect.width || 1);
    });
    ro.observe(el);
    setContainerWidth(el.clientWidth || 1);
    return () => ro.disconnect();
  }, [containerRef]);

  const usableWidth = Math.max(1, containerWidth - reservedWidth);
  const d = Math.max(duration, 0.1);
  // Zoom 0 means "fit": the whole match across the viewport, which is the only
  // sensible opening state for a thing whose job is finding a minute in ninety.
  const pps = zoom === 0 ? usableWidth / d : clamp(zoom, MIN_PPS, MAX_PPS);
  const totalWidth = Math.max(usableWidth, d * pps);
  const maxScrollX = Math.max(0, totalWidth - usableWidth);
  // Clamped on read rather than corrected in an effect: a resize or a zoom-out
  // shrinks maxScrollX, and an effect that fixes it up afterwards renders one
  // frame scrolled past the end first.
  const scrollX = Math.min(scrollXState, maxScrollX);

  const scrollTo = useCallback(
    (x: number) => setScrollX(clamp(x, 0, maxScrollX)),
    [maxScrollX],
  );

  const zoomAtCursor = useCallback(
    (nextPPS: number, cursorClientX: number) => {
      const el = containerRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const cursorOffset = cursorClientX - rect.left - reservedWidth;
      const timeAtCursor = (scrollX + cursorOffset) / pps;
      const clamped = clamp(nextPPS, Math.max(MIN_PPS, usableWidth / d), MAX_PPS);
      const nextMax = Math.max(0, d * clamped - usableWidth);
      setZoom(clamped);
      setScrollX(clamp(timeAtCursor * clamped - cursorOffset, 0, nextMax));
    },
    [containerRef, pps, scrollX, usableWidth, d, reservedWidth],
  );

  const zoomBy = useCallback(
    (factor: number) => {
      const el = containerRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      zoomAtCursor(pps * factor, rect.left + reservedWidth + usableWidth / 2);
    },
    [containerRef, pps, usableWidth, reservedWidth, zoomAtCursor],
  );

  const fitAll = useCallback(() => {
    setZoom(0);
    setScrollX(0);
    setFollowPaused(false);
  }, []);

  // Wheel: modifier zooms at the cursor, everything else pans. Registered by
  // hand because React's onWheel is passive and cannot preventDefault.
  const state = useRef({ pps, scrollX, maxScrollX });
  useEffect(() => {
    state.current = { pps, scrollX, maxScrollX };
  });
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      const s = state.current;
      if (e.ctrlKey || e.metaKey || e.altKey) {
        e.preventDefault();
        zoomAtCursor(s.pps * (e.deltaY > 0 ? 0.9 : 1.1), e.clientX);
        return;
      }
      // Two sources of a horizontal pan: shift+wheel, and a trackpad two-finger
      // swipe (deltaX). Dreambyte takes a plain vertical wheel here as well,
      // which is right for a full-screen NLE and wrong here: this timeline is
      // one block in a page that scrolls, and swallowing deltaY would trap the
      // page under it. Vertical wheel is left to the page.
      const delta = e.shiftKey ? e.deltaY : e.deltaX;
      if (!delta) return;
      e.preventDefault();
      const next = clamp(s.scrollX + delta, 0, s.maxScrollX);
      if (next !== s.scrollX) {
        setFollowPaused(true);
        setScrollX(next);
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [containerRef, zoomAtCursor]);

  return {
    pps,
    fitted: zoom === 0,
    scrollX,
    scrollTo,
    maxScrollX,
    usableWidth,
    followPaused,
    setFollowPaused,
    zoomIn: () => zoomBy(1.5),
    zoomOut: () => zoomBy(1 / 1.5),
    fitAll,
  };
}
