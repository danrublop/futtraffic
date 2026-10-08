"use client";

/** The match as an editor sees it, laid out the way an NLE lays out
 *  its timeline: a ruler across the top, a tool column
 *  down the left, named track headers beside it, and the tracks themselves on
 *  one scrolling surface with the playhead over everything.
 *
 *  Everything about how a timeline behaves is the NLE's and was not
 *  reinvented — pixels-per-second with a scroll offset (`use-timeline-zoom`),
 *  the virtualized nice-number ruler (`time-ruler`), rAF-coalesced scrubbing
 *  (`use-playhead-drag`), the Premiere-style playhead tab, the clip block with
 *  its dark label bar, the `--tl-*` palette. What is ours is what goes in the
 *  tracks: this app reads a match, it does not cut one, so there are no razor
 *  or trim tools and the two tracks are fixed — the film, and the clips you
 *  marked off it. */

import { useCallback, useEffect, useMemo, useRef } from "react";

import {
  LEFT_GUTTER,
  RULER_HEIGHT,
  TRACK_HEADER_WIDTH,
  TRACK_ROW_HEIGHT,
} from "@/components/timeline/constants";
import { TimeRuler, formatTimecode } from "@/components/timeline/time-ruler";
import { usePlayheadDrag } from "@/components/timeline/use-playhead-drag";
import { useTimelineZoom } from "@/components/timeline/use-timeline-zoom";
import type { Moment } from "@/lib/moments";
import { clockTenths } from "@/lib/timecode";
import type { Clip } from "@/lib/types";

const TICK_COLOR: Record<string, string> = { home: "bg-home", away: "bg-away" };

export interface TrackTimelineProps {
  dur: number;
  clips: Clip[];
  moments: Moment[];
  picked: Set<string>;
  editing: string | null;
  io: { in: number | null; out: number | null };
  onSeek: (t: number) => void;
  onPlayClip: (c: Clip) => void;
  /** Filled on mount with this timeline's two entry points.
   *
   *  `paint` is driven from the same per-frame loop that paints the bar on the
   *  video, so the playhead moves without React rendering
   *  anything. `resumeFollow` is called when playback starts: a hand scrub
   *  stops the view chasing the playhead, and the rule is that pressing
   *  play is what gives the chase back. */
  paintRef: React.RefObject<{
    paint: (t: number) => void;
    resumeFollow: () => void;
  } | null>;
}

export function TrackTimeline({ paintRef, ...p }: TrackTimelineProps) {
  const boxRef = useRef<HTMLDivElement>(null);
  const headRef = useRef<HTMLDivElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const z = useTimelineZoom(boxRef, p.dur, LEFT_GUTTER);
  const { pps, scrollX, usableWidth, scrollTo, followPaused, setFollowPaused } = z;

  // Everything worth landing on exactly, for the drag to snap to.
  const snapTimes = useMemo(
    () => p.moments.flatMap((m) => (m.end != null ? [m.t, m.end] : [m.t])),
    [p.moments],
  );
  const { isDragging, onPointerDown } = usePlayheadDrag({
    pps,
    scrollX,
    containerRef: boxRef,
    snapTimes,
    onSeek: p.onSeek,
    leftOffset: LEFT_GUTTER,
    onScrubStart: () => setFollowPaused(true),
  });

  /** The last time we were painted with. A zoom or a pan moves the playhead
   *  without the video moving, and paused there is no next frame to correct
   *  it on — so the effect below repaints from here whenever the view changes. */
  const lastT = useRef(0);

  /** DOM only. Split from the follow so the effect can call it without
   *  scheduling a render. */
  const place = useCallback(
    (t: number) => {
      if (tipRef.current) tipRef.current.textContent = clockTenths(t);
      const el = headRef.current;
      if (!el) return;
      const x = t * pps - scrollX;
      el.style.transform = `translateX(${x}px)`;
      // Off screen is hidden rather than clamped: a playhead pinned to the edge
      // of the view is a lie about where the video is.
      el.style.visibility = x < -1 || x > usableWidth + 1 ? "hidden" : "";
    },
    [pps, scrollX, usableWidth],
  );

  useEffect(() => {
    paintRef.current = {
      paint: (t: number) => {
        lastT.current = t;
        place(t);
        // Dreambyte's page-scroll rule: once the playhead passes 80% of the
        // view, jump so it sits at 25% and let it run again. Fires once a
        // screenful, so the setState costs nothing during playback.
        if (followPaused || isDragging) return;
        const rel = (t * pps - scrollX) / usableWidth;
        if (rel > 0.8 || rel < 0) scrollTo(t * pps - usableWidth * 0.25);
      },
      resumeFollow: () => setFollowPaused(false),
    };
    place(lastT.current);
    return () => {
      paintRef.current = null;
    };
  }, [
    paintRef,
    place,
    pps,
    scrollX,
    usableWidth,
    scrollTo,
    followPaused,
    isDragging,
    setFollowPaused,
  ]);

  // One row of marks, on the film itself. Side is still the colour — it was
  // never worth a track of its own.
  const marks = p.moments.filter((m) => m.kind !== "clip");

  return (
    <div
      ref={boxRef}
      className="relative h-full w-full shrink-0 overflow-hidden select-none"
      /* No border-top: the divider above IS that line, and drawn twice it
         reads as a 2px seam. */
      style={{ background: "var(--tl-bg)" }}
    >
      {/* Ruler row. Dreambyte fills the corner above the headers with a play
          button and a timecode; ours is EMPTY. It held zoom and fit for a
          while, and before that a 32px tool column down the left — three
          controls, pressed twice a session, taking a permanent bite out of
          every row. Zoom is ⌘/ctrl+wheel and pan is shift+wheel on the tracks
          themselves, which is where the hand already is. The corner cannot go:
          it is what keeps the ruler in step with the tracks beside it. */}
      <div className="flex">
        <div
          className="shrink-0"
          style={{
            width: LEFT_GUTTER,
            height: RULER_HEIGHT,
            background: "var(--tl-bg)",
            borderBottom: "1px solid var(--tl-border)",
            borderRight: "1px solid var(--tl-border)",
          }}
        />
        <div className="min-w-0 flex-1">
          <TimeRuler
            pps={pps}
            scrollX={scrollX}
            containerWidth={usableWidth}
            duration={p.dur}
            inPoint={p.io.in}
            outPoint={p.io.out}
          />
        </div>
      </div>

      {/* Content row: track headers | tracks. */}
      <div className="flex">
        {/* Track headers. */}
        <div className="shrink-0" style={{ width: TRACK_HEADER_WIDTH }}>
          <Header name="V1" sub="Film" />
          <Header name="V2" sub={`${p.clips.length} clip${p.clips.length === 1 ? "" : "s"}`} />
        </div>

        {/* One scrolling surface: the ruler above and both tracks are
            translated by the same scrollX, so nothing can drift out of step
            with the clock. */}
        <div
          className="relative min-w-0 flex-1 cursor-col-resize touch-none overflow-hidden"
          onPointerDown={onPointerDown}
        >
          {/* The film. One block for the whole match, every moment marked on
              it. ponytail: no filmstrip thumbnails — an NLE decodes its own
              media locally, this one would be a frame-every-few-seconds job on
              the server, and the matches list dropped its single thumbnail for
              exactly that cost. */}
          <Row scrollX={scrollX}>
            <Block left={0} width={Math.max(2, p.dur * pps)} color="var(--tl-clip-video)" label="Film" />
            {marks.map((m, i) => (
              <button
                key={`${m.t}-${i}`}
                type="button"
                title={`${m.label}${m.detail ? ` · ${m.detail}` : ""} · ${formatTimecode(m.t)}${
                  m.exact ? "" : " (minute only)"
                }`}
                onPointerDown={(e) => e.stopPropagation()}
                onClick={() => p.onSeek(m.t)}
                // 2px of paint inside 9px of hit area — narrow enough that
                // neighbours stay readable, wide enough to hit once the zoom
                // separates them.
                className="absolute inset-y-0 grid w-[9px] -translate-x-1/2 place-items-end pb-1"
                style={{ left: m.t * pps }}
              >
                {/* Minute-only marks are drawn short and faint: they are placed
                    to within 30s and a full-height one would read as a
                    timecode. */}
                <i
                  className={`block w-0.5 ${TICK_COLOR[m.side ?? ""] ?? "bg-muted-foreground"} ${
                    m.exact ? "h-6" : "h-3 opacity-60"
                  }`}
                />
              </button>
            ))}
          </Row>

          {/* The cut: the range you are marking, and the clips saved off it. */}
          <Row scrollX={scrollX} alt>
            {p.io.in != null && p.io.out != null && p.io.out > p.io.in && (
              <i
                className="border-primary bg-primary/20 absolute inset-y-0 block border-x-2"
                style={{ left: p.io.in * pps, width: (p.io.out - p.io.in) * pps }}
              />
            )}
            {p.clips.map((c) => (
              <Block
                key={c.id}
                left={c.start * pps}
                width={Math.max(2, (c.end - c.start) * pps)}
                color={p.picked.has(c.id) ? "var(--tl-clip-title)" : "var(--tl-clip-audio)"}
                ring={p.editing === c.id}
                label={c.name}
                title={`${c.name} (${formatTimecode(c.start)}–${formatTimecode(c.end)})`}
                onClick={() => p.onPlayClip(c)}
              />
            ))}
          </Row>
        </div>
      </div>

      {/* Playhead — the filled tab, the 1px line, the
          timecode bubble while you scrub. Outside the translated rows: paint
          writes its absolute x, so a scroll mid-frame cannot double-apply. */}
      <div
        className="pointer-events-none absolute z-20"
        style={{ top: 0, bottom: 0, left: LEFT_GUTTER, right: 0, overflow: "hidden" }}
      >
        <div ref={headRef} className="absolute inset-y-0" style={{ left: 0, width: 0 }}>
          <div
            className="pointer-events-auto absolute cursor-col-resize"
            style={{
              top: 0,
              left: -7,
              width: 14,
              height: 10,
              background: "var(--tl-playhead)",
              borderRadius: "0 0 3px 3px",
            }}
            onPointerDown={onPointerDown}
          />
          {isDragging && (
            <div
              ref={tipRef}
              className="absolute rounded-[3px] px-1 font-mono whitespace-nowrap text-white"
              style={{
                top: 12,
                left: 8,
                fontSize: 9,
                lineHeight: "14px",
                background: "var(--tl-playhead)",
              }}
            />
          )}
          <div
            className="absolute inset-y-0"
            style={{ left: -0.5, width: 1, background: "var(--tl-playhead)" }}
          />
        </div>
      </div>
    </div>
  );
}

/** A track header: the track's name, what is on it, and any control that
 *  belongs to the track rather than to the clock. */
function Header({
  name,
  sub,
  children,
}: {
  name: string;
  sub: string;
  children?: React.ReactNode;
}) {
  return (
    <div
      className="mt-px flex items-center gap-1.5 px-2"
      style={{
        height: TRACK_ROW_HEIGHT,
        background: "var(--tl-header-bg)",
        borderRight: "1px solid var(--tl-border)",
      }}
    >
      <span className="text-[12px] font-bold" style={{ color: "var(--tl-ctrl-text)" }}>
        {name}
      </span>
      <span className="min-w-0 flex-1 truncate text-[11px]" style={{ color: "var(--tl-toolbar-text)" }}>
        {sub}
      </span>
      {children}
    </div>
  );
}

/** One track row. The transform is on the row, not on each block: a row with
 *  sixty marks in it is one transform, not sixty positions to recompute. */
function Row({
  scrollX,
  alt,
  children,
}: {
  scrollX: number;
  alt?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div
      className="relative mt-px overflow-hidden"
      style={{
        height: TRACK_ROW_HEIGHT,
        background: alt ? "var(--tl-track-alt)" : "var(--tl-track-bg)",
      }}
    >
      <div className="absolute inset-0" style={{ transform: `translateX(${-scrollX}px)` }}>
        {children}
      </div>
    </div>
  );
}

/** A clip block: 3px inset top and bottom,
 *  a hairline border, and the name in a dark strip across the top so it stays
 *  legible whatever the block is filled with. */
function Block({
  left,
  width,
  color,
  label,
  title,
  ring,
  onClick,
}: {
  left: number;
  width: number;
  color: string;
  label: string;
  title?: string;
  ring?: boolean;
  onClick?: () => void;
}) {
  return (
    <div
      title={title}
      onPointerDown={onClick && ((e) => e.stopPropagation())}
      onClick={onClick}
      className={`absolute overflow-hidden rounded-[2px] ${onClick ? "cursor-pointer" : ""}`}
      style={{
        left,
        width,
        top: 3,
        bottom: 3,
        background: color,
        border: "1px solid var(--tl-clip-border)",
        boxShadow: ring ? "0 0 0 2px var(--accent)" : undefined,
      }}
    >
      {width > 30 && (
        <span
          className="pointer-events-none absolute top-0 right-0 left-0 truncate"
          style={{
            height: 15,
            lineHeight: "15px",
            padding: "0 6px",
            fontSize: 11,
            fontWeight: 600,
            color: "var(--tl-clip-label)",
            background: "rgba(0, 0, 0, 0.32)",
          }}
        >
          {label}
        </span>
      )}
    </div>
  );
}
