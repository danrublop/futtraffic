"use client";

/** The timeline's ruler.
 *
 *  `getTickInterval` and `formatTimecode` are that file's, unchanged — the
 *  nice-number series, the MIN_MAJOR_PX/MIN_MINOR_PX contract and the
 *  index-based tick loop are all load-bearing and were arrived at by fixing
 *  real bugs (crammed labels at mid zoom, major ticks blinking missing at
 *  non-integer intervals). Its test came across with them.
 *
 *  What was dropped: markers (this app draws its moments in lanes, not on the
 *  ruler) and frame-level display (a football clock has no frame column). */

import { RULER_HEIGHT } from "./constants";

// Minimum pixel spacing between adjacent major labels. Sized for the widest
// label we render ("H:MM:SS" ~ 52px at 10px monospace + padding).
const MIN_MAJOR_PX = 60;
// Minimum pixel spacing between adjacent minor ticks. Below this they
// visually merge into a solid line, which looks like noise.
const MIN_MINOR_PX = 8;

export function formatTimecode(s: number) {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`
    : `${m}:${String(sec).padStart(2, "0")}`;
}

/**
 * Premiere-style "nice" interval picker. We want the *pixel* spacing between
 * major ticks to stay roughly constant across zooms — only the time interval
 * each tick represents changes. We pick the smallest interval in a fixed
 * "nice number" series that yields at least MIN_MAJOR_PX between labels.
 *
 * The series is hand-tuned so adjacent intervals are 2×/2.5×/3× apart (no
 * ugly jumps like 1→10).
 */
export function getTickInterval(pps: number): { major: number; minor: number } {
  const candidates = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600, 7200];
  let major = candidates[candidates.length - 1];
  for (const c of candidates) {
    if (c * pps >= MIN_MAJOR_PX) {
      major = c;
      break;
    }
  }
  // Minor: divide major into the largest number of sub-ticks that still keeps
  // each minor >= MIN_MINOR_PX. We prefer 10ths, fall back to 5ths/4ths/2nds.
  let minor = major;
  for (const d of [10, 5, 4, 2]) {
    const m = major / d;
    if (m * pps >= MIN_MINOR_PX) {
      minor = m;
      break;
    }
  }
  return { major, minor };
}

interface Props {
  pps: number;
  scrollX: number;
  /** Width of the scrolling viewport, for virtualizing the tick loop. */
  containerWidth: number;
  duration: number;
  inPoint?: number | null;
  outPoint?: number | null;
}

/** Ticks are virtualized to the visible window: at 400 pps an hour of match is
 *  1.4 million pixels, and drawing every second of it is not a thing. */
export function TimeRuler({ pps, scrollX, containerWidth, duration, inPoint, outPoint }: Props) {
  const { major, minor } = getTickInterval(pps);

  // Snap the iteration to the minor-tick grid by index so non-integer minors
  // don't accumulate floating-point drift.
  const startIdx = Math.max(0, Math.floor(scrollX / pps / minor));
  const endIdx = Math.ceil(Math.min(duration, (scrollX + containerWidth) / pps) / minor);
  // Every Nth minor tick is a major tick (where N = major/minor, rounded).
  // Computing on the index avoids floating-point modulo errors that made major
  // ticks blink missing at non-integer intervals.
  const ratio = Math.max(1, Math.round(major / minor));

  const ticks: { time: number; major: boolean }[] = [];
  for (let i = startIdx; i <= endIdx; i++) ticks.push({ time: i * minor, major: i % ratio === 0 });

  return (
    <div
      className="relative shrink-0 overflow-hidden select-none"
      style={{ height: RULER_HEIGHT, background: "var(--tl-ruler-bg)" }}
    >
      <div className="relative h-full" style={{ transform: `translateX(${-scrollX}px)` }}>
        {ticks.map((t, i) => (
          <div key={i} className="absolute bottom-0" style={{ left: t.time * pps }}>
            <div
              style={{
                width: 1,
                height: t.major ? 8 : 3,
                background: t.major ? "var(--tl-ruler-tick)" : "var(--tl-ruler-tick-minor)",
                opacity: t.major ? 1 : 0.6,
              }}
            />
            {t.major && (
              <span
                className="absolute font-mono whitespace-nowrap"
                style={{
                  bottom: 9,
                  left: 3,
                  fontSize: 10,
                  lineHeight: 1,
                  color: "var(--tl-ruler-label)",
                }}
              >
                {formatTimecode(t.time)}
              </span>
            )}
          </div>
        ))}
        {inPoint != null && outPoint != null && outPoint > inPoint && (
          <div
            className="bg-primary/20 pointer-events-none absolute inset-y-0"
            style={{ left: inPoint * pps, width: (outPoint - inPoint) * pps }}
          />
        )}
      </div>
    </div>
  );
}
