"use client";

import { cn } from "@/lib/utils";
import type { Side } from "@/lib/types";

/* The momentum chart lives here, NOT in app/report/page.tsx where it was
   written: the Data browser's match Facts tab draws the same chart, and
   importing it from a route module pulled that whole page - and everything it
   imports - into the Data bundle. Same argument as components/rating-chip.tsx:
   two screens showing one picture share the component, not a route. */

/** THE PRESSURE CHART, as the reference draws it: home swells above the
 *  baseline in white, away below it in the chart blue, both as SMOOTH curves through
 *  the per-minute values - straight segments between ninety points read as
 *  noise, and the swing is the thing being shown. Half time is a dotted line
 *  down the middle; each goal is a ball above the chart for home and below
 *  it for away, at its minute; the clock runs underneath on a dotted rule.
 *
 *  Values are -1..+1 toward home (as written by a former feed importer, not in this repository). The curve
 *  is a Catmull-Rom spline with its control points kept on its own side of
 *  the baseline, so a smoothed area never crosses into the other team's. */
export function MomentumChart({
  momentum,
  fill = false,
}: {
  momentum: { values: number[]; goals?: { minute: number; side: Side }[] };
  /** Take the height the parent gives rather than a fixed 160px - the
   *  squad panel's Facts tab, which must not scroll on a short window. */
  fill?: boolean;
}) {
  // NO SERIES YET: a grey stand-in shape under a label, so the section keeps
  // its look until the match is processed. Drawn from a fixed wave, never
  // from anything about the match, and grey so it cannot be read as data.
  const ghost = momentum.values.length === 0;
  const values = ghost ? GHOST_MOMENTUM : momentum.values;
  const goals = ghost ? [] : momentum.goals;
  const n = values.length;
  const W = 100;
  const H = 44;
  const MID = H / 2;
  const AMP = H / 2 - 3;

  const area = (sign: 1 | -1) => {
    const pts = values.map((v, i): [number, number] => [
      n < 2 ? W / 2 : (i / (n - 1)) * W,
      MID - (sign === 1 ? Math.max(0, v) : Math.min(0, v)) * AMP,
    ]);
    const keep = (y: number) => (sign === 1 ? Math.min(MID, y) : Math.max(MID, y));
    let d = `M0,${MID}L${pts[0][0].toFixed(2)},${pts[0][1].toFixed(2)}`;
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[i - 1] ?? pts[i];
      const p1 = pts[i];
      const p2 = pts[i + 1];
      const p3 = pts[i + 2] ?? p2;
      const c1x = p1[0] + (p2[0] - p0[0]) / 6;
      const c1y = keep(p1[1] + (p2[1] - p0[1]) / 6);
      const c2x = p2[0] - (p3[0] - p1[0]) / 6;
      const c2y = keep(p2[1] - (p3[1] - p1[1]) / 6);
      d += `C${c1x.toFixed(2)},${c1y.toFixed(2)},${c2x.toFixed(2)},${c2y.toFixed(2)},${p2[0].toFixed(2)},${p2[1].toFixed(2)}`;
    }
    return `${d}L${W},${MID}Z`;
  };

  return (
    <div className={fill ? "flex min-h-0 flex-1 flex-col gap-2" : "grid gap-2"}>
      {/* Room above and below the curves for the balls, so a goal never
          sits on top of the swing it caused. */}
      <div className={cn("relative py-5", fill && "min-h-20 flex-1")}>
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className={cn("w-full", fill ? "h-full" : "h-40")} role="img"
             aria-label="Momentum across the match: home above the line, away below">
          <path d={area(1)} style={{ fill: ghost ? "var(--mute)" : "var(--foreground)" }} fillOpacity={ghost ? 0.18 : 0.92} />
          <path d={area(-1)} style={{ fill: ghost ? "var(--mute)" : "var(--chart-1)" }} fillOpacity={ghost ? 0.18 : 0.92} />
          <line
            x1={W / 2}
            y1={0}
            x2={W / 2}
            y2={H}
            style={{ stroke: "var(--mute)" }}
            strokeWidth={1.2}
            strokeDasharray="0.1 3"
            strokeLinecap="round"
            vectorEffect="non-scaling-stroke"
          />
        </svg>
        {ghost && (
          <span className="text-muted-foreground absolute inset-0 grid place-items-center text-xs">
            <span className="rounded-full bg-[var(--card)] px-2 py-0.5">Momentum appears once the match is processed</span>
          </span>
        )}
        {goals?.map((g, i) => (
          // The ball on its own, painted white through the icon as a mask -
          // no disc behind it - above the chart for home, below it for away.
          // The index is in the key: two goals by one side in one minute
          // (two in the 90th, as one fixture had) would repeat it.
          <span
            key={`${g.minute}-${g.side}-${i}`}
            aria-hidden
            className="absolute size-5 -translate-x-1/2 bg-[var(--foreground)] select-none"
            style={{
              left: `${Math.min(100, (g.minute / n) * 100)}%`,
              ...(g.side === "home" ? { top: 0 } : { bottom: 0 }),
              mask: "url(/soccer-ball.svg) center / contain no-repeat",
              WebkitMask: "url(/soccer-ball.svg) center / contain no-repeat",
            }}
          />
        ))}
      </div>
      <MinuteRule />
    </div>
  );
}

/** The placeholder's shape: a fixed, made-up wave, the same for every match. */
const GHOST_MOMENTUM = Array.from({ length: 90 }, (_, i) =>
  Number((0.6 * Math.sin(i / 5) * Math.cos(i / 13) + 0.15 * Math.sin(i / 2.3)).toFixed(3)),
);

/** The clock under the momentum chart, on a dotted rule: 0' · HT · FT. */
function MinuteRule() {
  return (
    <div className="readout flex items-center gap-2 text-xs font-semibold">
      <span>0&prime;</span>
      <span className="h-px flex-1 border-t border-dotted border-[var(--mute)]" />
      <span>HT</span>
      <span className="h-px flex-1 border-t border-dotted border-[var(--mute)]" />
      <span>FT</span>
    </div>
  );
}
