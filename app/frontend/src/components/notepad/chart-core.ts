import { max as d3max, sum as d3sum } from "d3-array";
import { scaleBand, scaleLinear, type ScaleBand, type ScaleLinear } from "d3-scale";
import {
  arc as d3arc,
  area as d3area,
  line as d3line,
  stack as d3stack,
  stackOffsetExpand,
  stackOrderNone,
} from "d3-shape";

import { EMPTY_BOARD, surfaceSvg, viewBox, viewOf } from "./pitch-data";

/** The geometry every figure in a note is built on.
 *
 *  D3 FOR THE MATHS, OURS FOR THE MARKS. d3-scale, d3-shape and d3-array are
 *  pure functions with no DOM — they hand back numbers and path strings — so
 *  they sit under an SVG-as-string renderer without dragging in a rendering
 *  model of their own. That matters more here than in most apps: the exported
 *  document is a self-contained HTML file with no script and no stylesheet to
 *  go and find, so anything that needs JavaScript at view time is unusable
 *  whatever else it offers.
 *
 *  WHY A LIBRARY AT ALL, having hand-rolled this once. Every rendering bug so
 *  far has been scale and axis arithmetic, not marks:
 *
 *    - a stacked chart scaled by its tallest SEGMENT rather than the stack,
 *      so the first column drew off the top of the plot;
 *    - axis labels reading 137.33, 103.00, 68.66 — the range divided by four,
 *      which is not what a tick is;
 *    - ten match names printed as one unbroken smear of characters.
 *
 *  scaleLinear().nice() answers the second, d3.stack the first, and the third
 *  is thinning that now lives in one place instead of four. None of that is
 *  clever; it is exactly the part that is tedious to get right and easy to get
 *  subtly wrong, which is what a library is for.
 *
 *  THE FOOTBALL VISUALS COME THROUGH HERE TOO. A pitch is not something any
 *  chart library draws, so those marks stay ours — but they take their frame,
 *  their wrapper and their sizing from this module like everything else. One
 *  sizing story, one print cap, one place a figure's outer shape is decided.
 */

export interface Pad {
  l: number;
  r: number;
  t: number;
  b: number;
}

export interface Frame {
  W: number;
  H: number;
  pad: Pad;
  plotW: number;
  plotH: number;
}

/** A USER UNIT IS APPROXIMATELY A PRINTED POINT, and the approximation is the
 *  honest part.
 *
 *  Every family used to draw into a 640-unit canvas whatever it contained, and
 *  CSS scaled that canvas to whatever width the column happened to be. Two
 *  measured consequences:
 *
 *    A figure's BOX was the canvas, not the picture. A doughnut 168 units
 *    across sat in 640, so three quarters of a full-width row was empty. That
 *    is the bug this file was opened for.
 *
 *    Type size was the declared size times an arbitrary layout number. The
 *    scale factor is `rendered width / W`, and with W pinned at 640 it ran
 *    0.40 to 0.58 across the layouts actually in use — so a label declared at
 *    11 printed at 6.3 pt full width, 4.4 pt three-up, 2.9 pt at `width: 50%`.
 *    No set of font sizes is right at all three.
 *
 *  W is now derived from the content, which narrows that band to roughly 0.9
 *  to 1.3. So a number here is within a point or so of what prints, and the
 *  sizes below are chosen against THIS DOCUMENT'S BODY, which is 6.3 pt
 *  (`.ft-doc` in globals.css), not against a generic print floor.
 *
 *  That distinction matters and it is why these are not IEEE's 8 pt. A figure
 *  whose labels print larger than the prose around it inverts the hierarchy —
 *  the audit found chart titles at 11.4 pt against a 10.5 pt H1, and figures
 *  shouting over the text was one of the things that made the page look wrong.
 *  A label at or just under body size is the right answer for a dense
 *  landscape report; 8 pt would be right for a document whose body was 10.
 *
 *  WHAT THIS DOES NOT FIX, plainly: a renderer that does not know its own
 *  rendered width cannot guarantee a printed point size. Only threading the
 *  target width through can. This narrows the error from a factor of two to a
 *  few tenths and it does not remove it.
 *
 *  CHANGING ANY OF THESE MEANS CHANGING globals.css TOO — the text sizes live
 *  there, as `px` inside a viewBox, which is the same thing as a user unit.
 *  They are named beside each entry.
 */
export const PT = {
  /** `.ft-chart__cat`, `.ft-chart__legend` — category and legend labels. */
  label: 6,
  /** `.ft-chart__val` — a value printed beside its mark. */
  value: 5.5,
  /** `.ft-chart__axis` — tick labels, the smallest type in any figure. */
  axis: 5,
  /** `.ft-chart__hero` — the single number that IS the chart. */
  hero: 20,
  /** Room for the y axis, its ticks, and a little air. */
  padL: 26,
  padR: 8,
  padT: 6,
  /** ...with a legend above the plot. */
  padTLegend: 16,
  padB: 16,
  /** What one category needs before its label runs into its neighbour. */
  slot: 46,
  /** A horizontal bar's row, and the bar inside it. */
  row: 13,
  rowMany: 18,
  bar: 8,
  barMany: 5,
  /** Mean character width at `label` size, for estimating a string's width
   *  without a DOM to measure it in. Errs wide: a little slack costs margin,
   *  too little clips. */
  charW: 3.2,
  /** Distance between two legend entries. Long enough for a ~14-character
   *  series name at `label` size plus its swatch. */
  legendStep: 70,
} as const;

/** Keep a derived size inside something a page can actually hold.
 *
 *  The FLOOR is doing more work than it looks. A three-category chart derives a
 *  width of about 170, and a two-up cell on a landscape sheet is 368 pt — so a
 *  low floor means the figure is scaled up 2x and its 6 pt labels print at 10,
 *  larger than the 6.3 pt body they sit beside. 320 pt is 113 mm, close enough
 *  to the common two-up column that the scale factor stays near 1 and the type
 *  prints at about the size it says. Small content still gets a big box; it
 *  gets a box with air in it rather than one that magnifies its own labels.
 *
 *  560 pt is 198 mm, past which a figure is a page of its own. */
export const fit = (v: number, lo = 320, hi = 560) => Math.max(lo, Math.min(hi, v));

/** A plot frame whose HEIGHT follows from its width.
 *
 *  Aspect is a real perceptual variable — Cleveland's banking result is about
 *  exactly this — but 45 degrees is a heuristic that did not survive
 *  replication, and a report needs figures that sit together on a page more
 *  than it needs each one individually optimal. So: one house ratio, clamped,
 *  and the width carries the information about how much there is to show.
 */
export const plotFrame = (W: number, pad: Pad, ratio = 1.85, lo = 120, hi = 210): Frame =>
  frame(W, Math.round(Math.max(lo, Math.min(hi, W / ratio))), pad);

export const frame = (W: number, H: number, pad: Pad): Frame => ({
  W,
  H,
  pad,
  plotW: W - pad.l - pad.r,
  plotH: H - pad.t - pad.b,
});

export const esc = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

export const n2 = (n: number) => Number(n.toFixed(2));

// ------------------------------------------------------------------ scales

/** Categories across the plot. `scaleBand` owns the arithmetic that turns
 *  "twelve players" into twelve slots with gaps — including the gap ratio,
 *  which is what stops adjacent bars reading as one shape. */
export const band = (labels: string[], f: Frame, inset = 0.28): ScaleBand<string> =>
  scaleBand<string>().domain(labels).range([f.pad.l, f.W - f.pad.r]).padding(inset);

/** Values up the plot. `.nice()` is the whole point: it rounds the domain out
 *  to human numbers so the ticks read 0, 5, 10, 15 rather than 0, 4.58, 9.16.
 *  `zero` because a bar that does not start at zero is a bar that lies about
 *  its own length; a line may start wherever the data does. */
export const linear = (hi: number, f: Frame, zero = true): ScaleLinear<number, number> =>
  scaleLinear()
    .domain([zero ? 0 : 0, Math.max(hi, Number.EPSILON)])
    .nice()
    .range([f.H - f.pad.b, f.pad.t]);

/** The same scale lying down — for a scatter's horizontal axis, where the
 *  value runs across the plot rather than up it. */
export const linearX = (hi: number, f: Frame): ScaleLinear<number, number> =>
  scaleLinear()
    .domain([0, Math.max(hi, Number.EPSILON)])
    .nice()
    .range([f.pad.l, f.W - f.pad.r]);

/** How a set of bars relates to each other. */
export type Stack = "none" | "stacked" | "pct";

export interface StackedRow {
  label: string;
  /** [y0, y1] per series, already offset — pct is expanded to 0-1. */
  spans: [number, number][];
  raw: number[];
  text: string[];
}

/** Stacked baselines, from d3.
 *
 *  `stackOffsetExpand` IS the hundred-percent-stacked rule, so that form is a
 *  different offset rather than a second pass that rescales every row by hand.
 */
export function stacked(
  rows: { label: string; values: number[]; text: string[] }[],
  seriesCount: number,
  mode: Stack,
): { rows: StackedRow[]; hi: number } {
  const n = Math.max(1, seriesCount);
  const keys = Array.from({ length: n }, (_, i) => String(i));
  const data = rows.map((r) => {
    const o: Record<string, number> = {};
    keys.forEach((k, i) => {
      o[k] = Math.max(0, r.values[i] ?? 0);
    });
    return o;
  });
  const layers = d3stack<Record<string, number>>()
    .keys(keys)
    .order(stackOrderNone)
    .offset(mode === "pct" ? stackOffsetExpand : undefined!)(data);

  const out = rows.map((r, ri) => ({
    label: r.label,
    raw: keys.map((_, i) => r.values[i] ?? 0),
    text: r.text,
    spans: layers.map((l) => [l[ri][0], l[ri][1]] as [number, number]),
  }));
  const hi =
    mode === "pct"
      ? 1
      : mode === "stacked"
        ? (d3max(data, (d) => d3sum(keys, (k) => d[k])) ?? 0)
        : (d3max(rows, (r) => d3max(r.values) ?? 0) ?? 0);
  return { rows: out, hi };
}

// -------------------------------------------------------------------- marks

export const linePath = (pts: [number, number][]) => d3line()(pts) ?? "";
export const areaPath = (pts: [number, number][], y0: number) =>
  d3area<[number, number]>()
    .x((d) => d[0])
    .y0(y0)
    .y1((d) => d[1])(pts) ?? "";
export const bandPath = (pts: [number, number][], base: [number, number][]) =>
  d3area<[number, number]>()
    .x((d) => d[0])
    .y0((_, i) => base[i][1])
    .y1((d) => d[1])(pts) ?? "";

/** A slice, or a ring segment when `hole` is set. */
export const slicePath = (a0: number, a1: number, r: number, hole = 0) =>
  d3arc()({ innerRadius: hole, outerRadius: r, startAngle: a0, endAngle: a1 }) ?? "";

// -------------------------------------------------------------------- chrome

/** Gridlines and their value labels, at ticks the scale chose. */
export function gridY(y: ScaleLinear<number, number>, f: Frame, fmt: (v: number) => string): string {
  return y
    .ticks(5)
    .map((v) => {
      const gy = y(v);
      return (
        `<line x1="${f.pad.l}" y1="${n2(gy)}" x2="${f.W - f.pad.r}" y2="${n2(gy)}" class="ft-chart__grid"/>` +
        `<text x="${f.pad.l - 5}" y="${n2(gy + PT.axis / 3)}" class="ft-chart__axis">${esc(fmt(v))}</text>`
      );
    })
    .join("");
}

export const clip = (label: string, max: number) =>
  label.length > max ? `${label.slice(0, max - 1)}…` : label;

/** Print every Nth category label, where N is what the longest name needs.
 *
 *  Ten fixture names across one figure ran together into an unbroken line of
 *  characters, which is worse than no axis — a smear looks like a label until
 *  you try to read it. Thinning an ORDERED axis is safe in a way dropping
 *  marks would not be: every point is still drawn, and the ones between two
 *  ticks are still between those two moments.
 */
export function labelStep(labels: string[], plotW: number, cap: number): number {
  const longest = Math.max(1, ...labels.map((l) => Math.min(l.length, cap)));
  const need = longest * PT.charW + 6;
  return Math.max(1, Math.ceil(labels.length / Math.max(1, Math.floor(plotW / need))));
}

export function catAxis(
  labels: string[],
  xOf: (i: number) => number,
  f: Frame,
  cap = 12,
): string {
  const step = labelStep(labels, f.plotW, cap);
  return labels
    .map((l, i) =>
      i % step
        ? ""
        : `<text x="${n2(xOf(i))}" y="${f.H - f.pad.b + PT.label + 3}" text-anchor="middle" class="ft-chart__cat">${esc(clip(l, cap))}</text>`,
    )
    .join("");
}

export function legend(items: { c: string; label: string }[], x: number, y: number): string {
  return items
    .map(
      (it, i) =>
        `<g transform="translate(${n2(x + i * PT.legendStep)},${n2(y)})">` +
        `<rect x="0" y="0.5" width="7" height="7" rx="1.5" fill="${it.c}" class="ft-swatch ft-swatch--s${i}"/>` +
        `<text x="11" y="7" class="ft-chart__legend">${esc(it.label)}</text></g>`,
    )
    .join("");
}

// ------------------------------------------------------------------ wrappers

/** The svg every generic figure is wrapped in. One place decides that a chart
 *  scales to its container and never overflows it. */
/** FILLS ITS COLUMN, AT AN ASPECT ITS CONTENT CHOSE.
 *
 *  `width: 100%`, deliberately, after trying the alternative. Declaring an
 *  absolute `{W}pt` makes one user unit exactly one printed point — which is
 *  the correct answer and the one the research recommends — but a figure then
 *  stops growing into the column it was given, and a three-category chart sat
 *  78 mm wide in a 130 mm cell with the rest white. That is the reported bug
 *  again, arriving from the other direction.
 *
 *  So the box still stretches. What changed, and it is most of the value:
 *  the box is now sized FROM THE CONTENT rather than being a 640-unit canvas
 *  every family inherited from the bar chart. That does two things.
 *
 *  It makes the box the picture — a doughnut no longer sits in three times its
 *  own width of nothing, which was the complaint.
 *
 *  And it collapses the range the type has to survive. The scale factor is
 *  `rendered width / W`, and with W fixed at 640 that ran 0.40 to 0.58 across
 *  the layouts in use, so a label declared at 11 printed anywhere from 2.9 to
 *  6.3 pt and no set of numbers was right at more than one width. With W
 *  tracking the content it runs about 0.9 to 1.3, so the sizes in PT are
 *  within a point or so of what actually prints.
 *
 *  WHAT THIS STILL DOES NOT FIX, stated plainly: a renderer that does not know
 *  its rendered width cannot guarantee a printed point size. Only passing the
 *  target width in can do that. This narrows the error from a factor of two to
 *  a few tenths; it does not remove it.
 */
export const chartSvgTag = (f: Frame, body: string, label: string, par?: string) =>
  `<svg viewBox="0 0 ${n2(f.W)} ${n2(f.H)}"${par ? ` preserveAspectRatio="${par}"` : ""} ` +
  `width="100%" role="img" aria-label="${esc(label)}" ` +
  `style="display:block;max-width:100%;height:auto${par === "none" ? ";height:180px" : ""}">` +
  body +
  `</svg>`;

/** The svg every PITCH figure is wrapped in.
 *
 *  Football is the part no chart library draws, so the marks are ours — but
 *  the frame, the wrapper and the sizing come from here like everything else.
 *  `overflow:hidden` is load-bearing: surfaceSvg paints grass past the view's
 *  edges so the surround is never a bald strip, and .ft-chart svg sets overflow
 *  VISIBLE for the sake of a bar chart's rightmost value label. Without the
 *  clip that grass spilled across the width of the document.
 */
export const pitchSvgTag = (body: string, label: string) => {
  const { box } = viewBox(viewOf("full"), false);
  return (
    `<svg class="ft-pitchfig" viewBox="${box}" preserveAspectRatio="xMidYMid meet" ` +
    `width="100%" role="img" aria-label="${esc(label)}" ` +
    `style="display:block;max-width:100%;height:auto;overflow:hidden">` +
    surfaceSvg(EMPTY_BOARD) +
    body +
    `</svg>`
  );
};
