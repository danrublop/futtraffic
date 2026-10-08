import {
  band,
  bandPath,
  catAxis,
  chartSvgTag,
  clip,
  fit,
  frame,
  plotFrame,
  PT,
  gridY,
  legend,
  linePath,
  linear,
  linearX,
  pitchSvgTag,
  slicePath,
  stacked,
  type Stack,
} from "./chart-core";
import {
  MAGNET_FILL,
  MAGNET_TEXT,
  mx,
  my,
} from "./pitch-data";
import type {
  FreezeExtra,
  GoalExtra,
  HeatExtra,
  MomentumExtra,
  PassExtra,
  PizzaExtra,
  QueryResult,
  ShotExtra,
  SpotsExtra,
  ZoneExtra,
} from "./query-data";

/** Charts as markup, drawn once so the live block and the exported file cannot
 *  differ — the same rule pitchSvg already enforces for the board.
 *
 *  SVG built as a string rather than JSX because the serializer runs headless,
 *  and self-contained rather than through a chart library because the document
 *  leaves the app: a note in somebody's inbox has no bundle to load.
 *
 *  COLOUR. One series is one hue and needs no legend — the title names it. Two
 *  series are the teams, and they keep the hue families the rest of the app
 *  already uses for home and away, stepped darker so they hold on both a dark
 *  screen and a printed page. The pair was run through the palette validator:
 *  lightness band, chroma floor, CVD separation (ΔE 15.4 protan / 26.3 tritan),
 *  normal-vision separation (22.6) and contrast all pass against both surfaces.
 *  Three series is not offered. A third hue would have to be chosen by eye, and
 *  a reader holding three legend swatches is reading a legend, not a chart.
 */

/** Validated for contrast and colour-blind separation; do not substitute by eye. */
export const SERIES_COLORS = ["#ec4899", "#8b5cf6"];
export const SINGLE_COLOR = "#3b82f6";

/** For the one form where the CATEGORIES are the colours rather than the
 *  series: a pie, or a doughnut. Six, in fixed order, never cycled — a seventh
 *  slice folds into "Other" instead of inventing a hue.
 *
 *  Validated against both surfaces: lightness band, chroma floor, normal-vision
 *  separation (worst adjacent 18.3) and contrast all pass. CVD separation is a
 *  WARN in the 6-8 band (amber vs green, ΔE 6.2 deutan), which is legal ONLY
 *  with secondary encoding — so every slice carries its category name as a
 *  direct label, and the relief is not optional decoration. */
/** One hue, light to dark, which is the rule for a magnitude — and a warm one,
 *  because that is the colour a report's zone graphics have always been. */
/** Re-exported: the stacking rule belongs to the geometry layer now, but
 *  every caller already imports it from here. */
export type { Stack } from "./chart-core";

export const HEAT = "#d1451f";

export const CATEGORY_COLORS = [
  "#3b82f6", "#e0529c", "#0f9d58", "#b0741a", "#8b5cf6", "#00919e",
];

/** Escape, and tolerate a cell that has no formatted text.
 *
 *  `text` is a row's formatted values, and it is legitimately EMPTY on a chart
 *  whose rows were written rather than queried — which is every chart a script
 *  builds (scripts/note-api.mts). Eleven sites in this file index into it, and
 *  ten of them did so unguarded, so a hand-written doughnut threw
 *  `Cannot read properties of undefined` out of a renderHTML and took the whole
 *  document with it. One guard here, not eleven `?? ""` at the call sites. */
const esc = (s: string | undefined) =>
  (s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

const n2 = (n: number) => Number(n.toFixed(2));

/** A class per series on every mark, so a THEME is a stylesheet rather than a
 *  second drawing routine.
 *
 *  The colour stays a `fill` ATTRIBUTE rather than moving into CSS: an exported
 *  note is opened with no stylesheet of ours guaranteed, and a chart that draws
 *  black without one is not a chart. A presentation attribute loses to any CSS
 *  rule, so `[data-theme] .ft-mark--s1 { fill: #fff }` still wins where the
 *  stylesheet is there. */
const markCls = (si: number) => `ft-mark ft-mark--s${si}`;

/** The colour of series `i`, honouring whatever the chart was set to.
 *
 *  `r.colors` is the coach's choice, frozen with the figure like everything
 *  else. Empty or short falls through to the validated palette above, so a
 *  chart nobody has recoloured is still the one the contrast and colour-vision
 *  checks were run against — and one they HAVE recoloured is theirs.
 */
const colorFor = (r: { colors?: string[] }, i: number, many: boolean) => {
  const set = r.colors?.[i];
  if (set) return set;
  return many ? SERIES_COLORS[i] ?? SERIES_COLORS[0] : SINGLE_COLOR;
};

/** The colour of category `i`, for the one form where the slices are the
 *  colours rather than the series. */
const catColor = (r: { colors?: string[] }, i: number) =>
  r.colors?.[i] || CATEGORY_COLORS[i];

/** Longest label decides the gutter, so nothing is clipped and nothing is
 *  centred over empty space. Capped, because one long name should not squeeze
 *  every bar into a sliver. */
const gutter = (rows: { label: string }[]) =>
  Math.min(130, Math.max(48, ...rows.map((r) => r.label.length * PT.charW + 8)));

/** Horizontal bars: the form for comparing a magnitude across categories whose
 *  names are words. Vertical columns would turn "Positional attack" into
 *  rotated text, which is a label nobody reads. */
export function barSvg(r: QueryResult): string {
  // A one-bar bar chart is not a chart. With nothing to compare against, the
  // bar encodes nothing the number does not already say, so the number IS the
  // chart. Falls out on a dimension a match happens to have one value for —
  // one side tracked, one position recorded.
  if (r.rows.length === 1 && r.series.length < 2) {
    const row = r.rows[0];
    return (
      chartSvgTag(
        frame(
          Math.max(120, (row.text[0] ?? "").length * PT.hero * 0.62, row.label.length * PT.charW),
          PT.label + PT.hero + 14,
          { l: 0, r: 0, t: 0, b: 0 },
        ),
        `<text x="0" y="${PT.label}" class="ft-chart__legend">${esc(row.label)}</text>` +
          `<text x="0" y="${PT.label + PT.hero + 4}" class="ft-chart__hero">${esc(row.text[0])}</text>`,
        `${row.label} — ${row.text[0]}`,
      )
    );
  }
  const many = r.series.length > 1;
  const g = gutter(r.rows);
  const rowH = many ? PT.rowMany : PT.row;
  const barH = many ? PT.barMany : PT.bar;
  const legendH = many ? PT.padTLegend : 0;
  // A horizontal bar chart's width is a MEASURE, not a count: the bars run
  // across it and the categories run down. So width is fixed by how long a bar
  // needs to be to be readable, and the HEIGHT carries the category count.
  const W = fit(g + 210, 220, 420);
  const H = legendH + r.rows.length * rowH + 8;
  const plot = W - g - 46;
  const scale = (v: number) => (r.max > 0 ? Math.max(0, (v / r.max) * plot) : 0);

  const legend = many
    ? `<g transform="translate(${g},4)">` +
      r.series
        .map(
          (s, i) =>
            `<g transform="translate(${i * 120},0)">` +
            `<rect x="0" y="1" width="9" height="9" rx="2" fill="${colorFor(r, i, true)}" class="ft-swatch ft-swatch--s${i}"/>` +
            `<text x="14" y="9.5" class="ft-chart__legend">${esc(s)}</text></g>`,
        )
        .join("") +
      `</g>`
    : "";

  const bars = r.rows
    .map((row, ri) => {
      const y0 = legendH + ri * rowH;
      const inner = row.values
        .map((v, si) => {
          const w = scale(v);
          // 2px gap between adjacent bars in a group, and a 4px rounded data
          // end anchored to the baseline.
          const y = y0 + (many ? 3 + si * (barH + 2) : (rowH - barH) / 2);
          return (
            `<g><title>${esc(row.label)}${many ? ` · ${esc(r.series[si])}` : ""} — ${esc(row.text[si])}</title>` +
            `<path d="${barPath(g, y, w, barH)}" fill="${colorFor(r, si, many)}" class="${markCls(si)}"/>` +
            `</g>` +
            // The value, always. A printed document has no hover, so a direct
            // label is not decoration here — it is the only way to read a
            // number off the page.
            `<text x="${n2(g + w + 6)}" y="${n2(y + barH - 1.5)}" class="ft-chart__val">${esc(row.text[si])}</text>`
          );
        })
        .join("");
      return (
        `<text x="${g - 8}" y="${n2(y0 + rowH / 2 + 3.5)}" class="ft-chart__cat">${esc(row.label)}</text>` +
        inner
      );
    })
    .join("");

  return (
    chartSvgTag(
      frame(W, H, { l: g, r: 0, t: legendH, b: 0 }),
      `${legend}${bars}`,
      `${r.measure.label} by ${r.dim.label}`,
    )
  );
}

/** A line, only ever over an ordered dimension. Joining unordered categories
 *  with a line asserts a progression between them that does not exist. */
export function lineSvg(r: QueryResult): string {
  const many = r.series.length > 1;
  // Width from how many points there are, height from the width: an ordered
  // series wants room along its axis, and a fixed 640 gave a three-point line
  // the same canvas as a ninety-minute one.
  const f = plotFrame(
    fit(PT.padL + Math.max(150, (r.rows.length - 1) * 26) + PT.padR),
    { l: PT.padL, r: PT.padR, t: many ? PT.padTLegend : PT.padT, b: PT.padB },
    1.95,
  );
  const y = linear(r.max, f);
  const n = r.rows.length;
  const xAt = (i: number) => f.pad.l + (n < 2 ? f.plotW / 2 : (i / (n - 1)) * f.plotW);

  const lines = (many ? r.series : [""])
    .map((_, si) => {
      const pts: [number, number][] = r.rows.map((row, i) => [xAt(i), y(row.values[si] ?? 0)]);
      const c = colorFor(r, si, many);
      const dots = r.rows
        .map(
          (row, i) =>
            `<g><title>${esc(row.label)}${many ? ` · ${esc(r.series[si])}` : ""} — ${esc(row.text[si])}</title>` +
            `<circle cx="${n2(xAt(i))}" cy="${n2(y(row.values[si] ?? 0))}" r="4" fill="${c}" ` +
            `stroke="var(--panel,#fff)" stroke-width="2" class="${markCls(si)}"/></g>`,
        )
        .join("");
      return (
        `<path d="${linePath(pts)}" fill="none" stroke="${c}" stroke-width="2" ` +
        `stroke-linejoin="round" class="ft-stroke ft-stroke--s${si}"/>${dots}`
      );
    })
    .join("");

  return chartSvgTag(
    f,
    (many ? legend(r.series.map((sname, i) => ({ c: colorFor(r, i, true), label: sname })), f.pad.l, 0) : "") +
      gridY(y, f, (v) => v.toFixed(r.measure.dp)) +
      lines +
      catAxis(r.rows.map((row) => row.label), xAt, f, 14),
    `${r.measure.label} by ${r.dim.label}`,
  );
}

/** The table. Always available, and not a fallback: it is the accessible view
 *  of every chart here, and the thing somebody copies into a spreadsheet. */
/** A column, rounded where it ENDS and square where it stands.
 *
 *  `rx` on a `<rect>` rounds all four corners, which put a 4px curve on the
 *  baseline — so every column appeared to float a hair above the axis it is
 *  measured from, and a stack of them read as separate tiles rather than one
 *  quantity. The rounding belongs at the free end only.
 *
 *  Clamped, because a column shorter than the radius would otherwise fold its
 *  own corners through each other and come out as a bow tie.
 */
export function colPath(x: number, y: number, w: number, h: number, r = 4): string {
  const k = Math.max(0, Math.min(r, w / 2, h));
  return (
    `M${n2(x)},${n2(y + h)}V${n2(y + k)}Q${n2(x)},${n2(y)} ${n2(x + k)},${n2(y)}` +
    `H${n2(x + w - k)}Q${n2(x + w)},${n2(y)} ${n2(x + w)},${n2(y + k)}` +
    `V${n2(y + h)}Z`
  );
}

/** A horizontal bar, rounded at the far end and square against the axis. */
export function barPath(x: number, y: number, w: number, h: number, r = 4): string {
  const k = Math.max(0, Math.min(r, w, h / 2));
  return (
    `M${n2(x)},${n2(y)}H${n2(x + w - k)}Q${n2(x + w)},${n2(y)} ${n2(x + w)},${n2(y + k)}` +
    `V${n2(y + h - k)}Q${n2(x + w)},${n2(y + h)} ${n2(x + w - k)},${n2(y + h)}` +
    `H${n2(x)}Z`
  );
}

export function tableHtml(r: QueryResult): string {
  const head =
    `<tr><th>${esc(r.dim.label)}</th>` +
    (r.series.length ? r.series.map((s) => `<th>${esc(s)}</th>`).join("") : `<th>${esc(r.measure.label)}</th>`) +
    `</tr>`;
  const body = r.rows
    .map(
      (row) =>
        `<tr><th scope="row">${esc(row.label)}</th>` +
        row.text.map((t) => `<td>${esc(t)}</td>`).join("") +
        `</tr>`,
    )
    .join("");
  return `<div class="ft-chart__scroll"><table class="ft-chart__table"><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
}

const legendRow = (items: { c: string; label: string }[]) =>
  `<p class="ft-chart__legendrow">` +
  items
    .map(
      (i) =>
        `<span class="ft-chart__key"><span class="ft-chart__swatch" style="background:${i.c}"></span>${esc(
          i.label,
        )}</span>`,
    )
    .join("") +
  `</p>`;

/** A shot map is a PITCH, and the pitch is the one the whiteboard draws.
 *
 *  Not a lookalike: `surfaceSvg` is the same function a board in the same note
 *  renders through, so the grass, the markings and the proportions are one
 *  object. A shot map that drew its own pitch would drift from the board beside
 *  it the first time either was touched.
 *
 *  A goal is filled and ringed; a miss is hollow. Colour carries the team and
 *  the ring carries the outcome, so neither rests on colour alone. */
export function shotMapSvg(e: ShotExtra): string {
  const dot = (p: ShotExtra["points"][number]) => {
    const c = p.side === "home" ? SERIES_COLORS[0] : SERIES_COLORS[1];
    const x = n2(mx(p.x));
    const y = n2(my(p.y));
    const title = `${p.side === "home" ? e.home : e.away} · ${p.goal ? "goal" : "no goal"} · ${
      p.period === 1 ? "first half" : "second half"
    }`;
    return (
      `<g><title>${esc(title)}</title>` +
      (p.goal
        ? `<circle cx="${x}" cy="${y}" r="1.9" fill="${c}" stroke="#ffffff" stroke-width="0.45"/>`
        : `<circle cx="${x}" cy="${y}" r="1.3" fill="none" stroke="${c}" stroke-width="0.5"/>`) +
      `</g>`
    );
  };
  // Misses first, so a goal is never hidden underneath one.
  const ordered = [...e.points].sort((a, b) => Number(a.goal) - Number(b.goal));
  return (
    pitchSvgTag(ordered.map(dot).join(""), "Shot map") +
    legendRow([
      { c: SERIES_COLORS[0], label: e.home },
      { c: SERIES_COLORS[1], label: e.away },
    ]) +
    `<p class="ft-chart__sub">Filled and ringed is a goal, hollow is a miss. Each side shoots at the goal it was attacking — home to the right, as on the boards.</p>`
  );
}

/** Momentum, drawn the way the match report draws it: the swing across the
 *  ninety, toward home above the axis and away below, with the goals marked on
 *  it. Ninety bars side by side is not this. */
export function momentumSvg(e: MomentumExtra): string {
  const n = e.values.length;
  if (!n) return "";
  const W = 640;
  const H = 180;
  const MID = H / 2;
  const AMP = MID - 14;
  const x = (i: number) => (n < 2 ? W / 2 : (i / (n - 1)) * W);

  const area = (sign: 1 | -1) => {
    let d = `M0,${MID}`;
    e.values.forEach((v, i) => {
      const c = sign === 1 ? Math.max(0, v) : Math.min(0, v);
      d += `L${n2(x(i))},${n2(MID - c * AMP)}`;
    });
    return `${d}L${W},${MID}Z`;
  };

  const goals = e.goals
    .map((g) => {
      const gx = n2((g.minute / n) * W);
      const c = g.side === "home" ? SERIES_COLORS[0] : SERIES_COLORS[1];
      return (
        `<g><title>${esc(`${g.minute}' ${g.side === "home" ? e.home : e.away}`)}</title>` +
        `<circle cx="${gx}" cy="${MID}" r="4.5" fill="${c}" stroke="var(--panel,#fff)" stroke-width="2"/></g>`
      );
    })
    .join("");

  return (
    `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" width="100%" role="img" ` +
    `aria-label="Momentum across the match" style="display:block;max-width:100%;height:180px">` +
    `<path d="${area(1)}" fill="${SERIES_COLORS[0]}" fill-opacity="0.85"/>` +
    `<path d="${area(-1)}" fill="${SERIES_COLORS[1]}" fill-opacity="0.85"/>` +
    `<line x1="${W / 2}" y1="0" x2="${W / 2}" y2="${H}" class="ft-chart__grid" stroke-dasharray="4 4"/>` +
    `<line x1="0" y1="${MID}" x2="${W}" y2="${MID}" class="ft-chart__grid"/>` +
    goals +
    `</svg>` +
    minuteAxis(n) +
    legendRow([
      { c: SERIES_COLORS[0], label: e.home },
      { c: SERIES_COLORS[1], label: e.away },
    ]) +
    `<p class="ft-chart__sub">Half time is the dashed line, at the middle of the series rather than at a clock reading. A dot on the axis is a goal.</p>`
  );
}

/** The clock under the swing.
 *
 *  HTML rather than <text> inside the SVG, because that SVG is drawn with
 *  `preserveAspectRatio="none"` so it fills whatever width it is given at a
 *  fixed height — which is the shape the match report draws momentum in, and
 *  which stretches any text inside it horizontally by the same factor. Ticks
 *  outside the viewport keep the shape AND stay readable.
 *
 *  Positioned by minute/total, the same fraction the goal dots use, so a tick
 *  and a goal at the same minute land on the same vertical.
 */
function minuteAxis(total: number): string {
  if (total < 10) return "";
  const step = total > 70 ? 15 : total > 30 ? 10 : 5;
  const marks: number[] = [];
  for (let m = 0; m <= total; m += step) marks.push(m);
  return (
    `<div class="ft-chart__ticks" aria-hidden="true">` +
    marks
      .map((m) => {
        const pct = (m / total) * 100;
        // The first and last would hang off the ends; pull them inside.
        const shift = pct < 2 ? "0" : pct > 98 ? "-100%" : "-50%";
        return (
          `<span style="left:${n2(pct)}%;transform:translateX(${shift})">${m}'</span>`
        );
      })
      .join("") +
    `</div>`
  );
}


// ------------------------------------------------------------- more forms

function svgLegend(series: string[], x: number, y: number, colors: string[] = []): string {
  return series
    .map(
      (label, i) =>
        `<g transform="translate(${n2(x + i * 120)},${n2(y)})">` +
        `<rect x="0" y="1" width="9" height="9" rx="2" fill="${colorFor({ colors }, i, true)}"/>` +
        `<text x="14" y="9.5" class="ft-chart__legend">${esc(label)}</text></g>`,
    )
    .join("");
}

/** Which category labels to print along an axis.
 *
 *  Ten match names across a 130mm figure printed as one unbroken smear —
 *  "Northbridge FCReal OviedoLevante v Real Socieda" — which is worse than no
 *  axis at all, because a smear looks like a label until you try to read it.
 *
 *  So: print every Nth, where N is however many it takes to give each one the
 *  room its own characters need. Thinning an ORDERED axis is safe in a way
 *  that dropping bars would not be — every point is still drawn, and the ones
 *  between two ticks are still between those two moments.
 */
function labelStep(rows: { label: string }[], plotW: number, cap: number): number {
  const longest = Math.max(1, ...rows.map((r) => Math.min(r.label.length, cap)));
  const need = longest * 5.4 + 8;
  return Math.max(1, Math.ceil(rows.length / Math.max(1, Math.floor(plotW / need))));
}

/** VERTICAL columns. The other half of the bar family, and the right one when
 *  the categories are short — minutes, halves, thirds, shirt numbers. Excel
 *  calls the vertical one a column chart and the horizontal one a bar chart,
 *  and the distinction is worth keeping: rotated word labels are unreadable,
 *  which is what horizontal bars exist to avoid. */
export function columnSvg(r: QueryResult, mode: Stack = "none"): string {
  const many = r.series.length > 1;
  // A slot per category. Three columns in a 640 canvas were three columns with
  // a hand of white between them.
  const f = plotFrame(
    fit(PT.padL + Math.max(1, r.rows.length) * PT.slot * (many ? 1.4 : 1) + PT.padR),
    { l: PT.padL, r: PT.padR, t: many ? PT.padTLegend : PT.padT, b: PT.padB },
    1.75,
  );
  const { rows, hi } = stacked(r.rows, r.series.length, mode);
  const x = band(r.rows.map((row) => row.label), f);
  const y = linear(hi, f);
  const stack = mode !== "none" && many;
  const bw = stack || !many ? x.bandwidth() : (x.bandwidth() - 2) / 2;
  const fmt = (v: number) =>
    mode === "pct" ? `${Math.round(v * 100)}%` : v.toFixed(r.measure.dp);

  const cols = rows
    .map((row, ri) => {
      const x0 = x(row.label) ?? 0;
      const bars = row.spans
        .map((sp, si) => {
          // A CLUSTERED column sits on the baseline. `stacked()` returns
          // cumulative spans whatever the mode, so reading them when nothing
          // is stacked lifted every bar by the one before it — and off the top
          // of the frame, because in that mode `hi` is the largest SINGLE
          // value rather than the largest total. The horizontal `barSvg` was
          // always right, so the same data looked correct as a Bar and broken
          // as a Column.
          const top = stack ? y(sp[1]) : y(row.raw[si] ?? 0);
          const h = Math.max(0, (stack ? y(sp[0]) : y(0)) - top);
          const bx = stack || !many ? x0 : x0 + si * (bw + 2);
          const text =
            mode === "pct"
              ? `${Math.round((sp[1] - sp[0]) * 100)}%`
              : (r.rows[ri].text[si] ?? "");
          return (
            `<g><title>${esc(row.label)}${many ? ` · ${esc(r.series[si])}` : ""} — ${esc(text)}</title>` +
            (stack
              ? `<rect x="${n2(bx)}" y="${n2(top)}" width="${n2(bw)}" height="${n2(h)}" ` +
                // A 2px surface ring, so touching segments read as two and not one.
                `fill="${colorFor(r, si, many)}" stroke="var(--panel,#fff)" stroke-width="2" class="${markCls(si)}"/>`
              : `<path d="${colPath(bx, top, bw, h)}" fill="${colorFor(r, si, many)}" class="${markCls(si)}"/>`) +
            `</g>` +
            // Only the clustered form gets a value on every bar; on a stack
            // they collide, and the axis carries it instead.
            (stack || h < 14
              ? ""
              : `<text x="${n2(bx + bw / 2)}" y="${n2(top - 4)}" text-anchor="middle" class="ft-chart__val">${esc(text)}</text>`)
          );
        })
        .join("");
      return bars;
    })
    .join("");

  return chartSvgTag(
    f,
    (many ? legend(r.series.map((sname, i) => ({ c: colorFor(r, i, true), label: sname })), f.pad.l, 0) : "") +
      gridY(y, f, fmt) +
      cols +
      catAxis(r.rows.map((row) => row.label), (i) => (x(r.rows[i].label) ?? 0) + x.bandwidth() / 2, f, 12),
    `${r.measure.label} by ${r.dim.label}`,
  );
}

/** An area, which is a line that also claims the space under it means
 *  something. Only over an ordered dimension, same rule as the line. */
export function areaSvg(r: QueryResult, mode: Stack = "none"): string {
  const many = r.series.length > 1;
  const stack = mode !== "none" && many;
  // Width from how many points there are, height from the width: an ordered
  // series wants room along its axis, and a fixed 640 gave a three-point line
  // the same canvas as a ninety-minute one.
  const f = plotFrame(
    fit(PT.padL + Math.max(150, (r.rows.length - 1) * 26) + PT.padR),
    { l: PT.padL, r: PT.padR, t: many ? PT.padTLegend : PT.padT, b: PT.padB },
    1.95,
  );
  const { rows, hi } = stacked(r.rows, r.series.length, stack ? mode : "none");
  const y = linear(hi, f);
  const n = r.rows.length;
  const xAt = (i: number) => f.pad.l + (n < 2 ? f.plotW / 2 : (i / (n - 1)) * f.plotW);
  const fmt = (v: number) => (mode === "pct" ? `${Math.round(v * 100)}%` : v.toFixed(r.measure.dp));

  const bands = (many ? r.series : [""])
    .map((_, si) => {
      const top: [number, number][] = rows.map((row, i) => [xAt(i), y(stack ? row.spans[si][1] : row.raw[si])]);
      const base: [number, number][] = rows.map((row, i) => [
        xAt(i),
        y(stack ? row.spans[si][0] : 0),
      ]);
      const c = colorFor(r, si, many);
      return (
        `<path d="${bandPath(top, base)}" fill="${c}" fill-opacity="0.22" class="${markCls(si)}"/>` +
        `<path d="${linePath(top)}" fill="none" stroke="${c}" stroke-width="2" ` +
        `stroke-linejoin="round" class="ft-stroke ft-stroke--s${si}"/>`
      );
    })
    .join("");

  return chartSvgTag(
    f,
    (many ? legend(r.series.map((sname, i) => ({ c: colorFor(r, i, true), label: sname })), f.pad.l, 0) : "") +
      gridY(y, f, fmt) +
      bands +
      catAxis(r.rows.map((row) => row.label), xAt, f, 14),
    `${r.measure.label} by ${r.dim.label}`,
  );
}

/** A pie, or a doughnut with a hole in it. The one form here where the
 *  CATEGORY carries the colour, which is why it needs its own palette and why
 *  every slice is labelled — six hues cannot be told apart from a legend alone.
 *
 *  Refuses past six slices rather than cycling: a seventh would repeat a hue
 *  already in use and two different things would be the same colour. */
export function pieSvg(r: QueryResult, hole = 0): string {
  const R = 84;
  const PAD = 12;
  const ROW = 26;
  const cx = PAD + R;
  const cy = PAD + R;
  const vals = r.rows.map((row) => Math.max(0, row.values[0] ?? 0));
  const total = vals.reduce((a, b) => a + b, 0);
  if (total <= 0) return barSvg(r);

  // THE BOX FROM THE CONTENT, not the bar chart's 640x214 canvas.
  //
  // A circle 168 units across sat in a box 640 wide: 26% of the figure was the
  // figure and the rest was a legend column sized for labels nobody had, so at
  // any rendered width the doughnut came out a third the size of the space it
  // was given. Same fault as the pizza and the radar, same fix - the content
  // decides the box.
  //
  // Text is measured in characters rather than laid out, because there is no
  // DOM here: this serializer runs headless. ~6 units per character at the
  // legend's 11px is close enough to keep the value column off the edge, and
  // erring wide costs a few units of margin while erring narrow clips a label.
  const labels = r.rows.map((row) => clip(row.label, 30));
  const values = r.rows.map((row, i) => `${row.text[0] ?? ""} (${((vals[i] / total) * 100).toFixed(0)}%)`);
  const wide = (xs: string[]) => Math.max(0, ...xs.map((s) => s.length)) * 6.2;
  const keyW = 16 + wide(labels) + 24 + wide(values) + PAD;
  const keyX = cx + R + PAD;
  const f = frame(
    keyX + keyW,
    Math.max(R * 2 + PAD * 2, r.rows.length * ROW + PAD * 2),
    { l: 0, r: 0, t: 0, b: 0 },
  );

  let a0 = 0;
  const slices = vals
    .map((v, i) => {
      const a1 = a0 + (v / total) * Math.PI * 2;
      const out =
        `<g transform="translate(${cx},${cy})"><title>${esc(r.rows[i].label)} — ${esc(r.rows[i].text[0])}</title>` +
        // A 2px surface ring: adjacent slices must not blend into one shape.
        `<path d="${slicePath(a0, a1, R, hole)}" fill="${catColor(r, i)}" ` +
        `stroke="var(--panel,#fff)" stroke-width="2" class="ft-mark ft-mark--c${i}"/></g>`;
      a0 = a1;
      return out;
    })
    .join("");

  // The direct labels the palette's CVD warning obliges, and the only way to
  // read a share off a printed page. Centred against the circle rather than
  // pinned to the top: three entries beside a 168-unit disc read as a caption
  // hanging off its shoulder.
  const keyTop = Math.max(PAD, (f.H - r.rows.length * ROW) / 2);
  const key = r.rows
    .map(
      (row, i) =>
        `<g transform="translate(${n2(keyX)},${n2(keyTop + i * ROW)})">` +
        `<rect x="0" y="0" width="10" height="10" rx="2" fill="${catColor(r, i)}" ` +
        `class="ft-swatch ft-swatch--c${i}"/>` +
        `<text x="16" y="9" class="ft-chart__legend">${esc(labels[i])}</text>` +
        `<text x="${n2(keyW - PAD)}" y="9" text-anchor="end" class="ft-chart__val">` +
        `${esc(values[i])}</text></g>`,
    )
    .join("");

  return chartSvgTag(f, slices + key, `${r.measure.label} by ${r.dim.label}`);
}

/** Two measures against each other, one dot per category. Needs a split, so
 *  each row carries an x and a y — without one there is only a single number
 *  per row and a scatter of it against nothing is a strip plot. */
export function scatterSvg(r: QueryResult): string {
  if (r.series.length < 2) return barSvg(r);
  const f = frame(300, 250, { l: PT.padL + 4, r: PT.padR, t: PT.padT, b: PT.padB + 10 });
  const xs = r.rows.map((row) => row.values[0] ?? 0);
  const ys = r.rows.map((row) => row.values[1] ?? 0);
  const y = linear(Math.max(...ys), f);
  const x = linearX(Math.max(...xs), f);

  const grid =
    y
      .ticks(5)
      .map((v) => {
        const gy = y(v);
        return (
          `<line x1="${f.pad.l}" y1="${n2(gy)}" x2="${f.W - f.pad.r}" y2="${n2(gy)}" class="ft-chart__grid"/>` +
          `<text x="${f.pad.l - 6}" y="${n2(gy + 3.5)}" class="ft-chart__axis">${esc(v.toFixed(r.measure.dp))}</text>`
        );
      })
      .join("") +
    x
      .ticks(5)
      .map(
        (v) =>
          `<text x="${n2(x(v))}" y="${f.H - 22}" text-anchor="middle" class="ft-chart__axis">${esc(v.toFixed(r.measure.dp))}</text>`,
      )
      .join("");

  const dots = r.rows
    .map(
      (row, i) =>
        `<g><title>${esc(row.label)} — ${esc(r.series[0])} ${esc(row.text[0])}, ` +
        `${esc(r.series[1])} ${esc(row.text[1])}</title>` +
        `<circle cx="${n2(x(xs[i]))}" cy="${n2(y(ys[i]))}" r="5" fill="${SINGLE_COLOR}" ` +
        `stroke="var(--panel,#fff)" stroke-width="2"/></g>`,
    )
    .join("");

  return chartSvgTag(
    f,
    grid +
      dots +
      `<text x="${n2(f.pad.l + f.plotW / 2)}" y="${f.H - 6}" text-anchor="middle" class="ft-chart__legend">${esc(r.series[0])}</text>` +
      `<text transform="translate(12,${n2(f.pad.t + f.plotH / 2)}) rotate(-90)" text-anchor="middle" class="ft-chart__legend">${esc(r.series[1])}</text>`,
    `${r.series[1]} against ${r.series[0]} by ${r.dim.label}`,
  );
}

/** Columns and a line together, ON ONE AXIS.
 *
 *  Excel's combo chart offers a secondary axis and it is the single most
 *  misread chart there is: two scales behind one picture means the crossing
 *  point of the two series is set by whoever picked the axis ranges, not by
 *  the data. Both series here are drawn against the same maximum. If they are
 *  not comparable on one scale, they are two charts. */
export function comboSvg(r: QueryResult): string {
  if (r.series.length < 2) return columnSvg(r);
  const W = 640;
  const H = 240;
  const pad = { l: 46, r: 12, t: 24, b: 34 };
  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;
  const max = r.max;
  const n = r.rows.length;
  const slot = plotW / Math.max(1, n);
  const barW = Math.min(46, slot * 0.6);
  const y = (v: number) => pad.t + plotH - (max > 0 ? (v / max) * plotH : 0);

  const grid = [0, 0.25, 0.5, 0.75, 1]
    .map((f) => {
      const gy = pad.t + plotH - f * plotH;
      return (
        `<line x1="${pad.l}" y1="${n2(gy)}" x2="${W - pad.r}" y2="${n2(gy)}" class="ft-chart__grid"/>` +
        `<text x="${pad.l - 6}" y="${n2(gy + 3.5)}" class="ft-chart__axis">${(max * f).toFixed(r.measure.dp)}</text>`
      );
    })
    .join("");

  const cx = (i: number) => pad.l + i * slot + slot / 2;
  const step = labelStep(r.rows, plotW, 12);
  const bars = r.rows
    .map(
      (row, i) =>
        `<g><title>${esc(row.label)} · ${esc(r.series[0])} — ${esc(row.text[0])}</title>` +
        `<path d="${colPath(
          cx(i) - barW / 2,
          y(row.values[0] ?? 0),
          barW,
          Math.max(0, pad.t + plotH - y(row.values[0] ?? 0)),
        )}" fill="${SERIES_COLORS[0]}"/></g>` +
        (i % step
          ? ""
          : `<text x="${n2(cx(i))}" y="${H - 12}" text-anchor="middle" class="ft-chart__cat">${esc(clip(row.label, 12))}</text>`),
    )
    .join("");
  const pts = r.rows.map((row, i) => `${n2(cx(i))},${n2(y(row.values[1] ?? 0))}`);
  const dots = r.rows
    .map(
      (row, i) =>
        `<g><title>${esc(row.label)} · ${esc(r.series[1])} — ${esc(row.text[1])}</title>` +
        `<circle cx="${n2(cx(i))}" cy="${n2(y(row.values[1] ?? 0))}" r="4.5" fill="${SERIES_COLORS[1]}" ` +
        `stroke="var(--panel,#fff)" stroke-width="2"/></g>`,
    )
    .join("");

  return (
    `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" ` +
    `aria-label="${esc(r.measure.label)} by ${esc(r.dim.label)}" ` +
    `style="display:block;max-width:100%;height:auto">` +
    svgLegend(r.series, pad.l, 0, r.colors ?? []) +
    `${grid}${bars}` +
    `<polyline points="${pts.join(" ")}" fill="none" stroke="${SERIES_COLORS[1]}" stroke-width="2" stroke-linejoin="round"/>${dots}` +
    `</svg>`
  );
}


// ----------------------------------------------------------- football, on grass

/** The same pitch every board and shot map is drawn on, so a note's pictures
 *  all agree about what a pitch looks like. */
/** A pitch figure: the shared wrapper, plus whatever marks go on the grass.
 *  The wrapper lives in chart-core so a board, a shot map and a bar chart all
 *  agree about how a figure is sized and clipped. */
function grass(body: string, label: string, sub?: string): string {
  return pitchSvgTag(body, label) + (sub ? `<p class="ft-chart__sub">${esc(sub)}</p>` : "");
}


// --------------------------------------------------------------- goal mouth
// THE ONE FIGURE THAT IS NOT ON THE GROUND. A goal is 7.32 m by 2.44 m and a
// penalty that missed has a position in THAT frame - over the bar, wide of the
// post - which no crop of the pitch can express. It is drawn in metres for the
// same reason the board is: a frame stored in percent is anisotropic, and a
// shot 30 cm inside the post would sit at a different distance from it
// depending on which way you measured.

/** Goal metres: 0-100 across the frame and 0-100 up it, into a space where the
 *  posts are 7.32 m apart and the bar is 2.44 m up. SVG y grows downward, so
 *  the height is measured from the crossbar rather than from the grass. */
const GW = 7.32;
const GH = 2.44;
const gx = (x: number) => (x / 100) * GW;
const gy = (y: number) => GH - (y / 100) * GH;

export function goalMouthSvg(e: GoalExtra): string {
  // Room outside the frame for the shots that missed. A goal graphic cropped to
  // the posts can only draw the ones that went in, which is the half of the
  // question nobody needs a picture for.
  const padX = 2.6;
  const padTop = 1.5;
  const padBot = 0.55;
  const W = GW + padX * 2;
  const H = GH + padTop + padBot;
  const post = 0.12;

  // The net, as a grid rather than a texture: it says "this area is the goal"
  // at a glance and prints as hairlines rather than as a grey smear.
  const step = GW / 12;
  const mesh =
    Array.from({ length: 11 }, (_, i) =>
      `<line x1="${n2((i + 1) * step)}" y1="0" x2="${n2((i + 1) * step)}" y2="${GH}"/>`).join("") +
    Array.from({ length: 4 }, (_, i) =>
      `<line x1="0" y1="${n2(((i + 1) * GH) / 5)}" x2="${GW}" y2="${n2(((i + 1) * GH) / 5)}"/>`).join("");

  const frame =
    `<rect x="0" y="0" width="${GW}" height="${GH}" fill="#eef1f4"/>` +
    `<g stroke="#ccd2da" stroke-width="0.022">${mesh}</g>` +
    // A GOAL IS WHITE, and that is not decoration: near-black posts were
    // invisible against the app's dark page - the figure read as a grey
    // rectangle floating in nothing - and a white frame with a grey edge reads
    // on a dark screen and on paper, which every other figure here also has to
    // do. Drawn with their real thickness and OUTSIDE the mouth, so a shot that
    // hit the inside of a post is inside the frame.
    `<g fill="#ffffff" stroke="#5c646d" stroke-width="0.035">` +
    `<rect x="${-post}" y="${-post}" width="${post}" height="${GH + post}"/>` +
    `<rect x="${GW}" y="${-post}" width="${post}" height="${GH + post}"/>` +
    `<rect x="${-post}" y="${-post}" width="${GW + post * 2}" height="${post}"/>` +
    `</g>` +
    `<line x1="${-padX}" y1="${GH}" x2="${GW + padX}" y2="${GH}" stroke="#7b838d" stroke-width="0.03"/>`;

  // THE NUMBER GOES IN THE DISC, the way the source draws it and the way a
  // magnet on the board already does. Set above the dot it collided with every
  // neighbour - and penalties cluster, which is the whole reason to plot them.
  const dot = (sh: GoalExtra["shots"][number]) => {
    const x = n2(gx(sh.x));
    const y = n2(gy(sh.y));
    const r = sh.label ? 0.3 : 0.19;
    return (
      `<g><title>${esc(sh.label ? `${sh.label} · ` : "")}${sh.scored ? "scored" : "missed"}</title>` +
      `<circle cx="${x}" cy="${y}" r="${r}" class="${markCls(sh.scored ? 0 : 1)}" ` +
      (sh.scored
        ? `fill="#e11d26" stroke="#ffffff" stroke-width="0.05"/>`
        : `fill="#ffffff" stroke="#1b1f24" stroke-width="0.06"/>`) +
      (sh.label
        ? `<text x="${x}" y="${y}" text-anchor="middle" dominant-baseline="central" ` +
          `font-size="0.26" fill="${sh.scored ? "#ffffff" : "#1b1f24"}" ` +
          `font-family="system-ui, sans-serif" font-weight="700">${esc(sh.label)}</text>`
        : "") +
      `</g>`
    );
  };

  // Misses first, so a goal is never hidden under one - the same rule the shot
  // map follows.
  const ordered = [...e.shots].sort((a, b) => Number(a.scored) - Number(b.scored));
  const scored = e.shots.filter((sh) => sh.scored).length;
  return (
    `<svg class="ft-goalfig" viewBox="${n2(-padX)} ${n2(-padTop)} ${n2(W)} ${n2(H)}" ` +
    `preserveAspectRatio="xMidYMid meet" width="100%" role="img" ` +
    `aria-label="${esc(`Goal mouth${e.who ? ` — ${e.who}` : ""}`)}" ` +
    `style="display:block;max-width:100%;height:auto">` +
    frame +
    ordered.map(dot).join("") +
    `</svg>` +
    // Its own legend rather than legendRow's squares: the marks here are a
    // filled dot and a ring, and a key that does not look like the thing it
    // keys is a key you have to decode twice.
    `<p class="ft-chart__legendrow">` +
    `<span class="ft-chart__key"><span class="ft-chart__swatch ft-chart__swatch--dot" ` +
    `style="background:#e11d26"></span>Scored (${scored})</span>` +
    `<span class="ft-chart__key"><span class="ft-chart__swatch ft-chart__swatch--ring">` +
    `</span>Missed (${e.shots.length - scored})</span></p>`
  );
}

/** Discs where people were, or where a team sheet says they line up.
 *
 *  A MEASURED map carries the caveat that matters more than the picture: sides
 *  swap at half time and the pipeline has no attacking direction to correct
 *  for it, so every mean lands nearer the middle than the player ever played.
 *  Read it as spread and pairings, never as "this is his position".
 */
export function spotsSvg(e: SpotsExtra): string {
  const discs = e.spots
    .map((s) => {
      const cx = mx(Math.max(2, Math.min(98, s.x)));
      const cy = my(Math.max(2, Math.min(98, s.y)));
      // The BOARD's discs, not a second palette. A position map and a board
      // drawn from the same match sat side by side in one note wearing
      // different colours for the same two teams, which reads as two different
      // pairs of teams.
      const side = s.side === "away" ? "away" : "home";
      return (
        `<g><title>${esc(s.label ? `#${s.label}` : "Unnumbered")}</title>` +
        `<circle cx="${n2(cx)}" cy="${n2(cy)}" r="2.1" fill="${MAGNET_FILL[side]}" ` +
        `stroke="#ffffff" stroke-width="0.25"/>` +
        (s.label
          ? `<text x="${n2(cx)}" y="${n2(cy + 0.8)}" text-anchor="middle" ` +
            `style="font:600 2.1px var(--font-mono,monospace);fill:${MAGNET_TEXT[side]}">${esc(s.label)}</text>`
          : "") +
        `</g>`
      );
    })
    .join("");
  return grass(
    discs,
    e.measured ? "Average positions" : "Line-up",
    e.measured
      ? "Each disc is one player's MEAN position. Sides swap at half time and the pipeline has no attacking direction to undo that, so every mean sits nearer the middle than the player played. Read the spread and who sits beside whom, not the position."
      : "Where the team sheet puts each shirt. Drawn, not measured.",
  );
}

/** Where a side actually spent the match.
 *
 *  Binned into cells rather than blurred into a cloud, and the cells are
 *  labelled with their share once they are worth reading. A scouting report
 *  never prints an unlabelled gradient: every zone graphic in one carries its
 *  own number, because "this bit is a slightly darker orange than that bit" is
 *  not a finding anybody can act on.
 *
 *  Warm ramp, one hue. Blue at eight percent opacity read as nothing at all —
 *  it was a picture whose caption had to explain that it was a picture.
 */
export function heatSvg(e: HeatExtra): string {
  // The grid comes from the FIGURE, not from a constant here. The binning
  // happens at freeze time now (see HeatExtra), and a note saved against a
  // 12x8 grid has to keep drawing as 12x8 whatever this file later prefers.
  const COLS = Math.max(1, e.cols);
  const ROWS = Math.max(1, e.rows);
  const bins = e.bins;
  const peak = Math.max(1, ...bins);
  const total = e.total || 1;
  const cells = bins
    .map((v, i) => {
      if (!v) return "";
      const c = i % COLS;
      const r = Math.floor(i / COLS);
      const x0 = mx((c / COLS) * 100);
      const y0 = my((r / ROWS) * 100);
      const w = mx(100 / COLS) - mx(0);
      const h = my(100 / ROWS) - my(0);
      const t = v / peak;
      const share = (v / total) * 100;
      return (
        `<g><title>${share.toFixed(1)}% of the time here</title>` +
        `<rect x="${n2(x0)}" y="${n2(y0)}" width="${n2(w)}" height="${n2(h)}" ` +
        `fill="${HEAT}" fill-opacity="${n2(0.1 + t * 0.8)}"/>` +
        // The number, wherever the cell is busy enough to carry one. Below
        // that it would be a page of 2%s.
        (share >= 2.5
          ? `<text x="${n2(x0 + w / 2)}" y="${n2(y0 + h / 2 + 1)}" text-anchor="middle" ` +
            `style="font:600 2.6px var(--font-mono,monospace);fill:#3a1206">${share.toFixed(0)}%</text>`
          : "") +
        `</g>`
      );
    })
    .join("");
  return grass(
    cells,
    `Where ${e.who} played`,
    `${e.who}. Share of the side's tracked positions in each zone — where the ` +
      "BODIES were, which is not where the ball was.",
  );
}

/** Bands of the pitch, each carrying a share. Thirds run across the length;
 *  channels run along it. Drawn on the pitch because "26% attacking third" is
 *  a sentence and this is a place. */
export function zonePitchSvg(e: ZoneExtra): string {
  const total = e.bands.reduce((a, b) => a + b.share, 0) || 1;
  const n = e.bands.length;
  const bands = e.bands
    .map((b, i) => {
      const t = b.share / Math.max(...e.bands.map((x) => x.share), 1);
      const from = (i / n) * 100;
      const size = 100 / n;
      const x0 = e.axis === "length" ? mx(from) : mx(0);
      const y0 = e.axis === "length" ? my(0) : my(from);
      const w = e.axis === "length" ? mx(size) - mx(0) : mx(100) - mx(0);
      const h = e.axis === "length" ? my(100) - my(0) : my(size) - my(0);
      const cx = e.axis === "length" ? mx(from + size / 2) : mx(50);
      const cy = e.axis === "length" ? my(50) : my(from + size / 2);
      return (
        `<g><title>${esc(b.label)} — ${((b.share / total) * 100).toFixed(0)}%</title>` +
        `<rect x="${n2(x0)}" y="${n2(y0)}" width="${n2(w)}" height="${n2(h)}" ` +
        `fill="${SINGLE_COLOR}" fill-opacity="${n2(0.12 + t * 0.6)}"/>` +
        `<text x="${n2(cx)}" y="${n2(cy - 1)}" text-anchor="middle" ` +
        `style="font:600 4px var(--font-mono,monospace);fill:#fff">${((b.share / total) * 100).toFixed(0)}%</text>` +
        `<text x="${n2(cx)}" y="${n2(cy + 3.6)}" text-anchor="middle" ` +
        `style="font:2.4px var(--font-mono,monospace);fill:#fff;opacity:.85">${esc(b.label)}</text></g>`
      );
    })
    .join("");
  return grass(bands, `${e.who} by zone`, `${e.who}. Shares of the whole, drawn where they happened.`);
}

// ------------------------------------------------- one moment, three figures
// A freeze frame, the hull it makes, and a coarse pitch-control grid. All read
// FreezeExtra; see it for why the moment is stored once and drawn three ways.

/** Convex hull, monotone chain. Mirrors model/engine/stats.py::_hull_area, which
 *  computes the AREA from the same construction and emits only the number - so
 *  the polygon has to be rebuilt here rather than read from the report. */
function hull(pts: [number, number][]): [number, number][] {
  if (pts.length < 3) return pts;
  const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: number[], a: number[], b: number[]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (src: [number, number][]) => {
    const out: [number, number][] = [];
    for (const q of src) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], q) <= 0) out.pop();
      out.push(q);
    }
    out.pop();
    return out;
  };
  return [...half(p), ...half([...p].reverse())];
}

/** The honesty line every figure off a frame slice carries. */
const seenNote = (e: FreezeExtra) =>
  `${e.seen} of ${e.squad} players visible at ${Math.floor(e.t_s / 60)}:` +
  `${String(Math.floor(e.t_s % 60)).padStart(2, "0")}` +
  (e.synthetic ? ". Positions are generated, not measured." : "") +
  (e.seen < e.squad
    ? " A shape drawn from part of a side is not that side's shape."
    : "");

export function freezeSvg(e: FreezeExtra, mode: "freeze" | "hull" | "control"): string {
  const disc = (p: FreezeExtra["players"][number], i: number) => {
    const c = p.side === "away" ? SERIES_COLORS[1] : SERIES_COLORS[0];
    return (
      `<g><title>${esc(p.label || "unnumbered")}</title>` +
      `<circle cx="${n2(mx(p.x))}" cy="${n2(my(p.y))}" r="1.5" fill="${c}" ` +
      `stroke="#ffffff" stroke-width="0.3" class="ft-mark ft-mark--c${i}"/>` +
      (p.label
        ? `<text x="${n2(mx(p.x))}" y="${n2(my(p.y) + 0.55)}" text-anchor="middle" ` +
          `style="font:1.5px var(--font-mono,monospace);fill:#fff">${esc(p.label)}</text>`
        : "") +
      `</g>`
    );
  };

  let under = "";

  if (mode === "hull") {
    // One hull per side, outfielders only in spirit - the caller decides who is
    // in `players`. Drawn under the discs so a vertex is still readable.
    for (const side of ["home", "away"]) {
      const own = e.players.filter((p) => p.side === side);
      if (own.length < 3) continue;
      const poly = hull(own.map((p) => [mx(p.x), my(p.y)] as [number, number]));
      const c = side === "away" ? SERIES_COLORS[1] : SERIES_COLORS[0];
      under +=
        `<polygon points="${poly.map(([x, y]) => `${n2(x)},${n2(y)}`).join(" ")}" ` +
        `fill="${c}" fill-opacity="0.13" stroke="${c}" stroke-width="0.35" ` +
        `stroke-dasharray="1.2 0.8"/>`;
    }
  }

  if (mode === "control") {
    // A GRID, NOT A VORONOI, and named pitch control rather than Voronoi for
    // exactly that reason. A true tessellation needs a Delaunay library this
    // document cannot carry - the file must open with no bundle - and at print
    // size a 42x27 grid of nearest-player cells is the same picture. Claiming
    // "Voronoi" for something that is not one is how a figure gets quoted as a
    // geometry it never computed.
    const COLS = 42;
    const ROWS = 27;
    // RUN-LENGTH ALONG THE ROW, and it is not an optimisation for its own sake.
    // The rendered markup is what the DOCUMENT stores - see HeatExtra, where
    // one figure reached 423 KB and two would have failed the save outright.
    // A cell grid is 1,134 rects; a side owns long unbroken runs of a row, so
    // encoding runs draws exactly the same picture in about a tenth of them.
    for (let r = 0; r < ROWS; r++) {
      const py = ((r + 0.5) / ROWS) * 100;
      let runSide = "";
      let runStart = 0;
      const flush = (endCol: number) => {
        if (!runSide) return;
        const col = runSide === "away" ? SERIES_COLORS[1] : SERIES_COLORS[0];
        const x0 = mx((runStart / COLS) * 100);
        const x1 = mx((endCol / COLS) * 100);
        under +=
          `<rect x="${n2(x0)}" y="${n2(my(((r + 1) / ROWS) * 100))}" ` +
          `width="${n2(x1 - x0)}" height="${n2(68 / ROWS)}" fill="${col}" ` +
          `fill-opacity="0.16" stroke="none"/>`;
      };
      for (let c = 0; c < COLS; c++) {
        const px = ((c + 0.5) / COLS) * 100;
        let best = Infinity;
        let side = "";
        for (const p of e.players) {
          const d = (p.x - px) ** 2 + (p.y - py) ** 2;
          if (d < best) {
            best = d;
            side = p.side;
          }
        }
        if (side !== runSide) {
          flush(c);
          runSide = side;
          runStart = c;
        }
      }
      flush(COLS);
    }
  }

  const ball = e.ball
    ? `<circle cx="${n2(mx(e.ball.x))}" cy="${n2(my(e.ball.y))}" r="0.9" ` +
      `fill="#ffffff" stroke="#14171b" stroke-width="0.35"/>`
    : "";

  const label =
    mode === "hull" ? "Team shape" : mode === "control" ? "Pitch control" : "Freeze frame";
  return grass(
    under + e.players.map(disc).join("") + ball,
    label,
    `${label}. ${seenNote(e)}`,
  );
}

// --------------------------------------------------- passes, three figures
// network / progressive / sonar, all off PassExtra.

/** Mean position per player, from the passes they made. The network's nodes. */
function passNodes(e: PassExtra, side: string) {
  const acc = new Map<string, { x: number; y: number; n: number }>();
  for (const p of e.passes) {
    if (p.side !== side || !p.from) continue;
    const a = acc.get(p.from) ?? { x: 0, y: 0, n: 0 };
    a.x += p.x0;
    a.y += p.y0;
    a.n += 1;
    acc.set(p.from, a);
  }
  return new Map(
    [...acc].map(([k, a]) => [k, { x: a.x / a.n, y: a.y / a.n, n: a.n }] as const),
  );
}

export function passSvg(e: PassExtra, mode: "network" | "progressive" | "sonar"): string {
  const note =
    (e.synthetic ? "Passes are generated, not measured. " : "") +
    (e.dropped ? `${e.dropped} more not drawn. ` : "");

  if (mode === "progressive") {
    // Direction and outcome, nothing aggregated. Failed passes under completed
    // ones so a completed arrow is never hidden by a failure.
    const arrow = (p: PassExtra["passes"][number]) => {
      const c = p.side === "away" ? SERIES_COLORS[1] : SERIES_COLORS[0];
      return (
        `<g><title>${esc(p.from)} → ${esc(p.to)}${p.ok ? "" : " (lost)"}</title>` +
        `<line x1="${n2(mx(p.x0))}" y1="${n2(my(p.y0))}" x2="${n2(mx(p.x1))}" ` +
        `y2="${n2(my(p.y1))}" stroke="${c}" stroke-width="${p.ok ? 0.3 : 0.22}" ` +
        `stroke-opacity="${p.ok ? 0.85 : 0.4}"${p.ok ? "" : ' stroke-dasharray="1 0.8"'}/>` +
        `<circle cx="${n2(mx(p.x1))}" cy="${n2(my(p.y1))}" r="${p.ok ? 0.5 : 0.35}" ` +
        `fill="${p.ok ? c : "none"}" stroke="${c}" stroke-width="0.2"/></g>`
      );
    };
    const ordered = [...e.passes].sort((a, b) => Number(a.ok) - Number(b.ok));
    return grass(
      ordered.map(arrow).join(""),
      "Passes",
      `Every pass, where it started and finished. Hollow is lost. ${note}`,
    );
  }

  if (mode === "sonar") {
    // A rose per player at his mean position: eight sectors, radius by how many
    // passes went that way. The figure a coach reads for "he only ever plays
    // backwards".
    const SECT = 8;
    let body = "";
    for (const side of ["home", "away"]) {
      const nodes = passNodes(e, side);
      const c = side === "away" ? SERIES_COLORS[1] : SERIES_COLORS[0];
      for (const [who, node] of nodes) {
        const bins = new Array(SECT).fill(0);
        for (const p of e.passes) {
          if (p.side !== side || p.from !== who) continue;
          const a = Math.atan2(-(p.y1 - p.y0), p.x1 - p.x0); // screen y is down
          bins[(Math.floor(((a + Math.PI * 2) % (Math.PI * 2)) / ((Math.PI * 2) / SECT))) % SECT]++;
        }
        const top = Math.max(1, ...bins);
        const cx = mx(node.x);
        const cy = my(node.y);
        body +=
          `<g><title>${esc(who)}, ${node.n} passes</title>` +
          bins
            .map((v, i) => {
              if (!v) return "";
              const r = (v / top) * 4.2;
              const a0 = (i / SECT) * Math.PI * 2;
              const a1 = ((i + 1) / SECT) * Math.PI * 2;
              // Screen coords: y grows downward, so negate the sine.
              const p0 = `${n2(cx + Math.cos(a0) * r)},${n2(cy - Math.sin(a0) * r)}`;
              const p1 = `${n2(cx + Math.cos(a1) * r)},${n2(cy - Math.sin(a1) * r)}`;
              return (
                `<path d="M ${n2(cx)},${n2(cy)} L ${p0} A ${n2(r)} ${n2(r)} 0 0 0 ${p1} Z" ` +
                `fill="${c}" fill-opacity="0.55" stroke="${c}" stroke-width="0.12"/>`
              );
            })
            .join("") +
          `<circle cx="${n2(cx)}" cy="${n2(cy)}" r="0.5" fill="#ffffff" stroke="${c}" ` +
          `stroke-width="0.25"/></g>`;
      }
    }
    return grass(body, "Pass sonar", `Where each player's passes went. ${note}`);
  }

  // network
  let body = "";
  for (const side of ["home", "away"]) {
    const nodes = passNodes(e, side);
    const c = side === "away" ? SERIES_COLORS[1] : SERIES_COLORS[0];
    const edges = new Map<string, number>();
    for (const p of e.passes) {
      if (p.side !== side || !p.from || !p.to || p.from === p.to || !p.ok) continue;
      // Undirected: a pairing is a relationship, and drawing both directions
      // as separate lines makes a two-way partnership look like two weak ones.
      const key = p.from < p.to ? `${p.from}|${p.to}` : `${p.to}|${p.from}`;
      edges.set(key, (edges.get(key) ?? 0) + 1);
    }
    const heaviest = Math.max(1, ...edges.values());
    for (const [key, w] of edges) {
      const [a, b] = key.split("|");
      const na = nodes.get(a);
      const nb = nodes.get(b);
      if (!na || !nb) continue;
      body +=
        `<g><title>${esc(a)} ↔ ${esc(b)}, ${w}</title>` +
        `<line x1="${n2(mx(na.x))}" y1="${n2(my(na.y))}" x2="${n2(mx(nb.x))}" ` +
        `y2="${n2(my(nb.y))}" stroke="${c}" stroke-width="${n2(0.15 + (w / heaviest) * 1.1)}" ` +
        `stroke-opacity="0.5"/></g>`;
    }
    const busiest = Math.max(1, ...[...nodes.values()].map((n) => n.n));
    for (const [who, node] of nodes) {
      body +=
        `<g><title>${esc(who)}, ${node.n} passes</title>` +
        `<circle cx="${n2(mx(node.x))}" cy="${n2(my(node.y))}" ` +
        `r="${n2(1.1 + (node.n / busiest) * 1.5)}" fill="${c}" stroke="#ffffff" ` +
        `stroke-width="0.3"/>` +
        `<text x="${n2(mx(node.x))}" y="${n2(my(node.y) + 0.55)}" text-anchor="middle" ` +
        `style="font:1.5px var(--font-mono,monospace);fill:#fff">${esc(who)}</text></g>`;
    }
  }
  return grass(
    body,
    "Pass network",
    "Nodes at each player's mean passing position, edges by volume. " +
      `Undirected, completed passes only. ${note}`,
  );
}

// -------------------------------------------------------------------- pizza
// ONE PLAYER AGAINST A POPULATION. See PizzaExtra for why this and not a radar.

/** The faint full-radius ring every wedge is read against. Without it a short
 *  wedge is just a short wedge; with it, it is visibly a fraction of the whole,
 *  which is the only thing a percentile means. */
const PIZZA_R = 88;
/** How far past the ring the outer labels sit. The box has to allow for it. */
const PIZZA_LABEL = 22;
/** ...and then for the LABEL ITSELF. The anchor point is not the text: a
 *  left-side label is anchored `end`, so it runs leftwards from its point, and
 *  a box sized to the anchor ring clips it. That is exactly what happened the
 *  first time this box was tightened - "Pass %" came off the left edge - so the
 *  side allowance is a named constant rather than a fudge in the width. Sized
 *  for `clip(label, 18)` at the axis size. */
const PIZZA_TEXT = 58;
/** Width the band key needs when there is one. Nothing when there is not. */
const PIZZA_KEY = 190;

/** THE BOX A ROUND FIGURE ACTUALLY NEEDS.
 *
 *  Every family inherited the bar chart's 640-wide canvas, and a circle cannot
 *  fill a 2.46:1 box: the radar drew a 191x229 picture into 640x260 and the
 *  element - which is sized to the VIEWBOX, not to the ink - came out four
 *  times the picture. On a page that reads as a small circle adrift in a
 *  full-width row, which is the reported bug.
 *
 *  So the aspect is derived from the content, which is the relationship
 *  mplsoccer states and this file had backwards: the figure's shape is the
 *  constraint and the box is the free variable, not the reverse. A key is only
 *  paid for when a key is drawn. */
function roundBox(withKey: boolean) {
  const ring = PIZZA_R + PIZZA_LABEL;
  const w = ring * 2 + PIZZA_TEXT * 2;
  // Vertically a label sits above or below its point rather than beside it, so
  // it costs a line rather than a word - much less than the sides.
  const h = ring * 2 + 26;
  return {
    f: frame(w + (withKey ? PIZZA_KEY : 0), h, { l: 0, r: 0, t: 0, b: 0 }),
    cx: ring + PIZZA_TEXT,
    cy: ring + 13,
    keyX: w + 6,
  };
}

export function pizzaSvg(e: PizzaExtra): string {
  const n = e.slices.length;
  if (!n) return `<p class="ft-chart__sub">Nothing to compare.</p>`;

  // One hue per band, in the order the bands first appear, so a report that
  // always lists attacking metrics first always colours them the same.
  const groups = [...new Set(e.slices.map((s) => s.group ?? ""))];
  const { f, cx, cy, keyX } = roundBox(groups.length > 1);
  const hue = (s: PizzaExtra["slices"][number]) =>
    groups.length > 1 ? CATEGORY_COLORS[groups.indexOf(s.group ?? "") % CATEGORY_COLORS.length] : SINGLE_COLOR;

  const step = (Math.PI * 2) / n;
  const wedges = e.slices
    .map((s, i) => {
      const a0 = i * step;
      const a1 = a0 + step;
      // Clamp rather than trust: a percentile outside 0-100 is a caller bug,
      // and a wedge drawn past the ring would read as better than best.
      const pct = Math.max(0, Math.min(100, s.pct));
      const r = (pct / 100) * PIZZA_R;
      const mid = a0 + step / 2;
      // The label sits OUTSIDE the ring and the number at the middle of its own
      // wedge. Both radii were tuned against a full-depth wedge, which is the
      // colliding case: at r = PIZZA_R the value sits at 0.62 of the way out
      // and the label a clear 22 beyond the rim, so they cannot meet. Putting
      // the value at `r - 14` instead pinned it to the rim and it overprinted
      // the label on every metric a player was best at.
      const lr = PIZZA_R + PIZZA_LABEL;
      const lx = cx + Math.sin(mid) * lr;
      const ly = cy - Math.cos(mid) * lr;
      const anchor = Math.sin(mid) > 0.1 ? "start" : Math.sin(mid) < -0.1 ? "end" : "middle";
      const vr = r * 0.62;
      const vx = cx + Math.sin(mid) * vr;
      const vy = cy - Math.cos(mid) * vr;
      return (
        `<g><title>${esc(s.label)} — ${esc(s.text)} (${pct.toFixed(0)}th)</title>` +
        `<path transform="translate(${cx},${cy})" d="${slicePath(a0, a1, PIZZA_R)}" ` +
        `fill="${hue(s)}" opacity="0.12"/>` +
        `<path transform="translate(${cx},${cy})" d="${slicePath(a0, a1, r)}" ` +
        `fill="${hue(s)}" stroke="var(--panel,#fff)" stroke-width="1.5" ` +
        `class="ft-mark ft-mark--c${i}"/>` +
        (r > 30
          ? `<text x="${n2(vx)}" y="${n2(vy + 3)}" text-anchor="middle" ` +
            `style="font:9px var(--font-mono,monospace);fill:#fff">${esc(s.text)}</text>`
          : "") +
        `<text x="${n2(lx)}" y="${n2(ly + 3)}" text-anchor="${anchor}" ` +
        `class="ft-chart__axis">${esc(clip(s.label, 18))}</text></g>`
      );
    })
    .join("");

  // The population is not a footnote. A percentile against ten team-mates and
  // one against a season of four hundred are different claims, and the figure
  // has to carry which.
  const key =
    groups.length > 1
      ? groups
          .map(
            (g, i) =>
              `<g transform="translate(${n2(keyX)},${n2(30 + i * 22)})">` +
              `<rect width="10" height="10" rx="2" fill="${CATEGORY_COLORS[i % CATEGORY_COLORS.length]}" ` +
              `class="ft-swatch ft-swatch--c${i}"/>` +
              `<text x="16" y="9" class="ft-chart__legend">${esc(clip(g || "Other", 22))}</text></g>`,
          )
          .join("")
      : "";

  return (
    chartSvgTag(f, wedges + key, `${e.who} by percentile`) +
    `<p class="ft-chart__sub">${esc(e.who)}. Percentile rank against ${e.population} ` +
    `player${e.population === 1 ? "" : "s"}; the ring is the whole field.</p>`
  );
}

/** THE SAME NUMBERS AS THE PIZZA, JOINED UP.
 *
 *  Offered because analysts ask for it and read it fluently, and kept second to
 *  the pizza on purpose. A radar joins its points, and the polygon that makes
 *  has an AREA which reads as a quantity and is not one: reorder the axes and
 *  the same player encloses a different area. So the wedge count and the ring
 *  are drawn, the values are printed at the vertices, and the fill is faint -
 *  everything that can be done to stop the shape being read as a total. When
 *  the audience is a coach rather than an analyst, use the pizza.
 */
export function radarSvg(e: PizzaExtra): string {
  const n = e.slices.length;
  if (n < 3) return pizzaSvg(e); // a triangle is the least a radar can be
  // A radar draws no key - its labels ARE the key - so it never pays for one.
  const { f, cx, cy } = roundBox(false);
  const R = PIZZA_R;

  const at = (i: number, r: number) => {
    const a = (i / n) * Math.PI * 2;
    return [cx + Math.sin(a) * r, cy - Math.cos(a) * r] as const;
  };

  // Rings at 25/50/75/100, so a vertex can be read against something.
  const rings = [0.25, 0.5, 0.75, 1]
    .map((t) => {
      const pts = Array.from({ length: n }, (_, i) => at(i, R * t))
        .map(([x, y]) => `${n2(x)},${n2(y)}`)
        .join(" ");
      return `<polygon points="${pts}" fill="none" stroke="var(--grid-line,#dfe3e8)" ` +
        `stroke-width="${t === 1 ? 0.8 : 0.5}"/>`;
    })
    .join("");

  const spokes = Array.from({ length: n }, (_, i) => {
    const [x, y] = at(i, R);
    return `<line x1="${cx}" y1="${cy}" x2="${n2(x)}" y2="${n2(y)}" ` +
      `stroke="var(--grid-line,#dfe3e8)" stroke-width="0.5"/>`;
  }).join("");

  const pts = e.slices.map((s, i) => at(i, (Math.max(0, Math.min(100, s.pct)) / 100) * R));
  const poly =
    `<polygon points="${pts.map(([x, y]) => `${n2(x)},${n2(y)}`).join(" ")}" ` +
    `fill="${SINGLE_COLOR}" fill-opacity="0.18" stroke="${SINGLE_COLOR}" stroke-width="1.4"/>`;

  const marks = e.slices
    .map((s, i) => {
      const [x, y] = pts[i];
      const [lx, ly] = at(i, R + PIZZA_LABEL);
      const anchor = lx > cx + 4 ? "start" : lx < cx - 4 ? "end" : "middle";
      return (
        `<g><title>${esc(s.label)} — ${esc(s.text)} (${s.pct.toFixed(0)}th)</title>` +
        `<circle cx="${n2(x)}" cy="${n2(y)}" r="1.6" fill="${SINGLE_COLOR}"/>` +
        `<text x="${n2(lx)}" y="${n2(ly + 3)}" text-anchor="${anchor}" ` +
        `class="ft-chart__axis">${esc(clip(s.label, 18))}</text>` +
        `<text x="${n2(lx)}" y="${n2(ly + 12)}" text-anchor="${anchor}" ` +
        `class="ft-chart__val">${esc(s.text)}</text></g>`
      );
    })
    .join("");

  return (
    chartSvgTag(f, rings + spokes + poly + marks, `${e.who} by percentile`) +
    `<p class="ft-chart__sub">${esc(e.who)}. Percentile rank against ${e.population} ` +
    `player${e.population === 1 ? "" : "s"}. The enclosed area is not a quantity — ` +
    `it changes with the order of the axes.</p>`
  );
}

export function chartSvg(r: QueryResult, render: string): string {
  // A source with a native visual gets it first: no arrangement of bars is a
  // shot map, and momentum is a shape rather than a set of values.
  if (render === "pitch" && r.extra?.kind === "shots") return shotMapSvg(r.extra);
  if (render === "momentum" && r.extra?.kind === "momentum") return momentumSvg(r.extra);
  if ((render === "spots" || render === "lineup") && r.extra?.kind === "spots") {
    return spotsSvg(r.extra);
  }
  if (render === "heat" && r.extra?.kind === "heat") return heatSvg(r.extra);
  if (render === "zonepitch" && r.extra?.kind === "zones") return zonePitchSvg(r.extra);
  if (render === "goal" && r.extra?.kind === "goal") return goalMouthSvg(r.extra);
  if (render === "pizza" && r.extra?.kind === "pizza") return pizzaSvg(r.extra);
  if (render === "radar" && r.extra?.kind === "pizza") return radarSvg(r.extra);
  if (r.extra?.kind === "freeze") {
    if (render === "freeze") return freezeSvg(r.extra, "freeze");
    if (render === "hull") return freezeSvg(r.extra, "hull");
    if (render === "control") return freezeSvg(r.extra, "control");
  }
  if (r.extra?.kind === "passes") {
    if (render === "network") return passSvg(r.extra, "network");
    if (render === "progressive") return passSvg(r.extra, "progressive");
    if (render === "sonar") return passSvg(r.extra, "sonar");
  }
  if (render === "table") return tableHtml(r);

  // Every branch below falls back rather than throwing, because `render` comes
  // out of a stored document and a hand-edited one can ask for a scatter of a
  // single series. The panel refuses those up front; this is the backstop, and
  // it draws SOMETHING true rather than nothing.
  const many = r.series.length > 1;
  if (render === "line" && r.ordered) return lineSvg(r);
  if (render === "area" && r.ordered) return areaSvg(r, "none");
  if (render === "areaStacked" && r.ordered && many) return areaSvg(r, "stacked");
  if (render === "column") return columnSvg(r, "none");
  if (render === "columnStacked" && many) return columnSvg(r, "stacked");
  if (render === "columnPct" && many) return columnSvg(r, "pct");
  if (render === "pie" && r.rows.length <= CATEGORY_COLORS.length && !many) return pieSvg(r);
  if (render === "doughnut" && r.rows.length <= CATEGORY_COLORS.length && !many) return pieSvg(r, 46);
  if (render === "scatter" && many) return scatterSvg(r);
  if (render === "combo" && many) return comboSvg(r);
  return barSvg(r);
}
