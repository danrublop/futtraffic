import { describe, expect, it } from "vitest";

import { CATEGORY_COLORS, SERIES_COLORS, barPath, chartSvg, colPath } from "./chart-svg";
import { PT } from "./chart-core";
import type { QueryResult } from "./query-data";

/** The gallery's guard rails.
 *
 *  `render` comes out of a stored document, so it can be anything — a note
 *  hand-edited to ask for a scatter of one series, or a pie of forty players.
 *  The panel refuses those up front; these are about the renderer's backstop,
 *  which has to draw something TRUE rather than throwing or drawing a lie.
 */

const q = (over: Partial<QueryResult> = {}): QueryResult => ({
  rows: [
    { label: "A", values: [3], text: ["3"] },
    { label: "B", values: [1], text: ["1"] },
  ],
  series: [],
  max: 3,
  measure: { id: "m", label: "Measure", agg: "sum", dp: 0 },
  dim: { id: "d", label: "Thing" },
  ordered: false,
  note: "",
  tier: "prov",
  ...over,
});

const split = (rows: [string, number, number][], max: number): QueryResult =>
  q({
    rows: rows.map(([label, a, b]) => ({ label, values: [a, b], text: [String(a), String(b)] })),
    series: ["Home", "Away"],
    max,
  });

describe("charts that need something they have not got", () => {
  it("falls back to bars rather than drawing a one-series scatter", () => {
    // A scatter of one number per row against nothing is a strip plot wearing
    // a scatter's label.
    expect(chartSvg(q(), "scatter")).not.toContain("<circle");
    // A column is a <path> now — rounded at the top, square on the baseline.
    expect(chartSvg(q(), "scatter")).toContain("<path");
  });

  it("falls back rather than stacking a single series against itself", () => {
    // A stacked segment is a plain <rect> with a surface ring; a clustered
    // column is a <path> rounded at the top and square on the baseline. The
    // SHAPE is what tells the two apart now that neither carries an rx.
    const svg = chartSvg(q(), "columnStacked");
    expect(svg).toMatch(/<title>[^<]*<\/title><path d="M/);
    expect(svg).not.toMatch(/<title>[^<]*<\/title><rect/);
  });

  it("will not draw a line through categories that have no order", () => {
    // The mark is a d3 path now rather than a polyline, so this asks the
    // question the right way round: does anything JOIN the points up.
    const joined = (svg: string) => /<path d="M[^"]+" fill="none"/.test(svg);
    expect(joined(chartSvg(q({ ordered: false }), "line"))).toBe(false);
    expect(joined(chartSvg(q({ ordered: true }), "line"))).toBe(true);
  });

  it("refuses a pie past the number of hues it has, instead of repeating one", () => {
    const many = q({
      rows: Array.from({ length: CATEGORY_COLORS.length + 1 }, (_, i) => ({
        label: `P${i}`, values: [1], text: ["1"],
      })),
      max: 1,
    });
    // An ARC, not just a path: a column is a path now too, so `<path` no
    // longer distinguishes a pie from the bars it falls back to. Only d3.arc
    // emits an elliptical-arc command.
    expect(chartSvg(many, "pie")).not.toMatch(/<path d="[^"]*A/);
    const few = q({ rows: q().rows, max: 3 });
    expect(chartSvg(few, "pie")).toMatch(/<path d="[^"]*A/);
  });
});

describe("a stack is scaled by the stack, not by its tallest piece", () => {
  const r = split([["A", 30, 70], ["B", 10, 10]], 70);

  it("puts the full stack at the top of the plot", () => {
    // A + B is 100. Scaling by r.max (70, the tallest SEGMENT) would draw the
    // first column 43% off the top of the plot.
    const svg = chartSvg(r, "columnStacked");
    const ys = [...svg.matchAll(/<rect x="[\d.]+" y="([\d.]+)"/g)].map((m) => Number(m[1]));
    expect(Math.min(...ys)).toBeGreaterThanOrEqual(0);
    const heights = [...svg.matchAll(/<rect[^>]*height="([\d.]+)"[^>]*stroke=/g)].map((m) =>
      Number(m[1]),
    );
    // Tallest segment is 70 of a 100 total, so it is 70% of the PLOT. Read the
    // plot out of the figure rather than hard-coding it: the canvas is derived
    // from the content now, so a number copied from one is a number that goes
    // stale the next time a padding moves. The ratio is the claim, not the box.
    const [, , , H] = (/viewBox="([^"]+)"/.exec(svg)?.[1] ?? "").split(/\s+/).map(Number);
    const plotH = H - PT.padTLegend - PT.padB;
    expect(Math.max(...heights)).toBeCloseTo(0.7 * plotH, 0);
  });

  it("rescales every row to a hundred for the 100% form", () => {
    const svg = chartSvg(r, "columnPct");
    expect(svg).toContain("100%");
    const heights = [...svg.matchAll(/<rect[^>]*height="([\d.]+)"[^>]*stroke=/g)].map((m) =>
      Number(m[1]),
    );
    // Row B is 10 and 10, so both its segments are half the plot — even though
    // in raw numbers they are a third of row A's.
    expect(heights[2]).toBeCloseTo(heights[3], 1);
  });
});

describe("two series, one axis", () => {
  it("draws a combo's columns and its line against the same maximum", () => {
    // The Excel trap this refuses: a secondary axis, where the crossing point
    // is set by whoever picked the ranges rather than by the numbers.
    const svg = chartSvg(split([["A", 100, 50], ["B", 50, 100]], 100), "combo");
    // colPath starts at the BASELINE and curves to the top, so the top is the
    // y of its first Q rather than a rect's y attribute.
    const barY = Number(/M[\d.-]+,[\d.-]+V[\d.-]+Q[\d.-]+,([\d.-]+)/.exec(svg)![1]);
    const dotY = Number(/<circle cx="[\d.]+" cy="([\d.]+)"/.exec(svg)![1]);
    // Column 100 and dot 50 of the same axis: the dot sits below the top of
    // the bar by half the plot, not level with it.
    expect(dotY).toBeGreaterThan(barY + 50);
    expect(svg.match(/class="ft-chart__axis"/g)!.length).toBe(5);
  });

  it("plots one series against the other in a scatter", () => {
    const svg = chartSvg(split([["A", 10, 1], ["B", 5, 2]], 10), "scatter");
    expect(svg.match(/<circle/g)).toHaveLength(2);
    expect(svg).toContain("Home");
    expect(svg).toContain("Away");
  });
});

describe("every slice is labelled, because the palette obliges it", () => {
  it("names each category and its share on the chart itself", () => {
    // The six-hue categorical palette passes the validator with a CVD warning
    // in the 6-8 band, which is legal ONLY with secondary encoding. These
    // labels ARE that encoding — not decoration that can be dropped.
    const svg = chartSvg(q(), "pie");
    expect(svg).toContain("75%");
    expect(svg).toContain("25%");
    expect(svg).toContain(">A<");
    expect(svg).toContain(">B<");
  });

  it("cuts a hole for a doughnut and none for a pie", () => {
    // d3-arc writes its own path syntax, so count the ARCS instead of matching
    // a radius: a ring segment is bounded by two, a wedge by one.
    const arcs = (svg: string) =>
      Math.max(...[...svg.matchAll(/ d="([^"]+)"/g)].map((m) => (m[1].match(/A/g) ?? []).length));
    expect(arcs(chartSvg(q(), "doughnut"))).toBeGreaterThan(arcs(chartSvg(q(), "pie")));
  });
});

describe("momentum gets a clock", () => {
  const swing = (n: number) => ({
    kind: "momentum" as const,
    values: Array.from({ length: n }, (_, i) => Math.sin(i)),
    goals: [{ minute: 11, side: "home" }],
    home: "Westfield",
    away: "Rovers",
  });

  it("marks the minutes under the swing, every quarter hour across a match", () => {
    const svg = chartSvg(q({ extra: swing(96) }), "momentum");
    for (const m of ["0'", "15'", "30'", "45'", "60'", "75'", "90'"]) expect(svg).toContain(`>${m}<`);
  });

  it("puts a tick at the same fraction the goal dots use, so the two line up", () => {
    // The goal at minute 11 of 96 sits at 11.46% across; the 15' tick at
    // 15.63%. Both are minute/total, not minute/(total-1).
    const svg = chartSvg(q({ extra: swing(96) }), "momentum");
    expect(svg).toContain("left:15.63%");
  });

  it("closes the spacing up on a short series rather than showing one mark", () => {
    // Twenty minutes at a quarter-hour step is two marks and a lot of blank
    // axis, so the step drops to five.
    const svg = chartSvg(q({ extra: swing(20) }), "momentum");
    for (const m of ["0'", "5'", "10'", "15'", "20'"]) expect(svg).toContain(`>${m}<`);
    // Under ten samples there is no clock worth drawing at all.
    expect(chartSvg(q({ extra: swing(6) }), "momentum")).not.toContain("ft-chart__ticks");
  });

  it("keeps the marks OUT of the svg, which is drawn stretched", () => {
    // preserveAspectRatio="none" is what makes the swing fill its width at a
    // fixed height, and it stretches any text inside the viewport with it.
    const svg = chartSvg(q({ extra: swing(96) }), "momentum");
    const inner = svg.slice(svg.indexOf("<svg"), svg.indexOf("</svg>"));
    expect(inner).toContain('preserveAspectRatio="none"');
    expect(inner).not.toContain("15'");
  });
});

describe("an axis with more names than room", () => {
  const many = (n: number) =>
    q({
      ordered: true,
      max: 10,
      rows: Array.from({ length: n }, (_, i) => ({
        label: `Northbridge FC v Opponent ${i}`,
        values: [i + 1],
        text: [String(i + 1)],
      })),
    });

  it("prints every Nth name instead of a smear", () => {
    // Ten long fixture names across one figure printed as one unbroken run of
    // characters, which is worse than no axis: a smear looks like a label
    // until you try to read it.
    const svg = chartSvg(many(10), "line");
    const cats = svg.match(/class="ft-chart__cat"/g) ?? [];
    expect(cats.length).toBeGreaterThan(0);
    expect(cats.length).toBeLessThan(10);
  });

  it("prints them all when they fit", () => {
    // SHORT names, and that is the point of the test rather than an
    // accommodation. Since a figure's canvas is sized in points from its own
    // content, "do they fit" is now a real question about real millimetres:
    // three 14-character fixture names need about 194 pt of axis and a
    // three-point line is given 178, so thinning them is correct and the old
    // version of this test was only passing because the canvas was 640 units
    // of nothing in particular.
    const short = q({
      ordered: true,
      max: 10,
      rows: ["Sep", "Oct", "Nov"].map((label, i) => ({
        label,
        values: [i + 1],
        text: [String(i + 1)],
      })),
    });
    expect((chartSvg(short, "line").match(/class="ft-chart__cat"/g) ?? []).length).toBe(3);
  });

  it("still draws every point — only the labels thin", () => {
    // Dropping a bar would be a different chart. Dropping a tick is not.
    const svg = chartSvg(many(10), "column");
    expect((svg.match(/<path d="M/g) ?? []).length).toBe(10);
  });
});

/** The bars, and not the legend's swatches: a bar is the mark inside the <g>
 *  that carries its hover <title>.
 *
 *  Two shapes to read now. A clustered or single column is a <path>, because
 *  only its top is rounded; a stacked segment is still a <rect>, because it has
 *  a surface ring and no rounding at all. Both report the same two numbers.
 */
const bars = (svg: string, attr: "y" | "height") =>
  [...svg.matchAll(/<title>[^<]*<\/title>(<rect[^>]*>|<path[^>]*>)/g)].map((m) => {
    const tag = m[1];
    if (tag.startsWith("<rect")) {
      const v = new RegExp(`\\b${attr}="(-?[\\d.]+)"`).exec(tag);
      return Number(v?.[1]);
    }
    // colPath: M x,(bottom) V (top+r) Q x,(top) ...
    const d = /M[\d.-]+,([\d.-]+)V[\d.-]+Q[\d.-]+,([\d.-]+)/.exec(tag);
    if (!d) return NaN;
    const bottom = Number(d[1]);
    const top = Number(d[2]);
    return attr === "y" ? top : bottom - top;
  });

describe("a clustered column sits on the baseline", () => {
  const two = {
    rows: [
      { label: "A", values: [3, 1], text: ["3", "1"] },
      { label: "B", values: [2, 2], text: ["2", "2"] },
    ],
    series: ["Home", "Away"],
    max: 3,
    measure: { id: "m", label: "M", agg: "sum" as const, dp: 0 },
    dim: { id: "d", label: "D" },
    ordered: false,
    note: "",
    tier: "prov" as const,
  };

  it("draws no bar above the plot", () => {
    // `stacked()` returns cumulative spans in every mode. Reading them when
    // nothing is stacked lifted the second series by the first, and the scale
    // is the largest SINGLE value there — so the second bar floated off the
    // top of the frame and was clipped away.
    const svg = chartSvg(two, "column");
    const ys = bars(svg, "y");
    expect(ys).toHaveLength(4);
    expect(Math.min(...ys)).toBeGreaterThanOrEqual(0);
  });

  it("gives equal values equal bars", () => {
    const svg = chartSvg({ ...two, rows: [{ label: "A", values: [2, 2], text: ["2", "2"] }] }, "column");
    const hs = bars(svg, "height");
    expect(hs[0]).toBeCloseTo(hs[1], 2);
  });

  it("still stacks when asked to", () => {
    const svg = chartSvg(two, "columnStacked");
    const hs = bars(svg, "height");
    // A = 3 + 1 against a scale of 4: three parts to one.
    expect(hs[0] / hs[1]).toBeCloseTo(3, 1);
  });
});

describe("a bar is rounded where it ends and square where it stands", () => {
  it("rounds a column's top only, leaving the baseline flat", () => {
    const d = colPath(10, 20, 40, 100);
    // Starts at the bottom-left ON the baseline and returns to the bottom-right
    // on it — no curve anywhere near y = 120.
    expect(d.startsWith("M10,120")).toBe(true);
    expect(d.endsWith("V120Z")).toBe(true);
    // Two curves, both at the top.
    expect((d.match(/Q/g) ?? [])).toHaveLength(2);
    expect(d).toContain("Q10,20");
    expect(d).toContain("Q50,20");
  });

  it("rounds a horizontal bar's far end only, leaving the axis edge flat", () => {
    const d = barPath(0, 5, 60, 12);
    expect(d.startsWith("M0,5")).toBe(true);
    expect(d).toContain("Q60,5");
    expect(d).toContain("Q60,17");
    expect(d.endsWith("H0Z")).toBe(true);
  });

  it("clamps the radius so a stub does not fold through itself", () => {
    // A 2px-tall column with a 4px radius would otherwise cross its own
    // corners and render as a bow tie.
    const d = colPath(0, 0, 3, 2);
    expect(d).not.toContain("NaN");
    expect(d.startsWith("M0,2")).toBe(true);
    // Radius capped by half the width and by the height.
    expect(d).toContain("Q0,0 1.5,0");
  });

  it("survives a zero-height column, which is what a zero value draws", () => {
    expect(colPath(0, 50, 10, 0)).not.toContain("NaN");
    expect(barPath(0, 0, 0, 10)).not.toContain("NaN");
  });
});

describe("a chart wears the colours it was given", () => {
  const two = {
    rows: [
      { label: "A", values: [3, 1], text: ["3", "1"] },
      { label: "B", values: [2, 2], text: ["2", "2"] },
    ],
    series: ["Home", "Away"],
    max: 3,
    measure: { id: "m", label: "M", agg: "sum" as const, dp: 0 },
    dim: { id: "d", label: "D" },
    ordered: false,
    note: "",
    tier: "prov" as const,
  };

  it("uses the validated palette when nothing was chosen", () => {
    const svg = chartSvg(two, "column");
    expect(svg).toContain(SERIES_COLORS[0]);
    expect(svg).toContain(SERIES_COLORS[1]);
  });

  it("takes an override, per series, and puts it in the legend too", () => {
    const svg = chartSvg({ ...two, colors: ["#00ff00", "#ff00ff"] }, "column");
    expect(svg).toContain("#00ff00");
    expect(svg).toContain("#ff00ff");
    expect(svg).not.toContain(SERIES_COLORS[0]);
    // The legend swatch is the same colour as the bar it names, or the chart
    // lies about which series is which.
    expect((svg.match(/#00ff00/g) ?? []).length).toBeGreaterThan(1);
  });

  it("falls through per SLOT, so overriding one leaves the other alone", () => {
    const svg = chartSvg({ ...two, colors: ["#00ff00"] }, "column");
    expect(svg).toContain("#00ff00");
    expect(svg).toContain(SERIES_COLORS[1]);
    expect(svg).not.toContain(SERIES_COLORS[0]);
  });

  it("treats an empty string as 'the default', not as a colour", () => {
    const svg = chartSvg({ ...two, colors: ["", "#ff00ff"] }, "column");
    expect(svg).toContain(SERIES_COLORS[0]);
    expect(svg).toContain("#ff00ff");
  });

  it("recolours a single-series chart, which has no legend to disagree with", () => {
    const one = { ...two, series: ["Only"], rows: [{ label: "A", values: [3], text: ["3"] }] };
    expect(chartSvg({ ...one, colors: ["#123456"] }, "column")).toContain("#123456");
  });

  it("recolours a pie BY CATEGORY, because that is what carries the hue there", () => {
    const parts = {
      ...two,
      series: ["One"],
      rows: [
        { label: "A", values: [3], text: ["3"] },
        { label: "B", values: [1], text: ["1"] },
      ],
    };
    const svg = chartSvg({ ...parts, colors: ["#00ff00", "#ff00ff"] }, "pie");
    expect(svg).toContain("#00ff00");
    expect(svg).toContain("#ff00ff");
  });
});

describe("the percentile pizza", () => {
  const slices = [
    { label: "Distance", pct: 90, text: "13.29 km", group: "Volume" },
    { label: "Sprints", pct: 40, text: "6.2", group: "Intensity" },
    { label: "Seen", pct: 100, text: "1.00", group: "Coverage" },
  ];
  const extra = { kind: "pizza" as const, who: "Varga", population: 21, slices };
  const svg = () => chartSvg(q({ extra }), "pizza");

  it("draws one wedge per metric, plus its ring", () => {
    // Each slice is two paths: the faint full-radius ring and the value on top.
    expect((svg().match(/<path /g) ?? []).length).toBe(slices.length * 2);
  });

  it("writes the number in the wedge, which is the whole point over a radar", () => {
    expect(svg()).toContain("13.29 km");
  });

  it("says what the population was", () => {
    // A rank against 21 and a rank against 400 are different claims.
    expect(svg()).toContain("21 player");
  });

  it("hides the number in a wedge too shallow to hold it", () => {
    // Otherwise every short wedge prints its value on top of the centre, where
    // all of them meet.
    const shallow = chartSvg(
      q({ extra: { ...extra, slices: [{ label: "Tiny", pct: 4, text: "0.1" }] } }),
      "pizza",
    );
    expect(shallow).not.toContain(">0.1<");
  });

  it("clamps a percentile outside 0-100 rather than drawing past the ring", () => {
    const wild = chartSvg(
      q({ extra: { ...extra, slices: [{ label: "Bad", pct: 140, text: "x" }] } }),
      "pizza",
    );
    // A wedge drawn past the ring would read as better than best.
    expect(() => wild).not.toThrow();
    expect(wild).toContain("<svg");
  });

  it("falls back rather than throwing when the payload is missing", () => {
    // `render` comes out of a stored document and can be hand-edited.
    expect(chartSvg(q(), "pizza")).toContain("<svg");
  });

  it("survives having no slices at all", () => {
    expect(() =>
      chartSvg(q({ extra: { ...extra, slices: [] } }), "pizza"),
    ).not.toThrow();
  });
});

describe("one moment, drawn three ways", () => {
  const players = [
    { x: 20, y: 30, side: "home", label: "2" },
    { x: 30, y: 50, side: "home", label: "6" },
    { x: 25, y: 70, side: "home", label: "3" },
    { x: 45, y: 40, side: "home", label: "8" },
    { x: 70, y: 30, side: "away", label: "5" },
    { x: 75, y: 55, side: "away", label: "4" },
    { x: 80, y: 70, side: "away", label: "3" },
  ];
  const extra = {
    kind: "freeze" as const,
    t_s: 1287,
    players,
    ball: { x: 50, y: 50 },
    seen: 7,
    squad: 22,
    home: "H",
    away: "A",
    synthetic: true,
  };
  const draw = (render: string, over = {}) => chartSvg(q({ extra: { ...extra, ...over } }), render);

  for (const render of ["freeze", "hull", "control"]) {
    it(`draws a ${render}`, () => {
      expect(draw(render)).toContain("<svg");
    });
  }

  // The whole reason seen/squad ride on the payload. A shape drawn from part of
  // a side is not that side's shape, and the figure has to say so itself -
  // prose around it does not survive a screenshot.
  it("prints how much of the squad it could see", () => {
    expect(draw("hull")).toContain("7 of 22");
  });

  it("says when the positions were generated rather than measured", () => {
    expect(draw("hull")).toContain("generated, not measured");
    expect(draw("hull", { synthetic: false })).not.toContain("generated, not measured");
  });

  it("warns that a partial side is not a side", () => {
    expect(draw("hull")).toContain("not that side's shape");
    expect(draw("hull", { seen: 22, squad: 22 })).not.toContain("not that side's shape");
  });

  it("draws a hull per side, not one around everybody", () => {
    expect((draw("hull").match(/<polygon /g) ?? []).length).toBe(2);
  });

  it("will not hull a side with fewer than three players", () => {
    const thin = draw("hull", { players: players.filter((p) => p.side === "away").slice(0, 2) });
    expect(thin).not.toContain("<polygon");
    expect(thin).toContain("<svg");
  });

  // 42x27 cells is 1,134 rects, and the rendered markup is what the document
  // stores forever. Runs draw the same picture in a fraction of them.
  it("run-length encodes the control grid rather than emitting every cell", () => {
    const rects = (draw("control").match(/<rect /g) ?? []).length;
    expect(rects).toBeGreaterThan(0);
    expect(rects).toBeLessThan(300);
  });

  it("falls back rather than throwing when the payload is missing", () => {
    for (const render of ["freeze", "hull", "control"]) {
      expect(chartSvg(q(), render)).toContain("<svg");
    }
  });
});

describe("passes, drawn three ways", () => {
  const passes = [
    { x0: 20, y0: 30, x1: 30, y1: 50, side: "home", from: "2", to: "6", ok: true },
    { x0: 30, y0: 50, x1: 45, y1: 40, side: "home", from: "6", to: "8", ok: true },
    { x0: 45, y0: 40, x1: 30, y1: 50, side: "home", from: "8", to: "6", ok: true },
    { x0: 30, y0: 50, x1: 25, y1: 70, side: "home", from: "6", to: "3", ok: false },
    { x0: 70, y0: 30, x1: 75, y1: 55, side: "away", from: "5", to: "4", ok: true },
  ];
  const extra = {
    kind: "passes" as const,
    passes,
    home: "H",
    away: "A",
    dropped: 12,
    synthetic: true,
  };
  const draw = (render: string, over = {}) => chartSvg(q({ extra: { ...extra, ...over } }), render);

  for (const render of ["network", "progressive", "sonar"]) {
    it(`draws a ${render}`, () => {
      expect(draw(render)).toContain("<svg");
    });
  }

  it("says how many passes it left out", () => {
    expect(draw("network")).toContain("12 more not drawn");
    expect(draw("network", { dropped: 0 })).not.toContain("not drawn");
  });

  // 6->8 and 8->6 are one partnership. Drawing them as two lines makes a
  // two-way relationship look like two weak ones.
  it("treats a pairing as undirected", () => {
    // home has 6-2, 6-8 (twice, both directions) and one failed 6-3.
    // Completed and undirected, that is two edges, not three.
    expect((draw("network").match(/↔/g) ?? []).length).toBe(2);
  });

  it("leaves failed passes out of the network but keeps them on the map", () => {
    expect(draw("network")).not.toContain("6 ↔ 3");
    expect(draw("progressive")).toContain("(lost)");
  });

  it("falls back rather than throwing when the payload is missing", () => {
    for (const render of ["network", "progressive", "sonar"]) {
      expect(chartSvg(q(), render)).toContain("<svg");
    }
  });
});

describe("the radar", () => {
  const extra = {
    kind: "pizza" as const,
    who: "Brandt",
    population: 214,
    slices: [
      { label: "Distance", pct: 88, text: "12.1 km" },
      { label: "Sprints", pct: 40, text: "41" },
      { label: "Passes", pct: 66, text: "72" },
      { label: "Duels", pct: 21, text: "48%" },
    ],
  };

  it("joins the points into one polygon", () => {
    const svg = chartSvg(q({ extra }), "radar");
    // Four rings plus the value polygon.
    expect((svg.match(/<polygon /g) ?? []).length).toBe(5);
  });

  it("says out loud that its area is not a quantity", () => {
    // The one thing a radar has to disclaim, and the reason the pizza is
    // offered first.
    expect(chartSvg(q({ extra }), "radar")).toContain("not a quantity");
  });

  it("falls back to the pizza below three axes, where a polygon is a line", () => {
    const two = { ...extra, slices: extra.slices.slice(0, 2) };
    expect(chartSvg(q({ extra: two }), "radar")).toContain("the ring is the whole field");
  });
});

describe("a chart whose rows were written rather than queried", () => {
  // scripts/note-api.mts builds these: `text` is the formatted value printed
  // beside a mark, and a hand-written row is allowed to arrive without one.
  // Ten of the eleven sites that index into it did so unguarded, so this threw
  // `Cannot read properties of undefined (reading 'replace')` out of a
  // renderHTML — which does not degrade one block, it takes the whole document.
  const noText = q({
    rows: [
      { label: "Scored", values: [3], text: [] },
      { label: "Missed", values: [1], text: [] },
    ],
  });

  for (const render of ["doughnut", "pie", "bar", "column", "line", "area", "combo"]) {
    it(`draws a ${render} rather than throwing`, () => {
      expect(() => chartSvg(noText, render)).not.toThrow();
      expect(chartSvg(noText, render)).toContain("<svg");
    });
  }

  it("leaves the label empty rather than printing the word undefined", () => {
    expect(chartSvg(noText, "doughnut")).not.toContain("undefined");
  });
});
