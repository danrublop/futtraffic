/** WRITING A NOTE FROM A SCRIPT — the agent's half of the note library.
 *
 *  The HTTP API was already complete for CRUD: GET/PUT/DELETE /api/notes/<path>,
 *  POST /api/notes/folder, POST /api/notes/move. What it could not do was make a
 *  note that LOOKS like one. The notepad's blocks store a data attribute AND the
 *  rendered markup beside it — an inline SVG board, a drawn chart — and only the
 *  editor's serializer produces that pair. An agent PUTting bare
 *  `<div data-pitch="...">` writes a file that renders once a human opens it in
 *  the editor and is blank everywhere else: the gallery card, the print export,
 *  the file mailed to a player.
 *
 *  So this is not a second API. It is the editor's own serializer, run headless
 *  against the same extensions the editor registers, so a note written here is
 *  byte-for-byte the note a coach would have got by clicking through the menus.
 *
 *      import { writeNote, h, p, pitch, chart, table } from "./note-api.mts";
 *      await writeNote("Scouting/Argentina.html", [h(1, "Argentina"), ...]);
 *
 *  Run:  npx vite-node --config vitest.config.mts scripts/<your-script>.mts
 *
 *  IMPORT THIS FIRST in a script that also wants notepad modules: the DOM shim
 *  below has to exist before any block module is loaded, and ES imports evaluate
 *  in source order.
 */
import { writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import { JSDOM } from "jsdom";
import type { Browser as PwBrowser } from "playwright";

// The blocks' renderHTML builds real elements, so a DOM has to exist before any
// of those modules are imported. Same shim as scripts/make-demo-note.mts.
const dom = new JSDOM("<!doctype html><html><body></body></html>");
const g = globalThis as unknown as Record<string, unknown>;
g.window = dom.window;
g.document = dom.window.document;
g.navigator ??= dom.window.navigator;
g.HTMLElement = dom.window.HTMLElement;
g.Element = dom.window.Element;
g.Node = dom.window.Node;

const { Editor } = await import("@tiptap/core");
const Document = (await import("@tiptap/extension-document")).default;
const Paragraph = (await import("@tiptap/extension-paragraph")).default;
const Text = (await import("@tiptap/extension-text")).default;
const Heading = (await import("@tiptap/extension-heading")).default;
const Bold = (await import("@tiptap/extension-bold")).default;
const Italic = (await import("@tiptap/extension-italic")).default;
const BulletList = (await import("@tiptap/extension-bullet-list")).default;
const OrderedList = (await import("@tiptap/extension-ordered-list")).default;
const ListItem = (await import("@tiptap/extension-list-item")).default;
const Blockquote = (await import("@tiptap/extension-blockquote")).default;
const HorizontalRule = (await import("@tiptap/extension-horizontal-rule")).default;

const grid = await import("@/components/notepad/table-grid");
// The print export, reused rather than reimplemented: the same wrapper the
// browser's print button builds, so a PDF and a printed page are one file.
const { documentHtml, documentRules } = await import("@/lib/doc-export");
const { docOrient } = await import("@/components/notepad/page-nodes");
const { fmt } = await import("@/components/notepad/query-data");
/** Re-exported so a figure a script assembles by hand rounds its labels the
 *  same way a queried one does, rather than keeping a second rounding rule. */
export { fmt };
const { PitchNode } = await import("@/components/notepad/pitch-node");
const { FigureNode } = await import("@/components/notepad/figure-node");
const { QueryNode } = await import("@/components/notepad/query-node");
const { StatNode } = await import("@/components/notepad/stat-node");
const { FaceNode } = await import("@/components/notepad/face-node");
const { PageBreakNode, PageNode, PageSetupNode } = await import("@/components/notepad/page-nodes");

export const { tableContent, tableFromRows, lockedTableFromRows, layoutRow } = grid;
export type Block = Record<string, unknown>;

export const API = process.env.FUTTRAFFIC_API ?? "http://127.0.0.1:8000";

// ------------------------------------------------------------------- blocks
// Thin builders over the ProseMirror JSON the panels already emit. They exist
// so a report script reads as a report and not as a tree literal.

export const h = (level: number, text: string): Block => ({
  type: "heading",
  attrs: { level },
  content: [{ type: "text", text }],
});

/** A paragraph. Empty text is an empty paragraph, not a text node holding "" —
 *  ProseMirror rejects the latter. */
export const p = (text = ""): Block =>
  text ? { type: "paragraph", content: [{ type: "text", text }] } : { type: "paragraph" };

export const ul = (items: string[]): Block => ({
  type: "bulletList",
  content: items.map((t) => ({ type: "listItem", content: [p(t)] })),
});

export const quote = (text: string): Block => ({ type: "blockquote", content: [p(text)] });
export const hr = (): Block => ({ type: "horizontalRule" });

/** PAPER, AND ITS MARGIN.
 *
 *  Belongs at the top of the document and there is only ever one. Landscape is
 *  the default, so a note only needs this to ask for portrait or to set a
 *  margin.
 *
 *  `margin` is millimetres on all four edges, clamped to 5-40: under about 5 mm
 *  most desktop printers cannot reach the edge and crop without saying so, and
 *  over 40 there is no measure left on A4 to put a figure in. Omit it for the
 *  paper's default - 12 mm landscape, 14 mm portrait, which is what every
 *  document printed with before the margin was authorable. */
export const pageSetup = (
  orient: "portrait" | "landscape",
  margin?: number,
): Block => ({
  type: "pageSetup",
  attrs: { orient, ...(margin ? { margin } : {}) },
});

/** Start the next block on a new sheet.
 *
 *  DO NOT PUT ONE BETWEEN TWO `page()` BLOCKS. A sheet already carries
 *  `break-before: page` in the print stylesheet, so a pageBreak between two of
 *  them is a SECOND break and prints a blank sheet between them - measured:
 *  seven of these turned a ten-page gallery into eleven pages with one of them
 *  entirely empty. Use it to break a FLOWING document, which is what it is for;
 *  a document built out of `page()` needs none at all. */
export const pageBreak = (): Block => ({ type: "pageBreak" });

/** A SHEET: a box the size of the printable page, holding what you put in it.
 *
 *  Not a break in a flow - a container. What fits prints on that sheet one to
 *  one; what does not hangs off the bottom edge where you can see it. See
 *  page-nodes.ts for why pagination here is authored rather than computed. */
export const page = (blocks: Block[]): Block => ({
  type: "page",
  content: blocks.length ? blocks : [{ type: "paragraph" }],
});

/** A player's face beside his name. `src` is a data URI baked in at build time
 *  - the document is read offline - and empty draws the silhouette, which is
 *  the same shape, so the page does not move when the photographs arrive. */
export const face = (opts: {
  name: string;
  sub?: string;
  /** The roster id, so a photo uploaded later can be found and baked in. */
  player?: string;
  src?: string;
}): Block => ({
  type: "face",
  attrs: {
    name: opts.name,
    sub: opts.sub ?? "",
    player: opts.player ?? "",
    src: opts.src ?? "",
  },
});

/** Who took each photograph and under what licence, keyed by player id.
 *
 *  Empty for photos a club uploaded itself, which are the club's own. The
 *  seeded ones are Wikimedia Commons and mostly CC-BY-SA, and the condition of
 *  using those is that the credit is printed with the picture - so a document
 *  that prints faces has to be able to print the line. */
export async function photoCredits(): Promise<Record<string, {
  title: string; artist: string; licence: string; page: string; file: string;
}>> {
  const res = await fetch(`${API}/api/players/photo-credits`);
  return res.ok ? await res.json() : {};
}

/** The headshot the club has for this player, as a data URI, or "" - which the
 *  face block draws as a silhouette. One fetch per player; the caller decides
 *  whether to pay for it. */
export async function headshot(playerId: string): Promise<string> {
  const res = await fetch(`${API}/api/players/${encodeURIComponent(playerId)}/photo`);
  if (!res.ok) return "";
  const type = res.headers.get("content-type") ?? "image/png";
  const b64 = Buffer.from(await res.arrayBuffer()).toString("base64");
  return `data:${type};base64,${b64}`;
}

/** A tactics board. Everything but the magnets has a default, because a board
 *  is usually a full green pitch with discs on it and nothing else. */
export const pitch = (opts: {
  magnets: { x: number; y: number; label: string; side: "home" | "away" | "ball" }[];
  shapes?: { kind: string; pts: number[]; color: string; label?: string }[];
  caption?: string;
  note?: string;
  /** Share of the column, 25-100. Below 100 pair it with `wrap` or the space
   *  beside it is simply empty. */
  width?: number;
  /** "left" or "right" floats the board and runs the prose down the other
   *  side - Google Docs calls it Wrap text. Omitted is Break text: the board
   *  owns its band of the page. Ignored inside a layout cell, where the cell
   *  is already the column. */
  wrap?: "left" | "right";
  view?: string;
  orient?: string;
  lines?: string;
  grass?: string;
  stripes?: boolean;
  tier?: string;
}): Block => ({
  type: "pitch",
  attrs: {
    magnets: opts.magnets,
    shapes: opts.shapes ?? [],
    view: opts.view ?? "full",
    orient: opts.orient ?? "h",
    lines: opts.lines ?? "standard",
    grass: opts.grass ?? "green",
    stripes: opts.stripes ?? false,
    caption: opts.caption ?? "",
    note: opts.note ?? "",
    tier: opts.tier ?? "",
    width: opts.width ?? 100,
    wrap: opts.wrap ?? "none",
  },
});

/** A chart, drawn from numbers this script already has.
 *
 *  `custom: true` because these rows were written rather than queried: the
 *  right-click menu must not offer a Refresh that would throw them away. */
export const chart = (opts: {
  title?: string;
  render?: string;
  rows: { label: string; values: number[]; text?: string[] }[];
  series?: string[];
  max?: number;
  unit?: string;
  dp?: number;
  label?: string;
  dimLabel?: string;
  ordered?: boolean;
  colors?: string[];
  width?: number;
  note?: string;
  tier?: string;
  /** A key of the stylesheet's chart themes. "report" is the scouting-report
   *  look: red bars, a hollow second series, a dashed grid. */
  theme?: string;
}): Block => ({
  type: "query",
  attrs: {
    title: opts.title ?? "",
    source: "manual",
    dim: "label",
    measure: "value",
    split: (opts.series ?? []).length > 1 ? "series" : "",
    render: opts.render ?? "column",
    limit: opts.rows.length,
    team: "",
    keep: [],
    sort: "",
    matchIds: [],
    colors: opts.colors ?? [],
    width: opts.width ?? 100,
    custom: true,
    // `text` is what the figure PRINTS beside a mark — the value label on a
    // bar, the number in a slice's title. A queried chart gets it from fmt();
    // one built here used to get an empty array, which read as a blank label
    // where there should be a number. Same formatter, so a hand-written chart
    // and a queried one round to the same string.
    rows: opts.rows.map((r) => ({
      label: r.label,
      values: r.values,
      text: r.text ?? r.values.map((v) => fmt(v, { dp: opts.dp ?? 0, unit: opts.unit ?? "" })),
    })),
    series: opts.series ?? [""],
    max: opts.max ?? Math.max(0, ...opts.rows.flatMap((r) => r.values)),
    extra: null,
    note: opts.note ?? "",
    tier: opts.tier ?? "",
    theme: opts.theme ?? "",
    unit: opts.unit ?? "",
    dp: opts.dp ?? 0,
    label: opts.label ?? "",
    dimLabel: opts.dimLabel ?? "",
    ordered: !!opts.ordered,
  },
});

/** WHERE THE SHOTS FINISHED IN THE GOAL, seen from behind the ball.
 *
 *  0-100 across the frame (0 is the left post) and 0-100 up it (0 is the
 *  ground). Outside that is a miss and is drawn where it went, which is the
 *  point of the figure: a taker who misses always misses to the same side. */
export const goalMouth = (opts: {
  title?: string;
  who?: string;
  shots: { x: number; y: number; scored: boolean; label?: string }[];
  note?: string;
  width?: number;
}): Block => ({
  type: "query",
  attrs: {
    title: opts.title ?? "",
    source: "manual",
    dim: "shot",
    measure: "count",
    split: "",
    render: "goal",
    limit: opts.shots.length,
    team: "",
    keep: [],
    sort: "",
    matchIds: [],
    colors: [],
    width: opts.width ?? 100,
    custom: true,
    // A native figure draws from `extra`; the rows are what a reader gets if
    // the picture cannot be drawn, so they say the same thing in words.
    rows: [
      { label: "Scored", values: [opts.shots.filter((sh) => sh.scored).length], text: [] },
      { label: "Missed", values: [opts.shots.filter((sh) => !sh.scored).length], text: [] },
    ],
    series: [""],
    max: opts.shots.length,
    extra: {
      kind: "goal",
      who: opts.who ?? "",
      shots: opts.shots.map((sh) => ({ ...sh, label: sh.label ?? "" })),
    },
    note: opts.note ?? "",
    tier: "",
    theme: "",
    unit: "",
    dp: 0,
    label: "Penalties",
    dimLabel: "Shot",
    ordered: false,
  },
});

/** ONE PLAYER AGAINST A POPULATION, ONE WEDGE PER METRIC.
 *
 *  The coach-facing comparison figure, and the reason it is this and not a
 *  radar: a radar joins its points, and the shape that draws has an AREA which
 *  reads as a quantity and is not one - reorder the axes and the same player
 *  gets a different area. A pizza joins nothing, and the number is written in
 *  the wedge instead of being read off a spoke.
 *
 *  `pct` is a PERCENTILE - a rank inside `population` - not a share and not a
 *  normalised value. `population` is printed on the figure because a rank
 *  against ten team-mates and a rank against a season are different claims
 *  wearing the same number, so it is required rather than defaulted.
 *
 *  `tier` matters here as much as anywhere: a percentile of a provisional
 *  number is still provisional, and ranking does not launder it.
 */
export const pizza = (opts: {
  who: string;
  population: number;
  slices: { label: string; pct: number; text: string; group?: string }[];
  title?: string;
  width?: number;
  note?: string;
  tier?: string;
}): Block => ({
  type: "query",
  attrs: {
    title: opts.title ?? "",
    source: "manual",
    dim: "metric",
    measure: "percentile",
    split: "",
    render: "pizza",
    limit: opts.slices.length,
    team: "",
    keep: [],
    sort: "",
    matchIds: [],
    colors: [],
    width: opts.width ?? 100,
    custom: true,
    // The rows say the same thing in words, for the reader who cannot see the
    // picture - the same bargain goalMouth() strikes.
    rows: opts.slices.map((s) => ({
      label: s.label,
      values: [s.pct],
      text: [s.text],
    })),
    series: [""],
    max: 100,
    extra: {
      kind: "pizza",
      who: opts.who,
      population: opts.population,
      slices: opts.slices.map((s) => ({
        label: s.label,
        pct: s.pct,
        text: s.text,
        ...(s.group ? { group: s.group } : {}),
      })),
    },
    note: opts.note ?? "",
    tier: opts.tier ?? "",
    theme: "",
    unit: "",
    dp: 0,
    label: "Percentile",
    dimLabel: "Metric",
    ordered: false,
  },
});

/** Shared spine for the figures that carry an `extra` payload and nothing else.
 *  Every one of them was repeating twenty-odd attribute defaults; the block is
 *  the payload plus a render id, and the rest is the query node's furniture. */
const figure = (
  render: string,
  extra: Record<string, unknown>,
  rows: { label: string; values: number[]; text?: string[] }[],
  opts: { title?: string; width?: number; note?: string; tier?: string; label?: string; dimLabel?: string } = {},
): Block => ({
  type: "query",
  attrs: {
    title: opts.title ?? "",
    source: "manual",
    dim: opts.dimLabel ?? "",
    measure: opts.label ?? "",
    split: "",
    render,
    limit: rows.length,
    team: "",
    keep: [],
    sort: "",
    matchIds: [],
    colors: [],
    width: opts.width ?? 100,
    custom: true,
    rows: rows.map((r) => ({ label: r.label, values: r.values, text: r.text ?? [] })),
    series: [""],
    max: Math.max(0, ...rows.flatMap((r) => r.values)),
    extra,
    note: opts.note ?? "",
    tier: opts.tier ?? "",
    theme: "",
    unit: "",
    dp: 0,
    label: opts.label ?? "",
    dimLabel: opts.dimLabel ?? "",
    ordered: false,
  },
});

/** ONE INSTANT, THREE WAYS. `render` picks which:
 *
 *    freeze   every visible player, as discs
 *    hull     the convex hull each side makes - compactness and width at a
 *             glance, and the figure stats.py computes the AREA of
 *    control   which side is nearest, over a grid
 *
 *  The same payload draws all three, so a note showing a moment two ways stores
 *  it once. `seen` against `squad` prints on every one of them: a shape drawn
 *  from 13 of 22 is not that side's shape, and the figure has to say so. */
export const freeze = (opts: {
  players: { x: number; y: number; side: "home" | "away"; label: string }[];
  render?: "freeze" | "hull" | "control";
  t_s?: number;
  ball?: { x: number; y: number };
  squad?: number;
  home?: string;
  away?: string;
  synthetic?: boolean;
  title?: string;
  width?: number;
  note?: string;
  tier?: string;
}): Block =>
  figure(
    opts.render ?? "freeze",
    {
      kind: "freeze",
      t_s: opts.t_s ?? 0,
      players: opts.players,
      ...(opts.ball ? { ball: opts.ball } : {}),
      seen: opts.players.length,
      squad: opts.squad ?? 22,
      home: opts.home ?? "Home",
      away: opts.away ?? "Away",
      synthetic: opts.synthetic ?? false,
    },
    [
      { label: "Home", values: [opts.players.filter((p) => p.side === "home").length] },
      { label: "Away", values: [opts.players.filter((p) => p.side === "away").length] },
    ],
    { ...opts, label: "Players", dimLabel: "Side" },
  );

/** PASSES, THREE WAYS: `network`, `progressive`, `sonar`. One payload again.
 *
 *  Cap what you pass in. What a block stores is what every copy of the document
 *  carries forever, and the heat map's 423 KB is the standing warning. */
export const passes = (opts: {
  passes: {
    x0: number; y0: number; x1: number; y1: number;
    side: "home" | "away"; from: string; to: string; ok: boolean;
  }[];
  render?: "network" | "progressive" | "sonar";
  home?: string;
  away?: string;
  dropped?: number;
  synthetic?: boolean;
  title?: string;
  width?: number;
  note?: string;
  tier?: string;
}): Block =>
  figure(
    opts.render ?? "network",
    {
      kind: "passes",
      passes: opts.passes,
      home: opts.home ?? "Home",
      away: opts.away ?? "Away",
      dropped: opts.dropped ?? 0,
      synthetic: opts.synthetic ?? false,
    },
    [
      { label: "Completed", values: [opts.passes.filter((p) => p.ok).length] },
      { label: "Lost", values: [opts.passes.filter((p) => !p.ok).length] },
    ],
    { ...opts, label: "Passes", dimLabel: "Outcome" },
  );

/** The pizza's numbers, joined into a polygon. Analyst-facing - see radarSvg
 *  for why this is offered second and not first. */
export const radar = (opts: Parameters<typeof pizza>[0]): Block => {
  const b = pizza(opts);
  (b.attrs as Record<string, unknown>).render = "radar";
  return b;
};

/** A table of imported numbers, held still — the shape every figure in a
 *  scouting report has. `bars` adds the share column the comparison pages use. */
export const table = (opts: {
  head: string[];
  rows: (string | number)[][];
  bars?: (number | null)[];
  /** The LOOK, as a key the stylesheet knows rather than a style the file
   *  carries. "report" is the scouting-report table: rows only, no vertical
   *  rules, a shaded total column. */
  variant?: string;
  /** A tone per body cell, row-aligned with `rows`: what the cell IS - a group
   *  heading, an indented member of one, a number, a quiet secondary figure.
   *  See globals.css for the vocabulary. */
  tones?: string[][];
}): Block =>
  lockedTableFromRows({
    head: opts.head,
    rows: opts.rows.map((r) => r.map((c) => String(c))),
    bars: opts.bars,
    variant: opts.variant,
    tones: opts.tones,
  }) as Block;

/** BLOCKS SIDE BY SIDE, IN PROPORTIONS YOU CHOOSE.
 *
 *      row([shotMap, shotTable], [2, 1])   // picture two thirds, words one
 *      row([a, b])                          // even halves
 *
 *  Two to four columns. `weights` are RATIOS, not units - they are normalised
 *  to per cent of whatever the row is itself given, so a row nested inside
 *  another row still works. Omit them for an even split.
 *
 *  THIS IS THE ONE THING THAT FILLS A PAGE. A figure on a line of its own gets
 *  a line of its own, and the rest of that line is white: measured across the
 *  block gallery, ink covered 15.7% of the sheet and the columns between the
 *  leftmost and rightmost mark spanned barely half the width. Two figures in a
 *  row is not a tidier version of two figures stacked, it is the difference
 *  between a report and a slide deck - which is the exact complaint the figure
 *  size caps in globals.css were added to answer, by shrinking every figure
 *  instead.
 *
 *  Ratios matter as much as the pairing. `layoutRow` has supported them since
 *  it was written and nothing ever passed any, so every row in every document
 *  was an even split - a tactics board and a paragraph of prose each taking
 *  half a page, when the report this is modelled on gives the pictures two
 *  thirds. Pass weights.
 */
export const row = (blocks: (Block | Block[])[], weights?: number[]): Block =>
  layoutRow(blocks, weights) as Block;

/** A table you can type in, for the columns a coach fills in himself. */
export const typable = (head: string[], rows: (string | number)[][]): Block =>
  tableFromRows(head, rows.map((r) => r.map((c) => String(c)))) as Block;

// ------------------------------------------------------------------ output

/** Blocks -> the exact HTML the editor would have saved. */
export function renderNote(blocks: Block[]): string {
  const editor = new Editor({
    extensions: [
      Document, Paragraph, Text, Heading, Bold, Italic,
      BulletList, OrderedList, ListItem, Blockquote, HorizontalRule,
      PitchNode, FigureNode, QueryNode, StatNode, FaceNode,
      PageSetupNode, PageNode, PageBreakNode, ...grid.TABLE_EXTENSIONS,
    ],
    content: { type: "doc", content: blocks },
  });
  return editor.getHTML();
}

// ------------------------------------------------------------------- paper
// PDF is the artefact a coach actually carries, and until now the only way to
// get one was a human clicking through a browser's print dialog. An agent that
// can write the note but not the PDF has not finished the job.
//
// WHY A REAL BROWSER. `documentHtml()` needs the `.ft-*` rules, and
// `documentRules()` reads them off `document.styleSheets` — a live CSSOM.
// JSDOM has a `document` but its `styleSheets` is empty, because no sheet was
// ever added to that DOM, so the guard at the top of documentRules() passes
// and it returns "" ANYWAY. That is the trap this file would otherwise walk
// into: a PDF that is laid out correctly and completely unstyled, with no
// error anywhere. So the stylesheet is harvested inside Chromium, where there
// is a real one, and an empty result throws rather than prints.
//
// Playwright is imported lazily, the same bargain app/server/agent.py strikes with
// the Anthropic SDK: a report script that only calls writeNote() must not need
// a 300 MB browser on disk to run.

/** The app's stylesheet, as a path Chromium can load. The `.ft-*` rules are
 *  hand-written CSS — no `@apply` among them, those are all in `@layer base`
 *  on `*`/`body`/`html` — so the raw file parses to the same rules the built
 *  app serves, and `.ft-paper` carries its own palette as literal hex. */
const GLOBALS_CSS = fileURLToPath(new URL("../src/app/globals.css", import.meta.url));

/** Blocks -> a PDF, laid out by the same engine that draws the editor.
 *
 *  `browser` is there for the caller with twenty reports to make: a Chromium
 *  launch is a second or two and laying a report out is milliseconds, so the
 *  launch is the whole runtime of a batch. One reused handle covers every case
 *  in sight; a pool would be inventing a problem.
 */
export async function renderPdf(
  blocks: Block[],
  opts: { title?: string; sub?: string; browser?: PwBrowser; strict?: boolean } = {},
): Promise<Buffer> {
  let chromium;
  try {
    ({ chromium } = await import("playwright"));
  } catch {
    throw new Error(
      "writePdf needs Playwright: npm i -D playwright && npx playwright install chromium",
    );
  }

  const browser = opts.browser ?? (await chromium.launch());
  try {
    const page = await browser.newPage();

    // Two passes over the one page. First an empty document wearing the app's
    // stylesheet, so there is a CSSOM to lift the rules out of; then the real
    // document, built with those rules baked into its <style>.
    await page.setContent("<!doctype html><html><body></body></html>");
    await page.addStyleTag({ path: GLOBALS_CSS });
    const rules = await page.evaluate(documentRules);
    if (!rules.trim()) {
      throw new Error(
        `no .ft-* rules found in ${GLOBALS_CSS} — the PDF would print unstyled`,
      );
    }

    const body = renderNote(blocks);
    const html = documentHtml(body, opts.title ?? "", rules, opts.sub ?? "");
    await page.setContent(html, { waitUntil: "load" });

    // ---- WHAT DOES NOT FIT, BEFORE IT IS PRINTED.
    // A sheet taller than the paper does not fail: Chromium spills it onto
    // another page, and a single block taller than the paper is CLIPPED
    // outright, silently, in the PDF. Neither is visible to whoever generated
    // it. The editor already shows this as a tinted band below the fold; a
    // script gets nothing, so it is measured here.
    //
    // Reported, not repaired. Shrinking to fit is the tempting answer and it is
    // the wrong one - Beamer's manual grades its own `shrink` "very evil" after
    // twenty years, because a figure silently scaled to 71% is a figure whose
    // type no longer matches the page it is on. The author, or the agent, gets
    // told and decides.
    const paperH = docOrient(body) === "portrait" ? 297 : 210;
    const over = await page.evaluate((mm: number) => {
      const perMm = 96 / 25.4;
      const out: { what: string; index: number; mm: number }[] = [];
      document.querySelectorAll<HTMLElement>(".ft-page").forEach((el, i) => {
        const h = el.getBoundingClientRect().height / perMm;
        if (h > mm + 0.5) out.push({ what: "sheet", index: i + 1, mm: h - mm });
      });
      // A block that cannot fit ANY page is the one that gets cut in half.
      document
        .querySelectorAll<HTMLElement>(".ft-chart, .ft-pitchfig, .ft-figure, .ft-grid-wrap")
        .forEach((el, i) => {
          const h = el.getBoundingClientRect().height / perMm;
          if (h > mm) out.push({ what: "figure", index: i + 1, mm: h - mm });
        });
      return out;
    }, paperH);

    if (over.length) {
      const lines = over
        .map((o) => `  ${o.what} ${o.index} overflows the ${paperH}mm page by ${o.mm.toFixed(0)}mm`)
        .join("\n");
      if (opts.strict) {
        throw new Error(`${over.length} block(s) do not fit:\n${lines}`);
      }
      console.warn(`warning: ${over.length} block(s) do not fit the page:\n${lines}`);
    }

    // preferCSSPageSize honours the @page rule documentHtml() emits, which is
    // where the pageSetup node's portrait/landscape choice ends up. Without it
    // Chromium prints US Letter and a landscape report comes out cropped.
    return await page.pdf({ preferCSSPageSize: true, printBackground: true });
  } finally {
    if (!opts.browser) await browser.close();
  }
}

/** Render and write to disk. Returns the byte count. */
export async function writePdf(
  out: string,
  blocks: Block[],
  opts: { title?: string; sub?: string; browser?: PwBrowser; strict?: boolean } = {},
): Promise<number> {
  const pdf = await renderPdf(blocks, opts);
  await writeFile(out, pdf);
  return pdf.length;
}

/** Render and PUT. Returns what the server stored, which is the only number
 *  worth checking against jobs.NOTE_MAX. */
export async function writeNote(path: string, blocks: Block[]): Promise<number> {
  const html = renderNote(blocks);
  const url = `${API}/api/notes/${path.split("/").map(encodeURIComponent).join("/")}`;
  const res = await fetch(url, {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ html }),
  });
  if (!res.ok) throw new Error(`PUT ${path} -> ${res.status} ${await res.text()}`);
  const { bytes } = (await res.json()) as { bytes: number };
  return bytes;
}

// ---------------------------------------------------------------- selfcheck
// `SELFCHECK=1 npx vite-node --config vitest.config.mts scripts/note-api.mts`
//
// Opt-in rather than "am I the entry module", because vite-node does not put
// the script it is running into process.argv at all.
//
// The one thing worth pinning: a block written here SERIALIZES to the drawn
// figure, not to an empty tagged div. That is the whole reason this file is not
// a curl command — everything else is the note library's HTTP API unchanged.
if (process.env.SELFCHECK) {
  const html = renderNote([
    h(1, "Self-check"),
    p("Prose."),
    pitch({ magnets: [{ x: 50, y: 50, label: "10", side: "home" }], caption: "A board" }),
    chart({ title: "A chart", rows: [{ label: "one", values: [3] }, { label: "two", values: [1] }] }),
    table({
      head: ["Player", "Passes"],
      rows: [["Varga", 72]],
      variant: "report",
      tones: [["", "num"]],
    }),
    face({ name: "10. Tomas Varga", sub: "34 years, 169 cm, left" }),
    pageBreak(),
    p("On the next sheet."),
  ]);
  const want = [
    ["board is drawn", "<svg"],
    ["the discs are on top of the arrows", 'r="1.2"'],
    ["board reads back", 'data-pitch="'],
    ["caption", "<figcaption>A board</figcaption>"],
    ["chart is drawn", 'class="ft-chart"'],
    ["chart reads back", 'data-qrows="'],
    ["table", "<table"],
    ["table cell", "Varga"],
    ["the table's look is a key, not a style", 'data-variant="report"'],
    ["a cell says what it is", 'data-tone="num"'],
    ["a face with no photograph is a silhouette", 'class="ft-face"'],
    ["and it is drawn, not a broken image", "<circle cx=\"20\""],
    ["a page break survives the round trip", 'data-pagebreak="1"'],
  ];
  if (html.includes("<style")) throw new Error("a note must carry no styles of its own");
  for (const [what, needle] of want) {
    if (!html.includes(needle)) throw new Error(`${what}: no ${needle} in the saved note`);
  }
  console.log(`ok — every block serialized, ${(html.length / 1024).toFixed(1)} KB`);

  // ---- paper. The trap first: this is what the PDF path must never do.
  if (documentRules() !== "") {
    throw new Error("documentRules() found rules under JSDOM — the trap has moved");
  }
  console.log("ok — documentRules() is empty under JSDOM, which is why Chromium harvests it");

  const doc = [
    h(1, "Paper"),
    p("First sheet."),
    chart({ title: "A chart", rows: [{ label: "one", values: [3] }] }),
    pageBreak(),
    p("Second sheet."),
  ];

  const { chromium } = await import("playwright");
  const browser = await chromium.launch();
  try {
    const land = await renderPdf(doc, { title: "Self-check", browser });
    if (land.subarray(0, 5).toString() !== "%PDF-") throw new Error("not a PDF");
    if (land.length < 4096) throw new Error(`suspiciously small PDF: ${land.length} B`);

    const pages = (b: Buffer) => (b.toString("latin1").match(/\/Type\s*\/Page[^s]/g) ?? []).length;
    if (pages(land) !== 2) throw new Error(`one pageBreak should be two sheets, got ${pages(land)}`);

    // The @page rule documentHtml() emits has to reach Chromium, or a landscape
    // report prints on a portrait sheet with its right-hand column missing.
    const box = (b: Buffer) => {
      const m = b.toString("latin1").match(/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)/);
      if (!m) throw new Error("no MediaBox in the PDF");
      return { w: +m[1], h: +m[2] };
    };
    const l = box(land);
    if (l.w <= l.h) throw new Error(`landscape is the default, got ${l.w}x${l.h}`);

    const port = await renderPdf([pageSetup("portrait"), ...doc], { title: "Portrait", browser });
    const pp = box(port);
    if (pp.w >= pp.h) throw new Error(`pageSetup("portrait") did not reach @page: ${pp.w}x${pp.h}`);

    // An empty note is a document with nothing in it, not a crash.
    const empty = await renderPdf([], { title: "Empty", browser });
    if (empty.subarray(0, 5).toString() !== "%PDF-") throw new Error("empty note did not render");

    console.log(
      `ok — ${(land.length / 1024).toFixed(1)} KB, ${pages(land)} sheets, ` +
        `landscape ${l.w}x${l.h}, portrait ${pp.w}x${pp.h}`,
    );
  } finally {
    await browser.close();
  }
}
