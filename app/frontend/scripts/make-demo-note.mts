/** Build one note that carries every surface the notepad can make.
 *
 *  Run:  npx vite-node --config vitest.config.mts scripts/make-demo-note.mts -- <matchId>
 *
 *  Not a fixture and not hand-written markup. Every block goes through the SAME
 *  builders the panels call and the SAME serializer the editor saves with, so
 *  what lands on disk is exactly what a coach would have got by clicking
 *  through the menus — including the refusals, where this match cannot answer.
 *
 *  It exists because "is every surface editable?" is not a question anybody
 *  should have to answer by clicking eighteen menu items. Open the note, and
 *  the three levels are side by side: cells you type in, figures you change
 *  through a picker, and a board you draw on.
 */
import { readFileSync, writeFileSync } from "node:fs";

import { JSDOM } from "jsdom";

// The blocks' renderHTML builds real elements, so a DOM has to exist before
// any of those modules are imported.
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

const { TABLE_EXTENSIONS, tableContent, tableFromRows, lockedTableFromRows, layoutRow } = await import(
  "@/components/notepad/table-grid"
);
const { PitchNode } = await import("@/components/notepad/pitch-node");
const { FigureNode } = await import("@/components/notepad/figure-node");
const { QueryNode } = await import("@/components/notepad/query-node");
const { runQuery, sourceOf, dimOf, measureOf, queryTitle, defaultRender } = await import(
  "@/components/notepad/query-data"
);
const { buildFigure, firstChoice } = await import(
  "@/components/notepad/figure-data"
);
const { DEFAULT_MAGNETS } = await import("@/components/notepad/pitch-data");
const { VISUALS } = await import("@/components/notepad/visuals");

type Block = Record<string, unknown>;

const matchId = process.argv[process.argv.length - 1];
const API = process.env.FUTTRAFFIC_API ?? "http://127.0.0.1:8000";

const report = await (await fetch(`${API}/api/matches/${matchId}/match.json`)).json();
const team = report?.meta?.home_name ?? "";
const season = await (await fetch(`${API}/api/season?team=${encodeURIComponent(team)}`)).json();

const h = (level: number, text: string): Block => ({
  type: "heading",
  attrs: { level },
  content: [{ type: "text", text }],
});
const p = (text: string): Block =>
  text ? { type: "paragraph", content: [{ type: "text", text }] } : { type: "paragraph" };

/** A chart or pitch visual, frozen exactly as VisualPanel freezes one. */
function visual(source: string, dim: string, measure: string, render: string, extra: Partial<Record<string, unknown>> = {}): Block | null {
  const r = runQuery({
    report,
    season,
    source,
    dim,
    measure,
    split: String(extra.split ?? ""),
    limit: Number(extra.limit ?? 10),
    team: String(extra.team ?? ""),
    render,
  });
  if (!r.rows.length && !r.extra) return null;
  return {
    type: "query",
    attrs: {
      title: queryTitle(r, String(extra.split ?? ""), render),
      source,
      dim,
      measure,
      split: String(extra.split ?? ""),
      render,
      limit: Number(extra.limit ?? 10),
      team: String(extra.team ?? ""),
      keep: [],
      width: Number(extra.width ?? 100),
      custom: false,
      rows: r.rows,
      series: r.series,
      max: r.max,
      extra: r.extra ?? null,
      note: r.refusal ?? r.note,
      tier: r.tier,
      unit: r.measure.unit ?? "",
      dp: r.measure.dp,
      label: r.measure.label,
      dimLabel: r.dim.label,
      ordered: r.ordered,
    },
  };
}

/** One of the three report figures, as a LOCKED table.
 *
 *  The same grid as every other table in the note. The imported numbers are
 *  held still, the question rides along in an attribute, and right-clicking it
 *  offers unlock, the subset chips and Refresh. */
function figure(kind: string, limit = 22): Block | null {
  const cols = firstChoice(kind, report);
  const f = buildFigure({ report, season, kind, cols, limit });
  if (!f.rows.length) return null;
  return lockedTableFromRows({
    head: f.head,
    rows: f.rows.map((r) => [r.name, ...r.cells]),
    bars: f.rows.map((r) => r.split ?? null),
    figure: JSON.stringify({ kind, cols, limit }),
  }) as Block;
}

/** The same figure as an EDITABLE table — the other half of the toggle. */
function figureAsTable(kind: string): Block[] {
  const cols = firstChoice(kind, report);
  const f = buildFigure({ report, season, kind, cols, limit: 12 });
  if (!f.rows.length) return [p("(nothing to show for this match)")];
  return [tableFromRows(f.head, f.rows.map((r) => [r.name, ...r.cells]))];
}

/** A table with live formulas already in it.
 *
 *  `tableContent` takes text and not attributes, so the formula cells are
 *  written here. The TEXT is the stale answer and the attribute is the
 *  question — which is the whole design: the exported note has no script, so
 *  what prints is a number that was already correct when it was saved. Open
 *  the note, put the caret in one and the recalculation runs. */
function formulaTable(): Block {
  const t = tableContent({
    rows: 5,
    cols: 3,
    header: true,
    cells: [
      ["Player", "Minutes", "Distance"],
      ["#7", "90", "11.4"],
      ["#9", "78", "10.2"],
      ["#11", "62", "9.1"],
      ["Total", "230", "30.7"],
    ],
  }) as unknown as { content: { content: { attrs?: Record<string, unknown> }[] }[] };
  const last = t.content[4].content;
  last[1].attrs = { ...(last[1].attrs ?? {}), formula: "=SUM(B2:B4)" };
  last[2].attrs = { ...(last[2].attrs ?? {}), formula: "=SUM(C2:C4)" };
  return t as unknown as Block;
}

const skipped: string[] = [];
const made: string[] = [];
const keep = (label: string, b: Block | null): Block[] => {
  if (!b) {
    skipped.push(label);
    return [p(`${label} — this match cannot answer that one.`)];
  }
  made.push(label);
  return [b];
};

const blocks: Block[] = [
  h(1, "Every surface, in one note"),
  p(
    "Built by scripts/make-demo-note.mts through the same builders the slash menu uses. " +
      "Three kinds of thing live here and they behave differently on purpose.",
  ),

  // ------------------------------------------------------------------ 1
  h(2, "1. Type in the cells"),
  p("Click any cell and type. Tab moves across and adds a row at the end of the table. The toolbar appears when the caret is inside, with the row and column commands and a formula box."),
  tableContent({ rows: 4, cols: 4, header: true, cells: [
    ["Player", "Minutes", "Distance", "Your note"],
    ["#7", "90", "11.4", ""],
    ["#9", "78", "10.2", ""],
    ["#11", "62", "9.1", ""],
  ] }),
  p("Formulas: put the caret in a cell and use the Σ box in the toolbar. The one below is already =SUM(C2:C4)."),
  formulaTable(),
  p("Paste a range straight out of Excel or a terminal and it becomes a table of its own."),

  // ------------------------------------------------------------------ 2
  h(2, "2. Imported numbers — a real table, held still"),
  p("Both are the SAME kind of table. Left is unlocked — type in it straight away. Right is locked: the imported numbers are held still. Right-click it for Unlock, Columns, Rows and Refresh, each opening a panel to the right."),
  layoutRow([
    [h(3, "Editable"), ...figureAsTable("playerTable")],
    [h(3, "Locked — right-click it"), ...keep("Player table (locked)", figure("playerTable", 12))],
  ]),
  h(3, "Comparison — home against away, with the share bar as a column"),
  ...keep("Comparison", figure("sections")),
  h(3, "Timeline — goals, cards, substitutions"),
  ...keep("Timeline", figure("events")),

  // ------------------------------------------------------------------ 3
  h(2, "3. Charts — pictures, with editable numbers behind them"),
  p("RIGHT-CLICK any chart. The menu gates its settings the way a spreadsheet does — Picture, Question, Numbers, Colours, Sort, Width, Caption — and each opens a panel to the right when you pass over it. Numbers is a grid of the actual values: change one and the picture redraws. Drag a chart's right edge to resize it."),
  h(3, "Column, and the same question as a bar"),
  ...keep("Column", visual("tracks", "player", "distance_km", "column", { limit: 8 })),
  ...keep("Bar", visual("tracks", "player", "top_speed_kmh", "bar", { limit: 8 })),
  h(3, "Split into two series — stacked, and 100% stacked"),
  ...keep("Stacked column", visual("tracks", "position", "distance_km", "columnStacked", { split: "side" })),
  ...keep("100% stacked", visual("tracks", "position", "distance_km", "columnPct", { split: "side" })),
  h(3, "Ordered dimensions — line and area"),
  ...keep("Line", visual("feed", "minute", "count", "line", { limit: 20 })),
  ...keep("Area", visual("feed", "minute", "count", "area", { limit: 20 })),
  h(3, "Parts of a whole — pie and doughnut"),
  ...keep("Pie", visual("territory", "third", "share", "pie")),
  ...keep("Doughnut", visual("teamFeed", "side", "count", "doughnut")),
  h(3, "Two measures — scatter and combo"),
  ...keep("Scatter", visual("tracks", "player", "distance_km", "scatter", { split: "side", limit: 12 })),
  ...keep("Column + line", visual("tracks", "position", "distance_km", "combo", { split: "side" })),
  h(3, "A query drawn as a table — a picture of one, not a table you can type in"),
  ...keep("Table render", visual("teamFeed", "side", "count", "table")),

  // ------------------------------------------------------------------ 4
  h(2, "4. On the pitch"),
  p("Football's own pictures. Same block as the charts — read-only, with the numbers behind them editable."),
  h(3, "Shot map"),
  ...keep("Shot map", visual("shots", "result", "count", "pitch")),
  h(3, "Average positions"),
  ...keep("Average positions", visual("tracks", "player", "distance_km", "spots", { team })),
  h(3, "Line-up — drawn, not measured"),
  ...keep("Line-up", visual("squad", "player", "rating", "lineup", { team })),
  h(3, "Heat map"),
  ...keep("Heat map", visual("tracks", "player", "distance_km", "heat", { team })),
  h(3, "Territory, drawn where it happened"),
  ...keep("Zones", visual("territory", "third", "share", "zonepitch")),
  h(3, "Momentum across the ninety"),
  ...keep("Momentum", visual("momentum", "minute", "pressure", "momentum")),

  // ------------------------------------------------------------------ 5
  h(2, "5. The board — draw on it"),
  p("Not a picture. Click it, and the whole window becomes a whiteboard: drag the discs, draw solid/dashed/wavy arrows, add zones and text. Cancel means cancel; the note gains one undo step per editing session, not one per stroke."),
  { type: "pitch", attrs: { magnets: DEFAULT_MAGNETS, shapes: [], stripes: false, caption: "Drag the discs — this is a real board", note: "A default 4-4-2 to draw over, not this team's shape." } },

  // ------------------------------------------------------------------ 6
  h(2, "6. Layout — blocks side by side"),
  p("A borderless table. Drag the divider between the columns."),
  layoutRow([
    [h(3, "Left"), p("Text flows inside its own cell and does not spill into the next one — which is the difference between this and newspaper columns.")],
    [h(3, "Right"), p("Anything can go in a cell: a chart, a board, another table.")],
  ]),

  // ------------------------------------------------------------------ 7
  h(2, "7. Ordinary document furniture"),
  { type: "bulletList", content: [
    { type: "listItem", content: [p("Bullet lists")] },
    { type: "listItem", content: [p("Bold, italic, underline, links, fonts and sizes from the toolbar")] },
  ] },
  { type: "orderedList", content: [
    { type: "listItem", content: [p("Numbered lists")] },
    { type: "listItem", content: [p("Headings at three levels")] },
  ] },
  { type: "blockquote", content: [p("A quote, for the thing somebody actually said.")] },
  { type: "horizontalRule" },
  p("Print, top left, is the export: it rebuilds this as a standalone page with no script in it. Every number above was frozen when the block was made, so the file works offline and on paper — formulas included, because what is stored is the answer and the question beside it."),
  p("Not shown here, because they are choices rather than blocks: the slash menu now asks for the picture first and filters everything after it to what that picture can honestly draw; the match scope offers a tick-list of named fixtures; the stat step offers the report's own sets — Top stats, Attack, Passes, Defense, Duels, and Goalkeeping when somebody kept goal — plus any set you save for the project."),
];

const editor = new Editor({
  extensions: [
    Document, Paragraph, Text, Heading, Bold, Italic,
    BulletList, OrderedList, ListItem, Blockquote, HorizontalRule,
    PitchNode, FigureNode, QueryNode, ...TABLE_EXTENSIONS,
  ],
  content: { type: "doc", content: blocks },
});

const html = editor.getHTML();
const out = process.env.OUT ?? "";
if (out) writeFileSync(out, html, "utf-8");

console.log(`built ${made.length} blocks, ${skipped.length} refused`);
if (skipped.length) console.log("  refused:", skipped.join(", "));
console.log(`  ${(html.length / 1024).toFixed(1)} KB`);
if (out) console.log(`  wrote ${out}`);
void readFileSync;
void sourceOf;
void dimOf;
void measureOf;
void defaultRender;
void VISUALS;
