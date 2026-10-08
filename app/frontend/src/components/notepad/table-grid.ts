import { Extension, Node, callOrReturn, getExtensionField, mergeAttributes } from "@tiptap/core";
import {
  addColumnAfter,
  addColumnBefore,
  addRowAfter,
  addRowBefore,
  columnResizing,
  deleteColumn,
  deleteRow,
  deleteTable,
  fixTables,
  goToNextCell,
  mergeCells,
  splitCell,
  TableView,
  tableEditing,
  toggleHeaderColumn,
  toggleHeaderRow,
} from "@tiptap/pm/tables";
import type { Node as PMNode } from "@tiptap/pm/model";
import type { EditorState, Transaction } from "@tiptap/pm/state";

import { formulaPlugin } from "./formula-plugin";
import { lockPlugin } from "./lock-plugin";

/** A real table. Rows, columns, cells you type in, borders you drag.
 *
 *  WHY THIS EXISTS, having already had three table-shaped blocks. Those are
 *  all FROZEN QUERY RESULTS: the notepad computed some rows and rendered them
 *  read-only. That is a report, not a table. It cannot take a column of a
 *  coach's own notes beside the numbers, it cannot have two cells merged for a
 *  heading, and there is no way to make an empty one and start typing. Every
 *  document editor worth the name has the other thing, and this is it.
 *
 *  Built on prosemirror-tables, which is ALREADY IN THE TREE — @tiptap/pm
 *  depends on it — so real tables cost no new dependency, only the schema
 *  wrappers below. That is the same engine Google Docs-class ProseMirror
 *  editors use, cell selection and column resizing included.
 *
 *  THE LAYOUT TABLE. Word and Google Docs both have two different things
 *  called columns. `Format > Columns` is a SECTION property where text flows
 *  out of the bottom of column one into the top of column two — newspaper
 *  columns, for prose. It is not what anyone uses to put a chart beside its
 *  table, because content does not flow between cells there; you want them
 *  pinned side by side. Both Google's own help and every guide to it say the
 *  same thing: if you do not want the text to spill, use a table with its
 *  borders turned off. So that is what `layout` is — one row, N cells, no
 *  rules, resizable. One concept doing both jobs instead of two.
 */

export interface TableAttrs {
  layout: boolean;
}

/** `tableRole` is prosemirror-tables' field, not TipTap's, so TipTap's config
 *  type does not know about it and the extensions below would not compile.
 *  Declaring it here is what makes `tableRole: "cell"` a typed field rather
 *  than a cast at four call sites. */
declare module "@tiptap/core" {
  // The parameter list has to match @tiptap/core's own declaration exactly, or
  // TypeScript refuses to merge the two.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  interface NodeConfig<Options = any, Storage = any> {
    tableRole?:
      | string
      | ((this: { name: string; options: Options; storage: Storage }) => string);
  }
}

const cellAttrs = {
  colspan: {
    default: 1,
    parseHTML: (el: HTMLElement) => Number(el.getAttribute("colspan")) || 1,
  },
  rowspan: {
    default: 1,
    parseHTML: (el: HTMLElement) => Number(el.getAttribute("rowspan")) || 1,
  },
  /** `=SUM(C2:C11)`, kept beside the value it produced. See table-formula.ts
   *  for why the VALUE is what lives in the cell and the formula only rides
   *  along: the exported note has no script to recompute anything. */
  formula: {
    default: null as string | null,
    parseHTML: (el: HTMLElement) => el.getAttribute("data-formula") || null,
    renderHTML: (attrs: Record<string, unknown>) =>
      attrs.formula ? { "data-formula": String(attrs.formula) } : {},
  },
  /** WHAT THIS CELL IS, in the report's vocabulary: a group heading, an
   *  indented member of one, the shaded total column, a number, a quiet
   *  secondary figure, an italic aside. A key into the stylesheet, never a
   *  style — a document is a file a person edits, and `style` takes anything.
   *  Unknown keys land as a class nothing matches, which draws an ordinary
   *  cell. */
  tone: {
    default: "",
    // Narrowed on the way IN, because this string becomes part of a class
    // name: a `data-tone="x ft-grid--layout"` in a hand-edited file would
    // otherwise add a second class to a cell.
    parseHTML: (el: HTMLElement) =>
      (el.getAttribute("data-tone") ?? "").replace(/[^a-z0-9-]/g, "").slice(0, 16),
    renderHTML: (attrs: Record<string, unknown>) => {
      const tone = String(attrs.tone ?? "").replace(/[^a-z0-9-]/g, "").slice(0, 16);
      return tone ? { "data-tone": tone, class: `ft-cell ft-cell--${tone}` } : {};
    },
  },
  /** A share, 0-1, drawn as a split bar behind the cell.
   *
   *  The comparison figure's last column was never a number — it is home's
   *  share of the pair, as a picture. Keeping it when that figure became a real
   *  table meant either a new inline node or an attribute the stylesheet draws.
   *  This is the attribute: no new node type, and it prints, because it is a
   *  background on a `<td>` rather than something that has to be laid out. */
  bar: {
    default: null as number | null,
    parseHTML: (el: HTMLElement) => {
      // The attribute has to be READ before it is coerced. `Number(null)` is
      // 0, which is a perfectly good share — so testing the number alone made
      // every cell in the document parse as `bar: 0` and wear the bar class.
      const raw = el.getAttribute("data-bar");
      if (raw == null) return null;
      const n = Number(raw);
      return Number.isFinite(n) && n >= 0 && n <= 1 ? n : null;
    },
    renderHTML: (attrs: Record<string, unknown>) =>
      attrs.bar == null
        ? {}
        : {
            "data-bar": String(attrs.bar),
            class: "ft-grid__bar",
            // NOT A TEXT CELL. It is a drawing of the two numbers beside it, so
            // a caret in it could only ever produce a cell that shows one thing
            // and means another. Clicking lands in the neighbouring cell
            // instead, which is what `contenteditable="false"` does inside an
            // editable root — and it costs nothing on the printed page.
            contenteditable: "false",
            // Inline, because `attr()` cannot feed a number into a gradient in
            // any browser that ships today. It rides in the saved file too,
            // which is what makes the bar survive into the printed page.
            style: `--bar:${Math.max(0, Math.min(1, Number(attrs.bar)))}`,
          },
  },
  // prosemirror-tables writes resized widths here, as one number per column
  // the cell spans. Null means "share what is left", which is what every cell
  // says until somebody drags a border.
  colwidth: {
    default: null as number[] | null,
    parseHTML: (el: HTMLElement) => {
      const raw = el.getAttribute("data-colwidth");
      const parts = raw ? raw.split(",").map((n) => Number(n)) : [];
      return parts.length && parts.every((n) => Number.isFinite(n) && n > 0) ? parts : null;
    },
    // AND AN INLINE WIDTH, which is the half that reaches paper. The editor
    // gets its column widths from the <colgroup> that GridTableView builds out
    // of this attribute at runtime - and a node view does not serialize, so an
    // exported document had the numbers on the cells and no colgroup to apply
    // them. Every layout row printed as an even split no matter what it asked
    // for. `.ft-grid` is table-layout: fixed, so a width on the first row's
    // cells IS the column, and a set of widths that does not add up to the
    // measure is scaled - which is why these can be proportions.
    renderHTML: (attrs: Record<string, unknown>) =>
      attrs.colwidth ? { "data-colwidth": (attrs.colwidth as number[]).join(",") } : {},
  },
  /** A SHARE OF THE ROW, in per cent: a column whose width is a design decision
   *  rather than something somebody dragged.
   *
   *  `colwidth` cannot do this job. It is pixels, and the editor turns it into a
   *  <colgroup> inside a node view - which does not serialize, so an exported
   *  document carried the numbers and had nothing to apply them with, and every
   *  layout row printed as an even split however it was written. Emitting those
   *  pixels as an inline width only moved the bug: 900 px of columns inside a
   *  cell 690 px wide makes the table overflow to the right rather than share
   *  it, and a pitch came out drawn across the prose beside it. A percentage
   *  cannot overflow, and it means the same thing nested one row deeper. */
  pct: {
    default: null as number | null,
    parseHTML: (el: HTMLElement) => {
      const n = Number(el.getAttribute("data-pct"));
      return Number.isFinite(n) && n > 0 && n <= 100 ? n : null;
    },
    renderHTML: (attrs: Record<string, unknown>) =>
      attrs.pct ? { "data-pct": String(attrs.pct), style: `width: ${attrs.pct}%` } : {},
  },
};

/** prosemirror-tables reads `tableRole` off the NodeSpec, and TipTap only
 *  copies the spec fields it knows about. This is the one hook that lets a
 *  field through. */
export const TableRole = Extension.create({
  name: "tableRole",
  extendNodeSchema(extension) {
    const context = { name: extension.name, options: extension.options, storage: extension.storage };
    return {
      tableRole: callOrReturn(getExtensionField(extension, "tableRole", context)),
    };
  },
});

/** prosemirror-tables' own view, wearing this document's classes.
 *
 *  `columnResizing` installs a node view of its own — it has to, because the
 *  drag handles need a <colgroup> to write widths into and ProseMirror's
 *  serializer will not make one. That view builds its <table> from scratch and
 *  ignores the node spec, so the class from `renderHTML` never reached the
 *  editor: the table was styled in the exported file and unstyled on screen.
 *  Subclassing is the smallest fix that keeps one class name for both. */
class GridTableView extends TableView {
  constructor(node: PMNode, cellMinWidth: number) {
    super(node, cellMinWidth);
    this.dom.classList.add("ft-grid-wrap");
    this.paint(node);
  }

  paint(node: PMNode) {
    this.table.className = node.attrs.layout ? "ft-grid ft-grid--layout" : "ft-grid";
    // Same reason the class is set here: this view builds its <table> from
    // scratch and never sees `renderHTML`, so an attribute written there
    // reaches the saved file and NOT the screen. The lock was invisible in the
    // editor and correct in the export, which is the worst way round.
    if (node.attrs.locked) this.table.setAttribute("data-locked", "1");
    else this.table.removeAttribute("data-locked");
    // Same again for the stored question. Nothing on screen reads it — the
    // menu reads ProseMirror's state — but a table whose attributes are
    // visible in the exported file and invisible in the inspector is a table
    // that costs somebody an afternoon the first time they go looking.
    if (node.attrs.figure) this.table.setAttribute("data-figure", String(node.attrs.figure));
    else this.table.removeAttribute("data-figure");
    // And the look, for the same reason: a variant set through the menu has to
    // change the table under the caret, not only the file it saves to.
    const variant = String(node.attrs.variant ?? "").replace(/[^a-z0-9-]/g, "");
    if (variant) this.table.setAttribute("data-variant", variant);
    else this.table.removeAttribute("data-variant");
  }

  update(node: PMNode) {
    const ok = super.update(node);
    // Toggling `layout` keeps the same view and only changes an attribute, so
    // without this the borders stay on after the table becomes a layout row.
    if (ok) this.paint(node);
    return ok;
  }
}

export const TableNode = Node.create({
  name: "table",
  group: "block",
  content: "tableRow+",
  tableRole: "table",
  isolating: true,
  // Not draggable: the drag handle would sit over the cells and every attempt
  // to select text in the first cell would start a drag of the whole table.

  addAttributes() {
    return {
      layout: {
        default: false,
        parseHTML: (el: HTMLElement) => el.classList.contains("ft-grid--layout"),
        renderHTML: () => ({}),
      },
      /** Imported numbers, held still.
       *
       *  A table of what the pipeline measured is not a table somebody should
       *  be able to change by leaning on the keyboard — the whole value of it
       *  is that it says what the match said. So it arrives locked, and
       *  unlocking is a deliberate act through the right-click menu. Once
       *  unlocked it is an ordinary table in every respect, edits included. */
      locked: {
        default: false,
        parseHTML: (el: HTMLElement) => el.getAttribute("data-locked") === "1",
        renderHTML: (attrs: Record<string, unknown>) => (attrs.locked ? { "data-locked": "1" } : {}),
      },
      /** HOW THE TABLE IS DRESSED, not what is in it.
       *
       *  A scouting report's table is not the app's table: no vertical rules, a
       *  hairline between rows, a shaded column for the total and a group row
       *  that reads as a heading. That is a LOOK, so it is a key into a fixed
       *  table of rules in the stylesheet rather than a second table node —
       *  "" is the app's own grid, and anything the stylesheet does not know
       *  falls back to it, which is what makes a hand-edited document safe. */
      variant: {
        default: "",
        parseHTML: (el: HTMLElement) => el.getAttribute("data-variant") ?? "",
        renderHTML: (attrs: Record<string, unknown>) =>
          attrs.variant ? { "data-variant": String(attrs.variant) } : {},
      },
      /** The question these rows answered, as JSON, so the menu can ask it
       *  again. Absent on a table somebody built by hand — which then has
       *  nothing to refresh, and the menu says so instead of offering it. */
      figure: {
        default: null as string | null,
        parseHTML: (el: HTMLElement) => el.getAttribute("data-figure") || null,
        renderHTML: (attrs: Record<string, unknown>) =>
          attrs.figure ? { "data-figure": String(attrs.figure) } : {},
      },
    };
  },

  parseHTML() {
    return [{ tag: "table" }];
  },

  renderHTML({ HTMLAttributes, node }) {
    const layout = !!node.attrs.layout;
    return [
      "table",
      mergeAttributes(HTMLAttributes, {
        class: layout ? "ft-grid ft-grid--layout" : "ft-grid",
      }),
      ["tbody", 0],
    ];
  },

  addProseMirrorPlugins() {
    return [
      // The drag handles on the column borders. `lastColumnResizable: false`
      // because the last border is the table's right edge, and dragging it
      // makes a table narrower than the text it sits in for no reason.
      columnResizing({ lastColumnResizable: false, View: GridTableView }),
      tableEditing({ allowTableNodeSelection: true }),
      // Formulas recalculate inside ProseMirror's own cycle, never from a React
      // effect writing back into the document. See formula-plugin.ts.
      formulaPlugin(),
      // A table of imported numbers holds still until somebody unlocks it.
      lockPlugin(),
    ];
  },

  addKeyboardShortcuts() {
    return {
      // Tab through the cells, the way it works in every other editor. It has
      // to lose to nothing else, which is why it is here and not a global.
      Tab: () => this.editor.commands.command(({ state, dispatch }) => nextCell(1)(state, dispatch)),
      "Shift-Tab": () =>
        this.editor.commands.command(({ state, dispatch }) => nextCell(-1)(state, dispatch)),
    };
  },

  onCreate() {
    // A pasted or hand-edited document can carry a table whose rows disagree
    // about how many cells they have. Every command below assumes a rectangle;
    // this is what makes it one, once, rather than each command guarding.
    const tr = fixTables(this.editor.state);
    if (tr) this.editor.view.dispatch(tr);
  },
});

export const TableRowNode = Node.create({
  name: "tableRow",
  content: "(tableCell | tableHeader)*",
  tableRole: "row",
  parseHTML() {
    return [{ tag: "tr" }];
  },
  renderHTML({ HTMLAttributes }) {
    return ["tr", HTMLAttributes, 0];
  },
});

export const TableCellNode = Node.create({
  name: "tableCell",
  // `block+`, so a cell can hold a chart, a board, a list — which is the whole
  // point of the layout table.
  content: "block+",
  tableRole: "cell",
  isolating: true,
  addAttributes: () => cellAttrs,
  parseHTML() {
    return [{ tag: "td" }];
  },
  renderHTML({ HTMLAttributes }) {
    return ["td", HTMLAttributes, 0];
  },
});

export const TableHeaderNode = Node.create({
  name: "tableHeader",
  content: "block+",
  tableRole: "header_cell",
  isolating: true,
  addAttributes: () => cellAttrs,
  parseHTML() {
    return [{ tag: "th" }];
  },
  renderHTML({ HTMLAttributes }) {
    return ["th", HTMLAttributes, 0];
  },
});

export const TABLE_EXTENSIONS = [TableRole, TableNode, TableRowNode, TableCellNode, TableHeaderNode];

// --------------------------------------------------------------------- shapes

/** One node of ProseMirror JSON, on its way into the document. Named because
 *  a panel now hands over a SEQUENCE — a heading, a table, and the line saying
 *  what the numbers rest on — and `Record<string, unknown>[]` at four call
 *  sites does not say that. */
export type Block = Record<string, unknown>;

/** Cell content for one string. Empty string means an empty paragraph, not a
 *  paragraph holding "" — ProseMirror rejects an empty text node. */
const cell = (type: "tableCell" | "tableHeader", text: string, tone = "") => ({
  type,
  ...(tone ? { attrs: { tone } } : {}),
  content: [
    text ? { type: "paragraph", content: [{ type: "text", text }] } : { type: "paragraph" },
  ],
});

/** A blank grid, optionally with a header row — the Google Docs grid picker's
 *  output. `layout` drops the rules and the header, which is the side-by-side
 *  container rather than a table of data. */
export function tableContent(opts: {
  rows: number;
  cols: number;
  header?: boolean;
  layout?: boolean;
  cells?: string[][];
  /** How the table is dressed. "" is the app's grid; see globals.css. */
  variant?: string;
  /** A tone per cell, same shape as `cells`. Blank leaves the cell plain. */
  tones?: string[][];
}) {
  const rows = Math.max(1, Math.min(50, opts.rows));
  // TWENTY-FOUR, not twelve. The cap was a guess and the source report walks
  // straight through it: the field-players grid is nineteen measures wide and
  // the corner-takers grid is twenty-two, both printed as ONE table across a
  // landscape sheet. Twelve meant splitting each into two tables that shared a
  // first column, which is two headers, two sets of rules and a paragraph of
  // apology where the source has one grid. A cap still exists because a runaway
  // column count is a broken caller, not a wide table.
  const cols = Math.max(1, Math.min(24, opts.cols));
  const header = !!opts.header && !opts.layout;
  return {
    type: "table",
    attrs: { layout: !!opts.layout, variant: opts.variant ?? "" },
    content: Array.from({ length: rows }, (_, r) => ({
      type: "tableRow",
      content: Array.from({ length: cols }, (_, c) =>
        cell(
          header && r === 0 ? "tableHeader" : "tableCell",
          opts.cells?.[r]?.[c] ?? "",
          opts.tones?.[r]?.[c] ?? "",
        ),
      ),
    })),
  };
}

/** A row of blocks, side by side. The blocks go in as-is, so a chart and the
 *  table it summarises end up level with each other.
 *
 *  A cell takes either one block or a stack of them, because the cell spec is
 *  `block+` and what goes beside a chart is now a heading, a table and its
 *  provenance line rather than a single figure. */
export function layoutRow(blocks: (Block | Block[])[], widths?: number[]) {
  const cols = Math.max(2, Math.min(4, blocks.length));
  // COLUMNS OF DIFFERENT WIDTHS, because a page is not made of equal thirds.
  // Every layout row was an even split, so a column of prose beside two pitches
  // got a third of the page and the pitches got a third each - and the report it
  // is copied from gives the pictures two thirds and the words one. Proportions
  // rather than pixels, and written as PER CENT of the row - see the `pct` cell
  // attribute - so [2, 1] is two thirds and one third of whatever the row is
  // itself given, nested inside another row included.
  const w = widths?.length === cols ? widths : null;
  const scale = w ? 100 / w.reduce((a, b) => a + b, 0) : 0;
  return {
    type: "table",
    attrs: { layout: true },
    content: [
      {
        type: "tableRow",
        content: Array.from({ length: cols }, (_, i) => {
          const stack = [blocks[i] ?? []].flat();
          return {
            type: "tableCell",
            attrs: w ? { pct: Math.round(w[i] * scale * 100) / 100 } : {},
            // A cell needs `block+`; a missing block would make the insert fail
            // silently rather than throw, which reads as a button that does
            // nothing at all.
            content: stack.length ? stack : [{ type: "paragraph" }],
          };
        }),
      },
    ],
  };
}

/** A frozen figure as a REAL table, held still.
 *
 *  This is what the player table, the comparison and the timeline are now. They
 *  used to be their own read-only block that happened to look like a table —
 *  the most table-shaped things in the document and the only ones with no caret
 *  in them. Same grid as everything else now, so the toolbar, the column
 *  widths, the merges and the formulas all apply the moment it is unlocked.
 *
 *  `bars` is the comparison's share column, one value per ROW, drawn behind the
 *  cell rather than in it. `figure` is the question, so the menu can ask again.
 */
/** How wide the share column is drawn, in pixels. Narrow on purpose: it is a
 *  proportion, and a proportion does not get wider by being read from further
 *  away. */
export const BAR_WIDTH = 96;

export function lockedTableFromRows(opts: {
  head: string[];
  rows: string[][];
  /** 0-1 per row, appended as a final bar cell. Omit for no bar column. */
  bars?: (number | null)[];
  /** JSON the right-click menu re-runs. */
  figure?: string;
  /** How the table is dressed. "" is the app's grid. */
  variant?: string;
  /** A tone per body cell, row-aligned with `rows` (the header row is not in
   *  it, so a caller does not have to carry a blank first row). */
  tones?: string[][];
}) {
  const bars = opts.bars ?? [];
  const wantsBar = bars.some((b) => b != null);
  const head = wantsBar ? [...opts.head, ""] : opts.head;
  const table = tableContent({
    rows: opts.rows.length + 1,
    cols: head.length,
    header: true,
    variant: opts.variant,
    cells: [head, ...opts.rows.map((r) => (wantsBar ? [...r, ""] : r))],
    // A COLUMN HEADER SITS OVER ITS COLUMN. It got no tone at all, so a
    // right-aligned number column carried a left-aligned label: "CHALLENGES /
    // WON" began at the left edge of a wide column and "126 / 90 - 71%" ended
    // at the right edge of it, with a thumb's width of white between the two
    // and no way to tell which header named which figure. Only the ALIGNMENT
    // carries up - a group row's fill is a property of that row, not of the
    // column - and a column counts as numeric if any body row says so.
    tones: opts.tones
      ? [head.map((_, c) => (opts.tones!.some((r) => r[c] === "num") ? "num" : "")), ...opts.tones]
      : undefined,
  }) as unknown as {
    attrs: Record<string, unknown>;
    content: { content: { attrs?: Record<string, unknown> }[] }[];
  };
  table.attrs = { ...table.attrs, locked: true, figure: opts.figure ?? null };
  if (wantsBar) {
    // A FIXED width, on every cell of the column. `.ft-grid` is
    // `table-layout: fixed`, so a column with no width shares the table
    // equally with the others — which gave a 90-pixel drawing a quarter of the
    // page and stretched the bar into a rule across the row. prosemirror-tables
    // reads `colwidth` off the cells to build its <colgroup>, and it has to be
    // on all of them or dragging another border re-derives the column and the
    // width goes back.
    table.content.forEach((row, i) => {
      const cells = row.content;
      const last = cells[cells.length - 1];
      last.attrs = { ...(last.attrs ?? {}), colwidth: [BAR_WIDTH] };
      const share = bars[i - 1];
      if (i === 0 || share == null) return;
      last.attrs = { ...last.attrs, bar: Math.max(0, Math.min(1, share)) };
    });
  }
  return table as unknown as Block;
}

/** Head and rows straight into an EDITABLE table. This is the bridge the
 *  notepad was missing: a query answers a question, and what lands is a table
 *  a coach can then put their own column beside. */
export function tableFromRows(head: string[], rows: string[][]) {
  return tableContent({
    rows: rows.length + 1,
    cols: head.length,
    header: true,
    cells: [head, ...rows],
  });
}

/** Rows out of a plain-text clipboard, or null when it is not a grid.
 *
 *  Only text/plain needs reading here. Excel, Sheets and Numbers all put an
 *  HTML table on the clipboard alongside the text, and ProseMirror already
 *  parses that through `parseHTML` on the nodes above — so the common copy
 *  works without this and always did. What arrives as text ALONE is a copy out
 *  of a terminal, a .tsv, or a plain editor, and that is the gap.
 *
 *  The refusals matter more than the parse. A tab in a sentence is not a grid,
 *  and neither is a ragged paste; guessing wrong turns a paragraph somebody
 *  meant to keep into a one-column table they have to undo. So: two lines or
 *  more, two columns or more, and every row the same width, or this declines
 *  and the ordinary paste happens.
 */
export function parseGrid(text: string): string[][] | null {
  const lines = text.replace(/\r\n?/g, "\n").replace(/\n+$/, "").split("\n");
  if (lines.length < 2) return null;
  const rows = lines.map((l) => l.split("\t"));
  const cols = rows[0].length;
  if (cols < 2) return null;
  return rows.every((r) => r.length === cols) ? rows : null;
}

// ------------------------------------------------------------------ commands

type Cmd = (state: EditorState, dispatch?: (tr: Transaction) => void) => boolean;

/** Everything the toolbar can do, in the order Google Docs lists them. Wrapped
 *  once here so the toolbar never imports prosemirror-tables itself. */
export const TABLE_COMMANDS: Record<string, Cmd> = {
  addRowBefore,
  addRowAfter,
  addColumnBefore,
  addColumnAfter,
  deleteRow,
  deleteColumn,
  mergeCells,
  splitCell,
  toggleHeaderRow,
  toggleHeaderColumn,
  deleteTable,
};

/** Tab between the cells, growing the table at its end.
 *
 *  `goToNextCell` alone returns false in the last cell, and a false here is not
 *  neutral: the keydown goes unprevented, the browser moves focus out of the
 *  contenteditable, and everything typed after that lands NOWHERE — not in the
 *  table, not in the document, gone. That is what this used to do, and it is
 *  the reason a table did not feel like a table.
 *
 *  So neither end falls through. Forwards out of the last cell adds a row, the
 *  way Excel and Google Docs both do. Backwards out of the first one has
 *  nothing to grow, so the key is simply swallowed and the caret stays put.
 */
export function nextCell(dir: 1 | -1): Cmd {
  return (state, dispatch) => {
    if (!inTable(state)) return false;
    if (goToNextCell(dir)(state, dispatch)) return true;
    if (dir === 1 && dispatch) {
      // One transaction, not two: `addRowAfter` writes the row and the caret
      // move rides along on the same tr, so Tab is a single undo step rather
      // than a row you have to undo twice to be rid of.
      addRowAfter(state, (tr) => {
        // `state.apply(tr).doc` IS `tr.doc`, so the selection the second
        // command finds needs no mapping to be valid on the first's document.
        goToNextCell(1)(state.apply(tr), (moved) => tr.setSelection(moved.selection));
        dispatch(tr);
      });
    }
    return true;
  };
}

export function inTable(state: EditorState): boolean {
  const $head = state.selection.$head;
  for (let d = $head.depth; d > 0; d -= 1) {
    if ($head.node(d).type.spec.tableRole === "row") return true;
  }
  return false;
}

/** The table the caret is in, and where it starts. Null outside one. */
export function tableAt(state: EditorState): { node: import("@tiptap/pm/model").Node; pos: number } | null {
  const $head = state.selection.$head;
  for (let d = $head.depth; d > 0; d -= 1) {
    const node = $head.node(d);
    if (node.type.spec.tableRole === "table") return { node, pos: $head.before(d) };
  }
  return null;
}
