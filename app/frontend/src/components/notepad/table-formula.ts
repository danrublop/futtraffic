import type { Node as PMNode } from "@tiptap/pm/model";
import { TableMap } from "@tiptap/pm/tables";
import FormulaParser, { DepParser, FormulaError, type Coord, type Ref } from "fast-formula-parser";

/** Formulas, in a table in a note.
 *
 *  THE VALUE IS FROZEN, THE FORMULA RIDES ALONG. A cell keeps `=SUM(C2:C11)`
 *  in an attribute and holds the computed number as its ordinary text. That is
 *  not a shortcut, it is the same rule every other block in this notepad obeys:
 *  the exported document is a self-contained file with no script and nothing to
 *  fetch, so anything that has to RUN to show a number cannot be in it. A note
 *  goes to a player as a PDF; a spreadsheet that needs a spreadsheet to read it
 *  does not.
 *
 *  So recalculation happens here, in the editor, and what leaves the building
 *  is text that was already correct when it was written.
 *
 *  ONE TABLE IS ONE SHEET. There are no cross-table references. A1 addresses
 *  the first cell of the table the formula is in, header row included — which
 *  is what a spreadsheet does, where the header IS row 1.
 */

/** 0-based column index to Excel's letters. 0 → A, 25 → Z, 26 → AA. */
export function colLabel(i: number): string {
  let n = i;
  let out = "";
  do {
    out = String.fromCharCode(65 + (n % 26)) + out;
    n = Math.floor(n / 26) - 1;
  } while (n >= 0);
  return out;
}

/** 0-based row and column to an A1 address. */
export const a1 = (row: number, col: number) => `${colLabel(col)}${row + 1}`;

/** A cell's text as a value the evaluator can use.
 *
 *  THE UNIT IS STRIPPED, AND THAT IS A DELIBERATE DIVERGENCE FROM EXCEL. Every
 *  measured figure in this product is printed with what it is measured in —
 *  "13.47 km", "24.1 km/h", "90 min" — because a bare number in a scouting
 *  report is how you get a distance read as a speed. Excel would call all of
 *  those text and answer #VALUE! to any sum of them, which would make formulas
 *  useless on the only tables anybody here actually has.
 *
 *  So a leading number wins and the trailing unit is ignored. A cell that does
 *  not START with a number is still text, so `SUM` over a column of names is
 *  0 and not a lie.
 */
export function cellValue(text: string): number | string {
  const t = text.trim();
  if (!t) return "";
  // Thousands separators only between digits, so "1,234.5 km" reads but a
  // comma-separated list does not become a number.
  const m = /^-?\d[\d,]*(\.\d+)?/.exec(t);
  if (!m) return t;
  const n = Number(m[0].replace(/,/g, ""));
  return Number.isFinite(n) ? n : t;
}

/** Every cell of a table, in row-major order, with where it starts. */
export interface Cell {
  /** Document position of the cell node itself. */
  pos: number;
  node: PMNode;
  row: number;
  col: number;
  text: string;
  formula: string | null;
}

/** The grid, read off prosemirror-tables' own map.
 *
 *  `TableMap` has already done the hard part — a table with merged cells is
 *  still a rectangle to it, and every address below is an index into that
 *  rectangle rather than a walk over rows that would miscount the moment
 *  something spans two of them. A merged cell answers to every address it
 *  covers, which is what a spreadsheet does too.
 */
export function readTable(table: PMNode, tableStart: number): {
  cells: Cell[];
  at: (row: number, col: number) => Cell | null;
  width: number;
  height: number;
} {
  const map = TableMap.get(table);
  const seen = new Map<number, Cell>();
  const byAddress = new Map<string, Cell>();

  for (let row = 0; row < map.height; row += 1) {
    for (let col = 0; col < map.width; col += 1) {
      const rel = map.map[row * map.width + col];
      let cell = seen.get(rel);
      if (!cell) {
        const node = table.nodeAt(rel);
        if (!node) continue;
        cell = {
          // `tableStart` is the position of the table node; +1 steps inside it,
          // which is what TableMap's offsets are relative to.
          pos: tableStart + 1 + rel,
          node,
          row,
          col,
          text: node.textContent,
          formula: (node.attrs.formula as string | null) || null,
        };
        seen.set(rel, cell);
      }
      byAddress.set(`${row}:${col}`, cell);
    }
  }

  return {
    cells: [...seen.values()],
    at: (row, col) => byAddress.get(`${row}:${col}`) ?? null,
    width: map.width,
    height: map.height,
  };
}

const SHEET = "T";
const CIRCULAR = "#CIRCULAR!";
const deps = new DepParser();

/** Is this reference a range, or a single cell? */
const isRange = (r: Ref): r is { from: Coord; to: Coord } => "from" in r;

/** The addresses a formula reads, as `row:col` keys. Anything the parser
 *  cannot make sense of contributes nothing rather than throwing — a broken
 *  formula is one bad cell, never a document that will not open. */
function refsOf(formula: string, row: number, col: number): string[] {
  let found: Ref[];
  try {
    found = deps.parse(formula, { row: row + 1, col: col + 1, sheet: SHEET });
  } catch {
    return [];
  }
  const out: string[] = [];
  for (const r of found) {
    if (isRange(r)) {
      for (let y = r.from.row; y <= r.to.row; y += 1) {
        for (let x = r.from.col; x <= r.to.col; x += 1) out.push(`${y - 1}:${x - 1}`);
      }
    } else {
      out.push(`${r.row - 1}:${r.col - 1}`);
    }
  }
  return out;
}

/** What a formula produced, as the text a cell should show. */
function render(result: unknown): string {
  if (result instanceof FormulaError) return result._error;
  if (result == null) return "";
  if (typeof result === "number") {
    if (!Number.isFinite(result)) return "#NUM!";
    // Excel's own default: show what is there, do not print 0.30000000000000004.
    return String(Math.round(result * 1e10) / 1e10);
  }
  if (typeof result === "boolean") return result ? "TRUE" : "FALSE";
  if (Array.isArray(result)) return render(result[0]?.[0] ?? result[0]);
  return String(result);
}

export interface Update {
  pos: number;
  /** The cell node, so the caller can rebuild its content in place. */
  node: PMNode;
  text: string;
}

/** Recalculate one table. Returns only the cells whose text actually changes,
 *  so a document with nothing to do produces no transaction at all — which is
 *  what stops the recalculation plugin from calling itself forever.
 */
export function recalcTable(table: PMNode, tableStart: number): Update[] {
  const grid = readTable(table, tableStart);
  const formulas = grid.cells.filter((c) => c.formula);
  if (!formulas.length) return [];

  /** Every cell's current value, replaced as each formula is evaluated. */
  const values = new Map<string, number | string>();
  for (let row = 0; row < grid.height; row += 1) {
    for (let col = 0; col < grid.width; col += 1) {
      const cell = grid.at(row, col);
      values.set(`${row}:${col}`, cell ? cellValue(cell.text) : "");
    }
  }

  const parser = new FormulaParser({
    onCell: ({ row, col }) => {
      const v = values.get(`${row - 1}:${col - 1}`);
      return v === undefined ? null : v;
    },
    onRange: ({ from, to }) => {
      const out: (number | string | null)[][] = [];
      for (let y = from.row; y <= to.row; y += 1) {
        const line: (number | string | null)[] = [];
        for (let x = from.col; x <= to.col; x += 1) {
          const v = values.get(`${y - 1}:${x - 1}`);
          line.push(v === undefined ? null : v);
        }
        out.push(line);
      }
      return out;
    },
  });

  // Evaluate in dependency order, so `=D2*2` where D2 is itself `=SUM(...)`
  // reads the sum and not the stale text under it. A cell that depends on
  // itself, directly or round a loop, says so instead of hanging.
  const byKey = new Map(formulas.map((c) => [`${c.row}:${c.col}`, c]));
  const done = new Map<string, string>();
  const visiting = new Set<string>();

  const evaluate = (key: string): string => {
    const cached = done.get(key);
    if (cached !== undefined) return cached;
    const cell = byKey.get(key);
    if (!cell || !cell.formula) return "";
    if (visiting.has(key)) {
      done.set(key, CIRCULAR);
      values.set(key, CIRCULAR);
      return CIRCULAR;
    }
    visiting.add(key);
    const body = cell.formula.replace(/^=/, "");
    // A cell is in the cycle if it reads itself, or reads anything that turned
    // out to be circular. Excel marks the whole loop, not just where it was
    // noticed, and a cell quietly computing from a #CIRCULAR! neighbour would
    // print a number that means nothing.
    let circular = false;
    for (const ref of refsOf(body, cell.row, cell.col)) {
      if (ref === key) circular = true;
      else if (byKey.has(ref) && evaluate(ref) === CIRCULAR) circular = true;
    }
    visiting.delete(key);

    // Set while we were recursing: a dependency came back round and found this
    // cell still on the stack. That verdict stands — evaluating anyway would
    // overwrite it with a number computed from stale text.
    const marked = done.get(key);
    if (marked !== undefined) return marked;
    if (circular) {
      done.set(key, CIRCULAR);
      values.set(key, CIRCULAR);
      return CIRCULAR;
    }

    let text: string;
    try {
      text = render(parser.parse(body, { row: cell.row + 1, col: cell.col + 1, sheet: SHEET }));
    } catch {
      // A formula the parser will not even read. Excel refuses these at the
      // point of typing; a document can hold one because a document can be
      // hand-edited.
      text = "#ERROR!";
    }
    done.set(key, text);
    values.set(key, cellValue(text));
    return text;
  };

  const updates: Update[] = [];
  for (const cell of formulas) {
    const key = `${cell.row}:${cell.col}`;
    const text = evaluate(key);
    if (text !== cell.text) updates.push({ pos: cell.pos, node: cell.node, text });
  }
  return updates;
}

/** Every table in the document, with where it starts. */
export function eachTable(doc: PMNode, fn: (table: PMNode, start: number) => void) {
  doc.descendants((node, pos) => {
    if (node.type.spec.tableRole !== "table") return true;
    fn(node, pos);
    // A table cannot hold another table's ROWS, but a layout cell can hold a
    // whole table — so this keeps descending rather than returning false.
    return true;
  });
}
