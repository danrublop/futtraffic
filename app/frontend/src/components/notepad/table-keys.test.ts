import { getSchema } from "@tiptap/core";
import Document from "@tiptap/extension-document";
import Paragraph from "@tiptap/extension-paragraph";
import Text from "@tiptap/extension-text";
import { Node as PMNode } from "@tiptap/pm/model";
import { EditorState, TextSelection, type Transaction } from "@tiptap/pm/state";
import { describe, expect, it } from "vitest";

import { TABLE_EXTENSIONS, nextCell, tableContent } from "./table-grid";

/** Tab at the edges of a table.
 *
 *  The bug this exists to hold down: `nextCell` returning FALSE at the last
 *  cell left the keydown unprevented, so the browser moved focus out of the
 *  editor and every keystroke after the Tab was swallowed by the page. Nothing
 *  threw and nothing was marked unsaved — the typing was simply gone. So what
 *  is asserted here is not only "a row appears" but "the key was handled",
 *  which is the half that loses the text when it regresses.
 */

const schema = getSchema([Document, Paragraph, Text, ...TABLE_EXTENSIONS]);

/** A state whose caret sits in the cell at (row, col) of a `rows`x`cols` grid. */
function stateAt(rows: number, cols: number, row: number, col: number) {
  const doc = PMNode.fromJSON(schema, {
    type: "doc",
    content: [tableContent({ rows, cols, header: false })],
  });
  let seen = -1;
  let pos = 0;
  doc.descendants((node, at) => {
    if (node.type.spec.tableRole !== "cell" && node.type.spec.tableRole !== "header_cell") return true;
    seen += 1;
    if (seen === row * cols + col) pos = at + 2; // inside the cell's paragraph
    return false;
  });
  const state = EditorState.create({ schema, doc });
  return state.apply(state.tr.setSelection(TextSelection.create(doc, pos)));
}

/** Run a command the way a keymap does, and report both halves. */
function run(state: EditorState, dir: 1 | -1) {
  let tr: Transaction | null = null;
  const handled = nextCell(dir)(state, (t) => {
    tr = t;
  });
  return { handled, next: tr ? state.apply(tr) : state };
}

/** Which cell the caret is in, counting from zero across the whole table. */
function caretCell(state: EditorState): number {
  const cells: number[] = [];
  state.doc.descendants((node, at) => {
    const role = node.type.spec.tableRole;
    if (role !== "cell" && role !== "header_cell") return true;
    cells.push(at);
    return false;
  });
  const head = state.selection.$head;
  for (let d = head.depth; d > 0; d -= 1) {
    const i = cells.indexOf(head.before(d));
    if (i !== -1) return i;
  }
  return -1;
}

const rowCount = (state: EditorState) => state.doc.firstChild!.childCount;

describe("Tab inside a table", () => {
  it("steps to the next cell without touching the shape", () => {
    const { handled, next } = run(stateAt(2, 3, 0, 0), 1);
    expect(handled).toBe(true);
    expect(caretCell(next)).toBe(1);
    expect(rowCount(next)).toBe(2);
  });

  it("adds a row at the last cell and lands the caret in it", () => {
    const { handled, next } = run(stateAt(2, 3, 1, 2), 1);
    expect(handled).toBe(true);
    expect(rowCount(next)).toBe(3);
    // First cell of the row that was just made.
    expect(caretCell(next)).toBe(6);
  });

  it("grows by ONE row, in ONE transaction, so a single undo takes it back", () => {
    const state = stateAt(2, 2, 1, 1);
    let count = 0;
    nextCell(1)(state, (tr) => {
      count += 1;
      expect(tr.docs.length).toBeGreaterThan(0);
      expect(state.apply(tr).doc.firstChild!.childCount).toBe(3);
    });
    expect(count).toBe(1);
  });

  it("HANDLES Shift-Tab in the first cell rather than letting focus escape", () => {
    // The assertion that matters is `handled`. Returning false here is what
    // used to throw the next keystrokes away.
    const { handled, next } = run(stateAt(2, 3, 0, 0), -1);
    expect(handled).toBe(true);
    expect(caretCell(next)).toBe(0);
    expect(rowCount(next)).toBe(2);
  });

  it("steps backwards between cells", () => {
    const { handled, next } = run(stateAt(2, 3, 1, 0), -1);
    expect(handled).toBe(true);
    expect(caretCell(next)).toBe(2);
  });

  it("declines outside a table, so Tab still belongs to the document", () => {
    const doc = PMNode.fromJSON(schema, {
      type: "doc",
      content: [{ type: "paragraph", content: [{ type: "text", text: "hi" }] }],
    });
    const state = EditorState.create({ schema, doc });
    expect(nextCell(1)(state, () => {})).toBe(false);
    expect(nextCell(-1)(state, () => {})).toBe(false);
  });
});
