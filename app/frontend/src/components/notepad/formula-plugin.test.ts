import { getSchema } from "@tiptap/core";
import Document from "@tiptap/extension-document";
import Paragraph from "@tiptap/extension-paragraph";
import Text from "@tiptap/extension-text";
import { Node as PMNode } from "@tiptap/pm/model";
import { EditorState, TextSelection } from "@tiptap/pm/state";
import { describe, expect, it } from "vitest";

import { formulaPlugin } from "./formula-plugin";
import { TABLE_EXTENSIONS } from "./table-grid";
import { readTable } from "./table-formula";

/** The recalculation pass, driven the way the editor drives it.
 *
 *  This is where the two failure modes that cannot be unit-tested in isolation
 *  live: a pass that appends a transaction forever, and a pass that deletes a
 *  formula because the coach typed a word somewhere else in the note.
 */

const schema = getSchema([Document, Paragraph, Text, ...TABLE_EXTENSIONS]);

const cell = (text: string, formula?: string) => ({
  type: "tableCell",
  attrs: formula ? { formula } : {},
  content: [text ? { type: "paragraph", content: [{ type: "text", text }] } : { type: "paragraph" }],
});

/** A doc of one paragraph and one table, with the plugin installed. */
function stateOf(rows: ReturnType<typeof cell>[][]) {
  const doc = PMNode.fromJSON(schema, {
    type: "doc",
    content: [
      { type: "paragraph", content: [{ type: "text", text: "notes" }] },
      {
        type: "table",
        attrs: { layout: false },
        content: rows.map((r) => ({ type: "tableRow", content: r })),
      },
    ],
  });
  return EditorState.create({ schema, doc, plugins: [formulaPlugin()] });
}

/** Apply a transaction and let the plugin append to it, exactly as a view does. */
function apply(state: EditorState, build: (s: EditorState) => EditorState["tr"]) {
  return state.apply(build(state));
}

/** Put the caret in a cell and then take it out again — which is how a typed
 *  formula is committed, and the only way to reach that path honestly. A
 *  freshly created state already has its selection at the start of the doc, so
 *  "select position 1" moves nothing and the plugin rightly does nothing. */
function typeAndLeave(state: EditorState, row: number, col: number) {
  const inside = apply(state, (s) =>
    s.tr.setSelection(TextSelection.create(s.doc, caretIn(s, row, col))),
  );
  return apply(inside, (s) => s.tr.setSelection(TextSelection.create(s.doc, 1)));
}

const grid = (state: EditorState) => {
  let out: ReturnType<typeof readTable> | null = null;
  state.doc.descendants((node, pos) => {
    if (node.type.spec.tableRole === "table" && !out) out = readTable(node, pos);
    return true;
  });
  return out!;
};

const textAt = (state: EditorState, row: number, col: number) =>
  grid(state).at(row, col)?.text ?? "";
const formulaAt = (state: EditorState, row: number, col: number) =>
  grid(state).at(row, col)?.formula ?? null;

/** The position just inside a cell's paragraph. */
const caretIn = (state: EditorState, row: number, col: number) =>
  (grid(state).at(row, col)?.pos ?? 0) + 2;

describe("the recalculation pass", () => {
  it("commits a typed formula when the caret leaves the cell", () => {
    const state = stateOf([[cell("2"), cell("3"), cell("=A1+B1")]]);
    const next = typeAndLeave(state, 0, 2);
    expect(formulaAt(next, 0, 2)).toBe("=A1+B1");
    expect(textAt(next, 0, 2)).toBe("5");
  });

  it("leaves the cell alone while the caret is still in it", () => {
    // Committing on every keystroke would evaluate "=SU" and replace what is
    // being typed with an error halfway through the word.
    const state = stateOf([[cell("2"), cell("3"), cell("=A1+")]]);
    const next = apply(state, (s) =>
      s.tr.setSelection(TextSelection.create(s.doc, caretIn(s, 0, 2))),
    );
    expect(textAt(next, 0, 2)).toBe("=A1+");
    expect(formulaAt(next, 0, 2)).toBeNull();
  });

  it("settles — a second pass over its own output appends nothing", () => {
    // The loop guard. If this ever returns a transaction, the editor recurses
    // until the stack ends and the note is unusable.
    const state = stateOf([[cell("2"), cell("3"), cell("=A1+B1")]]);
    const once = typeAndLeave(state, 0, 2);
    expect(textAt(once, 0, 2)).toBe("5");

    const twice = apply(once, (s) => s.tr.setSelection(TextSelection.create(s.doc, 3)));
    expect(textAt(twice, 0, 2)).toBe("5");
    expect(formulaAt(twice, 0, 2)).toBe("=A1+B1");
  });

  it("re-runs when a cell the formula READS changes", () => {
    const state = stateOf([[cell("2"), cell("3"), cell("5", "=A1+B1")]]);
    const next = apply(state, (s) => {
      const a = grid(s).at(0, 0)!;
      return s.tr.insertText("40", a.pos + 1, a.pos + 1 + a.node.content.size);
    });
    expect(textAt(next, 0, 2)).toBe("43");
  });

  it("KEEPS a formula when the coach types somewhere else in the note", () => {
    // The bug this is here for: a committed formula's text is its value and
    // never starts with "=", so a naive "no equals sign, drop the formula"
    // wiped every formula in the document on any keystroke anywhere.
    const state = stateOf([[cell("2"), cell("3"), cell("5", "=A1+B1")]]);
    const next = apply(state, (s) => s.tr.insertText(" more", 6));
    expect(formulaAt(next, 0, 2)).toBe("=A1+B1");
    expect(textAt(next, 0, 2)).toBe("5");
  });

  it("drops the formula when that cell itself is typed over", () => {
    // Which is what Excel does. A formula that survived being overwritten
    // would immediately overwrite the coach right back.
    const state = stateOf([[cell("2"), cell("3"), cell("5", "=A1+B1")]]);
    const next = apply(state, (s) => {
      const c = grid(s).at(0, 2)!;
      return s.tr.insertText("mine", c.pos + 1, c.pos + 1 + c.node.content.size);
    });
    expect(formulaAt(next, 0, 2)).toBeNull();
    expect(textAt(next, 0, 2)).toBe("mine");
  });

  it("does nothing at all to a table with no formulas in it", () => {
    const state = stateOf([[cell("Player"), cell("Distance")], [cell("#7"), cell("11.4 km")]]);
    const next = apply(state, (s) => s.tr.insertText("!", 6));
    expect(textAt(next, 1, 1)).toBe("11.4 km");
    expect(next.doc.firstChild!.textContent).toBe("notes!");
  });
});
