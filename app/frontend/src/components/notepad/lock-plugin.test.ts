import { getSchema } from "@tiptap/core";
import Document from "@tiptap/extension-document";
import Paragraph from "@tiptap/extension-paragraph";
import Text from "@tiptap/extension-text";
import { Node as PMNode } from "@tiptap/pm/model";
import { EditorState, TextSelection } from "@tiptap/pm/state";
import { describe, expect, it } from "vitest";

import { formulaKey } from "./formula-plugin";
import { formulaPlugin } from "./formula-plugin";
import { ALLOW, allows, inLockedTable, lockPlugin, tableUnder } from "./lock-plugin";
import { BAR_WIDTH, TABLE_EXTENSIONS, lockedTableFromRows } from "./table-grid";
import { readTable } from "./table-formula";

/** A locked table holds still.
 *
 *  What is guarded here is that "locked" means the CONTENT, not the node: the
 *  table must still be unlockable, still refreshable by the menu, and its
 *  formulas must keep answering — while typing, pasting, and every command on
 *  the table toolbar are refused.
 *
 *  `filterTransaction` and not `contenteditable="false"`, because the second
 *  only stops the keyboard. Each case below is a way to write into a table
 *  WITHOUT typing in it.
 */

const schema = getSchema([Document, Paragraph, Text, ...TABLE_EXTENSIONS]);

function stateOf(locked: boolean, opts: { figure?: string } = {}) {
  const table = lockedTableFromRows({
    head: ["Player", "Distance"],
    rows: [
      ["#7", "11.4"],
      ["#9", "10.2"],
    ],
    figure: opts.figure,
  }) as unknown as Record<string, unknown>;
  const doc = PMNode.fromJSON(schema, {
    type: "doc",
    content: [
      { type: "paragraph", content: [{ type: "text", text: "notes" }] },
      { ...table, attrs: { ...(table.attrs as object), locked } },
    ],
  });
  return EditorState.create({ schema, doc, plugins: [formulaPlugin(), lockPlugin()] });
}

const grid = (state: EditorState) => {
  let out: ReturnType<typeof readTable> | null = null;
  state.doc.descendants((node, pos) => {
    if (node.type.spec.tableRole === "table" && !out) out = readTable(node, pos);
    return true;
  });
  return out!;
};

/** Try to write into a cell. Returns the state after — unchanged if refused. */
function typeInto(state: EditorState, row: number, col: number, text: string, allow = false) {
  const c = grid(state).at(row, col)!;
  let tr = state.tr.insertText(text, c.pos + 1, c.pos + 1 + c.node.content.size);
  if (allow) tr = tr.setMeta(ALLOW, true);
  // A view consults the plugin before applying; applying directly would
  // bypass the rule this file exists to check.
  return allows(tr, state.doc) ? state.apply(tr) : state;
}

const textAt = (s: EditorState, r: number, c: number) => grid(s).at(r, c)?.text ?? "";
const tablePos = (s: EditorState) => {
  let pos = -1;
  s.doc.descendants((n, p) => {
    if (n.type.spec.tableRole === "table" && pos < 0) pos = p;
    return true;
  });
  return pos;
};

describe("a locked table", () => {
  it("refuses typing in a cell", () => {
    const s = stateOf(true);
    expect(textAt(typeInto(s, 1, 1, "999"), 1, 1)).toBe("11.4");
  });

  it("refuses a paste, a drop and anything else that writes without the keyboard", () => {
    const s = stateOf(true);
    const c = grid(s).at(1, 0)!;
    const tr = s.tr.replaceWith(
      c.pos + 1,
      c.pos + 1 + c.node.content.size,
      schema.nodes.paragraph.create(null, schema.text("pasted")),
    );
    expect(allows(tr, s.doc)).toBe(false);
  });

  it("refuses a structural command — deleting a row is still a write", () => {
    const s = stateOf(true);
    const table = tablePos(s);
    const node = s.doc.nodeAt(table)!;
    const row = node.firstChild!;
    const tr = s.tr.delete(table + 1, table + 1 + row.nodeSize);
    expect(allows(tr, s.doc)).toBe(false);
  });

  it("lets the table be UNLOCKED — the lock is on the content, not the node", () => {
    const s = stateOf(true);
    const tr = s.tr.setNodeAttribute(tablePos(s), "locked", false);
    expect(allows(tr, s.doc)).toBe(true);
    const next = s.apply(tr);
    expect(textAt(typeInto(next, 1, 1, "999"), 1, 1)).toBe("999");
  });

  it("lets the menu's own refresh through", () => {
    const s = stateOf(true);
    expect(textAt(typeInto(s, 1, 1, "9.9", true), 1, 1)).toBe("9.9");
  });

  it("lets the recalculation pass through, so a locked table can still hold formulas", () => {
    const s = stateOf(true);
    const c = grid(s).at(1, 1)!;
    const tr = s.tr
      .insertText("42", c.pos + 1, c.pos + 1 + c.node.content.size)
      .setMeta(formulaKey, true);
    expect(allows(tr, s.doc)).toBe(true);
  });

  it("leaves the rest of the document alone", () => {
    const s = stateOf(true);
    const tr = s.tr.insertText("!", 6);
    expect(allows(tr, s.doc)).toBe(true);
    expect(s.apply(tr).doc.firstChild!.textContent).toBe("notes!");
  });
});

describe("an unlocked table", () => {
  it("takes typing like any other", () => {
    const s = stateOf(false);
    expect(textAt(typeInto(s, 1, 1, "999"), 1, 1)).toBe("999");
  });
});

describe("what the menu asks", () => {
  it("finds the table under a position, and nothing outside one", () => {
    const s = stateOf(true);
    const c = grid(s).at(1, 0)!;
    expect(tableUnder(s, c.pos + 1)?.pos).toBe(tablePos(s));
    expect(tableUnder(s, 2)).toBeNull();
  });

  it("reports whether the caret is somewhere held still", () => {
    const s = stateOf(true);
    const c = grid(s).at(1, 0)!;
    const inside = s.apply(s.tr.setSelection(TextSelection.create(s.doc, c.pos + 2)));
    expect(inLockedTable(inside)).toBe(true);
    expect(inLockedTable(s.apply(s.tr.setSelection(TextSelection.create(s.doc, 2))))).toBe(false);
  });

  it("carries the question when there is one, and null when built by hand", () => {
    const withQ = stateOf(true, { figure: '{"kind":"playerTable","cols":"physical","limit":22}' });
    expect(withQ.doc.nodeAt(tablePos(withQ))!.attrs.figure).toContain("playerTable");
    const byHand = stateOf(false);
    expect(byHand.doc.nodeAt(tablePos(byHand))!.attrs.figure).toBeNull();
  });
});

describe("the share column", () => {
  it("adds a bar cell per row, clamped, and only when asked", () => {
    const t = lockedTableFromRows({
      head: ["Stat", "Home", "Away"],
      rows: [["Shots", "14", "9"], ["Corners", "6", "3"]],
      bars: [0.61, 2],
    }) as unknown as { content: { content: { attrs?: Record<string, unknown> }[] }[] };
    expect(t.content[0].content).toHaveLength(4);
    expect(t.content[1].content[3].attrs?.bar).toBe(0.61);
    expect(t.content[2].content[3].attrs?.bar).toBe(1);
    // A fixed width on EVERY cell of the column, header included. Without it
    // the column shares the fixed-layout table equally with the others and a
    // 96px drawing gets a quarter of the page.
    expect(t.content.map((r) => r.content[3].attrs?.colwidth)).toEqual([
      [BAR_WIDTH],
      [BAR_WIDTH],
      [BAR_WIDTH],
    ]);
    // The header has no share of its own.
    expect(t.content[0].content[3].attrs?.bar).toBeUndefined();
  });

  it("leaves the shape alone when no row has a share", () => {
    const t = lockedTableFromRows({
      head: ["Player", "Distance"],
      rows: [["#7", "11.4"]],
      bars: [null],
    }) as unknown as { content: { content: unknown[] }[] };
    expect(t.content[0].content).toHaveLength(2);
  });
});
