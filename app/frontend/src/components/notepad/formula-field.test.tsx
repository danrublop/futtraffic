import { Editor } from "@tiptap/core";
import Document from "@tiptap/extension-document";
import Paragraph from "@tiptap/extension-paragraph";
import Text from "@tiptap/extension-text";
import { TextSelection } from "@tiptap/pm/state";
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { FormulaField } from "./formula-field";
import { TABLE_EXTENSIONS } from "./table-grid";
import { readTable } from "./table-formula";

/** The formula bar, against a real editor.
 *
 *  This file is the reason the suite grew a jsdom half. The bar was written and
 *  checked by hand in a browser, which catches "it renders" and nothing about
 *  what happens when the caret moves while a formula is half-typed.
 *
 *  The editor here is a real TipTap instance with the real table extensions, so
 *  the recalculation plugin is running too: writing a formula through the bar
 *  and reading the CELL back is an end-to-end assertion, not a mock agreeing
 *  with itself.
 */

let editor: Editor | null = null;

afterEach(() => {
  editor?.destroy();
  editor = null;
});

/** An empty string is an EMPTY PARAGRAPH, never a paragraph holding "".
 *  ProseMirror rejects a zero-length text node, and TipTap answers an invalid
 *  document by quietly parsing an empty one — so getting this wrong does not
 *  throw, it hands every test a doc with no table in it. */
const cell = (text: string) => ({
  type: "tableCell",
  content: [text ? { type: "paragraph", content: [{ type: "text", text }] } : { type: "paragraph" }],
});

/** An editor holding one paragraph and one 1x3 table of numbers. */
function makeEditor() {
  editor = new Editor({
    extensions: [Document, Paragraph, Text, ...TABLE_EXTENSIONS],
    content: {
      type: "doc",
      content: [
        { type: "paragraph", content: [{ type: "text", text: "notes" }] },
        {
          type: "table",
          attrs: { layout: false },
          content: [{ type: "tableRow", content: [cell("2"), cell("3"), cell("")] }],
        },
      ],
    },
  });
  return editor;
}

const grid = (e: Editor) => {
  let out: ReturnType<typeof readTable> | null = null;
  e.state.doc.descendants((node, pos) => {
    if (node.type.spec.tableRole === "table" && !out) out = readTable(node, pos);
    return true;
  });
  return out!;
};

/** Put the caret inside the cell at (row, col), the way clicking it would. */
function caretTo(e: Editor, row: number, col: number) {
  const c = grid(e).at(row, col)!;
  e.view.dispatch(e.state.tr.setSelection(TextSelection.create(e.state.doc, c.pos + 2)));
}

const textAt = (e: Editor, row: number, col: number) => grid(e).at(row, col)?.text ?? "";
const formulaAt = (e: Editor, row: number, col: number) => grid(e).at(row, col)?.formula ?? null;

describe("the formula bar", () => {
  it("shows nothing at all when the caret is outside a table", () => {
    const e = makeEditor();
    e.view.dispatch(e.state.tr.setSelection(TextSelection.create(e.state.doc, 1)));
    render(<FormulaField editor={e} />);
    expect(screen.queryByRole("textbox")).toBeNull();
  });

  it("names the cell the caret is in, in A1", () => {
    const e = makeEditor();
    caretTo(e, 0, 2);
    render(<FormulaField editor={e} />);
    expect(screen.getByText("C1")).toBeTruthy();
    expect(screen.getByLabelText("Formula for C1")).toBeTruthy();
  });

  it("writes a formula into the cell, and the plugin computes it", () => {
    const e = makeEditor();
    caretTo(e, 0, 2);
    render(<FormulaField editor={e} />);
    const box = screen.getByLabelText("Formula for C1") as HTMLInputElement;

    fireEvent.change(box, { target: { value: "=A1+B1" } });
    fireEvent.keyDown(box, { key: "Enter" });

    expect(formulaAt(e, 0, 2)).toBe("=A1+B1");
    // The cell, not the attribute: this is the recalculation plugin running.
    expect(textAt(e, 0, 2)).toBe("5");
  });

  it("shows the formula back, not the value the cell now reads", () => {
    // The whole reason the bar exists. A cell showing 5 has nowhere to show
    // =A1+B1, and no way to fix a typo in it.
    const e = makeEditor();
    caretTo(e, 0, 2);
    const view = render(<FormulaField editor={e} />);
    const box = screen.getByLabelText("Formula for C1") as HTMLInputElement;
    fireEvent.change(box, { target: { value: "=A1+B1" } });
    fireEvent.keyDown(box, { key: "Enter" });

    caretTo(e, 0, 2);
    view.rerender(<FormulaField editor={e} />);
    expect((screen.getByLabelText("Formula for C1") as HTMLInputElement).value).toBe("=A1+B1");
    expect(textAt(e, 0, 2)).toBe("5");
  });

  it("clears the formula when the bar is emptied", () => {
    const e = makeEditor();
    caretTo(e, 0, 2);
    const view = render(<FormulaField editor={e} />);
    let box = screen.getByLabelText("Formula for C1") as HTMLInputElement;
    fireEvent.change(box, { target: { value: "=A1+B1" } });
    fireEvent.keyDown(box, { key: "Enter" });

    caretTo(e, 0, 2);
    view.rerender(<FormulaField editor={e} />);
    box = screen.getByLabelText("Formula for C1") as HTMLInputElement;
    fireEvent.change(box, { target: { value: "" } });
    fireEvent.keyDown(box, { key: "Enter" });
    expect(formulaAt(e, 0, 2)).toBeNull();
  });

  it("abandons a half-typed formula rather than carrying it to the next cell", () => {
    // The bug the cell-tagged draft exists to prevent: type into C1's bar,
    // move to A1, and the draft must not follow — writing it there would put a
    // formula in a cell nobody meant.
    const e = makeEditor();
    caretTo(e, 0, 2);
    const view = render(<FormulaField editor={e} />);
    fireEvent.change(screen.getByLabelText("Formula for C1"), { target: { value: "=SUM(" } });

    caretTo(e, 0, 0);
    view.rerender(<FormulaField editor={e} />);
    const box = screen.getByLabelText("Formula for A1") as HTMLInputElement;
    expect(box.value).toBe("");
    expect(formulaAt(e, 0, 0)).toBeNull();
  });

  it("takes Escape as abandon, leaving the cell as it was", () => {
    const e = makeEditor();
    caretTo(e, 0, 2);
    const view = render(<FormulaField editor={e} />);
    const box = screen.getByLabelText("Formula for C1") as HTMLInputElement;
    fireEvent.change(box, { target: { value: "=A1+B1" } });
    fireEvent.keyDown(box, { key: "Escape" });

    view.rerender(<FormulaField editor={e} />);
    expect(formulaAt(e, 0, 2)).toBeNull();
    expect(textAt(e, 0, 2)).toBe("");
  });

  it("refuses to store something that is not a formula", () => {
    const e = makeEditor();
    caretTo(e, 0, 2);
    render(<FormulaField editor={e} />);
    const box = screen.getByLabelText("Formula for C1") as HTMLInputElement;
    fireEvent.change(box, { target: { value: "just a note" } });
    fireEvent.keyDown(box, { key: "Enter" });
    expect(formulaAt(e, 0, 2)).toBeNull();
  });
});
