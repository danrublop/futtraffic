import { describe, expect, it } from "vitest";

import { layoutRow, parseGrid, tableContent, tableFromRows } from "./table-grid";

/** The shapes handed to ProseMirror. Everything here is about the schema
 *  string `tableRow+` / `(tableCell | tableHeader)*` / `block+`: a payload the
 *  schema rejects does not throw, it inserts NOTHING, so a wrong shape reads as
 *  a button that quietly does nothing at all. */

const rowTypes = (t: ReturnType<typeof tableContent>) =>
  t.content.map((r) => r.content.map((c) => c.type));

describe("a blank grid", () => {
  it("makes the size that was asked for", () => {
    const t = tableContent({ rows: 3, cols: 4 });
    expect(t.content).toHaveLength(3);
    expect(t.content[0].content).toHaveLength(4);
  });

  it("makes the first row header cells only when asked", () => {
    expect(rowTypes(tableContent({ rows: 2, cols: 2, header: true }))).toEqual([
      ["tableHeader", "tableHeader"],
      ["tableCell", "tableCell"],
    ]);
    expect(rowTypes(tableContent({ rows: 2, cols: 2 }))).toEqual([
      ["tableCell", "tableCell"],
      ["tableCell", "tableCell"],
    ]);
  });

  it("never gives a layout table a header, whatever it is asked", () => {
    // A borderless row holding a chart and a table has no header row — the
    // cells are containers, not data.
    expect(rowTypes(tableContent({ rows: 1, cols: 2, header: true, layout: true }))).toEqual([
      ["tableCell", "tableCell"],
    ]);
    expect(tableContent({ rows: 1, cols: 2, layout: true }).attrs.layout).toBe(true);
  });

  it("gives every cell a block to hold, because `block+` will not take none", () => {
    const cells = tableContent({ rows: 1, cols: 2 }).content[0].content;
    expect(cells.every((c) => c.content.length === 1)).toBe(true);
    expect(cells[0].content[0]).toEqual({ type: "paragraph" });
  });

  it("writes text as a paragraph with a text node, never an empty one", () => {
    const t = tableContent({ rows: 1, cols: 2, cells: [["hi", ""]] });
    expect(t.content[0].content[0].content[0]).toEqual({
      type: "paragraph",
      content: [{ type: "text", text: "hi" }],
    });
    // ProseMirror rejects a text node holding "", so a blank cell is a bare
    // paragraph rather than a paragraph containing nothing.
    expect(t.content[0].content[1].content[0]).toEqual({ type: "paragraph" });
  });

  it("clamps to a size a page can hold, in both directions", () => {
    expect(tableContent({ rows: 0, cols: 0 }).content).toHaveLength(1);
    expect(tableContent({ rows: 99, cols: 99 }).content).toHaveLength(50);
    expect(tableContent({ rows: 1, cols: 99 }).content[0].content).toHaveLength(24);
  });
});

describe("a row of blocks, side by side", () => {
  it("puts each block in its own cell of one borderless row", () => {
    const t = layoutRow([{ type: "query" }, { type: "figure" }]);
    expect(t.attrs.layout).toBe(true);
    expect(t.content).toHaveLength(1);
    expect(t.content[0].content.map((c) => c.content[0])).toEqual([
      { type: "query" },
      { type: "figure" },
    ]);
  });

  it("pads a single block to two cells rather than being dropped whole", () => {
    const t = layoutRow([{ type: "figure" }]);
    expect(t.content[0].content).toHaveLength(2);
    expect(t.content[0].content[1].content[0]).toEqual({ type: "paragraph" });
  });

  it("takes a STACK in one cell, because what goes beside a chart is now three blocks", () => {
    const t = layoutRow([
      [{ type: "heading" }, { type: "table" }, { type: "paragraph" }],
      { type: "query" },
    ]);
    expect(t.content[0].content.map((c) => c.content.length)).toEqual([3, 1]);
    expect(t.content[0].content[0].content.map((b) => b.type)).toEqual([
      "heading",
      "table",
      "paragraph",
    ]);
  });

  it("still gives an empty cell a paragraph, stack or not", () => {
    const t = layoutRow([{ type: "pitch" }, []]);
    expect(t.content[0].content[1].content).toEqual([{ type: "paragraph" }]);
  });

  it("never lays out more than four across", () => {
    const t = layoutRow(Array.from({ length: 7 }, (_, i) => ({ type: "p", i })));
    expect(t.content[0].content).toHaveLength(4);
  });
});

describe("a query answer, as a table you can then edit", () => {
  it("takes the head as the header row and the rest as cells", () => {
    const t = tableFromRows(["Player", "Distance"], [["#8", "11.0 km"], ["#9", "5.0 km"]]);
    expect(rowTypes(t)[0]).toEqual(["tableHeader", "tableHeader"]);
    expect(t.content).toHaveLength(3);
    expect(t.content[1].content[1].content[0]).toEqual({
      type: "paragraph",
      content: [{ type: "text", text: "11.0 km" }],
    });
  });
});

describe("a grid on the clipboard", () => {
  it("reads tab-separated lines as rows and columns", () => {
    expect(parseGrid("Player\tKm\nVarga\t12.1\nOkafor\t11.4")).toEqual([
      ["Player", "Km"],
      ["Varga", "12.1"],
      ["Okafor", "11.4"],
    ]);
  });

  it("survives Windows and a trailing newline, which is how a copy usually ends", () => {
    expect(parseGrid("a\tb\r\nc\td\r\n")).toEqual([
      ["a", "b"],
      ["c", "d"],
    ]);
  });

  it("keeps an empty cell rather than closing the gap", () => {
    expect(parseGrid("a\tb\n\td")).toEqual([
      ["a", "b"],
      ["", "d"],
    ]);
  });

  it("declines a ragged paste — that is prose with a tab in it", () => {
    expect(parseGrid("a\tb\nc\td\te")).toBeNull();
  });

  it("declines one line, and declines one column", () => {
    expect(parseGrid("a\tb")).toBeNull();
    expect(parseGrid("just\nsome\nlines")).toBeNull();
  });

  it("hands tableFromRows a head and a body it can use directly", () => {
    const g = parseGrid("Player\tKm\nVarga\t12.1")!;
    const t = tableFromRows(g[0], g.slice(1));
    expect(t.content).toHaveLength(2);
    expect(t.content[0].content.map((c) => c.type)).toEqual(["tableHeader", "tableHeader"]);
  });
});
