import { describe, expect, it } from "vitest";

import { columnsContent, columnsOf } from "./columns-node";

/** The schema says `column{2,4}`. Everything below is about that string: a row
 *  ProseMirror rejects does not throw, it inserts NOTHING, so a payload one
 *  column short is a click that silently does nothing at all. */
describe("building a row of columns", () => {
  it("gives an empty row the two columns the schema demands", () => {
    const c = columnsContent(2);
    expect(c.content).toHaveLength(2);
    expect(c.content[0]).toEqual({ type: "column", content: [{ type: "paragraph" }] });
  });

  it("clamps to what a page can hold, in both directions", () => {
    expect(columnsContent(1).content).toHaveLength(2);
    expect(columnsContent(9).content).toHaveLength(4);
  });

  it("puts each block in its own column", () => {
    const c = columnsOf([{ type: "query" }, { type: "figure" }]);
    expect(c.content.map((col) => col.content[0])).toEqual([{ type: "query" }, { type: "figure" }]);
  });

  it("pads a single block up to two columns rather than being dropped whole", () => {
    const c = columnsOf([{ type: "figure" }]);
    expect(c.content).toHaveLength(2);
    expect(c.content[1].content[0]).toEqual({ type: "paragraph" });
  });

  it("never builds a row wider than four, whatever it is handed", () => {
    const c = columnsOf(Array.from({ length: 7 }, (_, i) => ({ type: "p", i })));
    expect(c.content).toHaveLength(4);
    expect(c.content.map((col) => col.content[0])).toEqual([
      { type: "p", i: 0 }, { type: "p", i: 1 }, { type: "p", i: 2 }, { type: "p", i: 3 },
    ]);
  });
});
