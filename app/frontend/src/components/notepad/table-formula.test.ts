import { getSchema } from "@tiptap/core";
import Document from "@tiptap/extension-document";
import Paragraph from "@tiptap/extension-paragraph";
import Text from "@tiptap/extension-text";
import { Node as PMNode } from "@tiptap/pm/model";
import { describe, expect, it } from "vitest";

import { TABLE_EXTENSIONS } from "./table-grid";
import { a1, cellValue, colLabel, readTable, recalcTable } from "./table-formula";

/** The evaluator, over a real ProseMirror table.
 *
 *  Everything here goes through `recalcTable` rather than the parser directly,
 *  because the parser is a dependency and works — what is ours is the bridge:
 *  A1 addresses off TableMap, the unit-stripping coercion, the dependency
 *  order, and the refusal to hang on a cycle.
 */

const schema = getSchema([Document, Paragraph, Text, ...TABLE_EXTENSIONS]);

/** A table from a grid of strings. A cell starting with `=` is a formula, and
 *  arrives the way a committed one does: the attribute set, the text stale. */
function table(rows: string[][]) {
  const doc = PMNode.fromJSON(schema, {
    type: "doc",
    content: [
      {
        type: "table",
        attrs: { layout: false },
        content: rows.map((cells) => ({
          type: "tableRow",
          content: cells.map((c) => ({
            type: "tableCell",
            attrs: c.startsWith("=") ? { formula: c } : {},
            content: [c ? { type: "paragraph", content: [{ type: "text", text: c }] } : { type: "paragraph" }],
          })),
        })),
      },
    ],
  });
  return { doc, table: doc.firstChild!, start: 0 };
}

/** Run one pass and read the answers back by address. */
function values(rows: string[][]): Record<string, string> {
  const t = table(rows);
  const out: Record<string, string> = {};
  const grid = readTable(t.table, t.start);
  for (const u of recalcTable(t.table, t.start)) {
    const cell = grid.cells.find((c) => c.pos === u.pos)!;
    out[a1(cell.row, cell.col)] = u.text;
  }
  return out;
}

describe("A1 addressing", () => {
  it("counts columns the way a spreadsheet does, past Z", () => {
    expect([0, 1, 25, 26, 27, 51, 52].map(colLabel)).toEqual(["A", "B", "Z", "AA", "AB", "AZ", "BA"]);
  });

  it("numbers rows from one, header row included — the header IS row 1", () => {
    expect(a1(0, 0)).toBe("A1");
    expect(a1(9, 2)).toBe("C10");
  });
});

describe("reading a cell as a value", () => {
  it("takes the number and ignores the unit, which Excel would refuse", () => {
    // The whole product prints measured figures with what they are measured
    // in. #VALUE! on every real column would make formulas useless here.
    expect(cellValue("13.47 km")).toBe(13.47);
    expect(cellValue("24.1 km/h")).toBe(24.1);
    expect(cellValue("90 min")).toBe(90);
    expect(cellValue("1,234.5 km")).toBe(1234.5);
    expect(cellValue("-3.2 m")).toBe(-3.2);
  });

  it("leaves text as text, so a sum over names is 0 and not a lie", () => {
    expect(cellValue("Varga")).toBe("Varga");
    expect(cellValue("#7")).toBe("#7");
    expect(cellValue("")).toBe("");
  });
});

describe("recalculating a table", () => {
  it("sums a column of measured values, units and all", () => {
    expect(
      values([
        ["Player", "Distance"],
        ["#7", "11.40 km"],
        ["#9", "10.20 km"],
        ["Total", "=SUM(B2:B3)"],
      ]),
    ).toEqual({ B4: "21.6" });
  });

  it("does arithmetic across columns", () => {
    expect(
      values([
        ["a", "b", "c"],
        ["10", "4", "=B2/A2*100"],
      ]),
    ).toEqual({ C2: "40" });
  });

  it("evaluates in dependency order, not document order", () => {
    // C1 reads B1, which is itself a formula further DOWN the table. Reading
    // the stale text under B1 would answer 0.
    const out = values([["=B1*2", "=SUM(A2:A3)", ""], ["3", "", ""], ["4", "", ""]]);
    expect(out.B1).toBe("7");
    expect(out.A1).toBe("14");
  });

  it("says #CIRCULAR! instead of hanging", () => {
    const out = values([["=B1", "=A1"]]);
    expect(out.A1).toBe("#CIRCULAR!");
    expect(out.B1).toBe("#CIRCULAR!");
  });

  it("reports Excel's own errors rather than swallowing them", () => {
    expect(values([["10", "0", "=A1/B1"]]).C1).toBe("#DIV/0!");
  });

  it("carries a function that is not arithmetic", () => {
    expect(values([["12", '=IF(A1>11,"high","low")']]).B1).toBe("high");
    expect(values([["9", '=IF(A1>11,"high","low")']]).B1).toBe("low");
  });

  it("does not print floating-point dust", () => {
    // 0.1 + 0.2 is 0.30000000000000004 and no spreadsheet shows that.
    expect(values([["0.1", "0.2", "=A1+B1"]]).C1).toBe("0.3");
  });

  it("changes nothing when the answer already stands", () => {
    // The guarantee the recalculation plugin's loop protection rests on.
    const t = table([["2", "3", "=A1+B1"]]);
    const first = recalcTable(t.table, t.start);
    expect(first).toHaveLength(1);
    expect(first[0].text).toBe("5");

    const settled = table([["2", "3", "=A1+B1"]]);
    // Same table, but the formula cell already holds its answer.
    const doc = PMNode.fromJSON(schema, {
      type: "doc",
      content: [
        {
          type: "table",
          attrs: { layout: false },
          content: [
            {
              type: "tableRow",
              content: [
                { type: "tableCell", attrs: {}, content: [{ type: "paragraph", content: [{ type: "text", text: "2" }] }] },
                { type: "tableCell", attrs: {}, content: [{ type: "paragraph", content: [{ type: "text", text: "3" }] }] },
                {
                  type: "tableCell",
                  attrs: { formula: "=A1+B1" },
                  content: [{ type: "paragraph", content: [{ type: "text", text: "5" }] }],
                },
              ],
            },
          ],
        },
      ],
    });
    expect(settled.table).toBeTruthy();
    expect(recalcTable(doc.firstChild!, 0)).toEqual([]);
  });

  it("leaves a table with no formulas completely alone", () => {
    const t = table([
      ["Player", "Distance"],
      ["#7", "11.40 km"],
    ]);
    expect(recalcTable(t.table, t.start)).toEqual([]);
  });
});
