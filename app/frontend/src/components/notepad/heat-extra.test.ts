import { describe, expect, it } from "vitest";

import { HEAT_COLS, HEAT_ROWS, binHeat } from "./query-data";
import { parseExtra } from "./query-node";
import { heatSvg } from "./chart-svg";

/** The heat map's stored shape, and the notes saved before it changed.
 *
 *  One heat map used to be 423 KB of raw `{x,y}` inside an HTML attribute —
 *  140x the next largest block in a note, and two of them would pass the
 *  server's 1 MB cap and fail the save. The renderer only ever binned them to
 *  a fixed grid, so the counts are what the document keeps now.
 *
 *  What is guarded here is the MIGRATION as much as the encoding: two saved
 *  notes still hold the old form, and a parser that stopped understanding it
 *  would make those figures vanish on open.
 */

const asAttr = (o: unknown) => JSON.stringify(o);

describe("the stored heat map", () => {
  it("reads the binned form back as it was written", () => {
    const bins = binHeat([{ x: 5, y: 5 }, { x: 95, y: 95 }]);
    const back = parseExtra(
      asAttr({ kind: "heat", bins, cols: HEAT_COLS, rows: HEAT_ROWS, total: 2, who: "Rovers" }),
    );
    expect(back).toEqual({ kind: "heat", bins, cols: HEAT_COLS, rows: HEAT_ROWS, total: 2, who: "Rovers" });
  });

  it("still reads a note saved with raw points, and bins it on the way in", () => {
    const points = [{ x: 5, y: 5 }, { x: 5, y: 6 }, { x: 95, y: 95 }];
    const back = parseExtra(asAttr({ kind: "heat", points, who: "Rovers" }));
    expect(back?.kind).toBe("heat");
    const h = back as Extract<typeof back, { kind: "heat" }>;
    expect(h.total).toBe(3);
    expect(h.bins[0]).toBe(2);
    expect(h.bins[h.bins.length - 1]).toBe(1);
    expect(h.bins).toHaveLength(HEAT_COLS * HEAT_ROWS);
  });

  it("draws the old form and the new one identically", () => {
    // The point of the migration: same picture, a thousandth of the bytes.
    const points = [{ x: 5, y: 5 }, { x: 5, y: 6 }, { x: 40, y: 50 }, { x: 95, y: 95 }];
    const old = parseExtra(asAttr({ kind: "heat", points, who: "Rovers" }));
    const now = parseExtra(
      asAttr({
        kind: "heat",
        bins: binHeat(points),
        cols: HEAT_COLS,
        rows: HEAT_ROWS,
        total: points.length,
        who: "Rovers",
      }),
    );
    expect(heatSvg(old as never)).toBe(heatSvg(now as never));
  });

  it("is dramatically smaller than what it replaced", () => {
    // The number that made this worth doing. 8,000 points was the old cap.
    const points = Array.from({ length: 8000 }, (_, i) => ({
      x: (i * 7.13) % 100,
      y: (i * 3.37) % 100,
    }));
    const before = asAttr({ kind: "heat", points, who: "Rovers" }).length;
    const after = asAttr({
      kind: "heat",
      bins: binHeat(points),
      cols: HEAT_COLS,
      rows: HEAT_ROWS,
      total: points.length,
      who: "Rovers",
    }).length;
    expect(before).toBeGreaterThan(200_000);
    expect(after).toBeLessThan(1_000);
  });

  it("keeps the grid the figure was binned at, not this file's current one", () => {
    // A note saved against a different resolution must not be re-shaped to
    // whatever the constants say today.
    const back = parseExtra(
      asAttr({ kind: "heat", bins: [1, 2, 3, 4], cols: 2, rows: 2, total: 10, who: "Rovers" }),
    );
    const h = back as Extract<typeof back, { kind: "heat" }>;
    expect([h.cols, h.rows]).toEqual([2, 2]);
    expect(h.bins).toEqual([1, 2, 3, 4]);
  });

  it("pads a short bins array rather than refusing the figure", () => {
    const h = parseExtra(asAttr({ kind: "heat", bins: [5], cols: 2, rows: 2, total: 5, who: "R" }));
    expect((h as Extract<typeof h, { kind: "heat" }>).bins).toEqual([5, 0, 0, 0]);
  });

  it("falls back to the counted total when the stored one is missing", () => {
    const h = parseExtra(asAttr({ kind: "heat", bins: [2, 3], cols: 2, rows: 1, who: "R" }));
    expect((h as Extract<typeof h, { kind: "heat" }>).total).toBe(5);
  });

  it("refuses a heat payload with neither bins nor points", () => {
    expect(parseExtra(asAttr({ kind: "heat", who: "Rovers" }))).toBeNull();
  });
});
