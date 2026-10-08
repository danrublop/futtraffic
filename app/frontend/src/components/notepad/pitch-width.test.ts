import { describe, expect, it } from "vitest";

import { PitchNode } from "./pitch-node";

/** A BOARD'S WIDTH HAS TO SURVIVE THE ROUND TRIP.
 *
 *  The same class of bug as the payload whitelist in query-node: an attribute
 *  added to a node and to its view, and not to the half that reads a stored
 *  file back, serializes correctly and exports correctly and then reverts the
 *  moment anybody reopens the note. That one cost eight figures.
 *
 *  Exercised through the node's own attribute spec rather than through a live
 *  editor, because that is where the asymmetry lives: `renderHTML` writes
 *  `data-width` and `parseHTML` has to read the same name back.
 */
const attrs = PitchNode.config.addAttributes!.call(
  // The spec only reads `this.options` for other attributes; width needs none.
  { options: {}, name: "pitch", parent: undefined } as never,
) as Record<string, { default: unknown; parseHTML: (el: HTMLElement) => unknown; renderHTML: (a: Record<string, unknown>) => Record<string, string> }>;

const el = (v: string | null) => {
  const d = { getAttribute: (k: string) => (k === "data-width" ? v : null) };
  return d as unknown as HTMLElement;
};

describe("a board's width", () => {
  it("writes data-width and reads the same name back", () => {
    const written = attrs.width.renderHTML({ width: 50 });
    expect(written).toEqual({ "data-width": "50" });
    expect(attrs.width.parseHTML(el("50"))).toBe(50);
  });

  it("omits the attribute at full width, so old notes are untouched", () => {
    // Every board written before this attribute existed carries no data-width,
    // and none of them meant "a quarter of the column".
    expect(attrs.width.renderHTML({ width: 100 })).toEqual({});
    expect(attrs.width.parseHTML(el(null))).toBe(100);
  });

  it("refuses a width outside the range the handles can produce", () => {
    // A hand-edited file, or one from a future that widened the range. 100 is
    // the safe direction: a figure too big is visible, one at 2% is a dot.
    for (const bad of ["0", "5", "180", "-40", "wide", ""]) {
      expect(attrs.width.parseHTML(el(bad))).toBe(100);
    }
    for (const good of ["25", "60", "100"]) {
      expect(attrs.width.parseHTML(el(good))).toBe(Number(good));
    }
  });
});
