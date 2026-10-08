import { describe, expect, it } from "vitest";

import { parseExtra } from "./query-node";
import { goalMouthSvg } from "./chart-svg";
import type { GoalExtra } from "./query-data";

/** The goal mouth: the one figure in the report that is not on the ground.
 *
 *  What is guarded here is the thing that makes it worth having. A penalty map
 *  that only shows the ones that went IN is a picture of nothing - the question
 *  a coach brings to it is "where does he miss?" - so a shot outside the frame
 *  has to survive both the parser and the renderer, at the place it actually
 *  went.
 */

const attr = (o: unknown) => JSON.stringify(o);

const shots = (...s: GoalExtra["shots"]): GoalExtra => ({ kind: "goal", who: "x", shots: s });

describe("a shot in the goal frame, stored", () => {
  it("keeps a miss OUTSIDE the posts rather than clamping it onto them", () => {
    const e = parseExtra(
      attr({
        kind: "goal",
        shots: [
          { x: -18, y: 40, scored: false, label: "10" },
          { x: 50, y: 128, scored: false },
        ],
      }),
    );
    expect(e?.kind).toBe("goal");
    // Clamping to 0-100 would put every miss on the woodwork and make the
    // figure say the opposite of what happened.
    expect(e).toMatchObject({ shots: [{ x: -18, y: 40 }, { x: 50, y: 128 }] });
  });

  it("still bounds it, because a document is a file people edit", () => {
    const e = parseExtra(attr({ kind: "goal", shots: [{ x: 99999, y: -99999 }] }));
    expect(e).toMatchObject({ shots: [{ x: 170, y: -70 }] });
  });

  it("takes anything for a shot without throwing", () => {
    expect(parseExtra(attr({ kind: "goal", shots: [null, "x", {}] }))).toMatchObject({
      shots: [{ x: 0, y: 0, scored: false, label: "" }, {}, {}],
    });
    expect(parseExtra(attr({ kind: "goal" }))).toBeNull();
  });
});

describe("a shot in the goal frame, drawn", () => {
  it("fills a goal and rings a miss", () => {
    const svg = goalMouthSvg(shots(
      { x: 50, y: 50, scored: true, label: "" },
      { x: 50, y: 120, scored: false, label: "" },
    ));
    expect(svg).toContain('fill="#e11d26"');
    expect(svg).toContain('fill="#ffffff" stroke="#1b1f24"');
    expect(svg).toContain("Scored (1)");
    expect(svg).toContain("Missed (1)");
  });

  it("draws the misses first, so a goal is never hidden under one", () => {
    const svg = goalMouthSvg(shots(
      { x: 50, y: 50, scored: true, label: "" },
      { x: 51, y: 50, scored: false, label: "" },
    ));
    expect(svg.indexOf(">missed<")).toBeLessThan(svg.indexOf(">scored<"));
  });

  it("puts the shirt inside the disc", () => {
    const svg = goalMouthSvg(shots({ x: 50, y: 50, scored: true, label: "10" }));
    // Same x and y as the disc it belongs to: set above it, it collided with
    // every neighbour, and penalties cluster.
    const disc = /<circle cx="([\d.-]+)" cy="([\d.-]+)"/.exec(svg);
    expect(svg).toContain(`<text x="${disc?.[1]}" y="${disc?.[2]}"`);
    expect(svg).toContain(">10</text>");
  });
});
