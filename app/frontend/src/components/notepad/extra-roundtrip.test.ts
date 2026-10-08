import { describe, expect, it } from "vitest";

import { parseExtra } from "./query-node";
import type { Extra } from "./query-data";

/** THE TRIP A FIGURE ACTUALLY MAKES, and the one nothing was checking.
 *
 *  A block is written as ProseMirror JSON, serialized to HTML with its payload
 *  in `data-extra`, stored, and then PARSED BACK by the editor the next time
 *  anyone opens the note. `parseExtra` is a whitelist — it validates one
 *  `kind` at a time and returns null for anything it does not recognise, which
 *  is the right shape for something reading a file a human may have edited.
 *
 *  It also means adding a figure is a TWO-PLACE change, and the second place
 *  is invisible until you reopen a note. Three payload kinds — pizza, freeze,
 *  passes — were added to the renderers and to the Extra union and not here,
 *  so eight figures serialized correctly, exported to PDF correctly, and then
 *  fell back to a bar chart the moment the editor read them back. The PDF was
 *  right and the editor was wrong, which is the hardest version to notice.
 *
 *  So this test is not "does pizza parse". It is: EVERY kind in the union
 *  survives the trip, enumerated from a list that a new kind has to be added
 *  to. The list is the point.
 */

/** One minimal specimen per `kind`. A new member of the Extra union with no
 *  entry here fails the completeness test at the bottom. */
const SPECIMENS: Record<Extra["kind"], Extra> = {
  shots: {
    kind: "shots",
    home: "H",
    away: "A",
    points: [{ x: 80, y: 50, side: "home", goal: true, period: 1 }],
  },
  momentum: { kind: "momentum", values: [1, -2, 3], goals: [{ minute: 12, side: "home" }], home: "H", away: "A" },
  spots: {
    kind: "spots",
    spots: [{ x: 30, y: 40, side: "home", label: "6" }],
    home: "H",
    away: "A",
    measured: true,
  },
  heat: { kind: "heat", bins: [1, 2, 3], cols: 3, rows: 1, total: 6, who: "H" },
  zones: { kind: "zones", axis: "length", who: "H", bands: [{ label: "Def", share: 33 }] },
  goal: { kind: "goal", who: "N", shots: [{ x: 20, y: 30, scored: true, label: "1" }] },
  pizza: {
    kind: "pizza",
    who: "Brandt",
    population: 214,
    slices: [{ label: "Distance", pct: 88, text: "12.1 km", group: "Volume" }],
  },
  freeze: {
    kind: "freeze",
    t_s: 1287,
    players: [{ x: 30, y: 40, side: "home", label: "6" }],
    ball: { x: 50, y: 50 },
    seen: 1,
    squad: 22,
    home: "H",
    away: "A",
    synthetic: true,
  },
  passes: {
    kind: "passes",
    passes: [{ x0: 20, y0: 30, x1: 40, y1: 50, side: "home", from: "6", to: "8", ok: true }],
    home: "H",
    away: "A",
    dropped: 12,
    synthetic: true,
  },
};

const trip = (e: Extra) => parseExtra(JSON.stringify(e));

describe("every payload survives being written and read back", () => {
  for (const [kind, specimen] of Object.entries(SPECIMENS)) {
    it(`keeps a ${kind} a ${kind}`, () => {
      const back = trip(specimen);
      // Null is the failure that matters: chartSvg falls through to bars, so
      // the figure silently becomes a different figure rather than erroring.
      expect(back, `${kind} did not survive parseExtra — the editor will draw bars`).not.toBeNull();
      expect(back?.kind).toBe(kind);
    });
  }

  it("covers every kind the union declares", () => {
    // The guard on the guard. A new Extra member without a specimen above is a
    // compile error on SPECIMENS' type; this catches the runtime half, where
    // somebody adds a kind string that parseExtra has never heard of.
    const declared = Object.keys(SPECIMENS).sort();
    expect(declared.length).toBeGreaterThanOrEqual(9);
    for (const kind of declared) expect(trip(SPECIMENS[kind as Extra["kind"]])).not.toBeNull();
  });
});

describe("what the payloads keep, and what they recompute", () => {
  it("keeps a pizza's slices, its population and its bands", () => {
    const back = trip(SPECIMENS.pizza);
    expect(back).toMatchObject({ kind: "pizza", who: "Brandt", population: 214 });
    expect((back as { slices: unknown[] }).slices[0]).toMatchObject({
      label: "Distance",
      pct: 88,
      text: "12.1 km",
      group: "Volume",
    });
  });

  it("recomputes a freeze frame's `seen` from the players that survived", () => {
    // The figure's own honesty line. A file hand-edited to claim 22 while
    // carrying one player would otherwise print a claim the picture does not
    // support.
    const lying = { ...SPECIMENS.freeze, seen: 22 };
    expect(trip(lying)).toMatchObject({ seen: 1, squad: 22 });
  });

  it("adds passes it had to drop to the count the file already carried", () => {
    const one = SPECIMENS.passes as Extract<Extra, { kind: "passes" }>;
    const many = {
      ...one,
      dropped: 5,
      passes: Array.from({ length: 450 }, () => one.passes[0]),
    };
    const back = trip(many) as { passes: unknown[]; dropped: number };
    expect(back.passes).toHaveLength(400);
    expect(back.dropped).toBe(5 + 50);
  });

  it("is null, never an exception, for rubbish", () => {
    for (const bad of ["", "{", "null", '{"kind":"nope"}', '{"kind":"pizza"}']) {
      expect(parseExtra(bad)).toBeNull();
    }
  });
});
