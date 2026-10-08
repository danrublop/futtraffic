import { describe, expect, it } from "vitest";

import type { MatchReport, Play } from "@/lib/types";
import { DEFAULT_MAGNETS } from "./pitch-data";
import {
  EMPTY_FILTER,
  byClock,
  filterPlays,
  playBoard,
  playCaption,
  playChoices,
  plays,
} from "./play-data";

const play = (o: Partial<Play>): Play => ({
  id: "p1",
  type: "corner_attack",
  side: "home",
  period: 1,
  minute: 12,
  clock: "12:04",
  zone: "box",
  outcome: "shot",
  players: [7, 19],
  board: { magnets: DEFAULT_MAGNETS, shapes: [], view: "box", orient: "v" },
  t: 724,
  ...o,
});

const SET: Play[] = [
  play({ id: "a", type: "corner_attack", zone: "box", outcome: "goal", minute: 12, players: [7, 19] }),
  play({ id: "b", type: "counter_attack", zone: "middle_third", outcome: "shot", minute: 34, period: 1, players: [9] }),
  play({ id: "c", type: "corner_attack", zone: "box", outcome: "cleared", minute: 61, period: 2, side: "away", players: [19, 4] }),
  play({ id: "d", type: "high_press", zone: "middle_third", outcome: "lost", minute: 70, period: 2, players: [7] }),
];

const report = {
  meta: { home_name: "Westfield", away_name: "Rovers", facts: { plays: SET } },
} as unknown as MatchReport;

describe("finding a moment", () => {
  it("reads the plays off the match, and copes with a match that has none", () => {
    expect(plays(report)).toHaveLength(4);
    expect(plays(null)).toEqual([]);
    expect(plays({ meta: {} } as MatchReport)).toEqual([]);
  });

  it("returns everything when nothing is asked", () => {
    expect(filterPlays(SET, EMPTY_FILTER)).toHaveLength(4);
  });

  it("narrows on one axis", () => {
    expect(filterPlays(SET, { ...EMPTY_FILTER, type: "corner_attack" }).map((p) => p.id)).toEqual(["a", "c"]);
    expect(filterPlays(SET, { ...EMPTY_FILTER, outcome: "goal" }).map((p) => p.id)).toEqual(["a"]);
    expect(filterPlays(SET, { ...EMPTY_FILTER, period: "2" }).map((p) => p.id)).toEqual(["c", "d"]);
  });

  it("stacks axes — 'set pieces in the box, second half'", () => {
    const got = filterPlays(SET, { ...EMPTY_FILTER, type: "corner_attack", zone: "box", period: "2" });
    expect(got.map((p) => p.id)).toEqual(["c"]);
  });

  it("matches a player by membership, because several were involved", () => {
    expect(filterPlays(SET, { ...EMPTY_FILTER, player: "19" }).map((p) => p.id)).toEqual(["a", "c"]);
    expect(filterPlays(SET, { ...EMPTY_FILTER, player: "7" }).map((p) => p.id)).toEqual(["a", "d"]);
  });

  it("treats a half-typed shirt number as no filter rather than as zero", () => {
    expect(filterPlays(SET, { ...EMPTY_FILTER, player: "" })).toHaveLength(4);
  });

  it("returns nothing rather than everything for a combination that has none", () => {
    expect(filterPlays(SET, { ...EMPTY_FILTER, type: "high_press", zone: "box" })).toEqual([]);
  });

  it("reads a match forwards through the clock", () => {
    expect([...SET].sort(byClock).map((p) => p.minute)).toEqual([12, 34, 61, 70]);
  });
});

describe("the filter options", () => {
  it("only offer values this match actually has, with counts, commonest first", () => {
    expect(playChoices(SET, "type")).toEqual([
      { id: "corner_attack", label: "Corner, attacking", n: 2 },
      { id: "counter_attack", label: "Counter-attack", n: 1 },
      { id: "high_press", label: "High press", n: 1 },
    ]);
  });

  it("label a value nobody has named rather than dropping it", () => {
    const odd = [play({ id: "x", type: "rondo_exit" })];
    expect(playChoices(odd, "type")).toEqual([{ id: "rondo_exit", label: "rondo_exit", n: 1 }]);
  });
});

describe("a play's board", () => {
  it("comes back as a board the notepad can render", () => {
    const b = playBoard(SET[0]);
    expect(b.magnets.length).toBe(DEFAULT_MAGNETS.length);
    expect(b.view).toBe("box");
    expect(b.orient).toBe("v");
  });

  it("is narrowed by the same parsers a stored document goes through", () => {
    // match.json is a file a person or a script can edit, so a play must not be
    // able to put anything into the renderer a hand-edited document could not.
    const nasty = play({
      board: {
        magnets: [{ x: -50, y: 900, label: "toolong", side: "referee" }],
        shapes: [{ kind: "laser", pts: [1, 2, 3, 4], color: "url(#x)" }],
        view: "constructor",
        orient: "sideways",
        grass: "__proto__",
        lines: "toString",
      },
    });
    const b = playBoard(nasty);
    expect(b.magnets[0]).toEqual({ x: 0, y: 100, label: "too", side: "home" });
    expect(b.shapes).toEqual([]);
    expect(b.view).toBe("full");
    expect(b.orient).toBe("h");
    expect(b.grass).toBe("green");
    expect(b.lines).toBe("standard");
  });

  it("survives a play with no board at all", () => {
    const b = playBoard(play({ board: undefined }));
    expect(b.view).toBe("full");
    expect(b.magnets.length).toBeGreaterThan(0);
  });
});

describe("the caption", () => {
  it("names the fixture, what it was, when, how it ended and who", () => {
    expect(playCaption(SET[0], report)).toBe(
      "Westfield v Rovers — Corner, attacking · 12:04 — Goal — #7, #19",
    );
  });

  it("drops the parts a match cannot supply instead of printing blanks", () => {
    expect(playCaption(play({ players: [] }), null)).toBe("Corner, attacking · 12:04 — Shot");
  });
});
