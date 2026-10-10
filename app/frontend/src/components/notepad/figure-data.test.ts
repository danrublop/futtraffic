import { describe, expect, it } from "vitest";

import type { MatchReport } from "@/lib/types";
import { buildFigure, colSet, figureTitle, firstChoice, sectionId, COL_SETS, FIGURES } from "./figure-data";
import { parseFigureRows } from "./figure-node";

/** A finished match with tracks and NO event feed — which is every real match
 *  the pipeline has ever produced, and therefore the case that matters. */
const tracked = {
  meta: { home_name: "Westfield", away_name: "Rovers" },
  players: [
    { id: 1, team: "home", jersey: 8, minutes: 90, distance_km: 11.2, top_speed_kmh: 30.1, sprints: 24 },
    { id: 2, team: "home", jersey: null, minutes: 45, distance_km: 5.1, top_speed_kmh: 28.0, sprints: 9 },
    { id: 3, team: "unknown", jersey: null, minutes: 12, distance_km: 1.4, top_speed_kmh: 24.0, sprints: 2 },
  ],
} as unknown as MatchReport;

/** The demo match: tracks plus a hand-entered feed. */
const fed = {
  ...tracked,
  meta: {
    ...tracked.meta,
    facts: {
      player_stats: [
        { name: "Ana Ruiz", side: "home", pos: "Midfielder", stats: { rating: 7.8, minutes: 90, goals: 1, assists: 0, xg: 0.44, xa: 0.12 } },
        { name: "Sam Lee", side: "away", pos: "Goalkeeper", stats: { rating: 6.9, minutes: 90, goals: 0, assists: 0, xg: 0, xa: 0, saves: 4 } },
      ],
      top_stats: [{ label: "Ball possession", home: 72, away: 28, kind: "possession" }],
      sections: [
        {
          title: "Passes",
          rows: [
            { label: "Passes", home: 700, away: 274 },
            { label: "Accurate passes", home: "618 (88%)", away: "202 (74%)" },
            { label: "Formation", home: "4-2-3-1", away: "4-4-2" },
          ],
        },
      ],
      events: [
        { minute: "11'", side: "home", type: "goal", player: "Marco Ruiz", score: "1 - 0", assist: "Leo Brandt" },
        { minute: "30'", side: "away", type: "yellow_card", player: "Christopher McVey" },
        { minute: "HT", type: "period", score: "1 - 0" },
        { minute: "46'", side: "away", type: "substitution", in: "Milan Westbrook", out: "Joaquin Ellsworthy" },
      ],
    },
  },
} as unknown as MatchReport;

const build = (o: Partial<Parameters<typeof buildFigure>[0]> = {}) =>
  buildFigure({ report: tracked, cols: "physical", limit: 30, ...o });

describe("physical columns", () => {
  it("work on a match with no feed at all — the whole point of the set", () => {
    const f = build();
    expect(f.refusal).toBeUndefined();
    expect(f.head).toEqual(["Player", "Mins", "Distance", "km / 90", "Top speed", "Sprints"]);
    expect(f.rows).toHaveLength(3);
  });

  it("never invents a name, and says so", () => {
    const f = build();
    expect(f.rows.map((r) => r.name)).toEqual([
      "Westfield #8",
      "Westfield track 2",
      "Unassigned track 3",
    ]);
    expect(f.note).toMatch(/Shirt numbers, not names/);
  });

  it("puts per 90 beside the total, because a total ranks whoever played longest", () => {
    const [row] = build().rows;
    expect(row.cells).toEqual(["90", "11.20 km", "11.20 km", "30.1 km/h", "24"]);
    // The 45-minute player covers half the ground but the same per 90.
    expect(build().rows[1].cells[1]).toBe("5.10 km");
    expect(build().rows[1].cells[2]).toBe("10.20 km");
  });

  it("marks the side, and leaves it blank when the track has none", () => {
    expect(build().rows.map((r) => r.side)).toEqual(["home", "home", ""]);
  });

  it("honours the row limit", () => {
    expect(build({ limit: 2 }).rows).toHaveLength(2);
  });

  it("refuses when nothing was tracked", () => {
    const empty = { ...tracked, players: [] } as MatchReport;
    expect(build({ report: empty }).refusal).toMatch(/no tracked players/);
  });
});

describe("event columns", () => {
  it("point at the set that does work instead of rendering an empty table", () => {
    const f = build({ cols: "attack" });
    expect(f.rows).toHaveLength(0);
    expect(f.refusal).toMatch(/needs? the ball/i);
    expect(f.refusal).toMatch(/Physical/);
  });

  it("read the feed when the match carries one", () => {
    const f = build({ report: fed, cols: "top" });
    expect(f.rows.map((r) => r.name)).toEqual(["Ana Ruiz", "Sam Lee"]);
    expect(f.head[0]).toBe("Player");
    expect(f.head).toContain("Rating");
  });

  it("say entered, not measured — the feed is typed, not seen", () => {
    expect(build({ report: fed, cols: "top" }).note).toMatch(/Entered, not measured/);
    expect(build().note).toMatch(/Measured from the tracks/);
  });

  it("filter a goalkeeper-only set down to the keepers", () => {
    const f = build({ report: fed, cols: "gk" });
    expect(f.rows.map((r) => r.name)).toEqual(["Sam Lee"]);
  });
});

describe("the picker", () => {
  it("offers the measured set first, because it is the one that always works", () => {
    expect(COL_SETS[0]).toMatchObject({ id: "physical", source: "tracks" });
    expect(COL_SETS.filter((c) => c.source === "feed").length).toBeGreaterThan(3);
  });

  it("falls back rather than crashing on a set nobody defined", () => {
    expect(colSet("nope").id).toBe("physical");
    expect(build({ cols: "nope" }).head[0]).toBe("Player");
  });

  it("names the source in the title, so a pasted table is not ambiguous", () => {
    expect(figureTitle("playerTable", "physical", tracked)).toBe("Player stats — physical");
    expect(figureTitle("playerTable", "gk", tracked)).toBe("Player stats — goalkeeping");
    expect(figureTitle("events", "all", fed)).toBe("Timeline");
  });

  it("says so when there is no match at all", () => {
    expect(build({ report: null }).refusal).toMatch(/No match/);
  });
});

describe("the comparison figure", () => {
  const cmp = (o = {}) =>
    buildFigure({ report: fed, kind: "sections", cols: "passes", limit: 30, ...o });

  it("heads the columns with the two team names", () => {
    expect(cmp().head).toEqual(["", "Westfield", "Rovers"]);
  });

  it("splits a numeric row by home's share, for the bar", () => {
    const [passes] = cmp().rows;
    expect(passes.name).toBe("Passes");
    expect(passes.split).toBeCloseTo(700 / 974, 5);
  });

  it("reads the number in front of a qualified value", () => {
    // "618 (88%)" against "202 (74%)" splits on 618 and 202, not the percents.
    expect(cmp().rows[1].split).toBeCloseTo(618 / 820, 5);
  });

  it("leaves a non-numeric row without a bar rather than misparsing one", () => {
    const formation = cmp().rows[2];
    expect(formation.cells).toEqual(["4-2-3-1", "4-4-2"]);
    expect(formation.split).toBeUndefined();
  });

  it("offers top_stats as one more group, since it is the same shape", () => {
    expect(cmp().choices.map((c) => c.id)).toEqual(["top-stats", "passes"]);
    expect(
      buildFigure({ report: fed, kind: "sections", cols: "top-stats", limit: 30 }).rows[0].name,
    ).toBe("Ball possession");
  });

  it("does not offer it twice when the feed already has a group by that name", () => {
    // The demo carries BOTH `top_stats` and a `sections` group called "Top
    // stats". Two chips with the same slug means the second is unreachable.
    const both = {
      ...fed,
      meta: {
        ...fed.meta,
        facts: {
          ...fed.meta!.facts,
          sections: [
            { title: "Top stats", rows: [{ label: "Total shots", home: 33, away: 9 }] },
            ...fed.meta!.facts!.sections!,
          ],
        },
      },
    } as MatchReport;
    const ids = buildFigure({ report: both, kind: "sections", cols: "top-stats", limit: 30 }).choices.map((c) => c.id);
    expect(ids).toEqual(["top-stats", "passes"]);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("refuses on a match with no feed", () => {
    expect(
      buildFigure({ report: tracked, kind: "sections", cols: "passes", limit: 30 }).refusal,
    ).toMatch(/needs? the ball/i);
  });
});

describe("the timeline figure", () => {
  const tl = (cols = "all") => buildFigure({ report: fed, kind: "events", cols, limit: 30 });

  it("says each line in the words a reader uses", () => {
    expect(tl().rows.map((r) => `${r.name} ${r.cells[0]} ${r.cells[1]}`.trim())).toEqual([
      "11' Goal Marco Ruiz (1 - 0) assist Leo Brandt",
      "30' Yellow Christopher McVey booked",
      "HT Period 1 - 0",
      "46' Sub Milan Westbrook for Joaquin Ellsworthy",
    ]);
  });

  it("carries the side, and leaves a period marker without one", () => {
    expect(tl().rows.map((r) => r.side)).toEqual(["home", "away", "", "away"]);
  });

  it("filters to one kind of incident", () => {
    expect(tl("goal").rows).toHaveLength(1);
    expect(tl("substitution").rows[0].cells[1]).toMatch(/^Milan Westbrook for/);
  });

  it("refuses naming the kind that was asked for when there is none of it", () => {
    const noSubs = {
      ...fed,
      meta: {
        ...fed.meta,
        facts: { ...fed.meta!.facts, events: [{ minute: "11'", type: "goal", player: "X" }] },
      },
    } as MatchReport;
    expect(
      buildFigure({ report: noSubs, kind: "events", cols: "substitution", limit: 30 }).refusal,
    ).toMatch(/No subs/i);
  });
});

describe("switching figure kind", () => {
  it("carries a first choice that means something to the new kind", () => {
    expect(firstChoice("playerTable", fed)).toBe("physical");
    expect(firstChoice("events", fed)).toBe("all");
    expect(firstChoice("sections", fed)).toBe("top-stats");
  });

  it("falls back to the player table on a kind nobody defined", () => {
    expect(buildFigure({ report: tracked, kind: "nope", cols: "physical", limit: 5 }).head[0]).toBe(
      "Player",
    );
  });

  it("slugs a section title into something an attribute can hold", () => {
    expect(sectionId("Expected goals (xG)")).toBe("expected-goals-xg");
    expect(FIGURES.map((f) => f.id)).toEqual(["playerTable", "sections", "events"]);
  });
});

describe("what a stored figure may carry", () => {
  it("keeps a sentence-length cell, because a timeline row is prose", () => {
    // The cap was 24 when every cell was a figure like "13.56 km", and it cut
    // "Milan Westbrook for Joaquin Ellsworthy" mid-word in the PRINTED
    // document, where nothing can be scrolled to reveal the rest.
    const long = "Milan Westbrook for Joaquin Ellsworthy";
    const rows = parseFigureRows(JSON.stringify([{ name: "46'", side: "away", cells: ["Sub", long] }]));
    expect(rows[0].cells[1]).toBe(long);
  });

  it("still bounds it — a cell is not a paragraph", () => {
    const rows = parseFigureRows(JSON.stringify([{ name: "x", cells: ["a".repeat(500)] }]));
    expect(rows[0].cells[0].length).toBe(80);
  });
});
