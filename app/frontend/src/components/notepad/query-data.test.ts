import { describe, expect, it } from "vitest";

import type { MatchReport, Season } from "@/lib/types";
import {
  defaultRender,
  defaults,
  isComposition,
  labelsFor,
  measureOf,
  metricId,
  queryTitle,
  runColumns,
  runQuery,
  sourceOf,
  sources,
  teamsOf,
} from "./query-data";

const report = {
  meta: {
    home_name: "Westfield",
    away_name: "Rovers",
    facts: {
      player_stats: [
        { name: "Ana Ruiz", side: "home", pos: "Midfielder", stats: { goals: 2, passes: "618 (88%)", rating: 8 } },
        { name: "Kit Vale", side: "home", pos: "Midfielder", stats: { goals: 1, passes: 40, rating: 6 } },
        { name: "Sam Lee", side: "away", pos: "Defender", stats: { goals: 0, passes: 30, rating: 7 } },
      ],
      top_stats: [{ label: "Ball possession", home: 72, away: 28, kind: "possession" }],
      sections: [
        { title: "Passes", rows: [{ label: "Accurate passes", home: "618 (88%)", away: "202 (74%)" }] },
        // The same metric again, as the feed really sends it: a percent STRING
        // in a section and a bare number in top_stats.
        { title: "Possession", rows: [{ label: "Ball possession", home: "72%", away: "28%" }] },
        // A metric that ONLY ever arrives as a percent string, and one that
        // arrives as a made-of-attempted pair.
        { title: "Duels", rows: [
          { label: "Duel success", home: "55%", away: "45%" },
          { label: "Dribbles", home: "3/5 (60%)", away: "1/4 (25%)" },
        ] },
      ],
      lineups: {
        home: {
          formation: "4-3-3",
          xi: [
            { n: 1, name: "Keeper", pos: "Goalkeeper", x: 0.5, y: 0.06, rating: 7.5, age: 22, height: `6'5"` },
            { n: 8, name: "Ana Ruiz", pos: "Midfielder", x: 0.5, y: 0.4, rating: 8.1, age: 26, height: `5'10"` },
          ],
        },
        away: {
          formation: "4-4-2",
          xi: [{ n: 4, name: "Sam Lee", pos: "Defender", x: 0.5, y: 0.2, rating: 6.9, age: 30, height: `6'0"` }],
        },
      },
      subs: {
        home: [
          { n: 21, name: "Sub One", pos: "Midfielder", rating: 6.5, on: "63'", age: 20, height: `5'9"` },
          { n: 22, name: "Unused", pos: "Forward", age: 24, height: `6'1"` },
          // Came on, and the feed gave them no rating and no height. Both of
          // those used to arrive as 0 and drag the squad's averages down.
          { n: 23, name: "Late Sub", pos: "Forward", on: "88'" },
        ],
      },
      shots: {
        points: [
          { x: 0.1, y: 0.3, side: "home", goal: true, period: 1 },
          { x: 0.2, y: 0.4, side: "home", goal: false, period: 2 },
          { x: 0.8, y: 0.5, side: "away", goal: false, period: 1 },
        ],
      },
      momentum: { values: [0.1, 0.5, -0.2], goals: [] },
      attacking_zones: { home: [37, 25, 38], away: [28, 19, 53] },
      plays: [
        { id: "1", type: "corner_attack", zone: "box", outcome: "goal", side: "home", period: 1, minute: 5, clock: "5:00", players: [], board: {}, t: 0 },
        { id: "2", type: "corner_attack", zone: "box", outcome: "shot", side: "home", period: 2, minute: 60, clock: "60:00", players: [], board: {}, t: 0 },
        { id: "3", type: "counter_attack", zone: "middle_third", outcome: "shot", side: "away", period: 1, minute: 20, clock: "20:00", players: [], board: {}, t: 0 },
        { id: "4", type: "high_press", zone: "final_third", outcome: "lost", side: "home", period: 2, minute: 70, clock: "70:00", players: [], board: {}, t: 0 },
      ],
    },
  },
  teams: {
    home: {
      distance_m: 137273, top_speed_kmh: 24.7, sprints: 58,
      shape: { width_m: 37.6, depth_m: 46.3, defensive_line_m: 21.2, compactness_m2: 1776, stretch_index_m: 22 },
    },
    away: {
      distance_m: 120000, top_speed_kmh: 26.1, sprints: 41,
      shape: { width_m: 41.2, depth_m: 39.8, defensive_line_m: 28.4, compactness_m2: 1980, stretch_index_m: 25 },
    },
  },
  territory: { defensive_third: 25.9, middle_third: 48.1, attacking_third: 26.0 },
  players: [
    { id: 1, team: "home", jersey: 8, minutes: 90, distance_km: 11.0, top_speed_kmh: 30.1, sprints: 24, avg_position: [30, 34] },
    { id: 2, team: "home", jersey: 9, minutes: 45, distance_km: 5.0, top_speed_kmh: 28.0, sprints: 9, avg_position: [70, 20] },
    { id: 3, team: "away", jersey: 4, minutes: 90, distance_km: 9.0, top_speed_kmh: 29.0, sprints: 17, avg_position: [52, 50] },
  ],
  heatmaps: {
    "1": [[10, 34], [20, 30]],
    "2": [[80, 10]],
    "3": [[52, 50], [52, 51], [52, 52]],
  },
} as unknown as MatchReport;

const season: Season = {
  team: "Westfield",
  matches: [
    { id: "m1", name: "Week 1", created: 1, home_name: null, away_name: null, score: null },
    { id: "m2", name: "Week 2", created: 2, home_name: null, away_name: null, score: null },
  ],
  players: [
    {
      id: "ana", name: "Ana Ruiz", number: 8, position: "CM", team: "Westfield",
      matches: 2, minutes: 180, distance_km: 22, sprints: 40, top_speed_kmh: 30, km_per90: 11, sprints_per90: 20,
      by_match: {
        m1: { minutes: 90, distance_km: 12, sprints: 22, top_speed_kmh: 30, side: "home", jersey: 8 },
        m2: { minutes: 90, distance_km: 10, sprints: 18, top_speed_kmh: 29, side: "away", jersey: 8 },
      },
    },
    {
      id: "kit", name: "Kit Vale", number: 6, position: "CM", team: "Westfield",
      matches: 1, minutes: 20, distance_km: 3, sprints: 6, top_speed_kmh: 27, km_per90: 13.5, sprints_per90: 27,
      by_match: { m1: { minutes: 20, distance_km: 3, sprints: 6, top_speed_kmh: 27, side: "home", jersey: 6 } },
    },
  ],
  coverage: { bound: 2, total: 8 },
};

const q = (o: Partial<Parameters<typeof runQuery>[0]> = {}) =>
  runQuery({ report, season, source: "tracks", dim: "player", measure: "distance_km", split: "", limit: 20, ...o });

describe("asking the tracks", () => {
  it("breaks a measure down by a dimension, biggest first", () => {
    const r = q();
    expect(r.rows.map((x) => x.label)).toEqual(["Westfield #8", "Rovers #4", "Westfield #9"]);
    expect(r.rows[0].text[0]).toBe("11.00 km");
    expect(r.max).toBe(11);
  });

  it("rolls up when the dimension has fewer values than the facts", () => {
    const r = q({ dim: "side" });
    expect(r.rows.map((x) => [x.label, x.values[0]])).toEqual([
      ["Westfield", 16],
      ["Rovers", 9],
    ]);
  });

  it("counts rather than sums when the measure is a count", () => {
    expect(q({ dim: "side", measure: "count" }).rows.map((x) => x.values[0])).toEqual([2, 1]);
  });

  it("takes the best single reading for a peak, never a total", () => {
    const r = q({ dim: "side", measure: "top_speed_kmh" });
    expect(r.rows.find((x) => x.label === "Westfield")!.values[0]).toBe(30.1);
  });

  it("computes a rate as a ratio of sums, not a mean of rates", () => {
    // Westfield: (11 + 5) / (90 + 45) * 90 = 10.67. The mean of 11.0 and 10.0
    // would be 10.5, which weights a 45-minute sub like a full match.
    const r = q({ dim: "side", measure: "km_per90" });
    expect(r.rows.find((x) => x.label === "Westfield")!.values[0]).toBeCloseTo(10.67, 2);
  });
});

describe("splitting into two series", () => {
  it("draws one bar per side and keeps them on one scale", () => {
    const r = q({ source: "feed", dim: "position", measure: "goals", split: "side" });
    expect(r.series).toEqual(["Rovers", "Westfield"]);
    const mid = r.rows.find((x) => x.label === "Midfielder")!;
    expect(mid.values).toEqual([0, 3]);
  });

  it("never splits by the dimension it is already broken down by", () => {
    const r = q({ dim: "side", split: "side" });
    expect(r.series).toEqual([]);
    expect(r.rows).toHaveLength(2);
  });

  it("stops at two series rather than inventing a third colour", () => {
    const r = q({ source: "plays", dim: "outcome", split: "playType" });
    expect(r.series.length).toBeLessThanOrEqual(2);
  });
});

describe("asking the plays", () => {
  it("counts moments by any of their own facets", () => {
    expect(q({ source: "plays", dim: "playType", measure: "count" }).rows.map((x) => [x.label, x.values[0]]))
      .toEqual([["Corner, attacking", 2], ["Counter-attack", 1], ["High press", 1]]);
    expect(q({ source: "plays", dim: "zone", measure: "count" }).rows[0].label).toBe("Penalty box");
  });

  it("keeps an ordered dimension in its own order, not sorted by size", () => {
    const r = q({ source: "plays", dim: "period", measure: "count" });
    expect(r.rows.map((x) => x.label)).toEqual(["First half", "Second half"]);
    expect(r.ordered).toBe(true);
  });

  it("marks an unordered dimension as such, so nothing draws it as a line", () => {
    expect(q({ source: "plays", dim: "outcome" }).ordered).toBe(false);
  });
});

describe("asking the season", () => {
  it("breaks one player's matches apart, in order", () => {
    const r = q({ source: "season", dim: "match", measure: "distance_km" });
    expect(r.rows.map((x) => [x.label, x.values[0]])).toEqual([
      ["Week 1", 15],
      ["Week 2", 10],
    ]);
  });

  it("groups a position on its own totals", () => {
    const r = q({ source: "season", dim: "position", measure: "km_per90" });
    // (22 + 3) / (180 + 20) * 90 = 11.25
    expect(r.rows[0].values[0]).toBeCloseTo(11.25, 2);
  });

  it("always says what share of the tracking it rests on", () => {
    expect(q({ source: "season", dim: "player" }).note).toMatch(/2 of 8 tracked players/);
  });

  it("withholds sprints whatever the old season rows carry (22 and 18 above)", () => {
    const r = q({ source: "season", dim: "match", measure: "sprints" });
    expect(r.rows.flatMap((x) => x.values).filter((v) => v)).toEqual([]);
  });

  it("refuses when the season has not loaded", () => {
    expect(q({ source: "season", season: null }).refusal).toMatch(/season/i);
  });

  it("breaks the club apart by team, so two age groups can share a note", () => {
    // The whole point of taking notes off matches: a note is written across
    // teams, and a season fetched club-wide would otherwise pool the U15s and
    // the U17s into one anonymous list of names.
    const club: Season = {
      ...season,
      players: [
        ...season.players,
        {
          ...season.players[0],
          id: "u17", name: "Sam Okoro", team: "Westfield U17",
          by_match: { m2: { minutes: 90, distance_km: 9, sprints: 12, top_speed_kmh: 31, side: "home", jersey: 4 } },
        },
      ],
    };
    const r = q({ source: "season", season: club, dim: "side", measure: "distance_km" });
    expect(r.rows.map((x) => [x.label, x.values[0]])).toEqual([
      ["Westfield", 25],
      ["Westfield U17", 9],
    ]);
  });
});

describe("refusals", () => {
  it("names the ball when a feed source has nothing", () => {
    const bare = { players: report.players } as MatchReport;
    expect(q({ report: bare, source: "plays" }).refusal).toMatch(/needs? the ball|follows bodies/i);
  });

  it("points at the identity screen when no season row exists", () => {
    const none = { ...season, players: [] };
    expect(q({ source: "season", season: none }).refusal).toMatch(/identity screen/);
  });

  it("says so when there is no match", () => {
    expect(q({ report: null }).refusal).toMatch(/No match/);
  });
});

describe("what a source will answer", () => {
  it("only offers dimensions it actually carries", () => {
    expect(sourceOf("tracks").dims.map((d) => d.id)).toEqual(["player", "side"]);
    expect(sourceOf("plays").measures.map((m) => m.id)).toEqual(["count"]);
    expect(sourceOf("season").dims.map((d) => d.id)).toContain("match");
  });

  it("falls back rather than crashing on a source or measure nobody defined", () => {
    expect(sourceOf("nope").id).toBe("tracks");
    expect(q({ measure: "nope" }).measure.id).toBe(sourceOf("tracks").measures[0].id);
    expect(q({ dim: "nope" }).dim.id).toBe("player");
  });

  it("covers every part of the report a coach can see", () => {
    expect(sources(report).map((s) => s.id).sort()).toEqual([
      "feed", "momentum", "plays", "season", "shots", "squad",
      "teamFeed", "teams", "territory", "tracks", "zones",
    ]);
  });

  it("hands a new block a question its source can answer, that returns something", () => {
    for (const s of sources(report)) {
      const d = defaults(s.id, report);
      expect(s.dims.some((x) => x.id === d.dim), s.id).toBe(true);
      expect(s.measures.some((x) => x.id === d.measure), s.id).toBe(true);
      // A source that opens on a refusal is a source nobody will use twice.
      const r = runQuery({ report, season, source: s.id, ...d, split: "", limit: 20 });
      expect(r.refusal, `${s.id}: ${r.refusal}`).toBeUndefined();
    }
  });

  it("titles itself with the question that was asked", () => {
    expect(queryTitle(q(), "")).toBe("Distance by player");
    expect(queryTitle(q({ source: "feed", dim: "position", measure: "goals" }), "side")).toBe(
      "Goals by position, split by team",
    );
  });
});

describe("the rest of the report", () => {
  const at = (source: string, dim: string, measure: string, label: string) =>
    runQuery({ report, season, source, dim, measure, split: "", limit: 20 }).rows.find(
      (r) => r.label === label,
    );

  it("gives team distance in kilometres, not the metres it is stored in", () => {
    expect(at("teams", "side", "distance_km", "Westfield")!.text[0]).toBe("137.3 km");
  });

  it("averages a shape metric rather than adding two sides together", () => {
    expect(at("teams", "side", "width_m", "Rovers")!.values[0]).toBeCloseTo(41.2, 5);
  });

  it("reads a qualified team stat down to its leading quantity", () => {
    // "618 (88%)" is 618 accurate passes, not 618.88 of anything.
    expect(at("teamFeed", "side", metricId("Accurate passes"), "Westfield")!.values[0]).toBe(618);
  });

  it("offers a team metric once, however many groups repeat it", () => {
    const ids = sourceOf("teamFeed", report).measures.map((m) => m.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).toContain("ball-possession");
  });

  it("counts shots by how they ended, and by half", () => {
    expect(at("shots", "result", "count", "Goal")!.values[0]).toBe(1);
    expect(at("shots", "period", "count", "First half")!.values[0]).toBe(2);
  });

  it("turns a lineup height into centimetres", () => {
    expect(at("squad", "player", "height_cm", "Keeper")!.text[0]).toBe("196 cm");
  });

  it("averages a rating across a position instead of totalling it", () => {
    // Ana 8.1 and the sub 6.5 are both midfielders: 7.3, not 14.6.
    expect(at("squad", "position", "rating", "Midfielder")!.values[0]).toBeCloseTo(7.3, 5);
  });

  it("leaves an unused substitute out of the squad entirely", () => {
    const r = runQuery({
      report, season, source: "squad", dim: "player", measure: "count", split: "", limit: 20,
    });
    expect(r.rows.map((x) => x.label)).not.toContain("Unused");
    expect(r.rows.map((x) => x.label)).toContain("Sub One");
  });

  it("keeps momentum in clock order, so it can honestly be drawn as a line", () => {
    const r = runQuery({
      report, season, source: "momentum", dim: "minute", measure: "pressure", split: "", limit: 99,
    });
    expect(r.ordered).toBe(true);
    expect(r.rows.map((x) => x.label)).toEqual(["01", "02", "03"]);
  });

  it("breaks the attacking channels out per side", () => {
    const r = runQuery({
      report, season, source: "zones", dim: "channel", measure: "share", split: "side", limit: 9,
    });
    expect(r.rows.map((x) => x.label).sort()).toEqual(["Centre", "Left", "Right"]);
    expect(r.rows.find((x) => x.label === "Right")!.values).toEqual([53, 38]);
  });

  it("says what territory is NOT, because a green pitch reads as possession", () => {
    const r = runQuery({
      report, season, source: "territory", dim: "third", measure: "share", split: "", limit: 9,
    });
    expect(r.note).toMatch(/not who had the ball/);
  });
});

describe("sources with a visual of their own", () => {
  it("opens shots on a pitch and momentum on the swing, not on bars", () => {
    expect(defaultRender("shots")).toBe("pitch");
    expect(defaultRender("momentum")).toBe("momentum");
    expect(defaultRender("tracks")).toBe("bar");
  });

  it("carries the shots themselves, flipped to attack the way the boards do", () => {
    const r = runQuery({ report, season, source: "shots", dim: "side", measure: "count", split: "", limit: 20 });
    expect(r.extra?.kind).toBe("shots");
    const e = r.extra as Extract<typeof r.extra, { kind: "shots" }>;
    // The feed has home attacking the LEFT goal at x=0.1; every board in the
    // app has home attacking the right, so a shot map beside one must agree.
    expect(e.points[0].x).toBeCloseTo(90, 5);
    expect(e.points[0].goal).toBe(true);
    expect(e.home).toBe("Westfield");
  });

  it("carries the momentum series and the goals marked on it", () => {
    const r = runQuery({ report, season, source: "momentum", dim: "minute", measure: "pressure", split: "", limit: 99 });
    const e = r.extra as Extract<typeof r.extra, { kind: "momentum" }>;
    expect(e.values).toEqual([0.1, 0.5, -0.2]);
    expect(e.goals).toEqual([]);
  });

  it("gives a source with no visual of its own no payload to carry", () => {
    const r = runQuery({ report, season, source: "tracks", dim: "player", measure: "distance_km", split: "", limit: 5 });
    expect(r.extra).toBeUndefined();
  });
});

describe("an ordered dimension keeps the order its facts arrived in", () => {
  it("walks the pitch from back to front, not down the alphabet", () => {
    // The trap: sorting the labels reads fine for minutes and halves, and puts
    // the thirds in attacking-defensive-middle order, which is a pitch nobody
    // has ever played on.
    const r = q({ source: "territory", dim: "third", measure: "share" });
    expect(r.rows.map((x) => x.label)).toEqual([
      "Left third", "Centre third", "Right third",
    ]);
    expect(runColumns({
      report, season, source: "territory", dim: "third", measures: ["share"], limit: 9,
    }).rows.map((x) => x.label)).toEqual([
      "Left third", "Centre third", "Right third",
    ]);
  });

  it("still puts the clock in clock order, which is what the sort got right", () => {
    expect(q({ source: "momentum", dim: "minute", measure: "pressure", limit: 99 })
      .rows.map((x) => x.label)).toEqual(["01", "02", "03"]);
    expect(q({ source: "plays", dim: "period", measure: "count" })
      .rows.map((x) => x.label)).toEqual(["First half", "Second half"]);
  });
});

describe("a table, which is several measures at once", () => {
  const cols = (o: Partial<Parameters<typeof runColumns>[0]> = {}) =>
    runColumns({
      report, season, source: "tracks", dim: "player",
      measures: ["distance_km", "top_speed_kmh"], limit: 20, ...o,
    });

  it("heads the table with the dimension and then every measure, in order", () => {
    expect(cols().head).toEqual(["Player", "Distance", "Top speed"]);
  });

  it("keeps one row order across every column", () => {
    // Sorted by DISTANCE, the first column. Top speed disagrees about the
    // order, which is exactly the bug running one query per measure and
    // zipping the results would produce: #9 is fastest and third by distance.
    const r = cols();
    expect(r.rows.map((x) => x.label)).toEqual(["Westfield #8", "Rovers #4", "Westfield #9"]);
    expect(r.rows.map((x) => x.text)).toEqual([
      ["11.00 km", "30.1 km/h"],
      ["9.00 km", "29.0 km/h"],
      ["5.00 km", "28.0 km/h"],
    ]);
  });

  it("aggregates each column its own way in the same pass", () => {
    // Distance sums, top speed maxes. Rolled up to two rows, home's 11 + 5 is
    // 16 km but its top speed is 30.1, not 58.1.
    const r = cols({ dim: "side" });
    expect(r.rows.find((x) => x.label === "Westfield")!.values).toEqual([16, 30.1]);
  });

  it("marks which side a row is, and only where the source knows", () => {
    expect(cols().rows.map((x) => x.side)).toEqual(["home", "away", "home"]);
    // Territory has no side dimension at all; a dot there would be invented.
    const t = cols({ source: "territory", dim: "third", measures: ["share"] });
    expect(t.rows.every((x) => x.side === "")).toBe(true);
  });

  it("refuses with the reason rather than returning an empty table", () => {
    const r = runColumns({
      report: null, season, source: "tracks", dim: "player",
      measures: ["distance_km"], limit: 20,
    });
    expect(r.rows).toEqual([]);
    expect(r.refusal).toBe("No match is open.");
  });

  it("narrows the season to a range of matches", () => {
    const all = cols({ source: "season", dim: "player", measures: ["distance_km"] });
    expect(all.rows.find((x) => x.label === "Ana Ruiz")!.values[0]).toBe(22);
    const last = cols({
      source: "season", dim: "player", measures: ["distance_km"], matchIds: ["m2"],
    });
    expect(last.rows.find((x) => x.label === "Ana Ruiz")!.values[0]).toBe(10);
    // Kit played only week 1, so a week-2 table has no row for her at all.
    expect(last.rows.map((x) => x.label)).not.toContain("Kit Vale");
    expect(last.note).toContain("across 1 match");
  });

  it("counts an empty range as a range, not as no filter", () => {
    // The panel can hand down [] while the season is still loading; answering
    // it with the whole season would put a different question's numbers under
    // the label on screen.
    const r = cols({ source: "season", dim: "player", measures: ["distance_km"], matchIds: [] });
    expect(r.rows).toEqual([]);
  });
});

describe("knowing when a table needs its chart", () => {
  const comp = (source: string, dim: string, measure: string) => {
    const r = runColumns({ report, season, source, dim, measures: [measure], limit: 20 });
    return isComposition(r, measureOf(sourceOf(source, report), measure));
  };

  it("spots the parts of a whole the scouting report always draws", () => {
    expect(comp("territory", "third", "share")).toBe(true);
    expect(comp("zones", "channel", "share")).toBe(true);
    expect(comp("plays", "playType", "count")).toBe(true);
  });

  it("leaves a ranking alone — a bar chart of it says nothing the order does not", () => {
    expect(comp("tracks", "player", "distance_km")).toBe(false);
  });

  it("needs at least three parts, because two is a comparison and reads fine", () => {
    const r = runColumns({
      report, season, source: "tracks", dim: "side", measures: ["sprints"], limit: 20,
    });
    expect(isComposition(r, measureOf(sourceOf("tracks", report), "sprints"))).toBe(false);
  });

  it("never fires on a multi-column table, which has no single whole", () => {
    const r = runColumns({
      report, season, source: "territory", dim: "third", measures: ["share", "share"], limit: 20,
    });
    expect(isComposition(r, measureOf(sourceOf("territory", report), "share"))).toBe(false);
  });
});

describe("every source says who it is about", () => {
  it("splits cleanly into players and teams, with nothing left over", () => {
    const all = sources(report);
    expect(all.every((s) => s.subject === "player" || s.subject === "team")).toBe(true);
    // Both halves have to be non-empty in both scopes a panel can ask for, or
    // the Who chips offer a state with nothing behind it.
    for (const subject of ["player", "team"] as const) {
      expect(all.filter((s) => s.subject === subject && s.scope === "match").length).toBeGreaterThan(0);
    }
    expect(all.filter((s) => s.scope === "season").length).toBeGreaterThan(0);
  });
});

describe("one team, not both", () => {
  it("offers the names a source actually carries, and nothing where there is no side", () => {
    expect(teamsOf("tracks", report)).toEqual(["Rovers", "Westfield"]);
    // Territory is the fixture's, not either team's — offering a team filter
    // there would be a control that changes nothing.
    expect(teamsOf("territory", report)).toEqual([]);
  });

  it("narrows a ranking to one squad", () => {
    // A ranking of "the players" that mixes both squads is a ranking of
    // nobody's squad.
    const both = q();
    expect(both.rows).toHaveLength(3);
    const one = q({ team: "Westfield" });
    expect(one.rows.map((x) => x.label)).toEqual(["Westfield #8", "Westfield #9"]);
  });

  it("rescales the axis to the population on screen", () => {
    // Filtering after the roll-up would leave the bars drawn against the other
    // team's maximum, so the leader of the shown side would not reach the end.
    expect(q({ team: "Westfield" }).max).toBe(11);
    expect(q({ team: "Rovers" }).max).toBe(9);
  });

  it("narrows a table the same way", () => {
    const r = runColumns({
      report, season, source: "tracks", dim: "player",
      measures: ["distance_km"], limit: 20, team: "Rovers",
    });
    expect(r.rows.map((x) => x.label)).toEqual(["Rovers #4"]);
  });
});

describe("the pictures only football has", () => {
  it("puts one disc per player at their mean position, in percent of the pitch", () => {
    const r = q({ render: "spots" });
    expect(r.extra?.kind).toBe("spots");
    const e = r.extra as Extract<typeof r.extra, { kind: "spots" }>;
    expect(e.measured).toBe(true);
    expect(e.spots).toHaveLength(3);
    // 30 m along a 105 m pitch is 28.6% of it, not 30.
    expect(e.spots[0].x).toBeCloseTo((30 / 105) * 100, 4);
    expect(e.spots[0].label).toBe("8");
  });

  it("draws a team sheet as a DRAWN line-up, never as a measured one", () => {
    const e = q({ source: "squad", dim: "player", measure: "rating", render: "lineup" }).extra;
    expect(e?.kind).toBe("spots");
    expect((e as Extract<typeof e, { kind: "spots" }>).measured).toBe(false);
  });

  it("reflects the away half so both squads are not piled on one end", () => {
    const e = q({ source: "squad", dim: "player", measure: "rating", render: "lineup" })
      .extra as Extract<ReturnType<typeof q>["extra"], { kind: "spots" }>;
    const home = e.spots.filter((s) => s.side === "home").map((s) => s.x);
    const away = e.spots.filter((s) => s.side === "away").map((s) => s.x);
    expect(Math.max(...home)).toBeLessThan(50);
    expect(Math.min(...away)).toBeGreaterThan(50);
  });

  it("bins one side's sampled positions for a heat map", () => {
    const e = q({ render: "heat", team: "Westfield" }).extra;
    expect(e?.kind).toBe("heat");
    const h = e as Extract<typeof e, { kind: "heat" }>;
    // Tracks 1 and 2 are Westfield: two points and one. Track 3 is Rovers.
    expect(h.total).toBe(3);
    expect(h.who).toBe("Westfield");
    // COUNTS, not positions. The document used to carry up to 8,000 points to
    // draw ninety-six numbers; storing the numbers is the whole fix.
    expect(h.bins).toHaveLength(h.cols * h.rows);
    expect(h.bins.reduce((a, b) => a + b, 0)).toBe(3);
    // Pitch is 105x68 in the fixture, so [10,34] is 9.5% along and 50% down —
    // column 1 of 12, row 4 of 8 — and [20,30] is column 2, row 3.
    expect(h.bins[4 * h.cols + 1]).toBe(1);
    expect(h.bins[3 * h.cols + 2]).toBe(1);
    // [80,10] is Westfield's other track: 76% along, 14.7% down.
    expect(h.bins[1 * h.cols + 9]).toBe(1);
  });

  it("carries the thirds as bands along the pitch and the channels across it", () => {
    const t = q({ source: "territory", dim: "third", measure: "share", render: "zonepitch" }).extra;
    expect(t?.kind === "zones" && t.axis).toBe("length");
    expect(t?.kind === "zones" && t.bands[0].label).toBe("Left third");
    const z = q({ source: "zones", dim: "channel", measure: "share", render: "zonepitch" }).extra;
    expect(z?.kind === "zones" && z.axis).toBe("width");
  });

  it("hands a picture nothing when its source is not the one on screen", () => {
    // Asking the shot list for a heat map is not a fallback, it is a mistake,
    // and the renderer must not be handed a payload it would draw anyway.
    expect(q({ source: "shots", dim: "result", measure: "count", render: "heat" }).extra?.kind)
      .not.toBe("heat");
  });
});

describe("comparing named rows, not the top N", () => {
  it("offers exactly the rows the chart could draw", () => {
    // Read off the FACTS, so the picker can never list a name that would come
    // back as an empty row.
    expect(labelsFor({ report, season, source: "tracks", dim: "player" }))
      .toEqual(["Westfield #8", "Westfield #9", "Rovers #4"]);
    expect(labelsFor({ report, season, source: "tracks", dim: "player", team: "Rovers" }))
      .toEqual(["Rovers #4"]);
  });

  it("keeps the two that were picked and drops the rest", () => {
    const r = q({ keep: ["Rovers #4", "Westfield #8"] });
    expect(r.rows.map((x) => x.label)).toEqual(["Rovers #4", "Westfield #8"]);
  });

  it("keeps them in the order they were PICKED, not the measure's order", () => {
    // Sorted by distance, #8 leads. Picking #4 first has to mean #4 first, or
    // the pair swaps places every time the measure changes and the chart's
    // meaning moves under the reader.
    expect(q().rows[0].label).toBe("Westfield #8");
    expect(q({ keep: ["Rovers #4", "Westfield #8"] }).rows[0].label).toBe("Rovers #4");
  });

  it("ignores the row cap, because a picked set IS the answer", () => {
    const r = q({ keep: ["Westfield #8", "Westfield #9", "Rovers #4"], limit: 1 });
    expect(r.rows).toHaveLength(3);
  });

  it("cannot invent a row for a name that is not there", () => {
    const r = q({ keep: ["Westfield #8", "Nobody At All"] });
    expect(r.rows.map((x) => x.label)).toEqual(["Westfield #8"]);
  });

  it("compares columns in a table the same way", () => {
    const r = runColumns({
      report, season, source: "tracks", dim: "player",
      measures: ["distance_km", "top_speed_kmh"], limit: 20,
      keep: ["Rovers #4", "Westfield #8"],
    });
    expect(r.rows.map((x) => x.label)).toEqual(["Rovers #4", "Westfield #8"]);
  });
});

describe("a number nobody recorded is not a zero", () => {
  const q = (o: Record<string, unknown> = {}) =>
    runQuery({ report, season, source: "teamFeed", dim: "side", measure: "ball-possession", split: "", limit: 20, ...o });

  it("reads a percent string, and reads both sides off the same row", () => {
    // The regex used to demand the number be followed by end-of-string, space
    // or "(", so "72%" parsed as 0. The dedupe guard then ran on the home pass
    // only, so home kept top_stats (72) and away kept the section ("28%" -> 0).
    // The chart drew a 72-0 possession split, captioned as if it were the feed.
    expect(q().rows.map((r) => [r.label, r.values[0]])).toEqual([
      ["Westfield", 72],
      ["Rovers", 28],
    ]);
  });

  it("reads a percent that is the whole value", () => {
    // "55%" used to parse as 0, because the number had to be followed by end,
    // a space or "(". Every percent-only metric in the feed charted as zero.
    const r = runQuery({ report, season, source: "teamFeed", dim: "side", measure: "duel-success", split: "", limit: 20 });
    expect(r.rows.map((x) => x.values[0])).toEqual([55, 45]);
  });

  it("reads the successful half of a made-of-attempted pair", () => {
    const passes = runQuery({ report, season, source: "teamFeed", dim: "side", measure: "accurate-passes", split: "", limit: 20 });
    expect(passes.rows.map((x) => x.values[0])).toEqual([618, 202]);
    const dribbles = runQuery({ report, season, source: "teamFeed", dim: "side", measure: "dribbles", split: "", limit: 20 });
    expect(dribbles.rows.map((x) => x.values[0])).toEqual([3, 1]);
  });

  it("does not average a rating against the players who have none", () => {
    // Westfield: 7.5, 8.1, 6.5, and one substitute the feed never rated. The
    // mean of the three that exist is 7.37; dividing by four gives 5.53.
    const r = runQuery({ report, season, source: "squad", dim: "side", measure: "rating", split: "", limit: 20 });
    const west = r.rows.find((x) => x.label === "Westfield")!;
    expect(west.values[0]).toBeCloseTo((7.5 + 8.1 + 6.5) / 3, 2);
  });

  it("still counts every row, absent measure or not", () => {
    const r = runQuery({ report, season, source: "squad", dim: "side", measure: "count", split: "", limit: 20 });
    const west = r.rows.find((x) => x.label === "Westfield")!;
    expect(west.values[0]).toBe(4);
  });
});

describe("a split that does not fit is refused, not half-drawn", () => {
  it("keeps every row and says why it did not split", () => {
    // Four positions in the squad feed; the palette carries two series. This
    // used to take the first two alphabetically and drop the rest of the facts
    // on the floor, with no note and a row count that counted them.
    // Four positions in the squad feed against a palette that carries two.
    const r = runQuery({
      report, season, source: "squad", dim: "side", measure: "count", split: "position", limit: 20,
    });
    expect(r.series).toEqual([]);
    // Every player still counted: 4 for Westfield, 1 for Rovers.
    expect(r.rows.map((x) => x.values[0]).reduce((a, b) => a + b, 0)).toBe(5);
    expect(r.warning).toMatch(/Not split/);
  });

  it("splits when there are exactly two values", () => {
    const r = runQuery({
      report, season, source: "feed", dim: "position", measure: "goals", split: "side", limit: 20,
    });
    expect(r.series).toEqual(["Rovers", "Westfield"]);
    expect(r.warning).toBeUndefined();
  });
});

describe("ordering the rows on purpose", () => {
  const rows = () => q({ source: "tracks", dim: "player", measure: "distance_km", limit: 500 });

  it("still leads with the biggest when nothing was asked", () => {
    const v = rows().rows.map((r) => r.values[0] ?? 0);
    expect(v).toEqual([...v].sort((a, b) => b - a));
  });

  it("turns round for smallest first", () => {
    const v = q({ source: "tracks", dim: "player", measure: "distance_km", limit: 500, sort: "valueAsc" })
      .rows.map((r) => r.values[0] ?? 0);
    expect(v).toEqual([...v].sort((a, b) => a - b));
  });

  it("sorts by name in both directions", () => {
    const az = q({ source: "tracks", dim: "player", measure: "distance_km", limit: 500, sort: "label" })
      .rows.map((r) => r.label);
    expect(az).toEqual([...az].sort((a, b) => a.localeCompare(b)));
    const za = q({ source: "tracks", dim: "player", measure: "distance_km", limit: 500, sort: "labelDesc" })
      .rows.map((r) => r.label);
    expect(za).toEqual([...az].reverse());
  });

  it("leaves an ordered dimension alone unless asked otherwise", () => {
    // A match list sorted by size is a bar chart pretending to be a timeline.
    const natural = q({ source: "feed", dim: "minute", measure: "count", limit: 500 }).rows.map((r) => r.label);
    const asked = q({ source: "feed", dim: "minute", measure: "count", limit: 500, sort: "natural" })
      .rows.map((r) => r.label);
    expect(asked).toEqual(natural);
  });

  it("lets an ordered dimension be ranked when that is the question", () => {
    const v = q({ source: "feed", dim: "minute", measure: "count", limit: 500, sort: "value" })
      .rows.map((r) => r.values[0] ?? 0);
    expect(v).toEqual([...v].sort((a, b) => b - a));
  });

  it("never lets a sort overrule a comparison", () => {
    // Picking two players means those two IN THAT ORDER; re-sorting them by
    // whoever ran further would shuffle the pair every time the measure moved.
    const all = rows().rows.map((r) => r.label);
    const pair = [all[2], all[0]];
    const got = q({
      source: "tracks", dim: "player", measure: "distance_km", limit: 500,
      keep: pair, sort: "valueAsc",
    }).rows.map((r) => r.label);
    expect(got).toEqual(pair);
  });
});

describe("ordering a table's rows", () => {
  const cols = (over: Record<string, unknown> = {}) =>
    runColumns({
      report, season: null, source: "tracks", dim: "player",
      measures: ["distance_km", "top_speed_kmh"], limit: 500, ...over,
    });

  it("still ranks on the FIRST column when nothing was asked", () => {
    // Not the largest value in the row — a table ranks on the column somebody
    // picked first, which is the one they meant to rank by.
    const v = cols().rows.map((r) => r.values[0] ?? 0);
    expect(v).toEqual([...v].sort((a, b) => b - a));
  });

  it("keeps ranking on the first column when asked for biggest or smallest", () => {
    const desc = cols({ sort: "value" }).rows.map((r) => r.values[0] ?? 0);
    expect(desc).toEqual([...desc].sort((a, b) => b - a));
    const asc = cols({ sort: "valueAsc" }).rows.map((r) => r.values[0] ?? 0);
    expect(asc).toEqual([...asc].sort((a, b) => a - b));
    // Asserted on values[0] and not on the row maximum, which is the whole
    // distinction. Comparing against an ordering BY the second column was the
    // first version of this and it was worthless: whether the two orders
    // differ depends on the fixture, so it could pass by luck.
  });

  it("sorts by name when asked", () => {
    const az = cols({ sort: "label" }).rows.map((r) => r.label);
    expect(az).toEqual([...az].sort((a, b) => a.localeCompare(b)));
  });

  it("never lets a sort overrule a comparison", () => {
    const all = cols().rows.map((r) => r.label);
    const pair = [all[2], all[0]];
    expect(cols({ keep: pair, sort: "label" }).rows.map((r) => r.label)).toEqual(pair);
  });
});
