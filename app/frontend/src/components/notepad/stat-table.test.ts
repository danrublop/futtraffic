import { describe, expect, it } from "vitest";

import { SPRINTS_WITHHELD_WHY } from "@/lib/report";
import type { MatchReport, Season } from "@/lib/types";
import { buildTable, defaultTitle } from "./stat-table";

const report = {
  meta: { home_name: "Westfield", away_name: "Rovers" },
  teams: {
    home: { distance_m: 104000, top_speed_kmh: 31.2, sprints: 88, shape: {} },
    away: { distance_m: 99000, top_speed_kmh: 30.1, sprints: 74, shape: {} },
  },
  players: [
    { id: 1, team: "home", jersey: 8, minutes: 90, distance_km: 11.2, top_speed_kmh: 30.1, sprints: 24 },
    { id: 2, team: "home", jersey: null, minutes: 45, distance_km: 5.1, top_speed_kmh: 28.0, sprints: 9 },
    { id: 3, team: "away", jersey: 4, minutes: 90, distance_km: 9.8, top_speed_kmh: 29.0, sprints: 17 },
  ],
} as unknown as MatchReport;

const season: Season = {
  team: "Westfield",
  matches: [
    { id: "a", name: "wk1", created: 1, home_name: null, away_name: null, score: null },
    { id: "b", name: "wk2", created: 2, home_name: null, away_name: null, score: null },
  ],
  players: [
    {
      id: "ana-ruiz", name: "Ana Ruiz", number: 8, position: "CM", team: "Westfield",
      matches: 2, minutes: 180, distance_km: 22.0, sprints: 40, top_speed_kmh: 30.1,
      km_per90: 11.0, sprints_per90: 20, by_match: {},
    },
    {
      id: "sam-lee", name: "Sam Lee", number: 4, position: "CB", team: "Westfield",
      matches: 1, minutes: 30, distance_km: 4.5, sprints: 4, top_speed_kmh: 27.0,
      km_per90: 13.5, sprints_per90: 12, by_match: {},
    },
    {
      id: "kit-vale", name: "Kit Vale", number: 6, position: "CM", team: "Westfield",
      matches: 2, minutes: 20, distance_km: 3.0, sprints: 6, top_speed_kmh: 29.0,
      km_per90: 13.5, sprints_per90: 27, by_match: {},
    },
  ],
  coverage: { bound: 3, total: 8 },
};

const build = (o: Partial<Parameters<typeof buildTable>[0]>) =>
  buildTable({ scope: "match", group: "player", metric: "distance_km", limit: 10, report, season: null, ...o });

describe("a match table", () => {
  it("ranks tracks by shirt number and never by a name it does not have", () => {
    const t = build({});
    // Track 2 has no shirt number, so it is a track id — never a made-up name.
    expect(t.rows.map((r) => r.name)).toEqual(["#8", "#4", "Track 2"]);
    expect(t.rows[0].value).toBe("11.20 km");
    expect(t.note).toMatch(/Shirt numbers, not names/);
  });

  it("scales the bars against the top row, not against the pitch", () => {
    const t = build({});
    expect(t.rows[0].frac).toBe(1);
    expect(t.rows[1].frac).toBeCloseTo(9.8 / 11.2, 5);
  });

  it("honours the row limit", () => {
    expect(build({ limit: 2 }).rows).toHaveLength(2);
  });

  it("groups by team off the team totals, in kilometres", () => {
    const t = build({ group: "team" });
    expect(t.rows.map((r) => r.name)).toEqual(["Westfield", "Rovers"]);
    expect(t.rows[0].value).toBe("104.00 km");
  });

  it("refuses a position grouping instead of showing an empty table", () => {
    const t = build({ group: "position" });
    expect(t.rows).toHaveLength(0);
    expect(t.refusal).toMatch(/roster/);
  });

  it("refuses a season-only metric rather than rendering zeroes", () => {
    expect(build({ metric: "km_per90" }).refusal).toBeTruthy();
    expect(build({ metric: "matches" }).refusal).toBeTruthy();
  });

  it("says so when there is no match at all", () => {
    expect(build({ report: null }).refusal).toMatch(/No match/);
  });
});

describe("a season table", () => {
  const s = (o: Partial<Parameters<typeof buildTable>[0]> = {}) =>
    buildTable({ scope: "season", group: "player", metric: "km_per90", limit: 10, report: null, season, ...o });

  it("ranks people, because a season row exists only where a human confirmed one", () => {
    expect(s().rows.map((r) => r.name)).toEqual(["Sam Lee", "Kit Vale", "Ana Ruiz"]);
  });

  it("always states what share of the tracking it rests on", () => {
    expect(s().note).toMatch(/3 of 8 tracked players across 2 matches/);
  });

  it("carries the match count into the row, so a one-game leader reads as one", () => {
    expect(s().rows[0].sub).toContain("1 match");
    expect(s().rows[2].sub).toContain("2 matches");
  });

  it("groups a position on its own total over its own minutes, not a mean of means", () => {
    const t = s({ group: "position", metric: "km_per90" });
    const cm = t.rows.find((r) => r.name === "CM")!;
    // (22.0 + 3.0) / (180 + 20) * 90 = 11.25 — NOT (11.0 + 13.5) / 2 = 12.25,
    // which would weight a 20-minute cameo the same as two full matches.
    expect(cm.value).toBe("11.25 km");
    expect(cm.sub).toBe("2 players");
  });

  it("says where to go when nobody has a confirmed identity", () => {
    const t = buildTable({
      scope: "season", group: "player", metric: "km_per90", limit: 10, report: null,
      season: { ...season, players: [], coverage: { bound: 0, total: 8 } },
    });
    expect(t.refusal).toMatch(/identity screen/);
  });

  it("says where to go when nobody has a position", () => {
    const t = buildTable({
      scope: "season", group: "position", metric: "km_per90", limit: 10, report: null,
      season: { ...season, players: season.players.map((p) => ({ ...p, position: null })) },
    });
    expect(t.refusal).toMatch(/typed on the roster/);
  });

  it("refuses a home/away grouping, because a club is both across a season", () => {
    expect(s({ group: "team" }).refusal).toMatch(/home and away/);
  });
});

describe("season sprints", () => {
  it("are withheld with the report's reason, whatever old rows carry (40, 20/90 above)", () => {
    for (const metric of ["sprints", "sprints_per90"])
      for (const group of ["player", "position"] as const) {
        const t = buildTable({ scope: "season", group, metric, limit: 10, report, season });
        expect(t.rows, `${metric}/${group}`).toEqual([]);
        expect(t.refusal).toBe(SPRINTS_WITHHELD_WHY);
        expect(t.tier).toBe("withheld");
      }
  });
});

describe("defaultTitle", () => {
  it("names the scope, so a table pasted into a report is not ambiguous", () => {
    expect(defaultTitle("season", "position", "km_per90")).toBe(
      "Distance / 90 by position — all matches",
    );
    expect(defaultTitle("match", "player", "sprints")).toBe("Sprints — this match");
  });
});

describe("tier notes", () => {
  it("say something for every tier weaker than ok", async () => {
    // A missing key printed the SAME empty note as `ok`, so not_evaluated -
    // the tier most fields resolve to - read as the sturdiest one.
    const { TIER_NOTE } = await import("./stat-node");
    const { TIER } = await import("@/lib/report");
    for (const t of Object.keys(TIER)) {
      if (t !== "ok") expect(TIER_NOTE[t], t).toBeTruthy();
    }
  });
});
