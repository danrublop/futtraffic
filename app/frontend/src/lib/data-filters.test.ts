import { describe, expect, it } from "vitest";

import {
  byClub,
  byMatch,
  byNewest,
  clubRowsIn,
  confirmedIn,
  fixturesIn,
  feedClubFor,
  fixturesInvolving,
  matchRows,
  measuredById,
  seasonsOf,
  teamsInSeason,
} from "./data-filters";
import type { ClubRow, Fixture, PlayerRow } from "./season";
import type { ClubPlayer, Match, Season, Team } from "./types";

const fixture = (id: string, home: string, away: string): Fixture => ({
  id,
  home,
  away,
  score: "1 - 0",
  date: null,
  competition: null,
  round: null,
  demo: false,
});

const playerRow = (match: string, club: string, name: string): PlayerRow => ({
  match,
  club,
  opponent: "x",
  name,
  pos: null,
  stats: {},
});

const clubRow = (match: string, club: string): ClubRow => ({
  match,
  club,
  opponent: "x",
  home: true,
  gf: 1,
  ga: 0,
  stats: {},
});

const job = (id: string, created: number | null, name = id): Match =>
  ({ id, created, name, status: "done" }) as unknown as Match;

const team = (team_id: string, name: string, season: string): Team =>
  ({ team_id, name, season, age_band: "", gender: "", level: "", coach_ids: [] }) as unknown as Team;

describe("the feed's players", () => {
  const rows = [
    playerRow("m1", "Northbridge FC", "Varga"),
    playerRow("m1", "Eastvale FC", "Lindqvist"),
    playerRow("m2", "Northbridge FC", "Okafor"),
  ];

  it("narrows to one club", () => {
    expect(byClub(rows, "Eastvale FC").map((r) => r.name)).toEqual(["Lindqvist"]);
  });

  it("narrows to one match", () => {
    expect(byMatch(rows, "m2").map((r) => r.name)).toEqual(["Okafor"]);
  });

  it("composes", () => {
    expect(byMatch(byClub(rows, "Northbridge FC"), "m1").map((r) => r.name)).toEqual(["Varga"]);
  });

  it("treats an empty string as every, not none", () => {
    expect(byClub(rows, "")).toHaveLength(3);
    expect(byMatch(rows, "")).toHaveLength(3);
  });

  it("returns nothing for a value that matches nothing", () => {
    // Not everything. A filter that silently widens is worse than an empty table.
    expect(byClub(rows, "Westford FC")).toEqual([]);
  });
});

describe("fixtures", () => {
  const all = [fixture("m1", "Northbridge FC", "Eastvale FC"), fixture("m2", "Westford FC", "Northbridge FC")];

  it("narrows to a set of match ids", () => {
    expect(fixturesIn(all, new Set(["m2"])).map((f) => f.id)).toEqual(["m2"]);
  });

  it("CRITICAL an empty SET selects nothing; null is no filter", () => {
    // An empty set is a selection that matched nothing. Returning the whole
    // library for it answers a different question from the one on screen.
    expect(fixturesIn(all, new Set())).toEqual([]);
    expect(fixturesIn(all, null)).toHaveLength(2);
  });

  it("finds a club on either side of the fixture", () => {
    expect(fixturesInvolving(all, "Northbridge FC").map((f) => f.id)).toEqual(["m1", "m2"]);
    expect(fixturesInvolving(all, "Westford FC").map((f) => f.id)).toEqual(["m2"]);
    expect(fixturesInvolving(all, "")).toHaveLength(2);
  });
});

describe("club rows", () => {
  const rows = [clubRow("m1", "Northbridge FC"), clubRow("m1", "Eastvale FC"), clubRow("m2", "Westford FC")];

  it("narrows by matches and by club together", () => {
    expect(clubRowsIn(rows, new Set(["m1"]), "Eastvale FC")).toHaveLength(1);
  });

  it("an empty set selects nothing here too", () => {
    expect(clubRowsIn(rows, new Set(), "Eastvale FC")).toEqual([]);
  });

  it("null ids with no club is everything", () => {
    expect(clubRowsIn(rows, null)).toHaveLength(3);
  });
});

describe("our registry, narrowed to one match", () => {
  const players = [
    { id: "varga", name: "Varga", position: null, teams: [], matches: 2 },
    { id: "okafor", name: "Okafor", position: null, teams: [], matches: 1 },
    { id: "unbound", name: "Nobody Confirmed", position: null, teams: [], matches: 0 },
  ] as unknown as ClubPlayer[];

  const season = {
    team: "",
    matches: [],
    coverage: { bound: 3, total: 22 },
    players: [
      { id: "varga", by_match: { m1: {}, m2: {} } },
      { id: "okafor", by_match: { m1: {} } },
    ],
  } as unknown as Season;

  const measured = measuredById(season);

  it("keeps only the players with a confirmed track in that match", () => {
    expect(confirmedIn(players, measured, "m2").map((p) => p.id)).toEqual(["varga"]);
  });

  it("drops a player with no measured row at all", () => {
    // "Was this person confirmed in this match" is false for them. That is not
    // the same as the em dash in an unfiltered table, which means "not
    // confirmed anywhere yet".
    expect(confirmedIn(players, measured, "m1").map((p) => p.id)).toEqual([
      "varga",
      "okafor",
    ]);
  });

  it("an empty match id is every player, including the unconfirmed one", () => {
    expect(confirmedIn(players, measured, "")).toHaveLength(3);
  });

  it("survives a measured row with no by_match at all", () => {
    // match.json field shapes have migrated before; a required TS type is not a
    // runtime contract for file input.
    const thin = new Map([["varga", {} as never]]);
    expect(() => confirmedIn(players, thin, "m1")).not.toThrow();
    expect(confirmedIn(players, thin, "m1")).toEqual([]);
  });

  it("measuredById on a null season is an empty map, not a crash", () => {
    expect(measuredById(null).size).toBe(0);
  });
});

describe("squads by season", () => {
  const teams = [team("a", "U15", "2025/26"), team("b", "First team", "2026/27"), team("c", "U9", "")];

  it("narrows to one season", () => {
    expect(teamsInSeason(teams, "2026/27").map((t) => t.team_id)).toEqual(["b"]);
  });

  it("an empty season is every squad, including the one with none set", () => {
    expect(teamsInSeason(teams, "")).toHaveLength(3);
  });

  it("lists the seasons the ladder mentions, newest first, ignoring blanks", () => {
    expect(seasonsOf(teams)).toEqual(["2026/27", "2025/26"]);
  });
});

describe("the Matches join", () => {
  const jobs = [job("m1", 100), job("m2", 300), job("m3", 200)];
  const fixtures = [fixture("m1", "Northbridge FC", "Eastvale FC")];

  it("CRITICAL every job gets a row, with or without a feed", () => {
    // The whole point. A fixture-derived list is permanently EMPTY for an
    // academy filming its own matches, which is the primary persona.
    const rows = matchRows(jobs, fixtures);
    expect(rows).toHaveLength(3);
    expect(rows.map((r) => r.match.id)).toEqual(["m1", "m2", "m3"]);
  });

  it("attaches the fixture where the feed has one, null where it does not", () => {
    const rows = matchRows(jobs, fixtures);
    expect(rows[0].fixture?.home).toBe("Northbridge FC");
    expect(rows[1].fixture).toBeNull();
    expect(rows[2].fixture).toBeNull();
  });

  it("with no feed at all every row still exists", () => {
    expect(matchRows(jobs, [])).toHaveLength(3);
  });

  it("ignores a fixture with no job — the job list is the spine", () => {
    // A fixture whose job was deleted must not resurrect the match.
    const rows = matchRows(jobs, [...fixtures, fixture("gone", "A", "B")]);
    expect(rows.map((r) => r.match.id)).not.toContain("gone");
  });

  it("sorts newest first and puts an undated job last, not first", () => {
    const rows = byNewest(matchRows([...jobs, job("m4", null)], []));
    expect(rows.map((r) => r.match.id)).toEqual(["m2", "m3", "m1", "m4"]);
  });

  it("does not mutate its input", () => {
    const rows = matchRows(jobs, []);
    byNewest(rows);
    expect(rows.map((r) => r.match.id)).toEqual(["m1", "m2", "m3"]);
  });
});

describe("which club the feed calls our squad", () => {
  // Two clubs per match; ours is the one in all of them.
  const rows = [
    clubRow("m1", "Northbridge FC"),
    clubRow("m1", "Eastvale FC"),
    clubRow("m2", "Northbridge FC"),
    clubRow("m2", "Westford FC"),
    clubRow("m3", "Northbridge FC"),
    clubRow("m3", "Southmoor FC"),
  ];

  it("finds us behaviourally, not by name", () => {
    // The squad is called "First team" and the feed says "Northbridge FC". A name
    // comparison returns nothing; this returns the right club.
    expect(feedClubFor(rows, new Set(["m1", "m2", "m3"]))).toBe("Northbridge FC");
  });

  it("works on a subset of the squad's matches", () => {
    expect(feedClubFor(rows, new Set(["m1", "m2"]))).toBe("Northbridge FC");
  });

  it("REFUSES on one match, where both clubs are in all of it", () => {
    // Guessing here would attribute an opponent's season to our squad, which is
    // worse than showing nothing.
    expect(feedClubFor(rows, new Set(["m1"]))).toBe("");
  });

  it("refuses when no club spans every match", () => {
    // Two squads' matches mixed together: nobody is in all of them.
    const mixed = [clubRow("m1", "A"), clubRow("m1", "B"), clubRow("m2", "C"), clubRow("m2", "D")];
    expect(feedClubFor(mixed, new Set(["m1", "m2"]))).toBe("");
  });

  it("refuses a tie rather than picking one", () => {
    // The same two clubs played each other twice: both span every match.
    const twice = [clubRow("m1", "A"), clubRow("m1", "B"), clubRow("m2", "A"), clubRow("m2", "B")];
    expect(feedClubFor(twice, new Set(["m1", "m2"]))).toBe("");
  });

  it("refuses when there is no feed at all", () => {
    expect(feedClubFor([], new Set(["m1", "m2"]))).toBe("");
    expect(feedClubFor(rows, new Set())).toBe("");
  });
});
