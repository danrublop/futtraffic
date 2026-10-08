import { describe, expect, it } from "vitest";

import {
  agg,
  scoreOf,
  clubTotals,
  labelCoverage,
  opponentTotals,
  playerTotals,
  sectionsFor,
  type ClubRow,
  type PlayerRow,
} from "./season";

/** The shape agg() and labelCoverage() take: anything carrying a stats bag. */
type Row = { stats: Record<string, number> };

const club = (
  match: string,
  club: string,
  opponent: string,
  gf: number | null,
  ga: number | null,
  stats: Record<string, number> = {},
): ClubRow => ({ match, club, opponent, home: true, gf, ga, stats });

const player = (
  match: string,
  name: string,
  club: string,
  stats: Record<string, number>,
  pos: string | null = "Midfielder",
): PlayerRow => ({ match, club, opponent: "X", name, pos, stats });

describe("agg", () => {
  it("sums counts and counts the rows that carried the stat", () => {
    const rows: Row[] = [{ stats: { goals: 2 } }, { stats: {} }, { stats: { goals: 1 } }];
    expect(agg(rows, "goals")).toEqual({ value: 3, n: 2 });
  });

  it("means a rate instead of summing it", () => {
    // Two possessions summed is 121%, which is the bug this list exists for.
    const rows: Row[] = [{ stats: { "Ball possession": 60 } }, { stats: { "Ball possession": 40 } }];
    expect(agg(rows, "Ball possession")).toEqual({ value: 50, n: 2 });
  });

  it("means over the rows that HAVE it, not over all rows", () => {
    const rows: Row[] = [{ stats: { "Ball possession": 60 } }, { stats: {} }];
    expect(agg(rows, "Ball possession")).toEqual({ value: 60, n: 1 });
  });

  it("reports n=0 rather than inventing a zero", () => {
    expect(agg([{ stats: {} }], "sot")).toEqual({ value: 0, n: 0 });
  });

  it("ignores a non-finite value instead of poisoning the total", () => {
    const rows: Row[] = [{ stats: { goals: 2 } }, { stats: { goals: NaN } }];
    expect(agg(rows, "goals")).toEqual({ value: 2, n: 1 });
  });
});

describe("clubTotals", () => {
  const rows = [
    club("m1", "Northbridge FC", "Eastvale FC", 2, 1),
    club("m2", "Northbridge FC", "Getafe", 1, 1),
    club("m3", "Northbridge FC", "Osasuna", 0, 3),
    club("m1", "Eastvale FC", "Northbridge FC", 1, 2),
  ];

  it("tallies a record and goals per club", () => {
    const [rm, barca] = clubTotals(rows);
    expect(rm).toMatchObject({ club: "Northbridge FC", played: 3, w: 1, d: 1, l: 1, gf: 3, ga: 5 });
    expect(barca).toMatchObject({ club: "Eastvale FC", played: 1, w: 0, d: 0, l: 1 });
  });

  it("orders by matches played, so the library's own club leads", () => {
    expect(clubTotals(rows).map((c) => c.club)).toEqual(["Northbridge FC", "Eastvale FC"]);
  });

  it("counts an unscored match as played but not as a result", () => {
    const [rm] = clubTotals([...rows, club("m4", "Northbridge FC", "Levante", null, null)]);
    expect(rm.played).toBe(4);
    expect(rm.w + rm.d + rm.l).toBe(3);
    expect(rm.gf).toBe(3); // unchanged: a null score adds no goals
  });
});

describe("opponentTotals", () => {
  it("cuts one club's season by who it faced", () => {
    const rows = [
      club("m1", "Northbridge FC", "Eastvale FC", 2, 1),
      club("m2", "Northbridge FC", "Eastvale FC", 0, 1),
      club("m3", "Northbridge FC", "Getafe", 3, 0),
      club("m1", "Eastvale FC", "Northbridge FC", 1, 2),
    ];
    const got = opponentTotals(rows, "Northbridge FC");
    expect(got.map((o) => o.club)).toEqual(["Eastvale FC", "Getafe"]);
    expect(got[0]).toMatchObject({ played: 2, w: 1, l: 1, gf: 2, ga: 2 });
    // Eastvale FC's own row must not leak into Northbridge FC's opponents.
    expect(got.some((o) => o.club === "Northbridge FC")).toBe(false);
  });
});

describe("playerTotals", () => {
  it("groups by name AND club, so a transfer is two rows", () => {
    const rows = [
      player("m1", "A Player", "Northbridge FC", { goals: 1 }),
      player("m2", "A Player", "Northbridge FC", { goals: 2 }),
      player("m3", "A Player", "Getafe", { goals: 1 }),
    ];
    const got = playerTotals(rows);
    expect(got).toHaveLength(2);
    expect(got[0]).toMatchObject({ name: "A Player", club: "Northbridge FC", apps: 2 });
    expect(agg(got[0].rows, "goals")).toEqual({ value: 3, n: 2 });
  });

  it("takes the most recent position, not the first", () => {
    const got = playerTotals([
      player("m1", "P", "C", {}, "Defender"),
      player("m2", "P", "C", {}, "Attacker"),
    ]);
    expect(got[0].pos).toBe("Attacker");
  });
});

describe("labelCoverage", () => {
  it("reports how many rows carry each label, commonest first", () => {
    const rows: Row[] = [
      { stats: { minutes: 90, goals: 1 } },
      { stats: { minutes: 90 } },
      { stats: { minutes: 45, sot: 2 } },
    ];
    expect(labelCoverage(rows)).toEqual([
      { label: "minutes", n: 3 },
      { label: "goals", n: 1 },
      { label: "sot", n: 1 },
    ]);
  });
});

describe("sectionsFor", () => {
  // Mirrors the match report's own grouping so a season reads the way a single
  // match does.
  const rows = [
    { stats: { "Ball possession": 60, "Total shots": 10, Tackles: 4, Weird: 1 } },
    { stats: { "Ball possession": 40, "Total shots": 6, Tackles: 2, Weird: 3 } },
  ];

  it("groups the labels the way the report does", () => {
    const got = sectionsFor(rows);
    expect(got.map((s) => s.title)).toEqual(["Top stats", "Shots", "Defence", "Other"]);
  });

  it("means a rate and sums a count, the same as anywhere else", () => {
    const top = sectionsFor(rows).find((s) => s.title === "Top stats")!;
    expect(top.rows.find((r) => r.label === "Ball possession")!.agg.value).toBe(50);
    expect(top.rows.find((r) => r.label === "Total shots")!.agg.value).toBe(16);
  });

  it("CRITICAL sweeps an unknown label into Other rather than dropping it", () => {
    // The grouping is mirrored from the feed, so the feed can add a label this
    // file has never heard of. Losing a number the feed supplied would be worse
    // than showing it in the wrong group.
    const other = sectionsFor(rows).find((s) => s.title === "Other")!;
    expect(other.rows.map((r) => r.label)).toEqual(["Weird"]);
    expect(other.rows[0].agg.value).toBe(4);
  });

  it("omits a section whose labels are absent entirely", () => {
    // A row of em dashes teaches nothing.
    expect(sectionsFor([{ stats: { Tackles: 1 } }]).map((s) => s.title)).toEqual(["Defence"]);
  });

  it("survives rows with no stats at all", () => {
    expect(sectionsFor([{ stats: {} }])).toEqual([]);
    expect(sectionsFor([])).toEqual([]);
  });
});

/** THE SCORE PARSER, because every W/D/L tally, goals-for, goals-against and
 *  clean-sheet count in the Data tables is derived from it. The dash cases are
 *  the ones that mattered: an unplayed fixture used to read as a 0-0 draw. */
describe("scoreOf", () => {
  it("reads a played score, either dash", () => {
    expect(scoreOf("2 - 1")).toEqual([2, 1]);
    expect(scoreOf("0-0")).toEqual([0, 0]);
    expect(scoreOf(" 3 – 4 ")).toEqual([3, 4]);
  });

  it("refuses anything that is not two numbers", () => {
    // "-" is what an unplayed fixture carries, and it must not become 0-0.
    for (const s of ["-", " - ", "–", "", null, undefined, "3", "x-y", "2 - "])
      expect(scoreOf(s)).toBeNull();
  });
});
