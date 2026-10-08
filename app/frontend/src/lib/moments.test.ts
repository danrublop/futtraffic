import { describe, expect, it } from "vitest";

import { demoMoments, matchesQuery, minuteSecs, moments } from "./moments";
import type { MatchReport } from "./types";

describe("minuteSecs", () => {
  it("reads a minute, stoppage included, and refuses a period row", () => {
    expect(minuteSecs("48'")).toBe(2880);
    expect(minuteSecs("45+2'")).toBe(2820);
    expect(minuteSecs("HT")).toBeNull();
  });
});

describe("moments", () => {
  const report = {
    meta: {
      facts: {
        events: [
          { minute: "HT", type: "period", score: "1 - 0" },
          { minute: "11'", side: "home", type: "goal", player: "A", score: "1 - 0" },
        ],
        plays: [
          { id: "p1", type: "counter_attack", side: "away", period: 1, minute: 9, clock: "9:33", zone: "final_third", outcome: "shot", players: [], board: null, t: 573 },
        ],
      },
    },
  } as unknown as MatchReport;

  it("merges plays and feed events in time order, marking which clock each came off", () => {
    const m = moments(report, [{ t: 100, team: "home", label: "turnover" }]);
    expect(m.map((x) => [x.t, x.exact])).toEqual([
      [100, true],
      [573, true],
      [660, false],
    ]);
    // A period row carries no minute, so it is not seekable and is dropped.
    expect(m.some((x) => x.label.includes("1 - 0") && !x.exact)).toBe(true);
    expect(m[1].label).toBe("counter attack");
  });

  it("is empty, not broken, on a match with no facts", () => {
    expect(moments(null)).toEqual([]);
  });

  it("folds clips in by start time and keeps their length", () => {
    const clip = { id: "c1", start: 300, end: 312, name: "counter", tags: ["Press"], note: "" };
    const m = moments(report, [], [clip]);
    expect(m[0]).toMatchObject({ t: 300, end: 312, kind: "clip", label: "counter", exact: true });
    expect(m.map((x) => x.t)).toEqual([300, 573, 660]);
  });
});

/** Regression: every field below comes off a match.json on disk, and moments()
 *  runs in a useMemo during render. Before these guards a play written without
 *  `players` threw a TypeError mid-render, which blanks the film page rather
 *  than degrading it. The pipeline has migrated this file's shape before. */
describe("moments — a match.json missing fields it used to have", () => {
  const play = (extra: Record<string, unknown>) =>
    ({
      meta: {
        facts: {
          plays: [
            { id: "p1", type: "high_press", side: "home", period: 1, minute: 1,
              clock: "1:00", outcome: "shot", board: null, t: 60, ...extra },
          ],
        },
      },
    }) as unknown as MatchReport;

  it("survives a play with no players array", () => {
    const [m] = moments(play({ zone: "final_third" }));
    expect(m.who).toEqual([]);
    expect(m.label).toBe("high press");
  });

  it("survives a play with no zone, and does not print a leading separator", () => {
    const [m] = moments(play({ players: [7] }));
    expect(m.detail).toBe("shot");
  });

  it("survives a clip with no tags", () => {
    const clip = { id: "c1", start: 1, end: 2, name: "x", note: "" };
    const [m] = moments(null, [], [clip as never]);
    expect(m.who).toEqual([]);
    expect(m.detail).toBeUndefined();
  });

  it("names a play whose type is missing rather than rendering an empty row", () => {
    const [m] = moments(play({ type: undefined, players: [], zone: "box" }));
    expect(m.label).toBe("play");
  });
});

describe("matchesQuery", () => {
  const [play] = moments({
    meta: {
      facts: {
        plays: [
          { id: "p1", type: "high_press", side: "home", period: 1, minute: 1, clock: "1:00", zone: "final_third", outcome: "shot", players: [7, 9], board: null, t: 60 },
        ],
      },
    },
  } as unknown as MatchReport);

  it("reads the label, the detail and the names, and passes everything on empty", () => {
    expect(matchesQuery(play, "press")).toBe(true);
    expect(matchesQuery(play, "FINAL third")).toBe(true);
    expect(matchesQuery(play, "#9")).toBe(true);
    expect(matchesQuery(play, "  ")).toBe(true);
    expect(matchesQuery(play, "corner")).toBe(false);
  });
});

describe("demoMoments", () => {
  it("fills a match in time order, and the same way every time", () => {
    const a = demoMoments(3600);
    const b = demoMoments(3600);
    expect(a.length).toBeGreaterThan(50);
    expect(a).toEqual(b);
    expect(a.map((m) => m.t)).toEqual([...a.map((m) => m.t)].sort((x, y) => x - y));
    expect(a.every((m) => m.t <= 3600 && m.demo && m.exact)).toBe(true);
    // Both sides play, and every row names somebody.
    expect(new Set(a.map((m) => m.side))).toEqual(new Set(["home", "away"]));
    expect(a.every((m) => m.who.length === 1)).toBe(true);
    // A different match is a different sample.
    expect(demoMoments(2700)[0]).not.toEqual(a[0]);
  });
});
