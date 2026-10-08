import { describe, expect, it } from "vitest";

import type { TeamStat } from "@/lib/types";

import { aggregate, goalsFor } from "./squads-section";

const stat = (value: number | null, raw: string | null = null, pct: number | null = null): TeamStat => ({
  value,
  raw,
  pct,
});

/** THE SEASON TOTAL, which is the number a coach reads off the Squads table.
 *  Every case here is a way to print a believable wrong number: a share added
 *  up instead of averaged, a count averaged instead of added, metres printed
 *  as if they were kilometres. None of them throws. */
describe("aggregate", () => {
  it("averages a share, never adds it", () => {
    // Two matches at 52% and 68% possession is a 60% season, not 120%.
    expect(aggregate("Ball possession", [stat(52, "52%"), stat(68, "68%")], false)).toEqual({
      text: "60%",
      num: 60,
    });
    // Per 90 changes nothing for a share: it is already a rate.
    expect(aggregate("Ball possession", [stat(52, "52%"), stat(68, "68%")], true)?.text).toBe("60%");
  });

  it("adds a count, and per 90 divides by the matches that have it", () => {
    expect(aggregate("Total shots", [stat(12), stat(17)], false)).toEqual({ text: "29", num: 29 });
    expect(aggregate("Total shots", [stat(12), stat(17)], true)).toEqual({ text: "14.5", num: 14.5 });
  });

  it("adds a count that carries a share, and averages the share beside it", () => {
    const passes = [stat(343, "343 (80%)", 80), stat(567, "567 (90%)", 90)];
    expect(aggregate("Accurate passes", passes, false)).toEqual({ text: "910 (85%)", num: 910 });
  });

  it("reads distance in kilometres", () => {
    const d = [stat(108701), stat(106164)];
    expect(aggregate("Distance covered", d, false)?.text).toBe("214.9 km");
    expect(aggregate("Distance covered", d, true)?.text).toBe("107.4 km");
  });

  it("prints xG to two places, and signs a positive difference", () => {
    expect(aggregate("Expected goals (xG)", [stat(2.35), stat(1.67)], false)?.text).toBe("4.02");
    expect(aggregate("xG difference", [stat(1.2), stat(-0.5)], false)?.text).toBe("+0.70");
    expect(aggregate("xG difference", [stat(-0.4), stat(-0.3)], false)?.text).toBe("-0.70");
  });

  it("skips matches without the stat rather than counting them as zero", () => {
    // Per 90 over the two matches that HAVE it, not the three in scope.
    expect(aggregate("Total shots", [stat(12), stat(null), stat(18)], true)?.num).toBe(15);
  });

  it("says nothing when no match has the stat", () => {
    expect(aggregate("Total shots", [stat(null), stat(null)], false)).toBeNull();
    expect(aggregate("Total shots", [], false)).toBeNull();
  });
});

/** WHOSE GOALS. The Opponents table reads every match from the other end, so
 *  the same "2 - 1" is a win for one table and a loss for the other. A wrong
 *  flip swaps two plausible columns and hands the clean sheets to the wrong
 *  team — the acting-team-frame class of bug. */
describe("goalsFor", () => {
  it("reads our side of the score", () => {
    expect(goalsFor("2 - 1", true, false)).toEqual({ scored: 2, conceded: 1 });
    expect(goalsFor("2 - 1", false, false)).toEqual({ scored: 1, conceded: 2 });
  });

  it("flips to the opponent's side for the Opponents table", () => {
    // We were home and won 2-1: the opponent scored 1 and conceded 2.
    expect(goalsFor("2 - 1", true, true)).toEqual({ scored: 1, conceded: 2 });
    expect(goalsFor("2 - 1", false, true)).toEqual({ scored: 2, conceded: 1 });
  });

  it("keeps a clean sheet with the side that kept it", () => {
    // Away at 0-3: we conceded nothing, the home side conceded three.
    expect(goalsFor("0 - 3", false, false)?.conceded).toBe(0);
    expect(goalsFor("0 - 3", false, true)?.conceded).toBe(3);
  });

  it("refuses an unplayed fixture instead of calling it 0-0", () => {
    for (const s of ["-", " - ", "", null, undefined]) {
      expect(goalsFor(s, true, false)).toBeNull();
    }
  });
});
