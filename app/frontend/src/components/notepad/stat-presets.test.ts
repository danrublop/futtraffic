import { describe, expect, it } from "vitest";

import type { MatchReport } from "@/lib/types";
import { BUILT_IN, hasKeeper, presetsFor } from "./stat-presets";

/** The named stat sets.
 *
 *  What matters here is that they are the REPORT's groupings and not a second
 *  opinion — a hand-kept copy would drift the first time somebody added a
 *  column, and then "Defense" would mean two things in one product — and that
 *  a set is only offered when the source and the match can answer it.
 */

const withKeeper = {
  meta: { facts: { player_stats: [{ pos: "Goalkeeper" }, { pos: "Midfielder" }] } },
} as unknown as MatchReport;
const outfieldOnly = {
  meta: { facts: { player_stats: [{ pos: "Midfielder" }, { pos: "Defender" }] } },
} as unknown as MatchReport;

/** A source, shaped the way the panel hands them in. */
const src = (id: string, ids: string[]) => ({ id, measures: ids.map((m) => ({ id: m })) });

/** One source that can answer everything the built-ins ask. */
const everything = [src("feed", [...new Set(BUILT_IN.flatMap((p) => p.cols))])];

describe("which sets are offered", () => {
  it("carries the report's own groupings rather than a second list", () => {
    expect(BUILT_IN.map((p) => p.label)).toEqual([
      "Top stats",
      "Attack",
      "Passes",
      "Defense",
      "Duels",
      "Goalkeeping",
    ]);
  });

  it("offers goalkeeping only when somebody in the match kept goal", () => {
    const on = presetsFor(everything, withKeeper).map((p) => p.label);
    const off = presetsFor(everything, outfieldOnly).map((p) => p.label);
    expect(on).toContain("Goalkeeping");
    expect(off).not.toContain("Goalkeeping");
    // The rest are unaffected either way.
    expect(off).toContain("Defense");
  });

  it("drops a set the source cannot answer", () => {
    // The tracks know distance and nothing about tackles.
    const physical = [src("tracks", ["distance_km", "top_speed_kmh", "sprints", "minutes"])];
    expect(presetsFor(physical, withKeeper)).toEqual([]);
  });

  it("narrows a set to the columns that exist", () => {
    // rating and minutes are in Top stats AND Defense, which is why both
    // survive here — the narrowing is per set, not a partition.
    const out = presetsFor([src("feed", ["rating", "minutes", "goals", "tackles"])], outfieldOnly);
    expect(out.find((p) => p.label === "Top stats")?.cols).toEqual(["rating", "minutes", "goals"]);
    expect(out.find((p) => p.label === "Defense")?.cols).toEqual(["rating", "minutes", "tackles"]);
  });

  it("drops a set the source narrows to a single column", () => {
    // One measure is not a SET, and offering it as one would be a chip that
    // does the same thing as the chip beside it.
    const out = presetsFor([src("feed", ["goals", "tackles"])], outfieldOnly);
    expect(out).toEqual([]);
  });

  it("puts a coach's saved sets after the built-in ones, marked as theirs", () => {
    const out = presetsFor(everything, withKeeper, [
      { name: "Press", cols: ["tackles", "interceptions"] },
    ]);
    const mine = out[out.length - 1];
    expect(mine.label).toBe("Press");
    expect(mine.saved).toBe(true);
    expect(out.filter((p) => p.saved)).toHaveLength(1);
  });

  it("drops a saved set whose columns this source has none of", () => {
    const out = presetsFor([src("tracks", ["distance_km", "sprints"])], withKeeper, [
      { name: "Press", cols: ["tackles"] },
    ]);
    expect(out.filter((p) => p.saved)).toEqual([]);
  });
});

describe("finding a keeper", () => {
  it("asks the FEED, not the roster", () => {
    // A keeper who did not play is not in this match's numbers, and a
    // Goalkeeping set resolving to eight empty columns is worse than absent.
    expect(hasKeeper(withKeeper)).toBe(true);
    expect(hasKeeper(outfieldOnly)).toBe(false);
  });

  it("says no rather than throwing on a match with no feed", () => {
    expect(hasKeeper(null)).toBe(false);
    expect(hasKeeper({} as MatchReport)).toBe(false);
    expect(hasKeeper({ meta: { facts: {} } } as unknown as MatchReport)).toBe(false);
  });

  it("ignores a feed that is not a list", () => {
    expect(hasKeeper({ meta: { facts: { player_stats: "nope" } } } as unknown as MatchReport)).toBe(false);
  });
});

describe("which source answers a set", () => {
  const src = (id: string, ids: string[]) => ({ id, measures: ids.map((m) => ({ id: m })) });

  it("hands a set to the source that can answer MOST of it", () => {
    // `rating` and `minutes` exist in the squad list AND the feed, so "first
    // source with any overlap" gave Defense to the squad on the strength of
    // two columns it shares by accident — and the panel then said "7 of 6"
    // with every chip greyed out.
    const out = presetsFor(
      [
        src("squad", ["rating", "minutes", "age"]),
        src("feed", ["rating", "minutes", "def", "tackles", "interceptions", "clearances", "blocks"]),
      ],
      { meta: { facts: { player_stats: [{ pos: "Midfielder" }] } } } as never,
    );
    const def = out.find((p) => p.label === "Defense");
    expect(def?.source).toBe("feed");
    expect(def?.cols).toHaveLength(7);
  });

  it("gives every preset a source, so applying one can never miss", () => {
    const out = presetsFor(
      [src("feed", [...new Set(BUILT_IN.flatMap((p) => p.cols))])],
      { meta: { facts: { player_stats: [{ pos: "Goalkeeper" }] } } } as never,
      [{ name: "Press", cols: ["tackles", "interceptions"] }],
    );
    expect(out.length).toBeGreaterThan(0);
    expect(out.every((p) => p.source === "feed")).toBe(true);
  });
});
