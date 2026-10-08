import { describe, expect, it } from "vitest";

import {
  FIELD_TIER,
  TIER,
  buildRows,
  gapNote,
  playerLabel,
  teamName,
  tierOf,
} from "./report";
import type { FieldEnvelope, MatchReport, PlayerStats } from "./types";

/** Read against report.ts at md5 ac2023943f9117f1fb49ad6ccd30336e (observed_frac
 *  and the shape/formation tiers are not in the file yet). Where a case pins
 *  behaviour that looks wrong rather than intended, the comment says so. */

const player = (over: Partial<PlayerStats> = {}): PlayerStats => ({
  id: 1,
  team: "home",
  jersey: null,
  minutes: 90,
  distance_km: 10,
  top_speed_kmh: 30,
  sprints: 12,
  avg_position: [52.5, 34],
  clamped_frac: 0,
  ...over,
});

const report = (players: PlayerStats[], meta?: MatchReport["meta"]): MatchReport => ({
  meta,
  players,
});

describe("teamName", () => {
  it("uses the names from meta when they are there", () => {
    const r = report([], { home_name: "Ashford", away_name: "Selby" });
    expect(teamName(r, "home")).toBe("Ashford");
    expect(teamName(r, "away")).toBe("Selby");
  });

  it("falls back to Home/Away for absent or empty names", () => {
    expect(teamName(report([]), "home")).toBe("Home");
    expect(teamName(report([], { home_name: "" }), "home")).toBe("Home");
    expect(teamName(report([]), "away")).toBe("Away");
  });

  it("calls anything else Unassigned", () => {
    expect(teamName(report([]), "unknown")).toBe("Unassigned");
  });
});

describe("playerLabel", () => {
  it("uses the shirt number once identity has run", () => {
    expect(playerLabel(player({ jersey: 7 }), report([], { home_name: "Ashford" }))).toBe(
      "Ashford #7",
    );
  });

  it("says 'track' when there is no shirt number, never a bare #", () => {
    const label = playerLabel(player({ id: 42, jersey: null }), report([]));
    expect(label).toBe("Home track 42");
    expect(label).not.toContain("#");
  });

  it("keeps #0 — a valid shirt number, not a missing one", () => {
    expect(playerLabel(player({ jersey: 0 }), report([]))).toBe("Home #0");
  });
});

describe("buildRows", () => {
  it("returns nothing for an empty squad, however it is empty", () => {
    expect(buildRows(report([]))).toEqual([]);
    expect(buildRows({})).toEqual([]);
  });

  it("gives a lone player the whole of their side's distance", () => {
    const [row] = buildRows(report([player({ distance_km: 8, minutes: 90 })]));
    expect(row.share).toBe(100);
    expect(row.distance).toBe(8);
    expect(row.per90).toBe(8);
  });

  it("shares distance within a side, not across the match", () => {
    const rows = buildRows(
      report([
        player({ id: 1, team: "home", distance_km: 6 }),
        player({ id: 2, team: "home", distance_km: 2 }),
        player({ id: 3, team: "away", distance_km: 5 }),
      ]),
    );
    expect(rows.map((r) => r.share)).toEqual([75, 25, 100]);
  });

  it("normalises distance to 90 minutes, so a sub is not ranked last by default", () => {
    const rows = buildRows(
      report([
        player({ id: 1, minutes: 90, distance_km: 9 }),
        player({ id: 2, minutes: 30, distance_km: 4 }),
      ]),
    );
    expect(rows[0].per90).toBe(9);
    expect(rows[1].per90).toBe(12);
    // The totals still say the opposite, which is exactly why per90 exists.
    expect(rows[1].distance!).toBeLessThan(rows[0].distance!);
  });

  it("gives an unplayed player a zero per90 rather than a division by zero", () => {
    const [row] = buildRows(report([player({ minutes: 0, distance_km: 0 })]));
    expect(row.per90).toBe(0);
    expect(row.minutes).toBe(0);
    expect(row.share).toBe(0);
  });

  it("shows zero, not undefined, for metrics the backend left out", () => {
    const [row] = buildRows(
      report([
        player({
          minutes: undefined,
          distance_km: undefined,
          top_speed_kmh: undefined,
          sprints: undefined,
        }),
      ]),
    );
    expect(row.minutes).toBe(0);
    // REPINNED 2026-09-28: distance can be REFUSED (null) now, so absent reads as unknown, not 0.
    expect(row.distance).toBeNull();
    // REPINNED 2026-08-19, was toBe(0). Peak speed now refuses (null) wherever
    // the 9.5 m/s clamp bound, so `|| 0` became `?? null` in buildRows and an
    // absent peak reads as "we will not say" rather than as a stationary
    // player. 0 was the old value and would now be a real reading of zero.
    expect(row.peak).toBeNull();
    expect(row.sprints).toBeNull();   // withheld, never 0
    expect(row.share).toBeNull();
  });

  it("per90 is null, not NaN, when minutes are present but distance is missing or refused", () => {
    const [row] = buildRows(report([player({ minutes: 45, distance_km: undefined })]));
    expect(row.per90).toBeNull();
  });

  it("carries a refused peak through as null, never as zero", () => {
    // stats.py sends null where MAX_SPEED bound: the only figure it could have
    // published there is 34.2 km/h, which is the clamp, not the player. A 0
    // here would file that player among the slowest in the match.
    const rows = buildRows(
      report([
        player({ id: 1, top_speed_kmh: null }),
        player({ id: 2, top_speed_kmh: 27.4 }),
      ]),
    );
    expect(rows[0].peak).toBeNull();
    expect(rows[1].peak).toBe(27.4);
  });

  it("keeps the source player on the row, in input order", () => {
    const rows = buildRows(report([player({ id: 9 }), player({ id: 4 })]));
    expect(rows.map((r) => r.p.id)).toEqual([9, 4]);
  });

  it("groups the unknown side on its own", () => {
    const rows = buildRows(
      report([
        player({ id: 1, team: "unknown", distance_km: 3 }),
        player({ id: 2, team: "home", distance_km: 7 }),
      ]),
    );
    expect(rows[0].share).toBe(100);
    expect(rows[0].label).toBe("Unassigned track 1");
  });
});

describe("TIER vocabulary", () => {
  // The point of the tier is that the caveat travels with the number. A tier
  // with no `why` is a footnote-free claim, which is what this file exists to
  // prevent.
  it.each(["not_evaluated", "ok", "rel", "prov"] as const)("%s has a label and a reason", (tier) => {
    expect(TIER[tier].label.length).toBeGreaterThan(0);
    expect(TIER[tier].why.length).toBeGreaterThan(20);
  });

  it("keeps the three words the report and player pages both print", () => {
    expect(TIER.ok.label).toBe("sturdiest");
    expect(TIER.rel.label).toBe("relative");
    expect(TIER.prov.label).toBe("provisional");
  });

  it("never claims anything is validated", () => {
    for (const t of Object.values(TIER)) {
      expect(t.label).not.toMatch(/validated/i);
      expect(t.why).not.toMatch(/\bvalidated\b/i);
    }
  });
});

/** tierOf prefers what the backend COMPUTED for this match over the static
 *  table. The two are hand-kept copies of one contract, and the moment a caveat
 *  moves in stats.py the table here would keep rendering a stale badge. These
 *  pin the resolution order so that cannot happen silently. */
describe("tierOf", () => {
  const env = (confidence: FieldEnvelope["confidence"]): FieldEnvelope => ({
    value: null,
    units: "m",
    n: 1,
    confidence,
    provenance: "derived",
    caveat_flags: [],
  });

  it("falls back to the static table when the match ships no envelopes", () => {
    expect(tierOf("players.minutes").tier).toBe("not_evaluated");
    expect(tierOf("players.distance_km").tier).toBe("prov");
  });

  it("is not_evaluated for a field nobody has classified", () => {
    expect(tierOf("who.knows").tier).toBe("not_evaluated");
    expect(tierOf("who.knows", { "a.b": env("ok") }).tier).toBe("not_evaluated");
  });

  it("prefers the computed envelope over the table", () => {
    // The table's fallback is the weakest tier; this match says provisional,
    // and the match still wins. This is the drift the split tables would hide.
    expect(tierOf("players.minutes", { "players.minutes": env("prov") }).tier).toBe("prov");
  });

  it("keeps the UI copy while taking the computed tier", () => {
    const withWhy = Object.keys(FIELD_TIER).find((k) => FIELD_TIER[k].why);
    expect(withWhy).toBeDefined();
    const got = tierOf(withWhy!, { [withWhy!]: env("ok") });
    expect(got.tier).toBe("ok");
    expect(got.why).toBe(FIELD_TIER[withWhy!].why);
  });

  /** The backend keys per side; a UI header stands over both. */
  it("expands a generic path to the per-side keys the backend emits", () => {
    const fields = {
      "teams.home.distance_m": env("ok"),
      "teams.away.distance_m": env("ok"),
    };
    expect(tierOf("teams.distance_m", fields).tier).toBe("ok");
  });

  it("takes the WEAKEST side, never the sturdier one", () => {
    const fields = {
      "teams.home.distance_m": env("ok"),
      "teams.away.distance_m": env("prov"),
    };
    // A header covering both cannot be sturdier than the shakier side under it.
    expect(tierOf("teams.distance_m", fields).tier).toBe("prov");
  });

  it("expands a leaf path too (structure_stress -> .home/.away)", () => {
    expect(tierOf("structure_stress", { "structure_stress.home": env("rel") }).tier).toBe("rel");
  });

  it("prefers an exact key over the per-side expansion", () => {
    const fields = {
      territory: env("rel"),
      "territory.home": env("prov"),
    };
    expect(tierOf("territory", fields).tier).toBe("rel");
  });
});

/** THE HONESTY INVARIANT, and the one this session broke.
 *
 *  Contamination is 33.2% on the shipped match: a third of the detections under
 *  a track id belong to somebody else. Every figure computed per player is
 *  therefore a figure about a MIXTURE, and there is no tier above `prov` that
 *  can honestly stand over one — including `rel`, which is the trap: a share
 *  looks like the sturdy reading because calibration cancels out of a ratio,
 *  but contamination does not cancel, it MOVES running between two rows of the
 *  same side, which is exactly what a share of the side total measures.
 *  `derived.share` sat at `rel` here while stats.py's envelope already said
 *  `prov`, and nothing failed.
 *
 *  The `why` half is not decoration. Every one of these already read `prov`
 *  before contamination was measured, so the BADGE did not move when the reason
 *  changed from "never measured" to "measured, and a third of this row is
 *  someone else". Only `why` carries that, so a missing one is a caveat that
 *  renders as nothing. */
describe("per-player figures", () => {
  const PER_PLAYER = [
    "players.distance_km",
    "players.top_speed_kmh",
    "players.avg_position",
    "derived.share",
    "derived.per90",
  ];

  it.each(PER_PLAYER)("%s is provisional and says a row is not one player", (field) => {
    const t = FIELD_TIER[field];
    expect(t).toBeDefined();
    expect(t.tier).toBe("prov");
    expect(t.why).toMatch(/not one player/i);
  });

  /** THE GUARD THAT REPLACED SEVEN PINNED LITERALS.
   *
   *  Until 2026-09-12 these tests pinned the VALUES in the copy — 15.1%, 1.190,
   *  70.7%, 22.03 — on the reasoning that a pinned number cannot go stale
   *  silently. It went stale anyway, in the only way a pin cannot catch: the
   *  numbers stayed equal to themselves while the thing they described moved.
   *  One match's figures (118575) printed on every match's report, several of
   *  them measured at a configuration `app/server/jobs.py` does not run, and peak
   *  speed's median and ratio still shipping here after `model/engine/stats.py`
   *  withdrew them. A test asserting `toContain("22.03")` was holding the
   *  withdrawal OUT.
   *
   *  So the pin is inverted. A measured RESULT may not appear in this copy at
   *  all; the id in the envelope carries it instead (see `cite`). What is
   *  allowed is a CONSTANT OF THE PRODUCT, each of which can be read straight
   *  off the source and none of which is a claim about accuracy:
   *
   *    0.4 s      model/engine/stats.py DISTANCE_WINDOW_S
   *    9.5 m/s    model/engine/stats.py MAX_SPEED, and 34.2 km/h = that × 3.6
   *    30 s       model/engine/stats.py FORMATION_WINDOW_S
   *    3 m        model/engine/sceneprior.py pitch_polygon(margin_m=3.0)
   *    22 players / km/90 / per90   arithmetic facts about a football match
   *
   *  Add to the list only by adding a constant to the product. If a number
   *  needs a run to produce it, it does not belong in a string. */
  const PRODUCT_CONSTANTS = [
    "0.4 s",
    "9.5 m/s",
    "34.2 km/h",
    "\u00d7 3.6",
    "30 s",
    "3 m",
    "22 players",
    "km/90",
    "per90",
  ];

  it("carries no measured result anywhere in the tier copy", () => {
    const copy: [string, string][] = [
      ...Object.entries(TIER).map(([k, v]) => [`TIER.${k}`, v.why] as [string, string]),
      ...Object.entries(FIELD_TIER)
        .filter(([, v]) => v.why)
        .map(([k, v]) => [k, v.why!] as [string, string]),
    ];
    expect(copy.length).toBeGreaterThan(10); // the scan found something to scan

    for (const [field, why] of copy) {
      let stripped = why;
      for (const c of PRODUCT_CONSTANTS) stripped = stripped.split(c).join("");
      // Anything numeric left is a measurement someone typed in by hand.
      expect([field, stripped.match(/\d[\d.,]*/g)]).toEqual([field, null]);
    }
  });

  it("states contamination as a mechanism, not as one match's share", () => {
    // REPINNED 2026-09-12. It used to require 15.1% and 33.2% — measured on a
    // tracks file recording no detector, conf or stride, and re-measured since
    // on the shipped config to a different figure that is not a before/after
    // with it. The direction is the claim that survives: the stitch trades
    // purity for whole players, so there is no clean setting.
    const why = FIELD_TIER["players.avg_position"].why ?? "";
    expect(why).toMatch(/not one player/i);
    expect(why).toMatch(/stitch/i);
    expect(why).not.toContain("49.6%");
  });

  /** Added 2026-08-19 with model/validation/dist_eval.py, the first end-to-end
   *  comparison of distance and peak speed against ground truth. Both replaced
   *  a caveat that had been true when written: distance's said it had never
   *  been compared, and peak speed's said the filter made it read LOW. The
   *  ratios went in 2026-09-12; the two directions are what mattered and they
   *  are what is asserted. */
  it("gives the measured distance bias in BOTH directions", () => {
    const why = FIELD_TIER["players.distance_km"].why ?? "";
    // Over-read on observed frames and under-read across the half. One without
    // the other is half the story and flatters whichever reading is quoted.
    expect(why).toContain("HIGH");
    expect(why).toContain("LOW");
    expect(why).toMatch(/observed_frac/);
    expect(why).toMatch(/not one player/i);
    // The claim it replaced. It was true until 2026-08-17 and is not now.
    expect(why).not.toMatch(/never (been )?compared/i);
  });

  it("says peak speed is refused, and that it reads HIGH not low", () => {
    const why = FIELD_TIER["players.top_speed_kmh"].why ?? "";
    expect(why).toMatch(/refused/i);
    expect(why).toContain("HIGH");
    // The retired claim, kept as a negative because it is the one that was
    // BACKWARDS rather than merely stale.
    expect(why).not.toMatch(/reads systematically low/i);
    // WITHDRAWN in model/engine/stats.py on 2026-09-10 and shipped here for two more
    // days: 22.03 was a max-of-maxes read as a median, 2.6x was built on it,
    // and "teleported" failed at its own prediction (0 of the 179 peaks sat on
    // a raw input jump). A test that REQUIRED 22.03 is what kept them here.
    for (const gone of ["22.03", "2.6", "teleport", "9.72", "8.39", "179"]) {
      expect(why).not.toContain(gone);
    }
  });

  it("gives the team figures the same reasons as the player ones", () => {
    // teams.top_speed_kmh is a max over the rows, so one clamped player refuses
    // the side's headline too. Both were `{ tier }` with no `why` at all, which
    // renders as a badge with no explanation — quieter than a wrong one.
    expect(FIELD_TIER["teams.distance_m"].why).toBe(FIELD_TIER["players.distance_km"].why);
    expect(FIELD_TIER["teams.top_speed_kmh"].why).toBe(
      FIELD_TIER["players.top_speed_kmh"].why,
    );
  });

  it("stops saying nothing has been measured, now that something has", () => {
    expect(TIER.prov.why).not.toMatch(/never been measured/i);
    expect(TIER.prov.why).not.toMatch(/Nothing here has been compared/i);
    expect(TIER.prov.why).toMatch(/neither passed/);
  });

  it("does not print a team accuracy beside the words home and away", () => {
    // The rule this enforces: the side SPLIT is unsupervised clustering and the
    // side NAME is the larger cluster, so an accuracy for the split is not an
    // accuracy for the label, and printing one next to "home" asserts something
    // the product has no information about. Every figure that used to sit here
    // was measured on one match (70.7 / 93.0 / 1.18), retired (56.4), or
    // unreproducible (51.6).
    const team = FIELD_TIER["players.team"].why ?? "";
    expect(team).toMatch(/larger/);
    expect(team).toMatch(/name/i);
    for (const gone of ["70.7", "93.0", "1.18", "1.08", "56.4", "51.6", "2.50"]) {
      expect(team).not.toContain(gone);
    }
  });

  it("does not print one match's row count on every match", () => {
    // "58 of 361 rows" was 118575's. gapNote says it per row instead, off this
    // match's own data, which is the only honest place for it.
    expect(FIELD_TIER["players.observed_minutes"].why).not.toContain("361");
    expect(gapNote(player({ observed_frac: 0.5 } as Partial<PlayerStats>))).toContain("50%");
  });
});

/** The one route by which a measured value still reaches a screen from this
 *  file, and the reason the copy above could be emptied of numbers. */
describe("tierOf cites the envelope's measurement", () => {
  const withId = (id: string | null): FieldEnvelope => ({
    value: 1,
    units: "km",
    n: 1,
    confidence: "prov",
    justified_by: id,
    provenance: "measured",
    caveat_flags: [],
  });

  it("appends the api:// id the backend resolved for this match", () => {
    const t = tierOf("players.distance_km", {
      "players.distance_km": withId("api://product.distance.ratio_raw"),
    });
    expect(t.why).toContain("api://product.distance.ratio_raw");
    expect(t.why).toContain("api-golden.json");
    // Justified means MEASURED, never PASSED — teams.distance_m is justified by
    // a measurement saying distance is wrong. The tooltip must not imply a pass.
    expect(t.why).toMatch(/not one that passed/);
    // ...and the field's own reason is still there in front of it.
    expect(t.why).toContain(FIELD_TIER["players.distance_km"].why!);
  });

  it("says nothing when the envelope justifies nothing", () => {
    const t = tierOf("players.distance_km", { "players.distance_km": withId(null) });
    expect(t.why).toBe(FIELD_TIER["players.distance_km"].why);
    expect(t.why).not.toContain("api://");
  });

  it("cites the SAME envelope the badge came from, never a sturdier sibling", () => {
    // The per-side expansion picks the weakest side for the tier. The id has to
    // follow it, or the badge and its citation describe two different runs.
    const t = tierOf("teams.distance_m", {
      "teams.home.distance_m": { ...withId("api://strong"), confidence: "ok" },
      "teams.away.distance_m": withId("api://weak"),
    });
    expect(t.tier).toBe("prov");
    expect(t.why).toContain("api://weak");
    expect(t.why).not.toContain("api://strong");
  });
});
