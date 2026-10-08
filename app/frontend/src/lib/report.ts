import type { FieldEnvelope, MatchReport, PlayerStats } from "./types";

/** Derivations over match.json, and the one confidence vocabulary the report
 *  and player screens both use.
 *
 *  Nothing in this file invents a field. Everything comes out of
 *  stats.py::build_match — `players` carries distance_km, minutes,
 *  top_speed_kmh, sprints, avg_position and clamped_frac, and that is all.
 *
 *  AND NOTHING IN THIS FILE INVENTS A NUMBER EITHER, which it did until
 *  2026-09-12. Every measured value in the copy below was typed in by hand:
 *  one match's accuracy figures (118575) printed onto every match's report,
 *  several of them measured at a configuration the product does not run, two
 *  of them (peak speed's median and ratio) withdrawn in stats.py while this
 *  file went on showing them to readers. See report-copy (not published).
 *
 *  THE RULE NOW. A number may appear in this file only if it is
 *    (a) a CONSTANT OF THE PRODUCT — MAX_SPEED and its km/h conversion, the
 *        0.4 s distance window, OBSERVED_MIN — which is a fact about the code
 *        and can be read off it, or
 *    (b) COMPUTED FROM THIS MATCH's own rows, like `gapNote`'s percentage.
 *  A measured RESULT — an accuracy, a ratio, a count, a share — never. Those
 *  reach the screen the one way that stays true when the golden moves:
 *  `tierOf` appends the envelope's own `justified_by` id, which
 *  `stats.resolve` re-resolves against today's golden every time the match is
 *  served. The prose says the MECHANISM and the DIRECTION; the id says what
 *  was measured and lets the reader re-run it.
 */

export const DEFAULT_PITCH = { length: 105, width: 68 };

// ---------------------------------------------------------------------------
// The confidence vocabulary. Three tiers, one meaning each, and the tier
// travels WITH the number — on the column header, the tile, the bar — so that
// nothing on either page has to be read against a footnote.
//
// The wording is the held-out run (model/validation/tune_runs/holdout-windowed.md).
// "Validated" is not used for anything, because nothing here is: minutes and
// ID switches report NOT EVALUATED (the harness could not perturb them, so the
// test could not have failed), and distance's 1.2% is not an accuracy figure at
// all — validate.py puts the reference AND the measurement through the same
// path_length at the same 0.4 s window, so the chord's own error cancels
// identically on both sides. It measures noise-robustness. See stats.py.
// ---------------------------------------------------------------------------
export type Tier = "withheld" | "not_evaluated" | "ok" | "rel" | "prov";

/** Why sprints are null. The number is the product's own measurement, cited by
 *  document, not typed here: n21-product-now (not published) (N11 block). */
export const SPRINTS_WITHHELD_WHY =
  "Withheld, not reported: at the configuration that ships, the sprint count was measured against ground truth and reads several times too many, because the velocity it counts is fabricated after every occlusion. A count that far off is not a provisional number but a wrong one, so the product does not publish it until a measurement of it passes.";

/** True where stats.py withheld this field (caveat flag `sprints_withheld`). */
export const withheld = (field: string, fields?: Record<string, FieldEnvelope>): boolean =>
  !!computedEnv(field, fields)?.caveat_flags?.includes("sprints_withheld");

export const TIER: Record<Tier, { label: string; why: string }> = {
  /* A refusal, not a tier of confidence: the backend publishes no value. */
  withheld: {
    label: "withheld",
    why: "The product refuses to publish this figure: it was measured and found wrong, so no number is shown.",
  },
  /* The backend's default, and the honest one: `stats.metric()` now resolves
   * each field's tier against an `api://` measurement id, and a field whose id
   * is missing from `model/validation/api-golden.json` degrades to here. It ranks
   * BELOW provisional on purpose — provisional says "measured, and it did not
   * pass"; this says nobody measured it, which is the weaker claim. Most
   * fields read this today. */
  not_evaluated: {
    label: "not evaluated",
    why:
      "No accepted measurement stands behind this field. Every other tier on this page cites a re-runnable id in model/validation/api-golden.json; this one has none, so the number is what the code computed and nothing has ever checked whether it is right. It is not a claim that the figure is wrong — it is a claim that nobody knows.",
  },
  ok: {
    label: "sturdiest",
    why:
      "Depends on when a track starts and stops, not on pixel precision, so it does not carry the errors the distance figures do. Not independently tested either — the held-out run reports NOT EVALUATED, meaning it could not have failed. Also duration, not a timeline: a track lost and re-acquired under-reports.",
  },
  rel: {
    label: "relative",
    why:
      "A rank or a share. Every player went through the same camera, calibration and filter, so those biases largely cancel out of a ratio and the ordering survives where the absolute figure does not. One bias does NOT cancel: the 0.4 s window under-reads a player in proportion to how often they change direction, so it tracks playing position rather than hitting everyone equally.",
  },
  prov: {
    label: "provisional",
    why:
      "An absolute figure this repo does not stand behind. For most of them nobody has measured the error at all, and the one agreement figure that is sometimes quoted for them is not an accuracy figure: the reference and the measurement both go through the same 0.4 s window estimator, so its error cancels on both sides and what survives is noise-robustness. Distance and peak speed are the exceptions — both have now been compared with ground truth using the shipped detector, tracker and identity stages with a research calibration, and neither passed — and peak speed is refused outright wherever the clamp bound. Read the shape, not the value.",
  },
};

/** Said on every block drawn from a match whose tracks were generated.
 *
 *  The stats code on top of them is real and honest; its input is invented —
 *  an Ornstein-Uhlenbeck walk tuned to a fixed kilometre target, with nobody
 *  ever occluded and nobody ever substituted. Every distance, speed, sprint
 *  count and average position in the demo library comes out of that, and the
 *  envelope around it still says `provenance: "measured"`.
 *
 *  The flag has been in match.json since the importer was written. Nothing
 *  read it, which is the whole defect: a coach printing the starter document
 *  reads 13.5 km per player — above elite outfield range — under a note that
 *  says "Measured from the tracks".
 */
export const SYNTHETIC_NOTE =
  "Tracking for this match is GENERATED, not measured: the positions behind every distance, speed and average-position figure came out of a motion model rather than a camera.";

export const syntheticNote = (r: MatchReport | null | undefined): string =>
  r?.meta?.synthetic_tracking || r?.meta?.demo ? SYNTHETIC_NOTE : "";

// ---------------------------------------------------------------------------
// WHAT TIER EACH EMITTED FIELD CARRIES. Keyed by its path in match.json, so a
// field cannot reach a screen without one — `tierOf` falls back to
// `not_evaluated` rather than to nothing, which is the safe direction and the
// reason this is a lookup instead of a prop typed by hand at every call site.
// (This comment used to say it falls back to provisional. It stopped being true
// when the fallback moved to not_evaluated - see the note above TIER_RANK - and
// a copy of the old claim in query-rows.ts is how a second rank table drifted.)
// ---------------------------------------------------------------------------

/** Team shape, the formation label, its per-window series and the strided
 *  sample series all share one reason, and stats.py already names it: the key
 *  is literally called `window_fit_unstable`. The cost behind it is well
 *  behaved and is what `structure_stress` consumes; the label is an estimate
 *  wearing a badge. The shape metrics beside it are absolute metres that scale
 *  straight off the calibration.
 *
 *  The stability figures this used to quote (a label-change count, a mode
 *  share, a literature accuracy range) were measured on eight team-halves of
 *  one match and do not reproduce against the current output — see
 *  report-copy (not published). The instability is real; the numbers were not this
 *  match's. */
const SHAPE_WHY =
  "Team shape and formation ship under stats.py's own `window_fit_unstable` key: the formation is re-fitted independently in every 30 s window and the label it returns changes repeatedly within a half, so it is an estimate rather than a reading. The metres beside it are absolute distances that scale straight off the operator's clicked calibration, and the only figure the product holds for that calibration is its own fit residual — which is how well the solver matched the points that were clicked, not an error measured against anything on the pitch. Compare the two sides only through the paired difference, which is taken on the instants both were measurable; do not subtract the per-side figures, and do not quote either against another system.";

/** Every per-player physical figure carries this, and it is a different problem
 *  from the generic provisional caveat above: not "unmeasured", but measured and
 *  known to be mixed. See stats.py's `identity_contaminated`. */
const CONTAMINATED_WHY =
  "A player row is not one player. Measured against ground truth on this footage, a large share of the detections sitting under one track id have a different true player as their majority owner, and the stitch is what raises it: it buys whole players out of fragments and pays in purity, so there is no setting of this pipeline that is clean — fewer fragments and more mixing are the same knob. The error pools within a team rather than cancelling between players. Appearance cannot currently fix it either: at the torso resolution this footage offers, two players in the same kit barely separate at all, and an embedding does no better than the box height alone. Read the side totals, not the rows.";

/** Distance, measured against ground truth (model/validation/dist_eval.py). This
 *  REPLACED the "never compared end to end" line, which had been true since it
 *  was written and stopped being true on 2026-08-17. Two directions, one
 *  number: which one applies depends on whether the reader means the half or
 *  the frames we saw.
 *
 *  The ratios it used to print were that first run's, on one match. Re-measured
 *  on the shipped configuration the over-read replicates and everything else
 *  moved — coverage and the player-to-player spread most of all — so the
 *  direction is the claim and the digits were never this match's anyway. The
 *  id in the envelope is the number. */
const DISTANCE_WHY =
  "Measured against ground truth, not assumed, and it is wrong in two directions at once. On the frames a track was actually observed the distance reads HIGH. Over a WHOLE half the same tracks read LOW, and by more, because much of the half is never observed at all and the running inside those gaps is simply absent rather than estimated. `observed_frac` says which regime a row is in: as “how far did this player run” it is short, as “how far did the path we saw move” it is long. On the first match it was measured on, it was not detector noise, which would scale with the camera — the per-player error did not follow local ground resolution — and it was not jitter, because widening the 0.4 s window made it worse, not better. A player row is not one player — tracks swap people mid-way — but that is not what lengthens it: splitting every track wherever the player underneath changes leaves the measured distance where it was, and what does is not established. The reference is itself annotation on a coarse grid, which inflates the reference’s own path length — so the over-read is the flattering direction.";

/** Peak speed refuses rather than reports wherever MAX_SPEED bound, so this is
 *  the reason for a BLANK as much as for a number. It replaced a caveat that
 *  said the filter made peak speed read systematically low; measured, it reads
 *  high.
 *
 *  WITHDRAWN, and this file shipped them after stats.py pulled them: the
 *  "median of 22.03 m/s" was a max-of-maxes read as a median, the "2.6×" was
 *  built on it, and "the track teleported" failed at its own prediction. They
 *  are gone rather than restated. 9.5 m/s is MAX_SPEED in model/engine/stats.py and
 *  34.2 is that constant × 3.6; those two are the code, not a measurement. */
const PEAK_WHY =
  "Refused, not reported: this is blank wherever the 9.5 m/s clamp bound, because the only figure it could have published there is 34.2 km/h — that clamp × 3.6, a constant in our own source — which reads as an elite sprint while saying nothing about the player. Where a number does survive, the clamp never bound and the max is a real reading of the track; but a player row is not one player, so it is a max taken over more than one of them, and a maximum is precisely the statistic a single bad frame or a single swapped identity wins outright. Measured against ground truth it reads HIGH, not low — the caveat here used to say the opposite. Read it as the top of what the pipeline produced, never as an athlete's top speed.";

export interface FieldTier {
  tier: Tier;
  /** Field-specific reason, shown in addition to TIER[tier].why. */
  why?: string;
}

/** Observed time, as against `minutes` which is span. Added 2026-08-26 with the
 *  role work. stats.py:550 explains at length why `minutes` itself must not be
 *  redefined - it is "how long this track lasted" in the time-on-pitch bars, in
 *  MIN_MINUTES and in the validation harness - so this is the second number
 *  rather than a replacement. */
const OBSERVED_MINUTES_WHY =
  "How long this track was actually SEEN, as against `minutes`, which is the span from first sighting to last and counts every occlusion gap as time played. PUBLISHED BESIDE km/90, NOT YET USED BY IT: per90 still divides by `minutes`, so on any row with gaps it is short in the numerator (distance excludes the gaps) and long in the denominator (minutes does not) at the same time, and reads low twice over. How many rows that is on THIS match is not stated here because it is not known here — `gapNote` says it per row, off this match's own data. Switching per90 to this number re-bases a headline figure across every row, which is a change that deserves its own before-and-after rather than riding along with a role classifier.";

/** The role label. Kept deliberately weak: see stats.py CAVEATS.role_unvalidated
 *  and model/engine/roles.py. The `why` lives here because match.json ships
 *  caveat_flags KEYS and never prose. */
const ROLE_WHY =
  "Inferred from movement alone - how far this track roamed and where - and never scored, because it cannot be: the ground truth this pipeline is measured against holds 22 players labelled home and away and no officials at all, so there is nothing to check a referee or a coach against. The thresholds are geometry, not measurement. It answers only `staff` and `linesman`, refuses `referee` outright because that profile overlaps a box-to-box midfielder's, and it excludes nobody - a wrong role costs a wrong label and never removes a player from this table. Note also that the scene prior keeps only detections inside the pitch plus 3 m, so anyone standing in a technical area is filtered before tracking and can never be labelled `staff` at all.";

export const FIELD_TIER: Record<string, FieldTier> = {
  // players[]
  "players.minutes": { tier: "not_evaluated" },
  "players.distance_km": { tier: "prov", why: DISTANCE_WHY },
  "players.top_speed_kmh": { tier: "prov", why: PEAK_WHY },
  "players.sprints": { tier: "withheld", why: SPRINTS_WITHHELD_WHY },
  "players.avg_position": { tier: "prov", why: CONTAMINATED_WHY },
  "players.clamped_frac": { tier: "not_evaluated" },
  "players.observed_frac": { tier: "not_evaluated" },
  "players.observed_minutes": { tier: "not_evaluated", why: OBSERVED_MINUTES_WHY },
  "players.role": { tier: "prov", why: ROLE_WHY },
  "players.role_confidence": { tier: "prov", why: ROLE_WHY },
  "players.team": {
    tier: "prov",
    why:
      "The side is not a tag, and the NAME is not a reading. Shirt colour is clustered without supervision into two groups, and the group that gets called `home` is simply the larger one (model/engine/track.py) — with eleven a side, which cluster is larger is noise. So the split into two sides carries information and the words home and away carry none: nothing in the footage tells this pipeline which side is which, and no accuracy figure for the split may be read as an accuracy figure for the label. Detections also land on the wrong side of the split often enough to matter, and a wrong side moves a player's whole distance total into the opposing squad aggregate, so read every per-side total as provisional even where the per-player figures beside it are not.",
  },
  // derived in this file
  // NOT `rel`. A share is normally the sturdy reading — the calibration and
  // filter terms cancel out of a ratio — but contamination does not cancel:
  // it MOVES running between two rows of the same side, which is exactly what
  // a share of the side total is sensitive to. stats.py agrees: METRICS gives
  // `derived.share` the identity_contaminated flag, so its envelope ships
  // `prov`, and this entry read `rel` until the flag landed.
  "derived.share": { tier: "prov", why: CONTAMINATED_WHY },
  "derived.per90": { tier: "prov", why: DISTANCE_WHY },
  // teams[side]
  "teams.distance_m": { tier: "prov", why: DISTANCE_WHY },
  "teams.top_speed_kmh": { tier: "prov", why: PEAK_WHY },
  "teams.sprints": { tier: "withheld", why: SPRINTS_WITHHELD_WHY },
  "teams.shape": { tier: "prov", why: SHAPE_WHY },
  // not_evaluated, as the backend ships it: METRICS names no measurement for territory.
  territory: { tier: "not_evaluated" },
  // shape_series[side] — emitted by build_match, and until now read by nothing.
  "shape_series.formation": { tier: "prov", why: SHAPE_WHY },
  "shape_series.window_fit_unstable": { tier: "prov", why: SHAPE_WHY },
  "shape_series.samples": { tier: "prov", why: SHAPE_WHY },
  structure_stress: { tier: "prov", why: SHAPE_WHY },
  // The paired difference. Same tier as the rest of shape, different reason:
  // stats.py adds `team_purity` here and nowhere else, because a BETWEEN-team
  // difference is the one metric a single misassigned player destroys.
  relative_shape: {
    tier: "prov",
    why:
      "A difference between the two banks, so it is the metric team assignment breaks hardest: in a perturbation test, moving a single player of twenty-two into the wrong bank collapsed the measured width gap to a fraction of the truth and moving two overshot it — it fails chaotically, not gracefully. And which bank is called which is decided by cluster size, not by the footage, so the sign of this difference is not attributable to a named side. Computed only on the instants BOTH sides cleared the minimum player count, which is what makes it a difference at all.",
  },
};

/** `not_evaluated` is the fallback on purpose: an unknown field is one nobody
 *  has reasoned about, and that is exactly the one that must not read as
 *  measured. It is also what the backend now ships for it — `stats.metric()`
 *  resolves each tier against an `api://` id and degrades when the golden does
 *  not carry it — so the table and the envelope agree by construction rather
 *  than by anyone remembering to edit both.
 *
 *  THE FOUR `ok` ROWS ABOVE WERE STALE and read `ok` while the backend shipped
 *  `not_evaluated`; `tierOf` prefers the envelope so no screen was wrong, but
 *  the fallback for a match with no `fields` block was an assertion nothing
 *  stood behind. The tiers here are a FALLBACK, never a claim of their own. */
/** Weakest first, so min() over this is "the weakest link governs" — the same
 *  rule stats.metric() applies over a field's caveats on the way in. */
export const TIER_RANK: Record<Tier, number> = { withheld: -2, not_evaluated: -1, prov: 0, rel: 1, ok: 2 };

/** The sides a per-side envelope key can carry. `teams.distance_m` in the UI is
 *  a header over both, and the backend keys it per side. */
const SIDES = ["home", "away"] as const;

/** The ENVELOPE the backend actually computed for this field, or undefined.
 *
 *  Returns the whole envelope rather than its `confidence` because the tier is
 *  only half of what it carries: `justified_by` is the `api://` id of the
 *  measurement behind the tier, and that id is the only thing on this screen
 *  that stays true when the golden moves. It used to be dropped here, which is
 *  why the copy below had to hard-code values to say anything at all.
 *
 *  Exact key first. Failing that, the UI's generic path is expanded into the
 *  per-side keys the backend emits (`teams.distance_m` ->
 *  `teams.home.distance_m` + `teams.away.distance_m`, `structure_stress` ->
 *  `structure_stress.home` + `.away`) and the WEAKEST of them wins: a header
 *  standing over both sides cannot claim to be sturdier than the shakier one
 *  underneath it. The id then comes off that same weakest envelope, so the
 *  badge and the citation never describe two different measurements. */
function computedEnv(
  field: string,
  fields?: Record<string, FieldEnvelope>,
): FieldEnvelope | undefined {
  if (!fields) return undefined;
  if (fields[field]?.confidence) return fields[field];

  const parts = field.split(".");
  const candidates = SIDES.flatMap((s) => [
    // teams.distance_m -> teams.home.distance_m
    parts.length > 1 ? [parts[0], s, ...parts.slice(1)].join(".") : null,
    // structure_stress -> structure_stress.home
    `${field}.${s}`,
  ]).filter((k): k is string => k != null);

  const found = candidates
    .map((k) => fields[k])
    .filter((e): e is FieldEnvelope => e?.confidence != null);
  if (!found.length) return undefined;
  return found.reduce((a, b) =>
    TIER_RANK[b.confidence] < TIER_RANK[a.confidence] ? b : a,
  );
}

/** The measurement id, appended to the field's reason.
 *
 *  This is the ONLY route by which a measured value reaches a screen from this
 *  file now. It is deliberately not the value: `stats.resolve` re-resolves
 *  every envelope against today's golden on the way out, so an id is still
 *  right tomorrow and a typed-in number is right only until someone re-runs
 *  the harness — which is exactly what happened to every figure this file used
 *  to print.
 *
 *  "was made, not one that passed" is not hedging. `justified_by` means a
 *  measurement EXISTS for this field, never that it cleared a threshold;
 *  `teams.distance_m` is justified by a measurement saying distance is wrong.
 *  A tooltip that implied otherwise would be the same defect in a new place. */
const cite = (why: string | undefined, id?: string | null): string | undefined => {
  if (!id) return why;
  const s = `The tier on this field comes from ${id} in model/validation/api-golden.json — a measurement that was made, not one that passed.`;
  return why ? `${why} ${s}` : s;
};

/** The tier a field carries on THIS match.
 *
 *  Prefers what stats.py computed and shipped in `match.json.fields` over the
 *  static table below it. THE TWO DID DRIFT, and the comment here used to say
 *  they had not: when `identity_contaminated` landed on `derived.share` in
 *  stats.py the envelope dropped to `prov` while this table went on saying
 *  `rel`, and player-table.tsx was not reading the envelope at all, so the
 *  weaker badge never reached a screen. Fixed 2026-08-18, and the drift is now
 *  pinned by the per-player tests in report.test.ts.
 *
 *  Two things follow, and both are the quiet direction of failure:
 *   - a field MISSING from the table below still resolves, to `prov` with NO
 *     `why` — no wrong explanation, no explanation at all, which is harder to
 *     spot than a wrong one. Add the key.
 *   - the `why` is UI copy and lives ONLY here, so the backend can never
 *     supply it. A caller that takes `.tier` and drops `.why` renders a badge
 *     identical to a correct one. Pass the whole FieldTier. */
export const tierOf = (
  field: string,
  fields?: Record<string, FieldEnvelope>,
): FieldTier => {
  const env = computedEnv(field, fields);
  const local = FIELD_TIER[field];
  if (env?.caveat_flags?.includes("sprints_withheld")) return { tier: "withheld", why: SPRINTS_WITHHELD_WHY };
  if (env) return { tier: env.confidence, why: cite(local?.why, env.justified_by) };
  return local ?? { tier: "not_evaluated" };
};

// ---------------------------------------------------------------------------

export function teamName(report: MatchReport, side: string): string {
  if (side === "home") return report.meta?.home_name || "Home";
  if (side === "away") return report.meta?.away_name || "Away";
  return "Unassigned";
}

/** Jersey is null until identity works, which is the common case. Showing the
 *  track id in a "#" slot reads as a shirt number, which is a quiet lie. */
export function playerLabel(p: PlayerStats, report: MatchReport): string {
  return p.jersey != null
    ? `${teamName(report, p.team)} #${p.jersey}`
    : `${teamName(report, p.team)} track ${p.id}`;
}

// ---------------------------------------------------------------------------
// OCCLUSION. stats.py drops a whole 0.4 s window (ten samples at 25 fps) rather
// than closing it with a straight line across a path nobody saw, and until now
// that deletion was completely silent — a track could lose a fifth of its
// running and read as a confident number. `observed_frac` is the share of
// windows that survived.
//
// It is a FLOOR on the damage, not a correction factor. A window is k samples
// wide, not k seconds, so one straddling a gap swallows the gap's wall clock as
// well: measured on a constant-speed track, 8.2% of windows lost cost 11.1% of
// distance and 12.9% lost cost 47.1%. Never divide by it.
// ---------------------------------------------------------------------------

/** Below this share of measured windows a distance is no longer worth reading
 *  as a value. Not fitted — 0.95 is one occlusion event in twenty windows, and
 *  by then the under-count is already several percent. */
export const OBSERVED_MIN = 0.95;

/** types.ts is owned elsewhere this pass; stats.py::build_match emits the field
 *  today and older match.json files on disk do not have it. Absent means "not
 *  known", which must not read as "fully observed" — hence `?? null`. */
const observedFrac = (p: PlayerStats): number | null =>
  (p as PlayerStats & { observed_frac?: number }).observed_frac ?? null;

/** Why this player's distance is provisional beyond the usual reasons, or null.
 *  Consumed on the report screen; the tier is already `prov` for every absolute
 *  figure, so this is the REASON, not the tier. */
export function gapNote(p: PlayerStats): string | null {
  const f = observedFrac(p);
  if (f == null || f >= OBSERVED_MIN) return null;
  return `${((1 - f) * 100).toFixed(0)}% of this track's distance windows contained an unobserved gap and were dropped rather than guessed across. The distance is short by at least that much and probably more. Minutes still counts the gap time, so km/90 is short in the numerator and long in the denominator at once — read the rank, not the value.`;
}

export interface Row {
  p: PlayerStats;
  label: string;
  minutes: number;
  /** km, or null where stats.py REFUSED it (above MAX_SPEED x observed time). */
  distance: number | null;
  /** % of this player's own side's total distance; null with a refused distance */
  share: number | null;
  per90: number | null;
  /** km/h, or null where stats.py REFUSED it: MAX_SPEED bound on this track, so
   *  the only number it could have published was the clamp. Not zero — a
   *  refusal is "we will not say", and 0.0 would read as a player who stood
   *  still. Every render site must branch. */
  peak: number | null;
  /** null: WITHHELD (see SPRINTS_WITHHELD_WHY). Render "withheld", never 0. */
  sprints: number | null;
  /** Share of distance windows actually measured, or null on a match.json
   *  written before stats.py emitted it. */
  observed: number | null;
  /** Non-null when this player's distance was built from too few windows. */
  gapNote: string | null;
}

export function buildRows(report: MatchReport): Row[] {
  const players = report.players ?? [];
  const sideTotal: Record<string, number> = {};
  for (const p of players) {
    sideTotal[p.team] = (sideTotal[p.team] ?? 0) + (p.distance_km || 0);
  }
  return players.map((p) => ({
    p,
    label: playerLabel(p, report),
    minutes: p.minutes || 0,
    distance: p.distance_km ?? null,
    share: p.distance_km == null ? null : sideTotal[p.team] ? (p.distance_km / sideTotal[p.team]) * 100 : 0,
    // Per 90, not total: totals rank whoever played longest, which is a
    // substitution report dressed up as a fitness one.
    per90: p.distance_km == null ? null : p.minutes ? (p.distance_km / p.minutes) * 90 : 0,
    // `?? null`, never `|| 0`: the backend now sends null for a refused peak
    // and `|| 0` would turn "we will not say" into "0.0 km/h".
    peak: p.top_speed_kmh ?? null,
    sprints: p.sprints ?? null,
    observed: observedFrac(p),
    gapNote: gapNote(p),
  }));
}

/* `mean` and `rankOf` were deleted 2026-08-05 with the last screen that ranked
 * players against each other. They are not coming back: our identity error puts
 * sender accuracy at 9.1% against a 4.5% chance floor, so a rank between two
 * NAMED players is not a measurement this product can make. Sort by minutes, and
 * state a quantity with its band — see /checks. */

export const teamColor = (side: string) =>
  side === "home" ? "bg-home" : side === "away" ? "bg-away" : "bg-muted-foreground";

export const teamText = (side: string) =>
  side === "home" ? "text-home" : side === "away" ? "text-away" : "text-muted-foreground";
