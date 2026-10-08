import { PLAYER_STAT_COLS } from "@/lib/player-cols";
import { buildRows, playerLabel, syntheticNote, teamName, type Tier } from "@/lib/report";
import type { MatchReport, Season } from "@/lib/types";
import { PLAY_OUTCOMES, PLAY_TYPES, PLAY_ZONES, plays as matchPlays } from "./play-data";

/** Ask the match a question it was not built to answer.
 *
 *  Every other block in the notepad renders a shape somebody decided in
 *  advance: this squad ranked on distance, that group of stats home against
 *  away. This one is the general case — pick what to measure, pick what to
 *  break it down by, and get a table or a chart. Distance by position across
 *  the season. Plays by zone. Shots by player, split home and away.
 *
 *  A SOURCE decides what can be asked of it, which is the honest way round.
 *  `players[]` is measured and knows nothing about zones; `plays` knows about
 *  zones and nothing about distance; the season knows about matches and only
 *  for players somebody bound. Offering every dimension against every measure
 *  would mean most combinations quietly returning nothing, so a source carries
 *  its own dimension and measure lists and the picker only shows what fits.
 */

export interface Datum {
  dims: Record<string, string>;
  vals: Record<string, number>;
}

export interface Measure {
  id: string;
  label: string;
  agg: "sum" | "avg" | "max" | "count";
  dp: number;
  unit?: string;
  /** A rate is a ratio of SUMS, never a sum of rates: totalling each player's
   *  km/90 and dividing by the headcount weights a 12-minute substitute the
   *  same as somebody who played every week. */
  ratio?: { num: string; den: string; scale: number };
}

export interface Dim {
  id: string;
  label: string;
  /** Ordered dimensions can be drawn as a line; the rest can only be bars. */
  ordered?: boolean;
}

export interface Source {
  id: string;
  label: string;
  /** Who the rows are ABOUT. The `/` panels ask this first — a coach opens a
   *  table thinking "the players" or "the two teams", not thinking "tracks" or
   *  "teamFeed" — so it has to be a property of the source rather than a guess
   *  made from its dimension list. */
  subject: "player" | "team";
  scope: "match" | "season";
  dims: Dim[];
  measures: Measure[];
  note: string;
  tier: Tier;
}

/** A feed value as a number, or undefined when it is not one.
 *
 *  NOT zero. Zero is a reading — "no shots on target" — and a value the parser
 *  could not read is not a reading. The feed carries possession twice, once as
 *  `72` and once as `"72%"`; while `"72%"` returned 0 and the dedupe below kept
 *  a different row for each side, this drew a 72-0 possession split. A number
 *  nobody recorded is worse than a missing bar, because it is legible.
 *
 *  "618 (88%)" is 618, "72%" is 72, "3/5 (60%)" is 3 — the successful half of a
 *  made-of-attempted pair, which is what "successful dribbles" means. The
 *  number still has to be the whole leading token, or "4-2-3-1" reads as 4.
 */
const num = (v: unknown): number | undefined => {
  if (typeof v === "number") return Number.isFinite(v) ? v : undefined;
  const m = /^-?\d+(?:\.\d+)?(?=$|[\s(%/])/.exec(String(v ?? "").trim());
  return m ? Number(m[0]) : undefined;
};

/** Drop the keys nobody recorded, so an absent measure stays absent all the
 *  way to `finish` instead of arriving as a zero that drags an average down. */
const defined = (o: Record<string, number | undefined>): Record<string, number> =>
  Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined)) as Record<string, number>;

const D = {
  player: { id: "player", label: "Player" },
  position: { id: "position", label: "Position" },
  side: { id: "side", label: "Team" },
  match: { id: "match", label: "Match", ordered: true },
  period: { id: "period", label: "Half", ordered: true },
  playType: { id: "playType", label: "Play type" },
  zone: { id: "zone", label: "Zone" },
  outcome: { id: "outcome", label: "Outcome" },
  minute: { id: "minute", label: "Minute", ordered: true },
  channel: { id: "channel", label: "Channel" },
  result: { id: "result", label: "Result" },
} satisfies Record<string, Dim>;

/** A team metric's label as an id, so a measure this match happens to carry
 *  can live in a document attribute and be read back. */
export const metricId = (label: string) =>
  label.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

/** `6'5"` to centimetres. Height is in the lineup feed and is a real thing to
 *  break a squad down by, but not in the unit it arrives in. */
/** Undefined, not 0, for a height nobody recorded — a squad's mean height is
 *  not lowered by the players whose height the feed left blank. */
function heightCm(v: unknown): number | undefined {
  const m = /^(\d+)'\s*(\d+)?/.exec(String(v ?? ""));
  return m ? Math.round((Number(m[1]) * 12 + Number(m[2] ?? 0)) * 2.54) : undefined;
}

const COUNT: Measure = { id: "count", label: "How many", agg: "count", dp: 0 };

const PHYSICAL: Measure[] = [
  { id: "distance_km", label: "Distance", agg: "sum", dp: 2, unit: "km" },
  {
    id: "km_per90",
    label: "Distance / 90",
    agg: "sum",
    dp: 2,
    unit: "km",
    ratio: { num: "distance_km", den: "minutes", scale: 90 },
  },
  { id: "top_speed_kmh", label: "Top speed", agg: "max", dp: 1, unit: "km/h" },
  { id: "sprints", label: "Sprints", agg: "sum", dp: 0 },
  { id: "minutes", label: "Minutes", agg: "sum", dp: 0 },
  COUNT,
];

/** The feed's numeric columns, named the way the report screen names them so a
 *  chart and a table cannot label the same quantity differently. `rating` is
 *  averaged; everything else counts up. */
const FEED_MEASURES: Measure[] = [
  ...Object.entries(PLAYER_STAT_COLS)
    .filter(([id]) => !["duels", "ground_duels", "aerial_duels"].includes(id))
    .map(
      ([id, label]): Measure => ({
        id,
        label,
        agg: id === "rating" ? "avg" : "sum",
        dp: ["rating", "xg", "xa", "xgot_faced", "goals_prevented"].includes(id) ? 2 : 0,
      }),
    ),
  COUNT,
];

/** What the camera measured about a whole team. The shape metrics carry
 *  stats.py's own `window_fit_unstable` caveat and the territory shares are a
 *  share of POSITIONS rather than of possession — both said in the note. */
const TEAM_MEASURES: Measure[] = [
  { id: "distance_km", label: "Distance", agg: "sum", dp: 1, unit: "km" },
  { id: "top_speed_kmh", label: "Top speed", agg: "max", dp: 1, unit: "km/h" },
  { id: "sprints", label: "Sprints", agg: "sum", dp: 0 },
  { id: "width_m", label: "Width", agg: "avg", dp: 1, unit: "m" },
  { id: "depth_m", label: "Depth", agg: "avg", dp: 1, unit: "m" },
  { id: "defensive_line_m", label: "Defensive line", agg: "avg", dp: 1, unit: "m" },
  { id: "compactness_m2", label: "Compactness", agg: "avg", dp: 0, unit: "m²" },
  { id: "stretch_index_m", label: "Stretch index", agg: "avg", dp: 1, unit: "m" },
];

const SQUAD_MEASURES: Measure[] = [
  { id: "rating", label: "Rating", agg: "avg", dp: 2 },
  { id: "age", label: "Age", agg: "avg", dp: 1 },
  { id: "height_cm", label: "Height", agg: "avg", dp: 0, unit: "cm" },
  { id: "minutes_on", label: "Came on", agg: "avg", dp: 0, unit: "'" },
  COUNT,
];

/** Every team metric this match actually carries, deduped across the groups
 *  that repeat one (Total shots is in both "Top stats" and "Shots").
 *
 *  Computed from the report rather than declared, because the picker must not
 *  offer a metric this match has no row for — that is the whole "a source
 *  decides what can be asked of it" rule, applied one level finer. */
function teamFeedMeasures(report: MatchReport | null): Measure[] {
  const facts = report?.meta?.facts;
  const seen = new Map<string, Measure>();
  for (const label of (facts?.top_stats ?? []).map((t) => t.label)) {
    seen.set(metricId(label), { id: metricId(label), label, agg: "sum", dp: 2 });
  }
  for (const s of facts?.sections ?? []) {
    for (const r of s.rows) {
      const id = metricId(r.label);
      if (!seen.has(id)) seen.set(id, { id, label: r.label, agg: "sum", dp: 2 });
    }
  }
  return [...seen.values()];
}

/** The sources, for THIS match. A function rather than a constant because two
 *  of them can only know their measures once the report is in hand. */
export function sources(report: MatchReport | null = null): Source[] {
  return [
    ...STATIC_SOURCES.slice(0, 2),
    {
      id: "teamFeed",
      label: "Team stats",
    subject: "team",
      scope: "match",
      dims: [D.side],
      measures: teamFeedMeasures(report),
      note: "Entered, not measured — these come from the match feed, not from the camera.",
      tier: "prov",
    },
    ...STATIC_SOURCES.slice(2),
  ];
}

const STATIC_SOURCES: Source[] = [
  {
    id: "tracks",
    label: "Tracked players",
    subject: "player",
    scope: "match",
    dims: [D.player, D.side],
    measures: PHYSICAL,
    note: "Measured from the tracks. Shirt numbers, not names — the tracker reads a number or nothing.",
    tier: "prov",
  },
  {
    id: "teams",
    label: "Team shape",
    subject: "team",
    scope: "match",
    dims: [D.side],
    measures: TEAM_MEASURES,
    note:
      "Measured. The shape metres ship under stats.py's own window_fit_unstable caveat — read the difference between the sides, not either figure against another system.",
    tier: "prov",
  },
  {
    id: "territory",
    label: "Territory",
    subject: "team",
    scope: "match",
    dims: [{ id: "third", label: "Third", ordered: true }],
    measures: [{ id: "share", label: "Share of positions", agg: "sum", dp: 1, unit: "%" }],
    note:
      "A share of tracked POSITIONS per third — where the players were, not who had the ball. The ball it detects is not used here.",
    tier: "rel",
  },
  {
    id: "squad",
    label: "Squad",
    subject: "player",
    scope: "match",
    dims: [D.position, D.side, D.player],
    measures: SQUAD_MEASURES,
    note: "Entered, not measured — the lineup feed, not the camera.",
    tier: "prov",
  },
  {
    id: "shots",
    label: "Shots",
    subject: "team",
    scope: "match",
    dims: [D.side, D.period, D.result],
    measures: [COUNT],
    note:
      "Entered, not measured. Only where each shot was taken is recorded per shot — the situation and the xG behind a single one are not.",
    tier: "prov",
  },
  {
    id: "momentum",
    label: "Momentum",
    subject: "team",
    scope: "match",
    dims: [D.minute],
    measures: [{ id: "pressure", label: "Pressure", agg: "avg", dp: 2 }],
    note:
      "Entered, not measured — one value a minute, positive toward home. Nothing in the pipeline computes this.",
    tier: "prov",
  },
  {
    id: "zones",
    label: "Attacking channels",
    subject: "team",
    scope: "match",
    dims: [D.channel, D.side],
    measures: [{ id: "share", label: "Share of attacks", agg: "sum", dp: 0, unit: "%" }],
    note: "Entered, not measured — a share of attacks needs the ball.",
    tier: "prov",
  },
  {
    id: "plays",
    label: "Plays",
    subject: "team",
    scope: "match",
    dims: [D.playType, D.zone, D.outcome, D.side, D.period],
    measures: [COUNT],
    note: "Entered, not measured — the ball the pipeline detects is not used to find plays.",
    tier: "prov",
  },
  {
    id: "feed",
    label: "Match feed",
    subject: "player",
    scope: "match",
    dims: [D.player, D.side, D.position],
    measures: FEED_MEASURES,
    note: "Entered, not measured — these come from the match feed, not from the camera.",
    tier: "prov",
  },
  {
    id: "season",
    label: "All matches",
    subject: "player",
    scope: "season",
    // D.side is the CLUB's team here ("U15 Blue"), not home/away — the season
    // spans every team on the ladder, so without it a note comparing two age
    // groups would pool them into one anonymous list.
    dims: [D.player, D.position, D.side, D.match],
    measures: [...PHYSICAL.slice(0, 5), { id: "matches", label: "Matches", agg: "sum", dp: 0 }, COUNT],
    note: "",
    tier: "prov",
  },
];

/** Kept for the callers that only need the fixed list. */
export const SOURCES: Source[] = sources(null);

export const sourceOf = (id: string, report: MatchReport | null = null): Source => {
  const all = sources(report);
  return all.find((s) => s.id === id) ?? all[0];
};
/** Every value a dimension takes, in the order the facts produce them.
 *
 *  What the Compare picker lists. Read off the FACTS rather than guessed from
 *  the report, so it offers exactly the rows the chart could draw and never a
 *  name that would come back empty.
 */
export function labelsFor(opts: {
  report: MatchReport | null;
  season: Season | null;
  source: string;
  dim: string;
  team?: string;
  matchIds?: string[];
}): string[] {
  const src = sourceOf(opts.source, opts.report);
  const seen: string[] = [];
  for (const f of factsFor(src, opts.report, opts.season, opts.matchIds, opts.team)) {
    const v = f.dims[opts.dim];
    if (v && !seen.includes(v)) seen.push(v);
  }
  return seen;
}

/** The team names this source can be narrowed to, or [] when it has no side.
 *  Read off the facts rather than off the report, so a source that only ever
 *  carries one side does not offer the other. */
export function teamsOf(
  sourceId: string,
  report: MatchReport | null,
  season: Season | null = null,
): string[] {
  const src = sourceOf(sourceId, report);
  if (!src.dims.some((d) => d.id === "side")) return [];
  const seen = new Set<string>();
  for (const f of factsFor(src, report, season)) if (f.dims.side) seen.add(f.dims.side);
  return [...seen].sort();
}

export const measureOf = (s: Source, id: string) => s.measures.find((m) => m.id === id) ?? s.measures[0];
export const dimOf = (s: Source, id: string) => s.dims.find((d) => d.id === id) ?? s.dims[0];

// --------------------------------------------------------------------- facts

function trackFacts(report: MatchReport): Datum[] {
  return buildRows(report).map((r) => ({
    dims: {
      player: playerLabel(r.p, report),
      side: r.p.team === "unknown" ? "Unassigned" : teamName(report, r.p.team),
    },
    vals: {
      // Refused distance and withheld sprints are ABSENT, the same rule as peak speed below.
      ...(r.distance == null ? {} : { distance_km: r.distance }),
      minutes: r.minutes,
      // A peak speed the pipeline could not stand behind is null, not zero, so
      // the key is ABSENT rather than present-and-wrong. The pivot below counts
      // `k` only for facts that carried the measure, so an omitted value shrinks
      // the coverage it reports instead of dragging a mean towards zero.
      ...(r.peak == null ? {} : { top_speed_kmh: r.peak }),
      ...(r.sprints == null ? {} : { sprints: r.sprints }),
      // The mean of a player's own positions, in metres. NOT a role: sides
      // swap at half time, so every mean x lands near the middle. It is only
      // honest as a MAP of where somebody was, which is what draws it.
      avg_x_m: r.p.avg_position?.[0] ?? 0,
      avg_y_m: r.p.avg_position?.[1] ?? 0,
    },
  }));
}

function feedFacts(report: MatchReport): Datum[] {
  return (report.meta?.facts?.player_stats ?? []).map((p) => ({
    dims: {
      player: p.name,
      side: teamName(report, p.side),
      position: p.pos || "Unrecorded",
    },
    vals: defined(Object.fromEntries(Object.entries(p.stats).map(([k, v]) => [k, num(v)]))),
  }));
}

function playFacts(report: MatchReport): Datum[] {
  return matchPlays(report).map((p) => ({
    dims: {
      playType: PLAY_TYPES[p.type] ?? p.type,
      zone: PLAY_ZONES[p.zone] ?? p.zone,
      outcome: PLAY_OUTCOMES[p.outcome] ?? p.outcome,
      side: teamName(report, p.side),
      period: p.period === 1 ? "First half" : "Second half",
    },
    vals: {},
  }));
}

/** One fact per side: what the camera measured about the team as a whole. */
function teamFacts(report: MatchReport): Datum[] {
  return (["home", "away"] as const).flatMap((side) => {
    const t = report.teams?.[side];
    if (!t) return [];
    return [
      {
        dims: { side: teamName(report, side) },
        vals: {
          distance_km: (t.distance_m ?? 0) / 1000,
          // Absent, not zero - the same rule the player path above and
          // seasonFacts below both follow. A refused team peak that arrives as
          // 0 makes `ranked` drop the row and the copy then reads "this match
          // has no team total for top speed", when the total exists and it was
          // the reading that was declined.
          ...(t.top_speed_kmh == null ? {} : { top_speed_kmh: t.top_speed_kmh }),
          ...(t.sprints == null ? {} : { sprints: t.sprints }),
          ...Object.fromEntries(Object.entries(t.shape ?? {}).map(([k, v]) => [k, Number(v) || 0])),
        },
      },
    ];
  });
}

/** One fact per side per team metric the feed carries. */
function teamFeedFacts(report: MatchReport): Datum[] {
  const facts = report.meta?.facts;
  const rows = [
    ...(facts?.top_stats ?? []).map((t) => ({ label: t.label, home: t.home, away: t.away })),
    ...(facts?.sections ?? []).flatMap((s) => s.rows),
  ];
  return (["home", "away"] as const).flatMap((side) => {
    // One `seen` PER SIDE. Shared across both, the guard only ran on the home
    // pass, so home kept the first occurrence of a repeated metric and away
    // kept the last — two bars in one chart read off two different rows.
    const seen = new Set<string>();
    const vals: Record<string, number> = {};
    for (const r of rows) {
      const id = metricId(r.label);
      // First row wins, so a metric repeated across groups is one measure.
      if (seen.has(id)) continue;
      seen.add(id);
      const v = num(r[side]);
      if (v !== undefined) vals[id] = v;
    }
    return [{ dims: { side: teamName(report, side) }, vals }];
  });
}

function territoryFacts(report: MatchReport): Datum[] {
  const t = report.territory;
  if (!t) return [];
  return [
    // By pitch x, the same thirds for both sides: left / centre / right, whatever the keys say.
    ["Left third", t.defensive_third],
    ["Centre third", t.middle_third],
    ["Right third", t.attacking_third],
  ].map(([third, share]) => ({ dims: { third: String(third) }, vals: { share: Number(share) || 0 } }));
}

/** One fact per named player in the lineup feed — the only place a position,
 *  an age or a height exists for a match. */
function squadFacts(report: MatchReport): Datum[] {
  const facts = report.meta?.facts;
  const out: Datum[] = [];
  for (const side of ["home", "away"] as const) {
    const name = teamName(report, side);
    for (const p of facts?.lineups?.[side]?.xi ?? []) {
      out.push({
        dims: { position: p.pos || "Unrecorded", side: name, player: p.name },
        // No `minutes_on`: a starter did not come on, and averaging the
        // substitutes’ minutes against a pile of starter zeros is not an
        // average of anything.
        vals: defined({ rating: p.rating, age: p.age, height_cm: heightCm(p.height) }),
      });
    }
    for (const p of facts?.subs?.[side] ?? []) {
      // An unused substitute has no minute and is not a player who came on.
      if (!p.on) continue;
      out.push({
        dims: { position: p.pos || "Unrecorded", side: name, player: p.name },
        vals: defined({
          rating: p.rating,
          age: p.age,
          height_cm: heightCm(p.height),
          minutes_on: Number(String(p.on).replace(/\D/g, "")) || undefined,
        }),
      });
    }
  }
  return out;
}

function shotFacts(report: MatchReport): Datum[] {
  return (report.meta?.facts?.shots?.points ?? []).map((s) => ({
    dims: {
      side: teamName(report, s.side),
      period: s.period === 1 ? "First half" : "Second half",
      result: s.goal ? "Goal" : "No goal",
    },
    vals: {},
  }));
}

function momentumFacts(report: MatchReport): Datum[] {
  return (report.meta?.facts?.momentum?.values ?? []).map((v, i) => ({
    dims: { minute: String(i + 1).padStart(2, "0") },
    vals: { pressure: Number(v) || 0 },
  }));
}

function zoneFacts(report: MatchReport): Datum[] {
  const z = report.meta?.facts?.attacking_zones;
  if (!z) return [];
  const names = ["Left", "Centre", "Right"];
  return (["home", "away"] as const).flatMap((side) =>
    (z[side] ?? []).map((share, i) => ({
      dims: { channel: names[i] ?? String(i), side: teamName(report, side) },
      vals: { share: Number(share) || 0 },
    })),
  );
}

/** One fact per player per match, so a season can be broken down by either. */
function seasonFacts(season: Season, only?: string[]): Datum[] {
  const name = new Map(season.matches.map((m) => [m.id, m.name]));
  // An empty list is not "no filter" — it is a range that selected nothing,
  // and returning the whole season for it would quietly answer a different
  // question from the one on screen.
  const keep = only ? new Set(only) : null;
  return season.players.flatMap((p) =>
    Object.entries(p.by_match).filter(([mid]) => !keep || keep.has(mid)).map(([mid, s]) => ({
      dims: {
        player: p.name || p.id,
        position: p.position || "Unrecorded",
        side: p.team || "Unassigned",
        match: name.get(mid) ?? mid,
      },
      vals: {
        distance_km: s.distance_km,
        minutes: s.minutes,
        // Same rule as the player path above: a refused peak is an ABSENT key,
        // not a zero, so the pivot's coverage shrinks instead of a mean being
        // dragged down by matches the pipeline could not stand behind.
        ...(s.top_speed_kmh == null ? {} : { top_speed_kmh: s.top_speed_kmh }),
        // Sprints are WITHHELD at season level: ABSENT, never summed, whatever an old row holds.
        matches: 1,
      },
    })),
  );
}

// ----------------------------------------------------------------- the pivot

export interface QueryRow {
  label: string;
  /** One per series, aligned to `series`. */
  values: number[];
  /** Formatted, for the table and the direct labels on the bars. */
  text: string[];
}

export interface QueryResult {
  rows: QueryRow[];
  series: string[];
  /** The largest value anywhere, so every bar is drawn against one scale. */
  max: number;
  measure: Measure;
  dim: Dim;
  /** Whether this dimension can honestly be drawn as a line. */
  ordered: boolean;
  /** The colours this figure is drawn in, one per series (or per category, for
   *  a pie). Absent or short falls through to the validated palette — see
   *  colorFor in chart-svg.ts. Frozen with the figure like its rows. */
  colors?: string[];
  note: string;
  tier: Tier;
  refusal?: string;
  /** The answer stands, but something was asked for and not done. A refusal
   *  means no chart; this means a chart that is not quite the one requested,
   *  and it has to be said before the block goes in rather than after. */
  warning?: string;
  /** Some sources have a NATIVE visual that no bar chart can stand in for — a
   *  shot map is a pitch, momentum is the swing across the ninety. Those
   *  renders need the facts themselves rather than the roll-up, so the raw
   *  payload rides along beside the rows and is frozen with them. */
  extra?: Extra;
}

export interface ShotExtra {
  kind: "shots";
  points: { x: number; y: number; side: string; goal: boolean; period: number }[];
  home: string;
  away: string;
}

export interface MomentumExtra {
  kind: "momentum";
  values: number[];
  goals: { minute: number; side: string }[];
  home: string;
  away: string;
}

/** Discs on a pitch: an average position map, or a team sheet's shape. Both
 *  are the same payload because both are "these people, at these spots" — the
 *  difference is only whether the spots were measured or drawn, and that is
 *  what `measured` says. */
export interface SpotsExtra {
  kind: "spots";
  spots: { x: number; y: number; side: string; label: string }[];
  home: string;
  away: string;
  measured: boolean;
}

/** Where a side actually spent the match, ALREADY BINNED.
 *
 *  The report carries 600 sampled points per track and this used to freeze up
 *  to 8,000 of them into the document, as `{"x":12.345678901234,...}` inside an
 *  HTML attribute where every quote costs six characters. One heat map came to
 *  423 KB — 140x the next largest block in a note, and two of them would pass
 *  the server's 1 MB cap and fail the save outright.
 *
 *  The renderer bins those points into a fixed grid and throws the rest away,
 *  so the document was storing eight thousand numbers to draw ninety-six. Now
 *  the binning happens here, at freeze time, like every other computation in a
 *  block. Same picture, about a thousandth of the bytes.
 *
 *  `cols`/`rows` ride along rather than being read from a constant, so a saved
 *  note keeps drawing correctly the day somebody changes the grid — otherwise
 *  every note in existence would silently re-shape to a new resolution its
 *  counts were never binned at. `total` is the denominator for each cell's
 *  share, which used to be `points.length`. */
export interface HeatExtra {
  kind: "heat";
  /** Counts, row-major, `cols * rows` of them. */
  bins: number[];
  cols: number;
  rows: number;
  /** How many sampled positions went in, for the per-cell share. */
  total: number;
  who: string;
}

/** The grid a heat map is binned to. Lives here and not in the renderer
 *  because this is where the binning now happens; the renderer takes whatever
 *  the stored figure says it was binned at. */
export const HEAT_COLS = 12;
export const HEAT_ROWS = 8;

/** Percentage positions into bin counts. Out-of-range points are clamped
 *  rather than dropped: a position on the touchline is a real position, and
 *  the pitch dimensions the percentages were computed from are themselves
 *  approximate. */
export function binHeat(points: { x: number; y: number }[]): number[] {
  const bins = new Array(HEAT_COLS * HEAT_ROWS).fill(0);
  for (const pt of points) {
    const c = Math.max(0, Math.min(HEAT_COLS - 1, Math.floor((pt.x / 100) * HEAT_COLS)));
    const r = Math.max(0, Math.min(HEAT_ROWS - 1, Math.floor((pt.y / 100) * HEAT_ROWS)));
    bins[r * HEAT_COLS + c] += 1;
  }
  return bins;
}

/** Bands of a pitch with a share each — the thirds, or the three channels.
 *  `axis` says which way the bands run, because a third is a slice across the
 *  pitch and a channel is a slice along it. */
export interface ZoneExtra {
  kind: "zones";
  bands: { label: string; share: number }[];
  axis: "length" | "width";
  who: string;
}

/** WHERE IT WENT IN THE GOAL, seen from behind the ball.
 *
 *  The one figure in a scouting report that is not drawn on the pitch, and the
 *  reason it needed its own payload rather than another crop of the board: the
 *  goal mouth is a different PLANE. A pitch view is metres across the ground; a
 *  goal is 7.32 m across by 2.44 m up, and a shot that missed has a position in
 *  that frame - over the bar, wide of the post - which no crop of the ground
 *  can express.
 *
 *  Stored 0-100 across the goal and 0-100 up it, so the numbers survive being
 *  read by a human and do not encode the FIFA dimensions into every document.
 *  OUTSIDE that range is a miss and is drawn where it went, which is the whole
 *  point of the figure - a penalty taker who misses always misses to the same
 *  side, and that is a thing you find out by looking. */
export interface GoalExtra {
  kind: "goal";
  shots: { x: number; y: number; scored: boolean; label: string }[];
  who: string;
}

/** ONE PLAYER AGAINST A POPULATION, ONE WEDGE PER METRIC.
 *
 *  A radar joins its points, which draws a shape whose AREA looks like a
 *  quantity and is not one - reorder the axes and the same player gets a
 *  different area. A pizza does not join anything: each metric is its own
 *  wedge, so the only thing being compared is one radius against the ring, and
 *  the number is written in the wedge rather than left to be read off a spoke.
 *  That is why the coach-facing convention settled on it, and why mplsoccer's
 *  gallery leads with it.
 *
 *  `pct` is a PERCENTILE, 0-100 - a rank inside `population`, not a share and
 *  not a normalised value. `population` travels with it because a percentile
 *  against eleven team-mates and one against four hundred players in a season
 *  are different claims wearing the same number. */
export interface PizzaExtra {
  kind: "pizza";
  who: string;
  /** How many players the rank was taken against. Printed, always. */
  population: number;
  slices: {
    label: string;
    /** 0-100. The radius. */
    pct: number;
    /** What the number actually was, already formatted. Written in the wedge. */
    text: string;
    /** Optional banding - "Attacking", "Defending" - which colours the wedge.
     *  Absent everywhere means one colour, which is the honest default when
     *  nobody has grouped the metrics. */
    group?: string;
  }[];
}

/** EVERY VISIBLE PLAYER AT ONE INSTANT. A frame slice.
 *
 *  The payload behind three figures - the freeze frame itself, the convex hull
 *  a side's outfielders make, and a coarse pitch-control grid - because all
 *  three are the same question asked once and drawn three ways, and storing
 *  the moment once is the difference between one 2 KB attribute and three.
 *
 *  `seen` and `squad` are not decoration. A ball-following camera hides players,
 *  and a shape drawn from 13 of 22 is not the team's shape; every renderer here
 *  prints the pair so the reader can discount it. `synthetic` says the positions
 *  were generated rather than measured - a feed importer can write matches whose
 *  tracking is invented, and a figure that cannot tell you which is a figure
 *  that will eventually be quoted as measurement. */
export interface FreezeExtra {
  kind: "freeze";
  t_s: number;
  players: { x: number; y: number; side: string; label: string }[];
  ball?: { x: number; y: number };
  seen: number;
  squad: number;
  home: string;
  away: string;
  synthetic: boolean;
}

/** PASSES, AS ARROWS. One payload, three figures.
 *
 *  network      nodes at each player's mean position, edges weighted by volume
 *  progressive  the arrows themselves, direction and outcome
 *  sonar        per player, a rose of where his passes went
 *
 *  Capped by the builder rather than the renderer, for the reason HeatExtra
 *  spells out above: what a block stores is what every copy of the document
 *  carries forever, and a thousand passes in an HTML attribute is a quarter of
 *  a megabyte of quoting. Aggregation for the network happens at draw time
 *  because a few hundred items is cheap; the CAP is what keeps it cheap. */
export interface PassExtra {
  kind: "passes";
  passes: {
    x0: number; y0: number; x1: number; y1: number;
    side: string;
    /** Shirt or initials. The network and the sonar group on these. */
    from: string;
    to: string;
    ok: boolean;
  }[];
  home: string;
  away: string;
  /** How many the cap left behind, so a partial picture says it is one. */
  dropped: number;
  synthetic: boolean;
}

export type Extra =
  | ShotExtra | MomentumExtra | SpotsExtra | HeatExtra | ZoneExtra | GoalExtra
  | PizzaExtra | FreezeExtra | PassExtra;

interface Cell {
  sum: number;
  /** Facts in this group. What "How many" counts. */
  n: number;
  /** Facts in this group that carried the measure. What an average divides by. */
  k: number;
  max: number;
  num: number;
  den: number;
}

const cell = (): Cell => ({ sum: 0, n: 0, k: 0, max: 0, num: 0, den: 0 });

function finish(c: Cell, m: Measure): number {
  if (m.ratio) return c.den ? (c.num / c.den) * m.ratio.scale : 0;
  if (m.agg === "count") return c.n;
  // Divided by the facts that CARRIED the measure, not by every fact in the
  // group. Three midfielders, two rated 8.1 and 6.5 and one unrated, is a
  // mean of 7.30 and not 4.87.
  if (m.agg === "avg") return c.k ? c.sum / c.k : 0;
  if (m.agg === "max") return c.max;
  return c.sum;
}

/** A number as the document prints it. Takes the two fields it actually reads
 *  rather than a whole Measure, so a script assembling a chart by hand can
 *  label its rows the same way a queried one does. */
export const fmt = (v: number, m: { dp: number; unit?: string }) =>
  `${v.toFixed(m.dp)}${m.unit ? ` ${m.unit}` : ""}`;

const BUILDERS: Record<string, (r: MatchReport) => Datum[]> = {
  tracks: trackFacts,
  feed: feedFacts,
  plays: playFacts,
  teams: teamFacts,
  teamFeed: teamFeedFacts,
  territory: territoryFacts,
  squad: squadFacts,
  shots: shotFacts,
  momentum: momentumFacts,
  zones: zoneFacts,
};

/** A source's rows. One definition, because a chart and the table beside it
 *  showing the same thing have to BE the same numbers — and they only are if
 *  they came out of the same function. */
function factsFor(
  src: Source,
  report: MatchReport | null,
  season: Season | null,
  matchIds?: string[],
  team?: string,
): Datum[] {
  const all =
    src.scope === "season"
      ? season
        ? seasonFacts(season, matchIds)
        : []
      : report
        ? (BUILDERS[src.id]?.(report) ?? [])
        : [];
  // One team, not both. A ranking of "the players" that mixes the two squads
  // is a ranking of nobody's squad — a coach preparing their own side wants
  // theirs, and a coach scouting the opposition wants the other one.
  // Applied here rather than as a WHERE on the pivot so the roll-up, the
  // maximum and the axis are all computed over the population on screen.
  return team ? all.filter((f) => f.dims.side === team) : all;
}

/** Why a source has nothing, said in the words that name what to do about it. */
function emptyReason(src: Source): string {
  if (src.id === "tracks" || src.id === "teams" || src.id === "territory") {
    return "This match has no tracking yet.";
  }
  if (src.id === "season") {
    return "No player has a confirmed identity yet, so there is nothing to add up across matches. Assign tracks to your squad on the identity screen.";
  }
  return `This match carries no ${src.label.toLowerCase()}. That needs the ball, and the pipeline follows bodies.`;
}

/** The picked rows, in the order they were picked. Null when nothing was
 *  picked, so the caller falls through to its usual sort. A name that is not
 *  in the result is simply absent — a comparison cannot invent a row. */
function keepOrder<T extends { label: string }>(rows: T[], keep?: string[]): T[] | null {
  if (!keep?.length) return null;
  return keep.map((k) => rows.find((r) => r.label === k)).filter((r): r is T => !!r);
}

/** How the rows are ordered, when the caller says.
 *
 *  The default has always been "biggest first, unless the dimension has an
 *  order of its own" — which is right for the question a coach asks by picking
 *  a measure, and wrong the moment they want the same chart alphabetically, or
 *  smallest first to find who is not running. `sort` says which; absent keeps
 *  every existing figure exactly as it was drawn.
 *
 *  A COMPARISON still overrules it. Picking Varga then Okafor means those
 *  two in that order, and re-sorting them by whoever ran further would shuffle
 *  the pair under the reader every time the measure changed.
 */
export type SortBy = "" | "value" | "valueAsc" | "label" | "labelDesc" | "natural";

export const SORTS: { id: SortBy; label: string }[] = [
  { id: "value", label: "Biggest first" },
  { id: "valueAsc", label: "Smallest first" },
  { id: "label", label: "A to Z" },
  { id: "labelDesc", label: "Z to A" },
  { id: "natural", label: "As it happened" },
];

function applySort<T extends { label: string; values: number[] }>(
  rows: T[],
  how: SortBy,
  ordered: boolean,
  order: string[],
): T[] {
  const byNatural = () => rows.sort((a, b) => order.indexOf(a.label) - order.indexOf(b.label));
  const byValue = (dir: number) =>
    rows.sort((a, b) => dir * (Math.max(...b.values) - Math.max(...a.values)));
  const byLabel = (dir: number) => rows.sort((a, b) => dir * a.label.localeCompare(b.label));
  switch (how) {
    case "value":
      return byValue(1);
    case "valueAsc":
      return byValue(-1);
    case "label":
      return byLabel(1);
    case "labelDesc":
      return byLabel(-1);
    case "natural":
      return byNatural();
    default:
      // Unasked: the behaviour every figure already in a note was drawn with.
      return ordered ? byNatural() : byValue(1);
  }
}

export function runQuery(opts: {
  report: MatchReport | null;
  season: Season | null;
  source: string;
  dim: string;
  measure: string;
  /** "" for one series, or a dimension id to split into two. */
  split: string;
  limit: number;
  /** Season sources only: the subset of matches the range covers. Undefined
   *  means every match the season has. */
  matchIds?: string[];
  /** One side's rows only, by team NAME as the facts spell it. Empty or absent
   *  means both. */
  team?: string;
  /** How it is going to be drawn. Only the pitch visuals care: they need the
   *  facts themselves rather than the roll-up, and which facts depends on
   *  which picture. Absent is the same as a bar chart. */
  render?: string;
  /** How to order the rows. Absent is the historical default — see applySort. */
  sort?: SortBy;
  /** COMPARE these, and nothing else. Named dimension values, in the order
   *  they were picked. Empty or absent is the ordinary top-N. */
  keep?: string[];
}): QueryResult {
  const src = sourceOf(opts.source, opts.report);
  const dim = dimOf(src, opts.dim);
  const m = measureOf(src, opts.measure);
  const base = { rows: [], series: [], max: 0, measure: m, dim, ordered: !!dim.ordered, tier: src.tier };

  if (src.scope === "season" ? !opts.season : !opts.report) {
    return { ...base, note: "", refusal: src.scope === "season" ? "The season has not loaded." : "No match is open." };
  }

  const facts = factsFor(src, opts.report, opts.season, opts.matchIds, opts.team);

  if (!facts.length) {
    return {
      ...base,
      note: "",
      refusal: emptyReason(src),
    };
  }

  // A split is a second series, and only ever a second: two is the number a
  // reader can hold without a legend lookup on every bar, and the palette is
  // validated as a pair.
  //
  // It used to take the first two values ALPHABETICALLY and `continue` past
  // every fact belonging to the rest. "Goals by player split by position" kept
  // Attacker and Defender and silently dropped every midfielder and every
  // goalkeeper — 14 of 31 players on the demo feed — while the panel went on
  // reporting the row count as though they were there. The comment here used
  // to claim the remainder folded into "everything else". It did not fold; it
  // was deleted.
  //
  // So a split that does not fit is not half-drawn. The chart is drawn without
  // it and says why.
  const splitDim = opts.split && opts.split !== dim.id ? opts.split : "";
  const splitValues = splitDim
    ? [...new Set(facts.map((f) => f.dims[splitDim]).filter(Boolean))].sort()
    : [];
  const overflow = splitValues.length > 2 ? splitValues : null;
  const series = overflow || splitValues.length < 2 ? [] : splitValues;
  const splitWarning = overflow
    ? `Not split. ${dimOf(src, splitDim).label} takes ${overflow.length} values here (${overflow.join(", ")}) and one chart carries two series. Split on a two-value breakdown, or draw one chart for each.`
    : "";

  const table = new Map<string, Cell[]>();
  const width = Math.max(1, series.length);
  for (const f of facts) {
    const key = f.dims[dim.id];
    if (!key) continue;
    const si = series.length ? series.indexOf(f.dims[splitDim]) : 0;
    if (si < 0) continue;
    if (!table.has(key)) table.set(key, Array.from({ length: width }, cell));
    const c = table.get(key)![si];
    const raw = f.vals[m.ratio ? m.ratio.num : m.id];
    const v = raw ?? 0;
    c.sum += v;
    c.n += 1;
    if (raw !== undefined) c.k += 1;
    c.max = Math.max(c.max, v);
    if (m.ratio) {
      c.num += f.vals[m.ratio.num] ?? 0;
      c.den += f.vals[m.ratio.den] ?? 0;
    }
  }

  // Insertion order of the pivot, which is first-appearance order of the facts.
  const order = [...table.keys()];
  let rows: QueryRow[] = [...table.entries()].map(([label, cells]) => {
    const values = cells.map((c) => finish(c, m));
    return { label, values, text: values.map((v) => fmt(v, m)) };
  });

  // An ordered dimension keeps its own order — a match list sorted by size is
  // a bar chart pretending to be a timeline. Everything else leads with the
  // biggest, which is the question a coach asked by picking a measure.
  //
  // "Its own order" is the order the FACTS came out in, not alphabetical.
  // Sorting the labels reads the same for minutes and halves and is wrong for
  // the thirds of a pitch, which came out defensive-middle-attacking and sort
  // attacking-defensive-middle. The builder is the thing that knows the order;
  // the sort was quietly overruling it.
  //
  // A COMPARISON overrules both. Picking Varga then Okafor means those two,
  // in that order — not those two re-sorted by whoever ran further, which
  // would make the chart's order depend on its measure and shuffle the pair
  // under the reader every time they changed it.
  rows = keepOrder(rows, opts.keep) ?? applySort(rows, opts.sort ?? "", !!dim.ordered, order);
  if (!opts.keep?.length) rows = rows.slice(0, opts.limit);

  const max = Math.max(0, ...rows.flatMap((r) => r.values));
  const extra = nativeExtra(src.id, opts.report, opts.render, opts.team);
  if (!rows.length) {
    return { ...base, series, note: "", refusal: `Nothing has a ${dim.label.toLowerCase()} recorded.` };
  }
  return {
    ...base,
    rows,
    series,
    max,
    ...(extra ? { extra } : {}),
    ...(splitWarning ? { warning: splitWarning } : {}),
    note: [
      syntheticNote(opts.report),
      src.note || (src.id === "season" ? seasonNote(opts.season!, opts.matchIds) : ""),
      splitWarning,
    ]
      .filter(Boolean)
      .join(" "),
  };
}

/** The facts a native visual draws from.
 *
 *  Keyed on the RENDER as well as the source, because one source feeds more
 *  than one picture — tracks are an average-position map and a heat map, and
 *  the two need different payloads. Only one rides along, because only one is
 *  on screen, and freezing both into the document would double every chart's
 *  stored size for a picture nobody is looking at.
 */
function nativeExtra(
  sourceId: string,
  report: MatchReport | null,
  render?: string,
  team?: string,
): QueryResult["extra"] {
  const facts = report?.meta?.facts;
  const home = report ? teamName(report, "home") : "Home";
  const away = report ? teamName(report, "away") : "Away";
  const mine = (side: string) => !team || side === team;
  const who = team || `${home} and ${away}`;

  if (sourceId === "shots" && facts?.shots?.points?.length) {
    return {
      kind: "shots",
      // Flipped. The feed puts home attacking the LEFT goal; every board in
      // this app has home attacking the right. A shot map beside a board that
      // disagreed about which way the team was playing is worse than no map.
      points: facts.shots.points
        .filter((p) => mine(teamName(report!, p.side)))
        .map((p) => ({
          x: 100 - p.x * 100,
          y: p.y * 100,
          side: p.side,
          goal: !!p.goal,
          period: p.period,
        })),
      home,
      away,
    };
  }

  if (sourceId === "momentum" && facts?.momentum?.values?.length) {
    return {
      kind: "momentum",
      values: facts.momentum.values,
      goals: (facts.momentum.goals ?? []).map((g) => ({ minute: g.minute, side: g.side })),
      home,
      away,
    };
  }

  if (render === "spots" && sourceId === "tracks" && report) {
    const L = report.meta?.pitch?.length || 105;
    const W = report.meta?.pitch?.width || 68;
    const spots = buildRows(report)
      .filter((r) => r.p.avg_position && mine(teamName(report, r.p.team)))
      .map((r) => ({
        x: ((r.p.avg_position![0] ?? 0) / L) * 100,
        y: ((r.p.avg_position![1] ?? 0) / W) * 100,
        side: r.p.team === "unknown" ? "" : r.p.team,
        label: r.p.jersey != null ? String(r.p.jersey) : "",
      }));
    return spots.length ? { kind: "spots", spots, home, away, measured: true } : undefined;
  }

  if (render === "lineup" && sourceId === "squad" && report) {
    const spots: SpotsExtra["spots"] = [];
    for (const side of ["home", "away"] as const) {
      if (!mine(teamName(report, side))) continue;
      for (const pl of facts?.lineups?.[side]?.xi ?? []) {
        if (pl.x == null || pl.y == null) continue;
        // The team sheet draws each side in ITS OWN half, y running from its
        // own goal out. Home keeps that half, away is reflected into the other
        // one, or both squads pile up on the same end of the pitch.
        const along = (pl.y ?? 0) * 50;
        spots.push({
          x: side === "home" ? along : 100 - along,
          y: side === "home" ? (pl.x ?? 0) * 100 : 100 - (pl.x ?? 0) * 100,
          side,
          label: pl.n != null ? String(pl.n) : "",
        });
      }
    }
    return spots.length ? { kind: "spots", spots, home, away, measured: false } : undefined;
  }

  if (render === "heat" && sourceId === "tracks" && report) {
    const L = report.meta?.pitch?.length || 105;
    const W = report.meta?.pitch?.width || 68;
    const keep = new Set(
      (report.players ?? [])
        .filter((pl) => mine(teamName(report, pl.team)))
        .map((pl) => String(pl.id)),
    );
    const points: { x: number; y: number }[] = [];
    for (const [id, pts] of Object.entries(report.heatmaps ?? {})) {
      if (!keep.has(id)) continue;
      for (const xy of pts) points.push({ x: ((xy[0] ?? 0) / L) * 100, y: ((xy[1] ?? 0) / W) * 100 });
    }
    // No cap any more. The 8,000 was there to bound the document, and the
    // document now holds 96 counts however many positions went into them —
    // so every tracked point can be counted instead of the first eight
    // thousand, which silently favoured whichever tracks came first.
    return points.length
      ? {
          kind: "heat",
          bins: binHeat(points),
          cols: HEAT_COLS,
          rows: HEAT_ROWS,
          total: points.length,
          who,
        }
      : undefined;
  }

  if (render === "zonepitch" && sourceId === "territory" && report?.territory) {
    const t = report.territory;
    return {
      kind: "zones",
      axis: "length",
      who: "Both sides",
      bands: [
        { label: "Left third", share: t.defensive_third ?? 0 },
        { label: "Centre third", share: t.middle_third ?? 0 },
        { label: "Right third", share: t.attacking_third ?? 0 },
      ],
    };
  }

  if (render === "zonepitch" && sourceId === "zones" && report) {
    const z = facts?.attacking_zones;
    const side = team === away ? "away" : "home";
    const vals = z?.[side] ?? [];
    if (!vals.length) return undefined;
    return {
      kind: "zones",
      axis: "width",
      who: teamName(report, side),
      bands: ["Left", "Centre", "Right"].map((label, i) => ({ label, share: vals[i] ?? 0 })),
    };
  }

  return undefined;
}

/** The visual a source is drawn with unless somebody says otherwise. */
export function defaultRender(sourceId: string): string {
  if (sourceId === "shots") return "pitch";
  if (sourceId === "momentum") return "momentum";
  return "bar";
}

const seasonNote = (s: Season, only?: string[]) => {
  const n = only ? only.length : s.matches.length;
  return (
    `${s.coverage.bound} of ${s.coverage.total} tracked players across ${n} ` +
    `${n === 1 ? "match" : "matches"} carry a confirmed identity; the rest are not in this figure.`
  );
};

/** The default question a new block asks, per source. */
export function defaults(sourceId: string, report: MatchReport | null = null): { dim: string; measure: string } {
  const s = sourceOf(sourceId, report);
  return { dim: s.dims[0].id, measure: s.measures[0].id };
}

export function queryTitle(r: QueryResult, split: string, render = "bar"): string {
  // A native visual is not a breakdown, so it is not named like one: nobody
  // calls a shot map "How many by team", and an average-position map titled
  // "Distance by player" is a caption describing a different picture.
  if (render === "pitch" && r.extra?.kind === "shots") return "Shot map";
  if (render === "momentum" && r.extra?.kind === "momentum") return "Momentum";
  if ((render === "spots" || render === "lineup") && r.extra?.kind === "spots") {
    return r.extra.measured ? "Average positions" : "Line-up";
  }
  if (render === "heat" && r.extra?.kind === "heat") return `Where ${r.extra.who} played`;
  if (render === "zonepitch" && r.extra?.kind === "zones") return `${r.extra.who} by zone`;
  const by = ` by ${r.dim.label.toLowerCase()}`;
  const sp = split && split !== r.dim.id ? `, split by ${split === "side" ? "team" : split}` : "";
  return `${r.measure.label}${by}${sp}`;
}

// ------------------------------------------------------------------- columns

export interface ColumnRow {
  label: string;
  /** "home" | "away" | "" — the side dot, and nothing else. */
  side: string;
  values: number[];
  text: string[];
}

export interface ColumnResult {
  head: string[];
  rows: ColumnRow[];
  note: string;
  tier: Tier;
  refusal?: string;
}

/** Several measures at once, down one dimension. A TABLE, in other words.
 *
 *  `runQuery` answers one measure and can split it in two, which is what a bar
 *  chart can honestly draw. A table is the other shape: one row per player, a
 *  COLUMN per thing you wanted to compare, which is what a coach means by
 *  "pick what goes in it". Running runQuery once per measure and zipping the
 *  results would nearly work and would be wrong at the edge — each call sorts
 *  by its own measure, so row three of column one is a different player from
 *  row three of column two. One pass, one row order, N cells.
 *
 *  Measures all come from ONE source on purpose. Two sources means two keys —
 *  the feed knows names, the tracker knows shirt numbers — and a table that
 *  put them in one row would be asserting an identity join nobody made.
 */
export function runColumns(opts: {
  report: MatchReport | null;
  season: Season | null;
  source: string;
  dim: string;
  measures: string[];
  limit: number;
  matchIds?: string[];
  team?: string;
  keep?: string[];
  /** How to order the rows. Absent is the historical default. */
  sort?: SortBy;
}): ColumnResult {
  const src = sourceOf(opts.source, opts.report);
  const dim = dimOf(src, opts.dim);
  const ms = (opts.measures.length ? opts.measures : [src.measures[0].id]).map((id) =>
    measureOf(src, id),
  );
  const base = { head: [dim.label, ...ms.map((m) => m.label)], rows: [], tier: src.tier, note: "" };

  if (src.scope === "season" ? !opts.season : !opts.report) {
    return {
      ...base,
      refusal: src.scope === "season" ? "The season has not loaded." : "No match is open.",
    };
  }
  const facts = factsFor(src, opts.report, opts.season, opts.matchIds, opts.team);
  if (!facts.length) return { ...base, refusal: emptyReason(src) };

  // The side dot is only honest when the source actually carries a side. On a
  // source that does not, every row gets "" and the column of dots is absent
  // rather than a column of guesses.
  const hasSide = src.dims.some((d) => d.id === "side") && dim.id !== "side";
  const home = opts.report ? teamName(opts.report, "home") : "";

  const table = new Map<string, { cells: Cell[]; side: string }>();
  for (const f of facts) {
    const key = f.dims[dim.id];
    if (!key) continue;
    if (!table.has(key)) {
      table.set(key, {
        cells: ms.map(cell),
        side: hasSide ? (f.dims.side === home ? "home" : f.dims.side ? "away" : "") : "",
      });
    }
    const row = table.get(key)!;
    ms.forEach((m, i) => {
      const c = row.cells[i];
      const raw = f.vals[m.ratio ? m.ratio.num : m.id];
      const v = raw ?? 0;
      c.sum += v;
      c.n += 1;
      if (raw !== undefined) c.k += 1;
      c.max = Math.max(c.max, v);
      if (m.ratio) {
        c.num += f.vals[m.ratio.num] ?? 0;
        c.den += f.vals[m.ratio.den] ?? 0;
      }
    });
  }

  let rows: ColumnRow[] = [...table.entries()].map(([label, r]) => {
    const values = r.cells.map((c, i) => finish(c, ms[i]));
    return { label, side: r.side, values, text: values.map((v, i) => fmt(v, ms[i])) };
  });

  // Sorted by the FIRST column, which is the one somebody picked first and so
  // the one they meant to rank by. An ordered dimension keeps the order its
  // facts arrived in — see runQuery for why that is not the same as sorting
  // the labels.
  const order = [...table.keys()];
  // `applySort` ranks on the largest value in the row; a TABLE ranks on the
  // first column, which is the one somebody picked first and so the one they
  // meant to rank by. Same rule, different reading of "the value".
  const byFirst = (dir: number) => rows.sort((a, b) => dir * ((b.values[0] ?? 0) - (a.values[0] ?? 0)));
  rows =
    keepOrder(rows, opts.keep) ??
    (opts.sort === "value"
      ? byFirst(1)
      : opts.sort === "valueAsc"
        ? byFirst(-1)
        : opts.sort
          ? applySort(rows, opts.sort, !!dim.ordered, order)
          : dim.ordered
            ? rows.sort((a, b) => order.indexOf(a.label) - order.indexOf(b.label))
            : byFirst(1));
  if (!opts.keep?.length) rows = rows.slice(0, opts.limit);

  if (!rows.length) return { ...base, refusal: `Nothing has a ${dim.label.toLowerCase()} recorded.` };
  return {
    ...base,
    rows,
    note: [syntheticNote(opts.report), src.note || (src.id === "season" ? seasonNote(opts.season!, opts.matchIds) : "")]
      .filter(Boolean)
      .join(" "),
  };
}

/** A composition is a set of parts of one whole — the thirds of a pitch, the
 *  three attacking channels, how the shots ended. Those are the tables the
 *  scouting report never prints on their own: it puts the chart beside them,
 *  because "26%" means nothing until you can see it against the other two.
 *
 *  Detected from the numbers rather than from a list of source ids, so a
 *  measure added later gets the same treatment without anyone remembering to
 *  add it here: a single column of non-negative values, at least three rows,
 *  either summing to about a hundred or declared a share by its own unit.
 */
export function isComposition(r: ColumnResult, measure: Measure): boolean {
  if (r.rows.length < 3 || r.head.length !== 2) return false;
  const vals = r.rows.map((x) => x.values[0] ?? 0);
  if (vals.some((v) => v < 0)) return false;
  const total = vals.reduce((a, b) => a + b, 0);
  if (total <= 0) return false;
  return measure.unit === "%" || measure.agg === "count" || Math.abs(total - 100) < 1.5;
}
