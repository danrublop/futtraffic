import { SPRINTS_WITHHELD_WHY, teamName, tierOf, withheld, type Tier } from "@/lib/report";
import type { MatchReport, Season } from "@/lib/types";

/** What a `/table` block can ask for, and how the answer is built.
 *
 *  This is the block the scouting reports a coach copies are made of: a ranked
 *  list of a squad on one number, over one match or over a season. The picker
 *  is deliberately a small closed set rather than a query builder — three
 *  choices (scope, grouping, metric) cover the tables in a real report, and
 *  every combination that is NOT answerable is refused with the reason rather
 *  than rendered empty.
 *
 *  WHAT CANNOT BE ASKED, and why the picker has no option for it:
 *
 *   - A rank between two NAMED players IN ONE MATCH. Nothing in a match report
 *     carries a name; a track has a shirt number at best. Over a season the
 *     names come from bindings a HUMAN confirmed, which is a different claim
 *     and the only one this product can make (see jobs.season, lib/report.ts).
 *   - A position in a single match. Positions come off the roster, where a
 *     coach typed them. Nothing derives one from avg_position, because sides
 *     swap at half time and the pipeline has no attacking direction.
 */

export type Scope = "match" | "season";
export type Group = "player" | "position" | "team";

export interface Metric {
  id: string;
  label: string;
  /** The match.json path the tier is looked up under. */
  field: string;
  unit: string;
  dp: number;
  /** Absent = every scope. */
  scopes?: Scope[];
  /** Absent = every grouping. */
  groups?: Group[];
}

export const METRICS: Metric[] = [
  // Per 90 first, and it is the default for a reason report.ts already gives:
  // a season total ranks whoever played the most minutes, which is a
  // team-selection report dressed up as a fitness one.
  { id: "km_per90", label: "Distance / 90", field: "derived.per90", unit: "km", dp: 2, scopes: ["season"] },
  { id: "distance_km", label: "Distance", field: "players.distance_km", unit: "km", dp: 2 },
  { id: "top_speed_kmh", label: "Top speed", field: "players.top_speed_kmh", unit: "km/h", dp: 1 },
  { id: "sprints", label: "Sprints", field: "players.sprints", unit: "", dp: 0 },
  { id: "sprints_per90", label: "Sprints / 90", field: "derived.per90", unit: "", dp: 1, scopes: ["season"] },
  { id: "minutes", label: "Minutes", field: "players.minutes", unit: "", dp: 0 },
  { id: "matches", label: "Matches", field: "players.minutes", unit: "", dp: 0, scopes: ["season"] },
];

export const metricOf = (id: string): Metric =>
  METRICS.find((m) => m.id === id) ?? METRICS[1];

export const SCOPES: { id: Scope; label: string }[] = [
  { id: "match", label: "This match" },
  { id: "season", label: "All matches" },
];

export const GROUPS: { id: Group; label: string; scopes: Scope[] }[] = [
  { id: "player", label: "By player", scopes: ["match", "season"] },
  { id: "position", label: "By position", scopes: ["season"] },
  { id: "team", label: "By team", scopes: ["match"] },
];

/** One frozen row of a rendered table. Values are STRINGS, formatted at insert
 *  time, for the same reason `/stat` bakes its number in: the document is
 *  sandboxed and offline once it leaves, and a note that silently changed its
 *  figures after a re-run is a note you could not send to a player. */
export interface TableRow {
  name: string;
  /** The quiet second line — a side, a shirt number, a match count. */
  sub: string;
  value: string;
  /** For the bar rendering: 0-1 against the top row. */
  frac: number;
}

export interface TableData {
  rows: TableRow[];
  /** What the table stands on, in the coach's words. Always rendered. */
  note: string;
  tier: Tier;
  /** Non-null when the ask cannot be answered — rendered instead of rows. */
  refusal?: string;
}

const fmt = (n: number, m: Metric) =>
  `${n.toFixed(m.dp)}${m.unit ? ` ${m.unit}` : ""}`;

function ranked(
  raw: { name: string; sub: string; n: number | null }[],
  metric: Metric,
  limit: number,
): TableRow[] {
  // `n == null` is a REFUSED measurement, not a zero: a player whose peak the
  // clamp bound has no reading to rank, so they leave the table rather than
  // appearing at the bottom on a fabricated 0.0. The `> 0` test did this by
  // accident while the value was coerced; it is explicit now.
  const sorted = raw
    .filter((r): r is { name: string; sub: string; n: number } => r.n != null && r.n > 0)
    .sort((a, b) => b.n - a.n)
    .slice(0, limit);
  const top = sorted[0]?.n || 1;
  return sorted.map((r) => ({
    name: r.name,
    sub: r.sub,
    value: fmt(r.n, metric),
    frac: Math.max(0, Math.min(1, r.n / top)),
  }));
}

// ---------------------------------------------------------------------------

function fromMatch(
  report: MatchReport | null,
  group: Group,
  metric: Metric,
  limit: number,
): TableData {
  const tier = tierOf(metric.field, report?.fields).tier;
  if (!report) {
    return { rows: [], tier, note: "", refusal: "No match is open, so there are no numbers to rank." };
  }

  if (metric.id === "sprints" && withheld(metric.field, report.fields)) {
    return { rows: [], tier, note: "", refusal: SPRINTS_WITHHELD_WHY };
  }

  if (group === "team") {
    const raw = (["home", "away"] as const).flatMap((side) => {
      const t = report.teams?.[side];
      if (!t) return [];
      const n =
        metric.id === "distance_km"
          ? (t.distance_m || 0) / 1000
          : metric.id === "top_speed_kmh"
            ? t.top_speed_kmh || 0
            : metric.id === "sprints"
              ? t.sprints || 0
              : 0;
      return n ? [{ name: teamName(report, side), sub: side, n }] : [];
    });
    return {
      rows: ranked(raw, metric, limit),
      tier: tierOf(metric.field.replace("players.", "teams."), report.fields).tier,
      note: raw.length
        ? "Team totals over every tracked player on that side."
        : "",
      refusal: raw.length ? undefined : `This match has no team total for ${metric.label.toLowerCase()}.`,
    };
  }

  // group === "player". No names: the pipeline cannot attribute an identity to
  // a track, so a shirt number is the furthest this is allowed to go, and the
  // track id when even that is missing.
  const raw = (report.players ?? []).map((p) => ({
    name: p.jersey != null ? `#${p.jersey}` : `Track ${p.id}`,
    sub: p.team === "unknown" ? "unassigned" : teamName(report, p.team),
    n:
      metric.id === "distance_km"
        ? p.distance_km || 0
        : metric.id === "top_speed_kmh"
          ? p.top_speed_kmh || 0
          : metric.id === "sprints"
            ? p.sprints || 0
            : metric.id === "minutes"
              ? p.minutes || 0
              : 0,
  }));
  const rows = ranked(raw, metric, limit);
  return {
    rows,
    tier,
    note: rows.length
      ? "Shirt numbers, not names — this match's identities are what the tracker read, and it reads a number or nothing."
      : "",
    refusal: rows.length ? undefined : "This match has no tracked players yet.",
  };
}

function fromSeason(
  season: Season | null,
  group: Group,
  metric: Metric,
  limit: number,
): TableData {
  // Matches is a count of bindings a person made; minutes sums players.minutes,
  // which resolves not_evaluated per match, so its season total cannot be
  // sturdier than its parts. It read "ok" here, hard-coded.
  const tier: Tier =
    metric.id === "matches" ? "ok" : metric.id === "minutes" ? "not_evaluated" : "prov";
  // Withheld at season level as on the match report, whatever old match.json files hold.
  if (metric.id === "sprints" || metric.id === "sprints_per90") {
    return { rows: [], tier: "withheld", note: "", refusal: SPRINTS_WITHHELD_WHY };
  }
  if (!season) {
    return { rows: [], tier, note: "", refusal: "The season has not loaded." };
  }
  const { bound, total } = season.coverage;
  const played = season.matches.length;
  if (!season.players.length) {
    return {
      rows: [],
      tier,
      note: "",
      refusal:
        total === 0
          ? "No match has finished yet, so there is nothing to add up."
          : `None of the ${total} tracked players across ${played} ${played === 1 ? "match" : "matches"} has a confirmed identity yet. Assign tracks to your squad on the identity screen and they will join here.`,
    };
  }

  const pick = (p: (typeof season.players)[number]) =>
    metric.id === "km_per90"
      ? p.km_per90
      : metric.id === "distance_km"
          ? p.distance_km
          : metric.id === "top_speed_kmh"
            ? p.top_speed_kmh
            : metric.id === "minutes"
              ? p.minutes
              : p.matches;

  const cover =
    `${bound} of ${total} tracked players across ${played} ${played === 1 ? "match" : "matches"} ` +
    `carry a confirmed identity; the rest are not in this table.`;

  if (group === "position") {
    const withPos = season.players.filter((p) => p.position);
    if (!withPos.length) {
      return {
        rows: [],
        tier,
        note: "",
        refusal:
          "Nobody in the squad has a position recorded. Positions are typed on the roster — nothing derives one from where a player averaged, because the sides swap at half time.",
      };
    }
    const by = new Map<string, { minutes: number; km: number; peak: number; matches: number; n: number }>();
    for (const p of withPos) {
      const k = String(p.position);
      const g = by.get(k) ?? { minutes: 0, km: 0, peak: 0, matches: 0, n: 0 };
      g.minutes += p.minutes;
      g.km += p.distance_km;
      // A refused peak is not a candidate for a group's best reading.
      if (p.top_speed_kmh != null) g.peak = Math.max(g.peak, p.top_speed_kmh);
      g.matches += p.matches;
      g.n += 1;
      by.set(k, g);
    }
    const raw = [...by.entries()].map(([pos, g]) => ({
      name: pos,
      sub: `${g.n} ${g.n === 1 ? "player" : "players"}`,
      // A group's per-90 is its own total over its own minutes, never a mean of
      // the members' per-90s: that would weight a 12-minute substitute the same
      // as a player who started every week.
      n:
        metric.id === "km_per90"
          ? g.minutes
            ? (g.km / g.minutes) * 90
            : 0
          : metric.id === "distance_km"
              ? g.km
              : metric.id === "top_speed_kmh"
                ? g.peak
                : metric.id === "minutes"
                  ? g.minutes
                  : g.matches,
    }));
    const skipped = season.players.length - withPos.length;
    return {
      rows: ranked(raw, metric, limit),
      tier,
      note:
        cover +
        (skipped ? ` ${skipped} of them has no position recorded and is not grouped.` : ""),
    };
  }

  const raw = season.players.map((p) => ({
    name: p.name || p.id,
    sub: [p.number != null ? `#${p.number}` : "", p.position ?? "", `${p.matches} ${p.matches === 1 ? "match" : "matches"}`]
      .filter(Boolean)
      .join(" · "),
    n: pick(p),
  }));
  return { rows: ranked(raw, metric, limit), tier, note: cover };
}

export function buildTable(opts: {
  scope: Scope;
  group: Group;
  metric: string;
  limit: number;
  report: MatchReport | null;
  season: Season | null;
}): TableData {
  const m = metricOf(opts.metric);
  if (m.scopes && !m.scopes.includes(opts.scope)) {
    return { rows: [], tier: "prov", note: "", refusal: `${m.label} is only defined over ${m.scopes.join(" and ")}.` };
  }
  const groupOk = GROUPS.find((g) => g.id === opts.group)?.scopes.includes(opts.scope);
  if (!groupOk) {
    return {
      rows: [],
      tier: "prov",
      note: "",
      refusal:
        opts.group === "position"
          ? "Positions come off the roster and a single match has none, so a match cannot be grouped by position."
          : "A season has no home and away side to group by — the same club is both, on alternate weeks.",
    };
  }
  return opts.scope === "season"
    ? fromSeason(opts.season, opts.group, m, opts.limit)
    : fromMatch(opts.report, opts.group, m, opts.limit);
}

/** The heading a table gets if nobody types one. */
export function defaultTitle(scope: Scope, group: Group, metric: string): string {
  const m = metricOf(metric);
  const g = group === "player" ? "" : group === "position" ? " by position" : " by team";
  return `${m.label}${g} — ${scope === "season" ? "all matches" : "this match"}`;
}
