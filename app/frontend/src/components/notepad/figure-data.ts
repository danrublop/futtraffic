import { PLAYER_STAT_COLS, PLAYER_STAT_TABS, sortForTab, statCell } from "@/lib/player-cols";
import { buildRows, playerLabel, syntheticNote, teamName, tierOf, type Tier } from "@/lib/report";
import type { MatchEvent, MatchReport, Season, StatSection } from "@/lib/types";
import { dimOf, measureOf, runColumns, sourceOf } from "./query-data";

/** The figures a note can hold, out of the match report.
 *
 *  These are the shapes a scouting report is made of that the board and the
 *  ranked table do not cover: the wide player table (pages 16-17), the
 *  home-against-away comparison tables (6 and 9), and the event timeline (2).
 *  The report SCREEN has an interactive version of each; a document gets a
 *  frozen one, and both order and format through the same helpers so the two
 *  cannot disagree about the same match.
 *
 *  EVERY FIGURE DECLARES ITS SOURCE. `players[]` is what the camera measured
 *  and is present on every finished match. `meta.facts.*` is the event feed —
 *  entered rather than computed, and absent on a match the pipeline processed
 *  on its own. A figure reads one or the other, says which, and when its source
 *  is missing it refuses with the reason instead of rendering an empty table.
 *
 *  The two are never joined into one row. A feed row is keyed by NAME and a
 *  tracked row by TRACK ID; only an identity binding could join them, and
 *  identity reaches chance a few minutes into a match. Asserting that join
 *  silently is exactly the failure this whole codebase is written around.
 */

export type FigureKind = "playerTable" | "sections" | "events" | "query";

/** A table the coach built rather than one the report ships.
 *
 *  Stored in the figure's `cols` attribute as JSON, because a built table needs
 *  five things to be rebuilt and `cols` is one string. It rides along for the
 *  same reason the report figures' `cols` does: so Refresh can re-run the same
 *  question rather than the block being a dead snapshot.
 */
export interface QuerySpec {
  source: string;
  dim: string;
  measures: string[];
  limit: number;
  /** Season sources only. Absent means every match. */
  matchIds?: string[];
}

/** Read a spec back, tolerating anything. A figure whose spec will not parse
 *  is a figure that cannot refresh — never an exception that eats the note. */
export function parseQuerySpec(cols: string): QuerySpec | null {
  try {
    const raw = JSON.parse(cols) as Record<string, unknown>;
    const measures = Array.isArray(raw.measures)
      ? raw.measures.slice(0, 8).map((m) => String(m ?? "").slice(0, 60)).filter(Boolean)
      : [];
    if (!raw.source || !raw.dim || !measures.length) return null;
    return {
      source: String(raw.source).slice(0, 40),
      dim: String(raw.dim).slice(0, 40),
      measures,
      limit: Number(raw.limit) || 20,
      ...(Array.isArray(raw.matchIds)
        ? { matchIds: raw.matchIds.slice(0, 200).map((m) => String(m ?? "")) }
        : {}),
    };
  } catch {
    return null;
  }
}

export const FIGURES: { id: FigureKind; label: string; hint: string }[] = [
  { id: "playerTable", label: "Player table", hint: "Every player, many columns" },
  { id: "sections", label: "Comparison", hint: "Home against away, one group of stats" },
  { id: "events", label: "Timeline", hint: "Goals, cards and substitutions" },
];

export interface FigureRow {
  name: string;
  /** "home" | "away" | "" — drives the side dot, nothing else. */
  side: string;
  cells: string[];
  /** Comparison rows only: home's share of the pair, 0-1, for the split bar.
   *  Absent when the row's values are not numbers to begin with. */
  split?: number;
}

export interface FigureData {
  head: string[];
  rows: FigureRow[];
  /** What this figure rests on. Rendered, always. */
  note: string;
  tier: Tier;
  /** Non-null when the ask cannot be answered — rendered instead of rows. */
  refusal?: string;
  /** The subsets on offer for this kind, for the block's second picker. */
  choices: { id: string; label: string }[];
}

const NO_FEED =
  "This match has no event feed. Passes, shots and duels need the ball, and the pipeline follows bodies.";

/** A stat cell like "618 (88%)" or 72 down to the quantity in front of it, for
 *  the split bar. Null when there is no quantity to read.
 *
 *  The number has to be the WHOLE leading token — followed by the end of the
 *  string, a space, or an opening bracket. A bare `^\d+` would read "4-2-3-1"
 *  as 4 and "4-4-2" as 4, and draw a confident 50/50 bar under a row that says
 *  "Formation". A bar off a misparse is worse than no bar: this one would read
 *  as possession. */
function lead(v: string | number | undefined): number | null {
  if (typeof v === "number") return Number.isFinite(v) ? v : null;
  const m = /^-?\d+(?:\.\d+)?(?=$|[\s(])/.exec(String(v ?? "").trim());
  return m ? Number(m[0]) : null;
}

// ---------------------------------------------------------------- player table

const PLAYER_SETS = [
  { id: "physical", label: "Physical", source: "tracks" as const },
  ...PLAYER_STAT_TABS.map((t) => ({ id: t.id, label: t.label, source: "feed" as const })),
];

/** Kept exported under its old name: the block's picker and its tests were
 *  written against it before there was more than one figure kind. */
export const COL_SETS = PLAYER_SETS;
export const colSet = (id: string) => PLAYER_SETS.find((c) => c.id === id) ?? PLAYER_SETS[0];

const PHYSICAL_HEAD = ["Player", "Mins", "Distance", "km / 90", "Top speed", "Sprints"];
const setChoices = PLAYER_SETS.map(({ id, label }) => ({ id, label }));

/** Measured columns, off the tracks. Per 90 sits beside the total for the
 *  reason report.ts already gives: a total ranks whoever played longest. */
function fromTracks(report: MatchReport, limit: number): FigureData {
  const tier = tierOf("players.distance_km", report.fields).tier;
  const rows = buildRows(report)
    .sort((a, b) => b.minutes - a.minutes)
    .slice(0, limit)
    .map((r) => ({
      // No invented names: the tracker reads a shirt number or nothing, and
      // `playerLabel` is the one place that decides how to say so.
      name: playerLabel(r.p, report),
      side: r.p.team === "unknown" ? "" : r.p.team,
      cells: [
        String(Math.round(r.minutes)),
        r.distance == null ? "refused" : `${r.distance.toFixed(2)} km`,
        r.per90 ? `${r.per90.toFixed(2)} km` : "—",
        r.peak ? `${r.peak.toFixed(1)} km/h` : "—",
        r.sprints == null ? "withheld" : String(r.sprints),
      ],
    }));

  if (!rows.length) {
    return { head: PHYSICAL_HEAD, rows: [], tier, note: "", choices: setChoices, refusal: "This match has no tracked players yet." };
  }
  return {
    head: PHYSICAL_HEAD,
    rows,
    tier,
    choices: setChoices,
    note: "Measured from the tracks. Shirt numbers, not names — the tracker reads a number or nothing.",
  };
}

/** Event columns, off the entered feed. Everything here is `tagged` in the
 *  pipeline's own vocabulary: somebody typed it, the camera did not see it. */
function fromFeed(report: MatchReport, setId: string, limit: number): FigureData {
  const tab = PLAYER_STAT_TABS.find((t) => t.id === setId) ?? PLAYER_STAT_TABS[0];
  const head = ["Player", ...tab.cols.map((c) => PLAYER_STAT_COLS[c] ?? c)];
  const feed = report.meta?.facts?.player_stats ?? [];

  if (!feed.length) {
    return {
      head,
      rows: [],
      tier: "prov",
      note: "",
      choices: setChoices,
      refusal: `${NO_FEED} Switch to Physical for what was measured.`,
    };
  }

  const rows = sortForTab(feed, tab, false)
    .slice(0, limit)
    .map((p) => ({ name: p.name, side: p.side, cells: tab.cols.map((c) => statCell(p, c)) }));

  if (!rows.length) {
    return { head, rows: [], tier: "prov", note: "", choices: setChoices, refusal: `Nobody in this feed has ${tab.label.toLowerCase()} recorded.` };
  }
  return {
    head,
    rows,
    tier: "prov",
    choices: setChoices,
    note: "Entered, not measured — these come from the match feed, not from the camera.",
  };
}

// ------------------------------------------------------------------ comparison

export const sectionId = (title: string) =>
  title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

/** Home against away, one group at a time. Pages 6 and 9, and the header
 *  comparison on page 2 — the most repeated shape in the whole report.
 *
 *  One group per block rather than all seven in one, because a coach puts
 *  Passes under their passing paragraph, not a wall of every stat the feed
 *  carries under a heading that covers none of them. */
function fromSections(report: MatchReport, which: string): FigureData {
  const facts = report.meta?.facts;
  const sections: StatSection[] = [...(facts?.sections ?? [])];
  // `top_stats` is the same label/home/away shape under a different key, so it
  // joins the picker as one more group rather than needing its own figure —
  // but ONLY when `sections` does not already carry a group by that name. The
  // demo feed has both, and two chips with the same label and the same slug
  // means the second is unreachable: `find` on the id always returns the first.
  if (facts?.top_stats?.length && !sections.some((x) => sectionId(x.title) === "top-stats")) {
    sections.unshift({
      title: "Top stats",
      rows: facts.top_stats.map(({ label, home, away }) => ({ label, home, away })),
    });
  }
  const choices = sections.map((s) => ({ id: sectionId(s.title), label: s.title }));
  const head = ["", teamName(report, "home"), teamName(report, "away")];

  if (!sections.length) {
    return { head, rows: [], tier: "prov", note: "", choices, refusal: NO_FEED };
  }
  const s = sections.find((x) => sectionId(x.title) === which) ?? sections[0];
  const rows = s.rows.map((r) => {
    const h = lead(r.home);
    const a = lead(r.away);
    const total = (h ?? 0) + (a ?? 0);
    return {
      name: r.label,
      side: "",
      cells: [String(r.home ?? "—"), String(r.away ?? "—")],
      ...(h != null && a != null && total > 0 ? { split: h / total } : {}),
    };
  });
  return {
    head,
    rows,
    tier: "prov",
    choices,
    note: `${s.title} · entered, not measured — these come from the match feed, not from the camera.`,
  };
}

// -------------------------------------------------------------------- timeline

/** One line of the feed, in the words a reader uses. The pipeline cannot see
 *  any of this — it follows bodies; its ball rows exist but are pixel centres and
 *  unused — so every line is
 *  something a human recorded. */
function eventText(e: MatchEvent): string {
  if (e.type === "goal") {
    return [e.player, e.score ? `(${e.score})` : "", e.assist ? `assist ${e.assist}` : ""]
      .filter(Boolean)
      .join(" ");
  }
  if (e.type === "substitution") return `${e.in ?? "?"} for ${e.out ?? "?"}`;
  if (e.type === "yellow_card") return `${e.player ?? ""} booked`.trim();
  return e.score ? `${e.score}` : "";
}

const EVENT_LABEL: Record<string, string> = {
  goal: "Goal",
  yellow_card: "Yellow",
  substitution: "Sub",
  period: "Period",
};

function fromEvents(report: MatchReport, which: string): FigureData {
  const all = report.meta?.facts?.events ?? [];
  const choices = [
    { id: "all", label: "Everything" },
    { id: "goal", label: "Goals" },
    { id: "yellow_card", label: "Cards" },
    { id: "substitution", label: "Subs" },
  ];
  const head = ["Min", "", "Event"];

  if (!all.length) {
    return { head, rows: [], tier: "prov", note: "", choices, refusal: NO_FEED };
  }
  // A period marker is the spine of a timeline, so it survives every filter
  // except when the reader asked for one specific kind of incident.
  const rows = all
    .filter((e) => which === "all" || e.type === which)
    .map((e) => ({
      name: e.minute,
      side: e.side ?? "",
      cells: [EVENT_LABEL[e.type] ?? e.type, eventText(e)],
    }));

  if (!rows.length) {
    const label = choices.find((c) => c.id === which)?.label.toLowerCase() ?? which;
    return { head, rows: [], tier: "prov", note: "", choices, refusal: `No ${label} in this match.` };
  }
  return {
    head,
    rows,
    tier: "prov",
    choices,
    note: "Entered, not measured — the ball the pipeline detects is not used to find plays.",
  };
}

// ------------------------------------------------------------------------ api

export function buildFigure(opts: {
  report: MatchReport | null;
  kind?: string;
  cols: string;
  limit: number;
  /** Only a built table with a season source needs this; the report figures
   *  are all one match and ignore it. */
  season?: Season | null;
}): FigureData {
  if (opts.kind === "query") return fromQuery(opts.cols, opts.report, opts.season ?? null);
  const kind = (FIGURES.find((f) => f.id === opts.kind)?.id ?? "playerTable") as FigureKind;
  if (!opts.report) {
    return { head: [], rows: [], tier: "prov", note: "", choices: [], refusal: "No match is open, so there is nothing to show." };
  }
  const report = opts.report;
  const said = (f: FigureData): FigureData => {
    const extra = syntheticNote(report);
    return extra && f.note.indexOf(extra) < 0 ? { ...f, note: [extra, f.note].filter(Boolean).join(" ") } : f;
  };
  if (kind === "sections") return said(fromSections(report, opts.cols));
  if (kind === "events") return said(fromEvents(report, opts.cols));
  const set = colSet(opts.cols);
  return said(
    set.source === "tracks" ? fromTracks(report, opts.limit) : fromFeed(report, set.id, opts.limit),
  );
}

/** A built table, through the same runner the chart beside it uses. */
function fromQuery(cols: string, report: MatchReport | null, season: Season | null): FigureData {
  const spec = parseQuerySpec(cols);
  if (!spec) {
    return {
      head: [], rows: [], tier: "prov", note: "", choices: [],
      refusal: "This table's question could not be read back, so it cannot be refreshed.",
    };
  }
  const r = runColumns({ report, season, ...spec });
  return {
    head: r.head,
    rows: r.rows.map((x) => ({ name: x.label, side: x.side, cells: x.text })),
    note: r.note,
    tier: r.tier,
    choices: [],
    ...(r.refusal ? { refusal: r.refusal } : {}),
  };
}

/** The default subset when a figure kind is first inserted, or when the kind
 *  changes and the old subset means nothing to the new one. */
export function firstChoice(kind: string, report: MatchReport | null): string {
  if (kind === "events") return "all";
  if (kind === "sections") {
    return buildFigure({ report, kind, cols: "", limit: 1 }).choices[0]?.id ?? "top-stats";
  }
  return "physical";
}

export function figureTitle(kind: string, cols: string, report: MatchReport | null): string {
  if (kind === "query") return queryFigureTitle(cols, report);
  if (kind === "events") return "Timeline";
  if (kind === "sections") {
    const c = buildFigure({ report, kind, cols, limit: 1 }).choices.find((x) => x.id === cols);
    return c ? `${c.label} — home against away` : "Home against away";
  }
  return `Player stats — ${colSet(cols).label.toLowerCase()}`;
}

/** "Distance and Top speed by player". Named after what was asked for, so a
 *  coach scanning a note knows what a table is without reading its numbers. */
export function queryFigureTitle(cols: string, report: MatchReport | null): string {
  const spec = parseQuerySpec(cols);
  if (!spec) return "Table";
  const src = sourceOf(spec.source, report);
  const names = spec.measures.map((id) => measureOf(src, id).label);
  const list =
    names.length <= 1
      ? names[0]
      : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
  return `${list} by ${dimOf(src, spec.dim).label.toLowerCase()}`;
}
