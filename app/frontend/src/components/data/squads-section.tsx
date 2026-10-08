"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ArrowLeftRight, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";

import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHeader, TableRow } from "@/components/ui/table";
import {
  BallIcon,
  BAND_ROW,
  BootIcon,
  Head,
  Per90Toggle,
  pickable,
  Pill,
  Portrait,
  TableSkeleton,
} from "@/components/data/players-section";
import { RatingChip } from "@/components/rating-chip";
import { MomentumChart } from "@/components/momentum-chart";
import { cmp, useSort } from "@/components/sort-head";
import { api } from "@/lib/api";
import { useOrg } from "@/lib/club";
import { scoreOf, type SeasonTable } from "@/lib/season";
import type { Appearance, Coach, LineupRow, MatchInfo, MatchSide, TeamStat } from "@/lib/types";
import { DATA_TABLE } from "@/components/data/section-shell";
import { longDate } from "@/components/data/player-fact";
import { cn } from "@/lib/utils";

type Fixture = SeasonTable["fixtures"][number];
type Stats = Record<string, Record<string, TeamStat>>;

/** THE REPORT'S TEAMS STATS, regrouped. Most labels are the feed's own and
 *  the store keys each on its label run through metric_id (lowercase,
 *  anything else a hyphen), so the label is the lookup as well as the header.
 *
 *  FIVE ARE WORKED OUT per match rather than read (see `stat`): goals and
 *  goals conceded from the score, clean sheets from the goals conceded, and
 *  xG conceded / xG difference from the other side's xG. Over a season they
 *  total, or average under Per 90. */
const SECTIONS = [
  { id: "top", label: "Top stats", rows: ["Goals", "Goals conceded", "Clean sheets", "Ball possession", "Expected goals (xG)", "xG conceded", "Accurate passes", "Yellow cards"] },
  { id: "attack", label: "Attack", rows: ["Expected goals (xG)", "xG difference", "Shots on target", "Big chances", "Big chances missed", "Accurate passes", "Accurate long balls", "Accurate crosses", "Touches in opposition box", "Corners"] },
  // The xG breakdown lives here now; its headline is in Attack, and its own
  // pill was the room the Total / Per 90 switch needed on this row.
  { id: "shots", label: "Shots", rows: ["Total shots", "Shots on target", "Shots off target", "Blocked shots", "Hit woodwork", "Shots inside box", "Shots outside box", "xG open play", "xG set play", "xG non-penalty", "xG on target (xGOT)"] },
  { id: "passes", label: "Passes", rows: ["Passes", "Accurate passes", "Own half", "Opposition half", "Accurate long balls", "Accurate crosses", "Throws", "Offsides"] },
  // "Possession won final third" was asked for here and the feed does not
  // carry it - not in the stored team stats, not in the raw match JSON.
  { id: "defence", label: "Defence", rows: ["xG conceded", "Interceptions", "Clearances", "Keeper saves", "Tackles", "Blocks"] },
  { id: "duels", label: "Duels", rows: ["Duels won", "Ground duels won", "Aerial duels won", "Successful dribbles"] },
  { id: "discipline", label: "Discipline", rows: ["Yellow cards", "Red cards", "Fouls committed"] },
  { id: "physical", label: "Physical", rows: ["Distance covered", "Sprinting distance", "Number of sprints"] },
] as const;

/** app/server/db.py metric_id, the same rule. */
const metricId = (label: string) => label.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

/** A stat as the report prints it: the feed's own text where it has one
 *  ("343 (81%)", "52%"), metres as km, and xG to two places. */
function show(label: string, v: TeamStat | undefined): string {
  if (!v) return "—";
  if (v.raw) return v.raw;
  if (v.value == null) return "—";
  if (/distance/i.test(label)) return `${(v.value / 1000).toFixed(1)} km`;
  if (label === "xG difference") return `${v.value > 0 ? "+" : ""}${v.value.toFixed(2)}`;
  return Number.isInteger(v.value) ? String(v.value) : v.value.toFixed(2);
}

/** Enter or Space on a div that acts as a button. Space is preventDefault'd:
 *  without it the page scrolls behind the panel the press just opened. */
const activate = (go: () => void) => (ev: React.KeyboardEvent) => {
  if (ev.key !== "Enter" && ev.key !== " ") return;
  ev.preventDefault();
  go();
};

/** The result from our side, off a "home - away" score. */
function resultOf(score: string | null | undefined, home: boolean): "W" | "D" | "L" | null {
  const played = scoreOf(score);
  if (!played) return null;
  const [h, a] = played;
  return h === a ? "D" : h > a === home ? "W" : "L";
}

/** A stat's text in its cell: a share that rides on a count ("343 (81%)")
 *  goes on a small second line, so the column is as wide as the number and
 *  ten columns still fit. */
function StatText({ text }: { text: string }) {
  const [main, pct] = text.split(/ (?=\()/);
  return (
    <>
      {main}
      {pct && <span className="text-muted-foreground block text-[10px] leading-tight">{pct}</span>}
    </>
  );
}

function ResultBadge({ res }: { res: "W" | "D" | "L" | null }) {
  if (!res) return null;
  return (
    <span
      className={cn(
        "grid size-5 shrink-0 place-items-center rounded-[4px] text-[11px] font-semibold",
        res === "W" && "bg-[color-mix(in_oklab,var(--chart-5)_20%,transparent)] text-[var(--chart-5)]",
        res === "D" && "bg-[var(--card-hover)] text-[var(--mute)]",
        res === "L" && "bg-[color-mix(in_oklab,var(--bad)_20%,transparent)] text-[var(--bad)]",
      )}
    >
      {res}
    </span>
  );
}

/** One stat over several matches. A share that is ONLY a share (possession,
 *  "52%") is averaged, never added; a count that carries one ("343 (81%)")
 *  is added and its share averaged; distance reads in km.
 *
 *  PER 90 divides by the matches that have the stat. For a team that IS per
 *  90: a side plays every minute of every match. What it does not do is
 *  count stoppage time - a 98-minute match counts as 90, because the store
 *  does not keep a match's length. */
export function aggregate(label: string, vs: TeamStat[], per90: boolean): { text: string; num: number } | null {
  const nums = vs.filter((v) => v.value != null);
  if (!nums.length) return null;
  const total = nums.reduce((t, v) => t + (v.value as number), 0);
  const pcts = nums.map((v) => v.pct).filter((p): p is number => p != null);
  const pct = pcts.length ? pcts.reduce((a, b) => a + b, 0) / pcts.length : null;
  if (nums.every((v) => /^\s*[\d.]+%\s*$/.test(v.raw ?? ""))) {
    return { text: `${(total / nums.length).toFixed(0)}%`, num: total / nums.length };
  }
  const n = per90 ? total / nums.length : total;
  const fixed = /distance/i.test(label)
    ? `${(n / 1000).toFixed(1)} km`
    : /xg/i.test(label)
      ? `${label === "xG difference" && n > 0 ? "+" : ""}${n.toFixed(2)}`
      : per90 ? n.toFixed(1) : String(Math.round(n));
  return { text: pct != null ? `${fixed} (${pct.toFixed(0)}%)` : fixed, num: n };
}

/** Goals for and against, from ONE side of a "home - away" score, or null if
 *  it is not a played score.
 *
 *  `opponents` flips the side: the Opponents table reads each match from THEIR
 *  end, so their goals are the ones we conceded. Getting that flip wrong does
 *  not break anything - it swaps two believable columns and turns every clean
 *  sheet into the other team's - which is the acting-team-frame class of bug
 *  in an internal note. Pure and exported so the flip has a test. */
export function goalsFor(
  score: string | null | undefined,
  home: boolean,
  opponents: boolean,
): { scored: number; conceded: number } | null {
  const played = scoreOf(score);
  if (!played) return null;
  const [h, a] = played;
  const mineHome = opponents ? !home : home;
  return { scored: mineHome ? h : a, conceded: mineHome ? a : h };
}

/** THE TEAM STATS TABLE, two ways round - one per tab.
 *
 *  SQUAD (the Squads tab) is the Players table's shape: a row per squad, the
 *  report's Teams stats across under its section pills, over whatever the
 *  page's Season and Match dropdowns select - one match as the report prints
 *  it, a season as totals or per 90.
 *
 *  MATCHES (the All matches tab) is the other orientation: a row per game,
 *  the same columns, for setting one game against another.
 *
 *  OPPONENTS (the Opponents tab) is SQUAD for the other side: a row per club
 *  we played, ITS side of each match's stats, its record against us.
 *
 *  Either way the profile beside it is a team sheet and the squad's record,
 *  and only the table scrolls. */
export function SquadMatches({
  view,
  menu,
  onPlayer,
  teamId,
  season,
  matchId,
  fixtures,
}: {
  view: "squad" | "matches" | "opponents";
  /** The club's "..." menu, for the squad panel's top-right corner. */
  menu?: React.ReactNode;
  /** A player clicked on the team sheet, over to Players - with the match
   *  when there is one. */
  onPlayer?: (id: string, matchId?: string) => void;
  teamId?: string;
  season?: string;
  matchId?: string;
  fixtures: Fixture[];
}) {
  const { org } = useOrg();
  const club = org?.club.club_id ?? "";
  const [sides, setSides] = useState<MatchSide[] | null>(null);
  const [stats, setStats] = useState<Stats>({});
  const [oppStats, setOppStats] = useState<Stats>({});
  const [appear, setAppear] = useState<Appearance[]>([]);
  const [photos, setPhotos] = useState<Map<string, boolean>>(new Map());
  const [error, setError] = useState<string | null>(null);
  const [cat, setCat] = useState<string>(SECTIONS[0].id);
  const [per90, setPer90] = useState(false);
  const [picked, setPicked] = useState<string | null>(null);
  const [pickedSquad, setPickedSquad] = useState<string | null>(null);
  const sort = useSort("date");
  const squadSort = useSort("name");

  useEffect(() => {
    let live = true;
    api.getMatchSides().then(
      (r) => live && setSides(r),
      (e: unknown) => live && setError(e instanceof Error ? e.message : String(e)),
    );
    // Every player's numbers per match, for the team sheet's badges.
    api.getAppearances().then(
      (r) => live && setAppear(r),
      () => {},
    );
    // Only for the faces on the team sheet: which players have a photo.
    api.getPlayers().then(
      (ps) => live && setPhotos(new Map(ps.map((p) => [p.id, p.photo !== false]))),
      () => {},
    );
    return () => {
      live = false;
    };
  }, []);

  useEffect(() => {
    if (!club) return;
    let live = true;
    const fail = (e: unknown) => live && setError(e instanceof Error ? e.message : String(e));
    api.getTeamStats(club).then((s) => live && setStats(s), fail);
    // Always: xG conceded and xG difference read the other side's xG.
    api.getOpponentStats(club).then((s) => live && setOppStats(s), fail);
    return () => {
      live = false;
    };
  }, [club]);

  /** Our side in each match: the one our club played on. */
  const ours = useMemo(
    () => new Map((sides ?? []).filter((s) => s.club_id === club).map((s) => [s.match_id, s])),
    [sides, club],
  );
  /** And theirs: the other side of each of our matches. */
  const theirs = useMemo(
    () =>
      new Map(
        (sides ?? []).filter((s) => s.club_id !== club && ours.has(s.match_id)).map((s) => [s.match_id, s]),
      ),
    [sides, club, ours],
  );
  const opponents = view === "opponents";
  /** Whose stats the table reads - ours, or the opponent's side - and the
   *  side facing them. */
  const src = opponents ? oppStats : stats;
  const other = opponents ? stats : oppStats;

  /** Every match in scope, newest first - the fixtures arrive that way. */
  const rows = useMemo(
    () =>
      fixtures
        .filter((f) => {
          const s = ours.get(f.id);
          return (
            s &&
            (!teamId || s.team_id === teamId) &&
            (!season || f.competition === season) &&
            (!matchId || f.id === matchId)
          );
        })
        .map((f) => {
          const s = ours.get(f.id)!;
          const home = s.side === "home";
          return {
            f,
            side: s,
            home,
            opp: home ? f.away : f.home,
            res: resultOf(f.score, home),
            day: f.date?.slice(0, 10) ?? null,
          };
        }),
    [fixtures, ours, teamId, season, matchId],
  );
  type Row = (typeof rows)[number];

  /** One stat for one match, from the table's side. The DERIVED five are
   *  worked out here; everything else is read from the store. */
  const stat = (r: Row, label: string): TeamStat | undefined => {
    const n = (value: number): TeamStat => ({ value, raw: null, pct: null });
    const g = goalsFor(r.f.score, r.home, opponents);
    const xg = (s: Stats) => s[r.f.id]?.["expected-goals-xg"]?.value;
    switch (label) {
      case "Goals":
        return g ? n(g.scored) : undefined;
      case "Goals conceded":
        return g ? n(g.conceded) : undefined;
      case "Clean sheets":
        return g ? n(g.conceded === 0 ? 1 : 0) : undefined;
      case "xG conceded": {
        const v = xg(other);
        return v == null ? undefined : n(v);
      }
      case "xG difference": {
        const us = xg(src);
        const them = xg(other);
        return us == null || them == null ? undefined : n(us - them);
      }
      default:
        return src[r.f.id]?.[metricId(label)];
    }
  };

  /** The squads - or the opponents - with a match in scope, each with its
   *  matches. */
  const squads = useMemo(() => {
    const by = new Map<string, Row[]>();
    for (const r of rows) {
      const k = (opponents ? theirs.get(r.f.id)?.club_id : r.side.team_id) ?? "";
      const at = by.get(k);
      if (at) at.push(r);
      else by.set(k, [r]);
    }
    return [...by].map(([id, rs]) => {
      const team = opponents ? null : (org?.teams.find((t) => t.team_id === id) ?? null);
      return { id, team, name: opponents ? rs[0].opp : (team?.name ?? "Unassigned"), rs };
    });
  }, [rows, org, opponents, theirs]);
  type Squad = (typeof squads)[number];

  const section = SECTIONS.find((s) => s.id === cat) ?? SECTIONS[0];
  // With no rows there is nothing to filter on, so the header shows the
  // section's full label set instead of collapsing to the name column.
  const cols = rows.length ? section.rows.filter((l) => rows.some((r) => stat(r, l))) : [...section.rows];
  const single = rows.length === 1;
  // The opponent's record is from THEIR side: our win is their loss.
  const persp = (r: "W" | "D" | "L" | null) => (opponents ? (r === "W" ? "L" : r === "L" ? "W" : r) : r);
  const squadStat = (q: Squad, l: string) => {
    const vs = q.rs.map((r) => stat(r, l)).filter((v): v is TeamStat => !!v);
    if (single) return { text: show(l, vs[0]), num: vs[0]?.value ?? -Infinity };
    return aggregate(l, vs, per90);
  };

  const sorted = useMemo(() => {
    const num = (r: Row) =>
      sort.key === "date"
        ? Date.parse(r.f.date ?? "") || 0
        : (stat(r, sort.key)?.value ?? -Infinity);
    return [...rows].sort((a, b) => cmp(num(a), num(b), sort.asc));
    // eslint-disable-next-line react-hooks/exhaustive-deps -- stat reads src/other, listed
  }, [rows, src, other, sort.key, sort.asc]);

  // Names read A to Z first: useSort starts a new key descending, which is
  // right for every number here and backwards for a name.
  const squadsSorted = [...squads].sort((a, b) =>
    squadSort.key === "name"
      ? a.name.localeCompare(b.name) * (squadSort.asc ? -1 : 1)
      : cmp(squadStat(a, squadSort.key)?.num ?? -Infinity, squadStat(b, squadSort.key)?.num ?? -Infinity, squadSort.asc),
  );

  // WHAT THE PROFILE SHOWS. Match v match: the match clicked. Squad: that
  // squad over the scope, its team sheet the latest match in it.
  const squad = squads.find((q) => q.id === pickedSquad) ?? squadsSorted[0] ?? null;
  const selected =
    view === "matches"
      ? (rows.find((r) => r.f.id === picked) ?? sorted[0] ?? null)
      : (squad?.rs[0] ?? null);
  const team = org?.teams.find((t) => t.team_id === (selected?.side.team_id ?? teamId)) ?? null;
  const scopeRows = view === "matches" ? rows : (squad?.rs ?? []);

  // THE TEAM SHEET'S NUMBERS follow the table's scope: one match - the one
  // clicked in All matches, or the only one in scope - reads as that match;
  // a season averages the rating and totals goals and assists, per 90 when
  // the switch says so.
  const seasonal = view !== "matches" && !single;
  const sheetIds = seasonal ? scopeRows.map((r) => r.f.id) : selected ? [selected.f.id] : [];
  const sheetKey = sheetIds.join(",");
  const numbers = useMemo(() => {
    const ids = new Set(sheetKey.split(","));
    const out = new Map<string, PlayerNums>();
    for (const a of appear) {
      if (!ids.has(a.match_id)) continue;
      const t = out.get(a.player_id) ?? { ratings: [], goals: 0, assists: 0, mins: 0 };
      if (a.rating != null) t.ratings.push(a.rating);
      t.goals += a.goals ?? 0;
      t.assists += a.assists ?? 0;
      t.mins += a.minutes ?? 0;
      out.set(a.player_id, t);
    }
    return out;
  }, [appear, sheetKey]);

  // THE TOP THREES for the Info tab: this team's players only - the side it
  // played on in each match of the scope - by average rating, goals and
  // assists. Totals, whatever the switch says: a per-90 table of scorers is
  // topped by whoever scored in his only ten minutes.
  const tops = useMemo(() => {
    const sideOf = new Map(
      scopeRows.map((r) => [r.f.id, opponents ? (theirs.get(r.f.id)?.side ?? null) : r.side.side]),
    );
    const by = new Map<string, { id: string; name: string } & PlayerNums>();
    for (const a of appear) {
      if (!sideOf.has(a.match_id) || sideOf.get(a.match_id) !== a.side) continue;
      const t = by.get(a.player_id) ?? { id: a.player_id, name: a.name, ratings: [], goals: 0, assists: 0, mins: 0 };
      if (a.rating != null) t.ratings.push(a.rating);
      t.goals += a.goals ?? 0;
      t.assists += a.assists ?? 0;
      t.mins += a.minutes ?? 0;
      by.set(a.player_id, t);
    }
    const all = [...by.values()];
    const top = (f: (p: (typeof all)[number]) => number | null, fmt: (v: number) => string) =>
      all
        .map((p) => ({ p, v: f(p) }))
        .filter((x): x is { p: (typeof all)[number]; v: number } => x.v != null && x.v > 0)
        .sort((a, b) => b.v - a.v)
        .slice(0, 3)
        .map(({ p, v }) => ({ id: p.id, name: p.name, value: fmt(v) }));
    return {
      rated: top((p) => avgRating(p), (v) => v.toFixed(2)),
      scorers: top((p) => p.goals, String),
      assists: top((p) => p.assists, String),
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- scopeRows is keyed by sheetKey
  }, [appear, sheetKey, opponents, theirs]);

  const table =
    // Padding only. The header's size and colour come from DATA_HEAD, so this
    // table cannot disagree with Players about what a first row looks like.
    cn(DATA_TABLE, "[&_thead_th]:px-[3px] [&_tbody_td]:px-[3px] [&_tbody_td]:text-[13px]");
  const statHead = (i: number) =>
    cn("border-l border-[var(--grid-line)] text-center leading-tight whitespace-normal", i === cols.length - 1 && "!pr-3");
  const statCell = (i: number) =>
    cn("readout border-l border-[var(--grid-line)] text-center", i === cols.length - 1 && "!pr-3");

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <div className="flex min-w-0 flex-col gap-3 @min-[915px]:min-h-0 @min-[915px]:flex-1 @min-[915px]:flex-row">
        <Card className="min-w-0 flex-1 gap-0 py-0 @min-[915px]:min-h-0">
          {/* BAND_ROW, as the Players table's category row wears it: this row
              and the panel's name band beside it start on the same line, so
              they have to END on the same line too. Sized by its own content
              it came out 58 against the band's 57 and the two rules missed by
              a pixel. */}
          <div className={cn(BAND_ROW, "flex flex-wrap items-center gap-1 px-3 py-2")}>
            {SECTIONS.map((s) => (
              <Pill
                key={s.id}
                on={cat === s.id}
                onClick={() => {
                  setCat(s.id);
                  // A section without the sorted column hands the sort back.
                  const has = (k: string) => (s.rows as readonly string[]).includes(k);
                  if (sort.key !== "date" && !has(sort.key)) sort.on("date")();
                  if (squadSort.key !== "name" && !has(squadSort.key)) squadSort.on("name")();
                }}
              >
                {s.label}
              </Pill>
            ))}
            {/* Over a season: totals or per 90, as ONE button that says what
                it is showing and flips on a click - a two-part switch took
                the width of three pills and pushed itself onto a second line. */}
            {view !== "matches" && !single && (
              <Per90Toggle on={per90} onToggle={() => setPer90((v) => !v)} />
            )}
          </div>
          <CardContent className="overflow-x-auto p-0 @min-[915px]:min-h-0 @min-[915px]:flex-1 @min-[915px]:overflow-y-auto [&_[data-slot=table-container]]:overflow-visible">
            {error && <p className="text-muted-foreground p-4 text-sm">Could not load the squad&apos;s matches - {error}</p>}
            {!error && sides === null && <TableSkeleton rows={view === "squad" ? 1 : 10} />}
            {/* The header stays when there are no rows, like every data table. */}
            {!error && sides !== null && view !== "matches" && (
              <Table className={table}>
                <TableHeader>
                  <TableRow>
                    <Head k="name" sort={squadSort} className="w-px !pl-3">
                      {opponents ? "Opponent" : "Squad"}
                    </Head>
                    {cols.map((l, i) => (
                      <Head key={l} k={l} sort={squadSort} className={statHead(i)}>
                        {l}
                      </Head>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {squadsSorted.map((q) => {
                    const on = q.id === squad?.id;
                    const tally = (x: "W" | "D" | "L") => q.rs.filter((r) => persp(r.res) === x).length;
                    return (
                      <TableRow
                        key={q.id}
                        {...pickable(on, () => setPickedSquad(q.id))}
                      >
                        <TableCell className="w-px max-w-[140px] border-b border-[var(--grid-line)] !pl-3 whitespace-nowrap">
                          <span className="block truncate text-[13px] text-[var(--ink)]" title={q.name}>
                            {q.name}
                          </span>
                          <span className="text-muted-foreground block truncate text-xs">
                            {single
                              ? `v ${opponents ? (org?.club.name ?? "us") : q.rs[0].opp} · ${q.rs[0].f.score?.replace(/\s+/g, "")}`
                              : `${q.rs.length} ${q.rs.length === 1 ? "game" : "games"} · ${tally("W")}-${tally("D")}-${tally("L")}`}
                          </span>
                        </TableCell>
                        {cols.map((l, i) => (
                          <TableCell key={l} className={statCell(i)}>
                            <StatText text={squadStat(q, l)?.text ?? "—"} />
                          </TableCell>
                        ))}
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            )}
            {!error && sides !== null && view === "matches" && (
              <Table className={table}>
                <TableHeader>
                  <TableRow>
                    <Head k="date" sort={sort} className="w-px !pl-3">
                      Match
                    </Head>
                    {cols.map((l, i) => (
                      <Head key={l} k={l} sort={sort} className={statHead(i)}>
                        {l}
                      </Head>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sorted.map((r) => {
                    const on = r.f.id === selected?.f.id;
                    return (
                      <TableRow
                        key={r.f.id}
                        {...pickable(on, () => setPicked(r.f.id))}
                      >
                        <TableCell className="w-px max-w-[140px] border-b border-[var(--grid-line)] !pl-3 whitespace-nowrap">
                          <span className="block truncate text-[13px] text-[var(--ink)]" title={r.opp}>
                            {r.opp} <span className="text-muted-foreground">({r.home ? "H" : "A"})</span>
                          </span>
                          <span className="text-muted-foreground mt-0.5 flex items-center gap-1.5 text-xs">
                            <ResultBadge res={r.res} />
                            <span className="readout">{r.f.score?.replace(/\s+/g, "")}</span>
                            {/* No year: the Season filter already says it. */}
                            <span>· {longDate(r.day)?.replace(/, \d{4}$/, "")}</span>
                          </span>
                        </TableCell>
                        {cols.map((l, i) => (
                          <TableCell key={l} className={statCell(i)}>
                            <StatText text={show(l, stat(r, l))} />
                          </TableCell>
                        ))}
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            )}
            {!error && sides !== null && !rows.length && (
              <p className="text-muted-foreground p-4 text-sm">No matches for this squad in this season.</p>
            )}
          </CardContent>
        </Card>

        {/* ALWAYS OPEN once the table has loaded, like the player panel: with
            nothing selected it shows its fields with no data in them. */}
        {!error && sides !== null && (
          <SquadProfile
            key={`${view}-${selected?.f.id ?? "none"}`}
            // For an opponent, the same match seen from their side: their
            // team sheet, their result, us as the opponent.
            row={
              selected && opponents
                ? {
                    ...selected,
                    side: theirs.get(selected.f.id) ?? selected.side,
                    home: !selected.home,
                    opp: org?.club.name ?? "us",
                    res: persp(selected.res),
                  }
                : selected
            }
            name={opponents ? (squad?.name ?? selected?.opp ?? "Opponent") : (team?.name ?? "Squad")}
            // Name, season and competition are already in the header.
            facts={
              opponents
                ? [["Played us", String(scopeRows.length)]]
                : [
                    ["Age group", team?.age_band || "Senior"],
                    ["Level", team?.level || "—"],
                  ]
            }
            staff={
              opponents
                ? null
                : (org?.coaches ?? []).filter(
                    (c) => team && (team.coach_ids?.includes(c.coach_id) || c.team_ids?.includes(team.team_id)),
                  )
            }
            tops={tops}
            // All matches: Info is the match itself, not the table's numbers.
            match={
              view === "matches"
                ? { us: selected?.side ?? null, them: (selected && theirs.get(selected.f.id)) ?? null }
                : null
            }
            photos={photos}
            // Only our own players open - the Players table is the club's -
            // and from a single match, filtered to it.
            onPlayer={
              onPlayer && !opponents
                ? (id: string) => onPlayer(id, view === "matches" ? selected?.f.id : undefined)
                : undefined
            }
            numbers={numbers}
            seasonal={seasonal}
            per90={seasonal && per90}
            menu={menu}
            scope={
              view !== "matches" && !single
                ? `${season ?? "Every season"} · ${scopeRows.length} ${scopeRows.length === 1 ? "game" : "games"}${opponents ? " v us" : ""}`
                : null
            }
            // THE SCOPE IS A HEADING, not a subtitle. It says which season and
            // how many games the numbers below cover, which is the one thing
            // "Season stats" did not say - so it replaces that title rather
            // than sitting twice on the same card.
            record={scopeRows.map((r) => ({
              res: persp(r.res),
              score: r.f.score,
              home: opponents ? !r.home : r.home,
              opp: opponents ? (org?.club.name ?? "Us") : r.opp,
              day: r.day,
            }))}
          />
        )}
      </div>
    </div>
  );
}


/** THE PROFILE: the selected match's team sheet, and the squad itself. */
function SquadProfile({
  row,
  name,
  facts,
  staff,
  tops,
  match,
  photos,
  onPlayer,
  numbers,
  seasonal,
  per90,
  menu,
  scope,
  record,
}: {
  /** A season line for the header instead of one match's; the team sheet is
   *  then the latest match in it. */
  scope: string | null;
  /** Null when there is nothing to show: the panel keeps its fields, empty. */
  row: {
    f: Fixture;
    side: MatchSide;
    home: boolean;
    opp: string;
    res: "W" | "D" | "L" | null;
    day: string | null;
  } | null;
  /** The header: the squad's name, or the opponent's. */
  name: string;
  facts: [string, string][];
  /** The squad's staff, or null where there is no such thing (an opponent). */
  staff: Coach[] | null;
  tops: Record<"rated" | "scorers" | "assists", { id: string; name: string; value: string }[]>;
  /** Set in All matches: the Info tab tells the match's own story instead. */
  match: { us: MatchSide | null; them: MatchSide | null } | null;
  photos: Map<string, boolean>;
  /** Open a player in the Players table; only ours can be. */
  onPlayer?: (id: string) => void;
  /** Each player's numbers over the table's scope. */
  numbers: Map<string, PlayerNums>;
  /** A season rather than one match: "Season stats", the last XI. */
  seasonal: boolean;
  /** Goals and assists per 90 rather than totals. */
  per90: boolean;
  menu?: React.ReactNode;
  /** The scope's matches, newest first, from this team's side. */
  record: { res: "W" | "D" | "L" | null; score: string | null; home: boolean; opp: string; day: string | null }[];
}) {
  const [tab, setTab] = useState("lineup");
  const [sheet, setSheet] = useState<LineupRow[] | null>(null);
  // All matches: which side's team sheet, ours or the opponent's.
  const [theirs, setTheirs] = useState(false);
  const shown = match && theirs && match.them ? match.them : (row?.side ?? null);
  const [info, setInfo] = useState<MatchInfo | null>(null);
  const fid = row?.f.id ?? null;
  const sideKey = shown?.side ?? null;

  useEffect(() => {
    if (!fid || !sideKey) return;
    let live = true;
    api.getLineup(fid, sideKey).then(
      (r) => live && setSheet(r),
      () => live && setSheet([]),
    );
    return () => {
      live = false;
    };
  }, [fid, sideKey]);

  // The match's own story, for the Facts and Events tabs.
  const matchMode = match != null;
  useEffect(() => {
    if (!matchMode) return;
    if (!fid) return;
    let live = true;
    api.getMatchInfo(fid).then(
      (r) => live && setInfo(r),
      () => live && setInfo({ status: null, events: [], momentum: [] }),
    );
    return () => {
      live = false;
    };
  }, [fid, matchMode]);
  // Nothing to fetch: an empty sheet and an empty story, not a skeleton.
  const sheetNow = fid && sideKey ? sheet : [];
  const infoNow = fid ? info : { status: null, events: [], momentum: [] };

  const xi = (sheetNow ?? []).filter((p) => p.started && p.x != null && p.y != null);
  const bench = (sheetNow ?? []).filter((p) => !p.started);

  return (
    <Card className="w-full py-0 ring-0 @min-[915px]:min-h-0 @min-[915px]:w-[400px] @min-[915px]:shrink-0 @min-[995px]:w-[460px] @min-[1365px]:w-[560px]">
      <CardContent className="flex min-h-0 min-w-0 flex-1 flex-col gap-0 p-0">
        <div className={cn(BAND_ROW, "flex min-w-0 items-center gap-3 px-4 sm:px-6")}>
          <div className="min-w-0 flex-1">
            <p className="font-heading truncate text-lg font-medium">{name}</p>
            {/* The result line is not here any more - it is the heading over
                the numbers it describes, the same move the season's scope
                made. See the Lineup tab. */}
          </div>
          {menu && <div className="-mr-1 shrink-0">{menu}</div>}
          {/* The film, up here with the match's name. There was a Report button
              beside it; the panel you are reading IS what it opened. */}
          {match && (
            <div className="flex shrink-0 gap-1.5">
              {row ? (
                <Button variant="outline" size="sm" asChild>
                  <Link href={`/review?m=${encodeURIComponent(row.f.id)}`}>Watch</Link>
                </Button>
              ) : (
                <Button variant="outline" size="sm" disabled>
                  Watch
                </Button>
              )}
            </div>
          )}
        </div>

        {/* THE SAME SWITCHER THE PLAYER PANEL HAS, class for class: the two
            panels sit in the same column on the same surface and had two
            different tab strips. The rule sits ABOVE it - the band closes
            itself, as it does over there - so the switcher is fenced on one
            side and opens onto its content, instead of being boxed in by a
            second line underneath. */}
        {/* No border-t here: the band above closes itself (BAND_ROW), so a
            second line would double the rule. And nothing under the switcher
            either - the first block of every tab used to open with its own
            border-t, which fenced the strip on both sides and read as a third
            bar of chrome between the name and the content. */}
        <Tabs value={tab} onValueChange={setTab} className="px-4 pt-2 pb-4 sm:px-6">
          <TabsList className="w-full bg-[var(--card-hover)] [&_[data-slot=tabs-trigger][data-state=active]]:bg-[var(--card)]">
            <TabsTrigger value="lineup">Lineup</TabsTrigger>
            {match ? (
              <>
                <TabsTrigger value="facts">Facts</TabsTrigger>
                <TabsTrigger value="events">Events</TabsTrigger>
              </>
            ) : (
              <TabsTrigger value="info">Info</TabsTrigger>
            )}
          </TabsList>
        </Tabs>

        <div className="min-h-0 flex-1 overflow-y-auto">
          {/* THE TEAM SHEET, the reference's way: the XI where the feed plots
              them - own goal at the bottom - each a face with its rating and
              its goal and assist counts, the subs listed beside the pitch.
              Scaled into the height left, so the tab does not scroll. */}
          <div className={cn("flex h-full min-h-0 flex-col gap-3 px-4 pt-4 pb-4 sm:px-6", tab !== "lineup" && "hidden")}>
            <div className="flex items-center justify-between gap-2">
              <div className="min-w-0">
                {/* WHAT THESE NUMBERS ARE OF, as the title. "Match stats" and
                    "Season stats" named the table's kind, which is obvious
                    from the table, and left the actual answer - which game,
                    which season - in a subtitle up in the band. */}
                <h3 className="font-heading flex min-w-0 items-center gap-1.5 truncate text-base font-medium">
                  {seasonal ? (
                    (scope ?? "Season stats")
                  ) : row ? (
                    <>
                      <ResultBadge res={row.res} />
                      {/* The score never breaks; the opponent's name gives way. */}
                      <span className="readout shrink-0 whitespace-nowrap">{row.f.score}</span>
                      <span className="truncate font-normal">
                        v {row.opp} ({row.home ? "H" : "A"})
                      </span>
                    </>
                  ) : (
                    "Match stats"
                  )}
                </h3>
                <p className="text-muted-foreground flex min-w-0 items-center gap-1.5 truncate text-xs">
                  {seasonal ? "Last starting XI" : (longDate(row?.day) ?? "Starting XI")}
                  {shown?.formation ? ` · ${shown.formation}` : ""}
                  {!seasonal && shown?.team_rating != null && <RatingChip value={shown.team_rating} />}
                </p>
              </div>
              {/* All matches: a round switch with the Opponents tab's shield -
                  on, it shows the opponent's team sheet. */}
              {match?.them && row && (
                <button
                  type="button"
                  onClick={() => setTheirs((v) => !v)}
                  aria-pressed={theirs}
                  aria-label={theirs ? "Show our lineup" : `Show ${row.opp}'s lineup`}
                  title={theirs ? "Show our lineup" : `Show ${row.opp}'s lineup`}
                  className={cn(
                    "grid size-8 shrink-0 place-items-center rounded-full border transition-colors",
                    theirs
                      ? "bg-foreground text-background border-transparent"
                      : "border-[var(--hairline-strong)] text-[var(--ink-soft)] hover:bg-[var(--card-hover)]",
                  )}
                >
                  <Shield className="size-4" />
                </button>
              )}
            </div>
            {sheetNow === null ? (
              <Skeleton className="h-64 w-full" />
            ) : (
              <div className="flex min-h-0 flex-1 gap-3">
                <div className="flex min-h-0 min-w-0 flex-1 justify-center">
                  <div className="relative aspect-[68/105] h-full max-w-full rounded-[var(--radius-md)] bg-[color-mix(in_oklab,var(--ink)_6%,var(--card))]">
                    <svg viewBox="0 0 68 105" className="absolute inset-0 size-full" aria-hidden>
                      <g fill="none" stroke="color-mix(in oklab, var(--ink) 12%, var(--card))" strokeWidth="0.6">
                        <line x1="0" y1="52.5" x2="68" y2="52.5" />
                        <circle cx="34" cy="52.5" r="9.15" />
                        <path d="M13.84 0V16.5H54.16V0M24.84 0V5.5H43.16V0M26.69 16.5A9.15 9.15 0 0 0 41.31 16.5" />
                        <path d="M13.84 105V88.5H54.16V105M24.84 105V99.5H43.16V105M26.69 88.5A9.15 9.15 0 0 1 41.31 88.5" />
                      </g>
                    </svg>
                    {xi.map((p) => (
                      <LineupNode
                        key={p.player_id}
                        p={p}
                        num={numbers.get(p.player_id)}
                        per90={per90}
                        photo={photos.get(p.player_id) ?? false}
                        onOpen={onPlayer && photos.has(p.player_id) ? () => onPlayer(p.player_id) : undefined}
                      />
                    ))}
                  </div>
                </div>
                {bench.length > 0 && (
                  <div className="flex w-[136px] shrink-0 flex-col gap-0.5 overflow-y-auto">
                    <p className="text-muted-foreground pb-1 text-xs">Substitutes</p>
                    {bench.map((p) => {
                      const r = avgRating(numbers.get(p.player_id));
                      const open = onPlayer && photos.has(p.player_id) ? () => onPlayer(p.player_id) : undefined;
                      return (
                        <div
                          key={p.player_id}
                          role={open ? "button" : undefined}
                          tabIndex={open ? 0 : undefined}
                          onClick={open}
                          onKeyDown={open ? activate(open) : undefined}
                          title={open ? `Open ${p.name} in Players` : undefined}
                          className={cn(
                            "flex min-w-0 items-center gap-2 rounded-[var(--radius-sm)] py-1",
                            open && "cursor-pointer hover:bg-[var(--card-hover)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ink-soft)]",
                          )}
                        >
                          <Portrait id={p.player_id} name={p.name} photo={photos.get(p.player_id) ?? false} size={24} />
                          <div className="min-w-0 flex-1">
                            <p className="truncate text-xs text-[var(--ink)]">
                              {p.shirt != null && <span className="text-muted-foreground">{p.shirt} </span>}
                              {surname(p.name)}
                            </p>
                            <p className="text-muted-foreground truncate text-[10px]">
                              {p.on_minute != null ? `On ${p.on_minute}'` : "Unused"}
                            </p>
                          </div>
                          {r != null && <RatingChip value={r} />}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* THE SQUAD: its details, its staff, its form over the scope and
              its best players in it. */}
          {match && (
            <>
              <div className={cn("flex h-full min-h-0 flex-col gap-4 px-4 pt-4 pb-4 sm:px-6", tab !== "facts" && "hidden")}>
                <MatchFacts f={row?.f ?? null} us={match.us} them={match.them} info={infoNow} facts={facts} />
              </div>
              <div className={cn("px-4 pt-3 pb-4 sm:px-6", tab !== "events" && "hidden")}>
                <MatchEventsList info={infoNow} />
              </div>
            </>
          )}
          <div className={cn("grid gap-6 px-4 pt-4 pb-4 sm:px-6", (tab !== "info" || match) && "hidden")}>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-3">
              {facts.map(([l, v]) => (
                <Info key={l} v={v} l={l} />
              ))}
            </dl>

            {/* STAFF: a squad has two or three, so three slots side by side -
                one row, so the tab still fits without scrolling - filled from
                the club's staff list, which the "..." menu adds to. */}
            {staff && (
              <div className="grid gap-2">
                <h3 className="font-heading text-base font-medium">Staff</h3>
                <div className="grid grid-cols-3 gap-2">
                  {[...staff, ...Array(Math.max(0, 3 - staff.length)).fill(null)].map((c: Coach | null, i) =>
                    c ? (
                      <div key={c.coach_id} className="flex min-w-0 items-center gap-2 rounded-[var(--radius-md)] bg-[var(--card-hover)] px-2 py-2">
                        <Portrait name={c.name} size={24} />
                        <div className="min-w-0">
                          <p className="truncate text-xs text-[var(--ink)]" title={c.name}>{c.name}</p>
                          <p className="text-muted-foreground truncate text-[10px]">{c.role || "Staff"}</p>
                        </div>
                      </div>
                    ) : (
                      <div
                        key={`slot-${i}`}
                        className="text-muted-foreground grid place-items-center rounded-[var(--radius-md)] border border-dashed border-[var(--hairline-strong)] px-2 py-2.5 text-xs"
                      >
                        Open slot
                      </div>
                    ),
                  )}
                </div>
              </div>
            )}

            {/* TEAM FORM: the last five in scope, oldest to newest, the score
                on its result's colour and the opponent under it. */}
            <div className="grid gap-3">
                <h3 className="font-heading text-base font-medium">Team form</h3>
                <div className="grid grid-cols-5 gap-2">
                  {!record.length &&
                    Array.from({ length: 5 }, (_, i) => (
                      <div key={`empty-${i}`} className="grid justify-items-center gap-2">
                        <span className="readout text-muted-foreground rounded-[var(--radius-md)] bg-[var(--card-hover)] px-2 py-1 text-xs font-semibold">
                          –
                        </span>
                        <span className="size-8 rounded-full bg-[var(--card-hover)]" />
                      </div>
                    ))}
                  {record.slice(0, 5).reverse().map((x, i, arr) => (
                    <div key={`${x.day}-${i}`} className="grid justify-items-center gap-2" title={`v ${x.opp} · ${longDate(x.day) ?? ""}`}>
                      <span
                        className={cn(
                          "readout rounded-[var(--radius-md)] px-2 py-1 text-xs font-semibold whitespace-nowrap text-white",
                          x.res === "W" ? "bg-[#2f855a]" : x.res === "L" ? "bg-[#c0392b]" : "bg-[#6b7280]",
                          // The newest, marked as the reference marks it.
                          i === arr.length - 1 && "shadow-[0_3px_0_-1px_var(--mute)]",
                        )}
                      >
                        {(x.score ?? "–").replace(/\s*[-–]\s*/, " - ")}
                      </span>
                      <span className="grid size-8 place-items-center rounded-full bg-[var(--card-hover)] text-[10px] font-semibold text-[var(--ink-soft)]">
                        {initials(x.opp)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

            {/* THE TOP THREES, side by side as the reference has them. */}
            <div className="grid grid-cols-3 gap-4">
              {(
                [
                  ["rated", "Top rated"],
                  ["scorers", "Top scorers"],
                  ["assists", "Top assists"],
                ] as const
              ).map(([k, title]) => (
                <div key={k} className="grid min-w-0 content-start gap-1">
                  <h3 className="font-heading pb-1 text-sm font-medium">{title}</h3>
                  {tops[k].length ? (
                    tops[k].map((p, i) => (
                      <div key={p.id} className="flex min-w-0 items-center gap-2 border-b border-[var(--hairline)] py-2 last:border-0">
                        <Portrait id={p.id} name={p.name} photo={photos.get(p.id) ?? false} size={24} />
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-xs text-[var(--ink)]" title={p.name}>{surname(p.name)}</p>
                          <p className="text-muted-foreground truncate text-[10px]">{name}</p>
                        </div>
                        <span
                          className={cn(
                            "readout shrink-0 text-xs",
                            i === 0 && "bg-foreground text-background rounded-full px-1.5 py-0.5 font-semibold",
                          )}
                        >
                          {p.value}
                        </span>
                      </div>
                    ))
                  ) : (
                    <p className="text-muted-foreground text-xs">None</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

/** THE MATCH ITSELF, for All matches' Facts tab - nothing that is already a
 *  column in the table: when, where and what it was, both teams' shape and
 *  how the game swung. */
function MatchFacts({
  f,
  us,
  them,
  info,
  facts,
}: {
  f: Fixture | null;
  us: MatchSide | null;
  them: MatchSide | null;
  info: MatchInfo | null;
  /** The squad's own: age group and level. */
  facts: [string, string][];
}) {
  const when = f?.date ? new Date(f.date) : null;
  const home = us?.side === "home" ? us : them;
  const away = us?.side === "away" ? us : them;
  const venue = (f as { venue?: string | null } | null)?.venue ?? null;
  const homeName = f?.home ?? "Home";
  const awayName = f?.away ?? "Away";

  return (
    <>
      {/* "Sat, November 1, 1:30 PM", in the viewer's own time. Location is
          the stadium, which this import did not keep - it reads "—" until
          the feed's venue is brought in. Season is the competition as the
          Season filter names it. */}
      <dl className="grid grid-cols-2 gap-x-4 gap-y-3">
        <Info
          v={
            when
              ? `${when.toLocaleDateString(undefined, { weekday: "short", month: "long", day: "numeric" })}, ${when.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })}`
              : "—"
          }
          l="Kick-off"
        />
        <Info v={venue ?? "—"} l="Location" />
        <Info v={f?.competition ?? "—"} l="Season" />
        <Info v={info?.status ?? "—"} l="Status" />
        {facts.map(([l, v]) => (
          <Info key={l} v={v} l={l} />
        ))}
      </dl>
      <div className="border-t border-[var(--hairline)]" />

      <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-2 text-sm">
        <TeamShape name={homeName} side={home} />
        <span className="text-muted-foreground text-xs">v</span>
        <TeamShape name={awayName} side={away} right />
      </div>
      <div className="border-t border-[var(--hairline)]" />

      <div className="flex min-h-0 flex-1 flex-col gap-1.5">
        <p className="text-muted-foreground flex justify-between gap-2 text-xs">
          <span>Momentum</span>
          <span className="truncate">
            <span className="text-[var(--foreground)]">▲ {homeName}</span> ·{" "}
            <span className="text-[var(--chart-1)]">▼ {awayName}</span>
          </span>
        </p>
        {info === null ? (
          <Skeleton className="h-16 w-full" />
        ) : (
          // The match report's own chart, fed from the store: one value per
          // minute, goals as balls above and below. With no series it draws
          // its empty frame and says so.
          <>
            <MomentumChart
              fill
              momentum={{
                values: info.momentum.map((p) => p.value),
                goals: info.events
                  .filter((e) => (e.type === "goal" || e.type === "own_goal") && e.minute != null && e.side)
                  .map((e) => ({ minute: e.minute as number, side: e.side as "home" | "away" })),
              }}
            />
          </>
        )}
      </div>
    </>
  );
}

/** KEY EVENTS, All matches' own tab: home on the left and away on the right,
 *  in order, with the half-time line - subs listed now that there is room,
 *  and the tab scrolls when a match has a lot of them. */
function MatchEventsList({ info }: { info: MatchInfo | null }) {
  if (!info) return <Skeleton className="h-40 w-full" />;
  // Only the FIRST period marker: the import files full time as another "HT".
  const firstPeriod = info.events.find((e) => e.type === "period");
  const rows = info.events.filter((e) => e.type !== "period" || e === firstPeriod);
  if (!rows.length) return <p className="text-muted-foreground text-sm">No events for this match.</p>;
  return (
    <div className="grid">
      {rows.map((e, i) =>
        e.type === "period" ? (
          <div key={i} className="flex items-center gap-2 py-1.5">
            <span className="h-px flex-1 bg-[var(--grid-line)]" />
            <span className="readout text-[11px]">HT {e.score?.replace(/\s+/g, "")}</span>
            <span className="h-px flex-1 bg-[var(--grid-line)]" />
          </div>
        ) : (
          <div key={i} className="grid grid-cols-2 gap-3 py-1">
            {e.side === "home" ? <EventRow e={e} /> : <span />}
            {e.side === "away" ? <EventRow e={e} right /> : <span />}
          </div>
        ),
      )}
    </div>
  );
}

function TeamShape({ name, side, right = false }: { name: string; side: MatchSide | null; right?: boolean }) {
  return (
    <div className={cn("min-w-0", right && "text-right")}>
      <p className="truncate text-[var(--ink)]">{name}</p>
      <p className={cn("text-muted-foreground flex items-center gap-1.5 text-xs", right && "justify-end")}>
        {side?.formation ?? "—"}
        {side?.team_rating != null && <RatingChip value={side.team_rating} />}
      </p>
    </div>
  );
}

function EventRow({ e, right = false }: { e: MatchInfo["events"][number]; right?: boolean }) {
  const icon =
    e.type === "goal" || e.type === "own_goal" ? (
      <BallIcon label="Goal" className="mx-0 size-3.5 shrink-0" />
    ) : e.type === "sub" ? (
      <ArrowLeftRight aria-label="Substitution" className="text-muted-foreground size-3.5 shrink-0" />
    ) : (
      <span
        aria-label={e.type === "red_card" ? "Red card" : "Yellow card"}
        className={cn("h-3.5 w-2.5 shrink-0 rounded-[2px]", e.type === "red_card" ? "bg-bad" : "bg-warn")}
      />
    );
  return (
    <div className={cn("flex min-w-0 items-center gap-2 text-xs", right && "flex-row-reverse text-right")}>
      <span className="readout text-muted-foreground w-7 shrink-0">{e.minute != null ? `${e.minute}'` : ""}</span>
      {icon}
      <span className="min-w-0 truncate text-[var(--ink)]">
        {e.who ?? "—"}
        {e.type === "own_goal" && <span className="text-muted-foreground"> (og)</span>}
        {e.other && (e.type === "goal" || e.type === "own_goal") && (
          <span className="text-muted-foreground"> · {e.other}</span>
        )}
        {e.other && e.type === "sub" && <span className="text-muted-foreground"> for {e.other}</span>}
      </span>
    </div>
  );
}

/** One player's numbers over the table's scope. */
type PlayerNums = { ratings: number[]; goals: number; assists: number; mins: number };

const avgRating = (n?: PlayerNums) =>
  n?.ratings.length ? n.ratings.reduce((a, b) => a + b, 0) / n.ratings.length : null;

const surname = (name: string) => name.split(" ").slice(-1)[0];

/** A club's initials, the design system's stand-in for a crest. */
const initials = (name: string) =>
  name.split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0]).join("").toUpperCase() || "?";

/** One starter on the team sheet: the face, the rating pill on its top
 *  right, goals and assists as badges under it when there are any - per 90
 *  when asked - and the shirt number and surname below. */
function LineupNode({
  p,
  num,
  per90,
  photo,
  onOpen,
}: {
  p: LineupRow;
  num?: PlayerNums;
  per90: boolean;
  photo: boolean;
  /** Set for our own players: open them in the Players table. */
  onOpen?: () => void;
}) {
  const rating = avgRating(num) ?? p.rating;
  const shown = (v: number) =>
    per90 ? (num?.mins ? ((v / num.mins) * 90).toFixed(2) : "—") : String(v);
  const badge = "absolute -bottom-1.5 flex items-center gap-0.5 rounded-full bg-[var(--card)] px-1 py-px text-[10px] leading-none text-[var(--ink)] ring-1 ring-[var(--hairline-strong)]";
  return (
    <div
      className={cn(
        "group absolute flex w-20 -translate-x-1/2 -translate-y-1/2 flex-col items-center gap-1.5",
        onOpen && "cursor-pointer",
      )}
      style={{ left: `${(p.x as number) * 100}%`, top: `${(1 - (p.y as number)) * 100}%` }}
      title={onOpen ? `Open ${p.name} in Players` : `${p.shirt ?? ""} ${p.name}`}
      role={onOpen ? "button" : undefined}
      tabIndex={onOpen ? 0 : undefined}
      onClick={onOpen}
      onKeyDown={onOpen ? activate(onOpen) : undefined}
    >
      <div className={cn("relative rounded-full", onOpen && "group-hover:ring-2 group-focus-visible:ring-2 group-hover:ring-[var(--ink-soft)] group-focus-visible:ring-[var(--ink-soft)]")}>
        <Portrait id={p.player_id} name={p.name} photo={photo} size={38} />
        {rating != null && <RatingChip value={rating} className="absolute -top-2 -right-6" />}
        {(num?.assists ?? 0) > 0 && (
          <span className={cn(badge, "-left-4")}>
            <BootIcon label="Assists" className="size-2.5" />
            {shown(num!.assists)}
          </span>
        )}
        {(num?.goals ?? 0) > 0 && (
          <span className={cn(badge, "-right-4")}>
            <BallIcon label="Goals" className="size-3" />
            {shown(num!.goals)}
          </span>
        )}
      </div>
      <span className="max-w-full truncate text-[11px] leading-tight text-[var(--ink)]">
        {p.shirt != null && <span className="text-muted-foreground">{p.shirt} </span>}
        {surname(p.name)}
      </span>
    </div>
  );
}

/** A fact, value over label - the player panel's own cell. */
function Info({ v, l }: { v: string; l: string }) {
  return (
    <div className="min-w-0">
      <dd className="truncate text-sm font-medium text-[var(--ink)]">{v}</dd>
      <dt className="text-muted-foreground truncate text-xs">{l}</dt>
    </div>
  );
}
