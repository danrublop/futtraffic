"use client";

import { useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight, Shield, UserRound, Users, X } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { PitchIcon } from "@/components/pitch-icon";
import { ClubPanel } from "@/components/data/club-editor";
import { LoadFailure, mayExplainAbsence } from "@/components/data/load-state";
import { DATA_HEAD, ZEBRA } from "@/components/data/section-shell";
import { PlayerRegistry, TableSkeleton } from "@/components/data/players-section";
import { SquadMatches } from "@/components/data/squads-section";
import { MatchesTable } from "@/components/data/matches-section";
import { ClubTable, MatchTable, PlayerTable } from "@/components/data/tables";
import { useSeason } from "@/hooks/use-matches";
import { Pick } from "@/components/pick";
import { api } from "@/lib/api";
import { useOrg } from "@/lib/club";
import { clubRowsIn, feedClubFor } from "@/lib/data-filters";
import { parseView, type DataView } from "@/lib/data-nav";
import {
  agg,
  clubTotals,
  opponentTotals,
  playerTotals,
  sectionsFor,
  type SeasonTable,
  type StatSection,
} from "@/lib/season";
import type { Match } from "@/lib/types";
import { cn } from "@/lib/utils";

/** ONE DATABASE, ONE WAY TO LOOK AT IT.
 *
 *  This replaced five tabs that each invented their own page. Teams was a club
 *  editor followed by a nested four-tab strip; Opponents was a chip row over
 *  four stacked tables; Players was two tables under two different headings.
 *  Every one of them opened with a paragraph. Finding a number meant knowing
 *  which of five layouts had put it where.
 *
 *  So: a rail of TABLES on the left, one table at a time on the right, and the
 *  things that used to be drill-downs are FILTERS. Picking a squad is not a
 *  page any more, it is a value in a dropdown, which is what it always was.
 *
 *  Three rules the layout keeps:
 *
 *   - ONE table on screen. A page with four tables stacked is four questions
 *     answered at once and none of them asked.
 *   - ONE line of provenance per table, never a paragraph. Which KIND of number
 *     this is — measured, entered, or bound through a human's confirmation — is
 *     the product's central obligation and it fits on a line.
 *   - The filters that apply, and no others. A control that cannot narrow the
 *     table in front of you is clutter that also teaches the wrong thing.
 */

interface Entry {
  id: string;
  name: string;
  /** `ComponentType`, not lucide's own type: Matches uses the app's arena mark,
   *  which is an <img> rather than an SVG icon component. */
  icon: React.ComponentType<{ className?: string }>;
  /** One line. What kind of number this is and where it came from. */
  note: string;
  /** Which filters this table can honour. Anything not listed is not shown. */
  uses?: ("team" | "club" | "season" | "match")[];
  /** True when the note INTERPRETS a missing number ("an em dash means no
   *  confirmation yet"). Such a sentence is only true if the thing that
   *  supplies those numbers arrived — with the fetch failed, every cell is an
   *  em dash and the note would tell the coach something confident and wrong.
   *  This is the same gate the players table used to carry in its own caption;
   *  the caption moved up here, so the gate had to move with it. */
  explainsAbsence?: true;
}



const ENTRIES: Entry[] = [
  {
    id: "squads",
    name: "Squads",
    icon: Users,
    note: "Each squad's team stats from the report's Teams tab, over the season or match picked. Click a squad for its team sheet and record.",
    uses: ["team", "season", "match"],
  },
  {
    id: "our-players",
    name: "Players",
    icon: UserRound,
    note: "Everyone on our squads. Distance, km/90 and peak speed are MEASURED, counted only from tracks a human confirmed — an em dash is no confirmation yet, never a zero. Click a player for their profile.",
    uses: ["team", "season", "match"],
    explainsAbsence: true,
  },
  {
    id: "records",
    name: "Opponents",
    icon: Shield,
    note: "Every club we have played, with ITS side of each match's team stats and its record against us. Click one for their team sheet.",
    uses: ["team", "season", "match"],
  },
  {
    id: "all-matches",
    name: "All matches",
    icon: PitchIcon,
    note: "Every match a squad played, match against match, with its team stats from the report's Teams tab. Click a match for its team sheet.",
    uses: ["team", "season"],
  },
];

const entryOf = (id: string) => ENTRIES.find((e) => e.id === id) ?? ENTRIES[0];

export function DataBrowser({ initialTab }: { initialTab?: string }) {
  const [table, setTable] = useState<SeasonTable | null>(null);
  const [tableError, setTableError] = useState<string | null>(null);
  const { org } = useOrg();
  const { error: measuredError } = useSeason(undefined, true);

  // WHERE YOU ARE, as ONE snapshot - the table, the filters, and the player a
  // jump landed on - kept in a history, so Back and Forward walk it the way a
  // browser walks pages. Every move goes through `navigate`, which records it.
  // The filters live here so switching table keeps them; a table that cannot
  // honour one simply does not show it. `season` is null until someone picks:
  // the tables open on the CURRENT season, and "" is an explicit "Every season".
  const [hist, setHist] = useState(() => ({
    stack: [
      { ...parseView(initialTab), team: "", season: null as string | null, match: "", player: null as string | null },
    ],
    at: 0,
  }));
  const nav = hist.stack[hist.at];
  const view: DataView = { section: nav.section, id: nav.id };
  const navigate = (patch: Partial<typeof nav>) => {
    const next = { ...nav, ...patch };
    setHist((h) => ({ stack: [...h.stack.slice(0, h.at + 1), next], at: h.at + 1 }));
  };
  const step = (by: -1 | 1) => {
    const at = Math.max(0, Math.min(hist.stack.length - 1, hist.at + by));
    if (at === hist.at) return;
    setHist((h) => ({ ...h, at }));
  };
  const go = (next: DataView) => navigate({ ...next, player: null });
  const team = nav.team;
  const seasonPick = nav.season;
  const match = nav.match;
  const setTeam = (t: string) => navigate({ team: t });
  const setMatch = (m: string) => navigate({ match: m });
  const [club, setClub] = useState("");

  // TABLES STAY BUILT once opened: each is kept mounted, hidden, so going
  // back to one is instant and it keeps its sort, its pills and its place.
  const [visited, setVisited] = useState(() => new Set<string>([nav.section]));
  if (!visited.has(nav.section)) setVisited(new Set([...visited, nav.section]));

  useEffect(() => {
    let live = true;
    api.season().then(
      (t) => live && setTable(t),
      (e: unknown) => live && setTableError(e instanceof Error ? e.message : String(e)),
    );
    return () => {
      live = false;
    };
  }, []);

  const entry = entryOf(view.section);

  const clubs = useMemo(
    () => (table ? [...new Set(table.players.map((p) => p.club))].sort() : []),
    [table],
  );
  const fixtures = useMemo(
    () => (table ? [...table.fixtures].sort((a, b) => (b.date ?? "").localeCompare(a.date ?? "")) : []),
    [table],
  );

  // A season is a competition as the feed names it ("Premier League
  // 2025/2026"), and it narrows the Match list to its own fixtures.
  const seasons = useMemo(
    // In the order they were last played - `fixtures` is newest first.
    () => [...new Set(fixtures.map((f) => f.competition).filter((c): c is string => !!c))],
    [fixtures],
  );
  const season = seasonPick ?? seasons[0] ?? "";
  const pickSeason = (c: string) =>
    navigate({
      season: c,
      ...(c && match && fixtures.find((f) => f.id === match)?.competition !== c ? { match: "" } : {}),
    });

  /** A player clicked on a team sheet: over to Players with them picked -
   *  and, from a single match, that match as the filter. The other filters
   *  come along as they are. */
  const openPlayer = (id: string, matchId?: string) =>
    navigate({ section: "our-players", id: null, player: id, ...(matchId ? { match: matchId } : {}) });

  const uses = (f: "team" | "club" | "season" | "match") => entry.uses?.includes(f) ?? false;

  return (
    // Players fills the pane rather than growing it: its table scrolls inside
    // and the profile beside it stays put at full height.
    <div
      className={cn(
        "flex min-w-0 flex-col",
        (entry.id === "our-players" ||
          ((entry.id === "squads" || entry.id === "all-matches" || entry.id === "records") &&
            !view.id)) &&
          "h-full",
      )}
    >
      {/* ------------------------------------------------------------ tabs */}
      {/* ACROSS THE TOP, not down the side. Four names and a squad or two is
          not a navigation tree, and a 164px column spent saying so was 164px
          taken from the table - which is the only thing on this page anyone
          came for. Widthwise it is the same trade the reference screen makes:
          one strip of tabs, everything below it is the data.

          It also fixes a hole rather than just moving furniture. The rail was
          `hidden sm:flex`, so under 640px the Data section had NO navigation
          at all - whichever table you landed on was the only one reachable. A
          strip wraps instead of vanishing.

          Laid out like the Matches page: folder chips on the left, and the
          filters this table honours on the right of the same row. */}
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        {/* Back and forward through where you have been in here. */}
        <div className="flex shrink-0 overflow-hidden rounded-[var(--radius-md)] bg-[var(--card)]">
          {(
            [
              [-1, "Back", ChevronLeft],
              [1, "Forward", ChevronRight],
            ] as const
          ).map(([by, label, Icon]) => (
            <button
              key={label}
              type="button"
              onClick={() => step(by)}
              disabled={by === -1 ? hist.at === 0 : hist.at === hist.stack.length - 1}
              aria-label={label}
              title={label}
              className="grid h-11 w-9 place-items-center text-[var(--ink-soft)] transition-colors first:border-r first:border-[var(--hairline)] hover:bg-[var(--card-hover)] hover:text-[var(--ink)] disabled:opacity-35 disabled:hover:bg-transparent sm:h-9 sm:w-8"
            >
              <Icon className="size-4" />
            </button>
          ))}
        </div>
        {ENTRIES.map((e) => (
          <Tab
            key={e.id}
            entry={e}
            on={e.id === entry.id}
            measuredError={measuredError}
            onOpen={() => go({ section: e.id as DataView["section"], id: null })}
          />
        ))}

        <div className="ml-auto flex flex-wrap items-center gap-x-2 gap-y-2">
          {uses("team") && (
            <Pick
              label="Squad"
              value={team}
              onChange={setTeam}
              all="Every squad"
              items={(org?.teams ?? []).map((t) => ({ id: t.team_id, label: t.name }))}
            />
          )}
          {uses("club") && (
            <Pick
              label="Club"
              value={club}
              onChange={setClub}
              all="Every club"
              items={clubs.map((c) => ({ id: c, label: c }))}
            />
          )}
          {uses("season") && (
            <Pick
              label="Season"
              value={season}
              onChange={pickSeason}
              all="Every season"
              items={seasons.map((c) => ({ id: c, label: c }))}
              // Wide enough for "Premier League 2025/2026" whole.
              className="w-[212px]"
            />
          )}
          {uses("match") && (
            <Pick
              label="Match"
              value={match}
              onChange={setMatch}
              all="Every match"
              items={fixtures
                .filter((f) => !season || f.competition === season)
                .map((f) => ({ id: f.id, label: `${f.home} v ${f.away}` }))}
            />
          )}
        </div>
      </div>

      {/* --------------------------------------------------------- content */}
      <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-3 pt-3">

        {/* The tables whose numbers come from the measured season say so when
            it did not arrive, rather than rendering a column of em dashes. */}
        {measuredError && (entry.id === "our-players" || entry.id === "squads") && (
          <LoadFailure what="the measured season" error={measuredError} />
        )}
        {tableError && entry.id !== "squads" && (
          <LoadFailure what="the season table" error={tableError} />
        )}

        {/* Each table once opened stays mounted; only the one in view shows.
            A drill-down (a row's own page) is drawn on its own. */}
        {ENTRIES.filter((e) => visited.has(e.id)).map((e) => (
          <div
            key={e.id}
            className={cn("flex min-h-0 min-w-0 flex-1 flex-col", (e.id !== entry.id || view.id) && "hidden")}
          >
            <Body
              entry={e}
              table={table}
              team={team}
              club={club}
              season={season}
              onSeason={pickSeason}
              match={match}
              onMatch={setMatch}
              focus={e.id === "our-players" ? nav.player : null}
              onPlayer={openPlayer}
              selected={null}
              onSelect={(id) => go({ section: e.id as DataView["section"], id })}
            />
          </div>
        ))}
        {view.id && (
          <Body
            entry={entry}
            table={table}
            team={team}
            club={club}
            season={season}
            onSeason={pickSeason}
            match={match}
            onMatch={setMatch}
            focus={null}
            onPlayer={openPlayer}
            selected={view.id}
            onSelect={(id) => go({ section: view.section, id })}
          />
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ bodies */

function Body({
  entry,
  table,
  team,
  club,
  season,
  onSeason,
  match,
  onMatch,
  focus,
  onPlayer,
  selected,
  onSelect,
}: {
  entry: Entry;
  table: SeasonTable | null;
  team: string;
  club: string;
  season: string;
  onSeason: (competition: string) => void;
  match: string;
  onMatch: (id: string) => void;
  /** The player a jump landed on, for the Players table to pick. */
  focus: string | null;
  /** Open a player in Players - with a match, filtered to it. */
  onPlayer: (id: string, matchId?: string) => void;
  selected: string | null;
  onSelect: (id: string | null) => void;
}) {
  // A ROW OPENS ITS OWN STATS. A squad and a match are each a subject with
  // several tables behind them, and the table you clicked out of is the wrong
  // place to show four more. The detail keeps the browser's rule — one table on
  // screen — with a small strip to move between them.
  if (selected && entry.id === "squads")
    return <SquadDetail id={selected} table={table} onBack={() => onSelect(null)} />;
  if (selected && entry.id === "all-matches")
    return <MatchDetail id={selected} table={table} onBack={() => onSelect(null)} />;
  if (selected && entry.id === "records")
    return <ClubDetail club={selected} table={table} onBack={() => onSelect(null)} />;

  if (entry.id === "squads")
    return (
      <div className="flex min-h-0 flex-1 flex-col gap-3">
        <SquadMatches
          view="squad"
          onPlayer={(id) => onPlayer(id)}
          menu={<ClubPanel compact />}
          teamId={team || undefined}
          season={season || undefined}
          matchId={match || undefined}
          fixtures={table?.fixtures ?? []}
        />
      </div>
    );
  if (entry.id === "our-players")
    return (
      <PlayerRegistry
        focus={focus}
        teamId={team || undefined}
        season={season || undefined}
        onSeason={onSeason}
        matchId={match || undefined}
        onMatch={onMatch}
      />
    );
  if (entry.id === "all-matches")
    return (
      <SquadMatches
        view="matches"
        onPlayer={onPlayer}
        teamId={team || undefined}
        season={season || undefined}
        fixtures={table?.fixtures ?? []}
      />
    );

  // Everything below is feed-derived and needs the season table.
  if (!table) return <TableSkeleton rows={10} />;

  if (entry.id === "records") {
    return (
      <SquadMatches
        view="opponents"
        teamId={team || undefined}
        season={season || undefined}
        matchId={match || undefined}
        fixtures={table.fixtures}
      />
    );
  }

  // their-players
  const rows = table.players.filter(
    (p) => (!club || p.club === club) && (!match || p.match === match),
  );
  // Header always; the reason under it when the season has no entered feed.
  return (
    <>
      <PlayerTable rows={playerTotals(rows)} />
      {!rows.length && (
        <p className="text-muted-foreground p-4 text-sm">
          Nothing entered for this yet. These rows come from entered match data (team sheets,
          goals, xG), not from tracking.
        </p>
      )}
    </>
  );
}

/* ----------------------------------------------------------------- details */

/** One tab in the strip.
 *
 *  MARKED BY AN UNDERLINE, not by a filled pill. Sideways, a filled active tab
 *  reads as a button somebody is about to press rather than as the place you
 *  already are, and four of them in a row is four boxes competing with the
 *  table underneath. A rule under the live one is what the reference does and
 *  it is also what the strip's own bottom border makes free.
 *
 *  The underline is INK, not the accent. globals.css reserves Signal Blue for
 *  machine state, and which table you are looking at is not machine state.
 *
 *  It is drawn with a border on the button, pulled down by 1px to sit ON the
 *  strip's border rather than above it - so the active tab breaks the rule
 *  instead of stacking a second line under it.
 */
function Tab({
  entry,
  on,
  measuredError,
  onOpen,
}: {
  entry: Entry;
  on: boolean;
  measuredError: string | null;
  onOpen: () => void;
}) {
  const Icon = entry.icon;
  return (
    <button
      type="button"
      onClick={onOpen}
      aria-current={on ? "page" : undefined}
      // The note lives here rather than as a block above every table. It still
      // says which kind of number the table holds — measured, entered, or bound
      // through a human's confirmation — which is the one thing this product
      // cannot stop saying. A hover, not a paragraph.
      title={
        entry.explainsAbsence && !mayExplainAbsence(measuredError)
          ? `${entry.name} — the measured season did not load`
          : `${entry.name} — ${entry.note}`
      }
      // The Matches page's folder chip (Chip in app/page.tsx), class for class.
      className={cn(
        "flex h-11 shrink-0 items-center gap-2 rounded-[var(--radius-md)] border px-2.5 text-[12.5px] whitespace-nowrap transition-colors sm:h-9",
        on
          ? "border-[var(--hairline-strong)] bg-[var(--card-hover)] text-[var(--ink)]"
          : "border-transparent bg-[var(--card)] text-[var(--ink-soft)] hover:bg-[var(--card-hover)] hover:text-[var(--ink)]",
      )}
    >
      <Icon className="size-3.5 shrink-0 text-[var(--graphite)]" />
      {entry.name}
    </button>
  );
}

/** A PINNED ROW AND ITS TABLES.
 *
 *  Clicking a row does not replace the table with a different screen. The row
 *  you clicked stays put at the top — the same columns, the same numbers — and
 *  its own tables open underneath it. So the thing you selected is still on
 *  screen while you read what is behind it, and closing puts you back exactly
 *  where you were rather than at the top of a list you have to find your place
 *  in again.
 *
 *  `pin` is the table itself rendered down to one row, not a text breadcrumb.
 *  A breadcrumb would say the name and drop the numbers, and the numbers are
 *  the reason you clicked.
 */
function Detail({
  pin,
  tabs,
  onClose,
}: {
  pin: React.ReactNode;
  tabs: { id: string; label: string; body: React.ReactNode }[];
  onClose: () => void;
}) {
  const [open, setOpen] = useState(tabs[0]?.id ?? "");
  const body = tabs.find((t) => t.id === open) ?? tabs[0];
  return (
    <div className="grid min-w-0 gap-3">
      {pin}
      <div className="flex min-w-0 flex-wrap items-center gap-1">
        {tabs.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setOpen(t.id)}
            className={cn(
              "h-7 rounded-[var(--radius-md)] px-3 text-[12px] transition-colors",
              t.id === body?.id
                ? "bg-[var(--card)] text-[var(--ink)]"
                : "text-[var(--mute)] hover:text-[var(--ink-soft)]",
            )}
          >
            {t.label}
          </button>
        ))}
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          title="Close"
          className="text-muted-foreground ml-auto grid size-6 place-items-center rounded-[var(--radius-sm)] transition-colors hover:bg-[var(--card)] hover:text-[var(--ink)]"
        >
          <X className="size-3.5" />
        </button>
      </div>
      {body?.body}
    </div>
  );
}

/** ONE SQUAD: its record, who it played, its matches, its players.
 *
 *  The feed club is worked out from the MATCHES, not from the squad's name — a
 *  team keeps its id through a rename, so this club's squad is "First team"
 *  while every feed row still says "Northbridge FC". See `feedClubFor`.
 */
function SquadDetail({
  id,
  table,
  onBack,
}: {
  id: string;
  table: SeasonTable | null;
  onBack: () => void;
}) {
  const [matches, setMatches] = useState<Match[] | null>(null);

  useEffect(() => {
    let live = true;
    api.listMatches(id).then(
      (l) => live && setMatches(l),
      () => live && setMatches([]),
    );
    return () => {
      live = false;
    };
  }, [id]);

  const ids = useMemo(() => new Set((matches ?? []).map((m) => m.id)), [matches]);
  const feedClub = useMemo(
    () => (table ? feedClubFor(table.clubs, ids) : ""),
    [table, ids],
  );
  const feedRows = useMemo(
    () => (table && feedClub ? clubRowsIn(table.clubs, ids, feedClub) : []),
    [table, ids, feedClub],
  );
  // The other side of the same matches, so every stat can be read as ours
  // against theirs rather than as a number with nothing to sit beside.
  const theirRows = useMemo(
    () =>
      table && feedClub
        ? clubRowsIn(table.clubs, ids).filter((r) => r.club !== feedClub)
        : [],
    [table, ids, feedClub],
  );

  const tabs = [
    {
      id: "players",
      label: "Players",
      body: <PlayerRegistry teamId={id} />,
    },
    {
      id: "matches",
      label: "Matches",
      body: table ? (
        <MatchTable table={{ ...table, fixtures: table.fixtures.filter((f) => ids.has(f.id)) }} />
      ) : (
        <Skeleton className="h-48 w-full" />
      ),
    },
  ];

  if (feedRows.length) {
    tabs.push({
      id: "record",
      label: "Record",
      body: (
        <ClubTable
          rows={clubTotals(feedRows)}
          caption="This squad's season as the entered feed recorded it — a human typed these, they are not pipeline measurements."
        />
      ),
    });

    // EVERY TEAM STAT THE FEED CARRIES, grouped the way the match report groups
    // them, aggregated across the squad's matches and set against what the
    // opposition managed in the same games. A season read the way a single
    // match is read.
    for (const section of sectionsFor(feedRows)) {
      tabs.push({
        id: `s-${section.title}`,
        label: section.title,
        body: <StatSectionTable section={section} against={theirRows} />,
      });
    }

    tabs.push({
      id: "against",
      label: "Opponents",
      body: (
        <ClubTable
          rows={opponentTotals(feedRows, feedClub)}
          first="Opponent"
          caption={`The same season, cut by who they played. Goals are ${feedClub}'s.`}
        />
      ),
    });
  }

  return (
    <Detail
      pin={<ClubPanel selected={id} onSelect={() => onBack()} />}
      tabs={tabs}
      onClose={onBack}
    />
  );
}

/** ONE MATCH: every stat the library holds about it.
 *
 *  Both sides' team stats and both squads' player stats, from the ENTERED feed.
 *  The full report — momentum, shot map, formations — is its own page on `?m=`;
 *  this is the database's view of the same match, which is the numbers as rows.
 */
function MatchDetail({
  id,
  table,
  onBack,
}: {
  id: string;
  table: SeasonTable | null;
  onBack: () => void;
}) {
  if (!table) return <Skeleton className="h-64 w-full" />;
  const clubRows = table.clubs.filter((r) => r.match === id);
  const playerRows = table.players.filter((p) => p.match === id);

  const tabs = clubRows.length
    ? [
        {
          id: "teams",
          label: "Team stats",
          body: (
            <ClubTable
              rows={clubTotals(clubRows)}
              caption="Both sides of this match, as the entered feed recorded it."
            />
          ),
        },
        {
          id: "players",
          label: "Player stats",
          body: <PlayerTable rows={playerTotals(playerRows)} />,
        },
      ]
    : [
        {
          id: "none",
          label: "No feed",
          body: (
            <p className="text-muted-foreground max-w-[60ch] text-sm">
              No entered feed for this match, so there are no typed stats to show. What the
              pipeline measured of it is not on this page yet — the Data page shows the feed
              while the model is still being built.
            </p>
          ),
        },
      ];

  return (
    <Detail
      pin={<MatchesTable table={table} pinned={id} onOpen={() => onBack()} />}
      tabs={tabs}
      onClose={onBack}
    />
  );
}

/** ONE OPPONENT: their season, and who they played. */
function ClubDetail({
  club,
  table,
  onBack,
}: {
  club: string;
  table: SeasonTable | null;
  onBack: () => void;
}) {
  if (!table) return <Skeleton className="h-64 w-full" />;
  return (
    <Detail
      pin={
        <ClubTable
          rows={clubTotals(table.clubs)}
          caption={club}
          pinned={club}
          onPick={() => onBack()}
        />
      }
      tabs={[
        {
          id: "against",
          label: "Who they played",
          body: (
            <ClubTable
              rows={opponentTotals(table.clubs, club)}
              first="Opponent"
              caption={`Goals are ${club}'s, from their side of each match.`}
            />
          ),
        },
        {
          id: "matches",
          label: "Matches",
          body: (
            <MatchTable
              table={{
                ...table,
                fixtures: table.fixtures.filter((f) => f.home === club || f.away === club),
              }}
            />
          ),
        },
        {
          id: "players",
          label: "Players",
          body: (
            <PlayerTable rows={playerTotals(table.players.filter((p) => p.club === club))} />
          ),
        },
      ]}
      onClose={onBack}
    />
  );
}

/** One section of team stats: ours beside theirs, over the same matches.
 *
 *  `n/N` on a cell is the coverage rule the rest of the surface uses — the feeds
 *  disagree about which stats they carry, so a value drawn from fewer matches
 *  than the squad played says so rather than passing as a full-season total.
 */
function StatSectionTable({
  section,
  against,
}: {
  section: StatSection;
  against: { stats: Record<string, number> }[];
}) {
  return (
    <Card className="py-0 ring-0">
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <Table className={cn(ZEBRA, DATA_HEAD)} aria-label={section.title}>
            <TableHeader>
              <TableRow>
                <TableHead>{section.title}</TableHead>
                <TableHead className="text-right">This squad</TableHead>
                <TableHead className="text-right">Opponents</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {section.rows.map((r) => {
                const theirs = agg(against, r.label);
                return (
                  <TableRow key={r.label} className="hover:bg-muted/40">
                    <TableCell className="whitespace-nowrap">{r.label}</TableCell>
                    <StatCell a={r.agg} />
                    <StatCell a={theirs} />
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}

/** A number the feed carried, with how many matches carried it. */
function StatCell({ a }: { a: { value: number; n: number } }) {
  if (a.n === 0)
    return (
      <TableCell className="readout text-muted-foreground text-right">—</TableCell>
    );
  const dp = Number.isInteger(a.value) ? 0 : 2;
  return (
    <TableCell className="readout text-right whitespace-nowrap">
      {a.value.toFixed(dp)}
      <span className="text-muted-foreground ml-1.5 text-[11px]" title={`From ${a.n} matches`}>
        {a.n}
      </span>
    </TableCell>
  );
}

export { ENTRIES as DATA_ENTRIES };
export type { Match };
