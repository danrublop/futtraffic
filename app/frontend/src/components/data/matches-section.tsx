"use client";

import { useEffect, useMemo, useState } from "react";

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
import { cmp, SortHead, useSort } from "@/components/sort-head";
import { Chevron } from "@/components/data/tables";
import { LoadFailure } from "@/components/data/load-state";
import { DATA_HEAD, FilterField, SectionShell, ZEBRA } from "@/components/data/section-shell";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useMatches } from "@/hooks/use-matches";
import { api } from "@/lib/api";
import { matchRows, type MatchRow } from "@/lib/data-filters";
import { useSurfaceParams } from "@/components/surface-params";
import type { SeasonTable } from "@/lib/season";
import type { Match } from "@/lib/types";
import { matchDate, matchTitle, progressOf } from "@/lib/status";
import { cn } from "@/lib/utils";

/** EVERY MATCH IN THE LIBRARY, as a list you can sort and narrow.
 *
 *  THE SOURCE IS THE JOB LIST, NOT THE FEED, and that is the whole design.
 *  The obvious source is `SeasonTable.fixtures`, which is what the old match
 *  table rendered and what a first draft of this section used. It is wrong:
 *  fixtures come from the ENTERED feed, and an academy filming its own matches
 *  has no feed at all — `season-tables.tsx` computes `noFeed` from exactly that
 *  array and says so in its own comment. A fixture-derived Matches section is
 *  permanently EMPTY for the club this product is built for, and it would never
 *  show up in testing because the demo library is imported professional
 *  fixtures.
 *
 *  A JOB exists for every upload. So the job list is the spine and the feed is
 *  an enrichment: competition and score where a feed supplied them, an em dash
 *  where it did not. It also buys something a fixture list cannot have — the
 *  job carries processing state, so a match still being tracked appears here
 *  with a progress bar instead of arriving only once it finishes.
 *
 *  CLICKING A ROW DOES NOT OPEN A MATCH VIEW HERE. `app/report/page.tsx` is
 *  1537 lines and already IS the per-match page, wired to `?m=`. A second one
 *  inside this section would be two match reports drifting apart until one has
 *  the shot map and the other has the xG breakdown. A row sets the match and
 *  hands off to that page — see `open` below for why it navigates rather than
 *  splitting a pane.
 */
/** The table on its own, for the database browser: same rows, same hand-off,
 *  no section chrome. `teamId` narrows to one squad by asking the SERVER, which
 *  owns the job -> team slug. */
export function MatchesTable({
  table,
  teamId = "",
  onOpen,
  pinned,
}: {
  table: SeasonTable | null;
  teamId?: string;
  /** When set, the table renders down to this one row: it is the PIN above an
   *  open match, so the row you clicked keeps its numbers on screen. */
  pinned?: string | null;
  /** Given, a row opens the match's tables in place rather than navigating to
   *  its report. The database browser wants the numbers; the sidebar wants the
   *  report. Same table, two callers, one prop. */
  onOpen?: (id: string) => void;
}) {
  return (
    <MatchesSection
      table={table}
      teamId={teamId}
      chrome={false}
      onOpen={onOpen}
      pinned={pinned}
    />
  );
}

export function MatchesSection({
  table,
  teamId = "",
  chrome = true,
  onOpen,
  pinned,
}: {
  table: SeasonTable | null;
  teamId?: string;
  chrome?: boolean;
  onOpen?: (id: string) => void;
  pinned?: string | null;
}) {
  const params = useSurfaceParams();
  // Scoped by the header's team lens, which is the app's ONE team control. A
  // filter in here never writes it back.
  const { matches: allMatches, error, refresh } = useMatches();
  // One squad's matches come from the server, which owns the job -> team slug.
  const [scoped, setScoped] = useState<{ team: string; list: Match[] } | null>(null);
  useEffect(() => {
    if (!teamId) return;
    let live = true;
    api.listMatches(teamId).then(
      (list) => live && setScoped({ team: teamId, list }),
      () => live && setScoped({ team: teamId, list: [] }),
    );
    return () => {
      live = false;
    };
  }, [teamId]);
  const matches = teamId
    ? (scoped?.team === teamId ? scoped.list : null)
    : allMatches;
  const [competition, setCompetition] = useState("");
  const sort = useSort("date");

  const rows = useMemo(
    () => matchRows(matches ?? [], table?.fixtures ?? []),
    [matches, table],
  );

  const competitions = useMemo(
    () =>
      [...new Set(rows.map((r) => r.fixture?.competition).filter((c): c is string => !!c))].sort(),
    [rows],
  );

  const shown = useMemo(() => {
    const narrowed = rows.filter(
      (r) => !competition || r.fixture?.competition === competition,
    );
    const key = sort.key;
    return [...narrowed].sort((a, b) => {
      if (key === "match") {
        const s = matchTitle(a.match).localeCompare(matchTitle(b.match));
        return sort.asc ? s : -s;
      }
      if (key === "competition") {
        const s = (a.fixture?.competition ?? "").localeCompare(b.fixture?.competition ?? "");
        return sort.asc ? s : -s;
      }
      // Date, and the default. The JOB's stamp, not the feed's — every row has
      // one, and a row with none sorts last rather than first, which is what an
      // undefined would do through a subtraction.
      return cmp(a.match.created ?? -Infinity, b.match.created ?? -Infinity, sort.asc);
    });
  }, [rows, competition, sort.key, sort.asc]);

  // Pinned: this table IS the header of the open match, so it shows that row
  // and nothing else.
  const visible = pinned ? shown.filter((r) => r.match.id === pinned) : shown;

  /** Set the match and stay on the database. The report surface this used to
   *  hand off to is gone, so the fallback lands on Data with `?m=` set.
   *
   *  Both params are merged in, never the whole query string replaced — that
   *  is the bug that has shipped twice in this app already.
   */
  const open = (id: string) => {
    if (onOpen) return onOpen(id);
    params.set({ m: id, v: "data" }, { push: true });
  };

  const current = params.get("m");

  const shell = (
    <SectionShell
      title="Matches"
      count={matches === null ? undefined : `${shown.length} of ${rows.length}`}
      note="Every match in the library, whether or not a feed named it. Competition and score come from the entered feed and are absent when there is none; the date and the state come from the upload itself."
      filters={
        <>
          {competitions.length > 0 && (
            <FilterField label="Competition">
              <Choose
                value={competition}
                onChange={setCompetition}
                all="Every competition"
                items={competitions.map((c) => ({ id: c, label: c }))}
              />
            </FilterField>
          )}
        </>
      }
    >
      {error && <LoadFailure what="the match list" error={error} onRetry={() => void refresh()} />}
      <Card className="py-0 ring-0">
        <CardContent className="p-0">
          {matches === null && !error ? (
            <div className="p-4">
              <Skeleton className="h-5 w-full" />
            </div>
          ) : (
            <div className="overflow-x-auto">
              <Table className={cn(ZEBRA, DATA_HEAD)}>
                <TableHeader>
                  <TableRow>
                    <SortHead active={sort.key === "date"} onClick={sort.on("date")}>
                      Date
                    </SortHead>
                    <SortHead active={sort.key === "match"} onClick={sort.on("match")}>
                      Match
                    </SortHead>
                    <SortHead
                      active={sort.key === "competition"}
                      onClick={sort.on("competition")}
                    >
                      Competition
                    </SortHead>
                    <TableHead className="text-center">Score</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {visible.map((r) => (
                    <Row
                      key={r.match.id}
                      row={r}
                      on={r.match.id === current || r.match.id === pinned}
                      pinned={r.match.id === pinned}
                      onOpen={open}
                    />
                  ))}
                </TableBody>
              </Table>
              {/* Under the header, not in place of it. */}
              {shown.length === 0 && (
                <p className="text-muted-foreground max-w-[60ch] p-4 text-sm">
                  {rows.length === 0
                    ? "No matches yet. Upload one from the Matches screen and it appears here as soon as the upload starts, before any tracking has run."
                    : "No match matches these filters."}
                </p>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </SectionShell>
  );

  // Inside the database browser the heading and the note come from the rail, so
  // the shell would be a second title over the same table.
  return chrome ? shell : <>{shell.props.filters}{shell.props.children}</>;
}

function Row({
  row,
  on,
  pinned,
  onOpen,
}: {
  row: MatchRow;
  on: boolean;
  pinned?: boolean;
  onOpen: (id: string) => void;
}) {
  const { match, fixture } = row;
  const bar = progressOf(match);
  return (
    <TableRow
      onClick={() => onOpen(match.id)}
      aria-selected={on}
      className={cn("cursor-pointer", on ? "bg-muted/60" : "hover:bg-muted/40")}
    >
      <TableCell className="readout whitespace-nowrap">{matchDate(match) ?? "—"}</TableCell>
      <TableCell className="font-heading whitespace-nowrap">
        <span className="flex items-center gap-2.5">
          {matchTitle(match)}
          {/* WAS a State column reading "Ready" on every row of a finished
              library, with a filter above it. Both gone. What survives is the
              PROGRESS of a match still being tracked — the one thing here a
              fixture list could never show — inline beside the name, so a done
              library carries no column at all rather than a column of identical
              words. A conditional CELL would have given a processing row one
              more cell than the header and pulled the table out of line. */}
          {bar && (
            <span className="flex shrink-0 items-center gap-1.5">
              <span className="h-1 w-16 overflow-hidden rounded-full bg-[var(--card)]">
                <span
                  className="block h-full rounded-full bg-[var(--accent)] transition-[width] duration-500"
                  style={{ width: `${bar.value}%` }}
                />
              </span>
              <span className="readout text-[11px] text-[var(--mute)]">{bar.text}</span>
            </span>
          )}
        </span>
      </TableCell>
      {/* Feed columns. An em dash is not a gap in the data, it is a club that
          does not import a professional feed, which is the normal case. */}
      <TableCell className="text-muted-foreground whitespace-nowrap">
        {fixture?.competition ?? "—"}
      </TableCell>
      <TableCell className="readout text-center whitespace-nowrap">
        {fixture?.score ?? "—"}
      </TableCell>
      <TableCell className="w-8">
        <Chevron open={pinned} />
      </TableCell>
    </TableRow>
  );
}

/** A select whose empty value means "all of them". Radix refuses "" as an item
 *  value, so the all-of-them option carries a sentinel translated on both edges. */
function Choose({
  value,
  onChange,
  all,
  items,
}: {
  value: string;
  onChange: (v: string) => void;
  all: string;
  items: { id: string; label: string }[];
}) {
  const ALL = "~all";
  return (
    <Select value={value || ALL} onValueChange={(v) => onChange(v === ALL ? "" : v)}>
      <SelectTrigger size="sm" className="w-[190px]">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={ALL}>{all}</SelectItem>
        {items.map((i) => (
          <SelectItem key={i.id} value={i.id}>
            {i.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
