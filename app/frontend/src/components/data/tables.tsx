"use client";

import { useMemo } from "react";

import { Card, CardContent } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ChevronRight } from "lucide-react";
import { cmp, SortHead, useSort } from "@/components/sort-head";
import { agg, type Agg, type ClubTotals, type PlayerTotals, type SeasonTable } from "@/lib/season";
import { DATA_HEAD, ZEBRA } from "@/components/data/section-shell";
import { cn } from "@/lib/utils";

/** THE TABLES THE DATA BROWSER RENDERS.
 *
 *  Lifted out of season-tables.tsx unchanged. They were fine; what was wrong
 *  was the page around them — five tabs, each with its own header idiom, its
 *  own prose block and its own stack of two-to-four tables. One database
 *  renders one table at a time, so the tables are now just tables and the
 *  browser decides which one you are looking at.
 */

/** The club columns, in the order a coach reads them. `mean` marks the ones
 *  that are rates — lib/season.ts owns the actual sum-or-mean decision, this is
 *  only how the number is printed. */
const CLUB_COLS: { label: string; head: string; dp?: number }[] = [
  { label: "Ball possession", head: "Poss %", dp: 1 },
  { label: "Expected goals (xG)", head: "xG", dp: 2 },
  { label: "Total shots", head: "Shots" },
  { label: "Shots on target", head: "SoT" },
  { label: "Big chances", head: "Big ch." },
  { label: "Corners", head: "Corners" },
  { label: "Yellow cards", head: "Yel" },
  { label: "Red cards", head: "Red" },
];

const PLAYER_COLS: { label: string; head: string; dp?: number }[] = [
  { label: "minutes", head: "Mins" },
  { label: "goals", head: "Goals" },
  { label: "assists", head: "Assists" },
  { label: "xg", head: "xG", dp: 2 },
  { label: "xa", head: "xA", dp: 2 },
  { label: "rating", head: "Rating", dp: 2 },
  { label: "chances", head: "Chances" },
  { label: "acc_passes", head: "Acc. passes" },
];

function fmt(a: Agg, dp = 0): string {
  if (a.n === 0) return "—";
  return a.value.toFixed(dp);
}

/** A number with the matches behind it. The count only appears when it is
 *  SHORT of the appearances — printing "10/10" on every cell is noise, and
 *  printing nothing on "3/10" is the lie this component exists to avoid. */
function Cell({ a, of, dp = 0 }: { a: Agg; of: number; dp?: number }) {
  const short = a.n > 0 && a.n < of;
  return (
    <TableCell className="text-right">
      <span className={cn("readout", a.n === 0 && "text-muted-foreground")}>{fmt(a, dp)}</span>
      {short && (
        <span
          className="text-muted-foreground ml-1.5 text-[11px]"
          title={`From ${a.n} of ${of} matches — the others did not carry this stat`}
        >
          {a.n}/{of}
        </span>
      )}
    </TableCell>
  );
}

/** The affordance that says a row opens. Rotates when its row is the one
 *  pinned above an open detail, so the control also reads as the way back. */
export function Chevron({ open }: { open?: boolean }) {
  return (
    <ChevronRight
      className={cn(
        "size-3.5 shrink-0 text-[var(--graphite)] transition-transform",
        open && "rotate-90",
      )}
    />
  );
}

/* ------------------------------------------------------------------ tables */

export function ClubTable({
  rows,
  first = "Club",
  caption,
  onPick,
  pinned,
}: {
  rows: ClubTotals[];
  first?: string;
  /** No longer printed under the table — it is the table's accessible name
   *  instead. The `n/N` marker each short cell carries already explains itself
   *  on hover, which is where the footer's one load-bearing sentence went. */
  caption: string;
  /** Given, every row grows a chevron that opens that club's own tables. */
  onPick?: (club: string) => void;
  /** When set, the table renders down to this one row: it is the PIN above an
   *  open detail, so the row you clicked stays on screen with its numbers. */
  pinned?: string | null;
}) {
  const sort = useSort("played");

  const sorted = useMemo(() => {
    const col = CLUB_COLS.find((c) => c.label === sort.key);
    return [...rows].sort((a, b) => {
      if (col) return cmp(agg(a.rows, col.label).value, agg(b.rows, col.label).value, sort.asc);
      if (sort.key === "club") return sort.asc ? a.club.localeCompare(b.club) : b.club.localeCompare(a.club);
      const k = sort.key as "played" | "w" | "gf" | "ga";
      return cmp(a[k], b[k], sort.asc);
    });
  }, [rows, sort.key, sort.asc]);

  // Pinned: the table IS the header of the open detail, so it shows that row
  // and nothing else.
  const shown = pinned ? sorted.filter((c) => c.club === pinned) : sorted;

  return (
    <Card className="py-0 ring-0">
      <CardContent className="grid gap-0 p-0">
        <div className="overflow-x-auto">
          <Table className={cn(ZEBRA, DATA_HEAD)} aria-label={caption}>
            <TableHeader>
              <TableRow>
                <SortHead active={sort.key === "club"} onClick={sort.on("club")}>
                  {first}
                </SortHead>
                <SortHead active={sort.key === "played"} onClick={sort.on("played")} className="text-right">
                  P
                </SortHead>
                <TableHead className="text-right">W-D-L</TableHead>
                <SortHead active={sort.key === "gf"} onClick={sort.on("gf")} className="text-right">
                  GF
                </SortHead>
                <SortHead active={sort.key === "ga"} onClick={sort.on("ga")} className="text-right">
                  GA
                </SortHead>
                {CLUB_COLS.map((c) => (
                  <SortHead
                    key={c.label}
                    active={sort.key === c.label}
                    onClick={sort.on(c.label)}
                    className="text-right whitespace-nowrap"
                  >
                    {c.head}
                  </SortHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {shown.map((c) => (
                <TableRow
                  key={c.club}
                  onClick={onPick ? () => onPick(pinned ? "" : c.club) : undefined}
                  className={cn("hover:bg-muted/40", onPick && "cursor-pointer")}
                >
                  <TableCell className="font-heading whitespace-nowrap">{c.club}</TableCell>
                  <TableCell className="readout text-right">{c.played}</TableCell>
                  <TableCell className="readout text-right whitespace-nowrap">
                    {c.w}-{c.d}-{c.l}
                  </TableCell>
                  <TableCell className="readout text-right">{c.gf}</TableCell>
                  <TableCell className="readout text-right">{c.ga}</TableCell>
                  {CLUB_COLS.map((col) => (
                    <Cell key={col.label} a={agg(c.rows, col.label)} of={c.played} dp={col.dp} />
                  ))}
                  {onPick && (
                    <TableCell className="w-8">
                      <Chevron open={c.club === pinned} />
                    </TableCell>
                  )}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}

export function PlayerTable({ rows }: { rows: PlayerTotals[] }) {
  const sort = useSort("apps");

  const sorted = useMemo(() => {
    const col = PLAYER_COLS.find((c) => c.label === sort.key);
    return [...rows].sort((a, b) => {
      if (col) return cmp(agg(a.rows, col.label).value, agg(b.rows, col.label).value, sort.asc);
      if (sort.key === "name") return sort.asc ? a.name.localeCompare(b.name) : b.name.localeCompare(a.name);
      if (sort.key === "club") return sort.asc ? a.club.localeCompare(b.club) : b.club.localeCompare(a.club);
      return cmp(a.apps, b.apps, sort.asc);
    });
  }, [rows, sort.key, sort.asc]);

  return (
    <Card className="py-0 ring-0">
      <CardContent className="grid gap-0 p-0">
        <div className="overflow-x-auto">
          <Table className={cn(ZEBRA, DATA_HEAD)}>
            <TableHeader>
              <TableRow>
                <SortHead active={sort.key === "name"} onClick={sort.on("name")}>
                  Player
                </SortHead>
                <SortHead active={sort.key === "club"} onClick={sort.on("club")}>
                  Club
                </SortHead>
                <TableHead>Pos</TableHead>
                <SortHead active={sort.key === "apps"} onClick={sort.on("apps")} className="text-right">
                  Apps
                </SortHead>
                {PLAYER_COLS.map((c) => (
                  <SortHead
                    key={c.label}
                    active={sort.key === c.label}
                    onClick={sort.on(c.label)}
                    className="text-right whitespace-nowrap"
                  >
                    {c.head}
                  </SortHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {sorted.map((p) => (
                <TableRow key={`${p.name}|${p.club}`} className="hover:bg-muted/40">
                  <TableCell className="font-heading whitespace-nowrap">{p.name}</TableCell>
                  <TableCell className="text-muted-foreground whitespace-nowrap">{p.club}</TableCell>
                  <TableCell className="text-muted-foreground whitespace-nowrap">{p.pos ?? "—"}</TableCell>
                  <TableCell className="readout text-right">{p.apps}</TableCell>
                  {PLAYER_COLS.map((col) => (
                    <Cell key={col.label} a={agg(p.rows, col.label)} of={p.apps} dp={col.dp} />
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}

export function MatchTable({ table }: { table: SeasonTable }) {
  return (
    <Card className="py-0 ring-0">
      <CardContent className="grid gap-0 p-0">
        <div className="overflow-x-auto">
          <Table className={cn(ZEBRA, DATA_HEAD)}>
            <TableHeader>
              <TableRow>
                <TableHead>Date</TableHead>
                <TableHead>Competition</TableHead>
                <TableHead>Home</TableHead>
                <TableHead className="text-center">Score</TableHead>
                <TableHead>Away</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {table.fixtures.map((f) => (
                <TableRow key={f.id} className="hover:bg-muted/40">
                  <TableCell className="readout whitespace-nowrap">
                    {f.date ? f.date.slice(0, 10) : "—"}
                  </TableCell>
                  <TableCell className="text-muted-foreground whitespace-nowrap">
                    {f.competition ?? "—"}
                  </TableCell>
                  <TableCell className="font-heading whitespace-nowrap">{f.home}</TableCell>
                  <TableCell className="readout text-center whitespace-nowrap">
                    {f.score ?? "—"}
                  </TableCell>
                  <TableCell className="font-heading whitespace-nowrap">{f.away}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}
