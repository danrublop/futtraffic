"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeftRight, Bolt, ChevronLeft, ChevronRight, Star, Timer } from "lucide-react";

import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cmp, useSort } from "@/components/sort-head";
import { LoadFailure } from "@/components/data/load-state";
import { PLAYER_STAT_COLS, PLAYER_STAT_TABS } from "@/lib/player-cols";
import {
  EditableFacts,
  canSave,
  draftOf,
  savePlayer,
  type PlayerDraft,
} from "@/components/data/player-editor";
import { useOrg } from "@/lib/club";
import { useSeason } from "@/hooks/use-matches";
import { api } from "@/lib/api";
import { scoreOf } from "@/lib/season";
import type { Appearance, ClubPlayer, PlayerBio, Season } from "@/lib/types";
import { RatingChip } from "@/components/rating-chip";
import { DATA_TABLE } from "@/components/data/section-shell";
import { Fact, longDate } from "@/components/data/player-fact";
import { cn } from "@/lib/utils";

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="text-muted-foreground p-4 text-sm">{children}</p>;
}

/* ---------------------------------------------------------------------------
 * THE SQUAD HUB, over a real club.
 *
 * Built from FIFA's career-mode player database - its STATUS screen for the
 * shell and its STATS screen for the numbers, merged into one table because
 * there is no reason to make somebody tab between "who is this" and "what did
 * they do". What the game splits across two views is one question here.
 *
 * WHAT IS TAKEN, all of it layout, because layout is what makes that screen
 * recognisable: the shirt number and position rail on the left; the round face
 * at the head of the name; a dense zebra list; the cursor row INVERTED to a
 * pale band rather than tinted; the detail panel beside the list with one
 * headline number set large in its top-right; and the panel's per-competition
 * matrix, which lands here as per-MATCH with a Totals row under a rule.
 *
 * WHAT IS DROPPED, because it is a video game's model of a footballer and this
 * is a club's: Morale, Form, Fitness, Sharpness, Availability, Squad Role,
 * Contract, Player Performance. None of them exist outside a save file. There
 * is no honest column here for "Content" with a yellow face.
 *
 * OVR is dropped too, and REPLACED rather than deleted: the corner it occupied
 * now holds the feed's average match rating. That is the closest real analogue
 * - a number somebody assigns to a performance - and unlike an OVR it was not
 * invented here. It stays plain ink rather than FIFA's green-to-amber ramp,
 * because a rating is an ENTERED opinion and colouring it green would make
 * this product endorse a provider's verdict. globals.css is explicit that
 * colour here claims how far a number can be trusted, not whether it is good.
 *
 * MEASURED AND ENTERED STAY APART, which is the one thing the merge could not
 * blur. Goals, assists, minutes and rating are the feed's. Distance and km/90
 * are the pipeline's, off confirmed tracks only. They disagree - the feed has
 * Joao Pedro at 851 minutes and the tracker at 900, because an imported
 * fixture's tracking is synthetic and every synthetic track runs the full
 * match - so the table shows the FEED's minutes, the panel labels each block
 * with where it came from, and nothing sums across the line.
 * ------------------------------------------------------------------------- */

type Group = "GK" | "DEF" | "MID" | "ATT";

/** The feed's position words, folded onto FIFA's four rails. `roster.position`
 *  is free text a coach types, so this matches on a stem and falls through to
 *  MID rather than inventing a fifth group for a typo. */
export function group(pos: string | null | undefined): Group {
  const p = (pos ?? "").toLowerCase();
  if (p.startsWith("goal") || p === "gk") return "GK";
  if (p.startsWith("def") || /^(cb|lb|rb|lwb|rwb)$/.test(p)) return "DEF";
  if (
    p.startsWith("att") || p.startsWith("for") || p.startsWith("str") ||
    /^(cf|st|lw|rw)$/.test(p)
  ) {
    return "ATT";
  }
  return "MID";
}

/** Categorical, not ordinal. Chart tokens rather than literals so the badge
 *  follows the theme ladder into light mode; GK amber / MID green / ATT blue
 *  happens to land close to the game's own rails. */
const TINT: Record<Group, string> = {
  GK: "var(--chart-2)",
  DEF: "var(--chart-4)",
  MID: "var(--chart-5)",
  ATT: "var(--chart-1)",
};

const ORDER: Group[] = ["GK", "DEF", "MID", "ATT"];

/** THE ONE COLUMN TEMPLATE the Seasons tab runs on: the run's name, then
 *  appearances, minutes, goals, assists, yellow, red, rating. A season row and
 *  the match rows nested under it both wear it, which is what lets the list
 *  carry a single header. */
const RUN_COLS =
  "grid grid-cols-[minmax(0,1fr)_repeat(6,1.9rem)_2.9rem] items-center gap-x-1 text-center";

/** The profile's tabs, split by the QUESTION each answers rather than by data
 *  source: who is he, where does he play, and what did he do each season and
 *  each game - the last two also scope the table. */
const TABS = ["overview", "visuals", "seasons"] as const;

/** One of the app's own SVGs as a header glyph, painted in the TEXT colour
 *  through a mask - so it takes the same grey as the lucide icons beside it,
 *  where an inverted <img> came out a pale yellow. */
function MaskIcon({ src, label, className }: { src: string; label: string; className?: string }) {
  const m = `url(${src}) center / contain no-repeat`;
  return (
    <span
      role="img"
      aria-label={label}
      className={cn("mx-auto block bg-current", className)}
      style={{ mask: m, WebkitMask: m }}
    />
  );
}

/** Goals: the report's ball. Its drawing fills less of its box than a lucide
 *  glyph does, hence the larger box for the same apparent size. */
export const BallIcon = ({ label, className = "size-[19px]" }: { label: string; className?: string }) => (
  <MaskIcon src="/soccer-ball.svg" label={label} className={className} />
);

/** Appearances: the arena mark the Matches tab and the sidebar use. */
const MatchesIcon = ({ label }: { label: string }) => (
  <MaskIcon src="/soccer-arena.svg" label={label} className="size-4" />
);

/** Assists: a boot, drawn in lucide's own terms (24 box, 1.75 stroke, round
 *  joins) because lucide has none. */
export function BootIcon({ label, className = "mx-auto size-4" }: { label: string; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" role="img" aria-label={label} fill="none" stroke="currentColor"
         strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M4 5h5l1.5 4.5 6.8 2.1A3 3 0 0 1 20 14.5V16H4Z" />
      <path d="M6 16v2.5M10 16v2.5M14.5 16v2.5M18.5 16v2.5" />
      <path d="m10.5 10.6 1.2-1.1M12.9 11.3l1.2-1.1" />
    </svg>
  );
}

/** A TABLE'S SHAPE BEFORE ITS DATA: the header rule, then rows with a face,
 *  a name and a run of cells, each a placeholder - so the first load shows
 *  the table it will be and the numbers arrive into it, not a blank that
 *  becomes a table. */
export function TableSkeleton({ rows = 12, cols = 8 }: { rows?: number; cols?: number }) {
  const grid = { gridTemplateColumns: `minmax(0,1fr) repeat(${cols}, 3rem)` };
  return (
    <div aria-busy="true" aria-label="Loading">
      <div className="grid items-end gap-2 border-b border-[var(--grid-line)] px-3 py-3" style={grid}>
        <Skeleton className="h-3 w-16" />
        {Array.from({ length: cols }, (_, i) => (
          <Skeleton key={i} className="mx-auto h-3 w-8" />
        ))}
      </div>
      {Array.from({ length: rows }, (_, r) => (
        <div key={r} className="grid items-center gap-2 px-3 py-2.5" style={grid}>
          <span className="flex items-center gap-2.5">
            <Skeleton className="size-6 shrink-0 rounded-full" />
            <Skeleton className="h-3 w-28" />
          </span>
          {Array.from({ length: cols }, (_, i) => (
            <Skeleton key={i} className="mx-auto h-3 w-7" />
          ))}
        </div>
      ))}
    </div>
  );
}

/** A ROW THE KEYBOARD CAN REACH. Every table here opens its side panel by
 *  clicking a row, which is the whole interaction, and a bare `onClick` on a
 *  `<tr>` leaves that unreachable without a mouse. Enter and Space both
 *  activate - Space with preventDefault, or the page scrolls under the panel
 *  that just opened.
 *
 *  `aria-current`, not `aria-selected`: the latter is only defined for rows
 *  inside a grid, so on a plain table it announces nothing at all. */
export function pickable(on: boolean, choose: () => void, extra?: string) {
  return {
    tabIndex: 0,
    role: "button" as const,
    "aria-current": on ? ("true" as const) : undefined,
    onClick: choose,
    onKeyDown: (ev: React.KeyboardEvent) => {
      if (ev.key !== "Enter" && ev.key !== " ") return;
      ev.preventDefault();
      choose();
    },
    className: cn(
      "cursor-pointer transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[var(--ink-soft)]",
      extra,
      // A FAINT LIFT, on hover and on selection both, as the report's table
      // does it - the row stays dark and its text stays as it is. The bang is
      // needed: TableRow ships `hover:bg-muted/50` at the same specificity and
      // later in the sheet.
      on ? "!bg-[var(--card-hover)]" : "hover:!bg-[var(--card-hover)]",
    ),
  };
}

/** TOTALS OR PER 90, as ONE button that says which it is showing and flips
 *  on a click. The same control on every table here - a two-part switch took
 *  the width of three pills and pushed itself onto a second line. */
export function Per90Toggle({ on, onToggle }: { on: boolean; onToggle: () => void }) {
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-pressed={on}
      title={on ? "Showing per 90 - click for totals" : "Showing totals - click for per 90"}
      className="ml-auto flex items-center gap-1 rounded-full border border-[var(--hairline-strong)] px-2 py-1.5 text-[13px] font-medium whitespace-nowrap transition-colors hover:bg-[var(--card-hover)]"
    >
      {on ? "Per 90" : "Total"}
      <ArrowLeftRight className="text-muted-foreground size-3" />
    </button>
  );
}

/** The report's category pill, for the table's stat tabs. */
export function Pill({
  on,
  onClick,
  children,
}: {
  on: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={on}
      className={cn(
        "rounded-full px-2.5 py-1.5 text-[13px] font-medium transition-colors",
        on ? "bg-foreground text-background" : "bg-muted text-foreground/80 hover:text-foreground",
      )}
    >
      {children}
    </button>
  );
}
type Tab = (typeof TABS)[number];

/** The rail, as PLAIN TYPE. It was a filled pill, which put 23 competing
 *  rectangles down the left edge of a table whose job is to be scanned - the
 *  single busiest thing on the screen, and the game does not do it: FIFA's
 *  Pos column is ordinary text in a slightly lifted band. The hue survives on
 *  the glyphs, where it still separates the rails at a glance and paints
 *  nothing. `filled` is for the panel, where there is one of them and a solid
 *  chip reads as an identity badge rather than as noise. */
function PosBadge({ pos, filled = false }: { pos: string | null; filled?: boolean }) {
  const g = group(pos);
  return (
    <span
      title={pos ?? "Position not recorded"}
      className={cn(
        "readout inline-grid place-items-center text-[10px] leading-none font-medium tracking-wide",
        filled && "h-[18px] min-w-[34px] rounded-[var(--radius-xs)] px-1",
      )}
      style={{
        color: TINT[g],
        background: filled ? `color-mix(in oklab, ${TINT[g]} 14%, transparent)` : undefined,
      }}
    >
      {pos == null ? "—" : g}
    </span>
  );
}

/** A player's season as the FEED reported it. Everything here is entered. */
type Entered = {
  app: number;
  mins: number;
  goals: number;
  assists: number;
  /** Mean of the per-match ratings that HAVE one, or null if none do. Not a
   *  zero-filled mean: a match the provider left unrated would drag an average
   *  down as though the player had been terrible in it. */
  rating: number | null;
  /** Cards over the run. On Entered so a season row carries the same columns
   *  as the matches nested under it - one header for both. */
  yellow: number;
  red: number;
  age: number | null;
  shirt: number | null;
  /** null for an outfielder - the feed files no keeper metrics for them - and
   *  a number, possibly 0, for anyone who kept goal. */
  saves: number | null;
  cleanSheets: number | null;
};

/** A clean sheet needs a SUBSTANTIAL appearance, not just a zero.
 *
 *  `conceded == 0` alone reported five for one keeper's ten matches when
 *  the club only kept four. The extra one is a FIVE-MINUTE appearance: he came
 *  on at the death of a game another keeper had already conceded in, his own
 *  `conceded` is 0 for those five minutes, and counting it credits him with a
 *  shutout he had no part in. The feed's per-appearance conceded sums to 9
 *  across the run where the scorelines say 11, which is the same gap seen from
 *  the other end.
 *
 *  Sixty minutes is the bar most providers use and it reproduces the right
 *  answer here exactly - four. It is a THRESHOLD and therefore arguable, which
 *  is why it is a named constant with the number it was checked against rather
 *  than a bare 60 inline.
 */
const CLEAN_SHEET_MINUTES = 60;

export function enteredOf(rows: Appearance[]): Entered {
  const rated = rows.filter((r) => r.rating != null);
  return {
    app: rows.length,
    mins: rows.reduce((t, r) => t + (r.minutes ?? 0), 0),
    goals: rows.reduce((t, r) => t + (r.goals ?? 0), 0),
    assists: rows.reduce((t, r) => t + (r.assists ?? 0), 0),
    rating: rated.length
      ? rated.reduce((t, r) => t + (r.rating as number), 0) / rated.length
      : null,
    yellow: rows.reduce((t, r) => t + (r.yellow ?? 0), 0),
    red: rows.reduce((t, r) => t + (r.red ?? 0), 0),
    // The team sheet carries these per match and they can drift over a season
    // (a birthday, a change of shirt). The LATEST is the current fact, and the
    // rows arrive grouped by match id rather than by date, so this takes the
    // max rather than the last seen.
    age: rows.reduce<number | null>((a, r) => (r.age == null ? a : Math.max(a ?? 0, r.age)), null),
    shirt: rows.reduce<number | null>((s, r) => (r.shirt == null ? s : r.shirt), null),
    // NULL-CHECKED, not truthy-checked: a keeper with a quiet game has
    // `saves: 0`, which is a fact, and `!r.saves` would file him as an
    // outfielder for it.
    saves: rows.some((r) => r.saves != null)
      ? rows.reduce((t, r) => t + (r.saves ?? 0), 0)
      : null,
    cleanSheets: rows.some((r) => r.conceded != null)
      ? rows.filter((r) => r.conceded === 0 && (r.minutes ?? 0) >= CLEAN_SHEET_MINUTES).length
      : null,
  };
}

/** THE ROUND FACE at the head of the name - the one part of the game's list
 *  this product could not draw until recently.
 *
 *  It reads the photo store: headshots.py fills it from Wikimedia lead images
 *  under CC-BY-SA or public domain, and /api/players/{id}/photo serves what it
 *  found. Coverage is partial by nature - not every player has an article with
 *  a lead image - so `photo` gates the request and `onError` catches the row
 *  whose file went away between the list and the render. Both fall back to the
 *  silhouette, which still says the true thing: no picture of this person.
 *
 *  The credit does NOT travel with this thumbnail, and that is only defensible
 *  because the licence attaches to REDISTRIBUTION: a 26px avatar on the club's
 *  own screen is not a mailed report. /api/players/photo-credits is where the
 *  attribution lives, and anything that exports a face has to print it.
 */
export function Portrait({
  id,
  name,
  size,
  photo,
}: {
  id?: string;
  name: string;
  size: number;
  photo?: boolean;
}) {
  const [failed, setFailed] = useState(false);
  const real = Boolean(id) && photo !== false && !failed;

  return (
    <span
      style={{ width: size, height: size }}
      /* NO RING. A hairline around every face is another 23 outlines on a
         screen that already has banding and a header rule; the game's faces
         sit in the row unbordered. The card fill still reads as a disc behind
         a silhouette, which is the only case the ring was helping. */
      className="grid shrink-0 place-items-end overflow-hidden rounded-full bg-[var(--card)]"
    >
      {real ? (
        /* eslint-disable-next-line @next/next/no-img-element -- our own origin,
           one fixed size; the optimiser adds a hop for a 5 KB thumbnail. */
        <img
          src={`/api/players/${encodeURIComponent(id as string)}/photo`}
          alt=""
          aria-hidden
          title={name}
          onError={() => setFailed(true)}
          className="size-full object-cover object-top"
        />
      ) : (
        /* eslint-disable-next-line @next/next/no-img-element -- local static */
        <img
          src="/player.svg"
          alt=""
          aria-hidden
          title={name}
          className="h-[86%] w-auto opacity-60 dark:invert-[0.85]"
        />
      )}
    </span>
  );
}

/** A sortable header, the match report's way: the label, and a small round
 *  sort button on the column actually sorted - one indicator, not one per
 *  column - sitting ON the header's bottom rule, half above and half below,
 *  with the lit triangle saying which way. Opaque, so the rule passes behind
 *  it. Hover shades the header. Its own cell rather than the shared SortHead,
 *  whose arrow sits inline beside the label. */
export function Head({
  k,
  sort,
  className,
  children,
}: {
  k: string;
  sort: ReturnType<typeof useSort>;
  className?: string;
  children: React.ReactNode;
}) {
  const active = sort.key === k;
  return (
    <TableHead className={cn("relative transition-colors hover:bg-[var(--card-hover)]", className)}>
      <button
        type="button"
        onClick={sort.on(k)}
        className={cn("transition-colors hover:text-[var(--ink)]", active && "text-[var(--ink)]")}
      >
        {children}
      </button>
      {active && (
        <button
          type="button"
          onClick={sort.on(k)}
          aria-label={sort.asc ? "Sorted lowest first" : "Sorted highest first"}
          className="absolute bottom-0 left-1/2 z-10 grid size-5 -translate-x-1/2 translate-y-1/2 place-items-center rounded-full border border-[var(--grid-line-strong)] bg-[var(--panel)] text-[var(--ink)]"
        >
          <svg viewBox="0 0 10 12" className="size-2.5" aria-hidden>
            <path d="M5 0 10 5H0Z" fill="currentColor" opacity={sort.asc ? 1 : 0.4} />
            <path d="M5 12 10 7H0Z" fill="currentColor" opacity={sort.asc ? 0.4 : 1} />
          </svg>
        </button>
      )}
    </TableHead>
  );
}

/** A count that is genuinely zero, rendered as a dim zero rather than a bold
 *  one. A defender with no goals is not missing data - which is what an em
 *  dash would say - but it is also not a fact worth the same weight as a
 *  striker's 14, and a column of confident zeroes is what makes FIFA's stats
 *  page unreadable in preseason. */
function Count({ v, dp = 0, className }: { v: number | undefined; dp?: number; className?: string }) {
  return (
    <TableCell className={cn("readout text-right", className)}>
      {v == null ? (
        <span className="text-muted-foreground">—</span>
      ) : v === 0 ? (
        <span className="text-muted-foreground">0</span>
      ) : (
        v.toFixed(dp)
      )}
    </TableCell>
  );
}

/** The expected-goals columns, the only fractional ones in the report's set. */
const DECIMAL_COLS = new Set(["xg", "xa", "xgxa", "xgot_faced", "goals_prevented"]);

/* ------------------------------------------------------------- the bio card */


/** THE PITCH, and the positions he is actually played in.
 *
 *  Drawn rather than fetched: the standard markings in metres, and the feed's
 *  own coordinates on top - `right` and `top` as fractions, main position in
 *  blue, the rest grey. Positions the feed gives no coordinates for are NOT
 *  dropped silently; they are listed in the facts beside it.
 *
 *  It is a DRAWING and not a measurement, the same caveat appearance.plot_x
 *  carries in the schema: this is where a provider files him, not a mean of
 *  anything the tracker saw.
 */
function PositionPitch({ bio }: { bio: PlayerBio }) {
  const dots = (bio.positions ?? []).filter((p) => p.x != null && p.y != null);
  // Real markings in real metres, goal at top and bottom - the same approach
  // as the shot map: a 68 x 105 pitch, 40.32 x 16.5 boxes, 18.32 x 5.5 six-
  // yard boxes, 9.15 m circle and arcs. Mixed rather than translucent so a
  // line over the fill is one flat colour.
  const line = "color-mix(in oklab, var(--ink) 16%, var(--card))";
  return (
    <div className="relative aspect-[68/105] w-full rounded-[var(--radius-sm)] bg-[color-mix(in_oklab,var(--ink)_8%,var(--card))]">
      <svg viewBox="0 0 68 105" className="absolute inset-0 size-full" aria-hidden>
        <g fill="none" stroke={line} strokeWidth="0.8">
          <line x1="0" y1="52.5" x2="68" y2="52.5" />
          <circle cx="34" cy="52.5" r="9.15" />
          <path d="M13.84 0V16.5H54.16V0M24.84 0V5.5H43.16V0M26.69 16.5A9.15 9.15 0 0 0 41.31 16.5" />
          <path d="M13.84 105V88.5H54.16V105M24.84 105V99.5H43.16V105M26.69 88.5A9.15 9.15 0 0 1 41.31 88.5" />
        </g>
      </svg>
      {dots.map((d) => (
        <span
          key={d.label ?? `${d.x}-${d.y}`}
          title={`${d.label}${d.played ? ` — ${d.played} appearances` : ""}`}
          style={{ left: `${(d.x as number) * 100}%`, top: `${(d.y as number) * 100}%` }}
          className={cn(
            "absolute -translate-x-1/2 -translate-y-1/2 rounded-full px-2 py-1 text-[10px] leading-none font-semibold",
            d.main
              ? "bg-[color-mix(in_oklab,var(--accent-strong)_45%,black)] text-white"
              : "bg-[color-mix(in_oklab,var(--ink)_45%,var(--card))] text-[var(--card)]",
          )}
        >
          {d.short ?? d.label}
        </span>
      ))}
    </div>
  );
}

/** The feed's CamelCase enums, as English. "LeftFoot" and "FromCorner" are
 *  identifiers, not labels, and printing them raw is showing the reader the
 *  wire format. Split on the case boundary and lower everything after the
 *  first word, so an unmapped value the feed adds later still reads. */
export function humanise(v?: string | null): string | null {
  if (!v) return null;
  const words = v.replace(/([a-z0-9])([A-Z])/g, "$1 $2").split(" ");
  return words[0] + words.slice(1).map((w) => ` ${w.toLowerCase()}`).join("");
}

type Shot = NonNullable<PlayerBio["shots"]>[number];

/** The dropdown's no-filter value; Radix's Select refuses an empty string. */
const ALL_SHOTS = "all";

export const resultOf = (sh: Shot) =>
  sh.goal ? "Goal" : sh.on_target ? "Saved" : sh.blocked ? "Blocked" : "Off target";

/** Inside the 18-yard box: 16.5 m from the goal line, 20.16 m either side of
 *  the centre (y 34). */
export const inBox = (sh: Shot) => sh.x >= 105 - 16.5 && Math.abs(sh.y - 34) <= 20.16;

/** THE SEASON SHOT MAP, laid out as the reference: the map and its three
 *  totals, then one shot at a time with its detail.
 *
 *  THE ATTACKING THIRD ONLY, goal at the bottom: a whole pitch is 90% empty
 *  grass. x from 63 to 105 metres holds every shot this squad took.
 *
 *  WHICH WAY ROUND, checked rather than assumed. The feed's y increases toward
 *  the attacker's LEFT (mean shot y 41-44 for the left wingers, 29-30 for the
 *  right), so attacking down the page puts a left-sided player on the RIGHT of
 *  the image. That is geometry, not a bug, and the reference does it too.
 *
 *  ONE MARK PER SHOT, the same size, as the reference draws them: an outline
 *  for a miss or a block, filled for a save, a ball for a goal. The selected
 *  shot is ringed in red, with a line to where it crossed the goal line when
 *  the feed has that point.
 *
 *  THE FILTER is a dropdown of the feed's own categories, counted, one at a
 *  time. The totals, the map and the stepper all follow it.
 */
function ShotMap({
  bio,
  days,
  title,
}: {
  bio: PlayerBio;
  /** The days of the matches in scope; null for the whole feed season. */
  days: Set<string> | null;
  title: string;
}) {
  const [filter, setFilter] = useState<string | null>(null);
  const [at, setAt] = useState(0);

  // Newest first, so the stepper opens on the latest shot. SCOPED by the
  // page's filters: a shot carries the day it was taken, so a season or a
  // match narrows the map to the matches in it. The feed's season is wider
  // than the library - every competition the player appeared in - so under a
  // season this is the library's matches in it, not the feed's.
  const all = useMemo(
    () =>
      [...(bio.shots ?? [])].filter((sh) => !days || (sh.match_date != null && days.has(sh.match_date))).sort(
        (a, b) =>
          (b.match_date ?? "").localeCompare(a.match_date ?? "") || (b.min ?? 0) - (a.min ?? 0),
      ),
    [bio.shots, days],
  );

  const filters = useMemo(() => {
    const by = (field: "situation" | "foot") =>
      [...new Set(all.map((s) => s[field]).filter((v): v is string => !!v))].map((v) => ({
        id: `${field}:${v}`,
        label: humanise(v) ?? v,
        test: (s: Shot) => s[field] === v,
      }));
    return [
      { id: "goals", label: "Goals", test: (s: Shot) => s.goal },
      ...by("situation"),
      ...by("foot"),
      { id: "in", label: "Shots inside box", test: inBox },
      { id: "out", label: "Shots outside box", test: (s: Shot) => !inBox(s) },
    ]
      .map((f) => ({ ...f, n: all.filter(f.test).length }))
      .filter((f) => f.n > 0);
  }, [all]);

  // No shots is an empty map with zeroed totals, not a sentence: the layout
  // stays the same shape as a player who has them.
  const active = filters.find((f) => f.id === filter) ?? null;
  const shots = active ? all.filter(active.test) : all;
  const i = Math.min(at, shots.length - 1);
  const sel = shots[i] ?? null;
  const goals = shots.filter((s) => s.goal).length;
  const onT = shots.filter((s) => s.on_target).length;
  const xg = shots.reduce((t, s) => t + (s.xg ?? 0), 0);

  const TOP = 63; // metres from the far goal line; the map starts here
  const H = 105 - TOP;
  const pitch = "color-mix(in oklab, var(--ink) 8%, var(--card))";
  const line = "color-mix(in oklab, var(--ink) 16%, var(--card))";

  /** One shot's mark. Goals are a ball. */
  const mark = (sh: Shot, cx: number, cy: number, r: number, k: number) => {
    const pick = () => setAt(shots.indexOf(sh));
    const title = [resultOf(sh), sh.opponent ? `v ${sh.opponent}` : null,
      sh.min != null ? `${sh.min}'` : null, `xG ${(sh.xg ?? 0).toFixed(2)}`]
      .filter(Boolean).join(" · ");
    return sh.goal ? (
      <g key={k} onClick={pick} className="cursor-pointer">
        <title>{title}</title>
        <circle cx={cx} cy={cy} r={r * 1.05} fill="white" />
        <image href="/soccer-ball.svg" x={cx - r * 1.45} y={cy - r * 1.45}
               width={r * 2.9} height={r * 2.9} />
      </g>
    ) : (
      <circle key={k} cx={cx} cy={cy} r={r} onClick={pick} className="cursor-pointer"
              fill={sh.on_target ? "var(--chart-1)" : "transparent"}
              fillOpacity={sh.on_target ? 0.7 : 1}
              stroke="var(--chart-1)" strokeWidth={r * 0.25}>
        <title>{title}</title>
      </circle>
    );
  };

  // THE DRAWING TAKES WHAT IS LEFT. Everything else here has a fixed height,
  // so the map is the part that scales to the window, and the tab never
  // scrolls to reach the shot detail under it.
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <h3 className="font-heading min-w-0 text-base font-medium">{title}</h3>
        {/* THE FILTER, as a dropdown: the feed's own categories, counted. */}
        <Select
          value={filter ?? ALL_SHOTS}
          onValueChange={(v) => {
            setFilter(v === ALL_SHOTS ? null : v);
            setAt(0);
          }}
        >
          <SelectTrigger size="sm" className="w-[190px]" aria-label="Filter shots" disabled={!all.length}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ALL_SHOTS}>All shots {all.length}</SelectItem>
            {filters.map((f) => (
              <SelectItem key={f.id} value={f.id}>
                {f.label} {f.n}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* THE MAP ON THE LEFT, its totals in a column beside it. The row is
          only as tall as the map at its own aspect, so the shot detail sits
          right under it; on a short window the row shrinks and the map with
          it (max-h-full), so the tab still never scrolls. */}
      <div className="flex min-h-0 items-center gap-5">
      <svg viewBox={`0 0 68 ${H}`} preserveAspectRatio="xMinYMid meet"
           className="aspect-[68/42] max-h-full min-w-0 flex-1" role="img"
           aria-label={`${shots.length} shots, ${goals} goals, ${xg.toFixed(2)} expected goals`}>
        <rect x="0.4" y="0.4" width="67.2" height={H - 0.8} rx="2.5" fill={pitch}
              stroke={line} strokeWidth="0.6" />
        {/* Real markings in real metres: an 18-yard box 40.32 by 16.5, a
            six-yard box 18.32 by 5.5, the arc 9.15 m from the spot. */}
        <g fill="none" stroke={line} strokeWidth="0.6">
          <path d={`M13.84 ${H}V${H - 16.5}H54.16V${H}`} />
          <path d={`M24.84 ${H}V${H - 5.5}H43.16V${H}`} />
          <path d={`M26.69 ${H - 16.5}A9.15 9.15 0 0 1 41.31 ${H - 16.5}`} />
        </g>
        <rect x="30.34" y={H - 0.7} width="7.32" height="0.7" fill="var(--mute)" />
        {sel && sel.goal_y != null && (
          <line x1={sel.y} y1={sel.x - TOP} x2={sel.goal_y} y2={H}
                stroke="var(--bad)" strokeWidth="0.3" className="pointer-events-none" />
        )}
        {shots.map((sh, k) => mark(sh, sh.y, sh.x - TOP, 1.2, k))}
        {sel && (
          <circle cx={sel.y} cy={sel.x - TOP} r="1.9" fill="none" stroke="var(--bad)"
                  strokeWidth="0.35" className="pointer-events-none" />
        )}
      </svg>
        {/* Two by two, so the column is never taller than a small map. */}
        <dl className="grid shrink-0 grid-cols-2 gap-x-4 gap-y-3 text-center">
          {[
            [String(shots.length), "Shots"],
            [String(goals), "Goals"],
            [xg.toFixed(2), "xG"],
            [`${shots.length ? Math.round((onT / shots.length) * 100) : 0}%`, "On target"],
          ].map(([v, l]) => (
            <Cell key={l} v={v} l={l} readout />
          ))}
        </dl>
      </div>

      {/* THE SELECTED SHOT, in the same cells as the totals - value over
          label. The arrows walk the shots, newest first. */}
      <div className="grid gap-3 border-t pt-3">
        {/* Drawn with no shot too - dashes and dead arrows - so the tab keeps
            its height. */}
        <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setAt(Math.min(shots.length - 1, i + 1))}
              disabled={i >= shots.length - 1}
              aria-label="Earlier shot"
              className="grid size-7 shrink-0 place-items-center rounded-full bg-[var(--card-hover)] text-[var(--ink)] transition-opacity disabled:opacity-30"
            >
              <ChevronLeft className="size-4" />
            </button>
            <div className="min-w-0 flex-1">
              <p className="text-muted-foreground truncate text-center text-xs">
                {sel
                  ? [sel.opponent ? `v ${sel.opponent}` : null, sel.match_date,
                     sel.min != null ? `${sel.min}'` : null, `${i + 1} of ${shots.length}`]
                      .filter(Boolean).join(" · ")
                  : "No shots recorded"}
              </p>
              <dl className="mt-1.5 grid grid-cols-3 text-center">
                <Cell v={(sel && humanise(sel.foot)) ?? "—"} l="Shot type" />
                <Cell v={(sel && humanise(sel.situation)) ?? "—"} l="Situation" />
                <Cell v={sel ? resultOf(sel) : "—"} l="Result" />
              </dl>
            </div>
            <button
              type="button"
              onClick={() => setAt(Math.max(0, i - 1))}
              disabled={i <= 0}
              aria-label="Later shot"
              className="grid size-7 shrink-0 place-items-center rounded-full bg-[var(--card-hover)] text-[var(--ink)] transition-opacity disabled:opacity-30"
            >
              <ChevronRight className="size-4" />
            </button>
          </div>
      </div>
    </div>
  );
}

/** One of the shot map's six cells: the value over what it is. */
function Cell({ v, l, readout = false }: { v: string; l: string; readout?: boolean }) {
  return (
    <div className="min-w-0">
      <dd className={cn("truncate text-base", readout && "readout")}>{v}</dd>
      <dt className="text-muted-foreground text-xs">{l}</dt>
    </div>
  );
}


/** THE HEATMAP - where a player's touches actually were.
 *
 *  BINNED, then blurred, rather than drawn as a thousand overlapping blobs.
 *  A real heatmap is a kernel density estimate; a grid of counts under a
 *  Gaussian blur is the same thing computed in the order that a browser is
 *  good at, and at this size the two are indistinguishable. It also stays fast
 *  with Chalobah's 2,980 points, where a circle per touch would not.
 *
 *  NORMALISED TO A HIGH PERCENTILE, NOT THE MAXIMUM. One cell where a
 *  full-back stood for throw-ins can hold several times the count of anywhere
 *  else, and dividing by it prints an almost-empty pitch with a single red dot
 *  - technically the density and useless as a picture of where he played.
 *  Clamping at the 95th percentile of occupied cells keeps the shape.
 *
 *  The viewBox is the PITCH, 105x68 metres, because that is the unit the feed
 *  sends: no rescaling anywhere, and the markings land where they belong by
 *  construction rather than by fitting.
 */
function Heatmap({ bio }: { bio: PlayerBio }) {
  // Depend on `bio.heatmap` itself, not on a `?? []` default: the fallback is a
  // NEW empty array every render, so the memo below would rebin 2,980 points on
  // every paint for a player who has none.
  const grid = useMemo(() => {
    const pts = bio.heatmap ?? [];
    if (!pts.length) return null;
    const NX = 21;
    const NY = 14;
    const cells = new Float32Array(NX * NY);
    for (const p of pts) {
      const ix = Math.min(NX - 1, Math.max(0, Math.floor((p.x / 105) * NX)));
      const iy = Math.min(NY - 1, Math.max(0, Math.floor((p.y / 68) * NY)));
      cells[iy * NX + ix] += 1;
    }
    const hot = [...cells].filter((v) => v > 0).sort((a, b) => a - b);
    const p95 = hot[Math.floor(hot.length * 0.95)] || 1;
    return { NX, NY, cells, top: Math.max(1, p95), n: pts.length };
  }, [bio.heatmap]);

  // Green -> yellow -> red, the reference's ramp, with the low end fading to
  // nothing so an untouched pitch stays pitch-coloured rather than green.
  const ramp = (t: number): string => {
    if (t <= 0) return "rgba(0,0,0,0)";
    const c = Math.min(1, t);
    const [r, g] =
      c < 0.5 ? [Math.round(2 * c * 255), 200] : [255, Math.round(200 - (c - 0.5) * 2 * 200)];
    return `rgba(${r},${g},60,${(0.22 + 0.78 * c).toFixed(2)})`;
  };

  // ponytail: no points is an empty pitch, not a message - the grid below is
  // sized the same either way, so the picture stays put once the model fills it.
  const { NX, NY, cells, top, n } = grid ?? {
    NX: 21,
    NY: 14,
    cells: new Float32Array(0),
    top: 1,
    n: 0,
  };
  const cw = 105 / NX;
  const ch = 68 / NY;

  return (
    <>
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="font-heading text-base font-medium">Heatmap</h3>
        <p className="text-muted-foreground truncate text-sm">
          {bio.touches != null ? `Touches: ${bio.touches.toLocaleString()} · ` : ""}
          {bio.heatmap_season}
          {bio.heatmap_matches != null ? ` · ${bio.heatmap_matches} matches` : ""}
        </p>
      </div>
      {/* Scales into the space Overview leaves it, pinned to the top; the
          pitch colour is a rect inside the drawing, so it does not fill the
          spare room around it. */}
      <svg viewBox="0 0 105 68" preserveAspectRatio="xMidYMin meet"
           className="min-h-0 w-full flex-1" role="img"
           aria-label={`Touch heatmap over ${n} recorded positions`}>
        <rect width="105" height="68" rx="1.5" fill="color-mix(in oklab, var(--ink) 8%, var(--card))" />
        <defs>
          {/* 2 METRES of blur, stated in the same unit as everything else in
              this viewBox - which is the point of drawing in metres. */}
          <filter id="hm-blur" x="-10%" y="-10%" width="120%" height="120%">
            <feGaussianBlur stdDeviation="2" />
          </filter>
          {/* The blur spreads past the touchlines; clipped to the pitch. */}
          <clipPath id="hm-clip">
            <rect width="105" height="68" rx="1.5" />
          </clipPath>
        </defs>
        <g clipPath="url(#hm-clip)">
        <g filter="url(#hm-blur)">
          {[...cells].map((v, i) =>
            v > 0 ? (
              <rect
                key={i}
                x={(i % NX) * cw}
                y={Math.floor(i / NX) * ch}
                width={cw}
                height={ch}
                fill={ramp(v / top)}
              />
            ) : null,
          )}
        </g>
        </g>
        {/* Markings OVER the density, so the pitch stays legible under a hot
            zone. Real dimensions: an 16.5m box, a 9.15m centre circle. */}
        <g fill="none" stroke="var(--slate)" strokeOpacity="0.45" strokeWidth="0.3">
          <rect x="0.3" y="0.3" width="104.4" height="67.4" />
          {/* Attacking direction, at the halfway line where the reference puts
              it. Left is his own goal, right is the one he is shooting at. */}
          <path d="M50 3 l3 2.5 -3 2.5 M54 3 l3 2.5 -3 2.5"
                strokeOpacity="0.8" strokeWidth="0.9" strokeLinecap="round"
                strokeLinejoin="round" />
          <line x1="52.5" y1="0.3" x2="52.5" y2="67.7" />
          <circle cx="52.5" cy="34" r="9.15" />
          <rect x="0.3" y="13.85" width="16.5" height="40.3" />
          <rect x="88.2" y="13.85" width="16.5" height="40.3" />
        </g>
      </svg>
      {/* The arrow is drawn because the direction was CHECKED, not assumed.
          The first pass here left it off on the reasoning that sides swap at
          half time and a season aggregate cannot know which way anyone was
          facing. That was wrong: the feed normalises. Mean x over these 21
          squads' heatmaps orders perfectly by position - the keeper 11.6, the
          centre backs 35-40, the holding midfielders 48-50, the attacking
          midfielders 58-66, the wingers and strikers 69-72 - which only
          happens if every map has been flipped to attack left-to-right. */}
    </>
  );
}


/** PLAYER TRAITS - six percentiles as a hexagon.
 *
 *  The feed supplies both the numbers and the cohort, and the cohort is the
 *  half that carries the meaning: 87% is nothing until you know it is 87% of
 *  attacking midfielders and wingers. So the chart does not render without it.
 *
 *  THE ORDER IS THE FEED'S, untouched. Its items run clockwise from the
 *  top-right of its own hexagon, so preserving the order makes two players'
 *  shapes comparable at a glance - which is the only reason to draw a radar
 *  rather than six bars. Sorting them by value would make every player look
 *  like the same lopsided star.
 *
 *  Six axes either way: outfielders get chances/aerials/defensive/goals/shots/
 *  touches, keepers get sweeper/conceded/save%/high claims/clean sheets/long
 *  balls. Anything that is not six still draws - the geometry is computed from
 *  the count - it just stops being a hexagon.
 *
 *  WEDGES, NOT A RADAR POLYGON - the reference's form. The hexagon is cut
 *  into six sectors, one per trait, and each is filled from the centre out
 *  to its percentile. A radar polygon joins neighbouring values, so its AREA
 *  is an artefact of the axis order; a wedge is one value alone, and the
 *  number is printed beside it besides.
 *
 *  ONE THING IS NOT VERIFIED and should not be claimed: that further out is
 *  always better. It plainly is for goals and save percentage. "Goals
 *  conceded" is a keeper axis where the raw quantity runs the other way, and
 *  whether the feed inverts the percentile before publishing it is not
 *  something this data says. So nothing here labels an axis good or bad, and
 *  the fill is one flat colour rather than a red-to-green ramp that would be
 *  asserting exactly that.
 *
 *  One colour, --chart-1, rather than the club's: some club colours are
 *  near-black, and a fill nobody can see on a dark card is worse than one
 *  that does not match.
 */
/** The feed's two axis sets, in its own clockwise order. Every outfield role
 *  shares the first; keepers get the second. An empty chart still names its
 *  axes so the shape reads as "nothing recorded", not "failed to load". */
const OUTFIELD_TRAITS = ["Chances created", "Aerial duels", "Defensive actions", "Goals", "Shot attempts", "Touches"];
const KEEPER_TRAITS = ["Sweeper actions", "Goals conceded", "Save percentage", "High claims", "Clean sheets", "Long ball percentage"];

function TraitsHex({ bio, keeper = false }: { bio: PlayerBio; keeper?: boolean }) {
  const items = (bio.traits ?? []).filter((t) => typeof t.value === "number");
  // NO TRAITS IS AN EMPTY CHART, not a sentence. A hand-entered player has no
  // percentiles - nobody has published any about him - and swapping the shape
  // for a line of prose made the panel jump and read as a chart that failed to
  // load. Six blank axes say the same thing and say it in the chart's own
  // language: this is the shape, there is nothing in it yet.
  const empty = items.length < 3 || !bio.traits_group;
  const shown: { key?: string | null; label?: string | null; value?: number | null }[] = empty
    ? (keeper ? KEEPER_TRAITS : OUTFIELD_TRAITS).map((label) => ({ label, value: null }))
    : items;
  const axes = shown;

  const W = 300;
  const cx = W / 2;
  const cy = 115;
  const R = 64;
  const n = axes.length;
  const half = 180 / n; // a sector's half-width, in degrees

  const pt = (deg: number, r: number) => {
    const a = (deg * Math.PI) / 180;
    return [cx + r * Math.cos(a), cy - r * Math.sin(a)] as const;
  };
  // Item 0 points at 60 degrees and the rest run clockwise, which is where
  // the feed's first item sits on its own chart. The corners sit BETWEEN the
  // items, so each item owns the sector from one corner to the next.
  const axis = (i: number) => 60 - (360 / n) * i;
  const poly = (ps: (readonly [number, number])[]) =>
    ps.map((q) => q.map((v) => v.toFixed(1)).join(",")).join(" ");
  const ring = (r: number) => poly(axes.map((_, i) => pt(axis(i) + half, r)));
  const pct = (v: number) => `${Math.round(v * 100)}%`;
  // Short enough to sit beside the hexagon at this width; longer labels break
  // between words, as the reference's do.
  const wrap = (label: string) =>
    label.split(" ").reduce<string[]>((ls, w) => {
      const last = ls[ls.length - 1];
      if (last && `${last} ${w}`.length <= 13) ls[ls.length - 1] = `${last} ${w}`;
      else ls.push(w);
      return ls;
    }, []);

  return (
    // FILLS what Overview leaves it and scales to that, so the chart is never
    // scrolled to, pinned to the top (xMidYMin). The viewBox is cropped to the
    // drawing - the stacked labels reach y 30 and 200.
      <svg viewBox={`0 25 ${W} 180`} preserveAspectRatio="xMidYMin meet" className="min-h-0 w-full flex-1 overflow-visible" role="img"
           aria-label={
             empty
               ? "No percentiles recorded for this player"
               : `Percentiles: ${items.map((t) => `${t.label} ${pct(t.value as number)}`).join(", ")}`
           }>
        {!empty && items.map((t, i) => {
          const r = R * Math.max(0.02, t.value as number);
          return (
            <polygon key={`${t.key ?? i}-w`} fill="var(--chart-1)"
                     points={poly([[cx, cy], pt(axis(i) + half, r), pt(axis(i) - half, r)])} />
          );
        })}
        {/* Rings at a third and two thirds, and spokes to the corners, all
            OVER the fill, so a wedge can be read against them. */}
        {[1 / 3, 2 / 3].map((f) => (
          <polygon key={f} points={ring(R * f)} fill="none" stroke="var(--slate)"
                   strokeOpacity="0.6" strokeDasharray="2 2" />
        ))}
        <polygon points={ring(R)} fill="none" stroke="var(--slate)" strokeOpacity="0.8" />
        {axes.map((t, i) => {
          const [x, y] = pt(axis(i) + half, R);
          return <line key={`${t?.key ?? i}-s`} x1={cx} y1={cy} x2={x} y2={y}
                       stroke="var(--slate)" strokeOpacity="0.8" />;
        })}
        {shown.map((t, i) => {
          const th = axis(i);
          const [x, y] = pt(th, R * Math.cos((half * Math.PI) / 180) + 9);
          const up = Math.sin((th * Math.PI) / 180);
          const right = x > cx + 1;
          const anchor = right ? "start" : x < cx - 1 ? "end" : "middle";
          const value = typeof t.value === "number" ? pct(t.value) : "";
          const label = t.label ?? "";
          // Beside the hexagon on its flanks, or when the label is long: the
          // value on a line of its own over the label. Otherwise one line,
          // value nearest the hexagon.
          if (Math.abs(up) >= 0.3 && label.length <= 15) {
            return (
              <text key={`${t.key ?? i}-l`} x={x} y={y} textAnchor={anchor}
                    dominantBaseline="middle" fontSize="10">
                {right ? (
                  <>
                    {value && <tspan fill="var(--ink)" fontWeight="600">{value} </tspan>}
                    <tspan fill="var(--slate)">{label}</tspan>
                  </>
                ) : (
                  <>
                    <tspan fill="var(--slate)">{label}</tspan>
                    {value && <tspan fill="var(--ink)" fontWeight="600"> {value}</tspan>}
                  </>
                )}
              </text>
            );
          }
          const lines = value ? [value, ...wrap(label)] : wrap(label);
          const L = 12;
          const y0 = up >= 0.3 ? y - (lines.length - 1) * L : up <= -0.3 ? y : y - ((lines.length - 1) * L) / 2;
          return (
            <text key={`${t.key ?? i}-l`} textAnchor={anchor} dominantBaseline="middle" fontSize="10">
              {lines.map((ln, k) => (
                <tspan key={k} x={x} y={y0 + k * L}
                       fill={k === 0 && value ? "var(--ink)" : "var(--slate)"} fontWeight={k === 0 && value ? 600 : 400}>
                  {ln}
                </tspan>
              ))}
            </text>
          );
        })}
      </svg>
  );
}


/** THE CARD, over the reference's layout and only its real fields.
 *
 *  WHAT IS NOT HERE, because this product does not have it: the Follow button
 *  (there are no accounts anywhere in this app - see org.py) and the club
 *  crest (nothing stores one, and a crest guessed from a name is a logo on the
 *  wrong club).
 *
 *  The country shows its name and the feed's three-letter code, not a flag.
 *  Flag emoji key on two-letter ISO and the feed sends ENG, SCO, WAL - which
 *  have no country code and no emoji, only subdivision sequences. A lookup
 *  that silently renders the wrong flag for four home nations is worse than
 *  the letters.
 */
/** THE TOP ROW OF EITHER CARD: the table's category pills on one side, the
 *  player's face and name on the other. They sit side by side, so they are one
 *  height and each closes itself with its own rule - see the note in BioBand
 *  for why the rule has to live INSIDE both rows rather than under one of
 *  them. min-h, not h: when the pills wrap the row grows.
 *
 *  57, not a round 56: the border is inside the box, so 57 leaves 56 of
 *  content and a 40px face sits on an EVEN 8 above and below - the same 8 the
 *  tab switcher keeps under the rule. At 56 the face sat on 7.5 and no round
 *  padding below the line could match it. */
export const BAND_ROW = "min-h-[57px] border-b";

/** THE BAND, and it is the only part of the card that never leaves. Whatever
 *  tab is open, the face and the name stay above it - a panel that can be
 *  scrolled and switched needs one fixed thing saying whose it is. */
function BioBand({
  bio,
  photoId,
  photo,
  name,
  onName,
  onPhoto,
  photoPicked,
  menu,
}: {
  bio: PlayerBio | null;
  photoId: string;
  photo?: boolean;
  /** What to call him when there is no bio - the registry's name, or the id.
   *  While editing this is the DRAFT's name and `onName` is set, which is what
   *  turns the same line into the name field. */
  name?: string | null;
  onName?: (v: string) => void;
  /** Set while editing: the face becomes the photo picker. */
  onPhoto?: () => void;
  photoPicked?: string | null;
  menu?: React.ReactNode;
}) {
  return (
    // SAME HEIGHT, AND THE RULE INSIDE IT, as the table's category row beside
    // it (BAND_ROW there). Padding alone could never line the two rules up:
    // that one is the row's own border-b and sits in its last pixel, while
    // this one used to be the tabs' border-t and therefore started one pixel
    // BELOW the band - the gap you could still see after the heights matched.
    // Both rows are min-h-14 now, both own their rule, so the two lines land
    // on the same pixel and neither is tuned to the other's type.
    //
    // Below ~915px the pills wrap to a second line and nothing can align -
    // there the panel is under the table, not beside it.
    <div className={cn(BAND_ROW, "flex min-w-0 items-center gap-3 px-4 sm:px-6")}>
      {onPhoto ? (
        <button
          type="button"
          onClick={onPhoto}
          title={photoPicked ?? "Change photo"}
          className="relative shrink-0 rounded-full ring-offset-2 ring-offset-[var(--panel)] transition-shadow hover:ring-2 hover:ring-[var(--hairline-strong)]"
        >
          <Portrait id={photoId} name={name ?? photoId} photo={photo} size={40} />
          {/* A picked file has not been uploaded yet, so the face on screen is
              still the old one - the dot is the only thing that says a new one
              is going up with Save. */}
          {photoPicked && (
            // --accent, not --info: globals.css reserves the tier tokens (ok/info/bad)
            // for claims about how far a NUMBER can be trusted, and Signal Blue
            // for machine state. "A file is staged and uploads on Save" is
            // machine state.
            <span className="absolute right-0 bottom-0 size-2.5 rounded-full bg-[var(--accent)] ring-2 ring-[var(--panel)]" />
          )}
        </button>
      ) : (
        <Portrait id={photoId} name={bio?.name ?? photoId} photo={photo} size={40} />
      )}
      {/* THE NAME, AND NOTHING ELSE. The shirt number and the club used to
          ride along here; they are facts about him like his foot and his
          country, so they are down in the facts grid with the rest and this
          band is one line high.

          Editing types over it IN PLACE, at the same size and weight: the name
          is a fact like the others and it is edited where it is read. */}
      {onName ? (
        <input
          autoFocus
          value={name ?? ""}
          onChange={(e) => onName(e.target.value)}
          placeholder="Full name"
          aria-label="Name"
          // The band's own type, and no box: same rule as the cells below -
          // an underline only under the pointer or the caret, painted over the
          // text so the row keeps its height.
          className="font-heading min-w-0 flex-1 cursor-text border-0 bg-transparent p-0 text-base font-medium text-[var(--ink)] underline-offset-4 outline-none hover:underline hover:decoration-dotted focus:underline focus:decoration-dotted placeholder:font-normal placeholder:text-[var(--slate)]"
        />
      ) : (
        <p className="font-heading min-w-0 flex-1 truncate text-base font-medium">
          {bio?.name ?? name ?? photoId}
        </p>
      )}
      {menu}
    </div>
  );
}

/** THE BOLT, at the end of the name row: everything you can do TO a player
 *  rather than with his numbers.
 *
 *  A menu and not three buttons, because two of the three are rare - a squad
 *  is entered once and corrected occasionally - and a row of chrome beside a
 *  name would be read as part of the profile. It is the one control on this
 *  card that writes anything, so it is also the only one that looks like a
 *  control. */
function BandMenu({ onEdit, onAdd }: { onEdit: () => void; onAdd: () => void }) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          aria-label="Player actions"
          title="Player actions"
          className="text-muted-foreground -mr-1 shrink-0 rounded-[var(--radius-sm)] p-1.5 transition-colors hover:bg-[var(--card-hover)] hover:text-[var(--ink)]"
        >
          <Bolt className="size-[17px]" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-44">
        <DropdownMenuItem onSelect={onEdit}>Edit player</DropdownMenuItem>
        <DropdownMenuItem onSelect={onAdd}>Add player</DropdownMenuItem>
        {/* Nothing behind it yet, and it says so rather than opening an empty
            screen: transfers, retiring a profile, merging two of them. */}
        <DropdownMenuItem disabled>Manage player</DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

/** WHAT THE BOLT BECOMES while a profile is being edited.
 *
 *  In its place, not beside it: there is one control slot at the end of the
 *  name row, and edit mode is a state of the whole card rather than a thing
 *  you do to one field. It is also the only chrome that moves when edit mode
 *  opens - the facts under it stay exactly where they were - so this pair is
 *  what tells you the card is live.
 *
 *  Save is the affirmative, so it is the one that reads as a button; Cancel is
 *  quiet. Enter and Escape do the same two things from anywhere in the form -
 *  see the handlers in PlayerProfile. */
function BandEdit({
  onSave,
  onCancel,
  busy,
  ready,
}: {
  onSave: () => void;
  onCancel: () => void;
  busy: boolean;
  ready: boolean;
}) {
  return (
    <div className="flex shrink-0 items-center gap-1">
      <Button type="button" variant="ghost" size="sm" onClick={onCancel} disabled={busy}>
        Cancel
      </Button>
      <Button type="button" size="sm" onClick={onSave} disabled={busy || !ready}>
        {busy ? "Saving…" : "Save"}
      </Button>
    </div>
  );
}

/** WHO HE IS, as one section: the facts, the positions he is played in, and
 *  the pitch drawing of them beside. The Overview tab, and the part a coach
 *  reads first. */
function BioFacts({ bio }: { bio: PlayerBio }) {
  // EVERY FIELD, ALWAYS. A player with nothing recorded gets the same grid
  // with dashes in it, so an empty profile has the shape of a full one and
  // does not collapse into a different layout.
  const others = (bio.positions ?? []).filter((p) => !p.main).map((p) => p.short ?? p.label);
  const otherList = (others.length ? others : (bio.other_positions ?? [])).filter(Boolean);
  const facts: { value: string; label: string; sub?: string | null }[] = [
    { value: bio.height_cm != null ? `${bio.height_cm} cm` : "—", label: "Height" },
    { value: bio.age != null ? `${bio.age} years` : "—", label: "Age", sub: longDate(bio.birth_date) ?? "Age" },
    { value: bio.foot || "—", label: "Preferred foot" },
    { value: bio.country || "—", label: "Country", sub: bio.country_code ?? "Country" },
    { value: bio.primary_position || "—", label: "Primary position" },
    { value: otherList.length ? otherList.join(", ") : "—", label: "Other positions" },
    { value: bio.shirt != null ? `#${bio.shirt}` : "—", label: "Shirt number" },
    // SQUAD, not club: an academy is one club and many squads.
    { value: bio.team ? bio.team + (bio.on_loan ? " · on loan" : "") : "—", label: "Squad" },
  ];

  return (
    <div className="flex min-w-0 gap-4 px-4 py-3 sm:px-6">
      <div className="grid min-w-0 flex-1 grid-cols-2 content-start gap-x-4">
        {facts.map((f, i) => (
          <Fact key={`${f.label}-${i}`} {...f} />
        ))}
      </div>
      {/* 104px, measured not chosen: three badges on a 92px pitch left the
          outermost 3.7px off the touchline, and a position dot that looks
          like it is about to fall off the pitch reads as a bug. */}
      {/* Always drawn, empty when no position has a spot: dropping it would
          widen the facts beside it. */}
      <div className="w-[104px] shrink-0 self-center">
        <PositionPitch bio={bio} />
      </div>
    </div>
  );
}


/** THE PLAYER REGISTRY - one row per person in the club, and their profile.
 *
 *  THREE SOURCES, ONE ROW, ALL JOINED ON THE ROSTER ID. `/api/players` is the
 *  registry: who exists, in which squads, with what shirt. `api.getAppearances`
 *  is what the FEED said they did. `/api/season/measured` is what the pipeline
 *  MEASURED off confirmed tracks. All three key on the same server-minted
 *  roster id, which is the only join in this product that is not a guess about
 *  a name. A player with no measured row is not an error - it means no track
 *  has been confirmed as them yet, and the row says so with an em dash rather
 *  than a zero.
 *
 *  WHY THE MEASURED NUMBERS ARE SAFE HERE, when the season tables above
 *  deliberately refuse them: these totals are built ONLY from tracks a human
 *  confirmed on the identity screen. Identity reaches chance a few minutes into
 *  a match, so an unconfirmed track under a name would be one player's running
 *  reported as another's. The confirmed set is small; the profile's Confirmed
 *  matches row is the coverage, so the total never passes for the season.
 */
export function PlayerRegistry({
  matchId,
  teamId,
  onMatch,
  season: seasonProp,
  onSeason,
  focus,
}: {
  /** Narrow to one squad. A FILTER, defaulting to the whole club - not the
   *  app's team scope, which this section never writes. The default matters:
   *  a player climbing U15 -> U17 is one profile, and the whole point of this
   *  table is that a career does not stop at a squad. */
  teamId?: string;
  /** Show ONE match: the players who appeared in it, with that match's
   *  numbers instead of the season's. */
  matchId?: string;
  /** Where a match picked in the profile's Matches tab goes, so the page's
   *  Match dropdown shows it. Without one the pick is kept here. */
  onMatch?: (id: string) => void;
  /** Show ONE season (a competition as the feed names it), summed. */
  season?: string;
  /** The Season dropdown's setter, for picks made in the Seasons tab. */
  onSeason?: (competition: string) => void;
  /** A player to open, from a jump - a team sheet elsewhere, or Back. */
  focus?: string | null;
} = {}) {
  const [rows, setRows] = useState<ClubPlayer[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [appear, setAppear] = useState<Appearance[] | null>(null);
  const [stats, setStats] = useState<Record<string, Record<string, number>>>({});
  const [appearError, setAppearError] = useState<string | null>(null);
  const [cat, setCat] = useState(PLAYER_STAT_TABS[0].id);
  const [per90, setPer90] = useState(false);
  const [bios, setBios] = useState<Record<string, PlayerBio>>({});
  const [picked, setPicked] = useState<string | null>(null);
  // THE NEW-PLAYER FORM, which is the profile panel with no player behind it.
  const [adding, setAdding] = useState(false);
  const { org } = useOrg();
  // THE TABLE'S SCOPE: a season (a competition, as the store files it) or one
  // match, from the page's dropdowns or the profile's tabs. A match wins over
  // a season. Without the page's setters (a squad's own page) they are kept
  // here instead.
  const [localSeason, setLocalSeason] = useState("");
  const [localMatch, setLocalMatch] = useState("");
  const seasonPick = (onSeason ? seasonProp : localSeason) || null;
  const match = onMatch ? (matchId ?? "") : localMatch;
  const pickMatch = (id: string) => (onMatch ? onMatch(id) : setLocalMatch(id));
  const pickSeason = (c: string) => (onSeason ? onSeason(c) : setLocalSeason(c));
  // The ERROR is taken, not dropped. Losing it is what let a failed fetch
  // render as a confident claim about the squad - see data/load-state.tsx.
  const { season, error: measuredError, reload: reloadMeasured } = useSeason(undefined, true);
  const sort = useSort("rating");

  useEffect(() => {
    let live = true;
    api.getPlayers().then(
      (r) => live && setRows(r),
      (e: unknown) => live && setError(e instanceof Error ? e.message : String(e)),
    );
    return () => {
      live = false;
    };
  }, []);

  // A SEPARATE fetch, and a separately reported failure. The feed's half of
  // this table going missing must not look like the club having no players,
  // which is what one shared error state did to the first version of this.
  useEffect(() => {
    let live = true;
    api.getAppearances().then(
      (r) => live && setAppear(r),
      (e: unknown) => live && setAppearError(e instanceof Error ? e.message : String(e)),
    );
    return () => {
      live = false;
    };
  }, []);

  // NO ERROR STATE. A bio is the only thing here that is decoration rather
  // than an answer: without it the card falls back to the name and the numbers,
  // which is what this panel was a week ago. A red box over a squad list
  // because a height did not arrive would be louder than the loss.
  useEffect(() => {
    let live = true;
    api.getBios().then(
      (b) => live && setBios(b),
      () => {},
    );
    return () => {
      live = false;
    };
  }, []);

  /** Appearance rows per player, kept as rows and not only as totals: the
   *  panel's per-match matrix needs them individually. */
  const byPlayer = useMemo(() => {
    const m = new Map<string, Appearance[]>();
    for (const a of appear ?? []) {
      const at = m.get(a.player_id);
      if (at) at.push(a);
      else m.set(a.player_id, [a]);
    }
    return m;
  }, [appear]);

  const entered = useMemo(
    () => new Map([...byPlayer].map(([id, rs]) => [id, enteredOf(rs)])),
    [byPlayer],
  );

  /** The matches the table covers: one, a season's, or all (null). */
  const scopeIds = useMemo(() => {
    if (match) return [match];
    if (!seasonPick) return null;
    return [...new Set((appear ?? []).filter((a) => a.competition === seasonPick).map((a) => a.match_id))];
  }, [match, seasonPick, appear]);
  const scopeKey = scopeIds?.join(",") ?? "";
  // A season or a match picks its players from the appearances; until those
  // arrive the table cannot know who is in it, and must not say "nobody".
  const scopeLoading = scopeIds !== null && appear === null && !appearError;

  // The stats for the scope, summed server-side - the per-match rows are more
  // than the query endpoint returns. Same feed as the appearances, so the same
  // failure.
  useEffect(() => {
    let live = true;
    api.getSeasonStats(scopeKey ? scopeKey.split(",") : undefined).then(
      (s) => live && setStats(s),
      (e: unknown) => live && setAppearError(e instanceof Error ? e.message : String(e)),
    );
    return () => {
      live = false;
    };
  }, [scopeKey]);

  /** `entered`, over the scope only - the table's minutes and rating. A
   *  player with no appearance in it has no entry, and so no row. */
  const scoped = useMemo(() => {
    if (!scopeIds) return entered;
    const out = new Map<string, Entered>();
    for (const [id, rs] of byPlayer) {
      const inScope = rs.filter((a) => scopeIds.includes(a.match_id));
      if (inScope.length) out.set(id, enteredOf(inScope));
    }
    return out;
  }, [scopeIds, byPlayer, entered]);

  /** One season stat for one player, in the report's column ids. A player with
   *  no appearance gets nothing (an em dash); one who played and has no row for
   *  a stat gets 0, because the feed omits a stat line it has nothing on.
   *  Rating is the per-match average, never the sum.
   *
   *  PER 90 divides by the feed's minutes. Rating is already a per-match
   *  figure and minutes per 90 is 90, so both stay as they are; a player with
   *  no minutes has no rate, not a rate of zero. */
  const statOf = useCallback(
    (id: string, c: string): number | undefined => {
      const e = scoped.get(id);
      if (!e) return undefined;
      if (c === "rating") return e.rating ?? undefined;
      const s = stats[id] ?? {};
      const v = c === "xgxa" ? (s.xg ?? 0) + (s.xa ?? 0) : (s[c] ?? 0);
      if (!per90 || c === "minutes") return v;
      return e.mins ? (v / e.mins) * 90 : undefined;
    },
    [scoped, stats, per90],
  );

  // THE REPORT'S TABS, over the season. Goalkeeping shows keepers only, as it
  // does there.
  const tab = PLAYER_STAT_TABS.find((t) => t.id === cat) ?? PLAYER_STAT_TABS[0];

  const shown = useMemo(
    () =>
      (rows ?? []).filter(
        (p) =>
          (!scopeIds || scoped.has(p.id)) &&
          (!teamId || p.teams.some((t) => t.team_id === teamId)) &&
          (!tab.gkOnly || group(p.position) === "GK"),
      ),
    [rows, scopeIds, scoped, teamId, tab.gkOnly],
  );

  // A column the feed never sent for anyone on screen is dropped rather than
  // drawn as a column of zeroes it cannot vouch for.
  const cols = useMemo(() => {
    const has = (c: string): boolean =>
      c === "xgxa" ? has("xg") || has("xa") : shown.some((p) => stats[p.id]?.[c] != null);
    // Nobody on screen: nothing to drop on, so the tab's full header shows.
    return shown.length ? tab.cols.filter(has) : tab.cols;
  }, [tab, shown, stats]);

  const sorted = useMemo(() => {
    const num = (p: ClubPlayer, k: string): number => statOf(p.id, k) ?? -Infinity;
    return [...shown].sort((a, b) => {
      if (sort.key === "name") {
        const s = (a.name ?? a.id).localeCompare(b.name ?? b.id);
        return sort.asc ? s : -s;
      }
      // Position sorts DOWN THE RAILS - keeper, defence, midfield, attack -
      // and not alphabetically. That is the game's order and also the only
      // order a coach reads a squad list in; "Attacker" first is alphabetical
      // and nonsense.
      if (sort.key === "position") {
        const s = ORDER.indexOf(group(a.position)) - ORDER.indexOf(group(b.position));
        return sort.asc ? s : -s;
      }
      return cmp(num(a, sort.key), num(b, sort.key), sort.asc);
    });
  }, [shown, sort.key, sort.asc, statOf]);

  // OPEN THE TOP ROW ON ARRIVAL, once. A squad hub that starts with an empty
  // half is asking for a click before it will show you anything, and the row
  // it would show is the one already at the top of the list.
  //
  // The ref is what makes it "on arrival" rather than "whenever nothing is
  // selected". It must not re-fire when the sort or the squad
  // filter rewrites `sorted` - having the panel jump to a different player
  // because you changed a dropdown is not opening a default, it is taking the
  // selection away from you.
  //
  // A remount DOES open it again, which is the intent: leaving the section and
  // coming back is arriving.
  const opened = useRef(false);
  // A JUMP wins over the default: the player it names is opened the moment it
  // changes - adjusted during render, React's pattern for state that follows
  // a prop, so there is no frame showing the previous player first.
  const [seenFocus, setSeenFocus] = useState<string | null | undefined>(null);
  if (focus && focus !== seenFocus) {
    setSeenFocus(focus);
    setPicked(focus);
  }
  useEffect(() => {
    // Arriving by a jump counts as having arrived: the top row is only the
    // default when nothing named a player.
    if (opened.current || !sorted.length || seenFocus) return;
    opened.current = true;
    setPicked(sorted[0].id);
  }, [sorted, seenFocus]);

  // THE BEST RATING AMONG THE ROWS ON SCREEN, for the star pill. The report
  // stars the best in one MATCH; the same mark here means the best over the
  // season, because that is the set being looked at.
  //
  // OVER `shown`, NOT `entered`. The first attempt took the max of `entered`,
  // which is every appearance in the store - all 228 players of both sides -
  // so the maximum belonged to somebody at another club and no row of this club
  // ever matched it. The star simply never appeared, which is the quiet kind
  // of wrong: nothing errored and the column looked finished.
  const bestRating = useMemo(
    () =>
      shown.reduce(
        (top, p) => Math.max(top, scoped.get(p.id)?.rating ?? -Infinity),
        -Infinity,
      ),
    [shown, scoped],
  );

  // From every row, not the sorted ones, so a tab that filters (Goalkeeping)
  // does not close the profile of whoever it filtered out.
  const selected = (rows ?? []).find((p) => p.id === picked) ?? null;

  /** The club's squads, for the form's Squad dropdown. org.json owns the
   *  ladder (club-is-the-ladder-org-json-owns-it (internal note)); the squads
   *  seen on players are the fallback for an install whose org predates a
   *  roster file. */
  const teams = useMemo(() => {
    const seen = new Set<string>();
    for (const t of org?.teams ?? []) if (t.name) seen.add(t.name);
    for (const p of rows ?? []) for (const t of p.teams) if (t.team) seen.add(t.team);
    return [...seen];
  }, [org, rows]);

  /** BOTH STORES, after a write. The registry is not memoized and the bios
   *  are, so the editor drops the memo before calling this - see
   *  clearStoreCache in player-editor.tsx. */
  const reloadPlayers = useCallback((pid: string) => {
    api.getPlayers().then((r) => {
      setRows(r);
      setPicked(pid);
    }, () => {});
    api.getBios().then(setBios, () => {});
  }, []);

  // A BLANK STAND-IN while adding, so the panel renders its own empty form
  // instead of every field having to ask whether a player exists yet.
  const BLANK: ClubPlayer = {
    id: "",
    name: "",
    position: null,
    teams: [],
    matches: 0,
    photo: false,
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      {/* THE LIST STAYS WHOLE while a profile is open, and the profile sits
          BESIDE it - at every width a pointer is plausible, not only where
          everything fits without compromise.

          The list used to collapse to the one pinned row. That reads fine as a
          disclosure and it is not a database: the point of a squad hub is that
          you run DOWN the list while the panel keeps up, comparing player to
          player without losing your place.

          THE BREAKPOINT USED TO BE 1650 and it was measured honestly - table
          869 + panel 352 + gap + sidebar 252 + rail 110 puts "nothing clips"
          at about 1595. The mistake was treating "nothing clips" as the
          condition for putting the panel where it belongs. Under 1650 it
          dropped BELOW a 23-row table, so on a 1280 laptop - the common case,
          not the edge - the profile sat off-screen and the side-by-side
          reading this layout exists for never happened. Trading the whole
          feature for a scrollbar is the wrong way round.

          THE THRESHOLD MOVED WITH THE PANEL, twice, and both times because it
          was measured. It was `md` (768) while the panel was 300-352 wide.
          Widening it to 400-560 broke that: at 900 the shell takes 252 for the
          sidebar and 110 for the rail, the panel takes 400, and the table is
          left 126px for 414px of columns - a scrollbar with a table behind it.
          1200 is where the arithmetic closes (838 - 412 = 426, against 414
          needed), so that is where the split starts and below it the panel
          goes full width under the list.

          THOSE WERE WINDOW WIDTHS; THESE ARE CONTAINER WIDTHS. The same table
          also runs in a dock tab beside the chat, where the window is wide and
          the table is not. So the breakpoints are container queries against
          the surface (@container in workspace/page.tsx and dock.tsx), each the
          old window width less the sidebar, its border and the page's p-4:
          1200 / 1280 / 1650 -> 915 / 995 / 1365.

          SIDE BY SIDE, ONLY THE TABLE SCROLLS. The row takes the pane's
          height, the table scrolls inside its card, and the profile fills the
          column beside it instead of travelling with the page. */}
      <div className="flex min-w-0 flex-col gap-3 @min-[915px]:min-h-0 @min-[915px]:flex-1 @min-[915px]:flex-row">
        <Card className="min-w-0 flex-1 gap-0 py-0 @min-[915px]:min-h-0">
          {/* The report's category pills, pinned above the scrolling rows. A
              tab that drops the column being sorted on hands the sort to its
              own lead column, as the report's table does. */}
          <div className={cn(BAND_ROW, "flex flex-wrap items-center gap-1.5 px-3 py-2")}>
            {PLAYER_STAT_TABS.map((t) => (
              <Pill
                key={t.id}
                on={cat === t.id}
                onClick={() => {
                  setCat(t.id);
                  if (!["position", "name", ...t.cols].includes(sort.key)) sort.on(t.sort)();
                }}
              >
                {t.label}
              </Pill>
            ))}
            <Per90Toggle on={per90} onToggle={() => setPer90((v) => !v)} />
          </div>
          {/* ONE scroller for both directions. The Table's own wrapper scrolls
              sideways by default, and nested inside this it put the horizontal
              bar under the last row - out of sight until you scrolled down. */}
          <CardContent className="overflow-x-auto p-0 @min-[915px]:min-h-0 @min-[915px]:flex-1 @min-[915px]:overflow-y-auto [&_[data-slot=table-container]]:overflow-visible">
            {error && <Empty>Could not load players - {error}</Empty>}
            {measuredError && (
              <LoadFailure
                what="the measured season"
                error={measuredError}
                onRetry={reloadMeasured}
                className="m-3"
              />
            )}
            {appearError && (
              <LoadFailure
                what="the feed's appearances"
                error={appearError}
                className="m-3"
              />
            )}
            {!error && (rows === null || scopeLoading) && <TableSkeleton />}
            {/* THE HEADER EVEN WITH NO ROWS: an empty table keeps its column
                labels, so it reads as this table with nothing in it. */}
            {!error && rows !== null && !scopeLoading && (
              <div>
                {/* The match report's player table: sentence-case headers,
                    centred columns split by vertical rules, and rules between
                    rows only under the player. No banding. The sorted column's
                    header is shaded (see Head). */}
                {/* The header row stays put while the rows scroll under it. */}
                {/* The two-line header floor that keeps the first row of
                    numbers at the same height on every tab lives in
                    DATA_TABLE — see the note there. */}
                <Table className={cn(DATA_TABLE, "[&_thead_th]:px-1.5 [&_tbody_td]:px-1.5")}>
                  <TableHeader>
                    <TableRow>
                      {/* `!` on the edge padding: the table's own
                          `[&_thead_th]:px-1` outranks a plain class. */}
                      <Head k="position" sort={sort} className="w-[42px] !pl-3">
                        Pos
                      </Head>
                      {/* w-px shrinks Name to its longest name, so the spare
                          width goes to the stat columns instead of opening a
                          gap between the names and the numbers. */}
                      <Head k="name" sort={sort} className="w-px border-l border-[var(--grid-line)]">
                        Name
                      </Head>
                      {/* The report's full labels, wrapped between words the
                          way its header wraps them. */}
                      {cols.map((c, i) => (
                        <Head
                          key={c}
                          k={c}
                          sort={sort}
                          className={cn(
                            "border-l border-[var(--grid-line)] text-center leading-tight whitespace-normal",
                            i === cols.length - 1 && "!pr-3",
                          )}
                        >
                          {/* "xG + xA" held together, or it wraps to three lines. */}
                          {PLAYER_STAT_COLS[c].replaceAll(" + ", " + ")}
                        </Head>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {sorted.map((p) => {
                      const on = p.id === picked;
                      // The game's loan indicator, over the thing it actually
                      // means here: a player registered with a squad OTHER
                      // than the one being filtered to. The registry exists
                      // for the career that crosses squads, so the crossing
                      // has to stay visible even with a filter on - but a
                      // chip on every row for the common case is noise, so
                      // only the extra memberships are drawn.
                      const elsewhere = teamId
                        ? p.teams.filter((t) => t.team_id !== teamId)
                        : p.teams.length > 1
                          ? p.teams
                          : [];
                      return (
                        <TableRow
                          key={p.id}
                          {...pickable(on, () => setPicked(p.id), "group")}
                        >
                          <TableCell className="border-b border-[var(--grid-line)] !pl-3">
                            <PosBadge pos={p.position} />
                          </TableCell>
                          <TableCell className="font-heading w-px border-b border-l border-[var(--grid-line)] whitespace-nowrap">
                            <span className="flex items-center gap-2.5">
                              <Portrait
                                id={p.id}
                                name={p.name ?? p.id}
                                photo={p.photo}
                                size={26}
                              />
                              <span className="text-[13px] text-[var(--ink)]">
                                {p.name ?? p.id}
                              </span>
                              {elsewhere.map((t) => (
                                <span
                                  key={t.team_id}
                                  className={cn(
                                    "text-muted-foreground rounded-[var(--radius-xs)] px-1.5 py-0.5 text-[10px] whitespace-nowrap",
                                    "bg-[var(--card)]",
                                  )}
                                >
                                  {t.team ?? t.team_id}
                                  {t.number != null && ` #${t.number}`}
                                </span>
                              ))}
                            </span>
                          </TableCell>
                          {cols.map((c, i) => {
                            const v = statOf(p.id, c);
                            const cell = cn("border-l border-[var(--grid-line)] text-center", i === cols.length - 1 && "!pr-3");
                            // THE REPORT'S PILL for rating, not a bare number:
                            // the same component, so the two screens cannot
                            // drift.
                            return c === "rating" ? (
                              <TableCell key={c} className={cell}>
                                {v == null ? (
                                  <span className="readout text-muted-foreground">—</span>
                                ) : (
                                  <RatingChip value={v} best={v === bestRating} className="mx-auto" />
                                )}
                              </TableCell>
                            ) : (
                              <Count
                                key={c}
                                v={v}
                                dp={(per90 && c !== "minutes") || DECIMAL_COLS.has(c) ? 2 : 0}
                                className={cell}
                              />
                            );
                          })}
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
                {!shown.length && (
                  <Empty>
                    No players in this club yet. They are added on a match&apos;s pitch setup
                    screen, where a name gets attached to a track.
                  </Empty>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        {(selected || adding) && (
          <PlayerProfile
            // A fresh form per attempt, and a fresh panel when the player
            // changes: keying on the id is what throws the old draft away.
            key={adding ? "new-player" : selected!.id}
            player={adding ? BLANK : selected!}
            adding={adding}
            teams={teams}
            onAdd={() => setAdding(true)}
            onCancelAdd={() => setAdding(false)}
            onSaved={(pid) => {
              setAdding(false);
              reloadPlayers(pid);
            }}
            season={season}
            // NOTHING OF THE OLD PLAYER while adding. `selected` is still
            // whoever was open when the form was opened - the list keeps its
            // selection - so passing his bio through would put his name and
            // his face on the blank form.
            appearances={(!adding && selected && byPlayer.get(selected.id)) || []}
            entered={(!adding && selected && entered.get(selected.id)) || null}
            bio={(!adding && selected && bios[selected.id]) || null}
            seasonPick={seasonPick}
            onSeason={(c) => {
              pickSeason(c ?? "");
              pickMatch("");
            }}
            matchPick={match}
            onMatch={pickMatch}
            scopeIds={scopeIds}
          />
        )}
      </div>
    </div>
  );
}

/** THE DETAIL PANEL - the game's "PLAYER OVERVIEW & STATUS" and its "STATS"
 *  matrix, merged and stripped of everything that only exists in a save file.
 *
 *  The shape is copied closely because it is doing something specific: a
 *  titled bar, then the name large with ONE headline number set against it in
 *  the top-right, then a meta line, then label/value rows on a common right
 *  edge, then the matrix. That last pair is why the game's panel scans so
 *  fast - every value lands on the same x, so the eye runs down one edge
 *  instead of hunting.
 *
 *  The game's meta line is "GK | Age 27 | Spain". Age survives, because the
 *  team sheet carries it. Nationality does not: nothing in this product knows
 *  it, and a flag guessed from a name is exactly the class of error the
 *  identity work exists to avoid. The shirt number takes the slot instead.
 */
function PlayerProfile({
  player,
  season,
  appearances,
  entered,
  bio,
  seasonPick,
  onSeason,
  matchPick,
  onMatch,
  scopeIds,
  adding,
  teams,
  onSaved,
  onAdd,
  onCancelAdd,
}: {
  player: ClubPlayer;
  season: Season | null;
  appearances: Appearance[];
  entered: Entered | null;
  bio: PlayerBio | null;
  /** True while this panel IS the new-player form: the registry has no row
   *  for him yet, so `player` is a blank stand-in. */
  adding: boolean;
  /** The club's squads, for the form's Squad dropdown. */
  teams: string[];
  onSaved: (pid: string) => void;
  onAdd: () => void;
  onCancelAdd: () => void;
  /** The season (competition) the table is scoped to, if any. */
  seasonPick: string | null;
  onSeason: (competition: string | null) => void;
  /** The match the table is scoped to, or "". */
  matchPick: string;
  onMatch: (id: string) => void;
  /** The matches the table is scoped to, or null for all of them. */
  scopeIds: string[] | null;
}) {
  // NOT reset when the player changes, deliberately - see the tab strip.
  const [tab, setTab] = useState<Tab>("overview");

  // THE FORM, or null. Adding is always a form; editing is one the operator
  // asked for. Held here rather than in the row so that clicking down the
  // squad list leaves edit mode behind with the player it belonged to - a form
  // that survives a selection change is a form that saves the wrong person.
  // Initialised, not synchronised: the parent keys this panel on the player
  // (and on "new-player" while adding), so a different player is a different
  // component and the draft cannot outlive the person it belongs to. An
  // effect here would be a second way to say the same thing, one render late.
  const [draft, setDraft] = useState<PlayerDraft | null>(adding ? draftOf(null, null, teams) : null);
  const [busy, setBusy] = useState(false);
  const [photo, setPhoto] = useState<File | null>(null);
  /** Which season is showing its matches.
   *
   *  Starts on the season the page is scoped to, so the tab opens on the run
   *  you are already looking at rather than on a list of closed rows - but it
   *  is only the STARTING value. See the button in the Seasons tab for why it
   *  cannot go on being that scope. */
  const [openSeason, setOpenSeason] = useState<string | null>(seasonPick);
  const photoInput = useRef<HTMLInputElement>(null);
  const editing = draft !== null;

  // OVERVIEW IS THE ONLY TAB A FORM HAS. The other three are what the pipeline
  // and the feed recorded, which is not a thing you can type, so edit mode
  // pins the panel here - opened from Maps, editing takes you to the facts
  // rather than leaving you on a heatmap with a Save button over it.
  const openEdit = () => {
    setDraft(draftOf(player, bio, teams));
    setTab("overview");
  };

  const cancel = () => {
    setDraft(null);
    setPhoto(null);
    if (adding) onCancelAdd();
  };

  const commit = async () => {
    if (!draft || !canSave(draft) || busy) return;
    setBusy(true);
    try {
      const id = await savePlayer(adding ? null : player.id, draft, photo);
      setDraft(null);
      setPhoto(null);
      onSaved(id);
    } catch (e) {
      toast.error("Could not save", {
        description: e instanceof Error ? e.message : String(e),
      });
    } finally {
      setBusy(false);
    }
  };
  const names = useMemo(
    () => new Map((season?.matches ?? []).map((x) => [x.id, x])),
    [season],
  );

  /** One fixture from this player's side: the opponent, home or away, the
   *  result and the day. WHICH SIDE ARE WE comes from the player's own squad
   *  name - `season.team` is empty here, because this table is the whole club
   *  and fetches the unscoped season. With no side found, the opponent falls
   *  back to the fixture's name and there is no result, rather than a guess. */
  const fixture = (id: string) => {
    const f = names.get(id);
    if (!f) return null;
    const us = player.teams.map((t) => t.team).find(Boolean) ?? bio?.team;
    const home = f.home_name === us ? true : f.away_name === us ? false : null;
    const played = scoreOf(f.score);
    const res =
      home == null || !played
        ? null
        : played[0] === played[1] ? "D" : (played[0] > played[1]) === home ? "W" : "L";
    return {
      opp: (home == null ? f.name : home ? f.away_name : f.home_name) ?? f.name,
      res,
      score: f.score,
      // The fixture's UTC day, off the end of its name, where the importer
      // put it - not `created`, whose local date is a timezone question.
      day: /(\d{4}-\d{2}-\d{2})\s*$/.exec(f.name ?? "")?.[1] ?? null,
      created: f.created ?? 0,
    };
  };

  /** The days of the scope's matches, for the shot map: a shot is joined to
   *  a fixture by the day it was taken. */
  const scopeKey = scopeIds?.join(",") ?? "";
  const scopeDays = useMemo(() => {
    if (!scopeIds) return null;
    return new Set(scopeIds.map((id) => fixture(id)?.day).filter((d): d is string => !!d));
    // eslint-disable-next-line react-hooks/exhaustive-deps -- fixture reads `names`, keyed here by scopeKey
  }, [scopeKey, names]);

  /** THE SEASONS: this player's appearances grouped by competition, which is
   *  how the store files the season ("Premier League 2025/2026"). */
  const seasons = useMemo(() => {
    const by = new Map<string, Appearance[]>();
    for (const a of appearances) {
      const k = a.competition ?? "Unknown competition";
      const at = by.get(k);
      if (at) at.push(a);
      else by.set(k, [a]);
    }
    return [...by]
      .map(([comp, rs]) => ({ comp, e: enteredOf(rs) }))
      .sort((x, y) => y.comp.localeCompare(x.comp));
  }, [appearances]);

  /** THE MATCHES, newest first. */
  const games = [...appearances]
    .map((a) => ({ a, f: fixture(a.match_id) }))
    .sort((x, y) => (y.f?.created ?? 0) - (x.f?.created ?? 0));
  const club = player.teams.map((t) => t.team).find(Boolean) ?? bio?.team ?? "";

  return (
    // Beside the list it is the column's full height and never moves; only its
    // tab body scrolls, and only when a tab is taller than the pane. Stacked
    // (a container under 915px - a 1200px window on the page) it is in the flow.
    //
    // THE PANEL IS THE PRODUCT NOW. Every number the table used to carry is in
    // here, in four tabs, so the split of width between the two has been
    // reversed: the panel takes what it needs and the table keeps only enough
    // to identify a row and be scanned. 400 / 460 / 560, against 300 / 324 /
    // 352 two commits ago.
    //
    // ALL THREE ARE ARBITRARY VARIANTS, and mixing them with a named one was a
    // bug: `md: xl: min-[1650px]:` put the 1650 rule BEFORE `xl` in the
    // emitted sheet, so at 1700 the panel stayed 380 and the widest step never
    // applied at all. Same-type arbitrary variants sort by their value.
    <Card className="w-full py-0 ring-0 @min-[915px]:min-h-0 @min-[915px]:w-[400px] @min-[915px]:shrink-0 @min-[995px]:w-[460px] @min-[1365px]:w-[560px]">
      <CardContent
        className="flex min-h-0 min-w-0 flex-1 flex-col gap-0 p-0"
        // ENTER SAVES, ESCAPE CANCELS, from any box in the form - the keys a
        // form has even when it does not look like one. Enter is honoured only
        // from a text box: inside an open dropdown it is that menu's own key
        // for picking the highlighted row, and saving there would commit on
        // the keystroke that chose a country.
        onKeyDown={
          draft
            ? (e) => {
                // An open dropdown eats its own Escape - Radix marks that
                // keystroke handled when it closes the menu. Without this
                // check, Escape out of the country list threw the whole edit
                // away, because the same key bubbled up here.
                if (e.key === "Escape" && !e.defaultPrevented) {
                  e.preventDefault();
                  cancel();
                } else if (e.key === "Enter" && e.target instanceof HTMLInputElement) {
                  e.preventDefault();
                  void commit();
                }
              }
            : undefined
        }
      >
        {/* THE BIO, when the feed gave us one, and the old header when it did
            not. Not a fallback bolted on for safety: a club that entered its
            squad by hand has no bios at all, which is the ordinary case this
            product is actually for, and that squad still needs a profile with
            a face and a name on it. */}
        {/* THE FACE IS THE PHOTO PICKER while editing, and nothing else moves:
            a separate "change photo" button would be a second UI for the same
            fact, sitting beside the face it changes. */}
        <input
          ref={photoInput}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          hidden
          onChange={(e) => setPhoto(e.target.files?.[0] ?? null)}
        />
        {editing ? (
          <BioBand
            bio={bio}
            photoId={player.id}
            photo={player.photo}
            name={draft.name}
            onName={(v) => setDraft({ ...draft, name: v })}
            onPhoto={() => photoInput.current?.click()}
            photoPicked={photo?.name ?? null}
            menu={
              <BandEdit
                onSave={commit}
                onCancel={cancel}
                busy={busy}
                ready={canSave(draft)}
              />
            }
          />
        ) : (
          <BioBand
            bio={bio}
            photoId={player.id}
            photo={player.photo}
            name={player.name || (adding ? "New player" : null)}
            menu={<BandMenu onEdit={openEdit} onAdd={onAdd} />}
          />
        )}

        {/* THE TABS. This panel had eleven stacked sections and the last of
            them - 44 stat rows - sat below a heatmap, a shot map and a
            hexagon. Nothing was findable without scrolling past everything
            else, which is the complaint that produced them.

            THREE, split by the QUESTION each answers: who is he, where does
            he play, and what has he actually done. Not by data source, which
            is how a developer would cut it and is invisible to a coach.

            Seasons and Matches used to be two of them, and that was a split by
            table rather than by question: the matches WERE the season, listed
            flat, so reading "how did he do in the league" meant holding one
            tab in your head while scrolling the other. One list now, with a
            season opening onto its own games.

            The tab does NOT reset when the player changes, deliberately. Left
            on Maps, clicking down the squad list flips through their heatmaps
            one after another, which is the comparison this whole panel exists
            for. Resetting to Overview each time would break it every click. */}
        <Tabs
          value={tab}
          onValueChange={(v) => setTab(v as Tab)}
          // No border-t here: the band above closes itself now (BAND_ROW), so
          // a second line here would double the rule. pt-2 is the 8px the face
          // above the rule keeps below it - the switcher is not closer to the
          // line than the photo is.
          //
          // NOTHING UNDER THE SWITCHER EITHER: the first block of all four
          // tabs used to open with a border-t, which fenced the switcher on
          // both sides and read as a third bar of chrome between the name and
          // the content. The rules INSIDE a tab stay - they separate sections
          // from each other, which is a different job.
          className="px-4 pt-2 pb-4 sm:px-6"
        >
          {/* THE TABLE'S OWN TWO GREYS, so the switcher belongs to the same
              surface as the rows beside it: the track is --card-hover, the
              colour a row takes when you select it, and the active tab is
              --card, the ground the data table itself sits on. The default
              variant's near-black --input-bg read as a hole cut in the card. */}
          <TabsList className="w-full bg-[var(--card-hover)] [&_[data-slot=tabs-trigger][data-state=active]]:bg-[var(--card)]">
            {TABS.map((t) => (
              // THE OTHER THREE ARE DEAD WHILE HE IS BEING ENTERED. Visuals,
              // Seasons and Matches are all drawn from what the pipeline and
              // the feed recorded about a player, and a player who does not
              // exist in the registry yet has none of it - so those tabs would
              // open on three empty states in a row, which reads as a broken
              // panel rather than as an unfinished form. Overview is the form.
              // The same lock holds while EDITING an existing player: the form
              // is here, so leaving the tab would strand a half-typed draft
              // behind a heatmap.
              <TabsTrigger
                key={t}
                value={t}
                disabled={editing && t !== "overview"}
                title={editing && t !== "overview" ? "Finish editing first" : undefined}
                className="capitalize"
              >
                {t}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>

        {/* HIDDEN, not unmounted. Visuals owns the state a visitor sets -
            the shot map's filter and which shot the stepper is on - and
            Seasons and Matches own their scroll position. Unmounting throws
            all of it away, so a glance at Overview and back would silently
            reset the shot you were looking at. */}
        <div className="min-h-0 flex-1 overflow-y-auto">
        <div className={cn("flex h-full min-h-0 flex-col", tab !== "overview" && "hidden")}>
          {/* THE FORM IS THE FACTS, in the same grid and the same cells - an
              edit that opened a dialog over the panel would hide the thing
              being edited behind the form editing it. */}
          {draft ? (
            // The SAME BOX as BioFacts, down to the padding and the pitch
            // beside it, so opening edit mode moves nothing on screen.
            <div className="flex min-w-0 gap-4 px-4 py-3 sm:px-6">
              <EditableFacts draft={draft} setDraft={setDraft} teams={teams} />
              {/* THE PITCH IS ALWAYS HERE while editing, dots or no dots. It
                  draws the positions a player has been PLAYED in, which is the
                  feed's account and not something this form enters - so for a
                  new player it is an empty pitch, and an empty pitch is the
                  honest picture. Dropping it would also move the facts beside
                  it, which is the one thing edit mode must not do. */}
              <div className="w-[104px] shrink-0 self-center">
                <PositionPitch bio={bio ?? {}} />
              </div>
            </div>
          ) : (
            // The feed's age and shirt stand in until a bio records them.
            <BioFacts bio={{ age: entered?.age, shirt: entered?.shirt, ...bio }} />
          )}
          {/* ALWAYS, including for a player who does not exist yet: TraitsHex
              draws six blank axes when there are no percentiles, so the shape
              is there and empty rather than the section being absent. */}
          <div className="flex min-h-0 flex-1 flex-col gap-2 border-t px-4 pt-4 pb-3 sm:px-6">
            <h2 className="font-heading text-base font-medium">Player traits</h2>
            <TraitsHex
              bio={bio ?? {}}
              keeper={(draft ? draft.primary_position : bio?.primary_position ?? player.position) === "Goalkeeper" || player.position === "GK"}
            />
          </div>
        </div>

        {/* VISUALS: the shot map over the heatmap, both on screen at once,
            each scaling into its half so the tab never scrolls. The shot map
            is keyed on the player so its filter and stepper start fresh. */}
        <div className={cn("flex h-full min-h-0 flex-col", tab !== "visuals" && "hidden")}>
          {/* The shot map takes the height it needs and gives it back on a
              short window (flex-initial + min-h-0); the heatmap takes the
              rest, never less than 2/5 (basis): on a short window the two
              then shrink together instead of the heatmap going to a sliver.
              A fixed split left a band of nothing under the shot map. */}
          <div className="flex min-h-0 flex-initial flex-col px-4 pt-4 pb-3 sm:px-6">
            <ShotMap
              key={`${player.id}-${scopeKey}`}
              bio={bio ?? {}}
              days={scopeDays}
              title={scopeIds?.length === 1 ? "Match shot map" : "Season shot map"}
            />
          </div>
          <div className="flex min-h-0 flex-1 basis-2/5 flex-col gap-2 border-t px-4 pt-4 pb-3 sm:px-6">
            <Heatmap bio={bio ?? {}} />
          </div>
        </div>

        {/* SEASONS, EACH OPENING ONTO ITS OWN MATCHES.

            One list instead of two tabs. The matches WERE the seasons, listed
            flat and repeating the competition on every row; nesting them says
            the same thing once and keeps a season's games with the totals they
            add up to.

            ONE COLUMN SET, so one header. A season and a match answer the same
            questions - minutes, goals, assists, cards, rating - and only the
            appearances count is a season's alone, so a match simply leaves
            that cell empty. Two headers would have been two vocabularies for
            one list. */}
        <div className={cn("px-4 pb-4 sm:px-6", tab !== "seasons" && "hidden")}>
          {/* Pinned, so the icons stay over their columns as the list scrolls. */}
          <div className={cn(RUN_COLS, "text-muted-foreground bg-card sticky top-0 z-20 px-2 pt-3 pb-2")}>
            <span className="text-left text-sm">Seasons</span>
            <MatchesIcon label="Appearances" />
            <Timer className="mx-auto size-4" aria-label="Minutes" />
            <BallIcon label="Goals" />
            <BootIcon label="Assists" />
            <span aria-label="Yellow cards" className="bg-warn mx-auto h-3.5 w-2.5 rounded-[2px]" />
            <span aria-label="Red cards" className="bg-bad mx-auto h-3.5 w-2.5 rounded-[2px]" />
            <Star className="mx-auto size-4" aria-label="Rating" />
          </div>
          {seasons.map(({ comp, e }) => {
            const open = openSeason === comp;
            const played = games.filter(({ a }) => (a.competition ?? "Unknown competition") === comp);
            return (
              <div key={comp}>
                <button
                  type="button"
                  // OPEN IS ITS OWN STATE, not the scope. Tying it to the
                  // scoped season looked tidy and could not close: the Data
                  // page falls back to `seasons[0]` when the scope is cleared
                  // (data-browser.tsx), so clearing re-selected the row and it
                  // sprang open again. Opening still scopes - that is the
                  // useful half - but closing is now allowed to just close.
                  onClick={() => {
                    setOpenSeason(open ? null : comp);
                    onSeason(open ? null : comp);
                  }}
                  aria-expanded={open}
                  className={cn(
                    RUN_COLS,
                    "w-full rounded-[var(--radius-md)] px-2 py-2.5 transition-colors",
                    open ? "bg-[var(--card-hover)]" : "hover:bg-[var(--card-hover)]",
                  )}
                >
                  {/* No chevron. The row is the control - clicking it opens
                      or closes - and the open row is already lit, which is the
                      same thing an arrow would have said in less space. */}
                  <span className="min-w-0 text-left">
                    <span className="block truncate text-sm">{club || comp}</span>
                    <span className="text-muted-foreground block truncate text-xs">{comp}</span>
                  </span>
                  <span className="readout text-sm">{e.app}</span>
                  <span className="readout text-sm">{e.mins}</span>
                  <span className="readout text-sm">{e.goals}</span>
                  <span className="readout text-sm">{e.assists}</span>
                  <span className="readout text-sm">{e.yellow}</span>
                  <span className="readout text-sm">{e.red}</span>
                  {e.rating == null ? (
                    <span className="text-muted-foreground">—</span>
                  ) : (
                    <RatingChip value={e.rating} className="mx-auto" />
                  )}
                </button>

                {open && (
                  // Indented under the season, on the season's own left edge:
                  // the games belong to the row above them, and a hairline is
                  // what says so without another box.
                  <div className="ml-3 border-l border-[var(--hairline)] pl-1">
                    {played.map(({ a, f }) => (
                      <button
                        key={a.match_id}
                        type="button"
                        onClick={() => onMatch(matchPick === a.match_id ? "" : a.match_id)}
                        aria-pressed={matchPick === a.match_id}
                        className={cn(
                          RUN_COLS,
                          "w-full rounded-[var(--radius-md)] px-2 py-2 transition-colors",
                          matchPick === a.match_id ? "bg-[var(--card-hover)]" : "hover:bg-[var(--card-hover)]",
                        )}
                      >
                        <span className="min-w-0 text-left">
                          <span className="flex min-w-0 items-center gap-1.5">
                            <span className="min-w-0 truncate text-sm">{f?.opp ?? a.match_id}</span>
                            {f?.res && (
                              <span
                                className={cn(
                                  "grid size-4 shrink-0 place-items-center rounded-[3px] text-[10px] font-semibold",
                                  f.res === "W" && "bg-[color-mix(in_oklab,var(--chart-5)_20%,transparent)] text-[var(--chart-5)]",
                                  f.res === "D" && "bg-[var(--card-hover)] text-[var(--mute)]",
                                  f.res === "L" && "bg-[color-mix(in_oklab,var(--bad)_20%,transparent)] text-[var(--bad)]",
                                )}
                              >
                                {f.res}
                              </span>
                            )}
                            {f?.score && (
                              <span className="readout text-muted-foreground shrink-0 text-xs">
                                {f.score}
                              </span>
                            )}
                          </span>
                          <span className="text-muted-foreground block truncate text-xs">
                            {longDate(f?.day) ?? ""}
                          </span>
                        </span>
                        {/* The appearances column: a match IS one appearance,
                            so counting it here would just print 1 down the
                            column. Empty, and the season above owns it. */}
                        <span />
                        <span className="readout text-sm">{a.minutes ?? "—"}</span>
                        <span className="readout text-sm">{a.goals ?? 0}</span>
                        <span className="readout text-sm">{a.assists ?? 0}</span>
                        <span className="readout text-sm">{a.yellow}</span>
                        <span className="readout text-sm">{a.red}</span>
                        {a.rating == null ? (
                          <span className="text-muted-foreground">—</span>
                        ) : (
                          <RatingChip value={a.rating} className="mx-auto" />
                        )}
                      </button>
                    ))}
                    {/* A season with totals but no rows is a real state: the
                        appearances are there, the FIXTURES they belong to are
                        not in the library yet. */}
                    {!played.length && (
                      <p className="text-muted-foreground px-2 py-2 text-xs">
                        No matches from this season are in the library.
                      </p>
                    )}
                  </div>
                )}
              </div>
            );
          })}
          {!seasons.length && (
            <p className="text-muted-foreground text-sm">No appearances for this player.</p>
          )}
        </div>
        </div>
      </CardContent>
    </Card>
  );
}

