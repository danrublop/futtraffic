"use client";

import { Check, Link2, ListFilter, Pencil, Play, Search, Trash2, X } from "lucide-react";
import { useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { TrackRow } from "@/hooks/use-tracks";
import type { ClipEditor } from "@/hooks/use-clip-editor";
import type { Clip } from "@/lib/types";
import { cn } from "@/lib/utils";
import { matchesQuery, type Moment, type MomentKind } from "@/lib/moments";
import { clock } from "@/lib/timecode";

/** What each action is drawn in. Three families and a default: the ball moving
 *  (info), the ball being taken (warn), the ball ending up somewhere (primary).
 *  Anything unlisted — a feed's "Goal — Ramos", a clip's name — stays neutral,
 *  which is also what keeps a real match from being coloured in by accident. */
const KIND_TINT: Record<string, string> = {
  pass: "bg-info/15 text-info",
  cross: "bg-info/15 text-info",
  "throw-in": "bg-info/15 text-info",
  dribble: "bg-primary/15 text-primary",
  corner: "bg-primary/15 text-primary",
  shot: "bg-primary/15 text-primary",
  save: "bg-primary/15 text-primary",
  tackle: "bg-warn/15 text-warn",
  interception: "bg-warn/15 text-warn",
  foul: "bg-warn/15 text-warn",
};

const TEAM_DOT: Record<string, string> = {
  home: "bg-home",
  away: "bg-away",
  official: "bg-warn",
};

/** The rail beside the film: what happened (Events), what you cut (Clips), and
 *  what is drawn over the video (Effects).
 *
 *  Three sub-tabs rather than three panes, because the rail is 360px and the
 *  film needs the rest. Effects is here rather than on the transport bar
 *  because these are properties of the OVERLAY, not of the clock — you set them
 *  once and watch, and every one of them is also a key (B T D X). */
export interface ClipsPanelProps {
  /** Where the rail sits in the film's layout. The grid placement is the
   *  caller's business — the rail only knows it is a column. */
  className?: string;
  /** The clip editor, whole. This was eleven separate props; the hook that owns
   *  the state now names it, and the rail asks for the group. */
  clip: ClipEditor;
  /** The overlay, whole. `tracksError` is the reason a switch is dead: with
   *  nothing to draw they do nothing, and a dead switch with no reason on it
   *  reads as a bug. `offset` lives here rather than in the film's track header,
   *  where a bare number input beside a track name says nothing about what it
   *  does. */
  look: {
    overlay: boolean;
    trail: boolean;
    spotlight: boolean;
    hasTracks: boolean;
    tracksError: string | null;
    tracks: TrackRow[];
    sel: number | null;
    /** The overlay's frame offset, in frames. */
    offset: number;
    toggle: (k: "overlay" | "trail" | "spotlight") => void;
    select: (id: number | null) => void;
    setOffset: (n: number) => void;
  };
  /** Feed events, generated plays and clips, already in time order. Built in
   *  video-panel, which owns the report — this only draws them. */
  moments: Moment[];
  onSeek: (t: number) => void;
  onPlay: (c: Clip) => void;
  onReel: () => void;
}

const KINDS: { k: MomentKind; label: string }[] = [
  { k: "play", label: "Plays" },
  { k: "event", label: "Feed events" },
  { k: "clip", label: "Clips" },
];

const SIDES = [
  { s: "home", label: "Home" },
  { s: "away", label: "Away" },
  { s: "none", label: "Neither" },
];

/** Toggle one member of a Set without mutating the one React is holding. */
const flip = <T,>(set: Set<T>, v: T) => {
  const n = new Set(set);
  if (n.has(v)) n.delete(v);
  else n.add(v);
  return n;
};

export function ClipsPanel(p: ClipsPanelProps) {
  const ed = p.clip;
  const range =
    ed.in != null && ed.out != null && ed.out > ed.in ? `${(ed.out - ed.in).toFixed(1)}s` : null;

  // An empty set means "everything", not "nothing" — a filter nobody has opened
  // must not hide rows.
  const [q, setQ] = useState("");
  const [filterOpen, setFilterOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  /** Controlled, because the two buttons beside the tabs belong to Events and
   *  must not sit over Clips and Effects. */
  const [tab, setTab] = useState("moments");
  const [kinds, setKinds] = useState<Set<MomentKind>>(new Set());
  const [sides, setSides] = useState<Set<string>>(new Set());
  const [who, setWho] = useState<Set<string>>(new Set());
  const active = kinds.size + sides.size + who.size;

  const people = useMemo(() => {
    const n = (w: string) => (w.startsWith("#") ? Number(w.slice(1)) : NaN);
    // Shirt numbers first and in numeric order — sorted as text, #10 lands
    // before #2 — then everything named, alphabetically.
    return [...new Set(p.moments.flatMap((m) => m.who))].sort((a, b) => {
      const [x, y] = [n(a), n(b)];
      if (!isNaN(x) && !isNaN(y)) return x - y;
      if (!isNaN(x)) return -1;
      if (!isNaN(y)) return 1;
      return a.localeCompare(b);
    });
  }, [p.moments]);
  const shown = useMemo(
    () =>
      p.moments.filter(
        (m) =>
          matchesQuery(m, q) &&
          (!kinds.size || kinds.has(m.kind)) &&
          (!sides.size || sides.has(m.side ?? "none")) &&
          (!who.size || m.who.some((w) => who.has(w))),
      ),
    [p.moments, q, kinds, sides, who],
  );
  const followed = p.look.sel != null ? p.look.tracks.find((t) => t.id === p.look.sel) : undefined;

  const clearAll = () => {
    setKinds(new Set());
    setSides(new Set());
    setWho(new Set());
  };

  return (
    /* Not a card. The rail is a column of the editor, flush against the film
       beside it, and its lists run to the bottom of the page rather than into a
       scroll box two thirds of the way down. Below xl there is no room for a
       column, so it drops under the timeline and simply continues: the pane
       scrolls, the lists do not scroll inside themselves. */
    <div className={cn(// No border-l above xl: the draggable divider beside it is that line.
      "flex min-w-0 flex-col border-t xl:h-full xl:min-h-0 xl:border-t-0", p.className)}>
      <Tabs value={tab} onValueChange={setTab} className="flex min-h-0 flex-1 flex-col gap-0">
        {/* Search and filter ride on the TAB ROW, not on a strip of their own
            under it. The rail is 360px and that strip cost a full row of match
            for one text field that is empty almost all of the time — so both
            are buttons of the same size and style, and the typing happens in a
            dialog where there is room for it. */}
        <div className="flex shrink-0 items-center gap-2 p-2">
          <TabsList className="h-7 w-fit shrink-0">
            <TabsTrigger value="moments">Events</TabsTrigger>
            <TabsTrigger value="clips">Clips</TabsTrigger>
            <TabsTrigger value="look">Effects</TabsTrigger>
          </TabsList>
          {tab === "moments" && (
            <span className="ml-auto flex items-center gap-1.5">
              <Button
                size={q ? "sm" : "icon-sm"}
                variant={q ? "default" : "outline"}
                onClick={() => setSearchOpen(true)}
                className="shrink-0 max-sm:h-11 max-sm:w-11"
                aria-label="Search the timeline"
              >
                <Search />
                {q ? <span className="max-w-[10ch] truncate">{q}</span> : null}
              </Button>
              <Button
                size={active ? "sm" : "icon-sm"}
                variant={active ? "default" : "outline"}
                onClick={() => setFilterOpen(true)}
                className="shrink-0 max-sm:h-11 max-sm:w-11"
                aria-label="Filters"
              >
                <ListFilter />
                {active || null}
              </Button>
            </span>
          )}
        </div>

        {/* ------------------------------------------------------- events */}
        <TabsContent value="moments" className="mt-0 flex min-h-0 flex-1 flex-col">
          <Dialog open={searchOpen} onOpenChange={setSearchOpen}>
            <DialogContent className="sm:max-w-md">
              <DialogHeader>
                <DialogTitle>Search</DialogTitle>
                <DialogDescription>Events, plays and clips in this match.</DialogDescription>
              </DialogHeader>
              <form
                onSubmit={(ev) => {
                  ev.preventDefault();
                  setSearchOpen(false);
                }}
                className="grid gap-3"
              >
                <Input
                  autoFocus
                  value={q}
                  onChange={(ev) => setQ(ev.target.value)}
                  placeholder="Search events, plays, clips"
                  className="h-9"
                  aria-label="Search the timeline"
                />
                <div className="flex items-center gap-2">
                  {/* The count, because the list it is filtering is behind the
                      dialog and typing into a modal you cannot see the result
                      of is typing blind. */}
                  <span className="text-muted-foreground flex-1 text-xs">
                    {shown.length} of {p.moments.length} rows
                  </span>
                  <Button type="button" size="sm" variant="ghost" onClick={() => setQ("")} disabled={!q}>
                    <X /> Clear
                  </Button>
                  <Button type="submit" size="sm">
                    Done
                  </Button>
                </div>
              </form>
            </DialogContent>
          </Dialog>

          <Dialog open={filterOpen} onOpenChange={setFilterOpen}>
            <DialogContent className="sm:max-w-md">
              <DialogHeader>
                <DialogTitle>Filter</DialogTitle>
                <DialogDescription>An empty group filters nothing.</DialogDescription>
              </DialogHeader>
              <div className="grid gap-4">
                <Group label="Kind">
                  {KINDS.map(({ k, label }) => (
                    <Chip key={k} on={kinds.has(k)} onClick={() => setKinds((v) => flip(v, k))}>
                      {label}
                    </Chip>
                  ))}
                </Group>
                <Group label="Side">
                  {SIDES.map(({ s: side, label }) => (
                    <Chip
                      key={side}
                      on={sides.has(side)}
                      onClick={() => setSides((v) => flip(v, side))}
                    >
                      {label}
                    </Chip>
                  ))}
                </Group>
                {people.length > 0 && (
                  <Group label="Player or tag">
                    {/* Shirt numbers off the plays, names off the feed, tags off
                        your clips — whatever this match actually names. */}
                    {people.map((w) => (
                      <Chip key={w} on={who.has(w)} onClick={() => setWho((v) => flip(v, w))}>
                        {w}
                      </Chip>
                    ))}
                  </Group>
                )}
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground flex-1 text-xs">
                    {shown.length} of {p.moments.length} rows
                  </span>
                  <Button size="sm" variant="ghost" onClick={clearAll} disabled={!active}>
                    <X /> Clear
                  </Button>
                  <Button size="sm" onClick={() => setFilterOpen(false)}>
                    Done
                  </Button>
                </div>
              </div>
            </DialogContent>
          </Dialog>

          {p.moments.length === 0 ? (
            <p className="text-muted-foreground p-4 text-xs leading-relaxed">
              Nothing to list. This match carries no event feed and no plays — mark your own
              moments as clips instead.
            </p>
          ) : shown.length === 0 ? (
            <p className="text-muted-foreground p-4 text-xs leading-relaxed">
              No row matches. {active ? "Clear the filter" : "Try a shorter search"} to see all{" "}
              {p.moments.length}.
            </p>
          ) : (
            /* ponytail: no "currently playing" highlight — the playhead is
               driven by refs on the animation frame, and lifting it into state
               to light a row re-renders the whole panel 60x a second. */
            <ul className="p-1 xl:min-h-0 xl:flex-1 xl:overflow-y-auto">
              {shown[0]?.demo && (
                /* Said once, at the top, and never on the row: a row that has
                   to caveat itself is a row nobody reads. This list is only
                   ever ALL sample or ALL real — the film panel will not mix
                   them. */
                <li className="text-muted-foreground px-2 pt-1 pb-2 text-xs">
                  Sample events. This match carries no feed and no plays yet.
                </li>
              )}
              {shown.map((m, i) => (
                <li key={`${m.kind}-${m.t}-${i}`}>
                  <button
                    type="button"
                    onClick={() => (m.clip ? p.onPlay(m.clip) : p.onSeek(m.t))}
                    title={
                      m.exact
                        ? undefined
                        : "From the feed: a minute, not a timecode — this lands within 30s of the moment."
                    }
                    className="hover:bg-accent grid w-full grid-cols-[auto_auto_1fr] items-baseline gap-2 rounded-[var(--radius-md)] p-2 text-left"
                  >
                    <i
                      className={`size-2 shrink-0 translate-y-px rounded-full ${
                        m.kind === "clip"
                          ? "bg-info"
                          : (TEAM_DOT[m.side ?? ""] ?? "bg-muted-foreground")
                      }`}
                    />
                    <span className="text-muted-foreground shrink-0 font-mono text-xs tabular-nums">
                      {clock(m.t)}
                    </span>
                    <span className="min-w-0">
                      {/* Shirts first, then what they did: both as chips, so a
                          row is scanned rather than read. The shirt takes the
                          side's colour — the one thing every row of this list
                          has in common is that it belongs to one team or the
                          other. The ~ that used to mark a feed minute is gone
                          from the row; it is in the row's tooltip instead, where
                          it does not compete with the timecode. */}
                      <span className="flex min-w-0 items-center gap-1.5">
                        {m.who.slice(0, 3).map((w) => (
                          <span
                            key={w}
                            className={`shrink-0 rounded-[var(--radius-xs)] px-1.5 py-px text-[11px] font-medium ${
                              m.side === "away"
                                ? "bg-away/20 text-away"
                                : m.side === "home"
                                  ? "bg-home/20 text-home"
                                  : "bg-muted text-muted-foreground"
                            }`}
                          >
                            {w}
                          </span>
                        ))}
                        <span
                          className={`min-w-0 truncate rounded-[var(--radius-xs)] px-1.5 py-px text-[11px] font-medium ${
                            KIND_TINT[m.label.toLowerCase()] ?? "bg-muted text-muted-foreground"
                          }`}
                        >
                          {m.label}
                        </span>
                      </span>
                      {m.detail && (
                        <span className="text-muted-foreground mt-0.5 block truncate text-xs">
                          {m.detail}
                        </span>
                      )}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </TabsContent>

        {/* -------------------------------------------------------- clips */}
        <TabsContent value="clips" className="mt-0 flex min-h-0 flex-1 flex-col">
          <div className="grid shrink-0 gap-2 border-y p-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="bg-muted rounded-md px-2 py-1 font-mono text-xs tabular-nums">
                in <b className="text-primary">{ed.in != null ? clock(ed.in) : "—"}</b> · out{" "}
                <b className="text-primary">{ed.out != null ? clock(ed.out) : "—"}</b>
                {range ? ` · ${range}` : ""}
              </span>
              <Button
                size="xs"
                variant="outline"
                onClick={() => ed.in != null && p.onSeek(ed.in)}
                disabled={ed.in == null}
                className="max-sm:h-11 max-sm:px-3"
              >
                Go to in
              </Button>
              {(ed.in != null || ed.out != null || ed.editing) && (
                <Button
                  size="xs"
                  variant="ghost"
                  onClick={ed.clear}
                  className="max-sm:h-11 max-sm:px-3"
                >
                  {ed.editing ? "Cancel edit" : "Clear"}
                </Button>
              )}
            </div>

            <Input
              value={ed.name}
              onChange={(ev) => ed.setName(ev.target.value)}
              maxLength={120}
              placeholder="Name — e.g. counter from the left"
              className="h-8 max-sm:h-11"
            />

            <div className="flex flex-wrap gap-1 max-sm:gap-1.5">
              {ed.tags.map((t) => (
                <Badge
                  key={t}
                  asChild
                  variant={ed.tagSel.has(t) ? "default" : "outline"}
                  className="max-sm:h-9 max-sm:px-3 max-sm:text-xs"
                >
                  <button type="button" onClick={() => ed.toggleTag(t)}>
                    {t}
                  </button>
                </Badge>
              ))}
            </div>

            <Textarea
              value={ed.note}
              onChange={(ev) => ed.setNote(ev.target.value)}
              maxLength={2000}
              rows={2}
              placeholder="Note (optional)"
              className="min-h-14 text-xs"
            />

            <div className="flex items-center gap-2">
              <Button
                size="sm"
                onClick={ed.save}
                disabled={ed.saving}
                className="max-sm:h-11 max-sm:px-4"
              >
                {ed.editing ? "Save changes" : "Save clip"}
              </Button>
              {ed.error && <span className="text-bad text-xs">{ed.error}</span>}
            </div>
          </div>

          {ed.clips.length === 0 ? (
            <p className="text-muted-foreground p-4 text-xs leading-relaxed">
              None yet. Stop at a moment, press <Kbd>I</Kbd> then <Kbd>O</Kbd> to bracket it, name
              it and save. Tick several and <b className="text-foreground">Reel</b> plays them back
              to back.
            </p>
          ) : (
            <ul className="xl:min-h-0 xl:flex-1 xl:overflow-y-auto">
              {ed.clips.map((c) => (
                <li
                  key={c.id}
                  className={`grid grid-cols-[auto_1fr] items-start gap-2 border-b p-3 text-sm ${
                    ed.editing === c.id ? "bg-muted" : ""
                  }`}
                >
                  <input
                    type="checkbox"
                    className="accent-primary mt-1 size-3.5 max-sm:mt-0 max-sm:size-6"
                    checked={ed.picked.has(c.id)}
                    onChange={(ev) => ed.pick(c.id, ev.target.checked)}
                    aria-label={`Include ${c.name} in the reel`}
                  />
                  <div className="min-w-0">
                    <button
                      type="button"
                      onClick={() => p.onPlay(c)}
                      className="flex w-full min-w-0 items-baseline gap-2 text-left"
                    >
                      <span className="truncate font-medium">{c.name}</span>
                      <span className="text-muted-foreground shrink-0 font-mono text-xs tabular-nums">
                        {clock(c.start)}–{clock(c.end)}
                      </span>
                    </button>
                    {c.tags.length > 0 && (
                      <div className="mt-1 flex flex-wrap gap-1">
                        {c.tags.map((t) => (
                          <Badge key={t} variant="secondary" className="h-4 px-1.5 text-[10px]">
                            {t}
                          </Badge>
                        ))}
                      </div>
                    )}
                    {c.note && (
                      <p className="text-muted-foreground mt-1 text-xs break-words">{c.note}</p>
                    )}
                    <div className="mt-1.5 flex gap-0.5 max-sm:gap-1">
                      <Icon label="Play" onClick={() => p.onPlay(c)}>
                        <Play />
                      </Icon>
                      <Icon label="Edit" onClick={() => ed.edit(c)}>
                        <Pencil />
                      </Icon>
                      <Icon
                        label={ed.copied === c.id ? "Link copied" : "Copy link"}
                        onClick={() => ed.copyLink(c)}
                      >
                        {ed.copied === c.id ? <Check className="text-ok" /> : <Link2 />}
                      </Icon>
                      <Icon label="Delete" onClick={() => ed.remove(c)}>
                        <Trash2 />
                      </Icon>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          )}

          <div className="flex shrink-0 flex-wrap items-center gap-2 border-t p-3">
            <span className="text-muted-foreground flex-1 text-xs">
              {ed.picked.size ? `${ed.picked.size} picked` : "Tick clips for a reel"}
            </span>
            <Button
              size="xs"
              variant="ghost"
              onClick={() => ed.pickAll(true)}
              className="max-sm:h-11 max-sm:px-3"
            >
              All
            </Button>
            <Button
              size="xs"
              variant="ghost"
              onClick={() => ed.pickAll(false)}
              className="max-sm:h-11 max-sm:px-3"
            >
              None
            </Button>
            <Button
              size="sm"
              onClick={p.onReel}
              disabled={!ed.picked.size}
              className="max-sm:h-11 max-sm:px-4"
            >
              <Play /> Reel
            </Button>
          </div>
        </TabsContent>

        {/* --------------------------------------------------------- look */}
        <TabsContent value="look" className="mt-0 grid content-start gap-px border-y">
          {/* Every row is also a key, and the key is on the row: the transport
              bar has no room for five switches and these are not transport. */}
          <Switch
            on={p.look.overlay}
            onClick={() => p.look.toggle("overlay")}
            disabled={!p.look.hasTracks}
            label="Overlay"
            hint="B"
            why={p.look.tracksError}
          />
          <Switch
            on={p.look.trail}
            onClick={() => p.look.toggle("trail")}
            disabled={!p.look.hasTracks}
            label="Trail"
            hint="T"
            why={p.look.tracksError ?? "the path of the player you are following"}
          />
          <Switch
            on={p.look.spotlight}
            onClick={() => p.look.toggle("spotlight")}
            disabled={!p.look.hasTracks}
            label="Spotlight"
            hint="D"
            why={p.look.tracksError ?? "dim everyone else"}
          />

          <div className="flex items-center gap-2 p-3 text-sm">
            <span className="flex-1">Following</span>
            {followed ? (
              <>
                <span className="flex items-center gap-1.5 font-medium">
                  <i
                    className={`size-2 rounded-full ${
                      TEAM_DOT[followed.team ?? ""] ?? "bg-muted-foreground"
                    }`}
                  />
                  {followed.jersey != null ? `#${followed.jersey}` : "?"}
                  <span className="text-muted-foreground text-xs font-normal">t{followed.id}</span>
                </span>
                <Icon label="Stop following (X)" onClick={() => p.look.select(null)}>
                  <X />
                </Icon>
              </>
            ) : (
              <span className="text-muted-foreground text-xs">
                nobody — click a player on the video
              </span>
            )}
          </div>

          <div className="flex items-center gap-2 p-3 text-sm">
            <span className="flex-1">
              Offset
              <span className="text-muted-foreground block text-xs">
                frames — nudge it if the rings lag or lead the players
              </span>
            </span>
            <input
              type="number"
              step={1}
              value={p.look.offset}
              onChange={(e) => p.look.setOffset(parseInt(e.target.value, 10) || 0)}
              aria-label="Frame offset"
              className="bg-background h-8 w-16 rounded-md border px-2 text-right font-mono text-xs tabular-nums"
            />
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}

function Switch({
  on,
  onClick,
  disabled,
  label,
  hint,
  why,
}: {
  on: boolean;
  onClick: () => void;
  disabled?: boolean;
  label: string;
  hint: string;
  why: string | null;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      aria-label={label}
      disabled={disabled}
      onClick={onClick}
      className="hover:bg-accent flex items-center gap-2 p-3 text-left text-sm disabled:opacity-50"
    >
      <span className="flex-1">
        {label}
        {why && <span className="text-muted-foreground block text-xs">{why}</span>}
      </span>
      <kbd className="bg-muted text-muted-foreground rounded border px-1 py-0.5 font-mono text-[10px]">
        {hint}
      </kbd>
      <i
        className={`relative block h-4 w-7 shrink-0 rounded-full transition-colors ${
          on ? "bg-primary" : "bg-muted-foreground/40"
        }`}
      >
        <i
          className={`bg-background absolute top-0.5 block size-3 rounded-full transition-all ${
            on ? "left-3.5" : "left-0.5"
          }`}
        />
      </i>
    </button>
  );
}

function Icon({
  label,
  onClick,
  children,
}: {
  label: string;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          size="icon-xs"
          variant="ghost"
          onClick={onClick}
          aria-label={label}
          className="max-sm:size-11 max-sm:[&_svg:not([class*='size-'])]:size-5"
        >
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
}

/** One overlay switch. `why` is the reason, and it is the reason a DISABLED
 *  row is dead as well: with no tracks to draw these do nothing, and a dead
 *  switch with no reason on it reads as a bug. */

function Group({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-1.5">
      <span className="eyebrow">{label}</span>
      <div className="flex flex-wrap gap-1.5">{children}</div>
    </div>
  );
}

function Chip({
  on,
  onClick,
  children,
}: {
  on: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <Badge asChild variant={on ? "default" : "outline"} className="h-7 px-2.5">
      <button type="button" aria-pressed={on} onClick={onClick}>
        {children}
      </button>
    </Badge>
  );
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="bg-muted text-foreground rounded border px-1 py-0.5 font-mono text-[10px]">
      {children}
    </kbd>
  );
}
