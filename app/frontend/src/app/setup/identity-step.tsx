"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Merge, Scissors, Undo2, UserX } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { api } from "@/lib/api";
import type { Match, RosterPlayer, TrackSummary, TracksDigest } from "@/lib/types";

/** What jobs.py stores per track once a human has spoken.
 *
 *  `player` is the load-bearing field and the easiest one to omit: it is a
 *  season roster id, and `jobs.py :: _rebind` binds this track to that player
 *  ONLY when it is present and already known to the roster. Send name/jersey
 *  alone and the match still reads correctly — and the cross-match join, which
 *  is the actual product, silently never happens. This screen went out that way
 *  once; see `attachPlayer`.
 */
interface Assignment {
  player?: string;
  name?: string;
  jersey?: number | null;
  team?: string | null;
}

interface Pending {
  assign: Record<number, Assignment>;
  merge: [number, number][];
  /** One track that is really two people: everything from `at_frame` on
   *  becomes a new id. The server mints the id; the screen never guesses it. */
  split: { track: number; at_frame: number }[];
  exclude: number[];
  undo: number[];
}

const emptyPending = (): Pending => ({ assign: {}, merge: [], split: [], exclude: [], undo: [] });

/** The digest's `corrections` block is typed `unknown` because the API is
 *  Python. Read it defensively rather than casting: a shape change should show
 *  as "nothing assigned", never as a crash on the one screen a human uses. */
function readCorrections(d: TracksDigest | null) {
  const c = (d?.corrections ?? {}) as Record<string, unknown>;
  const assign: Record<number, Assignment> = {};
  if (c.assign && typeof c.assign === "object")
    for (const [k, v] of Object.entries(c.assign as Record<string, unknown>))
      if (v && typeof v === "object") assign[Number(k)] = v as Assignment;
  const exclude = Array.isArray(c.exclude) ? (c.exclude as number[]) : [];
  return { assign, exclude };
}

/** Operator minutes per match is the measurement that decides whether this is
 *  software or a services business. Wall clock would count lunch, so only
 *  intervals shorter than this accumulate. */
const IDLE_MS = 20_000;

export function IdentityStep({ match }: { match: Match }) {
  const [digest, setDigest] = useState<TracksDigest | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<Pending>(emptyPending);
  const [sel, setSel] = useState<ReadonlySet<number>>(new Set());
  const [last, setLast] = useState<number | null>(null);
  const [dirty, setDirty] = useState(false);
  const [name, setName] = useState("");
  const [jersey, setJersey] = useState("");
  const [position, setPosition] = useState("");
  const [team, setTeam] = useState<"home" | "away">("home");
  const [roster, setRoster] = useState<RosterPlayer[]>([]);

  /** The club this match belongs to, which is the roster's key. Without it
   *  there is nothing to file a season player under — `jobs.save_roster` drops
   *  any entry with a blank team — so the roster path is skipped entirely and
   *  this screen behaves exactly as it did before. */
  const club = match.team?.trim() || null;

  useEffect(() => {
    if (!club) return;
    let live = true;
    api
      .getRoster()
      .then((r) => live && setRoster(r.filter((p) => p.team === club && p.name)))
      // A roster that will not load must not block naming: it costs the season
      // join, not the match.
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [club]);

  // Kept in refs, not state: the clock must not re-render the track grid twice
  // a second, and nothing on screen reads it live.
  const activeMs = useRef(0);
  const sentMs = useRef(0);
  const lastEvt = useRef(0);
  const flushTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const tick = () => {
      const n = Date.now();
      if (lastEvt.current && n - lastEvt.current < IDLE_MS)
        activeMs.current += n - lastEvt.current;
      lastEvt.current = n;
    };
    const events = ["pointerdown", "keydown", "wheel", "pointermove"] as const;
    for (const e of events) addEventListener(e, tick, { passive: true });
    return () => {
      for (const e of events) removeEventListener(e, tick);
    };
  }, []);

  // Bumped after a structural edit: the server renumbers tracks, so the digest
  // has to be re-read or the next correction targets ids that no longer exist.
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let live = true;
    const load = async () => {
      try {
        const d = await api.getTracksDigest(match.id);
        if (!live) return;
        setDigest(d);
        setError(null);
      } catch (e) {
        if (live) setError(e instanceof Error ? e.message : String(e));
      }
    };
    void load();
    return () => {
      live = false;
    };
  }, [match.id, reloadKey]);

  const flush = useCallback(
    async (body: Pending, structural: boolean) => {
      const seconds = (activeMs.current - sentMs.current) / 1000;
      sentMs.current = activeMs.current;
      try {
        await api.saveIdentity(match.id, { ...body, operator_seconds: seconds });
        setDirty(false);
        // Merges renumber tracks server-side, so the view has to be re-read;
        // assignments are just labels and the optimistic view is already right.
        if (structural) {
          setSel(new Set());
          setReloadKey((k) => k + 1);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [match.id],
  );

  /** Structural edits flush at once because the server renumbers tracks;
   *  assignments batch, because every flush re-derives a tracks file that is
   *  tens of MB on a real match. */
  const queue = (patch: Partial<Pending>, structural: boolean) => {
    setDirty(true);
    setPending((p) => {
      const next: Pending = {
        assign: { ...p.assign, ...(patch.assign ?? {}) },
        merge: [...p.merge, ...(patch.merge ?? [])],
        split: [...p.split, ...(patch.split ?? [])],
        exclude: [...p.exclude, ...(patch.exclude ?? [])],
        undo: [...p.undo, ...(patch.undo ?? [])],
      };
      if (patch.undo?.length) {
        for (const id of patch.undo) delete next.assign[id];
        next.exclude = next.exclude.filter((i) => !patch.undo!.includes(i));
        next.merge = next.merge.filter(
          ([a, b]) => !patch.undo!.includes(a) && !patch.undo!.includes(b),
        );
        next.split = next.split.filter((s) => !patch.undo!.includes(s.track));
      }
      if (flushTimer.current) clearTimeout(flushTimer.current);
      if (structural) {
        void flush(next, true);
        return emptyPending();
      }
      flushTimer.current = setTimeout(() => {
        void flush(next, false);
        setPending(emptyPending);
      }, 1500);
      return next;
    });
  };

  const server = useMemo(() => readCorrections(digest), [digest]);
  const tracks: TrackSummary[] = digest?.tracks ?? [];

  const assignOf = (id: number): Assignment | undefined =>
    pending.assign[id] ?? server.assign[id];
  const isExcluded = (id: number) =>
    pending.exclude.includes(id) || server.exclude.includes(id);

  const named = tracks.filter((t) => assignOf(t.id)).length;
  const excluded = tracks.filter((t) => isExcluded(t.id)).length;
  const done = named + excluded;

  // ---- selection ----------------------------------------------------------
  const onCardClick = (id: number, e: React.MouseEvent) => {
    setSel((prev) => {
      const next = new Set(prev);
      if (e.shiftKey && last != null) {
        const order = tracks.map((t) => t.id);
        const [a, b] = [order.indexOf(last), order.indexOf(id)].sort((x, y) => x - y);
        for (const i of order.slice(a, b + 1)) next.add(i);
      } else if (e.metaKey || e.ctrlKey) {
        if (next.has(id)) next.delete(id);
        else next.add(id);
      } else {
        const only = next.size === 1 && next.has(id);
        next.clear();
        if (!only) next.add(id);
      }
      return next;
    });
    setLast(id);
  };

  // ---- actions ------------------------------------------------------------

  /** Give the assignment its season roster id, creating the player if this is
   *  the first week they have been named.
   *
   *  The id must come back from the server: `roster.py` mints it by
   *  accent-folding the name, and a locally-guessed slug that disagrees by one
   *  character binds to nothing without erroring.
   *
   *  A number alone is not a person — shirts get reassigned across a season —
   *  so only a typed name earns a roster entry. Everything else still assigns
   *  fine, it just stays local to this match. */
  const attachPlayer = async (a: Assignment, typed: string, pos: string): Promise<Assignment> => {
    if (!club || !typed) return a;
    const hit = roster.find((p) => p.name?.toLowerCase() === typed.toLowerCase());
    // A known player still gets an upsert when a position has been typed that
    // the roster does not have yet — otherwise the field is unfillable for
    // everyone who was named before positions existed.
    if (hit && (!pos || hit.position === pos)) return { ...a, player: hit.id };
    try {
      const next = await api.saveRoster([
        ...roster.filter((p) => p.id !== hit?.id),
        { ...(hit ?? {}), name: typed, number: a.jersey ?? hit?.number ?? null, team: club, position: pos || hit?.position || null },
      ]);
      const mine = next.filter((p) => p.team === club && p.name);
      setRoster(mine);
      const made = mine.find((p) => p.name?.toLowerCase() === typed.toLowerCase());
      return made ? { ...a, player: made.id } : a;
    } catch {
      // Naming the match is the operator's actual task; losing the season join
      // is bad, losing their click is worse.
      return a;
    }
  };

  const assign = async () => {
    if (!sel.size || (!name.trim() && !jersey.trim())) return;
    const j = jersey.trim() === "" ? null : Number(jersey);
    const typed = name.trim();
    const pos = position.trim();
    const ids = [...sel];
    setSel(new Set());
    setName("");
    setJersey("");
    setPosition("");
    const a = await attachPlayer(
      { name: typed || (j != null ? `#${j}` : ""), jersey: j, team },
      typed,
      pos,
    );
    queue(
      { assign: Object.fromEntries(ids.map((id) => [id, a])) as Record<number, Assignment> },
      false,
    );
  };

  const merge = () => {
    const ids = [...sel];
    if (ids.length < 2) return;
    // Keep the longest: it carries the most detections, so the fewest rows get
    // rewritten and its own id stays meaningful in the raw data.
    const keep = ids
      .map((i) => tracks.find((t) => t.id === i))
      .filter((t): t is TrackSummary => !!t)
      .sort((a, b) => b.n - a.n)[0]?.id;
    if (keep == null) return;
    queue({ merge: ids.filter((i) => i !== keep).map((i) => [keep, i] as [number, number]) }, true);
  };

  /** The one track being split, and where. Structural like a merge: the server
   *  renumbers, so it flushes at once and the digest is re-read. */
  const [splitting, setSplitting] = useState<TrackSummary | null>(null);
  const [splitAt, setSplitAt] = useState(0);
  const openSplit = () => {
    const t = tracks.find((x) => x.id === [...sel][0]);
    if (!t) return;
    setSplitAt(t.mid_f);
    setSplitting(t);
  };

  const exclude = () => sel.size && queue({ exclude: [...sel] }, true);
  const undo = () => sel.size && queue({ undo: [...sel] }, true);

  // ---- render -------------------------------------------------------------
  if (error && !digest)
    return (
      <Alert variant="destructive">
        <AlertTitle>No tracks to name yet</AlertTitle>
        <AlertDescription>
          {error}. This step needs tracking to have finished; it has nothing to do until
          then.
        </AlertDescription>
      </Alert>
    );

  if (!digest)
    return (
      <div className="grid gap-3">
        <Skeleton className="h-16 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );

  const groups: [string, TrackSummary[]][] = [
    ["To name", tracks.filter((t) => !assignOf(t.id) && !isExcluded(t.id))],
    ["Named", tracks.filter((t) => assignOf(t.id) && !isExcluded(t.id))],
    ["Not players", tracks.filter((t) => isExcluded(t.id))],
  ];

  return (
    <div className="grid gap-4">
      <Card>
        <CardContent className="grid gap-3">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
            <div className="min-w-0 flex-1">
              <div className="text-sm font-medium">
                {named} of {tracks.length} tracks named
                {excluded ? ` · ${excluded} marked not a player` : ""}
              </div>
              <Progress
                value={tracks.length ? (done / tracks.length) * 100 : 0}
                className="mt-2 h-1"
              />
            </div>
            <span className="text-muted-foreground text-xs">
              {dirty ? "saving…" : "saved"}
            </span>
          </div>
          <p className="text-muted-foreground text-xs">
            This step is optional — the report works without it. Skip it and every row is
            a <i>track</i> rather than a player: still measured, just not named. Names
            make it a team sheet and let a player be followed across matches.
          </p>
          {!club && (
            <p className="text-warn text-xs">
              This match has no team, so names stay local to it — nothing here will join
              up with next week. Set the team on the match to build a season profile.
            </p>
          )}
        </CardContent>
      </Card>

      {/* ---- the one toolbar --------------------------------------------- */}
      <Card className="sticky top-14 z-10">
        <CardContent className="flex flex-wrap items-center gap-2">
          {/* Native <datalist>, not a combobox: it is one element, it filters
              and keyboard-navigates itself, and it still accepts a name that is
              not on the list — which is every player's first week. */}
          <Input
            placeholder="Name"
            list="roster-names"
            autoComplete="off"
            value={name}
            onChange={(e) => {
              const v = e.target.value;
              setName(v);
              // Picking a known player fills their shirt number, so the common
              // case is type-two-letters-pick-enter.
              const hit = roster.find((p) => p.name?.toLowerCase() === v.toLowerCase());
              if (hit?.number != null) setJersey(String(hit.number));
              if (hit?.position) setPosition(hit.position);
            }}
            onKeyDown={(e) => e.key === "Enter" && void assign()}
            className="h-8 w-40"
          />
          <datalist id="roster-names">
            {roster.map((p) => (
              <option key={p.id} value={p.name ?? ""}>
                {p.number != null ? `#${p.number}` : ""}
              </option>
            ))}
          </datalist>
          <Input
            placeholder="#"
            inputMode="numeric"
            value={jersey}
            onChange={(e) => setJersey(e.target.value.replace(/\D/g, ""))}
            onKeyDown={(e) => e.key === "Enter" && void assign()}
            className="h-8 w-16"
          />
          {/* The position, typed once per player and kept on the roster. It is
              the ONLY honest source of one: nothing derives a position from
              where a player averaged, because the sides swap at half time and
              the pipeline has no attacking direction to correct for it. It is
              what lets the notepad group a season by position. */}
          <Input
            placeholder="Pos"
            title="Position — typed once, kept on the roster, used to group the season"
            aria-label="Position"
            value={position}
            onChange={(e) => setPosition(e.target.value.slice(0, 12))}
            onKeyDown={(e) => e.key === "Enter" && void assign()}
            className="h-8 w-16"
          />
          <Select value={team} onValueChange={(v) => setTeam(v === "away" ? "away" : "home")}>
            <SelectTrigger size="sm" className="w-28">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="home">Home</SelectItem>
              <SelectItem value="away">Away</SelectItem>
            </SelectContent>
          </Select>
          <Button
            size="sm"
            onClick={() => void assign()}
            disabled={!sel.size || (!name.trim() && !jersey.trim())}
          >
            Assign to {sel.size || "…"}
          </Button>
          <span className="bg-border mx-1 h-6 w-px" />
          <Button size="sm" variant="outline" onClick={merge} disabled={sel.size < 2}>
            <Merge /> Merge
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={openSplit}
            disabled={sel.size !== 1}
            title="One track that is really two players - cut it where they cross"
          >
            <Scissors /> Split
          </Button>
          <Button size="sm" variant="destructive" onClick={exclude} disabled={!sel.size}>
            <UserX /> Not a player
          </Button>
          <Button size="sm" variant="ghost" onClick={undo} disabled={!sel.size}>
            <Undo2 /> Undo
          </Button>
          <span className="text-muted-foreground ml-auto text-xs">
            Click to select · shift-click for a range · ⌘-click to add
          </span>
        </CardContent>
      </Card>

      <Dialog open={!!splitting} onOpenChange={(o) => !o && setSplitting(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Split track {splitting?.id}</DialogTitle>
            <DialogDescription>
              Scrub to the first frame that is the OTHER player. Everything from there on
              becomes a new track, which you then name separately.
            </DialogDescription>
          </DialogHeader>
          {splitting && (
            <div className="grid gap-2">
              {/* eslint-disable-next-line @next/next/no-img-element -- a job-scoped
                  crop route, not a static asset */}
              <img
                src={api.cropUrl(match.id, splitting.id, 220, splitAt)}
                alt={`track ${splitting.id} at frame ${splitAt}`}
                className="bg-muted mx-auto h-[220px] w-auto rounded"
              />
              <input
                type="range"
                min={splitting.first_f + 1}
                max={splitting.last_f}
                value={splitAt}
                onChange={(e) => setSplitAt(Number(e.target.value))}
                className="w-full"
              />
              <p className="text-muted-foreground text-center text-xs">
                frame {splitAt} of {splitting.first_f}-{splitting.last_f}
                {digest?.fps ? ` · ${(splitAt / (digest.fps as number)).toFixed(1)}s` : ""}
              </p>
            </div>
          )}
          <DialogFooter>
            <Button variant="ghost" onClick={() => setSplitting(null)}>
              Cancel
            </Button>
            <Button
              onClick={() => {
                if (splitting) queue({ split: [{ track: splitting.id, at_frame: splitAt }] }, true);
                setSplitting(null);
              }}
            >
              Split here
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {error && <p className="text-bad text-xs">{error}</p>}

      {tracks.length === 0 ? (
        <Alert>
          <AlertTitle>No tracks yet</AlertTitle>
          <AlertDescription className="text-muted-foreground">
            The pipeline has not produced any. Nothing to name until tracking finishes.
          </AlertDescription>
        </Alert>
      ) : (
        groups.map(([label, list]) =>
          list.length ? (
            <section key={label} className="grid gap-2">
              <h2 className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
                {label} · {list.length}
              </h2>
              <div className="grid grid-cols-[repeat(auto-fill,minmax(108px,1fr))] gap-2">
                {list.map((t) => {
                  const a = assignOf(t.id);
                  const ex = isExcluded(t.id);
                  const num = a?.jersey ?? t.jersey;
                  return (
                    <button
                      key={t.id}
                      type="button"
                      onClick={(e) => onCardClick(t.id, e)}
                      className={`bg-muted overflow-hidden rounded-md text-left outline-2 outline-transparent ${
                        sel.has(t.id) ? "outline-primary" : ""
                      } ${ex ? "opacity-40" : ""}`}
                    >
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={api.cropUrl(match.id, t.id)}
                        alt={`Track ${t.id}`}
                        loading="lazy"
                        className="bg-card block aspect-[11/20] w-full object-cover"
                        onError={(e) => (e.currentTarget.style.visibility = "hidden")}
                      />
                      <div className="grid gap-1 p-2">
                        <div className="flex items-center gap-1">
                          <Badge variant={a?.jersey != null ? "default" : "secondary"}>
                            {num != null ? `#${num}` : "?"}
                          </Badge>
                          {t.confidence != null && (
                            <span
                              className={`ml-auto text-[10px] ${
                                t.confidence < 0.6 ? "text-warn" : "text-muted-foreground"
                              }`}
                            >
                              {Math.round(t.confidence * 100)}%
                            </span>
                          )}
                        </div>
                        <div
                          className={`truncate text-xs ${a ? "text-ok" : "text-muted-foreground"}`}
                        >
                          {ex ? "not a player" : a?.name || "unnamed"}
                        </div>
                        <div className="text-muted-foreground text-[11px]">
                          id {t.id} · {t.seconds ?? Math.round((t.last_f - t.first_f) / (digest.fps || 25))}s
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </section>
          ) : null,
        )
      )}

      {tracks.length > 0 && (
        <Alert>
          <AlertTitle className="text-ok">
            {done === tracks.length
              ? "Every track is accounted for."
              : "You can stop at any point."}
          </AlertTitle>
          <AlertDescription className="text-muted-foreground">
            Corrections save as you go and the match rebuilds itself from them — there is
            no separate submit. Read the accuracy page before the report: it is the one
            that says whether to believe the numbers.
            <span className="mt-2 flex flex-wrap gap-3">
              <Link className="text-primary" href={`/checks?m=${encodeURIComponent(match.id)}`}>
                Accuracy →
              </Link>
              <Link className="text-primary" href={`/workspace?v=data&m=${encodeURIComponent(match.id)}`}>
                Match data →
              </Link>
            </span>
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}
