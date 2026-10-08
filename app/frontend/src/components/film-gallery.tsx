"use client";

import { useMemo, useState } from "react";
import { Files, Film, Filter, Folder, FolderPlus } from "lucide-react";
import Link from "next/link";

import { Chip } from "@/components/chip";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useMatches, useSeason } from "@/hooks/use-matches";
import { api } from "@/lib/api";
import { reloadOrg, teamLabel, useOrg, useTeam } from "@/lib/club";
import { useSurfaceParams } from "@/components/surface-params";
import { matchDate, matchTitle, statusOf, TONE_CLASS } from "@/lib/status";
import type { Match } from "@/lib/types";
import { cn } from "@/lib/utils";

/** THE FILM LIBRARY: what the film tab shows before a match is picked.
 *
 *  It used to show the "no match picked" gate — one sentence and a dialog. That
 *  is the wrong shape for the screen a coach opens most: the question is not
 *  "which job id" but "whose film am I watching".
 *
 *  FOLDERS ARE TEAMS, and only teams. A folder chip is the same one the match
 *  library draws, setting the same single team lens (lib/club.ts) — a second
 *  filter here would be two controls disagreeing about which squad you are in.
 *  A new folder is therefore a new SQUAD on the club ladder, saved through the
 *  same upsert the club screen uses, not a filing idea that exists only here.
 *
 *  A PLAYER IS A FILTER, NOT A FOLDER. It was a second row of chips, which put
 *  two rows of folders on screen claiming to be one hierarchy while a player
 *  actually cuts ACROSS the matches in a team rather than containing any of
 *  them. It is a dropdown now, and it is the season join rather than the squad
 *  file: `useSeason` returns only players a human confirmed on a track, with
 *  `by_match` naming the games. That is the only honest membership this app has
 *  — machine identity is 9.1% accurate — so filtering by a player shows film
 *  they are known to be in, and a squad member nobody has bound yet is not
 *  offered a filter that would silently mean "everything".
 *
 *  ponytail: no thumbnails. The match library dropped them because they cost a
 *  video decode per card per poll, and the same rule holds here.
 */
export function FilmGallery() {
  const params = useSurfaceParams();
  const { matches, error } = useMatches();
  const { org } = useOrg();
  const [team, pickTeam] = useTeam();
  const [player, setPlayer] = useState<string | null>(null);
  const [filterOpen, setFilterOpen] = useState(false);
  const [naming, setNaming] = useState(false);
  // The season join is every match.json on disk, so it is fetched when the
  // filter is actually asked for — not on the way into the gallery.
  const { season } = useSeason(team || undefined, filterOpen || Boolean(player));

  const squad = season?.players ?? [];
  // Derived, not stored: switching team drops the player on its own, because a
  // player who is not in this season's list simply does not resolve.
  const picked = squad.find((p) => p.id === player) ?? null;

  const shown = useMemo(
    () =>
      (matches ?? [])
        .filter((m) => !picked || m.id in picked.by_match)
        .sort((a, b) => (b.created ?? 0) - (a.created ?? 0)),
    [matches, picked],
  );

  // Picking a film both selects the match — the app's one match lens, so Data
  // and the report follow you — and drops `lib`, which is what was asking for
  // this gallery in the first place.
  const open = (id: string) => params.set({ m: id, lib: null }, { push: true });

  /** A new folder is a new squad. Teams are a WHOLE-LIST upsert server-side (a
   *  team missing from the list is removed from the ladder), so the existing
   *  ones go back with it; `club` and `coaches` are left out entirely, which is
   *  how save_org knows not to touch them. */
  const addFolder = async (raw: string) => {
    setNaming(false);
    const name = raw.trim();
    if (!name || !org) return;
    const saved = await api.saveOrg({
      teams: [
        ...org.teams.map((t) => ({
          team_id: t.team_id,
          name: t.name,
          age_band: t.age_band,
          gender: t.gender,
          level: t.level,
          season: t.season,
          coach_ids: t.coach_ids,
        })),
        { name },
      ],
    });
    reloadOrg();
    // Stand in the folder you just made. The id is server-minted, so it is only
    // knowable from the reply.
    const made = saved.teams.find((t) => t.name === name);
    if (made) pickTeam(made.team_id);
  };

  return (
    <div className="grid w-full min-w-0 content-start gap-4 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Chip label="All videos" icon={Files} active={!team} onClick={() => pickTeam("")} />
        {(org?.teams ?? []).map((t) => (
          <Chip
            key={t.team_id}
            label={teamLabel(t)}
            icon={Folder}
            active={team === t.team_id}
            onClick={() => pickTeam(t.team_id)}
          />
        ))}
        {naming && (
          <input
            autoFocus
            placeholder="Folder name"
            onBlur={(e) => void addFolder(e.currentTarget.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") e.currentTarget.blur();
              if (e.key === "Escape") setNaming(false);
            }}
            className="h-9 w-44 rounded-[var(--radius-md)] border border-[var(--accent)] bg-[var(--card)] px-2.5 text-[12.5px] text-[var(--ink)] focus-visible:outline-none"
          />
        )}

        {/* The editor's own buttons: the same icon-only outline square the
            film rail uses for search and filter, filled while it is doing
            something. A label on a button that is only ever one icon wide is
            a label the gallery pays for on every row. */}
        <span className="ml-auto flex items-center gap-1.5">
          <Button
            size="icon-sm"
            variant="outline"
            onClick={() => setNaming(true)}
            aria-label="New folder"
            title="New folder"
            className="shrink-0 max-sm:size-11"
          >
            <FolderPlus />
          </Button>
          <DropdownMenu open={filterOpen} onOpenChange={setFilterOpen}>
            <DropdownMenuTrigger asChild>
              <Button
                size="icon-sm"
                variant={picked ? "default" : "outline"}
                aria-label={picked ? `Filtered to ${picked.name ?? "one player"}` : "Filter by player"}
                title={picked ? (picked.name ?? "Filtered") : "Filter by player"}
                className="shrink-0 max-sm:size-11"
              >
                <Filter />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuItem onSelect={() => setPlayer(null)}>Everyone</DropdownMenuItem>
              {squad.map((p) => (
                <DropdownMenuItem key={p.id} onSelect={() => setPlayer(p.id)} className="gap-2">
                  <span className="min-w-0 flex-1 truncate">
                    {p.name ?? (p.number != null ? `#${p.number}` : p.id)}
                  </span>
                  <span className="text-muted-foreground shrink-0 text-xs">
                    {Object.keys(p.by_match).length}
                  </span>
                </DropdownMenuItem>
              ))}
              {season && squad.length === 0 && (
                <p className="text-muted-foreground px-2 py-2 text-xs">
                  No player has been confirmed on film here yet — name them on a
                  match and they can be filtered for.
                </p>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        </span>
      </div>

      {error && (
        <p className="text-sm text-[var(--danger)]">Could not reach the matches — {error}</p>
      )}

      {matches === null ? (
        <p className="text-[12px] text-[var(--mute)]">Loading…</p>
      ) : shown.length === 0 ? (
        <p className="max-w-[42ch] text-[12px] text-[var(--mute)]">
          {picked
            ? `No film of ${picked.name ?? "this player"} yet.`
            : team
              ? "No film in this folder yet."
              : null}{" "}
          <Link href="/workspace?v=data" className="underline underline-offset-2">
            Upload a match on the Data page
          </Link>{" "}
          and it opens here.
        </p>
      ) : (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-3">
          {shown.map((m) => (
            <FilmCard
              key={m.id}
              match={m}
              minutes={picked?.by_match[m.id]?.minutes}
              onOpen={() => open(m.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/** One match, as film. The status is on the card because a job still tracking
 *  plays back with no overlay on it, and a card that does not say so is a card
 *  that looks broken when you open it. */
function FilmCard({
  match,
  minutes,
  onOpen,
}: {
  match: Match;
  /** Minutes this player was confirmed for, while filtered to one. */
  minutes?: number;
  onOpen: () => void;
}) {
  const view = statusOf(match);
  // There is nothing to play until the whole file has arrived — the video route
  // serves the upload, not the pipeline's output.
  const playable = match.upload_complete;
  // `upload_complete` is true on feed-imported matches that never had a video,
  // so it is not enough on its own: the thumbnail 404s and the card falls back
  // to the icon rather than showing a broken image.
  const [broken, setBroken] = useState(false);

  return (
    <button
      type="button"
      onClick={onOpen}
      disabled={!playable}
      title={playable ? matchTitle(match) : view.detail}
      className={cn(
        "group grid overflow-hidden rounded-[var(--radius-md)] border border-[var(--hairline)] bg-[var(--card)] text-left transition-colors",
        playable ? "hover:bg-[var(--card-hover)]" : "opacity-60",
      )}
    >
      {/* A still from the match, or the icon when there is nothing to show.
          `loading="lazy"` IS the concurrency control: a library of fifty cards
          decodes only what is on screen, and the browser's own per-origin limit
          caps the rest — which is why there is no queue on the server side.
          Decorative, so no alt text; the title above already names the match. */}
      <span className="grid aspect-video place-items-center overflow-hidden bg-[var(--card-hover)]">
        {playable && !broken ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={api.thumbUrl(match.id, 240)}
            alt=""
            aria-hidden
            loading="lazy"
            onError={() => setBroken(true)}
            className="size-full object-cover"
          />
        ) : (
          <Film className="size-6 text-[var(--graphite)]" strokeWidth={1.5} />
        )}
      </span>
      <span className="grid gap-1 p-2.5">
        <span className="truncate text-[13px] text-[var(--ink)]">{matchTitle(match)}</span>
        <span className="flex items-center gap-2 text-[11px] text-[var(--mute)]">
          <span className="truncate">{matchDate(match) ?? ""}</span>
          {minutes != null && <span className="shrink-0">{Math.round(minutes)}′</span>}
          <span className={cn("ml-auto shrink-0", TONE_CLASS[view.tone])}>{view.label}</span>
        </span>
      </span>
    </button>
  );
}
