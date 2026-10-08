"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { useSurfaceParams } from "@/components/surface-params";
import { api } from "@/lib/api";
import { useTeam } from "@/lib/club";
import type { Match, MatchReport, Season } from "@/lib/types";
import { statusOf } from "@/lib/status";

/** Polling that stops on its own.
 *
 *  The old UI polled every 5s forever, and every poll re-listed every job. It
 *  also re-requested a thumbnail per card, which made the Python side decode a
 *  video frame per match per poll — those thumbnails are gone from the library
 *  now, but the rule they bought stands: poll only while something is actually
 *  moving, and stop when nothing is.
 */
const POLL_MS = 4000;

/** Every match, SCOPED TO THE CURRENT TEAM.
 *
 *  The scope is read here rather than passed in by each caller, which is what
 *  makes an academy usable at all: a U15 coach opening the app sees eleven
 *  matches, not the four hundred the club has filmed across nine age groups.
 *  Every list in the product — the library, the sidebar reports, the match
 *  switcher — goes through this hook, so they cannot disagree about what "the
 *  matches" are.
 *
 *  An empty scope means the whole club, and so does a scope naming a team that
 *  no longer exists: see lib/club.ts for why that degrades open rather than
 *  shut.
 */
export function useMatches() {
  const [team] = useTeam();
  const [matches, setMatches] = useState<Match[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const refresh = useCallback(async () => {
    try {
      const list = await api.listMatches(team || undefined);
      setMatches(list);
      setError(null);
      return list;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setMatches((prev) => prev ?? []);
      return null;
    }
  }, [team]);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      const list = await refresh();
      if (cancelled) return;
      // Keep going only while a job is live. A library of finished matches
      // costs nothing to leave open.
      if (list?.some((m) => statusOf(m).live)) {
        timer.current = setTimeout(tick, POLL_MS);
      }
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer.current) clearTimeout(timer.current);
    };
  }, [refresh]);

  return { matches, error, refresh };
}

/** The match named by `m` - the URL's on a page, the tab's inside a dock tab
 *  (see components/surface-params.tsx) - kept fresh while it is working. */
export function useMatch() {
  const id = useSurfaceParams().get("m");
  // Stamped with the id it was fetched for, so switching matches shows nothing
  // rather than the previous match's state. Clearing it in the effect instead
  // costs a second render and still renders the stale value once.
  const [got, setGot] = useState<{ id: string | null; match: Match | null }>({
    id: null,
    match: null,
  });
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const match = got.id === id ? got.match : null;

  const refresh = useCallback(async () => {
    if (!id) return null;
    try {
      const m = await api.getMatch(id);
      setGot({ id, match: m });
      setError(null);
      return m;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    }
  }, [id]);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    const tick = async () => {
      const m = await refresh();
      if (cancelled) return;
      if (m && statusOf(m).live) timer.current = setTimeout(tick, POLL_MS);
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer.current) clearTimeout(timer.current);
    };
  }, [id, refresh]);

  return { id, match, error, refresh };
}

/** The match named by `?m=` plus its match.json, once there is one.
 *
 *  Fetched exactly once per (match, has_match) — the report is a static file
 *  the pipeline wrote, so re-requesting it on every status poll would only
 *  re-parse the same megabyte. `loading` is true until we know whether there
 *  is a report at all, so a screen never flashes its empty state first.
 */
export function useReport() {
  const { id, match, error, refresh } = useMatch();
  const ready = match?.has_match ?? false;
  // Stamped with the match it was fetched for, rather than cleared in an
  // effect: clearing on the way in is a second render AND leaves one frame
  // where the previous match's numbers are on screen under the new name.
  const key = `${id}:${ready}`;
  const [got, setGot] = useState<{
    key: string;
    report: MatchReport | null;
    error: string | null;
  }>({ key: "", report: null, error: null });

  useEffect(() => {
    if (!id || !ready) return;
    let cancelled = false;
    api
      .getReport(id)
      .then((r) => !cancelled && setGot({ key, report: r, error: null }))
      .catch(
        (e) =>
          !cancelled &&
          setGot({
            key,
            report: null,
            error: e instanceof Error ? e.message : String(e),
          }),
      );
    return () => {
      cancelled = true;
    };
  }, [id, ready, key]);

  const fresh = got.key === key ? got : null;

  return {
    id,
    match,
    report: fresh?.report ?? null,
    error: error ?? fresh?.error ?? null,
    loading: Boolean(id) && !match && !error,
    refresh,
  };
}

/** Every finished match for a team, joined into per-player season rows.
 *
 *  `enabled` is a parameter rather than a call-site `if`, because this is used
 *  from a document block: a note can hold six tables and only two of them are
 *  season-scoped, and a hook cannot be called conditionally.
 *
 *  Cached at MODULE level, not in state. Six season tables in one document is
 *  six identical requests for a join the server does over every match.json on
 *  disk; the cache makes it one. `reload` drops the entry, which is what the
 *  Refresh button on a table needs — the whole point of a frozen table is that
 *  nothing else refetches behind it.
 */
const seasonCache = new Map<string, Promise<Season>>();

export function useSeason(team?: string, enabled = true) {
  const key = team ?? "";
  const [got, setGot] = useState<{ key: string; season: Season | null; error: string | null }>({
    key: "",
    season: null,
    error: null,
  });
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    let p = seasonCache.get(key);
    if (!p) {
      p = api.getSeason(team);
      seasonCache.set(key, p);
      // A failed fetch must not stay cached, or a blip becomes permanent.
      p.catch(() => seasonCache.delete(key));
    }
    p.then(
      (s) => !cancelled && setGot({ key, season: s, error: null }),
      (e: unknown) =>
        !cancelled &&
        setGot({ key, season: null, error: e instanceof Error ? e.message : String(e) }),
    );
    return () => {
      cancelled = true;
    };
  }, [key, team, enabled, nonce]);

  const reload = useCallback(() => {
    seasonCache.delete(key);
    setNonce((n) => n + 1);
  }, [key]);

  const fresh = got.key === key ? got : null;
  return { season: fresh?.season ?? null, error: fresh?.error ?? null, reload };
}
