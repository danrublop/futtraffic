"use client";

import { useCallback, useEffect, useState, useSyncExternalStore } from "react";

import { api } from "./api";
import type { Org } from "./types";

/** THE CLUB, and the team you are currently looking at.
 *
 *  An academy is a ladder of squads, so almost every screen in the app is
 *  really asking "…for which team?". Rather than add a team parameter to each
 *  one, the answer lives here once and the screens read it — the same shape as
 *  `?m=` for the match, except the team is NOT in the URL.
 *
 *  WHY NOT THE URL. The match is a thing you link to and send someone; the team
 *  is a lens you set once at the start of the season and forget. Putting it in
 *  the query string would mean every internal link had to carry it or silently
 *  drop you back to the whole club, which is the bug `withMatch` exists to
 *  avoid for `?m=` — one such parameter is a helper, two is a class of bug.
 *
 *  Persisted in localStorage, and deliberately NOT validated against the ladder
 *  here: a team removed from the club leaves a stale id, which resolves to no
 *  team, which reads as the whole club. That is the right failure — the coach
 *  sees everything, not nothing.
 */
const KEY = "futtraffic.team";

/** "" means the whole club, which is also what an unknown id degrades to. */
let current = "";
let hydrated = false;
const listeners = new Set<() => void>();

function read(): string {
  if (hydrated) return current;
  try {
    current = window.localStorage.getItem(KEY) ?? "";
  } catch {
    current = ""; // private mode: the scope is just not remembered
  }
  hydrated = true;
  return current;
}

function subscribe(fn: () => void) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function setTeam(teamId: string) {
  current = teamId || "";
  hydrated = true;
  try {
    if (current) window.localStorage.setItem(KEY, current);
    else window.localStorage.removeItem(KEY);
  } catch {
    /* not remembered, still applied for this session */
  }
  listeners.forEach((fn) => fn());
}

/** The team scope. The server snapshot is "" because these routes are
 *  prerendered for the desktop shell, and a scope read during SSR would not
 *  match the client's first paint. */
export function useTeam(): [string, (id: string) => void] {
  const team = useSyncExternalStore(subscribe, read, () => "");
  return [team, setTeam];
}

/** The club itself. Module-cached: the ladder is on the sidebar, the upload
 *  form, the club screen and the database tab at once, and it is one small
 *  object that changes only when somebody edits it. */
let cache: Promise<Org> | null = null;
const orgListeners = new Set<() => void>();

function load(): Promise<Org> {
  if (!cache) {
    cache = api.getOrg();
    cache.catch(() => {
      cache = null; // a blip must not become permanent
    });
  }
  return cache;
}

/** Drop the cache and re-notify everyone. Called after a save, so the sidebar
 *  and the upload form pick up a new team without a reload. */
export function reloadOrg() {
  cache = null;
  orgListeners.forEach((fn) => fn());
}

export function useOrg() {
  const [got, setGot] = useState<{ org: Org | null; error: string | null }>({
    org: null,
    error: null,
  });
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    const bump = () => setNonce((n) => n + 1);
    orgListeners.add(bump);
    return () => {
      orgListeners.delete(bump);
    };
  }, []);

  useEffect(() => {
    let live = true;
    load().then(
      (o) => live && setGot({ org: o, error: null }),
      (e: unknown) =>
        live &&
        setGot({ org: null, error: e instanceof Error ? e.message : String(e) }),
    );
    return () => {
      live = false;
    };
  }, [nonce]);

  const reload = useCallback(() => reloadOrg(), []);
  return { ...got, reload };
}

/** How a team is written wherever one line has to carry it: "U15 Boys · B".
 *  The age band is NOT prefixed — it is almost always already in the name, and
 *  "U15 · U15 Boys" is the kind of duplication that makes a sidebar unreadable. */
export function teamLabel(t: { name: string; level?: string }): string {
  return t.level ? `${t.name} · ${t.level}` : t.name;
}
