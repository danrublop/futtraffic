import { create } from "zustand";

/** THE DOCK — notes, film and data open beside the chat.
 *
 *  This brings back ONE thing from the pane manager deleted on 2026-09-12:
 *  several surfaces open at once. Nothing else. There are no splits and no
 *  arrangement - a tab strip, one visible tab, beside the chat and only on the
 *  chat. See database-is-the-backbone (not published) for what was removed and why.
 *
 *  Two rules carry the weight:
 *
 *  - ONLY THE ACTIVE TAB IS MOUNTED. A hidden film would hold its video and a
 *    tracks index of tens to hundreds of MB; a hidden note editor would keep
 *    autosaving a stale copy over any other writer (the agent PUTs notes
 *    too). Each tab keeps its PARAMS here instead, so coming back restores the
 *    match, the note and the table - and the note's own unmount flush saves
 *    the last keystrokes on the way out.
 *  - ONE TAB PER NOTE. Two editors on one file is last-write-wins with nothing
 *    on screen saying so: close one and its flush writes its copy over the
 *    other's edits. Opening a note that is already open focuses that tab.
 *
 *  Kept in localStorage, per browser, like every other view preference here;
 *  a stored shape this code no longer reads is dropped, never thrown on.
 */

export type DockKind = "note" | "film" | "data";

export interface DockTab {
  id: string;
  kind: DockKind;
  /** What the surface would otherwise read off the URL: m, n, tab, lib… */
  params: Record<string, string>;
}

/** A seek for a film tab, stamped so the same second twice still fires. */
export interface DockSeek {
  tabId: string;
  t: number;
  n: number;
}

interface DockState {
  tabs: DockTab[];
  active: string | null;
  /** Hidden keeps the tabs; the header buttons bring the dock back. */
  open: boolean;
  width: number;
  seek: DockSeek | null;

  /** Open a surface, or focus the tab already showing the same thing. */
  openTab: (kind: DockKind, params?: Record<string, string | null | undefined>) => string;
  close: (id: string) => void;
  focus: (id: string) => void;
  setParams: (id: string, patch: Record<string, string | null>) => void;
  setOpen: (open: boolean) => void;
  setWidth: (px: number) => void;
  seekFilm: (tabId: string, t: number) => void;
}

export const DOCK_KEY = "futtraffic.dock.v1";
export const DOCK_MIN = 320;
export const DOCK_DEFAULT = 520;
const KINDS: readonly DockKind[] = ["note", "film", "data"];

let seq = 0;
const newId = () => `t${Date.now().toString(36)}${(seq++).toString(36)}`;

const clean = (p: Record<string, string | null | undefined> = {}) =>
  Object.fromEntries(Object.entries(p).filter((e): e is [string, string] => typeof e[1] === "string" && e[1] !== ""));

/** The tab that already shows this, if one does: the same note, or a film or
 *  data tab on the same match. A second film of ANOTHER match is a new tab -
 *  that is the point of having tabs. */
export function sameTab(tabs: DockTab[], kind: DockKind, params: Record<string, string>): DockTab | undefined {
  if (kind === "note") {
    // No note chosen yet is the notes library: one of those is enough.
    return tabs.find((t) => t.kind === "note" && (t.params.n ?? "") === (params.n ?? ""));
  }
  return tabs.find((t) => t.kind === kind && (t.params.m ?? "") === (params.m ?? ""));
}

/** Whatever localStorage held, as tabs this code can render, or nothing. */
export function hydrate(raw: string | null): Pick<DockState, "tabs" | "active" | "open" | "width"> {
  const empty = { tabs: [], active: null, open: false, width: DOCK_DEFAULT };
  if (!raw) return empty;
  try {
    const d = JSON.parse(raw) as Partial<DockState>;
    const tabs = (Array.isArray(d.tabs) ? d.tabs : []).filter(
      (t): t is DockTab =>
        !!t &&
        typeof t.id === "string" &&
        KINDS.includes(t.kind) &&
        !!t.params &&
        typeof t.params === "object" &&
        Object.values(t.params).every((v) => typeof v === "string"),
    );
    const active = tabs.some((t) => t.id === d.active) ? (d.active as string) : (tabs[0]?.id ?? null);
    const width = typeof d.width === "number" && d.width >= DOCK_MIN ? d.width : DOCK_DEFAULT;
    return { tabs, active, open: tabs.length > 0 && d.open !== false, width };
  } catch {
    return empty;
  }
}

function read(): string | null {
  try {
    return window.localStorage.getItem(DOCK_KEY);
  } catch {
    return null; // private mode: the dock works, it just forgets on reload
  }
}

export const useDock = create<DockState>()((set, get) => ({
  ...hydrate(typeof window === "undefined" ? null : read()),
  seek: null,

  openTab: (kind, raw) => {
    const params = clean(raw);
    const have = sameTab(get().tabs, kind, params);
    if (have) {
      set({ active: have.id, open: true });
      return have.id;
    }
    const tab: DockTab = { id: newId(), kind, params };
    set((s) => ({ tabs: [...s.tabs, tab], active: tab.id, open: true }));
    return tab.id;
  },

  close: (id) =>
    set((s) => {
      const i = s.tabs.findIndex((t) => t.id === id);
      if (i < 0) return s;
      const tabs = s.tabs.filter((t) => t.id !== id);
      // The neighbour takes focus, the way a browser tab strip does.
      const active = s.active === id ? (tabs[Math.min(i, tabs.length - 1)]?.id ?? null) : s.active;
      return { tabs, active, open: tabs.length > 0 && s.open };
    }),

  focus: (id) => set((s) => (s.tabs.some((t) => t.id === id) ? { active: id, open: true } : s)),

  setParams: (id, patch) =>
    set((s) => {
      const tab = s.tabs.find((t) => t.id === id);
      if (!tab) return s;
      const params = clean({ ...tab.params, ...patch });
      // Choosing, inside a notes tab, a note another tab already has open:
      // go to that tab and drop this one, rather than mount a second editor.
      if (tab.kind === "note" && params.n) {
        const other = s.tabs.find((t) => t.id !== id && t.kind === "note" && t.params.n === params.n);
        if (other) return { tabs: s.tabs.filter((t) => t.id !== id), active: other.id };
      }
      return { tabs: s.tabs.map((t) => (t.id === id ? { ...t, params } : t)) };
    }),

  setOpen: (open) => set((s) => ({ open: open && s.tabs.length > 0 })),
  setWidth: (px) => set({ width: Math.max(DOCK_MIN, Math.round(px)) }),
  seekFilm: (tabId, t) => set((s) => ({ seek: { tabId, t, n: (s.seek?.n ?? 0) + 1 } })),
}));

useDock.subscribe((s) => {
  try {
    window.localStorage.setItem(
      DOCK_KEY,
      JSON.stringify({ tabs: s.tabs, active: s.active, open: s.open, width: s.width }),
    );
  } catch {
    /* not remembered; still works for this session */
  }
});
