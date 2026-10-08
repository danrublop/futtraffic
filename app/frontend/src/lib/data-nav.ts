/** WHERE YOU ARE INSIDE THE DATA SURFACE, as one string.
 *
 *  The Data page is one database browser: a rail of TABLES, one open at a
 *  time, and each one can have something selected. That is two pieces of
 *  state, and the obvious place to put them is the URL.
 *
 *  IT IS NOT THE URL, DELIBERATELY — and the reason changed. It used to be
 *  that `TabInstance.view` was persisted per pane and two Data panes could be
 *  on screen at once, so a global `?s=` would have made them fight over one
 *  value. The panes are gone. What is left is the simpler half of the same
 *  argument: this is ONE surface\'s place in itself, the Data browser keeps its
 *  own back/forward history over it, and a query parameter would be a second
 *  writer of a value that already has an owner.
 *   - lib/club.ts already argues the general case in writing: the match is in
 *     the URL because it is a thing you send someone, the team is not because
 *     it is a lens, and "one such parameter is a helper, two is a class of
 *     bug". This would have been the second and third.
 *
 *  The cost, stated rather than hidden: there is no shareable link to a player
 *  or a squad. Notes have one via `?n=`. The app is inconsistent about what is
 *  linkable and that is a trade, not an oversight.
 *
 *      "our-players"                  the table, nothing selected
 *      "our-players:tomas-varga"       the table and what is open in it
 *      "squads:northbridge-fc"
 *      "all-matches"
 *
 *  Pure, so the tests drive it directly and the components stay a render.
 */

/** Every table the browser can open. `store` is the raw store browser: one
 *  entry for routing purposes, even though what it opens is an escape hatch
 *  rather than a single table. */
export const DATA_SECTIONS = ["squads", "our-players", "records", "all-matches"] as const;

export type DataSection = (typeof DATA_SECTIONS)[number];

export const DEFAULT_SECTION: DataSection = "our-players";

export interface DataView {
  section: DataSection;
  /** What is open inside it, or null. A player id, a team id, a club name, a
   *  match id — the section decides what its ids mean, this only carries one. */
  id: string | null;
}

/** The names older builds wrote, mapped onto the section that answers them now.
 *
 *  These arrive from layouts saved in localStorage before the sections were
 *  named this way. They are NOT migrated on write — a stored layout is a cache,
 *  and rewriting every one of them to rename a string is a batch job that can
 *  half-run. They are translated on read instead, here, once.
 *
 *  A five-tab name was a PAGE of two to four tables. There is no single table
 *  that means the same thing, so each one resolves to the table a reader most
 *  likely wanted from that page, and lands somewhere real rather than nowhere. */
const LEGACY: Record<string, DataSection> = {
  // The five-tab era. Each tab was a PAGE of several tables; the name resolves
  // to the one table a reader most likely wanted from it.
  club: "squads",
  teams: "squads",
  players: "our-players",
  clubs: "records",
  opponents: "records",
  matches: "all-matches",
  // Tables that became a drill-down rather than a rail entry: an opponent's
  // players and a match's stats are reached by clicking the row they belong to.
  "their-players": "records",
  "feed-players": "records",
  "head-to-head": "records",
  "match-stats": "all-matches",
  // The raw SQL browser, which no longer has an entry at all.
  store: "all-matches",
  database: "all-matches",
  db: "all-matches",
};

const isSection = (s: string): s is DataSection =>
  (DATA_SECTIONS as readonly string[]).includes(s);

/** The separator between the section and what is open in it. A colon, because
 *  no section name contains one and no id we mint does either: roster ids are
 *  accent-folded slugs (roster.py), team ids are slugged names, and match ids
 *  are 32 hex characters. A club NAME can contain almost anything, which is why
 *  `split` below takes only the FIRST colon and hands the rest over whole. */
const SEP = ":";

/** Read a stored view string. Never throws, always lands on a real section.
 *
 *  Anything unrecognised falls back rather than rendering an empty surface: a
 *  layout written by a future build, a section that was renamed, a hand-edited
 *  localStorage entry. The failure mode that matters is a blank pane with no
 *  error, and this is where it gets prevented. */
export function parseView(view: string | null | undefined): DataView {
  const raw = (view ?? "").trim();
  if (!raw) return { section: DEFAULT_SECTION, id: null };

  const cut = raw.indexOf(SEP);
  const head = cut < 0 ? raw : raw.slice(0, cut);
  // Only the first colon splits. The remainder is the id verbatim, so a club
  // called "Northbridge FC: Castilla" survives the round trip.
  const tail = cut < 0 ? "" : raw.slice(cut + 1);

  const section = isSection(head) ? head : (LEGACY[head] ?? DEFAULT_SECTION);
  // A legacy name never carried a selection, and an unknown one cannot be
  // trusted to have meant this section's ids, so both drop the tail.
  const known = isSection(head) || false;
  return { section, id: known && tail ? tail : null };
}

/** The inverse. An empty or whitespace-only id serialises as the bare section
 *  rather than a dangling colon, so `parseView(serialiseView(v))` is stable. */
export function serialiseView(v: DataView): string {
  const id = (v.id ?? "").trim();
  return id ? `${v.section}${SEP}${id}` : v.section;
}

/** Narrow a view to a section that actually has this id.
 *
 *  A stored selection outlives the thing it names: a player leaves the club, a
 *  squad is deleted, a match is removed from the library. The section must open
 *  with nothing selected rather than with a detail pane bound to a ghost. */
export function resolveView(view: DataView, ids: readonly string[]): DataView {
  if (view.id && !ids.includes(view.id)) return { section: view.section, id: null };
  return view;
}

/** Merge one parameter into an existing query string.
 *
 *  WAS A BUG, twice. `router.push(`${pathname}?m=${id}`)` REPLACES the whole
 *  query, so setting the match silently drops `?v=` and bounces the workspace
 *  back to its default destination. It shipped that way in the chat panel and
 *  it is still live in the header's match switcher. Anything that sets a
 *  parameter goes through here.
 */
export function withParam(
  params: URLSearchParams | string,
  key: string,
  value: string | null,
): string {
  const next = new URLSearchParams(typeof params === "string" ? params : params.toString());
  if (value) next.set(key, value);
  else next.delete(key);
  return next.toString();
}
