import { PLAYER_STAT_TABS } from "@/lib/player-cols";
import type { MatchReport, StatPreset } from "@/lib/types";

/** Named sets of stats, so the measure step is a choice rather than a wall.
 *
 *  The feed offers about thirty-four numbers. Shown flat they are a wall of
 *  chips a coach reads left to right looking for "tackles", which is the
 *  opposite of what a picker is for. The report screen already groups the same
 *  numbers — Top stats, Attack, Passes, Defense, Duels, Goalkeeping — and that
 *  grouping is not a second opinion: `FEED_MEASURES` is built FROM
 *  `PLAYER_STAT_COLS`, the same dictionary `PLAYER_STAT_TABS` groups, so a
 *  preset means exactly the same thing in a chart as in a table.
 *
 *  That is the whole reason these are not a new list. A second hand-kept
 *  grouping would drift from the report's the first time somebody added a
 *  column, and then "Defense" would mean two things in one product.
 */
export interface Preset {
  id: string;
  label: string;
  cols: string[];
  /** WHICH SOURCE owns these columns.
   *
   *  Load-bearing. A preset's ids are the feed's, and applying them against
   *  whatever source happened to be selected produced a set of seven columns
   *  the source did not have — the panel said "7 of 6" and every chip stayed
   *  grey. A preset belongs to the source that can answer it, and says so. */
  source: string;
  /** Only offered when the match actually has a keeper in it. */
  gkOnly?: boolean;
  /** Saved by a coach rather than shipped. */
  saved?: boolean;
}

export const BUILT_IN = PLAYER_STAT_TABS.map((t) => ({
  id: t.id,
  label: t.label,
  cols: t.cols,
  gkOnly: t.gkOnly,
}));

/** Does this match have anybody who could answer a goalkeeping question?
 *
 *  Asked of the FEED rather than the roster, because a keeper who did not play
 *  is not in this match's numbers and a Goalkeeping preset that resolves to
 *  eight empty columns is worse than one that is not offered. */
export function hasKeeper(report: MatchReport | null): boolean {
  const rows = report?.meta?.facts?.player_stats;
  if (!Array.isArray(rows)) return false;
  return rows.some((r) => (r as { pos?: string })?.pos === "Goalkeeper");
}

/** The presets on offer for a source, given what this match holds.
 *
 *  A preset is only offered when the source can actually answer it — the
 *  tracks know distance and nothing about tackles — so the overlap is checked
 *  against the source's own measures rather than assumed.
 */
export function presetsFor(
  sources: { id: string; measures: { id: string }[] }[],
  report: MatchReport | null,
  saved: StatPreset[] = [],
): Preset[] {
  const keeper = hasKeeper(report);

  /** The source that can answer most of this set, and how much of it.
   *
   *  Most, not any: `rating` and `minutes` exist in more than one source, so
   *  "first source with an overlap" would hand Defense to the squad list on
   *  the strength of two columns it shares by accident. */
  const best = (cols: string[]) => {
    let winner: { source: string; cols: string[] } | null = null;
    for (const s of sources) {
      const has = new Set(s.measures.map((m) => m.id));
      const kept = cols.filter((c) => has.has(c));
      if (!winner || kept.length > winner.cols.length) winner = { source: s.id, cols: kept };
    }
    return winner;
  };

  const out: Preset[] = [];
  for (const p of BUILT_IN) {
    if (p.gkOnly && !keeper) continue;
    const w = best(p.cols);
    // Two overlapping columns is not a set worth offering as one.
    if (w && w.cols.length > 1) out.push({ ...p, cols: w.cols, source: w.source });
  }
  saved.forEach((p, i) => {
    const w = best(p.cols);
    if (w && w.cols.length > 0) {
      out.push({ id: `saved:${i}:${p.name}`, label: p.name, cols: w.cols, source: w.source, saved: true });
    }
  });
  return out;
}
