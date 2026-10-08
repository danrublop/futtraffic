/** SQL RESULTS -> THE SHAPE A FIGURE WANTS.
 *
 *  `POST /api/db/query` answers in the shape SQLite answers in: a list of
 *  column names and a list of tuples. Every chart and table block in the
 *  notepad wants something else - a label and its numbers - and until now the
 *  only code that produced that shape was `runQuery` in the browser, against
 *  in-memory report objects. A script had no way to get there, which is how
 *  report scripts ended up with thousands of lines of numbers typed in
 *  by hand.
 *
 *  These are the pure half of the bridge. They live under src/ rather than
 *  beside the script that uses them because vitest.config.mts only collects
 *  `src/**\/*.test.ts`, and a transform nobody can test is a transform that
 *  drifts. The half that talks to the network is scripts/query.mts.
 */
import { TIER_RANK, tierOf, type Tier } from "@/lib/report";

/** One row of a figure: what it is called, and its numbers. Matches the row
 *  shape query-node.ts stores, so these go straight into `chart()`. */
export interface ChartRow {
  label: string;
  values: number[];
}

export interface ChartRows {
  rows: ChartRow[];
  /** The value columns' names, which is what a legend prints. */
  series: string[];
  max: number;
}

/** The first column names the row; every column after it is a number.
 *
 *  That is the shape a `select label, ... group by label` returns and the shape
 *  a figure needs, so the convention is the contract rather than an argument.
 *  A column that is not a number becomes 0 rather than NaN: NaN propagates
 *  silently through a scale and draws a mark in the wrong place, and a chart
 *  that is quietly wrong is worse than one that is visibly empty.
 */
export function toChartRows(columns: string[], rows: unknown[][]): ChartRows {
  if (columns.length < 2) {
    throw new Error(
      `a figure needs a label column and at least one number column, got [${columns.join(", ")}]`,
    );
  }
  const series = columns.slice(1);
  const out = rows.map((r) => ({
    label: r[0] === null || r[0] === undefined ? "" : String(r[0]),
    values: r.slice(1).map(num),
  }));
  return { rows: out, series, max: Math.max(0, ...out.flatMap((r) => r.values)) };
}

/** SQLite hands back REAL as a number, but a SUM over no rows is null and a
 *  TEXT column is a string. Neither is a height. */
const num = (v: unknown): number => {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : 0;
};

/** THE HONESTY TIER FOR A METRIC NAME OUT OF THE STORE.
 *
 *  app/server/agent.py will not narrate a number without one, and the same rule has
 *  to hold for a figure: validation measured sprints at 63.8% MAE against the
 *  reference, and a distance whose track was seen for part of the match is a
 *  known under-count. The store keys metrics bare ('distance_km'); report.ts
 *  keys them by their path in match.json ('players.distance_km'), so this is
 *  the join between the two vocabularies.
 *
 *  `tierOf` falls back to `not_evaluated` for anything it does not know,
 *  which is the safe direction - an unknown metric is one nobody has reasoned
 *  about, and that is exactly the one that must not read as measured. So there
 *  is no default to write here. (Written against an older tierOf whose fallback
 *  was provisional; report.ts moved it deliberately and says why.)
 */
export function tierForMetric(metric: string, scope: "players" | "teams" = "players"): Tier {
  return tierOf(`${scope}.${metric}`).tier;
}

/** WHERE ONE VALUE SITS IN A POPULATION, 0-100.
 *
 *  The "weak" convention: the share of the population AT OR BELOW this value.
 *  So 72 reads as "72% of them are at or below him", 100 means nobody is above,
 *  and a player compared against only himself is 100 rather than undefined.
 *
 *  Spelled out because every percentile convention gives a different number for
 *  the same data and none of them is wrong - scipy alone ships four. An
 *  undocumented one is a figure nobody can reproduce, and a scouting report
 *  that cannot be reproduced is an opinion with a chart on it.
 *
 *  Ties count as at-or-below on purpose: three players who all ran exactly 10
 *  km are all in the same place, and splitting them by row order would invent a
 *  ranking out of the order rows came back from SQLite.
 */
export function percentileOf(population: number[], value: number): number {
  const vals = population.filter((v) => Number.isFinite(v));
  if (!vals.length) return 0;
  return (vals.filter((v) => v <= value).length / vals.length) * 100;
}

/** The weakest tier across several metrics, for a figure that mixes them.
 *
 *  Weakest wins for the same reason report.ts's own header rule does: a figure
 *  standing over two numbers cannot claim to be sturdier than the shakier one
 *  underneath it. */
//
// report.ts's TIER_RANK, not a copy of it. This file used to keep its own table
// of three tiers, and when report.ts added `not_evaluated` the copy did not
// follow: the lookup came back undefined, every comparison against undefined is
// false, and so the weakest of [prov, not_evaluated] read `prov` - a figure
// containing an unevaluated number stamped as sturdier than it was, depending
// only on the order its metrics were listed in.
export function weakestTier(tiers: Tier[]): Tier {
  return tiers.length
    ? tiers.reduce((a, b) => (TIER_RANK[b] < TIER_RANK[a] ? b : a))
    // Nothing to rank is nothing evaluated, the same fallback tierOf uses.
    : "not_evaluated";
}
