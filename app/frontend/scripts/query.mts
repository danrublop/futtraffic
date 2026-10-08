/** ASKING THE MATCH STORE FOR NUMBERS, FROM A SCRIPT.
 *
 *  The half of the report loop that was missing. note-api.mts can draw
 *  anything and could look nothing up: `chart()` takes rows the caller already
 *  holds, which is how report scripts ended up with thousands of lines of
 *  figures typed in by hand. The store has been reachable over HTTP the whole time -
 *  `POST /api/db/query`, read-only SQL, app/server/app.py - so this is thin on
 *  purpose.
 *
 *      import { chartFrom, tableFrom, q } from "./query.mts";
 *
 *      const shots = await chartFrom(
 *        "select zone, count(*) from event where match_id = ? and kind = 'shot' group by zone",
 *        [matchId],
 *        { title: "Shots by zone", metric: "shots" },
 *      );
 *
 *  WHY SQL AND NOT THE EDITOR'S PIVOT. runQuery in query-data.ts is 1,303 lines
 *  of typed sources, dimensions and measures, and it runs in the browser
 *  against report objects held in memory. Porting it to the server before
 *  knowing which of its eleven sources a report actually uses would buy the
 *  right end state too early and leave two copies of the football maths to
 *  drift. When three reports want the same query, that is the signal to
 *  promote it - not before.
 *
 *  IMPORT note-api.mts FIRST if you import both: its DOM shim has to exist
 *  before any block module loads. This file imports it, so importing this one
 *  first is also fine.
 */
import { API, chart, fmt, pizza, table, type Block } from "./note-api.mts";
import { percentileOf, toChartRows, tierForMetric, weakestTier } from "@/lib/query-rows";

export interface Result {
  columns: string[];
  rows: unknown[][];
}

/** ONE READ-ONLY STATEMENT AGAINST THE STORE.
 *
 *  Throws on a truncated answer rather than returning it. `/api/db/query` caps
 *  at DB_ROWS (5,000) and says so honestly in the response; a helper that
 *  passed that flag along as an ordinary result would turn a capped query into
 *  a figure that looks complete and is wrong, which is the one failure mode a
 *  document must not have. The caller's fix is a narrower WHERE or a LIMIT,
 *  and the message says so.
 */
export async function q(sql: string, params: unknown[] = []): Promise<Result> {
  let res: Response;
  try {
    res = await fetch(`${API}/api/db/query`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ sql, params }),
    });
  } catch (e) {
    // A report script that cannot reach the API must stop here, not write half
    // a note built on nothing.
    throw new Error(`${API} is not answering — is app/server/app.py running? (${String(e)})`);
  }
  if (!res.ok) throw new Error(`query failed ${res.status}: ${(await res.text()).slice(0, 400)}\n  ${sql}`);

  const out = (await res.json()) as Result & { truncated?: boolean };
  if (out.truncated) {
    throw new Error(
      `the store returned the row cap and there are more — this figure would be a partial answer ` +
        `wearing a complete one's label. Narrow the WHERE or add a LIMIT.\n  ${sql}`,
    );
  }
  return { columns: out.columns, rows: out.rows };
}

/** A CHART, DRAWN FROM THE STORE.
 *
 *  First column labels the row, the rest are its numbers - see toChartRows.
 *  `metric` is the store's name for what is being counted, and it is not
 *  decoration: it is what buys the figure its honesty tier, the same contract
 *  app/server/agent.py enforces on every number it narrates. Pass the metrics the
 *  query actually aggregates and the weakest one governs.
 */
export async function chartFrom(
  sql: string,
  params: unknown[],
  opts: {
    title?: string;
    render?: string;
    /** The store's metric name(s) behind the numbers, for the tier. */
    metric?: string | string[];
    scope?: "players" | "teams";
    theme?: string;
    unit?: string;
    dp?: number;
    colors?: string[];
    width?: number;
    note?: string;
    ordered?: boolean;
  } = {},
): Promise<Block> {
  const { metric, scope, ...rest } = opts;
  const res = await q(sql, params);
  const { rows, series, max } = toChartRows(res.columns, res.rows);
  const metrics = metric === undefined ? [] : [metric].flat();
  return chart({
    ...rest,
    rows,
    series,
    max,
    dimLabel: res.columns[0],
    tier: weakestTier(metrics.map((m) => tierForMetric(m, scope ?? "players"))),
  });
}

/** ONE PLAYER RANKED AGAINST EVERYONE THE STORE HAS, AS A PIZZA.
 *
 *  A percentile is a claim about a POPULATION, so the population is built here
 *  rather than left to the caller: one query per metric returning every
 *  player's season aggregate, ranked in JS where the tie rule is visible
 *  (see percentileOf) instead of buried in a window function.
 *
 *  One query per metric rather than one clever pivot: there are six or eight of
 *  them, they are indexed on `metric`, and a query a football person can read
 *  is worth more here than one round trip saved.
 *
 *  `minMinutes` is not optional decoration. Season aggregates include players
 *  who came on for four minutes, and a per-match average over a single cameo
 *  outranks a regular who played thirty times. Left at 0 the figure is still
 *  honest, just noisier - so the default says so rather than picking a
 *  threshold nobody chose.
 */
export async function pizzaFor(opts: {
  player: string;
  metrics: { metric: string; label: string; group?: string; dp?: number; unit?: string }[];
  title?: string;
  minMinutes?: number;
  width?: number;
}): Promise<Block> {
  const who = await q("select name from player where player_id = ?", [opts.player]);
  if (!who.rows.length) throw new Error(`no player ${opts.player} in the store`);
  const name = String(who.rows[0][0]);
  const floor = opts.minMinutes ?? 0;

  const slices: { label: string; pct: number; text: string; group?: string }[] = [];
  let population = 0;

  for (const m of opts.metrics) {
    // Every player's season figure for this metric, filtered by minutes played.
    const res = await q(
      "select stm.player_id, avg(stm.value) v from season_track_metric stm " +
        "where stm.metric = ? group by stm.player_id " +
        "having coalesce(sum(stm.minutes), 0) >= ?",
      [m.metric, floor],
    );
    const pop = res.rows.map((r) => Number(r[1])).filter((v) => Number.isFinite(v));
    const mine = res.rows.find((r) => String(r[0]) === opts.player);
    if (mine === undefined) {
      // Say which metric, rather than drawing a wedge at zero that reads as
      // "bad at this" when it means "no data for this".
      throw new Error(
        `${name} has no ${m.metric} in the store (or is under ${floor} minutes) — ` +
          `drop the metric or lower minMinutes`,
      );
    }
    const value = Number(mine[1]);
    population = Math.max(population, pop.length);
    slices.push({
      label: m.label,
      pct: percentileOf(pop, value),
      text: fmt(value, { dp: m.dp ?? 1, unit: m.unit }),
      ...(m.group ? { group: m.group } : {}),
    });
  }

  return pizza({
    who: name,
    population,
    slices,
    title: opts.title,
    width: opts.width,
    // Ranking does not launder a provisional number: a percentile of a
    // distance that is a known under-count is a percentile of an under-count.
    tier: weakestTier(opts.metrics.map((m) => tierForMetric(m.metric))),
  });
}

// -------------------------------------------------------------- team shape
// THE ONE FIGURE FAMILY A FEED CANNOT SELL YOU.
//
// Everything else in a scouting report can be bought: shots, xG, passes,
// possession. How wide a side played, how deep, how compact, and where its
// rear line sat MINUTE BY MINUTE comes out of positions, and positions are
// what this pipeline makes. stats.py computes it into shape_sample and nothing
// has ever drawn it - it is not even a source in the notepad.
//
// It needs no new renderer. Shape over time is a line chart, and W3 already
// draws those; what it needs is the honesty the raw table cannot express.

/** The columns shape_sample carries. Whitelisted rather than interpolated from
 *  the caller's string: this name goes into SQL, and a union type is a
 *  compile-time promise that a .mts script run through vite-node does not
 *  keep. */
const SHAPE_METRICS = {
  width_m: "Width",
  depth_m: "Depth",
  hull_area_m2: "Area covered",
  rear_line_m: "Defensive line",
  stretch_index_m: "Stretch",
} as const;

export type ShapeMetric = keyof typeof SHAPE_METRICS;

/** HOW A SIDE'S SHAPE MOVED THROUGH THE MATCH, BOTH TEAMS ON ONE AXIS.
 *
 *  `players` is the whole reason this is a function and not a query you write
 *  inline. schema.sql says it plainly beside the column: "A width taken off 7
 *  players is not a team's width." shape_sample carries `n_dedup` - how many
 *  bodies the sample was actually computed from - and on real footage it runs
 *  from 7 to 16 in the same half: 7 because the camera was following the ball
 *  and half the side was out of frame, 16 because a track fragmented and one
 *  player was counted twice. Averaging those together with the honest samples
 *  produces a smooth confident line that is partly measuring how many players
 *  happened to be visible.
 *
 *  So samples outside the band are dropped, and HOW MANY were dropped comes
 *  back on the figure. A filter that hides its own selectivity is the same
 *  failure as no filter - worse, because the line looks cleaner.
 */
export async function shapeSeries(opts: {
  match: string;
  metric: ShapeMetric;
  title?: string;
  /** Minutes per point. Five keeps ~18 points across a half, which reads. */
  bucket?: number;
  /** Acceptable `n_dedup`. Eleven a side, so nine to twelve allows one missing
   *  and one duplicated without letting a seven-player "width" in. */
  players?: [number, number];
  width?: number;
}): Promise<Block> {
  const col = SHAPE_METRICS[opts.metric];
  if (!col) throw new Error(`unknown shape metric ${opts.metric}`);
  const metric = opts.metric; // whitelisted above, safe to interpolate
  const bucket = opts.bucket ?? 5;
  const [lo, hi] = opts.players ?? [9, 12];

  // What the band costs, measured rather than assumed.
  const seen = await q(
    "select count(*) total, sum(n_dedup between ? and ?) kept from shape_sample where match_id = ?",
    [lo, hi, opts.match],
  );
  const total = Number(seen.rows[0]?.[0] ?? 0);
  const kept = Number(seen.rows[0]?.[1] ?? 0);
  if (!total) throw new Error(`no shape samples for match ${opts.match}`);
  if (!kept) {
    throw new Error(
      `every one of the ${total} shape samples for ${opts.match} was computed from ` +
        `outside ${lo}-${hi} players — widen \`players\` or do not draw this`,
    );
  }

  const fig = await chartFrom(
    `select cast(s.t_s / 60 / ? as int) * ? as minute,
            round(avg(case when s.side = 'home' then s.${metric} end), 1) as Home,
            round(avg(case when s.side = 'away' then s.${metric} end), 1) as Away
     from shape_sample s
     where s.match_id = ? and s.n_dedup between ? and ?
     group by minute order by minute`,
    [bucket, bucket, opts.match, lo, hi],
    {
      title: opts.title ?? `${col} through the match`,
      render: "line",
      ordered: true,
      dp: 1,
      unit: opts.metric === "hull_area_m2" ? "m²" : "m",
      width: opts.width,
      // No `metric`: tierForMetric would not know these names and would fall
      // back to not_evaluated. Shape IS evaluated, and provisional, because the
      // formation fit behind it is unstable - see report.ts, SHAPE_WHY - so the
      // tier is set on purpose here rather than left to a fallback that would
      // now under-state it.
    },
  );

  // The caveat rides ON the figure, not in the prose around it, because the
  // figure is what gets screenshotted out of the document.
  (fig.attrs as Record<string, unknown>).tier = "prov";
  (fig.attrs as Record<string, unknown>).note =
    `${kept} of ${total} samples, those taken from ${lo}-${hi} players. ` +
    `Minute buckets of ${bucket}.`;
  return fig;
}

/** A TABLE, DRAWN FROM THE STORE. Held still, like every imported figure. */
export async function tableFrom(
  sql: string,
  params: unknown[],
  opts: { head?: string[]; variant?: string; tones?: string[][] } = {},
): Promise<Block> {
  const res = await q(sql, params);
  return table({
    head: opts.head ?? res.columns,
    rows: res.rows.map((r) => r.map((c) => (c === null || c === undefined ? "" : String(c)))),
    variant: opts.variant,
    tones: opts.tones,
  });
}

// ---------------------------------------------------------------- selfcheck
// `SELFCHECK=1 npx vite-node --config vitest.config.mts scripts/query.mts`
//
// Needs app/server/app.py up, because the whole point of this file is the network
// half — the pure transforms are covered by src/lib/query-rows.test.ts in the
// ordinary suite. Opt-in for the same reason note-api.mts's is: vite-node does
// not put the script it runs into process.argv.
if (process.env.SELFCHECK) {
  const one = await q("select 1 as n");
  if (one.rows[0][0] !== 1) throw new Error("the store cannot count to one");
  console.log("ok — the store answers");

  // A capped answer must not come back as an ordinary one. 333k position rows
  // against a 5,000 cap, so this is the real path and not a mocked flag.
  let capped = false;
  try {
    await q("select match_id from position");
  } catch (e) {
    capped = /partial answer/.test(String(e));
  }
  if (!capped) throw new Error("a truncated result came back as a complete one");
  console.log("ok — a truncated result throws instead of becoming a figure");

  let bad = false;
  try {
    await q("select * from no_such_table");
  } catch (e) {
    bad = /no_such_table/.test(String(e));
  }
  if (!bad) throw new Error("a broken query did not surface the store's own message");
  console.log("ok — a broken query says what the store said");

  // The tier has to survive the trip, because a distance is a known
  // under-count and a figure that does not say so is the failure this
  // codebase cares most about.
  const c = await chartFrom(
    "select p.name, round(m.value, 2) from player_track_metric m " +
      "join player p using (player_id) where m.metric = ? order by 2 desc limit 5",
    ["distance_km"],
    { title: "Furthest", metric: "distance_km", unit: "km", dp: 2 },
  );
  const attrs = c.attrs as Record<string, unknown>;
  if (attrs.tier !== "prov") throw new Error(`distance must ride as provisional, got ${attrs.tier}`);
  const rows = attrs.rows as { label: string; values: number[] }[];
  if (rows.length !== 5) throw new Error(`expected 5 rows, got ${rows.length}`);
  if (!rows[0].label) throw new Error("the first column must label the row");
  console.log(`ok — ${rows.length} rows, tier ${attrs.tier}, top ${rows[0].label} ${rows[0].values[0]}`);

  const t = await tableFrom("select p.name from player p limit 3", [], { variant: "report" });
  if (!(t.content as unknown[])?.length) throw new Error("an empty table came back");
  console.log("ok — a table too");

  // Team shape: the figure family a feed cannot sell. Driven against whichever
  // match actually has samples, so this does not rot when the demo data moves.
  const some = await q(
    "select match_id from shape_sample group by match_id order by count(*) desc limit 1",
  );
  if (some.rows.length) {
    const match = String(some.rows[0][0]);
    const shape = await shapeSeries({ match, metric: "width_m" });
    const at = shape.attrs as Record<string, unknown>;
    if (at.tier !== "prov") throw new Error("shape must ride as provisional");
    // The selectivity has to be ON the figure. A filter that hides how much it
    // dropped is worse than no filter, because the line looks cleaner.
    if (!/\d+ of \d+ samples/.test(String(at.note))) {
      throw new Error(`the figure does not say what the band cost: ${at.note}`);
    }
    const rows = at.rows as { label: string; values: number[] }[];
    if (rows.length < 2) throw new Error("a series needs more than one point");
    console.log(`ok — shape over time, ${rows.length} points, ${at.note}`);

    // A band nothing can satisfy must refuse rather than draw an empty line.
    let refused = false;
    try {
      await shapeSeries({ match, metric: "width_m", players: [99, 100] });
    } catch (e) {
      refused = /widen `players`/.test(String(e));
    }
    if (!refused) throw new Error("an impossible player band still drew a figure");
    console.log("ok — a band nothing satisfies refuses instead of drawing nothing");
  } else {
    console.log("-- no shape samples in this store, shape checks skipped");
  }
}
