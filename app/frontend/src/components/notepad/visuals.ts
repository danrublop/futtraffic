import type { QueryResult } from "./query-data";

/** The chart gallery, in Excel's own families plus the ones only football has.
 *
 *  Shared, because the same list drives the `/visual` panel and the editor that
 *  opens on a chart already in the note. Two copies would drift, and the day
 *  they did, a chart you could insert would be a chart you could not change to.
 *
 *  `needs` is what the visual cannot be drawn without, and it is enforced in
 *  the picker rather than in the renderer: a chip that inserts a chart which
 *  quietly falls back to bars is worse than a chip that says why it is off.
 *
 *    split  — two series, so there is something to stack, pair or plot against
 *    order  — a dimension with an inherent order, so joining the points is a
 *             claim the data supports
 *    parts  — one series of non-negative values that are parts of one whole,
 *             and at most six of them, because the palette has six hues and a
 *             seventh would repeat one
 *
 *  `only` is the other direction: the sources a picture can be drawn from at
 *  all. A shot map is the shot list or nothing.
 *
 *  NOT OFFERED, deliberately: a second value axis. Excel's combo chart has one
 *  and it is the most misread chart there is — with two scales behind one
 *  picture, where the series cross is decided by whoever set the axis ranges.
 *  Combo here draws both against the same maximum.
 */
export interface Visual {
  id: string;
  label: string;
  group: string;
  needs?: "split" | "order" | "parts";
  /** Source ids this can be drawn from. Absent means any. */
  only?: string[];
  hint?: string;
  /** Set when no source can feed this yet. The chip stays visible and says
   *  what is missing, rather than inserting a chart that falls back to bars. */
  missing?: string;
}

const NO_PASSES = "Needs pass data — no source has passes yet.";
const NO_MOMENT = "Needs a one-moment source — none built yet.";

export const VISUALS: Visual[] = [
  { id: "column", label: "Column", group: "Column" },
  { id: "columnStacked", label: "Stacked", group: "Column", needs: "split" },
  { id: "columnPct", label: "100% stacked", group: "Column", needs: "split" },
  { id: "bar", label: "Bar", group: "Bar" },
  { id: "line", label: "Line", group: "Line", needs: "order" },
  { id: "area", label: "Area", group: "Line", needs: "order" },
  { id: "areaStacked", label: "Stacked area", group: "Line", needs: "order" },
  { id: "pie", label: "Pie", group: "Parts", needs: "parts" },
  { id: "doughnut", label: "Doughnut", group: "Parts", needs: "parts" },
  { id: "scatter", label: "Scatter", group: "Two measures", needs: "split" },
  { id: "combo", label: "Column + line", group: "Two measures", needs: "split" },
  {
    id: "pitch",
    label: "Shot map",
    group: "On the pitch",
    only: ["shots"],
    hint: "Every shot, where it was taken.",
  },
  {
    id: "spots",
    label: "Average positions",
    group: "On the pitch",
    only: ["tracks"],
    hint: "One disc per player, at their mean position. Sides swap at half time, so read the spread and the pairings, never the position itself.",
  },
  {
    // Its OWN id, not a second entry sharing "spots". Every lookup in the
    // panel is `VISUALS.find(v => v.id === visual)`, so two entries with one id
    // meant the first always won: clicking Line-up set the id to "spots",
    // which resolved back to Average positions, and the squad source could
    // never be reached at all. Both still draw through spotsSvg.
    id: "lineup",
    label: "Line-up",
    group: "On the pitch",
    only: ["squad"],
    hint: "Where the team sheet puts each shirt. Drawn, not measured.",
  },
  {
    id: "heat",
    label: "Heat map",
    group: "On the pitch",
    only: ["tracks"],
    hint: "Where a side's bodies actually were, binned. Says nothing about the ball.",
  },
  {
    id: "zonepitch",
    label: "Zones",
    group: "On the pitch",
    only: ["territory", "zones"],
    hint: "The shares, drawn where they happened.",
  },
  {
    id: "momentum",
    label: "Momentum",
    group: "On the pitch",
    only: ["momentum"],
    hint: "The swing across the ninety, positive toward home.",
  },
  {
    id: "goal",
    label: "Goal mouth",
    group: "On the pitch",
    only: ["penalties", "goalshots"],
    hint: "Where the shots finished in the frame, seen from behind the ball. Outside the posts is a miss, drawn where it went.",
  },
  {
    id: "pizza",
    label: "Percentile pizza",
    group: "Against a population",
    // No `only`: the ranks are computed by whoever builds the figure, not by a
    // source here. A script assembles one today (scripts/note-api.mts ::
    // pizza); the picker will offer it when a season source can rank a player
    // without one. Registered now so a note a script wrote opens in the editor
    // as the figure it is, rather than falling back to bars.
    hint: "One wedge per metric, its depth the player's rank in the population and the number written inside it. Preferred over a radar, whose joined-up area looks like a quantity and changes with the order of the axes.",
  },
  {
    id: "radar",
    label: "Radar",
    group: "Against a population",
    hint: "The pizza's numbers joined into a polygon. Analyst-facing: the enclosed area reads as a quantity and is not one, since it changes with the order of the axes. Prefer the pizza for a coach.",
  },
  {
    id: "freeze",
    missing: NO_MOMENT,
    label: "Freeze frame",
    group: "One moment",
    hint: "Every visible player at one instant, with how many of the squad that was.",
  },
  {
    id: "hull",
    missing: NO_MOMENT,
    label: "Team shape",
    group: "One moment",
    hint: "The convex hull each side's players make at one instant. Reads as compactness and width at a glance.",
  },
  {
    id: "control",
    missing: NO_MOMENT,
    label: "Pitch control",
    group: "One moment",
    hint: "Which side is nearest, over a grid. A coarse approximation of a Voronoi, and named for what it shows rather than for a tessellation it does not compute.",
  },
  {
    id: "network",
    missing: NO_PASSES,
    label: "Pass network",
    group: "Passing",
    hint: "Nodes at each player's mean passing position, edges by volume. Undirected and completed passes only.",
  },
  {
    id: "progressive",
    missing: NO_PASSES,
    label: "Pass map",
    group: "Passing",
    hint: "Every pass as an arrow, with its outcome. Hollow is lost.",
  },
  {
    id: "sonar",
    missing: NO_PASSES,
    label: "Pass sonar",
    group: "Passing",
    hint: "A rose per player: which way his passes actually went.",
  },
  { id: "table", label: "Table", group: "Other" },
];

export const VISUAL_GROUPS = [...new Set(VISUALS.map((v) => v.group))];

export const WHY: Record<string, string> = {
  split: "Needs two series — set Split, so there is a second value to stack or plot against.",
  order: "Needs a breakdown with an order. Joining unordered categories claims a progression that is not there.",
  parts: "Needs one series of up to six non-negative parts of a whole.",
};

/** Why this visual cannot be drawn from what is selected, or null when it can. */
export function blockedBy(v: Visual, r: QueryResult | null, sourceId: string): string | null {
  if (v.missing) return v.missing;
  if (v.only && !v.only.includes(sourceId)) {
    return `Drawn from ${v.only.length === 1 ? "one source" : "its own sources"} only — switch Data first.`;
  }
  if (!v.needs || !r) return null;
  if (v.needs === "split") return r.series.length > 1 ? null : WHY.split;
  if (v.needs === "order") return r.ordered ? null : WHY.order;
  const vals = r.rows.map((x) => x.values[0] ?? 0);
  const ok =
    r.series.length < 2 &&
    r.rows.length >= 2 &&
    r.rows.length <= 6 &&
    vals.every((x) => x >= 0) &&
    vals.some((x) => x > 0);
  return ok ? null : WHY.parts;
}

/** The visuals a source can be drawn as at all — used to narrow the gallery
 *  rather than show a wall of chips that are off for a reason nobody asked. */
export const visualsFor = (sourceId: string) =>
  VISUALS.filter((v) => !v.only || v.only.includes(sourceId));
