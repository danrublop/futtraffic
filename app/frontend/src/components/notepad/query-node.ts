import { Node, mergeAttributes } from "@tiptap/core";

import { tierNote } from "./stat-node";
import { chartSvg } from "./chart-svg";
import { HEAT_COLS, HEAT_ROWS, binHeat, type Extra, type QueryResult, type QueryRow } from "./query-data";

/** A question, and its answer, as a block.
 *
 *  The frozen thing here is the RESULT, not the query: rows, series and the
 *  scale they were drawn against. Same reason every other block freezes —
 *  the exported document is offline, and a chart that redrew itself after a
 *  re-run would be a chart you could not send to anyone. The question rides
 *  along so the view can offer a Refresh, which is a button.
 */

export interface QueryAttrs {
  title: string;
  source: string;
  dim: string;
  measure: string;
  split: string;
  render: string;
  limit: number;
  rows: QueryRow[];
  series: string[];
  max: number;
  note: string;
  tier: string;
  /** Enough of the measure to redraw and re-label without the source list. */
  unit: string;
  dp: number;
  label: string;
  dimLabel: string;
  ordered: boolean;
  /** The raw facts a native visual draws from, frozen with the rows. */
  extra: Extra | null;
  /** One side's rows only, by team name. "" is both. */
  team: string;
  /** The rows this figure compares, in the order they were picked. Empty is
   *  the ordinary top-N. Frozen with the answer like everything else, so a
   *  comparison a coach set up is still that comparison when it is refreshed. */
  keep: string[];
  /** Share of the column, 20-100. Not pixels: a note is read at whatever width
   *  the pane happens to be and printed at another one, so a chart pinned to
   *  620px is a chart that overflows one of them. */
  width: number;
  /** The rows were edited by hand, so Refresh would throw the edits away. The
   *  view says so and stops offering it. */
  custom: boolean;
  /** How the rows were ordered. Empty is the historical default. */
  sort: string;
  /** A key of the stylesheet's chart themes, or "" for the app's own look. */
  theme: string;
  /** The matches this covers, when a named set was picked. Empty is whatever
   *  the scope implies — this match, or the whole season. */
  matchIds: string[];
  /** One colour per series, or per category on a pie. Empty is the validated
   *  palette; anything here overrides it and is frozen with the figure, so a
   *  note that left the building keeps the colours it left with. */
  colors: string[];
}

/** Read the native payload back, narrowed. A hand-edited document must not be
 *  able to put anything into the pitch renderer that a board could not. */
export function parseExtra(value: string | null): Extra | null {
  if (!value) return null;
  try {
    const raw = JSON.parse(value) as Record<string, unknown>;
    const clamp = (n: unknown) => Math.max(0, Math.min(100, Number(n) || 0));
    const home = String(raw.home ?? "Home").slice(0, 40);
    const away = String(raw.away ?? "Away").slice(0, 40);
    if (raw.kind === "shots" && Array.isArray(raw.points)) {
      return {
        kind: "shots",
        home,
        away,
        points: raw.points.slice(0, 200).map((p) => {
          const q = (p ?? {}) as Record<string, unknown>;
          return {
            x: clamp(q.x),
            y: clamp(q.y),
            side: q.side === "away" ? "away" : "home",
            goal: !!q.goal,
            period: q.period === 2 ? 2 : 1,
          };
        }),
      };
    }
    if (raw.kind === "momentum" && Array.isArray(raw.values)) {
      return {
        kind: "momentum",
        home,
        away,
        values: raw.values.slice(0, 200).map((v) => {
          const n = Number(v);
          return Number.isFinite(n) ? Math.max(-1, Math.min(1, n)) : 0;
        }),
        goals: (Array.isArray(raw.goals) ? raw.goals : []).slice(0, 40).map((g) => {
          const q = (g ?? {}) as Record<string, unknown>;
          return { minute: Number(q.minute) || 0, side: q.side === "away" ? "away" : "home" };
        }),
      };
    }
    const clampPct = (v: unknown) => Math.max(0, Math.min(100, Number(v) || 0));
    if (raw.kind === "spots" && Array.isArray(raw.spots)) {
      return {
        kind: "spots",
        home,
        away,
        measured: !!raw.measured,
        spots: raw.spots.slice(0, 60).map((sp) => {
          const q = (sp ?? {}) as Record<string, unknown>;
          return {
            x: clampPct(q.x),
            y: clampPct(q.y),
            side: q.side === "away" ? "away" : q.side === "home" ? "home" : "",
            label: String(q.label ?? "").slice(0, 3),
          };
        }),
      };
    }
    if (raw.kind === "heat") {
      const who = String(raw.who ?? "").slice(0, 60);
      if (Array.isArray(raw.bins)) {
        const cols = Math.max(1, Math.min(64, Number(raw.cols) || HEAT_COLS));
        const rows = Math.max(1, Math.min(64, Number(raw.rows) || HEAT_ROWS));
        const bins = raw.bins.slice(0, cols * rows).map((n) => Math.max(0, Number(n) || 0));
        // A short `bins` is padded rather than refused. A hand-edited document
        // can carry one, and half a heat map drawn is more use than none.
        while (bins.length < cols * rows) bins.push(0);
        const declared = Number(raw.total) || 0;
        const counted = bins.reduce((a, b) => a + b, 0);
        return { kind: "heat", bins, cols, rows, total: declared || counted, who };
      }
      // A note saved before the binning moved to freeze time. Bin it here so
      // it draws identically, and it lands in the new shape on the next save.
      if (Array.isArray(raw.points)) {
        const points = raw.points.slice(0, 20000).map((pt) => {
          const q = (pt ?? {}) as Record<string, unknown>;
          return { x: clampPct(q.x), y: clampPct(q.y) };
        });
        return {
          kind: "heat",
          bins: binHeat(points),
          cols: HEAT_COLS,
          rows: HEAT_ROWS,
          total: points.length,
          who,
        };
      }
      return null;
    }
    if (raw.kind === "goal" && Array.isArray(raw.shots)) {
      // Bounded well OUTSIDE the frame, not clamped to it: a penalty dragged
      // wide belongs beside the post and not on it, and clamping would move
      // every miss onto the woodwork and make the figure say the opposite of
      // what happened.
      const bound = (v: unknown) => Math.max(-70, Math.min(170, Number(v) || 0));
      return {
        kind: "goal",
        who: String(raw.who ?? "").slice(0, 60),
        shots: raw.shots.slice(0, 60).map((sh) => {
          const q = (sh ?? {}) as Record<string, unknown>;
          return {
            x: bound(q.x),
            y: bound(q.y),
            scored: !!q.scored,
            label: String(q.label ?? "").slice(0, 3),
          };
        }),
      };
    }
    if (raw.kind === "zones" && Array.isArray(raw.bands)) {
      return {
        kind: "zones",
        axis: raw.axis === "width" ? "width" : "length",
        who: String(raw.who ?? "").slice(0, 60),
        bands: raw.bands.slice(0, 8).map((b) => {
          const q = (b ?? {}) as Record<string, unknown>;
          return { label: String(q.label ?? "").slice(0, 40), share: Number(q.share) || 0 };
        }),
      };
    }
    if (raw.kind === "pizza" && Array.isArray(raw.slices)) {
      return {
        kind: "pizza",
        who: String(raw.who ?? "").slice(0, 60),
        population: Math.max(0, Number(raw.population) || 0),
        // Twelve wedges is already more than a coach reads; past that the
        // labels collide whatever the box is.
        slices: raw.slices.slice(0, 12).map((s) => {
          const q = (s ?? {}) as Record<string, unknown>;
          return {
            label: String(q.label ?? "").slice(0, 40),
            pct: clamp(q.pct),
            text: String(q.text ?? "").slice(0, 24),
            ...(q.group ? { group: String(q.group).slice(0, 30) } : {}),
          };
        }),
      };
    }
    if (raw.kind === "freeze" && Array.isArray(raw.players)) {
      const squad = Math.max(0, Math.min(60, Number(raw.squad) || 0));
      const players = raw.players.slice(0, 40).map((pl) => {
        const q = (pl ?? {}) as Record<string, unknown>;
        return {
          x: clampPct(q.x),
          y: clampPct(q.y),
          side: q.side === "away" ? "away" : "home",
          label: String(q.label ?? "").slice(0, 3),
        };
      });
      const ball = (raw.ball ?? null) as Record<string, unknown> | null;
      return {
        kind: "freeze",
        t_s: Math.max(0, Number(raw.t_s) || 0),
        players,
        ...(ball ? { ball: { x: clampPct(ball.x), y: clampPct(ball.y) } } : {}),
        // `seen` is recomputed from what actually survived the parse rather
        // than trusted from the file. It is the figure's own honesty line, and
        // a document hand-edited to say 22 while carrying 9 players would
        // print a claim the picture does not support.
        seen: players.length,
        squad: squad || players.length,
        home,
        away,
        synthetic: !!raw.synthetic,
      };
    }
    if (raw.kind === "passes" && Array.isArray(raw.passes)) {
      const capped = raw.passes.slice(0, 400);
      return {
        kind: "passes",
        passes: capped.map((ps) => {
          const q = (ps ?? {}) as Record<string, unknown>;
          return {
            x0: clampPct(q.x0),
            y0: clampPct(q.y0),
            x1: clampPct(q.x1),
            y1: clampPct(q.y1),
            side: q.side === "away" ? "away" : "home",
            from: String(q.from ?? "").slice(0, 4),
            to: String(q.to ?? "").slice(0, 4),
            ok: !!q.ok,
          };
        }),
        home,
        away,
        // Whatever the file said it dropped, plus whatever the cap above just
        // dropped as well — otherwise a note carrying 900 passes would print
        // "0 more not drawn" over a picture missing five hundred of them.
        dropped: Math.max(0, Number(raw.dropped) || 0) + Math.max(0, raw.passes.length - capped.length),
        synthetic: !!raw.synthetic,
      };
    }
  } catch {
    /* a figure that cannot be parsed is an absent one, never an exception */
  }
  return null;
}

const esc = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

const strArray = (value: string | null, cap: number): string[] => {
  if (!value) return [];
  try {
    const raw: unknown = JSON.parse(value);
    return Array.isArray(raw) ? raw.slice(0, cap).map((v) => String(v ?? "").slice(0, 40)) : [];
  } catch {
    return [];
  }
};

/** Rows back out of a stored element, tolerating anything. A chart that cannot
 *  be parsed is an empty one, never an exception that eats the document. */
export function parseQueryRows(value: string | null): QueryRow[] {
  if (!value) return [];
  try {
    const raw: unknown = JSON.parse(value);
    if (!Array.isArray(raw)) return [];
    return raw
      .filter((r): r is Record<string, unknown> => !!r && typeof r === "object")
      .slice(0, 40)
      .map((r) => {
        const values = Array.isArray(r.values)
          ? r.values.slice(0, 2).map((v) => (typeof v === "number" && Number.isFinite(v) ? v : 0))
          : [];
        const text = Array.isArray(r.text)
          ? r.text.slice(0, 2).map((t) => String(t ?? "").slice(0, 24))
          : [];
        return { label: String(r.label ?? "").slice(0, 60), values, text };
      })
      .filter((r) => r.values.length > 0);
  } catch {
    return [];
  }
}

/** The stored attributes back into the shape the renderers take. Rebuilt here
 *  rather than stored whole, so a hand-edited document cannot hand the chart a
 *  measure or dimension object it invented. */
export function storedResult(a: QueryAttrs): QueryResult {
  return {
    rows: a.rows ?? [],
    series: (a.series ?? []).slice(0, 2),
    max: Number.isFinite(a.max) ? a.max : 0,
    measure: { id: a.measure, label: a.label || a.measure, agg: "sum", dp: a.dp ?? 0, unit: a.unit || undefined },
    dim: { id: a.dim, label: a.dimLabel || a.dim, ordered: a.ordered },
    ordered: !!a.ordered,
    colors: a.colors ?? [],
    note: a.note ?? "",
    tier: "prov",
    ...(a.extra ? { extra: a.extra } : {}),
  };
}

export const QueryNode = Node.create({
  name: "query",
  group: "block",
  atom: true,
  draggable: true,
  selectable: true,

  addAttributes() {
    const attr = (name: string, fallback = "") => ({
      default: fallback,
      parseHTML: (el: HTMLElement) => el.getAttribute(`data-${name}`) ?? fallback,
      renderHTML: (attrs: Record<string, unknown>) => ({ [`data-${name}`]: String(attrs[name] ?? "") }),
    });
    const int = (name: string, fallback: number) => ({
      default: fallback,
      parseHTML: (el: HTMLElement) => Number(el.getAttribute(`data-${name}`)) || fallback,
      renderHTML: (attrs: Record<string, unknown>) => ({ [`data-${name}`]: String(attrs[name] ?? fallback) }),
    });
    return {
      title: attr("title"),
      source: attr("source", "tracks"),
      dim: attr("qdim", "player"),
      measure: attr("qmeasure", "distance_km"),
      split: attr("split"),
      render: attr("render", "bar"),
      note: attr("note"),
      tier: attr("tier", "prov"),
      unit: attr("unit"),
      label: attr("label"),
      dimLabel: attr("dimlabel"),
      limit: int("limit", 12),
      dp: int("dp", 0),
      team: attr("team"),
      keep: {
        default: [] as string[],
        parseHTML: (el: HTMLElement) => strArray(el.getAttribute("data-keep"), 40),
        renderHTML: (attrs: Record<string, unknown>) =>
          (attrs.keep as string[])?.length ? { "data-keep": JSON.stringify(attrs.keep) } : {},
      },
      sort: attr("sort"),
      /** How the figure is DRESSED, not what it says. A key of a fixed table in
       *  the stylesheet, so nothing typed into a document reaches a selector -
       *  see globals.css. "" is the app's own look. */
      theme: attr("theme"),
      matchIds: {
        default: [] as string[],
        parseHTML: (el: HTMLElement) => strArray(el.getAttribute("data-matches"), 60),
        renderHTML: (attrs: Record<string, unknown>) =>
          (attrs.matchIds as string[])?.length
            ? { "data-matches": JSON.stringify(attrs.matchIds) }
            : {},
      },
      colors: {
        default: [] as string[],
        // Narrowed to CSS hex on the way in. A document is a file a person can
        // edit, and `fill` takes anything — including a url() pointing at a
        // filter, which is not a colour and not something a note should carry.
        parseHTML: (el: HTMLElement) =>
          strArray(el.getAttribute("data-colors"), 8).filter((c) => /^#[0-9a-f]{6}$/i.test(c)),
        renderHTML: (attrs: Record<string, unknown>) =>
          (attrs.colors as string[])?.length ? { "data-colors": JSON.stringify(attrs.colors) } : {},
      },
      width: {
        default: 100,
        parseHTML: (el: HTMLElement) =>
          Math.max(20, Math.min(100, Number(el.getAttribute("data-width")) || 100)),
        renderHTML: (attrs: Record<string, unknown>) =>
          Number(attrs.width) === 100 ? {} : { "data-width": String(attrs.width) },
      },
      custom: {
        default: false,
        parseHTML: (el: HTMLElement) => el.getAttribute("data-custom") === "1",
        renderHTML: (attrs: Record<string, unknown>) => (attrs.custom ? { "data-custom": "1" } : {}),
      },
      max: {
        default: 0,
        parseHTML: (el: HTMLElement) => Number(el.getAttribute("data-max")) || 0,
        renderHTML: (attrs: Record<string, unknown>) => ({ "data-max": String(attrs.max ?? 0) }),
      },
      ordered: {
        default: false,
        parseHTML: (el: HTMLElement) => el.getAttribute("data-ordered") === "1",
        renderHTML: (attrs: Record<string, unknown>) => (attrs.ordered ? { "data-ordered": "1" } : {}),
      },
      series: {
        default: [] as string[],
        parseHTML: (el: HTMLElement) => strArray(el.getAttribute("data-series"), 2),
        renderHTML: (attrs: Record<string, unknown>) => ({ "data-series": JSON.stringify(attrs.series ?? []) }),
      },
      extra: {
        default: null as Extra | null,
        parseHTML: (el: HTMLElement) => parseExtra(el.getAttribute("data-extra")),
        renderHTML: (attrs: Record<string, unknown>) =>
          attrs.extra ? { "data-extra": JSON.stringify(attrs.extra) } : {},
      },
      rows: {
        default: [] as QueryRow[],
        parseHTML: (el: HTMLElement) => parseQueryRows(el.getAttribute("data-qrows")),
        renderHTML: (attrs: Record<string, unknown>) => ({ "data-qrows": JSON.stringify(attrs.rows ?? []) }),
      },
    };
  },

  parseHTML() {
    return [{ tag: "div[data-qrows]" }];
  },

  renderHTML({ HTMLAttributes, node }) {
    const el = document.createElement("div");
    for (const [k, v] of Object.entries(mergeAttributes(HTMLAttributes, { class: "ft-chart" }))) {
      if (v != null) el.setAttribute(k, String(v));
    }
    const w = Number(node.attrs.width) || 100;
    if (w < 100) el.setAttribute("style", `width:${w}%`);
    const a = node.attrs as unknown as QueryAttrs;
    const r = storedResult(a);
    const tier = tierNote(a.tier);
    el.innerHTML =
      (a.title ? `<h4 class="ft-chart__title">${esc(a.title)}</h4>` : "") +
      (r.rows.length ? chartSvg(r, a.render) : "") +
      (a.note || tier
        ? `<p class="ft-chart__note">${esc([tier, a.note].filter(Boolean).join(" · "))}</p>`
        : "");
    return el;
  },
});
