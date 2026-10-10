import { Node, mergeAttributes } from "@tiptap/core";

import { tierNote } from "./stat-node";
import type { FigureRow } from "./figure-data";

/** A figure from the match report, as a block in the notepad.
 *
 *  `/stat` is one number and `/table` is a squad ranked on one metric. This is
 *  everything else a scouting report is made of: the wide player table (pages
 *  16-17), the home-against-away comparisons (6 and 9), and the event timeline
 *  (2). `kind` picks which, `cols` picks the subset within it, and both ride
 *  along so the view can rebuild — one node rather than three, because they
 *  differ only in how their rows were computed.
 *
 *  Head and rows are BAKED IN at insert, for the same two reasons the other
 *  blocks freeze theirs: the exported document is offline and cannot call the
 *  API, and a note whose figures changed after a re-run is one you could not
 *  send to a player. `cols` rides along so the view can offer a Refresh — a
 *  button somebody presses, never a background fetch.
 */

export interface FigureAttrs {
  kind: string;
  title: string;
  cols: string;
  limit: number;
  head: string[];
  rows: FigureRow[];
  note: string;
  tier: string;
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

/** Read rows back out of a stored element, tolerating anything. A figure that
 *  cannot be parsed is an empty one, never an exception that eats the note. */
export function parseFigureRows(value: string | null): FigureRow[] {
  if (!value) return [];
  try {
    const raw: unknown = JSON.parse(value);
    if (!Array.isArray(raw)) return [];
    return raw
      .filter((r): r is Record<string, unknown> => !!r && typeof r === "object")
      .slice(0, 60)
      .map((r) => ({
        name: String(r.name ?? "").slice(0, 60),
        side: r.side === "home" || r.side === "away" ? r.side : "",
        // 80, not 24. The cap was set when every cell was a figure like
        // "13.56 km"; a timeline cell is a sentence, and 24 cut "Milan
        // Westbrook for Joaquin Ellsworthy" to "...for Joaq" in the printed
        // document. Still bounded — a cell is not a paragraph.
        cells: Array.isArray(r.cells)
          ? r.cells.slice(0, 16).map((c) => String(c ?? "").slice(0, 80))
          : [],
        // Clamped, because it is a width: a split outside 0-1 draws a bar past
        // the end of its own cell.
        ...(typeof r.split === "number" && Number.isFinite(r.split)
          ? { split: Math.max(0, Math.min(1, r.split)) }
          : {}),
      }));
  } catch {
    return [];
  }
}

export const FigureNode = Node.create({
  name: "figure",
  group: "block",
  atom: true,
  draggable: true,
  selectable: true,

  addAttributes() {
    const attr = (name: string, fallback = "") => ({
      default: fallback,
      parseHTML: (el: HTMLElement) => el.getAttribute(`data-${name}`) ?? fallback,
      renderHTML: (attrs: Record<string, unknown>) => ({
        [`data-${name}`]: String(attrs[name] ?? ""),
      }),
    });
    return {
      kind: attr("kind", "playerTable"),
      title: attr("title"),
      cols: attr("cols", "physical"),
      note: attr("note"),
      tier: attr("tier", "prov"),
      limit: {
        default: 30,
        parseHTML: (el: HTMLElement) => Number(el.getAttribute("data-limit")) || 30,
        renderHTML: (attrs: Record<string, unknown>) => ({ "data-limit": String(attrs.limit ?? 30) }),
      },
      head: {
        default: [] as string[],
        parseHTML: (el: HTMLElement) => strArray(el.getAttribute("data-head"), 16),
        renderHTML: (attrs: Record<string, unknown>) => ({ "data-head": JSON.stringify(attrs.head ?? []) }),
      },
      rows: {
        default: [] as FigureRow[],
        parseHTML: (el: HTMLElement) => parseFigureRows(el.getAttribute("data-figrows")),
        renderHTML: (attrs: Record<string, unknown>) => ({
          "data-figrows": JSON.stringify(attrs.rows ?? []),
        }),
      },
    };
  },

  parseHTML() {
    return [{ tag: "div[data-figrows]" }];
  },

  renderHTML({ HTMLAttributes, node }) {
    // Like the board and the tables: a real element carrying the rendered
    // figure, so the saved file reads as a document rather than a tagged div.
    const el = document.createElement("div");
    for (const [k, v] of Object.entries(mergeAttributes(HTMLAttributes, { class: "ft-figure" }))) {
      if (v != null) el.setAttribute(k, String(v));
    }
    const a = node.attrs as unknown as FigureAttrs;
    // A comparison row carries a split bar in a trailing cell, so the header
    // needs the matching empty column or every row runs one cell wide.
    const split = (a.rows ?? []).some((r) => r.split != null);
    const head =
      (a.head ?? []).map((h) => `<th>${esc(h)}</th>`).join("") + (split ? "<th></th>" : "");
    const rows = (a.rows ?? [])
      .map(
        (r) =>
          `<tr><th scope="row"><span class="ft-figure__who">` +
          (r.side ? `<span class="ft-figure__dot ft-figure__dot--${esc(r.side)}"></span>` : "") +
          `${esc(r.name)}</span></th>` +
          r.cells.map((c) => `<td>${esc(c)}</td>`).join("") +
          (r.split == null
            ? ""
            : `<td class="ft-figure__splitcell"><span class="ft-figure__split">` +
              `<span style="width:${(r.split * 100).toFixed(1)}%"></span></span></td>`) +
          `</tr>`,
      )
      .join("");
    const tier = tierNote(a.tier);
    el.innerHTML =
      (a.title ? `<h4 class="ft-figure__title">${esc(a.title)}</h4>` : "") +
      (rows
        ? `<div class="ft-figure__scroll"><table class="ft-figure__table">` +
          `<thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>`
        : "") +
      (a.note || tier
        ? `<p class="ft-figure__note">${esc([tier, a.note].filter(Boolean).join(" · "))}</p>`
        : "");
    return el;
  },
});
