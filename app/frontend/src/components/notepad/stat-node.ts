import { Node, mergeAttributes } from "@tiptap/core";

/** A number from the match, as a block in the notepad.
 *
 *  The other half of what makes this a football notepad rather than a text box:
 *  you type `/stat`, pick a metric, and the figure lands in the document with
 *  the side it belongs to and — crucially — the confidence tier the pipeline
 *  assigned it.
 *
 *  The VALUE IS BAKED IN at insert time, not fetched when the document opens.
 *  Two reasons, and both are load-bearing:
 *   - the doc is sandboxed to an opaque origin and cannot call the API at all;
 *   - a note that silently changed its numbers after a re-run would be a note
 *     you could not trust, and this document gets sent to players.
 *
 *  So `value` is a snapshot. `field` and `side` are kept alongside it so a
 *  refresh command can be offered later — that is a button, not a background
 *  fetch.
 */

export interface StatAttrs {
  /** The report field this came from, e.g. "distance_km". */
  field: string;
  /** Shown above the number. */
  label: string;
  /** The formatted figure, exactly as the report renders it. */
  value: string;
  /** "home" | "away" | "" when the metric is not per-side. */
  side: string;
  /** The provenance tier: "ok" | "rel" | "prov" | "not_evaluated". Never
   *  invented here — it is whatever stats.py resolved, read through
   *  lib/report.ts's tierOf. */
  tier: string;
}

/** What each tier says in a document someone else will read. The pipeline's
 *  three-letter codes mean nothing to a player. */
export const TIER_NOTE: Record<string, string> = {
  ok: "",
  rel: "compare within this match only",
  prov: "provisional",
  // Missing until 2026-09-11, so the weakest tier printed the SAME empty note
  // as the strongest - and once tiers resolved at serve time most fields were
  // this one, every notepad stat reading as sturdy.
  not_evaluated: "not evaluated",
  withheld: "withheld - not published",
};

/** Read TIER_NOTE by a tier that came off a document attribute.
 *
 *  `Object.hasOwn` and not `TIER_NOTE[tier] ?? ""`, because `[]` walks the
 *  prototype chain: `data-tier="constructor"` returns the Object constructor,
 *  which is truthy, and the escaper downstream then calls `.replace` on a
 *  function and throws — taking the whole document's serialization with it. */
export const tierNote = (tier: string): string =>
  Object.hasOwn(TIER_NOTE, tier) ? TIER_NOTE[tier] : "";

const esc = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

export const StatNode = Node.create({
  name: "stat",
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
    return {
      field: attr("field"),
      label: attr("label"),
      value: attr("value", "—"),
      side: attr("side"),
      tier: attr("tier", "prov"),
    };
  },

  parseHTML() {
    return [{ tag: "div[data-field]" }];
  },

  renderHTML({ HTMLAttributes, node }) {
    // Like the pitch: a real element carrying the rendered figure, so the saved
    // file reads as a document rather than as an empty tagged div.
    const el = document.createElement("div");
    for (const [k, v] of Object.entries(mergeAttributes(HTMLAttributes, { class: "ft-stat" }))) {
      if (v != null) el.setAttribute(k, String(v));
    }
    const a = node.attrs as unknown as StatAttrs;
    const note = tierNote(a.tier);
    // Every interpolated value is escaped — `label` comes from the report and
    // `value` is formatted text, but neither is ours to trust by the time an
    // agent has written a document.
    el.innerHTML =
      `<span class="ft-stat__label">${esc(a.label)}${a.side ? ` · ${esc(a.side)}` : ""}</span>` +
      `<span class="ft-stat__value">${esc(a.value)}</span>` +
      (note ? `<span class="ft-stat__tier">${esc(note)}</span>` : "");
    return el;
  },
});
