import { Node, mergeAttributes } from "@tiptap/core";

/** The tactics board as a TipTap node. The board itself — its shape, its
 *  magnets and how it draws — is pitch-data.ts; this is only the schema and
 *  the serialization.
 *
 *  Three attributes rather than one `board` blob, because they are edited on
 *  completely different clocks: magnets change on every drag, shapes on every
 *  stroke, the view on a click. Separate attributes mean a pen stroke does not
 *  rewrite the magnet array in the undo stack, and a board saved before shapes
 *  existed still parses — `data-shapes` is simply absent and reads as none. */

import {
  pitchSvg,
  parseMagnets,
  parseShapes,
  parseView,
  parseLines,
  parseGrass,
  parseKits,
  parseOrient,
  DEFAULT_KITS,
  DEFAULT_MAGNETS,
  type Board,
} from "./pitch-data";

export * from "./pitch-data";

export const PitchNode = Node.create({
  name: "pitch",
  group: "block",
  atom: true, // one unit: you draw inside it, you do not type in it
  draggable: true,
  selectable: true,

  addAttributes() {
    return {
      magnets: {
        default: DEFAULT_MAGNETS,
        parseHTML: (el) => parseMagnets((el as HTMLElement).getAttribute("data-pitch")),
        renderHTML: (attrs) => ({ "data-pitch": JSON.stringify(attrs.magnets) }),
      },
      shapes: {
        default: [],
        parseHTML: (el) => parseShapes((el as HTMLElement).getAttribute("data-shapes")),
        renderHTML: (attrs) =>
          (attrs.shapes as unknown[])?.length
            ? { "data-shapes": JSON.stringify(attrs.shapes) }
            : {},
      },
      // The look of the board. Each one is written out only when it differs
      // from the default, so a plain board stays a plain tag in the saved file
      // and a document written before these existed still parses — an absent
      // attribute reads as the default rather than as an empty string.
      view: {
        default: "full",
        parseHTML: (el) => parseView((el as HTMLElement).getAttribute("data-view")),
        renderHTML: (attrs) =>
          attrs.view && attrs.view !== "full" ? { "data-view": String(attrs.view) } : {},
      },
      orient: {
        default: "h",
        parseHTML: (el) => parseOrient((el as HTMLElement).getAttribute("data-orient")),
        renderHTML: (attrs) => (attrs.orient === "v" ? { "data-orient": "v" } : {}),
      },
      lines: {
        default: "standard",
        parseHTML: (el) => parseLines((el as HTMLElement).getAttribute("data-lines")),
        renderHTML: (attrs) =>
          attrs.lines && attrs.lines !== "standard" ? { "data-lines": String(attrs.lines) } : {},
      },
      grass: {
        default: "green",
        parseHTML: (el) => parseGrass((el as HTMLElement).getAttribute("data-grass")),
        renderHTML: (attrs) =>
          attrs.grass && attrs.grass !== "green" ? { "data-grass": String(attrs.grass) } : {},
      },
      stripes: {
        // Off for a new board. READ the other way round, because a board saved
        // before this wrote nothing when stripes were on: absent means on, and
        // "0" means somebody turned them off. Flipping the parse too would
        // silently restripe every board already in a note.
        default: false,
        parseHTML: (el) => (el as HTMLElement).getAttribute("data-stripes") !== "0",
        renderHTML: (attrs) => (attrs.stripes === false ? { "data-stripes": "0" } : {}),
      },
      /** HOW THE TWO SIDES ARE DRESSED. Its own attribute rather than part of
       *  a settings blob for the same reason the magnets are: a kit is changed
       *  once a board and the magnets change on every drag, and one attribute
       *  holding both would rewrite the whole thing into the undo stack per
       *  disc. Written out only when it differs from the default, so a board
       *  nobody restyled stays a plain tag. */
      kits: {
        default: DEFAULT_KITS,
        parseHTML: (el) => parseKits((el as HTMLElement).getAttribute("data-kits")),
        renderHTML: (attrs) =>
          JSON.stringify(attrs.kits) === JSON.stringify(DEFAULT_KITS)
            ? {}
            : { "data-kits": JSON.stringify(attrs.kits) },
      },
      caption: {
        default: "",
        parseHTML: (el) => (el as HTMLElement).getAttribute("data-caption") ?? "",
        renderHTML: (attrs) => (attrs.caption ? { "data-caption": String(attrs.caption) } : {}),
      },
      // WHAT THIS BOARD RESTS ON. Every other block in the notepad carries a
      // note and a tier; this one carried a free-text caption and nothing
      // else — and it is the block that draws average positions, territory
      // shares and the synthetic plays, which are the three most claim-heavy
      // things in the document. A caption is a title, not a provenance.
      note: {
        default: "",
        parseHTML: (el) => (el as HTMLElement).getAttribute("data-note") ?? "",
        renderHTML: (attrs) => (attrs.note ? { "data-note": String(attrs.note) } : {}),
      },
      tier: {
        default: "",
        parseHTML: (el) => (el as HTMLElement).getAttribute("data-tier") ?? "",
        renderHTML: (attrs) => (attrs.tier ? { "data-tier": String(attrs.tier) } : {}),
      },
      /** Share of the column, 25-100. Not pixels — the same rule the chart's
       *  width follows, and for the same reason: a note is read at whatever
       *  width the pane happens to be and printed at another one.
       *
       *  A board had no width at all until now, so the only thing deciding how
       *  big one was drawn was a CSS cap, and the answer was "the same size
       *  everywhere" whether it sat alone on a sheet or in a third of a row.
       *  100 means fill whatever it is given, which is what every existing
       *  board carries by omission. */
      width: {
        default: 100,
        parseHTML: (el) => {
          const n = Number((el as HTMLElement).getAttribute("data-width"));
          return Number.isFinite(n) && n >= 25 && n <= 100 ? n : 100;
        },
        renderHTML: (attrs) =>
          Number(attrs.width) < 100 ? { "data-width": String(attrs.width) } : {},
      },
      /** TEXT BESIDE THE BOARD, or not. Google Docs calls these "Wrap text"
       *  and "Break text"; this is the same pair under the same two names.
       *
       *  "none" is Break text: the figure owns its band of the page and prose
       *  sits above and below it. "left"/"right" float it and the prose runs
       *  down the other side.
       *
       *  THE ANCHOR COMES FREE and that is why this is two lines rather than a
       *  subsystem. Docs tethers a positioned object to a PARAGRAPH and moves
       *  it when the text above changes; a CSS float is a sibling that
       *  precedes the text it wraps, so document order already is the anchor
       *  and "move with text" is simply what happens. Docs' own API cannot
       *  even express the rest of its UI - no page-relative origin, no way to
       *  create a positioned object at all - so there is nothing further worth
       *  copying.
       *
       *  Deliberately NOT offered: behind and in front of text. Both need the
       *  figure taken out of flow inside a sheet, and doc-export gives
       *  `.ft-page` `height: auto` so an over-full sheet spills onto a second
       *  physical page. An absolutely positioned figure on a spilled sheet
       *  lands on the wrong piece of paper and says nothing about it. */
      wrap: {
        default: "none",
        parseHTML: (el) => {
          const w = (el as HTMLElement).getAttribute("data-wrap");
          return w === "left" || w === "right" ? w : "none";
        },
        renderHTML: (attrs) =>
          attrs.wrap === "left" || attrs.wrap === "right"
            ? { "data-wrap": String(attrs.wrap) }
            : {},
      },
    };
  },

  parseHTML() {
    return [{ tag: "div[data-pitch]" }];
  },

  renderHTML({ HTMLAttributes, node }) {
    // A real DOM element, not an array spec: the saved file has to contain the
    // drawn board, and an array spec cannot express raw SVG markup. Building it
    // here means the serializer and the live NodeView both draw from
    // pitchSvg(), so what you export is what you saw.
    const el = document.createElement("div");
    for (const [k, v] of Object.entries(mergeAttributes(HTMLAttributes, { class: "ft-pitch" }))) {
      if (v != null) el.setAttribute(k, String(v));
    }
    // As an inline width, like the chart's: the stored file has to carry it or
    // a board resized in the editor prints at full width.
    const w = Number(node.attrs.width) || 100;
    if (w < 100) el.setAttribute("style", `width:${w}%`);
    el.innerHTML = pitchSvg(node.attrs as unknown as Board);
    const caption = String(node.attrs.caption ?? "");
    if (caption) {
      const c = document.createElement("figcaption");
      c.textContent = caption;
      el.appendChild(c);
    }
    // Rendered into the saved markup, so the printed page carries the caveat
    // and not only the screen.
    const note = String(node.attrs.note ?? "");
    if (note) {
      const n = document.createElement("p");
      n.className = "ft-pitch__note";
      n.textContent = note;
      el.appendChild(n);
    }
    return el;
  },
});
