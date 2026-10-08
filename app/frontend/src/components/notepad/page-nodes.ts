import { Node, mergeAttributes } from "@tiptap/core";

/** PAGES. Two nodes, because a document that leaves the app leaves as paper.
 *
 *  The notepad is one continuous surface and stays one: paginating a rich-text
 *  editor for real - reflowing every block as you type until it lands on a
 *  sheet - is a typesetting engine, and nobody asked for one. What a scouting
 *  report actually needs is smaller and exact:
 *
 *   - `pageBreak`  - "the next section starts on a new sheet", placed by hand
 *     where a coach wants it. On screen it is a labelled rule; on paper it is
 *     `break-after: page`, which is the browser's own pagination doing the work.
 *   - `pageSetup`  - which way up the paper is. It has to live IN the document,
 *     not in a setting somewhere, because the note is a file: it gets copied to
 *     another machine, opened in Finder, mailed to a player. A landscape report
 *     that arrives portrait because the preference stayed behind is a report
 *     with its tables cut off.
 *
 *  Landscape is the report format and portrait is a page of prose, so the
 *  document says which it is rather than the app deciding for everybody. A note
 *  with no `pageSetup` in it is LANDSCAPE, which is what every note written
 *  before this already printed as.
 */

export type Orient = "portrait" | "landscape";

/** LANDSCAPE IS THE DEFAULT, and it is not a preference of ours - it is what
 *  the exporter has always printed, because a scouting report's unit is a
 *  COLUMN and portrait gives you one of any useful width or three unreadably
 *  narrow ones. A note with no setup node in it is every note written before
 *  pages existed, and those already printed landscape; portrait is the thing
 *  you now ask for, for a page that is only prose. */
export const orientOf = (value: string | null | undefined): Orient =>
  value === "portrait" ? "portrait" : "landscape";

/** The orientation a saved note asks for, read off the stored HTML.
 *
 *  A string scan rather than a parse: the exporter has the file and not an
 *  editor, and standing up a ProseMirror schema to read one attribute out of
 *  the first tag would be a strange amount of machinery. */
export function docOrient(html: string): Orient {
  return /data-pagesetup="portrait"/.test(html) ? "portrait" : "landscape";
}

/** The default margin for each paper, in mm. These are the numbers that were
 *  hard-coded in doc-export's `page()` before the margin was authorable, so a
 *  document that says nothing prints exactly as it always did. */
export const MARGIN_DEFAULT: Record<Orient, number> = { landscape: 12, portrait: 14 };

/** Clamp to something a printer can actually do. Below ~5 mm most desktop
 *  printers cannot reach the edge and silently crop; above 40 mm on A4 there is
 *  no measure left to put a figure in. */
export function marginOf(v: string | number | null | undefined): number {
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? Math.max(5, Math.min(40, n)) : 0;
}

/** The margin the document asked for, or its paper's default. */
export function docMargin(html: string, orient: Orient = docOrient(html)): number {
  return marginOf(/data-margin="([\d.]+)"/.exec(html)?.[1]) || MARGIN_DEFAULT[orient];
}

declare module "@tiptap/core" {
  interface Commands<ReturnType> {
    pages: {
      /** Start a new sheet after the one the caret is in. */
      insertPage: () => ReturnType;
      /** Set the whole document's paper. Updates the existing setup node or
       *  puts one at the top; there is never more than one. */
      setOrientation: (orient: Orient) => ReturnType;
      insertPageBreak: () => ReturnType;
    };
  }
}

export const PageSetupNode = Node.create({
  name: "pageSetup",
  group: "block",
  atom: true,
  selectable: false,
  draggable: false,

  addAttributes() {
    return {
      orient: {
        default: "landscape" as Orient,
        parseHTML: (el: HTMLElement) => orientOf(el.getAttribute("data-pagesetup")),
        renderHTML: (attrs: Record<string, unknown>) => ({
          "data-pagesetup": orientOf(String(attrs.orient ?? "")),
        }),
      },
      /** THE PAPER MARGIN, IN MILLIMETRES, and 0 means "whatever the
       *  orientation's default is" rather than "no margin at all" - a document
       *  written before this attribute existed has no value for it, and a
       *  borderless page is not what those documents meant.
       *
       *  One number for all four edges. Book typography wants the canon's
       *  2:3:4:6 and an outer margin twice the inner, but that is a rule for a
       *  BOUND, TWO-SIDED object where the gutter is shared between facing
       *  pages. A scouting report is printed one side at a time and read flat,
       *  so the asymmetry buys nothing and costs a number nobody can explain. */
      margin: {
        default: 0,
        parseHTML: (el: HTMLElement) => marginOf(el.getAttribute("data-margin")),
        renderHTML: (attrs: Record<string, unknown>) =>
          Number(attrs.margin) > 0 ? { "data-margin": String(attrs.margin) } : {},
      },
    };
  },

  parseHTML() {
    return [{ tag: "div[data-pagesetup]" }];
  },

  renderHTML({ HTMLAttributes, node }) {
    const orient = orientOf(String(node.attrs.orient ?? ""));
    return [
      "div",
      mergeAttributes(HTMLAttributes, { class: "ft-pagesetup", contenteditable: "false" }),
      orient === "landscape" ? "A4 landscape" : "A4 portrait",
    ];
  },

  addCommands() {
    return {
      setOrientation:
        (orient: Orient) =>
        ({ state, tr, dispatch }) => {
          const first = state.doc.firstChild;
          if (first?.type.name === "pageSetup") {
            if (dispatch) dispatch(tr.setNodeAttribute(0, "orient", orient));
            return true;
          }
          if (dispatch) dispatch(tr.insert(0, this.type.create({ orient })));
          return true;
        },
    };
  },
});


/** A SHEET. Fixed size, and the content is the author's problem.
 *
 *  This exists because the other answer does not work. A flowing document can
 *  only be shown page-by-page if the editor fragments content exactly the way
 *  the printer does, and nothing in a browser exposes that. CSS multicol comes
 *  closest - it is the same fragmentation spec, honours the same break-inside
 *  rules - and on the eighteen-page report it produced 44 columns where Chrome
 *  printed 41. A preview that is 7% wrong is worse than none, because you would
 *  trust it.
 *
 *  So pagination is not COMPUTED here, it is AUTHORED - which is how the report
 *  being copied was made, and how every layout tool works. A page is a box the
 *  size of the printable sheet. What fits in the box prints on the sheet, one
 *  to one, and what does not fit hangs visibly off the bottom edge instead of
 *  silently reflowing somewhere you cannot see. That is the whole trade: you
 *  lose automatic reflow and you gain a document that is what it prints.
 *
 *  A page is an ordinary BLOCK, not a required wrapper: a note with no pages in
 *  it is a flowing document and still works exactly as it did. You reach for
 *  sheets when you are laying something out.
 *
 *  WHAT IS NOT HERE, and was tried. A plugin that measured the sheets and moved
 *  blocks across the fold - real pagination, block by block. It worked: the
 *  eighteen-page report reflowed itself from nineteen sheets to thirty and
 *  settled. It also killed the tab. Every move re-renders the document, the
 *  report is a megabyte of inline SVG and photographs, and a few hundred moves
 *  is not something a renderer survives; resuming the scan instead of
 *  restarting it and dropping from forty moves a frame to six did not change
 *  that. Reflow is worth having and is not worth an editor that dies on the
 *  document it was built for. It comes back when a block can be moved without a
 *  full redraw.
 */
export const PageNode = Node.create({
  name: "page",
  group: "block",
  content: "block+",
  // A sheet is a container you type inside; a selection must not run out of one
  // into the next, or a drag across two pages would delete the boundary.
  isolating: true,
  defining: true,

  parseHTML() {
    return [{ tag: "div[data-page]" }];
  },

  renderHTML({ HTMLAttributes }) {
    return ["div", mergeAttributes(HTMLAttributes, { class: "ft-page", "data-page": "1" }), 0];
  },

  addCommands() {
    return {
      insertPage:
        () =>
        ({ state, chain }) => {
          // After the page the caret is in, when it is in one - otherwise at
          // the end, which is what "new sheet" means in a flowing note.
          const { $from } = state.selection;
          for (let d = $from.depth; d > 0; d--) {
            if ($from.node(d).type.name === "page") {
              return chain()
                .insertContentAt($from.after(d), {
                  type: "page",
                  content: [{ type: "paragraph" }],
                })
                .run();
            }
          }
          return chain()
            .insertContentAt(state.doc.content.size, {
              type: "page",
              content: [{ type: "paragraph" }],
            })
            .run();
        },
    };
  },
});

export const PageBreakNode = Node.create({
  name: "pageBreak",
  group: "block",
  atom: true,
  selectable: true,
  draggable: true,

  parseHTML() {
    return [{ tag: "div[data-pagebreak]" }];
  },

  renderHTML({ HTMLAttributes }) {
    return [
      "div",
      mergeAttributes(HTMLAttributes, {
        class: "ft-pagebreak",
        "data-pagebreak": "1",
        contenteditable: "false",
      }),
    ];
  },

  addCommands() {
    return {
      insertPageBreak:
        () =>
        ({ commands }) =>
          // A paragraph after it, or a break at the end of the document leaves
          // no way to type on the page it just started.
          commands.insertContent([{ type: this.name }, { type: "paragraph" }]),
    };
  },
});
