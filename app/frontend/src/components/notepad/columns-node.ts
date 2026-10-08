import { Node, mergeAttributes } from "@tiptap/core";

/** Side-by-side blocks, because a scouting report is not one column.
 *
 *  Every page of the InStat report puts two or three things across a row — a
 *  shot map beside the shots table, three channel charts in a line — and until
 *  now the notepad could only stack. That is not a styling gap: a chart and the
 *  table it summarises READ as one figure when they are level with each other
 *  and as two unrelated blocks when one is under the other.
 *
 *  Deliberately equal widths, laid out with flex rather than a grid template.
 *  A grid needs the column count in the CSS, which means a rule per count and a
 *  count attribute that can disagree with how many children the node actually
 *  has. `flex: 1 1 0` needs neither: the columns are equal because there are
 *  N of them, whatever N is, and adding one is a document edit rather than an
 *  attribute change. Uneven widths are the thing this deliberately does not do
 *  yet — nothing in the report needs them.
 *
 *  `isolating` on the column is what keeps Backspace at the top of a column
 *  from eating the column beside it.
 */

export const ColumnNode = Node.create({
  name: "column",
  content: "block+",
  isolating: true,
  selectable: false,

  parseHTML() {
    return [{ tag: "div[data-column]" }];
  },

  renderHTML({ HTMLAttributes }) {
    return ["div", mergeAttributes(HTMLAttributes, { "data-column": "", class: "ft-col" }), 0];
  },
});

export const ColumnsNode = Node.create({
  name: "columns",
  group: "block",
  // Two to four. One column is a paragraph with extra steps; past four
  // nothing on a pitch-shaped figure is legible on A4.
  content: "column{2,4}",
  draggable: true,

  parseHTML() {
    return [{ tag: "div[data-columns]" }];
  },

  renderHTML({ HTMLAttributes, node }) {
    return [
      "div",
      mergeAttributes(HTMLAttributes, {
        // Not read by the CSS — carried so the export and anything reading the
        // saved file can see the intended width without counting children.
        "data-columns": String(node.childCount),
        class: "ft-cols",
      }),
      0,
    ];
  },
});

/** The insert payload for an N-column row of empty paragraphs. */
export const columnsContent = (n: number) => ({
  type: "columns",
  content: Array.from({ length: Math.max(2, Math.min(4, n)) }, () => ({
    type: "column",
    content: [{ type: "paragraph" }],
  })),
});

/** An N-column row already holding blocks. Anything short of N columns is
 *  padded with an empty paragraph, because `column{2,4}` will reject a row
 *  with one child and the whole insert fails silently when it does. */
export const columnsOf = (blocks: Record<string, unknown>[]) => ({
  type: "columns",
  content: Array.from({ length: Math.max(2, Math.min(4, blocks.length)) }, (_, i) => ({
    type: "column",
    content: [blocks[i] ?? { type: "paragraph" }],
  })),
});
