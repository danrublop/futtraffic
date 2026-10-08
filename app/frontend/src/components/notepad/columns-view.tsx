"use client";

import { NodeViewContent, NodeViewWrapper, type NodeViewProps } from "@tiptap/react";
import { Columns3, Minus, Plus } from "lucide-react";

/** A row of columns, with the two controls that change its shape.
 *
 *  Add and remove operate on the document, not on an attribute, because the
 *  column count IS the child count — there is no second copy of it to keep in
 *  step. Removing takes the LAST column and only when it is empty, so a click
 *  can never quietly delete a chart somebody put there; the note under the
 *  buttons says so rather than the button silently doing nothing.
 */
export function ColumnsView({ node, editor, getPos }: NodeViewProps) {
  const n = node.childCount;
  const editable = editor.isEditable;
  const last = node.lastChild;
  // "Empty" is one paragraph with no text in it — what a fresh column holds.
  const lastEmpty = !!last && last.childCount === 1 && last.textContent.trim() === "" &&
    last.firstChild?.type.name === "paragraph";

  const add = () => {
    const pos = getPos();
    if (typeof pos !== "number" || n >= 4) return;
    editor
      .chain()
      .focus()
      .insertContentAt(pos + node.nodeSize - 1, {
        type: "column",
        content: [{ type: "paragraph" }],
      })
      .run();
  };

  const remove = () => {
    const pos = getPos();
    if (typeof pos !== "number" || n <= 2 || !last || !lastEmpty) return;
    const to = pos + node.nodeSize - 1;
    editor.chain().focus().deleteRange({ from: to - last.nodeSize, to }).run();
  };

  return (
    <NodeViewWrapper className="ft-cols-wrap">
      <div className="group relative">
        <NodeViewContent className="ft-cols" data-columns={n} />

        {editable && (
          <div className="ft-controls absolute -top-3 right-0 flex items-center gap-0.5 rounded-[var(--radius-pill)] border border-[var(--hairline)] bg-[var(--panel)]/95 px-1.5 py-0.5 opacity-0 shadow-[var(--shadow-md)] transition-opacity group-hover:opacity-100 focus-within:opacity-100">
            <Columns3 size={11} strokeWidth={1.8} className="text-[var(--mute)]" />
            <span className="px-0.5 font-mono text-[10px] text-[var(--mute)]">{n}</span>
            <button
              type="button"
              onClick={remove}
              disabled={n <= 2 || !lastEmpty}
              title={
                n <= 2
                  ? "Two is the fewest — one column is a paragraph"
                  : lastEmpty
                    ? "Remove the last column"
                    : "The last column has something in it. Empty it first."
              }
              className="grid h-5 w-5 place-items-center rounded-[var(--radius-xs)] text-[var(--graphite)] transition-colors hover:bg-[var(--card)] hover:text-[var(--ink)] disabled:cursor-not-allowed disabled:opacity-35 disabled:hover:bg-transparent"
            >
              <Minus size={11} strokeWidth={2} />
            </button>
            <button
              type="button"
              onClick={add}
              disabled={n >= 4}
              title={n >= 4 ? "Four across is as narrow as a figure stays readable" : "Add a column"}
              className="grid h-5 w-5 place-items-center rounded-[var(--radius-xs)] text-[var(--graphite)] transition-colors hover:bg-[var(--card)] hover:text-[var(--ink)] disabled:cursor-not-allowed disabled:opacity-35 disabled:hover:bg-transparent"
            >
              <Plus size={11} strokeWidth={2} />
            </button>
          </div>
        )}
      </div>
    </NodeViewWrapper>
  );
}
