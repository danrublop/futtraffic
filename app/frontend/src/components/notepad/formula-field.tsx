"use client";

import { useState } from "react";
import type { Editor } from "@tiptap/react";
import { useEditorState } from "@tiptap/react";
import { Sigma } from "lucide-react";

import { cn } from "@/lib/utils";
import { a1, readTable } from "./table-formula";

/** The formula bar, for the cell the caret is in.
 *
 *  A committed formula's CELL shows its value — that is the whole design, and
 *  it is what makes the note printable. Which leaves one problem: having typed
 *  `=SUM(C2:C11)` and watched it turn into 134.5, there is nowhere to see the
 *  formula again, and no way to fix a typo in it short of retyping the lot.
 *
 *  So the toolbar shows it, the way Excel's own bar does, and editing it there
 *  writes the attribute straight back. A cell with no formula shows its
 *  address and takes one — which is also how you write a formula without
 *  disturbing what the cell currently reads.
 */
export function FormulaField({ editor }: { editor: Editor }) {
  const here = useEditorState({
    editor,
    selector: ({ editor: e }) => {
      const $head = e.state.selection.$head;
      for (let d = $head.depth; d > 0; d -= 1) {
        const role = $head.node(d).type.spec.tableRole;
        if (role !== "cell" && role !== "header_cell") continue;
        const pos = $head.before(d);
        for (let t = d; t > 0; t -= 1) {
          if ($head.node(t).type.spec.tableRole !== "table") continue;
          const grid = readTable($head.node(t), $head.before(t));
          const cell = grid.cells.find((c) => c.pos === pos);
          if (cell) return { pos, address: a1(cell.row, cell.col), formula: cell.formula };
        }
      }
      return null;
    },
    // The selector returns a fresh object every run, so without this the field
    // re-renders on every keystroke anywhere in the document and steals the
    // caret back mid-word.
    equalityFn: (a, b) =>
      a?.pos === b?.pos && a?.address === b?.address && a?.formula === b?.formula,
  });

  // The draft remembers WHICH cell it belongs to. Moving to another cell then
  // abandons it by simply not matching, instead of an effect that resets on
  // every change of position — which is a cascading render and, worse, one
  // frame in which the field shows the old cell's half-typed formula.
  const [draft, setDraft] = useState<{ pos: number; text: string } | null>(null);

  if (!here || !editor.isEditable) return null;

  const mine = draft?.pos === here.pos ? draft : null;
  const shown = mine ? mine.text : here.formula ?? "";

  const commit = (value: string) => {
    const next = value.trim();
    editor
      .chain()
      .command(({ tr, dispatch }) => {
        if (dispatch) tr.setNodeAttribute(here.pos, "formula", next.startsWith("=") ? next : null);
        return true;
      })
      .run();
    setDraft(null);
  };

  return (
    <span className="ft-controls flex items-center gap-1 border-l border-[var(--hairline)] pl-1.5">
      <Sigma size={12} strokeWidth={1.8} className="shrink-0 text-[var(--mute)]" />
      <span className="w-7 shrink-0 font-mono text-[10px] text-[var(--mute)]">{here.address}</span>
      <input
        value={shown}
        spellCheck={false}
        placeholder="=SUM(C2:C11)"
        aria-label={`Formula for ${here.address}`}
        onChange={(e) => setDraft({ pos: here.pos, text: e.target.value })}
        onBlur={() => mine && commit(mine.text)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            commit(shown);
            editor.commands.focus();
          }
          if (e.key === "Escape") {
            e.preventDefault();
            setDraft(null);
            editor.commands.focus();
          }
          // The table shortcuts below are the editor's, and a formula being
          // typed here is not in the editor.
          e.stopPropagation();
        }}
        className={cn(
          "h-6 w-44 rounded-[var(--radius-sm)] border bg-transparent px-1.5 font-mono text-[11px] outline-none transition-colors",
          here.formula
            ? "border-[var(--hairline-strong)] text-[var(--ink)]"
            : "border-[var(--hairline)] text-[var(--graphite)] focus:border-[var(--accent)]",
        )}
      />
    </span>
  );
}
