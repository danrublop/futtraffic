"use client";

import type { Editor } from "@tiptap/react";
import { useEditorState } from "@tiptap/react";
import {
  ArrowDownToLine,
  ArrowLeftToLine,
  ArrowRightToLine,
  ArrowUpToLine,
  Columns3,
  Heading,
  Merge,
  PanelLeft,
  Rows3,
  Split,
  Trash2,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { TABLE_COMMANDS, inTable } from "./table-grid";
import { FormulaField } from "./formula-field";

/** The controls for the table the caret is in.
 *
 *  Appears only when there IS a table under the caret, which is how both Word
 *  and Google Docs do it — the operations are meaningless anywhere else, and a
 *  permanent row of greyed-out table buttons is permanent noise in a document
 *  that mostly has no tables in it.
 *
 *  State comes from `useEditorState`, like the rest of the toolbar: reading
 *  `TABLE_COMMANDS.x(editor.state)` during render instead would be a lie the
 *  moment the caret moves without a re-render, and this bar's whole job is to
 *  say what is possible where the caret IS.
 */

// `addColumnBefore` and `toggleHeaderColumn` were implemented in
// TABLE_COMMANDS and reachable from nowhere: you could add a column to the
// right and never to the left, and a table whose first COLUMN is the labels —
// which is most of them here, the player's name down the side — had no way to
// say so. Both are one line each and were already written.
const BUTTONS = [
  { id: "addRowBefore", label: "Row above", Icon: ArrowUpToLine },
  { id: "addRowAfter", label: "Row below", Icon: ArrowDownToLine },
  { id: "addColumnBefore", label: "Column left", Icon: ArrowLeftToLine },
  { id: "addColumnAfter", label: "Column right", Icon: ArrowRightToLine },
  { id: "deleteRow", label: "Delete row", Icon: Rows3 },
  { id: "deleteColumn", label: "Delete column", Icon: Columns3 },
  { id: "mergeCells", label: "Merge cells — select more than one first", Icon: Merge },
  { id: "splitCell", label: "Split cell", Icon: Split },
  { id: "toggleHeaderRow", label: "Header row", Icon: Heading },
  { id: "toggleHeaderColumn", label: "Header column", Icon: PanelLeft },
  { id: "deleteTable", label: "Delete table", Icon: Trash2 },
] as const;

export function TableToolbar({ editor }: { editor: Editor }) {
  const s = useEditorState({
    editor,
    selector: ({ editor: e }) => ({
      here: inTable(e.state),
      // `dispatch === undefined` is how a ProseMirror command reports whether
      // it WOULD apply, without applying it.
      can: BUTTONS.map((b) => TABLE_COMMANDS[b.id](e.state, undefined)),
    }),
  });

  if (!s?.here || !editor.isEditable) return null;

  return (
    <div className="ft-controls flex shrink-0 items-center gap-0.5 border-l border-[var(--hairline)] pl-1.5">
      <FormulaField editor={editor} />
      {BUTTONS.map((b, i) => (
        <button
          key={b.id}
          type="button"
          title={b.label}
          aria-label={b.label}
          disabled={!s.can[i]}
          // onMouseDown, not onClick: a click moves focus out of the editor
          // first, and every one of these acts on where the caret was.
          onMouseDown={(e) => {
            e.preventDefault();
            editor
              .chain()
              .focus()
              .command(({ state, dispatch }) => TABLE_COMMANDS[b.id](state, dispatch))
              .run();
          }}
          className={cn(
            "grid h-6 w-6 place-items-center rounded-[var(--radius-xs)] transition-colors",
            s.can[i]
              ? "text-[var(--graphite)] hover:bg-[var(--card)] hover:text-[var(--ink)]"
              : "cursor-not-allowed text-[var(--mute)] opacity-35",
          )}
        >
          <b.Icon size={13} strokeWidth={1.7} />
        </button>
      ))}
    </div>
  );
}
