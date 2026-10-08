import type { Node as PMNode } from "@tiptap/pm/model";
import { Plugin, PluginKey } from "@tiptap/pm/state";

import { eachTable, recalcTable } from "./table-formula";

/** Keeping the numbers right, one transaction behind the typing.
 *
 *  `appendTransaction` is the only honest place for this. A formula's value
 *  depends on cells the user is editing RIGHT NOW, so it cannot be computed
 *  when the block is inserted and left alone the way a query's rows are — but
 *  it also must not be a React effect that writes to the document, because that
 *  is a second writer racing the editor. This runs inside ProseMirror's own
 *  cycle: the user's change and the recalculation it caused land as one undo
 *  step, and there is never a frame where the sum disagrees with the column.
 *
 *  THE LOOP THIS AVOIDS. Every transaction it appends is itself a transaction,
 *  so a pass that always returns something calls itself until the stack ends.
 *  Two things stop it: `recalcTable` returns only cells whose text actually
 *  CHANGES, and the appended transaction is stamped so the next pass knows it
 *  was ours.
 */
export const formulaKey = new PluginKey("ftFormula");

/** Does this cell hold a formula the user typed but has not committed? */
const typed = (text: string) => text.trim().startsWith("=") && text.trim().length > 1;

export function formulaPlugin() {
  return new Plugin({
    key: formulaKey,
    appendTransaction(trs, oldState, newState) {
      const changed = trs.some((t) => t.docChanged);
      const moved = !oldState.selection.eq(newState.selection);
      if (!changed && !moved) return null;
      // Our own recalculation coming back around.
      if (trs.some((t) => t.getMeta(formulaKey))) return null;

      const tr = newState.tr;
      let touched = false;

      /** The ranges the user just edited, in the NEW document's coordinates.
       *
       *  Load-bearing. Without it, "the cell no longer starts with = so drop
       *  its formula" fires on a cell nobody touched — type a word anywhere in
       *  the note and every formula in it is quietly deleted, because a
       *  committed formula's TEXT is its value and never starts with an equals
       *  sign. A formula is only dropped when the cell holding it was itself
       *  written in. */
      const edited: { from: number; to: number }[] = [];
      for (const t of trs) {
        if (!t.docChanged) continue;
        t.mapping.maps.forEach((map, i) => {
          const rest = t.mapping.slice(i + 1);
          map.forEach((_oldStart, _oldEnd, newStart, newEnd) => {
            edited.push({ from: rest.map(newStart, -1), to: rest.map(newEnd, 1) });
          });
        });
      }
      const wasEdited = (from: number, to: number) =>
        edited.some((r) => r.from <= to && r.to >= from);

      /** Where the caret is, as the position of the cell holding it. -1 when
       *  it is not in a table at all. */
      const caretCell = (() => {
        const $head = newState.selection.$head;
        for (let d = $head.depth; d > 0; d -= 1) {
          const role = $head.node(d).type.spec.tableRole;
          if (role === "cell" || role === "header_cell") return $head.before(d);
        }
        return -1;
      })();

      // 1. COMMIT. A cell reading "=SUM(C2:C11)" becomes a formula when the
      //    caret leaves it — which is when Excel commits too. Doing it on every
      //    keystroke would evaluate "=S" and "=SU" and replace what is being
      //    typed with #ERROR! halfway through the word.
      eachTable(newState.doc, (table, start) => {
        table.descendants((node, offset) => {
          const role = node.type.spec.tableRole;
          if (role !== "cell" && role !== "header_cell") return true;
          const pos = start + 1 + offset;
          if (pos === caretCell) return false; // still being typed in
          const text = node.textContent.trim();
          if (typed(text) && node.attrs.formula !== text) {
            tr.setNodeAttribute(pos, "formula", text);
            touched = true;
          } else if (!typed(text) && node.attrs.formula && wasEdited(pos, pos + node.nodeSize)) {
            // Typed over. In Excel that replaces the formula, and a formula
            // that survived being overwritten would immediately overwrite the
            // coach right back.
            tr.setNodeAttribute(pos, "formula", null);
            touched = true;
          }
          return false;
        });
      });

      // 2. RECALCULATE, against the document as step 1 left it.
      const doc = touched ? tr.doc : newState.doc;
      const edits: { pos: number; node: PMNode; text: string }[] = [];
      eachTable(doc, (table, start) => {
        edits.push(...recalcTable(table, start));
      });
      if (!edits.length) return touched ? tr.setMeta(formulaKey, true) : null;

      // Descending, so an earlier cell's replacement cannot shift a later
      // cell's recorded position out from under it.
      for (const e of edits.sort((a, b) => b.pos - a.pos)) {
        const cell = doc.nodeAt(e.pos);
        if (!cell) continue;
        const paragraph = newState.schema.nodes.paragraph.create(
          null,
          e.text ? newState.schema.text(e.text) : null,
        );
        tr.replaceWith(e.pos + 1, e.pos + 1 + cell.content.size, paragraph);
      }
      return tr.setMeta(formulaKey, true);
    },
  });
}
