import type { Node as PMNode } from "@tiptap/pm/model";
import { Plugin, PluginKey, type EditorState, type Transaction } from "@tiptap/pm/state";
import { AttrStep } from "@tiptap/pm/transform";

import { formulaKey } from "./formula-plugin";

/** A locked table refuses to be typed in.
 *
 *  The tables that carry imported numbers — the player table, the comparison,
 *  the timeline — are real tables now rather than read-only blocks, which is
 *  the point: same shape as every other table, same toolbar, same formulas.
 *  But their value is that they say what the match said, and a table you can
 *  silently overwrite by leaning on the keyboard does not say that any more.
 *
 *  So the numbers arrive held still and unlocking is a deliberate act, through
 *  the right-click menu. This is where "held still" is enforced, and it is
 *  `filterTransaction` rather than `contenteditable="false"` on the cells
 *  because the second only stops TYPING: paste, drag, undo into, and every
 *  command on the table toolbar would all still write.
 */
export const lockKey = new PluginKey("ftLock");

/** Set on a transaction that is allowed to write into a locked table anyway —
 *  the menu's own Refresh, and the unlock that precedes any editing. */
export const ALLOW = "ftLockAllow";

/** The locked table containing this position, if any. */
function lockedAt(doc: PMNode, pos: number): boolean {
  if (pos < 0 || pos > doc.content.size) return false;
  const $at = doc.resolve(pos);
  for (let d = $at.depth; d >= 0; d -= 1) {
    const node = $at.node(d);
    if (node.type.spec.tableRole === "table" && node.attrs.locked) return true;
  }
  return false;
}

/** Every position range a transaction writes to, in the document it started
 *  from. Attribute-only steps carry a single position instead of a range. */
function written(tr: Transaction, doc: PMNode): number[] {
  const out: number[] = [];
  let seen = doc;
  for (const step of tr.steps) {
    if (step instanceof AttrStep) {
      out.push(step.pos);
    } else {
      step.getMap().forEach((from, to) => {
        out.push(from, to);
      });
    }
    seen = step.apply(seen).doc ?? seen;
  }
  return out;
}

/** May this transaction be applied?
 *
 *  Exported so the tests can ask directly. `EditorState.filterTransaction` is
 *  not part of the public type, and a test that reaches through the state to
 *  find it is testing ProseMirror's internals rather than this rule.
 */
export function allows(tr: Transaction, doc: PMNode): boolean {
  if (!tr.docChanged) return true;
  // The menu's own writes, and the recalculation pass — a locked table can
  // still hold formulas, and they must keep answering.
  if (tr.getMeta(ALLOW) || tr.getMeta(formulaKey)) return true;
  // An attribute change ON the table is how it gets unlocked, and how the menu
  // re-points it at another subset. Only CONTENT is held still.
  if (tr.steps.every((s) => s instanceof AttrStep)) return true;
  return !written(tr, doc).some((pos) => lockedAt(doc, pos));
}

export function lockPlugin() {
  return new Plugin({
    key: lockKey,
    filterTransaction: (tr, state: EditorState) => allows(tr, state.doc),
  });
}

/** Is the caret in a locked table? The toolbar asks, so it can grey itself
 *  out rather than offering commands that will be silently refused. */
export function inLockedTable(state: EditorState): boolean {
  return lockedAt(state.doc, state.selection.$head.pos);
}

/** The table the caret is in, and whether it is locked — everything the
 *  right-click menu needs to describe itself. */
export function tableUnder(
  state: EditorState,
  pos: number,
): { pos: number; node: PMNode } | null {
  if (pos < 0 || pos > state.doc.content.size) return null;
  const $at = state.doc.resolve(pos);
  for (let d = $at.depth; d > 0; d -= 1) {
    const node = $at.node(d);
    if (node.type.spec.tableRole === "table") return { pos: $at.before(d), node };
  }
  return null;
}
