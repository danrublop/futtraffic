import type { AgentTab, UiAction } from "./agent";
import type { DockKind } from "./dock";

/** THE FOUR SURFACES, and the one translation the agent needs.
 *
 *  This file used to hold a whole editor: panes, a split axis, sizes, a
 *  reducer, and a serialiser that persisted the arrangement per destination
 *  and per match. All of it is gone. The workspace shows ONE surface at a
 *  time and the sidebar picks which — so there is no arrangement to reduce,
 *  nothing to persist, and no second place a surface can be.
 *
 *  What survives is the vocabulary: which surfaces exist, and how the agent's
 *  names map onto them.
 */
export const TAB_TYPES = ["chat", "data", "film", "docs"] as const;

export type TabType = (typeof TAB_TYPES)[number];

export interface TabInstance {
  type: TabType;
  /** Which table the data surface opens on ("all-matches", "our-players"…).
   *  Only `data` reads it; see lib/data-nav.ts for the grammar. */
  view?: string;
}

/** The agent speaks its own tab ids (the SSE contract); the app has its own.
 *  match/stats/lineup/squad are all the data surface — they differ only in
 *  which table opens first — and clips lives inside the film surface.
 *
 *  The SSE contract in lib/agent.ts is deliberately NOT changed here: the
 *  backend still says "stats" and "squad", and this is the one place that has
 *  to know what they mean now. That indirection is what let the match report
 *  be removed without touching app/server/agent.py, and it is what lets the panes
 *  go the same way — the agent still asks for "lineup" and still lands
 *  somewhere real.
 */
export function fromAgentTab(tab: AgentTab): TabInstance {
  switch (tab) {
    // A match and its team sheet both live in the All matches table now: pick
    // the row and the panel beside it carries the lineup, the facts and the
    // events.
    case "match":
    case "lineup":
      return { type: "data", view: "all-matches" };
    case "stats":
      return { type: "data", view: "squads" };
    case "squad":
      return { type: "data", view: "our-players" };
    case "data":
      return { type: "data" };
    case "docs":
      return { type: "docs" };
    case "film":
    case "clips":
      return { type: "film" };
  }
}

/** Where an agent action lands when the chat is on screen: a dock tab beside
 *  it, carrying the match or note the action named. The page it would have
 *  navigated to (fromAgentTab) is the same surface, so the two cannot drift -
 *  this only says which dock kind and which parameters. With no match named,
 *  the chat's own match is the one meant. */
export function dockTarget(
  action: UiAction,
  chatMatch: string | null,
): { kind: DockKind; params: Record<string, string> } {
  const m = action.match_id ?? chatMatch ?? undefined;
  const withM = (p: Record<string, string>) => (m ? { ...p, m } : p);
  if (action.action === "seek_film") return { kind: "film", params: withM({}) };
  const tab = fromAgentTab(action.tab);
  if (tab.type === "docs") return { kind: "note", params: action.note ? { n: action.note } : {} };
  if (tab.type === "film") return { kind: "film", params: withM({}) };
  return { kind: "data", params: withM(tab.view ? { tab: tab.view } : {}) };
}
