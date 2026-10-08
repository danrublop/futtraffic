import type { ChatSummary } from "./agent";
import type { Team } from "./types";

export interface ChatFolder {
  teamId: string;
  label: string;
  chats: ChatSummary[];
}

export interface SessionTree {
  /** Sessions in no squad: drawn first, as plain rows, with no folder. */
  loose: ChatSummary[];
  /** One folder per squad that has sessions, in the club's ladder order. */
  folders: ChatFolder[];
}

/** The sidebar's sessions — Claude Code's list, with a squad where the folder
 *  was.
 *
 *  A session in no squad is just a session: it sits at the top of the list as
 *  a row, not inside a "No squad" folder, because not picking a squad is the
 *  ordinary case and should not read as a category of its own.
 *
 *  Folders follow the club's LADDER order (youngest first, as the server sorts
 *  it), not alphabetical and not most-recent: the ladder is how every other
 *  screen lists squads, and a sidebar that reshuffled its folders after every
 *  answer would be impossible to find anything in. Sessions keep the server's
 *  newest-first order inside each group.
 *
 *  A session whose squad has since been removed from the club becomes a loose
 *  session rather than vanishing — the server keeps the id, so re-adding the
 *  squad files it again. Empty folders are not drawn.
 */
export function sessionTree(chats: ChatSummary[], teams: Pick<Team, "team_id" | "name">[]): SessionTree {
  const known = new Set(teams.map((t) => t.team_id));
  const loose: ChatSummary[] = [];
  const by = new Map<string, ChatSummary[]>();
  for (const c of chats) {
    if (c.team_id && known.has(c.team_id)) by.set(c.team_id, [...(by.get(c.team_id) ?? []), c]);
    else loose.push(c);
  }
  const folders = teams
    .filter((t) => by.has(t.team_id))
    .map((t) => ({ teamId: t.team_id, label: t.name, chats: by.get(t.team_id)! }));
  return { loose, folders };
}
