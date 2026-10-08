import { describe, expect, it } from "vitest";

import type { ChatSummary } from "./agent";
import { sessionTree } from "./chat-folders";

const chat = (id: string, team_id = ""): ChatSummary => ({ id, title: id, updated: "", team_id });
const TEAMS = [
  { team_id: "u9", name: "U9 Boys" },
  { team_id: "u15", name: "U15 Boys" },
  { team_id: "first", name: "First team" },
];

describe("sessionTree", () => {
  it("loose sessions first, then squads in ladder order, empty squads skipped", () => {
    const out = sessionTree([chat("a", "first"), chat("b"), chat("c", "u9"), chat("d", "first")], TEAMS);
    expect(out.loose.map((c) => c.id)).toEqual(["b"]);
    expect(out.folders.map((f) => [f.label, f.chats.map((c) => c.id)])).toEqual([
      ["U9 Boys", ["c"]],
      ["First team", ["a", "d"]],
    ]);
  });

  it("a session whose squad left the club becomes a loose session, not dropped", () => {
    const out = sessionTree([chat("a", "deleted-squad")], TEAMS);
    expect(out).toEqual({ loose: [chat("a", "deleted-squad")], folders: [] });
  });

  it("nothing to list, nothing drawn", () => {
    expect(sessionTree([], TEAMS)).toEqual({ loose: [], folders: [] });
  });
});
