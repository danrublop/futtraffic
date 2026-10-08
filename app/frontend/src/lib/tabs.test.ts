import { describe, expect, it } from "vitest";

import { dockTarget, fromAgentTab, TAB_TYPES } from "./tabs";

/** The reducer this file used to test is gone with the panes. What is left is
 *  the one piece of logic here that another system depends on: the SSE
 *  contract's tab names, which app/server/agent.py still speaks. */

describe("fromAgentTab", () => {
  it("lands every agent tab on a surface that exists", () => {
    // The backend still says match/stats/lineup/squad and is NOT changed when
    // a surface moves — this mapping absorbs it. The match report and the
    // panes are both gone; all four now land on a table in the database.
    expect(fromAgentTab("match")).toEqual({ type: "data", view: "all-matches" });
    expect(fromAgentTab("lineup")).toEqual({ type: "data", view: "all-matches" });
    expect(fromAgentTab("stats")).toEqual({ type: "data", view: "squads" });
    expect(fromAgentTab("squad")).toEqual({ type: "data", view: "our-players" });
    expect(fromAgentTab("film")).toEqual({ type: "film" });
    expect(fromAgentTab("clips")).toEqual({ type: "film" });
    expect(fromAgentTab("docs")).toEqual({ type: "docs" });
  });

  it("only ever names a surface the app has", () => {
    // The guard that matters: a mapping to a type not in TAB_TYPES would send
    // the agent somewhere the router falls back out of, silently.
    for (const t of ["match", "stats", "lineup", "squad", "film", "clips", "data", "docs"] as const) {
      expect(TAB_TYPES).toContain(fromAgentTab(t).type);
    }
  });
});

describe("dockTarget", () => {
  it("a film lands on the named match, or the chat's", () => {
    expect(dockTarget({ action: "open_tab", tab: "film", match_id: "m2" }, "m1")).toEqual({
      kind: "film",
      params: { m: "m2" },
    });
    expect(dockTarget({ action: "seek_film", t: 30 }, "m1")).toEqual({ kind: "film", params: { m: "m1" } });
    expect(dockTarget({ action: "open_tab", tab: "clips" }, null)).toEqual({ kind: "film", params: {} });
  });

  it("the table names keep their table, with the match", () => {
    expect(dockTarget({ action: "open_tab", tab: "stats" }, "m1")).toEqual({
      kind: "data",
      params: { tab: "squads", m: "m1" },
    });
    expect(dockTarget({ action: "open_tab", tab: "data" }, null)).toEqual({ kind: "data", params: {} });
  });

  it("a note is the note, never the chat's match", () => {
    expect(dockTarget({ action: "open_tab", tab: "docs", note: "club/press.html", match_id: "m1" }, "m1")).toEqual({
      kind: "note",
      params: { n: "club/press.html" },
    });
  });
});
