import { describe, expect, it } from "vitest";

import {
  DATA_SECTIONS,
  DEFAULT_SECTION,
  parseView,
  resolveView,
  serialiseView,
  withParam,
} from "./data-nav";

describe("parseView", () => {
  it("reads a bare section", () => {
    expect(parseView("squads")).toEqual({ section: "squads", id: null });
  });

  it("reads a section and its selection", () => {
    expect(parseView("our-players:tomas-varga")).toEqual({
      section: "our-players",
      id: "tomas-varga",
    });
  });

  it("splits on the FIRST colon only, so a club name survives", () => {
    // Club names come off a feed and can contain anything; ids we mint cannot.
    expect(parseView("records:Northbridge FC: Castilla")).toEqual({
      section: "records",
      id: "Northbridge FC: Castilla",
    });
  });

  it("lands on a real section for anything it does not recognise", () => {
    // A layout written by a future build, a renamed section, a hand-edited
    // localStorage entry. The failure that matters is a blank pane.
    for (const bad of ["", "   ", "nonsense", "SQUADS", "squads ", ":", ":x"]) {
      expect(DATA_SECTIONS).toContain(parseView(bad).section);
    }
    expect(parseView("nonsense").section).toBe(DEFAULT_SECTION);
    expect(parseView(null).section).toBe(DEFAULT_SECTION);
    expect(parseView(undefined).section).toBe(DEFAULT_SECTION);
  });

  it("drops a selection that came with an unrecognised section", () => {
    // The id was minted for some other section's namespace; carrying it over
    // would open a detail pane bound to a ghost.
    expect(parseView("nonsense:tomas-varga")).toEqual({
      section: DEFAULT_SECTION,
      id: null,
    });
  });

  describe("layouts saved by older builds", () => {
    // A five-tab name was a PAGE of two to four tables, so each resolves to the
    // table a reader most likely wanted from it.
    it("sends the club and teams tabs to Squads", () => {
      expect(parseView("club")).toEqual({ section: "squads", id: null });
      expect(parseView("teams")).toEqual({ section: "squads", id: null });
    });

    it("sends the clubs and opponents tabs to Records", () => {
      expect(parseView("clubs")).toEqual({ section: "records", id: null });
      expect(parseView("opponents")).toEqual({ section: "records", id: null });
    });

    it("sends the players tab to Our players", () => {
      expect(parseView("players")).toEqual({ section: "our-players", id: null });
    });

    it("sends the matches tab to All matches", () => {
      expect(parseView("matches")).toEqual({ section: "all-matches", id: null });
    });

    it("lands a table that became a drill-down on the row it belongs to", () => {
      // An opponent's players and a match's stats are reached by clicking the
      // row, so a saved layout naming them opens the table that holds that row.
      expect(parseView("their-players").section).toBe("records");
      expect(parseView("head-to-head").section).toBe("records");
      expect(parseView("match-stats").section).toBe("all-matches");
    });

    it("lands the retired SQL browser somewhere real", () => {
      // The raw store has no rail entry any more. A layout that names it must
      // still open something rather than an empty pane.
      for (const name of ["store", "database", "db"]) {
        expect(DATA_SECTIONS).toContain(parseView(name).section);
      }
    });
  });
});

describe("serialiseView", () => {
  it("round-trips a section and a selection", () => {
    for (const section of DATA_SECTIONS) {
      expect(parseView(serialiseView({ section, id: null }))).toEqual({ section, id: null });
      expect(parseView(serialiseView({ section, id: "abc-123" }))).toEqual({
        section,
        id: "abc-123",
      });
    }
  });

  it("writes a bare section for an empty id, never a dangling colon", () => {
    expect(serialiseView({ section: "squads", id: null })).toBe("squads");
    expect(serialiseView({ section: "squads", id: "" })).toBe("squads");
    expect(serialiseView({ section: "squads", id: "   " })).toBe("squads");
  });
});

describe("resolveView", () => {
  it("keeps a selection that still exists", () => {
    const v = { section: "our-players" as const, id: "tomas-varga" };
    expect(resolveView(v, ["tomas-varga", "sam-okafor"])).toEqual(v);
  });

  it("drops a selection whose subject is gone", () => {
    // A player leaves the club, a squad is deleted, a match is removed. The
    // section opens with nothing selected rather than bound to a ghost.
    expect(resolveView({ section: "our-players", id: "departed" }, ["tomas-varga"])).toEqual({
      section: "our-players",
      id: null,
    });
  });

  it("leaves a view with no selection alone", () => {
    expect(resolveView({ section: "squads", id: null }, [])).toEqual({
      section: "squads",
      id: null,
    });
  });
});

describe("withParam", () => {
  it("CRITICAL keeps every other parameter when it sets one", () => {
    // The bug this exists to stop: `router.push(`${pathname}?m=${id}`)` replaces
    // the whole query, drops ?v=, and bounces the workspace back to its default
    // destination. It shipped that way in the chat panel once already.
    const out = withParam("v=data&n=Notes%2FA.html", "m", "abc123");
    const got = new URLSearchParams(out);
    expect(got.get("v")).toBe("data");
    expect(got.get("n")).toBe("Notes/A.html");
    expect(got.get("m")).toBe("abc123");
  });

  it("replaces rather than appends when the key is already there", () => {
    const got = new URLSearchParams(withParam("v=data&m=old", "m", "new"));
    expect(got.getAll("m")).toEqual(["new"]);
    expect(got.get("v")).toBe("data");
  });

  it("removes the key for a null or empty value, keeping the rest", () => {
    const got = new URLSearchParams(withParam("v=data&m=abc", "m", null));
    expect(got.has("m")).toBe(false);
    expect(got.get("v")).toBe("data");
    expect(new URLSearchParams(withParam("v=data&m=abc", "m", "")).has("m")).toBe(false);
  });

  it("takes a URLSearchParams as well as a string", () => {
    const got = new URLSearchParams(withParam(new URLSearchParams("v=film"), "m", "x"));
    expect(got.get("v")).toBe("film");
    expect(got.get("m")).toBe("x");
  });

  it("encodes a value that needs it", () => {
    const got = new URLSearchParams(withParam("v=data", "id", "Northbridge FC & Co"));
    expect(got.get("id")).toBe("Northbridge FC & Co");
  });
});
