import { describe, expect, it } from "vitest";

import type { QueryResult } from "./query-data";
import { VISUALS, blockedBy, visualsFor } from "./visuals";

/** The gallery's rules, in one place because two pickers drive off them: the
 *  `/visual` panel and the editor that opens on a chart already in a note. The
 *  day those disagree, a chart you can insert is a chart you cannot change to. */

const r = (over: Partial<QueryResult> = {}): QueryResult => ({
  rows: [
    { label: "A", values: [3], text: ["3"] },
    { label: "B", values: [1], text: ["1"] },
    { label: "C", values: [2], text: ["2"] },
  ],
  series: [],
  max: 3,
  measure: { id: "m", label: "M", agg: "sum", dp: 0 },
  dim: { id: "d", label: "D" },
  ordered: false,
  note: "",
  tier: "prov",
  ...over,
});

const find = (label: string) => VISUALS.find((v) => v.label === label)!;

describe("which pictures a source can be drawn as", () => {
  it("keeps the ones with no source of their own, plus its own", () => {
    const ids = visualsFor("tracks").map((v) => v.label);
    expect(ids).toContain("Column");
    expect(ids).toContain("Average positions");
    expect(ids).toContain("Heat map");
    // A shot map is the shot list or nothing.
    expect(ids).not.toContain("Shot map");
    expect(ids).not.toContain("Momentum");
  });

  it("gives the shot list its map and none of the pitch pictures it cannot draw", () => {
    const ids = visualsFor("shots").map((v) => v.label);
    expect(ids).toContain("Shot map");
    expect(ids).not.toContain("Heat map");
  });
});

describe("why a picture is off", () => {
  it("refuses a source it cannot be drawn from at all", () => {
    expect(blockedBy(find("Shot map"), r(), "tracks")).toMatch(/switch Data/i);
    expect(blockedBy(find("Shot map"), r(), "shots")).toBeNull();
  });

  it("needs a second series before anything can be stacked or paired", () => {
    expect(blockedBy(find("Stacked"), r(), "tracks")).toMatch(/two series/i);
    expect(blockedBy(find("Scatter"), r(), "tracks")).toMatch(/two series/i);
    const two = r({ series: ["Home", "Away"], rows: [{ label: "A", values: [1, 2], text: ["1", "2"] }] });
    expect(blockedBy(find("Stacked"), two, "tracks")).toBeNull();
  });

  it("needs an order before it will join the points up", () => {
    expect(blockedBy(find("Line"), r(), "tracks")).toMatch(/order/i);
    expect(blockedBy(find("Line"), r({ ordered: true }), "tracks")).toBeNull();
  });

  it("needs parts of a whole, and no more of them than the palette has hues", () => {
    expect(blockedBy(find("Pie"), r(), "tracks")).toBeNull();
    const seven = r({
      rows: Array.from({ length: 7 }, (_, i) => ({ label: `P${i}`, values: [1], text: ["1"] })),
    });
    expect(blockedBy(find("Pie"), seven, "tracks")).toMatch(/six/i);
    const negative = r({ rows: [...r().rows, { label: "D", values: [-1], text: ["-1"] }] });
    expect(blockedBy(find("Pie"), negative, "tracks")).toMatch(/non-negative/i);
  });

  it("says nothing about a picture with no requirements", () => {
    expect(blockedBy(find("Column"), r(), "tracks")).toBeNull();
    expect(blockedBy(find("Table"), r(), "anything")).toBeNull();
  });
});

describe("the catalogue itself", () => {
  it("gives every entry a group, so the gallery has nowhere to put an orphan", () => {
    expect(VISUALS.every((v) => v.group)).toBe(true);
  });

  it("explains every pitch picture, because none of them is self-evident", () => {
    expect(VISUALS.filter((v) => v.group === "On the pitch").every((v) => v.hint)).toBe(true);
  });

  it("refuses the visuals nothing can feed yet, on every source", () => {
    for (const label of ["Pass network", "Pass map", "Pass sonar"])
      expect(blockedBy(find(label), r(), "tracks")).toMatch(/pass data/i);
    for (const label of ["Freeze frame", "Team shape", "Pitch control"])
      expect(blockedBy(find(label), r(), "tracks")).toMatch(/one-moment/i);
  });
});
