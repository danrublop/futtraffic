import { beforeEach, describe, expect, it } from "vitest";

import { DOCK_DEFAULT, hydrate, useDock } from "./dock";

const dock = () => useDock.getState();

beforeEach(() => {
  useDock.setState({ tabs: [], active: null, open: false, width: DOCK_DEFAULT, seek: null });
});

describe("openTab", () => {
  it("opens and focuses, and the same thing twice is one tab", () => {
    const a = dock().openTab("film", { m: "m1" });
    const b = dock().openTab("data");
    expect(dock().active).toBe(b);
    expect(dock().openTab("film", { m: "m1" })).toBe(a);
    expect(dock().tabs).toHaveLength(2);
    expect(dock().active).toBe(a);
    expect(dock().open).toBe(true);
  });

  it("a film of another match is a new tab", () => {
    dock().openTab("film", { m: "m1" });
    dock().openTab("film", { m: "m2" });
    expect(dock().tabs.map((t) => t.params.m)).toEqual(["m1", "m2"]);
  });

  it("drops empty params rather than storing them", () => {
    const id = dock().openTab("film", { m: null, lib: "" });
    expect(dock().tabs.find((t) => t.id === id)?.params).toEqual({});
  });
});

describe("one tab per note", () => {
  // Two editors on one file: close one, and its unmount flush writes its copy
  // over the other's edits, silently.
  it("opening a note already open focuses it", () => {
    const a = dock().openTab("note", { n: "club/press.html" });
    dock().openTab("data");
    expect(dock().openTab("note", { n: "club/press.html" })).toBe(a);
    expect(dock().tabs.filter((t) => t.kind === "note")).toHaveLength(1);
  });

  it("picking, in a library tab, a note another tab has open goes there and drops the library tab", () => {
    const open = dock().openTab("note", { n: "club/press.html" });
    const lib = dock().openTab("note");
    dock().setParams(lib, { n: "club/press.html" });
    expect(dock().tabs.map((t) => t.id)).toEqual([open]);
    expect(dock().active).toBe(open);
  });

  it("picking a note nobody has open just sets it", () => {
    const lib = dock().openTab("note");
    dock().setParams(lib, { n: "club/shape.html" });
    expect(dock().tabs).toEqual([{ id: lib, kind: "note", params: { n: "club/shape.html" } }]);
  });
});

describe("close", () => {
  it("the neighbour takes focus, and closing the last tab hides the dock", () => {
    const a = dock().openTab("film", { m: "m1" });
    const b = dock().openTab("film", { m: "m2" });
    const c = dock().openTab("film", { m: "m3" });
    dock().focus(b);
    dock().close(b);
    expect(dock().active).toBe(c);
    dock().close(c);
    expect(dock().active).toBe(a);
    dock().close(a);
    expect(dock().active).toBeNull();
    expect(dock().open).toBe(false);
  });
});

describe("hydrate", () => {
  it("restores what it can render and drops the rest", () => {
    const raw = JSON.stringify({
      tabs: [
        { id: "x", kind: "film", params: { m: "m1" } },
        { id: "y", kind: "pane", params: {} },
        { id: "z", kind: "note", params: { n: 5 } },
        null,
      ],
      active: "y",
      open: true,
      width: 10,
    });
    expect(hydrate(raw)).toEqual({
      tabs: [{ id: "x", kind: "film", params: { m: "m1" } }],
      active: "x",
      open: true,
      width: DOCK_DEFAULT,
    });
  });

  it("garbage, or nothing, is an empty dock - never a throw", () => {
    const empty = { tabs: [], active: null, open: false, width: DOCK_DEFAULT };
    expect(hydrate("{not json")).toEqual(empty);
    expect(hydrate(null)).toEqual(empty);
    expect(hydrate('"a string"')).toEqual(empty);
  });
});
