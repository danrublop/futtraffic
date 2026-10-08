import { describe, expect, it } from "vitest";

import { buildTree, cleanName, notesUnder, uniquePath } from "./notes";
import type { NoteEntry } from "./types";

const file = (path: string): NoteEntry => ({ path, dir: false, updated: 1, bytes: 1 });
const dir = (path: string): NoteEntry => ({ path, dir: true });

describe("buildTree", () => {
  it("nests, strips .html, and puts folders first", () => {
    const tree = buildTree([
      file("Zebra.html"),
      dir("Scouting"),
      file("Scouting/Rivals.html"),
      file("Scouting/Anderlecht.html"),
    ]);
    expect(tree.map((n) => n.name)).toEqual(["Scouting", "Zebra"]);
    expect(tree[0].children.map((n) => n.name)).toEqual(["Anderlecht", "Rivals"]);
    expect(tree[0].children[0].path).toBe("Scouting/Anderlecht.html");
  });

  it("keeps an empty folder — the whole reason the server lists dirs", () => {
    expect(buildTree([dir("Players")])).toEqual([
      { path: "Players", name: "Players", dir: true, children: [] },
    ]);
  });

  it("invents the parents of a folder created outside the app", () => {
    const tree = buildTree([file("A/B/C.html")]);
    expect(tree[0].name).toBe("A");
    expect(notesUnder(tree).map((n) => n.path)).toEqual(["A/B/C.html"]);
  });
});

describe("uniquePath", () => {
  it("steps aside rather than landing on a note that exists", () => {
    const taken = ["Scouting/Rivals.html", "Scouting/Rivals 2.html"];
    expect(uniquePath(taken, "Scouting", "Rivals")).toBe("Scouting/Rivals 3.html");
    expect(uniquePath(taken, "", "Rivals")).toBe("Rivals.html");
  });

  it("never mints a path the server would refuse", () => {
    expect(uniquePath([], "", "../etc/passwd")).toBe("etc passwd.html");
    expect(uniquePath([], "", "   ")).toBe("Untitled.html");
    expect(cleanName(".hidden")).toBe("hidden");
  });
});
