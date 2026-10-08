import type { NoteEntry } from "./types";

/** The note library as a tree, and the small path arithmetic around it.
 *
 *  Pure — no React, no fetch — because the tree is the one piece of this
 *  feature with rules worth testing: a folder whose parent is missing, a name
 *  that collides with a note already there, a rename that has to keep the note
 *  in its folder. The panel does the rendering.
 *
 *  A path is `Folder/Sub/Name.html`, exactly what the server stores. The `.html`
 *  is storage, not identity, and is never shown: `noteName` takes it off and
 *  `notePath` puts it back, so nothing else in the app has to know the library
 *  is a directory of HTML files.
 */

export interface NoteNode {
  /** Full path, the id everything else passes around. */
  path: string;
  /** The last segment, with `.html` off a note. What the tree prints. */
  name: string;
  dir: boolean;
  /** Files only: the listing row, for dates and snippets. */
  entry?: NoteEntry;
  /** Folders only. */
  children: NoteNode[];
}

export const noteName = (path: string): string =>
  (path.split("/").pop() ?? path).replace(/\.html$/, "");

/** The folder a path sits in, "" for the top level. */
export const folderOf = (path: string): string =>
  path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : "";

export const notePath = (folder: string, name: string): string =>
  `${folder ? `${folder}/` : ""}${name}.html`;

/** Every character the server's `_BAD_SEG` refuses, plus a leading dot, taken
 *  out here so a typed name is fixed rather than rejected. Kept deliberately in
 *  step with jobs.py — a name this lets through and the store refuses is a save
 *  that fails after the note is already on screen. */
export const cleanName = (s: string): string =>
  s
    .replace(/[\x00-\x1f/\\:*?"<>|]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^\.+/, "")
    .slice(0, 70)
    .trim();

/** `name`, or `name 2`, `name 3`… until nothing in `taken` has that path.
 *  Two notes in a folder cannot share a filename, and the server refuses a move
 *  onto an existing path rather than replacing it — so the caller has to pick a
 *  free name, not discover the clash from a 400. */
export function uniquePath(taken: Iterable<string>, folder: string, name: string): string {
  const seen = new Set(taken);
  const base = cleanName(name) || "Untitled";
  let candidate = notePath(folder, base);
  for (let i = 2; seen.has(candidate); i++) candidate = notePath(folder, `${base} ${i}`);
  return candidate;
}

/** Nest the flat listing. Folders sort before notes and both sort by name, so
 *  the tree reads the same on every machine whatever order the walk arrived in.
 *
 *  A row whose parent folder is not in the listing still lands in the right
 *  place: the parents are created as we go. That is not defensive padding — the
 *  library is a directory a coach can also open in Finder, and a folder created
 *  there between two listings is exactly this case. */
export function buildTree(entries: NoteEntry[]): NoteNode[] {
  const root: NoteNode = { path: "", name: "", dir: true, children: [] };
  const dirs = new Map<string, NoteNode>([["", root]]);

  const folder = (path: string): NoteNode => {
    const found = dirs.get(path);
    if (found) return found;
    const node: NoteNode = { path, name: noteName(path), dir: true, children: [] };
    dirs.set(path, node);
    folder(folderOf(path)).children.push(node);
    return node;
  };

  // Shallowest first, so a folder is never created twice under two names.
  for (const e of [...entries].sort((a, b) => a.path.localeCompare(b.path))) {
    if (e.dir) folder(e.path);
    else
      folder(folderOf(e.path)).children.push({
        path: e.path,
        name: noteName(e.path),
        dir: false,
        entry: e,
        children: [],
      });
  }

  const sort = (node: NoteNode) => {
    node.children.sort((a, b) =>
      a.dir === b.dir ? a.name.localeCompare(b.name) : a.dir ? -1 : 1,
    );
    node.children.forEach(sort);
  };
  sort(root);
  return root.children;
}

/** Every note under a node, in tree order — what the library page lists for a
 *  selected folder, and what a delete is about to take with it. */
export function notesUnder(nodes: NoteNode[]): NoteNode[] {
  return nodes.flatMap((n) => (n.dir ? notesUnder(n.children) : [n]));
}
