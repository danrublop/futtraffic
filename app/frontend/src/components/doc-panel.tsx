"use client";

import {
  ChevronLeft,
  FilePlus2,
  Plus,
  Folder,
  FolderPlus,
  Files,
  Images,
  MoreVertical,
  Pencil,
  Printer,
  Trash2,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useSurfaceParams } from "@/components/surface-params";
import { Notepad } from "@/components/notepad/notepad";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useReport } from "@/hooks/use-matches";
import { api, ApiError } from "@/lib/api";
import { documentHtml, documentRules, previewHtml } from "@/lib/doc-export";
import { starterDoc } from "@/lib/doc-template";
import {
  buildTree,
  cleanName,
  folderOf,
  noteName,
  notesUnder,
  uniquePath,
  type NoteNode,
} from "@/lib/notes";
import type { NoteEntry } from "@/lib/types";
import { park, unpark } from "@/lib/unsaved";
import { cn } from "@/lib/utils";

/** THE NOTES SURFACE: a gallery of the club's notes, and the editor.
 *
 *  WHAT CHANGED AND WHY. A note used to be one document per match, filed beside
 *  the match's video and its report, plus one scratch note in localStorage for
 *  when no match was selected. That put the note in the wrong drawer. The
 *  automated report is the artifact that belongs to one match; a NOTE is the
 *  thing a coach writes across several — an opponent seen three times, a player
 *  who climbed two age groups, a season review over every team on the ladder.
 *  None of those had a home, and the irony was that the notepad's own stat
 *  blocks could already query the whole library while the document holding them
 *  was pinned to a single job.
 *
 *  So notes are now ONE library for the club, organised the way documents are
 *  organised: folders, and a gallery of what is in the one you picked. No match
 *  id, no team id. The path is the identity — a rename is a move, and the
 *  server (jobs.py) stores the tree as actual directories, so it survives being
 *  copied to another machine.
 *
 *  The library is a GALLERY, not a file tree: chips across the top for the
 *  folders, cards below showing the top of each note as it actually looks, and
 *  the editor opening over the whole pane with a way back. Two screens, one
 *  going into the other, rather than a rail that is on screen whichever of them
 *  you are reading.
 *
 *  This file is the SHELL — listing, gallery, load/autosave. The document
 *  itself is components/notepad/notepad.tsx.
 *
 *  ponytail: moving is still native HTML5 drag and drop — a note onto a chip —
 *  and renaming is still an inline input, one code path for a folder and a
 *  note, rather than a modal per verb.
 */

/** Where the pre-library scratch note lived. Read once, on the first listing
 *  that comes back without it, and then dropped: the note goes into the library
 *  as `Scratch`, which is a real file on the server rather than one browser's
 *  localStorage. Delete this constant once no install can still be carrying
 *  one. */
const SCRATCH_KEY = "futtraffic:doc:scratch";

/** Mint an empty note and return its path. Exported because the sidebar's
 *  Notes "+" creates one too, and a button labelled New note that only
 *  navigates somewhere is a button that lies. */
export async function createNote(folder = ""): Promise<string> {
  const list = await api.notes();
  const path = uniquePath(list.map((e) => e.path), folder, "Untitled");
  // Seeded and saved before it is shown, so the row in the tree is a note on
  // disk — one that only becomes real after you type in it is a row that
  // vanishes when you click away.
  await api.putNote(path, starterDoc(null));
  return path;
}

export function DocPanel() {
  const params = useSurfaceParams();
  const selected = params.get("n");

  const [entries, setEntries] = useState<NoteEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const select = useCallback(
    // replace, not push: clicking through the gallery is browsing, and it
    // should not take twenty presses of Back to leave the notes tab.
    (path: string | null) => params.set({ n: path }),
    [params],
  );

  // The listing. Every mutation route answers with the whole list, so this is
  // the only place that fetches it and nothing has to guess what changed.
  useEffect(() => {
    let live = true;
    api
      .notes()
      .then(async (list) => {
        if (!live) return;
        setEntries(await adoptScratch(list));
      })
      .catch((e: unknown) => live && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      live = false;
    };
  }, []);

  const paths = useMemo(() => (entries ?? []).map((e) => e.path), [entries]);
  const tree = useMemo(() => buildTree(entries ?? []), [entries]);
  const isDir = useMemo(
    () => new Set((entries ?? []).filter((e) => e.dir).map((e) => e.path)),
    [entries],
  );
  /** Every folder, deepest path and all, in the order the chips print. */
  const folders = useMemo(
    () => (entries ?? []).filter((e) => e.dir).map((e) => e.path).sort((a, b) => a.localeCompare(b)),
    [entries],
  );

  /** The chip you are standing on: a folder, or "" for All. */
  const cwd = selected ? (isDir.has(selected) ? selected : folderOf(selected)) : "";

  const run = (p: Promise<NoteEntry[]>) =>
    p.then(setEntries, (e: unknown) => setError(e instanceof Error ? e.message : String(e)));

  const newNote = () => {
    createNote(cwd)
      .then(async (path) => {
        setEntries(await api.notes());
        select(path);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  };

  const newFolder = () => {
    const path = uniquePath(paths, "", "New folder").replace(/\.html$/, "");
    run(api.newFolder(path)).then(() => setRenaming(path));
  };

  const [renaming, setRenaming] = useState<string | null>(null);

  const rename = (path: string, raw: string) => {
    setRenaming(null);
    const name = cleanName(raw);
    if (!name || name === noteName(path)) return;
    const folder = folderOf(path);
    const to = isDir.has(path)
      ? uniquePath(paths, folder, name).replace(/\.html$/, "")
      : uniquePath(paths, folder, name);
    void run(api.moveNote(path, to)).then(() => selected === path && select(to));
  };

  const move = (from: string, toFolder: string) => {
    if (from === toFolder || folderOf(from) === toFolder) return;
    // Dropping a folder into its own subtree would delete it from the tree it
    // is being dropped into. The server refuses too; refusing here means the
    // drag simply does nothing instead of flashing an error.
    if (isDir.has(from) && (toFolder + "/").startsWith(from + "/")) return;
    const to = isDir.has(from)
      ? uniquePath(paths, toFolder, noteName(from)).replace(/\.html$/, "")
      : uniquePath(paths, toFolder, noteName(from));
    void run(api.moveNote(from, to)).then(() => selected === from && select(to));
  };

  const remove = (path: string) => {
    const inside = notesUnder(nodeAt(tree, path)?.children ?? []).length;
    if (inside && !window.confirm(`Delete “${noteName(path)}” and ${inside} notes inside it?`))
      return;
    void run(api.deleteNote(path)).then(() => {
      if (selected === path || selected?.startsWith(`${path}/`)) select(null);
    });
  };

  const openNote = entries && selected && !isDir.has(selected) ? selected : null;

  if (error)
    return <p className="text-sm text-[var(--danger)]">Could not reach the notes — {error}</p>;

  if (openNote)
    return <NoteEditor key={openNote} path={openNote} onBack={() => select(folderOf(openNote))} />;

  return (
    <Library
      folder={cwd}
      folders={folders}
      // A folder chip shows everything filed under it, its subfolders
      // included: standing on Scouting and being told it is empty because the
      // notes are one level down is the tree's answer, not a gallery's.
      notes={notesUnder(cwd ? (nodeAt(tree, cwd)?.children ?? []) : tree)}
      loading={entries === null}
      renaming={renaming}
      onPick={select}
      onOpen={select}
      onNew={newNote}
      onNewFolder={newFolder}
      onRename={rename}
      onStartRename={setRenaming}
      onMove={move}
      onDelete={remove}
    />
  );
}

const nodeAt = (nodes: NoteNode[], path: string): NoteNode | null => {
  for (const n of nodes) {
    if (n.path === path) return n;
    const found = n.dir ? nodeAt(n.children, path) : null;
    if (found) return found;
  }
  return null;
};

/* ---------------------------------------------------------------- gallery */

/** The library, as a gallery: a row of folder chips, then the notes.
 *
 *  WHAT CHANGED AND WHY. This was a file tree in a rail down the left, which is
 *  the shape a filesystem has rather than the shape the work has. A club's
 *  library is shallow — a folder per team, one for the club, a couple a coach
 *  made — and a tree spends a permanent column of the pane on twelve rows that
 *  never nest more than two deep. Chips say the same thing across one line and
 *  give the notes the whole pane.
 *
 *  ONE SORT, EVERYWHERE: most recently touched first. A gallery is for finding
 *  the thing you were just in, and alphabetical order buries it under whatever
 *  a coach happened to name things in 2019. The chips are the filter; recency
 *  is the order.
 *
 *  The chips are also the drop targets, so a note still moves by dragging it
 *  onto a folder — All being the way back out to the top level.
 */
function Library({
  folder,
  folders,
  notes,
  loading,
  renaming,
  onPick,
  onOpen,
  onNew,
  onNewFolder,
  onRename,
  onStartRename,
  onMove,
  onDelete,
}: {
  folder: string;
  folders: string[];
  notes: NoteNode[];
  loading: boolean;
  renaming: string | null;
  onPick: (path: string | null) => void;
  onOpen: (path: string) => void;
  onNew: () => void;
  onNewFolder: () => void;
  onRename: (path: string, name: string) => void;
  onStartRename: (path: string) => void;
  onMove: (from: string, toFolder: string) => void;
  onDelete: (path: string) => void;
}) {
  const recent = [...notes].sort((a, b) => (b.entry?.updated ?? 0) - (a.entry?.updated ?? 0));
  // The stylesheet scan is the expensive half of building a preview, and it is
  // the same for every card. Once per gallery, not once per note.
  const rules = useMemo(() => documentRules(), []);

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="ft-controls flex shrink-0 flex-wrap items-center gap-2">
        <Chip
          label="All"
          active={!folder}
          renaming={false}
          icon={Files}
          onClick={() => onPick(null)}
          onDrop={(from) => onMove(from, "")}
        />
        {folders.map((f) => (
          <Chip
            key={f}
            label={f}
            active={folder === f}
            renaming={renaming === f}
            icon={Folder}
            onClick={() => onPick(f)}
            onRename={(name) => onRename(f, name)}
            onStartRename={() => onStartRename(f)}
            onDelete={() => onDelete(f)}
            onDrop={(from) => onMove(from, f)}
          />
        ))}
        <span className="ml-auto flex items-center gap-2">
          <TileButton onClick={onNew} label="Create note" icon={Plus} />
          <TileButton onClick={onNewFolder} label="New folder" icon={FolderPlus} />
        </span>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {loading ? (
          <p className="text-[12px] text-[var(--mute)]">Loading…</p>
        ) : recent.length === 0 ? (
          <div className="grid h-full place-items-center">
            <div className="flex flex-col items-center gap-3">
              {/* The one sentence worth saying here is what a note is FOR, and
                  the thing that is new: it is not tied to a match. */}
              <p className="max-w-[34ch] text-center text-[12px] text-[var(--mute)]">
                A note belongs to the club, not to one match — pull numbers from
                any match, any team, any season into the same page.
              </p>
              <button
                type="button"
                onClick={onNew}
                className="flex h-7 items-center gap-1.5 rounded-[var(--radius-pill)] border border-[var(--hairline)] px-3 text-[12px] text-[var(--ink-soft)] transition-colors hover:bg-[var(--card)] hover:text-[var(--ink)]"
              >
                <FilePlus2 size={12} strokeWidth={1.5} /> Create note
              </button>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(210px,1fr))] gap-3 pb-2">
            {recent.map((n) => (
              <Card
                key={n.path}
                node={n}
                rules={rules}
                renaming={renaming === n.path}
                showFolder={!folder}
                onOpen={() => onOpen(n.path)}
                onRename={(name) => onRename(n.path, name)}
                onStartRename={() => onStartRename(n.path)}
                onDelete={() => onDelete(n.path)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/** A folder, as a tile: icon, name, and a ⋮ for what can be done to it.
 *
 *  The shape a documents app uses for a folder — Drive's row of tiles — rather
 *  than a text pill, because a folder is a THING you can rename, delete and
 *  drop notes onto, and a pill reads as a filter chip that merely toggles. The
 *  ⋮ is where those verbs live; double-click still renames, for the hand that
 *  already knows the gesture.
 */
function Chip({
  label,
  active,
  renaming,
  icon: Icon,
  onClick,
  onRename,
  onStartRename,
  onDelete,
  onDrop,
}: {
  label: string;
  active: boolean;
  renaming: boolean;
  icon: typeof Folder;
  onClick: () => void;
  onRename?: (name: string) => void;
  onStartRename?: () => void;
  onDelete?: () => void;
  onDrop: (from: string) => void;
}) {
  const [over, setOver] = useState(false);

  if (renaming && onRename) {
    return (
      <input
        autoFocus
        defaultValue={noteName(label)}
        onFocus={(e) => e.currentTarget.select()}
        onBlur={(e) => onRename(e.currentTarget.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") e.currentTarget.blur();
          if (e.key === "Escape") onRename(noteName(label));
        }}
        className="h-9 w-44 rounded-[var(--radius-md)] border border-[var(--accent)] bg-[var(--card)] px-2.5 text-[12.5px] text-[var(--ink)] focus-visible:outline-none"
      />
    );
  }

  return (
    <span
      onDragOver={(e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = "move";
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const from = e.dataTransfer.getData("text/plain");
        if (from) onDrop(from);
      }}
      className={cn(
        "flex h-9 items-center gap-1 rounded-[var(--radius-md)] border pr-1 transition-colors",
        active
          ? "border-[var(--hairline-strong)] bg-[var(--card-hover)] text-[var(--ink)]"
          : "border-transparent bg-[var(--card)] text-[var(--ink-soft)] hover:bg-[var(--card-hover)] hover:text-[var(--ink)]",
        over && "border-[var(--accent)] bg-[var(--accent)]/15",
      )}
    >
      <button
        type="button"
        onClick={onClick}
        onDoubleClick={onStartRename}
        title={onStartRename ? `${label} — double-click to rename` : label}
        className="flex h-9 max-w-[22ch] items-center gap-2 pr-1 pl-2.5 text-[12.5px]"
      >
        <Icon size={14} strokeWidth={1.7} className="shrink-0 text-[var(--graphite)]" />
        <span className="truncate">{label}</span>
      </button>
      {onDelete && onStartRename && (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              type="button"
              aria-label={`${label} actions`}
              className="grid size-6 shrink-0 place-items-center rounded-[var(--radius-sm)] text-[var(--stone)] transition-colors hover:bg-[var(--panel)] hover:text-[var(--ink)]"
            >
              <MoreVertical size={13} strokeWidth={1.7} />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="w-40">
            <DropdownMenuItem onSelect={onStartRename}>
              <Pencil className="size-4 text-[var(--graphite)]" /> Rename
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={onDelete} variant="destructive">
              <Trash2 className="size-4" /> Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      )}
    </span>
  );
}

/** One note. The card is a div rather than a button because it contains a
 *  frame, and a frame inside a button is interactive content nested inside
 *  interactive content — which no browser agrees on. Enter and Space are wired
 *  back up by hand. */
function Card({
  node,
  rules,
  renaming,
  showFolder,
  onOpen,
  onRename,
  onStartRename,
  onDelete,
}: {
  node: NoteNode;
  rules: string;
  renaming: boolean;
  showFolder: boolean;
  onOpen: () => void;
  onRename: (name: string) => void;
  onStartRename: () => void;
  onDelete: () => void;
}) {
  return (
    <div
      role="button"
      tabIndex={0}
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData("text/plain", node.path);
        e.dataTransfer.effectAllowed = "move";
      }}
      onClick={onOpen}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onOpen();
        }
        if (e.key === "F2") onStartRename();
      }}
      className="group flex cursor-default flex-col overflow-hidden rounded-[var(--radius-md)] border border-[var(--hairline)] text-left transition-colors hover:border-[var(--stone)] hover:bg-[var(--card)]"
    >
      <Preview entry={node.entry} rules={rules} />
      <span className="flex items-center gap-1 px-2.5 py-2">
        <span className="flex min-w-0 flex-1 flex-col gap-0.5">
          {renaming ? (
            <input
              autoFocus
              defaultValue={node.name}
              onClick={(e) => e.stopPropagation()}
              onFocus={(e) => e.currentTarget.select()}
              onBlur={(e) => onRename(e.currentTarget.value)}
              onKeyDown={(e) => {
                e.stopPropagation();
                if (e.key === "Enter") e.currentTarget.blur();
                if (e.key === "Escape") onRename(node.name);
              }}
              className="h-5 w-full rounded-[var(--radius-xs)] border border-[var(--accent)] bg-[var(--card)] px-1 text-[12px] text-[var(--ink)] focus-visible:outline-none"
            />
          ) : (
            <span
              onDoubleClick={(e) => {
                e.stopPropagation();
                onStartRename();
              }}
              title={`${node.name} — double-click to rename`}
              className="truncate text-[12px] text-[var(--ink)]"
            >
              {node.name}
            </span>
          )}
          <span className="truncate text-[10px] text-[var(--mute)]">
            {[showFolder ? folderOf(node.path) : "", when(node.entry?.updated)]
              .filter(Boolean)
              .join("  ·  ")}
          </span>
        </span>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          aria-label={`Delete ${node.name}`}
          title="Delete"
          className="hidden size-5 shrink-0 place-items-center rounded-[var(--radius-xs)] text-[var(--stone)] transition-colors group-hover:grid hover:text-[var(--danger)]"
        >
          <Trash2 size={10} strokeWidth={1.6} />
        </button>
      </span>
    </div>
  );
}

/** THE PICTURE ON A CARD: the top of the note, as it actually looks.
 *
 *  It was a snippet of prose beside the note's first SVG — a paraphrase, and a
 *  figure lifted out of the page it sits on. What makes a document
 *  recognisable is the shape of its own first screen: where the heading sits,
 *  whether it opens on a board or on three columns of prose, how dense it is.
 *  So the card renders the note's real markup, at a third scale, as a page.
 *
 *  In a sandboxed <iframe srcdoc>, which is the point: `sandbox=""` is every
 *  restriction on, so nothing in a note a coach pasted from somewhere can run
 *  a script, load a tracker, or reach this page — a guarantee from the platform
 *  rather than from a sanitiser of ours. `pointer-events: none` keeps the whole
 *  card one click target, and the frame is lazy so a library of eighty notes
 *  renders the dozen that are on screen.
 */
const SCALE = 0.3;

function Preview({ entry, rules }: { entry?: NoteEntry; rules: string }) {
  // No `bg-` in the shared part: two background classes on one element is a
  // race between two rules of equal weight, decided by their order in the
  // generated stylesheet rather than by the order they are written here.
  const box = "relative h-[150px] shrink-0 overflow-hidden border-b border-[var(--hairline)]";
  if (!entry?.head?.trim()) {
    return (
      <span className={`${box} grid place-items-center bg-[var(--card)]`}>
        <span className="text-[10px] text-[var(--mute)]">{entry?.snippet || "Empty"}</span>
      </span>
    );
  }
  return (
    <span className={`${box} bg-white`}>
      <iframe
        title=""
        aria-hidden
        tabIndex={-1}
        loading="lazy"
        sandbox=""
        srcDoc={previewHtml(entry.head, rules)}
        // Width in per cent of the card so the page fills it at any column
        // width, then scaled: 1/0.3 of the box across, and tall enough that the
        // scaled result overflows 150px rather than ending in white.
        style={{
          width: `${100 / SCALE}%`,
          height: 150 / SCALE,
          transform: `scale(${SCALE})`,
          transformOrigin: "top left",
        }}
        className="pointer-events-none absolute inset-0 border-0"
      />
    </span>
  );
}

/** The two verbs at the end of the chip row: a plus, and a folder with a plus.
 *  The tile a folder wears, at one square — same height, same radius, same
 *  fill, so they belong to the row rather than reading as leftovers from the
 *  rail it replaced. No words: these two symbols are the two most literal in
 *  the whole app, and the name is on the tooltip for the hand that hesitates. */
function TileButton({
  onClick,
  label,
  icon: Icon,
}: {
  onClick: () => void;
  label: string;
  icon: typeof FilePlus2;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      className="grid size-9 place-items-center rounded-[var(--radius-md)] border border-transparent bg-[var(--card)] text-[var(--graphite)] transition-colors hover:bg-[var(--card-hover)] hover:text-[var(--ink)]"
    >
      <Icon size={15} strokeWidth={1.7} />
    </button>
  );
}

/** Epoch seconds as the shortest thing that is still unambiguous. */
function when(t?: number): string {
  if (!t) return "";
  const d = new Date(t * 1000);
  const days = (Date.now() - d.getTime()) / 86_400_000;
  return days < 1
    ? d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })
    : d.toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

/* ----------------------------------------------------------------- editor */

type SaveState = "saved" | "dirty" | "saving" | "error";

/** 404 = a note in the tree whose file is gone; seed rather than fail. `saved`
 *  is false for a seeded document so autosave writes it, and true for one that
 *  came back from the server, which would otherwise be rewritten byte for byte
 *  on every open. */
/** ASK THE SERVER FOR THE PDF, rather than asking the browser to print.
 *
 *  Chrome draws its own furniture around anything it prints — the date and the
 *  document title across the top, the page URL and "1/8" along the bottom — and
 *  that is a checkbox in the READER'S settings that no page can reach.
 *  `@page { margin: 0 }` only persuades Chrome to skip it when the reader has
 *  not asked for it. A document handed to a coach cannot depend on that, so
 *  /export/pdf runs the same render headlessly and sends back bytes.
 */
async function savePdf(
  html: string,
  title: string,
  sub: string,
  onError: (m: string | null) => void,
): Promise<void> {
  onError(null);
  try {
    // The desktop build is a static export with no /export route; the shell
    // prints instead (desktop/electron/main.ts, "print-pdf").
    const desktop = (window as { futtrafficDesktop?: { printPdf?: (h: string) => Promise<Uint8Array> } })
      .futtrafficDesktop;
    if (desktop?.printPdf) {
      const pdf = await desktop.printPdf(documentHtml(html, title, documentRules(), sub));
      download(new Blob([new Uint8Array(pdf)], { type: "application/pdf" }), title);
      return;
    }
    const res = await fetch("/export/pdf", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ html, title, sub }),
    });
    if (!res.ok) {
      const { error } = (await res.json().catch(() => ({}))) as { error?: string };
      throw new Error(error ?? `PDF export failed (${res.status})`);
    }
    download(await res.blob(), title);
  } catch (e) {
    onError(e instanceof Error ? e.message : String(e));
  }
}

function download(blob: Blob, title: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${(title || "document").replace(/[/\\]/g, "-")}.pdf`;
  a.click();
  // On a timer rather than immediately: Safari has not finished reading the
  // blob by the time click() returns.
  window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
}

const CONFLICT_BTN =
  "flex h-7 shrink-0 items-center rounded-[var(--radius-sm)] px-1.5 text-[11px] text-[var(--ink)] transition-colors hover:bg-[var(--card)] disabled:opacity-40";

async function loadNote(path: string): Promise<{ html: string; saved: boolean; version: string }> {
  try {
    const { html, version } = await api.getNote(path);
    return { html, saved: true, version };
  } catch (e) {
    if (!(e instanceof ApiError && e.status === 404)) throw e;
    // A note that does not exist yet has the empty version, and the server
    // reads that as "there must be no file here" — so the first write of a
    // seeded document is itself guarded.
    return { html: starterDoc(null), saved: false, version: "" };
  }
}


/** The pre-library scratch note, lifted into the library the first time this
 *  browser sees a listing without it. One-way and idempotent: the key is
 *  dropped only after the server has the bytes. */
async function adoptScratch(list: NoteEntry[]): Promise<NoteEntry[]> {
  let html: string | null = null;
  try {
    html = window.localStorage.getItem(SCRATCH_KEY);
  } catch {
    return list; // private mode; there was never a scratch note to adopt
  }
  if (!html) return list;
  const path = uniquePath(list.map((e) => e.path), "", "Scratch");
  try {
    await api.putNote(path, html);
    window.localStorage.removeItem(SCRATCH_KEY);
    return await api.notes();
  } catch {
    return list; // keep the key: an adoption that failed must be retryable
  }
}

/** The note, open over the whole pane. `onBack` rather than a rail: the
 *  gallery and the document are two screens now, and the way out of a document
 *  is the same arrow it is in every reader — top left, before the title. */
function NoteEditor({ path, onBack }: { path: string; onBack: () => void }) {
  const [doc, setDoc] = useState<string | null>(null);
  // For `/stat`'s match-scoped sources: whichever match the header has
  // selected. The NOTE is not about that match — it is not about any — but a
  // coach reading one match and writing about it should not have to re-pick it.
  const { report } = useReport();
  /** The last html the server has confirmed. doc !== savedHtml = unsaved. */
  const [savedHtml, setSavedHtml] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  /** The PDF render runs on the server and takes a second or two on a big
   *  report, so the button says so rather than looking dead. */
  const [pdfBusy, setPdfBusy] = useState(false);
  const [pdfError, setPdfError] = useState<string | null>(null);
  /** The newest text, saved or not, and the newest the server has been handed.
   *  Refs rather than the states above because the unmount flush runs after the
   *  last render and cannot wait for one more. */
  const latest = useRef<string | null>(null);
  const sent = useRef<string | null>(null);
  /** The version the server last confirmed, quoted back on every write so a
   *  script that rewrote this note underneath us is caught rather than
   *  silently overwritten. A ref for the same reason as the two above: the
   *  unmount flush runs after the last render. */
  const version = useRef<string>("");
  /** A 409 stops the autosave until the reader picks a side — retrying with
   *  the same stale version could only ever 409 again. */
  const [conflict, setConflict] = useState(false);
  /** Bumped by "Load theirs" to remount the editor on the server's text; the
   *  editor reads `html` once, so a prop change alone would not show it. */
  const [rev, setRev] = useState(0);
  /** Set across that remount so the old editor's unmount flush cannot put the
   *  discarded text back in `latest`, where the panel's own flush would then
   *  write it over theirs. */
  const dropFlush = useRef(false);
  useEffect(() => {
    dropFlush.current = false;
  }, [rev]);

  useEffect(() => {
    let live = true;
    loadNote(path)
      .then(({ html, saved, version: v }) => {
        if (!live) return;
        version.current = v;
        // A write this browser lost on the way out of the note last time.
        // Preferred over the server's copy, because it is strictly newer and
        // it is the one nobody has seen since.
        const parked = unpark(path);
        setDoc(parked ?? html);
        if (saved && !parked) setSavedHtml(html);
        if (parked) {
          // Say what is about to happen, not what could. savedHtml is left
          // unset above, so this text is already dirty and the debounce writes
          // it a second from now without anyone pressing anything.
          setSaveError(
            "the last edit here could not be saved and was held in this tab — it is back on screen and is being saved now",
          );
        }
      })
      .catch((e: unknown) => {
        if (live) setLoadError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      live = false;
    };
  }, [path]);

  // Debounced autosave. Typing past an in-flight save simply re-arms it — the
  // newest text always wins the next round.
  useEffect(() => {
    if (doc === null || doc === savedHtml) return;
    latest.current = doc;
    if (conflict) return;
    const t = window.setTimeout(() => {
      setSaving(true);
      setSaveError(null);
      sent.current = doc;
      api.putNote(path, doc, version.current).then(
        ({ version: v }) => {
          setSaving(false);
          version.current = v;
          setSavedHtml(doc);
        },
        (e: unknown) => {
          setSaving(false);
          if (e instanceof ApiError && e.status === 409) setConflict(true);
          // 409 is not a network failure, it is somebody else's write. Say
          // which, because the two want different things from the reader: one
          // is "try again", the other is "your copy and theirs have diverged".
          setSaveError(
            e instanceof ApiError && e.status === 409
              ? "this note was changed somewhere else — your text is still here, but saving would overwrite theirs"
              : e instanceof Error
                ? e.message
                : String(e),
          );
        },
      );
    }, 1000);
    return () => window.clearTimeout(t);
  }, [path, doc, savedHtml, conflict]);

  /** Overwrite: the unguarded write, which is exactly what the reader chose. */
  const keepMine = () => {
    if (doc === null) return;
    setSaving(true);
    sent.current = doc;
    api.putNote(path, doc).then(
      ({ version: v }) => {
        version.current = v;
        setSavedHtml(doc);
        setConflict(false);
        setSaveError(null);
        setSaving(false);
      },
      (e: unknown) => {
        setSaving(false);
        setSaveError(e instanceof Error ? e.message : String(e));
      },
    );
  };

  /** Discard this tab's text and show the server's. */
  const loadTheirs = () => {
    api.getNote(path).then(
      ({ html, version: v }) => {
        version.current = v;
        latest.current = sent.current = html;
        dropFlush.current = true;
        setDoc(html);
        setSavedHtml(html);
        setConflict(false);
        setSaveError(null);
        setRev((r) => r + 1);
      },
      (e: unknown) => setSaveError(e instanceof Error ? e.message : String(e)),
    );
  };

  // The other half of the same hole: switching notes unmounts this mid-debounce
  // and the timer above is merely cleared. `path` is fixed for the instance's
  // whole life — the component is KEYED on it — so the flush can never land the
  // text of one note on top of another.
  //
  // It used to end `.catch(() => {})`. That was survivable while the only way
  // to lose was a network blip; with a version guard on the write it is not,
  // because a rejected stale write would be DISCARDED IN SILENCE — a guard
  // whose rejections are invisible turns a loud clobber into a quiet deletion,
  // which is worse than the clobber. So a flush that does not land parks its
  // text where the next open of this note will find it.
  useEffect(
    () => () => {
      const html = latest.current;
      if (html === null || html === sent.current) return;
      void api.putNote(path, html, version.current).catch(() => park(path, html));
    },
    [path],
  );

  const saveState: SaveState =
    saving ? "saving" : saveError ? "error" : doc !== null && doc !== savedHtml ? "dirty" : "saved";

  // The bar the editor already draws is the only header this pane gets: the
  // way out and the note's name on its left, print and the save state on its
  // right. A second strip above it was a second hairline and a second row of
  // 11px text saying nothing the toolbar could not hold.
  const lead = (
    <>
      <button
        type="button"
        onClick={onBack}
        title="Back to the notes"
        aria-label="Back to the notes"
        className="flex h-7 shrink-0 items-center rounded-[var(--radius-sm)] px-1 text-[var(--graphite)] transition-colors hover:bg-[var(--card)] hover:text-[var(--ink)]"
      >
        <ChevronLeft size={13} strokeWidth={1.7} />
        <Images size={13} strokeWidth={1.7} />
      </button>
      <span
        title={path}
        className="max-w-[16ch] shrink-0 truncate text-[11px] text-[var(--ink)]"
      >
        {noteName(path)}
      </span>
      <span aria-hidden className="mx-0.5 h-4 w-px shrink-0 bg-[var(--hairline)]" />
    </>
  );

  const status =
    // A failed PDF is loud, and first: it is a thing the reader just asked for
    // and did not get, where a save state is ambient.
    pdfError
      ? `PDF failed — ${pdfError}`
      : saveState === "saving"
        ? "Saving…"
        : saveState === "error"
          ? `Save failed — ${saveError}`
          : saveState === "dirty"
            ? "Unsaved changes"
            : doc !== null
              ? "Saved"
              : "";

  const trail = (
    <>
      <button
        type="button"
        onClick={async () => {
          if (!doc) return;
          setPdfBusy(true);
          await savePdf(doc, noteName(path), folderOf(path), setPdfError);
          setPdfBusy(false);
        }}
        disabled={doc === null || pdfBusy}
        title="Print to a PDF — rendered here, so no page numbers, URL or date from the browser's dialog"
        className="flex h-7 shrink-0 items-center gap-1 rounded-[var(--radius-sm)] px-1.5 text-[11px] text-[var(--graphite)] transition-colors hover:bg-[var(--card)] hover:text-[var(--ink)] disabled:opacity-40 disabled:hover:bg-transparent"
      >
        <Printer size={11} strokeWidth={1.7} /> {pdfBusy ? "…" : "Print"}
      </button>
      <span
        role="status"
        title={status}
        className={cn(
          "max-w-[18ch] shrink-0 truncate text-[11px]",
          pdfError || saveState === "error" ? "text-[var(--danger)]" : "text-[var(--mute)]",
        )}
      >
        {status}
      </span>
      {conflict && (
        <>
          <button type="button" onClick={keepMine} disabled={saving} title="Save this tab's text over the other copy" className={CONFLICT_BTN}>
            Keep mine
          </button>
          <button type="button" onClick={loadTheirs} title="Throw away this tab's text and open the saved copy" className={CONFLICT_BTN}>
            Load theirs
          </button>
        </>
      )}
    </>
  );

  return (
    /* `ft-note-pane` is where the board editor opens — see board-editor.tsx.
       It covers the note and nothing else, so the app's header and rail stay
       on screen; `relative` is what makes that an absolute box rather than a
       viewport one. */
    <div className="ft-note-pane relative flex h-full min-h-0 flex-col">
      {doc === null ? (
        // No editor yet, so no bar to hang them on — but the way back must
        // survive a note that will not load.
        <>
          <div className="ft-controls flex shrink-0 items-center gap-1 border-b border-[var(--hairline)] px-1 py-1">
            {lead}
            <span className="flex-1" />
            {trail}
          </div>
          <p className={cn("p-3 text-sm", loadError ? "text-[var(--danger)]" : "text-[var(--mute)]")}>
            {loadError ? `Could not load the note — ${loadError}` : "Loading…"}
          </p>
        </>
      ) : (
        <Notepad
          key={rev}
          html={doc}
          report={report}
          onChange={setDoc}
          onFlush={(html) => {
            if (!dropFlush.current) latest.current = html;
          }}
          lead={lead}
          trail={trail}
        />
      )}
    </div>
  );
}
