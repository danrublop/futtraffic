"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Editor } from "@tiptap/react";
import { EditorContent, ReactNodeViewRenderer, useEditor } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import { TextStyleKit } from "@tiptap/extension-text-style";

import type { MatchReport } from "@/lib/types";
import { Moon, Sun } from "lucide-react";

import { cn } from "@/lib/utils";
import { PitchNode } from "./pitch-node";
import { PitchView } from "./pitch-view";
import { StatNode } from "./stat-node";
import { FigureNode } from "./figure-node";
import { FigureView } from "./figure-view";
import { QueryNode } from "./query-node";
import { QueryView } from "./query-view";
// StatNode stays registered and StatPanel stays in the file: a note
// written before this can still hold a `/stat` block and must still
// render it. It is off the MENU because a lone number floating in a
// document is not a figure — no page of the scouting report has one —
// and every number it could show is a row of a table that says what it
// is next to.
import { PlayPanel } from "./slash-panels";
import { TablePanel, VisualPanel } from "./flow-panels";
import { ColumnNode, ColumnsNode } from "./columns-node";
import { TABLE_EXTENSIONS, inTable, layoutRow, parseGrid, tableFromRows, type Block } from "./table-grid";
import { ColumnsView } from "./columns-view";
import { Toolbar } from "./toolbar";
import { FaceNode } from "./face-node";
import { PageBreakNode, PageNode, PageSetupNode, orientOf } from "./page-nodes";
import { TableMenu, type MenuAt } from "./table-menu";
import { tableUnder } from "./lock-plugin";

/** The notepad: one surface, always the document.
 *
 *  There is no Edit/Preview toggle any more. It was there because the notepad
 *  stored raw HTML and the only way to write it was to type markup — which made
 *  a coach's notebook a code editor with a preview button. This is a rich text
 *  editor: what you see is the note, the way it is in Google Docs, and the HTML
 *  is an implementation detail that only matters when the file leaves.
 *
 *  What makes it a FOOTBALL notepad rather than a text box is two blocks, both
 *  on the `/` menu:
 *   - `/pitch` — the magnet board a coach already owns, with the discs draggable
 *     and their positions saved into the document.
 *   - `/stat`  — a number out of this match's report, carrying the confidence
 *     tier the pipeline gave it.
 *
 *  Storage stays HTML through the same /doc endpoint, because the promise of
 *  this feature is a file you can send to a player or print. Both custom blocks
 *  serialize to real rendered markup (an inline SVG board, a figure with the
 *  number in it) plus a data attribute that reads back into the editor, so the
 *  saved file is a document everywhere and a live board here.
 *
 *  Modelled on the notebook editor in Llamas Remote: TipTap + StarterKit, a
 *  custom Node per block with a React NodeView, `/` to insert, debounced save.
 */

const PitchWithView = PitchNode.extend({
  addNodeView() {
    return ReactNodeViewRenderer(PitchView);
  },
});

const FigureWithView = FigureNode.extend({
  addNodeView() {
    return ReactNodeViewRenderer(FigureView);
  },
});

const QueryWithView = QueryNode.extend({
  addNodeView() {
    return ReactNodeViewRenderer(QueryView);
  },
});

const ColumnsWithView = ColumnsNode.extend({
  addNodeView() {
    return ReactNodeViewRenderer(ColumnsView);
  },
});

interface MenuState {
  open: boolean;
  query: string;
  left: number;
  top: number;
  index: number;
  /** Doc position of the `/` that opened the menu. */
  from: number;
  /** Anchored from the BOTTOM instead of the top, because the caret is too
   *  near the foot of the pane for the menu to grow downwards. */
  flip: boolean;
  /** Set once a command needs a second choice — which stat. */
  mode: "commands" | "stats";
  /** What the menu is pinned to. A typed slash pins it to the caret, because
   *  that is where the coach is looking and where the block will land. The
   *  toolbar pins it to the toolbar — the hand is down there, and a menu that
   *  opens in the middle of the page is a menu you have to go and find. */
  anchor: "caret" | "bar";
}

const CLOSED: MenuState = {
  open: false,
  query: "",
  left: 0,
  top: 0,
  index: 0,
  from: 0,
  flip: false,
  mode: "commands",
  anchor: "caret",
};

export function Notepad({
  html,
  report,
  onChange,
  onFlush,
  lead,
  trail,
}: {
  /** The stored document. Read once — this is an uncontrolled editor. */
  html: string;
  /** This match's report, for `/stat`. Null with no match, which is fine: the
   *  command simply has nothing to offer and says so. */
  report: MatchReport | null;
  onChange: (html: string) => void;
  /** The last thing typed, handed over at unmount because `onChange` cannot
   *  be: it sets state on a component that is going away with this one. */
  onFlush?: (html: string) => void;
  /** Passed straight to the toolbar, which is this pane's only header: the way
   *  back and the note's name on the left, print and the save state on the
   *  right. */
  lead?: React.ReactNode;
  trail?: React.ReactNode;
}) {
  const [menu, setMenu] = useState<MenuState>(CLOSED);
  /** PRINT PREVIEW, in place. The app is dark and the page is not, and the only
   *  way to see a report as it will exist was to print it - which is a dialogue,
   *  a scroll and a guess, and nobody checks a layout that way. This puts the
   *  document on paper inside the editor, still editable, wearing the exact
   *  stylesheet the export wears (`.ft-paper` in globals.css is the one copy of
   *  it). The app around it stays dark, because a coach checking a page is not
   *  changing the theme of the application. */
  const [onPaper, setOnPaper] = useState(false);
  /** The right-click menu for a table, or null. */
  const [tableMenu, setTableMenu] = useState<MenuAt | null>(null);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** Reads the document as it stands. Set on every edit so the unmount flush
   *  does not have to reach for an editor that may already be torn down. */
  const readDoc = useRef<(() => string) | null>(null);
  const flush = useRef(onFlush);
  useEffect(() => {
    flush.current = onFlush;
  }, [onFlush]);

  /** The `/` menu's key handling, reached through a ref.
   *
   *  It has to be ProseMirror's `handleKeyDown` and not a React onKeyDown on a
   *  wrapping div: ProseMirror binds Enter, ArrowUp and ArrowDown itself, and
   *  it wins. A React handler on an ancestor DOES fire, but only after
   *  ProseMirror has already split the paragraph — so Enter inserted a line
   *  break, the caret moved off the `/`, and the menu closed without ever
   *  running the command. Returning true from here is what stops that.
   *
   *  A ref because the handler is registered once when the editor is created
   *  and would otherwise close over the first render's empty menu forever. It
   *  is assigned further down, where `results` exists. */
  const onMenuKey = useRef<(e: KeyboardEvent) => boolean>(() => false);
  /** The formatting bar, so the menu it opens can sit directly under it. */
  const bar = useRef<HTMLDivElement | null>(null);

  /** Put the document's own paper on the editor root, so the surface a coach
   *  types on is the shape of the page it prints to. The setting lives in the
   *  document (page-nodes.ts); this only mirrors it onto an attribute the
   *  stylesheet can see. */
  const paper = (e: Editor) => {
    const first = e.state.doc.firstChild;
    const orient = first?.type.name === "pageSetup"
      ? orientOf(String(first.attrs.orient ?? ""))
      : orientOf(null);
    e.view.dom.setAttribute("data-orient", orient);
  };

  const editor = useEditor({
    // The workspace prerenders (static export for the desktop shell), so the
    // editor must not render on the server.
    immediatelyRender: false,
    // TextStyleKit is what makes the font and size controls possible —
    // StarterKit already brings bold/italic/underline/lists/link/quote.
    extensions: [
      StarterKit, TextStyleKit, PitchWithView, StatNode,
      FigureWithView, QueryWithView, ...TABLE_EXTENSIONS,
      // Kept registered, off the menu. `/columns` now makes a borderless
      // table, which is what Word and Google Docs users make for the same job
      // and comes with draggable widths for free — but a note saved before
      // that would lose its layout if the node stopped parsing.
      ColumnsWithView, ColumnNode,
      // Paper, and the faces a scouting report is headed with.
      PageSetupNode, PageNode, PageBreakNode, FaceNode,
    ],
    content: html,
    editorProps: {
      attributes: {
        class: "ft-doc focus:outline-none",
      },
      handleKeyDown: (_view, event) => onMenuKey.current(event),
      handleDOMEvents: {
        /** Right-click on a table opens ITS menu, not the browser's.
         *
         *  Anywhere else the browser's own menu is left alone — spell-check,
         *  copy and "search for" are the reasons a coach right-clicks prose,
         *  and taking those away to show nothing useful would be a loss. */
        contextmenu: (view, event) => {
          const at = view.posAtCoords({ left: event.clientX, top: event.clientY });
          const found = at ? tableUnder(view.state, at.pos) : null;
          if (!found) return false;
          event.preventDefault();
          setTableMenu({ pos: found.pos, x: event.clientX, y: event.clientY });
          return true;
        },
      },
      /** A grid on the clipboard becomes a grid in the document.
       *
       *  Declines whenever there is an HTML half to the paste — that is the
       *  Excel/Sheets/Numbers case and ProseMirror already parses it through
       *  the table node's own `parseHTML`, better than this could. */
      handlePaste: (view, event) => {
        const data = event.clipboardData;
        if (!data || data.types.includes("text/html")) return false;
        const grid = parseGrid(data.getData("text/plain"));
        if (!grid) return false;
        // ponytail: a paste INSIDE a table drops into the one cell, the way any
        // other paste does. Filling a range outward from the caret needs
        // TableMap and the table grown to fit — worth doing the day somebody
        // pastes a column of numbers beside the players, not before.
        if (inTable(view.state)) return false;
        const node = view.state.schema.nodeFromJSON(tableFromRows(grid[0], grid.slice(1)));
        view.dispatch(view.state.tr.replaceSelectionWith(node).scrollIntoView());
        return true;
      },
    },
    onUpdate: ({ editor }) => {
      paper(editor);
      if (saveTimer.current) clearTimeout(saveTimer.current);
      readDoc.current = () => editor.getHTML();
      saveTimer.current = setTimeout(() => {
        saveTimer.current = null;
        onChange(editor.getHTML());
      }, 600);
      detectSlash();
    },
    onSelectionUpdate: () => detectSlash(),
    onCreate: ({ editor }) => paper(editor),
  });

  // A pending save must still land if the tab closes mid-debounce — and it did
  // not. This used to clear the timer and stop there, so the last 600 ms of
  // typing was thrown away every time the coach switched match, with the pill
  // still reading "Saved" because `doc` had never been told otherwise.
  useEffect(
    () => () => {
      if (!saveTimer.current) return;
      clearTimeout(saveTimer.current);
      const html = readDoc.current?.();
      if (html !== undefined) flush.current?.(html);
    },
    [],
  );

  const detectSlash = useCallback(() => {
    if (!editor) return;
    const { $from, empty } = editor.state.selection;
    if (!empty) return setMenu(CLOSED);
    const before = $from.parent.textBetween(0, $from.parentOffset, undefined, "￼");
    const m = /(?:^|\s)\/(\S*)$/.exec(before);
    if (!m) return setMenu(CLOSED);
    const query = m[1];
    const coords = editor.view.coordsAtPos($from.pos);
    // The menu is a list plus a panel that can be most of the viewport tall,
    // and the caret is wherever the coach happens to be typing — which is very
    // often the last line. Anchored naively it opens off the bottom of the
    // screen, and the panel is the half that gets lost.
    const WIDTH = 600;
    const NEEDS = 340;
    const flip = coords.bottom + NEEDS > window.innerHeight && coords.top > NEEDS;
    setMenu((prev) => ({
      ...prev,
      open: true,
      query,
      left: Math.max(8, Math.min(coords.left, window.innerWidth - WIDTH - 8)),
      top: flip ? coords.top : coords.bottom,
      flip,
      index: prev.open ? prev.index : 0,
      from: $from.pos - query.length - 1,
      mode: prev.open ? prev.mode : "commands",
      anchor: "caret",
    }));
  }, [editor]);

  /** Open the block menu from the toolbar.
   *
   *  It used to type the slash, so that the button and the keystroke could not
   *  drift. They did not drift; the menu went and opened wherever the caret
   *  happened to be — the middle of the page, or the title — and left a "/" in
   *  the document behind it if you changed your mind. Both of those are worse
   *  than two ways in.
   *
   *  So this touches the document not at all. It opens the same menu against
   *  the toolbar, with `word` standing in for what would have been typed after
   *  the slash, and works out where the block should land.
   */
  const openMenu = useCallback(
    (word?: string) => {
      if (!editor) return;
      const { $from } = editor.state.selection;
      // Not AT the caret: a coach who has not clicked into the document yet
      // has a caret in the title, and a chart inserted there splits the
      // heading in two. The block goes after whatever block the caret is in,
      // unless that block is an empty paragraph — which is a gap somebody
      // left for it.
      const gap = $from.depth === 1 && $from.parent.content.size === 0;
      setMenu({
        ...CLOSED,
        open: true,
        query: word ?? "",
        anchor: "bar",
        // Measured now rather than during render: the menu hangs directly
        // under the bar, whose height is whatever its content and the pane's
        // width make it.
        top: bar.current?.offsetHeight ?? 36,
        from: gap || !$from.depth ? $from.pos : $from.after(1),
      });
    },
    [editor],
  );

  /** Put the block in. From a typed slash that means replacing the `/query`
   *  that opened the menu; from the toolbar there is nothing to replace, and
   *  the block lands at the position worked out when the menu opened. */
  const replaceSlash = useCallback(
    (content: Block | Block[]) => {
      if (!editor) return;
      const chain = editor.chain().focus();
      if (menu.anchor === "bar") chain.insertContentAt(menu.from, content);
      else
        chain
          .deleteRange({ from: menu.from, to: editor.state.selection.$from.pos })
          .insertContent(content);
      chain.run();
      setMenu(CLOSED);
    },
    [editor, menu.from, menu.anchor],
  );

  /** `insert` is what a panel calls when the coach has finished choosing:
   *  replace the `/query` they typed, drop the node in, close the menu. */
  const insert = useCallback(
    (content: Block | Block[]) => replaceSlash(content),
    [replaceSlash],
  );

  const COMMANDS = useMemo(
    () => [
      // Ordered by what the block LOOKS like, not by where its numbers come
      // from. A coach opening this menu knows they want a table or a picture;
      // "chart vs figure vs stat table" was this codebase's filing system
      // showing through the UI.
      {
        id: "table",
        name: "Table",
        hint: "Rows and columns",
        panel: () => <TablePanel report={report} insert={insert} />,
      },
      {
        id: "visual",
        name: "Visual",
        hint: "Bars, a line, a map",
        panel: () => <VisualPanel report={report} insert={insert} />,
      },
      {
        id: "play",
        name: "Play",
        hint: "A moment, on a board",
        panel: () => <PlayPanel report={report} insert={insert} />,
      },
      {
        id: "pitch",
        name: "Board",
        hint: "A blank pitch",
        // No options to choose: a blank board IS the thing.
        run: () => replaceSlash({ type: "pitch" }),
      },
      {
        id: "columns",
        name: "Columns",
        hint: "Blocks side by side",
        // A borderless table, not a container of its own: that is what both
        // Word and Google Docs users build for this, and it arrives with
        // draggable widths and add-a-column already working.
        run: () => replaceSlash(layoutRow([{ type: "paragraph" }, { type: "paragraph" }])),
      },
      { id: "h1", name: "Heading", hint: "", run: () => replaceSlash({ type: "heading", attrs: { level: 2 } }) },
      { id: "h2", name: "Subheading", hint: "", run: () => replaceSlash({ type: "heading", attrs: { level: 3 } }) },
      { id: "ul", name: "Bullet list", hint: "", run: () => replaceSlash({ type: "bulletList", content: [{ type: "listItem", content: [{ type: "paragraph" }] }] }) },
      { id: "ol", name: "Numbered list", hint: "", run: () => replaceSlash({ type: "orderedList", content: [{ type: "listItem", content: [{ type: "paragraph" }] }] }) },
      { id: "quote", name: "Quote", hint: "", run: () => replaceSlash({ type: "blockquote", content: [{ type: "paragraph" }] }) },
      { id: "hr", name: "Divider", hint: "", run: () => replaceSlash({ type: "horizontalRule" }) },
    ],
    [replaceSlash, insert, report],
  );

  const results = useMemo(() => {
    if (!menu.open) return [];
    const q = menu.query.toLowerCase();
    return COMMANDS.filter((c) => c.name.toLowerCase().includes(q));
  }, [menu.open, menu.query, COMMANDS]);

  /** The command the panel belongs to. Hovering the list changes it; nothing
   *  about the panel is remembered between commands, which is the point — a
   *  chart's settings mean nothing to a table. */
  const active = results[menu.index];

  // Re-pointed after every render so the handler registered once on the editor
  // always sees the current menu. In an effect rather than in the render body
  // because a keystroke cannot arrive before the browser has painted, and
  // writing a ref during render is how you get a handler that disagrees with
  // what is on screen.
  //
  // `true` means "handled here" — that is what keeps Enter from splitting the
  // paragraph under the caret instead of running the command, and the arrows
  // from moving the caret off the `/`.
  useEffect(() => {
    onMenuKey.current = (e: KeyboardEvent) => {
      if (!menu.open) return false;
      if (e.key === "Escape") {
        setMenu(CLOSED);
        return true;
      }
      if (!results.length) return false;
      if (e.key === "ArrowDown") {
        setMenu((p) => ({ ...p, index: (p.index + 1) % results.length }));
        return true;
      }
      if (e.key === "ArrowUp") {
        setMenu((p) => ({ ...p, index: (p.index - 1 + results.length) % results.length }));
        return true;
      }
      if (e.key === "Enter" || e.key === "Tab") {
        // A panelled command has nothing to run yet — Enter just moves the
        // selection onto it, and the panel beside it is where the choosing
        // happens. Still handled, so Enter never splits the paragraph.
        results[menu.index]?.run?.();
        return true;
      }
      return false;
    };
  }, [menu.open, menu.index, results]);

  return (
    <div className="relative flex h-full min-h-0 flex-col">
      <div ref={bar}>
        {editor && (
          <Toolbar
            editor={editor}
            onInsert={openMenu}
            lead={lead}
            trail={
              <>
                {/* Day or night, which is all this toggle ever was: white
                    sheets on a light ground, or the app's dark surface. The
                    word "Paper" beside a page icon said the same thing in
                    eleven more pixels, and the glyph alone is the switch
                    every other app puts in a header. */}
                <button
                  type="button"
                  onClick={() => setOnPaper((p) => !p)}
                  aria-pressed={onPaper}
                  aria-label={onPaper ? "Back to the app's surface" : "See it on paper, as it prints"}
                  title={onPaper ? "Back to the app's surface" : "See it on paper, as it prints"}
                  className={cn(
                    "flex h-7 w-7 shrink-0 items-center justify-center rounded-[var(--radius-sm)] transition-colors",
                    onPaper
                      ? "bg-[var(--card)] text-[var(--ink)]"
                      : "text-[var(--graphite)] hover:bg-[var(--card)] hover:text-[var(--ink)]",
                  )}
                >
                  {onPaper ? <Sun size={13} strokeWidth={1.7} /> : <Moon size={13} strokeWidth={1.7} />}
                </button>
                {trail}
              </>
            }
          />
        )}
      </div>
      <EditorContent
        editor={editor}
        className={cn("ft-print-root min-h-0 flex-1 overflow-y-auto", onPaper && "ft-paper")}
      />

      {editor && tableMenu && (
        <TableMenu editor={editor} at={tableMenu} onClose={() => setTableMenu(null)} />
      )}

      {/* THE MENU IS TWO PANES. The list never moves; hovering a command
          opens its panel to the right, so running down the list flicks between
          panels without anything closing under the pointer.

          A command with options does not insert a default and leave you to fix
          it in the document — you set it up in the panel, against a preview of
          the answer, and what lands is what you asked for. */}
      {menu.open && (
        <div
          style={
            // Under the toolbar it was opened from — measured rather than
            // guessed, because the bar's height is whatever its own content
            // and the pane's width make it.
            menu.anchor === "bar"
              ? { position: "absolute", right: 16, top: menu.top + 4 }
              : menu.flip
                ? { position: "fixed", left: menu.left, bottom: window.innerHeight - menu.top + 4 }
                : { position: "fixed", left: menu.left, top: menu.top + 4 }
          }
          className="z-50 flex items-start gap-1"
          // The editor must not lose the caret the insert is about to replace.
          onMouseDown={(e) => e.preventDefault()}
        >
          <div
            role="listbox"
            className="max-h-72 w-52 shrink-0 overflow-y-auto rounded-[var(--radius-md)] border border-[var(--hairline)] bg-[var(--panel)] p-1 shadow-[var(--shadow-md)]"
          >
            {results.length === 0 ? (
              <p className="px-2 py-1.5 text-[12px] text-[var(--mute)]">Nothing matches.</p>
            ) : (
              results.map((r, i) => (
                <button
                  key={r.id}
                  type="button"
                  role="option"
                  aria-selected={i === menu.index}
                  onClick={() => !r.panel && r.run()}
                  onMouseEnter={() => setMenu((p) => ({ ...p, index: i }))}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-[var(--radius-sm)] px-2 py-1.5 text-left text-[12.5px] transition-colors",
                    i === menu.index
                      ? "bg-[var(--card)] text-[var(--ink)]"
                      : "text-[var(--ink-soft)] hover:bg-[var(--card)]",
                  )}
                >
                  <span className="min-w-0 flex-1 truncate">{r.name}</span>
                  {r.hint && <span className="shrink-0 text-[11px] text-[var(--mute)]">{r.hint}</span>}
                  {r.panel && <span aria-hidden className="shrink-0 text-[var(--mute)]">›</span>}
                </button>
              ))
            )}
          </div>

          {active?.panel && (
            <div className="max-h-[70vh] overflow-y-auto rounded-[var(--radius-md)] border border-[var(--hairline)] bg-[var(--panel)] shadow-[var(--shadow-md)]">
              {active.panel()}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
