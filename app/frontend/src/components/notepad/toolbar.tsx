"use client";

import { Fragment, useLayoutEffect, useRef, useState } from "react";
import type { Editor } from "@tiptap/react";
import { useEditorState } from "@tiptap/react";
import {
  BarChart3,
  Code,
  Columns2,
  Link2,
  List,
  ListOrdered,
  MoreHorizontal,
  Plus,
  Quote,
  RectangleHorizontal,
  RectangleVertical,
  ScissorsLineDashed,
  Spline,
  Table,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { TableToolbar } from "./table-toolbar";

/** The formatting bar, across the top of the document.
 *
 *  IT USED TO FLOAT, bottom-right, over the page. Two things were wrong with
 *  that: it sat ON the document, so it covered whatever was under it in the
 *  one part of the pane a coach is reading, and it moved with nothing — a bar
 *  that is somewhere in the middle of the surface has no edge to be found
 *  against. Docked at the top it is where every editor keeps one, above the
 *  page rather than on it, and the document below it is uncovered end to end.
 *
 *  IT IS THE ONLY HEADER. The document used to carry a second strip above this
 *  one holding the way back, the note's name and the save state — two bars,
 *  two hairlines, twice the chrome, for a pane whose whole point is the page.
 *  `lead` and `trail` are that row's contents, handed in by whoever owns the
 *  note, and they sit on the ends of this bar.
 *
 *  What does not fit goes into the `…` at the right rather than off the edge:
 *  a bar that scrolled sideways hid its own right-hand end with no sign it had
 *  one. Widths are measured once per item set and re-fitted on resize.
 *
 *  Button state is read with `useEditorState`, which subscribes to the editor's
 *  transactions. Reading `editor.isActive()` during render instead would be a
 *  lie the moment the caret moves without a re-render — the toolbar would show
 *  the formatting of wherever the caret used to be.
 */

const FONTS = [
  { label: "Sans", value: "" },
  { label: "Serif", value: "ui-serif, Georgia, serif" },
  { label: "Mono", value: "var(--font-mono)" },
  { label: "Condensed", value: "var(--font-condensed)" },
];

const SIZES = ["13px", "15px", "17px", "20px", "24px"];

/** The blocks worth a button of their own. The rest of the menu — headings,
 *  lists, a divider — is already on this bar or one key away, so putting it
 *  here twice would only make the bar longer. */
const BLOCKS = [
  { word: "table", label: "Table", icon: <Table size={13} strokeWidth={1.7} /> },
  { word: "visual", label: "Visual", icon: <BarChart3 size={13} strokeWidth={1.7} /> },
  { word: "play", label: "Play", icon: <Spline size={13} strokeWidth={1.7} /> },
  { word: "columns", label: "Columns", icon: <Columns2 size={13} strokeWidth={1.7} /> },
];

/** gap-1 between items, and the room the `…` button needs when there is one. */
const GAP = 4;
const MORE = 32;

/** How many of `widths` fit across `room`. When they do not all fit, the count
 *  is redone against the room left over by the `…` button — otherwise the
 *  button pushes the last item it just decided to keep straight back off. */
export function fitCount(widths: number[], room: number) {
  const upTo = (space: number) => {
    let used = 0;
    let k = 0;
    while (k < widths.length && used + widths[k] <= space) used += widths[k++];
    return k;
  };
  const all = upTo(room);
  return all === widths.length ? all : upTo(room - MORE);
}

/** How many of `count` items fit across the row, the rest belonging in the
 *  overflow menu.
 *
 *  Widths are cached: the buttons are a fixed size, so a resize is answered
 *  from the cache without touching the DOM. They are measured on the render
 *  where everything IS on the row — hence the reset to `count` whenever the
 *  item set changes — and in a LAYOUT effect, so the one wide frame is never
 *  painted. */
function useFit(count: number) {
  const ref = useRef<HTMLDivElement>(null);
  const widths = useRef<number[] | null>(null);
  const measuredAt = useRef(0);
  const [fit, setFit] = useState(count);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const size = () => {
      if (measuredAt.current !== count) widths.current = null;
      if (!widths.current) {
        // Nothing to measure until the row is showing everything, so ask for
        // that pass first. It is a layout effect: the wide frame never paints.
        if (el.children.length < count) return setFit(count);
        widths.current = [...el.children].map((c) => (c as HTMLElement).offsetWidth + GAP);
        measuredAt.current = count;
      }
      setFit(fitCount(widths.current, el.clientWidth));
    };
    size();
    const ro = new ResizeObserver(size);
    ro.observe(el);
    return () => ro.disconnect();
  }, [count, fit]);

  return { ref, fit };
}

/** One control, plus whether a rule goes in front of it. Each is exactly one
 *  child of the row, because the fit above counts children. */
interface Item {
  key: string;
  sep?: boolean;
  node: React.ReactNode;
}

export function Toolbar({
  editor,
  onInsert,
  lead,
  trail,
}: {
  editor: Editor;
  onInsert?: (word?: string) => void;
  /** Pinned to the left of the bar — the way back, and the note's name. */
  lead?: React.ReactNode;
  /** Pinned to the right — print, and the save state. */
  trail?: React.ReactNode;
}) {
  const s = useEditorState({
    editor,
    selector: ({ editor: e }) => ({
      bold: e.isActive("bold"),
      italic: e.isActive("italic"),
      underline: e.isActive("underline"),
      h1: e.isActive("heading", { level: 1 }),
      h2: e.isActive("heading", { level: 2 }),
      quote: e.isActive("blockquote"),
      ul: e.isActive("bulletList"),
      ol: e.isActive("orderedList"),
      code: e.isActive("codeBlock"),
      link: e.isActive("link"),
      font: (e.getAttributes("textStyle").fontFamily as string) ?? "",
      size: (e.getAttributes("textStyle").fontSize as string) ?? "",
      words: e.getText().trim().split(/\s+/).filter(Boolean).length,
      table: e.isActive("table"),
      page: e.isActive("page"),
      // The paper the whole document is on, which lives in the document - see
      // page-nodes.ts. Absent means portrait, which is every note written
      // before pages existed.
      landscape:
        e.state.doc.firstChild?.type.name !== "pageSetup" ||
        e.state.doc.firstChild.attrs.orient !== "portrait",
    }),
  });

  const chain = () => editor.chain().focus();

  const addLink = () => {
    const prev = (editor.getAttributes("link").href as string) ?? "";
    const url = window.prompt("Link to", prev);
    if (url === null) return; // cancelled — leave whatever was there
    if (url === "") return void chain().unsetLink().run();
    chain().setLink({ href: url }).run();
  };

  const items: Item[] = [];

  // Everything the slash menu can put in the document, on the bar the hand is
  // already on. A coach formatting a paragraph should not have to go back to
  // the keyboard and remember a character to get a chart under it — and each
  // of these opens the same panel the typed command does, because each of them
  // types that command.
  if (onInsert) {
    for (const b of BLOCKS)
      items.push({
        key: b.word,
        node: (
          <Btn on={false} label={b.label} onClick={() => onInsert(b.word)}>
            {b.icon}
          </Btn>
        ),
      });
    items.push({
      key: "more-blocks",
      node: (
        <Btn on={false} label="Everything else" onClick={() => onInsert()}>
          <Plus size={13} strokeWidth={2} />
        </Btn>
      ),
    });
  }

  items.push(
    {
      key: "bold",
      sep: true,
      node: (
        <Btn on={s.bold} label="Bold" onClick={() => chain().toggleBold().run()}>
          <span className="font-bold">B</span>
        </Btn>
      ),
    },
    {
      key: "italic",
      node: (
        <Btn on={s.italic} label="Italic" onClick={() => chain().toggleItalic().run()}>
          <span className="italic">I</span>
        </Btn>
      ),
    },
    {
      key: "underline",
      node: (
        <Btn on={s.underline} label="Underline" onClick={() => chain().toggleUnderline().run()}>
          <span className="underline underline-offset-2">U</span>
        </Btn>
      ),
    },
    {
      key: "h1",
      sep: true,
      node: (
        <Btn on={s.h1} label="Heading 1" onClick={() => chain().toggleHeading({ level: 1 }).run()}>
          <span className="text-[11px] font-semibold">H1</span>
        </Btn>
      ),
    },
    {
      key: "h2",
      node: (
        <Btn on={s.h2} label="Heading 2" onClick={() => chain().toggleHeading({ level: 2 }).run()}>
          <span className="text-[11px] font-semibold">H2</span>
        </Btn>
      ),
    },
    {
      key: "quote",
      node: (
        <Btn on={s.quote} label="Quote" onClick={() => chain().toggleBlockquote().run()}>
          <Quote size={13} strokeWidth={1.7} />
        </Btn>
      ),
    },
    {
      key: "ul",
      sep: true,
      node: (
        <Btn on={s.ul} label="Bullet list" onClick={() => chain().toggleBulletList().run()}>
          <List size={13} strokeWidth={1.7} />
        </Btn>
      ),
    },
    {
      key: "ol",
      node: (
        <Btn on={s.ol} label="Numbered list" onClick={() => chain().toggleOrderedList().run()}>
          <ListOrdered size={13} strokeWidth={1.7} />
        </Btn>
      ),
    },
    {
      key: "code",
      node: (
        <Btn on={s.code} label="Code block" onClick={() => chain().toggleCodeBlock().run()}>
          <Code size={13} strokeWidth={1.7} />
        </Btn>
      ),
    },
    {
      key: "link",
      node: (
        <Btn on={s.link} label="Link" onClick={addLink}>
          <Link2 size={13} strokeWidth={1.7} />
        </Btn>
      ),
    },
    // PAPER. Both of these change the printed document rather than the text,
    // and both are stored IN it — see page-nodes.ts for why the orientation
    // cannot be a preference.
    {
      key: "page",
      sep: true,
      node: (
        <Btn
          on={s.page}
          label="New sheet — a box the size of the printed page"
          onClick={() => chain().insertPage().run()}
        >
          <ScissorsLineDashed size={13} strokeWidth={1.7} />
        </Btn>
      ),
    },
    {
      key: "orient",
      node: (
        <Btn
          on={s.landscape}
          label={s.landscape ? "Landscape — click for portrait" : "Portrait — click for landscape"}
          onClick={() => chain().setOrientation(s.landscape ? "portrait" : "landscape").run()}
        >
          {s.landscape ? (
            <RectangleHorizontal size={13} strokeWidth={1.7} />
          ) : (
            <RectangleVertical size={13} strokeWidth={1.7} />
          )}
        </Btn>
      ),
    },
    {
      key: "font",
      sep: true,
      node: (
        <Select
          label="Font"
          value={s.font}
          onChange={(v) => (v ? chain().setFontFamily(v).run() : chain().unsetFontFamily().run())}
          options={FONTS}
        />
      ),
    },
    {
      key: "size",
      node: (
        <Select
          label="Size"
          value={s.size}
          onChange={(v) => (v ? chain().setFontSize(v).run() : chain().unsetFontSize().run())}
          options={[{ label: "Body", value: "" }, ...SIZES.map((v) => ({ label: v.replace("px", ""), value: v }))]}
        />
      ),
    },
  );

  // Only present when the caret is in a table, so the bar does not carry nine
  // table buttons through a document that has none.
  if (editor.isEditable && s.table)
    items.push({ key: "table-tools", node: <TableToolbar editor={editor} /> });

  // Last, so it is the first thing the row gives up when the pane narrows.
  items.push({
    key: "words",
    node: (
      <span className="readout px-1 text-[11px] whitespace-nowrap text-[var(--mute)]">
        {s.words} {s.words === 1 ? "word" : "words"}
      </span>
    ),
  });

  const { ref, fit } = useFit(items.length);
  const [more, setMore] = useState(false);

  return (
    <div className="ft-controls flex shrink-0 items-center gap-1 overflow-hidden border-b border-[var(--hairline)] px-1 py-1">
      {lead}
      <div ref={ref} className="flex min-w-0 flex-1 items-center gap-1 overflow-hidden">
        {items.slice(0, fit).map((i) => (
          <div key={i.key} className="flex shrink-0 items-center gap-1">
            {i.sep && <Sep />}
            {i.node}
          </div>
        ))}
      </div>

      {fit < items.length && (
        <DropdownMenu open={more} onOpenChange={setMore} modal={false}>
          <DropdownMenuTrigger asChild>
            <button
              type="button"
              title="More controls"
              aria-label="More controls"
              onMouseDown={(e) => e.preventDefault()}
              className="grid size-7 shrink-0 place-items-center rounded-[var(--radius-sm)] text-[var(--graphite)] transition-colors hover:bg-[var(--card)] hover:text-[var(--ink)]"
            >
              <MoreHorizontal size={13} strokeWidth={1.7} />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="flex w-auto max-w-64 flex-wrap items-center gap-1">
            {items.slice(fit).map((i) => (
              <Fragment key={i.key}>{i.node}</Fragment>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      )}

      {trail}
    </div>
  );
}

function Sep() {
  return <span aria-hidden className="mx-0.5 h-4 w-px shrink-0 bg-[var(--hairline)]" />;
}

function Btn({
  on,
  label,
  onClick,
  children,
}: {
  on: boolean;
  label: string;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      aria-pressed={on}
      // The editor must not lose the selection the button is about to format.
      onMouseDown={(e) => e.preventDefault()}
      onClick={onClick}
      className={cn(
        "grid size-7 shrink-0 place-items-center rounded-[var(--radius-sm)] text-[12px] transition-colors",
        on
          ? "bg-[var(--card)] text-[var(--ink)]"
          : "text-[var(--graphite)] hover:bg-[var(--card)] hover:text-[var(--ink)]",
      )}
    >
      {children}
    </button>
  );
}

function Select({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { label: string; value: string }[];
}) {
  return (
    <select
      title={label}
      aria-label={label}
      value={value}
      onMouseDown={(e) => e.stopPropagation()}
      onChange={(e) => onChange(e.target.value)}
      className="h-7 shrink-0 cursor-pointer rounded-[var(--radius-sm)] bg-transparent px-1 text-[11.5px] text-[var(--graphite)] outline-none transition-colors hover:bg-[var(--card)] hover:text-[var(--ink)]"
    >
      {options.map((o) => (
        <option key={o.value} value={o.value} className="bg-[var(--panel)] text-[var(--ink)]">
          {o.label}
        </option>
      ))}
    </select>
  );
}
