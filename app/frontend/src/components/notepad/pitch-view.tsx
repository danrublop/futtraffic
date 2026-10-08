"use client";

import { useRef, useState } from "react";
import { NodeViewWrapper, type NodeViewProps } from "@tiptap/react";
import { SlidersHorizontal } from "lucide-react";

import { BoardCanvas } from "./board-canvas";
import { BoardEditor } from "./board-editor";
import type { Board } from "./pitch-data";

/** The tactics board, as it appears IN the note.
 *
 *  It sits still. Nothing on it is draggable here and there is no palette
 *  underneath it — a board in a document is a figure, the same way a photograph
 *  is, and a coach scrolling their notes should not be able to nudge a marker
 *  out of place with a stray swipe.
 *
 *  Editing is a deliberate act: click Edit, the board opens over the whole
 *  window (board-editor.tsx), and Save puts it back down. That round trip is
 *  also what keeps the note's undo history sane — one step per board, not one
 *  per pen stroke.
 */
/** The four corners of a bounding box. Square, because a bounding box is a
 *  measuring frame and a round handle reads as a button. `dir` is which way the
 *  pointer travels to make the figure bigger — a block sits at the left of its
 *  column and cannot move, so a left-hand drag moves the RIGHT edge.
 *
 *  INSIDE the box, sitting on the frame, not floating outside it.
 *
 *  They were at -4px, and the frame they belong to is drawn 3px INSIDE the
 *  border box — so handle and frame were seven pixels apart and did not read as
 *  one object. Worse, anything outside the box overlaps whatever is next to it:
 *  two figures side by side in a layout row had the left one's handles sitting
 *  in the right one's column. And `.ft-grid-wrap` scrolls, so an outside handle
 *  is clippable by exactly the container that was already clipping the outline.
 *
 *  Flush to the frame's own corners, then. Nothing leaves the figure's box, so
 *  nothing overlaps a neighbour and nothing can be clipped. */
/** THE ONE CONTROL EVERY FIGURE WEARS.
 *
 *  A board had a pill reading "Edit" behind a pencil, a chart had a different
 *  pill reading "Edit" or "Done" behind a slider, in a different corner, at a
 *  different size. Two blocks that behave identically — hover, click, a panel
 *  opens — advertised themselves as two different kinds of thing.
 *
 *  So: one horizontal-sliders glyph, same place, same size, on every figure.
 *  It reads as "there are settings behind this", it needs no word, and a word
 *  was the part that differed. What the panel contains is the block's
 *  business; getting to it is not.
 */
export const FIGURE_BTN =
  "ft-controls absolute top-1.5 right-1.5 flex h-6 w-6 items-center justify-center " +
  "rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-[var(--panel)]/95 " +
  "text-[var(--ink)] opacity-0 shadow-[var(--shadow-md)] transition-opacity " +
  "hover:bg-[var(--card)] group-hover:opacity-100 focus-visible:opacity-100";

export const CORNERS = [
  { at: "top-[3px] left-[3px]", cur: "nwse-resize", dir: -1 as const, name: "top left" },
  { at: "top-[3px] right-[3px]", cur: "nesw-resize", dir: 1 as const, name: "top right" },
  { at: "bottom-[3px] left-[3px]", cur: "nesw-resize", dir: -1 as const, name: "bottom left" },
  { at: "bottom-[3px] right-[3px]", cur: "nwse-resize", dir: 1 as const, name: "bottom right" },
];

export function PitchView({ node, updateAttributes, editor, selected }: NodeViewProps) {
  const board = node.attrs as unknown as Board;
  const caption = String(node.attrs.caption ?? "");
  /** Closed, or open with the pane it fills. The host is read off the DOM when
   *  the button is clicked, not during render — the same lookup done in the
   *  render body reads a ref React has not committed yet. */
  const [open, setOpen] = useState<{ host: HTMLElement | null } | null>(null);
  const editable = editor.isEditable;
  const box = useRef<HTMLDivElement>(null);
  const width = Number(node.attrs.width) || 100;

  /** Drag a corner. Share of the column, in fives, never pixels — see the
   *  `width` attribute in pitch-node.ts for why. A board's height follows its
   *  own 1.5:1 pitch, so width is the only free variable and a corner scales
   *  the whole board; a height handle would only offer to distort it. */
  const startResize = (e: React.PointerEvent, dir: 1 | -1) => {
    e.preventDefault();
    e.stopPropagation();
    const host = box.current?.parentElement;
    if (!host || !box.current) return;
    const full = host.getBoundingClientRect().width;
    const startX = e.clientX;
    const startW = box.current.getBoundingClientRect().width;
    const move = (ev: PointerEvent) => {
      const px = startW + dir * (ev.clientX - startX);
      const pct = Math.round(((px / full) * 100) / 5) * 5;
      updateAttributes({ width: Math.max(25, Math.min(100, pct)) });
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  return (
    <NodeViewWrapper className="ft-pitch-wrap" data-drag-handle>
      <div
        ref={box}
        className="group relative"
        data-selected={selected ? "1" : undefined}
        style={width < 100 ? { width: `${width}%` } : undefined}
      >
        <BoardCanvas board={board} />

        {editable && (
          <button
            type="button"
            onClick={() =>
              setOpen({ host: box.current?.closest<HTMLElement>(".ft-note-pane") ?? null })
            }
            title="Edit the board"
            aria-label="Edit the board"
            // Only on hover: the point of the block is the board, and a button
            // parked on top of it permanently is a button in every printout
            // anybody looks at over a coach's shoulder.
            className={FIGURE_BTN}
          >
            <SlidersHorizontal size={13} strokeWidth={1.9} />
          </button>
        )}

        {/* The bounding box's handles. Shown when the board is selected, the
            way a picture's are in a document editor, and on hover so the board
            advertises that it can be resized at all. `ft-controls` is stripped
            on export, so none of this reaches paper. */}
        {editable &&
          CORNERS.map((c) => (
            <span
              key={c.name}
              onPointerDown={(e) => startResize(e, c.dir)}
              role="slider"
              aria-label={`Resize from the ${c.name}`}
              aria-valuenow={width}
              aria-valuemin={25}
              aria-valuemax={100}
              title={`${width}% — drag to resize`}
              style={{ cursor: c.cur }}
              className={`ft-controls absolute ${c.at} h-2 w-2 border border-[var(--accent)] bg-[var(--panel)] opacity-0 transition-opacity group-hover:opacity-100 ${
                selected ? "!opacity-100" : ""
              }`}
            />
          ))}
      </div>

      {caption && <figcaption className="ft-pitch-cap">{caption}</figcaption>}

      {open && (
        <BoardEditor
          board={board}
          caption={caption}
          /* Fill the note's pane, not the window. Found from the block rather
             than passed down through Notepad and the NodeView bridge, which
             TipTap renders outside React's tree anyway. */
          host={open.host}
          onCancel={() => setOpen(null)}
          onSave={(next, cap) => {
            // One transaction for the whole session, which is what makes ⌘Z in
            // the note take back "that board" rather than "that line".
            updateAttributes({ ...next, caption: cap });
            setOpen(null);
          }}
        />
      )}
    </NodeViewWrapper>
  );
}
