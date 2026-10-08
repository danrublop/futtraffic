"use client";

import { useEffect, useRef, useState } from "react";
import { NodeViewWrapper, type NodeViewProps } from "@tiptap/react";
import { SlidersHorizontal } from "lucide-react";

import { useReport, useSeason } from "@/hooks/use-matches";
import type { Season } from "@/lib/types";
import { tierNote } from "./stat-node";
import { chartSvg } from "./chart-svg";
import { ChartMenu } from "./chart-menu";
import { CORNERS, FIGURE_BTN } from "./pitch-view";
import { storedResult, type QueryAttrs } from "./query-node";
import { runQuery, sourceOf } from "./query-data";

/** A chart, as it appears IN the note.
 *
 *  It sits still, the way the board does. There used to be a strip of five
 *  selects and six chips under every one of these, on screen the whole time —
 *  a control panel bolted to a figure, taller than the small charts it sat
 *  under, and the loudest thing in a document otherwise made of prose. All of
 *  it moved into chart-editor.tsx, behind one button that appears on hover.
 *
 *  What IS direct here is the two things you reach for constantly: the title,
 *  which you click and type into, and the width, which you drag. Everything
 *  else is a deliberate act.
 */
export function QueryView({ node, updateAttributes, editor, selected }: NodeViewProps) {
  const a = node.attrs as unknown as QueryAttrs;
  const editable = editor.isEditable;
  const { match, report } = useReport();
  const src = sourceOf(a.source, report);
  const { season, reload } = useSeason(match?.team, src.scope === "season");
  /** Null when closed, otherwise where the menu hangs from — the pointer
   *  for a right-click, the button for a click on Edit. */
  const [open, setOpen] = useState<{ x: number; y: number } | null>(null);
  const [editingTitle, setEditingTitle] = useState(false);
  const box = useRef<HTMLDivElement>(null);

  /** The season arrives after the source is switched to it, so the rows frozen
   *  by that click are empty. Fill once, when the data lands, and never touch a
   *  chart that already has an answer — or a chart somebody typed by hand. */
  const applied = useRef<Season | null>(null);
  useEffect(() => {
    if (src.scope !== "season" || !season || applied.current === season) return;
    applied.current = season;
    if (a.rows?.length || a.custom) return;
    const r = runQuery({
      report, season, source: a.source, dim: a.dim, measure: a.measure,
      split: a.split, limit: a.limit, team: a.team, render: a.render,
    });
    updateAttributes({
      rows: r.rows, series: r.series, max: r.max, note: r.refusal ?? r.note,
      tier: r.tier, extra: r.extra ?? null, ordered: r.ordered,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src.scope, season, a.rows?.length]);

  /** Drag a corner. Width is a PERCENTAGE of the column, not pixels: a note is
   *  read at whatever width the pane happens to be and printed at another one,
   *  so a chart pinned to 620px overflows one of them.
   *
   *  ONE AXIS, FOUR HANDLES, and that is deliberate rather than unfinished. A
   *  figure's height comes from its viewBox — the box is derived from the
   *  content now, see chart-core — so width is the only free variable and
   *  dragging a corner scales the whole thing. Offering a height handle would
   *  be offering to distort it.
   *
   *  `dir` is which way the pointer has to travel to make it bigger: the two
   *  right-hand corners grow rightward, the two left-hand ones grow leftward.
   *  A block sits at the left of its column and cannot move, so a left-hand
   *  drag moves the RIGHT edge — the arithmetic is anchored on the current
   *  width rather than on an edge position, which makes both directions the
   *  same line of code and neither of them jump on the first pixel.
   */
  const startResize = (e: React.PointerEvent, dir: 1 | -1) => {
    e.preventDefault();
    e.stopPropagation();
    const host = box.current?.parentElement;
    if (!host || !box.current) return;
    const full = host.getBoundingClientRect().width;
    const startX = e.clientX;
    const startW = box.current.getBoundingClientRect().width;
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    const move = (ev: PointerEvent) => {
      const px = startW + dir * (ev.clientX - startX);
      // Five per cent steps, so two figures dragged to "about half" land on
      // the same half and a row of them lines up.
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


  const shown = a.rows?.length ? storedResult(a) : null;
  const width = Number(a.width) || 100;

  return (
    <NodeViewWrapper className="ft-chart-wrap">
      <div
        ref={box}
        className="ft-chart group relative"
        style={width < 100 ? { width: `${width}%` } : undefined}
        data-selected={selected ? "1" : undefined}
        /* The look, mirrored onto the live element. This view builds its own
           box and never sees `renderHTML`, so a theme set through the menu
           would otherwise reach the saved file and not the screen — the same
           trap the table view already documents. */
        data-theme={a.theme || undefined}
        /* Right-click opens the editor, the same gesture that opens a table's
           menu. The hover button stays: a chart has no other affordance, and a
           control you can only find by guessing at a mouse button is a control
           most people never find. */
        onContextMenu={(e) => {
          if (!editable) return;
          e.preventDefault();
          setOpen({ x: e.clientX, y: e.clientY });
        }}
      >
        {editingTitle && editable ? (
          <input
            autoFocus
            value={a.title}
            aria-label="Title"
            onChange={(e) => updateAttributes({ title: e.target.value.slice(0, 120) })}
            onBlur={() => setEditingTitle(false)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === "Escape") setEditingTitle(false);
            }}
            className="ft-chart__title w-full rounded-[var(--radius-xs)] border border-[var(--accent)] bg-transparent px-1 outline-none"
          />
        ) : (
          a.title && (
            <h4
              className="ft-chart__title"
              // A title you have to open a dialog to rename is a title nobody
              // renames. Click it.
              onClick={() => editable && setEditingTitle(true)}
              title={editable ? "Click to rename" : undefined}
              style={editable ? { cursor: "text" } : undefined}
            >
              {a.title}
            </h4>
          )
        )}

        {shown ? (
          <div dangerouslySetInnerHTML={{ __html: chartSvg(shown, a.render) }} />
        ) : (
          <p className="ft-chart__empty">{a.note || "Nothing to show yet."}</p>
        )}

        {shown && (a.note || tierNote(a.tier)) && (
          <p className="ft-chart__note">{[tierNote(a.tier), a.note].filter(Boolean).join(" · ")}</p>
        )}

        {editable && (
          <>
            {/* The same control the board wears, in the same corner, at the
                same size — see FIGURE_BTN. It used to sit ABOVE the box to keep
                clear of the title, which a square button in the corner no
                longer needs: it is a third the width of the pill it replaces
                and the titles it used to cover now clear it. The width, which
                the old pill printed beside itself, is in the corner handles'
                tooltip where the thing that changes it lives. */}
            <button
              type="button"
              onClick={(e) => {
                // The same menu the right button opens, hung off this button
                // rather than the pointer. One implementation, two ways in.
                const r = e.currentTarget.getBoundingClientRect();
                setOpen((v) => (v ? null : { x: r.left, y: r.bottom + 4 }));
              }}
              title={open ? "Close" : "Change the chart, the question or the numbers"}
              aria-label="Edit the chart"
              aria-expanded={!!open}
              className={FIGURE_BTN}
            >
              <SlidersHorizontal size={13} strokeWidth={1.9} />
            </button>

            {/* THE FOUR CORNERS. Visible when the figure is selected, the way
                a picture's handles are in a document editor, and on hover so
                the figure advertises that it can be resized at all. `ft-controls`
                is stripped on export, so none of this reaches paper. */}
            {CORNERS.map((c) => (
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
          </>
        )}
      </div>

      {/* A menu, not a dialog: the chart stays where it was and redraws under
          the panel as the choices change. */}
      {open && editable && (
        <ChartMenu
          attrs={a}
          report={report}
          season={season}
          onChange={updateAttributes}
          onReload={reload}
          onClose={() => setOpen(null)}
          at={open}
        />
      )}
    </NodeViewWrapper>
  );
}
