"use client";

import { useRef, useState } from "react";

import {
  isVertical,
  magnetSvg,
  mx,
  my,
  shapeHitSvg,
  shapeSvg,
  surfaceSvg,
  toPitch,
  viewBox,
  viewOf,
  type Board,
  type Magnet,
  type Shape,
  type ShapeKind,
} from "./pitch-data";

/** The drawing surface. Just the pitch — no palette, no panel, no chrome.
 *
 *  Split out from the block and the editor because the SAME canvas has to work
 *  in three places: sitting still inside a document, filling the editor, and
 *  (through pitchSvg) as markup in the saved file. Anything that decided how a
 *  magnet is drawn twice would eventually draw it two ways.
 *
 *  Pointer events rather than HTML5 drag: a magnet is dragged continuously and
 *  has to follow the finger, and setPointerCapture keeps the gesture when the
 *  pointer leaves the disc — which it always does, because a 1.2 m circle is
 *  smaller than the hand moving it.
 *
 *  Every gesture reports UP exactly once, on pointer up. Reporting per frame
 *  would put a hundred steps in the editor's history for one drag and re-render
 *  the document on every move; the SVG follows the pointer from local state in
 *  the meantime.
 */

export type Tool = "move" | "player" | ShapeKind;

/** Click within this of the first corner to close a polygon. Generous, because
 *  it is the only gesture on the board you have to aim at. */
const CLOSE_D = 2.5;

/** Freehand at pointer-event rate is thousands of samples for one stroke. Only
 *  keep a point once the pen has actually gone somewhere — 0.4% of the pitch,
 *  which is under a metre and well below what anyone can see. */
const PEN_MIN = 0.4;

/** A tap with an arrow tool is a mis-click, not a zero-length pass. */
const MIN_DRAG = 1.5;

export function BoardCanvas({
  board,
  editable = false,
  tool = "move",
  color = "white",
  side = "home",
  gear = "cone",
  picked = null,
  pickedShape = null,
  onPick,
  onPickShape,
  onChange,
  onNewText,
  className,
  boldness = 1,
  animate = false,
}: {
  board: Board;
  editable?: boolean;
  tool?: Tool;
  color?: string;
  side?: Magnet["side"];
  /** Which piece of equipment the placement tool drops, when `side` is
   *  "gear". A key of GEAR. */
  gear?: string;
  picked?: number | null;
  pickedShape?: number | null;
  onPick?: (i: number | null) => void;
  onPickShape?: (i: number | null) => void;
  onChange?: (patch: Partial<Board>) => void;
  onNewText?: (index: number) => void;
  className?: string;
  /** Line-width multiplier, for the style swatches. See surfaceSvg. */
  boldness?: number;
  /** Reveal the marks one at a time instead of all at once — a board walked
   *  through rather than presented. The discs are the starting picture and
   *  stay put; it is the arrows and zones that are the plan. Editor only: a
   *  saved board is a still, and so is the PDF it prints into. */
  animate?: boolean;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [live, setLive] = useState<Magnet[] | null>(null);
  const [draft, setDraft] = useState<Shape | null>(null);
  const dragging = useRef<{ index: number; moved: boolean } | null>(null);

  const { magnets, shapes } = board;
  const v = viewOf(board.view);
  const vert = isVertical(board.orient);
  const { box, aspect } = viewBox(v, vert);
  const shown = live ?? magnets;
  const set = (patch: Partial<Board>) => onChange?.(patch);

  /** Where a pointer is, in stored percent. Read off the DRAWN box rather than
   *  the element, because `meet` letterboxes: the SVG element can be wider than
   *  the pitch inside it, and measuring against the element's own rect puts
   *  every point metres out at exactly the crops where precision matters. */
  const at = (e: React.PointerEvent) => {
    const svg = svgRef.current;
    if (!svg) return null;
    const r = svg.getBoundingClientRect();
    const w = Math.min(r.width, r.height * aspect);
    const h = w / aspect;
    // A board that is not laid out yet has a zero-size rect, and dividing by it
    // gives NaN — which `clamp` cannot rescue and `parseMagnets` cannot reject,
    // because `typeof NaN === "number"`. A NaN coordinate saved into a document
    // is a disc that never draws again.
    if (!(w > 0) || !(h > 0)) return null;
    return toPitch(v, vert, (e.clientX - r.left - (r.width - w) / 2) / w, (e.clientY - r.top - (r.height - h) / 2) / h);
  };

  const capture = (e: React.PointerEvent) => {
    // Capture is what keeps a drag alive once the pointer leaves the 1.2 m disc
    // it started on. It is also allowed to fail: a pointer id that is not
    // currently down throws NotFoundError, and the gesture still works without
    // capture, so a throw here must not take the whole stroke with it.
    try {
      (e.currentTarget as Element).setPointerCapture(e.pointerId);
    } catch {
      /* no capture; move events still arrive by bubbling */
    }
    e.preventDefault();
    e.stopPropagation();
  };

  // --- gestures that start on the pitch itself ------------------------------

  const onPitchDown = (e: React.PointerEvent) => {
    if (!editable) return;
    const p = at(e);
    if (!p) return;
    if (tool === "move") {
      onPick?.(null);
      onPickShape?.(null);
      return;
    }
    // A POLYGON IS CLICKED, NOT DRAGGED, which is the one gesture on this board
    // that spans several pointer-downs: each click sets a corner, the last pair
    // of `pts` trails the cursor between them, and clicking the first corner
    // again closes it. No capture, because the gesture has to survive the
    // pointer going up.
    if (tool === "poly") {
      e.preventDefault();
      e.stopPropagation();
      if (!draft || draft.kind !== "poly") {
        setDraft({ kind: "poly", pts: [p.x, p.y, p.x, p.y], color });
        return;
      }
      const closes =
        draft.pts.length >= 8 && Math.hypot(p.x - draft.pts[0], p.y - draft.pts[1]) < CLOSE_D;
      if (closes) return closePoly();
      setDraft({ ...draft, pts: [...draft.pts, p.x, p.y] });
      return;
    }
    capture(e);
    if (tool === "player") {
      set({
        magnets: [...magnets, { ...p, label: "", side, ...(side === "gear" ? { gear } : {}) }],
      });
      // Equipment has no number and no side to change, so there is nothing for
      // the strip to offer and selecting it would only put a halo on a cone.
      if (side !== "gear") onPick?.(magnets.length);
      return;
    }
    if (tool === "text") {
      set({ shapes: [...shapes, { kind: "text", pts: [p.x, p.y], color, label: "" }] });
      onNewText?.(shapes.length);
      return;
    }
    setDraft({ kind: tool, pts: [p.x, p.y, p.x, p.y], color });
  };

  /** Drop the trailing cursor point and keep the corners. */
  const closePoly = () => {
    if (!draft || draft.kind !== "poly" || draft.pts.length < 8) return;
    set({ shapes: [...shapes, { ...draft, pts: draft.pts.slice(0, -2) }] });
    setDraft(null);
  };

  const onPitchMove = (e: React.PointerEvent) => {
    if (!draft) return;
    const p = at(e);
    if (!p) return;
    // A polygon's last pair is the cursor, not a corner — it follows the
    // pointer with no button down, which is what makes the next edge visible
    // before you commit to it.
    if (draft.kind === "poly") {
      setDraft({ ...draft, pts: [...draft.pts.slice(0, -2), p.x, p.y] });
      return;
    }
    if (draft.kind === "pen") {
      const n = draft.pts.length;
      if (Math.hypot(p.x - draft.pts[n - 2], p.y - draft.pts[n - 1]) > PEN_MIN) {
        setDraft({ ...draft, pts: [...draft.pts, p.x, p.y] });
      }
      return;
    }
    setDraft({ ...draft, pts: [draft.pts[0], draft.pts[1], p.x, p.y] });
  };

  const onPitchUp = () => {
    // A polygon outlives its pointer-ups; it ends when it is closed.
    if (!draft || draft.kind === "poly") return;
    setDraft(null);
    const [x0, y0] = draft.pts;
    const [x1, y1] = draft.pts.slice(-2);
    if (draft.kind === "pen" ? draft.pts.length < 6 : Math.hypot(x1 - x0, y1 - y0) < MIN_DRAG) return;
    set({ shapes: [...shapes, draft] });
  };

  // --- gestures that start on a disc ----------------------------------------

  const onMagnetDown = (i: number) => (e: React.PointerEvent) => {
    if (!editable) return;
    if (tool !== "move") return; // drawing over a disc must draw, not drag it
    capture(e);
    dragging.current = { index: i, moved: false };
    onPick?.(i);
    onPickShape?.(null);
    setLive(magnets);
  };

  const onMagnetMove = (i: number) => (e: React.PointerEvent) => {
    if (!live || dragging.current?.index !== i) return;
    const p = at(e);
    if (!p) return;
    dragging.current.moved = true;
    setLive(live.map((m, j) => (j === i ? { ...m, ...p } : m)));
  };

  const onMagnetUp = (i: number) => (e: React.PointerEvent) => {
    if (!live) return;
    const p = at(e);
    const moved = dragging.current?.moved;
    const next = p && moved ? live.map((m, j) => (j === i ? { ...m, ...p } : m)) : live;
    dragging.current = null;
    setLive(null);
    if (moved) set({ magnets: next });
  };

  const onShapeDown = (i: number) => (e: React.PointerEvent) => {
    if (!editable) return;
    if (tool !== "move") return;
    e.stopPropagation();
    onPickShape?.(i);
    onPick?.(null);
  };

  const cursor = !editable ? "default" : tool === "move" ? "default" : "crosshair";

  /** A half-drawn polygon belongs to the tool that started it. Switching tools
   *  is how you abandon one — otherwise a poly left open stays on the pitch
   *  while you drag discs around, and the next click with the arrow tool adds a
   *  corner to something you had forgotten about.
   *
   *  Adjusted during render rather than in an effect: an effect would paint the
   *  stale draft once before clearing it, and it is React's own answer for
   *  state that has to reset when a prop changes. */
  const [was, setWas] = useState(tool);
  if (was !== tool) {
    setWas(tool);
    setDraft(null);
  }

  const mark = (s: Shape, i: number) => (
    <g
      key={`s${i}`}
      onPointerDown={onShapeDown(i)}
      className={animate ? "ft-mark-in" : undefined}
      // The index IS the running order: marks are drawn in the order they were
      // made, which is the order they get explained in.
      style={animate ? ({ "--i": i } as React.CSSProperties) : undefined}
    >
      <g dangerouslySetInnerHTML={{ __html: shapeSvg(s, vert) }} pointerEvents="none" />
      {editable && <g dangerouslySetInnerHTML={{ __html: shapeHitSvg(s) }} />}
      {pickedShape === i && <Halo s={s} />}
    </g>
  );

  const content = (
    <>
      {/* surfaceSvg builds from fixed tables keyed by strings the parser has
          already narrowed, and it is shared with the exporter so the live board
          and the saved file cannot draw different pitches. */}
      <g dangerouslySetInnerHTML={{ __html: surfaceSvg(board, boldness) }} pointerEvents="none" />

      {shapes.map((s, i) => (s.kind === "zone" ? mark(s, i) : null))}

      {/* Arrows under the discs, like the exporter: a line across a
          player's number is unreadable, and a pointer over a disc that
          happens to sit on an arrow should grab the disc. */}
      {shapes.map((s, i) => (s.kind === "zone" ? null : mark(s, i)))}

      {shown.map((m, i) => (
        <g
          key={`m${i}`}
          onPointerDown={onMagnetDown(i)}
          onPointerMove={onMagnetMove(i)}
          onPointerUp={onMagnetUp(i)}
          onPointerCancel={onMagnetUp(i)}
          style={{ cursor: editable && tool === "move" ? "grab" : cursor }}
        >
          <g dangerouslySetInnerHTML={{ __html: magnetSvg(m, vert, board.kits) }} pointerEvents="none" />
          {picked === i && (
            <circle cx={mx(m.x)} cy={my(m.y)} r={1.9} fill="none" stroke="var(--accent)" strokeWidth={0.32} />
          )}
          {/* A fat invisible target: a 1.2 m disc is a few millimetres on a
              laptop and unhittable on a touchscreen. */}
          {editable && <circle cx={mx(m.x)} cy={my(m.y)} r={2.4} fill="transparent" />}
        </g>
      ))}

      {draft && (
        <g dangerouslySetInnerHTML={{ __html: shapeSvg(draft, vert) }} pointerEvents="none" opacity={0.9} />
      )}
    </>
  );

  return (
    <svg
      ref={svgRef}
      viewBox={box}
      preserveAspectRatio="xMidYMid meet"
      onPointerDown={onPitchDown}
      onPointerMove={onPitchMove}
      onPointerUp={onPitchUp}
      onPointerCancel={onPitchUp}
      // The other way to close a polygon, for anyone who does not fancy hitting
      // the first corner again.
      onDoubleClick={closePoly}
      className={className ?? "block w-full touch-none select-none"}
      // Square. A pitch is a rectangle with corner flags on it; a rounded one
      // reads as a card with a picture of grass on it. Same in the serialized
      // board (pitch-data.ts) so the live and the saved one agree.
      style={{ aspectRatio: aspect, cursor }}
    >
      {vert ? <g transform="rotate(-90 52.5 34)">{content}</g> : content}
    </svg>
  );
}

/** What "this one is selected" looks like for a mark rather than a disc. */
function Halo({ s }: { s: Shape }) {
  const p = s.pts.map((v, i) => (i % 2 === 0 ? mx(v) : my(v)));
  if (s.kind === "zone") {
    return (
      <rect
        x={Math.min(p[0], p[2])}
        y={Math.min(p[1], p[3])}
        width={Math.abs(p[2] - p[0])}
        height={Math.abs(p[3] - p[1])}
        fill="none"
        stroke="var(--accent)"
        strokeWidth={0.3}
        pointerEvents="none"
      />
    );
  }
  return (
    <circle cx={p[0]} cy={p[1]} r={1.4} fill="none" stroke="var(--accent)" strokeWidth={0.3} pointerEvents="none" />
  );
}
