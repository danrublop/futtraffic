"use client";

import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  Activity,
  ArrowRight,
  ChevronRight,
  Circle,
  Download,
  Layers,
  MousePointer2,
  MoveUpRight,
  PenLine,
  RotateCw,
  Square,
  TrafficCone,
  Trash2,
  Type,
  Undo2,
  Volleyball,
  Waypoints,
} from "lucide-react";

import { MatchPicker } from "@/components/match-picker";
import { useReport } from "@/hooks/use-matches";
import { syntheticNote } from "@/lib/report";
import { cn } from "@/lib/utils";
import { BoardCanvas, type Tool } from "./board-canvas";
import {
  EMPTY_BOARD,
  FORMATIONS,
  GEAR,
  GRASS,
  INK,
  KIT_COLOURS,
  LINE_TEMPLATES,
  MAGNET_FILL,
  AVG_POSITION_NOTE,
  TERRITORY_NOTE,
  VIEWS,
  fromAveragePositions,
  discDefs,
  isVertical,
  magnetSvg,
  pitchSvg,
  sideFormation,
  territoryShapes,
  viewBox,
  viewOf,
  type Board,
  type Kit,
  type Magnet,
  type ShapeKind,
} from "./pitch-data";

/** The board, open.
 *
 *  A separate surface from the note on purpose. A tactics board is not a figure
 *  you nudge in a margin — it is a thing you stand in front of — so in the
 *  document it sits still and this is what opens when you edit it.
 *
 *  THE PITCH IS THE SURFACE. There was a header across the top, a bordered rail
 *  down the left, a bordered strip along the bottom and a 224px column of
 *  settings down the right, and between them they took a third of the pane away
 *  from the only thing anybody opens this to look at. The pitch runs to every
 *  edge now and the controls are islands floating on it — the shape every board
 *  app a coach already uses has, and not a coincidence: a board is a thing read
 *  from across a room.
 *
 *  The draft is LOCAL until Save. That is the difference between this and a
 *  live-bound block: Cancel has to actually mean cancel, and the document has
 *  to gain exactly one history step per editing session rather than one per
 *  brush stroke — so ⌘Z in the note takes back "that board", not "that line".
 *  ⌘Z INSIDE the editor is a separate, board-local stack.
 */

/** THE RAIL, and it is the whole vocabulary of a board in eleven buttons.
 *
 *  Grouped, because a flat column of identical squares has to be read icon by
 *  icon: the pointer on its own, then the two sides you drag players out of,
 *  then the things that are not players — equipment and the ball — then the six
 *  ways of drawing on grass.
 *
 *  `opts` is the chevron, and the chevron is a promise: there is a choice
 *  behind this one, and picking the tool opens it beside the row. That is where
 *  the ink colours went — five permanent dots at the foot of the rail, on show
 *  while you dragged discs and irrelevant to most of the tools — and where the
 *  formations went, from a settings panel on the far side of the pane to the
 *  shirt they fill the pitch with.
 *
 *  NO ERASER, deliberately. It was a mode you had to enter and leave to delete
 *  one thing, next to a Remove that is one click on the thing itself. A mode
 *  whose whole job is destructive is also the one you least want to be in by
 *  accident.
 */
const TOOLS: {
  id: string;
  icon: typeof ArrowRight;
  label: string;
  hint: string;
  group: number;
  /** The tool this selects. Absent on the arrow, which selects whichever line
   *  style is current — see `line`. */
  tool?: Tool;
  /** What the placement tool puts down. */
  side?: Magnet["side"];
  /** The icon's colour, where the button stands in for a thing that has one. A
   *  black away disc on a black rail is not a button. */
  tint?: string;
  opts?: "gear" | "line" | "ink" | "kit";
}[] = [
  { id: "move", icon: MousePointer2, label: "Move", hint: "Drag the discs and marks", group: 0, tool: "move" },

  { id: "home", icon: Circle, label: "Home", hint: "Click to place one, or open its kit", group: 1, tool: "player", side: "home", opts: "kit" },
  { id: "away", icon: Circle, label: "Away", hint: "Click to place one, or open its kit", group: 1, tool: "player", side: "away", opts: "kit" },

  { id: "gear", icon: TrafficCone, label: "Equipment", hint: "Cones, poles, a mini goal", group: 2, tool: "player", side: "gear", tint: MAGNET_FILL.gear, opts: "gear" },
  { id: "ball", icon: Volleyball, label: "Ball", hint: "Click the pitch to place the ball", group: 2, tool: "player", side: "ball", tint: MAGNET_FILL.ball },

  { id: "pen", icon: PenLine, label: "Marker", hint: "Freehand", group: 3, tool: "pen", opts: "ink" },
  { id: "arrow", icon: MoveUpRight, label: "Arrow", hint: "Drag from one place to another", group: 3, opts: "line" },
  { id: "circle", icon: Circle, label: "Circle", hint: "Drag out an area", group: 3, tool: "circle", opts: "ink" },
  { id: "square", icon: Square, label: "Square", hint: "Box off an area", group: 3, tool: "zone", opts: "ink" },
  { id: "poly", icon: Waypoints, label: "Area", hint: "Click each corner, then the first again", group: 3, tool: "poly", opts: "ink" },
  { id: "text", icon: Type, label: "Label", hint: "Click the pitch to write on it", group: 3, tool: "text" },
];

/** The three lines every coaching diagram in the world already uses, behind one
 *  arrow button. They were three separate tools in the rail, which is three
 *  slots spent on what is one decision. */
const LINES: { id: ShapeKind; label: string; hint: string }[] = [
  { id: "pass", label: "Pass", hint: "Solid — the ball moves" },
  { id: "run", label: "Run", hint: "Dashed — the player moves" },
  { id: "dribble", label: "Carry", hint: "Wavy — the player takes it" },
];

/** The settings, behind three buttons instead of parked in a column. Grass, crop
 *  and orientation get touched once a board; the tools get touched every few
 *  seconds, and the two had the same amount of the pane. */
const PANELS: { id: string; icon: typeof Layers; label: string }[] = [
  { id: "look", icon: Layers, label: "Pitch" },
  { id: "match", icon: Activity, label: "From the match" },
  { id: "clear", icon: Trash2, label: "Clear" },
];

/** DARK, WHATEVER THE THEME IS. The rails sit on grass, not on the app's
 *  surface, and grass is dark in five of the seven palettes and light in two —
 *  a rail painted in theme tokens disappears into half of them. Dark with a
 *  light hairline reads on all seven. */
const RAIL = "rounded-[18px] bg-black/60 shadow-lg ring-1 ring-white/12 backdrop-blur-sm";
const RAIL_BTN =
  "relative grid size-9 shrink-0 place-items-center rounded-[10px] text-white/70 " +
  "transition-colors hover:bg-white/10 hover:text-white";
/** The selected tool is a filled tile, not a tinted one. It is the single piece
 *  of state you check without looking, from across a room. */
const RAIL_ON = "bg-[var(--accent)] text-[var(--on-accent,#fff)] hover:bg-[var(--accent)]";
const RAIL_RULE = "my-1 h-px w-6 shrink-0 bg-white/12";

/** A panel, as opposed to a rail: it holds the app's own chips, swatches and
 *  inputs, so it keeps the app's own surface. A dark card would mean restyling
 *  every control inside it, for a panel that is open two seconds at a time. */
const FLOAT =
  "rounded-[var(--radius-md)] border border-[var(--hairline)] bg-[var(--panel)] shadow-[var(--shadow-md)]";

export function BoardEditor({
  board,
  caption,
  onSave,
  onCancel,
  host,
}: {
  board: Board;
  caption: string;
  onSave: (board: Board, caption: string) => void;
  onCancel: () => void;
  /** The element the editor fills. The note's pane, normally — see
   *  `.ft-note-pane` — so the app's header and rail stay on screen while a
   *  board is being drawn. Null or missing falls back to the whole window. */
  host?: HTMLElement | null;
}) {
  const [draft, setDraft] = useState<Board>(board);
  const [cap, setCap] = useState(caption);
  const [tool, setTool] = useState<Tool>("move");
  const [color, setColor] = useState("white");
  const [side, setSide] = useState<Magnet["side"]>("home");
  const [picked, setPicked] = useState<number | null>(null);
  const [pickedShape, setPickedShape] = useState<number | null>(null);
  /** Which line the arrow button draws, and which piece of equipment the
   *  equipment button drops. Both live here rather than in the rail because
   *  they outlive a click on it: pick "Run" once and every arrow after it is a
   *  run until you say otherwise. */
  const [line, setLine] = useState<ShapeKind>("pass");
  const [gear, setGear] = useState("cone");
  /** Which settings panel is open, if any. One at a time: they float over the
   *  pitch, and two would cover the half of it you were drawing on. */
  const [panel, setPanel] = useState<string | null>(null);
  /** Which tool's chevron is open, and how far down the surface its row sits.
   *  Measured on the click rather than derived from the index: the rail scrolls
   *  when the pane is shorter than nine tools, and an offset computed from the
   *  list would then point at the wrong row. */
  const [flyout, setFlyout] = useState<{ id: string; top: number } | null>(null);
  const surface = useRef<HTMLDivElement>(null);
  /** Walk through the marks instead of showing the finished board. `replay`
   *  only exists to restart it: a CSS animation that has finished stays
   *  finished, so the canvas is remounted to play it again. */
  const [animated, setAnimated] = useState(false);
  const [replay, setReplay] = useState(0);

  /** Board-local undo. The document's own history only ever sees the Save, so
   *  without this a coach who drew one line too many would have to leave and
   *  come back. State rather than a ref because the button is disabled off it;
   *  bounded at 50 because an afternoon of strokes is not worth a megabyte. */
  const [past, setPast] = useState<Board[]>([]);
  const change = (patch: Partial<Board>) => {
    setPast((p) => [...p.slice(-49), draft]);
    setDraft({ ...draft, ...patch });
  };
  const undo = () => {
    // Read the stack, then set both — never setDraft inside setPast's updater.
    // React may run an updater twice (StrictMode, and any future re-render
    // pass), and an updater that also writes other state is a side effect that
    // runs twice with it.
    if (!past.length) return;
    setDraft(past[past.length - 1]);
    setPast(past.slice(0, -1));
    setPicked(null);
    setPickedShape(null);
  };

  const { id: matchId, report } = useReport();
  const measured = report?.players?.length ? report : null;
  // Computed up front so the chip can refuse before it is pressed rather than
  // silently placing a formation nobody measured.
  const avgMagnets = useMemo(
    () => (measured ? fromAveragePositions(measured.players ?? [], measured.meta?.pitch) : []),
    [measured],
  );
  const chosen = picked != null ? draft.magnets[picked] : undefined;
  const mark = pickedShape != null ? draft.shapes[pickedShape] : undefined;

  // A label typed onto the pitch should land where the caret is, not after a
  // trip to the panel — so the input for a just-placed label takes focus.
  const labelRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (mark?.kind === "text") labelRef.current?.focus();
  }, [pickedShape, mark?.kind]);

  // Escape leaves without saving, ⌘Z steps back. Bound on the window because
  // the canvas is an SVG and the panel is a pile of inputs — there is no one
  // element that reliably has focus while somebody is drawing.
  //
  // Re-bound whenever the undo stack changes, and NOT registered once with an
  // empty dependency list: a listener installed on the first render closes over
  // the stack as it was then, which is empty, so ⌘Z would do nothing forever.
  // Rebinding one window listener per stroke is not a cost worth outsmarting.
  //
  // An open panel eats the first Escape. It covers a corner of the pitch, so
  // "get this off the board" is the likelier request than "throw the board
  // away".
  const latest = useRef({ undo, onCancel, panel });
  useEffect(() => {
    latest.current = { undo, onCancel, panel };
  });
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = (e.target as HTMLElement)?.tagName === "INPUT";
      if (typing) return;
      if (e.key === "Escape") {
        if (latest.current.panel) setPanel(null);
        else latest.current.onCancel();
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        latest.current.undo();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  /** Lit when this row is what the pitch will do next. The placement tools all
   *  share one `tool`, so they are told apart by what they place; the arrow is
   *  lit for any of the three lines. */
  const isOn = (t: (typeof TOOLS)[number]) =>
    t.tool === "player"
      ? tool === "player" && side === t.side
      : t.tool
        ? tool === t.tool
        : LINES.some((l) => l.id === tool);

  /** Pick a pen — and recolour what is selected, so the swatch is also how you
   *  fix a line drawn in the wrong one. */
  const setInk = (k: string) => {
    setColor(k);
    if (pickedShape != null) {
      change({
        shapes: draft.shapes.map((sh, j) => (j === pickedShape ? { ...sh, color: k } : sh)),
      });
    }
  };

  /** The tool whose chevron is open, if any — the flyout renders from this
   *  rather than from the id, so a tool that loses its options cannot leave an
   *  empty panel behind. */
  const open = TOOLS.find((t) => t.id === flyout?.id && t.opts);
  /** Narrowed out here rather than inside the JSX: the callbacks below close
   *  over it, and a narrowing inside a conditional does not survive into a
   *  closure. */
  const kitSide =
    open?.opts === "kit" && (open.side === "home" || open.side === "away") ? open.side : null;

  /** A PNG of the board as it will be saved, not as it is being edited: built
   *  from pitchSvg, which is the same markup the note stores and the PDF
   *  prints, so what lands in the download is what a coach would have printed —
   *  no selection halos, no drag handles.
   *
   *  Rasterised rather than handed over as SVG, because the destination is a
   *  phone: this gets pasted into a group chat, and half of them will not
   *  render an SVG attachment at all. 2000px on the long side is a wall
   *  projector's worth and still under a megabyte.
   *
   *  ponytail: an <img> and a canvas. A serialiser library would buy nothing —
   *  the SVG is ours, self-contained, and carries no external fonts or images. */
  const [saving, setSaving] = useState(false);
  const [failed, setFailed] = useState(false);
  const exportPng = async () => {
    setSaving(true);
    setFailed(false);
    try {
      const { aspect } = viewBox(viewOf(draft.view), isVertical(draft.orient));
      const w = 2000;
      const h = Math.round(w / aspect);
      const svg = pitchSvg(draft)
        // AN SVG LOADED AS AN IMAGE IS A DOCUMENT, not a fragment, and a
        // document without a namespace does not parse — the <img> fires
        // `error` and nothing says why. Inline in the note the namespace is
        // implied by the HTML parser, which is why the board draws perfectly
        // everywhere else and only the export was blank.
        .replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ')
        // A percentage width has no meaning outside a layout, and the canvas
        // needs to know how big to be.
        .replace('width="100%"', `width="${w}" height="${h}"`);
      const img = new Image();
      await new Promise<void>((res, rej) => {
        img.onload = () => res();
        img.onerror = () => rej(new Error("could not draw the board"));
        img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
      });
      const c = document.createElement("canvas");
      c.width = w;
      c.height = h;
      c.getContext("2d")?.drawImage(img, 0, 0, w, h);
      const blob = await new Promise<Blob | null>((res) => c.toBlob(res, "image/png"));
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${(cap || "board").replace(/[^\w -]+/g, "").slice(0, 60).trim() || "board"}.png`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      // Said out loud in the pill. A download that silently does not happen is
      // indistinguishable from a browser that put the file somewhere unusual.
      setFailed(true);
    } finally {
      setSaving(false);
    }
  };

  const vert = isVertical(draft.orient);
  // The surface behind the canvas is painted the same grass, so the board reads
  // as one pitch running to the edges of the pane rather than a picture resting
  // on a desk. The canvas keeps its aspect; only the letterbox changes colour.
  const grass = GRASS[draft.grass]?.fill ?? GRASS.green.fill;

  // A portal either way, never left where the block sits: inside the document
  // tree the editor competes in the note's own stacking context and ends up
  // under the figure controls and the toolbar, and it would scroll away with
  // the text it was opened from.
  //
  // WHERE it lands is the whole difference. Into the note's pane it is an
  // `absolute` box filling that pane, and the app keeps its header and its
  // rail — you can still see which note you are in, and get out of it. Into
  // <body> it is `fixed` and swallows the window, which is what this did
  // always and is now only the fallback for a pane that isn't there.
  return createPortal(
    <div
      className={cn("z-50 overflow-hidden", host ? "absolute inset-0" : "fixed inset-0")}
      style={{ background: grass }}
    >
      {/* --- the pitch, edge to edge -------------------------------------- */}
      <div className="absolute inset-0 grid place-items-center">
        <BoardCanvas
          // Remounted on every replay, because a finished CSS animation stays
          // finished — this is the whole reason `replay` exists.
          key={replay}
          board={draft}
          animate={animated}
          editable
          tool={tool}
          color={color}
          side={side}
          gear={gear}
          picked={picked}
          pickedShape={pickedShape}
          onPick={setPicked}
          onPickShape={setPickedShape}
          onChange={change}
          onNewText={(i) => {
            setPickedShape(i);
            setTool("move");
          }}
          className="block max-h-full w-auto max-w-full touch-none select-none"
        />
      </div>

      {/* --- and the controls float on it ---------------------------------
          One `pointer-events-none` sheet holds them all, so the gaps between
          the islands are still pitch you can draw on. It is also the frame the
          tool flyout measures against. */}
      <div ref={surface} className="pointer-events-none absolute inset-0">
        {/* Top left, where a document keeps its name. */}
        <input
          value={cap}
          onChange={(e) => setCap(e.target.value.slice(0, 160))}
          placeholder="Caption — what this board shows"
          aria-label="Board caption"
          className={cn(
            FLOAT,
            "pointer-events-auto absolute top-2 left-2 h-7 w-64 max-w-[calc(100%-1rem)] px-2 text-[12px] text-[var(--ink)] outline-none placeholder:text-[var(--mute)] focus:border-[var(--accent)]",
          )}
        />

        {/* --- the tools -------------------------------------------------- */}
        <nav
          aria-label="Tools"
          className={cn(
            RAIL,
            "pointer-events-auto absolute top-1/2 left-2 flex max-h-[calc(100%-1rem)] -translate-y-1/2 flex-col items-center gap-1 overflow-y-auto p-1.5",
          )}
        >
          {TOOLS.map((t, i) => (
            <Fragment key={t.id}>
              {i > 0 && t.group !== TOOLS[i - 1].group && (
                <span aria-hidden className={RAIL_RULE} />
              )}
              <button
                type="button"
                title={`${t.label} — ${t.hint}`}
                aria-label={t.label}
                aria-pressed={isOn(t)}
                aria-expanded={t.opts ? flyout?.id === t.id : undefined}
                onClick={(e) => {
                  // The arrow has no tool of its own: it selects whichever of
                  // the three lines is current, and its flyout changes which.
                  setTool(t.tool ?? line);
                  if (t.side) setSide(t.side);
                  // Picking the tool opens its choice; picking it again puts
                  // the choice away. A tool with nothing behind it closes
                  // whatever was open, so the flyout never outlives its owner.
                  const b = e.currentTarget.getBoundingClientRect();
                  const s0 = surface.current?.getBoundingClientRect();
                  setFlyout((v) =>
                    t.opts && v?.id !== t.id
                      ? { id: t.id, top: b.top - (s0?.top ?? 0) + b.height / 2 }
                      : null,
                  );
                }}
                className={cn(RAIL_BTN, isOn(t) && RAIL_ON)}
              >
                {t.side === "home" || t.side === "away" ? (
                  <Disc side={t.side} kits={draft.kits} className="h-[18px] w-[18px]" />
                ) : (
                  <t.icon
                    size={16}
                    strokeWidth={1.7}
                    style={t.tint && !isOn(t) ? { color: t.tint } : undefined}
                  />
                )}
                {t.opts && (
                  <ChevronRight
                    size={9}
                    strokeWidth={2.2}
                    aria-hidden
                    className="absolute right-0 opacity-55"
                  />
                )}
              </button>
            </Fragment>
          ))}
        </nav>

        {/* What is behind a chevron, beside the row that owns it. */}
        {open && (
          <div
            className={cn(
              "pointer-events-auto absolute left-14",
              // A kit is a panel full of the app's own controls, so it keeps
              // the app's surface; everything else is two rows of swatches and
              // stays a dark pill on the grass.
              //
              // It is also the one flyout too tall to hang off its own row: a
              // panel centred on the Home button, which sits near the top of
              // the rail, has its own top half off the pane. So it takes the
              // full height instead and scrolls inside itself.
              open.opts === "kit"
                ? `${FLOAT} top-2 max-h-[calc(100%-1rem)] w-60 overflow-y-auto p-3`
                : `${RAIL} flex -translate-y-1/2 items-center gap-1.5 p-1.5`,
            )}
            style={open.opts === "kit" ? undefined : { top: flyout?.top }}
          >
            {kitSide ? (
              <KitPanel
                side={kitSide}
                kit={draft.kits[kitSide]}
                magnets={draft.magnets}
                onChange={(patch) =>
                  change({
                    kits: { ...draft.kits, [kitSide]: { ...draft.kits[kitSide], ...patch } },
                  })
                }
                onFormation={(name) =>
                  change({ magnets: sideFormation(name, kitSide, draft.magnets) })
                }
              />
            ) : open.opts === "gear" ? (
              Object.entries(GEAR).map(([k, g]) => (
                <button
                  key={k}
                  type="button"
                  title={g.label}
                  aria-label={g.label}
                  aria-pressed={gear === k}
                  onClick={() => setGear(k)}
                  className={cn(
                    "grid size-9 place-items-center rounded-[8px] transition-colors",
                    gear === k ? "bg-white/20" : "hover:bg-white/10",
                  )}
                >
                  {/* Drawn by the board's own builder, so the button cannot
                      promise a cone the pitch does not draw. */}
                  <svg viewBox={g.box} className="h-6 w-6">
                    <g dangerouslySetInnerHTML={{ __html: g.draw(MAGNET_FILL.gear) }} />
                  </svg>
                </button>
              ))
            ) : open.opts === "line" ? (
              <>
                {LINES.map((l) => (
                  <button
                    key={l.id}
                    type="button"
                    title={l.hint}
                    aria-pressed={line === l.id}
                    onClick={() => {
                      setLine(l.id);
                      setTool(l.id);
                    }}
                    className={cn(
                      "h-7 rounded-[8px] px-2 text-[11px] transition-colors",
                      line === l.id ? "bg-white/20 text-white" : "text-white/70 hover:bg-white/10",
                    )}
                  >
                    {l.label}
                  </button>
                ))}
                <span aria-hidden className="mx-0.5 h-5 w-px bg-white/15" />
                <Ink color={color} onPick={setInk} />
              </>
            ) : (
              <Ink color={color} onPick={setInk} />
            )}
          </div>
        )}

        {/* --- the settings ----------------------------------------------- */}
        <nav
          aria-label="Board settings"
          className={cn(
            RAIL,
            "pointer-events-auto absolute top-1/2 right-2 flex -translate-y-1/2 flex-col items-center gap-0.5 p-1",
          )}
        >
          {PANELS.map((pn) => (
            <button
              key={pn.id}
              type="button"
              title={pn.label}
              aria-label={pn.label}
              aria-pressed={panel === pn.id}
              onClick={() => setPanel((v) => (v === pn.id ? null : pn.id))}
              className={cn(RAIL_BTN, panel === pn.id && "bg-white/20 text-white")}
            >
              <pn.icon size={15} strokeWidth={1.7} />
            </button>
          ))}
        </nav>

        {panel && (
          <div
            className={cn(
              FLOAT,
              "pointer-events-auto absolute top-1/2 right-14 max-h-[calc(100%-1rem)] w-56 max-w-[calc(100%-5rem)] -translate-y-1/2 space-y-4 overflow-y-auto p-3",
            )}
          >
            {panel === "look" && (
              <>
                <Section title="Lines">
                  <div className="grid grid-cols-3 gap-1.5">
                    {Object.entries(LINE_TEMPLATES).map(([k, t]) => (
                      <Swatch
                        key={k}
                        on={draft.lines === k}
                        title={t.label}
                        onClick={() => change({ lines: k })}
                      >
                        <MiniPitch board={{ ...draft, lines: k }} />
                      </Swatch>
                    ))}
                  </div>
                </Section>

                <Section title="Pitch view">
                  {/* Wrapping, not a 2-column grid: the labels run from "Full
                      Pitch" to "Attacking Two-Thirds" and a chip never wraps
                      its own text, so a fixed cell just let the long ones
                      overlap their neighbour. */}
                  <div className="flex flex-wrap gap-1.5">
                    {Object.entries(VIEWS).map(([k, view]) => (
                      <Chip
                        key={k}
                        on={draft.view === k}
                        // Picking a crop also stands the pitch the way that
                        // crop is normally drawn. It is a default, not a lock —
                        // the toggle below still wins.
                        onClick={() => change({ view: k, orient: view.preferV ? "v" : "h" })}
                      >
                        {view.label}
                      </Chip>
                    ))}
                  </div>
                </Section>

                <Section title="Orientation">
                  <div className="flex gap-1.5">
                    <Chip on={!vert} onClick={() => change({ orient: "h" })}>
                      Across
                    </Chip>
                    <Chip on={vert} onClick={() => change({ orient: "v" })}>
                      <RotateCw size={11} strokeWidth={1.7} /> Upright
                    </Chip>
                  </div>
                </Section>

                <Section title="Grass">
                  <div className="grid grid-cols-3 gap-1.5">
                    {Object.entries(GRASS).map(([k, g]) => (
                      <Swatch
                        key={k}
                        on={draft.grass === k}
                        title={g.label}
                        onClick={() => change({ grass: k })}
                      >
                        <MiniPitch board={{ ...draft, grass: k }} />
                      </Swatch>
                    ))}
                  </div>
                  <Chip on={draft.stripes} onClick={() => change({ stripes: !draft.stripes })}>
                    Mown stripes
                  </Chip>
                </Section>
              </>
            )}

            {panel === "match" && (
              <Section title="From the match">
                {measured && avgMagnets.length ? (
                  <Chip
                    title="Place every tracked player at their average position. Provisional: a mean position is only as good as the track under it."
                    onClick={() =>
                      change({
                        magnets: avgMagnets,
                        note: [syntheticNote(measured), AVG_POSITION_NOTE].filter(Boolean).join(" "),
                        tier: "prov",
                      })
                    }
                  >
                    Average positions
                  </Chip>
                ) : measured ? (
                  <Chip disabled title="No track in this match carries an average position, and a board filled with a default 4-4-2 would be a formation nobody measured.">
                    Average positions
                  </Chip>
                ) : matchId ? (
                  <Chip disabled title="This match has no tracked players yet">
                    Average positions
                  </Chip>
                ) : (
                  <MatchPicker
                    trigger={
                      <Chip title="Choose a match to place its players">Average positions</Chip>
                    }
                  />
                )}
                {report?.territory && (
                  <Chip
                    title="Shade the thirds by where the tracked positions actually were. A share of positions, not possession — nothing here sees the ball."
                    onClick={() =>
                      change({
                        shapes: [...draft.shapes, ...territoryShapes(report.territory!)],
                        note: [syntheticNote(report), TERRITORY_NOTE].filter(Boolean).join(" "),
                        tier: "rel",
                      })
                    }
                  >
                    Territory
                  </Chip>
                )}
              </Section>
            )}

            {panel === "clear" && (
              <Section title="Clear">
                {draft.shapes.length > 0 && (
                  <Chip title="Remove every mark, keep the discs" onClick={() => change({ shapes: [] })}>
                    Marks
                  </Chip>
                )}
                <Chip onClick={() => change({ magnets: [] })}>Discs</Chip>
                {/* Back to a blank 4-4-2 on a plain pitch — the styling goes
                    with it, because "reset" that left a grid on chalk would
                    leave you hunting the panel for what is still switched on. */}
                <Chip onClick={() => change({ ...EMPTY_BOARD })}>Reset</Chip>
              </Section>
            )}
          </div>
        )}

        {/* --- what is picked, and the way out -----------------------------
            Left of the divider is about the ONE thing currently selected; with
            nothing selected it is the tool's own hint, so it never goes blank
            and never shows controls for nothing. Which side you are placing
            used to live here too and is in the rail's flyout now, where the
            tool that uses it is.

            Right of the divider: play it through, take a picture of it, step
            back, and the two ways out — a header bar's worth of buttons, in a
            pill that is only as wide as its words. */}
        <div
          className={cn(
            FLOAT,
            "pointer-events-auto absolute bottom-2 left-1/2 flex w-max max-w-[calc(100%-1rem)] -translate-x-1/2 flex-wrap items-center justify-center gap-1.5 px-2 py-1.5 text-[11px]",
          )}
        >
          {chosen?.side === "gear" ? (
            // Equipment has no number and no team. Offering the side chips here
            // would offer to turn a cone into a centre-back.
            <>
              <span className="px-1 text-[var(--mute)]">
                {GEAR[chosen.gear ?? ""]?.label ?? "Equipment"}
              </span>
              <Chip
                onClick={() => {
                  change({ magnets: draft.magnets.filter((_, j) => j !== picked) });
                  setPicked(null);
                }}
              >
                <Trash2 size={11} strokeWidth={1.7} /> Remove
              </Chip>
              <Chip onClick={() => setPicked(null)}>Done</Chip>
            </>
          ) : chosen ? (
            <>
              <input
                value={chosen.label}
                onChange={(e) =>
                  change({
                    magnets: draft.magnets.map((m, j) =>
                      j === picked ? { ...m, label: e.target.value.slice(0, 3) } : m,
                    ),
                  })
                }
                placeholder="No."
                aria-label="Disc label"
                className="h-6 w-12 rounded-[var(--radius-xs)] border border-[var(--hairline)] bg-[var(--input-bg)] px-1.5 text-center text-[11px] text-[var(--ink)] outline-none focus:border-[var(--accent)]"
              />
              {/* The name, which is printed BESIDE the disc rather than on it —
                  and only when the side's kit asks for one, so the field
                  appears with the thing it feeds. */}
              {chosen.side !== "ball" && draft.kits[chosen.side === "away" ? "away" : "home"].name !== "off" && (
                <input
                  value={chosen.name ?? ""}
                  onChange={(e) =>
                    change({
                      magnets: draft.magnets.map((m, j) =>
                        j === picked ? { ...m, name: e.target.value.slice(0, 24) } : m,
                      ),
                    })
                  }
                  placeholder="Name"
                  aria-label="Player name"
                  className="h-6 w-28 rounded-[var(--radius-xs)] border border-[var(--hairline)] bg-[var(--input-bg)] px-1.5 text-[11px] text-[var(--ink)] outline-none focus:border-[var(--accent)]"
                />
              )}
              {(["home", "away", "ball"] as const).map((sd) => (
                <Chip
                  key={sd}
                  on={chosen.side === sd}
                  onClick={() =>
                    change({
                      magnets: draft.magnets.map((m, j) => (j === picked ? { ...m, side: sd } : m)),
                    })
                  }
                >
                  <Dot side={sd} />
                  {sd}
                </Chip>
              ))}
              <Chip
                onClick={() => {
                  change({ magnets: draft.magnets.filter((_, j) => j !== picked) });
                  setPicked(null);
                }}
              >
                <Trash2 size={11} strokeWidth={1.7} /> Remove
              </Chip>
              <Chip onClick={() => setPicked(null)}>Done</Chip>
            </>
          ) : mark ? (
            <>
              {mark.kind === "text" ? (
                <input
                  ref={labelRef}
                  value={mark.label ?? ""}
                  onChange={(e) =>
                    change({
                      shapes: draft.shapes.map((sh, j) =>
                        j === pickedShape ? { ...sh, label: e.target.value.slice(0, 60) } : sh,
                      ),
                    })
                  }
                  placeholder="Write on the pitch"
                  aria-label="Pitch label"
                  className="h-6 w-56 max-w-[40vw] rounded-[var(--radius-xs)] border border-[var(--hairline)] bg-[var(--input-bg)] px-1.5 text-[11px] text-[var(--ink)] outline-none focus:border-[var(--accent)]"
                />
              ) : (
                <span className="px-1 text-[var(--mute)]">{MARK_NOTE[mark.kind]}</span>
              )}
              <Chip
                onClick={() => {
                  change({ shapes: draft.shapes.filter((_, j) => j !== pickedShape) });
                  setPickedShape(null);
                }}
              >
                <Trash2 size={11} strokeWidth={1.7} /> Remove
              </Chip>
              <Chip onClick={() => setPickedShape(null)}>Done</Chip>
            </>
          ) : (
            <span
              title={TOOLS.find(isOn)?.hint}
              className="max-w-[34ch] truncate px-1 text-[var(--mute)]"
            >
              {TOOLS.find(isOn)?.hint}
            </span>
          )}

          <span aria-hidden className="mx-0.5 h-4 w-px bg-[var(--hairline)]" />

          {/* Play it through. A board is explained in the order it was drawn,
              and a coach with a screen in front of players wants the marks to
              arrive one at a time rather than all at once. Pressing it while it
              is already on plays it again. */}
          <button
            type="button"
            role="switch"
            aria-checked={animated}
            onClick={() => {
              setAnimated(true);
              setReplay((n) => n + 1);
            }}
            onDoubleClick={() => setAnimated(false)}
            title={
              animated
                ? "Play it through again. Double-click to stop."
                : "Reveal the marks one at a time, in the order they were drawn"
            }
            className="flex h-6 items-center gap-1.5 rounded-[var(--radius-pill)] px-1.5 text-[11px] text-[var(--graphite)] transition-colors hover:bg-[var(--card)] hover:text-[var(--ink)]"
          >
            <span
              aria-hidden
              className={cn(
                "flex h-3.5 w-6 shrink-0 items-center rounded-full p-0.5 transition-colors",
                animated ? "bg-[var(--accent)]" : "bg-[var(--hairline-strong)]",
              )}
            >
              <span
                className={cn(
                  "size-2.5 rounded-full bg-white transition-transform",
                  animated && "translate-x-2.5",
                )}
              />
            </span>
            Animated
          </button>

          <button
            type="button"
            onClick={exportPng}
            disabled={saving}
            title="Download the board as a PNG — the same picture the note keeps, at 2000px"
            className={cn(
              "flex h-6 items-center gap-1 rounded-[var(--radius-pill)] px-2 text-[11px] transition-colors hover:bg-[var(--card)] hover:text-[var(--ink)] disabled:opacity-40",
              failed ? "text-[var(--danger)]" : "text-[var(--graphite)]",
            )}
          >
            <Download size={11} strokeWidth={1.7} />{" "}
            {saving ? "…" : failed ? "Export failed" : "Export"}
          </button>

          <button
            type="button"
            title="Undo (⌘Z)"
            aria-label="Undo"
            onClick={undo}
            disabled={!past.length}
            className="grid size-6 place-items-center rounded-[var(--radius-pill)] text-[var(--graphite)] transition-colors hover:bg-[var(--card)] hover:text-[var(--ink)] disabled:opacity-30 disabled:hover:bg-transparent"
          >
            <Undo2 size={12} strokeWidth={1.7} />
          </button>

          <span aria-hidden className="mx-0.5 h-4 w-px bg-[var(--hairline)]" />

          <button
            type="button"
            onClick={onCancel}
            className="h-6 rounded-[var(--radius-pill)] px-2 text-[11px] text-[var(--graphite)] transition-colors hover:bg-[var(--card)] hover:text-[var(--ink)]"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => onSave(draft, cap)}
            className="h-6 rounded-[var(--radius-pill)] bg-[var(--accent)] px-2.5 text-[11px] font-medium text-[var(--on-accent,#fff)] transition-opacity hover:opacity-90"
          >
            Save board
          </button>
        </div>
      </div>
    </div>,
    host ?? document.body,
  );
}

const MARK_NOTE: Record<string, string> = {
  pass: "A pass — the ball moves.",
  run: "A run — the player moves.",
  dribble: "Carried — the player takes it.",
  zone: "A zone.",
  pen: "Freehand.",
};

/** A tiny non-interactive board, for the style swatches. Drawn by the same
 *  canvas as the real thing, so a swatch cannot promise a look the board does
 *  not deliver — only with the lines fattened, because at this size the real
 *  widths are sub-pixel and every template would look the same. */
function MiniPitch({ board }: { board: Board }) {
  return (
    <BoardCanvas
      board={{ ...board, magnets: [], shapes: [], view: "full", orient: "h" }}
      boldness={5}
      className="block h-full w-full"
    />
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-1.5">
      <h3 className="text-[10.5px] tracking-wide text-[var(--mute)] uppercase">{title}</h3>
      <div className="flex flex-wrap gap-1.5">{children}</div>
    </section>
  );
}

function Swatch({
  on,
  title,
  onClick,
  children,
}: {
  on: boolean;
  title: string;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      title={title}
      aria-label={title}
      aria-pressed={on}
      onClick={onClick}
      className={cn(
        // Square, like every other pitch in the app. A rounded thumbnail of a
        // board is a rounded board, which is not a shape a pitch comes in.
        "overflow-hidden border transition-colors",
        on ? "border-[var(--accent)]" : "border-[var(--hairline)] hover:border-[var(--hairline-strong)]",
      )}
    >
      {children}
    </button>
  );
}

/** The pens. Four of the flyouts end in these, which is the whole reason they
 *  are a component rather than a fourth copy of the same five buttons. */
function Ink({ color, onPick }: { color: string; onPick: (k: string) => void }) {
  return (
    <>
      {Object.keys(INK).map((k) => (
        <button
          key={k}
          type="button"
          title={k}
          aria-label={`${k} pen`}
          aria-pressed={color === k}
          onClick={() => onPick(k)}
          className={cn(
            "size-5 shrink-0 rounded-full border-2 transition-transform",
            color === k ? "scale-110 border-white" : "border-white/25",
          )}
          style={{ background: INK[k] }}
        />
      ))}
    </>
  );
}

/** A disc, drawn by the board's own builder at whatever size it is asked for.
 *
 *  The rail used to wear a shirt icon for each side, which is a picture of a
 *  thing the board does not contain. This is the actual counter, in the actual
 *  kit — change the colour in the panel and the button that places them
 *  changes with it, which is the only honest way to label a tool that puts a
 *  specific object on the pitch. */
function Disc({
  side,
  kits,
  label = "",
  span = 3.6,
  className,
}: {
  side: Magnet["side"];
  kits: Board["kits"];
  label?: string;
  /** How much of the pitch, in metres, the box shows. Bigger means smaller. */
  span?: number;
  className?: string;
}) {
  const half = span / 2;
  return (
    <svg
      viewBox={`${(52.5 - half).toFixed(2)} ${(34 - half).toFixed(2)} ${span} ${span}`}
      className={className}
      aria-hidden
      dangerouslySetInnerHTML={{
        __html: discDefs(kits) + magnetSvg({ x: 50, y: 50, label, side }, false, kits),
      }}
    />
  );
}

/** TEAM STYLE: what one side wears, and it applies to every disc on that side.
 *
 *  Behind the chevron on the shirt itself, because that is the button that puts
 *  them on the pitch. It replaced the formations, which had been living here on
 *  the grounds that they also concern a whole side — but a formation is a thing
 *  you do once and a kit is a thing you set once, and only one of them belongs
 *  on a button you press eleven times. The formations are on the same panel,
 *  underneath, where they cost nothing.
 */
function KitPanel({
  side,
  kit,
  magnets,
  onChange,
  onFormation,
}: {
  side: "home" | "away";
  kit: Kit;
  magnets: Magnet[];
  onChange: (patch: Partial<Kit>) => void;
  onFormation: (name: string) => void;
}) {
  const set = (patch: Partial<Kit>) => onChange(patch);
  const kits = { home: kit, away: kit } as Board["kits"];
  return (
    <div className="space-y-3">
      {/* The disc itself, at the size and in the colours currently chosen. A
          preview that is the real renderer cannot lie about the result. */}
      <div className="grid h-20 place-items-center rounded-[var(--radius-sm)] bg-[var(--card)]">
        <Disc
          side={side}
          kits={kits}
          label={magnets.find((m) => m.side === side && m.label)?.label ?? "9"}
          span={6}
          className="h-16 w-16"
        />
      </div>

      <Row label="Appearance">
        <Seg
          on={!kit.outline}
          onClick={() => set({ outline: false })}
          title="A filled counter"
        >
          <span className="size-3 rounded-full bg-current" />
        </Seg>
        <Seg
          on={kit.outline}
          onClick={() => set({ outline: true })}
          title="Hollow — for the side the diagram is not about"
        >
          <span className="size-3 rounded-full border-[1.5px] border-current" />
        </Seg>
      </Row>

      <Row label="Style">
        <Seg on={kit.style !== "halves"} onClick={() => set({ style: "solid" })}>
          1 colour
        </Seg>
        <Seg on={kit.style === "halves"} onClick={() => set({ style: "halves" })}>
          2 colours
        </Seg>
      </Row>

      <Row label="Colour">
        <Palette value={kit.colour} onPick={(c) => set({ colour: c })} />
      </Row>

      {kit.style === "halves" && (
        <Row label="Second">
          <Palette value={kit.colour2} onPick={(c) => set({ colour2: c })} />
        </Row>
      )}

      <Row label="Size" nowrap>
        <input
          type="range"
          min={0.6}
          max={1.8}
          step={0.1}
          value={kit.size}
          aria-label="Disc size"
          onChange={(e) => set({ size: Number(e.target.value) })}
          className="h-1 flex-1 accent-[var(--accent)]"
        />
        <span className="w-6 shrink-0 text-right tabular-nums text-[var(--ink)]">
          {kit.size.toFixed(1)}
        </span>
      </Row>

      <Row label="Number">
        <Seg on={kit.number} onClick={() => set({ number: !kit.number })}>
          {kit.number ? "On" : "Off"}
        </Seg>
        {kit.number && (
          <>
            <Seg on={!kit.numberDark} onClick={() => set({ numberDark: false })} title="Light">
              <span className="size-3 rounded-full border border-[var(--hairline-strong)] bg-white" />
            </Seg>
            <Seg on={kit.numberDark} onClick={() => set({ numberDark: true })} title="Dark">
              <span className="size-3 rounded-full bg-[#16181c]" />
            </Seg>
          </>
        )}
      </Row>

      <Row label="Name">
        {(["off", "above", "under"] as const).map((v) => (
          <Seg key={v} on={kit.name === v} onClick={() => set({ name: v })}>
            {v}
          </Seg>
        ))}
      </Row>

      <div className="border-t border-[var(--hairline)] pt-2">
        <h3 className="mb-1.5 text-[10.5px] tracking-wide text-[var(--mute)] uppercase">
          Put a shape out
        </h3>
        <div className="flex flex-wrap gap-1.5">
          {Object.keys(FORMATIONS).map((name) => (
            <Chip key={name} title={`A ${name} of ${side} discs`} onClick={() => onFormation(name)}>
              {name}
            </Chip>
          ))}
        </div>
      </div>
    </div>
  );
}

/** A labelled line in the kit panel. The label column is fixed so the controls
 *  line up down the panel rather than each row starting wherever its own words
 *  ended. */
function Row({
  label,
  nowrap,
  children,
}: {
  label: string;
  /** For the one row whose controls are a slider and its readout, which read as
   *  one control and must not be split across two lines. */
  nowrap?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-2 text-[11px]">
      <span className="w-[3.8rem] shrink-0 text-[var(--mute)]">{label}</span>
      <div className={cn("flex flex-1 items-center gap-1", !nowrap && "flex-wrap")}>{children}</div>
    </div>
  );
}

/** One choice in a row of choices. */
function Seg({
  on,
  title,
  onClick,
  children,
}: {
  on: boolean;
  title?: string;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      title={title}
      aria-pressed={on}
      onClick={onClick}
      className={cn(
        "flex h-6 items-center justify-center gap-1 rounded-[var(--radius-sm)] border px-2 text-[11px] capitalize transition-colors",
        on
          ? "border-[var(--accent)] bg-[var(--accent)]/12 text-[var(--ink)]"
          : "border-[var(--hairline)] text-[var(--graphite)] hover:bg-[var(--card)] hover:text-[var(--ink)]",
      )}
    >
      {children}
    </button>
  );
}

/** The thirteen colours a kit can be. Keys, not colour strings — see
 *  KIT_COLOURS. */
function Palette({ value, onPick }: { value: string; onPick: (k: string) => void }) {
  return (
    <>
      {Object.entries(KIT_COLOURS).map(([k, hex]) => (
        <button
          key={k}
          type="button"
          title={k}
          aria-label={k}
          aria-pressed={value === k}
          onClick={() => onPick(k)}
          className={cn(
            "size-4 rounded-full border transition-transform",
            value === k
              ? "scale-115 border-[var(--ink)]"
              : "border-[var(--hairline-strong)] hover:scale-110",
          )}
          style={{ background: hex }}
        />
      ))}
    </>
  );
}

const Dot = ({ side }: { side: Magnet["side"] }) => (
  <span aria-hidden className="size-2 rounded-full" style={{ background: MAGNET_FILL[side] }} />
);

function Chip({
  on,
  disabled,
  title,
  onClick,
  children,
}: {
  on?: boolean;
  disabled?: boolean;
  title?: string;
  /** Optional so a Chip can be a MatchPicker trigger, which supplies its own
   *  click handler by cloning this element. */
  onClick?: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      title={title}
      onClick={onClick}
      className={cn(
        "flex h-6 items-center gap-1 rounded-[var(--radius-pill)] border px-2 text-[11px] whitespace-nowrap capitalize transition-colors",
        on
          ? "border-[var(--hairline-strong)] bg-[var(--card)] text-[var(--ink)]"
          : "border-[var(--hairline)] text-[var(--graphite)] hover:bg-[var(--card)] hover:text-[var(--ink)]",
        disabled && "cursor-default opacity-40 hover:bg-transparent hover:text-[var(--graphite)]",
      )}
    >
      {children}
    </button>
  );
}
