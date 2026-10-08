"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { ChevronRight } from "lucide-react";

import { cn } from "@/lib/utils";

/** The right-click menu, and the panels that open off it.
 *
 *  ONE CHROME, because a table and a chart answering the same gesture with two
 *  different-looking panels reads as two different products. Both now come
 *  through here: same corner radius, same rule, same row, same chevron saying
 *  "there is more this way".
 *
 *  GATED, the way a spreadsheet's own menus are. The root is a short list of
 *  what can be changed; picking one opens its choices in a panel TO THE RIGHT,
 *  and picking there can open another. Nothing is on screen until it is asked
 *  for, which is the difference between a menu and a control panel — and the
 *  reason a chart's settings can be exhaustive without the note becoming a
 *  wall of chips.
 *
 *  Never a dialog. No backdrop, no focus trap, nothing blanked: the figure
 *  stays where it was and redraws under the panel as the choices change.
 */

const PAD = 8;

/** Where a submenu hangs from: the row's top, and the list's two edges so it
 *  can open right or flip left. */
export interface Anchor {
  x: number;
  y: number;
  right: number;
}

/** Keep a floating panel inside the window.
 *
 *  Measured AFTER layout, not guessed from a width constant — a submenu's size
 *  depends on what is in it, and a right-click near the foot of a long note
 *  would otherwise open a panel below the fold. A menu you have to scroll to
 *  find is a menu that did not open.
 */
function useClamped(at: { x: number; y: number }, deps: unknown[] = []) {
  const ref = useRef<HTMLDivElement>(null);
  const [place, setPlace] = useState(at);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    setPlace({
      x: Math.max(PAD, Math.min(at.x, window.innerWidth - r.width - PAD)),
      y: Math.max(PAD, Math.min(at.y, window.innerHeight - r.height - PAD)),
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [at.x, at.y, ...deps]);
  return { ref, place };
}

/** The floating shell. Closes on Escape and on a click anywhere outside it —
 *  which for a cascade means outside the ROOT, since a submenu is a sibling in
 *  the DOM rather than a child. */
export function MenuShell({
  at,
  onClose,
  label,
  width = "14rem",
  children,
  extraInside,
}: {
  at: { x: number; y: number };
  onClose: () => void;
  label: string;
  width?: string;
  children: React.ReactNode;
  /** Panels that are visually part of this menu but not inside its box — a
   *  click in one of them must not count as a click away. */
  extraInside?: React.RefObject<HTMLElement | null>[];
}) {
  const { ref, place } = useClamped(at);

  useEffect(() => {
    const away = (e: MouseEvent) => {
      const t = e.target as Node;
      if (ref.current?.contains(t)) return;
      if (extraInside?.some((r) => r.current?.contains(t))) return;
      onClose();
    };
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("mousedown", away);
    // A right-click elsewhere is not a mousedown anywhere this listener sees,
    // so without it the old menu stayed up while the new one opened over it —
    // two menus on screen, and only one of them about the thing under the
    // pointer.
    window.addEventListener("contextmenu", away);
    window.addEventListener("keydown", esc);
    return () => {
      window.removeEventListener("mousedown", away);
      window.removeEventListener("contextmenu", away);
      window.removeEventListener("keydown", esc);
    };
  }, [onClose, ref, extraInside]);

  return (
    <div
      ref={ref}
      role="menu"
      aria-label={label}
      // Chrome that happens to live inside a document. Without this,
      // ProseMirror treats every keystroke in here as a keystroke in the note.
      contentEditable={false}
      suppressContentEditableWarning
      style={{ left: place.x, top: place.y, width }}
      className="ft-controls fixed z-50 max-w-[calc(100vw-1rem)] rounded-[var(--radius-md)] border border-[var(--hairline)] bg-[var(--panel)] p-1 text-[12px] shadow-[var(--shadow-lg)]"
    >
      {children}
    </div>
  );
}

/** A panel hanging off a menu row, opened to the right of it.
 *
 *  Flips to the LEFT when there is no room — a submenu that opens off the edge
 *  of the window is the failure this whole component exists to avoid, and the
 *  right-hand side is exactly where a menu opened near the right margin sits.
 */
export function SubPanel({
  anchor,
  width = "20rem",
  children,
  panelRef,
}: {
  /** The row this hangs off, in client coordinates. */
  anchor: { x: number; y: number; right: number };
  width?: string;
  children: React.ReactNode;
  panelRef?: React.RefObject<HTMLDivElement | null>;
}) {
  const own = useRef<HTMLDivElement>(null);
  const ref = panelRef ?? own;
  const [place, setPlace] = useState({ x: anchor.right, y: anchor.y });

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const fitsRight = anchor.right + r.width + PAD <= window.innerWidth;
    setPlace({
      x: fitsRight ? anchor.right : Math.max(PAD, anchor.x - r.width),
      y: Math.max(PAD, Math.min(anchor.y, window.innerHeight - r.height - PAD)),
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [anchor.x, anchor.y, anchor.right]);

  return (
    <div
      ref={ref}
      role="menu"
      contentEditable={false}
      suppressContentEditableWarning
      style={{ left: place.x, top: place.y, width }}
      className="ft-controls fixed z-50 max-h-[26rem] max-w-[calc(100vw-1rem)] overflow-y-auto rounded-[var(--radius-md)] border border-[var(--hairline)] bg-[var(--panel)] p-1.5 text-[12px] shadow-[var(--shadow-lg)]"
    >
      {children}
    </div>
  );
}

/** One row. Text, and a chevron when there is more this way.
 *
 *  NO ICON. A menu of six rows each carrying a little glyph is six glyphs
 *  competing with six words that already say the same thing, in a document
 *  whose whole visual language is text. The chevron stays because it is not
 *  decoration — it is the only thing that distinguishes a row that opens
 *  something from a row that does something.
 *
 *  HOVER OPENS IT, click does too. That is how a menu behaves everywhere else,
 *  and running down a list to see what is in each branch should not cost five
 *  clicks. A reason a row is unavailable rides in the tooltip rather than on
 *  the page — it is wanted about once, by the person who just tried it.
 */
export function MenuItem({
  hint,
  more,
  on,
  disabled,
  onPick,
  onHover,
  children,
}: {
  hint?: string;
  more?: boolean;
  on?: boolean;
  disabled?: boolean;
  onPick: (rect: Anchor) => void;
  /** Fired on hover as well, so a branch opens without a click. */
  onHover?: (rect: Anchor) => void;
  children: React.ReactNode;
}) {
  const where = (el: HTMLElement): Anchor => {
    const list = (el.parentElement ?? el).getBoundingClientRect();
    const own = el.getBoundingClientRect();
    return { x: list.left, y: own.top - 4, right: list.right + 2 };
  };
  return (
    <button
      type="button"
      role="menuitem"
      disabled={disabled}
      title={hint}
      onClick={(e) => onPick(where(e.currentTarget))}
      onMouseEnter={(e) => !disabled && onHover?.(where(e.currentTarget))}
      className={cn(
        "flex w-full items-center gap-2 rounded-[var(--radius-sm)] px-2 py-1.5 text-left transition-colors",
        disabled
          ? "cursor-not-allowed text-[var(--mute)] opacity-50"
          : on
            ? "bg-[var(--card)] text-[var(--ink)]"
            : "text-[var(--ink)] hover:bg-[var(--card)]",
      )}
    >
      <span className="flex-1 truncate">{children}</span>
      {more && <ChevronRight size={13} strokeWidth={1.8} className="shrink-0 text-[var(--mute)]" />}
    </button>
  );
}

export const MenuRule = () => <div aria-hidden className="my-1 h-px bg-[var(--hairline)]" />;
