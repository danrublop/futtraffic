"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";

import { cn } from "@/lib/utils";

/** One choice at a time, and the next set depends on the last.
 *
 *  The panels used to be a form: source, measure, breakdown, split, team,
 *  rows, compare, all on screen at once, most of them meaningless until the
 *  ones above were answered. That is a settings dialog, and it asks a coach to
 *  hold the whole shape of a query in their head before making the first
 *  decision.
 *
 *  A spreadsheet does not do that. It asks WHAT KIND OF PICTURE, and only then
 *  what goes in it — each click narrowing what the next click can be. So this
 *  is a stack of steps: the trail of what has been chosen along the top, one
 *  set of choices in the body, and a step advances the moment you pick.
 *
 *  Every crumb in the trail is a button back to that step, which is the part
 *  that makes a drill-down bearable — a wrong turn is one click to undo, not a
 *  reason to close the panel and start again.
 */

export interface Crumb {
  label: string;
  /** Back to the step that chose this. Absent for the step you are on. */
  onBack?: () => void;
}

export function Trail({ crumbs }: { crumbs: Crumb[] }) {
  if (!crumbs.length) return null;
  return (
    <div className="flex flex-wrap items-center gap-0.5 text-[10.5px]">
      {crumbs.map((c, i) => (
        <span key={`${c.label}-${i}`} className="flex items-center gap-0.5">
          {i > 0 && <ChevronRight size={10} strokeWidth={2} className="text-[var(--mute)]" />}
          {c.onBack ? (
            <button
              type="button"
              onClick={c.onBack}
              className="rounded-[var(--radius-xs)] px-1 py-0.5 text-[var(--graphite)] transition-colors hover:bg-[var(--card)] hover:text-[var(--ink)]"
            >
              {c.label}
            </button>
          ) : (
            <span className="px-1 py-0.5 text-[var(--ink)]">{c.label}</span>
          )}
        </span>
      ))}
    </div>
  );
}

export interface Choice {
  id: string;
  label: string;
  /** What this choice means, in the words that say what you get. */
  hint?: string;
  /** Why it is not available. A disabled choice that says nothing is a bug
   *  report waiting to be filed. */
  why?: string;
  /** Drawn to the left — a swatch, a glyph, a tiny preview. */
  icon?: React.ReactNode;
}

/** A step's choices, as rows you click.
 *
 *  Rows rather than chips because a choice that carries a sentence explaining
 *  it cannot be a chip, and every choice here should be able to: "one disc per
 *  player at their mean position" is the difference between picking the right
 *  picture and picking the one whose name sounded closest.
 */
export function Choices({
  items,
  onPick,
  value,
  columns,
}: {
  items: Choice[];
  onPick: (id: string) => void;
  /** Marks the current selection when a step is revisited. */
  value?: string;
  /** Lay out as a grid when the labels are short and there are many. */
  columns?: number;
}) {
  return (
    <div
      className={cn("grid gap-1", columns ? "" : "grid-cols-1")}
      style={columns ? { gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` } : undefined}
    >
      {items.map((it) => (
        <button
          key={it.id}
          type="button"
          disabled={!!it.why}
          title={it.why ?? it.hint}
          onClick={() => onPick(it.id)}
          className={cn(
            "flex items-center gap-2 rounded-[var(--radius-sm)] border px-2 py-1.5 text-left transition-colors",
            it.why
              ? "cursor-not-allowed border-[var(--hairline)] opacity-45"
              : value === it.id
                ? "border-[var(--accent)] bg-[var(--card)]"
                : "border-[var(--hairline)] hover:border-[var(--hairline-strong)] hover:bg-[var(--card)]",
          )}
        >
          {it.icon && <span className="grid size-6 flex-none place-items-center">{it.icon}</span>}
          <span className="min-w-0 flex-1">
            <span className="block truncate text-[11.5px] text-[var(--ink)]">{it.label}</span>
            {(it.why ?? it.hint) && (
              <span className="block text-[10px] leading-snug text-[var(--mute)]">
                {it.why ?? it.hint}
              </span>
            )}
          </span>
          {!it.why && <ChevronRight size={12} strokeWidth={1.8} className="flex-none text-[var(--mute)]" />}
        </button>
      ))}
    </div>
  );
}

/** The one control a step needs that is not a choice: going back. */
export function Back({ onClick, label }: { onClick: () => void; label: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex items-center gap-1 self-start rounded-[var(--radius-sm)] px-1 py-0.5 text-[11px] text-[var(--graphite)] transition-colors hover:bg-[var(--card)] hover:text-[var(--ink)]"
    >
      <ChevronLeft size={12} strokeWidth={1.9} /> {label}
    </button>
  );
}
