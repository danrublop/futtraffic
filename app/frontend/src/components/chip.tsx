"use client";

import { Folder } from "lucide-react";

import { cn } from "@/lib/utils";

/** A team, as a folder. The notes gallery's chip without the rename, the ⋮ and
 *  the drop target: a squad is not renamed from the match library — the club
 *  screen owns the ladder — and there is nothing to drag onto it.
 *
 *  The library filters by squad through a dropdown now, beside its other two
 *  filters. This stays because the FILM gallery still lays its ladder out as
 *  folders, and one chip drawn in one place beats two that drift apart. */
export function Chip({
  label,
  icon: Icon,
  active,
  onClick,
}: {
  label: string;
  icon: typeof Folder;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-pressed={active}
      className={cn(
        "flex h-11 max-w-[22ch] shrink-0 items-center gap-2 rounded-[var(--radius-md)] border px-2.5 text-[12.5px] transition-colors sm:h-9",
        active
          ? "border-[var(--hairline-strong)] bg-[var(--card-hover)] text-[var(--ink)]"
          : "border-transparent bg-[var(--card)] text-[var(--ink-soft)] hover:bg-[var(--card-hover)] hover:text-[var(--ink)]",
      )}
    >
      <Icon size={14} strokeWidth={1.7} className="shrink-0 text-[var(--graphite)]" />
      <span className="truncate">{label}</span>
    </button>
  );
}
