"use client";

import { useRouter, usePathname } from "next/navigation";
import { Check, ChevronsUpDown } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useMatch, useMatches } from "@/hooks/use-matches";
import { matchDate, matchTitle, statusOf, TONE_CLASS } from "@/lib/status";

/** Which match you are looking at, and how to change it — in the header, on
 *  every page. Previously the only way to switch was to go back to the list,
 *  which is why people lost track of what they were reading. */
export function MatchSwitcher() {
  const router = useRouter();
  const pathname = usePathname();
  const { id, match } = useMatch();
  const { matches } = useMatches();

  const go = (next: string) =>
    router.push(`${pathname}?m=${encodeURIComponent(next)}`);

  if (!id) {
    return (
      <div className="truncate px-2 text-[14px] font-medium text-[var(--ink)]">
        All matches
      </div>
    );
  }

  const label = match ? matchTitle(match) : "Loading…";
  const view = match ? statusOf(match) : null;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        {/* The pane's title, not a control that looks like one: 14px --ink, no
            resting background, the chevron only on hover. */}
        <Button
          variant="ghost"
          className="group h-8 min-w-0 gap-1.5 px-2 text-[14px] font-medium text-[var(--ink)] hover:bg-[var(--card)]"
        >
          <span className="truncate">{label}</span>
          {view && (
            <span className={`shrink-0 text-xs font-normal ${TONE_CLASS[view.tone]}`}>
              {view.label}
            </span>
          )}
          <ChevronsUpDown className="size-3.5 shrink-0 text-[var(--graphite)] opacity-0 transition-opacity group-hover:opacity-100 group-data-[state=open]:opacity-100" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-72">
        <DropdownMenuLabel>Switch match</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {(matches ?? []).map((m) => {
          const v = statusOf(m);
          return (
            <DropdownMenuItem key={m.id} onSelect={() => go(m.id)} className="gap-2">
              <Check className={`size-4 ${m.id === id ? "opacity-100" : "opacity-0"}`} />
              <span className="min-w-0 flex-1 truncate">{matchTitle(m)}</span>
              <span className="text-muted-foreground shrink-0 text-xs">
                {matchDate(m) ?? v.label}
              </span>
            </DropdownMenuItem>
          );
        })}
        <DropdownMenuSeparator />
        {/* Back to the library, IN PLACE: clearing `?m=` leaves you on the
            surface you are reading — the film tab shows its gallery again —
            where "All matches" navigates away from it entirely. */}
        <DropdownMenuItem onSelect={() => go("")}>Back to the library</DropdownMenuItem>
        <DropdownMenuItem onSelect={() => router.push("/")}>
          All matches
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
