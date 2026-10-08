"use client";

import { useState } from "react";
import { ArrowUpDown } from "lucide-react";

import { TableHead } from "@/components/ui/table";
import { cn } from "@/lib/utils";

/** The sortable column head, shared by every table on the Data surface.
 *
 *  It lived inside season-tables.tsx until the player registry became a real
 *  table and wanted the same thing. It cannot be imported FROM there — that
 *  file imports the registry — so it lives on its own rather than existing
 *  twice with two hover states.
 *
 *  Sorting is the whole point of a database table; a table you cannot reorder
 *  is a screenshot.
 */
export function SortHead({
  children,
  active,
  onClick,
  className,
}: {
  children: React.ReactNode;
  active: boolean;
  onClick: () => void;
  className?: string;
}) {
  return (
    <TableHead className={className}>
      <button
        type="button"
        onClick={onClick}
        className={cn(
          "inline-flex items-center gap-1 transition-colors hover:text-[var(--ink)]",
          active && "text-[var(--ink)]",
        )}
      >
        {children}
        <ArrowUpDown size={11} className={cn("shrink-0", active ? "opacity-70" : "opacity-25")} />
      </button>
    </TableHead>
  );
}

/** Sort state: a key, and a direction. A new key starts descending, because
 *  every numeric column on this surface is one where "most" is the question. */
export function useSort(initial: string) {
  const [key, setKey] = useState(initial);
  const [asc, setAsc] = useState(false);
  const on = (k: string) => () => {
    if (k === key) setAsc((a) => !a);
    else {
      setKey(k);
      setAsc(false);
    }
  };
  return { key, asc, on };
}

export const cmp = (a: number, b: number, asc: boolean) => (asc ? a - b : b - a);
