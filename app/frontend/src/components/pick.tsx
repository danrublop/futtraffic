"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

/** ONE FILTER DROPDOWN, shared by the Data browser and the match library.
 *
 *  It lived inside data-browser.tsx until the library wanted the same three
 *  filters above the same kind of table. Two screens that filter by squad must
 *  not be able to disagree about what the control looks like or what "every"
 *  reads as - the same argument as components/rating-chip.tsx. */
export function Pick({
  label,
  value,
  onChange,
  all,
  items,
  className = "w-[144px]",
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  all: string;
  items: { id: string; label: string }[];
  className?: string;
}) {
  const ALL = "~all";
  return (
    // NO VISIBLE LABEL: the value says which filter it is ("Every squad",
    // "Every match"), and three labels were three columns of width. The name
    // stays for anything reading the page aloud.
    <label className="flex items-center gap-1.5">
      <span className="sr-only">{label}</span>
      <Select value={value || ALL} onValueChange={(v) => onChange(v === ALL ? "" : v)}>
        {/* THE TAB CHIP'S CLOTHES, class for class (Tab in data-browser.tsx,
            which took them from the Matches page's folder chip). These sit in
            the same bar as those chips and do the same kind of job - narrowing
            what the table below shows - so a second control idiom on one row
            was just two ways of saying "click me".

            A filter that is actually filtering wears the SELECTED chip's
            colours. That is not decoration: "Every squad" and one squad look
            alike at a glance, and a table quietly showing a subset is the
            oldest way to misread a number in this product. */}
        {/* The height and radius are written in the trigger's OWN data-size
            form. SelectTrigger sizes itself with `data-[size=...]:h-*`, and an
            attribute selector outranks a plain `h-9` however late it is in the
            class list - the chip came out 28px against the tabs' 36 until this
            matched its shape. */}
        <SelectTrigger
          aria-label={label}
          className={cn(
            "flex shrink-0 items-center gap-2 border px-2.5 text-[12.5px] whitespace-nowrap transition-colors",
            "data-[size=default]:h-11 data-[size=default]:rounded-[var(--radius-md)] sm:data-[size=default]:h-9",
            value
              ? "border-[var(--hairline-strong)] bg-[var(--card-hover)] text-[var(--ink)]"
              : "border-transparent bg-[var(--card)] text-[var(--ink-soft)] hover:bg-[var(--card-hover)] hover:text-[var(--ink)]",
            className,
          )}
        >
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={ALL}>{all}</SelectItem>
          {items.map((i) => (
            <SelectItem key={i.id} value={i.id}>
              {i.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </label>
  );
}
