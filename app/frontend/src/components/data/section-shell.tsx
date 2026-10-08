"use client";

import { TableCell } from "@/components/ui/table";
import { cn } from "@/lib/utils";

/** THE PIECES EVERY DATA SECTION IS BUILT FROM.
 *
 *  Four sections — Players, Teams, Opponents, Matches — used to navigate three
 *  different ways: click-a-row-then-inner-tabs, pick-a-chip-then-a-stack, and
 *  dropdowns-plus-a-side-panel. Learning the page meant learning it three
 *  times. These are the pieces that make them read as one surface.
 *
 *  PIECES, NOT A COMPONENT THAT OWNS THE LAYOUT. There is deliberately no
 *  `DetailPane` here. The three detail bodies are an inline profile beside the
 *  table, a set of Radix tabs, and a navigation away to the Report — they have
 *  nothing in common but a name, and a component over them would be a switch
 *  statement in a disguise. Each section writes its own detail inline and takes
 *  its shared LOOK from `SectionShell` and `FilterBar`.
 *
 *  The precedent is `components/sort-head.tsx`: extracted as a shared piece,
 *  used by every table on this surface, generic about nothing.
 */

export function SectionShell({
  title,
  note,
  count,
  filters,
  actions,
  children,
}: {
  title: string;
  /** One line saying what this section IS. Not decoration: on a surface where
   *  measured, entered and bound numbers sit next to each other, saying which
   *  kind you are looking at is the product's central obligation. */
  note?: string;
  /** "22 players", "10 matches". Rendered beside the title, muted. */
  count?: string;
  filters?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="grid min-w-0 gap-3">
      <div className="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 className="text-sm font-medium">{title}</h2>
        {count && <span className="text-muted-foreground text-xs">{count}</span>}
        {actions && <span className="ml-auto flex items-center gap-2">{actions}</span>}
      </div>
      {note && (
        <p className="text-muted-foreground max-w-[92ch] -mt-1 text-xs leading-relaxed">{note}</p>
      )}
      {filters && <FilterBar>{filters}</FilterBar>}
      {children}
    </div>
  );
}

/** The row a section's narrowing controls sit in.
 *
 *  Every control in here is LOCAL to its section. None of them writes the
 *  global team scope — that belongs to the header switcher alone, because a
 *  filter that silently re-scopes the sidebar and the notepad is
 *  action-at-a-distance, and this surface had exactly that bug.
 */
export function FilterBar({ children }: { children: React.ReactNode }) {
  return <div className="flex min-w-0 flex-wrap items-center gap-2">{children}</div>;
}

/** A labelled control in the filter bar. */
export function FilterField({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="flex items-center gap-1.5">
      <span className="text-muted-foreground text-xs">{label}</span>
      {children}
    </label>
  );
}

/** A MEASURED cell: the number, or an em dash.
 *
 *  This existed twice in one file, six hundred lines apart, as `Num` and
 *  `Measured` — which is the clearest sign that file had stopped being one
 *  thing.
 *
 *  NEVER a zero for an absent value. A player nobody has bound to a track has
 *  not run zero kilometres; there is simply no reading. The em dash and the
 *  zero mean different things and the tables are read by people making
 *  selection decisions, so conflating them is not a rounding error.
 */
export function MeasuredCell({
  v,
  dp,
  unit = "",
  className,
}: {
  v?: number | null;
  dp: number;
  unit?: string;
  className?: string;
}) {
  return (
    <TableCell className={cn("readout text-right whitespace-nowrap", className)}>
      {v == null ? (
        <span className="text-muted-foreground">—</span>
      ) : (
        `${v.toFixed(dp)}${unit}`
      )}
    </TableCell>
  );
}

/** ALTERNATING ROWS, as one rule every table on this surface uses.
 *
 *  `:not(:hover)` is load-bearing, not tidiness. The zebra selector
 *  (`.zebra tbody tr:nth-child(odd)`) is more specific than a row's own
 *  `hover:bg-...` utility, so without it the stripe would win over the hover
 *  highlight and every other row would stop responding to the pointer — on a
 *  surface where rows are clickable, that reads as half the table being dead.
 */
export const ZEBRA =
  "[&_tbody_tr:nth-child(odd):not(:hover)]:bg-[var(--card)]";

/** THE FIRST ROW OF EVERY DATA TABLE, and the only copy of its look.
 *
 *  Players, Opponents, Squads, All matches and every sub-tab under them are
 *  one surface as far as a reader is concerned - they sit behind the same row
 *  of chips and answer the same question about different subjects. A header
 *  that was 11px here and 12px there, 44px tall on one and 40 on the next,
 *  made them read as separate products stitched together.
 *
 *  So the header's SIZE lives in here now, not in each table. Horizontal
 *  padding does not: ten stat columns want 3px and six library columns do not,
 *  and that is a real density difference rather than a drift.
 *
 *  THE HEIGHT IS TWO LINES WHETHER THE LABEL WRAPS OR NOT. These tables swap
 *  their columns under a row of tabs: Goalkeeping's labels all fit on one line
 *  and Duels' wrap to two, so a header sized to its own text makes the table -
 *  and the first row of numbers under it - jump by a line as you move between
 *  tabs. 2.5em is two lines at leading-tight, +1rem for the py-2, +1px for the
 *  cell's own bottom border - all three are inside the box under border-box,
 *  and leaving the border out is what made a two-line header 47px against a
 *  one-line header's 46: the text did not fit the floor and grew past it.
 *  Height on a table cell is a floor, so a rare three-line label still grows.
 *
 *  Sticky, on the card's own ground, so the rows pass behind the header rather
 *  than through it. --slate for the text: it clears AA on the card where
 *  --mute did not. */
export const DATA_HEAD =
  "[&_thead_th]:sticky [&_thead_th]:top-0 [&_thead_th]:z-20 [&_thead_th]:bg-card " +
  // The CELLS draw the rule; the row must not draw a second one under them.
  // Two stacked 1px borders are both a brighter line than either table meant
  // and a row one pixel taller than the other tables' - which is exactly the
  // kind of drift this constant exists to stop.
  "[&_thead_tr]:border-0 " +
  "[&_thead_th]:border-b [&_thead_th]:border-[var(--grid-line)] " +
  "[&_thead_th]:h-[calc(2.5em+1rem+1px)] [&_thead_th]:py-2 " +
  "[&_thead_th]:text-[12px] [&_thead_th]:leading-tight [&_thead_th]:font-normal " +
  "[&_thead_th]:text-[var(--slate)]";

/** THE TABLE DRESS the Data surface's own tables wear: the shared header,
 *  plus the body's own rules - no line between rows, and a taller cell than
 *  the shadcn default because these rows carry two lines in their first
 *  column. The match library and the club editor take DATA_HEAD alone; their
 *  bodies are a different shape. */
export const DATA_TABLE = DATA_HEAD + " [&_tbody_tr]:border-0 [&_tbody_td]:py-2.5";
