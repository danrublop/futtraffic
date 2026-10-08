"use client";

import { Plus, Trash2 } from "lucide-react";

import { cn } from "@/lib/utils";
import { storedResult, type QueryAttrs } from "./query-node";
import { fmt, type QueryRow } from "./query-data";
import { CATEGORY_COLORS, SERIES_COLORS, SINGLE_COLOR } from "./chart-svg";

/** The panels a chart's menu is made of.
 *
 *  This file used to BE the editor: four tabs, everything in them on screen at
 *  once, expanded under the figure. That shell is gone — chart-menu.tsx gates
 *  the same controls behind a right-click cascade, the way the table's menu
 *  does, so a document made of prose stops carrying a settings dialog in the
 *  middle of it.
 *
 *  What is left is the parts, which the menu hangs off its submenus: the grid
 *  of numbers, the colours, and the small labelled row and select the question
 *  panel is built from. Keeping them here rather than moving them means the
 *  diff that removed the shell removed only the shell.
 */


/** The colours, per series.
 *
 *  THE DEFAULTS ARE NOT ARBITRARY and the panel says so. `SERIES_COLORS` was
 *  chosen against a lightness band, a chroma floor, a colour-vision separation
 *  and a contrast check on both surfaces — there is a script that re-runs them.
 *  Overriding one is allowed, because a coach's club has colours and a report
 *  that cannot wear them is a report they will rebuild in PowerPoint. But the
 *  validated set stays one click away and is what an untouched chart uses.
 *
 *  A pie is the one form where the CATEGORIES carry the colour rather than the
 *  series, so that is what it offers instead — six, matching the six slices it
 *  will draw before a seventh folds into "Other".
 */
export function Colours({
  a,
  onChange,
}: {
  a: QueryAttrs;
  onChange: (patch: Partial<QueryAttrs>) => void;
}) {
  const byCategory = a.render === "pie" || a.render === "doughnut";
  const shown = storedResult(a);
  const slots = byCategory
    ? shown.rows.slice(0, CATEGORY_COLORS.length).map((row, i) => ({
        label: row.label,
        fallback: CATEGORY_COLORS[i],
      }))
    : (a.series?.length ? a.series : ["All"]).map((name, i) => ({
        label: name,
        fallback: a.series?.length > 1 ? SERIES_COLORS[i] ?? SERIES_COLORS[0] : SINGLE_COLOR,
      }));

  const set = (i: number, value: string | null) => {
    const next = [...(a.colors ?? [])];
    while (next.length < slots.length) next.push("");
    next[i] = value ?? "";
    // Trailing blanks mean "the default", so they are dropped rather than
    // stored — an attribute of empty strings is noise in every saved note.
    while (next.length && !next[next.length - 1]) next.pop();
    onChange({ colors: next });
  };

  return (
    <div className="flex flex-col gap-2 p-2.5">
      {slots.map((slot, i) => {
        const current = a.colors?.[i] || slot.fallback;
        const overridden = !!a.colors?.[i];
        return (
          <div key={i} className="flex items-center gap-2">
            <span className="w-28 shrink-0 truncate text-[11.5px] text-[var(--graphite)]" title={slot.label}>
              {slot.label}
            </span>
            <input
              type="color"
              aria-label={`Colour for ${slot.label}`}
              value={current}
              onChange={(e) => set(i, e.target.value)}
              className="h-6 w-9 cursor-pointer rounded-[var(--radius-xs)] border border-[var(--hairline)] bg-transparent p-0.5"
            />
            <input
              aria-label={`Hex for ${slot.label}`}
              value={current}
              spellCheck={false}
              onChange={(e) => {
                const v = e.target.value.trim();
                if (/^#[0-9a-f]{6}$/i.test(v)) set(i, v);
              }}
              className="h-6 w-20 rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-transparent px-1.5 font-mono text-[11px] text-[var(--ink)] outline-none focus:border-[var(--accent)]"
            />
            {(byCategory ? CATEGORY_COLORS : [...SERIES_COLORS, SINGLE_COLOR]).map((c) => (
              <button
                key={c}
                type="button"
                title={c}
                aria-label={`Set ${slot.label} to ${c}`}
                onClick={() => set(i, c)}
                style={{ background: c }}
                className="h-4 w-4 rounded-full border border-[var(--hairline-strong)]"
              />
            ))}
            {overridden && (
              <button
                type="button"
                onClick={() => set(i, null)}
                className="text-[11px] text-[var(--graphite)] underline hover:text-[var(--ink)]"
              >
                Default
              </button>
            )}
          </div>
        );
      })}
      <p className="pt-0.5 text-[10.5px] leading-snug text-[var(--mute)]">
        The defaults passed a lightness, chroma, colour-vision and contrast
        check against both light and dark. A colour picked here is yours and is
        not checked — which is fine for a club&rsquo;s own palette, and worth
        knowing before a chart goes to someone reading it on paper.
      </p>
    </div>
  );
}

/** The numbers, as a spreadsheet.
 *
 *  Editing one recomputes the axis maximum and re-formats the cell through the
 *  same `fmt` the query uses, so a typed number and a measured one are printed
 *  identically — otherwise a hand-edited row would be visibly a different kind
 *  of thing from the rows beside it, which is a lie in the other direction.
 */
export function DataGrid({
  a,
  onChange,
}: {
  a: QueryAttrs;
  onChange: (patch: Partial<QueryAttrs>) => void;
}) {
  const rows = a.rows ?? [];
  const width = Math.max(1, a.series?.length || 1);
  const measure = { id: a.measure, label: a.label || a.measure, agg: "sum" as const, dp: a.dp ?? 0, unit: a.unit || undefined };

  const commit = (next: QueryRow[]) =>
    onChange({
      rows: next,
      max: Math.max(0, ...next.flatMap((r) => r.values)),
      // Detached from the question. Said out loud rather than left implicit,
      // because the whole product rests on being able to tell a measured
      // number from a typed one.
      custom: true,
      extra: null,
      note: "Typed in by hand, not measured.",
      tier: "prov",
    });

  const setCell = (ri: number, ci: number, raw: string) => {
    const v = Number(raw);
    if (!Number.isFinite(v)) return;
    const next = rows.map((r, i) =>
      i !== ri
        ? r
        : {
            ...r,
            values: r.values.map((old, j) => (j === ci ? v : old)),
            text: r.text.map((old, j) => (j === ci ? fmt(v, measure) : old)),
          },
    );
    commit(next);
  };

  const setLabel = (ri: number, label: string) =>
    commit(rows.map((r, i) => (i === ri ? { ...r, label: label.slice(0, 60) } : r)));

  const addRow = () =>
    commit([
      ...rows,
      {
        label: `Row ${rows.length + 1}`,
        values: Array.from({ length: width }, () => 0),
        text: Array.from({ length: width }, () => fmt(0, measure)),
      },
    ]);

  return (
    <div className="flex flex-col gap-2">
      <div className="overflow-x-auto rounded-[var(--radius-md)] border border-[var(--hairline)]">
        <table className="w-full border-collapse text-[11.5px]">
          <thead>
            <tr>
              <th className="border-b border-[var(--hairline)] bg-[var(--card)] px-2 py-1.5 text-left text-[10px] font-medium tracking-wide text-[var(--mute)] uppercase">
                {a.dimLabel || "Row"}
              </th>
              {Array.from({ length: width }, (_, i) => (
                <th
                  key={i}
                  className="border-b border-[var(--hairline)] bg-[var(--card)] px-2 py-1.5 text-left text-[10px] font-medium tracking-wide text-[var(--mute)] uppercase"
                >
                  {a.series?.[i] ?? a.label ?? "Value"}
                </th>
              ))}
              <th className="w-8 border-b border-[var(--hairline)] bg-[var(--card)]" />
            </tr>
          </thead>
          <tbody>
            {rows.map((r, ri) => (
              <tr key={ri}>
                <td className="border-b border-[var(--hairline)] p-0">
                  <input
                    value={r.label}
                    aria-label={`Row ${ri + 1} name`}
                    onChange={(e) => setLabel(ri, e.target.value)}
                    className="w-full bg-transparent px-2 py-1.5 text-[var(--ink)] outline-none focus:bg-[var(--card)]"
                  />
                </td>
                {Array.from({ length: width }, (_, ci) => (
                  <td key={ci} className="border-b border-[var(--hairline)] p-0">
                    <input
                      // `inputMode` and not `type=number`: a spinner in a
                      // spreadsheet cell is a spinner nobody uses, and the
                      // browser's own number parsing swallows a stray comma.
                      inputMode="decimal"
                      value={String(r.values[ci] ?? 0)}
                      aria-label={`Row ${ri + 1} value ${ci + 1}`}
                      onChange={(e) => setCell(ri, ci, e.target.value)}
                      className="w-full bg-transparent px-2 py-1.5 text-right font-mono text-[var(--ink)] outline-none focus:bg-[var(--card)]"
                    />
                  </td>
                ))}
                <td className="border-b border-[var(--hairline)] text-center">
                  <button
                    type="button"
                    aria-label={`Delete row ${ri + 1}`}
                    title="Delete row"
                    disabled={rows.length <= 1}
                    onClick={() => commit(rows.filter((_, i) => i !== ri))}
                    className="grid h-6 w-6 place-items-center rounded-[var(--radius-xs)] text-[var(--mute)] hover:text-[var(--danger)] disabled:opacity-30"
                  >
                    <Trash2 size={11} strokeWidth={1.8} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={addRow}
          className="flex items-center gap-1 rounded-[var(--radius-sm)] border border-[var(--hairline)] px-2 py-1 text-[11px] text-[var(--graphite)] hover:text-[var(--ink)]"
        >
          <Plus size={11} strokeWidth={1.9} /> Row
        </button>
        <span className="text-[10.5px] leading-snug text-[var(--mute)]">
          Typing here detaches the chart from its question — the numbers become yours, and
          Refresh stops being offered rather than quietly overwriting them.
        </span>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------- pieces

export function Line({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-1">
      <span className="w-24 shrink-0 text-[9.5px] leading-tight tracking-wide text-[var(--mute)] uppercase">
        {label}
      </span>
      {children}
    </div>
  );
}

export function Sel({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: { id: string; label: string }[];
  onChange: (v: string) => void;
}) {
  return (
    <Line label={label}>
      <select
        aria-label={label}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="h-7 min-w-0 flex-1 cursor-pointer rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-transparent px-1.5 text-[11.5px] text-[var(--ink)] outline-none hover:border-[var(--hairline-strong)]"
      >
        {options.map((o) => (
          <option key={o.id} value={o.id} className="bg-[var(--panel)] text-[var(--ink)]">
            {o.label}
          </option>
        ))}
      </select>
    </Line>
  );
}

export function Chip({
  on,
  disabled,
  title,
  onClick,
  children,
}: {
  on?: boolean;
  disabled?: boolean;
  title?: string;
  onClick?: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "flex h-6 items-center gap-1 rounded-[var(--radius-pill)] border px-2 text-[11px] whitespace-nowrap transition-colors",
        disabled
          ? "cursor-not-allowed border-[var(--hairline)] text-[var(--mute)] opacity-40"
          : on
            ? "border-[var(--hairline-strong)] bg-[var(--card)] text-[var(--ink)]"
            : "border-[var(--hairline)] text-[var(--graphite)] hover:bg-[var(--card)] hover:text-[var(--ink)]",
      )}
    >
      {children}
    </button>
  );
}
