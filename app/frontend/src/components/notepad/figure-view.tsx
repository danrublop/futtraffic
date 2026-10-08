"use client";

import { useMemo } from "react";
import { NodeViewWrapper, type NodeViewProps } from "@tiptap/react";
import { RefreshCw } from "lucide-react";

import { useReport, useSeason } from "@/hooks/use-matches";
import { cn } from "@/lib/utils";
import { tierNote } from "./stat-node";
import { FIGURES, buildFigure, figureTitle, firstChoice, type FigureRow } from "./figure-data";

/** A report figure, live in the editor.
 *
 *  Two pickers: which figure, then which subset of it. Changing either rebuilds
 *  and re-freezes in one transaction, because the picked figures ARE the
 *  document; see figure-node.ts on why they are frozen and not fetched on open.
 *
 *  The refusals carry the weight here. On a real match every feed-backed figure
 *  is empty, so the sentence that says WHY — and points at the one subset that
 *  does work — is the thing a coach actually reads.
 */
export function FigureView({ node, updateAttributes, editor }: NodeViewProps) {
  const a = node.attrs as {
    kind: string;
    title: string;
    cols: string;
    limit: number;
    head: string[];
    rows: FigureRow[];
    note: string;
    tier: string;
  };
  const editable = editor.isEditable;
  const { report, match } = useReport();
  // Only a built table can span matches; the three report figures are one
  // match each and the season fetch stays off for them.
  const { season } = useSeason(match?.team, a.kind === "query");

  const built = useMemo(
    () => buildFigure({ report, season, kind: a.kind, cols: a.cols, limit: a.limit }),
    [report, season, a.kind, a.cols, a.limit],
  );
  const anySplit = a.rows.some((r) => r.split != null);

  /** Apply a picker change and re-freeze in one transaction — two would put a
   *  half-changed figure in the undo stack. */
  const change = (patch: Partial<typeof a>) => {
    const next = { ...a, ...patch };
    const f = buildFigure({ report, season, kind: next.kind, cols: next.cols, limit: next.limit });
    updateAttributes({
      ...patch,
      head: f.head,
      rows: f.rows,
      note: f.refusal ?? f.note,
      tier: f.tier,
      // Only overwrite a title the user has not written over.
      ...(a.title && a.title !== figureTitle(a.kind, a.cols, report)
        ? {}
        : { title: figureTitle(next.kind, next.cols, report) }),
    });
  };

  const stale =
    !built.refusal &&
    built.rows.length > 0 &&
    JSON.stringify(built.rows) !== JSON.stringify(a.rows);

  return (
    <NodeViewWrapper className="ft-figure-wrap">
      <div className="ft-figure" data-kind={a.kind}>
        {a.title && <h4 className="ft-figure__title">{a.title}</h4>}

        {a.rows.length > 0 ? (
          <div className="ft-figure__scroll">
            <table className="ft-figure__table">
              <thead>
                <tr>
                  {a.head.map((h, i) => (
                    <th key={i}>{h}</th>
                  ))}
                  {anySplit && <th />}
                </tr>
              </thead>
              <tbody>
                {a.rows.map((r, i) => (
                  <tr key={i}>
                    <th scope="row">
                      <span className="ft-figure__who">
                        {r.side && <span className={`ft-figure__dot ft-figure__dot--${r.side}`} />}
                        {r.name}
                      </span>
                    </th>
                    {r.cells.map((c, j) => (
                      <td key={j}>{c}</td>
                    ))}
                    {r.split != null && (
                      <td className="ft-figure__splitcell">
                        <span className="ft-figure__split">
                          <span style={{ width: `${(r.split * 100).toFixed(1)}%` }} />
                        </span>
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="ft-figure__empty">{a.note || built.refusal || "Nothing to show yet."}</p>
        )}

        {a.rows.length > 0 && (a.note || tierNote(a.tier)) && (
          <p className="ft-figure__note">{[tierNote(a.tier), a.note].filter(Boolean).join(" · ")}</p>
        )}
      </div>

      {editable && (
        <div className="ft-controls mt-1.5 flex flex-wrap items-center gap-1 text-[11px]">
          {/* The kind and subset chips switch between the three figures the
              REPORT ships. A built table is not one of them — its question
              lives in `cols` as a spec, and "Player table" would silently
              discard it — so it gets Refresh and nothing else. */}
          {a.kind !== "query" && FIGURES.map((f) => (
            <Chip
              key={f.id}
              on={a.kind === f.id}
              title={f.hint}
              // A subset name means nothing to a different kind, so changing
              // the kind carries its own first choice with it.
              onClick={() => change({ kind: f.id, cols: firstChoice(f.id, report) })}
            >
              {f.label}
            </Chip>
          ))}

          {a.kind !== "query" && (
            <span aria-hidden className="mx-0.5 h-4 w-px bg-[var(--hairline)]" />
          )}

          {built.choices.map((c) => (
            <Chip
              key={c.id}
              on={a.cols === c.id}
              title={
                c.id === "physical"
                  ? "Measured from the tracks — works on every finished match"
                  : "From the entered match feed — absent unless somebody recorded one"
              }
              onClick={() => change({ cols: c.id })}
            >
              {c.label}
            </Chip>
          ))}

          {a.kind !== "query" && (
            <span aria-hidden className="mx-0.5 h-4 w-px bg-[var(--hairline)]" />
          )}

          {a.kind === "playerTable" && (
          <select
            aria-label="Rows"
            value={a.limit}
            onChange={(e) => change({ limit: Number(e.target.value) })}
            className="h-6 cursor-pointer rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-transparent px-1 text-[11px] text-[var(--graphite)] outline-none hover:text-[var(--ink)]"
          >
            {[11, 16, 22, 30].map((n) => (
              <option key={n} value={n} className="bg-[var(--panel)] text-[var(--ink)]">
                Top {n}
              </option>
            ))}
          </select>
          )}

          <span className="flex-1" />

          {/* Never a background fetch. The numbers in a saved note change only
              when somebody presses this. */}
          <Chip
            title={
              stale
                ? "This match has been re-processed since these figures were taken"
                : "Take the figures again"
            }
            onClick={() => change({})}
          >
            <RefreshCw size={11} strokeWidth={1.7} />
            {stale ? "Numbers have changed — refresh" : "Refresh"}
          </Chip>
        </div>
      )}
    </NodeViewWrapper>
  );
}

function Chip({
  on,
  title,
  onClick,
  children,
}: {
  on?: boolean;
  title?: string;
  onClick?: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      className={cn(
        "flex h-6 items-center gap-1 rounded-[var(--radius-pill)] border px-2 text-[11px] transition-colors",
        on
          ? "border-[var(--hairline-strong)] bg-[var(--card)] text-[var(--ink)]"
          : "border-[var(--hairline)] text-[var(--graphite)] hover:bg-[var(--card)] hover:text-[var(--ink)]",
      )}
    >
      {children}
    </button>
  );
}
