"use client";

import { useMemo, useState } from "react";

import type { MatchReport, Play } from "@/lib/types";
import { BoardCanvas } from "./board-canvas";
import { Chip, Count, Panel, Refusal, Row } from "./panel-ui";
import {
  EMPTY_FILTER,
  PLAY_OUTCOMES,
  PLAY_TYPES,
  PLAY_ZONES,
  byClock,
  filterPlays,
  PLAY_NOTE,
  playBoard,
  playCaption,
  playChoices,
  plays as matchPlays,
  type PlayFilter,
} from "./play-data";

/** The second pane of the `/` menu, for the two commands that are not a query.
 *
 *  A command with options does not insert a default and leave you to fix it in
 *  the document. It opens a panel beside the list, you set it up there, and
 *  what lands is what you asked for. The list stays put the whole time, so
 *  hovering down it flicks between panels without anything closing.
 *
 *  `/table` and `/visual` live in flow-panels.tsx — they share a flow with each
 *  other and nothing with these two.
 */

export interface PanelProps {
  report: MatchReport | null;
  /** Put a node in the document and close the menu. */
  insert: (content: Record<string, unknown>) => void;
}

// -------------------------------------------------------------------- play

export function PlayPanel({ report, insert }: PanelProps) {
  const [f, setF] = useState<PlayFilter>(EMPTY_FILTER);
  const all = useMemo(() => matchPlays(report), [report]);
  const shown = useMemo(() => filterPlays(all, f).sort(byClock), [all, f]);
  const set = (patch: Partial<PlayFilter>) => setF({ ...f, ...patch });

  if (!all.length) {
    return (
      <Panel title="Play">
        <Refusal>
          This match has no plays. A play is a moment the pipeline picked out and drew, which needs
          the ball — and the pipeline follows bodies.
        </Refusal>
      </Panel>
    );
  }

  return (
    <Panel title="Play" subtitle={`${shown.length} of ${all.length}`}>
      <Row label="What">
        <Chip on={!f.type} onClick={() => set({ type: "" })}>Any</Chip>
        {playChoices(all, "type").map((c) => (
          <Chip key={c.id} on={f.type === c.id} onClick={() => set({ type: c.id })}>
            {PLAY_TYPES[c.id] ?? c.id} <Count n={c.n} />
          </Chip>
        ))}
      </Row>
      <Row label="Where">
        <Chip on={!f.zone} onClick={() => set({ zone: "" })}>Any</Chip>
        {playChoices(all, "zone").map((c) => (
          <Chip key={c.id} on={f.zone === c.id} onClick={() => set({ zone: c.id })}>
            {PLAY_ZONES[c.id] ?? c.id} <Count n={c.n} />
          </Chip>
        ))}
      </Row>
      <Row label="Ended">
        <Chip on={!f.outcome} onClick={() => set({ outcome: "" })}>Any</Chip>
        {playChoices(all, "outcome").map((c) => (
          <Chip key={c.id} on={f.outcome === c.id} onClick={() => set({ outcome: c.id })}>
            {PLAY_OUTCOMES[c.id] ?? c.id} <Count n={c.n} />
          </Chip>
        ))}
      </Row>
      <Row label="When">
        {(["", "1", "2"] as const).map((h) => (
          <Chip key={h || "all"} on={f.period === h} onClick={() => set({ period: h })}>
            {h === "" ? "Both halves" : h === "1" ? "First" : "Second"}
          </Chip>
        ))}
        <input
          value={f.player}
          onChange={(e) => set({ player: e.target.value.replace(/\D/g, "").slice(0, 2) })}
          placeholder="Shirt no."
          aria-label="Shirt number involved"
          className="h-6 w-20 rounded-[var(--radius-xs)] border border-[var(--hairline)] bg-[var(--input-bg)] px-1.5 text-[11px] text-[var(--ink)] outline-none focus:border-[var(--accent)]"
        />
      </Row>

      {/* The results ARE the picker: clicking a moment is what inserts it. */}
      {shown.length === 0 ? (
        <Refusal>Nothing matches all of that. Loosen one — the counts say what is there.</Refusal>
      ) : (
        <div className="grid max-h-64 grid-cols-2 gap-1.5 overflow-y-auto pr-0.5">
          {shown.map((p: Play) => (
            <button
              key={p.id}
              type="button"
              title="Add to the note"
              onClick={() =>
                insert({
                  type: "pitch",
                  attrs: {
                    ...playBoard(p),
                    caption: playCaption(p, report),
                    note: PLAY_NOTE,
                    tier: "prov",
                  },
                })
              }
              className="flex flex-col gap-1 rounded-[var(--radius-sm)] border border-[var(--hairline)] p-1 text-left transition-colors hover:border-[var(--accent)]"
            >
              <span className="grid h-16 place-items-center overflow-hidden">
                <BoardCanvas board={playBoard(p)} className="block max-h-16 w-auto max-w-full" />
              </span>
              <span className="flex items-baseline gap-1 px-0.5 text-[10px]">
                <span className="font-mono text-[var(--mute)]">{p.clock}</span>
                <span className="min-w-0 flex-1 truncate text-[var(--ink-soft)]">
                  {PLAY_TYPES[p.type] ?? p.type}
                </span>
              </span>
            </button>
          ))}
        </div>
      )}
    </Panel>
  );
}

// -------------------------------------------------------------------- stat

export function StatPanel({
  stats,
  insert,
}: {
  stats: { field: string; label: string; value: string; side: string; tier: string }[];
  insert: (content: Record<string, unknown>) => void;
}) {
  const [q, setQ] = useState("");
  const shown = stats.filter((s) => `${s.label} ${s.side}`.toLowerCase().includes(q.toLowerCase()));

  return (
    <Panel title="Stat" subtitle={`${shown.length}`}>
      {stats.length === 0 ? (
        <Refusal>This match has no numbers yet.</Refusal>
      ) : (
        <>
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Find a number"
            aria-label="Find a number"
            className="h-7 w-full rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-[var(--input-bg)] px-2 text-[11.5px] text-[var(--ink)] outline-none focus:border-[var(--accent)]"
          />
          <div className="grid max-h-64 gap-0.5 overflow-y-auto pr-0.5">
            {shown.map((s) => (
              <button
                key={`${s.field}:${s.label}:${s.side}`}
                type="button"
                onClick={() => insert({ type: "stat", attrs: s })}
                className="flex items-center gap-2 rounded-[var(--radius-sm)] px-2 py-1.5 text-left text-[11.5px] text-[var(--ink-soft)] transition-colors hover:bg-[var(--card)] hover:text-[var(--ink)]"
              >
                <span className="min-w-0 flex-1 truncate capitalize">
                  {s.label}
                  {s.side ? ` · ${s.side}` : ""}
                </span>
                <span className="shrink-0 font-mono text-[11px] text-[var(--mute)]">{s.value}</span>
              </button>
            ))}
          </div>
        </>
      )}
    </Panel>
  );
}
