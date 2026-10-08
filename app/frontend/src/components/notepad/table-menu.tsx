"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Editor } from "@tiptap/react";

import { useReport, useSeason } from "@/hooks/use-matches";
import { FIGURES, buildFigure, firstChoice, parseQuerySpec } from "./figure-data";
import { presetsFor } from "./stat-presets";
import { sourceOf } from "./query-data";
import { api } from "@/lib/api";
import type { StatPreset } from "@/lib/types";
import { ALLOW } from "./lock-plugin";
import { setCellAttr } from "@tiptap/pm/tables";

import { lockedTableFromRows } from "./table-grid";
import { MenuItem, MenuRule, MenuShell, SubPanel, type Anchor } from "./menu-ui";

/** The table's menu, on right-click. Same chrome and same gating as a chart's.
 *
 *  The lock is first because it is the one thing about an imported table that
 *  is not obvious: these numbers came out of the pipeline, and turning them
 *  into something you can type over should take a deliberate act rather than a
 *  stray keystroke.
 *
 *  Everything else is the question behind the rows, gated one level deep — the
 *  figure, its column set, how many rows. A table somebody built by hand has no
 *  question, so it gets the lock and nothing else rather than branches that
 *  would refuse.
 */

export interface MenuAt {
  /** Document position of the table node. */
  pos: number;
  x: number;
  y: number;
}

interface Spec {
  kind: string;
  cols: string;
  limit: number;
}

type Branch = "figure" | "cols" | "rows" | "style" | "tone" | null;

/** HOW A TABLE IS DRESSED. Keys into the stylesheet, never styles — see
 *  globals.css for why a document names a look instead of carrying one. */
const VARIANTS = [
  { id: "", label: "Grid", hint: "Rules on every side, the app's own table." },
  { id: "report", label: "Report", hint: "Rows only, the scouting-report look." },
];

/** What a CELL is, in the report's vocabulary. `tone` is a cell attribute, so
 *  these apply to whatever is selected — one cell, or a dragged block of them. */
const TONES = [
  { id: "", label: "Plain" },
  { id: "group", label: "Group heading" },
  { id: "sub", label: "Indented" },
  { id: "shade", label: "Shaded" },
  { id: "num", label: "Number" },
  { id: "muted", label: "Secondary" },
  { id: "note", label: "Aside" },
  { id: "strong", label: "Emphasis" },
];

const ROW_STEPS = [11, 16, 22, 30];

const parseSpec = (raw: string | null): Spec | null => {
  if (!raw) return null;
  try {
    const d = JSON.parse(raw) as Record<string, unknown>;
    const kind = String(d.kind ?? "");
    // "query" is a table BUILT from the match rather than one of the three the
    // report ships. Its `cols` is a whole QuerySpec instead of a subset name,
    // which `buildFigure` already understands — so it refreshes through the
    // same path and its columns can be swapped for a preset.
    if (kind !== "query" && !FIGURES.some((f) => f.id === kind)) return null;
    return { kind, cols: String(d.cols ?? ""), limit: Number(d.limit) || 22 };
  } catch {
    return null;
  }
};

export function TableMenu({
  editor,
  at,
  onClose,
}: {
  editor: Editor;
  at: MenuAt;
  onClose: () => void;
}) {
  const { report, match } = useReport();
  const { season } = useSeason(match?.team, false);
  const [branch, setBranch] = useState<Branch>(null);
  const [anchor, setAnchor] = useState<Anchor>({ x: 0, y: 0, right: 0 });
  const sub = useRef<HTMLDivElement>(null);

  const node = editor.state.doc.nodeAt(at.pos);
  const locked = !!node?.attrs.locked;
  const variant = String(node?.attrs.variant ?? "");
  const spec = useMemo(() => parseSpec((node?.attrs.figure as string) ?? null), [node]);

  // The project's saved sets, for a built table whose columns are a free list.
  const [saved, setSaved] = useState<StatPreset[]>([]);
  useEffect(() => {
    let live = true;
    api.getPresets().then((p) => live && setSaved(p), () => {});
    return () => {
      live = false;
    };
  }, []);

  /** A built table's own question, when it has one. */
  const query = useMemo(
    () => (spec?.kind === "query" ? parseQuerySpec(spec.cols) : null),
    [spec],
  );

  /** The sets this table's SOURCE can answer — built-ins and the coach's own.
   *  Only for a built table: the three report figures name a subset instead,
   *  and `built.choices` already offers those. */
  const presets = useMemo(() => {
    if (!query) return [];
    const src = sourceOf(query.source, report);
    return presetsFor([{ id: src.id, measures: src.measures }], report, saved);
  }, [query, report, saved]);

  const built = useMemo(
    () =>
      spec ? buildFigure({ report, season, kind: spec.kind, cols: spec.cols, limit: spec.limit }) : null,
    [spec, report, season],
  );

  if (!node) return null;

  const setAttr = (attrs: Record<string, unknown>) => {
    editor.view.dispatch(
      Object.entries(attrs)
        .reduce((tr, [k, v]) => tr.setNodeAttribute(at.pos, k, v), editor.state.tr)
        .setMeta(ALLOW, true),
    );
  };

  /** Ask the question again and replace the rows in place.
   *
   *  Never a background fetch, like every other figure in this notepad: the
   *  numbers in a saved note change when somebody presses this and at no other
   *  time. Replacing the whole table is what keeps a re-run honest — a match
   *  re-processed with a player fewer would otherwise leave a stale row behind.
   */
  const refresh = (next: Partial<Spec>) => {
    if (!spec) return;
    const s = { ...spec, ...next };
    const f = buildFigure({ report, season, kind: s.kind, cols: s.cols, limit: s.limit });
    if (!f.rows.length) return;
    const table = lockedTableFromRows({
      head: f.head,
      rows: f.rows.map((r) => [r.name, ...r.cells]),
      bars: f.rows.map((r) => r.split ?? null),
      figure: JSON.stringify(s),
    }) as unknown as Record<string, unknown>;
    const fresh = editor.state.schema.nodeFromJSON({
      ...table,
      attrs: { ...(table.attrs as object), locked },
    });
    editor.view.dispatch(
      editor.state.tr.replaceWith(at.pos, at.pos + node.nodeSize, fresh).setMeta(ALLOW, true),
    );
    onClose();
  };

  /** Set `tone` on every selected cell. prosemirror-tables already has the
   *  command; the only thing added here is the lock plugin's permission, so a
   *  locked table can still be RESTYLED - holding the numbers still is not the
   *  same as freezing how they look. */
  const setTone = (tone: string) => {
    setCellAttr("tone", tone)(editor.state, (tr) => editor.view.dispatch(tr.setMeta(ALLOW, true)));
    onClose();
  };

  const open = (b: Branch) => (rect: Anchor) => {
    setAnchor(rect);
    setBranch(b);
  };

  return (
    <>
      <MenuShell at={at} onClose={onClose} label="Table" extraInside={[sub]}>
        <MenuItem
          onPick={() => {
            setAttr({ locked: !locked });
            onClose();
          }}
          onHover={() => setBranch(null)}
        >
          {locked ? "Unlock" : "Lock"}
        </MenuItem>

        <MenuRule />
        <MenuItem more on={branch === "style"} onPick={open("style")} onHover={open("style")}>
          Style
        </MenuItem>
        <MenuItem more on={branch === "tone"} onPick={open("tone")} onHover={open("tone")}>
          Cells
        </MenuItem>

        {spec && (
          <>
            <MenuRule />
            {/* Not offered for a BUILT table. Its question is a whole spec —
                source, breakdown, measures — and "Player table" would silently
                discard it for a named subset of somebody else's columns. */}
            {!query && (
              <MenuItem more on={branch === "figure"} onPick={open("figure")} onHover={open("figure")}>
                Figure
              </MenuItem>
            )}
            {((built?.choices.length ?? 0) > 1 || presets.length > 0) && (
              <MenuItem more on={branch === "cols"} onPick={open("cols")} onHover={open("cols")}>
                Columns
              </MenuItem>
            )}
            {(spec.kind === "playerTable" || spec.kind === "query") && (
              <MenuItem more on={branch === "rows"} onPick={open("rows")} onHover={open("rows")}>
                Rows
              </MenuItem>
            )}
            <MenuRule />
            <MenuItem onPick={() => refresh({})} onHover={() => setBranch(null)}>
              Take the figures again
            </MenuItem>
          </>
        )}
      </MenuShell>

      {branch === "style" && (
        <SubPanel anchor={anchor} width="14rem" panelRef={sub}>
          {VARIANTS.map((v) => (
            <MenuItem
              key={v.id || "grid"}
              on={variant === v.id}
              onPick={() => {
                setAttr({ variant: v.id });
                onClose();
              }}
            >
              {v.label}
            </MenuItem>
          ))}
        </SubPanel>
      )}

      {branch === "tone" && (
        <SubPanel anchor={anchor} width="12rem" panelRef={sub}>
          {TONES.map((t) => (
            <MenuItem key={t.id || "plain"} onPick={() => setTone(t.id)}>
              {t.label}
            </MenuItem>
          ))}
        </SubPanel>
      )}

      {branch === "figure" && (
        <SubPanel anchor={anchor} width="12rem" panelRef={sub}>
          {FIGURES.map((f) => (
            <MenuItem
              key={f.id}
              on={spec?.kind === f.id}
              hint={f.hint}
              // A subset name means nothing to a different kind, so changing the
              // kind carries its own first choice with it.
              onPick={() => refresh({ kind: f.id, cols: firstChoice(f.id, report) })}
            >
              {f.label}
            </MenuItem>
          ))}
        </SubPanel>
      )}

      {branch === "cols" && (
        <SubPanel anchor={anchor} width="13rem" panelRef={sub}>
          {query
            ? presets.map((p) => (
                <MenuItem
                  key={p.id}
                  on={p.cols.join(",") === query.measures.join(",")}
                  hint={p.saved ? "Saved for this project" : undefined}
                  onPick={() =>
                    refresh({
                      cols: JSON.stringify({ ...query, source: p.source, measures: p.cols }),
                    })
                  }
                >
                  {p.label}
                </MenuItem>
              ))
            : built?.choices.map((c) => (
                <MenuItem key={c.id} on={spec?.cols === c.id} onPick={() => refresh({ cols: c.id })}>
                  {c.label}
                </MenuItem>
              ))}
        </SubPanel>
      )}

      {branch === "rows" && (
        <SubPanel anchor={anchor} width="9rem" panelRef={sub}>
          {ROW_STEPS.map((n) => (
            <MenuItem
              key={n}
              on={(query ? query.limit : spec?.limit) === n}
              onPick={() =>
                refresh(
                  query
                    ? { cols: JSON.stringify({ ...query, limit: n }), limit: n }
                    : { limit: n },
                )
              }
            >
              Top {n}
            </MenuItem>
          ))}
        </SubPanel>
      )}
    </>
  );
}
