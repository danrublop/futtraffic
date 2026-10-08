"use client";

import { useRef, useState } from "react";
import type { MatchReport, Season } from "@/lib/types";
import { Colours, DataGrid, Sel } from "./chart-editor";
import { storedResult, type QueryAttrs } from "./query-node";
import { SORTS, defaults, queryTitle, runQuery, sourceOf, sources, teamsOf, type SortBy } from "./query-data";
import { VISUALS, VISUAL_GROUPS, blockedBy, visualsFor } from "./visuals";
import { MenuItem, MenuRule, MenuShell, SubPanel, type Anchor } from "./menu-ui";

/** A chart's menu, gated the way a spreadsheet's is.
 *
 *  The controls used to be four tabs with everything in them on screen at once.
 *  That is a control panel, and it is the thing this notepad keeps taking apart:
 *  a document made of prose should not carry a settings dialog in the middle of
 *  it. So the root here is a SHORT LIST of what can be changed, and picking one
 *  opens its choices in a panel to the right. Nothing is on screen until it is
 *  asked for.
 *
 *  Same chrome as the table's menu, down to the corner radius — see menu-ui.tsx
 *  for why one gesture has to give one kind of answer everywhere.
 *
 *  The picture is two deep on purpose: the FAMILY first (Column, Bar, Line,
 *  Parts, On the pitch), then the form inside it. Eighteen visuals in one list
 *  is a wall; five families of three or four is a choice.
 */

type Branch =
  | "chart" | "family" | "question" | "data" | "colours" | "look" | "width" | "sort" | "caption" | null;

/** HOW THE FIGURE IS DRESSED, which is not what it says. A key into the
 *  stylesheet's chart themes — see globals.css. The marks carry their colour
 *  as an attribute and a class per series, so a theme is a stylesheet rather
 *  than a second drawing routine. */
const THEMES = [
  { id: "", label: "App", hint: "The product's own palette." },
  { id: "report", label: "Report", hint: "Scouting-report red, hollow second series, dashed grid." },
];

export function ChartMenu({
  attrs,
  report,
  season,
  at,
  onChange,
  onClose,
  onReload,
}: {
  attrs: QueryAttrs;
  report: MatchReport | null;
  season: Season | null;
  at: { x: number; y: number };
  onChange: (patch: Partial<QueryAttrs>) => void;
  onClose: () => void;
  onReload: () => void;
}) {
  const a = attrs;
  const [branch, setBranch] = useState<Branch>(null);
  const [family, setFamily] = useState<string | null>(null);
  const [anchor, setAnchor] = useState({ x: 0, y: 0, right: 0 });
  const [famAnchor, setFamAnchor] = useState({ x: 0, y: 0, right: 0 });
  const sub = useRef<HTMLDivElement>(null);
  const sub2 = useRef<HTMLDivElement>(null);

  const src = sourceOf(a.source, report);
  const teams = teamsOf(a.source, report, season);
  const shown = a.rows?.length ? storedResult(a) : null;
  const gallery = visualsFor(a.source);
  const groups = VISUAL_GROUPS.filter((g) => gallery.some((v) => v.group === g));

  /** Re-ask the question and re-freeze, exactly as the tabbed editor did.
   *  Every change routes through here so a chart cannot end up showing rows
   *  from one question under the title of another. */
  const ask = (patch: Partial<QueryAttrs>) => {
    const next = { ...a, ...patch };
    const r = runQuery({
      report,
      season,
      source: next.source,
      dim: next.dim,
      measure: next.measure,
      split: next.split,
      limit: next.limit,
      team: next.team,
      render: next.render,
      keep: next.keep,
      sort: next.sort as SortBy,
      matchIds: next.matchIds?.length ? next.matchIds : undefined,
    });
    onChange({
      ...patch,
      rows: r.rows,
      series: r.series,
      max: r.max,
      extra: r.extra ?? null,
      note: r.refusal ?? r.note,
      tier: r.tier,
      unit: r.measure.unit ?? "",
      dp: r.measure.dp,
      label: r.measure.label,
      dimLabel: r.dim.label,
      ordered: r.ordered,
      custom: false,
      ...(a.title && a.title !== queryTitle(storedResult(a), a.split, a.render)
        ? {}
        : { title: queryTitle(r, next.split, next.render) }),
    });
  };

  /** Hover and click do the same thing: show that branch. Toggling on a second
   *  click would fight the hover that just opened it. */
  const open = (b: Branch) => (rect: Anchor) => {
    setAnchor(rect);
    setBranch(b);
    setFamily(null);
  };

  const current = VISUALS.find((v) => v.id === a.render);

  return (
    <>
      <MenuShell at={at} onClose={onClose} label="Chart" extraInside={[sub, sub2]}>
        <MenuItem more on={branch === "chart"} onPick={open("chart")} onHover={open("chart")}>
          Picture
        </MenuItem>
        <MenuItem more on={branch === "question"} onPick={open("question")} onHover={open("question")}>
          Question
        </MenuItem>
        <MenuItem more on={branch === "data"} onPick={open("data")} onHover={open("data")}>
          Numbers
        </MenuItem>
        <MenuItem more on={branch === "colours"} onPick={open("colours")} onHover={open("colours")}>
          Colours
        </MenuItem>
        <MenuItem more on={branch === "look"} onPick={open("look")} onHover={open("look")}>
          Look
        </MenuItem>
        <MenuItem more on={branch === "sort"} onPick={open("sort")} onHover={open("sort")}>
          Sort
        </MenuItem>
        <MenuItem more on={branch === "width"} onPick={open("width")} onHover={open("width")}>
          Width
        </MenuItem>
        <MenuItem more on={branch === "caption"} onPick={open("caption")} onHover={open("caption")}>
          Caption
        </MenuItem>
        <MenuRule />
        <MenuItem
          disabled={a.custom}
          hint={
            a.custom
              ? "These numbers were typed in. Re-asking the question would throw them away."
              : undefined
          }
          onPick={() => {
            if (src.scope === "season") onReload();
            ask({});
            onClose();
          }}
          // A leaf, so hovering it puts away whatever branch was open.
          onHover={() => setBranch(null)}
        >
          Take the figures again
        </MenuItem>
      </MenuShell>

      {branch === "chart" && (
        <SubPanel anchor={anchor} width="13rem" panelRef={sub}>
          {groups.map((g) => (
            <MenuItem
              key={g}
              more
              on={family === g || current?.group === g}
              onPick={(rect) => {
                setFamAnchor(rect);
                setFamily(g);
              }}
              onHover={(rect) => {
                setFamAnchor(rect);
                setFamily(g);
              }}
            >
              {g}
            </MenuItem>
          ))}
        </SubPanel>
      )}

      {branch === "chart" && family && (
        <SubPanel anchor={famAnchor} width="14rem" panelRef={sub2}>
          {gallery
            .filter((v) => v.group === family)
            .map((v) => {
              const why = blockedBy(v, shown, a.source);
              return (
                <MenuItem
                  key={v.id}
                  on={a.render === v.id}
                  disabled={!!why}
                  hint={why ?? v.hint}
                  onPick={() => {
                    ask({ render: v.id });
                    onClose();
                  }}
                >
                  {v.label}
                </MenuItem>
              );
            })}
          {gallery.filter((v) => v.group === family).every((v) => blockedBy(v, shown, a.source)) && (
            <MenuItem disabled onPick={() => {}}>
              Nothing here fits
            </MenuItem>
          )}
        </SubPanel>
      )}

      {branch === "question" && (
        <SubPanel anchor={anchor} width="19rem" panelRef={sub}>
          <div className="flex flex-col gap-1.5 p-1">
            <Sel
              label="Data"
              value={a.source}
              options={sources(report).map((s) => ({ id: s.id, label: s.label }))}
              onChange={(v) => {
                const d = defaults(v, report);
                const first = visualsFor(v)[0];
                ask({
                  source: v,
                  ...d,
                  split: "",
                  team: "",
                  // The old picture may not exist for the new source at all.
                  render: VISUALS.some((x) => x.id === a.render && (!x.only || x.only.includes(v)))
                    ? a.render
                    : (first?.id ?? "column"),
                });
              }}
            />
            <Sel
              label="Measure"
              value={a.measure}
              options={src.measures.map((m) => ({ id: m.id, label: m.label }))}
              onChange={(v) => ask({ measure: v })}
            />
            <Sel
              label="By"
              value={a.dim}
              options={src.dims.map((d) => ({ id: d.id, label: d.label }))}
              onChange={(v) => ask({ dim: v })}
            />
            <Sel
              label="Split"
              value={a.split}
              options={[
                { id: "", label: "One series" },
                ...src.dims.filter((d) => d.id !== a.dim).map((d) => ({ id: d.id, label: d.label })),
              ]}
              onChange={(v) => ask({ split: v })}
            />
            {teams.length > 1 && (
              <Sel
                label="Team"
                value={a.team}
                options={[
                  { id: "", label: "Both" },
                  ...teams.map((t) => ({ id: t, label: t })),
                ]}
                onChange={(v) => ask({ team: v })}
              />
            )}
            <Sel
              label="Show"
              value={String(a.limit)}
              options={[
                { id: "5", label: "Top 5" },
                { id: "10", label: "Top 10" },
                { id: "20", label: "Top 20" },
                { id: "500", label: "All of it" },
              ]}
              onChange={(v) => ask({ limit: Number(v) })}
            />
          </div>
        </SubPanel>
      )}

      {branch === "data" && (
        <SubPanel anchor={anchor} width="24rem" panelRef={sub}>
          <DataGrid a={a} onChange={onChange} />
        </SubPanel>
      )}

      {branch === "colours" && (
        <SubPanel anchor={anchor} width="26rem" panelRef={sub}>
          <Colours a={a} onChange={onChange} />
        </SubPanel>
      )}

      {branch === "sort" && (
        <SubPanel anchor={anchor} width="12rem" panelRef={sub}>
          <MenuItem
            on={!a.sort}
            onPick={() => {
              ask({ sort: "" });
              onClose();
            }}
          >
            {a.ordered ? "As it happened" : "Biggest first"}
          </MenuItem>
          {SORTS.filter((o) => o.id !== (a.ordered ? "natural" : "value")).map((o) => (
            <MenuItem
              key={o.id}
              on={a.sort === o.id}
              onPick={() => {
                ask({ sort: o.id });
                onClose();
              }}
            >
              {o.label}
            </MenuItem>
          ))}
        </SubPanel>
      )}

      {branch === "caption" && (
        <SubPanel anchor={anchor} width="20rem" panelRef={sub}>
          <input
            autoFocus
            defaultValue={a.title}
            aria-label="Caption"
            placeholder="Says what the reader should take from it"
            onKeyDown={(e) => {
              if (e.key !== "Enter") return;
              e.preventDefault();
              onChange({ title: (e.target as HTMLInputElement).value.slice(0, 120) });
              onClose();
            }}
            onBlur={(e) => onChange({ title: e.target.value.slice(0, 120) })}
            className="h-7 w-full rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-transparent px-2 text-[12px] text-[var(--ink)] outline-none focus:border-[var(--accent)]"
          />
        </SubPanel>
      )}

      {branch === "look" && (
        <SubPanel anchor={anchor} width="15rem" panelRef={sub}>
          {THEMES.map((t) => (
            <MenuItem
              key={t.id || "app"}
              on={(a.theme ?? "") === t.id}
              onPick={() => {
                // The look only, never the question: a theme must not re-ask
                // and throw away rows somebody edited by hand.
                onChange({ theme: t.id });
                onClose();
              }}
            >
              {t.label}
            </MenuItem>
          ))}
        </SubPanel>
      )}

      {branch === "width" && (
        <SubPanel anchor={anchor} width="10rem" panelRef={sub}>
          {[40, 50, 60, 75, 100].map((w) => (
            <MenuItem
              key={w}
              on={a.width === w}
              onPick={() => {
                onChange({ width: w });
                onClose();
              }}
            >
              {w}%
            </MenuItem>
          ))}
        </SubPanel>
      )}
    </>
  );
}
