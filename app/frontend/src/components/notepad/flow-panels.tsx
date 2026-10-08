"use client";

import { ChevronRight } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { useSeason } from "@/hooks/use-matches";
import { cn } from "@/lib/utils";
import type { MatchReport, StatPreset } from "@/lib/types";
import { api } from "@/lib/api";
import { layoutRow, lockedTableFromRows, tableContent, tableFromRows, type Block } from "./table-grid";
import { tierNote } from "./stat-node";
import { presetsFor } from "./stat-presets";
import { FIGURES, buildFigure, figureTitle, firstChoice, queryFigureTitle } from "./figure-data";
import { Chip, Find, Panel, Refusal, Row, Select } from "./panel-ui";
import { VISUALS, VISUAL_GROUPS, WHY, blockedBy } from "./visuals";
import { Choices, Trail, type Choice, type Crumb } from "./wizard";
import {
  defaultRender,
  dimOf,
  fmt,
  isComposition,
  measureOf,
  queryTitle,
  labelsFor,
  runColumns,
  runQuery,
  sources,
  teamsOf,
  type Measure,
  type Source,
  SORTS,
  type SortBy,
} from "./query-data";

/** `/table` and `/visual`: one flow, two endings.
 *
 *  The `/` menu used to be organised by where a block's numbers came from —
 *  Chart, Figure, Table, Stat — which is the shape of this codebase and not the
 *  shape of the question. A coach opening the menu knows what they want the
 *  thing to LOOK like. So the menu is appearance-first, and both entries walk
 *  the same three steps underneath:
 *
 *      WHO    players or teams
 *      WHEN   this match, or a run of them
 *      WHAT   the measures, which are the columns of a table and the bars of a
 *             chart, picked out of one searchable list
 *
 *  `/visual` puts one step in front: which visual. That is the choice that
 *  changes what the other three are allowed to be — a shot map and a momentum
 *  swing are not "a chart of some measure", they are a specific source drawn
 *  its own way, so choosing one pins the rest and the panel says so.
 *
 *  THE SOURCE IS NEVER PICKED DIRECTLY. It falls out of which measures were
 *  chosen, which is why they are all in one list. What that list cannot do is
 *  let two sources into one figure: the feed keys rows by name and the tracker
 *  by shirt number, and a row holding both would be asserting an identity join
 *  nobody made. Picking across the line swaps the set rather than merging it.
 */

type Who = "player" | "team";

/** How many columns a built table may carry.
 *
 *  Eight, because that is the widest set the report screen already prints —
 *  Top stats is eight columns — and a cap that refused a preset the product
 *  ships would be a cap arguing with itself. It was six, hand-picked, and
 *  `setOnly` bypassed it entirely: applying a seven-column preset produced a
 *  panel reading "7 of 6". One constant now, used by both paths and by the
 *  line that reports it, so they cannot disagree again.
 */
const COL_MAX = 8;

interface FlowProps {
  report: MatchReport | null;
  insert: (content: Block | Block[]) => void;
}

/** A frozen answer, as a DOCUMENT rather than as a block: a heading, a real
 *  table you can type in, and the line saying what the numbers rest on.
 *
 *  This is what "Editable" means everywhere it is offered. It used to insert
 *  the bare table, which quietly dropped the provenance note — and a set of
 *  numbers with no statement of where they came from is the one thing this
 *  codebase will not ship. The note rides along as a paragraph, which the coach
 *  can then rewrite; the linked figure renders its own and cannot be touched.
 */
function asDocument(opts: {
  title: string;
  head: string[];
  rows: string[][];
  note: string;
  tier: string;
}): Block[] {
  const provenance = [tierNote(opts.tier), opts.note].filter(Boolean).join(" \u00b7 ");
  return [
    ...(opts.title
      ? [{ type: "heading", attrs: { level: 3 }, content: [{ type: "text", text: opts.title }] }]
      : []),
    tableFromRows(opts.head, opts.rows),
    ...(provenance ? [{ type: "paragraph", content: [{ type: "text", text: provenance }] }] : []),
  ];
}

// ------------------------------------------------------------------ the spine

/** Who / when / what, and the state behind them. Both panels drive this and
 *  neither owns it, so the two cannot drift into asking the same thing twice
 *  in two different orders. */
function useFlow(report: MatchReport | null, multi: boolean, only?: string[]) {
  const [who, setWho] = useState<Who>("player");
  const [season, setSeason] = useState(false);
  /** 0 means every match in the season; otherwise the last N. */
  const [lastN, setLastN] = useState(0);
  const [picked, setPicked] = useState<{ source: string; ids: string[] }>({ source: "", ids: [] });
  const [dim, setDim] = useState("");
  // null until somebody picks one, so the default can follow the dimension.
  const [chosenLimit, setChosenLimit] = useState<number | null>(null);
  const [chosenTeam, setTeam] = useState("");
  const [keep, setKeep] = useState<string[]>([]);
  const [find, setFind] = useState("");
  /** Named matches, when "Pick matches" was used. Beats lastN when set. */
  const [chosen, setChosen] = useState<string[]>([]);
  const [sort, setSort] = useState<SortBy>("");
  const [caption, setCaption] = useState("");

  // THE WHOLE CLUB, not the selected match's team. A note is not filed under a
  // match or a team any more, and this is the half of that which has teeth: a
  // season scoped to `match?.team` could not put the U15s and the U17s in one
  // table, so "how far do our midfielders run as they come up the ladder" had
  // no way to be asked. The season source carries a Team dimension, so the
  // wider fetch is read apart rather than silently pooled.
  //
  // Always, not only once somebody has asked for the season: the When step
  // offers "Last 3" and "Last 5" out of a count it cannot know until the
  // season has loaded, so gating the fetch on the answer hid the choices
  // behind themselves. The response is cached, so this costs one request.
  const { season: seasonData } = useSeason(undefined, true);

  const all = useMemo(() => sources(report), [report]);
  const groups = useMemo(() => {
    // A visual that can only be drawn from certain sources narrows the panel
    // to those, rather than the Who/When scope.
    if (only?.length) return all.filter((s) => only.includes(s.id));
    return all.filter((s) => s.subject === who && (s.scope === "season") === season);
  }, [all, who, season, only]);

  // The pinned source, or whatever the chosen measures came from, or the first
  // group so an empty panel still has something to describe.
  const src: Source | undefined = groups.find((s) => s.id === picked.source) ?? groups[0];

  const matchIds = useMemo(() => {
    if (!season || !seasonData) return undefined;
    // A named set beats a run of form: somebody who picked five matches by
    // hand meant those five, and quietly re-deriving them from `lastN` the
    // next time the season grew would answer a different question.
    if (chosen.length) return chosen;
    if (!lastN) return undefined;
    return seasonData.matches.slice(-lastN).map((m) => m.id);
  }, [season, lastN, seasonData, chosen]);

  // With one source and one measure there is nothing to choose, so the panel
  // does not pretend there is: a shot map is the shot list, drawn.
  const ids = only?.length === 1 && src ? [src.measures[0].id] : picked.ids;
  const dimObj = src ? dimOf(src, dim || src.dims[0].id) : null;
  const activeDim = dimObj?.id ?? "";
  // An ordered breakdown truncated to its top twenty is a different match:
  // momentum stops at minute 20, a season stops in October. Ranked lists are
  // the other way round — nobody wants every one of forty players by default.
  const limit = chosenLimit ?? (dimObj?.ordered ? 500 : 20);
  const teams = useMemo(
    () => (src ? teamsOf(src.id, report, seasonData) : []),
    [src, report, seasonData],
  );
  /** Everything the breakdown could show — what Compare offers. */
  const choices = useMemo(
    () =>
      src
        ? labelsFor({
            report,
            season: seasonData,
            source: src.id,
            dim: activeDim,
            team: teams.includes(chosenTeam) ? chosenTeam : undefined,
            matchIds,
          })
        : [],
    [src, report, seasonData, activeDim, chosenTeam, teams, matchIds],
  );
  // A name that is not in the current breakdown is not a comparison, it is an
  // empty row — which happens the moment the dimension or the team changes
  // under a picked set.
  const keeping = keep.filter((k) => choices.includes(k));
  // A team that is not in this source's rows is not a filter, it is an empty
  // result — which happens the moment the source changes under a chosen team.
  const team = teams.includes(chosenTeam) ? chosenTeam : "";

  const toggle = (sourceId: string, id: string) => {
    setPicked((p) => {
      // Across the line: swap rather than merge. See the header.
      if (p.source !== sourceId) return { source: sourceId, ids: [id] };
      if (!multi) return { source: sourceId, ids: [id] };
      const has = p.ids.includes(id);
      // Never down to nothing — an empty table is not a state worth reaching
      // by clicking the last column off.
      if (has && p.ids.length === 1) return p;
      return { source: sourceId, ids: has ? p.ids.filter((x) => x !== id) : [...p.ids, id].slice(0, COL_MAX) };
    });
    setDim("");
  };

  const setWhere = (next: Partial<{ who: Who; season: boolean; lastN: number }>) => {
    if (next.who !== undefined) setWho(next.who);
    if (next.season !== undefined) setSeason(next.season);
    if (next.lastN !== undefined) setLastN(next.lastN);
    // The measures belonged to the old scope's sources. Keeping them would
    // silently resolve to whatever the new source's first measure happens to
    // be, which is a different question wearing the old question's label.
    if (next.who !== undefined || next.season !== undefined) {
      setPicked({ source: "", ids: [] });
      setDim("");
    }
  };

  return {
    who, season, lastN, seasonData, groups, src, ids, activeDim, limit, find, matchIds, multi,
    ordered: !!dimObj?.ordered,
    teams, team, choices, keeping, chosen, sort, caption,
    setWhere, setDim, setLimit: setChosenLimit, setFind, setTeam, toggle, setSort, setCaption,
    /** Replace the selection wholesale — what applying a preset means. Going
     *  through `toggle` one id at a time would fight its own source-switching
     *  rule and land on whichever measure happened to be last. */
    setOnly(source: string, ids: string[]) {
      setPicked({ source, ids: ids.slice(0, COL_MAX) });
    },
    matchToggle(id: string) {
      setChosen((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]));
    },
    matchClear() { setChosen([]); },
    keepToggle(label: string) {
      setKeep((p) => (p.includes(label) ? p.filter((x) => x !== label) : [...p, label]));
    },
    keepClear() { setKeep([]); },
  };
}

type Flow = ReturnType<typeof useFlow>;

function Show({ f, steps }: { f: Flow; steps: number[] }) {
  return (
    <Row label="Show">
      {steps.map((n) => (
        <Chip key={n} on={f.limit === n} onClick={() => f.setLimit(n)}>
          {n === 500 ? "All of it" : `Top ${n}`}
        </Chip>
      ))}
    </Row>
  );
}

/** Pick the rows, rather than take the top N of them.
 *
 *  The panel could always ask "how many"; it could not ask "which". So a chart
 *  comparing two named players meant inserting the top twelve and deleting ten
 *  of them by hand — which the frozen block does not even allow.
 *
 *  Picked rows keep the ORDER THEY WERE PICKED IN, not the measure's order:
 *  otherwise the pair you are comparing swaps places every time you change what
 *  you are comparing them on, and the chart's meaning moves under the reader.
 */
function Compare({ f }: { f: Flow }) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  if (f.choices.length < 2) return null;
  const shown = q
    ? f.choices.filter((c) => c.toLowerCase().includes(q.toLowerCase()))
    : f.choices;

  return (
    <>
      <Row label="Compare">
        <Chip on={!f.keeping.length} onClick={() => { f.keepClear(); setOpen(false); }}>
          Top {f.limit === 500 ? "all" : f.limit}
        </Chip>
        <Chip on={f.keeping.length > 0} onClick={() => setOpen((v) => !v)}>
          {f.keeping.length ? `${f.keeping.length} picked` : "Pick rows"}
        </Chip>
      </Row>
      {open && (
        <div className="flex flex-col gap-1">
          <Find value={q} onChange={setQ} placeholder={`Find a ${f.src?.dims.find((d) => d.id === f.activeDim)?.label.toLowerCase() ?? "row"}`} />
          <div className="flex max-h-36 flex-wrap gap-1 overflow-y-auto pr-0.5">
            {shown.length === 0 && <Refusal>Nothing matches “{q}”.</Refusal>}
            {shown.map((c) => (
              <Chip key={c} on={f.keeping.includes(c)} onClick={() => f.keepToggle(c)}>
                {c}
              </Chip>
            ))}
          </div>
          {f.keeping.length > 0 && (
            <span className="text-[10px] leading-snug text-[var(--mute)]">
              In the order you picked them: {f.keeping.join(", ")}.
            </span>
          )}
        </div>
      )}
    </>
  );
}

/** The one list every measure lives in, grouped by where it comes from. */
function Measures({
  f,
  report,
  onPicked,
}: {
  f: Flow;
  report: MatchReport | null;
  onPicked?: () => void;
}) {
  const q = f.find.trim().toLowerCase();
  const [preset, setPreset] = useState("");
  const [saved, setSaved] = useState<StatPreset[]>([]);
  const [naming, setNaming] = useState(false);

  // The project's saved sets. One request, and a failure is silent on purpose:
  // presets are a convenience, and a panel that refuses to open because a
  // convenience did not load is worse than one that shows the built-ins.
  useEffect(() => {
    let live = true;
    api.getPresets().then(
      (p) => live && setSaved(p),
      () => {},
    );
    return () => {
      live = false;
    };
  }, []);

  // Keyed by VALUE — `groups` is rebuilt on every render and the array identity
  // would never match, so the memo would run every time anyway.
  const key = f.groups.map((s) => `${s.id}:${s.measures.map((m) => m.id).join("|")}`).join(",");
  const presets = useMemo(
    () => presetsFor(f.groups, report, saved),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [key, report, saved],
  );
  const chosen = presets.find((p) => p.id === preset);

  const shown = f.groups
    .map((s) => ({
      src: s,
      measures: s.measures.filter(
        (m) =>
          (!q || m.label.toLowerCase().includes(q)) &&
          // A preset NARROWS the wall; it does not become a different list.
          // The measures are still the source's own, still labelled the way
          // the report labels them.
          (!chosen || chosen.cols.includes(m.id)),
      ),
    }))
    .filter((g) => g.measures.length > 0);

  if (!f.groups.length) {
    return <Refusal>Nothing here can be broken down that way yet.</Refusal>;
  }

  return (
    <div className="flex flex-col gap-1">
      {presets.length > 0 && (
        <div className="flex flex-wrap items-center gap-1">
          <Chip on={!preset} onClick={() => setPreset("")}>
            Everything
          </Chip>
          {presets.map((p) => (
            <Chip
              key={p.id}
              on={preset === p.id}
              title={p.saved ? "Saved for this project" : undefined}
              onClick={() => {
                setPreset(p.id);
                // For a table, a preset IS the selection — that is what the
                // report screen's category pills do. For a chart, which takes
                // one measure, it only narrows what there is to choose from.
                // The preset's OWN source, not whatever is selected: these
                // ids are the feed's, and applying them against the tracks
                // produced seven columns that source did not have.
                if (f.multi) {
                  f.setOnly(p.source, p.cols);
                  onPicked?.();
                }
              }}
            >
              {p.label}
            </Chip>
          ))}
          {f.multi && f.ids.length > 1 && !naming && (
            <Chip title="Keep this set for every note in the project" onClick={() => setNaming(true)}>
              Save set
            </Chip>
          )}
        </div>
      )}
      {naming && (
        <input
          autoFocus
          placeholder="Name this set, then Enter"
          aria-label="Preset name"
          onKeyDown={(e) => {
            if (e.key === "Escape") return setNaming(false);
            if (e.key !== "Enter") return;
            const name = (e.target as HTMLInputElement).value.trim();
            if (!name) return setNaming(false);
            const next = [...saved.filter((p) => p.name !== name), { name, cols: f.ids }];
            setNaming(false);
            api.savePresets(next).then(setSaved, () => {});
          }}
          onBlur={() => setNaming(false)}
          className="h-6 w-full rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-transparent px-1.5 text-[11.5px] text-[var(--ink)] outline-none focus:border-[var(--accent)]"
        />
      )}
      <Find value={f.find} onChange={f.setFind} placeholder="Find a measure" />
      <div className="flex max-h-48 flex-col gap-1.5 overflow-y-auto pr-0.5">
        {shown.length === 0 && <Refusal>Nothing matches “{f.find}”.</Refusal>}
        {shown.map((g) => {
          const other = !!f.src && f.src.id !== g.src.id && f.ids.length > 0;
          return (
            <div key={g.src.id} className="flex flex-col gap-1">
              <span className="text-[9.5px] tracking-wide text-[var(--mute)] uppercase">
                {g.src.label}
              </span>
              <div className={cn("flex flex-wrap gap-1", other && "opacity-45")}>
                {g.measures.map((m) => (
                  <Chip
                    key={m.id}
                    on={f.src?.id === g.src.id && f.ids.includes(m.id)}
                    title={
                      other
                        ? `Starts a new set — ${g.src.label} and ${f.src!.label} key their rows differently and cannot share one.`
                        : undefined
                    }
                    onClick={() => { f.toggle(g.src.id, m.id); onPicked?.(); }}
                  >
                    {m.label}
                  </Chip>
                ))}
              </div>
            </div>
          );
        })}
      </div>
      {f.src && f.ids.length > 0 && f.multi && (
        <span className="text-[10px] leading-snug text-[var(--mute)]">
          {f.ids.length} of {COL_MAX} · all from {f.src.label}, because two sources key their
          rows differently and could not honestly share one.
        </span>
      )}
    </div>
  );
}

// ------------------------------------------------------------- the two steps
//                                                                both panels ask

const WHO: Choice[] = [
  { id: "player", label: "Players", hint: "One row for each name on the pitch" },
  { id: "team", label: "Teams", hint: "One row a side — home against away" },
];

const whenChoices = (matches: number): Choice[] => [
  { id: "match", label: "This match", hint: "The one open in the workspace" },
  {
    id: "all",
    label: "Every match",
    hint: matches ? `All ${matches} of the season so far` : "The season so far",
  },
  ...[3, 5, 10]
    .filter((n) => matches > n)
    .map((n) => ({ id: `last${n}`, label: `Last ${n}`, hint: "A run of form, not a season" })),
  ...(matches > 1
    ? [{ id: "pick", label: "Pick matches", hint: "Name them, in any order" }]
    : []),
];

const whenPick = (f: Flow, id: string) => {
  f.matchClear();
  if (id === "match") f.setWhere({ season: false, lastN: 0 });
  else if (id === "all") f.setWhere({ season: true, lastN: 0 });
  else if (id === "pick") f.setWhere({ season: true, lastN: 0 });
  else f.setWhere({ season: true, lastN: Number(id.slice(4)) });
};

const whenId = (f: Flow) =>
  !f.season ? "match" : f.chosen.length ? "pick" : f.lastN ? `last${f.lastN}` : "all";
const whenLabel = (f: Flow) =>
  !f.season
    ? "This match"
    : f.chosen.length
      ? `${f.chosen.length} matches`
      : f.lastN
        ? `Last ${f.lastN}`
        : "Every match";

/** Name the matches. A season is a list of fixtures a coach knows by heart, so
 *  the honest control is that list with ticks against it — not a range, which
 *  would make "the two derbies" impossible to ask for. */
function PickMatches({ f }: { f: Flow }) {
  const all = f.seasonData?.matches ?? [];
  return (
    <div className="flex max-h-64 flex-col gap-0.5 overflow-y-auto py-1">
      {all.map((m) => (
        <label
          key={m.id}
          className="flex cursor-pointer items-center gap-2 rounded-[var(--radius-sm)] px-1.5 py-1 text-[11.5px] hover:bg-[var(--card)]"
        >
          <input
            type="checkbox"
            checked={f.chosen.includes(m.id)}
            onChange={() => f.matchToggle(m.id)}
            className="accent-[var(--accent)]"
          />
          <span className="flex-1 truncate text-[var(--ink)]">{m.name ?? m.id.slice(0, 8)}</span>
        </label>
      ))}
      {!all.length && <Refusal>The season has not loaded any matches yet.</Refusal>}
    </div>
  );
}

/** A step whose answer is not a list of choices still needs a way onward. */
const Next = ({ onClick, label }: { onClick: () => void; label: string }) => (
  <button
    type="button"
    onClick={onClick}
    className="flex h-7 items-center justify-center gap-1 rounded-[var(--radius-sm)] border border-[var(--hairline-strong)] text-[11.5px] text-[var(--ink)] transition-colors hover:bg-[var(--card)]"
  >
    {label} <ChevronRight size={12} strokeWidth={1.8} />
  </button>
);

/** One side's rows only. Asked as a step of its own wherever both sides are in
 *  the source, because a ranking of "the players" that mixes both squads is a
 *  ranking of nobody's squad. */
const teamChoices = (f: Flow): Choice[] => [
  { id: "", label: "Both sides", hint: "Everyone on the pitch, in one list" },
  ...f.teams.map((t) => ({ id: t, label: t, hint: "This side only" })),
];

// ------------------------------------------------------------------- /table

type TStep = "kind" | "grid" | "which" | "who" | "when" | "pick" | "columns" | "detail";

const KINDS: Choice[] = [
  { id: "blank", label: "Blank table", hint: "Pick a size and type in it, like a spreadsheet" },
  {
    id: "build",
    label: "Build from the match",
    hint: "Choose the rows and the columns out of what was recorded",
  },
  ...FIGURES.map((x) => ({ id: x.id, label: x.label, hint: x.hint })),
];

export function TablePanel({ report, insert }: FlowProps) {
  const [step, setStep] = useState<TStep>("kind");
  const [kind, setKind] = useState("");
  const [live, setLive] = useState(false);
  const [cols, setCols] = useState("");
  const f = useFlow(report, true);

  const built = useMemo(() => {
    if (kind !== "build" || !f.src || !f.ids.length) return null;
    return runColumns({
      report,
      season: f.seasonData,
      source: f.src.id,
      dim: f.activeDim,
      measures: f.ids,
      limit: f.limit,
      matchIds: f.matchIds,
      team: f.team,
      keep: f.keeping,
      sort: f.sort,
    });
  }, [kind, report, f.src, f.ids, f.activeDim, f.limit, f.seasonData, f.matchIds, f.team, f.keeping, f.sort]);

  const preset = useMemo(
    () =>
      kind && kind !== "build" && kind !== "blank"
        ? buildFigure({ report, kind, cols, limit: 22 })
        : null,
    [report, kind, cols],
  );

  const spec = f.src
    ? JSON.stringify({
        source: f.src.id, dim: f.activeDim, measures: f.ids, limit: f.limit,
        ...(f.matchIds ? { matchIds: f.matchIds } : {}),
        ...(f.team ? { team: f.team } : {}),
        ...(f.keeping.length ? { keep: f.keeping } : {}),
      })
    : "";

  // One measure, and the rows are parts of a whole: that is the shape the
  // scouting report never prints bare — flanks, thirds, how the shots ended.
  // It gets its chart, beside it, in a two-column row.
  const measure: Measure | null = f.src && f.ids.length === 1 ? measureOf(f.src, f.ids[0]) : null;
  const withChart = !!built && !!measure && isComposition(built, measure);

  const addBuilt = () => {
    if (!built || !f.src) return;
    // Two different things, and the difference is the point.
    //
    // EDITABLE puts the answer into a real table: cells you can type in, a
    // column of your own notes beside the numbers, rows you can delete. Its
    // numbers are then yours — nothing will ever refresh them, exactly as
    // pasting values into Excel never re-runs the query.
    //
    // LINKED keeps the frozen figure, which remembers the question and can be
    // re-asked. You cannot type in it.
    // Locked is a real TABLE now, like the three report figures — the last
    // creation path still making a read-only `figure` atom. Its question is a
    // QuerySpec rather than a named subset, which `buildFigure` already
    // understands under the "query" kind, so the menu can re-ask it and swap
    // the columns for a preset.
    const figure = live
      ? asReportTable({
          kind: "query",
          cols: spec,
          title: f.caption || queryFigureTitle(spec, report),
          preset: {
            head: built.head,
            rows: built.rows.map((r) => ({ name: r.label, cells: r.text, split: undefined })),
            note: built.note,
            tier: built.tier,
            refusal: built.refusal,
          },
          locked: true,
          limit: f.limit,
        })
      : asDocument({
          // A heading somebody wrote beats one this code derived.
          title: f.caption || queryFigureTitle(spec, report),
          head: built.head,
          rows: built.rows.map((r) => [r.label, ...r.text]),
          note: built.refusal ?? built.note,
          tier: built.tier,
        });
    if (!withChart) return insert(figure);

    // The chart is the SAME question, run the way a chart runs it, so the two
    // halves of the row cannot disagree about a number.
    const render = defaultRender(f.src.id);
    const r = runQuery({
      report, season: f.seasonData, source: f.src.id, dim: f.activeDim,
      measure: f.ids[0], split: "", limit: f.limit, matchIds: f.matchIds, team: f.team,
      keep: f.keeping, render,
    });
    insert(
      layoutRow([
        {
          type: "query",
          attrs: {
            title: queryTitle(r, "", render),
            source: f.src.id, dim: f.activeDim, measure: f.ids[0], split: "", render,
            limit: f.limit, team: f.team, keep: f.keeping, width: 100, custom: false,
            rows: r.rows, series: r.series, max: r.max, extra: r.extra ?? null,
            note: r.refusal ?? r.note, tier: r.tier,
            unit: r.measure.unit ?? "", dp: r.measure.dp,
            label: r.measure.label, dimLabel: r.dim.label, ordered: r.ordered,
          },
        },
        figure,
      ]),
    );
  };

  const goto = (s: TStep) => () => setStep(s);
  const kindLabel = KINDS.find((k) => k.id === kind)?.label ?? "";
  const trail: Crumb[] = [
    { label: "Table", onBack: step === "kind" ? undefined : goto("kind") },
    ...(kind ? [{ label: kindLabel, onBack: step === "kind" ? goto("kind") : undefined }] : []),
    ...(kind === "build" && step !== "who"
      ? [
          {
            label: f.who === "team" ? "Teams" : "Players",
            onBack: step === "when" ? undefined : goto("who"),
          },
        ]
      : []),
    ...(kind === "build" && (step === "columns" || step === "detail")
      ? [{ label: whenLabel(f), onBack: step === "columns" ? undefined : goto("when") }]
      : []),
    ...(kind === "build" && step === "detail" && f.src ? [{ label: f.src.label }] : []),
  ];

  const action =
    step === "detail" && built?.rows.length
      ? {
          label: withChart ? "Add table and chart" : "Add table",
          run: addBuilt,
          hint: withChart ? "These are parts of one whole, so they land side by side" : undefined,
        }
      : step === "which" && preset?.rows.length
        ? {
            label: "Add table",
            run: () =>
              insert(
                asReportTable({
                  kind,
                  cols,
                  title: figureTitle(kind, cols, report),
                  preset,
                  locked: live,
                }),
              ),
          }
        : undefined;

  return (
    <Panel
      title={<Trail crumbs={trail} />}
      subtitle={
        step === "detail" && built?.rows.length ? `${built.rows.length} rows` : undefined
      }
      note={
        step === "detail" && f.ids.length && !built?.rows.length
          ? built?.refusal
          : step === "which" && !preset?.rows.length
            ? preset?.refusal
            : undefined
      }
      action={action}
    >
      {step === "kind" && (
        <Choices
          value={kind}
          items={KINDS}
          onPick={(id) => {
            setKind(id);
            if (id === "blank") setStep("grid");
            else if (id === "build") setStep("who");
            else {
              setCols(firstChoice(id, report));
              setStep("which");
            }
          }}
        />
      )}

      {step === "grid" && <Grid insert={insert} />}

      {step === "which" && preset && (
        <>
          <Choices
            value={cols}
            items={preset.choices.map((c) => ({ id: c.id, label: c.label }))}
            onPick={setCols}
          />
          <AsRow live={live} setLive={setLive} />
        </>
      )}

      {step === "who" && (
        <Choices
          value={f.who}
          items={WHO}
          onPick={(id) => {
            f.setWhere({ who: id as Who });
            setStep("when");
          }}
        />
      )}

      {step === "when" && (
        <Choices
          value={whenId(f)}
          items={whenChoices(f.seasonData?.matches.length ?? 0)}
          onPick={(id) => {
            whenPick(f, id);
            setStep(id === "pick" ? "pick" : "columns");
          }}
        />
      )}

      {step === "pick" && (
        <>
          <PickMatches f={f} />
          {f.chosen.length ? (
            <Next onClick={goto("columns")} label={`${f.chosen.length} picked — next`} />
          ) : (
            <Refusal>Tick the matches this should cover.</Refusal>
          )}
        </>
      )}

      {step === "columns" && (
        <>
          {f.src && (
            <Select
              label="Rows"
              value={f.activeDim}
              onChange={f.setDim}
              options={f.src.dims.map((d) => ({ id: d.id, label: d.label }))}
            />
          )}
          <Measures f={f} report={report} />
          {f.ids.length ? (
            <Next onClick={goto("detail")} label={`${f.ids.length} column${f.ids.length > 1 ? "s" : ""} — next`} />
          ) : (
            <Refusal>Pick what goes in the columns.</Refusal>
          )}
        </>
      )}

      {step === "detail" && (
        <>
          {f.teams.length > 1 && (
            <Row label="Team">
              {teamChoices(f).map((t) => (
                <Chip key={t.id} on={f.team === t.id} onClick={() => f.setTeam(t.id)}>
                  {t.label}
                </Chip>
              ))}
            </Row>
          )}
          <Show f={f} steps={[10, 20, 500]} />
          <Compare f={f} />
          <Select
            label="Sort"
            value={f.sort}
            onChange={(v) => f.setSort(v as SortBy)}
            options={[
              { id: "", label: f.ordered ? "As it happened" : "Biggest first" },
              ...SORTS.filter((o) => o.id !== (f.ordered ? "natural" : "value")),
            ]}
          />
          <Row label="Heading">
            <input
              value={f.caption}
              onChange={(e) => f.setCaption(e.target.value.slice(0, 120))}
              placeholder="What this table is"
              aria-label="Heading"
              className="h-6 flex-1 rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-transparent px-1.5 text-[11.5px] text-[var(--ink)] outline-none focus:border-[var(--accent)]"
            />
          </Row>
          <AsRow live={live} setLive={setLive} />
        </>
      )}
    </Panel>
  );
}

/** Editable or locked, offered wherever a frozen answer is about to land.
 *
 *  It used to sit on the last step of "Build from the match" only, so the three
 *  figures the report ships — the player table, the comparison, the timeline —
 *  had no way to arrive as anything but a block with no caret in it. They are
 *  the most table-looking things in the document and were the least editable.
 */
/** One of the report's three figures, as a REAL table.
 *
 *  Both halves of the toggle now produce the same node — the difference is one
 *  attribute. "Locked" keeps the imported numbers held still and remembers the
 *  question, so the right-click menu can change the subset or ask it again;
 *  "Editable" is the same table with the lock off and no question, which is
 *  what you want when the point is to type your own column beside the numbers.
 *
 *  What this replaces was a separate read-only block that merely LOOKED like a
 *  table. It was the most table-shaped thing in the document and the only one
 *  with no caret in it, which is the complaint that started all of this.
 */
function asReportTable(opts: {
  kind: string;
  cols: string;
  title: string;
  preset: { head: string[]; rows: { name: string; cells: string[]; split?: number }[]; note: string; tier: string; refusal?: string };
  locked: boolean;
  /** How many rows the question asked for, so Refresh asks for the same. */
  limit?: number;
}): Block[] {
  const { preset } = opts;
  const provenance = [tierNote(preset.tier), preset.refusal ?? preset.note]
    .filter(Boolean)
    .join(" \u00b7 ");
  return [
    { type: "heading", attrs: { level: 3 }, content: [{ type: "text", text: opts.title }] },
    lockedTableFromRows({
      head: preset.head,
      rows: preset.rows.map((r) => [r.name, ...r.cells]),
      bars: preset.rows.map((r) => r.split ?? null),
      figure: opts.locked
        ? JSON.stringify({ kind: opts.kind, cols: opts.cols, limit: opts.limit ?? 22 })
        : undefined,
    }) as Block,
    ...(provenance ? [{ type: "paragraph", content: [{ type: "text", text: provenance }] }] : []),
  ].map((b, i) =>
    // The lock rides on the table only, and only when it was asked for.
    i === 1 && !opts.locked
      ? ({ ...(b as Record<string, unknown>), attrs: { ...((b as { attrs?: object }).attrs ?? {}), locked: false } } as Block)
      : (b as Block),
  );
}

function AsRow({ live, setLive }: { live: boolean; setLive: (v: boolean) => void }) {
  return (
    <Row label="As">
      <Chip
        on={!live}
        title="A plain table \u2014 type in the cells straight away, add your own column, delete rows"
        onClick={() => setLive(false)}
      >
        Editable
      </Chip>
      <Chip
        on={live}
        title="The same table with the imported numbers held still. Right-click it to unlock, change the subset, or take the figures again."
        onClick={() => setLive(true)}
      >
        Locked
      </Chip>
    </Row>
  );
}

/** The Google Docs grid picker: hover out to the size you want, click.
 *
 *  A grid rather than two number inputs because picking 3x4 is a gesture, and
 *  because the shape you are about to get is the thing on screen. Rows and
 *  columns can both be changed afterwards from the toolbar, so nothing here is
 *  a decision anybody is stuck with.
 */
function Grid({ insert }: { insert: FlowProps["insert"] }) {
  const [hover, setHover] = useState({ r: 2, c: 3 });
  const [header, setHeader] = useState(true);
  const MAX_R = 8;
  const MAX_C = 8;

  return (
    <>
      <div
        className="flex w-fit flex-col gap-[3px] py-1"
        onMouseLeave={() => setHover({ r: 2, c: 3 })}
      >
        {Array.from({ length: MAX_R }, (_, r) => (
          <div key={r} className="flex gap-[3px]">
            {Array.from({ length: MAX_C }, (_, c) => (
              <button
                key={c}
                type="button"
                aria-label={`${r + 1} by ${c + 1}`}
                onMouseEnter={() => setHover({ r: r + 1, c: c + 1 })}
                onFocus={() => setHover({ r: r + 1, c: c + 1 })}
                onClick={() => insert(tableContent({ rows: r + 1, cols: c + 1, header }))}
                className={cn(
                  "h-[13px] w-[13px] border transition-colors",
                  r < hover.r && c < hover.c
                    ? "border-[var(--accent)] bg-[var(--accent)]/35"
                    : "border-[var(--hairline)] hover:border-[var(--hairline-strong)]",
                )}
              />
            ))}
          </div>
        ))}
      </div>
      <Row label="Size">
        <span className="font-mono text-[10.5px] text-[var(--mute)]">
          {hover.r} &times; {hover.c}
        </span>
      </Row>
      <Row label="First row">
        <Chip on={header} onClick={() => setHeader(true)}>Header</Chip>
        <Chip on={!header} onClick={() => setHeader(false)}>Plain</Chip>
      </Row>
      <Refusal>
        Rows, columns, merging and widths are all on the toolbar once the caret is in it.
      </Refusal>
    </>
  );
}

// ------------------------------------------------------------------ /visual

type VStep = "family" | "type" | "who" | "when" | "pick" | "measure" | "detail";

export function VisualPanel({ report, insert }: FlowProps) {
  const [step, setStep] = useState<VStep>("family");
  const [family, setFamily] = useState("");
  const [visual, setVisual] = useState("");
  const [split, setSplit] = useState("");

  // A picture drawn from its own sources narrows the panel to those. Choosing
  // "Shot map" IS choosing the shot list — which is why the first step offers
  // every visual rather than only the ones the current source can draw.
  // Filtering by the source made the native ones unreachable: you pick the
  // source BY picking the visual, so neither could ever come first.
  const current = VISUALS.find((v) => v.id === visual);
  const only = current?.only;
  const pin = only?.length === 1;
  const f = useFlow(report, false, only);

  const r = useMemo(() => {
    if (!f.src || !f.ids.length) return null;
    return runQuery({
      report, season: f.seasonData, source: f.src.id, dim: f.activeDim,
      measure: f.ids[0], split, limit: f.limit, matchIds: f.matchIds, team: f.team,
      keep: f.keeping, render: visual || "column", sort: f.sort,
    });
  }, [report, f.src, f.ids, f.activeDim, split, f.limit, f.seasonData, f.matchIds, f.team, f.keeping, visual, f.sort]);

  // Never leave a picture selected that the rest of the answer has just made
  // impossible — the block would go in as a silent fallback to bars.
  const currentBlock = current ? blockedBy(current, r, f.src?.id ?? "") : null;
  const effective = currentBlock ? "column" : visual || "column";
  const splits = f.src
    ? f.src.dims.filter((d) => d.id !== f.activeDim && d.id !== "player" && d.id !== "match")
    : [];

  /** Where a picture sends you next. One with its own source has no Who and no
   *  When to ask about — the source is already the answer to both — and a
   *  pinned one has no measure either. */
  const after = (id: string): VStep => {
    const o = VISUALS.find((v) => v.id === id)?.only;
    if (o?.length === 1) return "detail";
    return o?.length ? "measure" : "who";
  };

  const goto = (s: VStep) => () => setStep(s);
  const trail: Crumb[] = [
    { label: "Visual", onBack: step === "family" ? undefined : goto("family") },
    ...(family && step !== "family"
      ? [{ label: family, onBack: step === "type" ? undefined : goto("type") }]
      : []),
    ...(current && step !== "family" && step !== "type"
      ? [
          {
            label: current.label,
            onBack: step === after(visual) ? undefined : goto(after(visual)),
          },
        ]
      : []),
    ...(!only && (step === "when" || step === "measure" || step === "detail")
      ? [
          {
            label: f.who === "team" ? "Teams" : "Players",
            onBack: step === "when" ? undefined : goto("who"),
          },
        ]
      : []),
    ...(!only && (step === "measure" || step === "detail")
      ? [{ label: whenLabel(f), onBack: step === "measure" ? undefined : goto("when") }]
      : []),
  ];

  return (
    <Panel
      title={<Trail crumbs={trail} />}
      subtitle={step === "detail" && r?.rows.length ? `${r.rows.length} rows` : undefined}
      note={
        step === "detail" && !r?.rows.length ? (r?.refusal ?? "Nothing to show.") : undefined
      }
      action={
        step === "detail" && r?.rows.length && f.src
          ? {
              label: "Add visual",
              run: () =>
                insert({
                  type: "query",
                  attrs: {
                    // A caption a coach wrote beats a title this code derived.
                    title: f.caption || queryTitle(r, split, effective),
                    source: f.src!.id, dim: f.activeDim, measure: f.ids[0], split,
                    render: effective, limit: f.limit, team: f.team, sort: f.sort,
                    matchIds: f.matchIds ?? [],
                    keep: f.keeping, width: 100, custom: false,
                    rows: r.rows, series: r.series, max: r.max, extra: r.extra ?? null,
                    note: r.refusal ?? r.note, tier: r.tier,
                    unit: r.measure.unit ?? "", dp: r.measure.dp,
                    label: r.measure.label, dimLabel: r.dim.label, ordered: r.ordered,
                  },
                }),
            }
          : undefined
      }
    >
      {step === "family" && (
        <Choices
          columns={2}
          value={family}
          items={VISUAL_GROUPS.map((g) => ({
            id: g,
            label: g,
            hint: VISUALS.filter((v) => v.group === g).map((v) => v.label).join(" · "),
          }))}
          onPick={(g) => {
            setFamily(g);
            setStep("type");
          }}
        />
      )}

      {step === "type" && (
        <Choices
          value={visual}
          items={VISUALS.filter((v) => v.group === family).map((v) => ({
            id: v.id,
            label: v.label,
            // `needs` is not a refusal at this step: nothing has been measured
            // yet, so it is what the picture will ask you for, not a reason it
            // is off. The real refusal lands on the last step, where the
            // answer exists to check it against.
            hint: v.hint ?? (v.needs ? WHY[v.needs] : undefined),
            why: v.missing,
          }))}
          onPick={(id) => {
            setVisual(id);
            setStep(after(id));
          }}
        />
      )}

      {step === "who" && (
        <Choices
          value={f.who}
          items={WHO}
          onPick={(id) => {
            f.setWhere({ who: id as Who });
            setStep("when");
          }}
        />
      )}

      {step === "when" && (
        <Choices
          value={whenId(f)}
          items={whenChoices(f.seasonData?.matches.length ?? 0)}
          onPick={(id) => {
            whenPick(f, id);
            setStep(id === "pick" ? "pick" : "measure");
          }}
        />
      )}

      {step === "pick" && (
        <>
          <PickMatches f={f} />
          {f.chosen.length ? (
            <Next onClick={goto("measure")} label={`${f.chosen.length} picked — next`} />
          ) : (
            <Refusal>Tick the matches this should cover.</Refusal>
          )}
        </>
      )}

      {step === "measure" && (
        <>
          <Measures f={f} report={report} onPicked={goto("detail")} />
          {!f.ids.length && <Refusal>Pick what to measure.</Refusal>}
        </>
      )}

      {step === "detail" && (
        <>
          {currentBlock && <Refusal>{currentBlock}</Refusal>}
          {r?.warning && <Refusal>{r.warning}</Refusal>}
          {pin && !currentBlock && (
            <Refusal>{current?.hint ?? "Drawn from one source. Nothing to choose."}</Refusal>
          )}
          {/* Even a picture with its own source has a side. A heat map of BOTH
              teams is a picture of a football match, which everyone has already
              seen; the question is always where ONE of them played. */}
          {f.teams.length > 1 && (
            <Row label="Team">
              {teamChoices(f).map((t) => (
                <Chip key={t.id} on={f.team === t.id} onClick={() => f.setTeam(t.id)}>
                  {t.label}
                </Chip>
              ))}
            </Row>
          )}
          {!pin && (
            <>
              {f.src && (
                <Select
                  label="By"
                  value={f.activeDim}
                  onChange={(v) => { f.setDim(v); if (split === v) setSplit(""); }}
                  options={f.src.dims.map((d) => ({ id: d.id, label: d.label }))}
                />
              )}
              {splits.length > 0 && (
                <Select
                  label="Split"
                  value={split}
                  onChange={setSplit}
                  options={[
                    { id: "", label: "No split" },
                    ...splits.map((d) => ({ id: d.id, label: d.label })),
                  ]}
                />
              )}
              <Show f={f} steps={[6, 12, 20, 500]} />
              <Compare f={f} />
              {/* Ordering is a question about the ANSWER, so it belongs after
                  the answer exists — and it is off by default, because every
                  figure already in a note was drawn by the old rule. */}
              <Select
                label="Sort"
                value={f.sort}
                onChange={(v) => f.setSort(v as SortBy)}
                options={[
                  { id: "", label: f.ordered ? "As it happened" : "Biggest first" },
                  ...SORTS.filter((o) => o.id !== (f.ordered ? "natural" : "value")),
                ]}
              />
            </>
          )}
          <Row label="Caption">
            <input
              value={f.caption}
              onChange={(e) => f.setCaption(e.target.value.slice(0, 120))}
              placeholder="Says what the reader should take from it"
              aria-label="Caption"
              className="h-6 flex-1 rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-transparent px-1.5 text-[11.5px] text-[var(--ink)] outline-none focus:border-[var(--accent)]"
            />
          </Row>
          {r?.rows.length ? (
            <span className="font-mono text-[10px] text-[var(--mute)]">
              {r.rows[0].label} {fmt(r.rows[0].values[0], r.measure)}
              {r.rows.length > 1 ? ` · ${r.rows.length - 1} more` : ""}
            </span>
          ) : null}
        </>
      )}
    </Panel>
  );
}
