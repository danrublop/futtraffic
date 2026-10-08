/** The one list a coach scrubs by: everything in a match that has a time.
 *
 *  Two clocks meet here and they are not the same clock. A play carries video
 *  seconds; a feed event carries a minute and nothing finer. `exact` is which,
 *  and every renderer has to say so — a minute placed on a timeline looks like
 *  a timecode, and it is off by up to 30s. */
import type { Clip, MatchReport, Side } from "./types";

export type MomentKind = "play" | "event" | "clip";

/** One seekable moment: a feed event, a generated play, a possession change, or
 *  a clip somebody marked. `exact` separates the two clocks these come off — a
 *  play carries video seconds, a feed event carries only a minute, so its tick
 *  is placed to within 30s and must never be presented as frame-accurate. */
export interface Moment {
  t: number;
  /** Clips are the only moments with a length. Everything else is an instant. */
  end?: number;
  exact: boolean;
  kind: MomentKind;
  side?: Side;
  label: string;
  detail?: string;
  /** Everyone this moment names: shirt numbers on a play, names on a feed
   *  event, tags on a clip. What the filter and the search both read. */
  who: string[];
  outcome?: string;
  /** Set on `kind === "clip"`, so a row can play the clip rather than only seek. */
  clip?: Clip;
  /** SAMPLE, not measured. Set only by `demoMoments`; the panel says so on the
   *  list, because a fabricated pass at 12:04 is indistinguishable from a real
   *  one once it is a row in the same table. */
  demo?: boolean;
}

/** Underscores out, and undefined survives as "". Every field read here comes
 *  off a match.json on disk, and this runs inside a useMemo during render — a
 *  throw does not degrade the film page, it blanks it. The pipeline has already
 *  changed this file's shape once (the `.pre-plays` backups), so a missing key
 *  is a thing that happens, not a hypothetical. */
const words = (s: string | undefined) => (s ?? "").replace(/_/g, " ");

/** "45+2'" -> 2820. Period rows ("HT") carry no minute and are not seekable. */
export const minuteSecs = (m: string) => {
  const g = /^(\d+)(?:\+(\d+))?/.exec(m);
  return g ? (Number(g[1]) + Number(g[2] ?? 0)) * 60 : null;
};

/** Feed events + generated plays + possession changes + clips, one list, in
 *  time order. */
export function moments(
  report: MatchReport | null,
  possession: { t: number; team: Side; label: string }[] = [],
  clips: Clip[] = [],
): Moment[] {
  const f = report?.meta?.facts;
  const out: Moment[] = [];
  for (const e of f?.events ?? []) {
    const t = minuteSecs(e.minute);
    if (t == null) continue;
    out.push({
      t,
      exact: false,
      kind: "event",
      side: e.side,
      label:
        e.type === "goal"
          ? `Goal — ${e.player}${e.score ? ` (${e.score})` : ""}`
          : e.type === "yellow_card"
            ? `Yellow — ${e.player}`
            : `Sub — ${e.in} for ${e.out}`,
      detail: e.assist ? `assist ${e.assist}` : undefined,
      who: [e.player, e.assist, e.in, e.out].filter((x): x is string => Boolean(x)),
      outcome: e.type,
    });
  }
  for (const pl of f?.plays ?? [])
    out.push({
      t: pl.t,
      exact: true,
      kind: "play",
      side: pl.side,
      // ponytail: underscores out, not the notepad's PLAY_TYPES map — importing it
      // drags pitch-data (900 lines) in for eight strings.
      label: words(pl.type) || "play",
      detail: [words(pl.zone), pl.outcome].filter(Boolean).join(" · ") || undefined,
      who: (pl.players ?? []).map((n) => `#${n}`),
      outcome: pl.outcome,
    });
  for (const e of possession)
    out.push({
      t: e.t,
      exact: true,
      kind: "event",
      side: e.team,
      label: e.label,
      detail: "possession",
      who: [],
    });
  for (const c of clips)
    out.push({
      t: c.start,
      end: c.end,
      exact: true,
      kind: "clip",
      label: c.name,
      detail: c.note || (c.tags ?? []).join(" · ") || undefined,
      who: c.tags ?? [],
      clip: c,
    });
  return out.sort((a, b) => a.t - b.t);
}

/** Free text over everything a row shows, plus the names it carries. One
 *  lowercase pass — a match list is hundreds of rows, not thousands. */
export const matchesQuery = (m: Moment, q: string) => {
  const s = q.trim().toLowerCase();
  if (!s) return true;
  return `${m.label} ${m.detail ?? ""} ${m.who.join(" ")}`.toLowerCase().includes(s);
};

/* ------------------------------------------------------------------ sample */

/** The vocabulary a demo match is written in. Ball actions, not the notepad's
 *  PLAY_TYPES — those are phases ("positional attack"), and a rail you scrub is
 *  read as "what just happened", which is a pass, a tackle, a shot. */
const DEMO_ACTIONS: { label: string; zones: string[]; outcomes: string[] }[] = [
  { label: "Pass", zones: ["own third", "middle third", "final third"], outcomes: ["completed", "completed", "intercepted"] },
  { label: "Dribble", zones: ["middle third", "final third"], outcomes: ["beat his man", "dispossessed"] },
  { label: "Cross", zones: ["final third"], outcomes: ["cleared", "chance", "cut out"] },
  { label: "Tackle", zones: ["own third", "middle third"], outcomes: ["won", "foul"] },
  { label: "Interception", zones: ["middle third", "own third"], outcomes: ["won"] },
  { label: "Shot", zones: ["penalty box", "final third"], outcomes: ["on target", "wide", "blocked", "goal"] },
  { label: "Save", zones: ["penalty box"], outcomes: ["held", "parried"] },
  { label: "Corner", zones: ["final third"], outcomes: ["cleared", "chance"] },
  { label: "Throw-in", zones: ["middle third", "final third"], outcomes: ["retained", "lost"] },
  { label: "Foul", zones: ["own third", "middle third", "final third"], outcomes: ["free kick", "yellow card"] },
];

/** Outfield shirts, per side. Keepers are 1 and only appear on a save. */
const DEMO_SHIRTS = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 20, 23];

/** A match's worth of plausible events, for a match that carries none.
 *
 *  DETERMINISTIC on purpose: the same duration always produces the same list,
 *  so scrubbing, filtering and re-rendering do not reshuffle the rail under the
 *  hand. Every row it makes is flagged `demo`, and nothing here is ever mixed
 *  into a match that has real plays — see the film panel's call site.
 */
export function demoMoments(dur: number): Moment[] {
  // A tiny LCG rather than Math.random: the seed is the duration, so the list
  // is stable for a given match and different between two of them.
  let seed = (Math.round(dur) * 2654435761) >>> 0 || 20260830;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  const pick = <T,>(xs: T[]) => xs[Math.floor(rnd() * xs.length)];

  const out: Moment[] = [];
  let t = 4 + rnd() * 10;
  let side: Side = rnd() < 0.5 ? "home" : "away";
  while (t < dur) {
    const a = pick(DEMO_ACTIONS);
    // Possession turns over on the actions that lose the ball, and only then —
    // a rail that flips sides every row reads as noise.
    const outcome = pick(a.outcomes);
    const shirt = a.label === "Save" ? 1 : pick(DEMO_SHIRTS);
    out.push({
      t: Math.round(t * 10) / 10,
      exact: true,
      kind: "play",
      side,
      label: a.label,
      detail: `${pick(a.zones)} · ${outcome}`,
      who: [`#${shirt}`],
      outcome,
      demo: true,
    });
    if (/intercepted|dispossessed|cut out|lost|won|cleared|wide|blocked|held|parried/.test(outcome))
      side = side === "home" ? "away" : "home";
    t += 6 + rnd() * 22;
  }
  return out;
}
