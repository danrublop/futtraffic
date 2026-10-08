import { parseMagnets, parseShapes, parseView, parseOrient, parseGrass, parseKits, parseLines, type Board } from "./pitch-data";
import type { MatchReport, Play } from "@/lib/types";

/** Plays: the moments of a match, as boards.
 *
 *  A play IS a board plus metadata — see model/engine/plays.py. That is what
 *  makes this different from pasting a picture: a play dropped into a note is
 *  the same editable whiteboard as one a coach drew by hand, so the first thing
 *  they can do is move a disc on it and disagree.
 *
 *  Everything here is `tagged` in the pipeline's vocabulary. The pipeline
 *  follows bodies (ball rows exist but are pixel centres and unused), so it cannot
 *  know a moment was a
 *  corner or that a pass was intended. Today these are generated; when the
 *  2D-map work lands they will be computed, and nothing on this side changes
 *  except the sentence under the diagram.
 */

export const PLAY_TYPES: Record<string, string> = {
  positional_attack: "Positional attack",
  counter_attack: "Counter-attack",
  high_press: "High press",
  position_defence: "Position defence",
  corner_attack: "Corner, attacking",
  corner_defence: "Corner, defending",
  free_kick_attack: "Free kick",
  throw_in: "Throw-in",
};

export const PLAY_ZONES: Record<string, string> = {
  own_third: "Own third",
  middle_third: "Middle third",
  final_third: "Final third",
  box: "Penalty box",
};

export const PLAY_OUTCOMES: Record<string, string> = {
  goal: "Goal",
  shot: "Shot",
  chance: "Chance",
  cleared: "Cleared",
  lost: "Lost",
};

/** What a play can be filtered on. `any` is first on every axis because the
 *  common move is narrowing one thing, not building a query. */
export interface PlayFilter {
  type: string;
  zone: string;
  outcome: string;
  side: string;
  /** "", "1" or "2". */
  period: string;
  /** A shirt number, as typed. "" for everyone. */
  player: string;
}

export const EMPTY_FILTER: PlayFilter = {
  type: "",
  zone: "",
  outcome: "",
  side: "",
  period: "",
  player: "",
};

export const plays = (report: MatchReport | null): Play[] =>
  report?.meta?.facts?.plays ?? [];

/** Narrow a list of plays. Every axis is an equality test except the player,
 *  which is membership — "who was involved" is the question a coach actually
 *  asks, and it is the one axis where a play matches several values at once. */
export function filterPlays(all: Play[], f: PlayFilter): Play[] {
  const n = Number(f.player);
  return all.filter(
    (p) =>
      (!f.type || p.type === f.type) &&
      (!f.zone || p.zone === f.zone) &&
      (!f.outcome || p.outcome === f.outcome) &&
      (!f.side || p.side === f.side) &&
      (!f.period || String(p.period) === f.period) &&
      (!f.player || (Number.isFinite(n) && p.players?.includes(n))),
  );
}

/** The options actually present in this match, so a filter can never be set to
 *  something that returns nothing. Counts ride along: "Corner (6)" tells a
 *  coach whether it is worth opening before they open it. */
export function playChoices(all: Play[], key: keyof Play): { id: string; label: string; n: number }[] {
  const labels: Record<string, Record<string, string>> = {
    type: PLAY_TYPES,
    zone: PLAY_ZONES,
    outcome: PLAY_OUTCOMES,
  };
  const counts = new Map<string, number>();
  for (const p of all) {
    const v = String(p[key] ?? "");
    if (v) counts.set(v, (counts.get(v) ?? 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([id, n]) => ({ id, label: labels[key as string]?.[id] ?? id, n }));
}

/** A play's board, narrowed through the notepad's own parsers.
 *
 *  `Play.board` arrives as `unknown` on purpose: match.json is a file on disk
 *  that a person or a script can edit, and pitch-data.ts is the authority on
 *  what a board may contain. Running it through the same parsers the stored
 *  documents use means a hand-edited play cannot put anything into the renderer
 *  that a hand-edited document could not. */
export function playBoard(p: Play): Board {
  const raw = (p.board ?? {}) as Record<string, unknown>;
  const j = (v: unknown) => (v === undefined ? null : JSON.stringify(v));
  return {
    magnets: parseMagnets(j(raw.magnets)),
    shapes: parseShapes(j(raw.shapes)),
    view: parseView(typeof raw.view === "string" ? raw.view : null),
    orient: parseOrient(typeof raw.orient === "string" ? raw.orient : null),
    lines: parseLines(typeof raw.lines === "string" ? raw.lines : null),
    grass: parseGrass(typeof raw.grass === "string" ? raw.grass : null),
    kits: parseKits(raw.kits === undefined ? null : JSON.stringify(raw.kits)),
    stripes: raw.stripes !== false,
  };
}

/** The line under a play, in a scouting report's own format: what it was, when,
 *  and who. InStat prints "30.03.2022 Ecuador 1:1 Argentina 40:09"; ours adds
 *  the thing that matters more to the coach who filtered for it — the type and
 *  the shirt numbers. */
export function playCaption(p: Play, report: MatchReport | null): string {
  const meta = report?.meta;
  const fixture = [meta?.home_name, meta?.away_name].filter(Boolean).join(" v ");
  const who = p.players?.length ? `#${p.players.join(", #")}` : "";
  return [
    fixture,
    `${PLAY_TYPES[p.type] ?? p.type} · ${p.clock}`,
    PLAY_OUTCOMES[p.outcome] ?? p.outcome,
    who,
  ]
    .filter(Boolean)
    .join(" — ");
}

/** Sorted the way a coach reads a match: forwards through the clock. */
export const byClock = (a: Play, b: Play) => a.minute - b.minute;

/** What a play board rests on, and it is not much.
 *
 *  `plays.py` opens with "EVERYTHING HERE IS SYNTHETIC" and then claims every
 *  block that renders one carries that tier. It did not: the pitch node had a
 *  caption and no provenance, so a diagram captioned "Counter-attack · 63:12 —
 *  Goal — #7, #10, #4" printed in a scouting report with nothing to say that
 *  there is no minute-63 counter-attack, no goal, and that the shirt numbers
 *  were sampled at random out of the real team sheet.
 */
export const PLAY_NOTE =
  "Generated, not detected. The moment, its outcome and the shirts on the board are produced by a placeholder — nothing in the pipeline finds a play.";
