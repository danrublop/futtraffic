import type { Match } from "./types";

/** ONE place that turns a job state into words a coach understands and exactly
 *  ONE next action. Every screen reads this.
 *
 *  It exists because the old UI let each page invent its own wording, so the
 *  same job read "queued", "waiting" and "needs you" on three screens, and two
 *  of them offered four buttons of equal weight. A person looking at a match
 *  should never have to work out which control matters.
 */

export type Tone = "idle" | "busy" | "blocked" | "ready" | "failed";

export interface StatusView {
  /** Two or three words. Goes on the card and in the header. */
  label: string;
  /** One sentence, plain English, no jargon. What is happening or needed. */
  detail: string;
  tone: Tone;
  /** The single thing to do next, or null when there is nothing to do. */
  action: { label: string; href?: string; kind?: "retry" } | null;
  /** True while the pipeline is working and the UI should keep polling. */
  live: boolean;
}

const pct = (n: number) => `${Math.round(n * 100)}%`;

export function statusOf(match: Match): StatusView {
  const to = (path: string) => `${path}?m=${encodeURIComponent(match.id)}`;

  if (!match.upload_complete) {
    const have = match.uploaded_bytes;
    const want = match.total_bytes ?? 0;
    return {
      label: "Upload unfinished",
      detail: want
        ? `${(have / 1e9).toFixed(2)} of ${(want / 1e9).toFixed(2)} GB arrived. Pick the same file again to carry on.`
        : "The upload stopped before the whole file arrived.",
      tone: "blocked",
      action: { label: "Resume upload", href: "/" },
      live: false,
    };
  }

  switch (match.status) {
    case "calibrating":
      return {
        label: "Needs setup",
        detail:
          match.calibration_auto && !match.calibration_auto.accepted
            ? "We tried to find the pitch on our own and could not — mark it once by hand, about a minute."
            : "Mark where the pitch is, once. Click four corners on a still from your video — about a minute.",
        tone: "blocked",
        action: { label: "Mark the pitch", href: to("/setup") },
        live: false,
      };

    case "queued":
      return {
        label: "Queued",
        detail: "Waiting for a free slot. It will start on its own.",
        tone: "idle",
        action: null,
        live: true,
      };

    case "tracking":
      return {
        label: "Finding players",
        detail:
          "Watching every frame and following each player. This is the slow part — hours on a full match.",
        tone: "busy",
        action: null,
        live: true,
      };

    case "identifying":
      return {
        label: "Matching players",
        detail: "Joining the pieces of each player's run into one player.",
        tone: "busy",
        action: null,
        live: true,
      };

    case "done":
      return {
        label: "Ready",
        detail: "Processed. Watch it back or read the report.",
        tone: "ready",
        action: { label: "Watch", href: to("/review") },
        live: false,
      };

    case "failed":
      return {
        label: "Stopped",
        detail: match.error || "Something went wrong and it stopped.",
        tone: "failed",
        action: { label: "Try again", kind: "retry" },
        live: false,
      };
  }
}

/** Coarse progress for the bar. Deliberately not a time estimate: jobs.py
 *  reports stage-based progress with no per-frame callback, so any "12 minutes
 *  left" would be invented, and an invented ETA is the fastest way to lose
 *  trust in every other number on the page. */
export function progressOf(match: Match): { value: number; text: string } | null {
  if (!match.upload_complete && match.total_bytes) {
    const v = match.uploaded_bytes / match.total_bytes;
    return { value: v * 100, text: `${pct(v)} uploaded` };
  }
  if (match.status === "tracking" || match.status === "identifying") {
    return { value: match.progress * 100, text: pct(match.progress) };
  }
  return null;
}

export const TONE_CLASS: Record<Tone, string> = {
  idle: "text-muted-foreground",
  busy: "text-info",
  blocked: "text-warn",
  ready: "text-ok",
  failed: "text-bad",
};

export function matchTitle(match: Match): string {
  return (match.name || "").replace(/\s*\d{4}-\d{2}-\d{2}$/, "").trim() || "Untitled match";
}

/** The uploader stashes the date as a trailing " YYYY-MM-DD" on the name,
 *  because the create endpoint only takes `?name=`. Convention, not schema. */
export function matchDate(match: Match): string | null {
  const m = /(\d{4}-\d{2}-\d{2})\s*$/.exec(match.name || "");
  return m ? m[1] : null;
}

/** The upload name, but only when it says something the fixture headline and
 *  the date beside it do not.
 *
 *  The match row rendered `match.name` verbatim under a headline it usually
 *  repeats: "Westport FC 3 - 1 Harbour City FC" above, then
 *  "2026-08-06 · Westport FC v Harbour City FC 2026-08-06" below - the fixture
 *  twice and the date twice. The card's own comment says a line should only be
 *  kept when it "carries something the row does not"; this applies that to the
 *  name. Strip the date convention, drop both team names and the joining words,
 *  and if what is left is not a word, there was nothing to add.
 */
export function matchSubtitle(match: Match, home?: string, away?: string): string | null {
  const title = matchTitle(match);
  if (title === "Untitled match") return null;
  const rest = [home, away]
    .filter((s): s is string => !!s)
    .reduce((acc, team) => acc.split(team).join(" "), title)
    // "v", "vs" and "-" are how a fixture is written, not information.
    .replace(/\b(vs?|versus)\b/gi, " ")
    .replace(/[-–—|,]/g, " ")
    .trim();
  return /[a-z0-9]/i.test(rest) ? title : null;
}
