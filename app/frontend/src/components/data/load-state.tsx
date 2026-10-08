"use client";

import { RefreshCw, TriangleAlert } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/** A FETCH THAT FAILED MUST NEVER RENDER AS A CLAIM ABOUT THE DATA.
 *
 *  This exists because of a specific, live bug. The players table prints an em
 *  dash for a player with no confirmed tracking, and the caption underneath
 *  tells the coach exactly what that means:
 *
 *      "an em dash means no track has been confirmed yet, not a zero"
 *
 *  But the code loading those numbers destructured `{ season }` off `useSeason`
 *  and threw the `error` away. So when the measured endpoint failed, every
 *  player got an em dash and the caption made a specific, confident, WRONG
 *  statement about the squad — while the real cause was that the server could
 *  not be reached.
 *
 *  That is the exact failure mode this product exists to avoid, and it is worse
 *  than a blank screen: a blank screen tells you nothing, and this told you
 *  something false with a footnote explaining it.
 *
 *  So the rule these pieces enforce is narrow and absolute: WHEN A SOURCE
 *  FAILS, SAY SO, AND SUPPRESS ANY SENTENCE THAT INTERPRETS ITS ABSENCE. The
 *  section keeps rendering everything it still knows to be true — who is in the
 *  club is a different fetch and is still correct — and stops explaining what
 *  the missing half means.
 */

/** The strip a section shows above a table when one of its sources failed.
 *
 *  Deliberately NOT a full-page replacement. A section usually has more than
 *  one source: the registry knows who exists, the measured season knows what
 *  they ran. Losing the second is not a reason to hide the first. It IS a
 *  reason to stop describing what the missing numbers mean.
 */
export function LoadFailure({
  /** What could not be loaded, in the coach's words: "the measured season",
   *  "the club", "the match list". Not an endpoint path. */
  what,
  error,
  onRetry,
  className,
}: {
  what: string;
  error: string;
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <div
      role="alert"
      className={cn(
        "flex items-start gap-2 rounded-[var(--radius-md)] border border-[var(--warn)]/30 bg-[var(--warn)]/10 px-3 py-2 text-[12px]",
        className,
      )}
    >
      <TriangleAlert className="mt-0.5 size-3.5 shrink-0 text-[var(--warn)]" />
      <span className="min-w-0 flex-1 leading-relaxed">
        <span className="text-[var(--ink)]">Could not load {what}.</span>{" "}
        <span className="text-muted-foreground">
          The numbers it supplies are missing below, and that is a connection
          problem — not a statement about your squad.
        </span>
        <span className="text-muted-foreground readout mt-0.5 block text-[11px]">{error}</span>
      </span>
      {onRetry && (
        <Button variant="ghost" size="sm" onClick={onRetry} className="h-6 shrink-0 px-2 text-[11px]">
          <RefreshCw className="size-3" />
          Retry
        </Button>
      )}
    </div>
  );
}

/** A section's whole body could not load. Used when there is no partial truth
 *  worth rendering — the one source failed and there is nothing else to show. */
export function LoadEmpty({
  what,
  error,
  onRetry,
}: {
  what: string;
  error?: string | null;
  onRetry?: () => void;
}) {
  return (
    <div className="grid place-items-center gap-3 py-12 text-center">
      <p className="max-w-[46ch] text-[13px] text-[var(--ink-soft)]">Could not load {what}.</p>
      {error && <p className="readout max-w-[60ch] text-[11px] text-[var(--mute)]">{error}</p>}
      {onRetry && (
        <Button variant="outline" size="sm" onClick={onRetry}>
          <RefreshCw className="size-3.5" />
          Try again
        </Button>
      )}
    </div>
  );
}

/** Whether a caption that INTERPRETS missing data may be shown.
 *
 *  One line, but it is the whole rule, and having it as a named function means
 *  every section states the same thing rather than each one re-deciding: a
 *  sentence explaining what an absent number means is only honest when the
 *  thing that supplies those numbers actually arrived.
 */
export const mayExplainAbsence = (sourceError: string | null | undefined): boolean =>
  !sourceError;
