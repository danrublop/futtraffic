"use client";

import Link from "next/link";
import { useState } from "react";
import { AlertTriangle, Info, Loader2, RotateCw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useMatch } from "@/hooks/use-matches";
import { api } from "@/lib/api";
import { progressOf, statusOf } from "@/lib/status";

/** Two thin strips, at most, above the page.
 *
 *  This replaces the paragraph of caveats that used to sit on every screen.
 *  The full argument lives on /checks; repeating it six times taught people to
 *  scroll past it, which is the opposite of what it is for. One sentence, one
 *  link, and it stays out of the way.
 */
export function JobBanner() {
  const { match, refresh } = useMatch();
  const [retrying, setRetrying] = useState(false);
  const [dismissed, setDismissed] = useState(false);

  const view = match ? statusOf(match) : null;
  const progress = match ? progressOf(match) : null;

  const retry = async () => {
    if (!match) return;
    setRetrying(true);
    try {
      await api.retry(match.id);
      await refresh();
    } finally {
      setRetrying(false);
    }
  };

  return (
    <>
      {view && (view.tone === "busy" || view.tone === "blocked" || view.tone === "failed") && (
        <div className="flex items-center gap-3 border-b px-4 py-2 text-sm md:px-6">
          {view.tone === "busy" ? (
            <Loader2 className="text-info size-4 shrink-0 animate-spin" />
          ) : (
            <AlertTriangle
              className={`size-4 shrink-0 ${view.tone === "failed" ? "text-bad" : "text-warn"}`}
            />
          )}
          <span className="min-w-0 flex-1 truncate">
            <span className="font-medium">{view.label}.</span>{" "}
            <span className="text-muted-foreground">{view.detail}</span>
          </span>
          {progress && (
            <span className="text-muted-foreground hidden shrink-0 items-center gap-2 sm:flex">
              <Progress value={progress.value} className="h-1.5 w-24" />
              {progress.text}
            </span>
          )}
          {view.action?.kind === "retry" && (
            <Button size="sm" variant="outline" onClick={retry} disabled={retrying}>
              <RotateCw className={retrying ? "animate-spin" : ""} />
              Try again
            </Button>
          )}
          {view.action?.href && (
            <Button size="sm" asChild>
              <Link href={view.action.href}>{view.action.label}</Link>
            </Button>
          )}
        </div>
      )}

      {/* The calibration this match is using was solved for DIFFERENT footage.
          Never hidden: if the camera was nudged between matches, every distance
          on the page shifts, and being told is the only way anyone suspects it. */}
      {match?.calibration_inherited_from && !dismissed && (
        <div className="text-muted-foreground flex items-center gap-3 border-b px-4 py-2 text-xs md:px-6">
          <Info className="size-3.5 shrink-0" />
          <span className="min-w-0 flex-1">
            Pitch setup reused from <b className="text-foreground">{match.calibration_inherited_from}</b>{" "}
            — same ground, same club, same frame size. If the camera was moved or
            re-aimed since, redo it: the distances shift and nothing else will look wrong.
          </span>
          <Button size="sm" variant="ghost" asChild>
            <Link href={`/setup?m=${encodeURIComponent(match.id)}`}>Redo setup</Link>
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setDismissed(true)}>
            Dismiss
          </Button>
        </div>
      )}
    </>
  );
}
