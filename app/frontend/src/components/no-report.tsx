"use client";

import Link from "next/link";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { PickMatchGate } from "@/components/match-picker";
import { Skeleton } from "@/components/ui/skeleton";
import { useReport } from "@/hooks/use-matches";
import { api } from "@/lib/api";
import { statusOf, TONE_CLASS } from "@/lib/status";
import type { Match, MatchReport } from "@/lib/types";

/** Load the report, or say why there isn't one. All three results screens go
 *  through here so they cannot each invent their own empty state. */
export function ReportGate({
  children,
}: {
  children: (report: MatchReport, matchId: string, match: Match | null) => React.ReactNode;
}) {
  const { id, match, report, error, loading, refresh } = useReport();

  if (loading) return <Skeleton className="mx-auto h-64 w-full max-w-5xl" />;
  if (!id || !match || !match.has_match || (error && !report)) {
    return <NoReport match={match} error={error} onChange={refresh} />;
  }
  if (!report) return <Skeleton className="mx-auto h-64 w-full max-w-5xl" />;
  return <>{children(report, id, match)}</>;
}

/** The one empty state for every results screen.
 *
 *  It never invents wording: the sentence and the single next action both come
 *  out of statusOf(), so a match that reads "Needs setup" here reads "Needs
 *  setup" on the match list and offers the same button.
 */
export function NoReport({
  match,
  error,
  onChange,
}: {
  match: Match | null;
  error?: string | null;
  onChange?: () => void;
}) {
  const [retrying, setRetrying] = useState(false);

  // No match at all is not an error state, it is a question — so ask it here
  // rather than sending someone back to the list and losing the screen they
  // were on. See components/match-picker.tsx.
  if (!match) {
    return <PickMatchGate detail="Choose one to see its numbers." />;
  }

  const view = statusOf(match);
  const retry = async () => {
    setRetrying(true);
    try {
      await api.retry(match.id);
      onChange?.();
    } finally {
      setRetrying(false);
    }
  };

  // JobBanner already carries this sentence for the busy/blocked/failed tones.
  // Repeating it three inches lower is how a page ends up looking like it is
  // shouting; the button below is the part that is not duplicated.
  const banner = view.tone === "busy" || view.tone === "blocked" || view.tone === "failed";

  return (
    <Shell title="No numbers for this match yet">
      <p className="text-sm">
        <span className={TONE_CLASS[view.tone]}>{view.label}.</span>{" "}
        {!banner && <span className="text-muted-foreground">{view.detail}</span>}
      </p>
      {error && <p className="text-bad text-sm">{error}</p>}
      {view.action?.kind === "retry" ? (
        <Button size="sm" variant="outline" className="w-fit" onClick={retry} disabled={retrying}>
          {retrying ? "Retrying…" : view.action.label}
        </Button>
      ) : view.action?.href ? (
        <Button size="sm" asChild className="w-fit">
          <Link href={view.action.href}>{view.action.label}</Link>
        </Button>
      ) : null}
    </Shell>
  );
}

function Shell({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card className="mx-auto w-full max-w-5xl">
      <CardContent className="grid gap-3">
        <h2 className="font-medium">{title}</h2>
        {children}
      </CardContent>
    </Card>
  );
}
