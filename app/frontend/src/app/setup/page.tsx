"use client";

import { useState } from "react";
import { Check } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { PickMatchGate } from "@/components/match-picker";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useMatch } from "@/hooks/use-matches";
import { matchTitle } from "@/lib/status";

import { CalibrateStep } from "./calibrate-step";
import { IdentityStep } from "./identity-step";

/** The one manual step in the product, made as small as possible.
 *
 *  Two tabs rather than two pages because to the person doing it this is one
 *  job: tell the system about this match. Where the pitch is, and who is on it.
 */
export default function SetupPage() {
  const { id, match, error, refresh } = useMatch();
  const [tab, setTab] = useState<string | null>(null);

  if (!id)
    return (
      <PickMatchGate detail="Setup belongs to one match — its video is what you mark the pitch on." />
    );

  if (error && !match)
    return (
      <Alert variant="destructive" className="mx-auto max-w-2xl">
        <AlertTitle>Could not load this match</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );

  if (!match)
    return (
      <div className="mx-auto grid w-full max-w-6xl gap-3">
        <Skeleton className="h-10 w-64" />
        <Skeleton className="h-96 w-full" />
      </div>
    );

  const pitchDone = match.has_calibration;
  const current = tab ?? (pitchDone && match.has_tracks ? "names" : "pitch");

  return (
    <div className="mx-auto grid w-full max-w-6xl gap-4">
      <div>
        <h1 className="text-[18px] font-semibold tracking-[-0.2px]">Set up this match</h1>
        <p className="text-muted-foreground text-sm">
          {matchTitle(match)} — mark the pitch once, then name the players if you want
          names on the report.
        </p>
      </div>

      <Tabs value={current} onValueChange={setTab}>
        <TabsList>
          <TabsTrigger value="pitch">
            {pitchDone && <Check className="text-ok" />} 1 · Mark the pitch
          </TabsTrigger>
          <TabsTrigger value="names">2 · Name the players</TabsTrigger>
        </TabsList>

        <TabsContent value="pitch" className="mt-4">
          {/* Keyed: "Next match" changes ?m= under a mounted step, and the clicks
              and the still belong to the match they were made on. */}
          <CalibrateStep key={match.id} match={match} onSaved={refresh} />
        </TabsContent>

        <TabsContent value="names" className="mt-4">
          {match.has_tracks ? (
            <IdentityStep match={match} />
          ) : (
            <Alert>
              <AlertTitle>Nothing to name yet</AlertTitle>
              <AlertDescription className="text-muted-foreground">
                The pipeline has not finished following players in this match. This step
                appears on its own when it has — and it is optional even then.
              </AlertDescription>
            </Alert>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
