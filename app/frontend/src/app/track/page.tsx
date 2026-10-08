"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";

import { PlayerView } from "../players/[trackId]/player-view";

/** Desktop-only track page. The static export cannot serve /players/[trackId]
 *  (no server to render dynamic segments), so in the desktop build links point
 *  here — see lib/nav.ts. Web builds never link to this route. */
function TrackFromQuery() {
  const trackId = useSearchParams().get("t");
  if (!trackId) return null;
  return <PlayerView trackId={trackId} />;
}

export default function TrackPage() {
  // useSearchParams must sit inside Suspense in a static export.
  return (
    <Suspense>
      <TrackFromQuery />
    </Suspense>
  );
}
