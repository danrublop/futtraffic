"use client";

import { ToWorkspace } from "@/components/to-workspace";

/** The route survives as a REDIRECT onto the Data surface's Players view —
 *  see components/to-workspace.tsx.
 *
 *  This file used to also hold `Squads` and `Tracks`, the per-track table: the
 *  only screen that ever showed what the pipeline itself measured (distance,
 *  clamp share, gaps, per track). They lost their last importer when the match
 *  report was removed and were deleted rather than left as a dead route.
 */
export default function PlayersPage() {
  return <ToWorkspace view="data" tab="our-players" />;
}
