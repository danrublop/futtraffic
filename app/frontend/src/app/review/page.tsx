"use client";

import { ToWorkspace } from "@/components/to-workspace";

/** The Film route is a REDIRECT into the workspace's film destination. The
 *  viewer itself lives in components/video-panel.tsx and is rendered by the
 *  film pane; this exists only because copied clip URLs point at it, and it
 *  carries their ?t= and ?track= through. */
export default function ReviewPage() {
  return <ToWorkspace view="film" />;
}
