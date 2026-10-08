"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useEffect } from "react";

import type { TabType } from "@/lib/tabs";

/** Send a legacy surface route into the workspace as a destination.
 *
 *  `/report`, `/review` and `/players` were full pages before every surface
 *  became a pane. They are still linked from the match list, the player view,
 *  setup, and — the one that matters most — the clip URLs people copy and
 *  paste. Left as pages they were a trap: the header tools split within the
 *  workspace, so a report opened from the match list had nothing to split
 *  against and the tool opened its surface full width instead.
 *
 *  So they redirect rather than render. The whole query string is carried
 *  through, which is what keeps `?t=` (a clip's timestamp) and `?track=`
 *  working — VideoPanel reads both straight off the URL.
 *
 *  `replace`, not `push`: this URL should not sit in history as a step to go
 *  back to, or Back would bounce you straight forward again.
 */
export function ToWorkspace({ view, tab }: { view: TabType; tab?: string }) {
  const router = useRouter();
  const params = useSearchParams();

  useEffect(() => {
    const q = new URLSearchParams(params.toString());
    q.set("v", view);
    // `/players` is not a destination any more, it is one entity view inside
    // Data — so the redirect has to say WHICH view, or it lands on Facts.
    if (tab) q.set("tab", tab);
    router.replace(`/workspace?${q.toString()}`);
  }, [router, params, view, tab]);

  return null;
}
