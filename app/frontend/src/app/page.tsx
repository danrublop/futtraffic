"use client";

import { ToWorkspace } from "@/components/to-workspace";

/** THE MATCH LIBRARY IS GONE. This is its forwarding address.
 *
 *  It was a second list of matches: it read the JOB list and joined the season
 *  table onto it, while the Data page's All matches table reads the store. Two
 *  lists of one library is the same problem the match report was — see
 *  `database-is-the-backbone (not published)`.
 *
 *  What it owned and where that went:
 *    - Upload and search  -> `components/library-actions.tsx`, rendered beside
 *                            the Data page's filters. Upload is step one of the
 *                            product and could not go down with the page.
 *    - Job status, retry  -> `JobBanner`, which app-shell.tsx already draws
 *                            on EVERY page, so an in-flight or failed job is
 *                            more visible now, not less.
 *    - The folder chips   -> `components/chip.tsx`; the film gallery still
 *                            draws the same ladder.
 *
 *  A route rather than a `next.config.ts` redirect, because the desktop build
 *  is `output: "export"` and static exports carry no redirects.
 */
export default function HomePage() {
  return <ToWorkspace view="data" />;
}
