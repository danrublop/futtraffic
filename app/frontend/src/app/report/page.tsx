"use client";

import { ToWorkspace } from "@/components/to-workspace";

/** THE MATCH REPORT IS GONE. This is its forwarding address.
 *
 *  The report was a second way to read one match: it rendered `match.json`
 *  directly and made no network call, while every other surface reads the
 *  store. Two paths to the same numbers is how a screen and a table come to
 *  disagree, so the surface was removed and the database became the one place
 *  a number lives — see `database-is-the-backbone (not published)`.
 *
 *  The FILE stays, at ten lines, because people copied these URLs. A redirect
 *  is not a surface; it is what `/players` and `/review` already are. And it
 *  has to be a route rather than a `next.config.ts` redirect, because the
 *  desktop build is `output: "export"` and static exports have no redirects.
 *
 *  `v=data` and nothing else: the whole query string rides along, so `?m=` —
 *  the app's match lens — survives the hop.
 */
export default function ReportPage() {
  return <ToWorkspace view="data" />;
}
