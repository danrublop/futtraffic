/** Navigation helpers that know about the desktop static export.
 *
 *  A static export cannot render dynamic segments on demand, so in the
 *  desktop build /players/[trackId] does not exist and track links go to
 *  /track?t=<id> instead. NEXT_PUBLIC_FUTTRAFFIC_DESKTOP is inlined at build
 *  time — the branch costs nothing at runtime and web builds keep the
 *  pretty URLs. */

const DESKTOP = process.env.NEXT_PUBLIC_FUTTRAFFIC_DESKTOP === "1";

export function trackHref(trackId: number | string, matchId: string): string {
  const m = encodeURIComponent(matchId);
  return DESKTOP ? `/track?t=${trackId}&m=${m}` : `/players/${trackId}?m=${m}`;
}
