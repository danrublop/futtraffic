import { Star } from "lucide-react";

import { cn } from "@/lib/utils";

/** THE RATING PILL, shared by the report's player table and the data
 *  section's player registry.
 *
 *  Extracted rather than copied for the reason sortForTab already lives in
 *  lib/player-cols: two screens that show the same number must not be able to
 *  disagree about how it looks. A second implementation drifts by one shade
 *  the first time somebody adjusts a threshold, and a 6.9 that is orange on
 *  one screen and green on the other is worse than either colour.
 *
 *  GREEN AT 7.0 AND ABOVE, orange below - two buckets, because they read at a
 *  glance and a finer scale would be inventing precision the rating does not
 *  have. The hexes are literal and identical on both screens on purpose.
 *
 *  `best` is the blue star pill: the top rating of whatever set is on screen.
 *  What that set IS differs by caller - the match for the report, the squad
 *  for the registry - and that is the caller's business, which is why it is a
 *  prop and not computed here.
 */
export function RatingChip({
  value,
  best = false,
  className,
}: {
  value: number;
  best?: boolean;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "readout z-10 inline-flex w-fit items-center gap-1 rounded-md px-1.5 py-0.5 text-[10px] leading-none font-bold text-black",
        best ? "bg-[#3b82f6] text-white" : value >= 7 ? "bg-[#4ade80]" : "bg-[#fb923c]",
        className,
      )}
    >
      {value.toFixed(1)}
      {best && <Star className="size-2.5 fill-current" />}
    </span>
  );
}
