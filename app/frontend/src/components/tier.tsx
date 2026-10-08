"use client";

import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { TIER, type Tier as TierKey } from "@/lib/report";
import { cn } from "@/lib/utils";

/** The whole honesty apparatus, in one badge.
 *
 *  A badge plus a tooltip, never a paragraph: the argument repeated six times
 *  per screen is the argument nobody reads. /checks carries the long version.
 */
/* Colour is spent only where it makes a claim.
 *
 *   ok    cyan    — an instrument colour. Deliberately NOT green: green would
 *                   read as approval, and nothing here is approved.
 *   rel   chalk   — a rank is not a measurement, so it gets no hue at all.
 *   prov  sodium  — the one thing worth catching an eye. */
const COLOR: Record<TierKey, string> = {
  // Same chalk as `rel`, not a louder warning. This lands on most fields on
  // most screens, and a page where everything shouts is a page nobody reads.
  not_evaluated: "text-muted-foreground border-border border-dashed",
  ok: "text-ok border-ok/40",
  rel: "text-muted-foreground border-border",
  prov: "text-warn border-warn/40",
  withheld: "text-warn border-warn/40 border-dashed",
};

/** `why` is the FIELD-specific reason out of `tierOf`, and it is not optional
 *  decoration: the tier alone says how sturdy a number is, never what is wrong
 *  with THIS one. Contamination is the case that proves it — every per-player
 *  figure already read `provisional` before contamination was measured, so the
 *  badge did not move when the reason changed from "never measured" to
 *  "measured, and a third of this row is someone else". A tier computed and
 *  never rendered is indistinguishable from a correct one, so pass it. */
export function Tier({ t, why, className }: { t: TierKey; why?: string; className?: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge
          variant="outline"
          className={cn(
            "h-4 cursor-help px-1.5 text-[10px] tracking-wide uppercase",
            COLOR[t],
            className,
          )}
        >
          {TIER[t].label}
        </Badge>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs text-pretty">
        {why ? `${why} ${TIER[t].why}` : TIER[t].why}
      </TooltipContent>
    </Tooltip>
  );
}

/** A value the held-out run does not stand behind — set as a dimension with a
 *  tolerance, the way an engineering drawing states a measurement it knows the
 *  error of. Mono because it is an instrument reading; the dotted rule with end
 *  ticks (`.tol`) is the band.
 *
 *  Calm on purpose, never red: this appears on every distance for twenty-two
 *  players, and a page of warnings is a page nobody reads. */
export function Prov({ children }: { children: React.ReactNode }) {
  return <span className="tol readout text-muted-foreground">{children}</span>;
}
