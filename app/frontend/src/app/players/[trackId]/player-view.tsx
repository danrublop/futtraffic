"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ReportGate } from "@/components/no-report";
import { Occupancy, Pitch, flipY } from "@/components/pitch";
import { Prov, Tier } from "@/components/tier";
import { api } from "@/lib/api";
import { buildRows, DEFAULT_PITCH, teamColor, teamName, tierOf } from "@/lib/report";
import type { MatchReport, TrackSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

/** ONE TRACK, UNDER REVIEW.
 *
 *  This is not a player profile and it deliberately stops looking like one.
 *  Measured this session: applying our identity error takes sender accuracy
 *  from 100% to 9.1% against a 4.5% chance floor, and identity reaches chance
 *  roughly five minutes into a match. So every number below describes a
 *  TRAJECTORY the tracker held, and the question the page answers is whether
 *  that trajectory belongs to one person — not how well that person played.
 *
 *  Hence the order: identity and the speed clamp first, physical quantities
 *  second, and no comparison with anybody else at all.
 */

export function PlayerView({ trackId }: { trackId: string }) {
  return (
    <ReportGate>
      {(report, id) => <Track report={report} matchId={id} trackId={trackId} />}
    </ReportGate>
  );
}

/** The per-track digest row, for the one field match.json does not carry: the
 *  best jersey OCR confidence seen on this track (app.py::_summarise). Optional
 *  garnish — the digest 404s until tracks.json exists, and the page is complete
 *  without it, so a failure is swallowed rather than shown as an error. */
function useDigestRow(matchId: string, trackId: string) {
  const [row, setRow] = useState<{ key: string; t: TrackSummary | null }>({
    key: "",
    t: null,
  });
  const key = `${matchId}:${trackId}`;
  useEffect(() => {
    let off = false;
    api
      .getTracksDigest(matchId)
      .then((d) => {
        if (!off) setRow({ key, t: d.tracks.find((t) => String(t.id) === trackId) ?? null });
      })
      .catch(() => !off && setRow({ key, t: null }));
    return () => {
      off = true;
    };
  }, [matchId, trackId, key]);
  return row.key === key ? row.t : null;
}

function Track({
  report,
  matchId,
  trackId,
}: {
  report: MatchReport;
  matchId: string;
  trackId: string;
}) {
  const rows = useMemo(() => buildRows(report), [report]);
  const row = rows.find((r) => String(r.p.id) === trackId);
  const digest = useDigestRow(matchId, trackId);
  const back = `/workspace?v=data&m=${encodeURIComponent(matchId)}`;

  if (!row) {
    return (
      <div className="mx-auto grid w-full max-w-3xl gap-4">
        <Back href={back} />
        <Card>
          <CardContent className="grid gap-2">
            <h1 className="font-semibold">Track {trackId} is not in this match</h1>
            <p className="text-muted-foreground text-sm">
              Track ids are per match and are renumbered whenever a match is re-processed.
            </p>
            <Link
              href={`/players?m=${encodeURIComponent(matchId)}`}
              className="text-primary text-sm hover:underline"
            >
              See every track
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  const p = row.p;
  const meta = report.meta ?? {};
  const L = meta.pitch?.length ?? DEFAULT_PITCH.length;
  const W = meta.pitch?.width ?? DEFAULT_PITCH.width;
  const heat = report.heatmaps?.[String(p.id)] ?? [];
  const [ax, ay] = p.avg_position;
  const clamp = (p.clamped_frac || 0) * 100;

  return (
    <div className="mx-auto grid w-full max-w-3xl gap-4">
      <Back href={back} />

      <div className="flex items-center gap-3">
        <span
          className={cn(
            "grid size-11 shrink-0 place-items-center rounded-full text-lg text-white",
            "readout",
            p.jersey != null ? teamColor(p.team) : "bg-muted text-muted-foreground text-xs",
          )}
        >
          {p.jersey ?? "?"}
        </span>
        <div className="min-w-0">
          <h1 className="font-condensed truncate text-xl tracking-wide uppercase">
            Track {p.id}
          </h1>
          <p className="text-muted-foreground text-sm">
            {teamName(report, p.team)} · <span className="readout">{row.minutes.toFixed(0)}</span>{" "}
            min tracked
          </p>
        </div>
      </div>

      {/* ------------------------------------------------ the two review facts */}
      <div className="grid gap-3 sm:grid-cols-2">
        <Card className="gap-0 py-4">
          <CardContent className="grid gap-2 px-4">
            <div className="flex items-center justify-between gap-2">
              <span className="eyebrow">Identity</span>
              <Unverified />
            </div>
            <div
              className={cn(
                "readout text-2xl",
                p.jersey != null ? "text-warn" : "text-bad text-lg",
              )}
            >
              {p.jersey != null ? `#${p.jersey}` : "none read"}
            </div>
            <p className="text-muted-foreground text-xs leading-relaxed">
              {digest?.confidence != null ? (
                <>
                  Best jersey read on this track scored{" "}
                  <span className="readout text-foreground">{digest.confidence.toFixed(2)}</span>,
                  an upper bound over every frame.
                </>
              ) : p.jersey != null ? (
                "Read by jersey OCR. No per-read confidence is stored for this track."
              ) : (
                "OCR read no number, so this track cannot be joined to a person or to another match."
              )}
            </p>
          </CardContent>
        </Card>

        <Card className="gap-0 py-4">
          <CardContent className="grid gap-2 px-4">
            <div className="flex items-center justify-between gap-2">
              <span className="eyebrow">Speed clamp</span>
              <QTier field="players.clamped_frac" fields={report.fields} />
            </div>
            <div className={cn("readout text-2xl", clamp > 1 ? "text-warn" : "text-foreground")}>
              {clamp.toFixed(1)}
              <span className="text-muted-foreground ml-1 text-sm">% of steps</span>
            </div>
            <p className="text-muted-foreground text-xs leading-relaxed">
              Steps held at the 9.5 m/s ceiling — the one cheap correlate of an identity switch.
              {clamp > 1 && " Some of the distance below may belong to another player."}
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="flex flex-wrap gap-2">
        <Button asChild variant="outline" className="h-11 sm:h-9">
          <Link href={`/review?m=${encodeURIComponent(matchId)}&track=${p.id}`}>
            Check this track on video
          </Link>
        </Button>
      </div>

      {/* --------------------------------------------------------- quantities */}
      <Card>
        <CardContent className="grid gap-3">
          <h2 className="eyebrow">Measured on this track</h2>
          <dl className="grid gap-x-8 gap-y-3 sm:grid-cols-2">
            <Q
              label="Minutes"
              field="players.minutes"
              fields={report.fields}
              value={row.minutes.toFixed(0)}
              unit="min"
            />
            <Q
              label="Distance"
              field="players.distance_km"
              fields={report.fields}
              value={row.distance == null ? "refused" : row.distance.toFixed(2)}
              unit={row.distance == null ? undefined : "km"}
              band
            />
            <Q
              label="Peak speed"
              field="players.top_speed_kmh"
              fields={report.fields}
              value={row.peak == null ? "refused" : row.peak.toFixed(1)}
              unit={row.peak == null ? undefined : "km/h"}
              band
            />
            <Q
              label="Sprints"
              field="players.sprints"
              fields={report.fields}
              value={row.sprints == null ? "withheld" : String(row.sprints)}
              band
            />
          </dl>
          {row.gapNote && (
            <p className="text-warn text-xs leading-relaxed">{row.gapNote}</p>
          )}
        </CardContent>
      </Card>

      {/* ---------------------------------------------------------- occupancy */}
      <Card>
        <CardContent className="grid gap-3">
          <div className="flex items-center justify-between gap-2">
            <h2 className="eyebrow">Occupancy</h2>
            {/* avg_position, not a nameless "prov": it carries the same
                contamination flag as the distance above, and the mean of a
                mixed id is a mean of two people's positions. */}
            <QTier field="players.avg_position" fields={report.fields} />
          </div>
          <Pitch length={L} width={W}>
            <Occupancy
              points={heat}
              width={W}
              color={p.team === "home" ? "var(--home)" : "var(--away)"}
              id={String(p.id)}
            />
            <circle
              cx={ax}
              cy={flipY(ay, W)}
              r={2.4}
              className={cn(p.team === "home" ? "fill-home" : "fill-away")}
              stroke="white"
              strokeWidth={0.35}
            />
          </Pitch>
          <p className="text-muted-foreground text-xs leading-relaxed">
            Mean position{" "}
            <Prov>
              {ax.toFixed(1)} × {ay.toFixed(1)}
            </Prov>{" "}
            m, attacking left to right.{" "}
            {heat.length
              ? `${heat.length} sampled positions.`
              : "No sampled positions stored — marker only."}{" "}
            Read it for review, not for a role: two separated clouds, or a mean sitting where nobody
            plays, is what an identity switch looks like from here.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------- small parts

/** A bare badge for a field, with the reason `tierOf` returns attached. */
function QTier({ field, fields }: { field: string; fields?: MatchReport["fields"] }) {
  const t = tierOf(field, fields);
  return <Tier t={t.tier} why={t.why} />;
}

function Back({ href }: { href: string }) {
  return (
    <Link
      href={href}
      className="text-muted-foreground hover:text-foreground flex min-h-11 w-fit items-center gap-1.5 text-sm"
    >
      <ArrowLeft className="size-4" />
      Match report
    </Link>
  );
}

/** Identity gets its own badge rather than a tier: the three tiers grade how
 *  well a QUANTITY was measured, and identity is not a quantity — it is a
 *  claim that has been tested and failed. Rust, because rust means failed. */
function Unverified() {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge
          variant="outline"
          className="text-bad border-bad/40 h-4 cursor-help px-1.5 text-[10px] tracking-wide uppercase"
        >
          unverified
        </Badge>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs text-pretty">
        Identity is never verified against a person. Measured this session: injecting our identity
        error takes sender accuracy from 100% to 9.1% against a 4.5% chance floor, and identity
        reaches chance about five minutes into a match. Everything on this page describes the
        trajectory, not the player.
      </TooltipContent>
    </Tooltip>
  );
}

/** Label, value, tier — two columns on a phone, which is the most a phone can
 *  carry: the quantity and the band it was measured to.
 *
 *  The tier is RESOLVED, never passed: these three read `provisional` both
 *  before and after contamination was measured, so a literal here would have
 *  gone on saying "accuracy never measured" about a figure whose identity error
 *  now HAS a number. `tierOf` carries the reason with the badge. */
function Q({
  label,
  value,
  unit,
  field,
  fields,
  band,
}: {
  label: string;
  value: string;
  unit?: string;
  field: string;
  fields?: MatchReport["fields"];
  band?: boolean;
}) {
  const t = tierOf(field, fields);
  const n = (
    <>
      {value}
      {unit && <span className="text-muted-foreground ml-1 text-xs">{unit}</span>}
    </>
  );
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-dashed pb-2">
      <dt className="flex items-center gap-2 text-sm">
        {label}
        <Tier t={t.tier} why={t.why} />
      </dt>
      <dd className="readout text-lg">{band ? <Prov>{n}</Prov> : n}</dd>
    </div>
  );
}
