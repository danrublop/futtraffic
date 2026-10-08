"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { MatchPicker } from "@/components/match-picker";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useMatch } from "@/hooks/use-matches";
import { api } from "@/lib/api";
import { matchTitle, statusOf } from "@/lib/status";
import type { Match, MatchReport, PlayerStats, TracksDigest } from "@/lib/types";

/** 11 a side; officials are excluded downstream. */
const EXPECTED_TRACKS = 22;
/** stats.build_match drops tracks shorter than this. */
const DEBRIS_S = 10;
/** stats.MAX_SPEED, in the units shown here. */
const MAX_SPEED_KMH = 9.5 * 3.6;

type Grade = "ok" | "warn" | "fail" | "na";

const GRADE_CLASS: Record<Grade, string> = {
  ok: "text-ok",
  warn: "text-warn",
  fail: "text-bad",
  na: "text-muted-foreground",
};
const GRADE_WORD: Record<Grade, string> = {
  ok: "PASS",
  warn: "CHECK",
  fail: "FAIL",
  na: "n/a",
};

/** Physiology gates. Not opinions: a player at 0.5 km/90 or 45 km/h is a
 *  pipeline bug, and this exists to say so before the number reaches the report
 *  looking like a fact. Ported from web/results.html :: sanity(). */
function sanity(p: PlayerStats, L: number, matchMin: number | null) {
  const per90 = p.minutes && p.distance_km != null ? (p.distance_km / p.minutes) * 90 : null;
  // Keeper heuristic: an average position inside a penalty area. Crude, said to
  // be crude, and the only role signal the pipeline produces.
  const keeper = p.avg_position[0] <= 16.5 || p.avg_position[0] >= L - 16.5;
  const band: [number, number] = keeper ? [3, 5] : [8, 12];
  const checks: { k: string; s: Grade; t: string }[] = [];

  if (per90 == null) checks.push({ k: "dist", s: "na", t: "no minutes" });
  else
    checks.push({
      k: "dist",
      // Outside the band by more than half again is not "a quiet game", it is broken.
      s:
        per90 >= band[0] && per90 <= band[1]
          ? "ok"
          : per90 >= band[0] * 0.6 && per90 <= band[1] * 1.4
            ? "warn"
            : "fail",
      t: `${per90.toFixed(2)} km/90 vs ${band[0]}–${band[1]}`,
    });

  const spd = p.top_speed_kmh;
  checks.push({
    k: "speed",
    s:
      spd == null
        ? "na"
        : spd >= MAX_SPEED_KMH - 0.2
          ? "fail" // pinned at the clamp: an upper bound, not a measurement
          : spd >= 25 && spd <= 32
            ? "ok"
            : spd >= 18 && spd <= 34
              ? "warn"
              : "fail",
    t: spd == null ? "–" : `${spd.toFixed(1)} km/h vs 25–32`,
  });

  const cf = p.clamped_frac;
  checks.push({
    k: "clamp",
    s: cf == null ? "na" : cf > 0.05 ? "fail" : cf > 0.01 ? "warn" : "ok",
    t: cf == null ? "–" : `${(cf * 100).toFixed(1)}% of steps clamped`,
  });

  if (matchMin)
    checks.push({
      k: "min",
      s: p.minutes > matchMin * 1.02 ? "fail" : p.minutes < 5 ? "warn" : "ok",
      t: `${p.minutes} min of ${matchMin.toFixed(1)} filmed`,
    });

  const worst: Grade = checks.some((c) => c.s === "fail")
    ? "fail"
    : checks.some((c) => c.s === "warn")
      ? "warn"
      : checks.every((c) => c.s === "na")
        ? "na"
        : "ok";
  return { per90, keeper, checks, worst };
}

const fmtInt = (n: number | null | undefined) =>
  n == null ? "–" : Math.round(n).toLocaleString("en-US");
const fmtClock = (s: number | null | undefined) =>
  s == null
    ? "–"
    : `${s >= 3600 ? `${Math.floor(s / 3600)}h ` : ""}${Math.floor((s % 3600) / 60)}m ${Math.round(s % 60)}s`;

export function RunPanel() {
  const { id, match } = useMatch();
  // Keyed by the match it came from, so a switch to another match can never
  // render last match's numbers under this match's name while the fetch is in
  // flight — the one bug on an inspection page that must not happen.
  const [got, setGot] = useState<{
    id: string;
    report: MatchReport | null;
    digest: TracksDigest | null;
  } | null>(null);

  // Gated on the flags the status payload already carries. Asking for a
  // match.json that the pipeline has not written yet answers 404, and a handled
  // 404 still prints in the console — which trains people to ignore it.
  const hasMatch = match?.has_match ?? false;
  const hasTracks = match?.has_tracks ?? false;

  useEffect(() => {
    if (!id) return;
    let live = true;
    const load = async () => {
      const [report, digest] = await Promise.all([
        hasMatch ? api.getReport(id).catch(() => null) : null,
        hasTracks ? api.getTracksDigest(id).catch(() => null) : null,
      ]);
      if (live) setGot({ id, report, digest });
    };
    void load();
    return () => {
      live = false;
    };
  }, [id, hasMatch, hasTracks]);

  const report = got?.id === id ? got.report : null;
  const digest = got?.id === id ? got.digest : null;

  // Not a dead end: everything above this panel is real without a match, so
  // this offers the extra rather than blocking the page.
  if (!id)
    return (
      <Card>
        <CardContent className="flex flex-wrap items-center gap-3 text-sm">
          <span className="text-muted-foreground min-w-0 flex-1">
            Everything above is repo-level: it describes the estimator, not any one match. Choose
            a match to also see whether a particular run behaved.
          </span>
          <MatchPicker
            trigger={
              <Button size="sm" variant="outline">
                Choose a match
              </Button>
            }
          />
        </CardContent>
      </Card>
    );

  if (!match) return null;

  return (
    <Card className="min-w-0 gap-4">
      <CardHeader>
        <CardTitle className="text-sm">
          This run — {matchTitle(match)}
          <span className="text-muted-foreground ml-2 font-normal">
            job {match.id.slice(0, 8)}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="grid min-w-0 gap-4">
        <JobFacts match={match} />
        {report?.players?.length ? (
          <>
            <Health match={match} report={report} digest={digest} />
            <Sanity report={report} digest={digest} />
          </>
        ) : (
          <Alert variant="destructive">
            <AlertTitle>
              {statusOf(match).live
                ? "No numbers yet — this run has not reached build_match."
                : "No match.json — there is nothing to inspect."}
            </AlertTitle>
            <AlertDescription>
              {statusOf(match).live
                ? statusOf(match).detail
                : "Either the pipeline never ran or it crashed before build_match. Treat this as a failed run, not an empty one."}
            </AlertDescription>
          </Alert>
        )}
        {digest && <TrackQuality digest={digest} report={report} />}
      </CardContent>
    </Card>
  );
}

function JobFacts({ match }: { match: Match }) {
  const facts: [string, string][] = [
    ["status", statusOf(match).label],
    ["worker attempts", String(match.attempts ?? 0)],
    ["operator time", fmtClock(match.operator_seconds)],
    ["corrections applied", String(match.corrections ?? 0)],
    ["calibration on file", match.has_calibration ? "yes" : "no"],
  ];
  if (match.detector) facts.push(["detector", match.detector === "coreml" ? "CoreML" : "RF-DETR (PyTorch)"]);
  if (match.scene_prior === "active")
    facts.push(["off-pitch filter", `on — ${match.prior_rejected ?? 0} boxes removed`]);
  return (
    <div className="grid gap-2">
      <div className="flex flex-wrap gap-1.5">
        {facts.map(([k, v]) => (
          <span key={k} className="bg-muted rounded-md px-2 py-1 text-xs">
            <span className="text-muted-foreground">{k}</span>{" "}
            <b className="font-medium">{v}</b>
          </span>
        ))}
      </div>
      {match.calibration_inherited_from && (
        <Alert>
          <AlertTitle className="text-warn">
            The pitch setup was reused from “{match.calibration_inherited_from}”.
          </AlertTitle>
          <AlertDescription className="text-muted-foreground">
            It was not solved for this footage — only the frame size was checked. If the
            camera was nudged, every distance here is shifted and nothing else on this page
            would show it.{" "}
            <Link className="text-primary" href={`/setup?m=${encodeURIComponent(match.id)}`}>
              Redo it →
            </Link>
          </AlertDescription>
        </Alert>
      )}
      {match.identity_note && (
        <p className="text-warn text-xs">Identity stage: {match.identity_note}</p>
      )}
      {match.scene_prior && match.scene_prior !== "active" && (
        <p className="text-warn text-xs">Off-pitch filter: {match.scene_prior}</p>
      )}
      {match.ingest_error && (
        <p className="text-warn text-xs">
          Not in the database: {match.ingest_error}. This page still works; season totals and the
          Data pages will not include this match.
        </p>
      )}
    </div>
  );
}

function Tile({
  n,
  unit,
  label,
  sub,
  grade,
}: {
  n: string;
  unit?: string;
  label: string;
  sub?: string;
  grade?: Grade;
}) {
  return (
    <div className="bg-muted rounded-[var(--radius-md)] p-3">
      <div className={`text-lg leading-6 font-medium ${grade ? GRADE_CLASS[grade] : ""}`}>
        {n}
        {unit && (
          <span className="text-muted-foreground ml-0.5 text-xs font-normal">{unit}</span>
        )}
      </div>
      <div className="text-muted-foreground mt-1 text-[11px] leading-snug">
        {label}
        {sub && <span className="block">{sub}</span>}
      </div>
    </div>
  );
}

function digestStats(digest: TracksDigest | null) {
  if (!digest?.tracks?.length) return null;
  const fps = digest.fps || 25;
  const seconds = (t: (typeof digest.tracks)[number]) =>
    t.seconds ?? (t.last_f - t.first_f) / fps;
  const frames = Math.max(...digest.tracks.map((t) => t.last_f)) + 1;
  return {
    fps,
    frames,
    detections: digest.tracks.reduce((s, t) => s + t.n, 0),
    surviving: digest.tracks.filter((t) => seconds(t) >= DEBRIS_S).length,
    total: digest.tracks.length,
    seconds,
  };
}

function Health({
  match,
  report,
  digest,
}: {
  match: Match;
  report: MatchReport;
  digest: TracksDigest | null;
}) {
  const d = digestStats(digest);
  const meta = report.meta ?? {};
  const surviving = d?.surviving ?? report.players?.length ?? 0;
  const dev = d ? surviving - EXPECTED_TRACKS : null;
  const dpf = d && d.frames ? d.detections / d.frames : null;

  return (
    <div className="grid gap-2">
      <h3 className="text-sm font-medium">Run health</h3>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
        <Tile
          n={d ? fmtInt(d.frames) : "–"}
          label="Frames with a detection"
          sub={d ? `${(d.frames / d.fps / 60).toFixed(1)} min at ${d.fps} fps` : "needs tracks"}
        />
        <Tile
          n={dpf != null ? dpf.toFixed(1) : "–"}
          label="Detections / frame"
          sub="~1 per visible person"
          grade={
            dpf == null
              ? "na"
              : dpf >= 12 && dpf <= 30
                ? "ok"
                : dpf >= 6 && dpf <= 40
                  ? "warn"
                  : "fail"
          }
        />
        <Tile
          n={d ? fmtInt(d.total) : "–"}
          label="Tracks"
          sub={d ? `${fmtInt(d.total - d.surviving)} under ${DEBRIS_S}s` : ""}
        />
        <Tile
          n={fmtInt(surviving)}
          label={`Tracks ≥ ${DEBRIS_S}s`}
          sub={
            dev == null
              ? ""
              : dev === 0
                ? "exactly 22"
                : `${dev > 0 ? "+" : ""}${dev} vs 22 expected`
          }
          grade={dev == null ? "na" : Math.abs(dev) <= 2 ? "ok" : Math.abs(dev) <= 6 ? "warn" : "fail"}
        />
        <Tile
          n={meta.officials_excluded != null ? fmtInt(meta.officials_excluded) : "–"}
          label="Officials excluded"
          sub={meta.officials_excluded === 0 ? "a referee folded into a squad reads as a real player" : ""}
          grade={meta.officials_excluded === 0 ? "warn" : undefined}
        />
        <Tile
          n={meta.calibration_rms_m != null ? meta.calibration_rms_m.toFixed(2) : "–"}
          unit=" m"
          label="Calibration RMS"
          sub={
            match.calibration_source === "auto"
              ? "the drawing's residual on the painted lines, not a measured accuracy at players"
              : "the fit's residual on its own clicks, not a measured accuracy"
          }
          // A residual on the points the fit was solved from can say the fit
          // FAILED; it cannot say the map is right. So a large one fails and a
          // small one earns no grade - it used to earn "ok", on synthetic
          // matches at 0.00 m too.
          grade={
            meta.calibration_rms_m == null
              ? "na"
              : meta.calibration_rms_m > 0.5
                ? "fail"
                : undefined
          }
        />
        <Tile
          n={match.frame_shape ? `${match.frame_shape[0]}×${match.frame_shape[1]}` : "–"}
          label="Frame size"
          sub="what makes a calibration reusable"
        />
        <Tile
          n={String(report.players?.length ?? 0)}
          label="Rows in the report"
          sub="after debris and officials are dropped"
        />
      </div>
      <p className="text-muted-foreground text-xs">
        Throughput, wall clock and scene-prior rejections are not on disk:{" "}
        <code className="bg-muted rounded px-1 py-0.5">track.py</code> builds a{" "}
        <code className="bg-muted rounded px-1 py-0.5">_perf</code> block but writes only{" "}
        <code className="bg-muted rounded px-1 py-0.5">{"{fps, detections}"}</code>. Those
        tiles would be invented, so they are absent rather than empty.
        {meta.competition && ` Competition in meta: ${meta.competition}.`}
      </p>
    </div>
  );
}

function Sanity({
  report,
  digest,
}: {
  report: MatchReport;
  digest: TracksDigest | null;
}) {
  const players = report.players ?? [];
  const L = report.meta?.pitch?.length ?? 105;
  const d = digestStats(digest);
  const matchMin =
    d && d.frames && d.fps
      ? d.frames / d.fps / 60
      : Math.max(...players.map((p) => p.minutes || 0)) || null;

  const order: Record<Grade, number> = { fail: 0, warn: 1, ok: 2, na: 3 };
  const rows = players
    .map((p) => ({ p, r: sanity(p, L, matchMin) }))
    .sort((a, b) => order[a.r.worst] - order[b.r.worst]);
  const nFail = rows.filter((r) => r.r.worst === "fail").length;
  const nWarn = rows.filter((r) => r.r.worst === "warn").length;

  return (
    <div className="grid min-w-0 gap-2">
      <h3 className="text-sm font-medium">Physiology sanity checks</h3>
      <Alert>
        <AlertTitle className={nFail ? "text-bad" : nWarn ? "text-warn" : "text-ok"}>
          {nFail
            ? `${nFail} player${nFail === 1 ? "" : "s"} fail${nFail === 1 ? "s" : ""} a physical plausibility check.`
            : nWarn
              ? `${nWarn} player${nWarn === 1 ? "" : "s"} sit outside the expected range.`
              : `All ${rows.length} players fall inside the expected physical ranges.`}
        </AlertTitle>
        <AlertDescription className="text-muted-foreground">
          {nFail
            ? "A failing row is a pipeline bug, not an athlete. Do not publish this run until it is explained."
            : nWarn
              ? "Worth a look before anything here is quoted."
              : "Passing these checks means the numbers are not obviously broken. It does not mean they are correct."}
        </AlertDescription>
      </Alert>
      <div className="w-full overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Track</TableHead>
              <TableHead>Role</TableHead>
              <TableHead className="text-right">Min</TableHead>
              <TableHead className="text-right">km</TableHead>
              <TableHead className="text-right">km/90</TableHead>
              <TableHead className="text-right">Top speed</TableHead>
              <TableHead className="text-right">Clamped</TableHead>
              <TableHead className="text-right">Verdict</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map(({ p, r }) => (
              <TableRow key={p.id}>
                <TableCell className="whitespace-nowrap">
                  {p.jersey != null ? (
                    <span className={p.team === "away" ? "text-away" : "text-home"}>
                      {p.team} #{p.jersey}
                    </span>
                  ) : (
                    <span className="text-muted-foreground">
                      {p.team} track {p.id}
                    </span>
                  )}
                </TableCell>
                <TableCell className="text-muted-foreground">
                  {r.keeper ? "keeper?" : "outfield"}
                </TableCell>
                <TableCell className="text-right tabular-nums">{p.minutes}</TableCell>
                <TableCell className="text-right tabular-nums">{p.distance_km ?? "refused"}</TableCell>
                <TableCell className="text-right tabular-nums">
                  {r.per90 == null ? "–" : r.per90.toFixed(2)}
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {/* null is a refusal (the clamp bound), and `{null}` renders
                      as an empty cell, which reads as a missing track. */}
                  {p.top_speed_kmh ?? "refused"}
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {p.clamped_frac == null ? "–" : `${(p.clamped_frac * 100).toFixed(1)}%`}
                </TableCell>
                <TableCell className="text-right whitespace-nowrap">
                  {r.checks.map((c) => (
                    <Badge
                      key={c.k}
                      variant="outline"
                      title={c.t}
                      className={`ml-1 ${GRADE_CLASS[c.s]}`}
                    >
                      {GRADE_WORD[c.s]}
                    </Badge>
                  ))}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
      <p className="text-muted-foreground text-xs">
        Bands: outfield <b>8–12 km per 90</b>, keeper <b>3–5 km per 90</b>, peak speed{" "}
        <b>25–32 km/h</b>. Role is guessed from average position inside a penalty area,
        which is crude — a sweeper-keeper is missed, and a track flagged “keeper?” in
        midfield means the average position itself is wrong. Peak speed is hard-clamped at{" "}
        {MAX_SPEED_KMH.toFixed(1)} km/h in <code className="bg-muted rounded px-1">stats.py</code>,
        so a value on that ceiling is an upper bound rather than a measurement and fails
        here on purpose.
        {matchMin ? ` Filmed length taken as ${matchMin.toFixed(1)} min.` : ""} Per-90
        extrapolated from a short track multiplies its error with it.
      </p>
    </div>
  );
}

const BINS: [number, number, string][] = [
  [0, 10, "< 10s"],
  [10, 30, "10–30s"],
  [30, 60, "30s–1m"],
  [60, 300, "1–5m"],
  [300, 900, "5–15m"],
  [900, 1800, "15–30m"],
  [1800, Infinity, "30m +"],
];

function TrackQuality({
  digest,
  report,
}: {
  digest: TracksDigest;
  report: MatchReport | null;
}) {
  const d = digestStats(digest);
  if (!d) return null;
  const counts = BINS.map(
    ([lo, hi]) =>
      digest.tracks.filter((t) => d.seconds(t) >= lo && d.seconds(t) < hi).length,
  );
  const max = Math.max(1, ...counts);

  // Tracks per player: only meaningful once identity has run. Two tracks under
  // one jersey means the player was lost and re-acquired, and their distance is
  // split across two rows that each look complete.
  const byPlayer = new Map<string, number>();
  for (const t of digest.tracks.filter((t) => t.jersey != null))
    byPlayer.set(
      `${t.team ?? "?"} #${t.jersey}`,
      (byPlayer.get(`${t.team ?? "?"} #${t.jersey}`) ?? 0) + 1,
    );
  const frag = [...byPlayer.entries()].filter(([, n]) => n > 1).sort((a, b) => b[1] - a[1]);

  const clamp = new Map((report?.players ?? []).map((p) => [p.id, p.clamped_frac]));
  const unquotable = digest.tracks.filter((t) => (clamp.get(t.id) ?? 0) > 0.05).length;

  return (
    <div className="grid min-w-0 gap-2">
      <h3 className="text-sm font-medium">Track quality</h3>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Tile
          n={fmtInt(counts[0])}
          label={`Tracks under ${DEBRIS_S}s (tracker debris, dropped by stats.py)`}
          grade={counts[0] > d.total * 0.5 ? "fail" : counts[0] ? "warn" : "ok"}
        />
        <Tile n={fmtInt(d.surviving)} label="Tracks that reach the stats layer" />
        <Tile
          n={byPlayer.size ? fmtInt(frag.length) : "–"}
          label={
            byPlayer.size
              ? "Jerseys owning more than one track"
              : "No jersey assigned — fragmentation per player cannot be measured"
          }
          grade={frag.length ? "warn" : undefined}
        />
        <Tile
          n={fmtInt(unquotable)}
          label="Tracks that must not be quoted (clamp hit on >5% of steps)"
          grade={unquotable ? "fail" : undefined}
        />
      </div>
      <div className="grid gap-1.5">
        {BINS.map(([, , label], i) => (
          <div key={label} className="grid grid-cols-[72px_minmax(0,1fr)_48px] items-center gap-3 text-xs">
            <span className="text-muted-foreground">{label}</span>
            <div className="bg-muted h-3 overflow-hidden rounded">
              <div
                className={`h-full rounded ${i === 0 ? "bg-bad" : "bg-primary"}`}
                style={{ width: `${(counts[i] / max) * 100}%` }}
              />
            </div>
            <span className="text-right font-medium tabular-nums">{fmtInt(counts[i])}</span>
          </div>
        ))}
      </div>
      <p className="text-muted-foreground text-xs">
        A healthy full-match run is a handful of long tracks and very little else. A tall
        left-hand bar means the tracker is losing and re-acquiring players; those fragments
        are silently dropped by <code className="bg-muted rounded px-1">build_match</code>,
        so the report looks clean while the distance they carried has vanished.
      </p>
      {frag.length > 0 && (
        <Accordion type="single" collapsible>
          <AccordionItem value="frag">
            <AccordionTrigger>
              Jerseys split across more than one track ({frag.length})
            </AccordionTrigger>
            <AccordionContent>
              <ul className="text-muted-foreground grid gap-1 text-xs">
                {frag.map(([k, n]) => (
                  <li key={k}>
                    <b className="text-bad">{k}</b> — {n} tracks
                  </li>
                ))}
              </ul>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      )}
    </div>
  );
}
