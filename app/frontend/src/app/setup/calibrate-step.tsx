"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { ChevronLeft, ChevronRight, RotateCcw, Undo2 } from "lucide-react";

import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useMatches } from "@/hooks/use-matches";
import { api } from "@/lib/api";
import {
  calibrationPayload,
  DEFAULT_PITCH,
  LANDMARK_COUNT,
  LANDMARK_LABELS,
  landmarks,
  outliersOf,
  pitchLines,
  solveClicks,
  YD,
  type Fit,
  type PitchDims,
  type Pt,
} from "@/lib/pitch";
import type { Match, OsmPitches } from "@/lib/types";

/** The four dimensions that are NOT fixed by the laws of the game live in the
 *  top form; the rest are behind the accordion because changing them is a
 *  non-standard-venue decision, not a per-match one. */
const FIXED_FIELDS = [
  ["penalty_depth", "Penalty area depth"],
  ["penalty_width", "Penalty area width"],
  ["goal_box_depth", "Goal area depth"],
  ["goal_box_width", "Goal area width"],
  ["centre_circle_r", "Centre circle radius"],
] as const;

type Projection = "perspective" | "panoramic";

/** The placing lens: a square of this many screen pixels, the frame's own
 *  pixels magnified this many times. */
const LENS_PX = 220;
const LENS_ZOOM = 4;

interface Props {
  match: Match;
  onSaved: () => void;
}

export function CalibrateStep({ match, onSaved }: Props) {
  // Venue. Held in metres; the unit picker only changes how length/width are
  // typed, because everything downstream is metres.
  const [units, setUnits] = useState<"yd" | "m">("m");
  const [dims, setDims] = useState<PitchDims>(DEFAULT_PITCH);
  const [projection, setProjection] = useState<Projection>("perspective");
  const [strips, setStrips] = useState(5);

  const [frame, setFrame] = useState(0);
  const [src, setSrc] = useState(() => api.frameUrl(match.id, 0));
  const [nat, setNat] = useState<[number, number] | null>(null);
  const [frameError, setFrameError] = useState<string | null>(null);

  const [clicks, setClicks] = useState<Record<string, Pt>>({});
  const [skipped, setSkipped] = useState<ReadonlySet<string>>(new Set());
  const [history, setHistory] = useState<string[]>([]);
  const [pick, setPick] = useState<string | null>(null);
  const [zoom, setZoom] = useState(100);

  const [saving, setSaving] = useState(false);
  const [refusal, setRefusal] = useState<string | null>(null);
  const [savedRms, setSavedRms] = useState<number | null | undefined>(undefined);

  const ov = useRef<SVGSVGElement>(null);
  const drag = useRef<string | null>(null);
  const still = useRef<HTMLImageElement>(null);
  const lens = useRef<HTMLCanvasElement>(null);
  /** Where the dragged dot is, in frame pixels - null when nothing is held. A
   *  4096-wide panorama shown 1400 wide turns 9 px into 3, and on the far side
   *  9 px is 4-5 m (docs/experiments/n31-line-registration.md §2): the far dots of a whole
   *  measured click set were that far off the paint. The lens shows the frame's
   *  own pixels, four to one, under the dot while it is being placed. */
  const [held, setHeld] = useState<Pt | null>(null);
  /** The pitch drawing as an AUTOMATIC calibration's camera sees it. When this is
   *  set nobody has clicked anything: the dots below are a way to overrule it,
   *  not a step that is owed. */
  const [autoLines, setAutoLines] = useState<[number, number][][] | null>(null);
  const [widthM, setWidthM] = useState<number>(match.calibration_auto?.width_m ?? DEFAULT_PITCH.width);
  const [sizing, setSizing] = useState(false);
  /** "lat, lon" as any map app copies it. The lookup is by coordinates because
   *  that is the one thing every ground has and no geocoder has to guess. */
  const [where, setWhere] = useState("");
  const [osm, setOsm] = useState<OsmPitches | null>(null);
  const [osmWay, setOsmWay] = useState<number | undefined>();
  const [looking, setLooking] = useState(false);
  const lookUp = async () => {
    const parts = where.trim().split(/[,\s]+/);
    const [lat, lon] = parts.map((p) => (/^-?\d+(\.\d+)?$/.test(p) ? Number(p) : NaN));
    if (parts.length !== 2 || !Number.isFinite(lat) || !Number.isFinite(lon)) {
      setRefusal("Paste the ground's coordinates as: latitude, longitude");
      return;
    }
    setLooking(true);
    try {
      setOsm(await api.osmPitches(match.id, lat, lon));
    } catch (e) {
      setRefusal(e instanceof Error ? e.message : String(e));
    } finally {
      setLooking(false);
    }
  };
  const useWidth = async () => {
    setSizing(true);
    try {
      await api.setPitch(match.id, widthM, undefined, osmWay);
      setAutoLines(null); // that calibration was just dropped; the job re-calibrates under the new width
      setOsm(null);
      onSaved();
    } catch (e) {
      setRefusal(e instanceof Error ? e.message : String(e));
    } finally {
      setSizing(false);
    }
  };

  // Re-opening a match that already has a calibration should not mean clicking
  // eight landmarks again: the camera has not moved, so the previous clicks and
  // the venue dimensions are still the right starting point.
  // Guarded on has_calibration: probing a match that has none answers 404, and
  // a handled 404 still prints in the console. A page whose console is noisy by
  // design is a page nobody reads the console of.
  const hasCalibration = match.has_calibration;
  useEffect(() => {
    if (!hasCalibration) return;
    let live = true;
    api
      .getCalibration(match.id)
      .then((d) => {
        if (!live || !d) return;
        if (d.projection_type === "spherical") {
          setAutoLines(Object.values(d.auto?.overlay_px ?? {}));
          return;
        }
        if (!d.points) return;
        setClicks(d.points);
        if (d.pitch) {
          setUnits("m");
          setDims((p) => ({ ...p, ...d.pitch }));
        }
        if (d.projection_type === "perspective" || d.projection_type === "panoramic")
          setProjection(d.projection_type);
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [match.id, hasCalibration]);

  const lm = useMemo(() => landmarks(dims), [dims]);
  const nStrips = projection === "panoramic" ? Math.max(2, strips) : 1;
  const fit: Fit | null = useMemo(
    () => solveClicks(dims, clicks, nStrips),
    [dims, clicks, nStrips],
  );
  const bad = useMemo(() => outliersOf(fit), [fit]);

  /** Where each UNPLACED landmark probably is, so the operator drags a pitch
   *  into place instead of hunting 21 spots. Once four points are down the
   *  guesses come from the fit itself, so correcting a corner moves every
   *  landmark that is still a guess. Before that they come from a generic
   *  side-on view: the far touchline short and high in frame, the near one
   *  wide and low. A guess is NEVER saved - dragging one is what turns it
   *  into a click (`place`), which is the operator's own evidence. */
  const guesses = useMemo(() => {
    if (!nat) return {};
    const [W, H] = nat;
    const toImage =
      fit?.toImage ??
      (([x, y]: Pt): Pt => {
        const u = x / dims.length; // 0 at the left goal line, 1 at the right
        const v = y / dims.width; // 0 on the FAR touchline (top of frame)
        const half = 0.18 + 0.28 * v; // half-width of the pitch at this depth
        return [W * (0.5 + (u - 0.5) * 2 * half), H * (0.25 + 0.6 * v)];
      });
    const out: Record<string, Pt> = {};
    for (const [n, w] of lm) {
      if (clicks[n] || skipped.has(n)) continue;
      const q = toImage(w);
      if (isFinite(q[0]) && isFinite(q[1])) out[n] = q;
    }
    return out;
  }, [nat, fit, lm, clicks, skipped, dims.length, dims.width]);

  // An explicit pick from the list wins even if that landmark is already
  // placed — that is how the operator re-does one they got wrong.
  const target =
    pick && !skipped.has(pick)
      ? pick
      : (lm.map(([n]) => n).find((n) => !clicks[n] && !skipped.has(n)) ?? null);

  const placed = Object.keys(clicks).length;

  const loadFrame = useCallback(
    (f: number) => {
      const n = Math.max(0, f);
      setFrame(n);
      setFrameError(null);
      // No `w=`: H must map ORIGINAL video pixels to metres, because that is
      // the coordinate space the worker's detections arrive in. A resized still
      // would silently scale every distance.
      setSrc(api.frameUrl(match.id, n));
    },
    [match.id],
  );

  // ---- clicking -----------------------------------------------------------
  const svgPoint = (e: React.PointerEvent | React.MouseEvent): Pt | null => {
    const svg = ov.current;
    if (!svg) return null;
    const m = svg.getScreenCTM();
    if (!m) return null;
    const p = svg.createSVGPoint();
    p.x = e.clientX;
    p.y = e.clientY;
    const q = p.matrixTransform(m.inverse());
    return [q.x, q.y];
  };

  const place = (name: string, at: Pt) => {
    setClicks((c) => ({ ...c, [name]: at }));
    setSkipped((s) => {
      if (!s.has(name)) return s;
      const n = new Set(s);
      n.delete(name);
      return n;
    });
  };

  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (e.button === 2) return;
    const q = svgPoint(e);
    if (!q) return;
    const g = (e.target as Element).closest("[data-drag]");
    if (g instanceof SVGElement && g.dataset.drag) {
      drag.current = g.dataset.drag;
      setHeld(clicks[g.dataset.drag] ?? guesses[g.dataset.drag] ?? q);
      // Grabbing a guess places it where it already sits, so a landmark the
      // operator agrees with costs one click and the fit gains a point.
      if (!clicks[drag.current] && guesses[drag.current]) {
        place(drag.current, guesses[drag.current]);
        setHistory((h) => [...h, drag.current as string]);
      }
      ov.current?.setPointerCapture(e.pointerId);
      return;
    }
    if (!target) return;
    place(target, q);
    setHistory((h) => [...h, target]);
    setPick(null);
  };

  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!drag.current) return;
    const q = svgPoint(e);
    if (q) {
      place(drag.current, q);
      setHeld(q);
    }
  };

  // Drawn from the <img> already on the page: the still is served no-store, and
  // a second request for the lens would be a second, possibly different, frame.
  useEffect(() => {
    const c = lens.current;
    const im = still.current;
    if (!c || !im || !held) return;
    const g = c.getContext("2d");
    if (!g) return;
    const half = LENS_PX / LENS_ZOOM / 2;
    g.imageSmoothingEnabled = false;
    g.fillStyle = "#000";
    g.fillRect(0, 0, LENS_PX, LENS_PX);
    g.drawImage(im, held[0] - half, held[1] - half, 2 * half, 2 * half, 0, 0, LENS_PX, LENS_PX);
    g.strokeStyle = "#fff";
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(LENS_PX / 2, 0);
    g.lineTo(LENS_PX / 2, LENS_PX / 2 - 6);
    g.moveTo(LENS_PX / 2, LENS_PX / 2 + 6);
    g.lineTo(LENS_PX / 2, LENS_PX);
    g.moveTo(0, LENS_PX / 2);
    g.lineTo(LENS_PX / 2 - 6, LENS_PX / 2);
    g.moveTo(LENS_PX / 2 + 6, LENS_PX / 2);
    g.lineTo(LENS_PX, LENS_PX / 2);
    g.stroke();
  }, [held]);

  const clear = (name: string) => {
    setClicks((c) => {
      const n = { ...c };
      delete n[name];
      return n;
    });
    setHistory((h) => h.filter((x) => x !== name));
  };

  const undo = () => {
    const last = history[history.length - 1];
    if (!last) return;
    setHistory((h) => h.slice(0, -1));
    setClicks((c) => {
      const n = { ...c };
      delete n[last];
      return n;
    });
    setPick(last);
  };

  const reset = () => {
    setClicks({});
    setSkipped(new Set());
    setHistory([]);
    setPick(null);
    setRefusal(null);
    setSavedRms(undefined);
  };

  // ---- save ---------------------------------------------------------------
  const save = async () => {
    if (!fit) return;
    setSaving(true);
    setRefusal(null);
    try {
      await api.saveCalibration(
        match.id,
        calibrationPayload(dims, clicks, fit, projection, nStrips),
      );
      // The RMS to show is the one the SERVER has, read back through the same
      // file the worker will load — not the number this page computed a moment
      // ago and might disagree with.
      const stored = await api.getCalibration(match.id).catch(() => null);
      setSavedRms(stored?.rms_error_m ?? null);
      onSaved();
    } catch (e) {
      setRefusal(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  // ---- render -------------------------------------------------------------
  const verdict = verdictOf(fit, placed, [...bad]);
  // Where "Next" goes: the first other match still waiting for its pitch.
  const { matches } = useMatches();
  const next = matches?.find((m) => m.id !== match.id && !m.has_calibration);
  const inherited = match.calibration_inherited_from;

  return (
    <div className="grid gap-4">
      {autoLines && savedRms === undefined && (
        <Alert>
          <AlertTitle className="text-ok">
            Calibrated automatically from the painted lines — nothing to do here.
          </AlertTitle>
          <AlertDescription className="text-muted-foreground">
            The camera was found from this match&apos;s own pitch markings
            {match.calibration_auto?.via === "install" && match.calibration_auto.install_from
              ? `, starting from “${match.calibration_auto.install_from}” at the same ground`
              : ""}
            {match.calibration_auto?.hits && match.calibration_auto?.samples
              ? ` (${match.calibration_auto.hits} of ${match.calibration_auto.samples} points of the pitch drawing sit on paint)`
              : ""}
            . The red lines are where it puts the pitch: they should lie on the paint. If they are
            visibly off, mark the pitch by hand below and that will replace it.
            <span className="mt-2 block">
              <b className="text-foreground">Length {match.calibration_auto?.length_m?.toFixed(1) ?? "105.0"} m</b>,
              measured from the painted boxes and centre circle.{" "}
              <b className="text-foreground">Width {(match.calibration_auto?.width_m ?? 68).toFixed(1)} m</b>
              {match.calibration_auto?.width_assumed
                ? " is ASSUMED — this camera looks across the pitch, so the paint cannot tell it. If you know this ground's width, enter it; every across-pitch distance scales with it."
                : match.calibration_auto?.width_source === "osm"
                  ? " from OpenStreetMap's outline of this pitch (© OpenStreetMap contributors, ODbL)."
                  : " as given for this ground."}
            </span>
            <span className="mt-2 flex flex-wrap items-center gap-2">
              <span className="w-24">
                <NumberInput
                  value={widthM}
                  onChange={(v) => {
                    setWidthM(v);
                    setOsmWay(undefined); // typed over: it is the user's number now, not the map's
                  }}
                />
              </span>
              <span className="text-xs">m wide</span>
              <Button size="sm" variant="outline" onClick={useWidth} disabled={sizing || !(widthM >= 45 && widthM <= 90)}>
                {sizing ? "Recalibrating…" : "Use this width"}
              </Button>
            </span>
            <span className="mt-2 flex flex-wrap items-center gap-2">
              <input
                className="h-8 w-56 rounded-md border bg-background px-2 text-xs"
                placeholder="or look it up: latitude, longitude"
                aria-label="The ground's coordinates: latitude, longitude"
                value={where}
                onChange={(e) => setWhere(e.target.value)}
              />
              <Button size="sm" variant="ghost" onClick={lookUp} disabled={looking || !where.trim()}>
                {looking ? "Asking OpenStreetMap…" : "Find this pitch on OpenStreetMap"}
              </Button>
            </span>
            {osm && (
              <span className="mt-2 block text-xs">
                {osm.pitches.length === 0 && "OpenStreetMap has no football pitch within 300 m of that point."}
                {osm.pitches.map((p) => (
                  <span key={p.way} className="mt-1 flex flex-wrap items-center gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={!p.usable}
                      onClick={() => {
                        setWidthM(p.width);
                        setOsmWay(p.way);
                      }}
                    >
                      {p.length.toFixed(1)} × {p.width.toFixed(1)} m
                    </Button>
                    <span>
                      {p.name ?? p.surface ?? "pitch"}, {p.distance_m} m away —{" "}
                      {p.usable ? `usable: ${p.why}` : `not used: ${p.why}`}
                    </span>
                  </span>
                ))}
                <span className="mt-1 block opacity-70">{osm.attribution}</span>
              </span>
            )}
          </AlertDescription>
        </Alert>
      )}
      {!autoLines && match.calibration_auto && !match.calibration_auto.accepted && savedRms === undefined && (
        <Alert>
          <AlertTitle className="text-warn">We could not find this pitch on our own.</AlertTitle>
          <AlertDescription className="text-muted-foreground">
            {match.calibration_auto.why ??
              "Too little of the painted pitch could be matched to be sure of the camera."}{" "}
            Mark it by hand below — once per camera position.
          </AlertDescription>
        </Alert>
      )}
      {inherited && savedRms === undefined && (
        <Alert>
          <AlertTitle className="text-warn">
            This pitch setup was reused from “{inherited}”, not solved for this footage.
          </AlertTitle>
          <AlertDescription className="text-muted-foreground">
            The video is the same size, so the system copied that match&apos;s homography
            rather than asking you. If the camera has been nudged since, every distance
            on the report is shifted and nothing here will say so. Redo it below if you
            are not certain the mount has not moved.
          </AlertDescription>
        </Alert>
      )}

      <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1.9fr)_minmax(300px,1fr)]">
        {/* ---- the still ---------------------------------------------- */}
        <Card className="min-w-0 gap-3 py-3">
          <CardHeader className="gap-2 px-3">
            <div className="flex flex-wrap items-center gap-1.5">
              <Button variant="outline" size="sm" onClick={() => loadFrame(frame - 250)}>
                <ChevronLeft /> 250
              </Button>
              <Button variant="outline" size="sm" onClick={() => loadFrame(frame - 25)}>
                <ChevronLeft /> 25
              </Button>
              <Input
                type="number"
                min={0}
                step={25}
                value={frame}
                onChange={(e) => loadFrame(Number(e.target.value) || 0)}
                className="h-8 w-24"
                aria-label="Frame number"
              />
              <Button variant="outline" size="sm" onClick={() => loadFrame(frame + 25)}>
                25 <ChevronRight />
              </Button>
              <Button variant="outline" size="sm" onClick={() => loadFrame(frame + 250)}>
                250 <ChevronRight />
              </Button>
              <label className="text-muted-foreground ml-auto flex items-center gap-2 text-xs">
                Zoom
                <input
                  type="range"
                  min={100}
                  max={400}
                  step={10}
                  value={zoom}
                  onChange={(e) => setZoom(Number(e.target.value))}
                  className="accent-primary w-24"
                />
              </label>
            </div>
          </CardHeader>
          <CardContent className="px-3">
            {held && (
              <canvas
                ref={lens}
                width={LENS_PX}
                height={LENS_PX}
                aria-hidden
                className="border-border pointer-events-none fixed right-6 bottom-6 z-50 rounded-md border shadow-lg"
              />
            )}
            <div className="bg-muted max-h-[70vh] overflow-auto rounded-md">
              <div className="relative leading-none" style={{ width: `${zoom}%` }}>
                {/* Plain <img>: a same-origin dynamic JPEG the server marks
                    no-store. next/image would cache a frame it must not. */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  ref={still}
                  src={src}
                  alt={`Frame ${frame}`}
                  className="block h-auto w-full"
                  onLoad={(e) =>
                    setNat([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])
                  }
                  onError={() =>
                    setFrameError(
                      `Could not read frame ${frame}. The upload may still be in flight, or that frame is past the end — try a lower number.`,
                    )
                  }
                />
                {nat && (
                  <svg
                    ref={ov}
                    viewBox={`0 0 ${nat[0]} ${nat[1]}`}
                    preserveAspectRatio="none"
                    className="absolute inset-0 h-full w-full cursor-crosshair touch-none"
                    onPointerDown={onPointerDown}
                    onPointerMove={onPointerMove}
                    onPointerUp={() => {
                      drag.current = null;
                      setHeld(null);
                    }}
                    onContextMenu={(e) => {
                      const g = (e.target as Element).closest("[data-drag]");
                      if (g instanceof SVGElement && g.dataset.drag) {
                        e.preventDefault();
                        clear(g.dataset.drag);
                      }
                    }}
                  >
                    {autoLines && placed === 0 && (
                      <g fill="none" stroke="var(--bad)" strokeWidth={Math.max(nat[0], nat[1]) / 1400} pointerEvents="none">
                        {autoLines.map((pl, i) => (
                          <polyline key={i} points={pl.map((q) => `${q[0]},${q[1]}`).join(" ")} />
                        ))}
                      </g>
                    )}
                    <Overlay
                      dims={dims}
                      fit={fit}
                      clicks={clicks}
                      guesses={guesses}
                      bad={bad}
                      target={target}
                      r={Math.max(nat[0], nat[1]) / 220}
                    />
                  </svg>
                )}
              </div>
            </div>
            {frameError && <p className="text-warn mt-2 text-xs">{frameError}</p>}
          </CardContent>
        </Card>

        {/* ---- the panel ------------------------------------------------ */}
        <div className="grid min-w-0 content-start gap-4">
          <Card className="gap-3 py-3">
            <CardContent className="grid gap-3 px-3">
              <div className="bg-muted rounded-md p-3">
                <div className="text-muted-foreground text-xs">Click this next</div>
                <div className="mt-0.5 text-base font-medium">
                  {target ? (LANDMARK_LABELS[target] ?? target) : "All landmarks placed"}
                </div>
                <div className="text-muted-foreground mt-1 text-xs">
                  {placed} of {LANDMARK_COUNT} placed · 4 is the minimum. Drag each
                  dot onto its real spot. Grey dots are the starting guess and turn
                  coloured once you move them; only coloured dots are saved. Do the four corners
                  first and the rest follow. Right-click clears.
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2 text-center">
                <Stat n={String(placed)} unit={`/${LANDMARK_COUNT}`} label="Placed" />
                <Stat
                  n={fit ? fit.rms.toFixed(2) : "–"}
                  unit=" m"
                  label="RMS (inliers)"
                />
                <Stat
                  n={(dims.length / (units === "yd" ? YD : 1)).toFixed(0)}
                  unit={`×${(dims.width / (units === "yd" ? YD : 1)).toFixed(0)} ${units}`}
                  label="Venue"
                />
              </div>

              <Alert>
                <AlertDescription className={verdict.tone}>{verdict.text}</AlertDescription>
              </Alert>

              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="flex-1"
                  onClick={undo}
                  disabled={!history.length}
                >
                  <Undo2 /> Undo
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="flex-1"
                  onClick={reset}
                  disabled={!placed && !skipped.size}
                >
                  <RotateCcw /> Clear all
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Shut by default: the rings on the frame are the list. It is here for
              skipping a landmark the camera cannot see, or re-doing one by name. */}
          <Card className="py-0">
            <Accordion type="single" collapsible>
              <AccordionItem value="landmarks" className="border-none">
                <AccordionTrigger className="px-3 text-sm">Landmarks</AccordionTrigger>
                <AccordionContent className="px-0">
              <ScrollArea className="h-[38vh] min-h-56">
                <ul className="px-1">
                  {lm.map(([n]) => {
                    const done = !!clicks[n];
                    const skip = skipped.has(n);
                    return (
                      <li
                        key={n}
                        className={`flex items-center gap-2 rounded-md px-2 py-1.5 text-xs ${
                          n === target ? "bg-muted" : ""
                        }`}
                      >
                        <button
                          type="button"
                          onClick={() => {
                            setPick(n);
                            setSkipped((s) => {
                              const x = new Set(s);
                              x.delete(n);
                              return x;
                            });
                          }}
                          className="flex min-w-0 flex-1 items-center gap-2 text-left"
                        >
                          <span
                            className={`size-2 shrink-0 rounded-full ${
                              bad.has(n)
                                ? "bg-bad"
                                : done
                                  ? "bg-ok"
                                  : skip
                                    ? "border-muted-foreground border"
                                    : "bg-muted-foreground/40"
                            }`}
                          />
                          <span
                            className={`truncate ${skip ? "text-muted-foreground line-through" : ""}`}
                          >
                            {LANDMARK_LABELS[n] ?? n}
                          </span>
                        </button>
                        {bad.has(n) && (
                          <Badge variant="destructive" className="shrink-0">
                            outlier
                          </Badge>
                        )}
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 shrink-0 px-2 text-xs"
                          onClick={() => {
                            if (done) return clear(n);
                            setSkipped((s) => {
                              const x = new Set(s);
                              if (skip) x.delete(n);
                              else x.add(n);
                              return x;
                            });
                          }}
                        >
                          {done ? "clear" : skip ? "unskip" : "skip"}
                        </Button>
                      </li>
                    );
                  })}
                </ul>
              </ScrollArea>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </Card>
        </div>
      </div>

      {/* ---- venue --------------------------------------------------------- */}
      <Card className="py-0">
        <Accordion type="single" collapsible>
          <AccordionItem value="venue" className="border-none">
            <AccordionTrigger className="px-4">
              <div className="text-left">
                <div className="text-sm font-medium">
                  Venue: {(dims.length / (units === "yd" ? YD : 1)).toFixed(1)} ×{" "}
                  {(dims.width / (units === "yd" ? YD : 1)).toFixed(1)} {units} ·{" "}
                  {projection}
                </div>
                <div className="text-muted-foreground text-xs">
                  Type this venue&apos;s real dimensions — assuming 105×68 m puts a 4–6%
                  error on every distance.
                </div>
              </div>
            </AccordionTrigger>
            <AccordionContent className="grid gap-4 px-4">
              <div className="grid gap-3 sm:grid-cols-4">
                <Field label="Units">
                  <Select
                    value={units}
                    onValueChange={(v) => setUnits(v === "m" ? "m" : "yd")}
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="yd">yards</SelectItem>
                      <SelectItem value="m">metres</SelectItem>
                    </SelectContent>
                  </Select>
                </Field>
                <Field label={`Length, touchline (${units})`}>
                  <NumberInput
                    value={dims.length / (units === "yd" ? YD : 1)}
                    onChange={(v) =>
                      setDims((d) => ({ ...d, length: v * (units === "yd" ? YD : 1) }))
                    }
                  />
                </Field>
                <Field label={`Width, goal line (${units})`}>
                  <NumberInput
                    value={dims.width / (units === "yd" ? YD : 1)}
                    onChange={(v) =>
                      setDims((d) => ({ ...d, width: v * (units === "yd" ? YD : 1) }))
                    }
                  />
                </Field>
                <Field label="Projection">
                  <Select
                    value={projection}
                    onValueChange={(v) =>
                      setProjection(v === "panoramic" ? "panoramic" : "perspective")
                    }
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="perspective">
                        Perspective — one un-stitched camera
                      </SelectItem>
                      <SelectItem value="panoramic">
                        Panoramic — stitched / multi-lens
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </Field>
              </div>
              {projection === "panoramic" && (
                <div className="grid gap-3 sm:grid-cols-4">
                  <Field label="Strips">
                    <NumberInput value={strips} onChange={(v) => setStrips(v)} />
                  </Field>
                </div>
              )}
              <div className="grid gap-3 sm:grid-cols-5">
                {FIXED_FIELDS.map(([k, label]) => (
                  <Field key={k} label={`${label} (m)`}>
                    <NumberInput
                      value={dims[k]}
                      onChange={(v) => setDims((d) => ({ ...d, [k]: v }))}
                    />
                  </Field>
                ))}
              </div>
              <p className="text-muted-foreground text-xs">
                Those five are fixed by the laws of the game at every level worldwide.
                Change them only for a non-standard venue.
              </p>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </Card>

      {/* ---- save ---------------------------------------------------------- */}
      <Card>
        <CardContent className="grid gap-3">
          {projection === "panoramic" ? (
            <Alert>
              <AlertTitle className="text-warn">
                This pipeline will not accept a panoramic calibration.
              </AlertTitle>
              <AlertDescription className="text-muted-foreground">
                A single 3×3 matrix over a stitched frame measured <b>6.17 m</b> median
                position error against 65 published pitch keypoints. It does not fail
                loudly — it produces plausible wrong distances. The accepted export for a
                stitched rig is <code>remap</code>: flatten the panorama with the rig&apos;s
                own stitch parameters, then solve the flattened frame here as perspective.
                That drops the same measurement to <b>2.36 m</b>.
              </AlertDescription>
            </Alert>
          ) : refusal ? (
            <Alert variant="destructive">
              <AlertTitle>Refused by the pipeline</AlertTitle>
              {/* Verbatim. This paragraph is written for the person who just
                  clicked the landmarks; paraphrasing it throws that away. */}
              <AlertDescription className="whitespace-pre-wrap">{refusal}</AlertDescription>
            </Alert>
          ) : savedRms !== undefined ? (
            <Alert>
              <AlertTitle className="text-ok">Saved. This camera is calibrated.</AlertTitle>
              <AlertDescription className="text-muted-foreground">
                The server loaded it with the same call the worker makes, so it is known
                to be usable
                {savedRms != null ? (
                  <>
                    , and the file it stored reports an RMS of{" "}
                    <b className="text-foreground">{savedRms.toFixed(2)} m</b>
                  </>
                ) : null}
                . Processing starts on its own — finding players in a full match takes
                hours, and nothing else is needed from you.
              </AlertDescription>
            </Alert>
          ) : null}

          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant={savedRms !== undefined && next ? "outline" : "default"}
              onClick={save}
              disabled={!fit || projection === "panoramic" || saving}
            >
              {saving ? "Saving…" : savedRms !== undefined ? "Save again" : "Save to this match"}
            </Button>
            {savedRms !== undefined && (
              <>
                {next && (
                  <Button asChild>
                    <Link href={`/setup?m=${encodeURIComponent(next.id)}`}>
                      Next match <ChevronRight />
                    </Link>
                  </Button>
                )}
                <Button variant="outline" asChild>
                  <Link href={`/checks?m=${encodeURIComponent(match.id)}`}>
                    Check the run
                  </Link>
                </Button>
                <Button variant="ghost" asChild>
                  <Link href="/">Back to matches</Link>
                </Button>
              </>
            )}
            <span className="text-muted-foreground text-xs">
              {fit
                ? "Saving re-queues this match from the last finished stage."
                : "Place at least four landmarks that are not collinear."}
            </span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------- bits

function Stat({ n, unit, label }: { n: string; unit: string; label: string }) {
  return (
    <div className="bg-muted rounded-md p-2">
      <div className="text-lg leading-6 font-medium">
        {n}
        <span className="text-muted-foreground text-xs font-normal">{unit}</span>
      </div>
      <div className="text-muted-foreground text-[11px]">{label}</div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-1.5">
      <Label className="text-muted-foreground text-xs">{label}</Label>
      {children}
    </div>
  );
}

function NumberInput({
  value,
  onChange,
}: {
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <Input
      type="number"
      step="0.01"
      value={Number.isFinite(value) ? Number(value.toFixed(2)) : ""}
      onChange={(e) => {
        const v = parseFloat(e.target.value);
        if (Number.isFinite(v) && v > 0) onChange(v);
      }}
      className="h-8"
    />
  );
}

/** The reprojected markings either sit on the paint or they don't. This is the
 *  check the RMS cannot fake. */
function Overlay({
  dims,
  fit,
  clicks,
  guesses,
  bad,
  target,
  r,
}: {
  dims: PitchDims;
  fit: Fit | null;
  clicks: Record<string, Pt>;
  guesses: Record<string, Pt>;
  bad: ReadonlySet<string>;
  target: string | null;
  r: number;
}) {
  return (
    <>
      {/* Guesses first, so a placed point always draws over one. The same solid
          dot as a placed one, in grey: colour alone is what tells the operator
          what the machine assumed from what they themselves moved. */}
      {Object.entries(guesses).map(([n, [x, y]]) => (
        <g key={`g-${n}`} data-drag={n} style={{ cursor: "grab" }}>
          <circle
            cx={x}
            cy={y}
            r={r}
            fill="var(--muted-foreground)"
            fillOpacity={0.22}
            stroke={n === target ? "#fff" : "var(--muted-foreground)"}
            strokeWidth={r * 0.18}
          />
          <circle cx={x} cy={y} r={r * 0.16} fill="var(--muted-foreground)" />
        </g>
      ))}
      {fit && (
        <g
          fill="none"
          stroke="var(--primary)"
          strokeWidth={r * 0.22}
          strokeOpacity={0.9}
          pointerEvents="none"
        >
          {pitchLines(dims).map((pl, i) => {
            const pts = pl
              .map((w) => fit.toImage(w))
              .filter((q) => isFinite(q[0]) && isFinite(q[1]));
            return pts.length > 1 ? (
              <polyline key={i} points={pts.map((q) => `${q[0]},${q[1]}`).join(" ")} />
            ) : null;
          })}
        </g>
      )}
      {Object.entries(clicks).map(([n, [x, y]]) => {
        const c = bad.has(n) ? "var(--bad)" : n === target ? "#fff" : "var(--primary)";
        return (
          <g key={n} data-drag={n} style={{ cursor: "grab" }}>
            <circle
              cx={x}
              cy={y}
              r={r}
              fill={c}
              fillOpacity={0.22}
              stroke={c}
              strokeWidth={r * 0.18}
            />
            <circle cx={x} cy={y} r={r * 0.16} fill={c} />
          </g>
        );
      })}
    </>
  );
}

function verdictOf(
  fit: Fit | null,
  placed: number,
  bad: string[],
): { tone: string; text: React.ReactNode } {
  if (!fit)
    return {
      tone: "text-muted-foreground",
      text: `Place at least 4 landmarks to solve.${placed < 4 ? ` ${4 - placed} to go.` : ""}`,
    };
  if (bad.length)
    return {
      tone: "text-bad",
      text: `${bad.length} outlier${bad.length > 1 ? "s" : ""}: ${bad
        .map((b) => LANDMARK_LABELS[b] ?? b)
        .join(", ")}. These disagree with the rest of the fit — either the click is off or that marking is not where the dimensions say it is. The RMS excludes them, so it looks better than the fit is.`,
    };
  if (fit.rms < 0.3)
    return {
      tone: "text-ok",
      text: `RMS ${fit.rms.toFixed(2)} m over ${placed} landmarks, no outliers. Check the green overlay sits on the painted lines between your clicks too — that is the part the number cannot tell you.`,
    };
  if (fit.rms < 1.0)
    return {
      tone: "text-warn",
      text: `RMS ${fit.rms.toFixed(2)} m. Usable but loose. Add more landmarks, especially the far corners, or check the dimensions match this venue.`,
    };
  return {
    tone: "text-bad",
    text: `RMS ${fit.rms.toFixed(2)} m — that is metres of error on every position. Likely causes: wrong pitch dimensions, a mislabelled click, or a stitched panorama being solved as perspective.`,
  };
}
