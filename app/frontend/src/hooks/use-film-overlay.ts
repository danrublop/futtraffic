"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { useSurfaceParams } from "@/components/surface-params";
import type { TracksIndex, TrackRow } from "@/hooks/use-tracks";

/** Everything drawn OVER the video, and the state that decides what.
 *
 *  WHY IT MOVED. video-panel.tsx was 1693 lines. This is the half of it that
 *  only ever touches the canvas: which switches are on, who is followed, where
 *  a click lands, and the per-frame draw itself. Lifting it out means the panel
 *  is left with the video element, the transport and the layout.
 *
 *  Deliberately NOT a context. Its three consumers — the canvas, the rail, and
 *  (next) the toggle cluster on the stage — all render inside the film panel's
 *  own tree, and the rail is its direct child. A context between a parent and
 *  its own child is indirection that buys nothing and costs traceability.
 *
 *  NOTHING HERE GOES THROUGH REACT PER FRAME. 16k detections at 30fps must not
 *  re-render a tree, so the fit box, the hit list and the HUD text are refs and
 *  direct DOM writes. The only state is what changes seconds apart.
 */

const TRAIL_S = 3;
/** How near a click has to land, in CSS px, to grab a player. */
const PICK_PX = 44;
/** Detections are often sparse, so the last known position is reused — but past
 *  half a second that is not where the player is, and a stale ring is a lie. */
const STALE_S = 0.5;
const TAU = Math.PI * 2;

export const teamName = (t: string | null) =>
  t === "home" ? "Home" : t === "away" ? "Away" : t === "official" ? "Official" : "Unknown";
const nameOf = (tr: TrackRow) =>
  `${teamName(tr.team)} ${tr.jersey != null ? `#${tr.jersey}` : "?"} · track ${tr.id}`;
const label = (tr: TrackRow) => (tr.jersey != null ? `#${tr.jersey}` : "?");

interface Hit {
  id: number;
  x: number;
  y: number;
}
/** Maps a source-video pixel to a canvas pixel under object-fit: contain. */
interface Fit {
  s: number;
  ox: number;
  oy: number;
  dpr: number;
}

/** What the rail and the stage cluster both read and write. */
export interface FilmOverlay {
  overlay: boolean;
  trail: boolean;
  spotlight: boolean;
  hasTracks: boolean;
  /** Why a switch is dead. A dead switch with no reason on it reads as a bug. */
  tracksError: string | null;
  tracks: TrackRow[];
  sel: number | null;
  offset: number;
  toggle: (k: "overlay" | "trail" | "spotlight") => void;
  select: (id: number | null) => void;
  setOffset: (n: number) => void;

  // ---- the parts the film panel wires to the DOM
  /** Draw the canvas for one time. Called from the panel's per-frame loop. */
  paint: (t: number) => void;
  /** Recompute the letterbox fit. The panel calls this on resize and on a
   *  fullscreen change, both of which move the canvas without a render. */
  fit: () => void;
  /** The track under a click, or null. The panel decides what to do with that:
   *  a hit follows the player, a miss toggles playback. */
  hitTest: (clientX: number, clientY: number) => number | null;
  /** Set when the drawn positions cannot be pixels of this video. */
  coordWarn: string | null;
  /** Set when `?track=` names a track this run does not have. */
  trackWarn: string | null;
}

export function useFilmOverlay(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  stageRef: React.RefObject<HTMLDivElement | null>,
  data: TracksIndex | null,
  tracksError: string | null,
  dur: number,
  /** The three HUD chips over the top-left of the picture. Passed IN rather
   *  than created here and handed back: a ref read off a returned object during
   *  render is what react-hooks/refs exists to stop, and the panel is the thing
   *  that renders the spans anyway. */
  hud: {
    frameRef: React.RefObject<HTMLSpanElement | null>;
    followRef: React.RefObject<HTMLSpanElement | null>;
    noteRef: React.RefObject<HTMLSpanElement | null>;
  },
): FilmOverlay {
  // The SURFACE's params, not the page's. This logic moved here out of
  // video-panel.tsx, where it had already been switched to useSurfaceParams:
  // inside the dock the film is a tab with params of its own, so reading the
  // page URL's ?track= selects whatever the page happens to carry instead of
  // the player this film was opened on. It arrived with the old call because
  // the move and the switch happened on different branches.
  const search = useSurfaceParams();
  const { frameRef, followRef, noteRef } = hud;

  const fitRef = useRef<Fit>({ s: 1, ox: 0, oy: 0, dpr: 1 });
  const hitsRef = useRef<Hit[]>([]);
  const colorRef = useRef({
    home: "#C70101",
    away: "#4A7FE0",
    official: "#FF963F",
    unknown: "#9F9F9F",
    pick: "#61DF6E",
  });
  const [coordWarn, setCoordWarn] = useState<string | null>(null);

  const [overlayOn, setOverlayOn] = useState(true);
  const [trailOn, setTrailOn] = useState(true);
  const [spotOn, setSpotOn] = useState(false);
  const [offset, setOffset] = useState(0);

  const byId = useMemo(
    () => new Map((data?.tracks ?? []).map((t) => [t.id, t])),
    [data],
  );

  const trackQ = search.get("track");
  const wanted = trackQ ? Number(trackQ) : NaN;
  const deepTrack = data && byId.has(wanted) ? wanted : null;
  const trackWarn = trackQ && data && !byId.has(wanted) ? trackQ : null;
  const [selState, setSelState] = useState<number | null | undefined>(undefined);
  const sel = selState === undefined ? deepTrack : selState;
  const pick = (tid: number | null) => setSelState(tid != null && tid === sel ? null : tid);

  const fitBox = () => {
    const cv = canvasRef.current;
    const vid = videoRef.current;
    const stage = stageRef.current;
    if (!cv || !vid) return;
    // The stage takes the video's own shape rather than a guessed 16:9 — this
    // footage is a 4096×1080 panorama. Written to the element, not held in
    // state: it is a measurement of an external thing, and a render round trip
    // would lose the race with `loadedmetadata`, which on a local file fires
    // before React has attached a handler to the server-rendered <video>.
    if (stage && vid.videoWidth) {
      stage.style.aspectRatio = document.fullscreenElement
        ? ""
        : `${vid.videoWidth} / ${vid.videoHeight}`;
    }
    const r = cv.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = Math.max(1, Math.round(r.width * dpr));
    cv.height = Math.max(1, Math.round(r.height * dpr));
    const vw = vid.videoWidth || 1;
    const vh = vid.videoHeight || 1;
    // object-fit: contain. Once the stage hits its max height the letterbox
    // offsets stop being zero, which is when a naive overlay drifts.
    const s = Math.min(r.width / vw, r.height / vh);
    fitRef.current = {
      s: s * dpr,
      ox: ((r.width - vw * s) / 2) * dpr,
      oy: ((r.height - vh * s) / 2) * dpr,
      dpr,
    };
  };


  /** The strip over the picture. What is switched on and — the part that
   *  matters — when the followed player is NOT on screen, because a spotlight
   *  over empty grass otherwise reads as "he is standing there". */
  const paintHud = (t: number, missing: boolean) => {
    const tr = sel != null ? byId.get(sel) : undefined;
    setText(frameRef.current, data ? `frame ${data.fmin + Math.round(t * data.fps) + offset}` : "");
    setText(followRef.current, tr ? `following ${nameOf(tr)}` : "");
    setText(
      noteRef.current,
      !overlayOn
        ? "overlay off — B brings it back"
        : tr && missing
          ? "no detection at this frame — lost, occluded, or off pitch"
          : spotOn && sel == null
            ? "spotlight needs a player — click one"
            : tr && tr.jersey == null
              ? "no jersey number — identity unknown"
              : "",
    );
  };


  /** The canvas, for one time. The scrub bar and the reel are the panel's
   *  business; this only draws over the picture. */
  const paint = (t: number) => {
    const cv = canvasRef.current;
    const ctx = cv?.getContext("2d");
    if (!cv || !ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    hitsRef.current = [];

    // f = fmin + round(t x fps) + offset. fmin comes from the data, never
    // assumed: real ground-truth files start at 3000001, and assuming 0 puts
    // every ring off the end of the index and draws a convincing blank.
    if (!data || !data.n || !overlayOn) {
      paintHud(t, sel != null);
      return;
    }
    const k = data.fmin + Math.round(t * data.fps) + offset - data.fmin;
    if (k < 0 || k >= data.span) {
      paintHud(t, sel != null);
      return;
    }
    const at = data.prev[k];
    if (at < 0 || (k - at) / data.fps > STALE_S) {
      paintHud(t, sel != null);
      return;
    }

    const f = data.fmin + k;
    const s0 = data.start[at];
    const s1 = data.start[at + 1];
    const fit = fitRef.current;
    const C = colorRef.current;
    const r = Math.max(6, Math.min(18, 26 * fit.s));

    // Spotlight first, so it dims the VIDEO and the rings drawn after it stay
    // legible. Dimming afterwards greys out the thing being spotlighted.
    let sx = 0;
    let sy = 0;
    let missing = sel != null;
    if (sel != null) {
      for (let i = s0; i < s1; i++) {
        if (data.ID[i] === sel) {
          sx = fit.ox + data.X[i] * fit.s;
          sy = fit.oy + data.Y[i] * fit.s;
          missing = false;
        }
      }
    }
    if (spotOn && sel != null) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, cv.width, cv.height);
      if (!missing) ctx.ellipse(sx, sy - r * 1.1, r * 3.6, r * 3.2, 0, 0, TAU);
      ctx.fillStyle = "rgba(0,0,0,.74)";
      ctx.fill("evenodd");
      ctx.restore();
    }

    if (trailOn && sel != null) {
      const tr = byId.get(sel);
      if (tr) {
        const lo = f - TRAIL_S * data.fps;
        ctx.beginPath();
        ctx.lineWidth = Math.max(1.5, r * 0.22);
        ctx.strokeStyle = C.pick;
        ctx.globalAlpha = 0.75;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        let started = false;
        for (let i = 0; i < tr.idx.length; i++) {
          const j = tr.idx[i];
          const ff = data.F[j];
          if (ff < lo || ff > f) {
            started = false;
            continue;
          }
          const x = fit.ox + data.X[j] * fit.s;
          const y = fit.oy + data.Y[j] * fit.s;
          if (started) ctx.lineTo(x, y);
          else {
            ctx.moveTo(x, y);
            started = true;
          }
        }
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    for (let i = s0; i < s1; i++) {
      const tr = byId.get(data.ID[i]);
      if (!tr) continue;
      const x = fit.ox + data.X[i] * fit.s;
      const y = fit.oy + data.Y[i] * fit.s;
      hitsRef.current.push({ id: tr.id, x, y });
      const on = sel === tr.id;
      const base = sel != null && !on ? 0.32 : 1;
      const col =
        tr.team === "home"
          ? C.home
          : tr.team === "away"
            ? C.away
            : tr.team === "official"
              ? C.official
              : C.unknown;

      // A flattened ellipse at the feet point reads as contact with the pitch.
      // A circle centred there reads as a ball.
      ctx.globalAlpha = base * 0.27;
      ctx.beginPath();
      ctx.ellipse(x, y, r, r * 0.42, 0, 0, TAU);
      ctx.fillStyle = col;
      ctx.fill();
      ctx.globalAlpha = base;
      ctx.lineWidth = on ? 3 : 2;
      ctx.strokeStyle = col;
      ctx.stroke();

      if (on) {
        ctx.lineWidth = 2;
        ctx.strokeStyle = C.pick;
        ctx.beginPath();
        ctx.ellipse(x, y, r * 1.9, r * 0.92, 0, 0, TAU);
        ctx.stroke();
        chip(ctx, fit.s, x, y - r * 1.5, `${teamName(tr.team)} ${label(tr)}`, C.pick, "#000", null);
      }
      // Everyone else keeps the compact chip, or 22 name plates cover the pitch.
      chip(
        ctx,
        fit.s,
        x,
        y - r * 0.6,
        label(tr),
        tr.jersey != null ? col : "#3A3A3A",
        tr.jersey != null ? "#fff" : C.unknown,
        on ? C.pick : null,
      );
    }
    ctx.globalAlpha = 1;
    paintHud(t, missing);
  };

  // The palette lives in globals.css. Read it rather than repeating the hex, or
  // a ring and a table end up disagreeing about which team is which.
  useEffect(() => {
    const s = getComputedStyle(document.documentElement);
    const v = (n: string, fallback: string) => s.getPropertyValue(n).trim() || fallback;
    colorRef.current = {
      home: v("--home", "#C70101"),
      away: v("--away", "#4A7FE0"),
      official: v("--warn", "#FF963F"),
      unknown: v("--muted-foreground", "#9F9F9F"),
      pick: v("--primary", "#61DF6E"),
    };
  }, []);

  /** The single most likely way this page lies: positions in pitch metres, or
   *  from another resolution, drawn over this frame. Range alone does not catch
   *  it — 0–105 by 0–68 sits comfortably inside a 4096×1080 frame while every
   *  marker piles into the corner. Spread is the tell. */
  useEffect(() => {
    const vid = videoRef.current;
    if (!data || !data.n || !vid?.videoWidth) {
      setCoordWarn(null);
      return;
    }
    const vw = vid.videoWidth;
    const vh = vid.videoHeight;
    const stride = Math.max(1, Math.floor(data.n / 20000));
    let lox = Infinity;
    let hix = -Infinity;
    let loy = Infinity;
    let hiy = -Infinity;
    let seen = 0;
    let inside = 0;
    for (let i = 0; i < data.n; i += stride) {
      seen += 1;
      const x = data.X[i];
      const y = data.Y[i];
      if (x < lox) lox = x;
      if (x > hix) hix = x;
      if (y < loy) loy = y;
      if (y > hiy) hiy = y;
      if (x >= 0 && x <= vw && y >= 0 && y <= vh) inside += 1;
    }
    const inFrac = seen ? inside / seen : 0;
    const sx = (hix - lox) / vw;
    const sy = (hiy - loy) / vh;
    setCoordWarn(
      inFrac < 0.5
        ? `Only ${(inFrac * 100).toFixed(0)}% of tracked positions fall inside this video's ${vw}×${vh} frame.`
        : sx < 0.25 || sy < 0.15
          ? `Every tracked position sits in a ${(sx * 100).toFixed(1)}% × ${(sy * 100).toFixed(
              1,
            )}% patch of this video's ${vw}×${vh} frame — real tracks cover most of it.`
          : null,
    );
  }, [data, dur, videoRef]);

  /** The track under a click. Squared distances, because a square root per
   *  detection per click buys nothing when only the ordering matters. */
  const hitTest = (clientX: number, clientY: number) => {
    const cv = canvasRef.current;
    if (!cv) return null;
    const r = cv.getBoundingClientRect();
    const dpr = fitRef.current.dpr || 1;
    const x = (clientX - r.left) * dpr;
    const y = (clientY - r.top) * dpr;
    let best: number | null = null;
    let bd = (PICK_PX * dpr) ** 2;
    for (const h of hitsRef.current) {
      const d = (h.x - x) ** 2 + (h.y - y) ** 2;
      if (d < bd) {
        bd = d;
        best = h.id;
      }
    }
    return best;
  };

  return {
    overlay: overlayOn,
    trail: trailOn,
    spotlight: spotOn,
    hasTracks: !!data,
    tracksError: data ? null : `no tracking data: ${tracksError ?? "still loading"}`,
    tracks: data?.tracks ?? [],
    sel,
    offset,
    toggle: (k) =>
      k === "overlay"
        ? setOverlayOn((v) => !v)
        : k === "trail"
          ? setTrailOn((v) => !v)
          : setSpotOn((v) => !v),
    select: pick,
    setOffset,
    paint,
    fit: fitBox,
    hitTest,
    coordWarn,
    trackWarn,
  };
}

/** Write text to a chip and hide it when there is none. Exported because the
 *  chrome's hover bubble does the same thing for the same reason: these are
 *  written on pointermove and per frame, and neither is worth a render. */
export function setText(el: HTMLElement | null, s: string) {
  if (!el) return;
  if (el.textContent !== s) el.textContent = s;
  el.style.display = s ? "" : "none";
}

function chip(
  ctx: CanvasRenderingContext2D,
  s: number,
  x: number,
  y: number,
  text: string,
  bg: string,
  fg: string,
  ring: string | null,
) {
  const h = Math.max(13, 15 * s * 1.2);
  const pad = h * 0.34;
  ctx.font = `500 ${Math.round(h * 0.72)}px system-ui, sans-serif`;
  const w = ctx.measureText(text).width + pad * 2;
  const bx = x - w / 2;
  const by = y - h - h * 0.35;
  ctx.beginPath();
  ctx.roundRect(bx, by, w, h, h / 2);
  ctx.fillStyle = bg;
  ctx.fill();
  if (ring) {
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = ring;
    ctx.stroke();
  }
  ctx.fillStyle = fg;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x, by + h / 2 + 0.5);
}
