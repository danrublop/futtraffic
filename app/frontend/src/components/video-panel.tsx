"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle } from "lucide-react";

import { useSurfaceParams } from "@/components/surface-params";
import { ClipsPanel } from "@/app/review/clips-panel";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { FilmChrome } from "@/components/film-chrome";
import { FilmGallery } from "@/components/film-gallery";
import { TrackTimeline } from "@/components/track-timeline";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { useReport } from "@/hooks/use-matches";
import { useTracks } from "@/hooks/use-tracks";
import { api } from "@/lib/api";
import { demoMoments, moments } from "@/lib/moments";
import { useClipEditor } from "@/hooks/use-clip-editor";
import { useReel } from "@/hooks/use-reel";
import { fullscreenElement, useTransport } from "@/hooks/use-transport";
import { useFilmOverlay } from "@/hooks/use-film-overlay";


/* ---------------------------------------------------------------- constants */

/** How far the playhead may jump to land on an event. Measured event timing
 *  error is 0.40s median, so this is the widest snap that is still smaller than
 *  the uncertainty of the thing being snapped to. Anything larger would move
 *  the playhead further than we can say the event moved. */
const SNAP_S = 0.4;
/** How long the chrome stays after the pointer stops, while tp.playing. */
const IDLE_MS = 2500;

/** One possession change on the event rail — the fields a tick needs, named as
 *  `model/engine/events.py :: Event` names them.
 *
 *  NOTHING FILLS THIS YET. There is no events route on the API — app.py stops
 *  at clips and roster — so `EVENTS` is empty and the rail says so in words
 *  rather than drawing anything. Do not synthesise ticks from tracks.json: a
 *  tick asserts that the ball changed hands at a timecode, and possession is
 *  not a thing this page can derive from feet positions.
 */
interface PossessionEvent {
  /** Seconds from kickoff, with events.py's +0.20s tag offset already applied. */
  t: number;
  /** The side in possession AFTER the change. Team level only — per-player
   *  attribution is 9.1% accurate against 4.5% chance and is never shown. */
  team: "home" | "away";
  /** events.py `primary`: the feed's own coarsest label for the event. */
  label: string;
}

/** Wire this to the API when a route serves it. Empty is the honest state. */
const EVENTS: PossessionEvent[] = [];



const SHORTCUTS: [string, string][] = [
  ["Space", "play / pause"],
  ["← →", "back / forward 5s (hold Shift for 10s)"],
  ["J L", "back / forward 10s"],
  [", .", "one frame back / forward"],
  ["< >", "slower / faster"],
  ["I O", "mark clip in / out"],
  ["S", "save the clip"],
  ["R", "play the ticked clips as a reel"],
  ["B", "tracking overlay on / off"],
  ["T", "trail on the followed player"],
  ["D", "spotlight (dim everyone else)"],
  ["X", "stop following"],
  ["F", "fullscreen"],
  ["M", "mute"],
  ["Esc", "clear the range / stop the reel"],
  ["click", "a player to follow, elsewhere to play / pause"],
];

/* ------------------------------------------------------------------- page */

/** A tp.seek the workspace asked for, stamped so a repeat of the same timecode
 *  still lands. `n` increments per request; the panel seeks whenever a new
 *  stamp arrives, including the one waiting for it on mount. */
export interface FilmSeek {
  t: number;
  n: number;
}

/** The whole film viewer: player, tracking overlay, scrub bar, event rail,
 *  transport, clip marking and the clips panel. Extracted from the /review
 *  route untouched, so the route and the workspace's film tab render the same
 *  component and can never drift. */
export function VideoPanel({ seekRequest }: { seekRequest?: FilmSeek | null }) {
  const { id, report } = useReport();
  /** Dragged height of the video stage, px. null = the 16/9 default. */
  const [stageH, setStageH] = useState<number | null>(null);
  /** Dragged width of the events rail, px. Only read where the pane is wide
   *  enough to have two columns — below xl the rail is a row and this is
   *  inert. */
  const [railW, setRailW] = useState(360);
  const search = useSurfaceParams();
  const { data, progress, error: tracksError } = useTracks(id);

  // Written by the render loop. None of these go through React: 16k detections
  // at 30fps must not re-render a tree.
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const playerRef = useRef<HTMLDivElement>(null);
  /** The timeline's own painter, registered by TrackTimeline on mount. It
   *  moves its playhead and page-scrolls itself; this file only supplies t. */
  const tlPaint = useRef<{ paint: (t: number) => void; resumeFollow: () => void } | null>(null);
  const idleRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** Whether the pointer has asked for the playback chrome recently. Paused,
   *  the controls are the thing you came for and this does not govern — see the
   *  overlay's className. The only piece of player state allowed to be React
   *  state: it changes seconds apart, not frames. */
  const [chrome, setChrome] = useState(true);

  /** YouTube's rule: the chrome follows the pointer and gets out of the way
   *  IDLE_MS after it stops — but only while the video is running. Paused, the
   *  controls are the thing you came for, so they stay. */
  const bump = useCallback(() => {
    setChrome(true);
    if (idleRef.current) clearTimeout(idleRef.current);
    idleRef.current = setTimeout(() => {
      if (!videoRef.current?.paused) setChrome(false);
    }, IDLE_MS);
  }, []);

  useEffect(() => () => void (idleRef.current && clearTimeout(idleRef.current)), []);
  const drawRef = useRef<(t: number) => void>(() => {});
  /** The chrome's own painter, registered by FilmChrome on mount. The played
   *  bar, the buffered bar, the playhead and the clock all move every frame,
   *  so they are driven from here rather than through React. Same contract
   *  TrackTimeline uses below, for the same reason. */
  const barPaint = useRef<{ paint: (t: number) => void; paintBuffered: () => void } | null>(null);
  const refitRef = useRef<() => void>(() => {});
  const keyRef = useRef<(e: KeyboardEvent) => void>(() => {});
  const deepLinkRef = useRef(false);

  const [dur, setDur] = useState(0);

  /* Destructured for the redraw effect below: depending on `ov` itself would
     re-run it on every render, and these are the values that actually matter. */
  /** Driving the video element: play, tp.seek, speed, sound, fullscreen. `draw`
   *  goes in through the ref the render loop already uses, so a tp.seek repaints
   *  with the LATEST closure rather than the one this render captured. */
  const tp = useTransport(videoRef, playerRef, {
    dur,
    fps: data?.fps ?? null,
    draw: useCallback((t: number) => drawRef.current(t), []),
  });

  /** Everything drawn over the picture, and the state that decides what. One
   *  object, so the rail takes one prop instead of the eleven this used to be
   *  and the toggle cluster on the stage can read the same switches. */
  const frameRef = useRef<HTMLSpanElement>(null);
  const followRef = useRef<HTMLSpanElement>(null);
  const noteRef = useRef<HTMLSpanElement>(null);
  const ov = useFilmOverlay(videoRef, canvasRef, stageRef, data, tracksError, dur, {
    frameRef,
    followRef,
    noteRef,
  });
  // Stable identities pulled out so the effects below can name what they use
  // without depending on the whole transport object, which is new every render.
  const { setFsEl, setPlaying, seek } = tp;

  // Primitives for the redraw effect's dep array: depending on `ov` itself
  // would re-run it on every render, and these are the values that matter.
  const { overlay, trail, spotlight, sel: ovSel, offset: ovOffset } = ov;
  const [helpOpen, setHelpOpen] = useState(false);
  const [videoErr, setVideoErr] = useState(false);
  const [seekWarn, setSeekWarn] = useState(false);



  /** `?track=` follows one player from the moment the page opens, so a row in
   *  the report can say "watch this player" and land here with them selected.
   *
   *  Derived, not an effect: the id only becomes resolvable once the worker has
   *  finished, and an id that is NOT here means the link was built against an
   *  earlier run — re-tracking renumbers tracks — which has to be said rather
   *  than silently showing the whole team as if the link worked.
   */

  /* ------------------------------------------------------------ transport */


  /* ------------------------------------------------------------------ clips */

  /** Marking, naming, saving, deleting and picking clips. One object, so the
   *  rail takes one prop instead of the eleven this used to be. */
  /** Clips played back to back, client-side. Nothing here needs the server. */
  const reel = useReel(videoRef, tp.seek);

  const clip = useClipEditor(
    id,
    useCallback(() => videoRef.current?.currentTime ?? 0, []),
    tp.seek,
  );

  /** Everything with a time on it, in one list: the panel filters it and the
   *  timeline draws it. */
  const mo = useMemo(() => {
    const real = moments(report, EVENTS, clip.clips);
    // A match with plays or a feed shows those and nothing else. Only a match
    // that names NOTHING gets the sample, so a fabricated pass can never sit in
    // a list beside a measured one. 90 minutes when the video has not loaded a
    // duration yet — a demo does not need the real length.
    return real.some((m) => m.kind !== "clip")
      ? real
      : [...real, ...demoMoments(dur || 5400)].sort((a, b) => a.t - b.t);
  }, [report, clip.clips, dur]);

  /** Pulls a scrubbed time onto the nearest event inside SNAP_S. A no-op while
   *  no route serves events, which is the point: it costs nothing until then. */
  const snap = (t: number) => {
    let best = t;
    let bd = SNAP_S;
    for (const e of EVENTS) {
      const d = Math.abs(e.t - t);
      if (d < bd) {
        bd = d;
        best = e.t;
      }
    }
    return best;
  };

  /* -------------------------------------------------------------- overlay */

  const draw = (t: number) => {
    barPaint.current?.paint(t);
    tlPaint.current?.paint(t);
    reel.check(t);
    ov.paint(t);
  };


  /* --------------------------------------------------------- render loop */

  useEffect(() => {
    const vid = videoRef.current;
    if (!vid) return;
    let raf = 0;
    const step = (_now?: number, meta?: VideoFrameCallbackMetadata) => {
      drawRef.current(meta ? meta.mediaTime : vid.currentTime);
      pump();
    };
    const pump = () => {
      if (vid.paused || vid.ended) return;
      if (vid.requestVideoFrameCallback) vid.requestVideoFrameCallback(step);
      else raf = requestAnimationFrame(() => step());
    };
    const onPlay = () => {
      setPlaying(true);
      // Starts the idle countdown too: playback can begin from a key or from a
      // reel, with the pointer never moving, and the chrome has to leave anyway.
      bump();
      // A hand scrub stopped the timeline chasing the playhead; pressing play
      // is what gives the chase back.
      tlPaint.current?.resumeFollow();
      pump();
    };
    const onPause = () => {
      setPlaying(false);
      cancelAnimationFrame(raf);
      drawRef.current(vid.currentTime);
    };
    // Keeps the clock honest while paused-and-seeking, which neither rVFC nor a
    // stopped rAF loop would.
    const onTime = () => {
      if (vid.paused) drawRef.current(vid.currentTime);
    };
    vid.addEventListener("play", onPlay);
    vid.addEventListener("pause", onPause);
    vid.addEventListener("timeupdate", onTime);
    return () => {
      cancelAnimationFrame(raf);
      vid.removeEventListener("play", onPlay);
      vid.removeEventListener("pause", onPause);
      vid.removeEventListener("timeupdate", onTime);
    };
  }, [id, bump, setPlaying]);

  // Registration. A canvas that keeps the windowed scale after a fullscreen is
  // worse than no overlay: it puts every ring in the wrong place, confidently —
  // and Firefox and Safari do not always fire a resize when it happens.
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const refit = () => {
      refitRef.current();
      drawRef.current(videoRef.current?.currentTime ?? 0);
    };
    const ro = new ResizeObserver(refit);
    ro.observe(stage);
    const onFs = () => {
      setFsEl(fullscreenElement());
      window.setTimeout(refit, 60);
    };
    document.addEventListener("fullscreenchange", onFs);
    document.addEventListener("webkitfullscreenchange", onFs);
    return () => {
      ro.disconnect();
      document.removeEventListener("fullscreenchange", onFs);
      document.removeEventListener("webkitfullscreenchange", onFs);
    };
  }, [id, setFsEl]);

  /* ------------------------------------------------------------- keyboard */

  const onKey = (e: KeyboardEvent) => {
    const t = e.target as HTMLElement | null;
    if (
      t &&
      (t.tagName === "INPUT" ||
        t.tagName === "TEXTAREA" ||
        t.tagName === "SELECT" ||
        t.isContentEditable)
    ) {
      if (e.key === "Escape") t.blur();
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (helpOpen && e.key !== "?") return;
    const big = e.shiftKey ? 10 : 5;
    const hit = () => e.preventDefault();
    switch (e.key) {
      case " ":
      case "k":
      case "K":
        hit();
        tp.togglePlay();
        break;
      case "ArrowLeft":
        hit();
        tp.nudge(-big);
        break;
      case "ArrowRight":
        hit();
        tp.nudge(big);
        break;
      case "j":
      case "J":
        hit();
        tp.nudge(-10);
        break;
      case "l":
      case "L":
        hit();
        tp.nudge(10);
        break;
      case ",":
        hit();
        tp.frameStep(-1);
        break;
      case ".":
        hit();
        tp.frameStep(1);
        break;
      case "<":
        hit();
        tp.bumpSpeed(-1);
        break;
      case ">":
        hit();
        tp.bumpSpeed(1);
        break;
      case "i":
      case "I":
        hit();
        clip.markIn();
        break;
      case "o":
      case "O":
        hit();
        clip.markOut();
        break;
      case "s":
      case "S":
        hit();
        clip.save();
        break;
      case "f":
      case "F":
        hit();
        tp.toggleFullscreen();
        break;
      case "m":
      case "M":
        hit();
        tp.toggleMute();
        break;
      case "b":
      case "B":
        hit();
        ov.toggle("overlay");
        break;
      case "t":
      case "T":
        hit();
        ov.toggle("trail");
        break;
      case "d":
      case "D":
        hit();
        ov.toggle("spotlight");
        break;
      case "x":
      case "X":
        hit();
        ov.select(null);
        break;
      case "r":
      case "R":
        hit();
        reel.start(clip.clips.filter((c) => clip.picked.has(c.id)), true);
        break;
      case "?":
        hit();
        setHelpOpen((v) => !v);
        break;
      case "Escape":
        if (reel.running()) reel.end();
        else clip.clear();
        break;
    }
  };

  // One listener and one loop for the life of the page, both reading the
  // closure from the LATEST render out of a ref. Re-registering either on every
  // state change is how a video page starts dropping frames.
  useEffect(() => {
    drawRef.current = draw;
    keyRef.current = onKey;
    refitRef.current = ov.fit;
  });

  useEffect(() => {
    const h = (e: KeyboardEvent) => keyRef.current(e);
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  // Redraw once when something the overlay depends on changes and the video is
  // paused, so the loop is not running. Declared after the sync above, because
  // effects run in order and this one must see the current draw.
  useEffect(() => {
    drawRef.current(videoRef.current?.currentTime ?? 0);
  }, [data, overlay, trail, spotlight, ovSel, ovOffset]);

  /* ----------------------------------------------------------- deep links */

  /** A ui_action from the agent ("seek_film") arrives through the workspace
   *  as a prop, because the panel may not even be mounted when the frame
   *  lands. The stamp makes each request fire exactly once. */
  const lastSeekRef = useRef(0);
  useEffect(() => {
    if (!seekRequest || seekRequest.n === lastSeekRef.current) return;
    lastSeekRef.current = seekRequest.n;
    seek(seekRequest.t);
    // `seek` is memoised, and the stamp guard above keeps this a no-op except
    // on the render after a new request lands.
  }, [seekRequest, seek]);

  /** Idempotent, and hung off every media event that keeps firing rather than
   *  off `loadedmetadata` alone: the browser starts loading from the
   *  server-rendered markup, so on a local file the metadata can be there
   *  before React attaches a single handler — and then the scrub bar has no
   *  duration to place clip markers along. */
  const onMeta = () => {
    const vid = videoRef.current;
    if (!vid || !vid.videoWidth) return;
    if (vid.duration && vid.duration !== dur) setDur(vid.duration);
    // Seeking is the entire point of this page and it dies silently on a server
    // with no Range support. Only asked once there is enough loaded to answer.
    if (vid.readyState >= 3) setSeekWarn(!vid.seekable.length);
    ov.fit();
    drawRef.current(vid.currentTime);
    if (deepLinkRef.current) return;
    deepLinkRef.current = true;
    // `?t=12.5-30` opens with the range already marked, so whoever the link was
    // pasted to can save, re-cut or extend it.
    const q = (search.get("t") ?? "").split("-").map(Number);
    if (q.length === 2 && isFinite(q[0]) && isFinite(q[1]) && q[1] > q[0]) {
      // `?t=12.5-30` opens with the range already marked.
      clip.setRange(q[0], q[1]);
      tp.seek(q[0]);
    } else if (q.length === 1 && isFinite(q[0])) {
      tp.seek(q[0]);
    }
  };

  /* --------------------------------------------------------------- pointer */

  /** The time under the pointer, in the bubble above the bar. Written straight
   *  to the DOM — this fires on every pointermove and is not worth a render. */

  /** A player, or play/pause. The hit test is the overlay's — it owns the
   *  positions — and what to do with a miss is the panel's. */
  const onStageClick = (e: React.MouseEvent) => {
    const hit = ov.hitTest(e.clientX, e.clientY);
    if (hit != null) ov.select(hit);
    else tp.togglePlay();
  };

  /* ------------------------------------------------------------- rendering */

  // The film LIBRARY, not a gate: nothing picked, or the sidebar asked for it.
  if (!id || search.get("lib")) return <FilmGallery />;

  const d = dur || 1;

  return (
    /* No gutter between the film and the events beside it: the panel is a
       column of the editor, not a card floating next to one. When its
       CONTAINER is 64rem or wider the two columns fill it and scroll
       independently — the player and its timeline stay put while the events
       run the height of the page. A container query, not `xl:`: in the dock
       the window is wide and the film is not, and a viewport breakpoint drew
       two columns into 520px with the video squeezed to a sliver. 64rem on the
       page is the old 1280px window less the sidebar. */
    <div
      /* The column widths are dragged, so they are an inline style rather than
         a class — and the style is inert below 64rem, where this is a flex column
         and there are no columns to size. */
      style={{ gridTemplateColumns: `minmax(0,1fr) 1px ${railW}px` }}
      className="flex w-full flex-col @min-[64rem]:grid @min-[64rem]:h-full @min-[64rem]:min-h-0 @min-[64rem]:grid-rows-[auto_minmax(0,1fr)]"
    >
      <div className="grid min-w-0 content-start gap-3 @min-[64rem]:col-start-1 @min-[64rem]:row-start-1 @min-[64rem]:min-h-0 @min-[64rem]:overflow-y-auto">
        <div
          ref={playerRef}
          className={
            tp.fs
              ? "flex h-full flex-col justify-center bg-black"
              : "bg-card overflow-hidden"
          }
        >
          <div
            ref={stageRef}
            onClick={onStageClick}
            onDoubleClick={(e) => {
              e.preventDefault();
              tp.toggleFullscreen();
            }}
            onPointerMove={bump}
            onPointerLeave={() => tp.playing && setChrome(false)}
            style={tp.fs || !stageH ? undefined : { height: stageH }}
            /* ADJUSTABLE HEIGHT. CSS `resize-y` was tried first and cannot
               work here: the browser puts its grip in the bottom-right corner,
               which is exactly where the playback chrome sits, so the grip was
               never reachable. Hence the handle below — the editor's own
               divider gesture. Untouched, the stage keeps 16/9. */
            className={`relative w-full cursor-pointer overflow-hidden bg-black ${
              tp.fs ? "min-h-0 flex-1" : stageH ? "" : "aspect-video max-h-[92vh]"
            }`}
          >
            <video
              ref={videoRef}
              src={api.videoUrl(id)}
              playsInline
              preload="metadata"
              className="absolute inset-0 size-full object-contain"
              onLoadedMetadata={onMeta}
              onCanPlay={onMeta}
              onSeeked={() => {
                barPaint.current?.paintBuffered();
                drawRef.current(videoRef.current?.currentTime ?? 0);
              }}
              onProgress={() => {
                barPaint.current?.paintBuffered();
                onMeta();
              }}
              onEnded={reel.next}
              onError={() => setVideoErr(true)}
            />
            <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 size-full" />

            <FilmChrome
              tp={tp}
              clip={clip}
              moments={mo}
              dur={d}
              visible={chrome}
              snap={snap}
              onPlayClip={reel.play}
              onOpenHelp={() => setHelpOpen(true)}
              paintRef={barPaint}
              videoRef={videoRef}
            />

            <div className="pointer-events-none absolute top-2 left-2 flex flex-wrap gap-1.5 text-[11px] leading-4">
              <span
                ref={frameRef}
                className="text-muted-foreground rounded-md bg-black/60 px-2 py-1 tabular-nums backdrop-blur"
                style={{ display: "none" }}
              />
              <span
                ref={followRef}
                className="bg-primary text-primary-foreground rounded-md px-2 py-1 font-medium"
                style={{ display: "none" }}
              />
              <span
                ref={noteRef}
                className="text-warn rounded-md bg-black/70 px-2 py-1 backdrop-blur"
                style={{ display: "none" }}
              />
            </div>

            {reel.card && (
              <div className="absolute inset-0 grid place-items-center bg-black/85 p-6 text-center">
                <div>
                  <div className="text-primary text-[11px] tracking-[0.14em] uppercase">
                    {reel.card.tag}
                  </div>
                  <div className="my-2 text-2xl font-semibold tracking-[-0.3px]">{reel.card.head}</div>
                  <div className="text-muted-foreground text-sm">{reel.card.sub}</div>
                </div>
              </div>
            )}

            {videoErr && (
              /* A PLACEHOLDER FRAME, not an empty black rectangle: a pitch from
                 above, dimmed, with the reason over it — so a match with
                 nothing to play still reads as a film screen rather than a
                 broken one. It fills the stage the way a frame of the match
                 would, `object-cover`, and it is decorative, so it carries no
                 alt text. */
              <div className="absolute inset-0">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src="/pitch-placeholder.svg"
                  alt=""
                  aria-hidden
                  className="size-full object-cover opacity-40 select-none"
                />
                <p className="absolute inset-0 grid place-content-center place-items-center p-6 text-center text-sm text-white/90 [text-shadow:0_1px_3px_rgb(0_0_0/0.8)]">
                  This match has no playable video yet.
                  <br />
                  Clips and tags still work — they need only a time range.
                </p>
              </div>
            )}
          </div>

          {/* Dreambyte's timeline, with this match in it: the film as one
              track with its moments marked on it, the clips under it, and the
              playhead the bar above is driven from. It sits under the player
              rather than on it because a track you scrub is read while the
              video is still visible. */}
          {/* Inside the player so it survives fullscreen: a portal to
              document.body is not rendered while another element is fullscreen. */}
          <Dialog open={helpOpen} onOpenChange={setHelpOpen}>
            <DialogContent container={tp.fsEl ?? undefined} className="sm:max-w-lg">
              <DialogHeader>
                <DialogTitle>Keyboard</DialogTitle>
                <DialogDescription>Everything on this page has a key.</DialogDescription>
              </DialogHeader>
              <dl className="grid gap-1.5 sm:grid-cols-2">
                {SHORTCUTS.map(([k, what]) => (
                  <div key={k} className="flex items-baseline gap-2 text-xs">
                    <dt className="bg-muted shrink-0 rounded border px-1.5 py-0.5 font-mono">
                      {k}
                    </dt>
                    <dd className="text-muted-foreground">{what}</dd>
                  </div>
                ))}
              </dl>
            </DialogContent>
          </Dialog>
        </div>

        {progress && (
          <div className="grid gap-1">
            <Progress value={progress.total ? (progress.bytes / progress.total) * 100 : 100} className="h-1" />
            <p className="text-muted-foreground text-xs">
              Streaming tracks off the main thread — {(progress.bytes / 1e6).toFixed(1)}
              {progress.total ? ` of ${(progress.total / 1e6).toFixed(1)}` : ""} MB,{" "}
              {progress.n.toLocaleString()} detections indexed.
            </p>
          </div>
        )}

        {ov.coordWarn && (
          <Alert variant="destructive">
            <AlertTriangle />
            <AlertTitle>The overlay is not registered to this video</AlertTitle>
            <AlertDescription>
              {ov.coordWarn} tracks.json stores feet positions in the source video&apos;s image pixels;
              these are not pixels of this video — most likely pitch metres, or another resolution.
              Clips and tags are unaffected: a time range does not depend on the projection.
            </AlertDescription>
          </Alert>
        )}

        {seekWarn && (
          <Alert variant="destructive">
            <AlertTriangle />
            <AlertTitle>This video cannot be seeked</AlertTitle>
            <AlertDescription>
              The server answered without Range support, so the browser can only play straight
              through. Marking clips still works; jumping to them will not.
            </AlertDescription>
          </Alert>
        )}

        {ov.trackWarn && (
          <Alert variant="destructive">
            <AlertTriangle />
            <AlertTitle>No track {ov.trackWarn} in this match</AlertTitle>
            <AlertDescription>
              The link that sent you here was built from an earlier run — track ids are renumbered
              every time tracking re-runs. Every other player is still correct.
            </AlertDescription>
          </Alert>
        )}

      </div>

      {/* The rail's own divider. Same gesture as the one over the timeline:
          the border IS the handle, with a fat invisible strip over it, and it
          only exists where there are two columns to divide. */}
      <div
        role="separator"
        aria-label="Resize the events panel"
        onPointerDown={(e) => {
          const x0 = e.clientX;
          const w0 = railW;
          const move = (ev: PointerEvent) =>
            setRailW(Math.min(640, Math.max(240, w0 - (ev.clientX - x0))));
          const up = () => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
          };
          window.addEventListener("pointermove", move);
          window.addEventListener("pointerup", up);
        }}
        onDoubleClick={() => setRailW(360)}
        title="Drag to resize · double-click to reset"
        className="relative hidden cursor-col-resize bg-[var(--hairline)] transition-colors hover:bg-[var(--accent)] @min-[64rem]:col-start-2 @min-[64rem]:row-span-2 @min-[64rem]:row-start-1 @min-[64rem]:block"
      >
        <span aria-hidden className="absolute inset-y-0 -left-1.5 w-4" />
      </div>

      {/* SEVEN props, and two of them are whole objects. It was 28: eleven for
          the clip editor, eleven for the overlay, and the rest bookkeeping.
          The hooks own those groups now, so the rail asks for the group. */}
      <ClipsPanel
        className="@min-[64rem]:col-start-3 @min-[64rem]:row-span-2 @min-[64rem]:row-start-1"
        clip={clip}
        look={ov}
        moments={mo}
        onSeek={tp.seek}
        onPlay={reel.play}
        onReel={() => reel.start(clip.clips.filter((c) => clip.picked.has(c.id)), true)}
      />

      {/* THE TRACK SECTION, last in the source on purpose. Below xl the rail
          is not beside the film, it is under it — and the timeline used to be
          wedged between the two, so the film's own tracks sat above a list of
          events that pushed them off the screen. Last means the bottom of the
          page there, and the bottom-left cell of the grid when the rail IS
          beside it. */}
      {!tp.fs && (
        <div className="flex min-h-0 flex-col @min-[64rem]:col-start-1 @min-[64rem]:row-start-2">
          {/* THE DIVIDER, and it moves the boundary rather than growing one
              side: the film takes the height you drag it to and the tracks
              take everything left in the pane, so pulling up gives the
              timeline the room and pulling down gives it back. Pointer events
              on window, the way the workspace's pane divider does it, because
              a drag that leaves the 6px strip must not stop tracking. */}
          <div
            role="separator"
            aria-label="Resize the video"
            onPointerDown={(e) => {
              const y0 = e.clientY;
              const h0 = stageRef.current?.getBoundingClientRect().height ?? 0;
              const move = (ev: PointerEvent) =>
                setStageH(Math.max(180, h0 + ev.clientY - y0));
              const up = () => {
                window.removeEventListener("pointermove", move);
                window.removeEventListener("pointerup", up);
              };
              window.addEventListener("pointermove", move);
              window.addEventListener("pointerup", up);
            }}
            onDoubleClick={() => setStageH(null)}
            title="Drag to resize · double-click to reset"
            /* THE LINE IS THE HANDLE. No grip drawn on it: this is the border
               between the film and the tracks, and a pill floating in the
               middle of it is a second thing to aim at. Hairline with a fat
               invisible hit strip over it — the workspace divider's idiom —
               so the pointer finds it a few px either side. */
            className="relative h-px shrink-0 cursor-row-resize bg-[var(--tl-border)] transition-colors hover:bg-white/40"
          >
            <span aria-hidden className="absolute inset-x-0 -top-1.5 h-4" />
          </div>
          <div className="min-h-0 flex-1 overflow-hidden">
            <TrackTimeline
              dur={d}
              clips={clip.clips}
              moments={mo}
              picked={clip.picked}
              editing={clip.editing}
              io={{ in: clip.in, out: clip.out }}
              onSeek={tp.seek}
              onPlayClip={reel.play}
              paintRef={tlPaint}
            />
          </div>
        </div>
      )}

    </div>
  );
}
