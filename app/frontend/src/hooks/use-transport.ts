"use client";

import { useCallback, useState } from "react";

/** The speeds the menu offers. Nothing slower than a quarter reads as motion,
 *  and nothing faster than 2x is watchable on football. */
export const RATES = [0.25, 0.5, 1, 1.5, 2];

type FsDocument = Document & {
  webkitFullscreenElement?: Element | null;
  webkitExitFullscreen?: () => void;
};
type FsElement = HTMLElement & { webkitRequestFullscreen?: () => void };

/** Whatever the browser is currently showing fullscreen, webkit included.
 *  Exported because the panel has to mirror it into state on `fullscreenchange`
 *  and the vendor fallback should live in exactly one place. */
export const fullscreenElement = (): HTMLElement | null => {
  const doc = document as FsDocument;
  const el = doc.fullscreenElement ?? doc.webkitFullscreenElement ?? null;
  return el instanceof HTMLElement ? el : null;
};

export interface FilmTransport {
  playing: boolean;
  rate: number;
  muted: boolean;
  /** The element the browser is showing fullscreen, kept in STATE rather than
   *  read off a ref: the help dialog has to portal INTO it, and a portal to
   *  document.body is not rendered at all while something else is fullscreen. */
  fsEl: HTMLElement | null;
  fs: boolean;
  setFsEl: (el: HTMLElement | null) => void;
  setPlaying: (v: boolean) => void;

  /** The duration to lay controls out against: the element's if it has one,
   *  else the last one seen, else 1 so nothing divides by zero. */
  vdur: () => number;
  seek: (t: number) => void;
  nudge: (d: number) => void;
  /** One frame at the tracks' fps, or 25 when there are no tracks to ask. */
  frameStep: (d: number) => void;
  togglePlay: () => void;
  setSpeed: (r: number) => void;
  /** One step along RATES, clamped at both ends. */
  bumpSpeed: (d: number) => void;
  toggleMute: () => void;
  toggleFullscreen: () => void;
}

/** Driving the video element: play, seek, speed, sound, fullscreen.
 *
 *  Everything here is a thin wrapper over the element's own API, and that is
 *  the point — the element is the source of truth for whether it is playing,
 *  and this only mirrors what it reports into React so the buttons can draw.
 */
export function useTransport(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  /** What to make fullscreen: the player, not the stage, so the chrome and the
   *  help dialog come with it. */
  playerRef: React.RefObject<HTMLElement | null>,
  opts: {
    /** Last known duration, for laying out before metadata lands. */
    dur: number;
    /** Frame rate for single-stepping. Null when there are no tracks. */
    fps: number | null;
    /** Repaint after a seek. The video fires no frame callback while paused,
     *  so without this the canvas and the clock stay on the old time. */
    draw: (t: number) => void;
  },
): FilmTransport {
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [muted, setMuted] = useState(false);
  const [fsEl, setFsEl] = useState<HTMLElement | null>(null);

  const { dur, fps, draw } = opts;

  const vdur = () => videoRef.current?.duration || dur || 1;

  // Stable identity: the clip editor takes this as a dependency, and a `seek`
  // that changes every render would re-run its callbacks for nothing.
  const seek = useCallback(
    (t: number) => {
      const vid = videoRef.current;
      if (!vid) return;
      vid.currentTime = Math.max(0, Math.min(t, vid.duration || t));
      draw(vid.currentTime);
    },
    [videoRef, draw],
  );

  const nudge = (d: number) => seek((videoRef.current?.currentTime ?? 0) + d);

  const frameStep = (d: number) => {
    const vid = videoRef.current;
    if (!vid) return;
    vid.pause();
    seek(vid.currentTime + d / (fps || 25));
  };

  const togglePlay = () => {
    const vid = videoRef.current;
    if (!vid) return;
    if (vid.paused) void vid.play().catch(() => {});
    else vid.pause();
  };

  const setSpeed = (r: number) => {
    const vid = videoRef.current;
    if (!vid) return;
    vid.playbackRate = r;
    setRate(r);
  };

  const bumpSpeed = (d: number) => {
    const i = RATES.indexOf(rate);
    setSpeed(RATES[Math.max(0, Math.min(RATES.length - 1, (i < 0 ? 2 : i) + d))]);
  };

  const toggleMute = () => {
    const vid = videoRef.current;
    if (!vid) return;
    vid.muted = !vid.muted;
    setMuted(vid.muted);
  };

  const toggleFullscreen = () => {
    const doc = document as FsDocument;
    const el = playerRef.current as FsElement | null;
    if (doc.fullscreenElement || doc.webkitFullscreenElement) {
      (doc.exitFullscreen ?? doc.webkitExitFullscreen)?.call(doc);
    } else if (el) {
      (el.requestFullscreen ?? el.webkitRequestFullscreen)?.call(el);
    }
  };

  return {
    playing,
    rate,
    muted,
    fsEl,
    fs: fsEl !== null,
    setFsEl,
    setPlaying,
    vdur,
    seek,
    nudge,
    frameStep,
    togglePlay,
    setSpeed,
    bumpSpeed,
    toggleMute,
    toggleFullscreen,
  };
}
