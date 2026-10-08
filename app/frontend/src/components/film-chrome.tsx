"use client";

import { useEffect, useRef } from "react";
import {
  FastForward,
  Gauge,
  Keyboard,
  Maximize,
  Minimize,
  Pause,
  Play,
  Rewind,
  Volume2,
  VolumeX,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { setText, teamName } from "@/hooks/use-film-overlay";
import type { ClipEditor } from "@/hooks/use-clip-editor";
import type { FilmTransport } from "@/hooks/use-transport";
import { RATES } from "@/hooks/use-transport";
import type { Moment } from "@/lib/moments";
import { clock, clockTenths } from "@/lib/timecode";
import type { Clip } from "@/lib/types";

/** The controls drawn ON the video: the scrub bar and the transport row.
 *
 *  WHY IT IS ITS OWN FILE. It is the only part of the film panel that is about
 *  the video's own chrome rather than about the match, and keeping it here is
 *  what stops the panel growing every time a control is added — the overlay
 *  toggle cluster lands here next.
 *
 *  IT PAINTS ITSELF. The played bar, the buffered bar, the playhead and the
 *  clock all move every frame, so none of them go through React: the component
 *  registers a painter into `paintRef` on mount and the panel's per-frame loop
 *  calls it. Same contract TrackTimeline already uses in this same panel, for
 *  the same reason.
 */
export interface FilmChromeProps {
  tp: FilmTransport;
  clip: ClipEditor;
  /** Everything with a time on it, for the marks above the line. */
  moments: Moment[];
  /** Duration to lay the bar out against. Never 0 — callers pass `dur || 1`. */
  dur: number;
  /** Whether the chrome is being shown. Paused it always is, because then the
   *  controls are the thing you came for. */
  visible: boolean;
  /** Pull a scrubbed time onto a nearby event. */
  snap: (t: number) => number;
  onPlayClip: (c: Clip) => void;
  onOpenHelp: () => void;
  paintRef: React.RefObject<{ paint: (t: number) => void; paintBuffered: () => void } | null>;
  /** The video, for reading duration and buffered ranges. */
  videoRef: React.RefObject<HTMLVideoElement | null>;
}

export function FilmChrome({
  tp,
  clip,
  moments,
  dur,
  visible,
  snap,
  onPlayClip,
  onOpenHelp,
  paintRef,
  videoRef,
}: FilmChromeProps) {
  const scrubRef = useRef<HTMLDivElement>(null);
  const playedRef = useRef<HTMLElement>(null);
  const headRef = useRef<HTMLElement>(null);
  const bufRef = useRef<HTMLElement>(null);
  const clockRef = useRef<HTMLSpanElement>(null);
  const hoverRef = useRef<HTMLDivElement>(null);

  const paint = (t: number) => {
    const d = videoRef.current?.duration || 0;
    if (clockRef.current) {
      const s = `${clockTenths(t)} / ${clock(d)}`;
      if (clockRef.current.textContent !== s) clockRef.current.textContent = s;
    }
    const p = d ? Math.min(100, (t / d) * 100) : 0;
    if (playedRef.current) playedRef.current.style.width = `${p.toFixed(3)}%`;
    if (headRef.current) headRef.current.style.left = `${p.toFixed(3)}%`;
  };

  const paintBuffered = () => {
    const vid = videoRef.current;
    const el = bufRef.current;
    if (!vid || !el) return;
    const d = vid.duration || 0;
    if (!d || !vid.buffered.length) {
      el.style.width = "0";
      return;
    }
    // Only the range under the playhead matters: it is the one that decides
    // whether the next five seconds stutter.
    let lo = 0;
    let hi = 0;
    const t = vid.currentTime;
    for (let i = 0; i < vid.buffered.length; i++) {
      const a = vid.buffered.start(i);
      const b = vid.buffered.end(i);
      if (t >= a - 0.25 && t <= b + 0.25) {
        lo = a;
        hi = b;
        break;
      }
      if (b > hi) {
        lo = a;
        hi = b;
      }
    }
    el.style.left = `${((lo / d) * 100).toFixed(2)}%`;
    el.style.width = `${(((hi - lo) / d) * 100).toFixed(2)}%`;
  };

  // Registered in an effect, and re-registered every render so the panel's
  // frame loop always calls the LATEST closure. Same shape TrackTimeline uses.
  // Repaint once on registration, because a re-render with the video paused
  // has no next frame to correct the bar on.
  useEffect(() => {
    paintRef.current = { paint, paintBuffered };
    paint(videoRef.current?.currentTime ?? 0);
    paintBuffered();
    return () => {
      paintRef.current = null;
    };
  });

  /** The time under the pointer, in the bubble above the bar. Written straight
   *  to the DOM — this fires on every pointermove and is not worth a render. */
  const paintHover = (e: React.PointerEvent) => {
    const el = scrubRef.current;
    const tip = hoverRef.current;
    if (!el || !tip) return;
    const r = el.getBoundingClientRect();
    const f = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    tip.style.left = `${(f * 100).toFixed(2)}%`;
    setText(tip, clock(f * tp.vdur()));
  };

  const scrubTime = (e: React.PointerEvent) => {
    const el = scrubRef.current;
    if (!el) return 0;
    const r = el.getBoundingClientRect();
    return snap(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * tp.vdur());
  };

  const band =
    clip.in != null && clip.out != null && clip.out > clip.in
      ? { left: `${(clip.in / dur) * 100}%`, width: `${((clip.out - clip.in) / dur) * 100}%` }
      : null;

  // It lives ON the video under a scrim, the way a player people already know
  // does — and because in fullscreen there is no "below the video" to put it
  // in. Clicks stop here: the stage behind it toggles play and picks players.
  return (
    <div
        onClick={(e) => e.stopPropagation()}
        onDoubleClick={(e) => e.stopPropagation()}
        className={`absolute inset-x-0 bottom-0 cursor-default bg-gradient-to-t from-black/95 via-black/70 to-transparent pt-20 max-sm:pt-10 text-white transition-opacity duration-200 [&_button]:border-white/25 [&_button]:text-white [&_button]:hover:bg-white/15 ${
          visible || !tp.playing ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
      >
        {/* 44px of hit area with a 4px line drawn through the middle of
            it: a thumb has to be able to land on the timeline without
            being able to see where it lands. The line thickens and the
            head appears under the pointer — the only affordance a bar
            this thin has got. */}
        <div
          ref={scrubRef}
          className="group/bar relative mx-3 flex h-9 cursor-pointer touch-none items-center max-sm:h-11"
          onPointerDown={(e) => {
            const mk = (e.target as HTMLElement).dataset.clip;
            if (mk) {
              const c = clip.clips.find((x) => x.id === mk);
              if (c) onPlayClip(c);
              return;
            }
            tp.seek(scrubTime(e));
            e.currentTarget.setPointerCapture(e.pointerId);
          }}
          onPointerMove={(e) => {
            if (e.currentTarget.hasPointerCapture(e.pointerId)) tp.seek(scrubTime(e));
            paintHover(e);
          }}
          onPointerLeave={() => setText(hoverRef.current, "")}
          onPointerUp={(e) => e.currentTarget.releasePointerCapture(e.pointerId)}
        >
          {/* The time under the pointer. */}
          <div
            ref={hoverRef}
            className="pointer-events-none absolute bottom-full mb-1 -translate-x-1/2 rounded bg-black/85 px-1.5 py-0.5 font-mono text-[11px] tabular-nums"
            style={{ display: "none" }}
          />

          {/* Moments, as marks above the line. Deliberately not tappable:
              a 4px mark needs 44px of hit area, and 44px of hit area on
              the bar is a tap that seeks nowhere near where you aimed.
              Clicking a moment is the events panel's job. */}
          {moments
            .filter((e) => e.t >= 0 && e.t <= dur)
            .map((e, i) => (
              <i
                key={`${e.t}-${i}`}
                title={`${teamName(e.side ?? null)} · ${e.label} · ${
                  e.exact ? clockTenths(e.t) : `${clock(e.t)} (minute only)`
                }`}
                // A minute-only mark is drawn faint: it is placed to
                // within 30s and a solid one would read as a timecode.
                className={`absolute top-2 block h-1.5 w-0.5 -translate-x-1/2 ${
                  e.side === "away" ? "bg-away" : "bg-home"
                } ${e.exact ? "" : "opacity-50"}`}
                style={{ left: `${(e.t / dur) * 100}%` }}
              />
            ))}

          <div className="relative h-1 w-full overflow-hidden rounded-full bg-white/30 transition-[height] group-hover/bar:h-1.5">
            <i ref={bufRef} className="absolute inset-y-0 block bg-white/50" />
            <i ref={playedRef} className="bg-primary absolute inset-y-0 left-0 block" />
            {band && (
              <i
                className="border-primary bg-primary/30 absolute inset-y-0 block border-x-2"
                style={band}
              />
            )}
          </div>

          {/* Clips get their own strip below the line rather than sitting
              on it: on the line, every tap inside a clip plays it instead
              of seeking, which on touch is most of the bar. */}
          {clip.clips.map((c) => (
            <i
              key={c.id}
              data-clip={c.id}
              title={`${c.name} (${clock(c.start)}–${clock(c.end)})`}
              className={`absolute bottom-2 block h-1.5 rounded-sm max-sm:bottom-3 max-sm:h-2.5 ${
                clip.picked.has(c.id) ? "bg-primary" : "bg-info"
              }`}
              style={{
                left: `${(c.start / dur) * 100}%`,
                width: `${Math.max(0.4, ((c.end - c.start) / dur) * 100)}%`,
              }}
            />
          ))}

          <i
            ref={headRef}
            className="bg-primary pointer-events-none absolute block size-3 -translate-x-1/2 scale-0 rounded-full transition-transform group-hover/bar:scale-100"
          />
        </div>

        {/* transport */}
        <div className="flex flex-wrap items-center gap-0.5 px-2 pb-2 max-sm:gap-1">
          {/* Play, ten seconds either way, the clock, speed, sound,
              fullscreen. Jumping moment to moment is the events panel's
              job; five seconds and single frames are keys. */}
          <Key label={tp.playing ? "Pause (space)" : "Play (space)"} onClick={tp.togglePlay}>
            {tp.playing ? <Pause /> : <Play />}
          </Key>
          <Key label="Back 10s (J)" onClick={() => tp.nudge(-10)}>
            <Rewind />
          </Key>
          <Key label="Forward 10s (L)" onClick={() => tp.nudge(10)}>
            <FastForward />
          </Key>
          <span ref={clockRef} className="ml-2 font-mono text-xs tabular-nums text-white/85">
            0:00.0 / 0:00
          </span>
          <span className="flex-1" />
          <span className="contents max-sm:hidden">
            <DropdownMenu>
              <Tooltip>
                <TooltipTrigger asChild>
                  <DropdownMenuTrigger asChild>
                    <Button
                      size="icon-sm"
                      variant="ghost"
                      aria-label="Playback speed"
                      className="max-sm:size-11 max-sm:[&_svg:not([class*='size-'])]:size-5"
                    >
                      <Gauge />
                    </Button>
                  </DropdownMenuTrigger>
                </TooltipTrigger>
                <TooltipContent>Playback speed ({tp.rate}×)</TooltipContent>
              </Tooltip>
              <DropdownMenuContent align="end" className="min-w-24">
                <DropdownMenuRadioGroup
                  value={String(tp.rate)}
                  onValueChange={(v) => tp.setSpeed(Number(v))}
                >
                  {RATES.map((r) => (
                    <DropdownMenuRadioItem key={r} value={String(r)}>
                      {r}×
                    </DropdownMenuRadioItem>
                  ))}
                </DropdownMenuRadioGroup>
              </DropdownMenuContent>
            </DropdownMenu>
            <Key label={tp.muted ? "Unmute (M)" : "Mute (M)"} onClick={tp.toggleMute}>
              {tp.muted ? <VolumeX /> : <Volume2 />}
            </Key>
          </span>
          <Key label="Fullscreen (F)" onClick={tp.toggleFullscreen}>
            {tp.fs ? <Minimize /> : <Maximize />}
          </Key>
          <span className="contents max-sm:hidden">
            <Key label="Keyboard shortcuts (?)" onClick={onOpenHelp}>
              <Keyboard />
            </Key>
          </span>
      </div>
    </div>
  );
}

function Key({
  label,
  onClick,
  disabled,
  children,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          size="icon-sm"
          variant="ghost"
          onClick={onClick}
          disabled={disabled}
          aria-label={label}
          // 44px on a phone. Every one of these is a transport control and a
          // mis-hit on transport costs the operator the moment they were watching.
          className="max-sm:size-11 max-sm:[&_svg:not([class*=\'size-\'])]:size-5"
        >
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
}
