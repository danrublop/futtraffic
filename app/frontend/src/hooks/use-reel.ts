"use client";

import { useRef, useState } from "react";

import { clock } from "@/lib/timecode";
import type { Clip } from "@/lib/types";

/** How long the title card sits between clips in a reel. */
const CARD_MS = 1500;

interface Reel {
  list: Clip[];
  i: number;
  busy: boolean;
  cards: boolean;
}

/** The title card drawn over the picture between clips. */
export interface ReelCard {
  tag: string;
  head: string;
  sub: string;
}

export interface FilmReel {
  /** Non-null while a card is showing. The panel draws it over the video. */
  card: ReelCard | null;
  /** Whether a reel is playing. A reel can be running with no card on screen,
   *  so this is NOT `card !== null` — Esc has to stop one either way. A getter
   *  rather than state: it is read inside a keydown handler that is registered
   *  once for the life of the page, and it must not be a render-old value. */
  running: () => boolean;
  /** Play a list of clips back to back, with title cards between them. */
  start: (list: Clip[], cards: boolean) => void;
  /** One clip, no card. */
  play: (c: Clip) => void;
  /** Stop, wherever it is. */
  end: () => void;
  /** Called from the per-frame loop: has this clip reached its out-point? */
  check: (t: number) => void;
  /** Called from the video's `ended` event. */
  next: () => void;
}

/** Sequential playback of a list of clips.
 *
 *  Client-side only: this seeks one <video> element from clip to clip. There is
 *  no encoder and no concatenated file, which is why a reel needs nothing from
 *  the server and works on a match whose only asset is the upload.
 */
export function useReel(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  seek: (t: number) => void,
): FilmReel {
  const reelRef = useRef<Reel | null>(null);
  const [card, setCard] = useState<ReelCard | null>(null);

  const end = () => {
    reelRef.current = null;
    videoRef.current?.pause();
    setCard(null);
  };

  /** `busy` is the whole guard. Without it the end-of-clip test in the render
   *  loop and the video's own `ended` event both advance, and a seek that has
   *  not landed yet still reads the OLD currentTime — which is past the next
   *  clip's out-point, so the reel skips to the end in one frame. */
  const next = () => {
    const reel = reelRef.current;
    const vid = videoRef.current;
    if (!reel || reel.busy || !vid) return;
    reel.busy = true;
    reel.i += 1;
    if (reel.i >= reel.list.length) {
      end();
      return;
    }
    const c = reel.list[reel.i];
    vid.pause();
    if (reel.cards) {
      setCard({
        tag: c.tags.join(" · ") || "clip",
        head: c.name,
        sub: `${reel.i + 1} of ${reel.list.length} · ${clock(c.start)}–${clock(c.end)}`,
      });
    }
    window.setTimeout(
      () => {
        if (reelRef.current !== reel || reel.list[reel.i] !== c) return;
        setCard(null);
        seek(c.start);
        // Arm on the seek landing, or 400ms later if the browser decides the
        // playhead was already there and fires no `seeked`.
        const arm = () => {
          if (reelRef.current === reel && reel.list[reel.i] === c) reel.busy = false;
        };
        vid.addEventListener("seeked", arm, { once: true });
        window.setTimeout(arm, 400);
        void vid.play().catch(() => {});
      },
      reel.cards ? CARD_MS : 0,
    );
  };

  const start = (list: Clip[], cards: boolean) => {
    if (!list.length) return;
    reelRef.current = { list, i: -1, busy: false, cards };
    next();
  };

  const check = (t: number) => {
    const reel = reelRef.current;
    if (!reel || reel.busy) return;
    if (t >= reel.list[reel.i].end - 0.02) next();
  };

  return {
    card,
    running: () => reelRef.current !== null,
    start,
    play: (c: Clip) => start([c], false),
    end,
    check,
    next,
  };
}
