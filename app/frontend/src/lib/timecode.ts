/** Seconds to a clock, in one place.
 *
 *  There were four of these: video-panel had `clock` and `clockMs`, clips-panel
 *  had a near-copy of `clock`, track-timeline had `timecode`, and the ported
 *  ruler has `formatTimecode`. Three different formats between them, and two of
 *  them disagreed on negative input — clips-panel's rendered `-2:-30` for -90,
 *  because `Math.floor(-90 / 60)` is -2 and `-90 % 60` is -30.
 *
 *  `components/timeline/time-ruler.tsx` keeps its own `formatTimecode` and is
 *  NOT converted. That file is a faithful port of dreambyte's, its test came
 *  across with it, and the folder's headers say so. Editing it to save one
 *  small function would cost the next upstream comparison more than the
 *  duplication costs us. It also does something these do not: hours.
 */

/** "12:04". Negative-safe: the sign is pulled out before the split, so -90 is
 *  "-1:30" and not "-2:-30". A negative time is not normal, but the film's
 *  frame offset can walk the playhead behind zero and a garbled clock reads as
 *  a rendering bug rather than as the edge of the video. */
export const clock = (t: number) => {
  const s = Math.abs(t);
  return `${t < 0 ? "-" : ""}${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
};

/** "12:04.3". Tenths, not milliseconds — a scrub wants to see itself move
 *  between seconds, and a third decimal place is noise at 25fps. */
export const clockTenths = (t: number) => `${clock(t)}.${Math.floor(Math.abs(t) * 10) % 10}`;
