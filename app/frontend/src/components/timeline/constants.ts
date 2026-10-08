/** The timeline's fixed metrics.
 *
 *  The numbers are fixed, so the timeline reads at the same size its
 *  origin does: a 32px ruler, a 148px track header, a 32px tool column and
 *  48px rows. Only MAX_PPS is ours. */

/** Max pixels-per-second. Dreambyte allows 1500 so a single frame spans ~50px
 *  and is clickable; here the finest thing anybody aims at is a play, and a
 *  match is an hour, so 400 (a second ≈ 400px) is already past useful. */
export const MAX_PPS = 400;
/** Below this the whole match is a smear and the ruler has no labels left. */
export const MIN_PPS = 0.1;
export const RULER_HEIGHT = 32;
/** How near, in px, a drag has to pass a moment to land on it instead. */
export const SNAP_THRESHOLD = 5;
/** The track-name headers down the left. An editing timeline puts a 32px tool
 *  column beside them; this one has none, because it cuts nothing - its only
 *  tools are zoom and fit, and those sit in the corner above the headers. */
export const TRACK_HEADER_WIDTH = 148;
export const TRACK_ROW_HEIGHT = 48;
/** Everything left of time zero: the headers, and the width the ruler, the
 *  playhead and every scrub coordinate are offset by. */
export const LEFT_GUTTER = TRACK_HEADER_WIDTH;
