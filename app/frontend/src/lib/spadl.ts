/** ONE PITCH COORDINATE SPACE, NAMED, SO EVERY IMPORT STOPS RE-LITIGATING IT.
 *
 *  Three frames are already in play and nothing said how they relate:
 *
 *    metres    model/engine/pitch.py and lib/pitch.ts. Origin BOTTOM-LEFT, y up,
 *              corner_bl = [0,0] and corner_tl = [0,W]. The real pitch, whose
 *              dimensions the operator typed - NOT assumed.
 *    display   the board and every pitch figure. 0-100 along, 0-100 across,
 *              y DOWN because it is drawn straight into SVG, home attacks
 *              right. See pitch-data.ts :: Magnet.
 *    SPADL     what the football world publishes in. 105 x 68 m, origin
 *              bottom-left, y up, and the ACTING team always attacking right.
 *
 *  metres and SPADL already agree on origin and axis direction, so the only
 *  differences are a scale and an orientation rule. display differs from both
 *  by a y flip, which is the bug this file exists to stop being rediscovered.
 *
 *  WHY SPADL AT ALL. socceraction, mplsoccer, kloppy and every public dataset
 *  speak it, so an import that lands in this space needs no per-source
 *  argument about which corner is the origin. socceraction's spec:
 *  https://socceraction.readthedocs.io/en/latest/documentation/spadl/spadl.html
 *
 *  THE NORMALISATION IS LOSSY AND THAT IS DELIBERATE ON SPADL'S PART, NOT OURS.
 *  SPADL is fixed at 105 x 68. lib/pitch.ts's own docstring is the warning:
 *  NCAA allows 115-120 yd x 70-75 yd, and assuming 105 x 68 puts 4-6% on every
 *  distance. So `toSpadl` takes the real dimensions and there is no default -
 *  a caller that does not know the pitch must not be able to normalise against
 *  it by accident. Round-tripping through SPADL and back is exact; measuring a
 *  distance IN SPADL metres on a pitch that is not 105 x 68 is not.
 */
import type { PitchDims, Pt } from "@/lib/pitch";

/** SPADL's fixed pitch. Not a default for anything - the size the spec
 *  normalises TO, which is a different thing from the size a pitch IS. */
export const SPADL_LENGTH = 105;
export const SPADL_WIDTH = 68;

/** Just the two fields these conversions read, so a caller holding a
 *  `report.meta.pitch` (length and width only) can pass it without inventing
 *  penalty-box depths it does not have. */
export type Dims = Pick<PitchDims, "length" | "width">;

/** THE BOARD'S SPACE -> REAL METRES.
 *
 *  y is flipped: display y counts DOWN from the top touchline because it is an
 *  SVG coordinate, and metres count UP from the bottom one. */
export function displayToMetres([x, y]: Pt, d: Dims): Pt {
  return [(x / 100) * d.length, ((100 - y) / 100) * d.width];
}

/** REAL METRES -> THE BOARD'S SPACE. The inverse, y flip included. */
export function metresToDisplay([x, y]: Pt, d: Dims): Pt {
  return [(x / d.length) * 100, 100 - (y / d.width) * 100];
}

/** REAL METRES -> SPADL's 105 x 68.
 *
 *  A rescale, not a reframe: both are bottom-left origin with y up. Pass the
 *  pitch the numbers were measured on; see the note about 4-6% above. */
export function toSpadl([x, y]: Pt, d: Dims): Pt {
  return [(x / d.length) * SPADL_LENGTH, (y / d.width) * SPADL_WIDTH];
}

/** SPADL's 105 x 68 -> real metres on this pitch. */
export function fromSpadl([x, y]: Pt, d: Dims): Pt {
  return [(x / SPADL_LENGTH) * d.length, (y / SPADL_WIDTH) * d.width];
}

/** THE OTHER DIRECTION OF PLAY.
 *
 *  A point reflection through the centre spot, which is what turning a pitch
 *  round actually is - not a mirror. Both axes, because a team attacking the
 *  other way also has its left and right swapped.
 *
 *  This is the operation the event feed needs before ANY pitch join: the feed
 *  states each row in the acting team's own frame, so one side's rows are
 *  already left-to-right and the other's are not. Reflecting the wrong side
 *  puts every mark in the mirror position, which does not look broken - it
 *  looks like a team that attacked the wrong way, and it hides behind the
 *  median. Involutive, so applying it twice is identity and a double
 *  correction cancels rather than compounds.
 */
export function flip([x, y]: Pt, d: Dims): Pt {
  return [d.length - x, d.width - y];
}

/** The same reflection in the board's 0-100 space. */
export function flipDisplay([x, y]: Pt): Pt {
  return [100 - x, 100 - y];
}

/** PUT AN ACTING-TEAM POINT INTO THE BOARD'S FRAME, WHERE HOME ATTACKS RIGHT.
 *
 *  The one call a figure should need. SPADL states every action left-to-right
 *  for whoever is acting, so an away action has to be turned round to sit on a
 *  board that draws home to the right; a home action is already that way.
 */
export function spadlToBoard(p: Pt, side: "home" | "away", d: Dims): Pt {
  const metres = fromSpadl(p, d);
  return metresToDisplay(side === "away" ? flip(metres, d) : metres, d);
}
