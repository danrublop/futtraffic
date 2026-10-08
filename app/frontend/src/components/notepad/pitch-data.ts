/** The board's data and drawing, with no editor in it.
 *
 *  Split out from pitch-node.ts so the seed template and its tests can build a
 *  board without pulling ProseMirror into a node process — doc-template.ts needs
 *  DEFAULT_MAGNETS and nothing else.
 *
 *  A board is TWO layers: magnets (the discs) and shapes (everything drawn over
 *  them — passes, runs, dribbles, freehand, zones, text). They are separate
 *  because they behave differently: a magnet is a thing you pick up and put
 *  down, a shape is a stroke you make once and then keep or delete. Merging them
 *  into one "objects" array reads tidier and makes every hit-test and every
 *  toolbar branch check a discriminant it did not need to.
 */

export interface Magnet {
  /** 0-100 along the pitch, 0-100 across it. Home attacks to the RIGHT. */
  x: number;
  y: number;
  /** Shirt number, initials — whatever is on the disc. */
  label: string;
  /** The player's name, printed beside the disc when the side's kit asks for
   *  it. Separate from `label` because a number goes ON the disc and a name
   *  goes BESIDE it — a board that has to fit "Fernández" inside 2.4 m is a
   *  board of unreadable discs. */
  name?: string;
  side: "home" | "away" | "ball" | "gear";
  /** A key of GEAR, when side is "gear". A cone is not a player with a
   *  different colour — it has no number, no team and its own silhouette — but
   *  it IS something you pick up and put down, so it lives in the same array
   *  rather than in a third layer with its own hit-testing and its own drag. */
  gear?: string;
}

/** The marks a coach draws over the discs.
 *
 *  The three arrow kinds are the ones every coaching diagram in the world
 *  already uses and the ones the InStat reports use: a SOLID line is the ball
 *  moving without the player, a DASHED line is the player moving without the
 *  ball, a WAVY line is the player carrying it. Nothing here is configurable
 *  beyond that, because a board where the reader has to learn the author's
 *  private line vocabulary is a board that has to be explained in a caption.
 */
export type ShapeKind = "pass" | "run" | "dribble" | "pen" | "zone" | "circle" | "poly" | "text";

export interface Shape {
  kind: ShapeKind;
  /** Flat [x0,y0,x1,y1,…] in the same 0-100 space as the magnets. `pass`,
   *  `run` and `dribble` use the first and last point; `pen` and `poly` use all
   *  of them; `zone` and `circle` use two opposite corners; `text` uses the
   *  first. */
  pts: number[];
  /** A key of INK, never a raw colour — this string reaches innerHTML. */
  color: string;
  /** The label, for `text`. Ignored by every other kind. */
  label?: string;
}

/** WHAT A SIDE LOOKS LIKE.
 *
 *  A board is drawn for a particular pair of teams, and a coach who plays in
 *  red against blue should not have to explain that the salmon ones are theirs.
 *  Everything here is per side and applies to every disc on that side at once —
 *  a kit is a team's, not a player's, and per-disc styling is how a diagram
 *  ends up with fourteen different circles and no vocabulary.
 */
export interface Kit {
  /** A key of KIT_COLOURS. */
  colour: string;
  /** A key of KIT_COLOURS. Only drawn by the two-colour style. */
  colour2: string;
  /** "solid" or "halves". */
  style: string;
  /** Hollow rather than filled — the second appearance in the panel. Useful
   *  for the side a diagram is NOT about: an outline reads as furniture. */
  outline: boolean;
  /** Multiplies the 1.2 m disc radius. Clamped to 0.6-1.8 — smaller than that
   *  and the number is gone, bigger and eleven of them cover the pitch. */
  size: number;
  /** Print the number on the disc. */
  number: boolean;
  /** Dark number rather than light. */
  numberDark: boolean;
  /** "off", "above" or "under" — where the player's name goes. */
  name: string;
}

export const DEFAULT_KITS: Record<"home" | "away", Kit> = {
  home: {
    colour: "salmon",
    colour2: "white",
    style: "solid",
    outline: false,
    size: 1,
    number: true,
    numberDark: true,
    name: "off",
  },
  away: {
    colour: "ink",
    colour2: "white",
    style: "solid",
    outline: false,
    size: 1,
    number: true,
    numberDark: false,
    name: "off",
  },
};

/** The palette a kit is picked from. A key of this table is what gets stored,
 *  never a colour string — the board is one of the few parts of a saved
 *  document an agent can write, and every one of these reaches innerHTML. */
export const KIT_COLOURS: Record<string, string> = {
  salmon: "#f2a2ab",
  ink: "#16181c",
  white: "#f2f4f6",
  red: "#d8453c",
  maroon: "#8d2b3a",
  orange: "#f4a340",
  yellow: "#efc63c",
  green: "#35a05a",
  teal: "#149c8e",
  sky: "#6fb6e8",
  blue: "#3f7fd6",
  purple: "#8a63d2",
  grey: "#9aa3ad",
};

/** The board's whole state. */
export interface Board {
  magnets: Magnet[];
  shapes: Shape[];
  /** A key of VIEWS. Which part of the pitch is on screen — the data never
   *  moves, so switching from a full pitch to the penalty box reframes the
   *  drawing rather than rebuilding it. */
  view: string;
  /** "h" or "v". Which way up the pitch is drawn. Vertical is not a different
   *  board — it is the same coordinates turned a quarter turn, so a diagram
   *  drawn across the pitch survives being stood up. */
  orient: string;
  /** A key of LINE_TEMPLATES: which markings are painted. */
  lines: string;
  /** How each side is dressed. */
  kits: Record<"home" | "away", Kit>;
  /** A key of GRASS. */
  grass: string;
  /** Mown bands across the pitch. */
  stripes: boolean;
  /** What this board rests on. Empty for a board somebody drew from nothing —
   *  a diagram is a claim about a plan, not about a match. Set the moment the
   *  board takes anything from the report. */
  note?: string;
  tier?: string;
}

/** A 4-4-2 and a ball, which is where a coach starts drawing more often than
 *  from an empty board. Home defends the left. */
export const DEFAULT_MAGNETS: Magnet[] = [
  ...[
    [8, 50, "1"],
    [22, 20, "2"], [22, 40, "5"], [22, 60, "6"], [22, 80, "3"],
    [38, 20, "7"], [38, 40, "4"], [38, 60, "8"], [38, 80, "11"],
    [46, 38, "9"], [46, 62, "10"],
  ].map(([x, y, label]) => ({ x: x as number, y: y as number, label: String(label), side: "home" as const })),
  ...[
    [92, 50, "1"],
    [78, 20, "2"], [78, 40, "5"], [78, 60, "6"], [78, 80, "3"],
    [62, 20, "7"], [62, 40, "4"], [62, 60, "8"], [62, 80, "11"],
    [54, 38, "9"], [54, 62, "10"],
  ].map(([x, y, label]) => ({ x: x as number, y: y as number, label: String(label), side: "away" as const })),
  { x: 50, y: 50, label: "", side: "ball" },
];

export const EMPTY_BOARD: Board = {
  magnets: DEFAULT_MAGNETS,
  shapes: [],
  view: "full",
  orient: "h",
  lines: "standard",
  grass: "green",
  // Off. A scouting report's pitch is flat green — mown bands are television,
  // and under twenty numbered discs and a dozen arrows they are one more thing
  // competing with the marks somebody drew on purpose.
  stripes: false,
  kits: DEFAULT_KITS,
};

/** The shapes a coach reaches for, as fractions of their own half. Home
 *  defends the left, so an away shape is these mirrored. */
export const FORMATIONS: Record<string, [number, number][]> = {
  "4-4-2": [
    [8, 50],
    [22, 20], [22, 40], [22, 60], [22, 80],
    [38, 20], [38, 40], [38, 60], [38, 80],
    [46, 38], [46, 62],
  ],
  "4-3-3": [
    [8, 50],
    [22, 18], [22, 40], [22, 60], [22, 82],
    [34, 32], [36, 50], [34, 68],
    [47, 20], [48, 50], [47, 80],
  ],
  "4-2-3-1": [
    [8, 50],
    [22, 18], [22, 40], [22, 60], [22, 82],
    [31, 38], [31, 62],
    [42, 20], [42, 50], [42, 80],
    [49, 50],
  ],
  "3-5-2": [
    [8, 50],
    [21, 30], [20, 50], [21, 70],
    [33, 12], [34, 36], [32, 50], [34, 64], [33, 88],
    [47, 40], [47, 60],
  ],
  "5-3-2": [
    [8, 50],
    [20, 14], [21, 34], [20, 50], [21, 66], [20, 86],
    [34, 32], [33, 50], [34, 68],
    [47, 40], [47, 60],
  ],
  "3-4-3": [
    [8, 50],
    [21, 30], [20, 50], [21, 70],
    [34, 14], [33, 40], [33, 60], [34, 86],
    [47, 20], [48, 50], [47, 80],
  ],
};

/** Build a full board from a formation, with the away side mirrored across the
 *  halfway line so both teams face each other. */
export function fromFormation(name: string): Magnet[] {
  const shape = FORMATIONS[name] ?? FORMATIONS["4-4-2"];
  const shirt = (i: number) => String(i === 0 ? 1 : i + 1);
  return [
    ...shape.map(([x, y], i) => ({ x, y, label: shirt(i), side: "home" as const })),
    ...shape.map(([x, y], i) => ({ x: 100 - x, y, label: shirt(i), side: "away" as const })),
    { x: 50, y: 50, label: "", side: "ball" as const },
  ];
}

/** One side only, leaving the other team where it is. Half a coach's diagrams
 *  are one team's shape against nobody, and the other half are one team's
 *  shape against an opponent they have already placed by hand. */
export function sideFormation(name: string, side: "home" | "away", magnets: Magnet[]): Magnet[] {
  const shape = FORMATIONS[name] ?? FORMATIONS["4-4-2"];
  const shirt = (i: number) => String(i === 0 ? 1 : i + 1);
  const placed = shape.map(([x, y], i) => ({
    x: side === "home" ? x : 100 - x,
    y,
    label: shirt(i),
    side,
  }));
  return [...magnets.filter((m) => m.side !== side), ...placed];
}

/** The discs.
 *
 *  A scouting report puts the team it is ABOUT in one warm colour and the
 *  opposition in plain black, which is the right way round: the black dots are
 *  furniture you read past, and the coloured ones are the thing being said. A
 *  white-vs-blue pair reads as two equal teams, so a board about our press
 *  looked like a board about a match.
 */
export const MAGNET_FILL: Record<Magnet["side"], string> = {
  home: "#f2a2ab",
  away: "#16181c",
  ball: "#f4a340",
  // Training kit is equipment, not a team: one neutral colour so a pitch full
  // of cones does not read as a third side.
  gear: "#e8b04b",
};

/** A DISC IS A COUNTER ON A TABLE, not a coloured dot.
 *
 *  Lit from above, darker underneath, a lit edge across the top and a soft
 *  shadow on the grass beneath — which is what a physical tactics board looks
 *  like, and what makes a disc read as sitting ON the pitch rather than being
 *  painted onto it. It buys legibility for nothing as well: a flat black disc
 *  on ink-coloured grass is a hole, and the same disc with a lit top edge is an
 *  object.
 *
 *  Derived from the kit's colour rather than a hand-tuned table, because the
 *  colour is now a coach's choice out of thirteen. Mixing TOWARDS white and
 *  black by a fixed fraction works at both ends: 32% of the way to white lifts
 *  black to a visible grey and leaves a pale salmon still pale. */
const chan = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const mix = (h: string, to: number, amt: number) =>
  "#" +
  chan(h)
    .map((c) => Math.round(c + (to - c) * amt).toString(16).padStart(2, "0"))
    .join("");
const shadeOf = (h: string) => ({
  top: mix(h, 255, 0.32),
  bottom: mix(h, 0, 0.2),
  rim: mix(h, 0, 0.45),
});

/** The colour a side is actually drawn in, and its two other stops. */
export const kitColour = (k: Kit | undefined, which: 1 | 2 = 1) =>
  pick(KIT_COLOURS, (which === 1 ? k?.colour : k?.colour2) ?? "", KIT_COLOURS.salmon);

/** The ball and the equipment answer to nobody's kit. */
const FIXED_SHADE: Record<string, string> = { ball: "#f4a340", gear: "#e8b04b" };

/** The gradients and the shadow the discs reference, emitted once per board by
 *  surfaceSvg — which every drawing path already puts first, so a disc can
 *  never ask for paint that does not exist yet.
 *
 *  Only the colours this board actually wears, keyed by the palette key, which
 *  is narrowed by parseKits before it gets here. ids are document-global in
 *  SVG, so two boards on a page define the same ones twice — identical
 *  definitions, so the first wins and nothing looks different; the alternative
 *  is a per-board nonce threaded through a function the exporter calls
 *  headless. */
export function discDefs(kits: Record<"home" | "away", Kit> | undefined): string {
  const used: Record<string, string> = { ...FIXED_SHADE };
  for (const side of ["home", "away"] as const) {
    const k = kits?.[side];
    used[k?.colour ?? DEFAULT_KITS[side].colour] = kitColour(k, 1);
    // The second colour only when something draws it. Every gradient defined
    // here is bytes in every saved board.
    if (k?.style === "halves") used[k.colour2] = kitColour(k, 2);
  }
  return (
    `<defs>` +
    Object.entries(used)
      .map(([k, hex]) => {
        const c = shadeOf(hex);
        return (
          `<linearGradient id="ftDisc-${k}" x1="0" y1="0" x2="0" y2="1">` +
          `<stop offset="0" stop-color="${c.top}"/><stop offset="1" stop-color="${c.bottom}"/>` +
          `</linearGradient>`
        );
      })
      .join("") +
    // In metres: a 22 cm drop and a 16 cm blur under a 2.4 m disc.
    `<filter id="ftDiscShadow" x="-60%" y="-60%" width="220%" height="220%">` +
    `<feDropShadow dx="0" dy="0.22" stdDeviation="0.16" flood-color="#00000059"/>` +
    `</filter></defs>`
  );
}

/** Ink for a number sitting on MAGNET_FILL. The board itself reads this off
 *  the side's kit now — see `numberDark` — but chart-svg draws its own discs in
 *  the pass network, where there is no kit to ask. */
export const MAGNET_TEXT: Record<Magnet["side"], string> = {
  home: "#5e1420",
  away: "#ffffff",
  ball: "#5e3a0a",
  gear: "#3a2a08",
};

/** WHAT ELSE GOES ON A PITCH. A session plan is cones and goals as much as it
 *  is players, and a coach who cannot put a mini-goal on the board draws it as
 *  a rectangle and explains it in the caption.
 *
 *  Each entry draws itself around (0,0) in METRES and is translated into place,
 *  so the sizes below are real: a cone is knee-high, a mini-goal is three
 *  metres across, a pole is a pole. They are turned upright in a vertical board
 *  the way text is — a cone lying on its side is not a cone. */
export const GEAR: Record<
  string,
  { label: string; /** The viewBox a preview of it needs. */ box: string; draw: (c: string) => string }
> = {
  cone: {
    label: "Cone",
    box: "-1.3 -1.4 2.6 2.6",
    draw: (c) =>
      `<path d="M 0,-1.1 L 0.7,0.6 L -0.7,0.6 Z" fill="${c}" stroke="#00000055" stroke-width="0.1"/>` +
      `<rect x="-1" y="0.6" width="2" height="0.32" rx="0.14" fill="${c}" stroke="#00000055" stroke-width="0.1"/>`,
  },
  disc: {
    label: "Marker disc",
    box: "-1.2 -1.2 2.4 2.4",
    draw: (c) =>
      `<ellipse cx="0" cy="0" rx="0.95" ry="0.42" fill="${c}" stroke="#00000055" stroke-width="0.1"/>`,
  },
  pole: {
    label: "Pole",
    box: "-1.2 -2.7 2.4 5.1",
    draw: (c) =>
      `<rect x="-0.16" y="-2.4" width="0.32" height="4.2" fill="${c}" stroke="#00000055" stroke-width="0.08"/>` +
      `<ellipse cx="0" cy="1.8" rx="0.7" ry="0.3" fill="${c}" stroke="#00000055" stroke-width="0.08"/>`,
  },
  hurdle: {
    label: "Hurdle",
    box: "-1.4 -1 2.8 2.3",
    draw: (c) =>
      `<path d="M -1.1,0.9 L -1.1,-0.6 L 1.1,-0.6 L 1.1,0.9" fill="none" stroke="${c}" stroke-width="0.28" stroke-linecap="round"/>`,
  },
  ladder: {
    label: "Ladder",
    box: "-1.3 -3.3 2.6 6.6",
    draw: (c) =>
      `<rect x="-1" y="-3" width="2" height="6" fill="none" stroke="${c}" stroke-width="0.22"/>` +
      [-1.5, 0, 1.5]
        .map((y) => `<line x1="-1" y1="${y}" x2="1" y2="${y}" stroke="${c}" stroke-width="0.22"/>`)
        .join(""),
  },
  goal: {
    label: "Mini goal",
    box: "-1.8 -1.2 3.6 2.4",
    draw: (c) =>
      `<path d="M -1.5,0.9 L -1.5,-0.9 L 1.5,-0.9 L 1.5,0.9" fill="#ffffff22" stroke="${c}" stroke-width="0.3" stroke-linejoin="round"/>` +
      [-0.75, 0, 0.75]
        .map((x) => `<line x1="${x}" y1="-0.9" x2="${x}" y2="0.9" stroke="${c}" stroke-width="0.1"/>`)
        .join(""),
  },
};

/** The pens. A shape's `color` is a KEY of this table, never a colour string:
 *  the board is one of the few things in a saved document an agent can write,
 *  and every value here ends up inside innerHTML on the way out. */
export const INK: Record<string, string> = {
  white: "#ffffff",
  yellow: "#f7d154",
  red: "#e8564a",
  blue: "#5b9bf0",
  black: "#0d1117",
};

/** Which part of the pitch is on screen.
 *
 *  A crop, not a different board: the magnets and shapes stay in full-pitch
 *  coordinates and only the window over them changes, so a set-piece drawn in
 *  the box view is still in the right place when the board is switched back to
 *  the full pitch. `rot` turns the crop upright — the attacking goal at the
 *  top — which is how every set-piece diagram is drawn and is unreadable any
 *  other way at box scale.
 */
export interface View {
  x: number;
  y: number;
  w: number;
  h: number;
  /** The way round this crop is normally drawn. Picking the crop sets the
   *  board's orientation to match, and the coach can still turn it back —
   *  a hint, not a property of the crop. */
  preferV: boolean;
  label: string;
}

/** Crops are stated in METRES on a 105 x 68 pitch, which is also the space
 *  everything is drawn in — see the note above the pitch markings. */
export const VIEWS: Record<string, View> = {
  // The surround is 5 m, not 2.5. A scouting report always leaves a clear band
  // of grass outside the touchline; at 2.5 m the halfway line's ends and the
  // corner arcs sit right against the frame and the board reads as cropped.
  full: { x: -5, y: -5, w: 115, h: 78, preferV: false, label: "Full pitch" },
  half: { x: 49, y: -5, w: 61, h: 78, preferV: true, label: "Attacking half" },
  // The crop a scouting report actually draws its attacking diagrams in: the
  // move starts a few metres inside your own half and the frame has to hold
  // the deepest player in it, so "Attacking half" cuts the holding midfielder
  // out of the picture that is about him.
  attacking: { x: 25, y: -5, w: 85, h: 78, preferV: true, label: "Attacking two-thirds" },
  third: { x: 66.5, y: -5, w: 43.5, h: 78, preferV: true, label: "Final third" },
  box: { x: 78.5, y: 3, w: 31.5, h: 62, preferV: true, label: "Penalty box" },
};

export const viewOf = (name: string): View => pick(VIEWS, name, VIEWS.full);

/** Percent-of-pitch (how a board is STORED) to metres (how it is DRAWN). */
export const LENGTH_M = 105;
export const WIDTH_M = 68;
export const mx = (x: number) => (x / 100) * LENGTH_M;
export const my = (y: number) => (y / 100) * WIDTH_M;

/** Standing the pitch up turns the content group by rotate(-90 52.5 34) about
 *  the centre spot, which maps a pitch point (x,y) to (y + 18.5, 86.5 - x). The
 *  crop's window in that turned space is what viewBox has to describe.
 *
 *  A turn and not a second coordinate system, which is the whole reason a
 *  diagram drawn on a horizontal board is still right when it is stood up. */
const ROT_U = (LENGTH_M - WIDTH_M) / 2; // 18.5
const ROT_V = (LENGTH_M + WIDTH_M) / 2; // 86.5

export const isVertical = (orient: string) => orient === "v";

export function viewBox(v: View, vertical: boolean): { box: string; aspect: number } {
  if (vertical) {
    return { box: `${v.y + ROT_U} ${ROT_V - v.x - v.w} ${v.h} ${v.w}`, aspect: v.h / v.w };
  }
  return { box: `${v.x} ${v.y} ${v.w} ${v.h}`, aspect: v.w / v.h };
}

/** A point inside the drawn box, as a fraction of it, back to stored percent. */
export function toPitch(
  v: View,
  vertical: boolean,
  fx: number,
  fy: number,
): { x: number; y: number } {
  const [bx, by, bw, bh] = viewBox(v, vertical).box.split(" ").map(Number);
  const u = bx + fx * bw;
  const w = by + fy * bh;
  const p = vertical ? { x: ROT_V - w, y: u - ROT_U } : { x: u, y: w };
  return { x: clamp((p.x / LENGTH_M) * 100), y: clamp((p.y / WIDTH_M) * 100) };
}

const clamp = (n: number) => Math.min(100, Math.max(0, n));

// ---------------------------------------------------------------------------
// The pitch itself, drawn in METRES on a 105 x 68 field to the real FIFA
// dimensions — 16.5 m penalty area, 9.15 m centre circle, 7.32 m goal.
//
// Metres, and not the 0-100 percent the board is stored in, because that space
// is anisotropic: 100 units along the pitch is 105 m and 100 across is 68 m, so
// a circle drawn in it comes out an ellipse and a 20 m run drawn sideways looks
// longer than the same run drawn forwards. A coach's diagram that lies about
// distance is a diagram they will read a distance off. Storage stays in percent
// so boards saved before this still load; the conversion is mx()/my().
//
// Drawn once here so the editor's live board and the exported SVG cannot draw
// different pitches.
// ---------------------------------------------------------------------------
/** The real markings: goal areas, penalty areas and spots, the D, the centre
 *  circle, corner arcs and the goals themselves. */
const MARKINGS = `
  <circle cx="52.5" cy="34" r="9.15"/>
  <circle cx="52.5" cy="34" r="0.35" fill="#ffffff" stroke="none"/>
  <rect x="0" y="13.84" width="16.5" height="40.32"/>
  <rect x="0" y="24.84" width="5.5" height="18.32"/>
  <circle cx="11" cy="34" r="0.35" fill="#ffffff" stroke="none"/>
  <path d="M 16.5 26.69 A 9.15 9.15 0 0 1 16.5 41.31"/>
  <rect x="88.5" y="13.84" width="16.5" height="40.32"/>
  <rect x="99.5" y="24.84" width="5.5" height="18.32"/>
  <circle cx="94" cy="34" r="0.35" fill="#ffffff" stroke="none"/>
  <path d="M 88.5 26.69 A 9.15 9.15 0 0 0 88.5 41.31"/>
  <path d="M 0 1 A 1 1 0 0 0 1 0"/>
  <path d="M 104 0 A 1 1 0 0 0 105 1"/>
  <path d="M 105 67 A 1 1 0 0 0 104 68"/>
  <path d="M 1 68 A 1 1 0 0 0 0 67"/>
  <rect x="-1.6" y="30.34" width="1.6" height="7.32" stroke-opacity="0.85"/>
  <rect x="105" y="30.34" width="1.6" height="7.32" stroke-opacity="0.85"/>`;

const OUTLINE = `<rect x="0" y="0" width="105" height="68"/><line x1="52.5" y1="0" x2="52.5" y2="68"/>`;

/** The thirds, as a coach draws them when the point is WHERE rather than what. */
const THIRDS = `<line x1="35" y1="0" x2="35" y2="68"/><line x1="70" y1="0" x2="70" y2="68"/>`;

/** Five lanes across the pitch: two wings, two half-spaces, the middle. */
const CHANNELS = [13.6, 27.2, 40.8, 54.4]
  .map((y) => `<line x1="0" y1="${y}" x2="105" y2="${y}"/>`)
  .join("");

/** What the pitch is marked with. The dashed guides are a separate group so
 *  they read as the coach's own overlay rather than as real paint. */
export const LINE_TEMPLATES: Record<string, { label: string; solid: string; dashed: string }> = {
  standard: { label: "Full markings", solid: OUTLINE + MARKINGS, dashed: "" },
  thirds: { label: "Thirds", solid: OUTLINE + MARKINGS, dashed: THIRDS },
  channels: { label: "Channels", solid: OUTLINE + MARKINGS, dashed: CHANNELS },
  grid: { label: "Grid", solid: OUTLINE + MARKINGS, dashed: THIRDS + CHANNELS },
  halves: { label: "Halves only", solid: OUTLINE, dashed: "" },
  blank: { label: "Blank", solid: `<rect x="0" y="0" width="105" height="68"/>`, dashed: "" },
};

/** Pitch colours.
 *
 *  `green` is the default and it is the SCOUTING REPORT's green, not a
 *  television one: pale, flat, and quiet enough that a black opposition dot
 *  and a white marking both read cleanly on top of it. A saturated broadcast
 *  green with mown bands looks better on its own and worse the moment you put
 *  twenty numbered discs and a dozen arrows on it, which is the only thing a
 *  board is ever used for.
 */
export const GRASS: Record<string, { label: string; fill: string; band: string }> = {
  green: { label: "Green", fill: "#a9dcb4", band: "#00000008" },
  field: { label: "Field", fill: "#3f8f4f", band: "#00000018" },
  deep: { label: "Deep", fill: "#15321d", band: "#ffffff10" },
  teal: { label: "Teal", fill: "#17624f", band: "#00000018" },
  slate: { label: "Slate", fill: "#2b3440", band: "#ffffff0d" },
  ink: { label: "Ink", fill: "#14161a", band: "#ffffff0d" },
  chalk: { label: "Chalk", fill: "#e7ebee", band: "#00000010" },
};

/** Mown bands, eight across the length. Painted under the markings and over
 *  the base colour, so the pitch reads as a pitch and not as a green rectangle
 *  — which is most of why a board looks like a board. */
const stripeSvg = (band: string) =>
  Array.from({ length: 8 }, (_, i) =>
    i % 2
      ? `<rect x="${(i * 105) / 8}" y="0" width="${105 / 8}" height="68" fill="${band}"/>`
      : "",
  ).join("");

/** The whole playing surface for a board: colour, bands and markings.
 *
 *  Every value here comes out of a fixed table keyed by a string that
 *  `parseBoard` has already narrowed, so nothing a user or an agent typed
 *  reaches this markup. */
export function surfaceSvg(
  board: Pick<Board, "lines" | "grass" | "stripes"> & Partial<Pick<Board, "kits">>,
  /** Multiplies every line width. A 22 cm line on a 105 m pitch is a tenth of
   *  a pixel in a 60 px swatch — invisible, so every template would look
   *  identical in the picker that exists to tell them apart. */
  boldness = 1,
): string {
  const g = pick(GRASS, board.grass, GRASS.green);
  const t = pick(LINE_TEMPLATES, board.lines, LINE_TEMPLATES.standard);
  const chalky = board.grass === "chalk";
  const ink = chalky ? "#33383d" : "#ffffff";
  // A scouting report's markings are SOLID, not a suggestion. At 0.55 the
  // lines were fine on a dark broadcast green and went to mush the moment the
  // pitch became the pale one a report actually uses.
  const solid = chalky ? 0.65 : board.grass === "green" ? 0.95 : 0.55;
  return (
    discDefs(board.kits) +
    `<rect x="-8" y="-8" width="121" height="84" fill="${g.fill}"/>` +
    (board.stripes ? stripeSvg(g.band) : "") +
    `<g fill="none" stroke="${ink}" stroke-opacity="${solid}" stroke-width="${(0.22 * boldness).toFixed(2)}" stroke-linecap="square">${t.solid}</g>` +
    (t.dashed
      ? `<g fill="none" stroke="${ink}" stroke-opacity="0.34" stroke-width="${(0.18 * boldness).toFixed(2)}" stroke-dasharray="${(1.4 * boldness).toFixed(1)} ${(1.4 * boldness).toFixed(1)}">${t.dashed}</g>`
      : "")
  );
}

// ---------------------------------------------------------------------------
// Shapes -> markup. Shared by the live NodeView and the exporter for the same
// reason PITCH_LINES is: what you export has to be what you saw.
// ---------------------------------------------------------------------------

/** Everything below is in METRES. A disc is 2.4 m across, an arrowhead is 2 m
 *  long and a line is 22 cm thick — sizes that stay in proportion whether the
 *  board is showing the whole pitch or just the six-yard box. */
const DISC_R = 1.2;
const BALL_R = 0.7;
const HEAD = 2.0;
const STROKE = 0.32;

const n2 = (n: number) => Number(n.toFixed(2));

/** A shape's points, converted from stored percent to drawn metres. */
const inM = (s: Shape) => s.pts.map((v, i) => (i % 2 === 0 ? mx(v) : my(v)));

/** A dribble is drawn as a sine wave along the line. Six half-periods is enough
 *  to read as "carrying it" at any length a coach draws, and few enough that
 *  the path stays a handful of points rather than a curve library. */
function wavy(x0: number, y0: number, x1: number, y1: number): string {
  const dx = x1 - x0;
  const dy = y1 - y0;
  const len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len;
  const ny = dx / len;
  const amp = Math.min(1.4, len / 10);
  const steps = Math.max(8, Math.round(len));
  const pts: string[] = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const off = Math.sin(t * Math.PI * 6) * amp;
    pts.push(`${n2(x0 + dx * t + nx * off)},${n2(y0 + dy * t + ny * off)}`);
  }
  return `M ${pts.join(" L ")}`;
}

function head(x0: number, y0: number, x1: number, y1: number, stroke: string): string {
  const a = Math.atan2(y1 - y0, x1 - x0);
  const wing = (s: number) =>
    `${n2(x1 - HEAD * Math.cos(a - s))},${n2(y1 - HEAD * Math.sin(a - s))}`;
  return `<polygon points="${n2(x1)},${n2(y1)} ${wing(0.4)} ${wing(-0.4)}" fill="${stroke}"/>`;
}

const esc = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/** A flat [x,y,x,y…] list as an SVG `points` attribute. */
const pairs = (p: number[]) =>
  p.reduce<string[]>((out, v, i) => (i % 2 ? out : [...out, `${n2(v)},${n2(p[i + 1])}`]), []).join(" ");

const polyline = (p: number[]) =>
  `M ${p.reduce<string[]>((acc, v, i) => {
    if (i % 2 === 0) acc.push(`${n2(v)},${n2(p[i + 1])}`);
    return acc;
  }, []).join(" L ")}`;

/** A rotated view turns the whole content group by -90°, which would take the
 *  writing with it. Every piece of TEXT counter-rotates about its own anchor so
 *  a shirt number reads the right way up in the set-piece view — the one crop
 *  where the numbers matter most. Lines and discs are rotationally symmetric or
 *  meant to turn, so only text takes this. */
const upright = (rot: boolean, x: number, y: number) =>
  rot ? ` transform="rotate(90 ${n2(x)} ${n2(y)})"` : "";

/** One shape as markup. Every interpolated value is either a clamped number or
 *  a lookup into INK, except `label`, which is escaped. */
export function shapeSvg(s: Shape, rot = false): string {
  const c = pick(INK, s.color, INK.white);
  const p = inM(s);
  if (s.kind === "text") {
    return (
      `<text x="${n2(p[0])}" y="${n2(p[1])}" font-size="2.6" fill="${c}" ` +
      `font-family="system-ui, sans-serif" font-weight="600" dominant-baseline="middle" ` +
      `paint-order="stroke" stroke="#000000aa" stroke-width="0.7"` +
      `${upright(rot, p[0], p[1])}>${esc(s.label ?? "")}</text>`
    );
  }
  if (s.kind === "zone") {
    const x = Math.min(p[0], p[2]);
    const y = Math.min(p[1], p[3]);
    return (
      `<rect x="${n2(x)}" y="${n2(y)}" width="${n2(Math.abs(p[2] - p[0]))}" ` +
      `height="${n2(Math.abs(p[3] - p[1]))}" ` +
      `fill="${c}" fill-opacity="0.16" stroke="${c}" stroke-opacity="0.75" stroke-width="${STROKE}" ` +
      `stroke-dasharray="1.2 0.9"/>`
    );
  }
  if (s.kind === "circle") {
    // Two opposite corners, like a zone — so the same drag builds both and a
    // circle is an oval when you drag it into one, which is what a coach means
    // by "this area".
    return (
      `<ellipse cx="${n2((p[0] + p[2]) / 2)}" cy="${n2((p[1] + p[3]) / 2)}" ` +
      `rx="${n2(Math.abs(p[2] - p[0]) / 2)}" ry="${n2(Math.abs(p[3] - p[1]) / 2)}" ` +
      `fill="${c}" fill-opacity="0.16" stroke="${c}" stroke-opacity="0.75" ` +
      `stroke-width="${STROKE}" stroke-dasharray="1.2 0.9"/>`
    );
  }
  if (s.kind === "poly") {
    return (
      `<polygon points="${pairs(p)}" fill="${c}" fill-opacity="0.16" stroke="${c}" ` +
      `stroke-opacity="0.75" stroke-width="${STROKE}" stroke-linejoin="round" ` +
      `stroke-dasharray="1.2 0.9"/>`
    );
  }
  if (s.kind === "pen") {
    return (
      `<path d="${polyline(p)}" fill="none" stroke="${c}" stroke-width="${STROKE}" ` +
      `stroke-linecap="round" stroke-linejoin="round"/>`
    );
  }
  const [x0, y0] = [p[0], p[1]];
  const [x1, y1] = [p[p.length - 2], p[p.length - 1]];
  const dash = s.kind === "run" ? ` stroke-dasharray="1.6 1.2"` : "";
  const d = s.kind === "dribble" ? wavy(x0, y0, x1, y1) : `M ${n2(x0)},${n2(y0)} L ${n2(x1)},${n2(y1)}`;
  return (
    `<path d="${d}" fill="none" stroke="${c}" stroke-width="${STROKE}" stroke-linecap="round"${dash}/>` +
    head(x0, y0, x1, y1, c)
  );
}

/** The same geometry as shapeSvg, drawn fat and transparent. This is what a
 *  pointer actually hits: a 32 cm line is unclickable, and giving every stroke
 *  a wide invisible twin is a great deal less code than hit-testing points
 *  against segments by hand. */
export function shapeHitSvg(s: Shape): string {
  const p = inM(s);
  const fat = `fill="none" stroke="transparent" stroke-width="2.6" stroke-linecap="round" pointer-events="stroke"`;
  if (s.kind === "text") {
    return `<rect x="${n2(p[0] - 1)}" y="${n2(p[1] - 1.8)}" width="${n2(2 + (s.label?.length ?? 1) * 1.5)}" height="3.6" fill="transparent" pointer-events="all"/>`;
  }
  if (s.kind === "zone") {
    const x = Math.min(p[0], p[2]);
    const y = Math.min(p[1], p[3]);
    return `<rect x="${n2(x)}" y="${n2(y)}" width="${n2(Math.abs(p[2] - p[0]))}" height="${n2(Math.abs(p[3] - p[1]))}" fill="none" stroke="transparent" stroke-width="2.6" pointer-events="stroke"/>`;
  }
  if (s.kind === "circle") {
    return (
      `<ellipse cx="${n2((p[0] + p[2]) / 2)}" cy="${n2((p[1] + p[3]) / 2)}" ` +
      `rx="${n2(Math.abs(p[2] - p[0]) / 2)}" ry="${n2(Math.abs(p[3] - p[1]) / 2)}" ` +
      `fill="none" stroke="transparent" stroke-width="2.6" pointer-events="stroke"/>`
    );
  }
  if (s.kind === "poly") {
    return `<polygon points="${pairs(p)}" fill="none" stroke="transparent" stroke-width="2.6" pointer-events="stroke"/>`;
  }
  if (s.kind === "pen") return `<path d="${polyline(p)}" ${fat}/>`;
  return `<path d="M ${n2(p[0])},${n2(p[1])} L ${n2(p[p.length - 2])},${n2(p[p.length - 1])}" ${fat}/>`;
}

export function magnetSvg(
  m: Magnet,
  rot = false,
  /** How the two sides are dressed. Absent means the defaults, which is what a
   *  board written before kits existed carries. */
  kits: Record<"home" | "away", Kit> | undefined = DEFAULT_KITS,
): string {
  const cx = n2(mx(m.x));
  const cy = n2(my(m.y));

  if (m.side === "gear") {
    // Drawn around its own origin and moved into place, so each entry in GEAR
    // is a shape rather than a shape plus arithmetic. Stood upright in a
    // vertical board for the same reason a label is: a cone on its side is not
    // a cone.
    const g = GEAR[m.gear ?? ""] ?? GEAR.cone;
    return `<g transform="translate(${cx} ${cy})${rot ? " rotate(90)" : ""}">${g.draw(
      FIXED_SHADE.gear,
    )}</g>`;
  }

  const kit = m.side === "home" || m.side === "away" ? kits[m.side] : undefined;
  const key = m.side === "ball" ? "ball" : (kit?.colour ?? DEFAULT_KITS.home.colour);
  const key2 = kit?.colour2 ?? DEFAULT_KITS.home.colour2;
  const base = m.side === "ball" ? FIXED_SHADE.ball : kitColour(kit, 1);
  // Clamped here rather than trusted from the file: a size read out of a
  // document is a number an agent wrote.
  const r =
    m.side === "ball" ? BALL_R : DISC_R * Math.min(1.8, Math.max(0.6, Number(kit?.size) || 1));
  const turn = upright(rot, mx(m.x), my(m.y));

  const number =
    m.side === "ball" || !m.label || (kit && !kit.number)
      ? ""
      : `<text x="${cx}" y="${cy}" font-size="${n2(r * 1.25)}" text-anchor="middle" ` +
        `dominant-baseline="central" fill="${kit?.numberDark === false ? "#ffffff" : kit ? "#16181c" : "#5e1420"}" ` +
        `font-family="system-ui, sans-serif" font-weight="700"${turn}>${esc(m.label)}</text>`;

  // BESIDE the disc, never on it, and with a dark halo so it survives a pale
  // pitch — the same trick a text mark uses.
  const name =
    kit && kit.name !== "off" && m.name
      ? `<text x="${cx}" y="${n2(my(m.y) + (kit.name === "above" ? -(r + 0.9) : r + 1.9))}" ` +
        `font-size="1.5" text-anchor="middle" fill="#ffffff" font-family="system-ui, sans-serif" ` +
        `font-weight="600" paint-order="stroke" stroke="#000000aa" stroke-width="0.55"${turn}>` +
        `${esc(m.name.slice(0, 18))}</text>`
      : "";

  // HOLLOW. For the side a diagram is not about: an outline reads as furniture
  // and stops eleven black discs shouting over the four that matter.
  if (kit?.outline) {
    return (
      `<g${turn}>` +
      `<circle cx="${cx}" cy="${cy}" r="${n2(r)}" fill="#ffffff" fill-opacity="0.14" ` +
      `stroke="${base}" stroke-width="${n2(r * 0.2)}"/></g>` +
      number +
      name
    );
  }

  // The right half in the second colour. A path rather than a clip path, so a
  // two-colour disc costs one more element and no defs.
  const half =
    kit?.style === "halves"
      ? `<path d="M ${cx},${n2(my(m.y) - r)} A ${n2(r)},${n2(r)} 0 0 1 ${cx},${n2(my(m.y) + r)} Z" ` +
        `fill="url(#ftDisc-${key2})"/>`
      : "";

  return (
    `<g${turn} filter="url(#ftDiscShadow)">` +
    `<circle cx="${cx}" cy="${cy}" r="${n2(r)}" fill="url(#ftDisc-${key})" ` +
    `stroke="${mix(base, 0, 0.45)}" stroke-opacity="0.55" stroke-width="0.1"/>` +
    half +
    // The lit edge of a curved thing rather than a reflection: an ellipse
    // across the top, fading out before it reaches the middle.
    `<ellipse cx="${cx}" cy="${n2(my(m.y) - r * 0.34)}" rx="${n2(r * 0.66)}" ry="${n2(r * 0.36)}" ` +
    `fill="#ffffff" fill-opacity="0.26"/>` +
    `</g>` +
    number +
    name
  );
}

/** The board as standalone SVG markup — what a reader outside the editor sees.
 *  Kept as a string rather than JSX because renderHTML runs headless.
 *
 *  This string is assigned with innerHTML, so every value interpolated into it
 *  has to be one that cannot carry markup. Coordinates are numbers clamped by
 *  parseBoard, `side` and `kind` are narrowed to their literals, colours come
 *  from fixed tables keyed by those literals, and the two pieces of free text a
 *  user types — a magnet's label and a text shape's — go through esc(). A board
 *  is also the one part of a saved doc an agent can write, which is exactly why
 *  none of it is trusted here. */
export function pitchSvg(board: Board): string {
  const v = viewOf(board.view);
  const vert = isVertical(board.orient);
  const { box, aspect } = viewBox(v, vert);
  // ORDER IS THE POINT. Grass, then zones, then the arrows, then the discs on
  // top of everything. An arrow drawn over a player is a line crossing his
  // face - and it is exactly the arrows that matter most (a pass INTO a man, a
  // run OFF him) that end on a disc, so the whole vocabulary read wrong.
  // Zones stay underneath: a zone is a region of grass, not a mark on it.
  const body =
    surfaceSvg(board) +
    board.shapes.filter((s) => s.kind === "zone").map((s) => shapeSvg(s, vert)).join("") +
    board.shapes.filter((s) => s.kind !== "zone").map((s) => shapeSvg(s, vert)).join("") +
    board.magnets.map((m) => magnetSvg(m, vert, board.kits)).join("");
  const inner = vert ? `<g transform="rotate(-90 52.5 34)">${body}</g>` : body;
  return (
    `<svg viewBox="${box}" preserveAspectRatio="xMidYMid meet" width="100%" ` +
    // Square corners. A pitch is a rectangle with corner flags on it, and a
    // rounded one reads as a card with a picture of grass on it.
    `style="display:block;aspect-ratio:${aspect.toFixed(4)}">${inner}</svg>`
  );
}

// ---------------------------------------------------------------------------

/** Read a board back out of a stored element, tolerating anything. A board that
 *  cannot be parsed becomes the default eleven rather than an error.
 *
 *  AN EMPTY ARRAY IS NOT UNREADABLE. `[]` is a coach who deleted all 23 discs
 *  to draw a set piece with shapes alone, and it used to reload as a full
 *  4-4-2 sitting on top of their drawing — their deletion was literally
 *  unrepresentable. Absent falls back; present-and-empty is honoured. */
export function parseMagnets(value: string | null): Magnet[] {
  if (value === null || value === undefined || value === "") return DEFAULT_MAGNETS;
  try {
    const raw: unknown = JSON.parse(value);
    if (!Array.isArray(raw)) return DEFAULT_MAGNETS;
    if (raw.length === 0) return [];
    const out = raw
      .filter((m): m is Record<string, unknown> => !!m && typeof m === "object")
      .filter((m) => typeof m.x === "number" && typeof m.y === "number")
      .map((m) => ({
        x: clamp(m.x as number),
        y: clamp(m.y as number),
        label: typeof m.label === "string" ? m.label.slice(0, 3) : "",
        name: typeof m.name === "string" ? m.name.slice(0, 24) : undefined,
        side: SIDES.includes(m.side as Magnet["side"]) ? (m.side as Magnet["side"]) : ("home" as const),
        // Narrowed to a key of GEAR the same way every other string on the
        // board is, because this one reaches innerHTML through magnetSvg.
        gear:
          typeof m.gear === "string" && Object.hasOwn(GEAR, m.gear) ? m.gear : undefined,
      }));
    // Every entry was junk, which is a parse failure and not a deletion.
    return out.length ? out : DEFAULT_MAGNETS;
  } catch {
    return DEFAULT_MAGNETS;
  }
}

const SIDES: Magnet["side"][] = ["home", "away", "ball", "gear"];

/** One side's kit, out of whatever the file happens to hold. Every field falls
 *  back to the default on its own, so a board saved before a field existed —
 *  or one an agent wrote half of — comes back dressed rather than blank. */
function parseKit(raw: unknown, fallback: Kit): Kit {
  const k = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const colour = (v: unknown, d: string) =>
    typeof v === "string" && Object.hasOwn(KIT_COLOURS, v) ? v : d;
  const size = Number(k.size);
  return {
    colour: colour(k.colour, fallback.colour),
    colour2: colour(k.colour2, fallback.colour2),
    style: k.style === "halves" ? "halves" : "solid",
    outline: k.outline === true,
    size: Number.isFinite(size) ? Math.min(1.8, Math.max(0.6, size)) : fallback.size,
    number: k.number !== false,
    numberDark: typeof k.numberDark === "boolean" ? k.numberDark : fallback.numberDark,
    name: k.name === "above" || k.name === "under" ? k.name : "off",
  };
}

export function parseKits(value: string | null): Record<"home" | "away", Kit> {
  try {
    const raw = value ? (JSON.parse(value) as Record<string, unknown>) : {};
    return {
      home: parseKit(raw?.home, DEFAULT_KITS.home),
      away: parseKit(raw?.away, DEFAULT_KITS.away),
    };
  } catch {
    return DEFAULT_KITS;
  }
}

const KINDS: ShapeKind[] = ["pass", "run", "dribble", "pen", "zone", "circle", "poly", "text"];

/** How many points each kind is allowed to keep. A pen stroke is capped
 *  because a long drag at pointer-event rate is thousands of samples, and a
 *  document is not a place to store a thousand near-identical coordinates. */
const MAX_PTS = 400;

export function parseShapes(value: string | null): Shape[] {
  if (!value) return [];
  try {
    const raw: unknown = JSON.parse(value);
    if (!Array.isArray(raw)) return [];
    return raw
      .filter((s): s is Record<string, unknown> => !!s && typeof s === "object")
      .filter((s) => KINDS.includes(s.kind as ShapeKind) && Array.isArray(s.pts))
      .map((s) => ({
        kind: s.kind as ShapeKind,
        pts: (s.pts as unknown[])
          .filter((n): n is number => typeof n === "number" && Number.isFinite(n))
          .slice(0, MAX_PTS)
          .map(clamp),
        color: typeof s.color === "string" && Object.hasOwn(INK, s.color) ? s.color : "white",
        label: typeof s.label === "string" ? s.label.slice(0, 60) : undefined,
      }))
      .filter((s) => s.pts.length >= (s.kind === "text" ? 2 : s.kind === "poly" ? 6 : 4));
  } catch {
    return [];
  }
}

/** One narrowing rule for every string attribute on the board: it is a key of
 *  the table that owns it, or it is the default. These strings are interpolated
 *  into markup that is assigned with innerHTML, so "unknown means default" is
 *  the only safe reading — never "unknown means pass it through".
 *
 *  `Object.hasOwn` and NOT the `in` operator, which walks the prototype chain:
 *  `"constructor" in GRASS` is true, so `in` would accept `data-grass=
 *  "constructor"` and hand the render `Object` instead of a colour. A board is
 *  the one part of a saved document an agent can write, so the check has to be
 *  the one that only sees the keys actually in the table. */
const key = (table: Record<string, unknown>, fallback: string) => (value: string | null) =>
  value && Object.hasOwn(table, value) ? value : fallback;

/** Read a fixed table by a string that came from a document. Same reason. */
export const pick = <T,>(table: Record<string, T>, k: string, fallback: T): T =>
  Object.hasOwn(table, k) ? table[k] : fallback;

export const parseView = key(VIEWS, "full");
export const parseLines = key(LINE_TEMPLATES, "standard");
export const parseGrass = key(GRASS, "green");
export const parseOrient = (value: string | null): string => (value === "v" ? "v" : "h");

// ---------------------------------------------------------------------------

/** The board as this match was actually played: every tracked player dropped
 *  on their average position.
 *
 *  This is the import the notepad exists for — a coach draws the shape they
 *  meant, then puts the measured one beside it. `avg_position` is metres
 *  ([along the pitch, from the near touchline]) and the board is percentages,
 *  so it divides by the pitch the report was built against rather than a
 *  guessed 105x68.
 *
 *  Tier note, and it is the whole reason this is a separate command rather
 *  than the default board: `players.avg_position` is PROVISIONAL. A mean
 *  position is only as good as the track under it, and a track that swapped
 *  players has a mean belonging to neither. It is a picture of the match, not
 *  a measurement of a player, and the caller labels it that way.
 */
/** The thirds, shaded by where the play actually was.
 *
 *  `territory` is the share of all player-position samples that fell in each
 *  third. It is NOT possession — nothing here sees the ball — so the label says
 *  "of positions", and the tier behind it is `rel`: it is a share, and every
 *  player went through the same camera and calibration, so the biases largely
 *  cancel out of a ratio even where the absolute metres do not.
 *
 *  Built out of the same zone and text shapes a coach draws by hand, so it can
 *  be nudged, recoloured and deleted like anything else on the board rather
 *  than being a special layer with its own rules.
 */
export function territoryShapes(t: {
  defensive_third: number;
  middle_third: number;
  attacking_third: number;
}): Shape[] {
  const bands: [string, number, number, number][] = [
    // By pitch x, the same thirds for both sides: left / centre / right, whatever the keys say.
    ["Left", t.defensive_third, 0, 100 / 3],
    ["Centre", t.middle_third, 100 / 3, 200 / 3],
    ["Right", t.attacking_third, 200 / 3, 100],
  ];
  const top = Math.max(...bands.map(([, v]) => v)) || 1;
  return bands.flatMap(([name, share, x0, x1]): Shape[] => [
    // Shade by rank, not by an opacity computed off the share: three
    // near-identical greys read as "no finding", which is exactly what a
    // three-way split near 33% is.
    { kind: "zone", pts: [x0, 0, x1, 100], color: share === top ? "yellow" : "white" },
    // Short, because a label long enough to carry the caveat runs off the end
    // of its own band and into the next one. The caveat goes in the caption —
    // see TERRITORY_CAPTION, which the board fills in with this.
    { kind: "text", pts: [x0 + 12, 6], color: "white", label: `${name} ${share.toFixed(0)}%` },
  ]);
}

/** What the shaded thirds actually mean, for the caption under the board. The
 *  distinction is the whole honesty of the graphic: the pipeline follows bodies
 *  and its ball rows exist but are pixel centres and unused, so this is where the
 *  PLAYERS were, not who had it. */
export const TERRITORY_CAPTION =
  "Share of tracked positions per third — where the players were, not possession";

/** The same distinction as a NOTE rather than a caption, because it has to be
 *  said whether or not the coach has titled the board. It used to be written
 *  into the caption and only when the caption was still empty, so a board
 *  titled "Second-half press" got shaded thirds and no caveat at all. */
export const TERRITORY_NOTE =
  "Shaded by share of tracked POSITIONS per third — where the players were, not who had the ball. The pipeline follows bodies and never sees it.";

export const AVG_POSITION_NOTE =
  "One disc per tracked player at their mean position. Sides swap at half time, so read the spread and the pairings, never the position itself.";

export function fromAveragePositions(
  players: {
    team: string;
    jersey: number | null;
    id: number;
    avg_position?: [number, number] | null;
  }[],
  pitch?: { length: number; width: number },
): Magnet[] {
  const L = pitch?.length || 105;
  const W = pitch?.width || 68;
  const out = players
    .filter((p) => Array.isArray(p.avg_position) && p.team !== "unknown")
    .map((p) => {
      const [mx, my] = p.avg_position as [number, number];
      return {
        x: clamp((mx / L) * 100),
        y: clamp((my / W) * 100),
        // No names anywhere near this: a shirt number is the furthest the
        // pipeline can honestly go, and the track id when even that is absent.
        label: p.jersey != null ? String(p.jersey) : String(p.id),
        side: p.team === "away" ? ("away" as const) : ("home" as const),
      };
    });
  // EMPTY, not a 4-4-2. This used to fall back to DEFAULT_MAGNETS whenever no
  // track carried an average position — so the button filled the board with an
  // invented textbook shape under a tooltip promising measured ones, and
  // nothing downstream said otherwise. The caller refuses instead.
  return out;
}
