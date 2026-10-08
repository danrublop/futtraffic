import { describe, expect, it } from "vitest";

import {
  DEFAULT_MAGNETS,
  EMPTY_BOARD,
  FORMATIONS,
  GRASS,
  VIEWS,
  fromAveragePositions,
  fromFormation,
  DEFAULT_KITS,
  parseGrass,
  parseKits,
  parseLines,
  parseMagnets,
  parseOrient,
  parseShapes,
  parseView,
  pitchSvg,
  sideFormation,
  surfaceSvg,
  TERRITORY_CAPTION,
  territoryShapes,
  toPitch,
  viewBox,
  viewOf,
  type Board,
  type Kit,
  type Shape,
} from "./pitch-data";

describe("fromFormation", () => {
  it("mirrors the away side so the teams face each other", () => {
    const m = fromFormation("4-4-2");
    const home = m.filter((x) => x.side === "home");
    const away = m.filter((x) => x.side === "away");
    expect(home).toHaveLength(11);
    expect(away).toHaveLength(11);
    expect(m.filter((x) => x.side === "ball")).toHaveLength(1);
    // Keepers on opposite goal lines, same distance from each.
    expect(home[0].x).toBeCloseTo(100 - away[0].x);
    expect(home[0].x).toBeLessThan(50);
    expect(away[0].x).toBeGreaterThan(50);
  });

  it("falls back to 4-4-2 for a shape nobody defined", () => {
    expect(fromFormation("9-0-1")).toEqual(fromFormation("4-4-2"));
  });

  it("every formation fields eleven", () => {
    for (const name of Object.keys(FORMATIONS)) {
      expect(FORMATIONS[name], name).toHaveLength(11);
    }
  });
});

describe("fromAveragePositions", () => {
  const p = (
    id: number,
    team: string,
    avg: [number, number] | null,
    jersey: number | null = null,
  ) => ({ id, team, jersey, avg_position: avg });

  it("scales metres to percentages against the report's own pitch", () => {
    const m = fromAveragePositions(
      [p(1, "home", [52.5, 34]), p(2, "away", [84, 17])],
      { length: 105, width: 68 },
    );
    expect(m[0].x).toBeCloseTo(50);
    expect(m[0].y).toBeCloseTo(50);
    expect(m[1].x).toBeCloseTo(80);
    expect(m[1].y).toBeCloseTo(25);
  });

  it("uses a full-size pitch when the report does not say", () => {
    expect(fromAveragePositions([p(1, "home", [52.5, 34])])[0].x).toBeCloseTo(50);
  });

  it("labels by shirt number, and by track id when there is none", () => {
    const m = fromAveragePositions([p(7, "home", [10, 10], 9), p(8, "away", [90, 10])]);
    expect(m[0].label).toBe("9");
    expect(m[1].label).toBe("8");
  });

  it("drops players with no position or no side rather than stacking them at 0,0", () => {
    const m = fromAveragePositions([
      p(1, "home", [10, 10]),
      p(2, "home", null),
      p(3, "unknown", [20, 20]),
    ]);
    expect(m).toHaveLength(1);
  });

  it("returns nothing when nothing was tracked, rather than an invented shape", () => {
    // It used to fall back to DEFAULT_MAGNETS, so the "Average positions"
    // button filled the board with a textbook 4-4-2 that no camera had seen.
    // The caller now refuses instead.
    expect(fromAveragePositions([], { length: 105, width: 68 })).toEqual([]);
  });

  it("clamps a position outside the pitch", () => {
    const m = fromAveragePositions([p(1, "home", [200, -40])], { length: 105, width: 68 });
    expect(m[0].x).toBe(100);
    expect(m[0].y).toBe(0);
  });
});

describe("parseMagnets", () => {
  it("survives junk", () => {
    for (const bad of [null, "", "{", '"nope"', "[null,3]", '[{"x":"a","y":1}]']) {
      expect(parseMagnets(bad)).toEqual(DEFAULT_MAGNETS);
    }
  });

  it("honours an empty board, because emptying one is a thing a coach does", () => {
    // `[]` used to fall in with the junk and reload as a full 4-4-2 sitting on
    // top of whatever had been drawn, so deleting all 23 discs to make room
    // for a set piece was not a state the document could hold.
    expect(parseMagnets("[]")).toEqual([]);
  });

  it("clamps and narrows what it does accept", () => {
    const m = parseMagnets(
      JSON.stringify([{ x: -10, y: 400, label: "longlabel", side: "referee" }]),
    );
    expect(m[0]).toEqual({ x: 0, y: 100, label: "lon", side: "home" });
  });
});

describe("sideFormation", () => {
  it("replaces one side and leaves the other where the coach put it", () => {
    const start = fromFormation("4-4-2");
    const moved = start.map((m) =>
      m.side === "away" ? { ...m, x: 70, y: 12 } : m,
    );
    const next = sideFormation("4-3-3", "home", moved);
    expect(next.filter((m) => m.side === "home")).toHaveLength(11);
    // Every away disc is exactly where it was — including the hand-placed ones.
    expect(next.filter((m) => m.side === "away").every((m) => m.x === 70 && m.y === 12)).toBe(true);
    // ...and the ball is not duplicated or dropped.
    expect(next.filter((m) => m.side === "ball")).toHaveLength(1);
  });
});

describe("views", () => {
  it("a point round-trips through the crop it was measured in, either way up", () => {
    for (const name of Object.keys(VIEWS)) {
      for (const vertical of [false, true]) {
        const v = viewOf(name);
        // The centre of the drawn box is the centre of the crop, in pitch terms
        // — and standing the pitch up must not move it.
        const mid = toPitch(v, vertical, 0.5, 0.5);
        const cx = ((v.x + v.w / 2) / 105) * 100;
        const cy = ((v.y + v.h / 2) / 68) * 100;
        expect(mid.x, `${name}/${vertical}`).toBeCloseTo(Math.min(100, Math.max(0, cx)), 4);
        expect(mid.y, `${name}/${vertical}`).toBeCloseTo(Math.min(100, Math.max(0, cy)), 4);
      }
    }
  });

  it("upright puts the attacking goal at the top", () => {
    const box = viewOf("box");
    // Top of the drawn box is further up the pitch than the bottom of it...
    expect(toPitch(box, true, 0.5, 0).x).toBeGreaterThan(toPitch(box, true, 0.5, 1).x);
    // ...and across, the same drag is left-to-right instead.
    expect(toPitch(box, false, 1, 0.5).x).toBeGreaterThan(toPitch(box, false, 0, 0.5).x);
  });

  it("aspect matches the crop's real metres, so nothing is drawn stretched", () => {
    expect(viewBox(viewOf("full"), false).aspect).toBeCloseTo(115 / 78, 3);
    // Upright: the pitch WIDTH becomes the horizontal axis.
    expect(viewBox(viewOf("full"), true).aspect).toBeCloseTo(78 / 115, 3);
    expect(viewBox(viewOf("box"), true).aspect).toBeCloseTo(62 / 31.5, 3);
  });

  it("an unknown name falls back rather than reaching the markup", () => {
    expect(parseView("nope")).toBe("full");
    expect(parseView(null)).toBe("full");
    expect(viewOf("nope")).toBe(VIEWS.full);
    expect(parseLines('"><script>')).toBe("standard");
    expect(parseGrass("url(#x)")).toBe("green");
    expect(parseOrient("sideways")).toBe("h");
    expect(parseOrient("v")).toBe("v");
  });
});

/** A saved board is the one part of a document an agent can author, and every
 *  string on it indexes a fixed table on the way to innerHTML. `in` and plain
 *  `[]` both walk the prototype chain — `"constructor" in GRASS` is true — so a
 *  hand-written `data-grass="constructor"` used to hand the renderer `Object`
 *  instead of a colour and put `undefined` (or worse, a NaN viewBox) on screen. */
describe("prototype keys are not board styles", () => {
  const POISON = ["constructor", "__proto__", "toString", "hasOwnProperty", "valueOf"];

  it("never accepts one as a view, line template, grass or pen", () => {
    for (const k of POISON) {
      expect(parseView(k), k).toBe("full");
      expect(parseLines(k), k).toBe("standard");
      expect(parseGrass(k), k).toBe("green");
      expect(parseShapes(JSON.stringify([{ kind: "pass", pts: [1, 2, 3, 4], color: k }]))[0].color)
        .toBe("white");
    }
  });

  it("still draws a real pitch when a document asks for one", () => {
    for (const k of POISON) {
      const svg = pitchSvg({ ...EMPTY_BOARD, view: k, lines: k, grass: k });
      expect(svg, k).toContain(GRASS.green.fill);
      expect(svg, k).not.toContain("undefined");
      expect(svg, k).not.toContain("NaN");
      expect(svg, k).not.toContain("native code");
    }
  });
});

describe("surfaceSvg", () => {
  const surface = (b: Partial<Board>) => surfaceSvg({ ...EMPTY_BOARD, ...b });

  it("paints colour, bands and markings, and nothing else", () => {
    const s = surface({});
    expect(s).toContain(GRASS.green.fill);
    // Flat by default now — a report's pitch has no mown bands.
    expect(s).not.toContain(GRASS.green.band);
    expect(surfaceSvg({ ...EMPTY_BOARD, stripes: true })).toContain(GRASS.green.band);
    expect(s).toContain('r="9.15"'); // the centre circle
  });

  it("drops the bands when the stripes are off", () => {
    expect(surface({ stripes: false })).not.toContain(GRASS.green.band);
  });

  it("a guide template adds a dashed overlay, never more solid paint", () => {
    expect(surface({ lines: "standard" })).not.toContain("stroke-dasharray");
    const grid = surface({ lines: "grid" });
    expect(grid).toContain("stroke-dasharray");
    expect(grid).toContain('r="9.15"'); // ...and keeps the real markings
  });

  it("a blank pitch has no centre circle to draw over", () => {
    expect(surface({ lines: "blank" })).not.toContain('r="9.15"');
  });

  it("switches to dark ink on a light surface, so the lines stay visible", () => {
    expect(surface({ grass: "chalk" })).toContain("#33383d");
    expect(surface({ grass: "green" })).not.toContain("#33383d");
  });
});

describe("territoryShapes", () => {
  const t = { defensive_third: 25.9, middle_third: 48.1, attacking_third: 26.0 };

  it("covers the pitch end to end with three bands and three labels", () => {
    const s = territoryShapes(t);
    const zones = s.filter((x) => x.kind === "zone");
    expect(zones).toHaveLength(3);
    expect(s.filter((x) => x.kind === "text")).toHaveLength(3);
    expect(zones[0].pts[0]).toBe(0);
    expect(zones[2].pts[2]).toBe(100);
    // No gap between the bands.
    expect(zones[0].pts[2]).toBeCloseTo(zones[1].pts[0], 6);
  });

  it("marks the busiest third and calls the number a share of POSITIONS", () => {
    const s = territoryShapes(t);
    expect(s.filter((x) => x.kind === "zone").map((z) => z.color)).toEqual([
      "white",
      "yellow",
      "white",
    ]);
    expect(s.find((x) => x.kind === "text")!.label).toBe("Left 26%");
    // The pipeline does not see the ball, so the caption must say what this is
    // NOT before anybody reads a possession figure off a green pitch.
    expect(TERRITORY_CAPTION).toMatch(/not possession/);
  });
});

describe("parseShapes", () => {
  it("survives junk", () => {
    for (const bad of [null, "", "{", '"nope"', "[null]", '[{"kind":"pass"}]']) {
      expect(parseShapes(bad)).toEqual([]);
    }
  });

  it("drops an unknown kind and an unknown colour rather than passing it through", () => {
    const s = parseShapes(
      JSON.stringify([
        { kind: "laser", pts: [1, 2, 3, 4], color: "red" },
        { kind: "pass", pts: [1, 2, 3, 4], color: "url(#x)" },
      ]),
    );
    expect(s).toHaveLength(1);
    expect(s[0].color).toBe("white");
  });

  it("clamps coordinates and refuses a shape with too few points", () => {
    expect(parseShapes(JSON.stringify([{ kind: "pass", pts: [-5, 500], color: "red" }]))).toEqual([]);
    const s = parseShapes(JSON.stringify([{ kind: "pass", pts: [-5, 500, 20, 20], color: "red" }]));
    expect(s[0].pts).toEqual([0, 100, 20, 20]);
  });

  it("caps a freehand stroke so one drag cannot bloat the document", () => {
    const long = Array.from({ length: 2000 }, (_, i) => i % 100);
    const s = parseShapes(JSON.stringify([{ kind: "pen", pts: long, color: "white" }]));
    expect(s[0].pts.length).toBeLessThanOrEqual(400);
  });
});

describe("pitchSvg", () => {
  const board = (magnets = DEFAULT_MAGNETS, shapes: Shape[] = []) => ({
    ...EMPTY_BOARD,
    magnets,
    shapes,
  });

  it("escapes a label, because a magnet is something an agent can write", () => {
    const svg = pitchSvg(board([{ x: 1, y: 1, label: '"><script>', side: "home" }]));
    expect(svg).not.toContain("<script>");
    expect(svg).toContain("&quot;&gt;&lt;script&gt;");
  });

  it("escapes a pitch label too", () => {
    const svg = pitchSvg(
      board(DEFAULT_MAGNETS, [
        { kind: "text", pts: [10, 10], color: "white", label: '</text><script>x</script>' },
      ]),
    );
    expect(svg).not.toContain("<script>");
  });

  it("draws one disc per magnet on a self-contained svg", () => {
    const svg = pitchSvg(board());
    expect(svg.startsWith("<svg")).toBe(true);
    // Four circles of markings in PITCH_LINES (centre circle, centre spot, two
    // penalty spots), then one per magnet.
    expect((svg.match(/<circle/g) ?? []).length).toBe(DEFAULT_MAGNETS.length + 4);
  });

  it("gives a run a dash and a pass none, so the two cannot be confused", () => {
    const pass = pitchSvg(board([], [{ kind: "pass", pts: [10, 10, 40, 40], color: "white" }]));
    const run = pitchSvg(board([], [{ kind: "run", pts: [10, 10, 40, 40], color: "white" }]));
    expect(pass).not.toContain("stroke-dasharray");
    expect(run).toContain("stroke-dasharray");
    // Both carry an arrowhead: the direction is the information.
    expect(pass).toContain("<polygon");
    expect(run).toContain("<polygon");
  });

  it("keeps writing upright on an upright pitch, so a shirt number is readable", () => {
    const b = { ...board([{ x: 90, y: 50, label: "9", side: "home" }]), view: "box", orient: "v" };
    expect(pitchSvg(b)).toContain('transform="rotate(90');
    // ...and does not counter-rotate anything on a pitch that never turned.
    expect(pitchSvg({ ...b, orient: "h" })).not.toContain('transform="rotate(90');
  });

  it("draws zones and arrows under the discs", () => {
    const svg = pitchSvg(
      board([{ x: 50, y: 50, label: "9", side: "home" }], [
        { kind: "zone", pts: [10, 10, 40, 40], color: "yellow" },
        { kind: "pass", pts: [10, 10, 40, 40], color: "yellow" },
      ]),
    );
    // A zone is a region of grass and goes under everything. An arrow goes
    // under the DISCS too: the arrows that matter most end on a player - a
    // pass into a man, a run off him - and drawn on top they cross his number.
    const zone = svg.indexOf('fill-opacity="0.16"');
    const arrow = svg.indexOf("<polygon");
    const disc = svg.indexOf('r="1.2"');
    expect(zone).toBeGreaterThan(-1);
    expect(arrow).toBeGreaterThan(-1);
    expect(zone).toBeLessThan(arrow);
    expect(arrow).toBeLessThan(disc);
  });
});

/** THE THREE THINGS THE RAIL ADDED, and all three go through a parser that has
 *  to reject junk without deleting the board: a cone, an area you drag out and
 *  an area you click the corners of. Every string below reaches innerHTML. */
describe("equipment and the two new areas", () => {
  it("keeps a cone, and refuses one whose kind is made up", () => {
    const kept = parseMagnets(
      JSON.stringify([
        { x: 10, y: 10, label: "", side: "gear", gear: "goal" },
        { x: 20, y: 20, label: "", side: "gear", gear: '"><script>' },
      ]),
    );
    expect(kept.map((m) => m.side)).toEqual(["gear", "gear"]);
    expect(kept[0].gear).toBe("goal");
    // Unknown falls back to the default cone at draw time rather than being
    // written through to the markup.
    expect(kept[1].gear).toBeUndefined();
    expect(pitchSvg({ ...EMPTY_BOARD, magnets: kept })).not.toContain("<script>");
  });

  it("draws a cone as equipment, not as a disc", () => {
    const svg = pitchSvg({
      ...EMPTY_BOARD,
      magnets: [{ x: 50, y: 50, label: "9", side: "gear", gear: "cone" }],
    });
    // No number on it, and not the disc's circle.
    expect(svg).not.toContain(">9<");
    expect(svg).toContain("translate(");
  });

  it("keeps a circle and a polygon, and drops a polygon with two corners", () => {
    const shapes = parseShapes(
      JSON.stringify([
        { kind: "circle", pts: [10, 10, 30, 40], color: "white" },
        { kind: "poly", pts: [10, 10, 30, 10, 30, 30], color: "red" },
        { kind: "poly", pts: [10, 10, 30, 10], color: "red" },
      ]),
    );
    expect(shapes.map((s) => s.kind)).toEqual(["circle", "poly"]);
    const svg = pitchSvg({ ...EMPTY_BOARD, shapes });
    expect(svg).toContain("<ellipse");
    expect(svg).toContain("<polygon");
  });
});

/** THE KIT: what a side wears, and every field of it is read back out of a
 *  file an agent can write. */
describe("kits", () => {
  const dressed = (patch: Partial<Kit>) => ({
    ...EMPTY_BOARD,
    kits: { ...DEFAULT_KITS, home: { ...DEFAULT_KITS.home, ...patch } },
    magnets: [{ x: 50, y: 50, label: "9", name: "Okafor", side: "home" as const }],
  });

  it("draws the number on the disc and the name beside it", () => {
    const off = pitchSvg(dressed({ name: "off" }));
    expect(off).toContain(">9<");
    expect(off).not.toContain("Okafor");

    const under = pitchSvg(dressed({ name: "under" }));
    expect(under).toContain("Okafor");
    // Below the disc, which sits at y=34 on a full pitch.
    const y = Number(/Okafor/.exec(under) && /y="([\d.]+)"[^>]*>Okafor/.exec(under)?.[1]);
    expect(y).toBeGreaterThan(34);

    const above = pitchSvg(dressed({ name: "above" }));
    const yUp = Number(/y="([\d.]+)"[^>]*>Okafor/.exec(above)?.[1]);
    expect(yUp).toBeLessThan(34);
  });

  it("obeys the switches", () => {
    expect(pitchSvg(dressed({ number: false }))).not.toContain(">9<");
    // Hollow discs are not lit and cast nothing: they reference neither the
    // gradient nor the shadow, both of which stay defined for the other side.
    expect(pitchSvg(dressed({ outline: true }))).not.toContain('filter="url(#ftDiscShadow)"');
    // A second colour only when the style asks for one.
    expect(pitchSvg(dressed({ style: "solid", colour2: "blue" }))).not.toContain("ftDisc-blue");
    expect(pitchSvg(dressed({ style: "halves", colour2: "blue" }))).toContain("ftDisc-blue");
  });

  it("narrows every field, and dresses a board that predates them", () => {
    const k = parseKits(
      JSON.stringify({
        home: { colour: '"><script>', size: 99, style: "diagonal", name: "sideways" },
      }),
    );
    expect(k.home.colour).toBe(DEFAULT_KITS.home.colour);
    expect(k.home.size).toBe(1.8);
    expect(k.home.style).toBe("solid");
    expect(k.home.name).toBe("off");
    // Nothing there at all is the same as the defaults, not a blank kit.
    expect(parseKits(null)).toEqual(DEFAULT_KITS);
    expect(parseKits("not json")).toEqual(DEFAULT_KITS);
  });

  it("only defines the gradients the board actually wears", () => {
    const svg = pitchSvg({
      ...EMPTY_BOARD,
      kits: { home: { ...DEFAULT_KITS.home, colour: "blue" }, away: DEFAULT_KITS.away },
    });
    expect(svg).toContain('id="ftDisc-blue"');
    expect(svg).not.toContain('id="ftDisc-purple"');
  });
});
