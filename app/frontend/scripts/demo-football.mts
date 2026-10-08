/** GENERATED FOOTBALL, FOR FIGURES THAT HAVE NOTHING REAL TO DRAW YET.
 *
 *  A freeze frame and a pass set need two things the store does not hold: a
 *  FRAME SLICE (every player at one instant) and ON-BALL EVENTS. All 581 tracks
 *  across all 11 matches are `position_source = heatmap_sample`, the 600-point
 *  downsample, so there is no instant to slice; and the event table carries
 *  shots, subs, cards and goals only, no pass ever. See identity-scope (not published).
 *
 *  So this invents both, and every payload it produces is stamped
 *  `synthetic: true`, which each renderer prints on the figure as "generated,
 *  not measured". That is not politeness - a former match importer (not in
 *  this repository) wrote matches whose tracking was invented precisely so the
 *  app could be exercised, and it was emphatic that the whole thing rests on the difference being
 *  visible. A generated figure that cannot say so is one that gets quoted.
 *
 *  DETERMINISTIC. A fixed seed, so the gallery renders the same picture twice
 *  and a test can assert on it. Change SEED and every figure moves.
 */

/** mulberry32. Small, seeded, and good enough to place footballers. */
function rng(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const SEED = 20260905;

/** Display space: 0-100 along the pitch, 0-100 across, home attacking right.
 *  Same space as the board - see pitch-data.ts :: Magnet - so a generated
 *  figure and a drawn one can sit on the same page without a converter. */
type Spot = { x: number; y: number; label: string; side: "home" | "away" };

/** 4-3-3 against 4-4-2, as fractions of the pitch, home attacking right. */
const HOME: [number, number, string][] = [
  [6, 50, "1"], [20, 18, "2"], [18, 38, "5"], [18, 62, "6"], [20, 82, "3"],
  [40, 30, "8"], [38, 50, "4"], [40, 70, "10"], [62, 16, "7"], [66, 50, "9"], [62, 84, "11"],
];
const AWAY: [number, number, string][] = [
  [94, 50, "1"], [80, 20, "2"], [82, 40, "4"], [82, 60, "5"], [80, 80, "3"],
  [62, 22, "7"], [60, 44, "6"], [60, 56, "8"], [62, 78, "11"], [42, 42, "9"], [42, 58, "10"],
];

/** ONE INSTANT. `visible` drops players the way a ball-following camera does -
 *  the default keeps everyone, which is the fixed-wide case; pass a smaller
 *  number to see what a figure looks like when the footage hid half a side. */
export function freezeFrame(opts: { t_s?: number; visible?: number; seed?: number } = {}) {
  const r = rng(opts.seed ?? SEED);
  const t_s = opts.t_s ?? 1287;
  // Jitter off the formation, so it reads as a moment rather than a team sheet.
  const all: Spot[] = [
    ...HOME.map(([x, y, label]) => ({
      x: clamp(x + (r() - 0.5) * 9),
      y: clamp(y + (r() - 0.5) * 11),
      label,
      side: "home" as const,
    })),
    ...AWAY.map(([x, y, label]) => ({
      x: clamp(x + (r() - 0.5) * 9),
      y: clamp(y + (r() - 0.5) * 11),
      label,
      side: "away" as const,
    })),
  ];
  const squad = all.length;
  const keep = Math.max(3, Math.min(squad, opts.visible ?? squad));
  // Drop from the ends of the pitch first, which is what a camera following
  // the ball in midfield actually loses.
  const players = [...all]
    .sort((a, b) => Math.abs(a.x - 50) - Math.abs(b.x - 50))
    .slice(0, keep);
  return {
    kind: "freeze" as const,
    t_s,
    players,
    ball: { x: 52 + (r() - 0.5) * 6, y: 48 + (r() - 0.5) * 8 },
    seen: players.length,
    squad,
    home: "Real Arenas",
    away: "Club Deportivo",
    synthetic: true,
  };
}

/** A SET OF PASSES. Each one goes to a team-mate who is plausibly reachable,
 *  so a network drawn from them has the shape a formation would give it rather
 *  than a random graph. ~82% completed, which is about a league average. */
export function passSet(opts: { count?: number; cap?: number; seed?: number } = {}) {
  const r = rng((opts.seed ?? SEED) + 1);
  const want = opts.count ?? 220;
  const cap = opts.cap ?? 300;
  const sides: ["home" | "away", [number, number, string][]][] = [
    ["home", HOME],
    ["away", AWAY],
  ];
  const passes: {
    x0: number; y0: number; x1: number; y1: number;
    side: "home" | "away"; from: string; to: string; ok: boolean;
  }[] = [];

  for (let i = 0; i < want; i++) {
    const [side, shape] = sides[r() < 0.55 ? 0 : 1];
    const a = Math.floor(r() * shape.length);
    // Prefer a nearby team-mate: rank by distance, take one of the closest few.
    const near = shape
      .map((s, j) => ({ j, d: (s[0] - shape[a][0]) ** 2 + (s[1] - shape[a][1]) ** 2 }))
      .filter((s) => s.j !== a)
      .sort((p, q) => p.d - q.d)
      .slice(0, 4);
    const b = near[Math.floor(r() * near.length)].j;
    const ok = r() < 0.82;
    const [x0, y0, from] = shape[a];
    const [x1, y1, to] = shape[b];
    const j = () => (r() - 0.5) * 7;
    passes.push({
      x0: clamp(x0 + j()), y0: clamp(y0 + j()),
      // A lost pass ends short of its target, which is what makes the failed
      // arrows read as failures rather than as a second completed network.
      x1: clamp(x1 + j() + (ok ? 0 : (x0 - x1) * 0.35)),
      y1: clamp(y1 + j() + (ok ? 0 : (y0 - y1) * 0.35)),
      side, from, to, ok,
    });
  }
  return {
    kind: "passes" as const,
    passes: passes.slice(0, cap),
    home: "Real Arenas",
    away: "Club Deportivo",
    dropped: Math.max(0, passes.length - cap),
    synthetic: true,
  };
}

/** Percentile slices for a pizza or a radar. Real-looking, entirely invented. */
export function percentileSlices(seed = SEED + 2) {
  const r = rng(seed);
  const metrics: [string, string, string][] = [
    ["Distance", "km", "Volume"],
    ["Sprints", "", "Intensity"],
    ["Top speed", "km/h", "Intensity"],
    ["Passes", "", "Passing"],
    ["Pass %", "%", "Passing"],
    ["Duels won", "%", "Duels"],
  ];
  return metrics.map(([label, unit, group]) => {
    const pct = Math.round(12 + r() * 86);
    const value = unit === "km" ? 9 + r() * 4 : unit === "km/h" ? 28 + r() * 8 : Math.round(20 + r() * 70);
    return {
      label,
      pct,
      text: `${typeof value === "number" && unit === "km" ? value.toFixed(1) : Math.round(Number(value))}${unit ? ` ${unit}` : ""}`,
      group,
    };
  });
}

const clamp = (v: number) => Math.max(1, Math.min(99, v));
