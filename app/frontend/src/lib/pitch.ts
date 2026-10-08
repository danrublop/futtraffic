/** Pitch geometry and the click-to-calibrate solver.
 *
 *  Ported verbatim from web/calibrate.html — the maths is unchanged, only its
 *  container is. It mirrors model/engine/pitch.py :: Pitch.landmarks(); the names
 *  and their order must stay in lockstep, because Calibration.solve rejects an
 *  export with "unknown landmarks", which is a loud failure rather than a
 *  silent one.
 *
 *  # ponytail: duplicated from Python because the browser cannot import it.
 *  Upgrade path: serve the landmark list from app.py.
 *
 *  Self-check: `demo()` at the bottom. Run it with
 *    cp src/lib/pitch.ts /tmp/pitch.mts && \
 *      node --experimental-strip-types -e "import('/tmp/pitch.mts').then(m=>m.demo())"
 */

export type Pt = [number, number];

export interface PitchDims {
  length: number;
  width: number;
  penalty_depth: number;
  penalty_width: number;
  goal_box_depth: number;
  goal_box_width: number;
  centre_circle_r: number;
}

export const YD = 0.9144;

/** NCAA allows 115–120 yd × 70–75 yd. Assuming 105×68 m puts a 4–6% error on
 *  every distance shipped, which is why the operator types the real numbers. */
export const DEFAULT_PITCH: PitchDims = {
  // The FIFA standard, and what lib/report.ts already falls back to. The venue
  // panel is still where a real venue's numbers go.
  length: 105,
  width: 68,
  penalty_depth: 16.5,
  penalty_width: 40.32,
  goal_box_depth: 5.5,
  goal_box_width: 18.32,
  centre_circle_r: 9.15,
};

// Display text only - ids are what calibration.json stores. y=0 (the ids' "b") is the FAR touchline,
// top of the frame, as in autocal and calibrate-step's placement guesses; y=width ("t") the NEAR one.
export const LANDMARK_LABELS: Record<string, string> = {
  corner_bl: "Far-left corner",
  corner_tl: "Near-left corner",
  pen_l_bl: "Left box · far side on goal line",
  pen_l_tl: "Left box · near side on goal line",
  pen_l_br: "Left box · far outer",
  pen_l_tr: "Left box · near outer",
  goal_l_br: "Left 6yd box · far outer",
  goal_l_tr: "Left 6yd box · near outer",
  half_b: "Halfway line · far touchline",
  half_t: "Halfway line · near touchline",
  centre: "Centre spot",
  circle_b: "Centre circle · far side",
  circle_t: "Centre circle · near side",
  goal_r_bl: "Right 6yd box · far outer",
  goal_r_tl: "Right 6yd box · near outer",
  pen_r_bl: "Right box · far outer",
  pen_r_tl: "Right box · near outer",
  pen_r_br: "Right box · far side on goal line",
  pen_r_tr: "Right box · near side on goal line",
  corner_br: "Far-right corner",
  corner_tr: "Near-right corner",
};

export function landmarks(p: PitchDims): [string, Pt][] {
  const L = p.length,
    W = p.width,
    pd = p.penalty_depth,
    pw = p.penalty_width,
    gd = p.goal_box_depth,
    gw = p.goal_box_width,
    r = p.centre_circle_r,
    cy = W / 2;
  return [
    ["corner_bl", [0, 0]],
    ["corner_tl", [0, W]],
    ["pen_l_bl", [0, cy - pw / 2]],
    ["pen_l_tl", [0, cy + pw / 2]],
    ["pen_l_br", [pd, cy - pw / 2]],
    ["pen_l_tr", [pd, cy + pw / 2]],
    ["goal_l_br", [gd, cy - gw / 2]],
    ["goal_l_tr", [gd, cy + gw / 2]],
    ["half_b", [L / 2, 0]],
    ["half_t", [L / 2, W]],
    ["centre", [L / 2, cy]],
    ["circle_b", [L / 2, cy - r]],
    ["circle_t", [L / 2, cy + r]],
    ["goal_r_bl", [L - gd, cy - gw / 2]],
    ["goal_r_tl", [L - gd, cy + gw / 2]],
    ["pen_r_bl", [L - pd, cy - pw / 2]],
    ["pen_r_tl", [L - pd, cy + pw / 2]],
    ["pen_r_br", [L, cy - pw / 2]],
    ["pen_r_tr", [L, cy + pw / 2]],
    ["corner_br", [L, 0]],
    ["corner_tr", [L, W]],
  ];
}

export const LANDMARK_COUNT = 21;

/** Pitch markings as polylines in metres, for the overlay and the diagram. */
export function pitchLines(p: PitchDims): Pt[][] {
  const L = p.length,
    W = p.width,
    pd = p.penalty_depth,
    pw = p.penalty_width,
    gd = p.goal_box_depth,
    gw = p.goal_box_width,
    r = p.centre_circle_r,
    cy = W / 2;
  // Open polyline starting on the goal line: the fourth side is the goal line
  // itself, already drawn by the outline. Order matters — start at the goal
  // line so the FRONT edge (the one at depth d) actually gets a segment.
  const box = (gl: number, d: number, h: number): Pt[] => [
    [gl, cy - h / 2],
    [d, cy - h / 2],
    [d, cy + h / 2],
    [gl, cy + h / 2],
  ];
  const circle: Pt[] = Array.from({ length: 49 }, (_, i) => {
    const a = (i / 48) * 2 * Math.PI;
    return [L / 2 + r * Math.cos(a), cy + r * Math.sin(a)];
  });
  return [
    [
      [0, 0],
      [L, 0],
      [L, W],
      [0, W],
      [0, 0],
    ],
    [
      [L / 2, 0],
      [L / 2, W],
    ],
    circle,
    box(0, pd, pw),
    box(L, L - pd, pw),
    box(0, gd, gw),
    box(L, L - gd, gw),
  ];
}

// ------------------------------------------------------- 3x3 linear algebra

const mul3 = (A: number[], B: number[]): number[] => {
  const C = new Array<number>(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += A[i * 3 + k] * B[k * 3 + j];
      C[i * 3 + j] = s;
    }
  return C;
};

function inv3(M: number[]): number[] | null {
  const [a, b, c, d, e, f, g, h, i] = M;
  const A = e * i - f * h,
    B = f * g - d * i,
    C = d * h - e * g,
    det = a * A + b * B + c * C;
  if (!isFinite(det) || Math.abs(det) < 1e-14) return null;
  return [
    A / det,
    (c * h - b * i) / det,
    (b * f - c * e) / det,
    B / det,
    (a * i - c * g) / det,
    (c * d - a * f) / det,
    C / det,
    (b * g - a * h) / det,
    (a * e - b * d) / det,
  ];
}

const apply = (H: number[], p: Pt): Pt => {
  const w = H[6] * p[0] + H[7] * p[1] + H[8];
  return [
    (H[0] * p[0] + H[1] * p[1] + H[2]) / w,
    (H[3] * p[0] + H[4] * p[1] + H[5]) / w,
  ];
};

/** Hartley normalisation: centroid at 0, mean radius sqrt(2). Skipping it makes
 *  the 8x8 normal equations ill-conditioned at 1000+ px coordinates and the fit
 *  visibly drifts on a real 1080p frame. */
function hartley(pts: Pt[]): [number[], Pt[]] {
  let cx = 0,
    cy = 0;
  for (const p of pts) {
    cx += p[0];
    cy += p[1];
  }
  cx /= pts.length;
  cy /= pts.length;
  let d = 0;
  for (const p of pts) d += Math.hypot(p[0] - cx, p[1] - cy);
  const s = Math.SQRT2 / Math.max(d / pts.length, 1e-12);
  return [
    [s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1],
    pts.map((p): Pt => [(p[0] - cx) * s, (p[1] - cy) * s]),
  ];
}

/** Solve the 8x8 normal equations by Gauss-Jordan with partial pivoting. */
function solve8(N: number[][], c: number[]): number[] | null {
  const n = c.length,
    M = N.map((r, i) => r.concat([c[i]]));
  for (let k = 0; k < n; k++) {
    let piv = k;
    for (let i = k + 1; i < n; i++)
      if (Math.abs(M[i][k]) > Math.abs(M[piv][k])) piv = i;
    if (Math.abs(M[piv][k]) < 1e-14) return null;
    [M[k], M[piv]] = [M[piv], M[k]];
    for (let i = 0; i < n; i++) {
      if (i === k) continue;
      const f = M[i][k] / M[k][k];
      for (let j = k; j <= n; j++) M[i][j] -= f * M[k][j];
    }
  }
  return M.map((r, i) => r[n] / r[i]);
}

/** Weighted DLT. Weights are what let one homography per strip use EVERY
 *  correspondence while leaning on the nearby ones — picking a local subset
 *  instead lands on degenerate windows (the five halfway-line landmarks are
 *  adjacent in image x and collinear on the pitch).
 *
 *  # ponytail: least-squares with h33 fixed to 1, where calibrators.py takes
 *  the SVD null vector and pitch.py uses cv2.LMEDS. They agree to well under a
 *  centimetre on clean input; this is the operator preview, and `points` is
 *  exported too, so re-solving in Python is always available. */
function solveW(src: Pt[], dst: Pt[], w: number[]): number[] | null {
  if (src.length < 4) return null;
  const [Ts, a] = hartley(src),
    [Td, b] = hartley(dst);
  const N = Array.from({ length: 8 }, () => new Array<number>(8).fill(0)),
    c = new Array<number>(8).fill(0);
  const addRow = (row: number[], rhs: number) => {
    for (let i = 0; i < 8; i++) {
      c[i] += row[i] * rhs;
      for (let j = 0; j < 8; j++) N[i][j] += row[i] * row[j];
    }
  };
  for (let i = 0; i < a.length; i++) {
    const s = Math.sqrt(Math.max(w[i], 0));
    if (!s) continue;
    const [x, y] = a[i],
      [u, v] = b[i];
    addRow([x * s, y * s, s, 0, 0, 0, -u * x * s, -u * y * s], u * s);
    addRow([0, 0, 0, x * s, y * s, s, -v * x * s, -v * y * s], v * s);
  }
  const h = solve8(N, c);
  if (!h) return null;
  const Tdi = inv3(Td);
  if (!Tdi) return null;
  const H = mul3(Tdi, mul3(h.concat([1]), Ts));
  if (!isFinite(H[8]) || Math.abs(H[8]) < 1e-14) return null;
  return H.map((v) => v / H[8]);
}

/** Blend between the two bracketing strips. Continuous by construction — a hard
 *  per-strip switch would put a step in every player's velocity at each seam. */
function blendAt(mats: number[][], centres: number[], x: number, pt: Pt): Pt {
  if (mats.length === 1) return apply(mats[0], pt);
  const last = centres.length - 1;
  const xx = Math.min(Math.max(x, centres[0]), centres[last]);
  let i = 0;
  while (i < last - 1 && xx > centres[i + 1]) i++;
  const t = (xx - centres[i]) / (centres[i + 1] - centres[i] || 1e-9);
  const A = apply(mats[i], pt),
    B = apply(mats[i + 1], pt);
  return [A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t];
}

const median = (a: number[]): number => {
  const s = [...a].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

export interface Fit {
  /** Row-major 3x3, image px -> pitch metres. What gets posted. */
  H: number[];
  rms: number;
  /** 1 = inlier, 0 = rejected, parallel to the names passed to solve(). */
  keep: number[];
  names: string[];
  toImage: (X: Pt) => Pt;
  toPitch: (p: Pt) => Pt;
}

/** One solver for both projection types: perspective is nStrips = 1. */
export function fit(src: Pt[], dst: Pt[], nStrips: number): Omit<Fit, "names"> | null {
  const n = src.length;
  if (n < 4) return null;
  const xs = src.map((p) => p[0]),
    lo = Math.min(...xs),
    hi = Math.max(...xs);
  const S = Math.max(1, nStrips | 0);
  const centres =
    S === 1
      ? [(lo + hi) / 2]
      : Array.from({ length: S }, (_, i) => lo + ((hi - lo) * i) / (S - 1));
  const sigma = S === 1 ? Infinity : (hi - lo) / (S - 1) || 1;
  let keep = new Array<number>(n).fill(1),
    Hs: number[][] | null = null,
    err: number[] = [];
  for (let it = 0; it < 3; it++) {
    const next = centres.map((c) =>
      solveW(
        src,
        dst,
        src.map(
          (p, i) =>
            keep[i] * (Math.exp(-0.5 * Math.pow((p[0] - c) / sigma, 2)) + 0.05),
        ),
      ),
    );
    if (next.some((H) => !H)) return null;
    Hs = next as number[][];
    const solved = Hs;
    err = src.map((p, i) => {
      const q = blendAt(solved, centres, p[0], p);
      return Math.hypot(q[0] - dst[i][0], q[1] - dst[i][1]);
    });
    // Reject on median+MAD, not a multiple of the median: model mismatch is
    // smooth and spreads the MAD, a misclick is isolated and doesn't.
    const kept = err.filter((_, i) => keep[i]);
    const med = median(kept),
      mad = median(kept.map((e) => Math.abs(e - med)));
    const thr = Math.max(med + 6 * mad, 2 * med, 0.5);
    const nw: number[] = err.map((e) => (e <= thr ? 1 : 0));
    const rejected = nw.filter((v) => !v).length;
    // A fifth of the landmarks looking wrong means the MODEL is wrong, not the
    // clicks. Rejecting them would delete the evidence and report a lovely RMS.
    if (rejected > Math.max(1, Math.floor(n / 5))) break;
    if (nw.reduce((a, b) => a + b, 0) < 4) break;
    if (nw.every((v, i) => v === keep[i])) break;
    keep = nw;
  }
  if (!Hs) return null;
  const strips = Hs;
  const H = solveW(src, dst, keep);
  if (!H) return null;
  const inl = err.filter((_, i) => keep[i]);
  const rms = Math.sqrt(inl.reduce((s, e) => s + e * e, 0) / inl.length);
  const Hinv = strips.map(inv3);
  if (Hinv.some((m) => !m)) return null;
  const inverses = Hinv as number[][];
  const Hgi = inv3(H);
  if (!Hgi) return null;
  const toPitch = (p: Pt): Pt => blendAt(strips, centres, p[0], p);
  // Blending forward maps has no closed-form inverse (blending the inverses is
  // NOT it — that lands ~19px out on a stitched frame, which is visible on the
  // overlay). Quasi-Newton instead: measure the pitch-space residual, step by
  // it through the local inverse. Converges to sub-pixel in a few rounds.
  const toImage = (X: Pt): Pt => {
    let p = apply(Hgi, X);
    for (let k = 0; k < 8; k++) {
      const cur = toPitch(p);
      const dx = X[0] - cur[0],
        dy = X[1] - cur[1];
      if (Math.abs(dx) + Math.abs(dy) < 1e-9) break;
      const at = (q: Pt): Pt => blendAt(inverses, centres, p[0], q);
      const a = at(cur),
        b = at([cur[0] + dx, cur[1] + dy]);
      p = [p[0] + b[0] - a[0], p[1] + b[1] - a[1]];
    }
    return p;
  };
  return { H, rms, keep, toImage, toPitch };
}

/** Solve from named clicks. Returns null under four points, exactly as
 *  pitch.Calibration.solve raises there. */
export function solveClicks(
  p: PitchDims,
  clicks: Record<string, Pt>,
  nStrips = 1,
): Fit | null {
  const lm = landmarks(p);
  const names = lm.map(([n]) => n).filter((n) => clicks[n]);
  if (names.length < 4) return null;
  const map = Object.fromEntries(lm);
  const f = fit(
    names.map((n) => clicks[n]),
    names.map((n) => map[n]),
    nStrips,
  );
  return f ? { ...f, names } : null;
}

export function outliersOf(f: Fit | null): Set<string> {
  if (!f) return new Set();
  return new Set(f.names.filter((n, i) => !f.keep[i]));
}

/** The file app/server/jobs.py :: load_calibration will accept. Same shape as
 *  pitch.Calibration.save() plus `projection_type`. */
export function calibrationPayload(
  p: PitchDims,
  clicks: Record<string, Pt>,
  f: Fit,
  projection: "perspective" | "panoramic",
  nStrips: number,
): Record<string, unknown> {
  const out: Record<string, unknown> = {
    projection_type: projection,
    pitch: p,
    H: [f.H.slice(0, 3), f.H.slice(3, 6), f.H.slice(6, 9)],
    points: clicks,
    rms_error_m: f.rms,
    outliers: [...outliersOf(f)],
  };
  if (projection === "panoramic") out.n_strips = nStrips;
  return out;
}

/** Round-trip a synthetic camera: pitch -> image -> pitch. Mirrors
 *  model/engine/pitch.py :: _demo, which is the thing this must not drift from. */
export function demo(): void {
  const p: PitchDims = { ...DEFAULT_PITCH };
  const trueH = [12.0, 3.0, 300.0, 0.0, 9.0, 120.0, 0.0, 0.006, 1.0];
  const lm = landmarks(p);
  const clicks: Record<string, Pt> = {};
  for (const [name, world] of lm) clicks[name] = apply(trueH, world);

  const f = solveClicks(p, clicks);
  if (!f) throw new Error("solve failed on clean synthetic input");
  if (!(f.rms < 0.01)) throw new Error(`rms too high: ${f.rms}`);
  for (const [, world] of lm) {
    const back = f.toPitch(apply(trueH, world));
    if (Math.hypot(back[0] - world[0], back[1] - world[1]) > 1e-3)
      throw new Error("round trip drifted");
  }

  // A single misclicked corner must not wreck the fit — this is the whole
  // reason for more points than the minimum plus MAD rejection.
  const bad: Record<string, Pt> = { ...clicks };
  bad.corner_bl = [clicks.corner_bl[0] + 90, clicks.corner_bl[1] + 90];
  const badFit = solveClicks(p, bad);
  if (!badFit) throw new Error("solve failed with one outlier");
  if (!(badFit.rms < 0.5)) throw new Error(`outlier wrecked the fit: ${badFit.rms}`);
  if (![...outliersOf(badFit)].includes("corner_bl"))
    throw new Error(`outlier not named: ${[...outliersOf(badFit)]}`);

  // Four points is the floor, not a target; two must not solve.
  const four = Object.fromEntries(
    ["corner_bl", "corner_tl", "corner_br", "corner_tr"].map((k) => [k, clicks[k]]),
  );
  if (!(solveClicks(p, four)!.rms < 0.01)) throw new Error("four corners failed");
  if (solveClicks(p, { corner_bl: clicks.corner_bl, corner_tl: clicks.corner_tl }))
    throw new Error("should have refused 2 points");

  console.log("pitch.ts ok");
}
