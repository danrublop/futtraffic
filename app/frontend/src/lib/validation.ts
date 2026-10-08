import { readFile } from "node:fs/promises";
import path from "node:path";

/** Reads validate.py's reports off disk, server-side.
 *
 *  They are repo-level artefacts, not per-match output, and the job service does
 *  not serve them — the old UI found them by parsing an HTTP directory listing,
 *  which only worked when somebody happened to be serving the repo root. Here
 *  the server reads the file, so /checks tells the same story wherever it runs.
 */

export interface MetricResult {
  metric: string;
  target: number;
  gates: boolean;
  n: number;
  mae_pct: number | null;
  bias_pct: number | null;
  loa_lower: number | null;
  loa_upper: number | null;
  verdict: string;
  skipped_zero_gt: number;
  fitted: boolean;
  tee_pct: number | null;
  ccc: number | null;
  std_bias: number | null;
  calib_slope: number | null;
  calib_intercept: number | null;
  tee_rating: string;
}

export interface ValidationReport {
  status: string;
  /** Stamped into every report since 2026-08-04. A report that does not say
   *  what produced it can no longer be written — that is the whole lesson of
   *  model/validation/retired/. */
  config: Record<string, string | number>;
  source: string;
  reference_error_pct: number | null;
  n_player_matches: number;
  n_required: number;
  held_out_matches: string[];
  train_matches: string[];
  headline: Record<string, MetricResult>;
  fitted_train_split?: Record<string, MetricResult>;
  gate: string;
  caveats: string[];
}

/** The authoritative report. Anything else on disk either predates the
 *  configuration stamp or is retired. */
export const REPORT_REL = "model/validation/tune_runs/holdout-windowed.json";

const REPO_ROOT = path.join(process.cwd(), "..");

export async function readReport(
  rel: string = REPORT_REL,
): Promise<{ rel: string; data: ValidationReport } | null> {
  // model/validation/retired/ is kept as evidence of a measurement mistake that had
  // already been quoted downstream. Reading one would let a retracted figure
  // become the headline again, which is the exact failure this page exists to
  // prevent. See model/validation/retired/README.md.
  if (/(^|\/)retired(\/|$)/.test(rel))
    throw new Error(`refusing to read a retired report: ${rel}`);
  try {
    const text = await readFile(path.join(REPO_ROOT, rel), "utf8");
    return { rel, data: JSON.parse(text) as ValidationReport };
  } catch {
    return null;
  }
}

/** A run whose measured side was built by perturbing the ground truth compared
 *  two ESTIMATORS over the same positions. It can show the maths is sound. It
 *  can never show the product is accurate, because the tracker never ran. */
export function isSynthetic(r: ValidationReport): boolean {
  const d = r.config?.degrade;
  return d != null && d !== "none";
}

export const METRIC_NAME: Record<string, string> = {
  distance_m: "Total distance",
  minutes: "Minutes played",
  hsr_m: "High-speed running",
  sprints: "Sprints",
  peak_speed_ms: "Peak speed",
  id_switches: "Identity (ID switches)",
};

/** Measured on nothing at all. Sourced from the README's "Read this before
 *  quoting any number", metrics-2026-08 (not published) §0 and NOISE_TOLERANCE (not published) —
 *  not invented here. Each line names what would close it. */
export const NEVER_MEASURED: [string, string][] = [
  [
    "The tracking pipeline on real football video",
    "track.py has been scored end to end only on SoccerTrack v2 benchmark footage, never on a user's own match. Every other figure on this page comes from ground-truth trajectories with error injected into them, not from this detector and this tracker.",
  ],
  [
    "This camera, this venue",
    "The calibration is operator clicks on landmarks, fitted as one homography. No surveyed ground points, so its metre scale is unchecked here — and on a stitched panorama even perfect clicks leave metres of fit error.",
  ],
  [
    "Any comparison against a reference device",
    "No GPS vest, no optical system, no second instrument has ever been run on the same match. Agreement with another instrument is not error against truth, and we do not even have the agreement figure.",
  ],
  [
    "Jersey reading",
    "Identity is human-seeded by design. OCR at 20–50 px is unsolved and we do not claim otherwise; the operator binds tracks once per match and the algorithm propagates.",
  ],
  [
    "The filter bandwidth the distance claim is defined at",
    "Path length is scale-dependent — the same true trajectory measures 1836 m at 25 Hz and 1481 m at 1 Hz. This report states 0.4 s, which is why its distance figure is quotable as a bandwidth-specific agreement and not as accuracy.",
  ],
];

/** Figures that were published, then found to be measuring something else.
 *  Listed so they cannot quietly return. */
export const RETRACTED: [string, string][] = [
  [
    "distance MAE 410.8%, bias +120.5%, gate FAIL",
    "A harness artifact, retracted 2026-08-04. Reproduced bit-exactly at q=16, σ_px=0.5, 11 px injected noise — 32× the shipped q/σ ratio and the worst-scoring cell in tune.py's own grid. Nothing in the file recorded that, because build_report() did not stamp the configuration. It now does.",
  ],
  [
    "stats.py reads ~26% low",
    "Withdrawn 2026-08-04. It was scored against a reference asserting ~14 km per match, which is integrated quantisation dither rather than distance.",
  ],
];
