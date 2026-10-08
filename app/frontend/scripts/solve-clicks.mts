/** The operator's calibration, headless: clicks in, the exact payload the UI
 *  posts out. Lets a benchmark score the PRODUCT's solver instead of a Python
 *  re-implementation of it - pitch.ts rejects outliers on median+MAD and stops
 *  once a fifth look wrong, where model/engine/pitch.py uses cv2.LMEDS, and on a
 *  stitched panorama the two disagree about most of the points.
 *
 *  stdin:  {"pitch": PitchDims, "clicks": {name: [x, y]}}
 *  stdout: calibrationPayload(...) as JSON, or {"error": ...}
 *
 *  Run: npx vite-node --config vitest.config.mts scripts/solve-clicks.mts < in.json
 */
import { readFileSync } from "node:fs";
import { calibrationPayload, solveClicks, type PitchDims } from "../src/lib/pitch";

const { pitch, clicks } = JSON.parse(readFileSync(0, "utf8")) as {
  pitch: PitchDims;
  clicks: Record<string, [number, number]>;
};
const f = solveClicks(pitch, clicks);
process.stdout.write(
  JSON.stringify(
    f ? calibrationPayload(pitch, clicks, f, "perspective", 1) : { error: "solve failed" },
  ),
);
