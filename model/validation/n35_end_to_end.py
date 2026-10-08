"""N35: the product end to end under the zero-click calibration, beside the product as it ships.

A DIAGNOSTIC, declared before it ran: clip 118575 (ST2's train split), ONE configuration per arm, nothing
tuned on it here, one camera install. It says whether the pipeline RUNS through a spherical calibration
and which way its numbers move; it is not a held-out result and it ships nothing.

ARMS
    SHIP   N21's `.p21id`: the product as it ships - N6's clicked homography (n21-product-now (not published)).
    AUTO   `.p35id`: the same product path (st2_product.track then .identify: fresh detection, the
           scene prior, the tracker, the metric stitch, the identity stage) under
           `118575.p35.calibration.json`, the camera model/engine/autocal.py found from the paint with no clicks.
           `--build` makes it (about 8 min on an M3).

THE REFERENCE IS BIASED, so every ratio is printed beside its FLOOR. ST2's feed sits about 4% too far
from this camera in the pitch interior (audit-n33 (not published) finding 4) and is quantised to about a metre.
A correct map therefore reads movement about 0.96 of the feed's, distance a few percent under it, and
high-speed running well under it (HSR's elasticity to scale is about 4.5 -
research-stitched-panorama-calibration (not published) §6). The floor is TRUE positions pushed through the same
fragments (oracle="position"): what is left when the map is perfect, which is the fragments' own.
`no-owner` is the share of track rows with no IoU owner in the ground truth, which every figure excludes.

EXTENSION, declared 2026-09-17 before any of it ran: the same diagnostic, the same one configuration, on the six
other clips of the same install (`--seq`). AUTO's calibration there is the camera `model/validation/n34_autocal.py
--fit` found for that clip from its paint (n34_autocal.json), written as the product's own calibration file by
`--cal`. Nothing is tuned on them; all seven are ONE camera on ONE pitch, so seven clips are not seven tests of
the calibration - they are seven samples of the players.

    .venv/bin/python model/validation/n35_end_to_end.py --seq 118576 --cal     # writes <seq>.p35.calibration.json
    .venv/bin/python model/validation/n35_end_to_end.py --summary              # every clip scored so far, side by side
    .venv/bin/python model/validation/n35_end_to_end.py --build    # GPU: tracks and identifies under the zero-click calibration
    .venv/bin/python model/validation/n35_end_to_end.py            # scores both arms -> model/validation/out/stc2025/n35_e2e.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine"), os.path.join(ROOT, "app", "server")]
SEQ = sys.argv[sys.argv.index("--seq") + 1] if "--seq" in sys.argv else "118575"
OUT = os.path.join(ROOT, "model/validation/out/stc2025/n35_e2e.json" if SEQ == "118575" else f"model/validation/out/stc2025/n35_e2e_{SEQ}.json")


def write_cal():
    import calibrators as CAL  # noqa: PLC0415
    from pitch import Pitch  # noqa: PLC0415
    a = json.load(open(os.path.join(ROOT, "model/validation/out/stc2025/n34_autocal.json")))[SEQ]
    cal = CAL.SphericalCamera(Pitch(float(a.get("length_m") or 105.0), 68.0), a["camera"], (int(a["width"]), 1080),
                              auto={k: a.get(k) for k in ("accepted", "via", "hits", "samples", "line_rms_px")})
    path = os.path.join(ROOT, f"model/validation/out/stc2025/{SEQ}.p35.calibration.json")
    with open(path, "w") as fh:
        json.dump(cal.to_dict(), fh)
    print("->", path)


def build():
    import st2_product as ST  # noqa: PLC0415
    cal = ST._p(SEQ, ".p35.calibration.json")
    if not os.path.exists(cal):
        raise SystemExit(f"{cal} is missing: write it with model/engine/autocal.py (jobs.auto_calibrate) first")
    ST.track(SEQ, cal, tag=".p35")
    ST.identify(SEQ, cal, src=".p35", dst=".p35id")


def score():
    import calibrators as CAL  # noqa: PLC0415
    import dist_eval as D  # noqa: PLC0415
    import n31_scale as S  # noqa: PLC0415
    auto = CAL.load(f"{D.STC}/{SEQ}.p35.calibration.json")
    res = {}
    for tag, variant, cal in (("SHIP", "p21id", None), ("AUTO", "p35id", auto)):
        kw = {} if cal is None else {"our_cal": cal}
        f = D.evaluate_stc(SEQ, variant, oracle="position")
        d = D.evaluate_stc(SEQ, variant, oracle="drop", **kw)
        calx = cal if cal is not None else CAL.load(f"{D.STC}/{SEQ}.p05.calibration.json")
        tid, fr, g, uv, G, fps, drop = S.chord_inputs(SEQ, variant)
        sl = S.slopes(S.sums(tid, fr, g, calx.to_pitch(uv), G, fps))
        e = np.linalg.norm(calx.to_pitch(uv) - G, axis=1)
        e = e[np.isfinite(e)]
        dets = json.load(open(f"{D.STC}/{SEQ}.{variant}.tracks.json"))["detections"]
        res[tag] = {"rows": len(dets), "ids": len({x["id"] for x in dets}), "no_owner_share": round(float(drop), 4),
                    "pos_median_m": round(float(np.median(e)), 2), "pos_p90_m": round(float(np.percentile(e, 90)), 2),
                    "slope": {k: sl[k] for k in ("all", "x", "y", "far", "near")},
                    "distance_per_covered_m": round(d["squad"]["ours_m"] / d["squad"]["gt_matched_m"], 4),
                    "distance_floor": round(f["squad"]["ours_m"] / f["squad"]["gt_matched_m"], 4),
                    "hsr_ratio": round(d["speed"]["our_hsr_m"] / d["speed"]["gt_matched_hsr_m"], 3),
                    "hsr_floor": round(f["speed"]["our_hsr_m"] / f["speed"]["gt_matched_hsr_m"], 3)}
        r = res[tag]
        r["distance_over_floor"] = round(r["distance_per_covered_m"] / r["distance_floor"], 4)
        r["hsr_over_floor"] = round(r["hsr_ratio"] / r["hsr_floor"], 3)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    for tag, r in res.items():
        print(tag, json.dumps(r))
    print("->", OUT)


def summary():
    import glob  # noqa: PLC0415
    rows = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "model/validation/out/stc2025/n35_e2e*.json"))):
        q = os.path.basename(path)[len("n35_e2e_"):-len(".json")] or "118575"
        rows[q] = json.load(open(path))
    keys = (("pos_median_m", "position m"), ("distance_over_floor", "distance/floor"), ("hsr_over_floor", "HSR/floor"), ("ids", "ids"), ("no_owner_share", "no-owner"))
    print("clip    " + "  ".join(f"{name:>22s}" for _k, name in keys) + "             slope all (far, near)   SHIP -> AUTO")
    for q, r in rows.items():
        print(q + "  " + "  ".join(f"{r['SHIP'][k]:>9} -> {r['AUTO'][k]:<9}" for k, _n in keys)
              + f"  {r['SHIP']['slope']['all']:.3f} ({r['SHIP']['slope']['far']:.3f}, {r['SHIP']['slope']['near']:.3f}) -> {r['AUTO']['slope']['all']:.3f} ({r['AUTO']['slope']['far']:.3f}, {r['AUTO']['slope']['near']:.3f})")
    med = lambda arm, k: float(np.median([r[arm][k] for r in rows.values()]))  # noqa: E731
    print(f"MEDIAN over {len(rows)} clips  " + "  ".join(f"{name} {med('SHIP', k):.3f} -> {med('AUTO', k):.3f}" for k, name in keys[:3])
          + f"  slope {np.median([r['SHIP']['slope']['all'] for r in rows.values()]):.3f} -> {np.median([r['AUTO']['slope']['all'] for r in rows.values()]):.3f}")


if __name__ == "__main__":
    if "--summary" in sys.argv:
        raise SystemExit(summary())
    write_cal() if "--cal" in sys.argv else build() if "--build" in sys.argv else score()
