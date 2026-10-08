"""N41: does a lower detector confidence make a better PRODUCT? Protocol first.

WHY
    N40 measured that occluded and far-side players are detected weakly, not missed: on 118575 a confidence floor of
    0.3 instead of the shipped 0.5 lifts far-side occluded recall from 0.31 to 0.58 and isolated far recall from 0.64
    to 0.88, for about 1.4 more non-ground-truth boxes per frame (docs/experiments/n40-association.md §2). A recall table is not
    a product decision: extra boxes that form tracks inflate distance, cost identity, and the tracker's behaviour on
    weak boxes is not the detector's. N17 measured conf 0.3 at +0.02 HOTA on an older pipeline
    (registry-measures-configs-that-do-not-ship (internal note) warns what such numbers were made under). This is the
    same question asked of what ships today, under the split and rule below.

ARMS (the product's own path: model/engine/track.run with the clip's zero-click calibration -> scene prior, tracker,
      split + team-gated stitch -> jobs.apply_identity; the only difference is the detector's conf)
    C50   conf 0.5, what ships: the existing `.p35` / `.p35id` tracks (docs/experiments/n34-zero-click.md §7).
    C30   conf 0.3: `.p41c30` / `.p41c30id`, fresh detector passes (the cache keys on video and start, not conf).

POPULATION AND SPLIT (N40's).  DEV = 118575, 118576, 117093 - looked at freely.  HELD = 118577, 118578, 128057,
    128058 - run ONCE. Seven same-install clips; the split is by match, not by camera.

METRICS, per clip
    TrackEval on the stitched tracks (`.p41c30` vs `.p35`, model/validation/st2_mot.py): HOTA, DetA, DetRe, DetPr, AssA,
    IDF1, IDSW, and track ids. Then the identified tracks through dist_eval.evaluate_stc (oracle="drop", the N35
    columns): distance per covered metre RAW against the feed and over its floor, HSR ratio raw and over floor, and
    the share of rows with no owner (a rise here is false boxes forming tracks).

RULE, fixed now. C30 REPLACES 0.5 in the product iff, on HELD (means of four clips unless said):
      HOTA >= C50 + 0.010;   DetPr >= C50 - 0.020 on every clip;   IDSW <= 1.15 x C50;
      distance per covered metre RAW <= C50 + 0.010;   HSR ratio RAW <= C50;   no-owner share <= C50 + 0.02.
    Anything else: NOT SUPPORTED; 0.5 stays; the numbers are published. n = 1 candidate value (0.3, chosen from the
    recall table on ONE dev clip, which is why HELD exists).

    .venv/bin/python model/validation/n41_conf.py --build 118575      # detector at 0.3 + identity, one clip (~10 min GPU)
    .venv/bin/python model/validation/n41_conf.py --score dev          # TrackEval + distance/HSR on DEV -> n41_dev.json
    .venv/bin/python model/validation/n41_conf.py --score held         # ONCE; refuses if this file is uncommitted
    .venv/bin/python model/validation/n41_conf.py --show
"""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
STC = os.path.join(ROOT, "model/validation/out/stc2025")
DEV = ("118575", "118576", "117093")
HELD = ("118577", "118578", "128057", "128058")
CONF = 0.3
ARMS = {"C50": ("p35", "p35id"), "C30": ("p41c30", "p41c30id")}



def _git_env():
    """verify_claims.git_env - the environment MINUS every GIT_*.

    Without it an exported GIT_DIR (a hook, or any shell started inside one)
    BEATS cwd=, so this guard reads the OTHER tree's HEAD, where this path is
    usually clean - returncode 0, "registered", green. It fails SHORT, in the
    direction that serves an unregistered run as a registered one.
    a-green-gate-is-not-evidence-it-looked (internal note)
    """
    import sys as _sys  # noqa: PLC0415
    _sys.path.insert(0, os.path.join(ROOT, "model", "validation"))
    import verify_claims as _V  # noqa: PLC0415
    return _V.git_env()


def build(seq):
    import st2_product as ST  # noqa: PLC0415
    cal = ST._p(seq, ".p35.calibration.json")
    ST.track(seq, cal, tag=".p41c30", conf=CONF)
    ST.identify(seq, cal, src=".p41c30", dst=".p41c30id")


def score(which, out=None):
    import dist_eval as D  # noqa: PLC0415
    import n31_scale as S  # noqa: PLC0415
    import st2_mot as M  # noqa: PLC0415
    seqs = HELD if which == "held" else DEV
    if which == "held" and subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "model/validation/n41_conf.py"], cwd=ROOT, env=_git_env()).returncode:
        raise SystemExit("VOID: the held-out run must be made by the committed protocol")
    import calibrators as CAL  # noqa: PLC0415
    res = {}
    for arm, (raw, ided) in ARMS.items():
        M.VARIANT = f".{raw}"
        M.stage_gt(seqs, every=2)                       # the product tracks at stride 2; GT at every frame would halve recall
        M.stage_pred(seqs)
        te = M.run(seqs)
        res[arm] = {}
        for q in seqs:
            cal = CAL.load(os.path.join(STC, f"{q}.p35.calibration.json"))    # the metres come through the zero-click map, as N35
            d = D.evaluate_stc(q, ided, oracle="drop", our_cal=cal)
            f = D.evaluate_stc(q, ided, oracle="position")
            dets = json.load(open(os.path.join(STC, f"{q}.{raw}.tracks.json")))["detections"]
            res[arm][q] = {**te[q], "ids": len({r["id"] for r in dets}),
                           "dist_raw": round(d["squad"]["ours_m"] / d["squad"]["gt_matched_m"], 4),
                           "dist_over_floor": round((d["squad"]["ours_m"] / d["squad"]["gt_matched_m"]) / (f["squad"]["ours_m"] / f["squad"]["gt_matched_m"]), 4),
                           "hsr_raw": round(d["speed"]["our_hsr_m"] / d["speed"]["gt_matched_hsr_m"], 3),
                           "hsr_over_floor": round((d["speed"]["our_hsr_m"] / d["speed"]["gt_matched_hsr_m"]) / (f["speed"]["our_hsr_m"] / f["speed"]["gt_matched_hsr_m"]), 3),
                           "no_owner": round(float(S.chord_inputs(q, ided)[6]), 4)}
    json.dump(res, open(out or os.path.join(STC, f"n41_{which}.json"), "w"), indent=1)
    if out is None:
        show()


def show():
    for which in ("dev", "held"):
        p = os.path.join(STC, f"n41_{which}.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        seqs = list(r["C50"])
        keys = ("HOTA", "DetA", "DetRe", "DetPr", "AssA", "IDF1", "IDSW", "ids", "dist_raw", "dist_over_floor", "hsr_raw", "hsr_over_floor", "no_owner")
        print(f"== {which.upper()} ==")
        for q in seqs:
            print(q, " ".join(f"{k} {r['C50'][q][k]:.3f}->{r['C30'][q][k]:.3f}" if isinstance(r['C50'][q][k], float) else f"{k} {r['C50'][q][k]}->{r['C30'][q][k]}" for k in keys))
        m = lambda a, k: float(np.nanmean([r[a][q][k] for q in seqs]))  # noqa: E731
        print("MEAN " + " ".join(f"{k} {m('C50', k):.4f}->{m('C30', k):.4f}" for k in keys))
        if which == "held":
            ok = [m("C30", "HOTA") >= m("C50", "HOTA") + 0.010,
                  all(r["C30"][q]["DetPr"] >= r["C50"][q]["DetPr"] - 0.020 for q in seqs),
                  m("C30", "IDSW") <= 1.15 * m("C50", "IDSW"),
                  m("C30", "dist_raw") <= m("C50", "dist_raw") + 0.010,
                  m("C30", "hsr_raw") <= m("C50", "hsr_raw"),
                  m("C30", "no_owner") <= m("C50", "no_owner") + 0.02]
            names = ("HOTA +0.01", "DetPr held", "IDSW <= 1.15x", "distance raw", "HSR raw", "no-owner")
            print("RULE:", ", ".join(f"{n} {'ok' if o else 'FAIL'}" for n, o in zip(names, ok)), "->", "REPLACES 0.5" if all(ok) else "NOT SUPPORTED")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--build" in a:
        build(a[a.index("--build") + 1])
    elif "--score" in a:
        score(a[a.index("--score") + 1])
    else:
        show()
