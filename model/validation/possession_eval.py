"""PathCRF possession events vs the 118575 ball-action labels. No ball anywhere.

THE ONE COMPLETE COMBINATION IN THIS REPO, and it went unnoticed for weeks.
    `model/validation/touch_eval.py` needs a detected ball and is therefore capped by
    the ball picker: on the 118575 panorama it fired on 15.3% of sampled frames
    and the resulting score could not be attributed between the picker and the
    owner rule. This harness has no such problem because PathCRF uses no ball at
    all - only the 22 trajectories.

    118575 has BOTH halves of what that needs, already on disk:

        model/validation/soccertrack/118575_1st.tracks.json.gz   22 ids, METRES,
            72,101 frames at 25 fps, team + jersey, essentially no gaps
        data/footage/.../118575_12_class_events.json            907 PASS + 823 DRIVE
            + 111 HIGH PASS ... 1,026 half-1 rows

    Ground-truth positions, so the thing possession.py calls the binding
    constraint - positional accuracy - is 0.0 m here. Its own sensitivity table
    says that is where the model scores F1 0.60; at 1 m it scores 0.46. So this
    measures PathCRF's ceiling on our footage, not our tracking.

KEEPERS ARE PINNED FROM THE DATA, NOT DETECTED, and possession.py is emphatic
about why: node feature 0 is a +-1 LEFT/RIGHT indicator, and a wrong keeper
flips it for every node at once - a cliff, not a gradient (seeded F1 0.34, 0.33,
0.087, 0.076, the two low ones being exactly the two where the keeper came back
wrong). Here they are unambiguous, by mean x over the whole half:

    home id 1   jersey 12   mean x 12.75    <- left
    away id 12  jersey  1   mean x 89.32    <- right

    Every other track sits between 35 and 72. There is no judgement call.

WHAT IS SCORED. PathCRF emits a possession path; a possession CHANGE is its
event. Those are compared against the labels with the same greedy 1-1 matcher
and the same random-placement chance baseline `touch_eval` uses, so the two
routes - ball-based and ball-free - are directly comparable.

MEASURED 2026-09-05. 118575 half 1, 300-1800 s (25 min), GT tracks, trial 140.
586 possession changes against 529 CONTACT and 283 PASS-family labels:

                       prec   rec    F1  chance      lag
    CONTACT tol 1.0 s  0.55  0.61  0.58    0.46   -0.04
    CONTACT tol 2.5 s  0.66  0.73  0.70    0.60   -0.04
    PASS    tol 1.0 s  0.28  0.58  0.38    0.34   +0.01
    PASS    tol 2.5 s  0.44  0.90  0.59    0.51   -0.08

    AGAINST THE BALL-BASED ROUTE ON THE SAME FOOTAGE, `touch_eval` scored
    CONTACT tol 1.0 s at F1 0.32 with recall 0.21. This is nearly double, and
    the reason is structural rather than a better rule: the touch route is
    capped by ball coverage (recall ceiling 0.65, actual 0.26) and this route
    uses no ball at all, so it has no such ceiling. [WITHDRAWN - see below:]
    PASS RECALL 0.90 is the number to keep - 256 of 283 passes, where the ball
    route could not exceed 0.61 even with a perfect owner rule.

    [WITHDRAWN - see below:] It reproduces the paper. `possession.py` quotes
    event F1 0.6007; a 3-minute
    window scored 0.62 and 25 minutes scored 0.58, and the +0.12 margin over
    chance was identical at both scales (n=68 and n=586), so the estimate is
    stable rather than a small-sample artifact. Lag is |0.04| s or better on
    every row - unbiased, not merely close.

    REFUTED AS A RESULT, 2026-09-10. The same number of events spaced EVENLY,
    no model at all, scores higher on every row of the table above -
    `metronome_f1` / `metronome_rec` are returned beside each row now, and the
    two headline rows are registered as `product.possession.metronome.*`.
    Labels arrive every few seconds, so at these tolerances timing F1 cannot
    tell a model from a clock; the chance column was the weaker floor and the
    only one run. "Reproduces" compared this identity-free timing F1 with
    0.6007, OUR earlier transfer run in possession.py, not with the paper.
    PathCRF's product output is team possession, and that is what has to be
    scored: model-map (not published) §0.4 and §5 N14.

    READ IT AS A CEILING, NOT A PRODUCT NUMBER. These are GROUND-TRUTH
    positions, so positional error is 0.0 m. possession.py's own sensitivity
    table says 1 m takes F1 0.60 -> 0.46 and the vendor-remap pipeline is far
    worse than 1 m. This measures the MODEL on our footage; it says nothing
    about our tracking. The honest next step is to re-run this against pipeline
    tracks rather than GT and report both.

    Precision is the weak column (0.28 PASS at tol 1.0 s) and it is partly an
    artefact of the target: the model emits a change for every possession,
    including the DRIVE and duel touches that PASS-family excludes. Against all
    CONTACT labels precision is 0.55-0.66. Do not tune to PASS-only precision
    without deciding first whether a DRIVE is a false positive or a true one.
"""
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "model", "engine"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GT = "model/validation/soccertrack/118575_1st.tracks.json.gz"
F0 = 3000001          # the GT's first frame index; t = (f - F0) / fps
TEAM_SIZE = 11


def positions(path=GT, t0=None, t1=None):
    """GT tracks -> events.Positions, plus the two keeper ids.

    `t0`/`t1` in seconds slice the half; PathCRF decodes in 40 s segments and a
    full half is 2,884 s, so a slice is how you iterate without paying for all
    of it.
    """
    import events  # noqa: PLC0415

    d = json.load(gzip.open(path))
    fps = float(d["fps"])
    rows = d["detections"]
    ids = sorted({x["id"] for x in rows}, key=int)
    idx = {k: i for i, k in enumerate(ids)}
    frames = sorted({x["f"] for x in rows})
    if t0 is not None:
        frames = [f for f in frames if t0 <= (f - F0) / fps <= t1]
    fidx = {f: i for i, f in enumerate(frames)}

    XY = np.full((len(frames), len(ids), 2), np.nan)
    team_of = {}
    for x in rows:
        i = fidx.get(x["f"])
        if i is None:
            continue
        XY[i, idx[x["id"]]] = (x["x"], x["y"])
        team_of[x["id"]] = x["team"]
    mask = ~np.isnan(XY[..., 0])
    # A handful of frames are missing for one track; carry the last known
    # position rather than handing NaN to a second derivative.
    for j in range(XY.shape[1]):
        good = np.where(mask[:, j])[0]
        if len(good) < len(frames):
            XY[:, j, 0] = np.interp(np.arange(len(frames)), good, XY[good, j, 0])
            XY[:, j, 1] = np.interp(np.arange(len(frames)), good, XY[good, j, 1])

    teams = np.array([team_of[k] for k in ids])
    # Keepers by mean x within each team bank - see the module docstring on why
    # this is safe here and dangerous in general.
    mx = {k: float(np.nanmean(XY[:, idx[k], 0])) for k in ids}
    home = [k for k in ids if team_of[k] == "home"]
    away = [k for k in ids if team_of[k] == "away"]
    gks = [min(home, key=mx.get), min(away, key=mx.get),
           max(home, key=mx.get), max(away, key=mx.get)]
    left_gk = min(gks, key=mx.get)
    right_gk = max(gks, key=mx.get)
    pos = events.Positions(t=np.array([(f - F0) / fps for f in frames]),
                           XY=XY, mask=np.ones_like(mask), ids=ids, teams=teams)
    return pos, left_gk, right_gk, fps


def events_from(p):
    """Possession path -> event times, one per change of the possessing node."""
    out, prev = [], None
    for k in range(len(p.src)):
        cur = int(p.src[k])
        if cur != prev:
            out.append(float(p.t[k]))
            prev = cur
    return out


def score(t0=300.0, t1=900.0):
    """The numbers, as a dict. Keys are f"{target}.{tol}" -> the row.

    Split out of `main` because `api.py` needs a value, not a printed table, and
    regex-scraping a scorer's stdout is how three of the first twelve
    measurements were wired - the format changes and the number silently stops
    resolving.
    """
    import possession  # noqa: PLC0415
    import touch_eval as T  # noqa: PLC0415

    pos, lgk, rgk, fps = positions(t0=float(t0), t1=float(t1))
    # rms_error_m=0.0: these ARE the ground truth, so the gate is honestly clear.
    p = possession.infer(pos, lgk, rgk, fps=fps, rms_error_m=0.0)
    det = events_from(p)
    out = {"window": [float(t0), float(t1)], "frames": len(pos.t),
           "tracks": len(pos.ids), "keepers": [lgk, rgk],
           "steps": len(p.src), "changes": len(det), "rows": {}}
    for name, keep in (("PASS", T.PASSES), ("CONTACT", T.CONTACT)):
        gt = T.labels(T.LABELS, float(t0), float(t1), keep)
        for tol in (1.0, 2.5):
            tp, pr, rc, f1, lag = T.score(det, gt, tol)
            m_rec, m_f1 = T.metronome(len(det), gt, tol, float(t0), float(t1))
            out["rows"][f"{name}.{tol}"] = {
                "labels": len(gt), "tp": tp, "prec": pr, "rec": rc, "f1": f1,
                "chance": T.chance(len(det), gt, tol, float(t0), float(t1)),
                "metronome_f1": m_f1, "metronome_rec": m_rec, "lag": lag}
    return out


JOB = "data/jobs/118575_full_1st"


def score_pipeline(job=JOB, tracks="tracks.split.json", t0=300.0, t1=1800.0):
    """The same score against OUR tracks instead of ground truth. It does not run.

    THIS IS THE MEASUREMENT THE PRODUCT NEEDS AND THE ONE WE DO NOT HAVE. Every
    possession number in this file is on GROUND-TRUTH trajectories, so it is
    PathCRF's ceiling on our footage and says nothing about our tracking. The
    honest next step named in the module docstring is exactly this function.

    It raises, and the two reasons are read from `possession.py` rather than
    written here, so the day either gate moves this stops raising instead of
    lying about why:

        22 tracks required, the pipeline emits 413 fragments
        calibration rms 1.5 m required, the job's file says 2.85 m

    A gap that reports nothing is invisible; a gap that reports FAILED with its
    two blockers is a work item (sprint P2). Do not "fix" this by relaxing the
    gates - possession.py refuses above MAX_RMS_M because the failure mode is a
    confident possession path drawn over the wrong players.
    """
    import possession  # noqa: PLC0415

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rows = json.load(open(os.path.join(root, job, tracks)))["detections"]
    n_ids = len({r["id"] for r in rows})
    rms = float(json.load(open(os.path.join(root, job, "calibration.json")))
                ["rms_error_m"])
    need = possession.TEAM_SIZE * 2
    bad = []
    if n_ids != need:
        bad.append(f"PathCRF needs exactly {need} tracks, {tracks} has {n_ids}")
    if rms > possession.MAX_RMS_M:
        bad.append(f"calibration rms {rms:.2f} m > possession.MAX_RMS_M "
                   f"{possession.MAX_RMS_M} m")
    if bad:
        raise RuntimeError("possession cannot run on pipeline tracks: "
                           + "; ".join(bad))
    raise NotImplementedError(
        "both gates now pass - wire pipeline tracks into positions() and "
        "delete this line (sprint P2)")


def main(t0=300.0, t1=900.0):
    r = score(t0, t1)
    print(f"window {t0}-{t1}s  {r['frames']} frames  {r['tracks']} tracks  "
          f"keepers left={r['keepers'][0]} right={r['keepers'][1]}")
    print(f"PathCRF: {r['steps']} steps, {r['changes']} possession changes "
          f"({r['changes'] * 60 / (float(t1) - float(t0)):.1f}/min)")
    for name, label in (("PASS", "PASS / HIGH PASS / CROSS"),
                        ("CONTACT", "all CONTACT labels")):
        print(f"\n== {label}: {r['rows'][f'{name}.1.0']['labels']} labels ==")
        for tol in (1.0, 2.5):
            x = r["rows"][f"{name}.{tol}"]
            print(f"  tol {tol}s: tp {x['tp']:3d}  prec {x['prec']:.2f}  "
                  f"rec {x['rec']:.2f}  F1 {x['f1']:.2f}  "
                  f"chance {x['chance']:.2f}  lag {x['lag']:+.2f}")
    return r


if __name__ == "__main__":
    main(*sys.argv[1:])
