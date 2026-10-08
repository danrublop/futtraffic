"""N54: cubic or column term - the near touchline and the goal lines, held out, decide (N39 section 5, test (a)).

WHY  (n39-column-term (not published) section 5; n38-kappa-free (not published) section 3)
    N39 left two descriptions of the far-touchline effect standing: a cubic in the vertical law (kappa) and a
    two-lens column term (beta, d). Both were fitted to EVERY line, so each arm's agreement with the near
    touchline was in-sample, and the arm with more parameters wins that by construction. The two models
    disagree about where the correction lives: the cubic is a function of the ROW, the column term of the
    COLUMN, largest toward the frame edges and opposite in sign on the near touchline (N39 section 4: -2.9 to
    -5.2 px at the halfway column). So: fit each arm WITHOUT the near touchline and the two goal lines, then ask
    which one's camera lands on that paint. It is a held-out test, which in-sample fitting in N39 was not.

WRITTEN BEFORE ANY CLIP WAS FITTED BY THIS FILE
    Population: the seven clips of one install every calibration page since N30 has read (N36.SEQS), their
    40-frame medians, each clip's N36 paint-rule camera and length (n36_after.json), pitch 68 m wide.
    Protocol, identical to N39's except `leave_out`: paint rule (`native=`), radii AUTO_RADII[2:] = (8, 5, 3),
    three passes of `register`, square prior 1.0, soft-L1 at 2 px.
    HELD OUT: touch_near, goal_l, goal_r - searched, never fitted. Fitted: far touchline, halfway, boxes, circle.
    ARMS (n = 3, on top of the >= 34 configurations N36 and N39 count on these clips):
        rigid     ten-number spherical camera, kappa 0                   (control)
        kappa     the same, kappa FIXED at -0.34                          (N38 section 3: "if the next pre-registered
                                                                           test carries a fixed kappa, -0.34 is what
                                                                           these clips say, not -0.40")
        column    + beta, d free (N39's term, u_c 2048), kappa 0
    MEASURE: after the fit, `search` at radius 8 px with the arm's camera; signed normal residual per sample,
    + = the paint lies INSIDE the model's line (the normal points out of the pitch on perimeter samples).
    Near touchline: median per seventh of x in [4, 101] m (N39's bins, > 10 hits per bin), and the RMS of
    those bins, px. Goal lines: median |residual| over both, px (reported, not ruling).
    RULE:
        - A clip is scored for an arm when its fit is not refused (register returns the START camera on a
          refusal, so a refused fit must not be read) and at least 5 of 7 near-touchline bins are filled.
          Unscored clips are reported by name.
        - The test is VOID if rigid is scored on fewer than 5 clips (the camera cannot be fitted without the
          near line, and nothing below means anything).
        - Per clip, kappa vs column on the near-touchline bin RMS: the lower wins if lower by >= 0.25 px
          (half the search step); otherwise a tie.
        - CUBIC if kappa wins on >= 5 clips; COLUMN if column wins on >= 5; otherwise UNRESOLVED, and the
          next discriminator is a second install (N39 section 5 (b)), not another arm here.
        - Reported beside it, never ruling: each arm against rigid; goal-line median |px|; held-out metres.
    One install, seven clips all previously seen: this SELECTS which far-side term a pre-registered test on
    unseen footage should carry. It confirms nothing and nothing ships from it.

    .venv/bin/python model/validation/n54_near_profile.py            # self-check (no clip is read)
    .venv/bin/python model/validation/n54_near_profile.py --run      # the three arms -> out/stc2025/n54_near.json
    .venv/bin/python model/validation/n54_near_profile.py --show     # every number docs/experiments/n54-near-profile.md quotes
"""
import dataclasses
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
import n39_column_term as N39  # noqa: E402 - importing it patches autocal's _fwd/_inv with the column term

A, SEQS, STC = N39.A, N39.SEQS, N39.STC
OUT = os.path.join(STC, "n54_near.json")
HELD = ("touch_near", "goal_l", "goal_r")
ARMS = {"rigid": (False, 0.0), "kappa": (False, -0.34), "column": (True, 0.0)}
TIE_PX, MIN_BINS, WIN = 0.25, 5, 5


def held_residuals(bgr, cam, P):
    """-> {group: (x metres, y metres, signed px, metres per px)} for the held-out lines; + = paint inside."""
    rid, gr, nat = A.ridge_map(bgr), A.grass_map(bgr), A.native_map(bgr)
    X, T, grp = A.model_samples(P)
    out = np.array([A.OUTWARD.get(g, (0.0, 0.0)) for g in grp])
    hit, d, nrm = A.search(rid, cam, X, T, A.AUTO_RADII[2], grass=gr, outward=out, native=nat)
    r = ((N39.fwd(cam, X) - d) * nrm).sum(1)
    m = np.linalg.norm(N39.inv(cam, d + nrm) - N39.inv(cam, d), axis=1)
    return {g: (X[s, 0], X[s, 1], r[s], m[s]) for g in HELD for s in [hit & (grp == g)]}


def score(near_x, near_r):
    bins = N39.binned(near_x, near_r)
    ok = np.isfinite(bins)
    return bins, (float(np.sqrt(np.mean(np.square(np.asarray(bins)[ok])))) if ok.sum() >= MIN_BINS else None)


def verdict(rms):
    """rms: {arm: {clip: float | None}} -> (verdict, kappa wins, column wins, ties, scored clips)."""
    if sum(v is not None for v in rms["rigid"].values()) < MIN_BINS:
        return "VOID", 0, 0, 0, []
    both = [q for q in rms["kappa"] if rms["kappa"][q] is not None and rms["column"].get(q) is not None]
    kw = sum(rms["kappa"][q] <= rms["column"][q] - TIE_PX for q in both)
    cw = sum(rms["column"][q] <= rms["kappa"][q] - TIE_PX for q in both)
    v = "CUBIC" if kw >= WIN else "COLUMN" if cw >= WIN else "UNRESOLVED"
    return v, kw, cw, len(both) - kw - cw, both


def run():
    from pitch import Pitch  # noqa: PLC0415
    after = json.load(open(os.path.join(STC, "n36_after.json")))
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for arm, (column, kfix) in ARMS.items():
        if arm in res:
            continue
        A.VERT_CUBIC, R = kfix, {}
        for q in SEQS:
            bgr = cv2.imread(os.path.join(STC, f"{q}.median40.png"))
            P = dataclasses.replace(Pitch(105.0, 68.0), length=float(after[q]["length_m"]))
            rid, gr, nat, empty = A.ridge_map(bgr), A.grass_map(bgr), A.native_map(bgr), np.zeros((0, 2))
            cam = N39.start(after[q]["camera"], column, False)
            for _ in range(3):
                cam, rep = A.register(rid, cam, P, empty, empty, grass=gr, square=1.0, radii=A.AUTO_RADII[2:],
                                      native=nat, leave_out=HELD)
                if rep["refused"]:
                    break
            h = held_residuals(bgr, cam, P)
            bins, rms = score(h["touch_near"][0], h["touch_near"][2])
            goal = np.r_[h["goal_l"][2], h["goal_r"][2]]
            R[q] = {"camera": cam.tolist(), "refused": rep["refused"], "fit_rms_px": round(rep["line_rms_px"], 3),
                    "near_bins_px": [round(v, 2) for v in bins], "near_rms_px": None if rep["refused"] or rms is None else round(rms, 3),
                    "near_hits": [int(len(h["touch_near"][2])), int((np.array(A.model_samples(P)[2]) == "touch_near").sum())],
                    "near_median_m": round(float(np.median(h["touch_near"][2] * h["touch_near"][3])), 3) if len(h["touch_near"][2]) else None,
                    "goal_hits": int(len(goal)), "goal_median_abs_px": round(float(np.median(np.abs(goal))), 3) if len(goal) else None}
            if column:
                R[q]["beta_px"], R[q]["d_px"] = float(cam[10]), 1000.0 * float(cam[11])
            print(arm, q, {k: v for k, v in R[q].items() if k != "camera"}, flush=True)
        A.VERT_CUBIC = 0.0
        res[arm] = R
        json.dump(res, open(OUT, "w"), indent=1)


def show():
    r = json.load(open(OUT))
    print("== near touchline held out: signed px per seventh of x in [4, 101] m (+ = paint inside the model's line), bin RMS")
    for q in SEQS:
        for arm in ARMS:
            x = r[arm][q]
            print(f"{arm:7s} {q} " + " ".join(f"{v:+6.2f}" for v in x["near_bins_px"])
                  + f"  rms {x['near_rms_px']}  hits {x['near_hits'][0]}/{x['near_hits'][1]}  median {x['near_median_m']} m"
                  + f"  goal |px| {x['goal_median_abs_px']} ({x['goal_hits']})  fit rms {x['fit_rms_px']}"
                  + (f"  REFUSED {x['refused']}" if x["refused"] else "")
                  + (f"  beta {x['beta_px']:+.2f} d {x['d_px']:.0f}" if "beta_px" in x else ""))
    rms = {arm: {q: r[arm][q]["near_rms_px"] for q in SEQS} for arm in ARMS}
    for arm in ARMS:
        un = [q for q in SEQS if rms[arm][q] is None]
        vals = [v for v in rms[arm].values() if v is not None]
        print(f"{arm:7s} scored {len(vals)} of 7" + (f" (unscored {', '.join(un)})" if un else "")
              + (f", near bin RMS mean {np.mean(vals):.3f} px" if vals else ""))
    for arm in ("kappa", "column"):
        both = [q for q in SEQS if rms[arm][q] is not None and rms["rigid"][q] is not None]
        print(f"{arm:7s} vs rigid (reported, not ruling): lower by >= {TIE_PX} px on {sum(rms[arm][q] <= rms['rigid'][q] - TIE_PX for q in both)} of {len(both)}")
    v, kw, cw, ties, both = verdict(rms)
    print(f"RULE: kappa wins {kw}, column wins {cw}, ties {ties}, of {len(both)} clips scored for both -> {v}")


def selfcheck():
    none = dict.fromkeys(SEQS)
    assert verdict({"rigid": none, "kappa": none, "column": none})[0] == "VOID"
    base = dict.fromkeys(SEQS, 1.0)
    assert verdict({"rigid": base, "kappa": dict.fromkeys(SEQS, 0.5), "column": base})[:3] == ("CUBIC", 7, 0)
    assert verdict({"rigid": base, "kappa": base, "column": dict.fromkeys(SEQS, 0.5)})[:3] == ("COLUMN", 0, 7)
    assert verdict({"rigid": base, "kappa": base, "column": dict.fromkeys(SEQS, 0.8)})[0] == "UNRESOLVED"   # 0.2 px: a tie
    four = {q: (0.5 if i < 4 else 1.0) for i, q in enumerate(SEQS)}
    assert verdict({"rigid": base, "kappa": four, "column": base})[:2] == ("UNRESOLVED", 4)                # 4 of 7 is not 5
    x = np.arange(4, 101, 0.5)
    assert score(x, np.where(x < 75, 2.0, np.nan))[1] is not None and score(x, np.where(x < 30, 2.0, np.nan))[1] is None
    assert abs(score(x, np.full_like(x, -3.0))[1] - 3.0) < 1e-9
    print("n54 selfcheck ok: VOID, CUBIC, COLUMN, a 0.2 px tie, 4 of 7, and the 5-bin floor all behave")


if __name__ == "__main__":
    {"--run": run, "--show": show}.get(sys.argv[1] if len(sys.argv) > 1 else "", selfcheck)()
