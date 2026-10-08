"""N31: does registering the camera to the painted lines land players where the dots alone do not?

PROTOCOL, COMMITTED BEFORE THE LINE-REGISTERED CAMERAS ARE SCORED AGAINST ANY GROUND TRUTH.

STATUS OF THE DATA, said first
    N30's eight click sets have scored five maps and are no longer a held-out set. Everything here
    is a DEVELOPMENT result on ONE camera install (seven clips) plus one clip of another rig that
    the registration refuses. It can say whether the idea is worth a clean test; it cannot ship a map.
    Looked at before this was written, and disclosed: on 118575 only, three camera forms fitted to
    landmarks re-read BY HAND were scored at feet on N30's frames (equirect10 1.72 m, slope 0.962;
    docs/experiments/n31-line-registration.md §2). That probe is why this exists. The line-registered cameras
    themselves (model/validation/n31_linereg.py at 7386fc5, constants as committed, chosen on a synthetic
    panorama and on checks that read no ground truth) have been scored against nothing.

ARMS (n = 1 new configuration; three of N30's beside it for a like-for-like on the same frames)
    LINE     model/validation/out/stc2025/n31_linereg.json, one camera per clip, as `--fit` wrote it. Where
             the registration refused, LINE is REFUSED on that clip - never its click-only fallback.
    SHIP, SPH10, FISH11   n30_operator_maps.arms_for on the frozen clicks, as N30 scored them.
    CHAIN    ST2's own calibration (dist_eval._refmap), the dataset authors' map: a REFERENCE for
             what a careful human calibration reads on the same feet, not truth.

POPULATION
    position     MOT box feet on clip frames with f % 20 == 10, against the joined feed. N30 scored
                 f % 20 == 0; these are fresh frames of the SAME players 0.4 s later, so they refresh
                 frame noise and nothing else.
    scale, distance, HSR   `p21idn30` rows, stride 2 - the rows N30 used; there is no fresh version.
                 `oracle="drop"` removes rows with no IoU owner; that share is printed per clip.

METRICS
    A   median, p90, far-half median metres at feet, per clip and pooled over LINE's scored clips.
    B   n31_scale's chord slopes: all, x, y, far, near. Per clip and pooled.
    D   distance per covered metre over its FLOOR (true positions through the same fragments),
        SIGNED: ratio = (sum ours_m / sum gt_matched_m) / (floor's same sums). 1.00 is right, below
        is short. Per-clip sums are stored.
    H   HSR at MATCHED scale: the arm's map scaled by 1/slope_all per clip (dist_eval._ScaledCal),
        over the floor's HSR ratio. Reported, not ruled - N27's review showed HSR inherits the
        fragments' own over-read and the floor is the only fair comparator.

RULE, fixed now. LINE is SUPPORTED iff all of, on the clips where it is not refused:
    L1  its median at feet is below SPH10's AND below FISH11's on every clip;
    L2  its median at feet is within 0.5 m of CHAIN's on at least 6 of 7 clips (all but one);
    L3  its pooled slope is nearer 1 than SPH10's and FISH11's, AND its x, y, far and near pooled
        slopes lie within 0.05 of one another (a uniform scale is a camera that is right up to the
        reference; a non-uniform one is a wrong camera);
    L4  |signed distance ratio - 1| pooled is smaller than SPH10's and FISH11's.
    Anything else is NOT SUPPORTED. A refusal is reported as a refusal and scored by nothing.
    SUPPORTED means: build the clean test (a second rig, clicked once, this configuration frozen).
    It does not mean ship.

WHAT FAILING LOOKS LIKE, written first
    - LINE no better than SPH10: the dots were not the error and docs/experiments/n31-line-registration.md §2's probe was luck on one clip.
    - LINE better at feet but slopes still ~0.94 everywhere: the uniform under-read is the reference's.
    - LINE good near, bad far: the far touchline registered to the boards, not the paint.

    .venv/bin/python model/validation/n31_score.py --run     # ~35 min -> model/validation/out/stc2025/n31_score.json
    .venv/bin/python model/validation/n31_score.py --demo
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]

OUT = os.path.join(ROOT, "model/validation/out/stc2025/n31_score.json")
ARMS = ("LINE", "SHIP", "SPH10", "FISH11", "CHAIN")
RIVALS = ("SPH10", "FISH11")
NEAR_CHAIN_M, UNIFORM, PHASE = 0.5, 0.05, 10



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


def feet_pixels(seq, every_f=20, phase=PHASE):
    """dist_eval._feet_pixels on the frames with f % every_f == phase."""
    import dist_eval as D  # noqa: PLC0415
    feet = D.G_mot_feet(seq)
    uv, G = [], []
    for g, t in D.stc_gt(seq, D.STC_EVERY).items():
        for fr, xy in zip(t["f"].tolist(), t["xy"]):
            p = feet.get(fr, {}).get(int(g)) if fr % every_f == phase else None
            if p is not None:
                uv.append(p)
                G.append(xy)
    return np.array(uv, float), np.array(G, float)


def verdict(clip_median, pooled_slope, signed_distance):
    """The RULE on already-computed numbers. `clip_median[q][arm]` is None where an arm is refused."""
    qs = [q for q in clip_median if clip_median[q]["LINE"] is not None]
    lt = lambda a, b: a is not None and b is not None and a < b  # noqa: E731
    l1 = bool(qs) and all(lt(clip_median[q]["LINE"], clip_median[q][r]) for q in qs for r in RIVALS)
    near = sum(clip_median[q]["LINE"] <= clip_median[q]["CHAIN"] + NEAR_CHAIN_M for q in qs)
    l2 = bool(qs) and near >= len(qs) - 1
    s = pooled_slope
    parts = [s["LINE"][k] for k in ("x", "y", "far", "near")]
    l3 = (all(abs(s["LINE"]["all"] - 1) < abs(s[r]["all"] - 1) for r in RIVALS)
          and max(parts) - min(parts) <= UNIFORM)
    l4 = all(abs(signed_distance["LINE"] - 1) < abs(signed_distance[r] - 1) for r in RIVALS)
    return {"scored_clips": qs, "L1": l1, "L2": bool(l2), "near_chain": f"{near} of {len(qs)}", "L3": bool(l3),
            "L4": bool(l4), "verdict": "SUPPORTED" if l1 and l2 and l3 and l4 else "NOT SUPPORTED"}


def run():
    import camera_models as CM  # noqa: PLC0415
    import dist_eval as D  # noqa: PLC0415
    import n30_operator_maps as N  # noqa: PLC0415
    import n31_linereg as R  # noqa: PLC0415
    import n31_scale as S  # noqa: PLC0415
    seqs = N._check_freeze()
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "model/validation/n31_score.py", "model/validation/n31_linereg.py",
                       "model/validation/n31_scale.py"], cwd=ROOT, env=_git_env()).returncode:
        raise SystemExit("VOID: uncommitted change in N31's files - the rule that runs must be the committed one")
    reg = json.load(open(R.OUT))
    clip_median, clips, drop = {}, {}, {}
    errs = {a: [] for a in ARMS}
    slope_acc = {a: {} for a in ARMS}
    dist = {a: {"ours": 0.0, "gt": 0.0, "hsr": 0.0, "gt_hsr": 0.0} for a in (*ARMS[:-1], "FLOOR")}
    per_clip_sums = {}
    for q in seqs:
        payload = json.load(open(os.path.join(N.CLICKS, f"{q}.calibration.json")))
        cals, why, _rms = N.arms_for(payload, CM.T._wh(q))
        cals = {a: cals[a] for a in ("SHIP", "SPH10", "FISH11")}
        why = {a: why[a] for a in cals}
        cals["LINE"], why["LINE"] = CM.Fitted("equirect10", np.array(reg[q]["camera"], float)), reg[q]["refused"]
        cals["CHAIN"], why["CHAIN"] = N._Chain(q), None
        uv, G = feet_pixels(q)
        far = G[:, 1] < 34.0
        tid, f, g, uv2, G2, fps, drop[q] = S.chord_inputs(q, N.VARIANT)
        clip_median[q], clips[q], per_clip_sums[q], E, ACC = {}, {}, {}, {}, {}
        fr = D.evaluate_stc(q, N.VARIANT, oracle="position")
        per_clip_sums[q]["FLOOR"] = [fr["squad"]["ours_m"], fr["squad"]["gt_matched_m"], fr["speed"]["our_hsr_m"], fr["speed"]["gt_matched_hsr_m"]]
        for a in ARMS:
            if why[a]:
                clip_median[q][a], clips[q][a] = None, {"refused": why[a]}
                continue
            e = np.linalg.norm(np.asarray(cals[a].to_pitch(uv), float).reshape(-1, 2) - G, axis=1)
            ok = np.isfinite(e)
            acc = S.sums(tid, f, g, np.asarray(cals[a].to_pitch(uv2), float).reshape(-1, 2), G2, fps)
            sl = S.slopes(acc)
            E[a], ACC[a] = (e[ok], e[ok & far]), acc
            clip_median[q][a] = float(np.median(e[ok]))
            clips[q][a] = {"median": clip_median[q][a], "p90": float(np.percentile(e[ok], 90)),
                           "far_median": float(np.median(e[ok & far])), "feet": int(ok.sum()), "slope": sl}
            if a == "CHAIN":
                continue
            r = D.evaluate_stc(q, N.VARIANT, oracle="drop", our_cal=cals[a])
            m = D.evaluate_stc(q, N.VARIANT, oracle="drop", our_cal=D._ScaledCal(cals[a], 1.0 / sl["all"]))
            per_clip_sums[q][a] = [r["squad"]["ours_m"], r["squad"]["gt_matched_m"], m["speed"]["our_hsr_m"], m["speed"]["gt_matched_hsr_m"]]
        # pooled quantities only over the clips LINE is scored on, so every arm pools the same clips
        if why["LINE"]:
            print(q, "LINE REFUSED:", why["LINE"], {a: clip_median[q][a] and round(clip_median[q][a], 2) for a in ARMS}, flush=True)
            continue
        for a in ARMS:
            errs[a].append(E[a])
            for k, v in ACC[a].items():
                p = slope_acc[a].setdefault(k, [0.0, 0.0, 0])
                slope_acc[a][k] = [p[0] + v[0], p[1] + v[1], p[2] + v[2]]
        for a in dist:
            for k, x in zip(("ours", "gt", "hsr", "gt_hsr"), per_clip_sums[q][a]):
                dist[a][k] += x
        print(q, "drop", round(drop[q], 4), {a: round(clip_median[q][a], 2) for a in ARMS}, flush=True)
    pooled = {a: {"median": float(np.median(np.concatenate([x[0] for x in errs[a]]))),
                  "p90": float(np.percentile(np.concatenate([x[0] for x in errs[a]]), 90)),
                  "far_median": float(np.median(np.concatenate([x[1] for x in errs[a]])))} for a in ARMS}
    pooled_slope = {a: S.slopes(slope_acc[a]) for a in ARMS}
    floor_d, floor_h = dist["FLOOR"]["ours"] / dist["FLOOR"]["gt"], dist["FLOOR"]["hsr"] / dist["FLOOR"]["gt_hsr"]
    signed = {a: (dist[a]["ours"] / dist[a]["gt"]) / floor_d for a in dist if a != "FLOOR"}
    hsr_matched = {a: (dist[a]["hsr"] / dist[a]["gt_hsr"]) / floor_h for a in dist if a != "FLOOR"}
    v = verdict(clip_median, pooled_slope, signed)
    res = {"seqs": seqs, "clips": clips, "pooled": pooled, "pooled_slope": pooled_slope, "drop_share": drop,
           "signed_distance_over_floor": signed, "hsr_matched_scale_over_floor": hsr_matched,
           "floor": {"distance": floor_d, "hsr": floor_h}, "per_clip_sums": per_clip_sums,
           "sums_are": "ours_m, gt_matched_m, hsr_m at matched scale (floor: own scale), gt_matched_hsr_m",
           "verdict": v, "linereg_sha256": hashlib.sha256(open(R.OUT, "rb").read()).hexdigest(),
           "rule_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()}
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: res[k] for k in ("pooled", "pooled_slope", "signed_distance_over_floor",
                                          "hsr_matched_scale_over_floor", "verdict")}, indent=1))
    print(f"-> {OUT}")


def demo():
    flat = lambda v: {"all": v, "x": v, "y": v, "far": v, "near": v}  # noqa: E731
    cm = {f"c{i}": {"LINE": 1.4, "SPH10": 3.0, "FISH11": 2.8, "SHIP": 7.5, "CHAIN": 1.2} for i in range(7)}
    cm["odd"] = {"LINE": None, "SPH10": 1.5, "FISH11": 2.4, "SHIP": 11.0, "CHAIN": 4.6}       # refused: scored by nothing
    sl = {"LINE": flat(0.97), "SPH10": flat(0.94), "FISH11": flat(0.94)}
    sd = {"LINE": 0.98, "SPH10": 0.94, "FISH11": 0.94}
    v = verdict(cm, sl, sd)
    assert v["verdict"] == "SUPPORTED" and len(v["scored_clips"]) == 7, v
    cm["c3"]["LINE"] = 3.1                                                                  # loses one clip to a rival
    assert not verdict(cm, sl, sd)["L1"]
    cm["c3"]["LINE"] = 1.9                                                                  # one clip off the chain is allowed,
    assert verdict(cm, sl, sd)["L2"]
    cm["c4"]["LINE"] = 1.9                                                                  # two are not
    assert not verdict(cm, sl, sd)["L2"]
    cm["c3"]["LINE"] = cm["c4"]["LINE"] = 1.4
    assert not verdict(cm, {**sl, "LINE": {**flat(0.97), "far": 0.90}}, sd)["L3"]           # nearer 1 but not uniform
    assert not verdict(cm, sl, {**sd, "LINE": 1.07})["L4"]                                  # signed: 7% long loses to 6% short
    print("n31 score demo OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g_ = ap.add_mutually_exclusive_group(required=True)
    g_.add_argument("--run", action="store_true")
    g_.add_argument("--demo", action="store_true")
    a_ = ap.parse_args()
    (run if a_.run else demo)()
