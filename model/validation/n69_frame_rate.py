"""N69: does telling C-BIoU the true sampled rate improve tracking at what ships? Protocol first.

WHY
    model/engine/tracking.py builds trackers.CBIoUTracker with no kwargs, so frame_rate is the library's 30 while
    app/server/jobs.py feeds 25 fps clips at stride 2, one update() per detected frame. docs/reference/detection-tracking.md
    (Known limits, #tracker) lists this as OPEN, no effect claimed. This is that experiment.

WHAT frame_rate DOES IN THE INSTALLED LIBRARY (trackers 2.6.0, read, not assumed)
    core/base.py _compute_maximum_frames_without_update: max(1, ceil(frame_rate / 30 * lost_track_buffer)) - the lost
    buffer in UPDATES. That is the only place a fixed-rate caller feels it: the adapter passes no timestamp, so
    _predict_timing returns frame_step 1.0 with frame_rate None (Kalman F and Q unchanged), and _lost_track_time_budget
    returns None (the seconds budget, lost_track_buffer / 30, is unused).

    @M default 30,at_12.5 13,1.0 :: .venv/bin/python -c "import trackers;a=trackers.CBIoUTracker();b=trackers.CBIoUTracker(frame_rate=12.5);print('default',a.maximum_frames_without_update,'at_12.5',b.maximum_frames_without_update,a.maximum_time_without_update)"
       held fixed: installed trackers, CBIoUTracker defaults; A0's and A1's lost budget in updates, and the unused
       seconds budget. n=1, no sweep.
    @D 2.4 s, 1.04 s :: 30 / 12.5 ; 13 / 12.5   [the tag above; 12.5 = 25 fps / stride 2]

    So A1 SHORTENS how long a lost track survives, from 2.4 s to 1.04 s of video - the 1 s the library's default
    intends. A1 is exactly lost_track_buffer=13 at frame_rate 30; the question is the buffer, and the page says so.

ARMS (n=2 configurations: one baseline, one candidate; nothing else is tried)
    A0  shipped: CBIoUTracker() defaults.                                -> {seq}.p69a0.tracks.json
    A1  frame_rate = fps / stride = 12.5, lost_track_buffer LEFT at 30.  -> {seq}.p69a1.tracks.json
    No third arm. The only parameter frame_rate couples to is lost_track_buffer; the arm that keeps the buffer's
    SECONDS constant (buffer 72 at frame_rate 12.5 -> ceil(30) = 30 updates) is A0 bit for bit. minimum_consecutive_frames
    and track_activation_threshold do not read frame_rate and are out of scope (a separate question).

HELD FIXED
    The stored shipped-config detection caches {seq}.p35.dets.cache.json (2026-09-17; CoreML, conf 0.5, overlap 0.25,
    tile 1024, prior ['pitch'], stride 2, ball False, stitch True - read from each .p35.tracks.json config), replayed
    COMPLETE through model/engine/track.run: no detector, no prior re-filtering. The frozen N34 zero-click map
    {seq}.p35.calibration.json. Stitch on, team assignment, everything in track.run as the code stands. ball=False,
    because track.run refuses a ball=False cache under ball=True (the same constraint check.py track lives with).
    The ONLY difference between arms is tracking._RF_KINDS["cbiou"], patched in the child process.
    A0 is re-run, NOT read from the stored .p35.tracks.json: the tracker has changed since (empty frames reach
    update() from 2026-09-28), so the stored tracks are not today's baseline. Whether A0 equals them is printed.

POPULATION AND SPLIT (N41's, docs/experiments/n41-conf.md; the seven STC2025 clips with a .p35 cache; no clip dropped)
    DEV  = 118575, 118576, 117093 - looked at freely.   HELD = 118577, 118578, 128057, 128058 - scored ONCE.
    HELD is held out FOR THIS KNOB: no one has varied frame_rate on it; its A0 numbers were seen by N41/N42.
    Tuning: st2_product.TUNED = 128057 + 132831 is where the research .c03s config was tuned; the shipped C-BIoU
    constants were never tuned anywhere (no kwargs). 132831 has no .p35 cache and is not in the population. 128057 IS
    in HELD, so the rule must also hold without it (clause e). Every clip is reported, per clip.

METRICS, per clip, on the stitched tracks (the stage the product ships), GT staged at stride 2 (st2_mot.stage_gt(every=2))
    primary   HOTA.   secondary  DetA, AssA, IDSW, distinct track ids.
    not run   contamination: no scorer is wired at this config across these clips (contam_eval is a 118575 research
              harness); stated, not substituted. check.py track's golden sha WILL change under A1 by construction;
              shipping A1 means check.py track --rebless with this page as the reason. Not a metric.

DETERMINISM
    check.py's TRACK docstring records n=2 replays bit-identical. Here: A0 on 118575 is replayed twice; the two
    detections sha256 must match or the run is VOID. With deterministic replays there is no run-to-run noise to
    clear; the threshold below is a product-decision bar, N41's, not a noise band. Four clips give no significance
    test worth the name (a sign test on 4 cannot reach 0.05), so clause (b) is a consistency rule, not a p-value.

RULE, fixed now. Deltas are A1 - A0 on HELD.
    SHIP A1 iff all of:
      (a) mean HOTA delta >= +0.010
      (b) HOTA delta > 0 on at least 3 of 4 HELD clips
      (c) no HELD clip has HOTA delta < -0.010
      (d) mean IDSW(A1) <= 1.15 x mean IDSW(A0)
      (e) mean HOTA delta over 118577, 118578, 128058 (128057 dropped) >= +0.010
    WORSE        iff mean HOTA delta <= -0.010.
    NO EFFECT    iff |mean HOTA delta| < 0.010.
    NOT SUPPORTED otherwise (the gain is there but (b)-(e) fail). Only SHIP changes the product.
    n=1 candidate value, derived from the clip's frame rate and the job's stride, not chosen from any sweep.

RUN (phase 2; ~7 clips x 2 arms of cache replay plus TrackEval - an estimate, not timed)
    caffeinate -i -s .venv/bin/python model/validation/n69_frame_rate.py --build dev
    caffeinate -i -s .venv/bin/python model/validation/n69_frame_rate.py --score dev
    caffeinate -i -s .venv/bin/python model/validation/n69_frame_rate.py --build held
    caffeinate -i -s .venv/bin/python model/validation/n69_frame_rate.py --score held   # ONCE; refuses if uncommitted
    .venv/bin/python model/validation/n69_frame_rate.py --show

Page: n69-tracker-frame-rate (not published)
"""
import hashlib
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
FPS, STRIDE = 25.0, 2
ARMS = {"a0": {}, "a1": {"frame_rate": FPS / STRIDE}}


def _sha(p):
    with open(p) as fh:
        return hashlib.sha256(json.dumps(json.load(fh)["detections"], sort_keys=True).encode()).hexdigest()[:12]


def replay(seq, arm, out=None):
    import st2_product as ST  # noqa: PLC0415
    out = out or os.path.join(STC, f"{seq}.p69{arm}.tracks.json")
    if os.path.exists(out):
        return out
    p = lambda s: os.path.join(STC, f"{seq}{s}")  # noqa: E731
    code = ("import sys; sys.path.insert(0, 'model/engine'); import tracking, track; "
            f"tracking._RF_KINDS['cbiou'] = {ARMS[arm]!r}; "
            f"track.run({f'{ST.CLIPS}/{seq}.mp4'!r}, {out!r}, calib={p('.p35.calibration.json')!r}, "
            f"detector='coreml', stride={STRIDE}, ball=False, dets_cache={p('.p35.dets.cache.json')!r})")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
    return out


def build(which):
    for q in (HELD if which == "held" else DEV):
        for arm in ARMS:
            replay(q, arm)
    if which == "dev":   # determinism precheck
        again = replay("118575", "a0", out=os.path.join(STC, "118575.p69a0r.tracks.json"))
        a, b = _sha(os.path.join(STC, "118575.p69a0.tracks.json")), _sha(again)
        if a != b:
            raise SystemExit(f"VOID: A0 replay not deterministic ({a} vs {b})")
        print("A0 replay deterministic:", a)


def score(which):
    import protocol  # noqa: PLC0415
    import st2_mot as M  # noqa: PLC0415
    import verify_claims as V  # noqa: PLC0415
    seqs = HELD if which == "held" else DEV
    me = "model/validation/n69_frame_rate.py"
    if which == "held" and subprocess.run(["git", "diff", "--quiet", "HEAD", "--", me], cwd=ROOT, env=V.git_env()).returncode:
        raise SystemExit("VOID: the held-out run must be made by the committed protocol")
    res = {"protocol": protocol.protocol_hash(__doc__)}
    for arm in ARMS:
        M.VARIANT = f".p69{arm}"
        M.stage_gt(seqs, every=STRIDE)
        M.stage_pred(seqs)
        te = M.run(seqs)
        res[arm] = {}
        for q in seqs:
            f = os.path.join(STC, f"{q}.p69{arm}.tracks.json")
            dets = json.load(open(f))["detections"]
            res[arm][q] = {**{k: te[q][k] for k in ("HOTA", "DetA", "AssA", "IDSW")},
                           "ids": len({r["id"] for r in dets}), "sha": _sha(f)}
    res["a0_equals_stored_p35"] = {q: res["a0"][q]["sha"] == _sha(os.path.join(STC, f"{q}.p35.tracks.json")) for q in seqs}
    json.dump(res, open(os.path.join(STC, f"n69_{which}.json"), "w"), indent=1)
    show()


def verdict(r, seqs=HELD):
    d = {q: r["a1"][q]["HOTA"] - r["a0"][q]["HOTA"] for q in seqs}
    m = float(np.mean(list(d.values())))
    no57 = float(np.mean([d[q] for q in seqs if q != "128057"]))
    idsw = lambda a: float(np.mean([r[a][q]["IDSW"] for q in seqs]))  # noqa: E731
    ok = [m >= 0.010, sum(v > 0 for v in d.values()) >= 3, min(d.values()) >= -0.010,
          idsw("a1") <= 1.15 * idsw("a0"), no57 >= 0.010]
    if all(ok):
        return "SHIP A1", ok
    if m <= -0.010:
        return "WORSE", ok
    if abs(m) < 0.010:
        return "NO EFFECT", ok
    return "NOT SUPPORTED", ok


def show():
    for which in ("dev", "held"):
        p = os.path.join(STC, f"n69_{which}.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        seqs = list(r["a0"])
        print(f"== {which.upper()} (protocol {r['protocol']}) ==")
        for q in seqs:
            print(q, " ".join(f"{k} {r['a0'][q][k]}->{r['a1'][q][k]}" for k in ("HOTA", "DetA", "AssA", "IDSW", "ids")),
                  "A0==stored .p35:", r["a0_equals_stored_p35"][q])
        m = lambda a, k: float(np.mean([r[a][q][k] for q in seqs]))  # noqa: E731
        print("MEAN " + " ".join(f"{k} {m('a0', k):.4f}->{m('a1', k):.4f}" for k in ("HOTA", "DetA", "AssA", "IDSW", "ids")))
        if which == "held":
            v, ok = verdict(r)
            print("RULE:", dict(zip("abcde", ok)), "->", v)


def demo():
    mk = lambda h0, h1, i0=100, i1=100: {"a0": {q: {"HOTA": h0[k], "IDSW": i0} for k, q in enumerate(HELD)},  # noqa: E731
                                          "a1": {q: {"HOTA": h1[k], "IDSW": i1} for k, q in enumerate(HELD)}}
    base = [0.36, 0.37, 0.35, 0.38]
    assert verdict(mk(base, [x + 0.02 for x in base]))[0] == "SHIP A1"
    assert verdict(mk(base, [x - 0.02 for x in base]))[0] == "WORSE"
    assert verdict(mk(base, base))[0] == "NO EFFECT"
    assert verdict(mk(base, [x + 0.02 for x in base], i1=200))[0] == "NOT SUPPORTED"          # (d)
    assert verdict(mk(base, [base[0] + .01, base[1] + .01, base[2] + .08, base[3] - .005]))[0] == "NOT SUPPORTED"  # (e): 128057 carries it
    print("n69 rule ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--build" in a:
        build(a[a.index("--build") + 1])
    elif "--score" in a:
        score(a[a.index("--score") + 1])
    elif "--demo" in a:
        demo()
    else:
        show()
