"""The ball as a PATH, not a per-frame pick: Viterbi over N48's cached candidates.

WHY THIS RUNS
    N48 (n48-ball-detector-ab (not published)) measured, over six TeamTrack clips / 1800
    frames / conf 0.05 / 20 px gate, that the fine-tuned detector puts the true
    ball in the candidate list on 0.7150 of frames while `ball.BallPicker`
    returns it on 0.4056. That gap - 31% of every frame, with the answer already
    on the table - is the picker's, not the detector's.

WHAT THE SHIPPED PICKER ALREADY DOES WITH TIME, because "per-frame picker" is
not quite true and the scoping had to check before the rule could be written:
    `BallPicker.update` is memoryless IN ITS OUTPUT - the answer does not depend
    on the previous answer, and ball.py's self-check asserts exactly that. But it
    is NOT memoryless: `Scenery` keeps a 60-frame deque of EVERY candidate
    position and rejects any candidate that had some candidate within 8 px of it
    on >= 90% of the last >= 25 frames. So the picker uses temporal information,
    once, as a NEGATIVE gate (persistent => paint), and never as evidence FOR a
    candidate. After the three gates it is argmax(conf) within the frame.

WHERE THE 31% ACTUALLY GOES (`--why`, run before this rule was written, and it
moved the design)
    Over the 557 frames where `oracle` is right and the shipped picker is not -
    0.7150 - 0.4056 of 1800, N48's own gap - the loss splits three ways, each
    frame charged to the FIRST gate that deletes the true candidate:

        physical-size gate killed it               234   42%
        it survived every gate and lost argmax     231   41%
        scenery gate killed it                      92   17%
        on-the-pitch gate killed it                  0    0%

    and 252 of the 557 are one clip. `soccer_top test 1710-1740`'s ball travels
    0.9 px IN TOTAL over the 300 scored frames - it is a dead ball carried
    forward, the same degeneracy ball_cmp excluded 1680-1710 for, and both
    `Scenery` and the size gate delete a still object sitting on the pitch.
    `1740-1770` is nearly the same (101.9 px of travel, still on 243 of 299
    frame steps). So a THIRD of this population is a ball that does not move,
    which is also where N48's "one clip goes to 300/300" comes from.

    That is what a motion-continuity rule can and cannot touch. It cannot
    recover a candidate a gate already deleted (234 + 92 = 326 frames). It can
    only re-pick among what survives, so its whole headroom is the 231 frames
    where the true ball is gated IN and simply quieter than a decoy:
    true-candidate conf p50 0.078-0.162 against the winner's 0.272-0.494.

SECOND RULE, ARM D, FIXED BEFORE THAT RUN (committed 2026-09-23, nothing scored)
    N49 ruled `path` NOT SUPPORTED on arm B: 0.3856 against a 0.4556 bar. This is
    the same question asked of a different candidate list, and it is asked because
    the population changed in the way that matters, not because the answer was
    disliked. Arm D (N52's clean checkpoint, docs/experiments/n52-clean-ball-data.md §7b) puts
    the true ball in the list on 0.8822 of frames where arm B managed 0.7150, and
    `--why --arm D` splits its 741 lost frames as: scenery 295, size 37, pitch 0,
    and 409 that survive every gate and lose argmax(conf).

    population  ball_cmp.CLIPS, first 300 frames, off the COMMITTED `.cmpD`
                caches. 1800 frames, the same list arm B was scored on, so the
                two arms' path numbers compare to each other.
    baseline    `shipped` on arm D = 0.4706, not N48's 0.4056. The bar moves with
                it: a bar copied from another arm measures the arm, not the change.
    PASS        pooled `path` >= 0.5206 (shipped + 0.05) on the six clips together.
    HEADROOM, STATED FIRST  409/1800 = 0.2272 survive the gates, so the reachable
                ceiling for a re-pick is 0.6978 and +0.05 is 22% of it. Anything
                above 0.6978 is a bug in this file, not a result. The 295 frames
                the SCENERY gate deletes and the 37 the size gate deletes are NOT
                reachable by re-picking and are not part of this rule - 273 of the
                295 are one dead-ball clip, which is a gate question and gets its
                own pre-registration or none at all.
    VOID        (a)-(d) exactly as the arm-B rule below, evaluated on arm D.
    n           n=1. SIGMA_PX, MAX_STEP_PX, P_MISS and the cap are UNCHANGED from
                the arm-B run - not re-fitted to this population. Re-fitting them
                here would make this a sweep over a result already seen, which is
                what N50 and N51 both lost to.
    IF IT PASSES it is a statement about arm D's candidate list and nothing else;
                arm B's NOT SUPPORTED stands as the arm-B answer.

RULE, FIXED BEFORE THE SCORED RUN (committed with this file, nothing run yet)
    population  identical to ball_cmp's: the six clips of `ball_cmp.CLIPS`, the
                first 300 frames of each, off the COMMITTED `.cmpB` candidate
                caches (fine-tuned nano @640, conf 0.05). No re-detection. That
                is 1800 frames of the 4949 annotated in these clips' CSVs, the
                same subset
                N48 reports on, so these numbers compare to N48's and to
                nothing else.
    metric      pooled share of gt-present frames whose answer is within
                GATE_PX = 20 px, summed over clips exactly as ball_cmp pools -
                i.e. the number N48 prints as `shipped` 0.4056.
    arms        `shipped`  ball.BallPicker, untouched, as N48 scored it.
                `path`     THE ONE UNDER TEST: the identical gates (scenery,
                           on-pitch, physical size, each clip's own calibration
                           or none), then a Viterbi over the surviving
                           candidates instead of argmax(conf).
                `oracle`   the ceiling, unchanged.
                `centre`   the control, unchanged.
    PASS        pooled `path` >= pooled `shipped` + 0.05, i.e. >= 0.4556, on the
                six clips together. The bar is N48's own. HEADROOM IS STATED
                FIRST so the bar can be read against it: re-picking inside the
                gates can reach at most shipped + 231/1800 = 0.5339, so +0.05 is
                39% of what is available, and anything above 0.5339 is a bug in
                this file, not a result.
    VOID        (a) `centre` scores over 1% (ball_eval's rule);
                (b) the arms do not score the identical frame list (asserted);
                (c) `_gates` here stops reproducing `BallPicker` - `--score`
                    asserts that gates+argmax(conf) equals `shipped` on every
                    one of the 1800 frames, so a drift in ball.py voids the
                    comparison instead of silently changing its baseline;
                (d) model/engine/ball.py is modified on this branch. It is not:
                    the shipped picker must keep scoring 0.4056 here.
    n           n=1 configuration. The four constants below are set once, from
                physics and from the candidate geometry, and are NOT swept. They
                were chosen after reading the GT speed distribution of these six
                clips (p50 3-4 px/frame, max 16.1), so they are informed by the
                population - declared, not hidden. Both are deliberately loose
                bounds; the failure they could flatter (a decoy chain) is not one
                a loose speed limit can create.
                IF THE PRIMARY FAILS, a sensitivity sweep may be run as
                diagnosis; any number out of it carries its own n= and cannot be
                quoted as a headline. Rule 6.
    also report `wrong` and `silent` per arm. A picker that answers more often
                buys `correct` with `wrong`, and a gain that is only traded
                silence is not a gain.
    exploratory `path_nosc`: the same Viterbi with the SCENERY gate off, i.e.
                asking whether a path model can replace the gate rather than sit
                behind it. Reported, NEVER ruled on: it is a second
                configuration and the rule above is decided on `path` alone.

WHY A PATH MODEL COULD WORK NOW WHEN THREE TEMPORAL TERMS FAILED IN AUGUST
    ball.BallPicker's docstring records a soft speed prior (15.4% -> 5.0%) and a
    hard reachability gate (-> 4.4%), and gives the reason: both anchor this
    frame to the PREVIOUS PICK, which was wrong most of the time, so continuity
    entrenched the error. Two things differ here and both are load-bearing.
    First, Viterbi is a global optimum over the whole clip - it has no previous
    pick to be anchored to, and one confident stretch pays for the frames around
    it in either direction. Second, the per-frame evidence is no longer mostly
    wrong: the candidate is present on 0.7150 of frames, which is the condition
    that same docstring named ("a temporal model becomes correct here the moment
    the per-frame answer is right more often than not, AND NOT BEFORE").

    The honest counter-argument, written down before the run so it cannot be
    invented afterwards: smoothness does not favour a ball. A static decoy is
    maximally smooth, and a player's head is both smooth and louder than a ball.
    The scenery gate removes the persistent ones, so what the path has to beat is
    the transient decoys and the heads. If it loses to them, this file says so.

    .venv/bin/python model/validation/ball_path.py --why     # the split above
    .venv/bin/python model/validation/ball_path.py --score   # shipped vs path
    .venv/bin/python model/validation/ball_path.py --demo    # self-check
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ball_cmp as C  # noqa: E402
import ball_eval as B  # noqa: E402

sys.path.insert(0, os.path.join(B.ROOT, "model", "engine"))
import ball as ballmod  # noqa: E402

#: Which candidate cache this scores. N49 ruled on `.cmpB` - the DFL fine-tune -
#: and the ruling numbers on this page are that arm's. `--arm D` repoints it at
#: N52's clean checkpoint, whose candidate list is a DIFFERENT population: it
#: misses the ball on 11.8% of frames where B missed 28.5%, so the headroom a
#: re-pick can reach is not the 231/1800 measured below. Any number off another
#: arm carries that arm's name or it is not comparable to N49's.
SUFFIX = ".cmpB"

#: One frame of ball travel, in pixels, that costs nothing. The GT ball on the
#: moving clips here runs p50 3-4 px/frame and p90 ~10; a Gaussian at 12 px
#: leaves ordinary play unpenalised and charges ~8 nats at the 60 px cap.
SIGMA_PX = 12.0
#: Hard reachability, ~4x the fastest step these clips contain (16.1 px). A ball
#: struck at 30 m/s over a 4K top-down pitch (~36 px/m at 30 fps) moves ~36 px a
#: frame, so this is the physics bound, not a fit. Beyond it the path must pay a
#: MISS frame and re-enter, which is how occlusion and a hard clearance are
#: handled by the same mechanism.
MAX_STEP_PX = 60.0
#: The emission of the MISS state: "nothing visible this frame". Twice the
#: detector's own conf floor (ball_eval.CONF = 0.05), so any candidate the
#: detector is more than marginally sure of beats staying silent, and a candidate
#: at the floor does not.
P_MISS = 0.10


# ---------------------------------------------------------------- the gates

def _gates(cache: dict, clip: str, scenery: bool = True,
           ratio: tuple[float, float] | None = None) -> dict[int, tuple]:
    """{frame: (xy (N,2), conf (N,), keep (N,) bool)} - ball.BallPicker's gates.

    A REPLICATION of the gate block in `BallPicker.update`, not a refactor of it:
    the shipped picker must keep scoring exactly as N48 measured it, so it is not
    touched. `--score` asserts this replication plus argmax(conf) reproduces
    `shipped` on every scored frame, which is what makes the duplication safe -
    if ball.py moves, the comparison goes VOID instead of quietly re-basing.
    """
    H = cache["frame_h"]
    cal = B._clip_cal(clip)
    p = ballmod.BallPicker(cal=cal)
    # `ratio` overrides BallPicker.SIZE_RATIO for ball_gate's leave-one-clip-out
    # re-fit. Default None = the shipped bounds, so every N49 tag above still
    # replays exactly as it was recorded, and --score's equivalence assertion
    # keeps proving it.
    lo, hi = ratio if ratio is not None else p.SIZE_RATIO
    out: dict[int, tuple] = {}
    for f in sorted(cache["frames"]):
        row = cache["frames"][f]
        cands = [r for r in row if ballmod.plausible(tuple(r[:4]), H)]
        xy = np.array([[(r[0] + r[2]) / 2, (r[1] + r[3]) / 2] for r in cands],
                      float).reshape(-1, 2)
        conf = np.array([r[4] for r in cands], float)
        # ALWAYS fed, as the shipped picker feeds it: the memory is about where
        # candidates keep appearing, so skipping frames would blind it.
        flags = p.scenery.flags(xy)
        keep = np.ones(len(xy), bool) if not scenery else ~flags
        if keep.any() and p.cal is not None:
            m = p.cal.to_pitch(xy)
            pi = p.cal.pitch
            keep &= ((m[:, 0] > -p.MARGIN_M) & (m[:, 0] < pi.length + p.MARGIN_M) &
                     (m[:, 1] > -p.MARGIN_M) & (m[:, 1] < pi.width + p.MARGIN_M))
            if keep.any():
                obs = np.array([max(r[2] - r[0], r[3] - r[1]) for r in cands])
                r = obs / np.maximum(p._expected_px(xy), 1e-6)
                keep &= (r >= lo) & (r <= hi)
        out[f] = (xy, conf, keep)
    return out


# ---------------------------------------------------------------- the picker

def viterbi(gated: dict[int, tuple], sigma: float = SIGMA_PX,
            max_step: float = MAX_STEP_PX, p_miss: float = P_MISS,
            ) -> dict[int, np.ndarray | None]:
    """The single best ball path over the whole clip. {frame: xy or None}.

    States are this frame's surviving candidates plus one MISS ("the ball is
    occluded or out"). Emission is log(conf), or log(p_miss) for MISS.
    Transition between candidates is a Gaussian on the step, hard-capped at
    max_step; every transition in or out of MISS is free, which is what lets the
    path cross an occlusion and re-acquire somewhere else.

    ponytail: dense O(N^2) per frame over ~45 candidates - 300 frames is 600k
    float ops and the whole six-clip score runs in seconds. A gating structure
    is the upgrade if a state count ever reaches the thousands.
    """
    frames = sorted(gated)
    log_miss = float(np.log(p_miss))
    prev_sc: np.ndarray | None = None   # scores of [candidates..., MISS]
    prev_xy: np.ndarray | None = None
    back: list[np.ndarray] = []
    kept: list[np.ndarray] = []
    for f in frames:
        xy, conf, keep = gated[f]
        idx = np.flatnonzero(keep)
        kept.append(idx)
        em = np.append(np.log(np.maximum(conf[idx], 1e-6)), log_miss)
        n = len(em)
        if prev_sc is None:
            prev_sc, prev_xy = em, xy[idx]
            back.append(np.full(n, -1, int))
            continue
        # transition[i, j]: previous state i -> this state j
        m = len(prev_sc)
        T = np.zeros((m, n))
        if len(idx) and len(prev_xy):
            d = np.hypot(*(prev_xy[:, None, :] - xy[idx][None, :, :]).transpose(2, 0, 1))
            step = -0.5 * (d / sigma) ** 2
            step[d > max_step] = -np.inf
            T[:len(prev_xy), :len(idx)] = step
        # MISS row (last previous state) and MISS column stay 0: free entry and
        # free re-entry. An occluded ball must not have to pay twice.
        tot = prev_sc[:, None] + T
        bi = np.argmax(tot, axis=0)
        prev_sc = tot[bi, np.arange(n)] + em
        prev_xy = xy[idx]
        back.append(bi)
    if prev_sc is None:
        return {}
    out: dict[int, np.ndarray | None] = {}
    s = int(np.argmax(prev_sc))
    for t in range(len(frames) - 1, -1, -1):
        idx = kept[t]
        out[frames[t]] = None if s >= len(idx) else gated[frames[t]][0][idx[s]]
        s = int(back[t][s])
        if s < 0:
            break
    return out


def argmax_conf(gated: dict[int, tuple]) -> dict[int, np.ndarray | None]:
    """The shipped decision, off the same gate replication. Only `--score`'s
    equivalence assertion uses it; it exists to catch drift, not to be scored."""
    out: dict[int, np.ndarray | None] = {}
    for f, (xy, conf, keep) in gated.items():
        idx = np.flatnonzero(keep)
        out[f] = None if not len(idx) else xy[idx[int(np.argmax(conf[idx]))]]
    return out


# ---------------------------------------------------------------- the report

def why() -> dict:
    """Where the 31% goes. Population: the frames where a candidate sits within
    GATE_PX of truth AFTER ball.py's shape gate (i.e. `oracle` is right) and the
    shipped picker's answer does not. Every count is over that population."""
    tot = dict(lost=0, scenery=0, onpitch=0, size=0, conf=0)
    print(f"{'clip':46s} {'lost':>5s} {'scen':>5s} {'pitch':>6s} {'size':>5s} {'conf':>5s}"
          f" {'trueconf':>9s} {'winner':>7s} {'travel_px':>10s}")
    for clip in C.CLIPS:
        cache = B.load_cache(clip + SUFFIX)
        gt, _, _ = B.load_ball(clip)
        gated = _gates(cache, clip)
        frames = sorted(cache["frames"])
        n = dict(lost=0, scenery=0, onpitch=0, size=0, conf=0)
        tc, wc = [], []
        # the same gate chain again, but split, so the first gate to kill the
        # true candidate is attributable.
        nosc = _gates(cache, clip, scenery=False)
        for f in frames:
            g = gt.get(f)
            xy, conf, keep = gated[f]
            if g is None or not len(xy):
                continue
            d = np.hypot(*(xy - g).T)
            j = int(np.argmin(d))
            if d[j] > B.GATE_PX:
                continue                       # oracle was wrong here too
            idx = np.flatnonzero(keep)
            win = None if not len(idx) else int(idx[int(np.argmax(conf[idx]))])
            if win is not None and np.hypot(*(xy[win] - g)) <= B.GATE_PX:
                continue                       # shipped got it
            n["lost"] += 1
            if keep[j]:
                n["conf"] += 1
                tc.append(conf[j])
                if win is not None:
                    wc.append(conf[win])
            elif not nosc[f][2][j]:
                # died to a calibration gate even with scenery off; separate the
                # two by re-running the on-pitch test alone.
                cal = B._clip_cal(clip)
                if cal is None:
                    n["size"] += 1
                else:
                    m = cal.to_pitch(xy[j:j + 1])[0]
                    pi = cal.pitch
                    on = (-3.0 < m[0] < pi.length + 3.0) and (-3.0 < m[1] < pi.width + 3.0)
                    n["onpitch" if not on else "size"] += 1
            else:
                n["scenery"] += 1
        f300 = [f for f in frames if f in gt]
        P = np.array([gt[f] for f in f300])
        travel = float(np.hypot(*np.diff(P, axis=0).T).sum())
        for k in n:
            tot[k] += n[k]
        print(f"{clip:46s} {n['lost']:5d} {n['scenery']:5d} {n['onpitch']:6d} {n['size']:5d}"
              f" {n['conf']:5d} {np.median(tc) if tc else float('nan'):9.3f}"
              f" {np.median(wc) if wc else float('nan'):7.3f} {travel:10.1f}")
    print(f"{'POOLED':46s} {tot['lost']:5d} {tot['scenery']:5d} {tot['onpitch']:6d} "
          f"{tot['size']:5d} {tot['conf']:5d}")
    print("\n  lost  = oracle right, shipped not. scen/pitch/size = the first gate to"
          "\n          delete the true candidate; conf = it survived every gate and lost"
          "\n          argmax(conf). trueconf/winner are median confidences on those.")
    print(f"  HEADROOM for any re-pick inside the gates: {tot['conf']} frames of 1800.")
    return tot


def score(sigma: float = SIGMA_PX, max_step: float = MAX_STEP_PX,
          p_miss: float = P_MISS, check: bool = True, verbose: bool = True) -> dict:
    """shipped vs path vs oracle vs centre, identical frames, pooled as N48 pools."""
    METHODS = ["shipped", "path", "path_nosc", "oracle", "centre"]
    pooled = {m: [0, 0, 0, 0] for m in METHODS}       # correct, wrong, silent, gt
    if verbose:
        print(f"{'clip':46s} " + " ".join(f"{m:>10s}" for m in METHODS))
    per_clip = {}
    for clip in C.CLIPS:
        cache = B.load_cache(clip + SUFFIX)
        gt, _, _ = B.load_ball(clip)
        frames = sorted(cache["frames"])
        gt_here = {f: gt[f] for f in frames if f in gt}
        gated = _gates(cache, clip)
        preds = {
            "shipped": B.predict(cache, "shipped", gt_here),
            "path": viterbi(gated, sigma, max_step, p_miss),
            "path_nosc": viterbi(_gates(cache, clip, scenery=False), sigma, max_step, p_miss),
            "oracle": B.predict(cache, "oracle", gt_here),
            "centre": B.predict(cache, "centre", gt_here),
        }
        if check:
            # VOID (c): the gate replication must still BE the shipped picker.
            ship, mine = preds["shipped"], argmax_conf(gated)
            bad = [f for f in frames
                   if (ship[f] is None) != (mine[f] is None)
                   or (ship[f] is not None and float(np.hypot(*(ship[f] - mine[f]))) > 1e-6)]
            assert not bad, f"{clip}: gate replication drifted from ball.BallPicker on " \
                            f"{len(bad)} frames, e.g. {bad[:5]} - VOID, re-read _gates"
        sc = {m: B.score(gt_here, preds[m], frames, B.GATE_PX) for m in METHODS}
        # VOID (b): every arm scored on the same frames.
        assert len({sc[m]["gt_present"] for m in METHODS}) == 1, sc
        for m in METHODS:
            pooled[m][0] += sc[m]["correct"]
            pooled[m][1] += sc[m]["wrong"]
            pooled[m][2] += sc[m]["silent"]
            pooled[m][3] += sc[m]["gt_present"]
        per_clip[clip] = {m: sc[m]["correct"] for m in METHODS}
        if verbose:
            print(f"{clip:46s} " + " ".join(f"{sc[m]['correct']:10d}" for m in METHODS)
                  + f"   of {sc['shipped']['gt_present']}")
    rate = {m: round(pooled[m][0] / max(pooled[m][3], 1), 4) for m in METHODS}
    if not verbose:
        return {"rate": rate, "delta": round(rate["path"] - rate["shipped"], 4),
                "per_clip": per_clip, "pooled": pooled}
    print("\n" + f"{'POOLED rate':46s} " + " ".join(f"{rate[m]:10.4f}" for m in METHODS))
    print(f"{'  wrong':46s} " + " ".join(f"{pooled[m][1]:10d}" for m in METHODS))
    print(f"{'  silent':46s} " + " ".join(f"{pooled[m][2]:10d}" for m in METHODS))
    d = round(rate["path"] - rate["shipped"], 4)
    void = rate["centre"] > 0.01
    bar = C.MARGIN
    print(f"\ngt_present {pooled['shipped'][3]}  sigma {sigma} max_step {max_step} "
          f"p_miss {p_miss}  n=1 configuration")
    print(f"delta path - shipped {d:+.4f} (bar {bar:+.2f}) -> "
          + ("VOID: the centre control scores" if void
             else ("SUPPORTED" if d >= bar else "NOT SUPPORTED")))
    print(f"remaining gap to oracle {rate['oracle'] - rate['path']:+.4f}; "
          f"path_nosc (exploratory, not ruled on) {rate['path_nosc']:.4f}")
    return {"rate": rate, "delta": d, "per_clip": per_clip, "pooled": pooled}


def diag() -> None:
    """WHY the path loses, on the same frames. Diagnosis of a failed primary -
    every number here is descriptive, none of it is a result.

    Two questions. Did the path recover ANY of the 231 frames the rule said were
    its whole headroom (the true ball gated in, quieter than a decoy)? And when
    it is wrong, is its answer MOVING - i.e. did it lock onto a smooth decoy -
    or is it static?
    """
    tot = dict(head=0, head_won=0, ship_had=0, path_lost_ship=0)
    print(f"{'clip':46s} {'headroom':>9s} {'path won':>9s} {'ship had':>9s} {'path lost':>10s}"
          f" {'mv_p50':>7s} {'mv_p90':>7s} {'segs':>5s} {'seglen':>7s} {'gt_mv':>6s}")
    for clip in C.CLIPS:
        cache = B.load_cache(clip + SUFFIX)
        gt, _, _ = B.load_ball(clip)
        frames = sorted(cache["frames"])
        gt_here = {f: gt[f] for f in frames if f in gt}
        gated = _gates(cache, clip)
        path = viterbi(gated)
        ship = B.predict(cache, "shipped", gt_here)
        ok = lambda p, f: (p is not None and f in gt_here            # noqa: E731
                           and float(np.hypot(*(p - gt_here[f]))) <= B.GATE_PX)
        head = []      # the headroom frames: true candidate kept, shipped wrong
        for f in frames:
            xy, conf, keep = gated[f]
            g = gt_here.get(f)
            if g is None or not len(xy):
                continue
            d = np.hypot(*(xy - g).T)
            j = int(np.argmin(d))
            if d[j] <= B.GATE_PX and keep[j] and not ok(ship[f], f):
                head.append(f)
        head_won = sum(ok(path[f], f) for f in head)
        ship_had = [f for f in frames if ok(ship[f], f)]
        lost = sum(not ok(path[f], f) for f in ship_had)
        # the path's own motion, and how often it re-acquires
        mv, segs, cur = [], [], 0
        prev = None
        for f in frames:
            p_ = path.get(f)
            if p_ is None:
                if cur:
                    segs.append(cur)
                cur, prev = 0, None
                continue
            cur += 1
            if prev is not None:
                mv.append(float(np.hypot(*(p_ - prev))))
            prev = p_
        if cur:
            segs.append(cur)
        mv = np.array(mv) if mv else np.array([np.nan])
        segs = np.array(segs) if segs else np.array([0])
        P = np.array([gt_here[f] for f in sorted(gt_here)])
        gmv = float(np.median(np.hypot(*np.diff(P, axis=0).T)))
        tot["head"] += len(head)
        tot["head_won"] += head_won
        tot["ship_had"] += len(ship_had)
        tot["path_lost_ship"] += lost
        print(f"{clip:46s} {len(head):9d} {head_won:9d} {len(ship_had):9d} {lost:10d}"
              f" {np.nanmedian(mv):7.2f} {np.nanpercentile(mv, 90):7.1f} {len(segs):5d}"
              f" {np.median(segs):7.0f} {gmv:6.1f}")
    print(f"{'POOLED':46s} {tot['head']:9d} {tot['head_won']:9d} {tot['ship_had']:9d} "
          f"{tot['path_lost_ship']:10d}")
    print("\n  headroom = true candidate survives every gate, shipped picker wrong."
          "\n  path won = of those, how many the path got right."
          "\n  ship had / path lost = frames the shipped picker got right, and how many"
          "\n     of them the path then threw away."
          "\n  mv = the PATH'S OWN frame-to-frame movement in px; gt_mv is the ball's."
          "\n  segs/seglen = number of answered runs between MISS frames, and their median.")


def sweep() -> None:
    """Is it the CONSTANTS or the FORM? Diagnosis only - rule 6 applies and the
    n is printed with the table. No value out of this is a headline, and the
    primary was ruled on the committed sigma=12 / max_step=60 / p_miss=0.10."""
    grid = [(sg, ms, pm) for sg in (4.0, 12.0, 40.0)
            for ms in (30.0, 60.0, 200.0) for pm in (0.02, 0.10, 0.30)]
    print(f"{'sigma':>7s} {'max_step':>9s} {'p_miss':>7s} {'path':>8s} {'shipped':>8s} "
          f"{'delta':>8s}")
    best = None
    for sg, ms, pm in grid:
        r = score(sg, ms, pm, check=False, verbose=False)
        print(f"{sg:7.1f} {ms:9.1f} {pm:7.2f} {r['rate']['path']:8.4f} "
              f"{r['rate']['shipped']:8.4f} {r['delta']:+8.4f}")
        best = r if best is None or r["rate"]["path"] > best["rate"]["path"] else best
    print(f"\n  n={len(grid)} configurations, all on the SAME frames the rule is scored "
          f"on.\n  best path {best['rate']['path']:.4f} vs shipped "
          f"{best['rate']['shipped']:.4f} - a best-of-{len(grid)}, which is not a result.")


def demo() -> None:
    """The one thing this file adds is the path, so that is what gets broken.

    ball_eval._synth is the centre-spot failure in miniature: a ball moving 4 px
    a frame at conf 0.20 and a LOUDER stationary decoy at 0.40.

    WRITTEN DOWN BEFORE THE SCORED RUN, because it is the rule's own
    counter-argument arriving early: the path does NOT rescue the first 25
    frames here. A stationary decoy is perfectly smooth, so the motion term
    charges it nothing and log(conf) then prefers it at 0.40 over the ball at
    0.20 - the path takes the decoy for exactly as long as the scenery memory
    lets it, and switches the frame the gate deletes it - 174 of 200 against the
    shipped picker's 175, the one frame being the MISS the relock must pay
    because 1300 px is past MAX_STEP_PX. Smoothness is not evidence of being a
    ball. What the path can do is on the OTHER side of that gate, among the
    transient decoys, and that is what --score measures.
    """
    cache, gt = B._synth()
    frames = sorted(cache["frames"])
    gated = _gates(cache, "synth")
    got = viterbi(gated)
    s = B.score(gt, got, frames, B.GATE_PX)
    sh = B.score(gt, B.predict(cache, "shipped"), frames, B.GATE_PX)
    # It must track the ball once the decoy is gone - a broken backtrace or an
    # unreachable MISS state shows up here - and the ONLY frame it may lose to
    # the shipped picker is the single relock MISS described above.
    assert s["correct"] >= 170 and s["correct"] >= sh["correct"] - 1, (s, sh["correct"])
    assert s["silent"] == 1, f"the relock should cost exactly one MISS frame: {s}"

    # A path cannot teleport: candidates further apart than MAX_STEP_PX must be
    # separated by a MISS, never chained.
    jump = {0: (np.array([[100.0, 100.0]]), np.array([0.9]), np.array([True])),
            1: (np.array([[900.0, 900.0]]), np.array([0.9]), np.array([True]))}
    out = viterbi(jump)
    # ...so exactly one of the two frames is answered: the other is the MISS the
    # hop has to pass through. A version that chained them would answer both.
    assert (out[0] is None) != (out[1] is None), out
    # ...and with a competing reachable candidate the reachable one wins even
    # though it is quieter, which is the property argmax(conf) does not have.
    j2 = {0: (np.array([[100.0, 100.0]]), np.array([0.9]), np.array([True])),
          1: (np.array([[900.0, 900.0], [104.0, 100.0]]), np.array([0.9, 0.2]),
              np.array([True, True]))}
    assert abs(viterbi(j2)[1][0] - 104.0) < 1e-6, viterbi(j2)[1]

    # MISS must be reachable: an empty frame is silence, not the nearest guess.
    e = {0: (np.array([[100.0, 100.0]]), np.array([0.9]), np.array([True])),
         1: (np.zeros((0, 2)), np.zeros(0), np.zeros(0, bool))}
    assert viterbi(e)[1] is None, viterbi(e)

    print("ball_path demo: path tracks the ball once the gate clears the static decoy "
          "(it does NOT beat it before that), cannot teleport, can be silent")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--why", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--arm", choices=("B", "D"), default="B",
                    help="which candidate cache to read: B is N49's ruling arm, "
                         "D is N52's clean checkpoint (a different population)")
    a = ap.parse_args()
    # Rebinding the module constant, not threading a parameter through five
    # functions: every reader here goes through SUFFIX, and a run that reads two
    # arms at once is not a thing this file should be able to express.
    globals()["SUFFIX"] = f".cmp{a.arm}"
    if a.arm != "B":
        print(f"### ARM {a.arm} ({SUFFIX}) - NOT N49's ruling population. "
              f"The bar and the 231/1800 headroom on this page are arm B's.\n")
    if a.why:
        why()
    elif a.score:
        score()
    elif a.diag:
        diag()
    elif a.sweep:
        sweep()
    else:
        demo()
