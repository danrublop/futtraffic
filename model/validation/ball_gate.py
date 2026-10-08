"""The SIZE GATE, which N49 made the ball picker's largest single loss.

Replay over N48's committed candidate caches - no detection, no GPU. The only
pixels read are `n49_paint.measure`'s, and only to LABEL frames, never to score.

WHAT IS MEASURED FIRST (`--size`, run before the rule below was written)
    `BallPicker.SIZE_RATIO = (0.4, 3.0)` compares a candidate's box to what
    0.22 m spans at that ground point (`_expected_px`). N49
    ([`n49-ball-path (not published)`](../../docs/n49-ball-path.md)) charged 234 of 557
    lost frames to it, the largest single cause. Three facts change how that
    reads, and all three come out of the caches.

    1. THE GATE ONLY RUNS ON THREE OF THE SIX CLIPS. It needs a calibration and
       the three `soccer_side` clips have none, so all 234 losses come from
       `soccer_top`, i.e. from 900 frames, not 1800.

    2. THE FINE-TUNED DETECTOR DRAWS A DIFFERENT BOX FOR THE SAME BALL. Median
       true-ball box / expected, and the share inside the shipped bounds:

         clip            arm A (stock@1024)  arm C (stock@640)  arm B (nano@640)  GT box
         top 1710-1740    1.79  in 1.000      1.57  in 1.000     4.12  in 0.457   1.81
         top 1740-1770    4.46  in 0.000      3.77  in 0.391     4.48  in 0.000   2.01
         top 1770-1800    1.80  in 1.000      1.63  in 1.000     2.05  in 0.933   2.45

       The gate was fitted on the stock detector's boxes and the nano's are about
       twice the annotation on two of the three clips. The annotator's own box is
       itself 1.81-2.45x expected, which is NOT a 2x error in `_expected_px`:
       see the decomposition below, which charges 1.30x to the ball's own
       silhouette and the rest to the annotator's padding.

    3. AND IT IS "AT REST", NOT "ON THE PAINT", THAT BREAKS IT. The frames are
       split with `n49_paint.measure`'s `blob_px` (the bright component holding
       the GT centre) and the GT's own step, per
       [`n49-ball-on-the-paint (not published)`](../../docs/n49-ball-on-the-paint.md)
       (branch `ball-clip-gap`, f26108d), which found `1740-1770`'s ball fused
       with the penalty-area line for its last 246 frames. Arm B, pooled over
       the three calibrated clips:

         category                       n     ratio p50   inside (0.4, 3.0)
         open play (GT step > 0.5 px)   303      2.15           0.769
         at rest, clean blob            303      4.12           0.459
         at rest, FUSED blob (>=150 px)   1      2.40           1.000

       (These are `--size`'s own printout. The pre-registration first quoted a
       scratch pass that differed by up to two frames per row - 301 / 3.98 /
       0.462 and 0 fused - because it dropped frames with no previous GT step;
       corrected here rather than left to be quoted.)

       The fused frames contribute essentially NOTHING to this loss - ONE of
       them carries a candidate within 20 px - because on the rest the detector
       proposes nothing there at all. That is the other page's finding, and it
       means the two failures are disjoint. What the size gate deletes is a ball
       AT REST ON GRASS, whose box the nano draws at 4x expected. Open play is
       where the gate mostly works (0.769) and even there the p90 is 4.47,
       because the ball at a player's feet gets a fused box too.

RULE, FIXED BEFORE THE SCORED RUN (committed with this file)
    population  the six `ball_cmp.CLIPS`, first 300 frames each, the committed
                `.cmpB` caches, conf 0.05, 20 px gate - identical to N48 and N49,
                so the numbers compare to both and to nothing else.
    metric      pooled share of gt-present frames answered within 20 px by the
                shipped decision rule with ONLY `SIZE_RATIO` changed. The
                baseline is N48/N49's `shipped` 0.4056.
    HOW THE BOUNDS ARE FITTED, AND WHY THIS WAY. LEAVE-ONE-CLIP-OUT over the
                three calibrated clips: each is scored under bounds fitted on
                the OTHER two, so no clip is ever scored under a gate its own
                frames helped choose, and the hit rate cannot be consulted while
                choosing because the fold that sets a bound is never the fold it
                is scored on. The offered alternative - fit on the GT boxes
                alone - was rejected: the gate is applied to DETECTOR boxes, and
                on this population the GT box does not predict them (arm B runs
                ~2x the annotation on two clips and ~0.84x on the third), so a
                GT-only fit would fit the wrong distribution and would still
                have to be scored somewhere.
    the fit     over the fit clips' TRUE-ball ratios (the plausible candidate
                within 20 px of truth, pooled, NOT split by play state - the
                shipped gate cannot know which it is looking at):
                LO = floor(p1 * 10) / 10, HI = ceil(p99 * 10) / 10.
                One rule, no sweep, n=1.
    PASS        pooled over all six clips >= 0.4056 + 0.05 = 0.4556, every
                calibrated clip scored under bounds fitted without it. The three
                uncalibrated clips are unaffected by any bound and pool in
                unchanged, so the whole delta has to come from 900 frames.
    also report `wrong`, and the share of all plausible candidates the gate still
                keeps, against the shipped gate's share on the same frames. A
                gate that widens until it keeps everything has not fixed the
                loss, it has deleted the gate - and the hit rate alone cannot
                separate those, because argmax(conf) then has more decoys to
                lose to. That shows up as `wrong` and as the keep share.
    VOID        `centre` over 1%; the arms not scoring identical frames;
                `model/engine/ball.py` modified. It is not: the bounds are
                passed into `ball_path._gates`, whose default is the shipped
                pair and which asserts equivalence with `ball.BallPicker` on
                every default-bounds run (N49's VOID condition (c)).

--------------------------------------------------------------------------------
IS `_expected_px` ITSELF WRONG BY 2x? NO - AND THIS FILE SAID SO BADLY (`--expected`)
--------------------------------------------------------------------------------
    The "under-reads by roughly a factor of two" this file used to print was
    measured against the ANNOTATOR'S BOX, which is not the ball. `--expected`
    decomposes it against the ball's own pixels and the tables, the three checks
    that say the projection is sound, and every number's tag live in
    [`n50-ball-size-gate (not published)`](../../docs/n50-ball-size-gate.md) §6 - one
    copy, tagged, rather than a second one here drifting out of step with it.

    What the decomposition changes HERE: there is no plain error to correct, so
    the re-score under a "corrected" expectation is NOT run - an experiment needs
    a correction first. And it is the opposite of a bug report against ball.py:
    the thing outside the shipped ceiling is the nano drawing ~3.2x the ball's
    silhouette on a ball at rest, which is a DETECTOR defect, and rejecting it is
    the gate working.

--------------------------------------------------------------------------------
THE JOINT ARM (`--joint`), RULE FIXED BEFORE THE RUN, AND IT IS THE LAST ONE
--------------------------------------------------------------------------------
    N49 changed the picker behind the shipped gate and lost (0.3856 vs 0.4056).
    N50 changed the gate behind the shipped picker and lost (0.3772). Each one's
    failure is the other's cause - N50's wider gate hands the picker 15,611 extra
    candidates it cannot choose among, and N49's path has nothing to work with
    behind a gate that deleted the ball. The pair is the only untried
    combination, so it is tried once.

    population  unchanged: six `ball_cmp.CLIPS`, first 300 frames, `.cmpB`
                caches, conf 0.05, 20 px gate, 1800 frames.
    arm         the LEAVE-ONE-CLIP-OUT bounds of the rule above (each clip's
                fitted without it), and then `ball_path.viterbi` at its
                REGISTERED constants (sigma 12 px, max_step 60 px, p_miss 0.10)
                instead of argmax(conf). Nothing is re-tuned: both halves are
                the configurations already committed, so this is n=1 and not a
                third search.
    metric      pooled 20 px hit rate, as before, against `shipped` 0.4056.
    PASS        >= 0.4556 (shipped + 0.05). Also reported: `wrong`, `silent`,
                and each half alone on the same run, so "the pair beats both
                halves" is visible even if the bar is not cleared.
    THE PRIOR, WRITTEN DOWN FIRST. This is expected to FAIL. N49 measured that
                weakening the motion prior IMPROVES the path (sigma 40 beats
                sigma 12 beats sigma 4), i.e. continuity is not discriminating
                between ball and decoy at all, and the candidates a wider gate
                admits are the smooth, louder ones - heads and paint. A motion
                model has no purchase on them. The run is worth making only
                because it is cheap and it is the last cell of the 2x2.
    IF IT FAILS the conclusion is not "try a fourth picker". It is that the
                picker is not the lever: with 36.6 candidates a frame against
                COCO's 22.5, the lever is a detector that emits fewer decoys.
    VOID        as above: `centre` over 1%, unequal frame lists, ball.py touched.

    .venv/bin/python model/validation/ball_gate.py --size     # the measurement  #slow
    .venv/bin/python model/validation/ball_gate.py --expected # the decomposition  #slow
    .venv/bin/python model/validation/ball_gate.py --score    # the LOCO re-fit
    .venv/bin/python model/validation/ball_gate.py --joint    # gate AND picker
    .venv/bin/python model/validation/ball_gate.py --demo     # self-check
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ball_cmp as C  # noqa: E402
import ball_eval as B  # noqa: E402
import ball_path as P  # noqa: E402

sys.path.insert(0, os.path.join(B.ROOT, "model", "engine"))
import ball as ballmod  # noqa: E402

SUFFIX = ".cmpB"
#: A GT step above this is "the ball is in open play"; at or below it the
#: annotator did not have to move the box. 0.5 px is the annotation's own
#: quantisation, and it is a LABEL for reporting, never a gate.
MOVE_PX = 0.5
#: `n49_paint.measure`'s blob area above which the ball's bright silhouette has
#: fused with something (a line, a sock). The three clips sit at 49 / 745 / 51
#: px median, so anything between ~100 and ~500 splits them identically.
FUSE_PX = 150


def true_ratios(clip: str, suffix: str = SUFFIX) -> tuple[np.ndarray, list[int]]:
    """(ratios, frames) for the candidate nearest truth, where it is within the
    20 px gate - i.e. exactly the boxes the size gate is supposed to KEEP.

    Empty when the clip has no calibration, because then the gate never runs.
    """
    cal = B._clip_cal(clip)
    if cal is None:
        return np.zeros(0), []
    cache = B.load_cache(clip + suffix)
    gt, _, _ = B.load_ball(clip)
    p = ballmod.BallPicker(cal=cal)
    H = cache["frame_h"]
    out, fs = [], []
    for f in sorted(cache["frames"]):
        g = gt.get(f)
        if g is None:
            continue
        rows = [r for r in cache["frames"][f] if ballmod.plausible(tuple(r[:4]), H)]
        if not rows:
            continue
        xy = np.array([[(r[0] + r[2]) / 2, (r[1] + r[3]) / 2] for r in rows])
        d = np.hypot(*(xy - g).T)
        j = int(np.argmin(d))
        if d[j] > B.GATE_PX:
            continue
        exp = float(p._expected_px(xy[j:j + 1])[0])
        obs = max(rows[j][2] - rows[j][0], rows[j][3] - rows[j][1])
        out.append(obs / max(exp, 1e-6))
        fs.append(f)
    return np.array(out), fs


def fit_bounds(ratios: np.ndarray) -> tuple[float, float]:
    """The committed fit: p1/p99 of the true-ball ratios, rounded OUTWARD."""
    lo = math.floor(float(np.percentile(ratios, 1)) * 10) / 10
    hi = math.ceil(float(np.percentile(ratios, 99)) * 10) / 10
    return lo, hi


# ---------------------------------------------------------------- the measurement

def size() -> None:
    """Descriptive only, and it is what the rule was written from.

    Reads the video through `n49_paint.measure` to LABEL each frame open-play /
    at-rest-clean / at-rest-fused. That is the other agent's instrument, cited
    rather than rewritten - `n49-ball-on-the-paint (not published)`, f26108d.
    """
    import n49_paint  # noqa: PLC0415

    cal_clips = [c for c in C.CLIPS if B._clip_cal(c) is not None]
    lo0, hi0 = ballmod.BallPicker.SIZE_RATIO
    lab = {}
    for clip in cal_clips:
        gt, _, _ = B.load_ball(clip)
        rows = n49_paint.measure(clip)
        for r in rows:
            f = r["f"]
            mv = (float(np.hypot(*(gt[f] - gt[f - 1])))
                  if f in gt and f - 1 in gt else float("nan"))
            lab[(clip, f)] = ("open play" if mv > MOVE_PX else
                              ("at rest, FUSED" if r["blob_px"] >= FUSE_PX
                               else "at rest, clean"))

    print(f"{'clip':46s} {'arm':>3s} {'n':>5s} {'ratio p10':>9s} {'p50':>6s} {'p90':>6s}"
          f" {'GT ratio':>9s} {'inside':>7s}")
    for clip in C.CLIPS:
        if B._clip_cal(clip) is None:
            print(f"{clip:46s}  -- no calibration: the size gate never runs here")
            continue
        gt, wh, _ = B.load_ball(clip)
        p = ballmod.BallPicker(cal=B._clip_cal(clip))
        for arm, suf in (("A", ".cmpA"), ("C", ".cmpC"), ("B", ".cmpB")):
            r, fs = true_ratios(clip, suf)
            if not len(r):
                print(f"{clip:46s} {arm:>3s}     0   (no true candidate inside the gate)")
                continue
            cache = B.load_cache(clip + suf)
            gtr = []
            for f in fs:
                xy = np.array([gt[f]])
                gtr.append(wh[f] / max(float(p._expected_px(xy)[0]), 1e-6))
            print(f"{clip:46s} {arm:>3s} {len(r):5d} {np.percentile(r, 10):9.2f} "
                  f"{np.median(r):6.2f} {np.percentile(r, 90):6.2f} "
                  f"{np.median(gtr):9.2f} {float(((r >= lo0) & (r <= hi0)).mean()):7.3f}")

    print(f"\nBY PLAY STATE, pooled over the {len(cal_clips)} calibrated clips "
          f"(labels from n49_paint.measure, f26108d):")
    print(f"{'arm':>3s} {'category':>16s} {'n':>5s} {'p10':>6s} {'p50':>6s} {'p90':>6s}"
          f" {'inside':>7s}")
    for arm, suf in (("A", ".cmpA"), ("C", ".cmpC"), ("B", ".cmpB")):
        acc: dict[str, list[float]] = {}
        for clip in cal_clips:
            r, fs = true_ratios(clip, suf)
            for v, f in zip(r, fs):
                acc.setdefault(lab.get((clip, f), "open play"), []).append(float(v))
        for k in ("open play", "at rest, clean", "at rest, FUSED"):
            v = np.array(acc.get(k, []))
            if not len(v):
                print(f"{arm:>3s} {k:>16s} {0:5d}")
                continue
            print(f"{arm:>3s} {k:>16s} {len(v):5d} {np.percentile(v, 10):6.2f} "
                  f"{np.median(v):6.2f} {np.percentile(v, 90):6.2f} "
                  f"{float(((v >= lo0) & (v <= hi0)).mean()):7.3f}")
    print(f"\n  ratio = the TRUE ball's box / what 0.22 m spans at that ground point;"
          f"\n  inside = share inside the shipped SIZE_RATIO {(lo0, hi0)};"
          f"\n  population = frames where a plausible candidate sits within "
          f"{B.GATE_PX:.0f} px of truth."
          f"\n  'at rest, FUSED' is empty for arms A and B because on those frames the"
          f"\n  detector proposes NO candidate within the gate - a disjoint failure, and"
          f"\n  the subject of n49-ball-on-the-paint (not published), not of this gate.")


# ---------------------------------------------------------------- the scored run

def _void_c(clip, preds_shipped, g_old, frames):
    """VOID (c), RUN WHERE THE CLAIM IS MADE rather than in another file.

    This file's docstring says ball.py is unmodified "because the bounds are passed into
    ball_path._gates, which asserts equivalence with ball.BallPicker on every
    default-bounds run". That assertion lives in ball_path.score(check=True); score() and
    joint() here compute the default-bounds replication as `g_old` and never compared it
    to anything. So N50's 0.3772 and N51's 0.3344 were produced by functions whose stated
    VOID did not run in them - a promised gate absent from the code, which
    docs-guard-must-have-a-runnable-check (internal note) says has cost this repo its
    headline twice.
    """
    import numpy as np  # noqa: PLC0415
    mine = P.argmax_conf(g_old)
    bad = [f for f in frames
           if (preds_shipped[f] is None) != (mine[f] is None)
           or (preds_shipped[f] is not None
               and float(np.hypot(*(preds_shipped[f] - mine[f]))) > 1e-6)]
    assert not bad, (f"{clip}: the default-bounds gate replication drifted from "
                     f"ball.BallPicker on {len(bad)} frames, e.g. {bad[:5]} - VOID")


def score() -> dict:
    cal_clips = [c for c in C.CLIPS if B._clip_cal(c) is not None]
    ratios = {c: true_ratios(c)[0] for c in cal_clips}
    bounds = {c: fit_bounds(np.concatenate([ratios[o] for o in cal_clips if o != c]))
              for c in cal_clips}
    METHODS = ("shipped", "gate", "oracle", "centre")
    pooled = {m: [0, 0, 0] for m in METHODS}      # correct, wrong, gt_present
    keep = [0, 0]
    ship_keep = [0, 0]
    truekept = [0, 0]
    print(f"{'clip':46s} {'bounds':>12s} {'shipped':>8s} {'gate':>6s} {'oracle':>7s}"
          f" {'wrong':>6s} {'true kept new/old':>18s}")
    for clip in C.CLIPS:
        cache = B.load_cache(clip + SUFFIX)
        gt, _, _ = B.load_ball(clip)
        frames = sorted(cache["frames"])
        gt_here = {f: gt[f] for f in frames if f in gt}
        bnd = bounds.get(clip)
        g_new = P._gates(cache, clip, ratio=bnd)
        g_old = P._gates(cache, clip)
        preds = {"shipped": B.predict(cache, "shipped", gt_here),
                 "gate": P.argmax_conf(g_new),
                 "oracle": B.predict(cache, "oracle", gt_here),
                 "centre": B.predict(cache, "centre", gt_here)}
        _void_c(clip, preds["shipped"], g_old, frames)
        sc = {m: B.score(gt_here, preds[m], frames, B.GATE_PX) for m in METHODS}
        assert len({sc[m]["gt_present"] for m in METHODS}) == 1, sc
        for m in METHODS:
            pooled[m][0] += sc[m]["correct"]
            pooled[m][1] += sc[m]["wrong"]
            pooled[m][2] += sc[m]["gt_present"]
        for g, acc in ((g_new, keep), (g_old, ship_keep)):
            acc[0] += int(sum(int(k.sum()) for _, _, k in g.values()))
            acc[1] += int(sum(len(k) for _, _, k in g.values()))
        # DESCRIPTIVE, not part of the rule: how often the true candidate
        # survives each gate. It separates "the gate deleted the ball" from
        # "the ball was there and argmax took a decoy", which is the whole
        # reading of this result.
        tk = [0, 0]
        for i, g in enumerate((g_new, g_old)):
            for f in frames:
                x, _, k = g[f]
                gt_f = gt_here.get(f)
                if gt_f is None or not len(x):
                    continue
                dd = np.hypot(*(x - gt_f).T)
                j = int(np.argmin(dd))
                tk[i] += int(dd[j] <= B.GATE_PX and bool(k[j]))
        truekept[0] += tk[0]
        truekept[1] += tk[1]
        lab = f"{bnd[0]:.1f}-{bnd[1]:.1f}" if bnd else "no cal"
        print(f"{clip:46s} {lab:>12s} {sc['shipped']['correct']:8d} "
              f"{sc['gate']['correct']:6d} {sc['oracle']['correct']:7d} "
              f"{sc['gate']['wrong']:6d} {tk[0]:9d}/{tk[1]:<8d}")
    rate = {m: round(pooled[m][0] / max(pooled[m][2], 1), 4) for m in METHODS}
    d = round(rate["gate"] - rate["shipped"], 4)
    print(f"\n{'POOLED rate':46s} {'':>12s} {rate['shipped']:8.4f} {rate['gate']:6.4f} "
          f"{rate['oracle']:7.4f} {pooled['gate'][1]:6d}")
    print(f"gt_present {pooled['shipped'][2]}, centre {rate['centre']:.4f}")
    print(f"true candidate survives the gate on {truekept[0]} frames (re-fitted) vs "
          f"{truekept[1]} (shipped); of the {truekept[0] - truekept[1]} restored, the "
          f"picker converts {pooled['gate'][0] - pooled['shipped'][0]:+d}")
    print(f"candidates kept: re-fitted gate {keep[0]}/{keep[1]} "
          f"({keep[0] / max(keep[1], 1):.4f}), shipped gate {ship_keep[0]}/{ship_keep[1]} "
          f"({ship_keep[0] / max(ship_keep[1], 1):.4f})")
    print(f"bounds, each fitted WITHOUT the clip it is used on: "
          + "; ".join(f"{c[-9:]} {b[0]:.1f}-{b[1]:.1f}" for c, b in bounds.items()))
    print(f"\ndelta gate - shipped {d:+.4f} (bar {C.MARGIN:+.2f}) -> "
          + ("VOID: the centre control scores" if rate["centre"] > 0.01
             else ("SUPPORTED" if d >= C.MARGIN else "NOT SUPPORTED"))
          + "   n=1 fitting rule, leave-one-clip-out")
    return {"rate": rate, "bounds": bounds, "delta": d, "keep": keep}


def expected() -> None:
    """Is `_expected_px` wrong by 2x? Decompose it against the ball's own pixels.

    Four references for the same object, per calibrated clip: what the shipped
    probe returns, the same probe rotated 90 degrees in the pitch plane, the
    ball's bright silhouette (`n49_paint.measure`'s `blob_px`, as an equivalent
    disc diameter) and the annotator's box. Plus the calibration's own scale
    against 19,800 annotated PLAYER positions, which is what would catch a
    global scale error. Descriptive; nothing here is scored.
    """
    import n49_paint  # noqa: PLC0415
    from paths import FOOTAGE2  # noqa: PLC0415
    from teamtrack_tracks import load as load_tracks  # noqa: PLC0415

    def probe(p, xy, d):
        m = p.cal.to_pitch(xy)
        Hi = np.linalg.inv(np.asarray(p.cal.H, float))
        q = np.c_[m + np.asarray(d, float), np.ones(len(m))] @ Hi.T
        w = np.where(np.abs(q[:, 2:3]) < 1e-9, 1e-9, q[:, 2:3])
        return np.hypot(*((q[:, :2] / w) - xy).T)

    for clip in [c for c in C.CLIPS if B._clip_cal(c) is not None]:
        cal = B._clip_cal(clip)
        p = ballmod.BallPicker(cal=cal)
        gt, wh, _ = B.load_ball(clip)
        rows = n49_paint.measure(clip)
        fs = [r["f"] for r in rows if r["f"] in gt]
        xy = np.array([gt[f] for f in fs])
        ex, ey = probe(p, xy, [ballmod.BALL_DIAMETER_M, 0.0]), \
            probe(p, xy, [0.0, ballmod.BALL_DIAMETER_M])
        scale = probe(p, xy, [1.0, 0.0])
        dia = np.sqrt(4 * np.array([r["blob_px"] for r in rows if r["f"] in gt], float) / np.pi)
        box = np.array([wh[f] for f in fs], float)
        half = xy[:, 1] < np.median(xy[:, 1])
        print(f"\n{clip}  n={len(fs)}")
        print(f"  scale {np.median(scale):5.2f} px/m -> expected_px {np.median(ex):5.2f} "
              f"(y-probe {np.median(ey):5.2f}, ratio {np.median(ey / ex):4.2f})")
        print(f"  ball silhouette {np.median(dia):5.2f} px = {np.median(dia / ex):4.2f}x; "
              f"annotator box {np.median(box):5.2f} px = {np.median(box / ex):4.2f}x "
              f"= {np.median(box / dia):4.2f}x the ball")
        # A still ball sits on one row for all 300 frames, so there is no split to
        # make and printing a nan would read as a measurement.
        if half.any() and (~half).any():
            print(f"  by image row, silhouette/expected: {np.median((dia / ex)[half]):4.2f} "
                  f"vs {np.median((dia / ex)[~half]):4.2f}")
        else:
            print("  by image row: the ball never leaves one row here (still ball), no split")
        # A global scale error would put the players on the wrong-sized pitch.
        tr = load_tracks(FOOTAGE2 / f"{clip.replace('videos', 'annotations')}.csv")
        P = np.array([[r["x"], r["y"]] for k, v in tr.items() if not k.startswith("BALL")
                      for r in v[:2000]])
        m = cal.to_pitch(P)
        print(f"  {len(P)} player positions -> pitch x {np.percentile(m[:, 0], 1):.1f}"
              f"-{np.percentile(m[:, 0], 99):.1f}, y {np.percentile(m[:, 1], 1):.1f}"
              f"-{np.percentile(m[:, 1], 99):.1f} of {cal.pitch.length:.1f} x "
              f"{cal.pitch.width:.1f} m (p1-p99)")
    print("\n  A 2x error in expected_px would show as a 2x scale error here and does not."
          "\n  The residual is the ball's own silhouette reading ~1.3x a 6 px prediction,"
          "\n  which is a threshold bleeding ~1 px per side, and the annotator's padding.")


def joint() -> dict:
    """The last cell of the 2x2: the re-fitted gate AND the path picker."""
    cal_clips = [c for c in C.CLIPS if B._clip_cal(c) is not None]
    ratios = {c: true_ratios(c)[0] for c in cal_clips}
    bounds = {c: fit_bounds(np.concatenate([ratios[o] for o in cal_clips if o != c]))
              for c in cal_clips}
    METHODS = ("shipped", "gate", "path", "joint", "oracle", "centre")
    pooled = {m: [0, 0, 0, 0] for m in METHODS}     # correct, wrong, silent, gt
    print(f"{'clip':46s} " + " ".join(f"{m:>8s}" for m in METHODS))
    for clip in C.CLIPS:
        cache = B.load_cache(clip + SUFFIX)
        gt, _, _ = B.load_ball(clip)
        frames = sorted(cache["frames"])
        gt_here = {f: gt[f] for f in frames if f in gt}
        g_new = P._gates(cache, clip, ratio=bounds.get(clip))
        g_old = P._gates(cache, clip)
        preds = {"shipped": B.predict(cache, "shipped", gt_here),
                 "gate": P.argmax_conf(g_new),
                 "path": P.viterbi(g_old),
                 "joint": P.viterbi(g_new),
                 "oracle": B.predict(cache, "oracle", gt_here),
                 "centre": B.predict(cache, "centre", gt_here)}
        _void_c(clip, preds["shipped"], g_old, frames)
        sc = {m: B.score(gt_here, preds[m], frames, B.GATE_PX) for m in METHODS}
        assert len({sc[m]["gt_present"] for m in METHODS}) == 1, sc
        for m in METHODS:
            pooled[m][0] += sc[m]["correct"]
            pooled[m][1] += sc[m]["wrong"]
            pooled[m][2] += sc[m]["silent"]
            pooled[m][3] += sc[m]["gt_present"]
        print(f"{clip:46s} " + " ".join(f"{sc[m]['correct']:8d}" for m in METHODS))
    rate = {m: round(pooled[m][0] / max(pooled[m][3], 1), 4) for m in METHODS}
    print(f"\n{'POOLED rate':46s} " + " ".join(f"{rate[m]:8.4f}" for m in METHODS))
    print(f"{'  wrong':46s} " + " ".join(f"{pooled[m][1]:8d}" for m in METHODS))
    print(f"{'  silent':46s} " + " ".join(f"{pooled[m][2]:8d}" for m in METHODS))
    d = round(rate["joint"] - rate["shipped"], 4)
    print(f"\ngt_present {pooled['shipped'][3]}; sigma {P.SIGMA_PX} max_step {P.MAX_STEP_PX} "
          f"p_miss {P.P_MISS}, bounds fitted leave-one-clip-out; n=1, nothing re-tuned")
    print(f"delta joint - shipped {d:+.4f} (bar {C.MARGIN:+.2f}) -> "
          + ("VOID: the centre control scores" if rate["centre"] > 0.01
             else ("SUPPORTED" if d >= C.MARGIN else "NOT SUPPORTED")))
    print(f"best of the three changes: {max(rate['gate'], rate['path'], rate['joint']):.4f} "
          f"against shipped {rate['shipped']:.4f}; oracle {rate['oracle']:.4f}")
    return {"rate": rate, "delta": d}


def demo() -> None:
    """The fit and the hold-out are the two decisions here, so they are what
    gets broken - on synthetic inputs, where the answer is known."""
    # THE FIT ROUNDS OUTWARD, or it would cut the very boxes it was fitted on.
    r = np.array([1.0] * 50 + [2.34] * 50)
    lo, hi = fit_bounds(r)
    assert lo <= r.min() and hi >= r.max(), (lo, hi)
    assert (lo, hi) == (1.0, 2.4), (lo, hi)
    # ...and it is a gate, not a hull: a 1-in-1000 tail stays outside.
    assert fit_bounds(np.append(np.full(1000, 2.0), 90.0))[1] < 90.0

    # LEAVE-ONE-CLIP-OUT must never let a clip set the bound it is scored under.
    rs = {"a": np.full(10, 1.0), "b": np.full(10, 5.0), "c": np.full(10, 9.0)}
    for c in rs:
        fit = np.concatenate([rs[o] for o in rs if o != c])
        assert not set(fit.tolist()) & set(rs[c].tolist()), c
        assert fit_bounds(fit) != fit_bounds(rs[c]), "the held-out clip moved its own bound"

    # THE OVERRIDE REACHES THE GATE - and with no calibration it must be inert,
    # which is this file's first finding and the reason the side clips cannot
    # move under any bound.
    cache, gt = B._synth()
    assert B._clip_cal("synth") is None
    wide = P._gates(cache, "synth", ratio=(0.0, 1e9))
    n_wide = sum(int(k.sum()) for _, _, k in wide.values())
    n_ship = sum(int(k.sum()) for _, _, k in P._gates(cache, "synth").values())
    assert n_wide == n_ship, "no calibration: the size bounds must change nothing"
    print("ball_gate demo: the fit rounds outward, LOCO holds its clip out, "
          "and with no calibration the bounds are inert")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--expected", action="store_true")
    ap.add_argument("--joint", action="store_true")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.size:
        size()
    elif a.score:
        score()
    elif a.expected:
        expected()
    elif a.joint:
        joint()
    else:
        demo()
