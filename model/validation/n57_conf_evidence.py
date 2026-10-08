"""N57: the dead-ball split, and whether the detector's CONFIDENCE is evidence.

    .venv/bin/python model/validation/n57_conf_evidence.py --split
    .venv/bin/python model/validation/n57_conf_evidence.py --lr
    .venv/bin/python model/validation/n57_conf_evidence.py --demo

WHY THIS RUNS
    Four pre-registered picker changes have now lost (N49 arm B, N50's re-fitted
    size gate, N51's joint arm, N49 §5b on arm D). N55 reads the four as one
    mistake - they add temporal integration to a problem that is not weak-signal
    - and names the per-candidate likelihood as what was never tested. Before
    building one, two things have to be settled, and both are arithmetic over
    caches that already exist.

PART A - THE POPULATION SPLIT, which N49 §7 asked for and nobody did
    Two of the six scored clips are DEAD BALLS: `soccer_top test 1710-1740`'s GT
    ball travels 0.9 px in total over 300 frames and `1740-1770`'s 101.9, against
    1160-1624 px for the other four. Pooling them is why "the picker loses X%"
    has never meant one thing: a fix for a ball at rest and a fix for a ball in
    play are different fixes, and the scenery gate deletes one population by
    design.

    RULE, fixed here: a clip is DEAD-BALL if its GT ball's mean speed over the
    scored frames is **< 1.0 px/frame**. That is a property of the ball, not of
    any detector or picker, and the gap it sits in is 10x wide on both sides
    (0.003 and 0.34 px/frame against 3.9-5.4). Every picker number from here on
    is reported for both populations or it is not reported.

PART B - IS CONFIDENCE EVIDENCE?
    N49's Viterbi used `log(conf)` as its emission term, i.e. it ASSUMED a higher
    confidence argues for a candidate being the ball. On the frames it loses, the
    true candidate sits at conf p50 0.078-0.162 while the winner sits at
    0.272-0.494. If confidence does not separate the ball from its decoys, that
    emission was not merely weak - it was pointed the wrong way, and four losses
    become one bug rather than four dead ends.

    RULE, fixed before the run. Population: `ball_cmp.CLIPS`, the committed
    `.cmpD` candidate caches, every frame with a GT ball, split by PART A. For
    each frame the TRUE candidate is the gated candidate nearest truth within
    GATE_PX; every other candidate in that frame is a DECOY. Bin conf into
    deciles of the pooled candidate distribution and compute the empirical
    likelihood ratio LR = p(conf | true) / p(conf | decoy) per bin.

    READING, fixed now so the answer cannot be chosen after seeing it, by the
    AUC of conf as a ranker of true candidates against decoys - the
    Mann-Whitney statistic, i.e. P(conf of a random true > conf of a random
    decoy). It is the exact question the emission term asks:
        AUC >= 0.55   CONFIDENCE IS EVIDENCE. log(conf) was directionally right
                      and the four losses are not explained by its sign.
        0.45-0.55     CONFIDENCE IS NOISE. The emission carried no information;
                      a likelihood built on it was a coin weighted by nothing.
        AUC <= 0.45   CONFIDENCE IS ANTI-EVIDENCE. log(conf) had the WRONG SIGN,
                      and every result that used it is confounded by that.
    The first draft of this rule read a Spearman rho over ten LR bins against a
    +-0.3 band. That was discarded BEFORE any real data was scored, because the
    null sd of rho over ten points is ~0.33: the band admitted most of the null,
    and a fixture built to be pure noise read as EVIDENCE. The LR table is still
    printed - it is how the shape is read - but the verdict is the AUC's.
    Reported for both populations separately and pooled. n=1 configuration: one
    binning (deciles), one gate (GATE_PX, unchanged), no sweep. If a second
    binning is ever run, that n moves and this stops being a pre-registration.

    THIS IS A DIAGNOSIS, NOT A CHANGE. Nothing here proposes a picker, and no
    outcome of it is a result about the product. It decides which of N55's
    candidates is worth building, and its own verdict is one of three words.
"""
import collections
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
DEAD_PX_PER_FRAME = 1.0
BINS = 10
FLAT = 0.3


def split():
    """{clip: 'dead'|'in-play'} by the GT ball's own mean speed. No detector."""
    import ball_cmp as M
    import ball_eval as B

    out = {}
    for clip in M.CLIPS:
        gt, _c, _n = B.load_ball(clip)
        cache = B.load_cache(clip + M.ARMS["D"]["suffix"])
        fs = [f for f in sorted(gt) if f in cache.get("frames", cache)]
        step = [float(np.hypot(*(gt[b] - gt[a]))) for a, b in zip(fs, fs[1:])]
        mean = float(np.mean(step)) if step else 0.0
        out[clip] = ("dead" if mean < DEAD_PX_PER_FRAME else "in-play", mean, len(fs))
    return out


def conf_pairs():
    """(conf, is_true) for every candidate in every GT frame, per population."""
    import ball_cmp as M
    import ball_eval as B

    pop = split()
    rows = collections.defaultdict(list)
    for clip in M.CLIPS:
        gt, _c, _n = B.load_ball(clip)
        cache = B.load_cache(clip + M.ARMS["D"]["suffix"])
        frames = cache.get("frames", cache)
        for f in [x for x in sorted(gt) if x in frames]:
            cands = B._dets(frames[f]) if hasattr(B, "_dets") else frames[f]
            if not cands:
                continue
            d = [(float(np.hypot(*(B._centre(c["xyxy"]) - gt[f]))), c) for c in cands]
            near, best = min(d, key=lambda t: t[0])
            for dist, c in d:
                rows[pop[clip][0]].append((float(c.get("conf", 0.0)),
                                           c is best and near <= B.GATE_PX))
    return rows


def lr(rows):
    """Empirical likelihood ratio by conf decile, and its Spearman rho."""
    conf = np.array([c for c, _ in rows])
    true = np.array([t for _, t in rows], bool)
    if true.sum() < BINS or (~true).sum() < BINS:
        return None
    edges = np.quantile(conf, np.linspace(0, 1, BINS + 1))
    edges[-1] += 1e-9
    # LAPLACE-SMOOTHED, because an empty bin is INFORMATIVE. Dropping bins with
    # no true candidate discards exactly the evidence a separating signal
    # produces - the first cut did, and a fixture built to read EVIDENCE came
    # back UNDECIDABLE because 8 of its 10 bins held decoys only.
    out = []
    for i in range(BINS):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        pt = ((true & m).sum() + 1) / (true.sum() + BINS)
        pd = (((~true) & m).sum() + 1) / ((~true).sum() + BINS)
        out.append((float(edges[i]), float(edges[i + 1]), float(pt), float(pd), float(pt / pd)))
    idx = np.arange(BINS, dtype=float)
    val = np.log(np.array([r[4] for r in out], float))
    ri, rv = np.argsort(np.argsort(idx)), np.argsort(np.argsort(val))
    rho = float(np.corrcoef(ri, rv)[0, 1])
    return out, rho


def auc(rows):
    """P(conf of a random true candidate > conf of a random decoy), ties at 0.5."""
    t = np.array([c for c, y in rows if y], float)
    d = np.array([c for c, y in rows if not y], float)
    if len(t) < 2 or len(d) < 2:
        return float("nan")
    order = np.argsort(np.concatenate([t, d]), kind="mergesort")
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    vals = np.concatenate([t, d])
    # average ranks within ties, or a detector that emits one repeated confidence
    # would read as a perfect ranker
    for v in np.unique(vals):
        m = vals == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:len(t)].sum() - len(t) * (len(t) + 1) / 2) / (len(t) * len(d)))


def verdict(a):
    if not np.isfinite(a):
        return "UNDECIDABLE (too few of one class)"
    return ("CONFIDENCE IS EVIDENCE" if a >= 0.55 else
            "CONFIDENCE IS ANTI-EVIDENCE" if a <= 0.45 else "CONFIDENCE IS NOISE")


def show_split():
    for clip, (pop, mean, n) in sorted(split().items(), key=lambda kv: kv[1][1]):
        print(f"{pop:8s} {mean:7.3f} px/frame  n={n:4d}  {clip.split('__')[-1]}")
    print(f"\nrule: dead-ball if the GT ball's mean speed < {DEAD_PX_PER_FRAME} px/frame")


def show_lr():
    rows = conf_pairs()
    allrows = [r for v in rows.values() for r in v]
    for name, rs in list(rows.items()) + [("POOLED", allrows)]:
        res = lr(rs)
        if res is None:
            print(f"\n{name}: too few of one class"); continue
        table, rho = res
        a = auc(rs)
        n_t = sum(1 for _, t in rs if t)
        print(f"\n{name}: {len(rs)} candidates, {n_t} true, {len(rs) - n_t} decoys")
        print(f"  {'conf bin':>16} {'p(true)':>9} {'p(decoy)':>9} {'LR':>8}")
        for lo, hi, pt, pd, ratio in table:
            print(f"  {lo:7.3f}-{hi:6.3f} {pt:9.4f} {pd:9.4f} {ratio:8.3f}")
        print(f"  AUC = {a:.4f} (shape: Spearman rho over LR bins {rho:+.3f})"
              f"  ->  {verdict(a)}")


def ranks():
    """Where the TRUE candidate sits by confidence, per frame, unconditionally.

    ADDED AFTER the AUC result, and for a reason that improves the page rather
    than decorating it: the conclusion drawn from AUC 0.8895 - "the problem is the
    number of competitors" - rested on treating candidates as independent draws,
    which is exactly what N55 says they are not. This measures the quantity that
    argument was approximating. It is a diagnosis, carries no bar, and changes no
    verdict: the AUC and its three-way reading were fixed in 4804496 and stand.

    Unconditional on purpose. N49 §2's "~12th" is conditioned on the frames the
    picker LOSES, which is the wrong denominator for "how hard is this argmax".
    """
    import ball_cmp as M
    import ball_eval as B

    pop = split()
    out = collections.defaultdict(list)
    for clip in M.CLIPS:
        gt, _c, _n = B.load_ball(clip)
        cache = B.load_cache(clip + M.ARMS["D"]["suffix"])
        frames = cache.get("frames", cache)
        for f in [x for x in sorted(gt) if x in frames]:
            cands = B._dets(frames[f])
            if not cands:
                continue
            d = [(float(np.hypot(*(B._centre(c["xyxy"]) - gt[f]))), c) for c in cands]
            near, best = min(d, key=lambda t: t[0])
            if near > B.GATE_PX:
                continue          # no true candidate present: not an argmax this could win
            cs = sorted(cands, key=lambda c: -float(c.get("conf", 0.0)))
            out[pop[clip][0]].append(cs.index(best) + 1)
            out["POOLED"].append(cs.index(best) + 1)
    for name in ("dead", "in-play", "POOLED"):
        r = np.array(out[name])
        if not r.size:
            continue
        print(f"{name:8s} n={r.size:5d}  rank of the true candidate by conf: "
              f"p50 {np.median(r):5.1f}  p90 {np.percentile(r, 90):5.1f}  "
              f"TOP-1 on {100 * (r == 1).mean():5.1f}% of frames  "
              f"top-3 {100 * (r <= 3).mean():5.1f}%")
    return out


def demo():
    """The reading rule, on synthetic data where the answer is known."""
    rng = np.random.default_rng(0)
    # confidence that SEPARATES: true candidates drawn high, decoys low
    good = [(float(rng.uniform(0.6, 1.0)), True) for _ in range(300)] + \
           [(float(rng.uniform(0.0, 0.5)), False) for _ in range(3000)]
    assert verdict(auc(good)) == "CONFIDENCE IS EVIDENCE", auc(good)
    # confidence that carries NOTHING: both drawn from one distribution
    none = [(float(rng.uniform(0, 1)), i < 300) for i in range(3300)]
    assert verdict(auc(none)) == "CONFIDENCE IS NOISE", auc(none)
    # confidence pointed the WRONG WAY
    bad = [(float(rng.uniform(0.0, 0.4)), True) for _ in range(300)] + \
          [(float(rng.uniform(0.5, 1.0)), False) for _ in range(3000)]
    assert verdict(auc(bad)) == "CONFIDENCE IS ANTI-EVIDENCE", auc(bad)
    # a detector emitting ONE confidence for everything ranks nothing; ties must
    # read as noise, not as a perfect ranker
    flat = [(0.7, i < 300) for i in range(3300)]
    assert verdict(auc(flat)) == "CONFIDENCE IS NOISE", auc(flat)
    print("n57 demo: the reading rule separates evidence, noise and anti-evidence")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--split" in a:
        show_split()
    elif "--lr" in a:
        show_lr()
    elif "--rank" in a:
        ranks()
    else:
        demo()
