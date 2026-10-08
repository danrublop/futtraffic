"""G1: the floors an event model has to clear, before an event model exists.

WHY THIS RUNS FIRST
    The sprint plan set P1's gate at "beat PathCRF's F1 0.58". A constant-interval
    emitter scores 0.558 on the same half with the same scorer. The gate was a bar a
    metronome clears, and nobody noticed because §11 was written and not run.

    So: every floor is computed here, in the SAME units as the model's metric, before
    any model is trained. A number that does not beat these is not a result.

THE FOUR FLOORS
    chance      uniformly random spots, n_det = n_gt. The absolute floor.
    metronome   one event every mean-gap seconds, class drawn from the prior. Catches
                a model that learned the CLOCK instead of the football - events here
                come every 1.6 s, so the clock alone is worth a lot.
    shift-copy  PASS and DRIVE are the release and reception of ONE ball movement
                (measured: 99.9% different player, 95.0% same team, flight p50 1.40 s).
                So emitting DRIVE at every PASS + 1.40 s is a model-free predictor of
                38% of the corpus. Any DRIVE head must beat it.
    prior       predict nothing but the class prior at the GT times. Isolates "can it
                localise" from "can it classify".

THE CLOCK THE FLOORS SIT ON - CHANGED 2026-09-19, AND EVERY FLOOR MOVED
    Until this date `load` read bas_index.csv directly: every row with footage, at
    the raw label time. st2_train scores its model on st2_cache's LANDED rows at
    the CORRECTED row (st2_index's LABEL OFFSET addendum - the labels were about a
    second early). So the floors and the thing they were supposed to judge sat on
    different clocks over different populations, which is the same shape of error
    st2-p1-step6 (not published) was spent removing one level down. `load` now reads the
    cache. `--raw` keeps the old path runnable, and st2-p1-step7 (not published) carries
    both tables, because a floor that moved because the clock moved is not the
    same floor and quietly replacing it would hide that.

UNITS, WHICH IS WHERE THE PLAN WENT WRONG
    `touch_eval.chance` returns mean F1; the headline metric is mAP. They are not
    comparable: a random PASS detector scores AP ~0.08 but precision/recall ~0.28 at
    its operating point. Every floor here is reported BOTH ways, and every baseline
    carries a confidence so its AP is defined at all.

`ponytail: AP is 11-point interpolated to match the upstream scorer, not the
sklearn definition. Upgrade path is vendoring src/evaluation/bas_map.py wholesale
once its schema adapter is written.`
"""
import argparse
import collections
import csv
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from st2_index import EVENT_SPLIT  # noqa: E402

INDEX = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data/soccertrack_v2/bas_index.csv")
MODEL = "model/models/event_model.pt"      # sprint P1's output. It does not exist yet.
SPLIT = EVENT_SPLIT     # 132831 is "excluded" and never loaded by any split
TOL = (1.0, 2.0)


def load(split, corrected=True):
    """The labels every floor is measured on: {(match, period): [(t, label, side, actor)]}.

    CORRECTED (the default, and the only honest option since 2026-09-19) reads
    st2_cache's .npz - the LANDED rows at their corrected row's feed clock, which
    is the exact population and clock st2_train scores its model on. `raw=True`
    restores the old path, straight off bas_index.csv at the raw label time, and
    exists only so the floors it produced stay re-runnable beside the new ones.

    THE TWO ARE NOT THE SAME FLOOR. On 117093 P1 the raw path sees 471 PASSes of
    1138 rows and times the first at 0.68 s; the cache sees 469 of 1134 and times
    it at 1.68 s - the one-second LABEL OFFSET of st2_index's PROTOCOL addendum.
    A floor on one clock cannot judge a model on the other, and the whole point of
    step 6 was to stop having two clocks.
    """
    if not corrected:
        rows = [r for r in csv.DictReader(open(INDEX)) if r["no_footage"] == "0"]
        out = collections.defaultdict(list)
        for r in rows:
            if SPLIT[r["match"]] != split:
                continue
            out[(r["match"], int(r["period"]))].append(
                (float(r["t_s"]), r["label"], r["team"], r["player_id"]))
        for v in out.values():
            v.sort()
        return dict(out)
    import st2_cache  # noqa: PLC0415
    from st2_index import HALF_MS  # noqa: PLC0415
    out = {}
    for m in sorted(x for x, s in SPLIT.items() if s == split):
        c = st2_cache.load(m)
        for k in (0, 1):
            sel = (c["ev_period"] == k) & (c["ev_drop"] == "")
            t = c[f"p{k}_t_ms"]
            out[(m, k)] = sorted(
                (float(t[r] - k * HALF_MS) / 1000, str(lab), int(sd), int(a))
                for r, a, lab, sd in zip(c["ev_row"][sel], c["ev_actor"][sel],
                                         c["ev_label"][sel], c["ev_side"][sel]))
    return out


def _greedy(times, gt, tol):
    """One-to-one nearest match, first come. -> the gt index for each time, or -1.

    THE ONE MATCHER. ap_tolerant, prf and attribution all call it, so "greedy
    one-to-one" cannot mean three slightly different things in three places -
    which is what it meant until 2026-09-19, and the demo's stacked-prediction
    assertion only guarded two of them.
    """
    matched, out = [False] * len(gt), []
    for t in times:
        best, bd = -1, tol + 1e-9
        for j, g in enumerate(gt):
            if matched[j]:
                continue
            d = abs(t - g)
            if d <= tol and d < bd:
                best, bd = j, d
        if best >= 0:
            matched[best] = True
        out.append(best)
    return out


def ap_tolerant(pred, gt, tol):
    """11-point interpolated AP with a temporal tolerance. `pred` is (t, score, ...),
    ranked by score descending; greedy one-to-one, as the upstream scorer does."""
    if not gt:
        return float("nan")
    tp_fp = [int(j >= 0) for j in
             _greedy([p[0] for p in sorted(pred, key=lambda p: -p[1])], gt, tol)]
    tp, pr = 0, []
    for i, v in enumerate(tp_fp, 1):
        tp += v
        pr.append((tp / len(gt), tp / i))
    return sum(max((p for r, p in pr if r >= k / 10), default=0.0)
               for k in range(11)) / 11 if pr else 0.0


def prf(pred, gt, tol):
    """Precision/recall/F1 at the operating point - the units `touch_eval` reports."""
    if not gt or not pred:
        return 0.0, 0.0, 0.0
    tp = sum(j >= 0 for j in _greedy(sorted(p[0] for p in pred), gt, tol))
    p, r = tp / len(pred), tp / len(gt)
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


# ---------------------------------------------------------------- baselines
def b_chance(per, cls, rng):
    """Uniform random spots, n_det = n_gt, random confidence."""
    out = {}
    for k, ev in per.items():
        n = sum(1 for e in ev if e[1] == cls)
        hi = max(e[0] for e in ev)
        out[k] = [(rng.uniform(0, hi), rng.random()) for _ in range(n)]
    return out


def b_metronome(per, cls, rng):
    """One event every mean-gap seconds. Catches a model that learned the clock."""
    out = {}
    for k, ev in per.items():
        n = sum(1 for e in ev if e[1] == cls)
        hi = max(e[0] for e in ev)
        if n == 0:
            out[k] = []
            continue
        step = hi / n
        out[k] = [(step * (i + 0.5), rng.random()) for i in range(n)]
    return out


def b_shift(per, cls, rng, src="PASS", lag=1.40):
    """Emit `cls` at every GT `src` + lag. Only meaningful for the PASS/DRIVE pair."""
    return {k: [(e[0] + lag, rng.random()) for e in ev if e[1] == src]
            for k, ev in per.items()}


BASELINES = (("chance", b_chance), ("metronome", b_metronome),
             ("shift-copy*", b_shift))


def floors(split="val", seed=0, corrected=True, only=None):
    """The whole table, as a dict, so a gate can be CHECKED and not read.

    -> {"n_events":, "periods":, "matches": [...],
        "cls": {CLASS: n},
        "AP"/"F1": {f"{baseline}.{CLASS}.{tol}": value}}

    `report` prints this. Nothing recomputes it: a floor that exists in two
    places is a floor that can disagree with itself, which is exactly the drift
    postmortem (not published) §B is a list of.
    """
    per = load(split, corrected)
    if only:
        per = {k: v for k, v in per.items() if k[0] == only}
    cls_n = collections.Counter(e[1] for ev in per.values() for e in ev)
    out = {"split": split, "periods": len(per), "corrected": corrected,
           "n_events": sum(len(v) for v in per.values()),
           "matches": sorted({k[0] for k in per}), "cls": dict(cls_n),
           "AP": {}, "F1": {}}
    for tol in TOL:
        for cls in cls_n:
            for bname, fn in BASELINES:
                if fn is b_shift and cls not in ("DRIVE", "PASS"):
                    continue
                src = "PASS" if cls == "DRIVE" else "DRIVE"
                lag = 1.40 if cls == "DRIVE" else -1.40
                # One rng PER CELL, seeded from the cell's own name. Sharing a
                # single stream made every floor depend on the order the table
                # was walked in - reordering the loop moved chance DRIVE AP by
                # 0.008, which a golden snapshot would report as a regression.
                rng = random.Random(f"{seed}|{bname}|{cls}|{tol}")
                aps, f1s = [], []
                for _ in range(5):
                    pr = (fn(per, cls, rng, src, lag) if fn is b_shift
                          else fn(per, cls, rng))
                    a = [ap_tolerant(pr[k], [e[0] for e in ev if e[1] == cls], tol)
                         for k, ev in per.items()]
                    f = [prf(pr[k], [e[0] for e in ev if e[1] == cls], tol)[2]
                         for k, ev in per.items()]
                    aps.append(np.nanmean(a))
                    f1s.append(np.mean(f))
                out["AP"][f"{bname}.{cls}.{tol}"] = float(np.mean(aps))
                out["F1"][f"{bname}.{cls}.{tol}"] = float(np.mean(f1s))
    return out


def attribution_many(pairs, tol=1.0):
    """`attribution` over several (pred, gt) PERIODS, matched inside each period and
    pooled over all of them. Matching across a period boundary would let a spot in
    the first half claim an event in the second, which is not a thing that can
    happen; pooling the hits afterwards is."""
    out, pool = {}, []
    for pred, gt in pairs:
        for cls, hits in _hits(pred, gt, tol).items():
            out.setdefault(cls, [[], 0])
            out[cls][0] += hits
            out[cls][1] += len(gt[cls])
            pool += hits
    res = {c: (len(h), n) + _means(h) for c, (h, n) in out.items()}
    res["all"] = (len(pool), sum(v[1] for v in res.values())) + _means(pool)
    return res


def _hits(pred, gt, tol):
    """{class: [(correct, nearest, chance) per matched spot]} - the matching itself."""
    out = {}
    for cls, g in gt.items():
        p = sorted(pred.get(cls, []), key=lambda x: -x[1])
        out[cls] = [(int(p[i][2] == g[j][1]), g[j][2], g[j][3])
                    for i, j in enumerate(_greedy([x[0] for x in p], [e[0] for e in g], tol))
                    if j >= 0]
    return out


def attribution(pred, gt, tol=1.0):
    """ATTRIBUTION GIVEN A CORRECT SPOT, on the matched set and its two baselines.

    `pred` is {class: [(t, score, actor)]}, `gt` is {class: [(t, actor, nearest,
    chance)]} where `nearest` is that event's nearest-player credit against the
    feed's ball and `chance` is 1/on-pitch. Spots are matched to GT exactly as
    ap_tolerant matches them - same `_greedy`, same order, same tolerance - so
    "given a correct spot" means the spots that scored, not a second selection.

    -> {class: (n_matched, n_gt, model, nearest, chance)}, plus "all" pooled over
    every matched pair. The baselines are averaged over THE MATCHED SET ONLY, so
    they answer "on these events, how well would naming the player nearest the
    ball have done" - sprint §3's identical-matched-set requirement.

    NEVER QUOTE THE MODEL COLUMN ALONE. It is the most gameable metric on this
    task: emit only easy isolated events and it climbs while recall collapses.
    n_matched / n_gt is the recall it was bought at and belongs on the same line.
    """
    return attribution_many([(pred, gt)], tol)


def _means(hits):
    """Model / nearest / chance, NaN-skipping: an event whose frame carries no feed
    ball has no nearest-player credit and must not delete the column for the rest.
    One such event in 117093 P1 was enough to NaN the whole table before this."""
    cols = []
    for i in range(3):
        v = [h[i] for h in hits if np.isfinite(h[i])]
        cols.append(float(np.mean(v)) if v else float("nan"))
    return tuple(cols)


# ---------------------------------------------------------------- attribution
def _credit(d, actor):
    """Nearest-player credit for one event. `d` is each player's distance to the
    ball, NaN for players off the pitch. A k-way tie containing the actor scores
    1/k - positions sit on a 1 m grid, so ties are real."""
    on = np.isfinite(d)
    near = on & (d <= np.nanmin(d) + 1e-6)
    return (1.0 / near.sum() if near[actor] else 0.0), 1.0 / on.sum()


def nearest_player(split="val"):
    """G1 floor 3: name the on-pitch player nearest the feed's ball at the label
    frame, over events st2_cache LANDED. -> {match: {group: (n, acc, chance)}},
    plus each match's count of landed events dropped for want of a ball."""
    import st2_cache  # noqa: PLC0415
    out = {}
    for m in sorted(x for x, s in SPLIT.items() if s == split):
        c = st2_cache.load(m)
        cr, labs, no_ball = [], [], 0
        for i in np.flatnonzero(c["ev_drop"] == ""):
            k, r, a = c["ev_period"][i], c["ev_row"][i], c["ev_actor"][i]
            b = c[f"p{k}_ball"][r]
            if np.isnan(b[0]):
                no_ball += 1
                continue
            cr.append(_credit(np.linalg.norm(c[f"p{k}_xy"][r] - b, axis=1), a))
            labs.append(str(c["ev_label"][i]))
        cr, labs = np.array(cr), np.array(labs)
        res = {}
        for g in ("PASS", "DRIVE", "all"):
            sel = labs == g if g != "all" else np.ones(len(labs), bool)
            res[g] = (int(sel.sum()), float(cr[sel, 0].mean()), float(cr[sel, 1].mean()))
        out[m] = {"groups": res, "no_ball": no_ball, "landed": int((c["ev_drop"] == "").sum())}
    return out


def report_nearest(split="val"):
    r = nearest_player(split)
    print(f"G1 NEAREST-PLAYER - split={split}, GT feed, label frame, "
          f"132831 excluded; EVENT_SPLIT, not ST2's official split")
    for m, v in r.items():
        print(f"{m}: landed {v['landed']}, no ball {v['no_ball']} "
              f"({v['no_ball'] / v['landed']:.3f} dropped)")
        for g, (n, acc, ch) in v["groups"].items():
            print(f"  {g:6s} n={n:5d}  nearest {acc:.3f}  chance {ch:.3f}")


def model_map(split="val", tol=1.0, model=MODEL):
    """The event model's AP against the floors above. THERE IS NO EVENT MODEL.

    Registered, and left raising, on purpose. The floors are only half a gate:
    a table of numbers nothing is compared against cannot tell you the product
    is failing, it can only tell you what failing would look like. This is the
    other half, and until sprint P1 trains something it reports FAILED with the
    reason - which is a work item, where an unregistered gap is invisible.
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not os.path.exists(os.path.join(root, model)):
        raise FileNotFoundError(
            f"no event model at {model}: sprint P1 (geometry-only per-player "
            f"classifier) is not trained, so nothing in this repo is scored "
            f"against the {split} floors. The floors themselves resolve.")
    raise NotImplementedError(
        f"{model} exists - wire its predictions through ap_tolerant at tol {tol}")


def report(split="val", seed=0, corrected=True):
    r = floors(split, seed, corrected)
    names = [n for n, _ in BASELINES]
    print(f"G1 FLOORS - split={split}, {r['periods']} periods, "
          f"{r['n_events']} events, matches {r['matches']} "
          f"(132831 excluded; EVENT_SPLIT, not ST2's official split)\n"
          f"labels: {'CORRECTED - st2_cache landed rows on the feed clock'
                     if corrected else 'RAW - bas_index.csv at the label clock, SUPERSEDED 2026-09-19'}\n")
    print("* shift-copy is GT-FED: it is handed every ground-truth PASS time and\n"
          "  emits DRIVE at +1.40 s (and PASS at GT DRIVE -1.40 s). It is a floor,\n"
          "  not a deployable predictor, and the asterisk has always meant that.\n")
    for tol in TOL:
        print(f"--- tolerance +-{tol:.1f} s " + "-" * 52)
        print(f"{'class':26s} {'n':>5s} | " +
              " | ".join(f"{n:>16s}" for n in names))
        print(f"{'':26s} {'':>5s} | " + " | ".join(f"{'AP':>7s} {'F1':>8s}" for _ in names))
        for cls, n in collections.Counter(r["cls"]).most_common():
            cells = [f"{r['AP'][k]:7.3f} {r['F1'][k]:8.3f}"
                     if (k := f"{b}.{cls}.{tol}") in r["AP"]
                     else f"{'-':>7s} {'-':>8s}" for b in names]
            print(f"{cls:26s} {n:5d} | " + " | ".join(cells))
        print()


def demo():
    """Self-check: a perfect predictor must score 1.0, chance must not, and the
    shift-copy must beat chance on DRIVE - the whole point of having it."""
    per = load("val")
    gt = [e[0] for e in per[("117093", 0)] if e[1] == "PASS"]
    assert abs(ap_tolerant([(t, 1.0) for t in gt], gt, 1.0) - 1.0) < 1e-9
    assert prf([(t, 1.0) for t in gt], gt, 1.0)[2] == 1.0
    # THE CENTRAL CLAIM IS ONE-TO-ONE MATCHING, so break it and require a fail.
    # Three predictions stacked on one GT event must score recall 1/3, not 1.
    # Deleting `if matched[j]: continue` from ap_tolerant used to leave this
    # demo printing "ok" while every floor in the table shifted.
    one = [gt[0]]
    stacked = [(gt[0] + d, 1.0) for d in (-0.2, 0.0, 0.2)]
    # `prf` is where this is testable: one-to-one gives tp=1 of 3 predictions,
    # so precision is exactly 1/3. Without it tp=3 and recall would exceed 1.0.
    # (AP cannot test it at n_gt=1 - 11-point interpolation hits recall 1.0 on
    # the first TP at precision 1.0, so it reads 1.000 either way.)
    p_, r_, _ = prf(stacked, one, 1.0)
    assert abs(r_ - 1.0) < 1e-9, r_
    assert abs(p_ - 1 / 3) < 1e-9, f"prf matching is not one-to-one: precision {p_}"

    # ap_tolerant needs >=2 GT to show it: two GT far apart, two predictions
    # stacked on each. One-to-one scores 2 TP + 2 FP -> AP 0.849; without it all
    # four match -> AP 1.000. (At n_gt=1 both read 1.000, which is why the
    # earlier version of this check was blind.)
    two = [gt[0], gt[0] + 600.0]
    quad = [(two[0] - .2, 1.), (two[0] + .2, 1.), (two[1] - .2, 1.), (two[1] + .2, 1.)]
    a_ = ap_tolerant(quad, two, 1.0)
    assert a_ < 0.95, f"ap_tolerant is not one-to-one: AP {a_:.3f}"

    # nearest-player credit: actor alone at the ball, a two-way tie, a miss,
    # and players off the pitch (NaN) must not count toward chance.
    assert _credit(np.array([0.5, 3.0, np.nan]), 0) == (1.0, 0.5)
    assert _credit(np.array([1.0, 1.0, 4.0]), 1)[0] == 0.5
    assert _credit(np.array([1.0, 2.0, 4.0]), 2)[0] == 0.0

    # THE CLOCK. The corrected load and the raw one are different populations on
    # different clocks, and this asserts the direction and size of the gap rather
    # than trusting the docstring: the raw labels are the EARLY ones.
    raw = load("val", corrected=False)[("117093", 0)]
    cor = per[("117093", 0)]
    assert len(raw) == 1138 and len(cor) == 1134, (len(raw), len(cor))
    r0 = min(e[0] for e in raw if e[1] == "PASS")
    c0 = min(e[0] for e in cor if e[1] == "PASS")
    assert abs((c0 - r0) - 1.0) < 1e-6, (r0, c0)

    # ATTRIBUTION: a spotter that is right on time and on the player scores 1.0;
    # right on time and wrong on the player scores 0.0 at the SAME n_matched -
    # which is the whole point of separating the two. A spot outside tolerance
    # matches nothing and lowers n_matched, never the accuracy.
    g = {"PASS": [(10.0, 3, 1.0, 0.05), (30.0, 7, 0.0, 0.05)]}
    ok = attribution({"PASS": [(10.0, 0.9, 3), (30.0, 0.8, 7)]}, g)["PASS"]
    assert ok[:3] == (2, 2, 1.0) and ok[3] == 0.5, ok        # nearest floor 1 of 2
    bad = attribution({"PASS": [(10.0, 0.9, 5), (30.0, 0.8, 5)]}, g)["PASS"]
    assert bad[:3] == (2, 2, 0.0), bad
    half_ = attribution({"PASS": [(10.0, 0.9, 3), (99.0, 0.8, 7)]}, g)["PASS"]
    assert half_[:3] == (1, 2, 1.0), half_

    rng = random.Random(0)
    ch = np.mean([ap_tolerant(b_chance(per, "DRIVE", rng)[k],
                              [e[0] for e in ev if e[1] == "DRIVE"], 1.0)
                  for k, ev in per.items()])
    sh = np.mean([ap_tolerant(b_shift(per, "DRIVE", rng)[k],
                              [e[0] for e in ev if e[1] == "DRIVE"], 1.0)
                  for k, ev in per.items()])
    assert sh > ch * 2, (sh, ch)
    print(f"demo ok: one-to-one holds, perfect 1.000, "
          f"chance DRIVE AP {ch:.3f}, shift-copy(GT-fed) {sh:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--nearest", action="store_true", help="G1 nearest-player attribution floor")
    ap.add_argument("--raw", action="store_true",
                    help="the SUPERSEDED floors, off bas_index.csv at the raw label clock")
    a = ap.parse_args()
    demo() if a.demo else report_nearest(a.split) if a.nearest \
        else report(a.split, corrected=not a.raw)
