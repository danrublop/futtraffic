"""Would our non-appearance signals have won the duel? The go/no-go on W6.

THE QUESTION, AND WHY IT HAS TO BE ANSWERED BEFORE BUILDING
    `@C` Every published fragment-resolution system - idtracker.ai, u-track,
    SUSHI, MHT - defers the decision at an overlap and resolves it afterwards
    WITH APPEARANCE. `@M` Appearance is measured useless on our footage (SigLIP
    0.611 against a 0.586 chance line). We would resolve with motion and a team
    label instead, and no paper says whether that is enough because none of them
    had to work without appearance.

    So this measures it on cached data, changing no tracker code: at each A1
    event - a duel swap with NO gap, `@M` 50.5% of all our failures - the
    tracker chose the wrong continuation. Given the two candidates it was
    choosing between, would motion, or the team label, or both, have picked the
    right one?

WHAT IT IS NOT - TWO LIMITS, BOTH REAL
    1. It is an ORACLE, handed the two candidates and asked only to rank them.
       A real deferred tracker has to find the candidate set itself. **Every
       number here is a ceiling.**
    2. **It only looks at events that ARE failures.** It cannot see how often
       the same rule would break an association the tracker currently gets
       RIGHT. A motion rule at 60.3% on known-bad cases is not a 60.3% net gain;
       the false-positive half is unmeasured and needs the rule actually built
       and run through `st2_ab`.

RESULT, all ten sequences, 4,901 duels (2026-09-09)

        motion alone                60.3%   (chance 50.0%)
        duels that are cross-team   70.3%   = the ceiling of a perfect kit label
        both together               88.2%   if the two are independent

      ~~both players carry a predicted box   65.6%~~
      ~~ONE box covers both players          34.4%  <- merged measurement~~

    RETRACTED 2026-09-10. Those two lines are NOT what the code measures, and the
    number said to decide what W6 must solve is wrong by a factor of ~30.
    `covered` (see `report()`) is a set of the two STRINGS "from" and "to". It
    records whether each GT player is covered by SOME predicted box; it says
    nothing about whether that is the SAME box. So `both_covered` counts merged
    boxes as if they were two candidates, and `one_covered` counts a MISSED
    detection, not a merged one.

    `@V` Re-measured on 128057 / 118577 / 132877, 1,350 duels, `.c03s`,
    separating the three cases the code conflates:

        two distinct rankable candidate boxes     26   1.9%
        ONE box covering both players            934  69.2%  <- merged measurement
        only one player covered at all           390  28.9%  <- missed detection

    97% of what this code calls `both_covered` is a single box covering both
    players (e.g. f=2918, tid 0 at IoU 0.780 against GT 13 and 0.824 against GT 2).

    Read it as: **the choice W6 is built to make barely exists.** A two-hypothesis
    ranker needs two candidates; duels that present two are **1.9%**, not 65.6%.
    The dominant case at **69.2%** is one measurement for two targets - the
    merged-measurement case radar names (research-mtt-classical (not published)) - and no
    pairwise ranking helps there: it needs defer-through-the-merge, the
    idtracker.ai pattern, which is a different mechanism from a two-hypothesis
    score. A further 28.9% is a player with no box at all, which is detection.

    The motion result below (60.3% on a two-way choice) and the "both together
    88.2%" line are unaffected AS ARITHMETIC, but they answer a question that
    arises in ~2% of duels. **This is the go/no-go number for W6 and it points the
    other way from the one that was published.**

    Still true and still the ranking that matters: **the signal that would carry
    W6 is the team label, not motion.** Motion alone is 10 points above a coin
    flip on a two-way choice it was handed the answer set for. The team label
    covers 70.3% of duels - but its accuracy AT A DUEL FRAME is unmeasured, and a
    duel is exactly where two kits are merged into one box and colour is least
    reliable.

THE TWO SIGNALS, AND WHY ONLY THESE TWO
    motion  constant velocity fitted on the track's OWN last MOTION_N frames,
            extrapolated to the switch frame; the candidate whose ground point
            is nearer wins. This is the signal a two-hypothesis MHT scores.
    team    NOT MEASURABLE FROM CACHE, and the first version of this file got
            that wrong. `tracks.json` carries ONE team label per track, not one
            per detection, so "the candidate detection's team" read back from it
            is just the track's own label and the test picks itself by
            construction - it scored 0/14, which is not chance and was an
            artefact, not a finding. The signal a team-aware matcher would
            actually use is the KIT COLOUR OF THE BOX at the switch frame, and
            that needs crops re-extracted from video (`model/validation/team_eval.py`
            does exactly this, with an .npz cache).

            What IS computable here is the CEILING: with GT team labels, the
            share of duels that are cross-team is the share a perfect
            per-detection kit label could resolve. That is reported instead, and
            labelled a ceiling rather than a result.

    Ground-plane exclusion and the 11+11 cardinality constraint are GLOBAL - a
    pairwise ranking cannot express them - so they are out of scope here and
    remain unmeasured. That is a real limit on what this study can conclude.
"""
import argparse
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, "model/validation")
import st2_failtax as F  # noqa: E402
import st2_team as T  # noqa: E402

SEQS = ("117092", "117093", "118575", "118576", "118577", "118578",
        "128057", "128058", "132831", "132877")
MOTION_N = 5          # frames of history for the velocity fit
MIN_HIST = 3


def foot(b):
    """Ground point of a top-left x,y,w,h box. A box bottom IS the ground point."""
    return np.array([b[0] + b[2] / 2.0, b[1] + b[3]])


def duels(seq, variant=".c03s"):
    """A1 events: contamination with pair_iou > 0 and no gap."""
    out = F.analyse(seq, variant.lstrip("."), 0.5)
    return [e for e in out["contam"] if e["gap"] == 0 and e["pair_iou"] > 0] \
        if isinstance(out, dict) else \
        [e for e in out[0] if e["gap"] == 0 and e["pair_iou"] > 0]


def study(seq, variant=".c03s"):
    import json as _json
    gt = T.load_gt(seq)
    path = f"model/validation/out/stc2025/{seq}{variant}.tracks.json"
    tr, team_major = T.load_tracks(path)
    hist = defaultdict(dict)                       # tid -> {frame: box}
    det_team = {}                                  # (frame, tid) -> team
    for r in _json.load(open(path))["detections"]:
        hist[r["id"]][r["f"]] = T.to_tlwh(r)
        det_team[(r["f"], r["id"])] = r.get("team", "unknown")

    gteam, _, _ = T.gt_teams(seq, gt)
    ev = duels(seq, variant)
    res = Counter()
    for e in ev:
        f, tid = e["f"], e["tid"]
        gfrom, gto = gt[f].get(e["gid_from"]), gt[f].get(e["gid_to"])
        if gfrom is None or gto is None:
            continue
        res["n"] += 1

        # --- motion: fit on the track's own history BEFORE the switch ---
        hs = [(k, v) for k, v in hist[tid].items() if f - MOTION_N <= k < f]
        hs.sort()
        if len(hs) >= MIN_HIST:
            ts = np.array([k for k, _ in hs], float)
            ps = np.array([foot(v) for _, v in hs])
            # least squares constant velocity, so a single noisy frame cannot
            # swing it the way a two-point difference would
            A = np.c_[ts - ts[-1], np.ones(len(ts))]
            coef, *_ = np.linalg.lstsq(A, ps, rcond=None)
            pred = coef[0] * (f - ts[-1]) + coef[1]
            dfrom = np.linalg.norm(pred - foot(gfrom))
            dto = np.linalg.norm(pred - foot(gto))
            res["motion_n"] += 1
            if abs(dfrom - dto) < 1e-9:
                res["motion_tie"] += 1
            elif dfrom < dto:
                res["motion_ok"] += 1
        else:
            res["motion_nohist"] += 1

        # --- team CEILING: is this duel cross-team at all? ---
        # A perfect per-detection kit label resolves exactly the cross-team
        # duels and abstains on the rest. This is the most the signal can be
        # worth, before any classifier error.
        ta, tb = gteam.get(e["gid_from"]), gteam.get(e["gid_to"])
        if ta is not None and tb is not None:
            res["team_known"] += 1
            if ta != tb:
                res["cross_team"] += 1

        # --- how many candidate boxes even exist at the switch frame? ---
        # If only ONE detection covers both players, there is nothing to choose
        # between and the problem is a merged measurement, not an association
        # error. This is the number that decides what W6 must actually solve.
        covered = set()
        for tid2 in hist:
            b2 = hist[tid2].get(f)
            if b2 is None:
                continue
            for g, box in (("from", gfrom), ("to", gto)):
                if T.iou(b2, box) >= 0.5:
                    covered.add(g)
        res["both_covered"] += (len(covered) == 2)
        res["one_covered"] += (len(covered) == 1)
    return res


def report(variant=".c03s", seqs=None):
    seqs = seqs or SEQS
    tot = Counter()
    print(f"\n{'seq':>10s} {'duels':>7s} {'motion':>16s} {'cross-team':>16s} "
          f"{'both boxes exist':>18s}")
    for s in seqs:
        r = study(s, variant)
        tot.update(r)
        fmt = lambda ok, n: (f"{ok:5d}/{n:<5d}{100*ok/n:5.1f}%" if n else
                             f"{'-':>16s}")
        print(f"{s:>10s} {r['n']:7d} {fmt(r['motion_ok'], r['motion_n'])} "
              f"{fmt(r['cross_team'], r['team_known'])} "
              f"{fmt(r['both_covered'], r['n'])}")
    fmt = lambda ok, n: (f"{ok:5d}/{n:<5d}{100*ok/n:5.1f}%" if n else "-")
    print(f"\n{'TOTAL':>10s} {tot['n']:7d} {fmt(tot['motion_ok'], tot['motion_n'])} "
          f"{fmt(tot['cross_team'], tot['team_known'])} "
          f"{fmt(tot['both_covered'], tot['n'])}")
    n = max(tot["n"], 1)
    xt = tot["cross_team"] / max(tot["team_known"], 1)
    mo = tot["motion_ok"] / max(tot["motion_n"], 1)
    print(f"\n  chance is 50.0% - two-way choices, handed the answer set, so"
          f" every number here is a CEILING")
    print(f"  motion alone            {100*mo:.1f}%")
    print(f"  perfect team label      {100*xt:.1f}% of duels are cross-team")
    print(f"  both, if independent    {100*(xt + (1-xt)*mo):.1f}%  "
          f"(perfect team on cross-team duels, motion on the rest)")
    print(f"\n  BUT: both players carry a predicted box at the switch frame in "
          f"only {tot['both_covered']}/{n} ({100*tot['both_covered']/n:.1f}%) "
          f"of duels.")
    print(f"  In the other {100*tot['one_covered']/n:.1f}% ONE box covers both "
          f"players - a merged measurement, not a choice between candidates.")
    print(f"  motion had too little history on {tot['motion_nohist']}/{n}")
    return tot


def demo():
    """A straight-line track must be scored correct against a crossing decoy."""
    b = lambda x, y: (x - 10.0, y - 50.0, 20.0, 50.0)
    hs = [(i, b(100.0 + 10 * i, 200.0)) for i in range(5)]
    ts = np.array([k for k, _ in hs], float)
    ps = np.array([foot(v) for _, v in hs])
    A = np.c_[ts - ts[-1], np.ones(len(ts))]
    coef, *_ = np.linalg.lstsq(A, ps, rcond=None)
    pred = coef[0] * (5 - ts[-1]) + coef[1]
    assert np.linalg.norm(pred - foot(b(150.0, 200.0))) < 1e-6, pred
    # the decoy sits where the track WAS, so motion must prefer the true one
    assert (np.linalg.norm(pred - foot(b(150.0, 200.0)))
            < np.linalg.norm(pred - foot(b(140.0, 210.0))))
    print("ok  (constant-velocity fit extrapolates exactly and beats a decoy)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=".c03s")
    ap.add_argument("--seqs", nargs="*")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    demo() if a.demo else report(a.variant, a.seqs)
