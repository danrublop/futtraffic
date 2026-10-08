"""The VETO check: do clean frames exist either side of a duel at all?

WHAT THIS CAN AND CANNOT DO
    The temporal proposal is: never classify inside an overlap. Cut at the
    crossing, label each fragment from its OWN clean frames, and resolve the
    duel by matching two labelled fragments to two.

    That requires clean frames to exist. This measures whether they do. It is a
    NECESSARY condition and nothing more - **it can veto the plan, it cannot
    approve it.** Even with fragments on both sides, nothing here says the
    matching works, and `@M` ~~34.4%~~ **69.2%** of duels are merged measurements
    with no fragments during the merge by construction.
    (CORRECTED 2026-09-10: the 34.4% came from `st2_oracle_duel.py`, which
    measured a different partition than its labels claimed. `@V` Re-measured on
    1,350 duels: one box covering both players **69.2%**, only one player covered
    at all 28.9%, two distinct rankable boxes **1.9%**. The merged share is twice
    what this line said, which makes the veto this file can issue MORE binding,
    not less.)

DEFINITIONS, chosen before running it
    clean       a GT box whose max IoU with any OTHER GT box is < 0.1
    window      +/- 1 s = +/- 25 frames around the switch frame
    the bar     >= 5 clean frames for BOTH participants; and separately
                >= 3 clean BEFORE and >= 3 clean AFTER for both, which is what
                a two-sided fragment match actually needs

`@M` RESULT, 4,901 duels over all ten matches - AND THE WINDOW IS THE ANSWER

        window    both >= 5    >= 3 each side    neither has any
        +/- 1 s     56.2%          28.3%             24.2%
        +/- 2 s     76.9%          50.4%             11.4%
        +/- 3 s     83.7%          62.9%              6.7%
        +/- 5 s     90.5%          74.0%              3.4%
        +/-10 s     97.0%          83.8%              1.1%

    The first run used +/-1 s and read as a partial veto at 28.3%. That was the
    WINDOW, not the football: +/-1 s was picked before running and is simply too
    tight. At a realistic +/-3 to 5 s, **63-74% of duels have clean fragments on
    both sides**, so the temporal route is not vetoed.

    THE WINDOW IS NOT FREE, which is why this is reported as a curve rather than
    a number. A clean frame five seconds from the duel is five seconds in which
    that track can be corrupted by a DIFFERENT duel before its label is used. The
    right window is an engineering choice with a cost on each side, and it should
    be swept again against the actual matcher rather than settled here.

    `@D` What it implies for the ceiling, at +/-3 s and assuming ~95% resolution
    where two-sided fragments exist: 0.629 x 0.95 + 0.371 x 0.412 = **~75%**
    against today's 41.2%. At +/-5 s, ~81%. Both well short of the ~93% a
    sibling survey derived, because that derivation assumed clean frames simply
    exist. The 95% itself is UNMEASURED and is the next thing to measure.
"""
import argparse
import sys
from collections import Counter

sys.path.insert(0, "model/validation")
import st2_failtax as F  # noqa: E402
import st2_team as T  # noqa: E402

SEQS = ("117092", "117093", "118575", "118576", "118577", "118578",
        "128057", "128058", "132831", "132877")
W = 25          # +/- 1 s at 25 fps
CLEAN_IOU = 0.1


def clean_counts(gt, gid, f):
    """(clean before, clean after) for one player in the window."""
    before = after = 0
    for k in range(f - W, f + W + 1):
        g = gt.get(k)
        if not g or gid not in g:
            continue
        if F.max_overlap(g, gid) < CLEAN_IOU:
            if k < f:
                before += 1
            elif k > f:
                after += 1
    return before, after


def study(seq, variant=".c03s"):
    gt = T.load_gt(seq)
    out = F.analyse(seq, variant.lstrip("."), 0.5)
    ev = [e for e in (out["contam"] if isinstance(out, dict) else out[0])
          if e["gap"] == 0 and e["pair_iou"] > 0]
    r = Counter()
    for e in ev:
        f = e["f"]
        cf = clean_counts(gt, e["gid_from"], f)
        ct = clean_counts(gt, e["gid_to"], f)
        r["n"] += 1
        r["both5"] += (sum(cf) >= 5 and sum(ct) >= 5)
        r["both_2sided"] += (min(cf) >= 3 and min(ct) >= 3)
        r["neither"] += (sum(cf) == 0 and sum(ct) == 0)
        r["one_side_only"] += ((sum(cf) >= 5) != (sum(ct) >= 5))
        r["cf"] += sum(cf); r["ct"] += sum(ct)
    return r


def report(seqs=None, variant=".c03s"):
    seqs = seqs or SEQS
    tot = Counter()
    print(f"\n{'seq':>10s} {'duels':>7s} {'both>=5 clean':>15s} "
          f"{'>=3 each side':>15s} {'neither has any':>16s}")
    for s in seqs:
        r = study(s, variant)
        tot.update(r)
        n = max(r["n"], 1)
        print(f"{s:>10s} {r['n']:7d} {r['both5']:6d} {100*r['both5']/n:6.1f}% "
              f"{r['both_2sided']:6d} {100*r['both_2sided']/n:6.1f}% "
              f"{r['neither']:7d} {100*r['neither']/n:6.1f}%")
    n = max(tot["n"], 1)
    print(f"\n{'TOTAL':>10s} {tot['n']:7d} {tot['both5']:6d} "
          f"{100*tot['both5']/n:6.1f}% {tot['both_2sided']:6d} "
          f"{100*tot['both_2sided']/n:6.1f}% {tot['neither']:7d} "
          f"{100*tot['neither']/n:6.1f}%")
    print(f"\n  window +/-{W} frames (+/-1 s), clean = max IoU with any other "
          f"GT box < {CLEAN_IOU}")
    print(f"  mean clean frames in window: {tot['cf']/n:.1f} (from-player), "
          f"{tot['ct']/n:.1f} (to-player)")
    print(f"  only ONE participant clears 5: {100*tot['one_side_only']/n:.1f}%")
    print(f"\n  THIS IS A VETO TEST. Passing it does not mean the matching works.")
    return tot


def demo():
    """A player alone must be clean; two boxes on top of each other must not."""
    gt = {10: {1: (0., 0., 20., 50.), 2: (500., 0., 20., 50.)},
          11: {1: (0., 0., 20., 50.), 2: (2., 0., 20., 50.)}}
    assert F.max_overlap(gt[10], 1) < CLEAN_IOU
    assert F.max_overlap(gt[11], 1) > CLEAN_IOU
    g = {f: {1: (0., 0., 20., 50.), 2: (500., 0., 20., 50.)}
         for f in range(0, 60)}
    b, a = clean_counts(g, 1, 30)
    assert b == W and a == W, (b, a)
    print(f"ok  (isolated player reads {b} clean before / {a} after; "
          f"overlapping pair is not clean)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="*")
    ap.add_argument("--variant", default=".c03s")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    demo() if a.demo else report(a.seqs, a.variant)
