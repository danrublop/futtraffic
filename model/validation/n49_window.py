"""What N48's six 300-frame windows actually contain, and what the pooled
`oracle` would be without the dead-ball ones.

This re-reads N48's committed candidate caches and SoccerTrack's ball truth;
it detects nothing and writes nothing at all. The exclusion
column is a SENSITIVITY, not a proposal: dropping the one clip that scores
worst is the filter CLAUDE.md warns about, and both drops are printed so the
page cannot quote one without the other.

    .venv/bin/python model/validation/n49_window.py
    .venv/bin/python model/validation/n49_window.py --demo   # the row still equals N48's
"""
import contextlib, io, sys, os
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "model", "engine"))
sys.path.insert(0, os.path.join(ROOT, "model", "validation"))
import ball_eval as B      # noqa: E402
import ball_cmp as C       # noqa: E402

DEAD = ["soccer_top__test__videos__D_20220220_1_1710_1740",
        "soccer_top__test__videos__D_20220220_1_1740_1770"]
ARMS = (("A", ".cmpA"), ("C", ".cmpC"), ("B", ".cmpB"))


def main():
    hits, frozen = {}, {}
    for c in C.CLIPS:
        xy, _, _ = B.load_ball(c)
        P = np.array([xy[f] for f in sorted(xy)[:C.FRAMES]])
        v = np.hypot(*(np.diff(P, axis=0).T))
        frozen[c] = int((v < 0.5).sum())
        for a, s in ARMS:
            # B.report prints a full per-picker table; this file wants one number.
            with contextlib.redirect_stdout(io.StringIO()):
                r = B.report(c, B.GATE_PX, s)
            hits[(a, c)] = r["scores"]["oracle"]["correct"]
    print(f"\n{'clip':32s} {'frozen/299':>11s} {'med px/frame':>13s} {'gt box px':>10s} "
          f"{'A':>5s} {'C':>5s} {'B':>5s}")
    for c in C.CLIPS:
        xy, wh, _ = B.load_ball(c)
        P = np.array([xy[f] for f in sorted(xy)[:C.FRAMES]])
        v = np.hypot(*(np.diff(P, axis=0).T))
        box = np.median([wh[f] for f in sorted(wh)[:C.FRAMES]])
        print(f"{c[-32:]:32s} {frozen[c]:11d} {np.median(v):13.2f} {box:10.1f} "
              + " ".join(f"{hits[(a, c)]:5d}" for a, _ in ARMS))
    print(f"\n{'population':34s} {'n':>5s} {'A':>7s} {'C':>7s} {'B':>7s} {'B-C':>7s}")
    for drop, name in (([], "all six (N48 as published)"),
                       (DEAD[1:], "minus 1740_1770 ONLY (do not)"),
                       (DEAD[:1], "minus 1710_1740 only"),
                       (DEAD, "minus both dead-ball windows")):
        keep = [c for c in C.CLIPS if c not in drop]
        r = {a: sum(hits[(a, c)] for c in keep) / (C.FRAMES * len(keep)) for a, _ in ARMS}
        print(f"{name:34s} {C.FRAMES * len(keep):5d} {r['A']:7.4f} {r['C']:7.4f} {r['B']:7.4f} "
              f"{r['B'] - r['C']:+7.4f}")


def demo():
    """The one thing that must not drift: this file's "all six" row IS N48's
    published pooled oracle. If the caches or the pooling ever change, the
    sensitivity table below it is comparing against a number nobody published."""
    hits = {}
    for c in C.CLIPS:
        with contextlib.redirect_stdout(io.StringIO()):
            hits[c] = B.report(c, B.GATE_PX, ".cmpB")["scores"]["oracle"]["correct"]
    pooled = round(sum(hits.values()) / (C.FRAMES * len(C.CLIPS)), 4)
    assert pooled == 0.7150, f"arm B pooled oracle {pooled}, N48 published 0.7150"
    assert set(DEAD) <= set(C.CLIPS), DEAD
    print(f"n49_window demo ok: arm B pooled oracle {pooled} == N48's published 0.7150")


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
