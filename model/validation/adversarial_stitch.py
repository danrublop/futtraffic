"""A self-check must fail when the function's CENTRAL CLAIM is falsified.

THE RULE OF THUMB, IN TWO PARTS
    Not "when the output changes shape". Not "when a count changes". A self-check
    earns its place only if you have tried to find a one-line edit that makes the
    function WRONG and leaves the check GREEN, and failed to find one.

    Operationally:

        for each load-bearing sentence in the docstring, write the one-line
        mutation that falsifies that sentence, and require the check to die.

    A docstring sentence with no mutation behind it is a comment, not a claim.

    AND THE PART THAT IS EASIER TO GET WRONG, because it looks like diligence:

        A CHECK THAT CARRIES ITS OWN COPY OF THE ARITHMETIC UNDER TEST IS NOT A
        CHECK. It is a second implementation that happens to agree with itself.

    Yesterday's orientation check was blind for exactly this reason: it recomputed
    the ball's pitch coordinate inline, so mutating the real conversion left it
    green. It only started catching a flipped y AFTER `ball_to_pitch` was
    extracted - that is, after there was ONE implementation for both the pipeline
    and the check to call. The extraction was the fix; the assertion was already
    there and already useless.

    This is why the two parts are one rule. Duplication is invisible to review -
    a recomputed expected value reads as rigour - and it is trivially visible to
    mutation, because a duplicated check is precisely a check no mutation can
    kill. So the mutation board below is not just a way to grade checks; it is
    the detector for duplication, and it is the only one that works.

    THREE WAYS TO WRITE AN ASSERTION THAT CANNOT DUPLICATE
        RELATIONAL    tie two calls of the real function together
                      (`_feasible([a,b,c]) == max(_link(a,b), _link(b,c))`)
        ORDERING      demand a rank, never a value
                      (`cost(on-course) < cost(backwards)`)
        ORACLE        a literal constant, ideally the one already in the
                      docstring (`abs(cost - 1.60) < 0.05`)
    Every assertion in harden() below is one of these three, and _selfcheck()
    asserts that none of them reaches for np.hypot.

WHY THIS FILE EXISTS
    Four of this repo's five self-checks pass on deliberately broken code. Deleting
    a one-to-one constraint from a scorer, flipping a coordinate axis, writing
    half-size images - all green. A coordinate-frame bug that deleted 45% of the
    pitch shipped under a passing check. The checks assert shapes and counts; every
    bug was semantic. `stitch.py`'s check is the best of them and it still has the
    hole, because it was written by asking "what should I test?" instead of "how
    would I break this without you noticing?".

    The second question has a mechanical answer, so it gets a mechanical harness.

HOW TO READ THE OUTPUT
    KILLED    the check died on that mutation. Good; the claim is defended.
    SURVIVED  the code is now wrong and the check still says OK. That is a hole,
              and it is the only line in the output worth reading.
    NOTAPPLIED the mutation's source text is gone - stitch.py was refactored under
              this file. Loud on purpose: a silently-inapplicable mutation is a
              vacuous pass, which is the exact disease being treated.

    Run:  .venv/bin/python model/validation/adversarial_stitch.py
          .venv/bin/python model/validation/adversarial_stitch.py --selfcheck   (fast)

`ponytail: source-text substitution against one module, not a mutation-testing
framework. Eight named mutations that each falsify a named sentence beat a
thousand random ones you will never read. If a second module needs this, lift
MUTANTS out and pass it in - do not install mutmut.`
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "model" / "engine" / "stitch.py"
sys.path.insert(0, str(ROOT / "model" / "engine"))


# (id, the docstring sentence it falsifies, old text, new text)
MUTANTS = [
    ("temporal-exclusion",
     "'a stitched id must never hold two detections in the same frame'",
     "    if a.end >= b.start:\n        return None",
     "    if False and a.end >= b.start:\n        return None"),

    ("speed-gate",
     "'a join implying more than MAX_SPEED m/s is refused'",
     "    if d > max_speed * gap_s + slack_m:\n        return None",
     "    if False:\n        return None"),

    ("gap-ceiling",
     "MAX_GAP_S: 'Longest gap to bridge'",
     "    if gap_s > max_gap_s:\n        return None",
     "    if False:\n        return None"),

    ("team-gate",
     "'Two components confidently on OPPOSITE sides are not one player'",
     "            if a and b and not (a & b):\n                return None",
     "            if False:\n                return None"),

    ("worst-junction",
     "_feasible: 'A component costs what its WORST junction costs'",
     "        worst = max(worst, c)",
     "        worst = c"),

    ("vel-clamp",
     "_vel: 'CLAMPED TO MAX_SPEED, because the unclamped version was worse "
     "than useless'",
     "    return v if n <= MAX_SPEED else v * (MAX_SPEED / n)",
     "    return v"),

    ("coast-direction",
     "_link: 'extrapolate each end along its own velocity'",
     "    fwd = float(np.hypot(*(b.xy[0] - (a.xy[-1] + _vel(a, -1, fps) * gap_s))))",
     "    fwd = float(np.hypot(*(b.xy[0] - (a.xy[-1] - _vel(a, -1, fps) * gap_s))))"),

    ("cost-is-distance",
     "_link: 'The cost is a pitch-plane MISS DISTANCE ... Deliberately not the "
     "implied speed'",
     "    return min(d, fwd, bwd)",
     "    return min(d, fwd, bwd) / max(gap_s, 1e-9)"),
]


def load(mutation=None):
    """stitch.py as a live module, optionally with one line changed.

    Nothing is written to disk. `model/engine/stitch.py` is never touched.
    """
    src = SRC.read_text()
    if mutation:
        _, _, old, new = mutation
        if old not in src:
            return None
        src = src.replace(old, new, 1)
    m = types.ModuleType("stitch_under_test")
    m.__file__ = str(SRC)
    exec(compile(src, str(SRC), "exec"), m.__dict__)   # noqa: S102 - our own file
    return m


def dies(fn, m) -> bool:
    """True if `fn(m)` raises. Anything raising counts: an assert is the check
    working, an exception is the check crashing, and both are 'not green'."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn(m)
        return False
    except BaseException:                              # noqa: BLE001 - that is the point
        return True


# --------------------------------------------------------------- the hardening
# Four assertions. Each one is the docstring's OWN measurement, executed. They are
# written here rather than in model/engine/stitch.py so nothing tracked is modified;
# they belong inside `_selfcheck` and should be moved there.
def harden(m) -> None:
    """The four claims stitch.py's own check does not defend."""
    tk = lambda tid, f0, f1, x0, x1, y=34.0: m._Tracklet(          # noqa: E731
        tid, np.arange(f0, f1),
        np.column_stack([np.linspace(x0, x1, f1 - f0), np.full(f1 - f0, y)]))
    link = lambda a, b: m._link(a, b, 25.0, m.MAX_SPEED, m.SLACK_M, m.MAX_GAP_S)  # noqa: E731

    # 1. _feasible returns the WORST junction, not the last one and not the mean.
    #    "a mean lets one 40 m leap hide behind six 2 m ones" - so build exactly
    #    that: a 24 m junction followed by a 0.2 m one, and demand the 24.
    a = tk(1, 0, 26, 0.0, 1.0)          # 1 m/s right
    b = tk(2, 150, 176, 30.0, 31.0)     # 5 s later, 29 m away, 1 m/s right
    c = tk(3, 300, 326, 31.2, 32.2)     # 5 s later, 0.2 m away
    j1, j2 = link(a, b), link(b, c)
    assert j1 > 20 and j2 < 1, (j1, j2)          # the fixture is the fixture
    got = m._feasible([a, b, c], 25.0, m.MAX_SPEED, m.SLACK_M, m.MAX_GAP_S)
    assert abs(got - max(j1, j2)) < 1e-9, (
        f"a component must cost its WORST junction: junctions {j1:.2f} m and "
        f"{j2:.2f} m scored {got:.2f} m")

    # 2. _vel is clamped, and the docstring's own numbers say by how much. One
    #    frame of 0.4 m far-side wobble is 10 m/s; clamped to 9 it coasts a 19.6 m
    #    join to 1.60 m, unclamped to 0.40 m - cheaper than a genuine 1 m join, so
    #    greedy agglomeration does the wild merge FIRST. That is the association
    #    ceiling this file blames elsewhere, and it is one `min` away.
    wobble = m._Tracklet(9, np.array([0, 1]), np.array([[0.0, 34.0], [0.4, 34.0]]))
    far = m._Tracklet(10, np.arange(51, 76),
                      np.column_stack([np.full(25, 20.0), np.full(25, 34.0)]))
    cost = link(wobble, far)             # 19.6 m across 2.0 s
    assert abs(cost - 1.60) < 0.05, (
        f"unclamped velocity: a 19.6 m join over one frame of 0.4 m wobble costs "
        f"{cost:.2f} m, and the docstring says 1.60 m")
    assert cost > link(tk(11, 0, 26, 0.0, 0.0), tk(12, 51, 76, 1.0, 1.0)), \
        "a wobble-coasted 19.6 m join must not outrank a genuine 1 m join"

    # 3. Coasting runs FORWARD. A sprinter found 12 m ahead of where they were
    #    heading is innocent; the same 12 m behind them is somebody else. Sign
    #    flips are the bug class that deleted 45% of the pitch today.
    run = tk(20, 0, 26, 0.0, 6.0)                       # 6 m/s, rightwards
    ahead = tk(21, 76, 101, 18.0, 24.0)                 # 2 s on, exactly on course
    behind = tk(22, 76, 101, -6.0, 0.0)                 # 2 s on, 12 m the wrong way
    assert link(run, ahead) < link(run, behind) - 5.0, (
        f"coasting is not directional: on-course {link(run, ahead):.2f} m vs "
        f"backwards {link(run, behind):.2f} m")

    # 4. The cost is a miss DISTANCE, not an implied speed. 1 m over 0.2 s must
    #    rank ahead of 2 m over 4 s; by speed those are 5.0 and 0.5 m/s and the
    #    order inverts. Measured: distance separates at AUC 0.842, speed 0.751.
    near = link(tk(30, 0, 26, 0.0, 0.0), tk(31, 31, 56, 1.0, 1.0))
    slow = link(tk(32, 0, 26, 0.0, 0.0), tk(33, 126, 151, 2.0, 2.0))
    assert near < slow, (
        f"cost ranks by implied speed, not distance: 1 m/0.2 s scored {near:.3f} "
        f"and 2 m/4.0 s scored {slow:.3f}")


def main() -> int:
    base = load()
    assert base is not None
    # Both checks must be green on the REAL file first, or every KILLED below is
    # meaningless. This is the assertion that stops a broken harness reading as a
    # perfect score.
    for name, fn in (("_selfcheck", lambda _m: _m._selfcheck()), ("harden", harden)):
        if dies(fn, base):
            print(f"FATAL: {name} fails on unmutated model/engine/stitch.py")
            fn(base)                      # again, uncaught, for the traceback
            return 2

    print("Two independent checks, run separately. The columns are not a ladder:\n"
          "the gates are stitch.py's to defend and the cost semantics are harden's.\n"
          "A mutation surviving BOTH is a hole.\n")
    print(f"{'mutation':20s} {'_selfcheck':>12s} {'harden':>10s}   falsifies")
    holes = 0
    for mut in MUTANTS:
        mid, claim = mut[0], mut[1]
        m = load(mut)
        if m is None:
            print(f"{mid:20s} {'NOTAPPLIED':>12s} {'-':>10s}   "
                  f"source text gone - stitch.py changed under this file")
            holes += 1
            continue
        a = "KILLED" if dies(lambda _m: _m._selfcheck(), m) else "SURVIVED"
        b = "KILLED" if dies(harden, m) else "SURVIVED"
        holes += (a == "SURVIVED" and b == "SURVIVED")
        print(f"{mid:20s} {a:>12s} {b:>10s}   {claim}")

    n_new = sum(1 for mut in MUTANTS
                if (m := load(mut)) is not None
                and not dies(lambda _m: _m._selfcheck(), m) and dies(harden, m))
    print(f"\n{n_new} of {len(MUTANTS)} mutations survive model/engine/stitch.py's own "
          f"self-check and are killed by harden().")
    print("harden() belongs inside stitch._selfcheck; it lives here only because "
          "this task may not modify tracked files.")
    return 1 if holes else 0


def _selfcheck() -> None:
    """The harness, not stitch. Two ways this file could lie, both asserted.

    It could report KILLED because loading failed rather than because the check
    caught anything, and it could report a green board because the mutations no
    longer apply. Both are the vacuous-pass failure this file exists to name.
    """
    assert load() is not None, "cannot load model/engine/stitch.py at all"
    applied = [mut[0] for mut in MUTANTS if load(mut) is not None]
    assert len(applied) == len(MUTANTS), \
        f"mutations no longer applicable: {set(m[0] for m in MUTANTS) - set(applied)}"
    assert not dies(harden, load()), "harden() fails on unmutated stitch.py"
    # dies() must distinguish green from raised, or the whole board is noise.
    assert dies(lambda _m: (_ for _ in ()).throw(AssertionError("x")), None)
    assert not dies(lambda _m: None, None)

    # THE DUPLICATION RULE, MECHANICALLY. harden() must never recompute what
    # _link computes; it may only call it and compare. If someone later adds a
    # "helpful" expected cost inline, this fires - which is the check the
    # orientation bug did not have.
    import inspect  # noqa: PLC0415 - self-check only
    body = inspect.getsource(harden)
    for op in ("np.hypot", "np.linalg.norm", "** 0.5", "math.sqrt"):
        assert op not in body, (
            f"harden() recomputes the cost ({op}) instead of calling _link. A "
            f"check that duplicates the logic under test agrees with itself and "
            f"nothing else.")
    print(f"adversarial_stitch self-check OK ({len(applied)}/{len(MUTANTS)} "
          f"mutations apply, harden() green on the real file, no duplicated "
          f"arithmetic)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    sys.exit(_selfcheck() or 0 if ap.parse_args().selfcheck else main())
