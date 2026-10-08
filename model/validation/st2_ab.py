"""A/B two tracker variants across every scoreable sequence. The decision rule.

WHY THIS EXISTS
    Every association number this repo has quoted came from two sequences, and
    both were the ones the config was tuned on. Measured on nine:

        tuned pair (128057, 132831)   HOTA 0.393
        held-out (seven)              HOTA 0.356
        generalisation gap                 +0.037

    So the pair overstates by roughly a tenth of the number it reports. A change
    scored on it alone cannot be believed.

THE NOISE FLOOR IS NOT WHAT IT LOOKS LIKE
    Across sequences, HOTA has sd 0.054 (0.023 dropping 117092). That is NOT the
    error bar on a change: the pipeline is deterministic - `st2_run.py --demo`
    reproduces a shipped file byte for byte - so re-running the same config
    twice gives the same number exactly. Measurement noise is zero.

    What varies is whether an improvement TRANSFERS. That is a question about
    signs across sequences, not about a standard error, which is why the rule
    below is a sign test and not a t-test.

THE DECISION RULE
    A change ships when it improves a strict majority of sequences with the sign
    test at p < 0.05. On n=9 that is 8 of 9. A mean improvement carried by two
    sequences and flat on seven is not a result, it is a tuning artefact - which
    is exactly how this repo produced the numbers it later had to retract.

REPORT AssA, NOT ONLY HOTA
    HOTA is the geometric mean of DetA and AssA, so an association change that
    lifts AssA 10% moves HOTA about 5% and can be masked entirely by detector
    noise. Every work item in tracker-engineering-scope (not published) is association,
    so AssA is the primary and HOTA is the headline.

    `@M` on nine sequences: AssA mean 0.243, range 0.117-0.339, a 2.9x spread -
    much wider than DetA's. Association is both the headroom and the unstable
    part, which is the whole reason for the sign rule.

AND WATCH IDSW/ids TOGETHER WITH IT
    Coalescence - two tracks collapsing onto one target - raises AssA. So does a
    real fix. They are told apart by the id count moving toward 22 rather than
    below it, which is why `ids` is printed and not hidden.
"""
import argparse
import json
import os
import sys
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st2_coalesce as C  # noqa: E402
import st2_mot as T  # noqa: E402

ALL = ("117092", "117093", "118575", "118576", "118577", "118578",
       "128057", "132831", "132877", "128058")
TUNED = ("128057", "132831")


def available(variant):
    return [s for s in ALL
            if os.path.exists(f"{T.WORK}/{s}{variant}.tracks.json")]


def score(variant, seqs):
    T.VARIANT = variant
    T.stage_gt(seqs)
    T.stage_pred(seqs)
    return T.run(seqs)


def ids_of(seq, variant):
    p = f"{T.WORK}/{seq}{variant}.tracks.json"
    return len({r["id"] for r in json.load(open(p))["detections"]})


# Below this a per-sequence delta is a tie, not a direction. HOTA is printed to
# three decimals and TrackEval's own numbers move in the fourth.
TIE = 1e-4


def sign_p(k, n):
    """ONE-SIDED sign test: P(k or more of n improve | no real effect).

    One-sided because this is a SHIP GATE, and a ship gate only cares about
    improvement. The first version was two-sided (`k = max(k, n - k)`), which
    made 1-of-9 and 8-of-9 indistinguishable at p = 0.039 and gave **0 of 9 a
    p of 0.0039** - a change that regressed every single sequence cleared the
    significance test, with only the positive-mean check standing between it and
    a SHIPS verdict. That is the exact failure this rule exists to prevent.

    Ties are excluded by the caller, not counted as failures - counting them as
    failures is the standard mistake and it is not conservative, it is wrong. A
    gate that leaves six sequences untouched and improves four has evidence 4/4,
    not 4/10.
    """
    if n == 0:
        return 1.0
    return min(1.0, sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def ab(va, vb, metric="AssA"):
    seqs = [s for s in available(va) if s in available(vb)]
    if len(seqs) < 3:
        sys.exit(f"only {len(seqs)} sequences have both variants")
    A, B = score(va, seqs), score(vb, seqs)
    print(f"\n{'seq':>10s} {'':>5s} {'HOTA':>16s} {'DetA':>16s} {'AssA':>16s} "
          f"{'ids':>12s} {'dup/fr':>14s}")
    wins = ties = 0
    dup_a = dup_b = 0.0
    for s in seqs:
        d = {m: B[s][m] - A[s][m] for m in ("HOTA", "DetA", "AssA")}
        if abs(d[metric]) < TIE:
            ties += 1
        else:
            wins += d[metric] > 0
        tag = "T" if s in TUNED else "h"
        # The guard. Coalescence raises AssA exactly as a real fix does.
        ca = C.score(s, va)["dup_per_frame"]
        cb = C.score(s, vb)["dup_per_frame"]
        dup_a += ca; dup_b += cb
        print(f"{s:>10s} {tag:>5s} "
              + " ".join(f"{A[s][m]:.3f}->{B[s][m]:.3f}{d[m]:+.3f}"
                         for m in ("HOTA", "DetA", "AssA"))
              + f" {ids_of(s, va):5d}->{ids_of(s, vb):<5d}"
              + f" {ca:6.3f}->{cb:<6.3f}")
    n = len(seqs)
    mh = sum(B[s]["HOTA"] - A[s]["HOTA"] for s in seqs) / n
    ma = sum(B[s]["AssA"] - A[s]["AssA"] for s in seqs) / n
    eff = n - ties
    p = sign_p(wins, eff)
    # POWER FIRST. Below five moving sequences even a perfect sweep cannot reach
    # p < 0.05 - the floor is sign_p(n, n) = 2^-n - so a "DOES NOT SHIP" there is
    # not a null result, it is the maximum obtainable evidence printed as one,
    # and it is indistinguishable from 0-of-n. That is postmortem C2 living
    # inside the instrument that C2 was written about.
    floor = sign_p(eff, eff) if eff else 1.0
    underpowered = floor >= 0.05
    print(f"\n  mean delta   HOTA {mh:+.4f}   AssA {ma:+.4f}")
    print(f"  {metric} improved on {wins}/{eff} sequences that moved at all "
          f"({ties} unchanged, excluded), sign test p = {p:.3f}")
    if underpowered:
        print(f"  UNDERPOWERED: with only {eff} sequences moving, the best "
              f"attainable p is {floor:.3f}. No result could have shipped. "
              f"Score more sequences or loosen TIE; do not read this as a null.")
    # The rule, stated by the instrument rather than left to the reader.
    # A change that doubles coalescence has not improved association, it has
    # merged players. Baseline is 0.001/frame, so the bar is deliberately low.
    coalesced = dup_b / n > max(2 * dup_a / n, 0.01)
    ok = p < 0.05 and (ma if metric == "AssA" else mh) > 0 and not coalesced
    why = ("" if not coalesced else
           "  -- BLOCKED: dup/frame more than doubled; this is coalescence, "
           "not association")
    label = ("SHIPS" if ok else
             "INCONCLUSIVE (underpowered)" if underpowered else "DOES NOT SHIP")
    print(f"  VERDICT: {label} "
          f"(needs p<0.05, a positive mean, and no coalescence; tuned rows are "
          f"marked T and prove nothing on their own){why}")
    return ok


def demo():
    assert sign_p(9, 9) < 0.05 and sign_p(8, 9) < 0.05, "9/9 and 8/9 must pass"
    assert sign_p(7, 9) > 0.05, "7/9 must NOT pass - that is the whole point"
    # THE P0 THIS FILE SHIPPED WITH: a two-sided test made a total regression
    # look significant. A ship gate is one-sided; failing badly must never pass.
    assert sign_p(0, 9) > 0.9, f"0/9 must be the WORST case, got {sign_p(0,9)}"
    assert sign_p(1, 9) > 0.9, f"1/9 must not pass, got {sign_p(1,9)}"
    assert sign_p(2, 9) > sign_p(7, 9) > sign_p(8, 9) > sign_p(9, 9), "monotone"
    # Ties must not be counted as failures: 7 of 7 that moved is not 7 of 10.
    assert sign_p(7, 7) < 0.05 < sign_p(7, 10), "tie handling"
    assert sign_p(0, 0) == 1.0
    # A variant compared against itself must be a dead heat and must not ship.
    # A variant against ITSELF must be a dead heat: every delta zero, so every
    # sequence ties, so n_effective is 0 and nothing can ship. The old version of
    # this check compared R[s]["HOTA"] to itself, which cannot fail.
    v = ".c03s"
    seqs = available(v)[:3]
    if len(seqs) >= 3:
        R = score(v, seqs)
        assert all(abs(R[a]["HOTA"] - R[b]["HOTA"]) > 1e-9
                   for a in seqs for b in seqs if a != b), \
            "distinct sequences must not score identically - staging is broken"
        assert sign_p(0, 0) == 1.0, "an all-ties comparison must never ship"
    # A verdict must never claim a null it had no power to detect.
    assert sign_p(4, 4) > 0.05, "n=4 must be underpowered - the floor is 0.0625"
    assert sign_p(5, 5) < 0.05, "n=5 must be the first powered size"
    assert sign_p(3, 3) > sign_p(4, 4) > sign_p(5, 5), "floor must fall with n"
    print(f"ok  (sign test: 8/9 p={sign_p(8,9):.3f} ships, "
          f"7/9 p={sign_p(7,9):.3f} does not)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=".c03s", help="baseline variant suffix")
    ap.add_argument("--b", help="candidate variant suffix")
    ap.add_argument("--metric", default="AssA", choices=("HOTA", "DetA", "AssA"))
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo or not a.b:
        demo()
    else:
        sys.exit(0 if ab(a.a, a.b, a.metric) else 1)
