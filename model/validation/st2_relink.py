"""W6: cut tracks at a duel, label each fragment off the duel, re-link by kit.

WHAT THIS IS
    `st2_fragment.py` measured the idea beside the tracker, on ground-truth boxes
    and ground-truth duels. This is the idea INSIDE the tracker: it finds duels in
    OUR OWN output, votes each fragment's kit over ITS OWN clean frames, and swaps
    the post-duel tails when the kit says the tracker took the wrong continuation.

    Output is a new variant, so `st2_ab.py` can score it against the input under
    the sign rule and the coalescence guard. That is the whole point - every
    number before this one was an oracle and could not see what the rule BREAKS.

`@M` WHAT THE MEASUREMENT SAID, and what it implies for the thresholds here
    CORRECTED 2026-09-10. This paragraph used to read "decides 83.8% of them at
    97.4% precision, against 42.7% at 96.5% for a single frame" - which is 83.8%
    over the 46% of duels that HAD fragments against a baseline over ALL duels.
    On a common denominator (`st2_fragment.py`'s own unconditional table):

        +/-1 s   decides 38.8% at 97.4%   <- FEWER than the baseline
        +/-2 s   decides 57.6% at 95.5%   <- the defensible setting
        +/-5 s   decides 71.0% at 93.2%
        single   decides 42.7% at 96.5%   <- baseline

    No window beats the baseline on both coverage and precision. What survives is
    the shape: reading the kit AWAY from the duel frame converts abstentions into
    decisions at 93-97% precision. Precision still falls as the window widens, so
    this still CASCADES: try the tight window first and widen only where no
    fragment exists. `st2_fragment.py` has the full table.

`@M` RESULT: IT DOES NOT SHIP, AND THE REASON IS THE FINDING

    `model/validation/st2_ab.py --a .c03s --b .c03s_rl` over all ten matches:

        mean HOTA -0.0011, mean AssA -0.0013
        AssA improved on 2 of the 4 sequences that moved, p = 1.000
        six sequences unchanged; 118576 lost 0.014 AssA from two swaps

    Eleven swaps across ten matches is not an intervention, it is a rounding
    error - and two of them were wrong enough to cost 118576 more than the other
    nine gained.

    **The decision was never the bottleneck. Finding the event is.** `@M` Of the
    duels ground truth says happened, the share that appear in OUR OWN output as
    two tracks overlapping - the only shape this code can act on:

        128057   624 GT duels ->  59 visible ( 9.5%)
        118576   732 GT duels ->  83 visible (11.3%)
        132831   303 GT duels ->  62 visible (20.5%)

    **RETRACTED THE SAME DAY. The 9.5% was MY THRESHOLD, not the tracker.**
    `DUEL_IOU` was 0.30. An audit pointed out that is below the measured median
    pair IoU of 0.352, so it cuts genuine duels before any tracker failure does.
    `@M` Sweeping it:

        threshold        128057            118576
        IoU >= 0.30    59 ( 9.5%)        83 (11.3%)
        IoU >= 0.15   392 (62.8%)       453 (61.9%)
        IoU >= 0.05   617 (98.9%)       721 (98.5%)

    `@M` And `.c03` (pre-stitch, 121 tracks) gives the same counts as `.c03s`
    (53 tracks), so "the stitch destroyed the evidence" was wrong too. The events
    are there. I could not see them.

    `@M` Re-run at IoU 0.15 - 68 swaps instead of 11 - and the sign flips:

        IoU 0.30:  mean AssA -0.0013, 2/4 improved, p = 1.000
        IoU 0.15:  mean AssA +0.0014, 6/8 improved, p = 0.145

    (The 0.15 p-value read 0.289 when this was first written. `st2_ab.sign_p` was
    two-sided then; `4a6bcb0` made the gate one-sided, which is the correct test
    for "did it improve", and 6/8 one-sided is 37/256 = 0.145. The verdict is
    unchanged.)

    **Still DOES NOT SHIP** - p = 0.145 is not 0.05 - but positive and pointing
    the right way. So the honest statement is *"a post-hoc pass is measurably
    positive and not yet significant"*, NOT *"it cannot work"*. The claim that the
    deferral must move inside the matcher is unproven, and it was an expensive
    thing to conclude from a constant I picked without checking it against the
    distribution it was cutting.

THREE RULES THAT KEEP IT CONSERVATIVE, because a wrong swap is unrepairable
    1. It only ever swaps when both sides read a CONFIDENT, DIFFERENT kit. An
       abstention leaves the tracker's own decision alone. An abstention costs a
       fragmented track, which the stitch can repair; a wrong swap welds two
       players and nothing downstream can.
    2. It requires a minimum number of clean frames on each side. A vote of one
       is not a vote.
    3. It never invents or deletes a detection. Same boxes, same count, new ids -
       the same contract `stitch.apply` keeps, which is why detection precision
       and recall cannot move and any change in DetA would be a bug.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

sys.path.insert(0, "model/validation")
sys.path.insert(0, "model/engine")
import st2_team as T  # noqa: E402
from st2_fragment import feat, scan, whiten_fit  # noqa: E402
from st2_mot import WORK  # noqa: E402 - root-relative, not cwd-relative
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download
CLIPS = _ST2 + "/mot/clips"
DUEL_IOU = 0.30        # two of our boxes this close IS the duel we measured
CLEAN_IOU = 0.10
N_VOTE = 5
MIN_VOTE = 3           # a vote of one is not a vote
WINDOWS = (25, 50, 125)   # the cascade: +/-1 s, +/-2 s, +/-5 s


def load(seq, variant):
    rows = json.load(open(f"{WORK}/{seq}{variant}.tracks.json"))
    by_f = defaultdict(dict)
    for r in rows["detections"]:
        by_f[r["f"]][r["id"]] = T.to_tlwh(r)
    return rows, by_f


def duels(by_f):
    """Frames where two of OUR tracks overlap hard, collapsed to one per pair-run."""
    hot = defaultdict(list)
    for f in sorted(by_f):
        ids = list(by_f[f])
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if T.iou(by_f[f][ids[i]], by_f[f][ids[j]]) >= DUEL_IOU:
                    a, b = sorted((ids[i], ids[j]))
                    hot[(a, b)].append(f)
    out = []
    for (a, b), fs in hot.items():
        run = [fs[0]]
        for f in fs[1:]:
            if f - run[-1] > 5:
                out.append((a, b, run[len(run) // 2]))
                run = []
            run.append(f)
        if run:
            out.append((a, b, run[len(run) // 2]))
    return out


def clean(by_f, tid, f, lo, hi, n=N_VOTE):
    """Frames where this track's box overlaps no OTHER track. Nearest first."""
    ok = []
    for k in range(lo, hi):
        b = by_f.get(k, {}).get(tid)
        if b is None:
            continue
        if max((T.iou(b, o) for i, o in by_f[k].items() if i != tid), default=0.0) < CLEAN_IOU:
            ok.append(k)
    return sorted(ok, key=lambda k: abs(k - f))[:n]


def relink(seq, variant=".c03s", out_variant=None, verbose=True):
    rows, by_f = load(seq, variant)
    ev = duels(by_f)
    if verbose:
        print(f"  {seq}: {len(ev)} duel events among our own tracks", flush=True)

    # Plan every vote first, at the tightest window that yields fragments.
    plan, need = [], defaultdict(set)
    for a, b, f in ev:
        for w in WINDOWS:
            ba = clean(by_f, a, f, f - w, f)
            bb = clean(by_f, b, f, f - w, f)
            aa = clean(by_f, a, f, f + 1, f + w + 1)
            ab = clean(by_f, b, f, f + 1, f + w + 1)
            if min(len(ba), len(bb), len(aa), len(ab)) >= MIN_VOTE:
                plan.append((a, b, f, ba, bb, aa, ab))
                for fr in ba + aa:
                    need[fr].add(a)
                for fr in bb + ab:
                    need[fr].add(b)
                break

    # Kit model from OUR OWN tracks, sampled across the clip - no ground truth.
    fr_all = sorted(by_f)
    picks = fr_all[::max(1, len(fr_all) // 40)][:40]
    imgs = scan(seq, set(picks) | set(need))
    samp = defaultdict(list)
    for f in picks:
        if f in imgs:
            for tid, b in by_f[f].items():
                v = feat(imgs[f], b)
                if v is not None:
                    samp[tid].append(v)
    ids = [i for i in sorted(samp) if len(samp[i]) >= 3]
    if len(ids) < 4:
        if verbose:
            print(f"  {seq}: too few tracks to fit a kit model, passing through")
        return write(rows, rows["detections"], seq, variant, out_variant, 0)
    M, cen, _ = whiten_fit(np.array([np.median(samp[i], axis=0) for i in ids]))

    lab = {}
    for f, tids in need.items():
        if f not in imgs:
            continue
        for tid in tids:
            b = by_f.get(f, {}).get(tid)
            if b is None:
                continue
            v = feat(imgs[f], b)
            if v is not None:
                lab[(f, tid)] = int(np.argmin(
                    [np.linalg.norm(v @ M - c) for c in cen]))

    def vote(frames, tid):
        v = [lab[(fr, tid)] for fr in frames if (fr, tid) in lab]
        return Counter(v).most_common(1)[0][0] if len(v) >= MIN_VOTE else None

    swaps, seen = [], Counter()
    for a, b, f, ba, bb, aa, ab in plan:
        pa, pb = vote(ba, a), vote(bb, b)
        qa, qb = vote(aa, a), vote(ab, b)
        if None in (pa, pb, qa, qb):
            seen["no_vote"] += 1
            continue
        if pa == pb or qa == qb:
            seen["abstain"] += 1        # rule 1: only confident, DIFFERENT kits
            continue
        seen["decided"] += 1
        if qa != pa:                    # the tails are on the wrong tracks
            swaps.append((a, b, f))
            seen["swapped"] += 1

    out = apply_swaps(rows["detections"], swaps)
    if verbose:
        print(f"  {seq}: decided {seen['decided']}, abstained {seen['abstain']}, "
              f"no-vote {seen['no_vote']} -> {len(swaps)} swaps", flush=True)
    return write(rows, out, seq, variant, out_variant, len(swaps))


def apply_swaps(dets, swaps):
    """Exchange the post-duel tails of two ids. Same boxes, same count, new ids."""
    ids = {r["id"] for r in dets}
    nxt = max(ids) + 1 if ids else 1
    out = [dict(r) for r in dets]
    for a, b, f in sorted(swaps, key=lambda s: s[2]):
        tmp = nxt
        nxt += 1
        for r in out:
            if r["f"] > f and r["id"] == a:
                r["id"] = tmp
        for r in out:
            if r["f"] > f and r["id"] == b:
                r["id"] = a
        for r in out:
            if r["id"] == tmp:
                r["id"] = b
    return out


def write(rows, dets, seq, variant, out_variant, n_swaps):
    ov = out_variant or f"{variant}_rl"
    p = f"{WORK}/{seq}{ov}.tracks.json"
    json.dump({**rows, "detections": dets,
               "_relink": {"swaps": n_swaps, "duel_iou": DUEL_IOU,
                           "windows": list(WINDOWS), "min_vote": MIN_VOTE,
                           "source": f"{seq}{variant}"}}, open(p, "w"))
    return p


def demo():
    """Swapping tails must preserve every box and exchange exactly the tails."""
    dets = ([{"f": f, "id": 1, "x": 1.0, "y": 1.0, "w": 2.0, "h": 2.0} for f in range(10)]
            + [{"f": f, "id": 2, "x": 9.0, "y": 9.0, "w": 2.0, "h": 2.0} for f in range(10)])
    out = apply_swaps(dets, [(1, 2, 4)])
    assert len(out) == len(dets), "no box may be added or removed"
    assert Counter(r["id"] for r in out) == Counter(r["id"] for r in dets), \
        "id COUNTS must be preserved, only their assignment changes"
    before = {(r["f"], r["id"]) for r in out if r["f"] <= 4}
    assert before == {(f, i) for f in range(5) for i in (1, 2)}, before
    tail1 = [r for r in out if r["f"] > 4 and r["id"] == 1]
    assert all(abs(r["x"] - 9.0) < 1e-9 for r in tail1), "id 1 must carry 2's tail"
    # a no-op swap list must be byte-identical
    assert apply_swaps(dets, []) == dets
    print(f"ok  ({len(dets)} boxes preserved; tails exchanged after f=4; "
          f"empty swap list is a no-op)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="*")
    ap.add_argument("--variant", default=".c03s")
    ap.add_argument("--duel-iou", type=float, default=DUEL_IOU)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        demo()
    else:
        globals()["DUEL_IOU"] = a.duel_iou
        seqs = a.seqs or sorted({s.split(".")[0] for s in os.listdir(WORK)
                                 if s.endswith(f"{a.variant}.tracks.json")})
        for s in seqs:
            relink(s, a.variant)
