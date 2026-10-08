"""The SUFFICIENT test: does labelling fragments off the duel actually resolve it?

WHAT THIS IS FOR
    `st2_cleanframes.py` was the veto - it proved clean frames EXIST either side
    of a duel (63-74% two-sided at a realistic window). Existing is necessary and
    not sufficient. This runs the thing itself: label each fragment from its own
    clean frames with the SHIPPED reader, then resolve the duel by comparing
    labels, and score it against the number the single-frame reader gets on the
    same duels.

    `@M` The baseline it must beat, measured over 1,925 crops on all ten matches
    (`model/validation/st2_kit_parity.py`):

        single crop at the duel frame:  41.2% resolved, 1.5% wrong, 57.3% abstain

    Same duels, same reader, same resolution rule. The ONLY thing that changes is
    which frames the label is read from. That is the point: it isolates the
    temporal idea from everything else.

THE RULE, identical to the single-frame test so the comparison is honest
    own = vote over gid_from's clean frames BEFORE the duel   (the track's own history)
    a   = vote over gid_from's clean frames AFTER             (the correct continuation)
    b   = vote over gid_to's   clean frames AFTER             (the impostor)

    a == b        -> abstain   (both continuations read as the same kit)
    a == own      -> resolved
    otherwise     -> wrong

WHAT IT STILL CANNOT SEE, and this is the honest limit
    Like the oracle before it, this only looks at events that are ALREADY
    FAILURES. It cannot count associations the same rule would BREAK that the
    tracker currently gets right. A number here is an upper bound on the gain and
    a lower bound on nothing. Only `st2_ab.py` settles the net, and that needs
    the rule built into the tracker rather than measured beside it.

    It also uses GROUND TRUTH boxes, so localisation error is removed - the same
    ceiling caveat that applies to every number in st2_kit_parity.

`@M` RESULT - 1,200 cross-team duels, all ten matches, four windows

    All duels, including those with no usable fragment:

        window   resolved   wrong   abstain   no fragment
        +/-1 s     37.8%     1.0%     7.5%      53.8%
        +/-2 s     55.0%     2.6%    12.3%      30.1%
        +/-3 s     60.5%     4.2%    13.6%      21.8%
        +/-5 s     66.2%     4.8%    15.4%      13.7%
        single      41.2%     1.5%    57.3%        n/a     <- the baseline

    THE NUMBER THAT MATTERS IS THE DECISION RATE AT PRECISION, because an
    abstention and an error are not the same cost. An abstention leaves a track
    fragmented, which the stitch can repair; a confident wrong association welds
    two players together and nothing downstream repairs it.

        window   decides   precision   (of duels that HAD fragments)
        +/-1 s    83.8%      97.4%
        +/-2 s    82.4%      95.5%
        +/-3 s    82.6%      93.6%
        +/-5 s    82.1%      93.3%
        single    42.7%      96.5%     <- baseline, over all duels

    RETRACTED 2026-09-10 - the sentence that stood here compared two different
    denominators, and on a common one it inverts:

        ~~**At +/-1 s this is not a trade at all.** The fragment vote decides
        83.8% of duels at 97.4% precision where the single frame decides 42.7%
        at 96.5% - it nearly doubles the decision rate AND is more precise.~~

    83.8% is `(ok + wrong) / scored`, and `scored` EXCLUDES the 53.8% of duels
    with no usable fragment at +/-1 s. 42.7% is over ALL duels. `@D` Put both on
    all duels, straight off the unconditional table above:

        window   decides (ALL duels)   precision
        +/-1 s        38.8%              97.4%
        +/-2 s        57.6%              95.5%
        +/-3 s        64.7%              93.5%
        +/-5 s        71.0%              93.2%
        single        42.7%              96.5%   <- baseline

    **At +/-1 s the fragment vote decides FEWER duels than the single frame** -
    38.8% against 42.7% - for +0.9 points of precision. It is a trade, and at the
    tight window it is a losing one on coverage. **No configuration beats the
    baseline on both axes.** The defensible version is +/-2 s: **57.6% at 95.5%**,
    which is +14.9 points of coverage for -1.0 point of precision.

    What survives, and it is the real finding: reading the label AWAY from the
    duel frame converts abstentions into decisions at 93-97% precision. Reading
    the label off the duel frame was simply the wrong place to read it. That
    holds. "Nearly doubles the decision rate" does not.

    The trade appears only when the window is widened to buy coverage: no-fragment
    falls 53.8% -> 13.7% while precision falls 97.4% -> 93.3%. So the window is a
    real knob with a cost on each side, and the obvious shape is a CASCADE - try
    +/-1 s, widen only where no fragment exists - which should reach the wide
    window's coverage at close to the tight window's precision. Untested.

    ~~`@D` Scale, if it holds: A1 is 50.5% of all failures and 70.3% of A1 is
    cross-team, so at +/-5 s this addresses 0.505 x 0.703 x 0.662 = ~23% of all
    tracker failures, against ~1.7% newly broken.~~

    CORRECTED 2026-09-10 - the middle factor was mislabelled, and the reachable
    version of this chain is an order of magnitude smaller.

    **70.3% is not a property of A1.** It is the cross-team share of all 4,901 GT
    duels, whether or not the tracker failed there (commit 79f6dcf). `@V` A1's
    own cross-team share is 695/927 = **75.0%**
    (`failure-taxonomy-association (not published)` section 7), stated four lines from the
    70.3% in that same commit. `@D` Corrected:

        0.505 x 0.750 x 0.662 = ~25% addressed at +/-5 s
        0.505 x 0.750 x 0.048 = ~1.8% newly broken

    So the relabelling moves the headline UP; the error was in the label. Three
    things it still is not:

    1. **Three populations.** 0.505 is the two TUNED sequences; 0.750 and 0.662
       are all ten. Independence between the factors is assumed, never argued.
    2. **A ceiling, on GT boxes and GT duels.** See the limit above: this cannot
       see associations the rule would BREAK that are currently RIGHT.
    3. **Not what a post-hoc pass can reach.** `@D` Multiplying by the share of GT
       duels visible in our own output as two overlapping tracks puts the
       reachable share near **2.5%** (0.251 x 0.095). Treat that last factor as
       soft - the 9.5% behind it was itself retracted (b03bb78: DUEL_IOU 0.30 sits
       below the median pair IoU of 0.352; at 0.15 the visible share is 62.8%).

    **And it has now been measured rather than projected.** `@M` `st2_relink.py`
    built the post-hoc pass and scored it: 68 swaps across ten matches at IoU
    0.15, **mean AssA +0.0014, 6 of the 8 that moved, p = 0.145 -> DOES NOT
    SHIP**. Positive, not significant, and nothing like 23%. Only st2_ab settles
    this, and it did.

WHY IT READS THE VIDEO IN ONE LINEAR PASS
    The first version called `cap.set(CAP_PROP_POS_FRAMES, f)` per wanted frame.
    `@M` That costs ~1.2 s each on these 4096x1080 h264 files, because a random
    seek re-decodes from the nearest keyframe - and the sweep wants ~600 unique
    frames per sequence x 10 sequences x 4 windows, so it ran for EIGHT HOURS
    without finishing. Now: one pass per sequence with `grab()` (no decode) and
    `retrieve()` only on the frames actually wanted. Same frames, one decode of
    the file instead of 600 seeks into it.
"""
import argparse
import random
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

sys.path.insert(0, "model/validation")
sys.path.insert(0, "model/engine")
import st2_failtax as F  # noqa: E402
import st2_team as T  # noqa: E402
from track import team_by_shirt  # noqa: E402
import os
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download

CLIPS = _ST2 + "/mot/clips"
SEQS = ("117092", "117093", "118575", "118576", "118577", "118578",
        "128057", "128058", "132831", "132877")
CLEAN_IOU = 0.1
N_VOTE = 5          # clean frames per fragment; nearest to the duel win
N_DUELS = 120       # per sequence, sampled with a fixed seed


def feat(img, box):
    """team_by_shirt takes CORNERS; MOT GT is (x, y, w, h). See st2_kit_parity."""
    x, y, w, h = box
    v = np.asarray(team_by_shirt(img, (int(x), int(y), int(x + w), int(y + h))),
                   float)
    return None if v.size != 3 or not np.any(v) else v


def whiten_fit(X, rounds=3, k=2):
    """_kmeans' within-cluster metric, returning the transform as well."""
    X = np.asarray(X, np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    lab = cv2.kmeans(X, k, None, crit, 20, cv2.KMEANS_PP_CENTERS)[1].ravel()
    M, Z = np.eye(X.shape[1]), X
    for _ in range(rounds):
        S, n = np.zeros((X.shape[1],) * 2), 0
        for j in range(k):
            A = X[lab == j].astype(np.float64)
            if len(A) < 2:
                continue
            S += (A - A.mean(0)).T @ (A - A.mean(0))
            n += len(A) - 1
        w, V = np.linalg.eigh(S / max(n, 1) + 1e-9 * np.eye(X.shape[1]))
        M = V / np.sqrt(np.maximum(w, 1e-12))
        Z = np.ascontiguousarray(X @ M, np.float32)
        lab = cv2.kmeans(Z, k, None, crit, 20, cv2.KMEANS_PP_CENTERS)[1].ravel()
    return M, np.array([Z[lab == j].mean(0) for j in range(k)]), lab


def scan(seq, wanted):
    """{frame: decoded image} for `wanted`, in ONE linear pass. See the docstring.

    grab() advances without decoding, which is the whole trick; retrieve() pays
    the decode only on the frames we actually asked for.
    """
    cap = cv2.VideoCapture(f"{CLIPS}/{seq}.mp4")
    want, out, i = set(wanted), {}, 0
    hi = max(want) if want else -1
    while i <= hi:
        if not cap.grab():
            break
        if i in want:
            ok, img = cap.retrieve()
            if ok:
                out[i] = img
        i += 1
    cap.release()
    return out


_REF_CACHE = {}


def reference(seq, gt, n_frames=40):
    """Per-player kit label and the model, from frames spread across the clip."""
    if seq in _REF_CACHE:
        return _REF_CACHE[seq]
    fr = sorted(gt)
    picks = fr[::max(1, len(fr) // n_frames)][:n_frames]
    imgs = scan(seq, picks)
    samp = defaultdict(list)
    for f, img in imgs.items():
        for gid, b in gt[f].items():
            v = feat(img, b)
            if v is not None:
                samp[gid].append(v)
    ids = sorted(samp)
    M, cen, lab = whiten_fit(np.array([np.median(samp[i], axis=0) for i in ids]))
    _REF_CACHE[seq] = (dict(zip(ids, lab)), M, cen)
    return _REF_CACHE[seq]


def clean_frames(gt, gid, f, lo, hi):
    """Frames in [lo, hi) where this player's box overlaps nobody, nearest first."""
    out = [k for k in range(lo, hi)
           if gt.get(k) and gid in gt[k] and F.max_overlap(gt[k], gid) < CLEAN_IOU]
    return sorted(out, key=lambda k: abs(k - f))[:N_VOTE]


def study(seq, window, variant=".c03s"):
    gt = T.load_gt(seq)
    lab, M, cen = reference(seq, gt)
    out = F.analyse(seq, variant.lstrip("."), 0.5)
    ev = [e for e in (out["contam"] if isinstance(out, dict) else out[0])
          if e["gap"] == 0 and e["pair_iou"] > 0
          and lab.get(e["gid_from"]) is not None and lab.get(e["gid_to"]) is not None
          and lab[e["gid_from"]] != lab[e["gid_to"]]          # cross-team only
          and e["gid_from"] in gt.get(e["f"], {}) and e["gid_to"] in gt[e["f"]]]
    random.Random(2).shuffle(ev)
    ev = ev[:N_DUELS]

    # Collect every (frame, gid) the votes need, so each frame is decoded ONCE.
    need, plan = defaultdict(set), []
    for e in ev:
        f = e["f"]
        b = clean_frames(gt, e["gid_from"], f, f - window, f)
        a = clean_frames(gt, e["gid_from"], f, f + 1, f + window + 1)
        c = clean_frames(gt, e["gid_to"], f, f + 1, f + window + 1)
        if not (b and a and c):
            plan.append((e, None))
            continue
        for fr in b:
            need[fr].add(e["gid_from"])
        for fr in a:
            need[fr].add(e["gid_from"])
        for fr in c:
            need[fr].add(e["gid_to"])
        plan.append((e, (b, a, c)))

    imgs = scan(seq, need.keys())
    got = {}
    for fr, img in imgs.items():
        for gid in need[fr]:
            if gid in gt.get(fr, {}):
                v = feat(img, gt[fr][gid])
                if v is not None:
                    got[(fr, gid)] = int(np.argmin(
                        [np.linalg.norm(v @ M - c) for c in cen]))

    def vote(frames, gid):
        v = [got[(fr, gid)] for fr in frames if (fr, gid) in got]
        return Counter(v).most_common(1)[0][0] if v else None

    r = Counter()
    for e, fr in plan:
        r["n"] += 1
        if fr is None:
            r["no_fragment"] += 1
            continue
        b, a, c = fr
        own, ga, gb = (vote(b, e["gid_from"]), vote(a, e["gid_from"]),
                       vote(c, e["gid_to"]))
        if own is None or ga is None or gb is None:
            r["no_fragment"] += 1
            continue
        r["scored"] += 1
        if ga == gb:
            r["abstain"] += 1
        elif ga == own:
            r["ok"] += 1
        else:
            r["wrong"] += 1
    return r


def report(window, seqs=None):
    seqs = seqs or SEQS
    tot = Counter()
    print(f"\n=== window +/-{window} frames (+/-{window/25:.0f} s), "
          f"{N_VOTE} clean frames per fragment ===", flush=True)
    print(f"{'seq':>10s} {'duels':>7s} {'resolved':>10s} {'wrong':>8s} "
          f"{'abstain':>9s} {'no frag':>9s}")
    for s in seqs:
        r = study(s, window)
        tot.update(r)
        n = max(r["n"], 1)
        print(f"{s:>10s} {r['n']:7d} {100*r['ok']/n:9.1f}% {100*r['wrong']/n:7.1f}% "
              f"{100*r['abstain']/n:8.1f}% {100*r['no_fragment']/n:8.1f}%",
              flush=True)
    n = max(tot["n"], 1)
    print(f"{'TOTAL':>10s} {tot['n']:7d} {100*tot['ok']/n:9.1f}% "
          f"{100*tot['wrong']/n:7.1f}% {100*tot['abstain']/n:8.1f}% "
          f"{100*tot['no_fragment']/n:8.1f}%")
    print(f"  single-frame baseline on the same duels: "
          f"41.2% resolved / 1.5% wrong / 57.3% abstain")
    if tot["scored"]:
        print(f"  of the {tot['scored']} that HAD fragments: "
              f"{100*tot['ok']/tot['scored']:.1f}% resolved, "
              f"{100*tot['wrong']/tot['scored']:.1f}% wrong, "
              f"{100*tot['abstain']/tot['scored']:.1f}% abstain")
    return tot


def demo():
    """The vote must beat one bad frame, and the resolution rule must be sane."""
    got = {(1, 7): 0, (2, 7): 0, (3, 7): 1}          # one frame disagrees
    v = [got[(f, 7)] for f in (1, 2, 3)]
    assert Counter(v).most_common(1)[0][0] == 0, "majority must survive one bad frame"
    # resolution rule
    for own, a, b, want in ((0, 0, 1, "ok"), (0, 1, 0, "wrong"),
                            (0, 1, 1, "abstain"), (1, 1, 0, "ok")):
        got_it = "abstain" if a == b else ("ok" if a == own else "wrong")
        assert got_it == want, (own, a, b, got_it, want)
    gt = {f: {1: (0., 0., 20., 50.), 2: (900., 0., 20., 50.)} for f in range(100)}
    assert clean_frames(gt, 1, 50, 40, 50) == [49, 48, 47, 46, 45], \
        clean_frames(gt, 1, 50, 40, 50)
    # scan() must return exactly the frames asked for, and no others.
    import os
    seq = "128057"
    if os.path.exists(f"{CLIPS}/{seq}.mp4"):
        want = [0, 7, 31]
        g = scan(seq, want)
        assert set(g) == set(want), (sorted(g), want)
        assert all(v is not None and v.size for v in g.values())
    print("ok  (majority survives a bad frame; rule maps ok/wrong/abstain; "
          "clean frames come back nearest-first)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=75)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--seqs", nargs="*")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        demo()
    elif a.sweep:
        for w in (25, 50, 75, 125):
            report(w, a.seqs)
    else:
        report(a.window, a.seqs)
