"""Is the kit label still readable AT a duel? The number W6 turns on.

WHY
    `@M` model/validation/st2_oracle_duel.py: 70.3% of duels are cross-team, and that is
    the ceiling of a perfect kit label - the signal that would carry a
    two-hypothesis resolver. Motion alone is only 60.3% against a 50% chance
    line. So W6's value rests almost entirely on whether the kit label works AT
    THE MOMENT OF A DUEL, and nothing had measured that.

    There is reason to expect it not to. A duel is exactly where two players'
    boxes overlap (median pair IoU 0.352), so a torso crop taken there contains
    some of the OTHER kit. `@M` The overall per-detection figure is 70.7%
    (model/validation/team_eval.py); this asks what it is where it matters.

METHOD, AND IT IS DELIBERATELY GENEROUS
    Team model: per-player median torso colour over frames sampled across the
    clip, k-means k=2 - the same construction `st2_team.gt_teams` uses, so the
    reference labels are that function's.

    Then, on GROUND TRUTH boxes (not ours), classify a single frame's torso
    crop to the nearer centre and ask whether it matches that player's own
    reference label. Using GT boxes removes localisation error entirely, so this
    is the CEILING of a per-detection kit label - a real one reads a predicted
    box, which at a duel is worse placed than GT.

    Controls are the SAME players in frames where their GT box overlaps nobody.
    Duel minus control is the effect of the overlap itself, with player, kit,
    lighting and sequence held fixed - which is the comparison that matters, not
    the absolute number.

RESULT (2026-09-09)

    single crop, 3 sequences, 360 each:

        at a duel   68.6%        control (same players, no overlap)  83.6%
        delta       -15.0 points

    pair resolution, 5 sequences, 974 CROSS-TEAM duels - the decisive one:

        resolved correctly   41.6%
        resolved wrongly      3.9%
        ABSTAINED            54.5%   both players read as the SAME kit

    `@D` So the team signal is worth far less than the oracle's "perfect label"
    line suggested. Folding these in: on cross-team duels it gives
    0.416 + 0.545 x 0.603 (motion on the abstentions) = 74.5%, and over all
    duels 0.703 x 0.745 + 0.297 x 0.603 = **70.3%** against motion's 60.3%.
    The oracle's 88.2% assumed a perfect label; the real ceiling is ~70%.

    THE ABSTENTION IS THE INTERESTING HALF - AND IT IS NOT FIXABLE BY CROPPING.
    54.5% is not the classifier being unsure, it is both crops reading as the
    SAME kit. The obvious diagnosis was that `st2_team.torso` takes the
    upper-middle 50% of a box which at a duel contains the other player, so a
    duel-aware crop should rescue it. `@M` Six variants were run on the same 974
    cross-team duels - masking the other player's box, a narrower column, both,
    and per-pixel nearest-centre voting instead of a median:

        baseline        41.6% resolved   3.9% wrong   54.5% abstain
        mask-other      41.5%            6.6%         52.0%
        narrow          39.5%            7.3%         53.2%
        narrow+mask     40.1%            8.8%         51.0%
        pixel-vote      39.7%            4.8%         55.4%
        mask+vote       40.3%            6.7%         53.0%

    **Not one beats the baseline.** Masking trades abstentions for outright
    errors at about one for one.

    `@M` The reason, measured on 1,148 duel crops, and it is the useful part:

        median torso crop                                149 px  (~12 x 12)
        share of those pixels covered by the OTHER box    70.6%
        crops where masking leaves fewer than 6 px         38.5%

    So the diagnosis was right and the fix is arithmetically impossible. At a
    duel the other player owns **70.6% of the torso crop**, and removing him
    leaves too little to classify - in 38.5% of crops, nothing usable at all.
    A pixel-accurate segmentation would not save it either: the visible sliver
    of the occluded player is a handful of pixels.

    **This closes the single-frame route and points at the temporal one.** The
    label has to be decided on the clean frames either side of the collision and
    propagated through it - which is what every system in
    research-mot-identical-targets (not published) does, and it is now a measured
    conclusion here rather than a borrowed one.
"""
import argparse
import random
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

sys.path.insert(0, "model/validation")
import st2_failtax as F  # noqa: E402
import st2_team as T  # noqa: E402
import os
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download

SEQS = ("117092", "117093", "118575", "118576", "118577", "118578",
        "128057", "128058", "132831", "132877")
CLIPS = _ST2 + "/mot/clips"
N_PER = 120


def centres(seq, gt):
    """Reference team model and per-player label, from st2_team's own method."""
    lab, _, _ = T.gt_teams(seq, gt)
    cap = cv2.VideoCapture(f"{CLIPS}/{seq}.mp4")
    frames = sorted(gt)
    picks = frames[::max(1, len(frames) // 40)][:40]
    samples = defaultdict(list)
    for f in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        limg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        for gid, box in gt[f].items():
            c = T.torso(limg, box)
            if c is not None:
                samples[gid].append(np.median(c.reshape(-1, 3), axis=0))
    cap.release()
    cen = {}
    for k in (0, 1):
        v = [np.median(samples[g], axis=0) for g in samples if lab.get(g) == k]
        if v:
            cen[k] = np.median(v, axis=0)
    return lab, cen


def sample_frames(seq, gt, variant=".c03s"):
    """(duel (frame, gid) pairs, control (frame, gid) pairs) for the same players."""
    out = F.analyse(seq, variant.lstrip("."), 0.5)
    ev = [e for e in (out["contam"] if isinstance(out, dict) else out[0])
          if e["gap"] == 0 and e["pair_iou"] > 0]
    duel = []
    for e in ev:
        for g in (e["gid_from"], e["gid_to"]):
            if g in gt.get(e["f"], {}):
                duel.append((e["f"], g))
    # controls: same players, frames where their box overlaps nothing
    who = {g for _, g in duel}
    ctrl = []
    for f in sorted(gt):
        for g in who & set(gt[f]):
            if F.max_overlap(gt[f], g) == 0.0:
                ctrl.append((f, g))
    random.Random(0).shuffle(duel)
    random.Random(1).shuffle(ctrl)
    return duel[:N_PER], ctrl[:N_PER]


def read(seq, pairs, gt, lab, cen):
    """Classify one torso crop per (frame, gid). -> (n, correct)."""
    cap = cv2.VideoCapture(f"{CLIPS}/{seq}.mp4")
    by_f = defaultdict(list)
    for f, g in pairs:
        by_f[f].append(g)
    n = ok_n = 0
    for f in sorted(by_f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        got, img = cap.read()
        if not got:
            continue
        limg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        for g in by_f[f]:
            if g not in lab or len(cen) < 2:
                continue
            c = T.torso(limg, gt[f][g])
            if c is None:
                continue
            col = np.median(c.reshape(-1, 3), axis=0)
            pick = min(cen, key=lambda k: np.linalg.norm(col - cen[k]))
            n += 1
            ok_n += (pick == lab[g])
    cap.release()
    return n, ok_n


def report(seqs=None, variant=".c03s"):
    seqs = seqs or SEQS
    tot = Counter()
    print(f"\n{'seq':>10s} {'at duel':>18s} {'control':>18s} {'delta':>8s}")
    for s in seqs:
        gt = T.load_gt(s)
        lab, cen = centres(s, gt)
        d, c = sample_frames(s, gt, variant)
        dn, dok = read(s, d, gt, lab, cen)
        cn, cok = read(s, c, gt, lab, cen)
        tot["dn"] += dn; tot["dok"] += dok; tot["cn"] += cn; tot["cok"] += cok
        f = lambda a, b: f"{a:4d}/{b:<4d}{100*a/b:6.1f}%" if b else f"{'-':>18s}"
        dd = (dok/dn - cok/cn)*100 if dn and cn else float("nan")
        print(f"{s:>10s} {f(dok, dn):>18s} {f(cok, cn):>18s} {dd:+7.1f}")
    f = lambda a, b: f"{a:4d}/{b:<4d}{100*a/b:6.1f}%" if b else "-"
    print(f"\n{'TOTAL':>10s} {f(tot['dok'], tot['dn']):>18s} "
          f"{f(tot['cok'], tot['cn']):>18s} "
          f"{100*(tot['dok']/max(tot['dn'],1) - tot['cok']/max(tot['cn'],1)):+7.1f}")
    print("\n  GT boxes, so this is the CEILING of a per-detection kit label.")
    print("  Chance is 50.0%. Control = same players, frames with zero overlap.")
    return tot


def demo():
    """A synthetic crop nearer its own centre must classify to it."""
    cen = {0: np.array([50.0, 120.0, 130.0]), 1: np.array([200.0, 130.0, 120.0])}
    col = np.array([60.0, 122.0, 129.0])
    assert min(cen, key=lambda k: np.linalg.norm(col - cen[k])) == 0
    col2 = np.array([190.0, 128.0, 121.0])
    assert min(cen, key=lambda k: np.linalg.norm(col2 - cen[k])) == 1
    print("ok  (nearest-centre assignment picks the right kit on both sides)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="*")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    demo() if a.demo else report(a.seqs)


def pair_resolve(seq, variant=".c03s", n_per=200):
    """The decisive measure: would the kit label RESOLVE the duel?

    A two-hypothesis resolver does not need each crop classified in isolation.
    It needs the two candidates TOLD APART, and the one matching the track's own
    accumulated team picked. The track was on `gid_from`, so its team is
    `lab[gid_from]`, and the resolution is correct iff the from-player still
    reads as that team AND the to-player does not.

    Measured directly rather than derived from single-crop accuracy, because
    the two crops at one duel share lighting, frame and the overlap itself and
    their errors are anything but independent.
    """
    gt = T.load_gt(seq)
    lab, cen = centres(seq, gt)
    out = F.analyse(seq, variant.lstrip("."), 0.5)
    ev = [e for e in (out["contam"] if isinstance(out, dict) else out[0])
          if e["gap"] == 0 and e["pair_iou"] > 0]
    ev = [e for e in ev if lab.get(e["gid_from"]) is not None
          and lab.get(e["gid_to"]) is not None
          and lab[e["gid_from"]] != lab[e["gid_to"]]          # cross-team only
          and e["gid_from"] in gt.get(e["f"], {})
          and e["gid_to"] in gt[e["f"]]]
    random.Random(2).shuffle(ev)
    ev = ev[:n_per]
    by_f = defaultdict(list)
    for e in ev:
        by_f[e["f"]].append(e)
    cap = cv2.VideoCapture(f"{CLIPS}/{seq}.mp4")
    r = Counter()
    for f in sorted(by_f):
        got, img = cap.read() if False else (None, None)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        got, img = cap.read()
        if not got:
            continue
        limg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        for e in by_f[f]:
            pick = {}
            for key, g in (("from", e["gid_from"]), ("to", e["gid_to"])):
                c = T.torso(limg, gt[f][g])
                if c is None:
                    break
                col = np.median(c.reshape(-1, 3), axis=0)
                pick[key] = min(cen, key=lambda k: np.linalg.norm(col - cen[k]))
            if len(pick) < 2:
                continue
            r["n"] += 1
            own = lab[e["gid_from"]]
            if pick["from"] == pick["to"]:
                r["abstain"] += 1                    # both read the same kit
            elif pick["from"] == own:
                r["ok"] += 1
            else:
                r["wrong"] += 1
    cap.release()
    return r
