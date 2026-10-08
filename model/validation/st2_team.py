"""Is our association failure a TEAM problem? Measure, do not assume.

WHY THIS EXISTS
    We know association is the constraint (oracle AssA 0.523 vs pipeline ~0.27).
    "Association" is not one failure. It is two, and they have different fixes:

      GT-side switch   one player's track id changes      -> identity LOST
      track-side switch one track id changes player       -> identity STOLEN

    A team label is a hard constraint that can only ever block a *cross-team*
    track-side switch. It cannot re-link a fragmented player, and it cannot
    separate two team-mates. So the ceiling on "fix teams, fix association" is
    exactly the cross-team share of track-side switches - and this measures it.

TEAM GROUND TRUTH
    The MOT GT ships bare ids 1..22 with no team column, and joining them to the
    roster in raw/<match>/*_metadata.xml needs the pitch->panorama chain plus the
    clip's unknown offset into the full match. Colour is cheaper and its own
    answer: if two kits do not separate by colour, a team label is not free
    anyway. Per id: median torso colour over sampled frames, k-means k=2. The
    printed margin says whether to trust the split.

OUR TEAM FIELD
    NOT the authority - `model/validation/team_eval.py` is. That scores against
    SoccerTrack's real per-detection `team` truth on 118575 and reports 70.7%
    per detection / 93.0% per track, with a hard 73.1% ceiling because stitched
    tracks weld opposite-side players together. The line printed here is a weak
    cross-check on a DIFFERENT pair (the raw-panorama challenge clips) against a
    colour proxy rather than truth, best-case label mapping. Quote team_eval.py.

WHAT IT FOUND (128057 + 132831, c03s, IoU 0.5)
    GT-side switches      663     identity LOST      - team cannot help
    track-side switches  1171     identity STOLEN
      cross-team          833  =  71.1%              vs 51.9% by chance
    ceiling if team assignment were perfect and hard-enforced: 833/1834 = 45.4%

    Stable at IoU 0.3 (45.0%). So team assignment is worth real association -
    but it is not the whole of it: 36% of the errors are fragmentation, which a
    team label cannot touch, and the arrow also runs the other way. Team
    accuracy here is 88-93% per track, agreeing with team_eval.py's 93.0%, and
    `model/engine/track.py` measured that welded tracks are what cap it at 73.1% on
    118575 - not the labeller.

    FIRST VERSION OF THIS FILE WAS WRONG. It read tracks.json x,y as top-left,
    so every box sat a full box height off and only 17-19 of 22 players ever
    matched. It reported a 27.6% ceiling and a 51% team accuracy. See `to_tlwh`.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download

GT_SRC = _ST2 + "/mot"
CLIPS = f"{GT_SRC}/clips"
TRACKS = "model/validation/out/stc2025"
SEQS = ("128057", "132831")   # challenge pair, same as st2_mot.SEQS
IOU_MIN = 0.5   # overridden by --iou; see sensitivity note in the ledger
N_SAMPLE = 40


def load_gt(seq):
    """frame -> {gt_id: (x, y, w, h)}. GT is 0-based, as is tracks.json."""
    by_f = defaultdict(dict)
    with open(f"{GT_SRC}/{seq}.txt") as fh:
        for line in fh:
            p = line.split(",")
            by_f[int(p[0])][int(p[1])] = tuple(float(v) for v in p[2:6])
    return by_f


def to_tlwh(r):
    """tracks.json is BOTTOM-CENTRE x,y + w,h. GT is MOT top-left.

    This convention is not guessable from the field names and reading it wrong
    offsets every box by a full box height, which on a 22x41 px player destroys
    IoU outright while still *looking* like a tracker that merely misses a lot.
    It is `st2_mot.stage_pred`'s conversion and `demo()` pins the two together,
    so this cannot drift from the harness that produces the HOTA numbers.
    """
    return (r["x"] - r["w"] / 2.0, r["y"] - r["h"], r["w"], r["h"])


def load_tracks(path):
    by_f, team = defaultdict(dict), {}
    for d in json.load(open(path))["detections"]:
        by_f[d["f"]][d["id"]] = to_tlwh(d)
        if d.get("team"):
            team.setdefault(d["id"], Counter())[d["team"]] += 1
    return by_f, {i: c.most_common(1)[0][0] for i, c in team.items()}


def torso(frame, box):
    """Upper-middle of the box: shirt, minimal grass and minimal shorts."""
    x, y, w, h = box
    x0, x1 = int(x + 0.25 * w), int(x + 0.75 * w)
    y0, y1 = int(y + 0.15 * h), int(y + 0.45 * h)
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return None
    crop = frame[max(y0, 0):y1, max(x0, 0):x1]
    return None if crop.size < 12 else crop


def gt_teams(seq, by_f):
    """Per GT id: median torso colour over sampled frames -> k-means k=2."""
    cap = cv2.VideoCapture(f"{CLIPS}/{seq}.mp4")
    frames = sorted(by_f)
    picks = frames[::max(1, len(frames) // N_SAMPLE)][:N_SAMPLE]
    samples = defaultdict(list)
    for f in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        for gid, box in by_f[f].items():
            c = torso(lab, box)
            if c is not None:
                samples[gid].append(np.median(c.reshape(-1, 3), axis=0))
    cap.release()
    ids = sorted(samples)
    X = np.array([np.median(samples[i], axis=0) for i in ids])
    from sklearn.cluster import KMeans
    km = KMeans(2, n_init=10, random_state=0).fit(X)
    d = np.linalg.norm(X[:, None] - km.cluster_centers_[None], axis=2)
    margin = np.abs(d[:, 0] - d[:, 1])           # Lab distance to own vs other
    return dict(zip(ids, km.labels_)), dict(zip(ids, margin)), len(picks)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def match(gt, tr, thr=None):
    """Hungarian on IoU. Greedy would double-count in a crowd, which is the
    exact regime we are measuring."""
    gids, tids = list(gt), list(tr)
    if not gids or not tids:
        return {}
    M = np.array([[iou(gt[g], tr[t]) for t in tids] for g in gids])
    r, c = linear_sum_assignment(-M)
    k = IOU_MIN if thr is None else thr
    return {tids[j]: gids[i] for i, j in zip(r, c) if M[i, j] >= k}


def runs(seq_of_pairs):
    """[(frame, val)] -> number of value changes, and the change pairs."""
    out, prev = [], None
    for _, v in seq_of_pairs:
        if prev is not None and v != prev:
            out.append((prev, v))
        prev = v
    return out


def analyse(seq, variant):
    global IOU_MIN
    gt = load_gt(seq)
    tpath = f"{TRACKS}/{seq}.{variant}.tracks.json"
    if not os.path.exists(tpath):
        sys.exit(f"no tracks: {tpath}")
    tr, our_team = load_tracks(tpath)

    team, margin, n_pick = gt_teams(seq, gt)
    sizes = Counter(team.values())
    print(f"\n=== {seq} ({variant}) ===")
    print(f"team clusters from {n_pick} sampled frames: "
          f"{sizes[0]}/{sizes[1]} of {len(team)} ids, "
          f"min margin {min(margin.values()):.1f} Lab, "
          f"median {np.median(list(margin.values())):.1f}")

    by_track, by_gt = defaultdict(list), defaultdict(list)
    for f in sorted(set(gt) & set(tr)):
        for tid, gid in match(gt[f], tr[f]).items():
            by_track[tid].append((f, gid))
            by_gt[gid].append((f, tid))

    track_sw = [sw for t in by_track.values() for sw in runs(t)]
    gt_sw = [sw for g in by_gt.values() for sw in runs(g)]
    cross = [a for a in track_sw if team.get(a[0]) != team.get(a[1])]

    # The colour split is an ESTIMATE (it lands 10/12, not the roster's 11/11),
    # so repeat on ids whose cluster assignment is not marginal.
    hi = {i for i, m in margin.items() if m >= np.median(list(margin.values()))}
    conf = [a for a in track_sw if a[0] in hi and a[1] in hi]
    conf_x = [a for a in conf if team[a[0]] != team[a[1]]]
    # ...and its OWN chance baseline: the high-margin half is not team-balanced,
    # so 52% is the wrong yardstick for it.
    hs = Counter(team[i] for i in hi)
    hn = sum(hs.values())
    conf_chance = (sum(hs[k] * (hn - hs[k]) for k in hs) / (hn * (hn - 1))
                   if hn > 1 else 0.0)

    # Chance: a track that steals a uniformly random other player is cross-team
    # 11/21 = 52.4% of the time. Anything at chance is not a team signal.
    sizes_l = sorted(sizes.values())
    n = sum(sizes_l)
    chance = sum(sizes[k] * (n - sizes[k]) for k in sizes) / (n * (n - 1))

    print(f"tracks {len(by_track)} for {len(by_gt)} matched players")
    print(f"GT-side switches   (identity LOST, team cannot help): {len(gt_sw)}")
    print(f"track-side switches(identity STOLEN)                : {len(track_sw)}")
    print(f"  ...of which CROSS-team (a team label could block)  : {len(cross)}"
          f"  = {100 * len(cross) / max(1, len(track_sw)):.1f}%")
    print(f"  ...of which SAME-team (team label is no help)      : "
          f"{len(track_sw) - len(cross)}")
    print(f"  chance cross-team if switches were random pairs     : "
          f"{100 * chance:.1f}%")
    print(f"  high-confidence colour ids only                     : "
          f"{len(conf_x)}/{len(conf)} = "
          f"{100 * len(conf_x) / max(1, len(conf)):.1f}% "
          f"(its own chance {100 * conf_chance:.1f}%, "
          f"subset {hs[0]}/{hs[1]})")

    # Our own team field, scored best-case.
    MIN_SUP = 25   # a track seen <1 s cannot fairly score a team label
    ours = {t: our_team[t] for t in by_track
            if t in our_team and len(by_track[t]) >= MIN_SUP}
    truth = {t: team[Counter(g for _, g in by_track[t]).most_common(1)[0][0]]
             for t in ours if by_track[t]}
    if truth:
        lab = sorted({ours[t] for t in truth})
        best = 0
        for flip in (0, 1):
            agree = sum(1 for t in truth
                        if lab.index(ours[t]) ^ flip == truth[t])
            best = max(best, agree)
        print(f"[weak x-check, see team_eval.py] our `team` vs colour: "
              f"{best}/{len(truth)} = {100 * best / len(truth):.1f}% "
              f"(tracks with >={MIN_SUP} matched frames)")
    return len(track_sw), len(cross), len(gt_sw)


def demo():
    # The box convention must be the harness's, not a re-derivation of it.
    src = open("model/validation/st2_mot.py").read()
    assert 'x = r["x"] - r["w"] / 2.0' in src and 'y = r["y"] - r["h"]' in src, \
        "st2_mot changed its box convention; to_tlwh must follow it"
    r = {"x": 100.0, "y": 200.0, "w": 20.0, "h": 40.0}
    assert to_tlwh(r) == (90.0, 160.0, 20.0, 40.0)
    assert abs(iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-9
    assert iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 10, 10)) - (50 / 150)) < 1e-9
    assert runs([(0, "a"), (1, "a"), (2, "b"), (3, "a")]) == [("a", "b"), ("b", "a")]
    assert runs([(0, "a")]) == []
    # Hungarian must not hand one GT box to two tracks.
    m = match({1: (0, 0, 10, 10), 2: (100, 0, 10, 10)},
              {7: (0, 0, 10, 10), 8: (1, 0, 10, 10)})
    assert sorted(m.values()) == [1], m
    print("ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="c03s")
    ap.add_argument("--seqs", nargs="*", default=list(SEQS))
    ap.add_argument("--iou", type=float, default=IOU_MIN)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    globals()["IOU_MIN"] = a.iou
    if a.demo:
        demo()
    else:
        tot = [analyse(s, a.variant) for s in a.seqs]
        sw, cr, gsw = (sum(x[i] for x in tot) for i in range(3))
        print(f"\nTOTAL: track-side {sw}, cross-team {cr} "
              f"({100 * cr / max(1, sw):.1f}%), GT-side {gsw}")
        print(f"CEILING if team assignment were perfect and hard-enforced: "
              f"{cr} of {sw + gsw} association errors = "
              f"{100 * cr / max(1, sw + gsw):.1f}%")
