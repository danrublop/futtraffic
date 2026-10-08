"""The duel drop, measured on the SHIPPED reader rather than the test harness.

WHY THIS FILE EXISTS
    `st2_kit_at_duel.py` reads a crop with `st2_team.torso` + median Lab +
    nearest centre. **The product does not do that.** `model/engine/track.py ::
    team_by_shirt` takes per-pixel grey-world chromaticity plus mean L over the
    FULL box width, and `_kmeans` clusters it in the pooled within-cluster
    metric. Those are different code paths and the gap between them is not
    small: `@M` 78.3% for mean Lab against 93.8% for the shipped pair.

    So every duel number measured before this file described the harness, not
    the thing users run. This re-measures against the ship path.

`@M` RESULT, ALL TEN SEQUENCES, GT boxes (so still a ceiling)

                            harness readout      SHIPPED reader
        control (clear)          83.6%               **92.5%**   (2,400 crops)
        at a duel                68.6%               **69.8%**   (1,925 crops)
        drop                     -15.0 pts           **-22.6 pts**
        pair resolved            41.6%                 41.2%
        pair wrong                3.9%                **1.5%**
        pair abstain             54.5%                 57.3%

    Stable: the same measurement on three sequences gave 92.8 / 70.4 / -22.4 /
    39.9 / 0.2 / 60.0, so nothing here rests on which clips were picked.

    The 92.5% control lands on the 93.8% `team_eval.py` records independently,
    which is the check that this is finally the product being measured.

    Two things change in the reading. The shipped reader is MUCH better when it
    can see the player and **collapses harder at a duel** - a 22-point fall, not
    15 - so the case for deciding the label on clean frames and propagating it
    is stronger, not weaker. And it almost never guesses wrong (0.2%); it
    abstains instead (60.0%), which is the right failure and the one a
    fragment-level vote can actually fix.

A BOX-CONVENTION BUG NEARLY MADE THIS A FINDING
    `team_by_shirt` takes CORNERS (x1, y1, x2, y2); MOT GT is (x, y, w, h). The
    first run of this file passed the latter, producing 57 usable crops of an
    expected 600, a 63.6% control BELOW the weaker readout, and 100% abstention.
    Every one of those was a symptom of the conversion, not a property of the
    reader. `demo()` pins the convention so it cannot drift.
"""
import sys, random
from collections import Counter, defaultdict
import cv2, numpy as np
sys.path.insert(0, "model/validation"); sys.path.insert(0, "model/engine")
import st2_failtax as F, st2_team as T
from track import team_by_shirt
import os
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download
CLIPS = _ST2 + "/mot/clips"

def whiten_fit(X, rounds=3, k=2):
    """_kmeans' within-cluster metric, but returning the TRANSFORM so new
    points can be projected into the same space."""
    X = np.asarray(X, np.float32); M = np.eye(X.shape[1], dtype=np.float64)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    lab = cv2.kmeans(X, k, None, crit, 20, cv2.KMEANS_PP_CENTERS)[1].ravel()
    Z = X
    for _ in range(rounds):
        S = np.zeros((X.shape[1], X.shape[1])); n = 0
        for j in range(k):
            A = X[lab == j].astype(np.float64)
            if len(A) < 2: continue
            S += (A - A.mean(0)).T @ (A - A.mean(0)); n += len(A) - 1
        w, V = np.linalg.eigh(S / max(n, 1) + 1e-9 * np.eye(X.shape[1]))
        M = V / np.sqrt(np.maximum(w, 1e-12))
        Z = np.ascontiguousarray(X @ M, np.float32)
        lab = cv2.kmeans(Z, k, None, crit, 20, cv2.KMEANS_PP_CENTERS)[1].ravel()
    cen = np.array([Z[lab == j].mean(0) for j in range(k)])
    return M, cen, lab

def ref(seq, gt, n_frames=40):
    cap = cv2.VideoCapture(f"{CLIPS}/{seq}.mp4"); fr = sorted(gt)
    picks = fr[::max(1, len(fr)//n_frames)][:n_frames]
    samp = defaultdict(list)
    for f in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
        if not ok: continue
        for gid, b in gt[f].items():
            v = feat(img, b)
            if v is not None: samp[gid].append(v)
    cap.release()
    ids = sorted(samp); X = np.array([np.median(samp[i], axis=0) for i in ids])
    M, cen, lab = whiten_fit(X)
    return dict(zip(ids, lab)), M, cen

def feat(img, b):
    """box is GT (x, y, w, h); team_by_shirt wants CORNERS (x1, y1, x2, y2)."""
    x, y, w, h = b
    v = team_by_shirt(img, (int(x), int(y), int(x + w), int(y + h)))
    v = np.asarray(v, float)
    return None if v.size != 3 or not np.any(v) else v


def classify(img, box, M, cen):
    v = feat(img, box)
    if v is None: return None
    z = v @ M
    return int(np.argmin([np.linalg.norm(z - c) for c in cen]))

def demo():
    """Pin the box convention that produced a fake finding on the first run."""
    img = np.zeros((80, 60, 3), np.uint8)
    img[20:50, 10:40] = (40, 40, 200)          # a red-ish torso block
    corners = team_by_shirt(img, (10, 10, 40, 60))
    assert corners.shape == (3,) and np.any(corners), corners
    # the same box read as (x, y, w, h) samples somewhere else entirely
    wrong = team_by_shirt(img, (10, 10, 30, 50))
    assert not np.allclose(corners, wrong), "conventions must differ"
    x, y, w, h = 10.0, 10.0, 30.0, 50.0
    assert feat(img, (x, y, w, h)) is not None
    assert np.allclose(feat(img, (x, y, w, h)),
                       team_by_shirt(img, (10, 10, 40, 60)))
    print("ok  (team_by_shirt takes CORNERS; feat() converts from MOT x,y,w,h)")


if "--demo" in sys.argv:
    demo(); sys.exit(0)

tot = Counter()
SEQS = sys.argv[1:] or ["117092", "117093", "118575", "118576", "118577",
        "118578", "128057", "128058", "132831", "132877"]
for seq in SEQS:
    gt = T.load_gt(seq); lab, M, cen = ref(seq, gt)
    out = F.analyse(seq, "c03s", 0.5)
    ev = [e for e in (out["contam"] if isinstance(out,dict) else out[0])
          if e["gap"]==0 and e["pair_iou"]>0
          and lab.get(e["gid_from"]) is not None and lab.get(e["gid_to"]) is not None
          and e["gid_from"] in gt.get(e["f"],{}) and e["gid_to"] in gt[e["f"]]]
    cross = [e for e in ev if lab[e["gid_from"]] != lab[e["gid_to"]]]
    random.Random(2).shuffle(cross); cross = cross[:200]
    who = {g for e in cross for g in (e["gid_from"], e["gid_to"])}
    ctrl = [(f, g) for f in sorted(gt) for g in who & set(gt[f])
            if F.max_overlap(gt[f], g) == 0.0]
    random.Random(1).shuffle(ctrl); ctrl = ctrl[:240]
    by_f = defaultdict(list)
    for e in cross: by_f[e["f"]].append(("duel", e))
    for f, g in ctrl: by_f[f].append(("ctrl", g))
    cap = cv2.VideoCapture(f"{CLIPS}/{seq}.mp4")
    for f in sorted(by_f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
        if not ok: continue
        for kind, item in by_f[f]:
            if kind == "ctrl":
                p = classify(img, gt[f][item], M, cen)
                if p is None: continue
                tot["cn"] += 1; tot["cok"] += (p == lab[item])
            else:
                e = item; own = lab[e["gid_from"]]
                pf = classify(img, gt[f][e["gid_from"]], M, cen)
                pt = classify(img, gt[f][e["gid_to"]], M, cen)
                if pf is None or pt is None: continue
                tot["dn"] += 1; tot["dok_single"] += (pf == own)
                if pf == pt: tot["abstain"] += 1
                elif pf == own: tot["ok"] += 1
                else: tot["wrong"] += 1
    cap.release()
n, c = max(tot["dn"],1), max(tot["cn"],1)
print(f"\nSHIPPED READER (team_by_shirt + within-cluster metric), {len(SEQS)} sequences")
print(f"  single crop at duel   {100*tot['dok_single']/n:5.1f}%   ({tot['dn']} crops)")
print(f"  single crop control   {100*tot['cok']/c:5.1f}%   ({tot['cn']} crops)")
print(f"  delta                 {100*(tot['dok_single']/n - tot['cok']/c):+5.1f} points")
print(f"\n  pair on cross-team duels: resolved {100*tot['ok']/n:.1f}%  "
      f"wrong {100*tot['wrong']/n:.1f}%  abstain {100*tot['abstain']/n:.1f}%")
print(f"\n  harness readout (st2_kit_at_duel): 68.6% duel / 83.6% control, "
      f"41.6% resolved / 54.5% abstain")


