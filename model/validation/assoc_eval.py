"""Score ASSOCIATION against ground truth, without re-running the detector.

Why this exists
---------------
Association is the binding constraint on this pipeline: the first full-half run
consumed 3,973 track ids for 22 players (median track 1.5 s, longest 11.2 min of
47.9). Every idea for fixing that - a Kalman, a longer max_lost, appearance
embeddings, depth-aware gating - is a guess until it is measured, and measuring
it by re-running `track.py` costs ~17 hours a half because the DETECTOR is the
expensive part.

The detector is also the part that does not change. `dets.cache.json` already
holds every detection for all 71,850 frames, so association can be replayed on
its own in seconds. That is the whole point of this file: it turns an overnight
experiment into a loop you can actually iterate in.

What it measures
----------------
`model/validation/soccertrack/118575_1st.tracks.json.gz` is real ground truth for this
exact half: 22 ids, each present for all 72,101 frames, positions in METRES.
That makes honest metrics possible rather than proxies:

  IDF1          the standard identity F1 - how well a predicted id corresponds
                to one true player over time. The metric that matches the
                failure; MOTA would mostly reward detection, which already works.
  fragments     mean predicted tracks per GT player. 1.0 is perfect. The first
                run scored ~181, which is the finding stated as one number.
  id switches   how often a GT player's assigned prediction id changes.
  life          mean predicted track length, in seconds.
  recall        detection recall, reported BY DEPTH BAND, because the failure is
                not uniform: far-side players are 8-30 px and near-side 70-115.

Matching is done in METRES on the pitch, not in pixels, because that is the
space the two sources share and the space the product's claims are made in.

Usage
-----
    python model/validation/assoc_eval.py --frames 15000            # first 10 min
    python model/validation/assoc_eval.py --max-lost 75 --kalman    # try a change
    python model/validation/assoc_eval.py --selfcheck
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model", "engine"))

import calibrators  # noqa: E402
from tracking import BEST, get_tracker  # noqa: E402

JOB = "data/jobs/118575_full_1st"
GT = "model/validation/soccertrack/118575_1st.tracks.json.gz"

# How close a prediction must be to a GT player to count as that player. 2 m is
# deliberately generous: the calibration's own held-out error is around 1 m, so a
# tighter gate would score the CALIBRATION rather than the association, which is
# not what this file is for.
GATE_M = 2.0


def load_gt(path: str = GT) -> tuple[dict[int, dict[str, np.ndarray]], int]:
    """GT positions in metres, indexed by frame.

    The file numbers frames from 3,000,001; the offset is removed here so both
    sides speak the same 0-based frame index as the video.
    """
    d = json.load(gzip.open(path))
    rows = d["detections"]
    base = min(r["f"] for r in rows)
    per_frame: dict[int, dict[str, np.ndarray]] = collections.defaultdict(dict)
    for r in rows:
        per_frame[r["f"] - base][str(r["id"])] = np.array([r["x"], r["y"]])
    return per_frame, len(per_frame)


def run_tracker(cache: dict, cal, frames: int, tracker_name: str, **kw) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Replay association over cached detections. Returns frame -> [(id, xy_m)]."""
    # cal goes in for the "-pitch" trackers, which associate on the pitch plane
    # and cannot be built without one. Harmless for the pixel-space ones, which
    # never see it.
    if tracker_name.endswith("-pitch"):
        kw["cal"] = cal
    tr = get_tracker(tracker_name, **kw)
    out: dict[int, list[tuple[int, np.ndarray]]] = {}
    for f in range(frames):
        dets = cache["frames"].get(str(f))
        if dets is None:
            continue
        got = tr.update(dets, None)
        if not got:
            out[f] = []
            continue
        # Feet, matching track.py :: track — the homography maps the ground plane.
        px = np.array([[(d["xyxy"][0] + d["xyxy"][2]) / 2, d["xyxy"][3]] for d in got])
        xy = cal.to_pitch(px)
        out[f] = list(zip((d["id"] for d in got), xy))
    return out


def greedy_match(gt_pts: dict, cand_pts: dict, gate: float = GATE_M) -> dict:
    """{gt id: candidate key} - each GT takes its nearest UNUSED candidate inside `gate`.

    THE one per-frame position match in this harness. It was written out four
    times - here, twice in team_eval.py and once in jersey_eval.py - and four
    copies of a matcher is four chances for the population a number describes to
    differ from the population its neighbour describes.

    Greedy nearest-neighbour rather than a full Hungarian per frame. With 22 GT
    players spread over a 105x68 m pitch and a 2 m gate the two agree almost
    everywhere, and greedy keeps a 70,000-frame sweep interactive - which is the
    entire reason this file exists.

    GT-MAJOR, and the order is part of the answer: `gt_pts` is walked in its own
    iteration order and each GT claims a candidate before the next is looked at,
    so a different order can give a different pairing where two GT players
    contest one detection. That is the behaviour the four copies already had and
    it is preserved exactly; anything that needs order-independence wants the
    Hungarian, not this.
    """
    used, out = set(), {}
    for gid, g in gt_pts.items():
        best, bd = None, gate
        for k, p in cand_pts.items():
            if k in used:
                continue
            d = float(np.hypot(*(p - g)))
            if d < bd:
                best, bd = k, d
        if best is not None:
            used.add(best)
            out[gid] = best
    return out


def score(pred: dict[int, list[tuple[int, np.ndarray]]], gt: dict[int, dict[str, np.ndarray]],
          frames: int, fps: float = 25.0) -> dict:
    """IDF1, fragments per GT player, id switches, track life, recall.

    The per-frame match is `greedy_match` - see it for the gate and the order.
    """
    assigned: dict[str, list[int]] = collections.defaultdict(list)  # gt id -> pred id per frame
    tp = fn = fp = 0
    life = collections.Counter()
    recall_band = collections.Counter()
    total_band = collections.Counter()

    for f in range(frames):
        g = gt.get(f, {})
        p = pred.get(f, [])
        for pid, _ in p:
            life[pid] += 1
        m = greedy_match(g, {i: pxy for i, (_pid, pxy) in enumerate(p)})
        for gid, gxy in g.items():
            best = m.get(gid)
            band = min(9, int(gxy[1] // 6.8))  # 10 bands across the 68 m width
            total_band[band] += 1
            if best is None:
                fn += 1
                assigned[gid].append(-1)
            else:
                tp += 1
                recall_band[band] += 1
                assigned[gid].append(p[best][0])
        fp += len(p) - len(m)

    # IDF1 over the assignment: for each GT player the best-matching predicted
    # id, then the usual identity precision/recall against it.
    idtp = 0
    frags, switches = [], 0
    for gid, seq in assigned.items():
        real = [x for x in seq if x != -1]
        if not real:
            frags.append(0)
            continue
        c = collections.Counter(real)
        idtp += c.most_common(1)[0][1]
        frags.append(len(c))
        switches += sum(1 for a, b in zip(real, real[1:]) if a != b)

    n_gt = sum(len(gt.get(f, {})) for f in range(frames))
    n_pred = sum(len(pred.get(f, ())) for f in range(frames))
    idf1 = 2 * idtp / (n_gt + n_pred) if (n_gt + n_pred) else 0.0
    return {
        "frames": frames,
        "IDF1": round(idf1, 4),
        "fragments_per_player": round(float(np.mean(frags)), 1) if frags else 0.0,
        "id_switches": switches,
        "pred_tracks": len(life),
        "mean_track_life_s": round(float(np.mean(list(life.values()))) / fps, 2) if life else 0.0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        "recall_by_width_band": {
            b: round(recall_band[b] / total_band[b], 3) for b in sorted(total_band) if total_band[b]
        },
    }


def evaluate(frames: int = 15000, tracker_name: str = BEST, job: str = JOB, **kw) -> dict:
    cache = json.load(open(f"{job}/dets.cache.json"))
    cal = calibrators.load(f"{job}/calibration.json")
    gt, _ = load_gt()
    t0 = time.perf_counter()
    pred = run_tracker(cache, cal, frames, tracker_name, **kw)
    res = score(pred, gt, frames)
    res["tracker"] = tracker_name
    res["settings"] = kw
    res["seconds"] = round(time.perf_counter() - t0, 1)
    return res


def _selfcheck() -> None:
    """Scoring must be right before any number off it is believed.

    A synthetic pair: one GT player walking a straight line, and predictions
    that follow perfectly but are cut in two. Perfect detection, one break.
    """
    gt = {f: {"1": np.array([float(f) * 0.01, 34.0])} for f in range(100)}
    pred = {f: [(1 if f < 50 else 2, np.array([float(f) * 0.01, 34.0]))] for f in range(100)}
    r = score(pred, gt, 100)
    assert r["recall"] == 1.0, r
    assert r["precision"] == 1.0, r
    assert r["fragments_per_player"] == 2.0, r
    assert r["id_switches"] == 1, r
    # Half the frames carry the winning id, so IDF1 is 0.5 on a clean split.
    assert abs(r["IDF1"] - 0.5) < 1e-6, r

    # A prediction outside the gate is a miss AND a false positive, never a
    # silent pass — the gate is the one thing that must not be generous by bug.
    far = {f: [(1, np.array([float(f) * 0.01, 40.0]))] for f in range(100)}
    r2 = score(far, gt, 100)
    assert r2["recall"] == 0.0 and r2["precision"] == 0.0, r2
    print("assoc_eval.py ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=15000, help="frames to score (25 fps)")
    ap.add_argument("--tracker", default=BEST)
    ap.add_argument("--max-lost", type=int, default=None)
    ap.add_argument("--iou-thr", type=float, default=None)
    ap.add_argument("--expand", type=float, default=None)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
    else:
        kw = {k: v for k, v in
              (("max_lost", a.max_lost), ("iou_thr", a.iou_thr), ("expand", a.expand))
              if v is not None}
        print(json.dumps(evaluate(a.frames, a.tracker, **kw), indent=2))
