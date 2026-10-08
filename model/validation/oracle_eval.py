"""ORACLE: is 33.2% contamination a DETECTION problem or an ASSOCIATION problem?

Why this exists
---------------
Every fix on the table costs differently depending on the answer, and until now
nothing in this repo could tell them apart. `contam_eval.py` says 33.2% of
detections sit under an id whose dominant true player is somebody else, but a
detection that was never made cannot be associated wrongly, and a detection
placed 3 m off a player is an association hazard the tracker did not create. So
the number confounds two failures with different fixes: a better detector, or a
better association.

This file separates them the only way that is honest - by replacing the
detector with a perfect one and changing NOTHING else.

    real  : the shipped detections from dets.cache.json
    oracle: one box per GT player per frame, placed where the calibration and
            the geometry say that player's box is

Both go through the SAME tracker (`tracking.get_tracker`), the same projection,
and the same scorer (`contam_eval.owners` / `contam_eval.contamination`,
imported, not reimplemented). The frames are the same frames. The only variable
is the boxes.

    contamination(oracle) ~ contamination(real)  -> ASSOCIATION is the problem.
                                                    The detector is not what is
                                                    costing us identity, and
                                                    every association idea on
                                                    the list is live.
    contamination(oracle) << contamination(real) -> DETECTION is the problem.
                                                    No solver rescues it; the
                                                    boxes have to get better
                                                    first.

WHAT AN ORACLE BOX IS, AND WHAT IT IS NOT
    It is the GT pitch position pushed back through the homography to the pixel
    the foot projects to, given a box height from `sceneprior.expected_height_px`
    at that ground point and the real median aspect (w/h = 0.466, measured off
    dets.cache.json rather than assumed).

    It is NOT the annotator's box - SoccerTrack 118575 publishes positions in
    metres and no image-space boxes at all (`data/footage/README.md` notes the
    bbox XML is missing). So this oracle inherits the calibration: a perfect
    detector as seen THROUGH our homography. That is the right oracle for this
    question, because the tracker only ever sees boxes through that same
    homography - but it means the oracle is not a bound on a better calibration.

    AND IT MUST BE SMOOTHED FIRST, which is not a nicety. SoccerTrack GT sits on
    a 1.05 x 0.68 m grid and HOLDS STILL for ~95% of consecutive frames, then
    jumps a whole cell (`dist_eval.py` verifies the grid in its own self-check).
    Fed straight in, that is a box that is motionless for twenty frames and then
    teleports ~18 px near-side - which destroys IoU on a 10x24 px box and is
    adversarial for a motion-model tracker in a way real detections never are.
    Measured: raw quantised GT boxes gave 215 ids over 200 frames against the
    real detector's 27, i.e. the "perfect" detector scored four times WORSE.
    That is annotation quantisation being read as player motion. A centred
    moving average over `--gt-smooth` frames removes it without inventing
    motion, and both variants are printed so the confound stays visible.

    It is also PERFECT RECALL by construction: 22 boxes every frame, no misses,
    no false positives, no occlusion. Real detection recall through the tracker
    is 0.628. So the oracle answers "if the detector were perfect", not "if the
    detector were 10% better".

USE --clip. THE 118575 PATH IS KEPT AS A DEMONSTRATION THAT IT CANNOT WORK.
    118575 GT is metres on a 1.05x0.68 m grid: **88.9% of consecutive frames are
    zero-displacement**, p50 step 0.0000 m. De-quantising it is a modelling
    choice that SETS THE ANSWER - measured over 6,000 frames, the oracle scores
    14.9% / 18.1% / 6.8% / 0.1% at smoothing windows 1 / 5 / 11 / 25. That is
    the harness's free parameter, not the detector, and no window is defensible.

    The `soccer_top` 30 s clips have what 118575 lacks: **real per-frame
    annotated boxes** (`bb_left`, `bb_top`, `bb_width`, `bb_height` in the raw
    SoccerTrack CSVs), 0.7% zero-step, p50 1.0 px. So --clip needs no
    de-quantisation, no synthetic height and no round-trip through the
    homography - the oracle boxes ARE the annotator's boxes.

    Its own limit is length. `second-match (not published)` shows identity metrics on a
    short slice reporting the opposite of the truth as a clean pass (0.0%
    contamination at 47 frames against a true 33.2%; 5.4% at 1,000 frames). At
    900 frames the reading is ASYMMETRIC and must be read that way: a HIGH
    number is real evidence that association fails even with perfect boxes; a
    LOW number is not evidence of anything, because real tracks score low there
    too.

    python model/validation/oracle_eval.py --selfcheck
    python model/validation/oracle_eval.py --clip soccer_top__test__videos__D_20220220_1_1680_1710
    python model/validation/oracle_eval.py --frames 15000        # the 118575 demonstration
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "model", "engine"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import calibrators  # noqa: E402
import sceneprior  # noqa: E402
from tracking import get_tracker  # noqa: E402

import contam_eval  # noqa: E402 - attribution and scoring are THEIRS
from contam_eval import contamination, owners  # noqa: E402

JOB = os.path.join(ROOT, "data/jobs/118575_full_1st")
GT = os.path.join(ROOT, "model/validation/soccertrack/118575_1st.tracks.json.gz")

# Measured off dets.cache.json, first 400 frames: median box 28.9 x 56.4 px.
# Guessing this would have made the oracle boxes a different SHAPE from the real
# ones, and C-BIoU associates on box overlap - so the shape is part of what is
# being held constant, not a cosmetic detail.
ASPECT_W_H = 0.466


def load_gt_full(path: str = GT) -> dict[int, dict[str, np.ndarray]]:
    """frame -> {gt player id: (x, y) metres}. Same rebasing as contam_eval."""
    rows = json.load(gzip.open(path))["detections"]
    base = min(r["f"] for r in rows)
    per: dict[int, dict[str, np.ndarray]] = collections.defaultdict(dict)
    for r in rows:
        per[r["f"] - base][str(r["id"])] = np.array([r["x"], r["y"]], float)
    return per


def to_px(cal, xy_m: np.ndarray) -> np.ndarray:
    """Pitch metres -> the pixels that project to them. Inverse of to_pitch.

    Asserted, not assumed: `to_pitch` is only H for the perspective and remap
    calibrators, and silently returning garbage here would move every number in
    this file. Same guard as dist_eval.to_px, and for the same reason.
    """
    H = np.asarray(cal.H, float)
    p = np.c_[xy_m, np.ones(len(xy_m))] @ np.linalg.inv(H).T
    px = p[:, :2] / p[:, 2:]
    err = float(np.abs(np.asarray(cal.to_pitch(px)) - xy_m).max())
    if err > 1e-6:
        raise SystemExit(f"{type(cal).__name__}.to_pitch is not H (round-trip {err:.3g} m); "
                         "this harness cannot push GT backwards through it")
    return px


def smooth_gt(gt: dict[int, dict[str, np.ndarray]], win: int) -> dict[int, dict[str, np.ndarray]]:
    """Centred moving average per player, in metres, over `win` frames.

    Removes the annotation grid, not the player. win=1 returns the input, which
    is how the raw-quantised control is produced.
    """
    if win <= 1:
        return gt
    frames = sorted(gt)
    per: dict[str, list] = collections.defaultdict(list)
    for f in frames:
        for gid, xy in gt[f].items():
            per[gid].append((f, xy))
    out: dict[int, dict[str, np.ndarray]] = collections.defaultdict(dict)
    k = np.ones(win) / win
    for gid, seq in per.items():
        fs = [f for f, _ in seq]
        arr = np.array([xy for _, xy in seq], float)
        if len(arr) < win:
            sm = arr
        else:
            # 'edge' padding, so the first and last half-windows are not dragged
            # toward zero - a player is not at the corner flag at kickoff.
            pad = win // 2
            sm = np.column_stack([
                np.convolve(np.pad(arr[:, c], pad, mode="edge"), k, mode="valid")[:len(arr)]
                for c in (0, 1)])
        for f, xy in zip(fs, sm):
            out[f][gid] = xy
    return out


def load_clip_boxes(clip: str) -> tuple[dict[int, dict[str, tuple]], str]:
    """frame -> {player key: (x1, y1, x2, y2)} from the raw SoccerTrack CSV.

    The CSV is a THREE-ROW header - TeamID / PlayerID / Attributes - over one
    column per (player, attribute). BALL columns are dropped: this file scores
    player identity, and ball.py is measured elsewhere.

    Rows are 1-BASED while video frames are 0-based (`ball_eval.py` paid for
    that lesson - an unrebased join shifts every position ~5 px), so the frame
    index is rebased here.
    """
    import csv as _csv
    path = os.path.join(ROOT, "data/footage/soccertrack",
                        clip.replace("__videos__", "__annotations__") + ".csv")
    if not os.path.exists(path):
        raise SystemExit(f"no annotation CSV for {clip}: {path}")
    with open(path, newline="") as fh:
        rows = list(_csv.reader(fh))
    teams, players, attrs = rows[0], rows[1], rows[2]
    cols: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for i in range(1, len(attrs)):
        if teams[i] == "BALL" or players[i] == "BALL":
            continue
        cols[f"{teams[i]}_{players[i]}"][attrs[i]] = i

    out: dict[int, dict[str, tuple]] = {}
    base = None
    for r in rows[4:]:
        if not r or not r[0].strip():
            continue
        fr = int(float(r[0]))
        base = fr if base is None else base
        per: dict[str, tuple] = {}
        for key, ci in cols.items():
            try:
                left = float(r[ci["bb_left"]])
                top = float(r[ci["bb_top"]])
                w = float(r[ci["bb_width"]])
                h = float(r[ci["bb_height"]])
            except (ValueError, KeyError, IndexError):
                continue
            if w <= 0 or h <= 0:
                continue
            per[key] = (left, top, left + w, top + h)
        out[fr - base] = per
    return out, path


def oracle_dets(g: dict[str, np.ndarray], cal, cam) -> list[dict]:
    """One perfect box per GT player, in the detector's own output contract."""
    if not g:
        return []
    xy_m = np.array(list(g.values()), float)
    feet = to_px(cal, xy_m)
    h = np.asarray(sceneprior.expected_height_px(cam, xy_m), float)
    w = h * ASPECT_W_H
    return [{"xyxy": (float(fx - wi / 2), float(fy - hi),
                      float(fx + wi / 2), float(fy)),
             "conf": 0.99, "cls": "person"}
            for (fx, fy), wi, hi in zip(feet, w, h)]


def replay(dets_by_frame, frames: int, cal, tracker: str = "best") -> list[dict]:
    """Run the shipped tracker and emit rows in the tracks.json contract.

    x, y are FEET IN IMAGE PIXELS, matching track.py - because that is what
    contam_eval.owners expects and it projects them itself.
    """
    tr = get_tracker(tracker)
    rows: list[dict] = []
    for f in range(frames):
        dets = dets_by_frame(f)
        if dets is None:
            continue
        for d in tr.update(dets, None):
            x1, _y1, x2, y2 = d["xyxy"]
            rows.append({"f": f, "id": int(d["id"]), "team": "home",
                         "jersey": None, "x": (x1 + x2) / 2.0, "y": y2})
    return rows


def _run(name: str, dets_by_frame, frames: int, cal, gt, stride: int) -> dict:
    rows = replay(dets_by_frame, frames, cal)
    own = owners(rows, cal, gt, stride)
    c = contamination(rows, own)
    ids = len({r["id"] for r in rows})
    seen = len(set(own.values()))
    # contamination() already returns a PERCENTAGE, not a fraction.
    print(f"  {name:24s} contaminated {c['contaminated']:6.1f}%   ids {ids:5d}   "
          f"impure {c['impure_ids']:5d}   ids/player {c['ids_per_gt_player']:6.1f}   "
          f"attributed {len(own):7d}   seen {seen}")
    return {**c, "ids": ids, "rows": len(rows), "attributed": len(own)}


def main_clip(clip: str, stride: int, real_tracks: str | None = None) -> None:
    """The honest oracle: the annotator's own boxes, through the shipped tracker."""
    cal = calibrators.load(os.path.join(ROOT, f"model/validation/calib_{clip}.json"))
    boxes, path = load_clip_boxes(clip)
    frames = max(boxes) + 1

    # GT in metres for attribution, from the SAME boxes - bottom-centre, which
    # is what track.py emits and what to_pitch expects.
    gt: dict[int, dict[str, np.ndarray]] = {}
    for fr, per in boxes.items():
        if not per:
            continue
        keys = list(per)
        feet = np.array([[(b[0] + b[2]) / 2, b[3]] for b in per.values()], float)
        met = np.asarray(cal.to_pitch(feet), float)
        gt[fr] = dict(zip(keys, met))

    print(f"{os.path.basename(path)}\n{frames} frames, {len(gt[0])} annotated players, "
          f"GT stride {stride}, gate {contam_eval.GATE_M} m, tracker 'best'\n")
    _run("oracle (annotated boxes)",
         lambda f: [{"xyxy": b, "conf": 0.99, "cls": "person"}
                    for b in boxes.get(f, {}).values()],
         frames, cal, gt, stride)
    if real_tracks:
        # The CONTROL, and the reason a 900-frame reading is usable at all: both
        # arms are handicapped by the same clip length, so the DELTA is valid
        # even though neither absolute is. Same tracker, same scorer, same
        # frames, same GT - only the boxes differ.
        rows = json.load(open(real_tracks))["detections"]
        own = owners(rows, cal, gt, stride)
        c = contamination(rows, own)
        ids = len({r["id"] for r in rows})
        print(f"  {'real (shipped detector)':24s} contaminated {c['contaminated']:6.1f}%   "
              f"ids {ids:5d}   impure {c['impure_ids']:5d}   "
              f"ids/player {c['ids_per_gt_player']:6.1f}   attributed {len(own):7d}   "
              f"seen {c['gt_players_seen']}")
        print(f"\n  recall: the real arm attributed {len(own)} detections where the "
              f"oracle attributed {len(gt) * len(gt[0]) // stride}\n  at the same "
              f"stride - so read contamination next to how little it saw.")

    print("\n  READ ASYMMETRICALLY. 900 frames is 30 s, and identity metrics on a\n"
          "  slice this short flatter (second-match (not published): 5.4% at 1,000 frames\n"
          "  against a true 33.2%). A high number here is evidence; a low one is not.")


def main(frames: int, stride: int, smooth: int) -> None:
    cal = calibrators.load(os.path.join(JOB, "calibration.json"))
    cam = sceneprior.camera_from_homography(cal.H, (11878, 870))
    gt = load_gt_full()
    cache = json.load(open(os.path.join(JOB, "dets.cache.json")))["frames"]

    print(f"{frames} frames, GT stride {stride}, gate {contam_eval.GATE_M} m, "
          f"tracker 'best'\n")
    sm = smooth_gt(gt, smooth)
    real = _run("real (dets.cache)", lambda f: cache.get(str(f)), frames, cal, gt, stride)
    _run("oracle, GT as-published", lambda f: oracle_dets(gt.get(f, {}), cal, cam),
               frames, cal, gt, stride)
    orac = _run(f"oracle, smoothed w={smooth}",
                lambda f: oracle_dets(sm.get(f, {}), cal, cam), frames, cal, gt, stride)

    print("\n  the as-published row is a CONTROL, not a result: it carries the "
          "1.05x0.68 m\n  annotation grid as synthetic teleports. Read the smoothed row.\n")
    d = real["contaminated"] - orac["contaminated"]
    print(f"  delta {d:+.1f} points of contamination attributable to DETECTION")
    print("  " + ("ASSOCIATION is the binding constraint - a perfect detector still leaves "
                  f"{orac['contaminated']:.1f}%" if orac["contaminated"] > 10.0 else
                  "DETECTION is the binding constraint - a perfect detector nearly clears "
                  "it, and no association solver substitutes for that"))


def _selfcheck() -> None:
    """The oracle must be perfect, and the scorer must be able to fail it."""
    cal = calibrators.load(os.path.join(JOB, "calibration.json"))
    cam = sceneprior.camera_from_homography(cal.H, (11878, 870))
    gt = load_gt_full()

    # 1. An oracle box must project back to the metre it came from. If this
    #    drifts, every "oracle" number below is measuring the round-trip.
    f0 = min(gt)
    g = gt[f0]
    dets = oracle_dets(g, cal, cam)
    assert len(dets) == len(g), (len(dets), len(g))
    feet = np.array([[(d["xyxy"][0] + d["xyxy"][2]) / 2, d["xyxy"][3]] for d in dets])
    back = np.asarray(cal.to_pitch(feet), float)
    err = float(np.abs(back - np.array(list(g.values()), float)).max())
    assert err < 1e-6, err

    # 2. Boxes must be the right size to be people, or C-BIoU is associating
    #    shapes the real detector never produces.
    hs = np.array([d["xyxy"][3] - d["xyxy"][1] for d in dets])
    assert 5 < hs.min() and hs.max() < 200, (hs.min(), hs.max())

    # 3. A PERFECT association over oracle boxes must score 0% contaminated.
    #    This is the case that would silently pass if owners() were broken, so
    #    it is asserted before any real number is believed.
    rows, gid_of = [], {}
    for f in range(f0, f0 + 200):
        for k, (gid, xy) in enumerate(gt[f].items()):
            px = to_px(cal, xy[None, :])[0]
            gid_of[gid] = k
            rows.append({"f": f, "id": k, "team": "home", "jersey": None,
                         "x": float(px[0]), "y": float(px[1])})
    own = owners(rows, cal, gt, 10)
    assert len(own) > 0, "nothing attributed - the gate or the frame base is wrong"
    perfect = contamination(rows, own)
    assert perfect["contaminated"] == 0.0, perfect

    # 4. ...and a fully welded association must NOT score 0, or the metric
    #    cannot see the failure it exists to detect.
    welded = [{**r, "id": 0} for r in rows]
    bad = contamination(welded, owners(welded, cal, gt, 10))
    assert bad["contaminated"] > 0.5, bad

    print("oracle_eval.py self-check OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--clip", default=None,
                    help="a soccer_top clip stem - the honest oracle, real annotated boxes")
    ap.add_argument("--real-tracks", default=None,
                    help="a tracks.json from the shipped detector on the SAME clip - "
                         "the control that makes a 900-frame reading usable")
    ap.add_argument("--frames", type=int, default=15000)
    ap.add_argument("--stride", type=int, default=contam_eval.STRIDE)
    ap.add_argument("--gt-smooth", type=int, default=11,
                    help="centred moving-average window, frames, over GT metres. "
                         "Removes the 1.05x0.68 m annotation grid. 1 disables")
    a = ap.parse_args()
    _selfcheck()
    if a.clip:
        main_clip(a.clip, a.stride, a.real_tracks)
    elif not a.selfcheck:
        main(a.frames, a.stride, a.gt_smooth)
