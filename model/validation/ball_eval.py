"""THE BALL, against ground truth, for the first time.

WHY THIS EXISTS
    Every number in `ball.py` is a shape measurement over a handful of frames.
    Its gates were fitted to box aspect ratios, its self-check asserts those
    gates, and nothing in the file has ever been compared to a known ball
    position. That is not a small gap, because the failure it hides is the one
    a proximity picker is built to produce: on the DFL clip the picker locks
    onto the CENTRE SPOT of the centre circle - white, round, ~8 px, on grass,
    and it never moves - and a picker scored on self-consistency finds that
    maximally consistent. It passes every gate in the file, by construction.

    The ground truth was already on disk. SoccerTrack ships the ball as a 22nd
    "player" in the same annotation CSVs the player harnesses read, and
    `teamtrack_tracks.load` skipped it with one `continue`. It is now selectable
    (`want_ball=True`) and this is what reads it.

WHAT THE GROUND TRUTH IS, AND THE ONE THING IT CANNOT TELL YOU
    7,649 ball positions over 7,650 annotated frames across the nine
    SoccerTrack clips - 99.99%. The ball is labelled on essentially EVERY
    frame, which is the answer to "how often is the ball annotated" and also
    the limitation: the annotators did not mark occlusion. In real football the
    ball spends much of the match at a player's feet, and SoccerTrack carries
    a position through those stretches anyway. So `silent` below - truth says a
    position, the picker says nothing - conflates "the picker missed a visible
    ball" with "the ball was under a player and nothing could have seen it".
    It is an upper bound on the picker's misses, not a count of them.

    The boxes are annotation boxes, not measurements. One clip (test 1680_1710)
    is a constant 20x20 on every frame; the rest vary 4-20 px. The ball's true
    size is ~10 px at 4K (0.22 m at ~36 px/m), which is what sizes the gate.

THE GATE, AND WHY IT IS NOT DOING THE WORK
    GATE_PX = 20 px at 3840x2160: two ball diameters, ~0.55 m on the pitch, and
    roughly the annotator's own box. The report prints the same scores at 10,
    20, 40 and 100 px. If a conclusion only holds at one gate it is a threshold
    artifact and the sweep will say so - the observed failures are 300-1500 px
    out, so no plausible gate rescues them.

THE THREE WAYS THIS METRIC COULD LIE, AND WHAT STOPS EACH
  1. A PICKER THAT ALWAYS ANSWERS scores well on "share correct" if the wrong
     answers are not counted. `correct` and `wrong` are both reported, always,
     over the same denominator, and they are separate columns.
  2. A METRIC OVER LABELLED FRAMES ONLY silently measures the easy subset.
     Every rate here prints its numerator and denominator, and n is stated
     once per table. Here truth is present on ~100% of frames so the two
     denominators coincide, but they are still printed apart.
  3. A CHECK THAT CANNOT FAIL. `centre` is a constant "always return the middle
     of the frame" picker, scored by the identical code on the identical
     frames. If it ever scores well the metric is broken, and `_selfcheck`
     asserts that it does not.

WHAT IT MEASURED, 2026-08-20, AND WHAT THE FIX MOVED. n = 1,800 frames, two
SoccerTrack top-down clips at 3840x2160, CoreML RF-DETR medium 576 tiled
1024/0.10, conf 0.05. Share of truth-present frames answered within 20 px:

        picker            clip A          clip B
        proximity picker    0.0%            0.0%    <- what shipped until now
        conf                4.8%            5.6%
        shipped (now)      15.4%           19.3%    scenery + pitch + size gates
        shipped_nocal       8.9%           13.8%    the same, no calibration
        oracle             20.2%           28.6%    the detector's ceiling
        centre              0.4%            0.0%    the control

    Wrong answers fell from 84.7% / 100.0% to 57.2% / 75.1%. That number matters
    more than the hit rate: the picker is now willing to say nothing, and a
    wrong ball position is worse than none because everything downstream
    believes it.

    THE ORIGINAL FINDING, kept because it is what the gates were built against.
    The proximity picker was right on ZERO of 1,800 frames. It was not merely
    inaccurate, it was stationary: median frame-to-frame movement of its output
    was 0.015 px and 0.250 px, against a ball that moves ~5 px a frame. On both
    clips it parked at y ~ 2100 - the bottom edge of the frame - while the ball
    never left y 364-1630. It had locked onto painted line markings, on footage
    with nothing to do with the DFL clip that failure was first seen on. It
    answered on 85% and 100% of frames, so any metric reporting only "share of
    frames with a ball" would have called that 85% and 100% coverage.

THE DETECTOR / PICKER SPLIT
    Candidates are cached to disk once, then every picker is a pure replay over
    that cache. `oracle` picks the cached candidate nearest to truth, so it is
    exactly "could ANY picker have been right on this frame". oracle - picker
    is the picker's fault; 1 - oracle is the detector's.

    python model/validation/ball_eval.py --selfcheck
    python model/validation/ball_eval.py --cache            # ~7 min, 900 4K frames
    python model/validation/ball_eval.py                    # score the cache
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "model", "engine"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ball as ballmod  # noqa: E402
from paths import FOOTAGE2, VALIDATION  # noqa: E402
from teamtrack_tracks import load as load_tracks  # noqa: E402

#: Default clip: 900 frames of 3840x2160 top-down, ball 8-13 px, ball labelled
#: on all 900. Its player annotations already back three other harnesses here.
CLIP = "soccer_top__train__videos__D_20220220_1_0000_0030"

#: Two ball diameters at 4K. See the header for why, and for the sweep that
#: shows the answer does not depend on it.
GATE_PX = 20.0
GATES = (10.0, 20.0, 40.0, 100.0)

#: The detector threshold. Deliberately far below the shipped 0.5: the point of
#: the split is to separate "never proposed" from "chose wrongly", and a high
#: threshold would move the true ball into the first bucket and flatter the
#: picker. On the frame this was probed on, the true ball came in at 0.204 and
#: the loudest decoy at 0.392, so 0.5 would have discarded both.
CONF = 0.05


# ---------------------------------------------------------------- ground truth

def load_ball(clip: str = CLIP) -> tuple[dict[int, np.ndarray], dict[int, float], int]:
    """({frame: (x, y) px}, {frame: box size px}, n_annotated_frames).

    Frames are rebased to 0 to match the video: SoccerTrack numbers its rows
    from 1 and the video's first frame is 0, so joining them raw would shift
    every position by one frame of ball travel (~5 px/frame here, a quarter of
    the gate). Checked in `_selfcheck`, not assumed.

    A frame missing from the returned dict is "the ball is not annotated here",
    which is a state and not a hole - see the header on what it does and does
    not mean.
    """
    csv = FOOTAGE2 / f"{clip.replace('videos', 'annotations')}.csv"
    tr = load_tracks(csv, want_ball=True)
    rows = tr.get("BALL_BALL", [])
    if not rows:
        raise SystemExit(f"no BALL column in {csv}")
    base = min(r["f"] for r in rows)
    xy = {r["f"] - base: np.array([r["x"], r["y"]]) for r in rows}
    wh = {r["f"] - base: max(r["w"], r["h"]) for r in rows}
    return xy, wh, len(rows)


# ---------------------------------------------------------------- the cache

def cache_path(clip: str) -> str:
    return os.path.join(VALIDATION, f"ball_cands_{clip}.json.gz")


def extents_path(clip: str) -> str:
    return os.path.join(VALIDATION, f"ball_extents_{clip}.json.gz")


def build_extents(clip: str = CLIP, suffix: str = "") -> str:
    """`ball.blob_extent` for every cached candidate, so the marking gate can
    be REPLAYED off the cache like every other picker here.

    The gate is the one part of `BallPicker` that reads the image rather than
    the boxes, which would otherwise force a video decode per scored
    configuration and break the detector/picker split this file is built on.
    It is a pure function of (frame, box), so caching it changes nothing about
    what is measured - the same threshold sees the same numbers.

    ponytail: a sidecar keyed on the box, not a sixth column in the candidate
    cache. The candidate caches on disk predate this and are expensive to
    rebuild; a sidecar leaves them valid.
    """
    import cv2  # noqa: PLC0415

    cache = load_cache(clip + suffix)
    video = str(FOOTAGE2 / f"{clip}.mp4")
    cap = cv2.VideoCapture(video)
    out: dict[str, list] = {}
    t0 = time.time()
    for f in range(max(cache["frames"]) + 1):
        ok, frame = cap.read()
        if not ok:
            break
        row = cache["frames"].get(f)
        if row is None:
            continue
        out[str(f)] = [ballmod.blob_extent(frame, tuple(r[:4])) for r in row]
        if f and f % 200 == 0:
            print(f"  extents {f}  {time.time() - t0:.0f}s", flush=True)
    cap.release()
    p = extents_path(clip + suffix)
    with gzip.open(p, "wt") as fh:
        json.dump({"clip": clip, "frames": out}, fh)
    print(f"{len(out)} frames of blob extents -> {p}")
    return p


def load_extents(clip: str) -> dict[int, list]:
    p = extents_path(clip)
    if not os.path.exists(p):
        return {}
    d = json.load(gzip.open(p))
    return {int(k): v for k, v in d["frames"].items()}


def _wsha(path: str | None) -> str | None:
    """sha256 of a weights file, or None. Cheap enough to do once per cache build."""
    if not path or not os.path.exists(path):
        return None
    import hashlib  # noqa: PLC0415
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_cache(clip: str = CLIP, n_frames: int | None = None, conf: float = CONF,
                detector: str = "coreml", weights: str | None = None,
                tile: int | None = None, suffix: str = "",
                variant: str | None = None, resolution: int | None = None) -> str:
    """Run the REAL detector over the clip and save every ball candidate.

    Cached, because the picker is the thing under test and it is free while the
    detector is 0.44 s a frame at 4K. Caching the candidates rather than the
    picker's output is what makes the detector/picker split possible at all:
    `oracle` re-picks from the same list and answers "could any picker have got
    this frame right".

    Frames are read SEQUENTIALLY. Seeking a 4K h264 stream per frame costs more
    than the detector does, and `cap.set` on this container lands on the wrong
    frame often enough to poison a 20 px gate.
    """
    import cv2  # noqa: PLC0415
    import detectors  # noqa: PLC0415

    video = str(FOOTAGE2 / f"{clip}.mp4")
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = total if n_frames is None else min(n_frames, total)
    # `tile` MUST match what a finetuned model was trained at. This one saw the
    # ball at NATIVE scale inside 640 px tiles; running it behind the default
    # 1024 px tiles resizes each tile to the model's 640 and shrinks the ball by
    # a third, which is a different object from the one it learnt. A ball
    # detector evaluated at the wrong tile size measures the tiling, not the
    # model.
    kw = {"conf": conf, "ball": True}
    if weights:
        kw["weights"] = weights
    # A finetuned checkpoint must be loaded into the architecture it was trained
    # in: RFDETRDetector defaults to "medium", and a nano checkpoint read as
    # medium returns nothing at all (ball-detector-trained-not-yet-scored (internal note),
    # the first of the two wrapper bugs).
    if variant:
        kw["variant"] = variant
    if tile is not None:
        kw["tile"] = tile
    # RESOLUTION, which no ball call site has ever passed. RFDETRNanoConfig defaults it to
    # 384 while run_d1/training_config.json records the checkpoint was TRAINED at 640, so
    # every ball number in this repo is a 640-trained model run at 384 input - the 640
    # crop resized down a further 1.67x, which is the opposite of what this file's own
    # "saw the ball at NATIVE scale inside 640 px tiles" comment claims. Exposed so the
    # size of that can be measured rather than argued.
    if resolution is not None:
        kw["resolution"] = resolution
    det = detectors.get_detector(detector, **kw)

    p = cache_path(clip + suffix)
    out: dict[str, list] = {}
    frame_h = frame_w = 0
    t0 = time.time()

    def flush() -> None:
        """Checkpoint. Long runs get killed here, and a cache of the first 400
        frames is a measurement; a cache of nothing is a wasted afternoon."""
        with gzip.open(p, "wt") as fh:
            # weights/tile/variant too: arms A and C of ball_cmp differ ONLY in tile,
            # so without them a cache is distinguishable only by its caller's suffix and
            # a reused suffix would compare a config against itself undetectably.
            json.dump({"clip": clip, "video": video, "conf": conf, "detector": detector,
                       # THE HASH, not only the path. A cache that records where the
                       # weights were cannot prove WHICH weights they were, so VOID (d)
                       # "the checkpoint's sha256 beside the number" was satisfied in form
                       # only: a second checkpoint written over the first leaves stale
                       # caches and a fresh-looking hash with no warning.
                       "weights": weights, "weights_sha256": _wsha(weights),
                       "tile": tile, "variant": variant,
                       "resolution": resolution,
                       "frame_h": frame_h, "frame_w": frame_w, "n_frames": len(out),
                       "frames": out}, fh)

    for f in range(want):
        ok, frame = cap.read()
        if not ok:
            print(f"video ended at frame {f} of {total}", flush=True)
            break
        frame_h, frame_w = frame.shape[:2]
        cands = [d for d in det.detect(frame) if d["cls"] in detectors.BALL_CLASSES]
        out[str(f)] = [[round(float(v), 2) for v in d["xyxy"]] + [round(float(d["conf"]), 4)]
                       for d in cands]
        if f % 50 == 0:
            el = time.time() - t0
            print(f"  frame {f}/{want}  {len(cands)} cands  "
                  f"{el:.0f}s elapsed, ~{el / max(f, 1) * want:.0f}s total", flush=True)
        if f and f % 100 == 0:
            flush()
    cap.release()
    flush()
    print(f"{len(out)} frames, {sum(len(v) for v in out.values())} ball candidates -> {p}")
    return p


def load_cache(clip: str = CLIP) -> dict:
    # `clip` arrives with the suffix already appended (see `report`); adding it
    # again here referenced a name this scope does not have, so every call
    # raised NameError and the report path could not run at all.
    p = cache_path(clip)
    if not os.path.exists(p):
        raise SystemExit(f"no candidate cache at {p}\n"
                         f"  build it: .venv/bin/python model/validation/ball_eval.py --cache")
    d = json.load(gzip.open(p))
    d["frames"] = {int(k): v for k, v in d["frames"].items()}
    return d


# ---------------------------------------------------------------- the metric

def score(gt: dict[int, np.ndarray], pred: dict[int, np.ndarray | None],
          frames: list[int], gate_px: float = GATE_PX) -> dict:
    """The whole metric, as one pure function so `_selfcheck` can break it.

    Every count is over the SAME frame list, and every rate carries the
    denominator it was taken over. `correct` and `wrong` are separate and both
    reported: a picker that answers on every frame drives `wrong` up, and any
    summary that quotes one without the other is the trap this file exists to
    avoid.

    `unverifiable` is a picker answer on a frame truth does not label. It is
    neither a hit nor a miss, and lumping it into either is how a metric starts
    lying about the frames it cannot see.
    """
    n = len(frames)
    present = correct = wrong = silent = unverifiable = 0
    errs: list[float] = []
    for f in frames:
        g = gt.get(f)
        p = pred.get(f)
        if g is None:
            unverifiable += p is not None
            continue
        present += 1
        if p is None:
            silent += 1
            continue
        d = float(np.hypot(*(np.asarray(p) - g)))
        errs.append(d)
        if d <= gate_px:
            correct += 1
        else:
            wrong += 1
    e = np.array(errs) if errs else np.array([np.nan])
    pct = (float("nan"),) * 3 if not errs else (
        float(np.median(e)), float(np.percentile(e, 90)), float(e.max()))
    return {
        "n": n, "gate_px": gate_px,
        "gt_present": present, "returns": correct + wrong + unverifiable,
        "correct": correct, "wrong": wrong, "silent": silent,
        "unverifiable": unverifiable,
        # Denominators, spelled out, so no rate here can be quoted without one.
        "correct_of_n": correct / max(n, 1),
        "correct_of_present": correct / max(present, 1),
        "wrong_of_n": wrong / max(n, 1),
        "wrong_of_present": wrong / max(present, 1),
        "silent_of_present": silent / max(present, 1),
        "err_p50": pct[0], "err_p90": pct[1], "err_max": pct[2],
    }


# ---------------------------------------------------------------- the pickers

def _dets(row: list) -> list[dict]:
    """One cached row -> the dict list `BallPicker.update` expects."""
    return [{"xyxy": tuple(r[:4]), "conf": r[4], "cls": "sports ball"} for r in row]


def _centre(box: list) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def _clip_cal(clip: str):
    """The clip's calibration, or None. Never fatal - a missing file only means
    the picker runs on its shape and scenery gates, which is a real
    configuration and is scored as `shipped_nocal`."""
    path = os.path.join(VALIDATION, f"calib_{clip}.json")
    if not os.path.exists(path):
        return None
    import calibrators  # noqa: PLC0415
    return calibrators.load(path)


#: Stands in for the frame `BallPicker.update` would be handed. The picker only
#: asks `frame is not None` before calling `blob_extent`, and `blob_extent` is
#: replaced by the cached lookup for the duration - see `predict`.
_FRAME_STANDIN = object()


def predict(cache: dict, method: str, gt: dict[int, np.ndarray] | None = None,
            extents: dict[int, list] | None = None,
            ) -> dict[int, np.ndarray | None]:
    """{frame: xy or None} for one picker, replayed over the cached candidates.

    `shipped` is `ball.BallPicker` as it stands, unmodified and unconfigured.
    `_mark` is the same picker with `marking_gate=True` (track.py's
    `--ball-markings`), replayed off the cached blob extents.
    `conf` is argmax-confidence every frame - the cold-start rule, applied
    always, which is the simplest thing that could work.
    `oracle` picks the plausible candidate nearest truth, so it is the ceiling
    any picker over this detector could reach, and the split between the
    detector's failures and the picker's.
    `centre` is the constant middle of the frame. It is here to fail.
    """
    H, W = cache["frame_h"], cache["frame_w"]
    frames = sorted(cache["frames"])
    out: dict[int, np.ndarray | None] = {}

    if method.startswith("shipped"):
        # WITH the clip's calibration where there is one: two of the picker's
        # three gates - on the pitch, and 0.22 m across - are questions only a
        # calibration can answer, and scoring the picker without one measures a
        # configuration the job path never runs. `shipped_nocal` is the other
        # half, for footage that has no calibration at all.
        mark = method.endswith("_mark")
        cal = None if "nocal" in method else _clip_cal(cache.get("clip", ""))
        if mark and not extents:
            raise SystemExit(f"{method} needs the blob-extent sidecar; build it with "
                             f"--extents")
        p = ballmod.BallPicker(cal=cal, marking_gate=mark)
        real_extent = ballmod.blob_extent
        try:
            for f in frames:
                row = cache["frames"][f]
                frame = None
                if mark:
                    lut = {tuple(r[:4]): e for r, e in zip(row, extents.get(f, []))}
                    ballmod.blob_extent = (
                        lambda _f, box, window=6.0, _l=lut: _l.get(tuple(box)))
                    frame = _FRAME_STANDIN
                b = p.update(_dets(row), H, f, frame)
                out[f] = np.array(b.xy) if b.seen else None
        finally:
            ballmod.blob_extent = real_extent
        return out

    for f in frames:
        row = cache["frames"][f]
        if method == "centre":
            out[f] = np.array([W / 2, H / 2])
            continue
        keep = [r for r in row if ballmod.plausible(tuple(r[:4]), H)]
        if not keep:
            out[f] = None
        elif method == "conf":
            out[f] = _centre(max(keep, key=lambda r: r[4]))
        elif method == "oracle":
            g = (gt or {}).get(f)
            out[f] = (_centre(min(keep, key=lambda r: float(np.hypot(*(_centre(r) - g)))))
                      if g is not None else _centre(max(keep, key=lambda r: r[4])))
        elif method == "any":
            # UNGATED, straight off the detector: does the model see the ball at
            # all, before ball.py's shape gate gets a vote. The difference
            # between this and `oracle` is what the gate costs.
            g = (gt or {}).get(f)
            out[f] = (min((_centre(r) for r in row),
                          key=lambda c: float(np.hypot(*(c - g))))
                      if row and g is not None else None)
        else:
            raise SystemExit(f"unknown picker {method!r}")
    return out


# ---------------------------------------------------------------- the report

def _row(name: str, s: dict) -> str:
    return (f"  {name:19s} {s['correct']:5d} {s['correct_of_present']:7.1%} "
            f"{s['wrong']:6d} {s['wrong_of_present']:7.1%} "
            f"{s['silent']:6d} {s['silent_of_present']:7.1%} "
            f"{s['unverifiable']:5d}  {s['err_p50']:7.1f} {s['err_p90']:8.1f}")


def report(clip: str = CLIP, gate_px: float = GATE_PX, suffix: str = "") -> dict:
    cache = load_cache(clip + suffix)
    gt, wh, n_ann = load_ball(clip)
    frames = sorted(cache["frames"])
    n = len(frames)
    gt_here = {f: gt[f] for f in frames if f in gt}

    sizes = np.array([wh[f] for f in frames if f in wh])
    ncand = np.array([len(cache["frames"][f]) for f in frames])

    print(f"\nCLIP {clip}")
    print(f"  {cache['frame_w']}x{cache['frame_h']}, detector {cache['detector']} "
          f"conf {cache['conf']}, n = {n} frames scored")
    print(f"  GROUND TRUTH: ball annotated on {len(gt_here)}/{n} scored frames "
          f"({len(gt_here) / n:.2%}); {n_ann} annotated in the clip's CSV.")
    print("    SoccerTrack does not mark occlusion, so this is 'a position was "
          "recorded',")
    print("    NOT 'the ball was visible'. Misses below are an upper bound.")
    print(f"  GT box size: median {np.median(sizes):.1f} px, "
          f"range {sizes.min():.1f}-{sizes.max():.1f}. Gate = {gate_px:.0f} px.")
    print(f"  DETECTOR: {int(ncand.sum())} ball candidates over {n} frames "
          f"(median {np.median(ncand):.0f}/frame, "
          f"{(ncand == 0).sum()} frames with none).")

    extents = load_extents(clip + suffix)
    methods = ["shipped", "shipped_nocal", "conf", "oracle", "any", "centre"]
    if extents:
        # --ball-markings, on the same cached candidates, so the only thing that
        # changed between the two rows is the gate.
        methods[1:1] = ["shipped_mark", "shipped_nocal_mark"]
    preds = {m: predict(cache, m, gt_here, extents) for m in methods}
    scores = {m: score(gt_here, p, frames, gate_px) for m, p in preds.items()}

    print(f"\n  denominator for the % columns: gt_present = {scores['shipped']['gt_present']} "
          f"frames (of n = {n})")
    print("  picker                right       wrong        silent   unver   err_p50  err_p90")
    for m in methods:
        print(_row(m, scores[m]))
    print("    shipped = ball.BallPicker with this clip's calibration (the job path).")
    if extents:
        print("    *_mark  = the same, marking_gate=True (track.py --ball-markings).")
    print("    shipped_nocal = the same picker with no calibration: scenery + shape only.")
    print("    conf    = argmax(conf) every frame.")
    print("    oracle  = the gated candidate NEAREST truth: the ceiling over this detector.")
    print("    any     = nearest candidate ignoring ball.py's shape gate: the raw detector.")
    print("    centre  = a constant (W/2, H/2). It is the control; if it scores, stop reading.")

    o, s = scores["oracle"], scores["shipped"]
    print(f"\n  SPLIT over gt_present = {s['gt_present']}:")
    print(f"    detector never proposed it   {s['gt_present'] - o['correct']:5d} "
          f"({1 - o['correct_of_present']:.1%})")
    print(f"    detector had it, picker lost {o['correct'] - s['correct']:5d} "
          f"({(o['correct'] - s['correct']) / max(s['gt_present'], 1):.1%})")
    print(f"    picker right                 {s['correct']:5d} ({s['correct_of_present']:.1%})")
    dropped = sum(not ballmod.plausible(tuple(r[:4]), cache["frame_h"])
                  for f in frames for r in cache["frames"][f])
    print(f"    (ball.py's shape gate discards {dropped} of {int(ncand.sum())} "
          f"candidates before the picker sees them; 'any' minus 'oracle' is what "
          f"that costs.)")

    print(f"\n  GATE SWEEP, correct/{s['gt_present']} gt-present frames "
          "(if a finding needs one gate it is an artifact):")
    print("    gate_px  " + "".join(f"{m:>10s}" for m in ("shipped", "conf", "oracle", "centre")))
    sweep = {}
    for g in GATES:
        sweep[g] = {m: score(gt_here, preds[m], frames, g) for m in
                    ("shipped", "conf", "oracle", "centre")}
        print(f"    {g:7.0f}  " + "".join(f"{sweep[g][m]['correct_of_present']:9.1%} "
                                          for m in ("shipped", "conf", "oracle", "centre")))
    print()
    return {"clip": clip, "n": n, "scores": scores,
            "no_cand": int((ncand == 0).sum()), "n_cand": int(ncand.sum()),
            "wh": cache["frame_w"] * cache["frame_h"]}


#: Every SoccerTrack clip with ball truth. Six 4K top-down, three 6500x1000
#: side panoramas - the two footage types ball.py's marking gate disagrees on.
ALL_CLIPS = [
    "soccer_top__train__videos__D_20220220_1_0000_0030",
    "soccer_top__train__videos__D_20220220_1_0030_0060",
    "soccer_top__test__videos__D_20220220_1_1680_1710",
    "soccer_top__test__videos__D_20220220_1_1710_1740",
    "soccer_top__test__videos__D_20220220_1_1740_1770",
    "soccer_top__test__videos__D_20220220_1_1770_1800",
    "soccer_side__train__videos__F_20200220_1_0000_0030",
    "soccer_side__train__videos__F_20200220_1_0030_0060",
    "soccer_side__train__videos__F_20200220_1_0060_0090",
]


def report_all(gate_px: float = GATE_PX, clips: list[str] | None = None) -> dict:
    """Every clip, then the pooled total. The pooled row is a SUM over frames,
    not a mean over clips: a mean of nine per-clip rates weights a 750-frame
    clip the same as a 900-frame one and is not a rate of anything.

    RETURNS the pooled table it prints - {"pooled": {picker: {correct, n, rate}},
    "gt_present":, "clips":, "no_cand":} - because api.py needs the number and
    the alternative is regex-scraping a printed percentage back into a float.
    """
    clips = clips or ALL_CLIPS
    rows = []
    for c in clips:
        if not os.path.exists(cache_path(c)):
            print(f"SKIP {c}: no candidate cache")
            continue
        rows.append(report(c, gate_px))

    cols = ["shipped", "shipped_mark", "shipped_nocal", "shipped_nocal_mark",
            "conf", "oracle", "any", "centre"]
    print("=" * 118)
    print(f"ALL CLIPS, share of gt-present frames within {gate_px:.0f} px")
    print(f"{'clip':46s} {'n':>5s} {'noca':>5s} " + "".join(f"{c[:9]:>10s}" for c in cols))
    tot = {c: [0, 0] for c in cols}
    tn = tno = 0
    for r in rows:
        s = r["scores"]
        gp = s["shipped"]["gt_present"]
        tn += gp
        tno += r["no_cand"]
        print(f"{r['clip']:46s} {gp:5d} {r['no_cand']:5d} " +
              "".join(f"{s[c]['correct_of_present']:9.1%} " if c in s else f"{'-':>9s} "
                      for c in cols))
        for c in cols:
            if c in s:
                tot[c][0] += s[c]["correct"]
                tot[c][1] += gp
    print("-" * 118)
    print(f"{'POOLED':46s} {tn:5d} {tno:5d} " +
          "".join(f"{tot[c][0] / max(tot[c][1], 1):9.1%} " for c in cols))
    for group, label in (("soccer_top__", "POOLED soccer_top (4K top-down)"),
                         ("soccer_side__", "POOLED soccer_side (6500x1000)")):
        g = {c: [0, 0] for c in cols}
        gn = gno = 0
        for r in rows:
            if not r["clip"].startswith(group):
                continue
            s = r["scores"]
            gp = s["shipped"]["gt_present"]
            gn += gp
            gno += r["no_cand"]
            for c in cols:
                if c in s:
                    g[c][0] += s[c]["correct"]
                    g[c][1] += gp
        if gn:
            print(f"{label:46s} {gn:5d} {gno:5d} " +
                  "".join(f"{g[c][0] / max(g[c][1], 1):9.1%} " for c in cols))
    print("  noca = frames on which the detector proposed NO ball candidate at all.")
    print("  A blank column is a clip with no blob-extent sidecar (--extents).")
    return {"gate_px": gate_px, "clips": [r["clip"] for r in rows],
            "gt_present": tn, "no_cand": tno,
            "pooled": {c: {"correct": tot[c][0], "n": tot[c][1],
                           "rate": tot[c][0] / tot[c][1] if tot[c][1] else 0.0}
                       for c in cols}}


def shipped_detector_settings() -> dict:
    """The detector config a job runs, read where the product reads it: tile, overlap and
    conf from shipped_config.product_config(), ball conf from detectors.BALL_CONF, the merge
    threshold from n70_class_merge, the input size from the CoreML model's file name. The
    detector itself is a runtime choice in jobs.py (coreml or rfdetr) and, as in
    shipped_config, is not compared."""
    import re  # noqa: PLC0415
    import detectors as D  # noqa: PLC0415
    import n70_class_merge as N  # noqa: PLC0415
    import shipped_config as SC  # noqa: PLC0415
    p = SC.product_config()
    res = re.search(r"-(\d+)\.mlpackage$", os.path.basename(str(D.COREML_MODEL).rstrip("/")))
    return {"tile": p["tile"], "overlap": p["overlap"], "conf": p["conf"],
            "ball_conf": D.BALL_CONF, "merge_thr": N.MERGE_THR,
            "resolution": int(res.group(1)) if res else f"unparsed:{D.COREML_MODEL}"}


def check_settings(d: dict, want: dict) -> None:
    """Refuse candidates made at a detector config other than the one that ships. The
    merge hash alone let product.ball.* read OK after a tile, conf or model change."""
    for c, cc in d["clips"].items():
        got = cc.get("settings")
        if not got:
            raise RuntimeError(f"{c}: no detector settings recorded - cannot tell what made it")
        bad = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
        if bad:
            raise RuntimeError(f"{c}: made at another detector config than ships, "
                               f"{{key: (fixture, shipped)}} = {bad}")


def score_m0(path: str, clips: list[str] | None = None, gate_px: float = GATE_PX) -> dict:
    """product.ball.*'s producer: the SHIPPED merge's candidates (n70_class_merge.export ->
    fixtures/ball/m0_cands.json.gz, committed), scored as n70_class_merge.clip_arms scores M0.

    -> {"clips": {clip: {"shipped/s1"|"oracle/s1"|"shipped/s2": {correct, wrong, silent, gt_present}}},
        "pooled": {same keys: {correct, n, rate}}}. Stride 2 feeds the picker even frames only, as
    jobs.py does. Raises (the registry records FAILED) when the file is absent, a clip is missing,
    the merge that made it is not the merge in detectors.py now, or its recorded detector
    settings are not the shipped ones (check_settings) - never a partial pool."""
    import n70_class_merge as N  # noqa: PLC0415
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} absent - build it: {N.EXPORT_CMD}")
    d = json.load(gzip.open(path))
    if d.get("merge_src_sha256") != N.merge_src_sha():
        raise RuntimeError(f"{path} was made by another detectors._merge - regenerate: {N.EXPORT_CMD}")
    check_settings(d, shipped_detector_settings())
    out: dict = {"clips": {}, "pooled": {}}
    for c in clips or ALL_CLIPS:
        if c not in d["clips"]:
            raise KeyError(f"{c} not in {path}")
        cc = d["clips"][c]
        frames = {int(f): v for f, v in cc["frames"].items()}
        gt = load_ball(c)[0]
        r = out["clips"][c] = {}
        for stride, methods in ((1, ("shipped", "oracle")), (2, ("shipped",))):
            cache = {**cc, "frames": {f: v for f, v in frames.items() if f % stride == 0}}
            fl = sorted(cache["frames"])
            g = {f: gt[f] for f in fl if f in gt}
            for m in methods:
                s = score(g, predict(cache, m, g), fl, gate_px)
                r[f"{m}/s{stride}"] = {k: s[k] for k in ("correct", "wrong", "silent", "gt_present")}
    for k in ("shipped/s1", "oracle/s1", "shipped/s2"):
        c = sum(r[k]["correct"] for r in out["clips"].values())
        n = sum(r[k]["gt_present"] for r in out["clips"].values())
        out["pooled"][k] = {"correct": c, "n": n, "rate": c / n}
    return out


# ---------------------------------------------------------------- self-check

def _synth(n: int = 200, decoy: tuple[float, float] = (1600.0, 600.0),
           ) -> tuple[dict, dict[int, np.ndarray]]:
    """A cache and its truth: one moving ball, one stationary louder decoy.

    This is the centre-spot failure in miniature. The decoy never moves, so it
    is maximally self-consistent to a proximity picker, and it is more
    confident than the ball on the cold-start frame - which is the only frame
    where `BallPicker` reads confidence at all.
    """
    frames, gt = {}, {}
    for f in range(n):
        x, y = 300.0 + 4.0 * f, 500.0 + 2.0 * f
        gt[f] = np.array([x, y])
        frames[str(f)] = [[x - 5, y - 5, x + 5, y + 5, 0.20],
                          [decoy[0] - 5, decoy[1] - 5, decoy[0] + 5, decoy[1] + 5, 0.40]]
    cache = {"clip": "synth", "detector": "synth", "conf": 0.05,
             "frame_h": 2160, "frame_w": 3840, "frames": {int(k): v for k, v in frames.items()}}
    return cache, gt


def _selfcheck() -> None:
    """Everything here must be able to FAIL. That is the point of the file."""
    frames_syn = list(range(200))
    cache, gt = _synth()

    # 1. THE METRIC MUST REWARD THE TRUTH. A picker handed the true position
    #    scores 100% and zero wrong; if this does not hold nothing below means
    #    anything.
    perfect = score(gt, dict(gt), frames_syn)
    assert perfect["correct"] == 200 and perfect["wrong"] == 0, perfect

    # 2. THE METRIC MUST PUNISH THE DECOY. Same frames, same code, the decoy
    #    chosen every time: the score has to collapse AND the wrong count has
    #    to rise, because a metric that only tracks 'correct' lets an
    #    always-answering picker hide.
    bad = score(gt, {f: np.array([1600.0, 600.0]) for f in frames_syn}, frames_syn)
    assert bad["correct"] == 0, bad
    assert bad["wrong"] == 200, bad
    assert bad["correct_of_present"] < 0.5 < perfect["correct_of_present"]

    # 3. THE CONTROL MUST FAIL. A constant frame centre is the classic metric
    #    that scores well by accident; if it ever does, the metric is broken.
    ctr = score(gt, predict(cache, "centre"), frames_syn)
    assert ctr["correct_of_present"] < 0.05, f"frame centre scored {ctr}"

    # 4. SILENCE IS NOT CORRECTNESS. A picker that answers nothing must score 0
    #    correct and 0 wrong - it must not be able to buy a good score by
    #    abstaining, and it must not be indistinguishable from the decoy.
    mute = score(gt, {f: None for f in frames_syn}, frames_syn)
    assert mute["correct"] == 0 and mute["wrong"] == 0 and mute["silent"] == 200, mute

    # 5. THE REAL PICKER MUST BEAT THE DECOY. This assertion used to say the
    #    OPPOSITE - it pinned the bug, because at the time the picker took the
    #    louder decoy on the cold-start frame and then followed its own buffer
    #    into it for ever. The decoy is stationary and the ball is not, which is
    #    the one difference ball.Scenery reads, so this is the synthetic version
    #    of the failure that scored 0 of 1,800 real frames.
    shipped = score(gt, predict(cache, "shipped"), frames_syn)
    assert shipped["correct_of_present"] > 0.8, \
        f"the stationary decoy captured the picker again: {shipped}"
    #    It cannot be perfect and should not pretend to be: the scenery memory
    #    needs a second of evidence before it will call anything scenery, so the
    #    first ~25 frames go to the louder decoy. Those are counted, not hidden.
    assert shipped["wrong"] >= 20, shipped
    #    ...and the detector proposed the true ball on every one of those
    #    frames, so any remaining loss is the PICKER's and the split must say so.
    orc = score(gt, predict(cache, "oracle", gt), frames_syn)
    assert orc["correct"] == 200, orc

    #    The no-calibration path is scored too, and must ALSO resist the decoy:
    #    scenery alone is what does this, and it is the only gate available on
    #    footage nobody has calibrated.
    nocal = score(gt, predict(cache, "shipped_nocal"), frames_syn)
    assert nocal["correct_of_present"] > 0.8, nocal

    # 5b. THE MARKING GATE, REPLAYED OFF THE SIDECAR, must behave as it does on
    #     a real frame: a synthetic pitch with one painted stripe and one ball,
    #     the stripe louder. Off, the picker takes the stripe; on, it must not.
    #     If the extent lookup is not wired to the picker this passes silently
    #     with both rows identical, so both directions are asserted.
    import cv2  # noqa: PLC0415
    img = np.full((400, 400, 3), (60, 160, 60), np.uint8)
    cv2.rectangle(img, (200, 0), (206, 399), (245, 245, 245), -1)
    cv2.circle(img, (100, 100), 5, (245, 245, 245), -1)
    line_box, ball_box = [197.0, 197.0, 209.0, 209.0], [94.0, 94.0, 106.0, 106.0]
    mc = {"clip": "synth", "detector": "synth", "conf": 0.05, "frame_h": 400,
          "frame_w": 400, "frames": {0: [line_box + [0.9], ball_box + [0.2]]}}
    mex = {0: [ballmod.blob_extent(img, tuple(line_box)),
               ballmod.blob_extent(img, tuple(ball_box))]}
    off = predict(mc, "shipped_nocal")[0]
    on = predict(mc, "shipped_nocal_mark", extents=mex)[0]
    assert abs(off[0] - 203) < 3, f"the marking gate was already on: {off}"
    assert on is not None and abs(on[0] - 100) < 3, f"cached extents did not gate: {on}"

    # 6. UNVERIFIABLE IS NEITHER. Drop truth on half the frames: the answers on
    #    unlabelled frames must not land in correct or wrong, and the
    #    denominator must shrink to match.
    half = {f: v for f, v in gt.items() if f % 2 == 0}
    s = score(half, dict(gt), frames_syn)
    assert s["gt_present"] == 100 and s["correct"] == 100 and s["unverifiable"] == 100, s
    assert s["correct_of_present"] == 1.0 and s["correct_of_n"] == 0.5, s

    # 7. THE GT JOIN. SoccerTrack rows start at frame 1 and the video at 0, so
    #    an unrebased join is off by one frame of ball travel. Frame 0 must
    #    exist and the ball must move like a ball, not like a player.
    csv = FOOTAGE2 / f"{CLIP.replace('videos', 'annotations')}.csv"
    if csv.exists():
        xy, wh, n_ann = load_ball(CLIP)
        assert 0 in xy and min(xy) == 0, f"GT not rebased to 0: {min(xy)}"
        assert n_ann == 900, n_ann
        pts = np.array([xy[f] for f in sorted(xy)])
        assert pts[:, 0].max() < 3840 and pts[:, 1].max() < 2160, pts.max(0)
        step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        assert 1.0 < np.median(step) < 60.0, f"{np.median(step)} px/frame is not a ball"
        assert 4.0 <= np.median(list(wh.values())) <= 25.0, np.median(list(wh.values()))
    else:
        print("  (SKIP: SoccerTrack annotations not on disk; synthetic checks ran)")

    # The candidates must have been made at the detector config that ships. Planted:
    # a tile change and a fixture with no settings both refuse; the real one passes.
    want = {"tile": 1024, "conf": 0.5}
    for bad in ({"tile": 640, "conf": 0.5}, None):
        try:
            check_settings({"clips": {"c": {"settings": bad}}}, want)
            raise AssertionError(f"candidates at settings {bad} scored as the shipped detector's")
        except RuntimeError:
            pass
    check_settings({"clips": {"c": {"settings": {"tile": 1024, "conf": 0.5, "x": 1}}}}, want)
    m0 = os.path.join(ROOT, "model/validation/fixtures/ball/m0_cands.json.gz")
    if os.path.exists(m0):
        check_settings(json.load(gzip.open(m0)), shipped_detector_settings())
    print("ball_eval.py ok")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=CLIP)
    ap.add_argument("--cache", action="store_true", help="run the detector and cache candidates")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--detector", default="coreml")
    ap.add_argument("--weights", default=None, help="finetuned checkpoint for --cache")
    ap.add_argument("--variant", default=None,
                    help="architecture of --weights (nano/small/medium); wrong = no boxes")
    ap.add_argument("--tile", type=int, default=None,
                    help="tile size; must match what --weights was trained at")
    ap.add_argument("--suffix", default="", help="write/read a separate cache under this suffix")
    ap.add_argument("--gate", type=float, default=GATE_PX)
    ap.add_argument("--extents", action="store_true",
                    help="decode the video and cache blob_extent per candidate, "
                         "which is what --ball-markings needs to be replayable")
    ap.add_argument("--all", action="store_true", help="every clip in ALL_CLIPS, then pooled")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
        return 0
    if a.all:
        if a.cache or a.extents:
            for c in ALL_CLIPS:
                if a.cache and not os.path.exists(cache_path(c)):
                    build_cache(c, a.frames, a.conf, a.detector, a.weights, a.tile,
                                variant=a.variant)
                if a.extents and not os.path.exists(extents_path(c)):
                    build_extents(c)
        report_all(a.gate)
        return 0
    if a.cache:
        build_cache(a.clip, a.frames, a.conf, a.detector, a.weights, a.tile, a.suffix,
                    variant=a.variant)
    if a.extents:
        build_extents(a.clip, a.suffix)
    report(a.clip, a.gate, a.suffix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
