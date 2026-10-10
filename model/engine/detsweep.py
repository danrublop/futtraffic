"""Measure a detector configuration against ground truth, BY PITCH-WIDTH BAND.

    python model/engine/detsweep.py --selfcheck                    # free, no GPU
    python model/engine/detsweep.py -n 96 --every 700 --only "base r768"
    # ...and the 30 s loop, which is the fast one (900-frame 4K clip):
    python model/engine/detsweep.py --clip soccer_top__test__videos__D_20220220_1_1680_1710 \
        --first 0 --every 100 -n 8 --only "base r576"

`--clip` sets --gt, --video and --calib together because those three must agree:
a GT file from one clip against the video of another scores zero and reads as a
detector regression. It also switches the GT to PIXEL space - the 30 s clips
carry x,y in image pixels and 118575 carries metres, and the wrong answer there
reports recall 0 rather than raising.

Two things are different on those clips and both are in your favour: the
calibration is rms 0.029 m against the panorama's 2.845, so the 2 m gate is a
DETECTOR gate there rather than the calibration gate it is on 118575; and a run
is 30 s of footage instead of 48 minutes. What you lose is the pixel column -
the camera is near-nadir, the single-view recovery refuses (see sceneprior.
MAX_CAMERA_H_M), and every px-space column and the "+prior" rows go nan or are
skipped rather than zero, because a zero there reads as "found nothing".

The 460 DFL clips cannot be used here at all: they carry no ground truth, only a
Name,Folder Path manifest. They are for throughput and eyeballing.

Why this file exists
--------------------
Every detector idea on this footage looks like an improvement if you score it on
what the detector already found - det/frame, mean box size, confidence, tiles
hit. Five such metrics were believed in one session and all five were measuring
nothing. The only honest score is against
`model/validation/soccertrack/118575_1st.tracks.json.gz`: 22 real ids, every frame,
positions in METRES.

It also scores in two spaces on purpose, because the far touchline fails in both
and they are different failures:

  metres  - a detection counts if its FOOT projects within 2 m of a GT player.
            This is the product metric; the tracker and every stat live here.
  pixels  - a detection counts if its foot lands within 0.6 x the geometrically
            expected player height of where GT projects INTO THE IMAGE.
            This is the detector's own metric: was there a box on that player?

The gap between the two is calibration, not detection, and on 118575 it is
enormous at the far touchline: one image ROW is 0.57 m there (12 rows per 6.8 m
band), so a 4 px error in the box bottom - one shin - is a 2 m miss. Measured on
3,000 cached frames of the shipping config:

    band  0(far)     1       2       3       4       5       6       7    8   9(near)
    m@2m   0.179   0.320  0.464  0.772  0.665  0.856  0.949  0.990  0.993  0.861
    px     0.458   0.754  0.891  0.986  0.975  0.965  0.973  0.979  0.931  0.839

So bands 1-3 are NOT a detection failure - the boxes are there and the metres
are wrong. Band 0 is a real one: 46% of far-touchline players have no box at all.
A detector sweep can only move the px column; the m column is calibrators.py's
problem, and any far-side "detection fix" measured only in metres is measuring
the calibration.

Usage
-----
    .venv/bin/python model/engine/detsweep.py --selfcheck        # scoring only, no GPU
    .venv/bin/python model/engine/detsweep.py -n 96 --every 700 --only "base r768"

ONE PROCESS PER CONFIGURATION - the --only loop below, not a single run. This
machine has 16 GB and MPS does not hand memory back, so a second RF-DETR in the
same process OOMs at 19 GiB and drops to fp32, which quietly makes every
seconds-per-frame after the first one a lie:

    for c in "base r768 t1024" "base+far tgt128"; do
        .venv/bin/python model/engine/detsweep.py -n 40 --only "$c"; done

Boxes are cached in /tmp/detsweep_dets.json, so re-scoring at another threshold,
another gate, or with the prior on and off costs nothing. The key carries the
DETECTOR's confidence as well as the clip and the frames, because "detect at
0.15 and threshold after the tile merge" and "detect at 0.5" are two different
runs that used to collide on one key - the same trap as `--dets-cache`
(model/engine/detectors.py, BALL_CONF, "ONE TRAP").

THE BAND-0 RE-MEASUREMENT (2026-09-12) - protocol, written before the run
-------------------------------------------------------------------------
WHY. `0a5a7cd` retired the scene prior's height gate; `track.make_prior` now
applies the pitch polygon alone. `model/engine/detectors.py`'s conf sweep was taken
`prior=True` under BOTH gates, and `model-map (not published)` quotes its band-0 cell
as "the shipped detector config" while arguing the far touchline is the
bottleneck. That cell no longer describes anything the product runs.

POPULATION  118575 first half, 40 frames from 3000 every 1500 (60 s apart, so
            no two samples hold the same play), every GT player in each = 880
            GT player-frames. NOT held out: 118575 is the only panorama in this
            repo with per-frame GT and it is the footage the product runs, so
            every detection number here is on it. n=1 - one prior, no arms.
HELD FIXED  coreml r576 t1024 through detectors.get_detector('coreml') - the
            shipped detector - the job's 65-point remap calibration, the
            11878x870 band canvas, GT in metres, GATE_M 2.0, GATE_H 0.6, and
            `prior=True` routed through track.make_prior, i.e. whatever the
            product's default gates are on the day it runs.
CHANGED     the prior, and nothing else: ("pitch", "height") -> ("pitch",).
METRIC      band-0 recall in PIXELS - a box within 0.6 expected player heights
            of where GT projects into the image. Bands 1 and 2, precision_px,
            band-0 metres and overall metres go beside it, because one band is
            not a result on its own and the px/m gap is calibration.
FLOOR       the null control the old table records at 0.069 - the same boxes
            against a frame 60 s away. A band-0 figure at or under that is box
            density flattering itself, not detection.
DROP SHARE  the prior IS a filter, so every row states det/frame with it and
            without it. The height gate's own drop share is the difference
            between the two tables.
PASS RULE   band-0 px recall at conf 0.50 rises against the old table's 0.328.
FAILING     it comes back at or below 0.328. Then the height gate was not what
            cost band 0, the sentence in model-map (not published) stands on its
            original number, and all this run changes is the label on the
            table. Publish it either way, next to the old table, not over it.

TWO PASSES, because the coreml row's DETECTOR conf is pinned at 0.5 (configs()
below) and the old table predates that pin (it landed in 166c1f5):

  S (shipped)  det_conf 0.5, scored at conf 0.50. The operating point
               app/server/jobs.py runs. Its recall_m / recall_px must reproduce
               api://product.detect.recall_m and .recall_px exactly; that is
               the consistency check, and if it does not, nothing else here is
               believed.
  L (ladder)   det_conf 0.15, scored at 0.50 / 0.30 / 0.25 / 0.20 - the OLD
               table's protocol exactly, so its cells are comparable to the old
               cells one for one. Its 0.50 row against S is the first
               measurement of what detecting low and thresholding after the
               tile merge actually costs; configs() asserts it is nonzero and
               nothing had checked.

THE TILE-OVERLAP DECISION (2026-09-12) - protocol, written before the run
-------------------------------------------------------------------------
WHY. `7c2cd35` took OVERLAP 0.25 -> 0.10 at 00:28 on 2026-08-19, nine minutes
after `ce869b8` published the conf table at 00:19, and chose it on a sweep at
CONF 0.2 - where band 1 is saturated (0.900 at 0.25 against 0.929 at 0.10) -
while the product ships at conf 0.5. `298e62b` re-measured at 0.5 on 40 frames
and found band 1 at 0.729 (overlap 0.25) against 0.529 (overlap 0.10): fourteen
GT player-frames of seventy, far outside the +-6 points at one sigma that
`7c2cd35` itself records for a band that size. Either the product has been
shipping a mid-pitch recall regression since 2026-08-19, or 40 frames saw a
ghost - which is the same mistake `7c2cd35` documents itself making in the
other direction. THIS RUN IS THE REPLICATION, and it is confirmatory: band 1 is
the primary because the 40-frame run named band 1, not because it is the band
that moved when the table was read.

POPULATION  118575 first half, 96 frames from 3000 every 700 (28 s apart) -
            2,112 GT player-frames, and 96 frames is the sample size `7c2cd35`
            itself used to overturn its own 40-frame reading. NOT held out:
            118575 is the only panorama here with per-frame GT and it is the
            footage the product runs. n=2 configurations tried (two overlaps),
            both published, neither selected on a sweep.
HELD FIXED  coreml r576 t1024 through detectors.get_detector('coreml'), the
            DETECTOR asked for conf 0.5 (det_conf=0.5 - what app/server/jobs.py
            runs; the ladder's 0.15-then-threshold is a different operating
            point and is not this question), the job's 65-point remap
            calibration, the 11878x870 band canvas, GATE_M 2.0, GATE_H 0.6,
            prior=True through track.make_prior (pitch polygon alone, as
            shipped), and the same 96 frames for both arms - the two arms are
            PAIRED, frame for frame.
CHANGED     OVERLAP, and nothing else: 0.10 (shipped) against 0.25 (pre-
            7c2cd35). It enters the box-cache key, so an arm cannot replay the
            other arm's boxes - the same trap as det_conf above and as
            --dets-cache (model/engine/detectors.py, BALL_CONF, "ONE TRAP").
METRIC      band-1 recall in PIXELS at conf 0.50, paired over the 96 frames.
            Reported beside it, every one of them required: bands 0/1/2 and all
            ten, overall image recall, image precision (precision_px, on-pitch
            - recall bought in precision is not free, and more marginal boxes
            into an online tracker is how two players weld into one id), metre
            recall, per-band n, the drop share (det/frame with the prior and
            without, both arms), and COMPUTE - tiles per frame and the
            detector's own seconds per frame, because this is the shipped path
            and 13 -> 16 tiles is a product cost on a 16 GiB machine.
FLOOR       the null control: the same boxes against a frame 28 s away. A band
            figure at or under it is box density flattering itself.
PASS RULE, and it decides in BOTH directions:
  REVERT to 0.25 iff all three hold at conf 0.50:
    (a) the paired 95% bootstrap CI on band-1 image recall (0.25 minus 0.10;
        percentile, 2,000 resamples of the 96 frames, seed 0) lies strictly
        above zero;
    (b) the point estimate is >= 5 points on band 1 OR >= 1 point on overall
        image recall - the size that pays for +23% tiles per frame and the
        1.27x detector wall clock `7c2cd35` measured (0.372 -> 0.473 s/frame);
    (c) image precision at 0.25 is not more than 1 point below 0.10's.
  KEEP 0.10 if the CI contains zero, or the point estimate favours 0.10. The
    40-frame reading was then a ghost and `298e62b`'s attribution stands as an
    attribution without being a regression.
  DECISION NEEDED - change nothing, publish both numbers - if (a) holds and
    (b) or (c) fails: a real but small gain paid for in throughput or in
    precision is a product trade, not a measurement, and not this run's to
    make.
  UNDERPOWERED is a sub-case of KEEP and is said OUT LOUD: CI contains zero
    AND its half-width >= 5 points means this sample cannot see the effect the
    40-frame run reported, and the report states the frames a two-sided test
    at 80% power would need rather than pretending to a null.
FAILING     any outcome but the first leaves model/engine/detectors.py's OVERLAP at
            0.10. `overlap_decision()` below applies the rule in code and
            returns the verdict, so it is not re-read off a table afterwards.
APPLIED     8472cef: the rule fired REVERT and OVERLAP is 0.25. The shipped
            arm is now read from detectors.OVERLAP - it was a literal
            `arms=(0.10, 0.25)` default naming 0.10 as shipped, false from the
            commit that added it - so `--overlap-decision` run bare scores 0.10
            as the CANDIDATE against the shipped 0.25. This protocol's own run,
            taken while 0.10 shipped, is replayed with ship=0.10,
            candidate=0.25: tile-overlap (not published) section 3.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import calibrators  # noqa: E402
import sceneprior  # noqa: E402
from detectors import RFDETRDetector, TiledDetector, geometry_tiles  # noqa: E402

JOB = "data/jobs/118575_full_1st"
GT = "model/validation/soccertrack/118575_1st.tracks.json.gz"
VIDEO = "data/footage/118575_panorama_1st_half.mp4"
CALIB = None        # None = f"{JOB}/calibration.json"; --clip sets it explicitly

# The SoccerTrack 30 s clips, which are the ONLY short footage in this repo that
# can be scored at all - the 460 DFL clips carry no ground truth of any kind,
# just a Name,Folder Path manifest. Each is 900 frames of 3840x2160 with
# per-detection GT and a perspective calibration fitted at rms 0.029 m, against
# 118575's 2.845 m. A far better test bed than the panorama, and 30 s long.
CLIPS = "soccer_top__{split}__videos__D_20220220_1_{span}"

GATE_M = 2.0        # same gate as model/validation/assoc_eval.py, deliberately
GATE_H = 0.6        # px gate, as a fraction of expected player height
BAND_M = 6.8        # 68 m of pitch width in ten bands


def load_gt(path: str = GT, cal=None) -> dict[int, np.ndarray]:
    """frame -> (N,2) GT positions in metres. Frame numbers are offset by
    3,000,001 in the file; the minimum is subtracted so both sides speak the
    same 0-based index as the video.

    `cal` is for the SoccerTrack 30 s clips, whose `.tracks.json.gz` carries
    x,y in IMAGE PIXELS rather than the metres 118575 carries - the converter
    writes box bottom-centre, and the metres live in a sibling calibration
    file. Passing the clip's calibration projects them; passing None keeps the
    118575 contract. Getting this backwards does not fail, it reports a recall
    of zero, so the caller states the space rather than the file being sniffed.
    """
    rows = json.load(gzip.open(path))["detections"]
    base = min(r["f"] for r in rows)
    per: dict[int, list] = collections.defaultdict(list)
    for r in rows:
        per[r["f"] - base].append((r["x"], r["y"]))
    out = {f: np.array(v, float) for f, v in per.items()}
    if cal is not None:
        out = {f: np.asarray(cal.to_pitch(v), float) for f, v in out.items()}
    return out


def sample_frames(frames: list[int], cal):
    """Yields undistorted (11878x870) band frames, one at a time.

    A GENERATOR, not a list, and that is not style: this machine has 16 GB and
    one band frame is 31 MB, so holding 40 of them costs 1.2 GB next to a model
    that MPS already reports at 19 GiB. Holding them measured 7% CPU and no
    progress - the run was swapping, not detecting. Seeking costs ~30 ms, which
    is nothing against 1.4 s of detection.
    """
    cap = cv2.VideoCapture(VIDEO)
    try:
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, raw = cap.read()
            if not ok:
                raise SystemExit(f"could not read frame {f} of {VIDEO}")
            # Only a RemapCalibrator has tables to apply. The SoccerTrack 30 s
            # clips are a plain perspective fit on the raw frame, so flattening
            # is not merely unnecessary there, it would raise.
            yield cal.undistort_frame(raw) if getattr(cal, "_mapx", None) is not None else raw
    finally:
        cap.release()


def score(dets: list[list[dict]], gts: list[np.ndarray], cal, cam,
          conf: float = 0.5, prior: bool = False,
          image_wh: tuple | None = None) -> dict:
    """Recall in both spaces, per band, plus precision. Matching is exclusive
    greedy nearest-first: one detection can satisfy exactly one GT player, so a
    single box in a crowd cannot cover three of them."""
    hit_m = collections.Counter()
    hit_px = collections.Counter()
    # Metre error of boxes that ARE on the player. Recall in pixels can be
    # saturated while the product metric is not, because the far side converts
    # 0.57 m per image row - so a config can only help there by placing the FOOT
    # better, and this is the column that shows it.
    err: dict[int, list] = collections.defaultdict(list)
    tot = collections.Counter()
    tp = fp = fp_in = tp_px = fp_px = 0
    n_det = 0
    # The pitch polygon is not decoration: track.py drops every box whose foot
    # falls outside it, so a false positive in the stand is one the product
    # never sees. Magnifying the far field magnifies the CROWD behind it, and
    # scoring that as lost precision would be scoring a filter that exists.
    # cam is None on a near-nadir clip, where the single-view recovery refuses
    # (sceneprior.MAX_CAMERA_H_M). Everything in PIXEL space needs it - the
    # polygon, the GT projection, and the expected-height gate the px column
    # divides by - so those columns go nan rather than zero, because a zero
    # here reads as "the detector found nothing" and nan reads as "not asked".
    # The METRE column needs only cal.to_pitch and is unaffected, which is the
    # column that matters anyway: on these clips the calibration is rms 0.029 m
    # against the panorama's 2.845, so the 2 m gate is a detector gate here and
    # not the calibration gate it is on 118575.
    # track.make_prior ITSELF, not a copy of its rules: this said "exactly what
    # track.py keeps" while listing the gates by hand, so when the height gate
    # came off the shipped default (N19/N20) the copy would have gone on
    # scoring a filter the product no longer runs.
    # The polygon is needed for the per-band bookkeeping below WHATEVER the
    # shipped prior does with it, so it is built here rather than borrowed.
    poly = None if cam is None else sceneprior.pitch_polygon(cam, cal).astype(np.float32)
    keep = None
    if prior and cam is not None:
        from track import make_prior  # noqa: PLC0415 - avoids a cycle at import
        if image_wh is None:
            raise ValueError("score(prior=True) needs image_wh: make_prior "
                             "checks the clicks against the frame size")
        keep = make_prior(cal, image_wh)
    for boxes, gt in zip(dets, gts):
        b = np.array([d["xyxy"] for d in boxes if d["conf"] >= conf], float).reshape(-1, 4)
        if keep is not None and len(b):
            b = b[keep(b)]
        n_det += len(b)
        feet = np.column_stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]])
        met = cal.to_pitch(feet) if len(feet) else np.zeros((0, 2))
        if cam is not None:
            p = np.column_stack([gt, np.ones(len(gt))])
            q = (cam.G @ p.T).T
            uv = q[:, :2] / q[:, 2:3]
            want = sceneprior.expected_height_px(cam, gt)
        bands = np.minimum(9, np.floor(gt[:, 1] / BAND_M).astype(int))
        for band in bands:
            tot[int(band)] += 1
        if not len(b):
            continue
        d_m = np.linalg.norm(met[None, :, :] - gt[:, None, :], axis=2)
        if cam is None:
            on_pitch = np.ones(len(feet), bool)
            spaces = ((d_m, GATE_M, hit_m),)
        else:
            d_px = np.linalg.norm(feet[None, :, :] - uv[:, None, :], axis=2) / want[:, None]
            on_pitch = np.array([cv2.pointPolygonTest(poly, (float(u), float(v)), False) >= 0
                                 for u, v in feet])
            spaces = ((d_m, GATE_M, hit_m), (d_px, GATE_H, hit_px))
        for D, gate, hit in spaces:
            taken_g, taken_d = set(), set()
            for i, j in zip(*np.unravel_index(np.argsort(D, axis=None), D.shape)):
                if D[i, j] >= gate:
                    break
                if i in taken_g or j in taken_d:
                    continue
                taken_g.add(int(i))
                taken_d.add(int(j))
                hit[int(bands[i])] += 1
                if D is not d_m:
                    err[int(bands[i])].append(float(d_m[i, j]))
            miss_in = sum(1 for j in range(len(b)) if j not in taken_d and on_pitch[j])
            if D is d_m:
                tp += len(taken_d)
                fp += len(b) - len(taken_d)
                fp_in += miss_in
            else:
                tp_px += len(taken_d)
                fp_px += miss_in
    n = sum(tot.values())
    return {
        "recall_m": sum(hit_m.values()) / n,
        "recall_px": sum(hit_px.values()) / n if cam is not None else float("nan"),
        # Precision is capped below 1: GT holds the 22 players and not the ~3
        # officials, who are real people the detector is right to find.
        "precision": tp / max(1, tp + fp),
        "precision_on_pitch": tp / max(1, tp + fp_in) if cam is not None else float("nan"),
        # Precision in the image, on the pitch: the share of on-pitch boxes that
        # sit on a real player. The metric one is not a precision at all at the
        # far side - a box ON a player 3 m out in metres counts against it.
        "precision_px": (tp_px / max(1, tp_px + fp_px)) if cam is not None else float("nan"),
        "det_per_frame": n_det / max(1, len(dets)),
        "band_m": {b: hit_m[b] / tot[b] for b in sorted(tot)},
        "band_px": {b: (hit_px[b] / tot[b] if cam is not None else float("nan"))
                    for b in sorted(tot)},
        "band_n": dict(sorted(tot.items())),
        "err_far_m": float(np.median(sum((err[b] for b in (0, 1, 2)), []) or [np.nan])),
        "err_near_m": float(np.median(sum((err[b] for b in (7, 8, 9)), []) or [np.nan])),
    }


def _fmt(name: str, r: dict, sec: float) -> str:
    far_m = np.mean([r["band_m"][b] for b in (0, 1, 2) if b in r["band_m"]])
    far_px = np.mean([r["band_px"][b] for b in (0, 1, 2) if b in r["band_px"]])
    return (f"{name:28s} {r['det_per_frame']:6.1f} {r['recall_m']:7.3f} {far_m:7.3f} "
            f"{r['recall_px']:7.3f} {far_px:7.3f} {r['precision']:7.3f} "
            f"{r['precision_on_pitch']:7.3f} {r['precision_px']:7.3f} "
            f"{r['err_far_m']:7.2f} {sec:7.2f}")


HEAD = (f"{'config':28s} {'det/fr':>6s} {'rec_m':>7s} {'far_m':>7s} "
        f"{'rec_px':>7s} {'far_px':>7s} {'prec':>7s} {'prec_in':>7s} {'prec_px':>7s} "
        f"{'ferr_m':>7s} {'s/fr':>7s}")


def bands(r: dict, label: str = "") -> str:
    """Per-band recall in both spaces - the headline number hides the problem."""
    ks = sorted(r["band_m"])
    return (f"\n    [{label}]" if label else ""
            ) + ("\n    band   " + " ".join(f"{k:6d}" for k in ks)
            + "\n    n      " + " ".join(f"{r['band_n'][k]:6d}" for k in ks)
            + "\n    m@2m   " + " ".join(f"{r['band_m'][k]:6.3f}" for k in ks)
            + "\n    px     " + " ".join(f"{r['band_px'][k]:6.3f}" for k in ks))


def configs(uniform, plan) -> dict:
    """name -> detector kwargs. `uniform` is the shipping 16-tile grid and
    `plan(**kw)` builds a geometry plan for this frame.

    `plan=None` returns only the uniform-grid family. Geometry plans need a
    recovered camera, and on a near-nadir clip there is not one - and the plans
    are built EAGERLY here, so this has to be decided before the dict is made
    rather than filtered out of it afterwards.

    Geometry passes cover the FAR FIELD ONLY, and that is not laziness: the
    band is 870 px tall, so no square tile can exceed 870, and the uniform grid
    is therefore already as coarse as the near field can be tiled. The uniform
    grid IS the near-field plan; the only thing left to buy is far-field
    magnification, and the far field is cheap to buy because the pitch is a
    trapezoid - the far touchline is 1,954 px wide out of 11,878.
    """
    uniform_only = {
        "base r576 t1024": dict(res=576, tile=1024),
        "base r768 t1024": dict(res=768, tile=1024),
        "base r1024 t1024": dict(res=1024, tile=1024),
        "base r768 t768": dict(res=768, tile=768),
        # THE SHIPPED DETECTOR: app/server/jobs.py builds get_detector("coreml"),
        # the rfdetr-medium-576 mlpackage tiled at 1024 on the ANE - built here
        # through that same factory. AT THE JOB'S CONF, not the sweep's 0.15:
        # the tile merge ranks whole boxes over clipped ones, so detecting low
        # and thresholding after the merge can lose a player a job keeps.
        # Score this row at conf 0.5 only. Last, so a machine without the
        # worker still runs the PyTorch rows before it fails.
        "coreml r576 t1024": dict(res=576, tile=1024, backend="coreml", det_conf=0.5),
    }
    if plan is None:
        return uniform_only
    return {
        **uniform_only,
        "base+far tgt96": dict(res=768, plan=uniform + plan(res=768, target_px=96,
                                                            max_h_px=48)),
        "base+far tgt128": dict(res=768, plan=uniform + plan(res=768, target_px=128,
                                                             max_h_px=48)),
        "base+far tgt160": dict(res=768, plan=uniform + plan(res=768, target_px=160,
                                                             max_h_px=48)),
        "far only tgt128": dict(res=768, plan=plan(res=768, target_px=128, max_h_px=48)),
    }


def _prepare(n_frames: int, every: int, first: int, only: str | None,
             gt_pixels: bool):
    """Everything both `sweep` and `run` need before a box is detected.

    Extracted so `run` can hand `api.py` a number without re-deriving the
    sample, the calibration or the tile plan. Two copies of this setup is two
    places for the frame sample to drift, and a measurement scored on a
    different sample than the printed table is the same class of bug as
    postmortem (not published) A2.
    """
    cal = calibrators.load(CALIB or f"{JOB}/calibration.json")
    gt_all = load_gt(GT, cal if gt_pixels else None)
    idx = [f for f in range(first, first + n_frames * every, every) if f in gt_all]
    if not idx:
        raise SystemExit(
            f"no sampled frame is in the ground truth: --first {first} --every {every} "
            f"-n {n_frames} spans frames {first}..{first + n_frames * every} and the GT "
            f"holds {min(gt_all)}..{max(gt_all)}. A 30 s clip needs a smaller --first.")
    gts = [gt_all[f] for f in idx]
    if getattr(cal, "_mapx", None) is not None:
        h, w = cal._mapx.shape[:2]
    else:
        cap = cv2.VideoCapture(VIDEO)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    # Degrade exactly as track.make_prior does rather than dying: on the
    # top-down 30 s clips the recovery is near-degenerate and now refuses, and
    # a sweep that cannot score the DETECTOR because the PRIOR is unavailable
    # would be measuring the wrong thing anyway. The "+prior" rows are dropped
    # and said to be dropped; the plain rows are the answer on that footage.
    try:
        cam = sceneprior.camera_from_homography(cal.H, (w, h))
    except ValueError as e:
        print(f"scene prior unavailable ({e}); +prior rows are SKIPPED, not zero")
        cam = None
    # The geometry plans need a camera; without one only the uniform grids can
    # be built, which is the whole "base r*" family.
    cfgs = configs(TiledDetector(_stub(), tile=1024).grid(w, h),
                   None if cam is None else
                   (lambda **kw: geometry_tiles(cam, cal, (w, h), **kw)))
    cfgs = {k: v for k, v in cfgs.items() if only is None or only in k}
    return cal, cam, gts, idx, cfgs, w, h


#: Box cache for re-scoring one detection run at many thresholds. Keyed on
#: config name, clip and frames - NOT on weights, detector or remap - so a
#: caller that needs a fresh detection (model/validation/measurements.py) sets None.
BOX_CACHE = "/tmp/detsweep_dets.json"


#: Detector confidence when a configuration does not pin one. Every row is
#: re-scored above this, so it is a floor on the ladder, not an operating point.
DET_CONF = 0.15


def _key(name: str, idx: list[int], det_conf: float, overlap: float) -> str:
    """The box-cache key. The clip is in it, because without it a soccer_top run
    silently scores the 118575 boxes still sitting in /tmp under the same config
    name. The DETECTOR's conf is in it for the same reason one step in: boxes
    detected at 0.15 and thresholded to 0.5 are not the boxes detected at 0.5 -
    the tile merge ranks whole boxes over clipped ones - and until 2026-09-12
    the two shared a key, so asking for the second replayed the first. The TILE
    OVERLAP is in it for the third instance of the same bug: 13 tiles and 16
    tiles over one frame are two detection runs, and the 2026-09-12 overlap
    arms had to be kept apart by hand in two /tmp files until this line."""
    return f"{name}|{os.path.basename(VIDEO)}|{idx[0]}|{len(idx)}|d{det_conf}|o{overlap}"


def _boxes(name: str, kw: dict, idx: list[int], cal, w: int, h: int) -> dict:
    """Detect (or read back) one configuration's boxes over the sampled frames."""
    cache_path = BOX_CACHE
    cache = (json.load(open(cache_path))
             if cache_path and os.path.exists(cache_path) else {})
    from detectors import OVERLAP  # noqa: PLC0415 - the shipped default, read late
    ov = kw.get("overlap", OVERLAP)
    key = _key(name, idx, kw.get("det_conf", DET_CONF), ov)
    if key not in cache:
        if kw.get("backend") == "coreml":
            from detectors import get_detector  # noqa: PLC0415
            det = get_detector("coreml", conf=kw.get("det_conf", DET_CONF),
                               tile=kw.get("tile", 1024), overlap=ov)
        else:
            det = TiledDetector(RFDETRDetector(variant="medium", resolution=kw["res"],
                                               conf=kw.get("det_conf", DET_CONF)),
                                tile=kw.get("tile", 1024), plan=kw.get("plan"),
                                overlap=ov)
        out, secs = [], []
        for img in sample_frames(idx, cal):
            t0 = time.perf_counter()
            out.append(det.detect(img))
            secs.append(time.perf_counter() - t0)
        # Frame 0 pays for tracing and the first MPS allocation; the median
        # of the rest is what a long run actually costs per frame.
        cache[key] = {"dets": out, "sec": float(np.median(secs[1:])),
                      "tiles": len(det.grid(w, h)), "overlap": ov}
        if cache_path:
            json.dump(cache, open(cache_path, "w"))
    return cache[key]


def run(name: str = "base r768 t1024", conf: float = 0.5, prior: bool = True,
        n_frames: int = 40, every: int = 1500, first: int = 3000,
        gt_pixels: bool = False, det_conf: float | None = None,
        overlap: float | None = None, per_frame: bool = False) -> dict:
    """ONE configuration at ONE confidence -> `score`'s dict, plus the scope.

    The accessor `sweep` never had: it prints a table and returns None, so the
    only way to quote a detection number was to regex its stdout. `n` here is
    GT PLAYER-FRAMES, not frames - recall's denominator is players.

    `det_conf` overrides what the DETECTOR is asked for, which is a different
    knob to `conf`: `conf` thresholds boxes that already exist, `det_conf`
    decides which ones the detector emits in the first place, and on a tiled
    merge those two are not the same operating point. It is in the cache key and
    in the returned scope, so a run cannot quietly borrow the other one's boxes.
    Leave it None and the configuration's own value stands (the shipped coreml
    row pins 0.5 - see configs()).

    `overlap` is the third knob of the same kind and it is in the cache key too:
    13 tiles and 16 tiles over one frame are two detection runs. None takes the
    shipped `detectors.OVERLAP`.

    `per_frame` additionally returns ONE `score` dict per frame, which is what
    makes two arms comparable PAIRED rather than as two pooled rates - see
    `overlap_decision`. It is the same scorer, called a frame at a time, so
    there is no second matching path to drift.
    """
    cal, cam, gts, idx, cfgs, w, h = _prepare(n_frames, every, first, name, gt_pixels)
    if name not in cfgs:
        raise SystemExit(f"no configuration named {name!r}; have {sorted(cfgs)}")
    if prior and cam is None:
        raise SystemExit("prior=True but the scene prior is unavailable on this footage")
    kw = dict(cfgs[name])
    if det_conf is not None:
        kw["det_conf"] = det_conf
    if overlap is not None:
        kw["overlap"] = overlap
    b = _boxes(name, kw, idx, cal, w, h)
    r = score(b["dets"], gts, cal, cam, conf, prior=prior, image_wh=(w, h))
    out = {**r, "config": name, "conf": conf, "prior": prior,
           "det_conf": kw.get("det_conf", DET_CONF), "overlap": b["overlap"],
           "sec_per_frame": b["sec"], "tiles": b["tiles"],
           "frames": len(idx), "gt_player_frames": int(sum(len(g) for g in gts)),
           "first": idx[0], "every": every, "video": os.path.basename(VIDEO)}
    if per_frame:
        out["per_frame"] = [score([d], [g], cal, cam, conf, prior=prior,
                                  image_wh=(w, h)) for d, g in zip(b["dets"], gts)]
    return out


def _paired_ci(a: list, b: list, n: list, boot: int = 2000, seed: int = 0) -> dict:
    """Percentile bootstrap on a POOLED recall difference, resampling FRAMES.

    `a`, `b` are per-frame hit counts for one arm each and `n` the per-frame GT
    count, identical for both arms because both arms see the same frames. The
    statistic is the ratio of sums, (sum a - sum b) / sum n, which is the same
    estimator as the printed point difference rather than a mean of per-frame
    rates - a frame holding one band-0 player must not outvote a frame holding
    five.

    Resampling FRAMES and not players is the pairing: two players in one frame
    share a tiling, a crowd and a play, so they are not two independent draws,
    and 7c2cd35's +-6 points at one sigma is the unpaired interval that treats
    them as if they were.
    """
    a, b, n = (np.asarray(x, float) for x in (a, b, n))
    point = float((a.sum() - b.sum()) / max(1.0, n.sum()))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(boot, len(n)))
    d = (a - b)[idx].sum(1) / np.maximum(1.0, n[idx].sum(1))
    return {"delta": point, "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)), "p_gt0": float((d > 0).mean()),
            "n_gt": int(n.sum()), "frames": int(len(n))}


def _hits(rows: list[dict], band: int | None) -> list[int]:
    """Per-frame image-space hits for one band (None = every band)."""
    if band is None:
        return [int(round(sum(r["band_px"][b] * r["band_n"][b] for b in r["band_n"])))
                for r in rows]
    return [int(round(r["band_px"].get(band, 0.0) * r["band_n"].get(band, 0)))
            for r in rows]


def _gtn(rows: list[dict], band: int | None) -> list[int]:
    return [sum(r["band_n"].values()) if band is None else r["band_n"].get(band, 0)
            for r in rows]


def overlap_decision(n_frames: int = 96, every: int = 700, first: int = 3000,
                     conf: float = 0.5, det_conf: float = 0.5,
                     candidate: float = 0.10, ship: float | None = None,
                     band: int = 1, boot: int = 2000, seed: int = 0,
                     min_band: float = 0.05, min_all: float = 0.01,
                     max_prec_loss: float = 0.01) -> dict:
    """Does the tile overlap move from what ships to `candidate`? The pass rule
    in the module docstring, in code, so the verdict is produced by the run
    rather than read off a table afterwards. The delta is candidate minus
    shipped throughout.

    `ship=None` is detectors.OVERLAP, resolved here and in no other place; the
    result carries it and print_overlap_decision reads it back. It replaces an
    `arms=(0.10, 0.25)` default whose first element was "shipped" - a copy of
    OVERLAP added in 8472cef, the commit that moved OVERLAP to 0.25 - while the
    printer re-derived the shipped arm a second way, by sorting. Pass `ship`
    only to replay a decision taken while something else shipped:
    tile-overlap (not published) section 3 replays 2026-09-12's with ship=0.10,
    candidate=0.25. `candidate` defaults to the protocol's other arm.

    Every other default here is the declared protocol: 96 frames every 700 from
    3000, conf 0.50 with the DETECTOR asked for 0.5 (what app/server/jobs.py runs),
    band 1 as the primary because 298e62b's 40-frame reading named band 1, a
    paired percentile bootstrap at 2,000 resamples and seed 0, and the three
    gates - a CI strictly above zero, a material size (>= 5 points on band 1 or
    >= 1 point overall), and no more than 1 point of image precision.
    """
    from detectors import OVERLAP  # noqa: PLC0415 - the shipped default, read late
    ship = OVERLAP if ship is None else ship
    if candidate == ship:
        raise SystemExit(f"candidate overlap {candidate} is the shipped arm - nothing to decide")
    cand, arms = candidate, (ship, candidate)
    r = {ov: run("coreml r576 t1024", conf, True, n_frames, every, first,
                 det_conf=det_conf, overlap=ov, per_frame=True) for ov in arms}
    # The prior's drop share, both arms: boxes a frame before it and after it.
    raw = {ov: run("coreml r576 t1024", conf, False, n_frames, every, first,
                   det_conf=det_conf, overlap=ov)["det_per_frame"] for ov in arms}
    ci = {b: _paired_ci(_hits(r[cand]["per_frame"], b), _hits(r[ship]["per_frame"], b),
                        _gtn(r[ship]["per_frame"], b), boot, seed)
          for b in (0, 1, 2, None)}
    # THE FLOOR, on the shipped arm and free: the same boxes against the next
    # sampled frame's ground truth, 28 s away. Whatever it scores is what box
    # density alone buys, and a band figure at or under it is not detection.
    cal, cam, gts, idx, cfgs, w, h = _prepare(n_frames, every, first,
                                              "coreml r576 t1024", False)
    bx = _boxes("coreml r576 t1024",
                {**cfgs["coreml r576 t1024"], "det_conf": det_conf, "overlap": ship},
                idx, cal, w, h)
    null = score(bx["dets"], gts[1:] + gts[:1], cal, cam, conf, prior=True,
                 image_wh=(w, h))
    prec_loss = r[ship]["precision_px"] - r[cand]["precision_px"]
    a = ci[band]["lo"] > 0
    b_ = ci[band]["delta"] >= min_band or ci[None]["delta"] >= min_all
    c = prec_loss <= max_prec_loss
    if a and b_ and c:
        verdict = f"REVERT to overlap {cand}"
    elif a:
        verdict = ("DECISION NEEDED: real but "
                   + ("small" if not b_ else f"paid for in {prec_loss:.3f} precision"))
    else:
        half = (ci[band]["hi"] - ci[band]["lo"]) / 2
        # UNDERPOWERED needs the CI to CONTAIN zero (module docstring); failing
        # gate (a) only gives lo <= 0. A CI wholly below zero is the candidate
        # measurably losing, not a sample too small to see it.
        verdict = f"KEEP overlap {ship}" + (
            f" - UNDERPOWERED, ci95 half-width {half:.3f}"
            if half >= min_band and ci[band]["hi"] >= 0 else "")
    return {"arms": {ov: {k: r[ov][k] for k in
                          ("band_px", "band_m", "band_n", "recall_px", "recall_m",
                           "precision_px", "det_per_frame", "tiles", "sec_per_frame",
                           "gt_player_frames", "frames")} | {"det_per_frame_no_prior": raw[ov]}
                     for ov in arms},
            "ci": ci, "precision_loss": prec_loss,
            "null": {k: null[k] for k in ("band_px", "recall_px")},
            "gates": {"a_ci": a, "b_size": b_, "c_precision": c},
            "verdict": verdict, "band": band, "conf": conf, "det_conf": det_conf,
            "ship": ship, "candidate": cand, "n_configurations": len(arms)}


def print_overlap_decision(d: dict) -> None:
    ship, cand = d["ship"], d["candidate"]
    print(f"{d['arms'][ship]['frames']} frames, "
          f"{d['arms'][ship]['gt_player_frames']} GT player-frames, conf "
          f"{d['conf']:.2f}, det_conf {d['det_conf']:.2f}, n={d['n_configurations']} "
          f"configurations")
    print(f"{'overlap':>8s} {'tiles':>6s} {'s/fr':>6s} {'det/fr':>7s} {'raw/fr':>7s} "
          + " ".join(f"{'b' + str(b):>7s}" for b in (0, 1, 2))
          + f" {'rec_px':>7s} {'prec_px':>7s} {'rec_m':>7s}")
    for ov in (ship, cand):
        a = d["arms"][ov]
        print(f"{ov:8.2f} {a['tiles']:6d} {a['sec_per_frame']:6.3f} "
              f"{a['det_per_frame']:7.2f} {a['det_per_frame_no_prior']:7.2f} "
              + " ".join(f"{a['band_px'][b]:7.3f}" for b in (0, 1, 2))
              + f" {a['recall_px']:7.4f} {a['precision_px']:7.4f} {a['recall_m']:7.4f}")
    print("  band n   " + " ".join(f"{d['arms'][ship]['band_n'][b]:7d}" for b in (0, 1, 2)))
    print(f"  null (overlap {ship}, GT 28 s away)   "
          + " ".join(f"{d['null']['band_px'][b]:7.3f}" for b in (0, 1, 2))
          + f" {d['null']['recall_px']:7.4f}")
    for b in (0, 1, 2, None):
        c = d["ci"][b]
        print(f"  paired delta band {str(b):>4s}  n_gt {c['n_gt']:5d}  "
              f"{c['delta']:+.4f}  ci95 [{c['lo']:+.4f}, {c['hi']:+.4f}]  "
              f"P(>0) {c['p_gt0']:.3f}")
    print(f"  gates a_ci={d['gates']['a_ci']} b_size={d['gates']['b_size']} "
          f"c_precision={d['gates']['c_precision']} "
          f"(precision loss {d['precision_loss']:+.4f})")
    print(f"VERDICT: {d['verdict']}")


def sweep(n_frames: int = 40, every: int = 1500, first: int = 3000,
          confs: tuple = (0.15, 0.3, 0.5), only: str | None = None,
          gt_pixels: bool = False) -> None:
    """`only` runs just the configurations whose name contains it, and every
    configuration's boxes are cached to /tmp. Both exist for one reason: MPS
    does not hand memory back, so a second RF-DETR built in the same process
    OOMs at 19 GiB and falls back to fp32 - which then reports a wrong
    seconds-per-frame for every configuration after the first. One process per
    configuration is the fix; the cache is what makes re-scoring free.
    """
    cal, cam, gts, idx, cfgs, w, h = _prepare(n_frames, every, first, only, gt_pixels)
    print(f"{len(idx)} frames of {w}x{h}, {sum(len(g) for g in gts)} GT player-frames")
    print(HEAD)
    for name, kw in cfgs.items():
        b = _boxes(name, kw, idx, cal, w, h)
        out, sec, n_tiles = (b[k] for k in ("dets", "sec", "tiles"))
        for c in confs:
            plain = score(out, gts, cal, cam, c)
            print(_fmt(f"{name} t{n_tiles} c{c}", plain, sec), flush=True)
            rp = None
            if cam is not None:
                rp = score(out, gts, cal, cam, c, prior=True, image_wh=(w, h))
                print(_fmt("  +prior", rp, sec), flush=True)
            # The null. The same boxes scored against a DIFFERENT frame's ground
            # truth (60 s away, so the players are somewhere else): whatever it
            # scores is what box density alone buys, and a real number has to
            # beat it. At conf 0.15 the detector emits 55 boxes for 22 players,
            # and a metric that cannot tell that from finding them is the sixth
            # metric in an earlier run that measured nothing.
            print(_fmt("  null", score(out, gts[1:] + gts[:1], cal, cam, c), sec), flush=True)
            if c == confs[-1]:
                # LABELLED, because it was not. This table is the +prior result
                # where there is one, and an all-zero prior table is
                # indistinguishable from "the detector found nothing" unless it
                # says which of the two rows above it belongs to.
                print(bands(rp if rp is not None else plain,
                            f"+prior c{c}" if rp is not None else f"no prior c{c}"),
                      flush=True)


def _stub():
    class _S:
        resolution = 768
    return _S()


def _selfcheck() -> None:
    """Scoring must be right before any number off it is believed."""
    cal = calibrators.load(f"{JOB}/calibration.json")
    cam = sceneprior.camera_from_homography(cal.H, (11878, 870))
    gt = load_gt()
    assert min(gt) == 0 and len(gt) > 70000, (min(gt), len(gt))
    g = gt[30000]
    assert g.shape == (22, 2), g.shape

    # Perfect detections: put a box exactly where the geometry says each player
    # is, and both spaces must read 1.0. If they do not, the projection or the
    # matcher is wrong and every sweep number is noise.
    p = np.column_stack([g, np.ones(len(g))])
    q = (cam.G @ p.T).T
    uv = q[:, :2] / q[:, 2:3]
    ht = sceneprior.expected_height_px(cam, g)
    perfect = [{"xyxy": (u - 5, v - hh, u + 5, v), "conf": 0.9, "cls": "person"}
               for (u, v), hh in zip(uv, ht)]
    r = score([perfect], [g], cal, cam)
    assert r["recall_m"] == 1.0 and r["recall_px"] == 1.0 and r["precision"] == 1.0, r
    assert all(v == 1.0 for v in r["band_m"].values()), r["band_m"]

    # One box cannot satisfy two players: matching is exclusive.
    r1 = score([perfect[:1]], [g], cal, cam)
    assert abs(r1["recall_m"] - 1 / 22) < 1e-9, r1["recall_m"]

    # A box 3 m up-pitch of every player passes no 2 m gate...
    off = cal.to_pitch(np.column_stack([uv[:, 0], uv[:, 1]]))
    assert np.median(np.linalg.norm(off - g, axis=1)) < 0.5, "to_pitch/G disagree"
    shifted = [{"xyxy": (u - 5, v - hh, u + 5, v), "conf": 0.9, "cls": "person"}
               for (u, v), hh in zip(uv + np.array([0.0, 40.0]), ht)]
    r2 = score([shifted], [g], cal, cam)
    assert r2["recall_m"] < 0.3, r2["recall_m"]

    # ...and confidence filtering is applied, not ignored.
    faint = [{**d, "conf": 0.2} for d in perfect]
    assert score([faint], [g], cal, cam, conf=0.5)["recall_m"] == 0.0
    assert score([faint], [g], cal, cam, conf=0.15)["recall_m"] == 1.0

    # The prior keeps geometrically-correct boxes and drops a crowd-shaped one
    # in the stand. If it dropped the players too, every "+prior" row would be
    # measuring the filter rather than the detector.
    crowd = [{"xyxy": (500.0, 20.0, 560.0, 80.0), "conf": 0.9, "cls": "person"}]
    rp = score([perfect + crowd], [g], cal, cam, prior=True, image_wh=(11878, 870))
    assert rp["recall_px"] > 0.9, rp["recall_px"]
    assert rp["det_per_frame"] == 22, rp["det_per_frame"]
    # ...and the prior it applies is track.make_prior itself, so the gates this
    # scores follow the shipped default instead of a copy that drifts from it.
    import track  # noqa: PLC0415
    tall = [{"xyxy": (u - 5, v - 3 * hh, u + 5, v), "conf": 0.9, "cls": "person"}
            for (u, v), hh in list(zip(uv, ht))[:1]]
    kept = score([perfect + tall], [g], cal, cam, prior=True, image_wh=(11878, 870))
    assert kept["det_per_frame"] == 22 + len(tall) * (
        "height" not in track.make_prior.__defaults__[0]), kept["det_per_frame"]

    # The box cache separates two detector confidences. Without this the
    # shipped row (det_conf 0.5) and the conf ladder (det_conf 0.15) share a
    # key, and whichever ran second silently reported the first one's boxes -
    # which is how a conf sweep reports that confidence does nothing.
    assert _key("c", [3000] * 40, 0.15, 0.10) != _key("c", [3000] * 40, 0.5, 0.10)
    assert _key("c", [3000], 0.5, 0.10) != _key("d", [3000], 0.5, 0.10)
    # ...and the tile overlap, the third knob of the same shape: 13 tiles and 16
    # tiles over one frame are two detection runs, and until 2026-09-12 they
    # collided on one key and had to be kept apart by hand in two /tmp files.
    assert _key("c", [3000], 0.5, 0.10) != _key("c", [3000], 0.5, 0.25)

    # THE PASS RULE ITSELF, on hand-made per-frame rows, because a rule that
    # cannot fire in both directions is not a rule. Ten frames, band 1 holding
    # two GT players each: the candidate arm hits 18 of 20 and the shipped arm
    # 8 of 20, which is a +0.50 difference no 2,000-resample interval can
    # straddle...
    big = _paired_ci([2] * 9 + [0], [1] * 8 + [0, 0], [2] * 10)
    assert big["delta"] == 0.5 and big["lo"] > 0, big
    # ...and the same totals shuffled so the two arms tie must NOT clear zero.
    tie = _paired_ci([1, 2, 0, 2, 1, 2, 0, 1, 2, 1], [2, 1, 1, 1, 2, 1, 1, 2, 0, 1],
                     [2] * 10)
    assert tie["delta"] == 0.0 and tie["lo"] < 0 < tie["hi"], tie
    # A hit count is a rate times a count, and a rate the scorer prints to three
    # places would round 7/22 back to 6 if it were taken off the printed table.
    row = {"band_px": {0: 7 / 22, 1: 0.0}, "band_n": {0: 22, 1: 3}}
    assert _hits([row], 0) == [7] and _hits([row], 1) == [0], _hits([row], 0)
    assert _hits([row], None) == [7] and _gtn([row], None) == [25]
    print("detsweep.py self-check OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--overlap-decision", action="store_true",
                    help="the tile-overlap pass rule in this file's docstring: "
                         "two overlaps, paired over the same frames at the SHIPPED "
                         "conf, and the verdict it produces")
    ap.add_argument("-n", "--frames", type=int, default=40)
    ap.add_argument("--every", type=int, default=1500,
                    help="frames between samples; 1500 is 60 s, so no two samples "
                         "hold the same play")
    ap.add_argument("--only", default=None,
                    help="run only configurations whose name contains this; run one "
                         "process per configuration, or MPS OOMs and drops to fp32")
    ap.add_argument("--first", type=int, default=3000,
                    help="first frame to sample; 3000 skips the panorama walk-out. "
                         "A 900-frame clip needs 0")
    ap.add_argument("--clip", default=None,
                    help="a SoccerTrack 30 s clip stem, e.g. "
                         "soccer_top__test__videos__D_20220220_1_1680_1710. Sets "
                         "--gt, --video and --calib together and switches the GT to "
                         "pixel space, because those three must agree")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--gt-space", choices=("metres", "pixels"), default=None,
                    help="what x,y in the GT file MEAN. 118575 is metres, the 30 s "
                         "clips are pixels. Wrong answer reports recall 0, not an error")
    a = ap.parse_args()
    _selfcheck()
    if a.clip:
        # One flag, not four, because a GT file from one clip against the video
        # of another scores zero and looks like a detector regression.
        GT = f"model/validation/{a.clip}.tracks.json.gz"
        VIDEO = f"data/footage/soccertrack/{a.clip}.mp4"
        CALIB = f"model/validation/calib_{a.clip}.json"
        if a.gt_space is None:
            a.gt_space = "pixels"
        for p in (GT, VIDEO, CALIB):
            if not os.path.exists(p):
                raise SystemExit(f"--clip {a.clip}: missing {p}")
    GT = a.gt or GT
    VIDEO = a.video or VIDEO
    CALIB = a.calib or CALIB
    if a.overlap_decision:
        print_overlap_decision(overlap_decision())
    elif not a.selfcheck:
        sweep(a.frames, a.every, a.first, only=a.only,
              gt_pixels=(a.gt_space == "pixels"))
