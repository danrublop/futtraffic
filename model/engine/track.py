"""Video -> tracks.json. Detection + tracking only; stats.py does the maths.

Detector and tracker are both pluggable (see detectors.py, tracking.py) so the
pipeline runs end to end with zero weights (`--detector stub`) and swaps in a
real model without touching this file.

FRAME NUMBERS DO NOT CHANGE WITH THE STRIDE: a strided run emits fewer rows
carrying their ORIGINAL indices at the ORIGINAL fps (f = i * stride).
Manual: docs/reference/detection-tracking.md (#contract; stride evidence: #evidence-stride)

Licensing note, load-bearing: RF-DETR is Apache-2.0, NOT Ultralytics YOLO
(AGPL-3.0). AGPL triggers on network use, so serving analysis over the web on
an Ultralytics-derived model would oblige us to publish the server source.
Do not "just swap in YOLO" - see README.
"""
import argparse
import hashlib
import inspect
import json
import os
import time
from collections import Counter
from collections.abc import Iterator

import cv2
import numpy as np

import ball as _ball
import detectors
import rolenet as _rolenet
from detectors import get_detector
from tracking import best_available, get_tracker


def team_by_shirt(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Shirt colour as (chroma_b, chroma_g, lightness). Upper-middle of the box.

    Deliberately crude and deliberately NOT a face or body-appearance embedding.
    Kit colour is the only appearance signal on the default path; rolenet (off by
    default) names a role from the crop and carries no identity. See README on BIPA.

    Grey-world chromaticity, not mean Lab: lighting varies more across a half
    than two kits differ, and dividing each pixel by its brightness removes it.
    L is kept as the third channel because it alone separates a white kit from a
    black one. detection-tracking-evidence (not published)#evidence-teams
    """
    x1, y1, x2, y2 = box
    h = y2 - y1
    # Clamp: detectors emit boxes that run off the frame edge, and a negative
    # index here silently samples the far side of the image - a wrong kit
    # colour, which is a wrong team, which is a wrong stat line.
    top, bot = max(0, y1 + int(0.2 * h)), max(0, y1 + int(0.5 * h))
    torso = frame[top:bot, max(0, x1):max(0, x2)]
    if torso.size == 0:
        return np.zeros(3)
    px = torso.reshape(-1, 3).astype(np.float32)
    # Per-PIXEL normalisation, not per-crop: a crop that is half sunlit and half
    # shadowed averages to neither, and the far touchline is routinely both.
    chroma = (px / (px.sum(axis=1, keepdims=True) + 1e-6)).mean(axis=0)[:2] * 255.0
    lightness = float(cv2.cvtColor(torso, cv2.COLOR_BGR2Lab)[:, :, 0].mean())
    # Scaled to a comparable magnitude so the vector is readable in a debugger;
    # _kmeans rescales it anyway, so the scale is not load-bearing.
    return np.array([chroma[0], chroma[1], lightness])


_KM_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)


def _kmeans(X: np.ndarray, k: int, rounds: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """k-means in the pooled WITHIN-cluster metric. -> (labels, centres).

    Plain k-means lets the widest-spread channel decide the split, and on kit
    colour that is the lighting. So the axes are rescaled by the pooled
    WITHIN-cluster scatter and re-clustered (the tied-covariance mixture idea).
    Within-cluster, not total: total-covariance sphering pulls two clean kits
    together until k=3 can cut one in half (_selfcheck_teams fails on it).
    detection-tracking-evidence (not published)#evidence-teams
    Returns centres in the SPHERED space, because the caller's separation test
    is a ratio of distances and the two must be measured in the same metric.
    """
    X = np.ascontiguousarray(np.asarray(X, np.float32))
    lab = cv2.kmeans(X, k, None, _KM_CRIT, 20, cv2.KMEANS_PP_CENTERS)[1].ravel()
    Z = X
    for _ in range(rounds):
        if len(X) < k + 2 or X.shape[1] < 2:
            break
        S = np.zeros((X.shape[1], X.shape[1]))
        n = 0
        for j in range(k):
            A = X[lab == j].astype(np.float64)
            if len(A) < 2:
                continue
            S += (A - A.mean(0)).T @ (A - A.mean(0))
            n += len(A) - 1
        w, V = np.linalg.eigh(S / max(n, 1) + 1e-9 * np.eye(X.shape[1]))
        Z = np.ascontiguousarray(X @ (V / np.sqrt(np.maximum(w, 1e-12))), np.float32)
        lab = cv2.kmeans(Z, k, None, _KM_CRIT, 20, cv2.KMEANS_PP_CENTERS)[1].ravel()
    cen = np.array([Z[lab == j].mean(0) if (lab == j).any() else np.zeros(Z.shape[1])
                    for j in range(k)])
    return lab, cen


def _by_kit(colours: dict[int, list]) -> dict[int, str]:
    """Three-means on kit colour -> home / away / official.

    Clusters PER DETECTION, then each track takes the majority label of its own
    detections: a track that swapped between players has a mean colour belonging
    to neither kit, and a vote survives a minority of mixed frames. Three
    clusters when the smallest is a genuine third kit (officials); otherwise two,
    so k=3 never deletes half a side. A referee folded into a team would be a
    phantom high-distance profile.
    docs/reference/detection-tracking.md#teams (evidence: #evidence-teams)
    """
    # Flatten to detections, remembering which track each came from. A track
    # with no usable crop at all drops out here exactly as it did before.
    tids, rows = [], []
    for i, cs in colours.items():
        for c in cs:
            if c.any():
                tids.append(i)
                rows.append(c)
    ids = sorted(set(tids))
    if len(ids) < 2:
        return {i: "home" for i in ids}

    X = np.stack(rows).astype(np.float32)
    tids = np.asarray(tids)
    # SEEDED, and 20 restarts inside _kmeans rather than 5. k-means is seeded
    # randomly and this is not a convex problem: the same 118575 input at 5
    # restarts landed anywhere between 54% and 71% team accuracy depending on
    # nothing but the RNG. A squad list that changes when you re-run the same
    # job is not a measurement, and it is not something a coach can be shown.
    cv2.setRNGSeed(0)

    def km(k: int):
        return _kmeans(X, k)

    def vote(lab, names: dict) -> dict[int, str]:
        """Each track takes the majority label of its OWN detections.

        Ties break toward the label that is rarer overall, so a track split
        evenly between a team and the officiating crew is called an official.
        The asymmetry is deliberate: stats.build_match drops officials from
        `players`, so a wrong 'official' costs one row while a wrong team label
        corrupts a squad aggregate that a coach reads.
        """
        freq = {k: int((lab == k).sum()) for k in names}
        out = {}
        for i in ids:
            counts = np.bincount(lab[tids == i], minlength=max(names) + 1)
            best = max(names, key=lambda k: (counts[k], -freq[k]))
            out[i] = names[best]
        return out

    def two() -> dict[int, str]:
        lab, _ = km(2)
        big = 0 if (lab == 0).sum() >= (lab == 1).sum() else 1
        return vote(lab, {big: "home", 1 - big: "away"})

    if len(ids) < 6:
        return two()

    lab, cen = km(3)
    sizes = sorted((int((lab == k).sum()), k) for k in range(3))
    (n_small, k_small), rest = sizes[0], [k for _, k in sizes[1:]]

    # Two independent tests, and BOTH must pass. Size alone is not enough: with
    # no officials on the pitch, k=3 happily carves one team into halves, and a
    # size threshold accepts the smaller half as a crew - deleting five real
    # players from the match. Separation is what tells the cases apart. A third
    # KIT is chosen to contrast with both teams, so it sits far from both
    # centres; a split HALF sits right next to the half it was cut from.
    #
    # The 25% is now a share of DETECTIONS rather than of tracks, which is the
    # sturdier denominator: a crew of three is a small share of detections
    # however many fragments the tracker cut them into.
    d_team = float(np.linalg.norm(cen[rest[0]] - cen[rest[1]]))
    d_small = min(float(np.linalg.norm(cen[k_small] - cen[k])) for k in rest)
    #
    # Both tests only work in the SPHERED metric, which is why _kmeans returns its
    # centres there. detection-tracking-evidence (not published)#evidence-teams
    #
    # ponytail: fixed 25% / 0.5x thresholds, not learned. Ceiling is a heavily
    # reduced side or a keeper kit close to a team kit; upgrade path is per-track
    # confidence instead of a hard label.
    if n_small > max(1, int(0.25 * len(X))) or d_small < 0.5 * d_team:
        return two()

    big = rest[0] if (lab == rest[0]).sum() >= (lab == rest[1]).sum() else rest[1]
    other = rest[0] if big == rest[1] else rest[1]
    return vote(lab, {k_small: "official", big: "home", other: "away"})


def _officials_by_contrast(colours: dict[int, list], labels: dict[int, str],
                           thresh: float = 7.0, min_n: int = 25) -> dict[int, str]:
    """Relabel a track whose kit matches NEITHER team as an official.

    A track whose mean kit is far from BOTH teams' median, in MAD units, is an
    official. Across TRACKS, not detections: a track mean averages the lighting
    away, so a referee stands out where per detection he disappears. Median and
    MAD because the outlier is inside the sample. Tracks shorter than `min_n`
    detections are not judged.

    ponytail: fixed `thresh` MADs / `min_n` detections, not learned. It misses a
    referee whose kit differs from a team mostly in lightness, and colour cannot
    separate a goalkeeper from a referee; a detector that NAMES them can
    (assign_teams step 3). detection-tracking-evidence (not published)#evidence-teams
    """
    means, counts = {}, {}
    for i, cs in colours.items():
        good = [c for c in cs if np.any(c)]
        if good:
            means[i], counts[i] = np.mean(good, axis=0), len(good)

    stats = []
    for t in ("home", "away"):
        pts = np.array([means[i] for i in means
                        if labels.get(i) == t and counts[i] >= min_n])
        # Four tracks per side before this fires at all. With fewer, the MAD is
        # measuring two players rather than a kit, and every third track looks
        # like an official.
        if len(pts) >= 4:
            med = np.median(pts, axis=0)
            stats.append((med, np.maximum(np.median(np.abs(pts - med), axis=0) * 1.4826, 1e-3)))
    if len(stats) < 2:
        return labels

    out = dict(labels)
    for i in means:
        # Short tracks are excluded rather than judged: a handful of detections
        # is a mean with no stability, and on 118575 the 1-5 detection fragments
        # score 6-10 while being ordinary players in shadow.
        if counts[i] < min_n or labels.get(i) == "official":
            continue
        if min(float(np.linalg.norm((means[i] - med) / mad)) for med, mad in stats) > thresh:
            out[i] = "official"
    return out


#: Detector class names that are an OFFICIAL, not a player. A COCO checkpoint
#: emits "person" for everyone and none of these ever appear, so this is inert
#: until a class-aware model is wired in - which is the point: colour cannot
#: separate a referee from a keeper (_officials_by_contrast) and a name can.
OFFICIAL_CLASSES = frozenset({"referee", "linesman", "assistant_referee"})

#: ...and the names that positively assert a PLAYER. "person" is deliberately
#: NOT one: it is what a COCO checkpoint calls everyone including the referee,
#: so treating it as a claim would let a class-blind model overrule the colour
#: gate for the whole pitch.
PLAYER_NAMES = frozenset({"player", "goalkeeper", "keeper"})


def assign_teams(colours: dict[int, list],
                 classes: dict[int, "Counter"] | None = None) -> dict[int, str]:
    """Kit colour -> home / away / official, per track.

    Three stages, in increasing order of authority:

      1. `_by_kit` clusters the kits and can pull out a CROWD in a third kit -
         on the DFL clip, four coaches and a photographer in black.
      2. `_officials_by_contrast` catches the INDIVIDUAL who matches neither
         team and is too rare to be a cluster. It has a measured ceiling: a
         referee in yellow against green scores 4.9 where a real player scores
         4.4, so it does not reach every one of them.
      3. `classes`, when a detector supplies them, OVERRIDES both. A name is not
         a guess: it is the only signal that separates a referee from a
         goalkeeper, who contrast with both teams by exactly the same rule.

    `classes` is `{track_id: Counter(class_name)}` - the detector's own votes
    over that track's detections, majority wins. A COCO checkpoint emits
    "person" for everyone, so passing them changes nothing; that is what makes
    this safe to wire in ahead of the model.
    """
    labels = _officials_by_contrast(colours, _by_kit(colours))
    if not classes:
        return labels
    for tid, votes in classes.items():
        if not votes:
            continue
        name = votes.most_common(1)[0][0]
        if name in OFFICIAL_CLASSES:
            labels[tid] = "official"
        elif name in PLAYER_NAMES and labels.get(tid) == "official":
            # The detector says PLAYER (or goalkeeper) and colour said official.
            # Trust the name: this is the goalkeeper case that colour gets
            # wrong, and calling a keeper an official deletes him from his squad
            # (stats.build_match drops officials from `players`). Re-derive his
            # side from his own kit against the two team centres.
            labels[tid] = _nearest_team(colours, labels, tid)
    return labels


def _nearest_team(colours: dict[int, list], labels: dict[int, str], tid: int) -> str:
    """The side whose kit centre this track's mean colour is nearest.

    Used only to put a track BACK on a team after a class name overruled the
    colour gate. A keeper's kit is unlike either outfield kit by construction,
    so this is a coin toss dressed as a measurement - which is why it is only
    ever reached for a track the detector already named a player, never used to
    label anyone in the first place.
    """
    good = [c for c in colours.get(tid, ()) if np.any(c)]
    if not good:
        return "home"
    m = np.mean(good, axis=0)
    best, bd = "home", float("inf")
    for t in ("home", "away"):
        pts = [np.mean([c for c in colours[i] if np.any(c)], axis=0)
               for i in colours if labels.get(i) == t
               and any(np.any(c) for c in colours[i])]
        if not pts:
            continue
        d = float(np.linalg.norm(m - np.mean(pts, axis=0)))
        if d < bd:
            best, bd = t, d
    return best


def make_prior(cal, image_wh: tuple, gates: tuple = ("pitch",)):
    """Geometry filter from the calibration, or None if we can't build one.

    Two gates in principle: the pitch polygon (crowd, dugout, bench, adjacent
    pitches) and the expected player height at each ground point. ONLY THE
    POLYGON SHIPS: the height gate lost on both a bad and a good calibration and
    is kept off by default so `gates` can re-run the experiments.
    docs/reference/calibration.md#scene-prior (evidence: #evidence-scene-prior)

    Returns None rather than raising: a bad or missing calibration must degrade
    to "no filtering", never to "no tracking".
    """
    # A calibration that knows its own outline in the image supplies it. A spherical camera's H is a
    # stand-in that is metres out on a panorama (calibrators.SphericalCamera); a polygon drawn from it
    # would cut real players off the far side, which is what the clicked homography did before it.
    if hasattr(cal, "pitch_polygon_px"):
        if "pitch" not in gates:
            return None                          # gates=() means what it says for this camera too
        # Same failure as the off-frame clicks below: an outline for another frame size empties the run.
        cwh = tuple(int(v) for v in getattr(cal, "image_wh", image_wh))
        if cwh != tuple(int(v) for v in image_wh):
            print(f"  WARNING calibration was fitted on a {cwh[0]}x{cwh[1]} frame, not this "
                  f"{image_wh[0]}x{image_wh[1]} one; running unfiltered. Re-calibrate against THIS video.", flush=True)
            return None
        own =np.asarray(cal.pitch_polygon_px(), dtype=np.float32)
        if len(own) < 8:
            print("  scene prior unavailable (the calibration projects no pitch outline); running unfiltered", flush=True)
            return None
        if "height" in gates:
            print("  scene prior: the height gate needs a pinhole camera and is skipped for this calibration", flush=True)

        def keep_own(boxes: np.ndarray) -> np.ndarray:
            feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]])
            return np.array([cv2.pointPolygonTest(own, (float(u), float(v)), False) >= 0 for u, v in feet], bool)
        keep_own.gates = ("pitch",)
        return keep_own
    try:
        import sceneprior
        cam = sceneprior.camera_from_homography(cal.H, image_wh)
    except Exception as e:                       # noqa: BLE001 - never fatal
        print(f"  scene prior unavailable ({e}); running unfiltered", flush=True)
        return None

    # Clicks on a DIFFERENT frame size pass every check above, then both gates
    # reject every box and the run looks like "the detector found nothing".
    # Off-frame clicks are the unambiguous tell (calibrate.html cannot click
    # outside the image). calibration-evidence (not published)#evidence-scene-prior
    w, h = image_wh
    off = [n for n, (u, v) in getattr(cal, "points", {}).items()
           if not (0 <= u <= w and 0 <= v <= h)]
    if off:
        print(f"  WARNING calibration was clicked on a different frame size - "
              f"{len(off)} landmark(s) outside this {w}x{h} frame ({', '.join(sorted(off))}); "
              f"running unfiltered. Re-calibrate against THIS video.", flush=True)
        return None

    # yscale resamples one axis only, and sceneprior assumes SQUARE pixels; the
    # remap stays correct, only the prior derived from it is not.
    # calibration-evidence (not published)#evidence-scene-prior
    if float(getattr(cal, "yscale", 1.0)) != 1.0:
        print(f"  scene prior disabled: maps_yscale={cal.yscale:g} makes pixels "
              f"anisotropic and sceneprior assumes square ones; running unfiltered",
              flush=True)
        return None
    if not cam.mount_ok:
        print(f"  WARNING camera height {cam.height_m:.1f} m is below 6 m - stride "
              f"oscillation alone can break the 5% distance gate (groundpoint.py)",
              flush=True)
    poly = sceneprior.pitch_polygon(cam, cal).astype(np.float32)

    def keep(boxes: np.ndarray) -> np.ndarray:
        ok = np.ones(len(boxes), bool)
        if "pitch" in gates:
            feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]])
            ok &= np.array([cv2.pointPolygonTest(poly, (float(u), float(v)), False) >= 0
                            for u, v in feet], bool)
        if "height" in gates:
            ok &= sceneprior.plausible_height(cam, cal, boxes)
        return ok

    # The filter carries the gates it applies, so a run can RECORD what it used
    # instead of re-deriving it - a second derivation is a second copy, and a
    # copy of a gate set is how 0a5a7cd shipped the height gate for a day.
    keep.gates = tuple(gates)
    return keep


def track(frames: Iterator[np.ndarray], fps: float, detector, tracker,
          progress: int = 0, prior=None, cache: dict | None = None,
          checkpoint=None, checkpoint_every: int = 2000, stride: int = 1,
          ball: bool = False, roles=None, cal=None,
          ball_markings: bool = False, ball_detector=None,
          cache_rejected: dict | None = None) -> dict:
    """The whole detect-track-assign loop. Frames in, tracks.json payload out.

    `cache` is an optional detection cache, `{frame_index: [det, ...]}` holding
    what the detector AND the prior already agreed on. Hits are per FRAME, so a
    partial cache resumes rather than being ignored: frames present are replayed,
    frames absent are detected and added. `checkpoint(cache, n_frames)` is called
    every `checkpoint_every` newly-detected frames so a killed run loses minutes
    rather than hours. Frames are still decoded either way - the
    kit-colour crop needs pixels, and decode is nothing next to RF-DETR, which
    is 92-98% of wall clock (README). This exists because tuning the TRACKER
    used to cost a full detector pass per setting: measured 1.17 fps uncropped
    and 0.2 fps cropped, so a four-point sweep was hours instead of one pass.

    `stride` says only how the caller SPACED the frames it is handing over - it
    does no skipping itself (`_frames` does) - so that the rows come out
    numbered in the source video's own frames. `f = i * stride`, and see the
    module docstring for why they are not renumbered.

    `cache_rejected` ({frame: n}, beside `cache`) keeps each detected frame's prior
    rejections, so a replay reports the same prior_rejected as the run that detected it.
    """
    # Guarded here rather than in run(), because this is where the stride turns
    # into a frame NUMBER: at 0 every row would be filed under frame 0, and the
    # run would look plausible right up to the point where distance is infinite.
    if stride < 1:
        raise SystemExit(f"stride must be >= 1, got {stride}")
    dets_out: list[dict] = []
    classes: dict[int, Counter] = {}
    ball_out: list[dict] = []
    # The calibration is handed to the picker: its on-the-pitch and physical-size gates need it.
    # ball-evidence (not published)#evidence-gates
    picker = _ball.BallPicker(cal=cal, marking_gate=ball_markings) if ball else None
    # Appearance votes, kept SEPARATE from the detector's own `classes`: a COCO
    # detector says "person" for everyone, so without this there is nothing to
    # vote on. rolenet.vote() then decides which tracks it is sure enough about.
    role_votes: dict[int, Counter] = {}
    colours: dict[int, list[np.ndarray]] = {}
    rejected = 0
    n_new = 0
    t0 = time.perf_counter()
    f = -1
    role_crops: list = []
    role_ids: list[int] = []
    for i, frame in enumerate(frames):
        # ORIGINAL video frame index, never a renumbering. Module docstring.
        f = i * stride
        # Per-frame, not per-run. A whole-run `replay` flag meant a cache was
        # all or nothing, so a job that died at frame 6,000 of 72,101 - which is
        # what happened on 2026-08-04 - threw away nine hours of detection and
        # had to start over. With a per-frame hit, a partial cache resumes.
        hit = cache is not None and f in cache
        if hit:
            # Already past the prior when it was cached; re-filtering would
            # double-count `rejected` and, if the calibration changed since,
            # silently mix two geometries in one run.
            dets = [dict(d) for d in cache[f]]
            for d in dets:
                d["xyxy"] = tuple(d["xyxy"])
            rejected += (cache_rejected or {}).get(f, 0)
        else:
            if detector is None:
                # Reachable one way: a cache written `complete` over N frames,
                # then replayed against a longer range. run() skips building the
                # detector on a complete cache, so frame N has nothing to detect
                # with. Say that, rather than dying on NoneType.detect or - far
                # worse - returning empty frames that look like a quiet match.
                raise SystemExit(
                    f"frame {f} is not in the detection cache and no detector was "
                    f"built, because the cache is marked complete. It covers "
                    f"{len(cache)} frames; you asked for more. Re-run without "
                    f"--dets-cache, or delete the cache and rebuild it.")
            dets = detector.detect(frame)
            if ball_detector is not None:
                # A SEPARATE, FINE-TUNED BALL DETECTOR, merged into `dets` rather than
                # handled beside them. Everything below already splits ball candidates
                # out by class, so contributing here means the split, the picker and -
                # the reason this is the right line - the DETECTION CACHE all work
                # unchanged: the cache stays "what the detectors produced", so a replay
                # still has the ball. Its boxes come from a different tile grid (640 vs
                # 1024) and a different input square, which is why it is its own pass.
                dets = dets + [d for d in ball_detector.detect(frame)
                               if d.get("cls") in detectors.BALL_CLASSES]
        # Ball candidates leave the player path HERE, before both the prior and
        # the tracker. The prior's height gate is a person gate - a 9 px ball is
        # never a plausible player - and the tracker would spend an id per
        # candidate on an object there is exactly one of (see ball.py). The
        # split is unconditional: a cache written by a --ball run replayed
        # without one would otherwise feed ball boxes into association.
        cands = [d for d in dets if d.get("cls") in detectors.BALL_CLASSES]
        if cands:
            dets = [d for d in dets if d.get("cls") not in detectors.BALL_CLASSES]
        if picker is not None:
            b = picker.update(cands, frame.shape[0], f, frame)
            if b.seen:
                # Box CENTRE, not the feet: a ball in flight is not on the
                # ground and its bottom edge means nothing (ball.Ball).
                ball_out.append({"f": f, "x": float(b.xy[0]), "y": float(b.xy[1])})
        if not hit and prior is not None and dets:
            boxes = np.array([d["xyxy"] for d in dets], dtype=np.float64)
            keep = prior(boxes)
            rejected += int((~keep).sum())
            if cache_rejected is not None and not keep.all():
                cache_rejected[f] = int((~keep).sum())
            # Filter BEFORE the tracker, not after: a crowd detection that
            # reaches the tracker can steal an association from a real player,
            # and no downstream filter can undo an identity that was taken.
            dets = [d for d, k in zip(dets, keep) if k]
        if cache is not None and not hit:
            n_new += 1
            # The ball candidates go in too (unfiltered - the prior is a person gate),
            # or a resumed job replays those frames with no ball; a hit splits them
            # back out above. Caches written before 2026-09-28 simply have none.
            cache[f] = [{"xyxy": tuple(float(v) for v in d["xyxy"]),
                         "conf": float(d.get("conf", 1.0)),
                         "cls": d.get("cls", "person")} for d in cands + dets]
            # Checkpoint. The detector is 92-98% of wall clock, so this is the
            # only work worth protecting; the tracker replays from cache in
            # minutes. Writing the whole dict each time is O(n) but n is ~78 MB
            # at a full half and this fires ~35 times in nine hours.
            # Counted in frames DETECTED, not in frame indices: at stride 2
            # f + 1 is always odd, so an index test would never fire and a
            # nine-hour run would checkpoint exactly never.
            if checkpoint and checkpoint_every and n_new % checkpoint_every == 0:
                checkpoint(cache, f + 1)
        for d in tracker.update(dets, frame):
            fx1, fy1, fx2, fy2 = (float(v) for v in d["xyxy"])
            tid = int(d["id"])
            # Ints are needed to slice a crop; they must NEVER reach the ground
            # point. Truncating there adds 0.289 px of quantisation noise and a
            # -0.5 px bias on the vertical axis - the axis the camera's grazing
            # angle amplifies to 0.45-2.07 m per pixel. Measured cost through
            # stats.py: +6.5 to +21.3 percentage points of distance error.
            colours.setdefault(tid, []).append(
                team_by_shirt(frame, (int(fx1), int(fy1), int(fx2), int(fy2))))
            # The detector's own word on WHAT this is, voted per track. Inert on
            # a COCO checkpoint (everyone is "person"); the one signal that
            # separates a referee from a goalkeeper on a class-aware one.
            classes.setdefault(tid, Counter())[d.get("cls", "person")] += 1
            if roles is not None:
                c = _rolenet.crop(frame, (fx1, fy1, fx2, fy2))
                if c is not None:
                    role_crops.append(c)
                    role_ids.append(tid)
            dets_out.append({
                "f": f, "id": tid,
                # Feet, not centroid: the homography maps the ground plane, and
                # a player's centroid floats above it by half their height.
                "x": (fx1 + fx2) / 2, "y": fy2,
                # The box SIZE, kept at 1 decimal. Nothing measures with it -
                # every metre comes from (x, y) through the homography - but an
                # overlay that sizes its marker off a constant is unreadable at
                # both ends of a wide shot, and re-detecting a half to recover
                # two floats per row is nine hours. ~14% on the file.
                "w": round(fx2 - fx1, 1), "h": round(fy2 - fy1, 1),
            })
        # One batch per FRAME: ~24 crops of 48x96 is a millisecond next to
        # RF-DETR, and batching across frames would mean holding pixels for a
        # whole video to save nothing measurable.
        if roles is not None and role_crops:
            for tid, (nm, _) in zip(role_ids, roles(role_crops)):
                role_votes.setdefault(tid, Counter())[nm] += 1
            role_crops, role_ids = [], []

        if progress and f and f % progress == 0:
            print(f"  frame {f}  {(f + 1) / (time.perf_counter() - t0):.1f} fps", flush=True)

    # The per-detection colours go in whole. Averaging here is what produced a
    # 7.0/17.0 split on real footage where the truth is 11/11 - see assign_teams.
    # The appearance model OVERRIDES the detector's own names where it has a
    # confident majority, because on the shipped COCO checkpoint those names are
    # all "person" and carry no information. A class-aware detector's names win
    # where it has none - rolenet abstains far more often than it speaks.
    named = {tid: Counter({nm: 1}) for tid, nm in _rolenet.vote(role_votes).items()}
    teams = assign_teams(colours, {**classes, **named})
    for d in dets_out:
        # "unknown", NOT "home": assign_teams drops any track whose mean kit
        # colour is all-zero (an empty torso crop - a box clipped to the frame
        # edge, or one that got no pixels), and defaulting those to home made
        # ONE bank absorb every classification failure. Measured on the 2k
        # slice: 96.1% of the away bank was really away, but only 69.4% of the
        # home bank was really home. frames.py :: team_frames drops "unknown", so
        # the failure costs detections instead of corrupting a squad.
        d["team"] = teams.get(d["id"], "unknown")
        d["jersey"] = None  # jersey OCR is a separate stage; see README

    elapsed = time.perf_counter() - t0
    # `frames` and `throughput_fps` are counted in SOURCE frames, so throughput
    # stays "how fast do we get through the video" and a stride shows up as the
    # speed-up it is. `sampled` is what the detector actually saw.
    return {"fps": float(fps), "detections": dets_out, "ball": ball_out,
            "_perf": {"frames": f + 1, "seconds": round(elapsed, 2),
                      "throughput_fps": round((f + 1) / elapsed, 2) if elapsed else 0.0,
                      "prior_rejected": rejected,
                      "stride": stride, "sampled": (f // stride) + 1 if f >= 0 else 0}}


def _frames(cap: cv2.VideoCapture, max_frames: int | None,
            stride: int = 1) -> Iterator[np.ndarray]:
    """Every `stride`-th frame. `max_frames` counts SOURCE frames either way, so
    a stride changes how densely a run samples the video, never how much of it
    it covers. Skipped frames are grabbed, not decoded - `grab` skips the colour
    conversion, which is the only part of decode worth saving here."""
    n = 0
    while max_frames is None or n < max_frames:
        ok, frame = cap.read()
        if not ok:
            return
        n += 1
        yield frame
        for _ in range(stride - 1):
            if max_frames is not None and n >= max_frames:
                return
            if not cap.grab():
                return
            n += 1


def stitch_tracks(rows: list[dict], cal, fps: float) -> list[dict]:
    """Join the fragments association left behind, then re-vote each team.

    The team label gates the merge - two components confidently on opposite
    banks are not one player - and is then RE-DERIVED from the merged component,
    because stats.py :: player_stats hands a whole track the team of its first row and
    that first fragment is no more authoritative than any other.
    """
    import stitch as _st  # noqa: PLC0415

    # SEVER first, then join: stitching a track that already holds two players
    # spreads the error. The split and the team gate limit the purity cost of
    # stitching; they do not remove it. docs/reference/detection-tracking.md#stitch-trade
    rows = _st.split(rows, cal, fps=fps)
    labels = {int(d["id"]): d["team"] for d in rows}
    rows = _st.apply(rows, _st.stitch(_st.tracklets(rows, cal), fps, teams=labels))
    votes: dict[int, Counter] = {}
    for d in rows:
        votes.setdefault(int(d["id"]), Counter())[d["team"]] += 1
    for d in rows:
        d["team"] = votes[int(d["id"])].most_common(1)[0][0]
    return rows


def _calib_digest(path: str | None) -> str | None:
    """Content hash of the calibration file, or None if there isn't one.

    The cache holds boxes the scene prior ALREADY filtered (see track(): a hit
    is not re-filtered), so replaying it under a different calibration mixes two
    geometries in one run and nothing downstream can tell. Hashing the CONTENT
    rather than comparing the path is the whole point: on the job path the new
    calibration is written over the old one at the same path, so a path
    comparison would never notice.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


#: Cache settings keys introduced after caches were on disk (see run()).
LATE_KEYS = ("model_sha256",)
#: The detection cache's format, stamped in its settings. 2 = ball candidates are stored with the
#: boxes (17022d4, 2026-09-28). A cache without it was written before that and holds NO ball
#: candidates, so a ball run refuses it (it would replay ballless, in silence); a run without the
#: ball has nothing to lose and replays it as a late key - the committed track fixture is one.
CACHE_VERSION = 2


def _settings_mismatch(was: dict | None, now: dict, late: tuple = LATE_KEYS) -> list[str]:
    """Every detection setting a cache recorded that this run does not share.

    Same reason as _calib_digest, for the settings instead of the geometry: a
    hit replays its boxes as they are, so the gates they already passed, the
    threshold, the tiling and the detector all decide what this run sees. Found
    on the overlap - every cache written before 8472cef was tiled at 0.10, and a
    run at the shipped 0.25 would have replayed one without a word. Strict on
    purpose: a key the cache never recorded counts as a difference, not a pass.
    `was` None is a cache that predates the check, and run() writes its tracks
    unstamped rather than guessing what it was detected under.
    """
    if was is None:
        return []
    # `k not in was`, not `was.get(k) != now[k]`: the second reads a key the cache
    # never recorded as None and so ACCEPTS it whenever this run's value is None
    # too - prior_gates on an uncalibrated run - which is the opposite of strict.
    was = {**{k: None for k in late}, **was}     # run() has set these None in `now` too
    return [f"{k} {was[k]!r} in the cache, {now[k]!r} in this run" if k in was
            else f"{k} not recorded in the cache, {now[k]!r} in this run"
            for k in now if k not in was or was[k] != now[k]]


def run(video: str, out: str, detector: str = "rfdetr", tracker: str = "best",
        conf: float = 0.5, weights: str | None = None, resolution: int | None = None,
        max_frames: int | None = None, start: int = 0, variant: str = "medium",
        calib: str | None = None, remap: str | None = None,
        remap_yscale: float = 1.0, dets_cache: str | None = None,
        stride: int = 1, ball: bool = False, roles: bool = False,
        ball_markings: bool = False, stitch: bool = True,
        prior_gates: tuple | None = None, ball_model: str | None = None) -> dict:
    """Detect, track, label teams and stitch one video; write `out` (tracks.json).

    `conf` is the player detection threshold, a model score in [0, 1]; the ball
    uses min(conf, detectors.BALL_CONF). `stride` detects every Nth frame and
    keeps original frame numbers. `calib` enables the scene prior and stitching.
    `prior_gates`: None defers to make_prior's default gates; () applies no gate
    (the calibration is still used to flatten and stitch); a tuple names the
    gates. `dets_cache` replays or resumes detections, and refuses one detected
    under other settings. `tracker="best"` is resolved, and stamped as
    `tracker_resolved`.

    Writes {"fps", "fps_source", "detections": [{f, id, x, y, w, h, team, jersey}], "ball"
    (only when non-empty), "_perf"} to `out`, and returns the in-memory result
    (same keys; "ball" always present). `fps_source` is "container", or "fallback"
    when the container reported no usable fps (a WARNING is printed).
    Contract: docs/reference/detection-tracking.md#contract
    """
    import calibrators  # noqa: PLC0415

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    # A container that reports no usable fps runs at 25, and says so in tracks.json.
    # docs/reference/detection-tracking.md#contract
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_source = "container" if fps > 0 else "fallback"   # NaN > 0 is False
    if fps_source == "fallback":
        fps = 25.0
        print(f"WARNING: {video} reports no fps; timing assumes the fallback {fps}", flush=True)
    wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    # calibrators.load, not pitch.Calibration.load: the latter REFUSES anything
    # but a plain perspective file, which is exactly what a panorama rig cannot
    # produce. Everything this function needs off a calibration - .H for the
    # scene prior, .pitch - every Calibrator exposes.
    cal = calibrators.load(calib) if calib else None

    # Flatten BEFORE detection. A stitched panorama is not a plane, so no
    # homography undoes it and no downstream stage can compensate: on the real
    # SoccerTrack panorama a single homography fitted straight to the raw frame
    # is 6.2 m off at the median against the dataset's own 65 keypoints, and
    # 2.4 m after the remap (calibrators._measure_real). Detection, the pitch
    # polygon, the height prior and the homography all assume perspective, so
    # this is the one place the correction belongs.
    flatten = None
    if remap:
        # Two sources of truth for the canvas is a silent metres-scale error,
        # not a warning: the calibration's points were measured on ITS canvas.
        # The crop is the same trap with a bigger lever - a calibration solved
        # on the pitch-band crop is ~300 rows and 2.9x, and projecting vendor
        # canvas pixels through it puts every player off the far end of the map.
        if isinstance(cal, calibrators.RemapCalibrator) and (
                cal.yscale != remap_yscale or cal.crop is not None or cal.scale != 1.0):
            raise SystemExit(
                f"--remap disagrees with {calib}, which was solved at yscale "
                f"{cal.yscale}, crop {cal.crop}, scale {cal.scale}; drop --remap "
                f"and let the calibration supply its own tables")
        mx, my = calibrators.load_maps(remap, remap_yscale)
        flatten = lambda f: cv2.remap(f, mx, my, cv2.INTER_LINEAR)  # noqa: E731
    elif isinstance(cal, calibrators.RemapCalibrator):
        # A remap calibration carries the path to its own tables AND the yscale
        # they were solved at, so --calib alone is enough; needing both flags is
        # how you end up projecting raw-panorama pixels through a
        # remapped-space homography, or through one solved on a 3x taller canvas.
        flatten = cal.undistort_frame
    if flatten is not None:
        # The prior and the calibration both live in REMAPPED space now, so the
        # frame size they are checked against must be the remapped one. Probing
        # with one blank frame also fails fast on missing tables - the
        # alternative is discovering it 70000 frames in.
        try:
            wh = flatten(np.zeros((wh[1], wh[0], 3), np.uint8)).shape[1::-1]
        except ValueError as e:
            raise SystemExit(f"{calib}: {e}; pass --remap <dir>") from e
        print(f"  remap active: frames flattened to {wh[0]}x{wh[1]} before detection",
              flush=True)

    # prior_gates reaches make_prior so a run can keep the calibration (which a
    # remap needs to flatten the frame at all) and still detect WITHOUT a gate.
    # model/validation/st2_product.py (N19/N20) is why the seam exists.
    #
    # None means DEFER, and it has to: this argument used to default to
    # ("pitch", "height") and pass it on explicitly, which made make_prior's own
    # default unreachable from app/server/jobs.py and left the height gate shipping
    # for a day after 0a5a7cd retired it. One default, in one place, and the
    # seam carries no opinion of its own. `()` still means no gate at all, which
    # is why this tests `is None` and not truthiness.
    if cal is None:
        prior = None
    elif prior_gates is None:
        prior = make_prior(cal, wh)
    else:
        prior = make_prior(cal, wh, prior_gates)

    dkw: dict = {}
    if detector in ("rfdetr", "cuda"):
        # cuda is RFDETRDetector(device="cuda") - same constructor, so it takes
        # the same knobs. Without it --detector cuda --conf 0.25 ran at 0.5.
        dkw = {"conf": conf, "variant": variant, "ball": ball}
        dkw.update({k: v for k, v in (("weights", weights), ("resolution", resolution)) if v})
    elif detector == "tensorrt":
        # A baked engine has its variant and input square fixed at build time,
        # so only the threshold means anything here.
        dkw = {"conf": conf, "ball": ball}
    elif detector == "coreml":
        # The .mlpackage is exported at ONE variant and ONE input square, so
        # --variant and --resolution have nowhere to go; passing them would be a
        # TypeError rather than a silent no-op, which is the right failure but a
        # confusing one. --conf and -w (the model bundle) are the two that mean
        # something here.
        dkw = {"conf": conf, "ball": ball}
        if weights:
            dkw["weights"] = weights
    frames = _frames(cap, max_frames, stride)
    if flatten is not None:
        frames = (flatten(f) for f in frames)
    # The settings the PRODUCT varies that decide which boxes a detection pass
    # produces - written into the cache header and, with stride and stitch, into
    # the tracks' stamp, so the two cannot describe different runs. Stride is not
    # one: a cache is keyed by the original frame index and replays at any stride.
    # NOT recorded, so NOT guarded: weights, variant, resolution, remap and
    # remap_yscale. app/server/jobs.py passes none of them; a research run that varies
    # one must give it its own --dets-cache. `tile` is the default of the constructor
    # this detector name builds (None = untiled). `model_sha256` is the weights'
    # identity (detectors.model_identity, the one check.py detect compares).
    # THE FINE-TUNED BALL DETECTOR, opt-in and OFF by default. `ball_model` is a path -
    # detectors.BALL_MODEL is where the product would find one - and asking for it moves
    # the ball onto arm D1 of N52 at the geometry it was trained in. The PRIMARY detector
    # then stops contributing ball boxes, so the two do not both answer: `ball` still
    # gates the picker, but `dkw["ball"]` goes False. Its sha256 goes into the cache
    # settings (`ball_model`), so a cache replays only under the ball model that made it.
    #
    # Not the default because of its cost: a second per-frame pass at a smaller tile
    # is several times the detection bill. detection-tracking-evidence (not published)#evidence-ball
    ball_sha = None
    if ball and ball_model:
        if not os.path.exists(ball_model):
            raise SystemExit(f"no ball model at {ball_model} - it is distributed like the "
                             f"CoreML bundle (gitignored, or FT_BALL_MODEL)")
        # THE LICENCE GATE, before the weights load. Arm D1 (TeamTrack, MIT) and arm B (DFL,
        # competition use only) are both a 117 MB checkpoint_best_ema.pth, so the path cannot
        # say which one is there. Refuse anything but the one that may ship.
        _h = hashlib.sha256()
        with open(ball_model, "rb") as _fh:
            for _c in iter(lambda: _fh.read(1 << 20), b""):
                _h.update(_c)
        if _h.hexdigest() != detectors.BALL_MODEL_SHA256:
            raise SystemExit(
                f"the ball model at {ball_model} hashes {_h.hexdigest()[:16]}..., not the "
                f"{detectors.BALL_MODEL_SHA256[:16]}... this product may ship. N52 arm D1 is "
                f"TeamTrack/MIT; arm B is DFL, COMPETITION USE ONLY and must never ship, and "
                f"the two are indistinguishable by filename.")
        ball_sha = _h.hexdigest()
    ident = detectors.model_identity(detector, weights, variant)
    settings = {
        "prior_gates": list(prior.gates) if prior is not None else None,
        "conf": conf, "detector": detector, "ball": ball,
        "overlap": detectors.OVERLAP,
        "tile": detectors.tile_of(detector),
        "model_sha256": ident and ident["sha256"],
        "cache_version": CACHE_VERSION,
        # the fine-tuned ball detector's sha256, None when the primary answers for the ball
        "ball_model": ball_sha,
    }
    # The detection cache. Loaded, the detector is never constructed - which is
    # the point: get_detector("rfdetr") alone builds and traces the model.
    cached: dict = {}
    cached_rej: dict = {}
    complete = False
    stamped = True
    if dets_cache and os.path.exists(dets_cache):
        with open(dets_cache) as fh:
            raw = json.load(fh)
        if raw.get("start") != start or raw.get("video") != os.path.basename(video):
            raise SystemExit(
                f"{dets_cache} was built from {raw.get('video')} at start "
                f"{raw.get('start')}, not {os.path.basename(video)} at {start}; "
                f"delete it or pass a different --dets-cache")
        # A missing calib_sha is a cache written before this check existed;
        # accept it rather than throwing away hours of detection already on disk.
        was, now = raw.get("calib_sha"), _calib_digest(calib)
        if was is not None and was != now:
            raise SystemExit(
                f"{dets_cache} was built behind a DIFFERENT {raw.get('calib')} "
                f"({was}, now {now}). Its boxes are already filtered by that "
                f"calibration's scene prior, so replaying them here would mix two "
                f"geometries silently. Delete it and re-detect.")
        # A ball run needs a cache that stored ball candidates (CACHE_VERSION): one with no
        # settings at all is older still, and would replay with no ball on any frame.
        if ball and raw.get("settings") is None:
            raise SystemExit(
                f"{dets_cache} predates recorded settings, so it predates stored ball candidates "
                f"(cache_version {CACHE_VERSION}); a ball run would replay it with no ball. "
                f"Delete it and re-detect.")
        # Keys added after caches already existed: one that never recorded it replays
        # (like a missing calib_sha) and the run stamps it None - unknown, not guessed.
        # `ball_model` is late only while this run uses none: a cache from before it was
        # recorded holds the PRIMARY detector's ball boxes, never a ball model's.
        late = LATE_KEYS + (() if ball else ("cache_version",)) + (("ball_model",) if ball_sha is None else ())
        for k in late:
            if raw.get("settings") is not None and raw["settings"].get(k) is None:
                settings[k] = None
        bad = _settings_mismatch(raw.get("settings"), settings, late)
        if bad:
            raise SystemExit(
                f"{dets_cache} was detected under different settings: "
                f"{'; '.join(bad)}. Its boxes replay as they are, so this run "
                f"would silently be that configuration. Delete it and re-detect, "
                f"or pass a different --dets-cache.")
        # Replayable - discarding hours of detection is the worse failure - but
        # nothing may claim to know what it was detected under, including a
        # resume that appends to it.
        stamped = raw.get("settings") is not None
        if not stamped:
            print(f"  detection cache: {dets_cache} predates recorded settings "
                  f"(overlap, conf, gates unknown) - tracks written UNSTAMPED", flush=True)
        cached = {int(k): v for k, v in raw["frames"].items()}
        cached_rej = {int(k): v for k, v in raw.get("rejected", {}).items()}
        complete = bool(raw.get("complete"))
        print(f"  detection cache: {'replaying' if complete else 'RESUMING from'} "
              f"{len(cached)} frames in {dets_cache}", flush=True)

    def _write_cache(c: dict, done: int, final: bool = False) -> None:
        # tmp-then-rename: a torn write on a partial cache would destroy the
        # very thing this exists to protect. Same idiom as jobs.py's state I/O.
        tmp = f"{dets_cache}.tmp"
        with open(tmp, "w") as fh:
            json.dump({"video": os.path.basename(video), "start": start,
                       "calib": calib, "calib_sha": _calib_digest(calib),
                       "settings": settings if stamped else None,
                       "complete": final, "rejected": cached_rej, "frames": c}, fh)
        os.replace(tmp, dets_cache)
        print(f"  detection cache: {'wrote' if final else 'checkpoint'} "
              f"{len(c)} frames ({done} seen) -> {dets_cache}", flush=True)

    # `complete` and not merely `cached`: a PARTIAL cache still needs the
    # detector for the frames it is missing, and get_detector() alone builds and
    # traces the model, so skipping it on a resume would crash at the first gap.
    # Built ONCE, after the ball-model decision: only one detector answers for the ball.
    if ball_sha is not None and "ball" in dkw:
        dkw["ball"] = False
    det = None if complete else get_detector(detector, **dkw)
    # The ball detector likewise: a complete cache already holds its boxes.
    ball_det = None if complete or ball_sha is None else get_detector(
        "rfdetr", conf=conf, ball=True, variant=detectors.BALL_VARIANT, tile=detectors.BALL_TILE,
        resolution=detectors.BALL_RESOLUTION, weights=ball_model)
    tracker = best_available() if tracker == "best" else tracker     # stamped as what ran
    result = track(frames, fps, det, get_tracker(tracker), ball_detector=ball_det,
                   progress=200, prior=prior,
                   cache=cached if dets_cache else None,
                   cache_rejected=cached_rej if dets_cache else None,
                   checkpoint=(_write_cache if dets_cache and not complete else None),
                   stride=stride, ball=ball, cal=cal, ball_markings=ball_markings,
                   roles=_rolenet.RoleNet() if roles else None)
    cap.release()

    if dets_cache and not complete:
        _write_cache(cached, len(cached), final=True)

    # Written explicitly, not by dumping `result`: tracks.json is a contract
    # stats.py and the web layer both read, and _perf is ours alone.
    # ponytail: whole match held in RAM and written as one JSON blob. Ceiling is
    # ~135k frames x ~20 players = a few GB and a ~400MB file; upgrade path is
    # streaming JSONL, but team assignment needs every track's mean colour
    # first, so it would need a second pass. Not worth it until it hurts.
    # STITCH. Joins the fragments online association left, gated by team label. It
    # needs a calibration (the cost is metres), so an uncalibrated run ships
    # unstitched. `stitch=False` is stamped, so an unstitched artefact can never pass
    # for the product. docs/reference/detection-tracking.md#stitch
    if stitch and cal is not None and result["detections"]:
        before = len({int(d["id"]) for d in result["detections"]})
        result["detections"] = stitch_tracks(result["detections"], cal, float(result["fps"]))
        print(f"stitched {before} tracks -> "
              f"{len({int(d['id']) for d in result['detections']})}")

    p = result["_perf"]
    # What this run actually used, stamped next to the tracks so any artefact can
    # be diffed against what ships (model/validation/shipped_config.py). prior_gates is
    # read off the filter object, not recomputed, and is None when make_prior
    # declined to build one - which is the case app/server/jobs.py's own comment says
    # is otherwise indistinguishable from a filter that is working.
    # No stamp at all off a cache that never recorded its settings: this run's
    # settings are not the ones its boxes were detected under, and shipped_config
    # reads a missing stamp as UNSTAMPED but believes a present one.
    if stamped:
        p["config"] = {**settings, "stride": stride, "stitch": stitch, "tracker_resolved": tracker}
    with open(out, "w") as fh:
        # _perf ships WITH the tracks. It used to be built and thrown away, which
        # meant throughput, wall clock and - the one that matters -
        # prior_rejected could never be shown for any run. prior_rejected is how
        # you tell "the geometry filter is working" from "the geometry filter
        # silently failed to load", and those look identical without it.
        # Consumers read {fps, detections}; an extra key is additive.
        # `ball` only when there is one: a run without --ball writes exactly the
        # file it always did, so nothing downstream sees a new empty key.
        json.dump({"fps": result["fps"], "fps_source": fps_source, "detections": result["detections"],
                   **({"ball": result["ball"]} if result["ball"] else {}),
                   "_perf": p}, fh)
    result["fps_source"] = fps_source
    ids = {d["id"] for d in result["detections"]}
    if ball:
        print(f"ball on {len(result['ball'])}/{p['frames']} frames - see ball.py on why "
              f"'a ball' is not 'the ball'")
    print(f"{len(result['detections'])} detections, {len(ids)} tracks over {p['frames']} frames "
          f"@ {p['throughput_fps']} fps -> {out}")
    return result


def _selfcheck() -> None:
    """StubDetector + IoUTracker end to end. Guards the output contract that
    stats.py and the web layer both read - if any of these change, they break."""
    from detectors import StubDetector
    from tracking import IoUTracker

    # Two red shirts, two blue, so kit-colour clustering has something to find.
    # The painter runs the same schedule as the detector under test, so the
    # pixels really are where the boxes say they are.
    painter = StubDetector(n=4)
    frames, boxes_per_frame = [], []
    for _ in range(30):
        img = np.full((360, 640, 3), 40, np.uint8)
        boxes = painter.detect(img)
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = (int(v) for v in b["xyxy"])
            img[y1:y2, x1:x2] = (200, 40, 40) if i < 2 else (40, 40, 200)
        frames.append(img)
        boxes_per_frame.append(boxes)

    r = track(iter(frames), 25.0, StubDetector(n=4), IoUTracker())
    d = r["detections"]

    assert set(r) >= {"fps", "detections"} and isinstance(r["fps"], float)
    assert d, "no detections"
    for x in d:
        assert set(x) == {"f", "id", "team", "jersey", "x", "y", "w", "h"}, sorted(x)
        assert isinstance(x["f"], int) and isinstance(x["id"], int)
        assert x["team"] in ("home", "away") and x["jersey"] is None
        assert isinstance(x["x"], float) and isinstance(x["y"], float)
        assert x["w"] > 0 and x["h"] > 0, x

    # Ids stable: 4 players seen for 30 frames must be 4 tracks, not 120.
    ids = {x["id"] for x in d}
    assert len(ids) == 4, f"{len(ids)} ids for 4 stub players - association is broken"
    assert len(d) == 4 * len(frames), len(d)

    # Feet, not centroid. The stub box is 24x60 at a known place on frame 0;
    # y must be the BOTTOM edge. A centroid would read 30px higher, which is
    # ~2m of ground-plane error after the homography and silently wrong.
    b0 = boxes_per_frame[0][0]["xyxy"]
    first = min(d, key=lambda x: (x["f"], x["x"]))
    # w/h are the BOX, not a nominal size - an overlay sizes its marker off them.
    assert (first["w"], first["h"]) == (24.0, 60.0), (first["w"], first["h"])
    assert first["y"] == float(int(b0[3])), (first["y"], b0)
    assert first["y"] != (b0[1] + b0[3]) / 2, "y is the centroid, not the feet"
    assert first["x"] == float((int(b0[0]) + int(b0[2])) / 2), (first["x"], b0)

    # The ball leaves the player path before the tracker. It used to be
    # associated as a very small person, which spends an id per candidate on the
    # one object there is exactly one of.
    class _WithBall:
        def __init__(self):
            self.inner, self.n = StubDetector(n=2), 0

        def detect(self, frame):
            out = self.inner.detect(frame)
            out.append({"xyxy": (100.0 + self.n, 100.0, 110.0 + self.n, 110.0),
                        "conf": 0.9, "cls": "sports ball"})
            self.n += 1
            return out

    grey = [np.full((360, 640, 3), 40, np.uint8) for _ in range(10)]
    rb = track(iter(grey), 25.0, _WithBall(), IoUTracker(), ball=True)
    assert len(rb["ball"]) == 10, rb["ball"]
    assert len(rb["detections"]) == 2 * 10, len(rb["detections"])
    assert len({x["id"] for x in rb["detections"]}) == 2, "the ball became a track"
    # Centre of the box, and it moves with it.
    assert rb["ball"][0]["x"] == 105.0 and rb["ball"][1]["x"] == 106.0, rb["ball"][:2]
    # Without --ball the candidate is still kept out of association, and no
    # ball rows are written.
    off = track(iter(grey), 25.0, _WithBall(), IoUTracker())
    assert off["ball"] == [] and len({x["id"] for x in off["detections"]}) == 2

    # A SEPARATE BALL DETECTOR MUST ACTUALLY REACH THE PICKER, and its boxes must not
    # reach association. Plumbing that silently contributes nothing is the failure this
    # repo keeps finding: a promised path absent from the result
    # (internal note `docs-guard-must-have-a-runnable-check`). The primary here emits NO
    # ball, so every ball row below can only have come from the second detector.
    class _BallOnly:
        def __init__(self):
            self.n = 0

        def detect(self, frame):
            out = [{"xyxy": (200.0 + self.n, 200.0, 210.0 + self.n, 210.0),
                    "conf": 0.9, "cls": "sports ball"}]
            self.n += 1
            return out

    two = track(iter(grey), 25.0, StubDetector(n=2), IoUTracker(), ball=True,
                ball_detector=_BallOnly())
    assert len(two["ball"]) == 10, two["ball"]
    assert two["ball"][0]["x"] == 205.0 and two["ball"][1]["x"] == 206.0, two["ball"][:2]
    assert len({x["id"] for x in two["detections"]}) == 2, "the second detector's ball " \
                                                          "became a track"
    # and with no second detector the same primary writes NO ball at all, so the rows
    # above are not something the primary could have produced.
    none = track(iter(grey), 25.0, StubDetector(n=2), IoUTracker(), ball=True)
    assert none["ball"] == [], none["ball"]

    # Both kit colours must survive to a team label, or the referee/GK story
    # downstream has nothing to work with.
    assert {x["team"] for x in d} == {"home", "away"}

    # Referee separation, on Lab-ish colour vectors. A referee folded into a team
    # is a phantom high-distance profile - referees outrun most outfielders.
    rng = np.random.default_rng(0)
    def kit(base, n, start, obs=4):
        # A LIST per track: assign_teams clusters detections and votes per
        # track, so one mean colour per track is no longer its input. See the
        # measured 7.0/17.0 failure in that docstring.
        return {start + i: [np.array(base, float) + rng.normal(0, 3, 3)
                            for _ in range(obs)] for i in range(n)}
    crew = assign_teams({**kit((60, 180, 90), 11, 0),     # red-ish
                         **kit((60, 90, 180), 11, 100),   # blue-ish
                         **kit((200, 128, 200), 2, 200)}) # bright official kit
    assert sum(v == "official" for v in crew.values()) == 2, crew
    assert sum(v == "home" for v in crew.values()) == 11, crew
    assert sum(v == "away" for v in crew.values()) == 11, crew

    # No officials on the pitch: k=3 must NOT carve a team in half and delete it.
    both = assign_teams({**kit((60, 180, 90), 11, 0), **kit((60, 90, 180), 11, 100)})
    assert "official" not in both.values(), both
    assert sum(v == "home" for v in both.values()) == 11, both

    # Scene prior: must reject off-pitch and wrong-sized boxes, keep real ones,
    # and above all must never turn a bad calibration into a dead pipeline.
    import sceneprior
    from pitch import Calibration, Pitch

    W, H = 1920, 1080
    f_px, cam_h = 1400.0, 12.0
    C = np.array([52.5, -25.0, cam_h])
    fw = np.array([0.0, 25.0, -cam_h]); fw /= np.linalg.norm(fw)
    rt = np.cross(fw, [0, 0, 1.0]); rt /= np.linalg.norm(rt)
    Rm = np.column_stack([rt, np.cross(fw, rt), fw]).T
    Km = np.array([[f_px, 0, W / 2], [0, f_px, H / 2], [0, 0, 1.0]])
    Gt = (Km @ np.column_stack([Rm, -Rm @ C]))[:, [0, 1, 3]]
    cal = Calibration(pitch=Pitch(105.0, 68.0), H=np.linalg.inv(Gt / Gt[2, 2]))
    cam = sceneprior.camera_from_homography(cal.H, (W, H))
    keep = make_prior(cal, (W, H))

    centre = cv2.perspectiveTransform(
        np.array([[[52.5, 34.0]]]), Gt / Gt[2, 2]).reshape(2)
    ph = float(sceneprior.expected_height_px(cam, np.array([[52.5, 34.0]]))[0])
    good = [centre[0] - 6, centre[1] - ph, centre[0] + 6, centre[1]]
    boxes = np.array([
        good,                                        # a real player, right size
        [good[0], good[1] - ph * 1.6, good[2], good[3]],   # 2.6x too tall
        [good[0], good[1] + ph * 0.7, good[2], good[3]],   # 0.3x too short
        [5, 5, 17, 5 + ph],                          # top-left corner: off pitch
    ])
    # WHAT SHIPS: the polygon only. The wrong-SIZE boxes survive on purpose -
    # the height gate lost on both a bad and a good calibration (see
    # make_prior), and its cost was real players, not these.
    assert list(keep(boxes)) == [True, True, True, False], keep(boxes)
    # The gate still works where it is asked for, so the two experiments that
    # judged it are re-runnable and a future one can re-open the question.
    assert list(make_prior(cal, (W, H), ("pitch", "height"))(boxes)) == \
        [True, False, False, False]
    # (the corner box is off-pitch AND the wrong size there, so height alone
    #  rejects it too - it is not a discriminating case for that gate)
    assert list(make_prior(cal, (W, H), ("height",))(boxes)) == [True, False, False, False]

    # A broken calibration degrades to unfiltered, never to zero detections -
    # silently dropping every box would look exactly like "the detector found
    # nothing", which is the least debuggable failure this pipeline could have.
    # a calibration that knows its own outline supplies it: a spherical camera's H is a stand-in, and a
    # polygon drawn from it would cut players off the far side of a panorama
    import autocal as _ac  # noqa: PLC0415
    import calibrators as _cals  # noqa: PLC0415
    _P = Pitch(105.0, 68.0)
    _sph = _cals.SphericalCamera(_P, _ac.camera(_P, (4096, 1080), fov_deg=156.0, tilt_deg=19.0, height_m=7.5,
                                                behind_m=15.0, along_m=0.0, pan_deg=0.0), (4096, 1080))
    _keep = make_prior(_sph, (4096, 1080))
    _foot = lambda m: _sph.to_image(np.array([m], float))[0]  # noqa: E731
    _box = lambda m: [_foot(m)[0] - 8, _foot(m)[1] - 40, _foot(m)[0] + 8, _foot(m)[1], 1.0]  # noqa: E731
    assert _keep.gates == ("pitch",), _keep.gates
    assert _keep(np.array([_box([52.5, 34.0]), _box([2.0, 1.0]), _box([52.5, -12.0]), _box([52.5, 80.0])])).tolist() == \
        [True, True, False, False], "centre and far corner are on the pitch; the far stand and behind the camera side are not"
    assert make_prior(_sph, (1920, 1080)) is None, "a spherical outline for another frame size must run unfiltered"

    broken = Calibration(pitch=Pitch(105.0, 68.0), H=np.eye(3))
    # THE SEAM CARRIES NO OPINION. track.run used to default prior_gates to a
    # COPY of make_prior's gate set and pass it on explicitly, so moving
    # make_prior's default (0a5a7cd) changed nothing for app/server/jobs.py and the
    # height gate shipped for another day. lint_truth's DEFAULT TRAP cannot see
    # this: it reads ast.Constant and a tuple default is an ast.Tuple.
    import inspect  # noqa: PLC0415
    assert inspect.signature(run).parameters["prior_gates"].default is None, \
        "track.run must DEFER to make_prior's default, not carry a copy of it"
    # ...and the filter it builds says which gates it applies, so a run STAMPS
    # what it used rather than re-deriving it (model/validation/shipped_config.py).
    _mp = inspect.signature(make_prior).parameters["gates"].default
    assert make_prior(cal, (W, H)).gates == tuple(_mp)
    assert make_prior(cal, (W, H), ("pitch", "height")).gates == ("pitch", "height")
    assert make_prior(cal, (W, H), ()).gates == (), "() must stay 'no gate at all'"

    assert make_prior(broken, (W, H)) is None

    # ...and neither must a calibration clicked on a DIFFERENT frame size. This
    # is the one that actually shipped: it solves, it recovers a camera, and
    # then it rejects every box. Nothing else in this file catches it.
    assert make_prior(Calibration(pitch=Pitch(105.0, 68.0), H=cal.H,
                                  points={"corner_bl": (140.0, 2.0 * H)}), (W, H)) is None
    assert make_prior(Calibration(pitch=Pitch(105.0, 68.0), H=cal.H,
                                  points={"corner_bl": (140.0, H - 1.0)}), (W, H)) is not None

    # An anisotropically resampled remap canvas breaks sceneprior's square-pixel
    # assumption; the prior must stand down rather than reject real players.
    import calibrators  # noqa: PLC0415
    inner = calibrators.PerspectiveCalibrator(cal)
    assert make_prior(calibrators.RemapCalibrator(cal.pitch, None, None, inner,
                                                  yscale=3.0), (W, H)) is None
    assert make_prior(calibrators.RemapCalibrator(cal.pitch, None, None, inner,
                                                  yscale=1.0), (W, H)) is not None
    # ...but a pitch-band crop with an ISOTROPIC upscale must NOT stand it down.
    # That is the whole reason the crop exists rather than yscale: it buys the
    # same detail (24 -> 60 px players on the real panorama) and keeps square
    # pixels, so the prior stays on. A guard that fired here would silently
    # trade the geometry filter for the resolution.
    assert make_prior(calibrators.RemapCalibrator(cal.pitch, None, None, inner,
                                                  crop=(390, 690), scale=2.9), (W, H)) is not None

    _selfcheck_teams()
    _selfcheck_resume()
    _selfcheck_cache_replay()
    _selfcheck_stride()
    _selfcheck_remap()
    print("track.py ok")


def _selfcheck_teams() -> None:
    """A track that swapped players votes; it does not average into limbo.

    The regression this guards shipped for months and was invisible: kit colours
    were averaged per track, so a track that swapped between a red player and a
    blue one got PURPLE - a colour neither team wears, which k-means then filed
    under whichever team happened to be nearer. On real 118575 footage that
    turned a true 11/11 into 7.0/17.0 per frame, and the away side fell below
    stats.SHAPE_MIN_PLAYERS so half the shape series simply stopped existing.
    Nothing raised. Both teams still had players. It just quietly moved five of
    them to the wrong side.
    """
    red, blue = np.array([80.0, 180.0, 140.0]), np.array([120.0, 110.0, 80.0])

    def jit(c, n, s=3.0):
        rng = np.random.default_rng(0)
        return [c + rng.normal(0, s, 3) for _ in range(n)]

    # Eleven clean tracks a side, plus ONE track that swaps mid-way: 6 frames of
    # red then 6 of blue. Its mean is the midpoint and belongs to neither.
    colours = {}
    for i in range(11):
        colours[i] = jit(red, 20)
    for i in range(11, 22):
        colours[i] = jit(blue, 20)
    colours[99] = jit(red, 6) + jit(blue, 6)

    teams = assign_teams(colours)
    sides = {teams[i] for i in range(11)}, {teams[i] for i in range(11, 22)}
    assert len(sides[0]) == 1 and len(sides[1]) == 1, teams
    assert sides[0] != sides[1], "the two kits collapsed into one team"
    n = Counter(teams[i] for i in range(22))
    assert n["home"] == 11 and n["away"] == 11, n

    # The swapper is called by its majority, not by its blend. With a 6/6 tie
    # the vote is decided by the rarer label, but either team is defensible -
    # what is NOT defensible is it landing anywhere else.
    assert teams[99] in ("home", "away"), teams[99]

    # The mean of that track is ~(100,145,110): between the kits and close to
    # neither. This is the number that used to decide its team.
    m = np.mean(colours[99], axis=0)
    assert np.linalg.norm(m - red) > 40 and np.linalg.norm(m - blue) > 40, m

    # A lopsided swapper follows its majority, with no tie to break. Compare
    # WITHIN one call: "home" is whichever cluster is larger, so between two
    # calls with different inputs the two names can swap wholesale. That
    # instability is real and pre-existing - it is why the operator binds a side
    # once per match rather than the pipeline naming one - but it is not what
    # this assert is about.
    colours[99] = jit(red, 18) + jit(blue, 2)
    t = assign_teams(colours)
    assert t[99] == t[0], f"majority vote ignored: {t[99]} != {t[0]} (18 red, 2 blue)"
    colours[99] = jit(red, 2) + jit(blue, 18)
    t = assign_teams(colours)
    assert t[99] == t[11], f"majority vote ignored: {t[99]} != {t[11]} (2 red, 18 blue)"

    # Officials: a genuinely small, well-separated third kit is kept.
    yellow = np.array([200.0, 120.0, 200.0])
    colours = {i: jit(red, 20) for i in range(11)}
    colours.update({i: jit(blue, 20) for i in range(11, 22)})
    colours.update({i: jit(yellow, 20) for i in (90, 91, 92)})
    t3 = assign_teams(colours)
    assert sum(v == "official" for v in t3.values()) == 3, Counter(t3.values())
    assert sum(v == "home" for v in t3.values()) == 11, Counter(t3.values())

    # ...but a SPLIT team is not. Two shades of the same kit must not be read as
    # a crew, because accepting it deletes that half-side from the match.
    colours = {i: jit(red, 20) for i in range(11)}
    colours.update({i: jit(blue, 20) for i in range(11, 19)})
    colours.update({i: jit(blue + 6.0, 20) for i in range(19, 22)})
    t4 = assign_teams(colours)
    assert "official" not in t4.values(), Counter(t4.values())

    # ONE referee, in a kit no cluster is big enough to find. This is the DFL
    # case: the third cluster is spent on the touchline crowd, and a single
    # person in a fourth kit is left inside a team unless he is tested against
    # the teams' own spread. jit() puts team-mates within ~3 units of each
    # other, so the referee at 60+ units is far outside both.
    colours = {i: jit(red, 40) for i in range(11)}
    colours.update({i: jit(blue, 40) for i in range(11, 22)})
    colours.update({i: jit(np.array([70.0, 60.0, 200.0]), 40) for i in (80, 81, 82, 83)})  # crowd
    colours[99] = jit(yellow, 40)                                            # the referee
    t5 = assign_teams(colours)
    assert t5[99] == "official", f"the referee stayed a player: {t5[99]}"
    assert sum(v == "official" for v in t5.values()) == 5, Counter(t5.values())
    assert Counter(t5.values())["home"] == 11, Counter(t5.values())

    # A short track is NOT judged: a handful of detections is a mean with no
    # stability, and on 118575 the 1-5 detection fragments all look like
    # officials while being ordinary players in shadow.
    colours[98] = jit((red + blue) / 2, 4)
    t6 = assign_teams(colours)
    assert t6[98] != "official", "a 4-detection fragment was called an official"

    # A DETECTOR'S NAME beats both of them, in both directions. This is inert on
    # the COCO checkpoint that ships (everyone is "person") and is the whole
    # point of the class-aware path: colour puts a keeper and a referee in the
    # same bucket, and only a name separates them.
    cls_votes = {99: Counter({"player": 40}),          # colour says official
                 80: Counter({"referee": 40})}         # colour says a team
    t7 = assign_teams(colours, cls_votes)
    assert t7[99] in ("home", "away"), f"the name did not rescue the player: {t7[99]}"
    assert t7[80] == "official", f"the name did not catch the referee: {t7[80]}"
    # ...and a checkpoint that only ever says "person" changes nothing at all.
    same = assign_teams(colours, {i: Counter({"person": 9}) for i in colours})
    assert same == assign_teams(colours), "a COCO checkpoint changed the labels"


def _selfcheck_cache_replay() -> None:
    """A full cache replay gives the fresh run's ball rows and prior_rejected.
    Both were lost before 2026-09-28: ball boxes never reached the cache (6 ball rows
    fresh, 0 replayed) and rejections were counted on misses only, so a resumed job
    read 0 and said its working prior "may not have loaded"."""
    from tracking import IoUTracker  # noqa: PLC0415

    class D:
        def detect(self, fr):
            return [{"xyxy": (100, 100, 130, 180), "conf": .9, "cls": "person"},
                    {"xyxy": (1100, 100, 1130, 180), "conf": .9, "cls": "person"},   # prior drops it
                    {"xyxy": (300, 300, 309, 309), "conf": .6, "cls": "sports ball"}]

    def prior(b):
        return b[:, 0] < 1000

    frames = [np.full((720, 1280, 3), 90, np.uint8) for _ in range(6)]
    cache, rej = {}, {}
    fresh = track(iter(frames), 25.0, D(), IoUTracker(), cache=cache, cache_rejected=rej,
                  stride=2, ball=True, prior=prior)
    replay = track(iter(frames), 25.0, None, IoUTracker(), cache=cache, cache_rejected=rej,
                   stride=2, ball=True, prior=prior)
    assert fresh["ball"] and replay["ball"] == fresh["ball"], (fresh["ball"], replay["ball"])
    assert replay["_perf"]["prior_rejected"] == fresh["_perf"]["prior_rejected"] == 6, \
        (fresh["_perf"]["prior_rejected"], replay["_perf"]["prior_rejected"])
    assert replay["detections"] == fresh["detections"]


def _selfcheck_stride() -> None:
    """A strided run emits ORIGINAL frame numbers, and stats reads them as time.

    Two things can silently break and both are expensive. If `f` were renumbered,
    every metre downstream would be right and every SECOND wrong by the stride -
    `stitch.split` divides its teleport cut by a hard-coded 25, and app.py seeks
    the video with `cap.set(POS_FRAMES, f)`. If `f` were NOT multiplied by the
    stride at all, a stride-1 detection cache would stop replaying at any other
    stride. Neither shows up as an exception, so the frame numbers are asserted
    directly and then handed to `stats.player_stats` to prove they read as time.
    """
    import stats  # noqa: PLC0415
    from detectors import StubDetector  # noqa: PLC0415
    from pitch import Calibration, Pitch  # noqa: PLC0415
    from tracking import IoUTracker  # noqa: PLC0415

    def frames(n=20):
        return iter([np.full((360, 640, 3), 40, np.uint8) for _ in range(n)])

    # stride is the CALLER's spacing: 20 frames handed over at stride 2 are
    # frames 0..38 of the video.
    r = track(frames(), 25.0, StubDetector(n=4), IoUTracker(), stride=2)
    got = sorted({d["f"] for d in r["detections"]})
    assert got == list(range(0, 40, 2)), got[:6]
    assert r["fps"] == 25.0, "the fps must stay the video's, not fps/stride"
    assert r["_perf"]["stride"] == 2 and r["_perf"]["sampled"] == 20, r["_perf"]

    # A stride-1 cache replays at stride 2 by frame number, with no re-detection.
    cache: dict = {}
    track(frames(40), 25.0, StubDetector(n=4), IoUTracker(), cache=cache)
    calls = [0]

    class Counting:
        def detect(self, frame):
            calls[0] += 1
            return []

    track(frames(), 25.0, Counting(), IoUTracker(), cache=dict(cache), stride=2)
    assert calls[0] == 0, f"a stride-1 cache must replay at stride 2: {calls[0]} misses"

    # _frames skips with grab() and still counts SOURCE frames, so -n covers the
    # same span of video whatever the stride.
    class Cap:
        """Minimal VideoCapture: read() decodes, grab() must not."""
        def __init__(self, n): self.i, self.n, self.read_at = 0, n, []
        def read(self):
            if self.i >= self.n:
                return False, None
            self.read_at.append(self.i)
            self.i += 1
            return True, np.zeros((4, 4, 3), np.uint8)
        def grab(self):
            self.i += 1
            return self.i <= self.n

    c = Cap(100)
    assert len(list(_frames(c, 10, 2))) == 5 and c.read_at == [0, 2, 4, 6, 8], c.read_at
    c = Cap(100)
    assert len(list(_frames(c, 10, 1))) == 10, "stride 1 must be the old behaviour"
    c = Cap(7)                                   # the video ends mid-stride
    assert len(list(_frames(c, None, 3))) == 3 and c.read_at == [0, 3, 6], c.read_at

    # stride 0 would file every row under frame 0. It must not be reachable.
    try:
        track(frames(2), 25.0, StubDetector(n=4), IoUTracker(), stride=0)
        raise AssertionError("stride 0 was accepted")
    except SystemExit:
        pass

    # ...and the numbers read as TIME. 5 m/s for 20 s, sampled every other frame:
    # stats must still measure ~100 m over ~20 s, because it takes its cadence
    # from the track and not from fps. Renumbering would halve the seconds.
    cal = Calibration(pitch=Pitch(105.0, 68.0), H=np.eye(3))
    tr = [{"f": f, "id": 1, "team": "home", "jersey": None,
           "x": 5.0 * (f / 25.0), "y": 34.0} for f in range(0, 501, 2)]
    ps = stats.player_stats(tr, cal, 25.0)
    assert abs(ps.minutes - 20.0 / 60.0) < 1e-6, ps.minutes
    assert abs(ps.distance_m - 100.0) < 2.0, ps.distance_m
    assert ps.observed_frac == 1.0, "a strided track must not read as all-gaps"
    # The distance window is still 0.4 s of WALL CLOCK - 5 samples, not 10.
    assert abs(stats.path_length(np.array([[5.0 * f / 25.0, 34.0] for f in range(0, 501, 2)]),
                                 np.arange(0, 501, 2) / 25.0,
                                 np.ones(250, bool)) - 100.0) < 2.0


def _selfcheck_resume() -> None:
    """A partial detection cache RESUMES; it is neither ignored nor trusted whole.

    The regression this guards is expensive and silent in both directions. Before
    per-frame hits, `replay` was decided once for the whole run, so a cache
    holding 6,000 of 72,101 frames was treated as complete and the remaining
    66,101 frames came back empty - a run that "succeeded" with a fifth of the
    match. The opposite failure is re-detecting frames already cached, which
    costs hours at 2.2 fps. Both produce a plausible file, so only an assert on
    the detector's call count can tell them apart.
    """
    from detectors import StubDetector
    from tracking import IoUTracker

    class Counting:
        """Wraps the stub and records which frames actually hit the detector.

        `start` seeds StubDetector's internal frame counter. The stub advances
        `_f` on every call, so a fresh one on a RESUMED run would emit frame-0
        boxes for frame 8 and the output comparison below would fail for a
        reason that has nothing to do with the cache. A real detector is a pure
        function of the pixels it is given; the stub is not, and the test has to
        pay for that difference rather than be confused by it.
        """
        def __init__(self, start: int = 0):
            self.inner, self.calls = StubDetector(n=4), 0
            self.inner._f = start

        def detect(self, frame):
            self.calls += 1
            return self.inner.detect(frame)

    def frames():
        return iter([np.full((360, 640, 3), 40, np.uint8) for _ in range(20)])

    # Cold: no cache, every frame detected, cache filled.
    cold_det, cache = Counting(), {}
    cold = track(frames(), 25.0, cold_det, IoUTracker(), cache=cache)
    assert cold_det.calls == 20, cold_det.calls
    assert len(cache) == 20, len(cache)

    # Simulate a kill at frame 8: keep only what a checkpoint would have saved.
    partial = {k: v for k, v in cache.items() if k < 8}
    warm_det = Counting(start=8)
    warm = track(frames(), 25.0, warm_det, IoUTracker(), cache=partial)
    assert warm_det.calls == 12, f"resumed run re-detected: {warm_det.calls} != 12"
    assert len(partial) == 20, len(partial)
    # The resumed run must be indistinguishable from the cold one. If it is not,
    # the cache is replaying something the detector would not have produced.
    assert warm["detections"] == cold["detections"], "resume changed the output"

    # A complete cache still skips the detector entirely - the original point of
    # the cache, and the reason run() only passes det=None when complete is set.
    full_det = Counting()
    track(frames(), 25.0, full_det, IoUTracker(), cache=dict(cache))
    assert full_det.calls == 0, full_det.calls

    # Checkpoints fire on newly-detected frames only, at the stated interval.
    seen, cp_cache = [], {}
    track(frames(), 25.0, Counting(), IoUTracker(), cache=cp_cache,
          checkpoint=lambda c, n: seen.append(n), checkpoint_every=5)
    assert seen == [5, 10, 15, 20], seen
    # Resuming past a checkpoint must not re-fire it for cached frames.
    seen2, part2 = [], {k: v for k, v in cp_cache.items() if k < 10}
    track(frames(), 25.0, Counting(), IoUTracker(), cache=part2,
          checkpoint=lambda c, n: seen2.append(n), checkpoint_every=5)
    assert seen2 == [15, 20], seen2


def _selfcheck_ball_model(vid: str, cpath: str, d: str, plain_ball_cache: str) -> None:
    """ball_model: the primary detector built ONCE (with its ball off), no detector at all on a
    complete cache, and the ball model's sha256 in the cache settings, refused when it differs."""
    from unittest import mock  # noqa: PLC0415
    from detectors import StubDetector  # noqa: PLC0415
    fake = os.path.join(d, "ball.pth")
    with open(fake, "wb") as fh:
        fh.write(b"not really a checkpoint")
    sha = hashlib.sha256(b"not really a checkpoint").hexdigest()
    built: list = []

    class _BallStub:
        def detect(self, frame):
            return [{"xyxy": (200.0, 200.0, 209.0, 209.0), "conf": 0.9, "cls": "sports ball"}]

    def fake_get(name, **kw):
        built.append((name, kw.get("ball")))
        return _BallStub() if kw.get("weights") == fake else StubDetector(n=2)

    bm = os.path.join(d, "bm.cache.json")
    with mock.patch.object(detectors, "BALL_MODEL_SHA256", sha), \
            mock.patch.object(detectors, "model_identity", lambda *a, **k: None), \
            mock.patch(f"{__name__}.get_detector", fake_get):
        run(vid, os.path.join(d, "bm1.json"), detector="coreml", calib=cpath, dets_cache=bm,
                ball=True, ball_model=fake)
        assert built == [("coreml", False), ("rfdetr", True)], f"detectors built {built}"
        raw = json.load(open(bm))
        assert all(any(b["cls"] == "sports ball" for b in v) for v in raw["frames"].values()), \
            "the ball model's boxes never reached the cache"
        assert raw["settings"]["ball_model"] == sha
        built.clear()
        run(vid, os.path.join(d, "bm2.json"), detector="coreml", calib=cpath, dets_cache=bm,
            ball=True, ball_model=fake)
        assert built == [], f"a complete cache still built {built}"
        for kw, cache_ in (({}, bm), ({"ball_model": fake}, plain_ball_cache)):
            try:
                run(vid, os.path.join(d, "bm3.json"), detector="coreml" if cache_ == bm else "stub",
                    calib=cpath, dets_cache=cache_, ball=True, **kw)
                raise AssertionError(f"a cache replayed under another ball model ({kw}, {cache_})")
            except SystemExit as e:
                assert "ball_model" in str(e), e


def _selfcheck_remap() -> None:
    """The remap really reaches the detector, and --calib alone is enough.

    This is the regression that shipped once already: undistort_frame existed,
    was documented, was tested in isolation, and was called by nothing - so the
    detector kept seeing raw panorama frames and every number downstream was
    quietly metres out. The only assert that catches that is one on the frames
    the detector actually receives.
    """
    import os, tempfile  # noqa: PLC0415

    import calibrators  # noqa: PLC0415
    from pitch import Calibration, Pitch  # noqa: PLC0415

    SW, SH, DW, DH = 320, 180, 200, 100     # source vs remapped canvas: different
    with tempfile.TemporaryDirectory() as d:
        vid = os.path.join(d, "v.mp4")
        vw = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (SW, SH))
        assert vw.isOpened(), "no mp4 writer; cannot self-check the frame path"
        for _ in range(6):
            vw.write(np.full((SH, SW, 3), 60, np.uint8))
        vw.release()

        gy, gx = np.mgrid[0:DH, 0:DW]
        maps = os.path.join(d, "maps.npz")
        np.savez(maps, mapx=(gx * SW / DW).astype(np.float32),
                 mapy=(gy * SH / DH).astype(np.float32))

        cal = calibrators.RemapCalibrator(
            Pitch(105.0, 68.0), *calibrators.load_maps(maps, 2.0),
            calibrators.PerspectiveCalibrator(Calibration(pitch=Pitch(105.0, 68.0),
                                                          H=np.eye(3))),
            maps=maps, yscale=2.0)
        cpath = os.path.join(d, "c.json")
        cal.save(cpath)
        back = calibrators.load(cpath)
        assert isinstance(back, calibrators.RemapCalibrator), type(back)

        seen: list[tuple] = []

        class Spy:
            def detect(self, frame):
                seen.append(frame.shape[:2])
                return []

        real = globals()["get_detector"]
        globals()["get_detector"] = lambda *a, **kw: Spy()
        try:
            run(vid, os.path.join(d, "t.json"), detector="stub", remap=maps)
            assert seen and set(seen) == {(DH, DW)}, seen
            # --calib alone must imply the remap AND the yscale it was solved
            # at - two settings that can disagree is how you project raw pixels
            # through a remapped-space homography, or one off by 2x.
            seen.clear()
            run(vid, os.path.join(d, "t2.json"), detector="stub", calib=cpath)
            assert seen and set(seen) == {(2 * DH, DW)}, seen
            # ...and no remap at all still hands over the untouched source frame.
            seen.clear()
            run(vid, os.path.join(d, "t3.json"), detector="stub")
            assert seen and set(seen) == {(SH, SW)}, seen
            assert json.load(open(os.path.join(d, "t3.json")))["fps_source"] == "container"
            # A capture reporting 0 fps runs at the fallback and the file SAYS so.
            real_cap = cv2.VideoCapture

            class ZeroFps:
                def __init__(self, *a):
                    self._c = real_cap(*a)

                def get(self, prop):
                    return 0.0 if prop == cv2.CAP_PROP_FPS else self._c.get(prop)

                def __getattr__(self, k):
                    return getattr(self._c, k)

            cv2.VideoCapture = ZeroFps
            try:
                run(vid, os.path.join(d, "t3z.json"), detector="stub")
            finally:
                cv2.VideoCapture = real_cap
            z = json.load(open(os.path.join(d, "t3z.json")))
            assert z["fps_source"] == "fallback" and z["fps"] == 25.0, (z.get("fps_source"), z["fps"])

            # A cache holds boxes the scene prior ALREADY filtered, so replaying
            # it behind a calibration that has since changed mixes two
            # geometries and nothing downstream can see it. The guard must be on
            # the file's CONTENT: the job path overwrites calibration.json in
            # place, so its path never changes and a path check never fires.
            cache = os.path.join(d, "dets.json")
            run(vid, os.path.join(d, "t4.json"), detector="stub", calib=cpath,
                dets_cache=cache)
            assert json.load(open(cache))["calib_sha"] == _calib_digest(cpath)
            calibrators.RemapCalibrator(
                Pitch(100.0, 64.0), *calibrators.load_maps(maps, 2.0),
                calibrators.PerspectiveCalibrator(
                    Calibration(pitch=Pitch(100.0, 64.0), H=np.eye(3))),
                maps=maps, yscale=2.0).save(cpath)      # same path, new geometry
            try:
                run(vid, os.path.join(d, "t5.json"), detector="stub", calib=cpath,
                    dets_cache=cache)
                raise AssertionError("a cache from another calibration was accepted")
            except SystemExit as e:
                assert "geometries" in str(e), e
            # ...and a cache with no calib_sha at all is still replayable, or
            # this check would invalidate every cache already on disk.
            raw = json.load(open(cache))
            del raw["calib_sha"]
            json.dump(raw, open(cache, "w"))
            run(vid, os.path.join(d, "t6.json"), detector="stub", calib=cpath,
                dets_cache=cache)
            # THE STAMP, from a real run(): the configuration is written next to
            # the tracks, and a run with an unusable calibration records
            # prior_gates None - distinguishable, at last, from a working filter.
            stamp = json.load(open(os.path.join(d, "t6.json")))["_perf"]["config"]
            assert set(stamp) >= {"prior_gates", "stride", "conf", "overlap", "tile"}, stamp
            assert stamp["detector"] == "stub" and stamp["stride"] == 1, stamp
            # what RAN: "best" resolved, and the stub is untiled (this read 1024 before 2026-09-28)
            assert stamp["tracker_resolved"] == best_available(quiet=True) and stamp["tile"] is None, stamp
            # THE SETTINGS. The header and the stamp describe one run, and a
            # cache detected at another overlap or threshold is refused - replayed,
            # it would silently BE that configuration.
            assert stamp["stitch"] is True, "the product stitches, and its stamp must say so"
            assert json.load(open(cache))["settings"] == {
                k: v for k, v in stamp.items() if k not in ("stride", "stitch", "tracker_resolved")}, \
                "header and stamp disagree"
            run(vid, os.path.join(d, "t6ns.json"), detector="stub", calib=cpath,
                dets_cache=cache, stitch=False)
            assert json.load(open(os.path.join(d, "t6ns.json")))["_perf"]["config"]["stitch"] is False, \
                "an unstitched run must not stamp as the product"
            was_overlap = detectors.OVERLAP
            detectors.OVERLAP = 0.10
            try:
                run(vid, os.path.join(d, "t7.json"), detector="stub", calib=cpath,
                    dets_cache=cache)
                raise AssertionError("a cache tiled at another overlap was accepted")
            except SystemExit as e:
                assert "overlap" in str(e), e
            finally:
                detectors.OVERLAP = was_overlap
            try:
                run(vid, os.path.join(d, "t8.json"), detector="stub", calib=cpath,
                    conf=0.3, dets_cache=cache)
                raise AssertionError("a cache at another conf was accepted")
            except SystemExit as e:
                assert "conf" in str(e), e
            # ...except a LATE key: a cache from before model_sha256 existed still replays.
            raw = json.load(open(cache))
            del raw["settings"]["model_sha256"]
            json.dump(raw, open(cache, "w"))
            run(vid, os.path.join(d, "t8a.json"), detector="stub", calib=cpath, dets_cache=cache)
            # CACHE_VERSION: a cache from before ball candidates were stored replays for a run
            # without the ball, and is REFUSED by a ball run (it would replay with no ball).
            bc = os.path.join(d, "ball.cache.json")
            run(vid, os.path.join(d, "t8c.json"), detector="stub", calib=cpath, dets_cache=bc, ball=True)
            assert json.load(open(bc))["settings"]["cache_version"] == CACHE_VERSION
            for strip in ("cache_version", "settings"):
                raw = json.load(open(bc))
                (raw["settings"] if strip == "cache_version" else raw).pop(strip)
                old_c = os.path.join(d, f"old_{strip}.cache.json")
                json.dump(raw, open(old_c, "w"))
                try:
                    run(vid, os.path.join(d, "t8d.json"), detector="stub", calib=cpath, dets_cache=old_c,
                        ball=True)
                    raise AssertionError(f"a ball run replayed a cache without {strip}")
                except SystemExit as e:
                    assert "cache_version" in str(e), e
            raw = json.load(open(cache))
            del raw["settings"]["cache_version"]
            json.dump(raw, open(cache, "w"))
            run(vid, os.path.join(d, "t8e.json"), detector="stub", calib=cpath, dets_cache=cache)
            assert json.load(open(os.path.join(d, "t8e.json")))["_perf"]["config"]["cache_version"] is None
            _selfcheck_ball_model(vid, cpath, d, bc)
            # A header that recorded SOME settings but not one this run has is a
            # difference, not a pass - even when this run's value is None, which
            # prior_gates is here (the stub calibration builds no prior).
            raw = json.load(open(cache))
            assert raw["settings"]["prior_gates"] is None, raw["settings"]
            del raw["settings"]["prior_gates"]
            json.dump(raw, open(cache, "w"))
            try:
                run(vid, os.path.join(d, "t8b.json"), detector="stub", calib=cpath,
                    dets_cache=cache)
                raise AssertionError("a cache missing a recorded setting was accepted")
            except SystemExit as e:
                assert "prior_gates not recorded" in str(e), e
            # ...and one that recorded nothing still replays, UNSTAMPED - the
            # pair with t6 above, which is what lets this assert fail.
            raw = json.load(open(cache))
            del raw["settings"]
            json.dump(raw, open(cache, "w"))
            run(vid, os.path.join(d, "t9.json"), detector="stub", calib=cpath,
                dets_cache=cache)
            assert "config" not in json.load(open(os.path.join(d, "t9.json")))["_perf"], \
                "tracks off a cache with unknown settings must not carry a stamp"
        finally:
            globals()["get_detector"] = real

        # stitch_tracks, on the calibration built above. One player cut into two
        # fragments must come back as one id; an opposing-team fragment sitting
        # in the same place must NOT be absorbed, which is the gate; and the
        # merged player takes the team its detections vote for rather than the
        # one its first fragment happened to carry.
        # Identity homography, so an image coordinate IS a pitch metre and the
        # speed gate can be reasoned about directly.
        flat = Calibration(pitch=Pitch(), H=np.eye(3))

        def frag(tid, f0, f1, x, team):
            return [{"f": f, "id": tid, "x": x + 0.05 * (f - f0), "y": 34.0,
                     "team": team, "jersey": None} for f in range(f0, f1)]

        # One player in two fragments, plus an opposing player alongside them.
        rows = (frag(1, 0, 100, 10.0, "home") + frag(2, 110, 200, 15.5, "home")
                + frag(3, 0, 200, 10.2, "away"))
        got = stitch_tracks(list(rows), flat, 25.0)
        assert len({r["id"] for r in got}) == 2, \
            f"the two fragments of one player must merge: {sorted({r['id'] for r in got})}"

        # The gate: relabel the second fragment onto the other bank and the same
        # join is refused, because two components confidently on opposite banks
        # are not one player.
        split = (frag(1, 0, 100, 10.0, "home") + frag(2, 110, 200, 15.5, "away")
                 + frag(3, 0, 200, 10.2, "away"))
        assert len({r["id"] for r in stitch_tracks(list(split), flat, 25.0)}) == 3, \
            "an opposite-team fragment must not be absorbed"

        # An UNLABELLED fragment is not blocked, and the merged player takes the
        # team its detections vote for rather than whatever it arrived with.
        unl = (frag(1, 0, 100, 10.0, "home") + frag(2, 110, 200, 15.5, "unknown"))
        merged = stitch_tracks(list(unl), flat, 25.0)
        assert len({r["id"] for r in merged}) == 1, "an unlabelled fragment must still join"
        assert {r["team"] for r in merged} == {"home"}, \
            f"the merged player must re-vote to home: {[r['team'] for r in merged][:3]}"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", nargs="?", help="omit to run the self-check")
    p.add_argument("-o", "--out", default="tracks.json")
    p.add_argument("-d", "--detector", default="rfdetr")
    p.add_argument("-t", "--tracker", default="best",
                   help="best = C-BIoU when the trackers package is present, "
                        "else the built-in bytetrack")
    p.add_argument("-w", "--weights", default=None)
    p.add_argument("-c", "--conf", type=float, default=0.5)
    p.add_argument("-r", "--resolution", type=int, default=None,
                   help="detector input size; raise it if small players are missed")
    p.add_argument("-n", "--max-frames", type=int, default=None)
    p.add_argument("-s", "--start", type=int, default=0, help="first frame to read")
    p.add_argument("--calib", default=None,
                   help="calibration.json; enables the pitch mask + player-size "
                        "prior, which reject crowd and wrong-sized boxes for free")
    p.add_argument("--remap", default=None,
                   help="dir of *mapx.npy/*mapy.npy (or an .npz) that flattens a "
                        "stitched panorama. Applied to every frame BEFORE detection; "
                        "--calib must then be a calibration solved on the FLAT image. "
                        "Implied when --calib is a remap calibration.")
    p.add_argument("--remap-yscale", type=float, default=1.0,
                   help="resample the flattened canvas vertically. Vendor tables "
                        "size the canvas for their pipeline, not ours - SoccerTrack's "
                        "undersample the pitch ~2.9x, i.e. a 60px player becomes 24px. "
                        "3 buys ~1:1 at 3x the detector pixels, but it makes pixels "
                        "ANISOTROPIC and so DISABLES the scene prior (see make_prior). "
                        "Prefer a calibration solved with a pitch-band crop and an "
                        "isotropic scale (calibrators.pitch_band), which buys the same "
                        "detail and keeps the prior. Ignored when the tables come from "
                        "--calib, which stores its own.")
    p.add_argument("--dets-cache", default=None,
                   help="detections json to replay, or to write if absent. Skips "
                        "the detector entirely on replay, which is 92-98%% of "
                        "wall clock - use it to tune the TRACKER without paying "
                        "for detection again. Refuses a cache built from a "
                        "different video or start frame.")
    p.add_argument("--ball-markings", action="store_true",
                   help="reject ball candidates whose white blob is a painted LINE "
                        "rather than a compact object. Removes every line pickup on "
                        "1080p broadcast footage and costs 3-6 points against the "
                        "4K top-down ball truth - see ball.blob_extent for both")
    p.add_argument("--roles", action="store_true",
                   help="name each track player/goalkeeper/referee from its crops "
                        "(model/models/role_clf.pt) and let a confident referee vote "
                        "override kit colour. Measured on 1080p broadcast footage; "
                        "it abstains on the panorama - see rolenet.py")
    p.add_argument("--ball", action="store_true",
                   help="also detect and pick the ball (COCO 'sports ball' + ball.py's "
                        "shape gate). Off by default: on footage the gate cannot carry, "
                        "it emits confident legs - see detectors.BALL_CLASSES")
    p.add_argument("--stride", type=int, default=1,
                   help="detect every Nth frame. 1 (default) changes nothing; the "
                        "product's stride is set in app/server/jobs.py. Its identity "
                        "cost is unmeasured - docs/reference/detection-tracking.md#stride. "
                        "Rows keep their ORIGINAL frame numbers at the ORIGINAL "
                        "fps, so a stride-1 --dets-cache replays at any stride.")
    p.add_argument("-v", "--variant", default="medium",
                   help="rfdetr size: nano/small/medium/base/large. Pick on measured "
                        "small-player recall on YOUR footage, not on paper mAP.")
    p.add_argument("--ball-model", nargs="?", const=detectors.BALL_MODEL, default=None,
                   help="take the ball from the FINE-TUNED nano detector (N52 arm D1) "
                        "instead of the primary detector's sports-ball class. Bare flag "
                        "uses detectors.BALL_MODEL; a path overrides it. Needs --ball. "
                        "OFF by default because it is a SECOND per-frame pass: accuracy "
                        "0.3167 -> 0.5850 measured on six clips of two matches "
                        "(docs/experiments/n65-every-ball-number-was-measured-at-384.md), at a "
                        "MEASURED 3.82x detection cost (0.94 -> 3.59 s/frame at 4K, "
                        "tile 640 making ~40 tiles against tile 1024's ~15) and with no "
                        "CoreML fast path. Do not enable it on real jobs as it stands.")
    a = p.parse_args()
    if a.video:
        run(a.video, a.out, a.detector, a.tracker, a.conf, a.weights, a.resolution,
            a.max_frames, a.start, a.variant, a.calib, a.remap, a.remap_yscale,
            a.dets_cache, a.stride, a.ball, a.roles, a.ball_markings,
            ball_model=a.ball_model)
    else:
        _selfcheck()
