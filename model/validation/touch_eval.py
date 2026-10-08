"""Touches and passes from tracks + ball, in PIXELS, scored against real labels.

WHY THIS EXISTS AND WHAT IT IS NOT
    `possession.py` infers a possession path from 22 trajectories with no ball at
    all, and is the better answer when it can run. It cannot run here: PathCRF is
    not vendored on this machine, it demands exactly 22 tracks, and it needs
    metres. This file is the opposite trade - it needs a ball, and in exchange it
    needs no calibration, no fixed roster and no third-party weights. It runs on
    any footage the ball picker survives.

    Owner = nearest player to the ball. Owner changes, held for HOLD samples, are
    touches. Consecutive touches are passes. That is the whole model, and its
    ceiling is the ball picker underneath it (see ball.py).

DISTANCE IS TO THE FEET, DIVIDED BY BOX HEIGHT, AND BOTH HALVES MATTER
    The first version measured to the player's BOX and it called a ball sailing
    over a player's head a touch, because the box spans that y. So the distance
    is to the ground point only.

    It is then divided by box height because height is the only depth cue an
    uncalibrated frame has - a player 60 m away is half the pixels and every
    pixel distance around them means twice as much. One threshold in
    body-heights works near and far; one threshold in pixels works at exactly
    one depth. `ponytail: 1.8 m per body height is implied and never written
    down. Upgrade path is a calibration, which turns this into metres.`

CONSTANTS COME FROM DFL'S 3,468 PASS ANNOTATIONS, NOT FROM TASTE
    GAP_S was 3.0 by guess. DFL's own labels put the median inter-event gap at
    3.76 s and p90 at 13.36 s, so a 3 s cutoff was discarding more than half of
    every real interval. The same labels killed a second assumption: a 30 s clip
    holds a median of 2.9 passes and 3.7 ball events, NOT the ~20 touches an
    eyeball guess suggests. Do not re-tune these by eye.

SCORING, AND THE ONE NUMBER THAT MAKES THE REST READABLE
    Labels are `118575_12_class_events.json`, SoccerNet ball-action format, 1,026
    half-1 rows whose `position` (ms) agrees with `gameTime` on all 2,035 rows to
    within 1.5 s and spans 0.6-2871.3 s of a 2874.0 s video.

    Two targets, and the second is the fair one. A PASS is labelled when the
    passer KICKS it; a touch fires when the RECEIVER gets it, so touches lag
    passes by the flight time and `lag` reports that. Every CONTACT label marks a
    player touching the ball, which is what a touch IS - so CONTACT is the
    like-for-like set.

    CONTACT labels run at ~25/min. At a 2.5 s tolerance a detection thrown at
    random is inside SOME label's window almost always, so raw F1 here is close
    to meaningless. `chance()` throws the same NUMBER of detections uniformly at
    random, 300 times, and that line - not zero - is what a result has to beat.

FIRST MEASUREMENT, 2026-09-03. 118575 half 1, frames 7,500-12,000 (300-480 s),
coreml + stride 2 + --ball, remapped to 11878x870. 20 touches, NEAR_H 1.0:

                        tol 1.0 s                 tol 2.5 s
                  prec   rec    F1  chance   prec   rec    F1  chance
    PASS-family   0.30  0.14  0.19   0.26    0.80  0.36  0.50   0.43
    CONTACT       0.75  0.21  0.32   0.26    0.95  0.26  0.41   0.33

    IT BEATS CHANCE BY 5-8 F1 POINTS AND NO MORE. On n=20 detections in one
    3-minute window that is a signal, not a result. Two things in it are real
    enough to build on: precision is high and recall is low - when it fires it
    is usually on a genuine ball contact, it just rarely fires - and the median
    lag is -0.2 s against CONTACT but +1.0 s against PASS, which is the pass
    flight time showing up exactly where the model says it should (a touch is
    the RECEPTION; the PASS label is the kick).

    WHERE THE MISSING RECALL LIVES, which is the useful half. Fraction of labels
    with ANY ball detection within tolerance - the ceiling this logic could
    reach with a perfect owner rule:

        PASS-family   tol 1.0 s  0.48    tol 2.5 s  0.61
        CONTACT       tol 1.0 s  0.53    tol 2.5 s  0.65

    So of CONTACT at 2.5 s: 35% of labels have no ball at all, and the touch
    rule converts 0.26/0.65 = 40% of the rest (PASS-family: 59%). The loss is
    roughly half detector, half owner rule. The detector half is the panorama -
    the picker fired on 344 of ~2,250 sampled frames (15.3%) here against 41% on
    DFL 1080p native, which is the format effect ball.py predicts.

    THIS SLICE CANNOT SEPARATE THE TWO CLEANLY and no data in this repo can.
    118575 has event labels and NO ground-truth ball; the SoccerTrack
    `soccer_top__*` clips have ground-truth ball and NO event labels. They never
    overlap. Isolating the owner rule needs labels and a ball on the same
    footage - DFL's competition videos, or SoccerNet Ball Action Spotting.

    ^ THAT PARAGRAPH WAS WRONG, and stayed wrong for two days. It is true of the
    files that happened to be on disk and false of the dataset they came from.
    SoccerTrack v2 ships `ball/` - PER-FRAME BALL TRUTH for all 20 halves - next
    to the BAS labels for the same footage, under CC BY 4.0. It is kept here
    because the reasoning was sound and the inventory was not: check the source,
    not the download. `gt_isolate()` is the experiment it said was impossible.

THE OWNER RULE IS FINE. THE BALL WAS ALWAYS THE PROBLEM (2026-09-05)
    `gt_isolate()`, 118575 half 1 ENTIRE (2,884 s, 1,020 CONTACT / 535
    PASS-family labels), ground-truth ball + ground-truth tracks, metres:

        near_m  det          CONTACT tol 1.0 s          PASS tol 2.5 s
                        prec   rec    F1  chance    prec   rec    F1  chance
           1.0  614     0.76  0.46  0.57    0.37    0.62  0.71  0.66    0.51
           1.5  974     0.65  0.62  0.63    0.44    0.47  0.85  0.60    0.52

    F1 0.63 at a +0.19 margin over chance, against 0.32 for the same rule on a
    DETECTED ball. The rule roughly doubles when the ball stops being the
    bottleneck, and its lag at near_m 1.0 is +0.00 s - unbiased to two decimals.

    THE COMPARISON THAT DECIDES THE ROADMAP:

        owner rule + ground-truth ball    F1 0.63   (+0.19 over chance)
        owner rule + detected ball        F1 0.32   (+0.06)
        PathCRF, NO BALL AT ALL           F1 0.58   (+0.12)

    PathCRF reaches 92% of what this rule manages with a PERFECT ball, while
    needing no ball at all. Since a perfect ball is exactly what the panorama
    cannot give (the picker fires on 15.3% of frames there), the ball-free route
    is not merely competitive, it is the one that survives contact with real
    footage. Spend effort on trajectories, not on the ball.

    ^ WITHDRAWN 2026-09-10: PathCRF's row does not clear a same-count
    metronome (`metronome()` below; model-map (not published) §0.4), so it cannot
    carry this conclusion. The owner rule's row has not been compared with
    one either.

THE DFL CLIP BENCH IS A NULL RESULT, AND THE REASON IS THE LABEL SET (2026-09-05)
    `bench()` below, 50 play + 50 background clips of `alejopaullier/dfl-clips`:

        NEAR_H   play/s   bg/s   ratio      <- 1.00 is no signal
           0.6    0.119  0.190    0.63
           1.0    0.308  0.325    0.95
           2.0    0.716  0.508    1.41
           3.0    0.885  0.611    1.45

    At any threshold that still MEANS possession (<=1.0 body-heights, ~1.8 m)
    the rule fires slightly MORE in background than in event clips. It only
    clears 1.0 at 2-3 body-heights, i.e. 3.6-5.4 m, where "nearest player" has
    stopped meaning "has the ball" and a touch is just the nearest body changing
    as the ball rolls.

    THE NEGATIVE CLASS IS NOT NEGATIVE FOR THIS QUESTION. "background" means no
    annotated play/challenge/throwin - it does NOT mean no ball contact. Six
    background clips eyeballed are ordinary live football, 22 players, ball in
    play (`data/dfl_clips/background_sheet.jpg`), and the median gap between
    annotated events is 3.76 s against a 2.52 s background clip, so they are
    carved straight out of in-play football that is full of touches. The
    contrast this dataset offers is "was there an annotated EVENT", which is the
    competition's task and not ours.

    ONE CONFOUND DIED HERE, WHICH IS WORTH THE RUN. Ball coverage is 0.41 on
    play against 0.37 on background - equal. So the ball detector does NOT
    separate the classes and cannot be credited for any ratio above 1. An n=3
    smoke test had shown 0.41 vs 0.07 and that was pure noise.

    HYPOTHESIS THE NUMBERS SUGGEST, NOT A FINDING. The deficit at NEAR_H 0.6 is
    the wrong way round for a short-clip artifact (the first confirmed owner is
    a free touch, and dividing it by a play clip's shorter 1.92 s should inflate
    play, not depress it). A pass may be marked by the ball being UNOWNED in
    flight rather than by an owner change. Test that before tuning this further.
"""
import json
import sys

import numpy as np

# ---------------------------------------------------------------- the detector

NEAR_H = 0.6      # ball within this many body-heights of the feet = possession
HOLD = 2          # samples a new owner must hold before the change counts
GAP_S = 8.0       # longer than this and the ball was lost, not passed


def _d(bx, p):
    """Ball point to a player's ground point, in body-heights."""
    return (((p["x"] - bx[0]) ** 2 + (p["y"] - bx[1]) ** 2) ** 0.5) / p["h"]


def touches(d):
    """tracks.json (with `ball`) -> [{f, t, id, team, x, y}], one per owner change."""
    by_f = {}
    for x in d["detections"]:
        by_f.setdefault(x["f"], []).append(x)
    owner, cand, n, out = None, None, 0, []
    for b in d["ball"]:
        ps = by_f.get(b["f"], [])
        if not ps:
            continue
        p = min(ps, key=lambda q: _d((b["x"], b["y"]), q))
        who = p["id"] if _d((b["x"], b["y"]), p) <= NEAR_H else None
        n = n + 1 if who == cand else 1
        cand = who
        if n >= HOLD and cand is not None and cand != owner:
            out.append({"f": b["f"], "t": round(b["f"] / d["fps"], 2), "id": cand,
                        "team": p["team"], "x": round(b["x"]), "y": round(b["y"])})
            owner = cand
    return out


def passes(ts):
    """Consecutive touches inside GAP_S. `same_team` is NOT trustworthy - kit
    colour is 70.7% per detection, so a TURNOVER here is as likely to be a team
    label error as a real one."""
    return [{"t0": a["t"], "t1": b["t"], "src": a["id"], "dst": b["id"],
             "same_team": a["team"] == b["team"], "dt": round(b["t"] - a["t"], 2)}
            for a, b in zip(ts, ts[1:]) if b["t"] - a["t"] <= GAP_S]


# ------------------------------------------------------------------ the scorer

PASSES = {"PASS", "HIGH PASS", "CROSS"}
CONTACT = PASSES | {"DRIVE", "HEADER", "SHOT", "THROW IN", "FREE KICK",
                    "BALL PLAYER BLOCK", "PLAYER SUCCESSFUL TACKLE"}
LABELS = "data/footage/soccertrack_calib_118575/118575_12_class_events.json"


def labels(path, t0, t1, keep, half="1"):
    """SoccerNet-format labels -> sorted event times in seconds.

    Two shapes, because the copy in `data/footage/` predates the dataset release: the
    old one is {"annotations": [...]} with `team` blank on every row, the
    SoccerTrack v2 download is {"actions": [...]} and carries `team` on all rows
    and `player_id` on 99.7% of them. Prefer the v2 file - it is strictly richer
    and it is what makes per-player scoring possible at all.
    """
    d = json.load(open(path))
    rows = d.get("actions", d.get("annotations", []))
    return sorted(t for t in (float(x["position"]) / 1000.0 for x in rows
                              if x["label"] in keep and x["gameTime"].startswith(half))
                  if t0 <= t <= t1)


def touches_m(frames, XY, ball, live, fps, near_m=1.5, hold=HOLD):
    """The owner rule in METRES, on ground truth. The isolation experiment.

    Everything else in this file feeds the rule a DETECTED ball and so measures
    the picker and the rule welded together. This takes SoccerTrack v2's
    per-frame ball truth and the ground-truth tracks, so whatever it scores is
    the rule alone. `near_m` replaces NEAR_H because with metres there is no
    depth to normalise away - the body-height divisor exists only to fake this.

    `live` is the ball's own status flag; frames where it is 0 are skipped
    rather than treated as position (0,0).
    """
    out, owner, cand, n = [], None, None, 0
    for i in range(len(frames)):
        if not live[i]:
            cand, n = None, 0
            continue
        d2 = ((XY[i, :, 0] - ball[i, 0]) ** 2 + (XY[i, :, 1] - ball[i, 1]) ** 2)
        j = int(np.nanargmin(d2))
        who = j if d2[j] ** 0.5 <= near_m else None
        n = n + 1 if who == cand else 1
        cand = who
        if n >= hold and cand is not None and cand != owner:
            out.append(frames[i] / fps)
            owner = cand
    return out


def score(det, gt, tol):
    """Greedy nearest-first 1-1 match -> (tp, precision, recall, f1, median lag).

    1-1 matters: without it one dense burst of detections scores every nearby
    label and reads as recall it did not earn.
    """
    ud, ug, lag = set(), set(), []
    for _, i, j in sorted((abs(d - g), i, j)
                          for i, d in enumerate(det) for j, g in enumerate(gt)
                          if abs(d - g) <= tol):
        if i in ud or j in ug:
            continue
        ud.add(i)
        ug.add(j)
        lag.append(det[i] - gt[j])
    tp = len(lag)
    p = tp / len(det) if det else 0.0
    r = tp / len(gt) if gt else 0.0
    lag.sort()
    return (tp, p, r, 2 * p * r / (p + r) if p + r else 0.0,
            lag[len(lag) // 2] if lag else float("nan"))


def chance(n, gt, tol, t0, t1, trials=300, seed=0):
    """Mean F1 of n detections thrown uniformly at random. THE BASELINE TO BEAT."""
    import random  # noqa: PLC0415
    rng = random.Random(seed)
    return sum(score(sorted(rng.uniform(t0, t1) for _ in range(n)), gt, tol)[3]
               for _ in range(trials)) / trials


def metronome(n, gt, tol, t0, t1, phases=20):
    """(mean recall, mean F1) of n detections spaced EVENLY, over `phases` offsets.

    THE FLOOR THAT BINDS. `chance` scatters n events at random; a clock puts
    one in every gap, and when labels come every few seconds a tolerance near
    the gap scores a clock like a model. On 118575 half 1 the PathCRF
    possession changes scored BELOW this at both the CONTACT F1 and the PASS
    recall they were published on (model-map (not published) §0.4). No timing number
    here means anything without this beside it."""
    if n <= 0 or t1 <= t0:      # nothing emitted, or no window: nothing matched
        return (0.0, 0.0)
    step = (t1 - t0) / n
    runs = [score([t0 + step * (i + k / phases) for i in range(n)], gt, tol)
            for k in range(phases)]
    return (sum(r[2] for r in runs) / phases, sum(r[3] for r in runs) / phases)


def report(tracks_path, labels_path=LABELS, start="0"):
    """`start` is track.py's -s, in FRAMES, and leaving it out silently scores the
    wrong minutes of football.

    track.py seeks with `cap.set(POS_FRAMES, start)` and then counts from zero,
    so a `-s 7500` run emits frames 0..N and tracks.json records the offset
    NOWHERE. Its docstring promises frame numbers survive `--stride`; nothing
    says they do not survive `-s`. Scored without this, a slice from 300-480 s
    was compared against the labels for 0-180 s and landed exactly where two
    unrelated windows should - at chance.
    """
    d = json.load(open(tracks_path))
    if not d.get("ball"):
        raise SystemExit(f"{tracks_path} has no `ball` - rerun track.py with --ball")
    off = int(start) / d["fps"]
    fr = [x["f"] for x in d["detections"]]
    t0, t1 = min(fr) / d["fps"] + off, max(fr) / d["fps"] + off
    print(f"window {t0:.0f}-{t1:.0f}s ({t1 - t0:.0f}s), ball on {len(d['ball'])} frames")
    for name, keep in (("PASS / HIGH PASS / CROSS", PASSES), ("all CONTACT labels", CONTACT)):
        gt = labels(labels_path, t0, t1, keep)
        print(f"\n== {name}: {len(gt)} labels ({len(gt) * 60 / (t1 - t0):.1f}/min) ==")
        print(f"{'NEAR_H':>6} {'det':>4} |" + "|".join(
            f"  tol {t}s: {'tp':>3} {'prec':>5} {'rec':>5} {'F1':>5} {'lag':>6} {'chance':>6} "
            for t in (1.0, 2.5)))
        for nh in (0.4, 0.6, 0.8, 1.0):
            global NEAR_H
            NEAR_H = nh
            det = [x["t"] + off for x in touches(d)]
            cells = []
            for tol in (1.0, 2.5):
                tp, p, r, f1, lag = score(det, gt, tol)
                cells.append(f"           {tp:3d} {p:5.2f} {r:5.2f} {f1:5.2f} {lag:+6.2f} "
                             f"{chance(len(det), gt, tol, t0, t1):6.2f} ")
            print(f"{nh:6.1f} {len(det):4d} |" + "|".join(cells))
        NEAR_H = 0.6


def bench(zip_path="data/dfl_clips/dfl-clips.zip", n=60, seed=0, stride=2):
    """DFL event clips vs background clips: does a touch fire where an event is?

    `alejopaullier/dfl-clips` (CC0) cuts one clip per annotation of the DFL
    competition - 2,849 play, 598 challenge, 172 throwin - plus 1,802 background
    clips holding no event. That turns spotting into a two-sample question with
    a real negative class, which no other data here has: a touch rate inside
    event clips against the SAME rate inside background.

    NORMALISE BY DURATION OR THIS LIES. Background clips are every one of them
    exactly 2.52 s; play clips run 0.84-4.48 s, median 1.92. Background is 31%
    longer on average, so touches-per-CLIP is biased toward background before a
    single frame is decoded. Touches per second is the comparison.
    """
    import os  # noqa: PLC0415
    import random  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import zipfile  # noqa: PLC0415

    import cv2  # noqa: PLC0415
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "model", "engine"))
    from detectors import get_detector  # noqa: PLC0415
    from track import track  # noqa: PLC0415
    from tracking import get_tracker  # noqa: PLC0415

    z = zipfile.ZipFile(zip_path)
    mp4 = [x for x in z.namelist() if x.endswith(".mp4")]
    rng = random.Random(seed)
    sel = {k: rng.sample([x for x in mp4 if x.split("/")[-1].startswith(k + "-")], n)
           for k in ("play", "background")}
    out_dir = os.path.join(os.path.dirname(zip_path), "bench")
    os.makedirs(out_dir, exist_ok=True)

    # ONE detector for every clip. Built per clip it would spawn a CoreML worker
    # 120 times and startup would dominate the measurement.
    det = get_detector("coreml", conf=0.5, ball=True)
    res = {}
    for kind, members in sel.items():
        rows = []
        for m in members:
            p = os.path.join(out_dir, m.split("/")[-1])
            if not os.path.exists(p):
                with z.open(m) as s, open(p, "wb") as o:
                    o.write(s.read())
            dur = float(subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", p], capture_output=True, text=True).stdout.strip())
            cap = cv2.VideoCapture(p)
            frames, i = [], 0
            while True:
                ok, im = cap.read()
                if not ok:
                    break
                if i % stride == 0:
                    frames.append(im)
                i += 1
            cap.release()
            r = track(iter(frames), 25.0 / stride, det, get_tracker("best"), ball=True)
            rows.append({"tracks": r, "dur": dur, "ball": len(r["ball"]),
                         "frames": len(frames), "name": m.split("/")[-1]})
            print(f"  {kind:11} {rows[-1]['name']:28} "
                  f"{rows[-1]['ball']}/{rows[-1]['frames']} ball frames", flush=True)
        res[kind] = rows

    print(f"\n{'class':12} {'clips':>5} {'secs':>7} {'ball frac':>10}")
    for kind, rows in res.items():
        print(f"{kind:12} {len(rows):5d} {sum(r['dur'] for r in rows):7.0f} "
              f"{sum(r['ball'] for r in rows) / sum(r['frames'] for r in rows):10.2f}")

    # SWEEP, because one NEAR_H is a guess and the tracks are already paid for.
    global NEAR_H
    keep = NEAR_H
    print(f"\n{'NEAR_H':>6} | {'play /s':>8} {'bg /s':>8} {'ratio':>7} | "
          f"{'play >=1':>9} {'bg >=1':>7}")
    for nh in (0.6, 0.8, 1.0, 1.5, 2.0, 3.0):
        NEAR_H = nh
        cell = {}
        for kind, rows in res.items():
            n_t = [len(touches(r["tracks"])) for r in rows]
            cell[kind] = (sum(n_t) / sum(r["dur"] for r in rows),
                          sum(t >= 1 for t in n_t) / len(rows))
        rp, rb = cell["play"][0], cell["background"][0]
        print(f"{nh:6.1f} | {rp:8.3f} {rb:8.3f} {rp / rb if rb else float('inf'):7.2f} | "
              f"{cell['play'][1]:9.2f} {cell['background'][1]:7.2f}")
    NEAR_H = keep
    print("\nratio 1.00 = no signal. Ball coverage differs by class too - a ratio "
          "above 1 that\nmatches the ball-fraction ratio is the DETECTOR "
          "separating them, not the touch rule.")
    return res


GT_TRACKS = "model/validation/soccertrack/118575_1st.tracks.json.gz"
GT_BALL = "data/soccertrack_v2/ball__118575_1st_ball.npz"
V2_LABELS = "data/soccertrack_v2/bas__118575__118575_12_class_events.json"
BALL_F_OFF = 3_000_000      # track_f = ball_f + this. Verified, see gt_isolate.
PITCH = (52.5, 34.0)        # ball npz is CENTRE-origin; tracks are CORNER-origin


def gt_isolate(half=GT_TRACKS, ball=GT_BALL, lab=V2_LABELS):
    """Owner rule on GROUND-TRUTH ball + GROUND-TRUTH tracks. No detector at all.

    ALIGNMENT, VERIFIED NOT ASSUMED. Ball frames run 1..72,101 and track frames
    3,000,001..3,072,101, so `track_f = ball_f + 3,000,000`. The ball npz is
    CENTRE-origin metres (x -57.75..57.75) while the tracks are CORNER-origin
    (0..105), so the ball needs +52.5/+34. With both applied the nearest player
    to the ball sits at a median 1.72 m - a ball at somebody's feet, and within
    noise of the 1.77 m possession-location error possession.py reports. With
    the shift omitted it is 33.79 m. That is the check; it is not subtle.
    """
    import gzip  # noqa: PLC0415

    b = np.load(ball)  # plain arrays only; no allow_pickle, see the security note
    bxy = np.stack([b["x"] + PITCH[0], b["y"] + PITCH[1]], 1)
    live = b["status"] != 0
    bframe = b["frame"].astype(np.int64) + BALL_F_OFF

    d = json.load(gzip.open(half))
    fps = float(d["fps"])
    ids = sorted({x["id"] for x in d["detections"]}, key=int)
    col = {k: i for i, k in enumerate(ids)}
    row = {f: i for i, f in enumerate(bframe)}
    XY = np.full((len(bframe), len(ids), 2), np.nan)
    for x in d["detections"]:
        i = row.get(x["f"])
        if i is not None:
            XY[i, col[x["id"]]] = (x["x"], x["y"])

    t0, t1 = float(bframe[0] - BALL_F_OFF) / fps, float(bframe[-1] - BALL_F_OFF) / fps
    print(f"{len(bframe)} frames ({t1 - t0:.0f}s), {len(ids)} tracks, "
          f"ball in play {live.mean():.3f}")
    # Returned as well as printed. `api.py` needs a value, and a printed table is
    # not a value - three of the first twelve measurements regex-scraped a
    # scorer's stdout, and each one breaks the day the format is tidied.
    out = {"frames": len(bframe), "secs": t1 - t0, "tracks": len(ids),
           "ball_in_play": float(live.mean()), "rows": {}}
    for name, keep in (("PASS", PASSES), ("CONTACT", CONTACT)):
        gt = labels(lab, t0, t1, keep)
        print(f"\n== {name}: {len(gt)} labels ({len(gt) * 60 / (t1 - t0):.1f}/min) ==")
        print(f"{'near_m':>6} {'det':>5} |" + "|".join(
            f"  tol {t}s: {'prec':>5} {'rec':>5} {'F1':>5} {'chance':>6} {'lag':>6} "
            for t in (1.0, 2.5)))
        for nm in (0.5, 1.0, 1.5, 2.0, 3.0):
            det = touches_m(bframe - BALL_F_OFF, XY, bxy, live, fps, near_m=nm)
            cells = []
            for tol in (1.0, 2.5):
                _, p, r, f1, lag = score(det, gt, tol)
                ch = chance(len(det), gt, tol, t0, t1, trials=40)
                out["rows"][f"{name}.{nm}.{tol}"] = {
                    "labels": len(gt), "det": len(det), "prec": p, "rec": r,
                    "f1": f1, "chance": ch, "lag": lag}
                cells.append(f"            {p:5.2f} {r:5.2f} {f1:5.2f} "
                             f"{ch:6.2f} {lag:+6.2f} ")
            print(f"{nm:6.1f} {len(det):5d} |" + "|".join(cells))
    return out


def _demo():
    """Synthetic asserts for both halves. No footage, no labels, no weights."""
    # DETECTOR: A holds the ball, then B does. One touch each, one pass between.
    d = {"fps": 25.0, "detections": [], "ball": []}
    for f in range(0, 20, 2):
        for i, x in ((1, 100.0), (2, 400.0)):
            d["detections"].append({"f": f, "id": i, "x": x, "y": 200.0,
                                    "w": 40.0, "h": 80.0, "team": "home"})
        d["ball"].append({"f": f, "x": 100.0 if f < 10 else 400.0, "y": 195.0})
    ts = touches(d)
    assert [t["id"] for t in ts] == [1, 2], ts
    ps = passes(ts)
    assert len(ps) == 1 and (ps[0]["src"], ps[0]["dst"]) == (1, 2), ps

    # A ball 3 body-heights overhead is NOT a touch - the bug this gate exists for.
    high = {"fps": 25.0, "detections": d["detections"],
            "ball": [{"f": f, "x": 100.0, "y": 200.0 - 3 * 80.0} for f in range(0, 20, 2)]}
    assert touches(high) == [], touches(high)

    # SCORER
    tp, p, r, f1, lag = score([1.1, 5.1], [1.0, 5.0, 9.0], 1.0)
    assert (tp, round(p, 3), round(r, 3), round(lag, 3)) == (2, 1.0, 0.667, 0.1)
    assert score([50.0], [1.0, 5.0], 1.0)[0] == 0            # nothing in range
    assert score([1.0, 1.05], [1.0], 1.0)[0] == 1            # one GT, one match
    assert score([1.0, 5.0], [1.0, 5.0], 1.0)[3] == 1.0
    assert 0.0 < chance(3, [10.0, 200.0, 500.0], 1.0, 0.0, 600.0) < 0.2
    print("ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        bench(*sys.argv[2:])
    elif len(sys.argv) < 2:
        _demo()
    else:
        report(*sys.argv[1:])
