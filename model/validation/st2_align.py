"""G0: does the video frame at a labelled event actually show that event?

WHY THIS RUNS BEFORE ANY TRAINING
    Every expensive mistake in this repo has been a silent misalignment that
    produced plausible numbers - the event feed in the acting team's frame, tracks
    in remapped space, `-s` renumbering frames, a golden file generated on a 5x
    stretched clock. None of them crashed. All of them were visible in one picture.

    So: take labelled events, cut the frame at each one, crop to where the ball is,
    caption it with the label, and LOOK. A throw-in tile must show a throw-in.

THE PROJECTION, WHICH IS NOT A HOMOGRAPHY
    A single pitch->panorama homography does NOT work here: fitting one to the 65
    hand-annotated keypoints gives 103 px median error and 11/65 inliers, because
    the panorama is STITCHED, not a single perspective view. The shipped chain is

        pitch metres --H--> remapped px --mapx lookup--> panorama px

    `118575_homography.npy` maps CORNER-origin pitch metres (0..105, 0..68) to the
    remapped canvas; `118575_mapx.npy` is OpenCV fixed-point CV_16SC2, so
    `mapx[v, u]` is the source panorama pixel for remapped pixel (u, v).
    Measured against the 65 keypoints: **median 7.8 px**. That is the check.

BALL ORIENTATION
    `ball/*.npz` is CENTRE-origin metres, and SoccerTrack's convention is "y towards
    the touchline nearer the camera". The keypoints put pitch y=68 at pixel y~598-715
    (bottom of frame, i.e. near side), so corner_y = ball_y + 34, unflipped.

THE CONTACT SHEET IS NOT THE PROOF - `--probe` IS
    A 460 px tile cannot tell a PASS from a HIGH PASS, and one still frame cannot
    show a pass at all. The sheet rules out the catastrophic failures (wrong half,
    wrong minute, wrong coordinate frame) and nothing finer.

    `--probe` is the decisive test, and it needs no eyes. Each class has a physical
    signature in the ball track, so sweep a time offset and see where it sharpens:
    OUT must sit ON a boundary, SHOT near a goal, PASS in open play. Measured on
    118575 period 0 - the ball annotation STOPS within 0.2 s of every OUT label,
    and its last position is 0.00 m from the line. That pins alignment to ~5 frames
    without anyone squinting at a tile.

`ponytail: 118575 is the only match with calibration on disk, so the sheet runs on
one match. --probe needs only ball npz + index, so it runs on all 20 halves.`
"""
import csv
import json
import os
import random
import re
import sys

import cv2
import numpy as np

CAL = "data/footage/soccertrack_calib_118575"
VIDEO = "data/footage/118575_panorama_1st_half.mp4"
BALL = "data/soccertrack_v2/ball__118575_1st_ball.npz"
INDEX = "data/soccertrack_v2/bas_index.csv"
OUT = "model/validation/out/G0_alignment_118575.jpg"
MATCH, PERIOD = "118575", 0
COLS, ROWS = 6, 4
CROP_W, CROP_H = 900, 506       # 16:9 around the ball, ~24 m of pitch at centre
TILE_W = 460


def ball_to_pitch(bx, by):
    """Ball centre-origin metres -> CORNER-origin metres, the projector's frame.

    One function, used by `main`, `probe` and `demo` alike. It was three copies,
    and a self-check that carries its own copy of the arithmetic it is checking
    cannot catch a sign error in the other two - which is exactly what happened.
    """
    return [[bx + 52.5, by + 34.0]]


def projector():
    """pitch metres (corner origin) -> panorama pixel. Returns (fn, keypoint error)."""
    H = np.load(f"{CAL}/{MATCH}_homography.npy")
    mx = np.load(f"{CAL}/{MATCH}_mapx.npy")

    def fn(xy):
        P = np.asarray(xy, np.float64).reshape(-1, 1, 2)
        R = cv2.perspectiveTransform(P, H).reshape(-1, 2)
        u = np.clip(np.round(R[:, 0]).astype(int), 0, mx.shape[1] - 1)
        v = np.clip(np.round(R[:, 1]).astype(int), 0, mx.shape[0] - 1)
        return np.stack([mx[v, u, 0], mx[v, u, 1]], 1).astype(float)

    kp = json.load(open(f"{CAL}/{MATCH}_keypoints.json"))
    P = np.array([[float(n) for n in re.findall(r"-?[\d.]+", k)] for k in kp])
    Q = np.array(list(kp.values()), np.float64)
    return fn, float(np.median(np.linalg.norm(fn(P) - Q, axis=1)))


def pick(n, seed=0):
    """n events spread over classes and over time, each with the ball actually seen."""
    z = np.load(BALL, allow_pickle=True)
    bx, by, st = z["x"], z["y"], z["status"]
    rows = [r for r in csv.DictReader(open(INDEX))
            if r["match"] == MATCH and int(r["period"]) == PERIOD]
    by_cls = {}
    for r in rows:
        f = int(r["frame"])
        if not (0 <= f < len(st)) or st[f] == 0:
            continue                                   # ball not annotated here
        by_cls.setdefault(r["label"], []).append((r, f))

    rng = random.Random(seed)
    order = sorted(by_cls, key=lambda k: len(by_cls[k]))   # rarest class first
    out = []
    while len(out) < n and any(by_cls.values()):
        for k in order:
            if not by_cls[k] or len(out) >= n:
                continue
            out.append(by_cls[k].pop(rng.randrange(len(by_cls[k]))))
    out.sort(key=lambda rf: rf[1])
    return out, bx, by


def main():
    proj, kp_err = projector()
    print(f"projection check: median {kp_err:.1f} px against 65 hand keypoints")
    if kp_err > 25:
        sys.exit(f"projection is broken ({kp_err:.1f} px) - fix before reading the sheet")

    picks, bx, by = pick(COLS * ROWS)
    cap = cv2.VideoCapture(VIDEO)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    th = int(TILE_W * CROP_H / CROP_W)
    sheet = np.zeros((ROWS * th, COLS * TILE_W, 3), np.uint8)

    for i, (r, f) in enumerate(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        px, py = proj(ball_to_pitch(bx[f], by[f]))[0]
        x0 = int(np.clip(px - CROP_W / 2, 0, W[0] - CROP_W))
        y0 = int(np.clip(py - CROP_H / 2, 0, W[1] - CROP_H))
        tile = frame[y0:y0 + CROP_H, x0:x0 + CROP_W]
        cv2.circle(tile, (int(px - x0), int(py - y0)), 26, (0, 230, 255), 2)
        tile = cv2.resize(tile, (TILE_W, th))
        cap_txt = f"{r['label']}  {float(r['t_s']) // 60:.0f}:{float(r['t_s']) % 60:04.1f}"
        sub = f"{r['team']}  #{r['player_id']}  f{f}"
        cv2.rectangle(tile, (0, th - 34), (TILE_W, th), (0, 0, 0), -1)
        cv2.putText(tile, cap_txt, (6, th - 20), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, sub, (6, th - 5), cv2.FONT_HERSHEY_SIMPLEX, .40, (190, 190, 190), 1, cv2.LINE_AA)
        R, C = divmod(i, COLS)
        sheet[R * th:(R + 1) * th, C * TILE_W:(C + 1) * TILE_W] = tile
        print(f"  {i:2d} {r['label']:26s} t={float(r['t_s']):7.1f}s  ball px ({px:6.0f},{py:5.0f})")
    cap.release()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\n-> {OUT}   ({sheet.shape[1]}x{sheet.shape[0]})")
    print("LOOK AT IT. The circle is where the ball is said to be. Each caption must "
          "match what the tile shows. If it does not, nothing downstream is worth running.")


def probe(match=MATCH, period=0, ball=BALL):
    """Alignment, measured rather than eyeballed. Each class has a ball-track
    signature; sweep an offset and see where it sharpens."""
    z = np.load(ball, allow_pickle=True)
    bx, by, st = z["x"], z["y"], z["status"]
    rows = [r for r in csv.DictReader(open(INDEX))
            if r["match"] == match and int(r["period"]) == period]

    def seen(f):
        return None if not (0 <= f < len(st)) or st[f] == 0 else (bx[f], by[f])

    outs = [int(r["frame"]) for r in rows if r["label"] == "OUT"]
    print(f"{match} p{period}: OUT events (n={len(outs)}) vs a time offset")
    print(f"{'offset s':>9s} {'ball annotated':>15s} {'median edge dist m':>19s}")
    for off in (-2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2):
        v = [seen(f + int(round(off * 25))) for f in outs]
        ok = [p for p in v if p]
        d = [min(52.5 - abs(a), 34.0 - abs(b)) for a, b in ok]
        md = f"{np.median(d):19.2f}" if d else f"{'-':>19s}"
        print(f"{off:9.1f} {100 * len(ok) / max(len(v), 1):14.0f}% {md}")

    print("\nper-class signature at offset 0")
    def line(lab, fn):
        f = [int(r["frame"]) for r in rows if r["label"] == lab]
        ok = [p for p in (seen(i) for i in f) if p]
        tail = fn(ok) if ok else "ball never annotated (dead ball - expected)"
        print(f"  {lab:26s} n={len(f):4d}  seen {100 * len(ok) / max(len(f), 1):3.0f}%   {tail}")
    line("OUT", lambda o: "median %.2f m from a boundary" % np.median(
        [min(52.5 - abs(a), 34.0 - abs(b)) for a, b in o]))
    line("THROW IN", lambda o: "median |y| %.1f m (touchline 34.0)" % np.median([abs(b) for a, b in o]))
    line("SHOT", lambda o: "median %.1f m from the nearer goal" % np.median(
        [float(np.hypot(52.5 - abs(a), b)) for a, b in o]))
    line("GOAL", lambda o: "median %.1f m from the nearer goal" % np.median(
        [float(np.hypot(52.5 - abs(a), b)) for a, b in o]))
    line("PASS", lambda o: "median |x| %.1f m |y| %.1f m (open play)" % (
        np.median([abs(a) for a, b in o]), np.median([abs(b) for a, b in o])))
    rng = np.random.default_rng(0)
    o = [p for p in (seen(int(f)) for f in rng.integers(0, len(st), 4000)) if p]
    md = float(np.median([min(52.5 - abs(a), 34.0 - abs(b)) for a, b in o]))
    print(f"  {'RANDOM FRAMES':26s} n=4000  seen {100 * len(o) / 4000:3.0f}%   "
          f"median {md:.1f} m from a boundary")


def demo():
    """Self-check: the projection must reproduce the hand keypoints, and every
    picked event must have a ball reading and land inside the video."""
    proj, err = projector()
    assert err < 25, f"projection {err:.1f} px"
    picks, bx, by = pick(24)
    assert len(picks) == 24, len(picks)
    n = int(cv2.VideoCapture(VIDEO).get(cv2.CAP_PROP_FRAME_COUNT))
    for r, f in picks:
        assert 0 <= f < n, (r, f)
        p = proj(ball_to_pitch(bx[f], by[f]))[0]
        assert 0 <= p[0] < 4096 and 0 <= p[1] < 1080, (r, p)
    assert len({r["label"] for r, _ in picks}) >= 8, "classes not spread"

    # THE CENTRAL CLAIM IS THE ORIENTATION, and neither the contact sheet nor
    # --probe can check it: `min(52.5-|x|, 34-|y|)` is flip-invariant, so a
    # sign error on the ball's y passes both. Pin it to the geometry instead -
    # SoccerTrack's +y is the touchline NEARER the camera, and the keypoints put
    # pitch y=68 low in the frame. So a ball at large positive centre-origin y
    # must project BELOW a ball at large negative y.
    # ...and it must run through the BALL conversion, not hardcoded pitch coords,
    # or a sign error in `by[f] + 34.0` sails past it.
    zb = np.load(BALL, allow_pickle=True)
    bx_, by_, st_ = zb["x"], zb["y"], zb["status"]
    ok_ = st_ != 0
    hi = int(np.argmax(np.where(ok_, by_, -1e9)))     # ball nearest the camera
    lo = int(np.argmin(np.where(ok_, by_, +1e9)))     # ball furthest away
    p_hi = proj(ball_to_pitch(bx_[hi], by_[hi]))[0]
    p_lo = proj(ball_to_pitch(bx_[lo], by_[lo]))[0]
    assert p_hi[1] > p_lo[1] + 100, (
        f"ball y orientation flipped: +y={by_[hi]:.1f}m -> row {p_hi[1]:.0f}, "
        f"-y={by_[lo]:.1f}m -> row {p_lo[1]:.0f}; +y is the NEAR touchline so it "
        f"must project LOWER in the frame")
    print(f"demo ok: projection {err:.1f} px, y orientation pinned, 24 events "
          f"over {len({r['label'] for r, _ in picks})} classes, all in frame")


if __name__ == "__main__":
    (demo if "--demo" in sys.argv else probe if "--probe" in sys.argv else main)()
