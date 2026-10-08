"""Why the detector loses the ball on ONE of N48's six clips: look at the pixels.

Answers open question 2 of n48-ball-detector-ab (not published). Detects nothing - it
replays N48's committed candidate caches (`.cmpB`), reads SoccerTrack's ball
truth, and reads the video. Everything it writes goes to
model/validation/out/n49_ball_clip_gap/ and nowhere else.

WHAT IT MEASURES, AND WHY THESE THREE NUMBERS

  blob_px   area of the connected BRIGHT component (HSV V>165, S<70 - bright and
            unsaturated, which pitch paint, a white kit and the ball all are)
            that contains the GT ball centre, inside a 121x121 window. A ball
            alone on grass is its own component, ~50 px. A ball at rest ON a
            painted line is one component WITH the line, and its outline is no
            longer closed. This is the variable that separates the three clips.
  elong     that component's PCA axis ratio. A line through the window is
            extreme, a lone ball ~1.
  contrast  ball disc mean (r<=4, grayscale) minus the median of the NON-bright
            pixels in an r 8-13 ring. It is here to be refuted: the clip the
            detector fails on has the HIGHEST contrast of the three, so dimness
            is not the explanation.
  nearest   distance from GT to the nearest arm-B candidate centre, px. <= 20 is
            ball_eval.GATE_PX, the same gate N48 ruled on.

  blob_px is a heuristic and fires on a white kit too. On `1710-1740`, where a
  white-kitted player stands over a motionless ball, it reads 49 - so it
  discriminates in practice. That is an observation on three clips, not a claim
  about the measure in general.

  ponytail: HSV threshold and a connected component, no segmentation model. If
  the prevalence run in n49-ball-on-the-paint (not published) ever needs this to be
  right on shadowed paint, the upgrade path is a local threshold, not a net.

    .venv/bin/python model/validation/n49_paint.py            # the numbers
    .venv/bin/python model/validation/n49_paint.py --sheets   # + contact sheets
    .venv/bin/python model/validation/n49_paint.py --demo     # self-check
"""
import argparse
import os
import sys
import json

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "model", "engine"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ball_eval as B  # noqa: E402
from paths import FOOTAGE2, VALIDATION  # noqa: E402

#: The anomaly clip and the two 30 s windows either side of it - same match,
#: same camera. They are the control: whatever explains 1740-1770 has to leave
#: its neighbours alone.
CLIPS = ["soccer_top__test__videos__D_20220220_1_1710_1740",
         "soccer_top__test__videos__D_20220220_1_1740_1770",
         "soccer_top__test__videos__D_20220220_1_1770_1800"]
#: N48 scored the first 300 frames of each clip. Same window here, or the
#: numbers describe a different population from the one being explained.
N = 300
#: The frame this clip's ball comes to rest on the line. Read off the hit map,
#: not fitted: arm B answers on every frame before it and none after.
SETTLE = 54
W = 60
OUT = os.path.join(VALIDATION, "out", "n49_ball_clip_gap")


def _bright(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 2] > 165) & (hsv[:, :, 1] < 70)).astype(np.uint8)


def _blob(patch):
    """(area, elongation) of the bright component under the patch centre."""
    bright = _bright(patch)
    n, lab = cv2.connectedComponents(bright)
    cid = lab[W, W]
    if cid == 0:                        # centre not bright: take the nearest blob
        ys, xs = np.nonzero(bright)
        cid = lab[ys, xs][np.argmin((ys - W) ** 2 + (xs - W) ** 2)] if ys.size else 0
    ys, xs = np.nonzero(lab == cid) if cid else (np.array([]), np.array([]))
    if ys.size < 2:
        return int(ys.size), 1.0
    ev = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]).astype(float)))
    return int(ys.size), float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))


def measure(clip):
    xy, _, _ = B.load_ball(clip)
    cache = B.load_cache(clip + ".cmpB")
    cap = cv2.VideoCapture(str(FOOTAGE2 / f"{clip}.mp4"))
    out = []
    # Sequential, like ball_eval.build_cache: seeking this container lands on
    # the wrong frame often enough to poison a 20 px gate.
    for f in range(N):
        ok, im = cap.read()
        if not ok:
            break
        g = xy[f]
        x, y = int(round(g[0])), int(round(g[1]))
        p = cv2.copyMakeBorder(im, W, W, W, W, cv2.BORDER_REPLICATE)[y:y + 2 * W + 1,
                                                                    x:x + 2 * W + 1]
        area, elong = _blob(p)
        gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY).astype(float)
        Y, X = np.ogrid[:2 * W + 1, :2 * W + 1]
        d = np.hypot(Y - W, X - W)
        ring = (d >= 8) & (d <= 13) & (_bright(p) == 0)
        near = min([float(np.hypot((r[0] + r[2]) / 2 - g[0], (r[1] + r[3]) / 2 - g[1]))
                    for r in cache["frames"].get(f, [])], default=9e9)
        out.append({"f": f, "blob_px": area, "elong": round(elong, 1),
                    "contrast": round(float(gray[d <= 4].mean() -
                                            (np.median(gray[ring]) if ring.sum() else np.nan)), 1),
                    "nearest": round(near, 1)})
    cap.release()
    return out


def sheets(clip, frames, name, r=80, scale=3, cols=5):
    """One crop per sampled frame on the GT position: GT box red, arm-B
    candidates yellow. The point of the whole file is that these are lookable-at."""
    xy, wh, _ = B.load_ball(clip)
    cache = B.load_cache(clip + ".cmpB")
    cap = cv2.VideoCapture(str(FOOTAGE2 / f"{clip}.mp4"))
    tiles = {}
    for f in range(max(frames) + 1):
        ok, im = cap.read()
        if not ok:
            break
        if f not in set(frames):
            continue
        g = xy[f]
        x0, y0 = int(g[0]) - r, int(g[1]) - r
        pad = cv2.copyMakeBorder(im, r, r, r, r, cv2.BORDER_CONSTANT)
        c = cv2.resize(pad[y0 + r:y0 + 3 * r, x0 + r:x0 + 3 * r],
                       (2 * r * scale,) * 2, interpolation=cv2.INTER_NEAREST)
        s = wh[f] * scale / 2
        cv2.rectangle(c, (int(r * scale - s),) * 2, (int(r * scale + s),) * 2, (0, 0, 255), 1)
        for b in cache["frames"].get(f, []):
            px = ((b[0] + b[2]) / 2 - g[0]) * scale + r * scale
            py = ((b[1] + b[3]) / 2 - g[1]) * scale + r * scale
            if 0 <= px < 2 * r * scale and 0 <= py < 2 * r * scale:
                cv2.circle(c, (int(px), int(py)), 6, (0, 255, 255), 1)
        cv2.putText(c, f"f{f} ({int(g[0])},{int(g[1])})", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles[f] = c
    cap.release()
    fs = sorted(tiles)
    rows = []
    for i in range(0, len(fs), cols):
        row = [tiles[f] for f in fs[i:i + cols]]
        row += [np.zeros_like(row[0])] * (cols - len(row))
        rows.append(np.hstack(row))
    p = os.path.join(OUT, name)
    cv2.imwrite(p, np.vstack(rows))
    print(f"  {p}  {len(fs)} frames")


def _context(clip, frame):
    """The whole pitch with the ball circled. A crop cannot show that the ball
    is on the penalty-area line while play is 2,000 px away."""
    xy, _, _ = B.load_ball(clip)
    cap = cv2.VideoCapture(str(FOOTAGE2 / f"{clip}.mp4"))
    im = None
    for _ in range(frame + 1):
        ok, im = cap.read()
    cap.release()
    g = xy[frame]
    cv2.circle(im, (int(g[0]), int(g[1])), 40, (0, 0, 255), 6)
    p = os.path.join(OUT, f"context_f{frame}.jpg")
    cv2.imwrite(p, cv2.resize(im, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"  {p}  ball at ({g[0]:.0f}, {g[1]:.0f})")


def demo():
    """The measure must separate a disc on grass from a disc on a bar."""
    grass = np.full((2 * W + 1, 2 * W + 1, 3), (60, 120, 70), np.uint8)
    ball = grass.copy()
    cv2.circle(ball, (W, W), 6, (240, 240, 240), -1)
    a_alone, e_alone = _blob(ball)
    line = ball.copy()
    cv2.line(line, (W - 3, 0), (W - 3, 2 * W), (240, 240, 240), 9)
    a_line, e_line = _blob(line)
    assert 80 < a_alone < 250, a_alone
    assert a_line > 5 * a_alone and e_line > 2 * e_alone, (a_line, a_alone, e_line, e_alone)
    assert _blob(grass)[0] == 0
    print(f"n49_paint demo ok: lone ball {a_alone} px elong {e_alone:.1f}; "
          f"on a line {a_line} px elong {e_line:.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sheets", action="store_true", help="also write the contact sheets")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args(argv)
    if a.demo:
        return demo()
    os.makedirs(OUT, exist_ok=True)
    res = {c: measure(c) for c in CLIPS}
    json.dump(res, open(os.path.join(OUT, "paint.json"), "w"))
    print(f"\n{'clip':10s} {'armB hit':>9s} {'blob_px med':>12s} {'elong med':>10s} "
          f"{'contrast med':>13s}")
    for c, rows in res.items():
        v = {k: np.array([r[k] for r in rows]) for k in ("blob_px", "elong", "contrast", "nearest")}
        print(f"{c[-9:]:10s} {int((v['nearest'] <= 20).sum()):4d}/{len(rows):<4d} "
              f"{np.median(v['blob_px']):12.0f} {np.median(v['elong']):10.1f} "
              f"{np.median(v['contrast']):13.1f}")
    v = {k: np.array([r[k] for r in res[CLIPS[1]]]) for k in ("blob_px", "elong", "nearest")}
    for lo, hi, nm in ((0, SETTLE, "f0-53   ball in play/at feet"),
                       (SETTLE, N, "f54-299 ball at rest on the line")):
        m = slice(lo, hi)
        print(f"  1740_1770 {nm}: hit {int((v['nearest'][m] <= 20).sum()):3d} of {hi - lo}  "
              f"blob_px med {np.median(v['blob_px'][m]):.0f}  "
              f"elong med {np.median(v['elong'][m]):.1f}")
    if a.sheets:
        print()
        f = list(range(0, N, 20))
        for c in CLIPS:
            sheets(c, f, f"gap_{c[-9:]}.png")
        sheets(CLIPS[1], [0, 30, 55, 60, 70, 100, 150, 200, 250, 290],
               "zoom_1740_1770.png", r=26, scale=10)
        _context(CLIPS[1], 150)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
