"""Measure the detector's per-frame positional noise in PIXELS. Unblocks the filter.

`stats.SIGMA_PX` is currently a guess, and the audit showed the consequence: no
single (ACCEL_Q, SIGMA_PX) pair is both unbiased and noise-robust, so the pair
we ship loses ~26% of the distance. The right value is not a tuning choice, it
is a property of the detector we can measure - but only against ground truth in
IMAGE SPACE, which SoccerTrack does not provide (its GT is pitch metres).

TeamTrack does. `footage2/soccer_top__*__annotations__*.csv` carries per-frame
bounding boxes for 22 players from a single fixed PERSPECTIVE camera, MIT
licensed. This measures:

  1. detection recall (how many of the 22 the detector finds), and
  2. the positional error of the GROUND POINT (box bottom-centre) against GT,
     which is exactly sigma_px

Run: .venv/bin/python measure_sigma.py [--detector stub] [--frames 40]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from paths import FOOTAGE2 as FOOTAGE, VALIDATION


def load_teamtrack(csv_path: Path) -> dict[int, np.ndarray]:
    """{frame_index: (N,4) xyxy boxes}. Wide CSV: 3 header rows then one row
    per frame, with bb_height/bb_left/bb_top/bb_width repeating per player."""
    rows = list(csv.reader(open(csv_path)))
    if len(rows) < 4:
        return {}
    teams, players, attrs = rows[0][1:], rows[1][1:], rows[2][1:]

    # Group the flat column list into one (player -> {attr: col}) mapping.
    cols: dict[tuple, dict] = {}
    for i, (t, p, a) in enumerate(zip(teams, players, attrs)):
        if p == "BALL" or not a:
            continue                      # the ball is not a player; v1 ignores it
        cols.setdefault((t, p), {})[a] = i

    out = {}
    for r in rows[3:]:
        if not r or not r[0].strip():
            continue
        try:
            fi = int(float(r[0]))
        except ValueError:
            continue
        vals = r[1:]
        boxes = []
        for c in cols.values():
            try:
                l = float(vals[c["bb_left"]]); t_ = float(vals[c["bb_top"]])
                w = float(vals[c["bb_width"]]); h = float(vals[c["bb_height"]])
            except (KeyError, ValueError, IndexError):
                continue                  # a player off-frame has blank cells
            if w > 0 and h > 0:
                boxes.append([l, t_, l + w, t_ + h])
        if boxes:
            out[fi] = np.array(boxes, float)
    return out


def ground_point(boxes: np.ndarray) -> np.ndarray:
    """Bottom-centre. The same convention track.py writes into tracks.json."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    return np.column_stack([(b[:, 0] + b[:, 2]) / 2.0, b[:, 3]])


def match(gt: np.ndarray, det: np.ndarray, max_px: float):
    """Greedy nearest ground-point matching. Returns (gt_idx, det_idx, dists).

    Greedy, not Hungarian: at this stage recall is far from 1, so most GT boxes
    have no candidate at all and the optimal assignment is not the interesting
    quantity. If recall gets high this should become linear_sum_assignment,
    because greedy invents matches when players are close.
    """
    if len(gt) == 0 or len(det) == 0:
        return [], [], np.array([])
    g, d = ground_point(gt), ground_point(det)
    dist = np.linalg.norm(g[:, None, :] - d[None, :, :], axis=2)
    gi, di, out = [], [], []
    used_g, used_d = set(), set()
    for _ in range(min(len(g), len(d))):
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        if dist[i, j] > max_px:
            break
        gi.append(i); di.append(j); out.append(dist[i, j])
        used_g.add(i); used_d.add(j)
        dist[i, :] = np.inf; dist[:, j] = np.inf
    return gi, di, np.array(out)


def scale_map(cal, wh: tuple, n: int = 24) -> np.ndarray:
    """Metres-per-pixel at each of n x n image points. (n*n, 2), largest first.

    The two singular values of the Jacobian ARE the local metre scale along the
    two principal image directions, so their ratio is the anisotropy that
    stats.kalman_ground's R is built to represent. Reporting it is the only way
    to know whether the anisotropic term is doing any work on a given camera.
    """
    import stats  # noqa: PLC0415 - avoids importing torch-adjacent deps at module load
    w, h = wh
    gx, gy = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, n))
    uv = np.column_stack([gx.ravel(), gy.ravel()])
    J = stats.homography_jacobian(np.asarray(cal.H, float), uv)
    return np.linalg.svd(J, compute_uv=False)


def _pct(name: str, e: np.ndarray, unit: str) -> str:
    return (f"  {name:22s} median {np.median(e):6.3f}  mean {e.mean():6.3f}  "
            f"p95 {np.percentile(e, 95):6.3f}  {unit}   (n={len(e)})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector", default="stub")
    ap.add_argument("--tracker", default="iou")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--clip", default="", help="substring of the annotation filename")
    # TeamTrack numbers its annotation rows from 1; cv2 numbers video frames
    # from 0. Guessing wrong shifts every comparison by one frame, which at
    # 2-4 px of player motion per frame is most of the sigma being measured.
    ap.add_argument("--frame-offset", type=int, default=0,
                    help="video frame index = annotation index + this")
    ap.add_argument("--max-px", type=float, default=60.0,
                    help="a detection further than this from any GT box is a miss")
    a = ap.parse_args()

    ann = sorted(p for p in FOOTAGE.glob("soccer_top__*__annotations__*.csv")
                 if a.clip in p.name)
    if not ann:
        print("SKIP: no TeamTrack soccer_top annotations in footage2/")
        return 0
    csv_path = ann[0]
    vid = FOOTAGE / csv_path.name.replace("annotations", "videos").replace(".csv", ".mp4")
    if not vid.exists():
        print(f"SKIP: {vid.name} not present")
        return 0

    gt = load_teamtrack(csv_path)
    if not gt:
        print(f"SKIP: no boxes parsed from {csv_path.name}")
        return 0

    import cv2
    from detectors import get_detector
    print(f"clip   {vid.name}")
    print(f"GT     {len(gt)} annotated frames, "
          f"{np.mean([len(v) for v in gt.values()]):.1f} boxes/frame")
    hs = np.concatenate([v[:, 3] - v[:, 1] for v in gt.values()])
    print(f"       player pixel heights: median {np.median(hs):.0f}, "
          f"range {hs.min():.0f}-{hs.max():.0f}")

    # The homography from model/validation/teamtrack_calib.py, if this clip has one.
    # Without it only the pixel number is available, and the pixel number is not
    # what the filter consumes - R is metres, via the Jacobian.
    from pitch import Calibration  # noqa: PLC0415
    cpath = VALIDATION / f"calib_{vid.stem}.json"
    cal = Calibration.load(str(cpath)) if cpath.exists() else None
    print(f"calib  {cpath.name if cal else 'NONE - pixel numbers only'}"
          + (f"  rms {cal.rms_error_m:.3f} m, pitch {cal.pitch.length:.1f}x"
             f"{cal.pitch.width:.1f} m" if cal else ""))

    det = get_detector(a.detector, **({"conf": a.conf, "tile": a.tile}
                                      if a.detector == "rfdetr" else {}))
    cap = cv2.VideoCapture(str(vid))
    # Spread the sample across the whole clip. Taking the first N frames samples
    # one second of a 30 s clip: the players never leave the third they started
    # in, and the per-region breakdown below reports an empty attacking third.
    allk = sorted(gt)
    keys = [allk[i] for i in np.linspace(0, len(allk) - 1, min(a.frames, len(allk))).astype(int)]
    n_gt = n_det = n_hit = 0
    errs, hit_gp, miss_gp = [], [], []
    wh = None
    for fi in keys:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi + a.frame_offset)
        ok, frame = cap.read()
        if not ok:
            continue
        wh = (frame.shape[1], frame.shape[0])
        boxes = np.array([d["xyxy"] for d in det.detect(frame)], float).reshape(-1, 4)
        gi, di, dd = match(gt[fi], boxes, a.max_px)
        n_gt += len(gt[fi]); n_det += len(boxes); n_hit += len(gi)
        errs.append(dd)
        g, d = ground_point(gt[fi]), ground_point(boxes)
        # Keep the ERROR VECTOR and WHERE it happened, not just its length: the
        # whole question is whether error grows with distance from the camera,
        # and a pooled median cannot answer that.
        if len(gi):
            hit_gp.append(np.column_stack([g[gi], d[di] - g[gi]]))
        miss_gp.append(g[[i for i in range(len(g)) if i not in set(gi)]])
    cap.release()

    e = np.concatenate(errs) if errs else np.array([])
    recall = n_hit / n_gt if n_gt else 0.0
    print(f"\ndetector={a.detector} conf={a.conf} tile={a.tile} over {len(keys)} frames")
    print(f"  recall           {recall*100:5.1f}%   ({n_hit}/{n_gt} GT boxes matched)")
    print(f"  detections/frame {n_det/max(len(keys),1):5.1f}   (GT has "
          f"{n_gt/max(len(keys),1):.1f})")
    if not len(e):
        print("  no matches - cannot estimate SIGMA_PX")
        return 0

    hits = np.concatenate(hit_gp)          # (N,4): gt_u, gt_v, du, dv
    print("\npixel space")
    print(_pct("ground-point error", e, "px"))
    print(_pct("  |du| (image x)", np.abs(hits[:, 2]), "px"))
    print(_pct("  |dv| (image y)", np.abs(hits[:, 3]), "px"))
    print(f"  bias (du, dv)          {hits[:, 2].mean():+6.2f}, {hits[:, 3].mean():+6.2f} px"
          "   <- a systematic offset is a convention mismatch, not noise")
    # stats.SIGMA_PX is a per-axis Gaussian sigma, and the median of an error
    # MAGNITUDE is not that. Estimate each axis from its own median absolute
    # deviation about its own bias: a constant offset is a convention mismatch
    # the filter should not be told to expect as noise.
    sig = [float(np.median(np.abs(hits[:, i] - np.median(hits[:, i]))) / 0.6745)
           for i in (2, 3)]
    print(f"\n  robust per-axis sigma: u {sig[0]:.2f} px, v {sig[1]:.2f} px "
          f"(anisotropy {max(sig)/min(sig):.2f}:1 in PIXELS)")
    print(f"  => SIGMA_PX ~ {np.hypot(*sig)/np.sqrt(2):.2f} px (isotropic equivalent) "
          f"at recall {recall*100:.1f}%; error magnitude median {np.median(e):.2f} px "
          f"(stats.py uses {__import__('stats').SIGMA_PX})")

    if cal is None:
        return 0

    sv = scale_map(cal, wh)
    print(f"\nhomography scale over the frame (m/px): "
          f"major {sv[:, 0].min():.4f}-{sv[:, 0].max():.4f}, "
          f"minor {sv[:, 1].min():.4f}-{sv[:, 1].max():.4f}")
    print(f"  anisotropy (major/minor): median {np.median(sv[:, 0]/sv[:, 1]):.3f}, "
          f"max {np.max(sv[:, 0]/sv[:, 1]):.3f}"
          "   <- 1.0 means the anisotropic R term is doing nothing")

    gt_m = cal.to_pitch(hits[:, :2])
    err_m = cal.to_pitch(hits[:, :2] + hits[:, 2:]) - gt_m
    en = np.linalg.norm(err_m, axis=1)
    print("\nmetre space (the units R is actually in)")
    print(_pct("ground-point error", en, "m"))
    print(_pct("  along pitch length", np.abs(err_m[:, 0]), "m"))
    print(_pct("  across pitch width", np.abs(err_m[:, 1]), "m"))

    # Recall and error, by where on the pitch. This is the entanglement check:
    # if the thirds with the worst recall also carry the largest error, the
    # pooled sigma above is flattering by exactly the amount it cannot see.
    miss_m = cal.to_pitch(np.concatenate(miss_gp)) if any(len(m) for m in miss_gp) \
        else np.empty((0, 2))
    L, W = cal.pitch.length, cal.pitch.width
    print("\n  by pitch third (length) / half (width):     recall   err median   p95")
    for label, hsel, msel in (
        ("defensive third", gt_m[:, 0] < L / 3, miss_m[:, 0] < L / 3),
        ("middle third", (gt_m[:, 0] >= L / 3) & (gt_m[:, 0] < 2 * L / 3),
         (miss_m[:, 0] >= L / 3) & (miss_m[:, 0] < 2 * L / 3)),
        ("attacking third", gt_m[:, 0] >= 2 * L / 3, miss_m[:, 0] >= 2 * L / 3),
        ("near touchline y<W/2", gt_m[:, 1] < W / 2, miss_m[:, 1] < W / 2),
        ("far touchline y>=W/2", gt_m[:, 1] >= W / 2, miss_m[:, 1] >= W / 2),
    ):
        nh, nm = int(hsel.sum()), int(msel.sum())
        r = nh / (nh + nm) if nh + nm else float("nan")
        sub = en[hsel]
        med = f"{np.median(sub):6.3f}" if nh else "     -"
        p95 = f"{np.percentile(sub, 95):6.3f}" if nh else "     -"
        print(f"    {label:22s} {r*100:5.1f}%   {med} m   {p95} m   "
              f"(n={nh}, missed {nm})")

    print("\nCaveat: recall and error are entangled. A detector that only finds "
          "big near players\nreports a flattering sigma, because the far side it "
          "missed is where error is worst.")
    return 0


def _selfcheck() -> None:
    """Parser and matcher, without needing a model."""
    ann = sorted(FOOTAGE.glob("soccer_top__*__annotations__*.csv"))
    if ann:
        gt = load_teamtrack(ann[0])
        assert gt, "parsed no frames from real TeamTrack GT"
        n = np.mean([len(v) for v in gt.values()])
        assert 5 < n <= 22, f"{n} boxes/frame is not a football match"
        hs = np.concatenate([v[:, 3] - v[:, 1] for v in gt.values()])
        assert 5 < np.median(hs) < 200, np.median(hs)

    # Matching: an exact overlay must give zero error, and a shifted box must
    # give exactly the shift - if this is wrong every sigma below is wrong.
    g = np.array([[100., 100., 120., 160.], [500., 300., 520., 360.]])
    _, _, d0 = match(g, g.copy(), 60.0)
    assert np.allclose(d0, 0), d0
    shifted = g + np.array([3.0, 0, 3.0, 0])
    _, _, d1 = match(g, shifted, 60.0)
    assert np.allclose(d1, 3.0), d1
    # Beyond max_px is a miss, not a bad match.
    _, _, d2 = match(g, g + np.array([500.0, 0, 500.0, 0]), 60.0)
    assert len(d2) == 0, d2
    print("measure_sigma.py ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
