"""Draw our tracks on an STC2025 clip, next to the ground truth we are scored on.

WHY A SECOND OVERLAY
    `model/engine/overlay.py` is the product renderer and it needs a calibration -
    it draws a minimap and pitch lines. These clips are scored in RAW panorama
    pixels with no calibration (and 132831's shipped homography is degenerate
    anyway), so `render()` cannot run on them. The part worth reusing is the
    drawing convention, not the pipeline: `marker` and `TEAM_BGR` are imported,
    never re-drawn here, so a ring means the same thing in both.

    Ours = team-coloured ring + track id. GT = thin grey box + player id. Where
    a grey box has no ring, we missed a player; where a ring has no grey box, we
    invented one (or it is the referee - the GT holds 22 players and no
    officials, so every referee we draw is a false positive by construction).

BOX CONVENTION
    tracks.json is BOTTOM-CENTRE x,y + w,h; the MOT GT is top-left. Getting this
    backwards is not visible as a crash, it just draws everything one box height
    low. `st2_team.to_tlwh` owns the conversion and demo() pins it to st2_mot.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overlay import (FONT, TEAM_BGR, UNKNOWN_BGR, densify, marker,  # noqa: E402
                     pitch_model)
from pitch import Pitch  # noqa: E402

_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download
GT_SRC = _ST2 + "/mot"
CLIPS = f"{GT_SRC}/clips"
from st2_mot import WORK as TRACKS  # noqa: E402 - root-relative, one definition
GT_BGR = (150, 150, 150)
LINE_BGR = (80, 240, 255)
RAW = _ST2 + "/raw"


def pitch_of(seq):
    """Pitch dimensions from the match's own metadata, not from our defaults.

    ST2 ships `<pitch width="105" height="68"/>`; `width` there is the LENGTH.
    Assuming 105x68 would be right for every match in this dataset and wrong
    the first time it is not, silently, by a metre.
    """
    x = open(f"{RAW}/{seq}/{seq}_tracker_box_metadata.xml", encoding="utf-8").read()
    m = re.search(r'<pitch width="([\d.]+)" height="([\d.]+)"', x)
    return Pitch(length=float(m.group(1)), width=float(m.group(2)))


def projector(seq, frame_wh):
    """pitch metres (corner origin) -> clip pixel, plus a validity mask.

    The chain is ST2's, not ours: a single pitch->panorama homography does NOT
    fit this footage because the panorama is STITCHED (st2_align.py: 103 px
    median, 11/65 inliers). The real chain is

        pitch metres --H--> remapped px --mapx lookup--> panorama px

    and it measures 7.8 px median on 118575's 65 keypoints. 128057 ships no
    keypoints file, so there is no number here - only the drawn line against
    the painted one, which is the check `overlay.pitch_lines` was written for.

    OUT-OF-TABLE VERTICES ARE DROPPED, NOT CLAMPED. Clamping a vertex to the
    edge of the remap table draws a confident straight line down the side of
    the image that reads as a calibration result rather than as missing data.
    """
    H = np.asarray(np.load(f"{RAW}/{seq}/{seq}_homography.npy"), np.float64)
    mx = np.load(f"{RAW}/{seq}/{seq}_mapx.npy")
    W, Hgt = frame_wh
    # A homography is defined only up to scale, so H and -H are the same
    # projection with every homogeneous w negated. Pin the sign on the centre
    # spot - in front of the camera for any calibration that means anything -
    # before trusting w > 0, or the test discards every vertex instead of none.
    if (np.array([52.5, 34.0, 1.0]) @ H.T)[2] < 0:
        H = -H

    def fn(xy):
        A = np.asarray(xy, np.float64).reshape(-1, 2)
        Ph = np.column_stack([A, np.ones(len(A))]) @ H.T
        w = Ph[:, 2]
        # BEHIND THE CAMERA. A point behind the lens still lands on a perfectly
        # ordinary pixel, so without this a pitch corner off the end of a
        # touchline rig draws a confident line across the crowd.
        ok = w > 1e-9
        wsafe = np.where(ok, w, 1.0)
        R = np.column_stack([Ph[:, 0] / wsafe, Ph[:, 1] / wsafe])
        u = np.round(R[:, 0]).astype(int)
        v = np.round(R[:, 1]).astype(int)
        ok &= (u >= 0) & (u < mx.shape[1]) & (v >= 0) & (v < mx.shape[0])
        out = np.full((len(u), 2), -1.0)
        uu, vv = np.clip(u, 0, mx.shape[1] - 1), np.clip(v, 0, mx.shape[0] - 1)
        out[:, 0] = mx[vv, uu, 0]
        out[:, 1] = mx[vv, uu, 1]
        ok &= (out[:, 0] >= 0) & (out[:, 0] < W) & (out[:, 1] >= 0) & (out[:, 1] < Hgt)
        return out, ok

    return fn


def line_polys(seq, frame_wh):
    """Pitch markings as clip-pixel polylines, split at every dropped vertex."""
    fn = projector(seq, frame_wh)
    out = []
    for poly in (densify(p) for p in pitch_model(pitch_of(seq))):
        pts, ok = fn(poly)
        run = []
        for p, good in zip(pts, ok):
            if good:
                run.append(p)
            elif len(run) > 1:
                out.append(np.array(run, np.int32)); run = []
            else:
                run = []
        if len(run) > 1:
            out.append(np.array(run, np.int32))
    return out


def load(seq, variant, f0, f1):
    ours = defaultdict(list)
    for d in json.load(open(f"{TRACKS}/{seq}.{variant}.tracks.json"))["detections"]:
        if f0 <= d["f"] < f1:
            ours[d["f"]].append(d)
    gt = defaultdict(list)
    with open(f"{GT_SRC}/{seq}.txt") as fh:
        for line in fh:
            p = line.split(",")
            f = int(p[0])
            if f0 <= f < f1:
                gt[f].append((int(p[1]), *(float(v) for v in p[2:6])))
    return ours, gt


def run(seq, variant, start_s, dur_s, out, width, show_gt=True, lines=False):
    cap = cv2.VideoCapture(f"{CLIPS}/{seq}.mp4")
    if not cap.isOpened():
        raise SystemExit(f"cannot open clip {seq}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f0 = int(round(start_s * fps))
    f1 = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), f0 + int(round(dur_s * fps)))
    ours, gt = load(seq, variant, f0, f1)

    scale = width / W
    size = (width, int(round(H * scale)))
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"avc1"), fps, size)
    if not vw.isOpened():                      # avc1 is absent on some builds
        vw = cv2.VideoWriter(out.replace(".mp4", ".avi"),
                             cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
        out = out.replace(".mp4", ".avi")

    polys = line_polys(seq, (W, H)) if lines else []
    if lines:
        drawn = sum(len(p) for p in polys)
        want = sum(len(densify(p)) for p in pitch_model(pitch_of(seq)))
        print(f"pitch lines: {len(polys)} polylines, {drawn}/{want} vertices "
              f"inside the remap table")

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    n = 0
    for f in range(f0, f1):
        ok, img = cap.read()
        if not ok:
            break
        for poly in polys:
            cv2.polylines(img, [poly], False, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.polylines(img, [poly], False, LINE_BGR, 2, cv2.LINE_AA)
        if show_gt:
            for gid, x, y, w, h in gt[f]:
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                              GT_BGR, 2, cv2.LINE_AA)
                cv2.putText(img, str(gid), (int(x), int(y) - 4), FONT, 0.5,
                            GT_BGR, 2, cv2.LINE_AA)
        for d in ours[f]:
            # x,y IS the ground point - bottom-centre. No arithmetic to get wrong.
            marker(img, int(round(d["x"])), int(round(d["y"])), d["w"], d["h"],
                   str(d["id"]), TEAM_BGR.get(d.get("team"), UNKNOWN_BGR))
        cv2.putText(img, f"{seq} .{variant}  f{f}  {f / fps:6.2f}s"
                    f"   ours={len(ours[f])} gt={len(gt[f])}",
                    (12, 30), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(cv2.resize(img, size, interpolation=cv2.INTER_AREA))
        n += 1
    cap.release()
    vw.release()
    print(f"{out}  {n} frames  {n / fps:.1f}s  {size[0]}x{size[1]}")
    return out


NAMES = ["outline", "halfway", "centre circle", "penalty L", "penalty R",
         "goal box L", "goal box R"]


def check(seq, frames=(300, 1500, 3000, 4500), search_px=30, step=0.5):
    """Signed PERPENDICULAR residual: drawn marking vs painted marking, in px.

    The first version of this used a distance transform - distance to the
    nearest white pixel in ANY direction - and reported 3.3 px median on
    128057. That number was too good, and the render disagreed with it: the
    right penalty box and the centre circle are visibly drawn OUTSIDE the
    paint. A nearest-white-pixel search will happily match the goal frame, the
    "University of Tsukuba" lettering outside the touchline, a white-shirted
    player, or the ADJACENT line of the same box, so a line drawn 15 px off can
    score 2 px against its neighbour.

    Two changes fix it, and both matter:

      - Search only along the line's NORMAL, +/- search_px. A line can then
        only match paint it is actually perpendicular to.
      - Report the residual SIGNED. An unsigned median hides exactly the defect
        the render shows: a centre circle drawn uniformly too large averages to
        a small error in every direction at once. A signed median does not.

    Paint outside the pitch is masked out, which removes the lettering, the
    stands and the goal frames. Unmatched vertices are reported, never folded
    into the median: a goal line under a crowd has no paint to be near.

    SIGN. The normal is (-ty, tx) on the direction of travel, so its meaning is
    per-marking, not global. `pitch_model` walks the centre circle
    anticlockwise, which makes +normal point INWARD there - so the circle's
    +5.0 means the paint is inside the drawn line, i.e. we draw the circle too
    large. That is what the render shows.

    128057, four frames:

        marking        signed  |med|   p90  unmatched
        halfway          -1.5    1.5   4.0       1.2%   <- on the paint
        outline          +0.0    3.0  17.5      32.8%
        centre circle    +5.0    7.0  13.0      29.8%   <- one-sided
        penalty R        +0.0    5.0  22.0      22.9%
        ALL              +0.0    3.0  16.0

    The circle is not a wrong radius. Sweeping it from 8.80 m to 9.45 m moves
    the SIGNED residual monotonically (+2.0 to +9.0) but the absolute one
    floors at 6.5 px and never reaches the paint's own half-width, so no radius
    fits. It is local calibration error near the image centre - what a single
    global homography costs on a STITCHED panorama - and it is the same order
    as st2_align.py's 7.8 px keypoint median on 118575.

    The unmatched shares are high here and were near zero under the distance
    transform. That is the point: most of those vertices never had paint
    perpendicular to them, and the old method was scoring them against
    whatever white pixel happened to be nearest.
    """
    cap = cv2.VideoCapture(f"{CLIPS}/{seq}.mp4")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fn = projector(seq, (W, H))
    P = pitch_of(seq)
    groups = [densify(g) for g in pitch_model(P)]

    # On-pitch mask: the touchline needs paint AT the boundary, so allow a
    # small margin, but far less than the distance to the lettering.
    e = 1.5
    xs = np.linspace(-e, P.length + e, 300); ys = np.linspace(-e, P.width + e, 200)
    ring = np.concatenate([np.stack([xs, np.full_like(xs, -e)], 1),
                           np.stack([np.full_like(ys, P.length + e), ys], 1),
                           np.stack([xs[::-1], np.full_like(xs, P.width + e)], 1),
                           np.stack([np.full_like(ys, -e), ys[::-1]], 1)])
    rp, rok = fn(ring)
    inside = np.zeros((H, W), np.uint8)
    if rok.any():
        cv2.fillPoly(inside, [rp[rok].astype(np.int32)], 1)

    ts = np.arange(-search_px, search_px + step, step)
    per = {n: [] for n in NAMES}
    seen = {n: [0, 0] for n in NAMES}
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        paint = ((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 150) & (inside > 0))
        for name, g in zip(NAMES, groups):
            pts, good = fn(g)
            for i in range(1, len(pts) - 1):
                if not (good[i] and good[i - 1] and good[i + 1]):
                    continue
                tx, ty = pts[i + 1] - pts[i - 1]
                L = float(np.hypot(tx, ty))
                if L < 1e-6:
                    continue
                nx, ny = -ty / L, tx / L          # unit normal
                seen[name][1] += 1
                u = np.round(pts[i][0] + ts * nx).astype(int)
                v = np.round(pts[i][1] + ts * ny).astype(int)
                m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                hit = np.zeros(len(ts), bool)
                hit[m] = paint[v[m], u[m]]
                if hit.any():
                    per[name].append(ts[np.argmin(np.where(hit, np.abs(ts), 1e9))])
                else:
                    seen[name][0] += 1
    cap.release()
    print(f"\n{seq}: drawn vs painted, signed perpendicular, {len(frames)} frames")
    print(f"{'marking':>14s} {'signed':>8s} {'|med|':>7s} {'p90':>7s} {'unmatched':>10s}")
    allv = []
    for n in NAMES:
        v = np.array(per[n]); allv += list(v)
        u, t = seen[n]
        if len(v):
            print(f"{n:>14s} {np.median(v):+8.1f} {np.median(np.abs(v)):7.1f} "
                  f"{np.percentile(np.abs(v), 90):7.1f} {100 * u / max(1, t):9.1f}%")
        else:
            print(f"{n:>14s} {'-':>8s} {'-':>7s} {'-':>7s} {100 * u / max(1, t):9.1f}%")
    if not allv:
        print(f"{'ALL':>14s}   no vertex found paint on its normal within "
              f"{search_px} px - calibration unusable")
        return float("nan")
    a = np.abs(allv)
    print(f"{'ALL':>14s} {np.median(allv):+8.1f} {np.median(a):7.1f} "
          f"{np.percentile(a, 90):7.1f}")
    return float(np.median(a))


def demo():
    from st2_team import to_tlwh
    assert to_tlwh({"x": 100.0, "y": 200.0, "w": 20.0, "h": 40.0}) \
        == (90.0, 160.0, 20.0, 40.0), "box convention drifted from st2_team"
    assert TEAM_BGR["home"] != TEAM_BGR["away"] != TEAM_BGR["official"]
    # ST2 calls the 105 m axis `width`; pitch_of must not hand it to Pitch.width.
    p = pitch_of("128057")
    assert p.length > p.width, f"pitch axes swapped: {p.length}x{p.width}"
    # A dropped vertex must break the polyline, never snap to the frame edge.
    fn = projector("128057", (4096, 1080))
    corners = [(52.5, 34.0), (0, 0), (105, 0), (105, 68), (0, 68)]
    pts, ok = fn(corners)
    assert ok.all(), f"pitch corners must project into the clip: {ok}"
    assert 0 < pts[0][0] < 4096 and 0 < pts[0][1] < 1080, pts[0]
    # A vertex that leaves the remap table is DROPPED, never snapped to an edge.
    _, bad = fn([(1e6, 34.0)])
    assert not bad[0], "an out-of-table vertex was accepted"
    # Densifying must add points without moving the ends of the line.
    d = densify([(0.0, 0.0), (105.0, 0.0)], step_m=0.5)
    assert len(d) > 200 and d[0] == (0.0, 0.0) and d[-1] == (105.0, 0.0), len(d)
    assert densify([(0.0, 0.0), (1.0, 0.0)])[0] == (0.0, 0.0)
    print("ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="128057")
    ap.add_argument("--variant", default="c03s")
    ap.add_argument("--start", type=float, default=120.0)
    ap.add_argument("--dur", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--no-gt", action="store_true")
    ap.add_argument("--lines", action="store_true",
                    help="draw the pitch model through ST2's calibration chain")
    ap.add_argument("--out", default="")
    ap.add_argument("--check", action="store_true",
                    help="measure drawn-vs-painted residual instead of rendering")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        demo()
    elif a.check:
        check(a.seq)
    else:
        out = a.out or f"model/validation/out/stc2025/{a.seq}.{a.variant}.overlay.mp4"
        run(a.seq, a.variant, a.start, a.dur, out, a.width, not a.no_gt, a.lines)
