"""The ball: one position per frame, picked from the detector's ball-class boxes. Not a tracker.

Gates (shape, scenery, on the pitch, physical size, opt-in marking), then the most confident
survivor, with no memory of earlier picks. `Ball.xy` is the box CENTRE in the input pixels; a
frame where nothing survives is not seen and is never interpolated. A detected ball is a
position, not a possession or an event.
docs/reference/ball.md
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

# A sphere's box is square at every distance and angle; the range admits motion blur.
# ball-evidence (not published)#evidence-shape
ASPECT_MIN, ASPECT_MAX = 0.55, 1.8

# Sanity bounds on size, as a fraction of frame height. Below the floor it is
# noise; above the ceiling it is a person. Deliberately wide - this is a
# guardrail, not a fit.
SIZE_MIN_FRAC, SIZE_MAX_FRAC = 0.002, 0.05

#: A football's diameter in metres (IFAB Law 2), for the expected box size at a ground point.
#: ball-evidence (not published)#evidence-shape
BALL_DIAMETER_M = 0.22

# No speed gate, relock or position buffer: each temporal term lowered the score.
# ball-evidence (not published)#evidence-gates

@dataclass
class Ball:
    """One frame's answer. `xy` is the box CENTRE in pixels.

    Centre, not the bottom edge: a player's ground point is their feet, but a
    ball in flight is not on the ground and its bottom edge means nothing.
    Converting to pitch metres is therefore only valid while the ball is ON the
    ground, and this module does not attempt it - a calibration turns a ground
    point into metres, and a ball at head height is not one.
    """

    f: int
    xy: tuple[float, float]
    conf: float
    wh: tuple[float, float]
    #: False when nothing passed the gates this frame - the ball is occluded,
    #: out of frame, or was never detected. NOT interpolated: inventing a
    #: position is how an occlusion becomes fake distance travelled.
    seen: bool = True


def aspect_ok(w: float, h: float) -> bool:
    """A round object's box is square. The measured discriminator."""
    if w <= 0 or h <= 0:
        return False
    return ASPECT_MIN <= (w / h) <= ASPECT_MAX


def plausible(box: tuple[float, float, float, float], frame_h: int) -> bool:
    """Shape and size gates, before any tracking decision is made."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if not aspect_ok(w, h):
        return False
    big = max(w, h) / max(frame_h, 1)
    return SIZE_MIN_FRAC <= big <= SIZE_MAX_FRAC


#: Marking gate (opt-in): reject a candidate whose white blob is longer than this many boxes.
#: ball-evidence (not published)#evidence-marking
MAX_BLOB_EXTENT = 2.5


def blob_extent(frame: np.ndarray, box: tuple[float, float, float, float],
                window: float = 6.0) -> float | None:
    """Longest side of the white blob under `box`, in multiples of the box; None when the window
    has no pixels. One frame only, so a panning camera cannot spoil it. It costs true balls on
    every clip with ball truth, which is why the gate is opt-in.
    ball-evidence (not published)#evidence-marking

    ponytail: a local white-vs-grass threshold, not lines.paint_mask. paint_mask
    wants a whole frame and its largest green component; the question here is
    local and runs per candidate per frame, which is a different cost.
    """
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1, 1.0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    r = int(round(window * side / 2))
    X0, Y0 = max(0, int(cx - r)), max(0, int(cy - r))
    win = frame[Y0:int(cy + r), X0:int(cx + r)]
    if win.size == 0 or min(win.shape[:2]) < 5:
        return None

    hsv = cv2.cvtColor(win, cv2.COLOR_BGR2HSV)
    # Unsaturated AND bright relative to THIS window: grass is saturated green,
    # paint and a ball are neither. A global threshold would move with the
    # floodlights and the shadow of the stand.
    mask = ((hsv[:, :, 1] < 90) &
            (hsv[:, :, 2] > np.percentile(hsv[:, :, 2], 75))).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    px = min(max(int(cx) - X0, 0), mask.shape[1] - 1)
    py = min(max(int(cy) - Y0, 0), mask.shape[0] - 1)
    i = int(lab[py, px])
    if i == 0:
        # The exact centre pixel is not white - a dark seam on the ball, or a
        # box off by a pixel. Take the nearest white component instead of
        # calling it clean, because "no white here" would PASS the gate and
        # that is the wrong way to fail.
        yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
        near = (np.hypot(xx - px, yy - py) <= side) & (lab > 0)
        if not near.any():
            return 0.0
        i = int(np.bincount(lab[near]).argmax())
    return max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) / side


class Scenery:
    """Flags candidates that some candidate sat near, in pixels, for most of the recent picker
    calls: paint and fixed markings, while the camera holds still.
    ball-evidence (not published)#evidence-scenery

    ponytail: a fixed radius in pixels, and it assumes the camera is roughly
    still over the window. A hard pan invalidates it for `window` frames, which
    costs recall and never costs a false ball. Upgrade path is to register the
    frames first (pancheck) and ask the question in pitch metres.
    """

    def __init__(self, radius: float = 8.0, window: int = 60, share: float = 0.9,
                 min_frames: int = 25):
        # min_frames: calls held before any flag, so a ball briefly at rest is not scenery.
        # `window` and `min_frames` count picker calls, not video frames: under a stride they
        # span stride times more video.
        # ball-evidence (not published)#evidence-scenery
        self.radius, self.window, self.share = radius, window, share
        self.min_frames = min_frames
        self._hist: deque[np.ndarray] = deque(maxlen=window)

    def flags(self, xy: np.ndarray) -> np.ndarray:
        """(N,) True where this candidate looks like a fixed marking."""
        out = np.zeros(len(xy), bool)
        if len(self._hist) >= self.min_frames and len(xy):
            hits = np.zeros(len(xy), int)
            for past in self._hist:
                if len(past):
                    d = np.hypot(*(xy[:, None, :] - past[None, :, :]).transpose(2, 0, 1))
                    hits += (d.min(1) <= self.radius).astype(int)
            out = hits >= self.share * len(self._hist)
        self._hist.append(np.asarray(xy, float).reshape(-1, 2))
        return out


class BallPicker:
    """Pick the one ball from a frame's candidates: gates, then argmax(conf). MEMORYLESS.

    Without `cal` only the shape and scenery gates run; on-the-pitch and physical size need it.
    docs/reference/ball.md#entry-point
    ball-evidence (not published)#evidence-gates
    """

    #: Observed box over the expected ball size at that ground point. Wide: boxes run large
    #: on a small object, and an airborne ball is above the ground plane.
    #: ball-evidence (not published)#evidence-gates
    SIZE_RATIO = (0.4, 3.0)
    #: Metres off the pitch rectangle a ball may be: throw-ins are taken from outside the line.
    MARGIN_M = 3.0

    def __init__(self, cal=None, scenery: "Scenery | None" = None,
                 marking_gate: bool = False):
        self.cal = cal
        self.scenery = scenery if scenery is not None else Scenery()
        # Off by default: it costs true balls on every clip with ball truth.
        # ball-evidence (not published)#evidence-marking
        self.marking_gate = marking_gate

    def _expected_px(self, xy: np.ndarray) -> np.ndarray:
        """Pixels that BALL_DIAMETER_M spans at each ground point, asked of the calibration's own
        map: through `jacobian` where the calibrator class has one (its `H` is then a stand-in),
        else back through inv(H) (perspective, remap).
        ball-evidence (not published)#evidence-gates

        ponytail: `jacobian` is looked up on the CLASS, so UndistortWrapper/BowWrapper (which
        delegate attributes to their inner calibrator) take the H path and miss their own
        pre-warp. Neither is written by the product; give them a jacobian if one ever is.
        """
        if getattr(type(self.cal), "jacobian", None) is not None:
            J = np.asarray(self.cal.jacobian(xy), float).reshape(-1, 2, 2)
            return np.hypot(*(np.linalg.pinv(J) @ np.array([BALL_DIAMETER_M, 0.0])).T)
        m = self.cal.to_pitch(xy)
        Hi = np.linalg.inv(np.asarray(self.cal.H, float))
        q = np.c_[m + np.array([BALL_DIAMETER_M, 0.0]), np.ones(len(m))] @ Hi.T
        w = np.where(np.abs(q[:, 2:3]) < 1e-9, 1e-9, q[:, 2:3])
        return np.hypot(*((q[:, :2] / w) - xy).T)

    def update(self, dets: list[dict], frame_h: int, f: int = 0,
               frame: np.ndarray | None = None) -> Ball:
        cands = [d for d in dets
                 if d.get("cls") in ("sports ball", "ball") and plausible(d["xyxy"], frame_h)]
        xy = np.array([[(d["xyxy"][0] + d["xyxy"][2]) / 2,
                        (d["xyxy"][1] + d["xyxy"][3]) / 2] for d in cands],
                      float).reshape(-1, 2)

        # ALWAYS fed, even on a frame nothing survives: the scenery memory is
        # about where candidates KEEP appearing, and skipping the frames the
        # picker gives up on would blind it to exactly the repeat offenders.
        keep = ~self.scenery.flags(xy)

        if keep.any() and self.cal is not None:
            m = self.cal.to_pitch(xy)
            p = self.cal.pitch
            keep &= ((m[:, 0] > -self.MARGIN_M) & (m[:, 0] < p.length + self.MARGIN_M) &
                     (m[:, 1] > -self.MARGIN_M) & (m[:, 1] < p.width + self.MARGIN_M))
            if keep.any():
                obs = np.array([max(d["xyxy"][2] - d["xyxy"][0],
                                    d["xyxy"][3] - d["xyxy"][1]) for d in cands])
                ratio = obs / np.maximum(self._expected_px(xy), 1e-6)
                keep &= (ratio >= self.SIZE_RATIO[0]) & (ratio <= self.SIZE_RATIO[1])

        # Marking gate: a single-frame test, which a panning camera cannot defeat.
        # ball-evidence (not published)#evidence-marking
        if keep.any() and frame is not None and self.marking_gate:
            for j in np.flatnonzero(keep):
                e = blob_extent(frame, cands[j]["xyxy"])
                if e is not None and e > MAX_BLOB_EXTENT:
                    keep[j] = False

        if not keep.any():
            # Not seen rather than a guess: everything downstream would believe a wrong ball.
            # ball-evidence (not published)#evidence-picker
            return Ball(f=f, xy=(float("nan"), float("nan")), conf=0.0,
                        wh=(0.0, 0.0), seen=False)

        idx = np.flatnonzero(keep)
        i = int(idx[np.argmax([cands[j]["conf"] for j in idx])])
        b = cands[i]
        return Ball(f=f, xy=(float(xy[i][0]), float(xy[i][1])), conf=float(b["conf"]),
                    wh=(b["xyxy"][2] - b["xyxy"][0], b["xyxy"][3] - b["xyxy"][1]))


def _selfcheck() -> None:
    """The gates are the product here, so they are what gets asserted."""
    # The measured discriminator: DFL boxes pass, panorama boxes do not.
    assert aspect_ok(8.7, 9.0), "the DFL ball must pass"
    assert not aspect_ok(53.8, 12.7), "the panorama 4:1 box must NOT pass"
    assert aspect_ok(12.0, 8.0), "motion blur along travel must still pass"

    H = 1080
    assert plausible((100, 100, 109, 109), H)
    assert not plausible((100, 100, 154, 113), H)          # 4:1, a leg
    assert not plausible((100, 100, 101, 101), H)          # 1 px, noise
    assert not plausible((100, 100, 200, 200), H)          # 100 px, a person

    ball = lambda x, y, c=0.3, s=9: {  # noqa: E731
        "xyxy": (x, y, x + s, y + s), "conf": c, "cls": "sports ball"}

    # THE CENTRE-SPOT FAILURE: a bright decoy that never moves must lose to a dimmer thing
    # that does, once the scenery memory has seen it sit still.
    p = BallPicker()
    for i in range(40):
        p.update([ball(500, 500, 0.9), ball(100 + 3 * i, 200, 0.2)], H, i)
    got = p.update([ball(500, 500, 0.9), ball(220, 200, 0.2)], H, 40)
    assert got.seen and abs(got.xy[0] - 224.5) < 1, f"locked onto the static decoy: {got}"

    # ...and a ball that genuinely sits still for a moment is NOT scenery on the
    # strength of a few frames - the window is 60 and the share is 0.9.
    q = BallPicker()
    for i in range(5):
        q.update([ball(500, 500, 0.9)], H, i)
    assert q.update([ball(500, 500, 0.9)], H, 5).seen, "a briefly still ball was discarded"

    # Nothing plausible -> not seen, and NOT interpolated. Inventing a position
    # is how an occlusion becomes fake distance travelled.
    assert not BallPicker().update([ball(0, 0, 0.9, 400)], H, 0).seen

    # MEMORYLESS: two identical frames give the same answer whatever came before them.
    a, b = BallPicker(), BallPicker()
    for i in range(10):
        b.update([ball(900, 900, 0.9)], H, i)     # b has a long, wrong history
    frame = [ball(300, 300, 0.4), ball(310, 305, 0.8)]
    assert a.update(frame, H, 99).xy == b.update(frame, H, 99).xy, \
        "the picker's answer depends on its history"

    # THE MARKING GATE. A synthetic pitch: green, with one white stripe and one
    # white disc the size of a ball. The stripe must be rejected however
    # confident it is, and the disc kept - this is the centre-circle failure
    # that scenery cannot see once the camera pans.
    img = np.full((400, 400, 3), (60, 160, 60), np.uint8)
    cv2.rectangle(img, (200, 0), (206, 399), (245, 245, 245), -1)   # a painted line
    cv2.circle(img, (100, 100), 5, (245, 245, 245), -1)             # a ball
    line_box = (197.0, 197.0, 209.0, 209.0)
    ball_box = (94.0, 94.0, 106.0, 106.0)
    assert blob_extent(img, line_box) > MAX_BLOB_EXTENT, blob_extent(img, line_box)
    assert blob_extent(img, ball_box) < MAX_BLOB_EXTENT, blob_extent(img, ball_box)

    g = BallPicker(marking_gate=True)
    picked = g.update([{"xyxy": line_box, "conf": 0.9, "cls": "sports ball"},
                       {"xyxy": ball_box, "conf": 0.2, "cls": "sports ball"}],
                      400, 0, img)
    assert picked.seen and abs(picked.xy[0] - 100) < 3, f"took the line marking: {picked}"

    # ...and it is OFF unless asked for, because it costs real balls on the
    # footage this repo has truth for. Off, the line marking is taken.
    assert BallPicker().update([{"xyxy": line_box, "conf": 0.9, "cls": "sports ball"}],
                               400, 0, img).seen
    # ...and with no frame it cannot run at all, rather than erroring: a caller
    # that has only boxes still gets the other gates.
    assert BallPicker(marking_gate=True).update(
        [{"xyxy": line_box, "conf": 0.9, "cls": "sports ball"}], 400, 0).seen

    # The physical size gate needs a calibration and must reject a box that is
    # the wrong size to be 0.22 m at that point on the ground.
    class _Cal:
        # H maps IMAGE -> PITCH, the same direction Calibrator.to_pitch goes, so
        # the two must agree: 8 px per metre is H = diag(1/8, 1/8, 1). A stub
        # whose H and to_pitch disagree tests nothing and passes anyway.
        H = np.array([[0.125, 0.0, 0.0], [0.0, 0.125, 0.0], [0.0, 0.0, 1.0]])
        pitch = type("P", (), {"length": 105.0, "width": 68.0})()

        @staticmethod
        def to_pitch(pts):
            return np.asarray(pts, float) / 8.0

    c = BallPicker(cal=_Cal())
    assert abs(c._expected_px(np.array([[200.0, 200.0]]))[0] - 8 * 0.22) < 1e-6, \
        c._expected_px(np.array([[200.0, 200.0]]))
    # 8 px per metre, so 0.22 m is 1.76 px; SIZE_RATIO admits up to 3x that.
    assert not c.update([ball(200, 200, 0.9, 40)], H, 0).seen, "a 40 px 'ball' passed"
    assert c.update([ball(200, 200, 0.9, 3)], H, 1).seen

    # A calibration whose H is only a stand-in (SphericalCamera) must be asked through its own
    # jacobian. Here the map is 4 px/m and H claims 8.
    class _Sph(_Cal):
        @staticmethod
        def to_pitch(pts):
            return np.asarray(pts, float) / 4.0

        @staticmethod
        def jacobian(pts):
            return np.tile(np.eye(2) / 4.0, (len(np.asarray(pts).reshape(-1, 2)), 1, 1))

    got = BallPicker(cal=_Sph())._expected_px(np.array([[200.0, 200.0]]))[0]
    assert abs(got - 4 * 0.22) < 1e-9, f"expected size read through H, not the map: {got}"

    # ...and reject anything off the pitch, however confident. 68 m at 8 px/m
    # puts the far touchline at 544 px, so 2000 px is deep in the stands.
    assert not BallPicker(cal=_Cal()).update([ball(2000, 2000, 0.99, 3)], H, 0).seen, \
        "a ball in the stands passed"

    print("ball.py ok")


if __name__ == "__main__":
    _selfcheck()
