"""Does this footage hold still enough for ONE calibration?

WHY THIS EXISTS
    Every calibration in this repo is a single 3x3 solved once and reused for a
    whole match. That is correct for a bolted-down camera and WRONG BY
    CONSTRUCTION for anything an operator drives, and the wrongness is silent:
    the homography keeps returning finite, plausible-looking metres long after
    it stopped describing the camera. calcheck tells you how good a calibration
    is on the frame it was solved on. Nothing told you how long it stays true.

    SCOPE: this repository targets a single FIXED panoramic camera. Panning,
    operator-driven and broadcast footage is OUT OF SCOPE; this module stays as
    the instrument that tells you a clip is outside that scope (overlay.py uses
    it), not as support for such footage.

    That question decides whether a format is supportable at all, so it should
    be a measurement and not an assumption. Twice this session a claim about
    this pipeline turned out to rest on an unchecked belief about the footage.

WHAT IT MEASURES, AND WHY IT IS NOT CIRCULAR
    Reuse a calibration solved at frame A on a later frame B, and ask how far
    off it is IN METRES. The trick is that this needs no ground truth and no
    second calibration: a point painted on the pitch does not move, so if the
    same physical point sits at pixel p in frame A and pixel p' in frame B, then
    cal(p') - cal(p) is exactly the error you would make by reusing cal. The
    frame-to-frame map that carries p to p' comes from SIFT + RANSAC on the
    static scene - stands, hoardings, paint - with the players outliered away.
    Nothing in that chain is fitted to the calibration, which is what keeps this
    from being another guard that agrees with itself.

    It is a LOWER bound on the damage. It captures the camera moving. It does
    not capture the calibration being wrong to begin with; that is calcheck's
    job, and both numbers are needed.

MEASURED during development on an external 30 s DFL broadcast-style clip
(1920x1080, 25 fps). That footage is NOT bundled with this repository.
    Calibrate at t=0 and reuse it (21 on-pitch probes, median error):

        t =  1 s   0.29 m      still inside a stride
        t =  2 s   0.98 m
        t =  3 s   3.57 m
        t =  8 s  12.81 m      a fifth of the pitch length
        t = 29 s  20.99 m  (max 35.55 m)

    So on an operator-driven camera one homography is worth about TWO SECONDS.
    Not "degrades gracefully" - a pan is a step function, and this clip goes
    from a third of a metre to three and a half between t=1 s and t=3 s. Note
    also that the error comes BACK DOWN when the operator swings back through
    the original framing (4.0 m at t=20 s, having been 12.8 m at t=8 s), so a
    guard that samples one late frame and finds a small number proves nothing.

    And the reason the camera moves at all matters as much as the movement:
    across the same 30 s, projecting a 1 m pitch grid through the per-frame
    geometry puts only 52-61% OF THE PITCH IN SHOT at any instant (median 57%).
    It is a ball-following tactical camera, not a wide camera. So even a perfect
    per-frame homography cannot produce a per-player distance total here - for
    roughly half the players there is no pixel to measure.

    Reproduce:  python model/engine/pancheck.py VIDEO CALIBRATION.json
    (on a panning clip and calibration of your own; none is bundled)

WHAT THIS FILE IS NOT
    It is not a tracker and not a stabiliser. Registering frame N against a
    reference is how the QUESTION is asked; it is not a proposed fix. Chaining
    image registration over a half accumulates drift without bound, which is
    why per-frame geometry would have to re-anchor to markings whose metres are
    known, which does not drift (lines.py finds the markings; panning is out of
    scope, so nothing here builds that geometry). Use this to decide whether a
    format needs per-frame geometry, then go and build the per-frame geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from pitch import Calibration, Pitch

#: Lowe ratio and RANSAC pixel threshold for the static-scene registration.
#: Both are the textbook defaults; neither was tuned to make an answer come out.
RATIO = 0.75
RANSAC_PX = 3.0

#: Below this many RANSAC inliers the registration is not evidence of anything.
#: Measured on the external DFL clip above (not bundled), a good pair returns
#: 250-4000 and the worst pairing
#: in the clip (t=0 against t=30, a 1.6x zoom apart) still returns 251.
MIN_INLIERS = 40


def _sift(nfeatures: int = 4000):
    return cv2.SIFT_create(nfeatures=nfeatures)


def register(a: np.ndarray, b: np.ndarray, sift=None) -> tuple[np.ndarray | None, int]:
    """Homography carrying pixels of frame `a` onto frame `b`, and its inliers.

    Grayscale in. This is a STATIC-SCENE registration: it is valid because the
    stands, hoardings and paint dominate the feature count and the players are
    a RANSAC minority. It says nothing about the scene being planar - a pure
    rotation/zoom of a rigid scene is exactly a homography, which is the case a
    tripod-mounted operator camera produces.
    """
    sift = sift or _sift()
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None, 0
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m, n in pairs if m.distance < RATIO * n.distance]
    if len(good) < MIN_INLIERS:
        return None, len(good)
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    n = 0 if mask is None else int(mask.sum())
    if H is None or not np.isfinite(H).all() or n < MIN_INLIERS:
        return None, n
    return H, n


def _warp(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(pts, np.float32).reshape(-1, 1, 2), H.astype(np.float32)).reshape(-1, 2)


def drift_m(cal, H_ref_to_now: np.ndarray, probe_px: np.ndarray) -> np.ndarray:
    """Metres of error from reusing `cal` after the camera has moved.

    `probe_px` are pixels in the REFERENCE frame - the frame `cal` was solved
    on - ideally the image positions of pitch locations you care about.
    `H_ref_to_now` carries those pixels to where the same physical points sit
    now. The error is the difference the calibration reports between the two,
    because the physical points did not move.
    """
    probe_px = np.asarray(probe_px, float).reshape(-1, 2)
    return np.hypot(*(cal.to_pitch(_warp(H_ref_to_now, probe_px)) - cal.to_pitch(probe_px)).T)


def pitch_probes(cal, image_wh: tuple[int, int], nx: int = 8, ny: int = 5,
                 margin_m: float = 5.0) -> np.ndarray:
    """Pixels of an evenly spread grid of ON-PITCH points visible in the frame.

    Probing in PITCH space rather than image space on purpose: an image-space
    grid puts half its points on the crowd, where nobody is ever measured, and
    a metre of error there is not a metre of error in a stat.
    """
    p = cal.pitch
    gx, gy = np.meshgrid(np.linspace(margin_m, p.length - margin_m, nx),
                         np.linspace(margin_m, p.width - margin_m, ny))
    px = _warp(np.linalg.inv(cal.H), np.column_stack([gx.ravel(), gy.ravel()]))
    w, h = image_wh
    inside = (px[:, 0] > 0) & (px[:, 0] < w) & (px[:, 1] > 0) & (px[:, 1] < h)
    return px[inside]


@dataclass
class Hold:
    """How long one calibration stays inside a tolerance."""

    tol_m: float
    fps: float
    n_probes: int
    #: (seconds, median_m, p90_m, max_m, inliers), reference frame first.
    samples: list[tuple[float, float, float, float, int]] = field(default_factory=list)

    @property
    def hold_s(self) -> float:
        """Seconds until the median error first exceeds tol_m. inf if never."""
        for t, med, *_ in self.samples:
            if med > self.tol_m:
                return t
        return float("inf")

    @property
    def verdict(self) -> str:
        h = self.hold_s
        if h == float("inf"):
            return "fixed camera - one calibration is valid for the whole clip"
        if h < 60:
            return f"operator-driven - one calibration is worth {h:.1f} s, per-frame geometry required"
        return f"drifts slowly - one calibration holds {h:.0f} s"


@dataclass
class Consistency:
    """Self-consistency of a per-frame calibration, in metres, per frame lag.

    THE TWO LAGS ANSWER DIFFERENT QUESTIONS AND ONLY ONE OF THEM IS ABOUT DRIFT.

      k = 1     JITTER. Frame-to-frame disagreement, and the number that decides
                whether distance is measurable at all. Distance is a sum of
                displacements, so independent noise of sigma per axis fabricates
                distance at roughly (sigma * f / v)^2 - measured elsewhere as
                23.9 km of phantom running per 90 min at sigma = 0.10 m and
                25 Hz. A method can look excellent at k = 750 and be useless.
      k = 750   DRIFT. Where the whole clip has ended up. This is what `hold()`
                already measures for ONE reused calibration.

    Published methods are ranked on per-frame metrics that cannot see k = 1 at
    all: PTZ-SLAM's per-frame-independent variant beats its own filtered version
    on MEAN pan error (0.10 vs 0.17 deg) and loses 16x on the MAXIMUM (15.89 vs
    0.96 deg). Mean is what leaderboards score; the tail is what destroys a sum
    of displacements. So this instrument reports the spread, not just the middle.
    """

    fps: float
    n_probes: int
    #: lag (frames) -> (median_m, p90_m, max_m, n_samples)
    lags: dict[int, tuple[float, float, float, int]] = field(default_factory=dict)

    def row(self, k: int) -> str:
        med, p90, mx, n = self.lags[k]
        return (f"  lag {k:5d} ({k / self.fps:6.2f} s)   median {med:7.4f} m   "
                f"p90 {p90:7.4f} m   max {mx:8.4f} m   n={n}")

    def __str__(self) -> str:
        return "\n".join(self.row(k) for k in sorted(self.lags))


def consistency(video: str, cal_at, lags=(1, 25, 750), samples: int = 20,
                nx: int = 8, ny: int = 5, sift=None) -> Consistency:
    """Does a per-frame calibration agree with itself about where the pitch is?

    `cal_at(f)` returns the Calibration valid for frame `f`, or None to skip it.
    Pass `lambda _f: cal` to score a single reused calibration, which is the
    instrument's own control: with one calibration this reduces EXACTLY to
    `drift_m`, so a disagreement between the two is a bug in this function.

    THE MEASUREMENT NEEDS NO GROUND TRUTH, which is the entire point - panning
    clips of this kind usually carry no labels of any kind. A static point on the pitch does not
    move. So carry its pixels from frame t to frame t+k with a static-scene
    registration (`register`, SIFT on stands and paint, players are a RANSAC
    minority) and ask the two frames' calibrations where it is. Both answers
    are in metres and they should be the same answer. The residual IS the
    calibration's error, with nothing to compare against.

    Probes are ON-PITCH (`pitch_probes`), because a metre of error in the crowd
    is not a metre of error in a stat.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sift = sift or _sift()
        out = Consistency(fps=fps, n_probes=0)

        def gray(i):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if ok else None

        for k in lags:
            if k >= total:
                continue
            # Spread the sample over the clip rather than taking the first N:
            # a pan is not uniform, and the first second of a broadcast clip is
            # usually the least dynamic part of it.
            starts = np.linspace(0, total - k - 1, samples).astype(int)
            errs = []
            for t in starts:
                ca, cb = cal_at(int(t)), cal_at(int(t + k))
                if ca is None or cb is None:
                    continue
                a, b = gray(int(t)), gray(int(t + k))
                if a is None or b is None:
                    continue
                M, _n = register(a, b, sift)
                if M is None:
                    continue
                probes = pitch_probes(ca, (w, h), nx, ny)
                if not len(probes):
                    continue
                out.n_probes = max(out.n_probes, len(probes))
                d = np.hypot(*(np.asarray(cb.to_pitch(_warp(M, probes)), float)
                               - np.asarray(ca.to_pitch(probes), float)).T)
                errs.append(d)
            if errs:
                e = np.concatenate(errs)
                out.lags[k] = (float(np.median(e)), float(np.percentile(e, 90)),
                               float(e.max()), int(len(e)))
        return out
    finally:
        cap.release()


def centre_scatter(cal_at, frames, image_wh: tuple[int, int]) -> dict:
    """How far the recovered CAMERA CENTRE wanders. It physically cannot.

    The sharpest label-free test published: BroadTrack reports that NBJW's
    estimated focal point "can travel up to 20 meters along a single sequence"
    while a correct solve keeps it in a tight neighbourhood of the tripod. A
    camera bolted to a gantry that appears to move 20 m is a bad solve, proven
    with no annotation at all.

    One confound, stated because it will bite: for a distant pitch the camera
    POSITION and the FOCAL LENGTH are near-degenerate - uncertainty in one is
    absorbed by the other without visible perspective distortion (BroadTrack
    calls it the vertigo effect). So read the two together; a tight centre with
    a wandering focal is the same disease wearing different clothes.
    """
    import sceneprior  # noqa: PLC0415
    cs, fs = [], []
    for i in frames:
        c = cal_at(int(i))
        if c is None:
            continue
        try:
            cam = sceneprior.camera_from_homography(c.H, image_wh)
        except ValueError:
            continue                       # near-degenerate; not a centre at all
        cs.append(cam.centre_m)
        fs.append(cam.focal_px)
    if len(cs) < 2:
        return {"n": len(cs)}
    cs = np.asarray(cs, float)
    span = np.hypot(*(cs - np.median(cs, axis=0)).T)
    return {"n": len(cs), "centre_p50_m": float(np.median(span)),
            "centre_max_m": float(span.max()),
            "focal_px_median": float(np.median(fs)),
            "focal_px_spread": float(np.percentile(fs, 90) - np.percentile(fs, 10))}


def hold(video: str, cal, ref: int = 0, step: int = 25, limit: int | None = None,
         tol_m: float = 1.0) -> Hold:
    """Reuse `cal` (solved on frame `ref`) across the clip and time the decay.

    Every frame is compared against the REFERENCE, never against its
    predecessor. Chaining would let registration error compound and would
    measure the chain rather than the camera.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    want = set(range(ref, limit or total, step))
    grabbed: dict[int, np.ndarray] = {}
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            grabbed[i] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        i += 1
    cap.release()
    if ref not in grabbed:
        raise SystemExit(f"reference frame {ref} not in {video}")

    probes = pitch_probes(cal, wh)
    if len(probes) < 4:
        raise SystemExit("the calibration puts almost no pitch inside the frame")
    out = Hold(tol_m=tol_m, fps=fps, n_probes=len(probes))
    sift = _sift()
    for n in sorted(grabbed):
        if n == ref:
            out.samples.append(((n - ref) / fps, 0.0, 0.0, 0.0, 0))
            continue
        H, inl = register(grabbed[ref], grabbed[n], sift)
        if H is None:
            continue                       # unregisterable: no evidence, not zero error
        e = drift_m(cal, H, probes)
        out.samples.append(((n - ref) / fps, float(np.median(e)),
                            float(np.percentile(e, 90)), float(e.max()), inl))
    return out


# --------------------------------------------------------------------------
# Turning DETECTED LINES into calibration points.
#
# lines.py hands back segments; pitch.Calibration wants named points. The step
# between them is the correspondence - which detected line is the halfway line -
# and this does NOT solve it. The caller supplies the labels; this only does the
# arithmetic, which is the part worth not doing by hand: a Hough endpoint is
# noisy, so each labelled line is refitted to the paint pixels underneath it and
# the points come out of the refitted intersections.
# --------------------------------------------------------------------------

def refit(paint: np.ndarray, seg: tuple[float, float, float, float],
          band_px: float = 7.0) -> tuple[np.ndarray, int]:
    """Total-least-squares refit of one segment to the paint beneath it.

    Returns (a, b, c) for a*x + b*y + c = 0 with (a, b) unit, and the pixel
    count it was fitted from. Hough's endpoints are quantised to whatever
    fragment won the vote; the paint under the line is hundreds of pixels and
    fixes both the angle and the offset.
    """
    x1, y1, x2, y2 = map(float, seg)
    d = np.array([x2 - x1, y2 - y1])
    length = float(np.hypot(*d))
    if length < 1e-6:
        raise ValueError("degenerate segment")
    d /= length
    nrm = np.array([-d[1], d[0]])
    ys, xs = np.nonzero(paint)
    P = np.column_stack([xs, ys]).astype(float)
    rel = P - [x1, y1]
    t, s = rel @ d, rel @ nrm
    Q = P[(np.abs(s) < band_px) & (t > -0.02 * length) & (t < 1.02 * length)]
    if len(Q) < 30:
        raise ValueError(f"only {len(Q)} paint pixels under this segment")
    c = Q.mean(0)
    direction = np.linalg.svd(Q - c)[2][0]
    n = np.array([-direction[1], direction[0]])
    return np.array([n[0], n[1], -float(n @ c)]), len(Q)


def points_from_lines(paint: np.ndarray, labels) -> dict[str, tuple[float, float]]:
    """Labelled pitch lines -> clicks for pitch.Calibration.solve.

    `labels` is a sequence of (segment, axis, value): axis "x" for a line of
    constant pitch x (goal line, penalty-box face, halfway line), "y" for
    constant pitch y (touchlines, the long edges of a box). Every x-line is
    intersected with every y-line and the result is named "(x, y)" in metres,
    which is the literal form pitch.resolve_landmarks already accepts.

    Two families is a requirement, not a convenience. Lines of one family are
    parallel on the pitch, so they meet at a vanishing point and constrain the
    homography barely at all between them; the intersections are what carry the
    scale. And note what this buys: intersections are allowed to be VIRTUAL. The
    penalty-box face crosses the near touchline off the bottom of the frame, and
    that off-screen crossing is still a perfectly good correspondence because
    both lines were measured where they ARE visible.
    """
    xs = [(refit(paint, s)[0], v) for s, ax, v in labels if ax == "x"]
    ys = [(refit(paint, s)[0], v) for s, ax, v in labels if ax == "y"]
    if len(xs) < 2 or len(ys) < 2:
        raise SystemExit(f"need >=2 lines in each family, got {len(xs)} x and {len(ys)} y")
    out = {}
    for lx, vx in xs:
        for ly, vy in ys:
            A = np.array([lx[:2], ly[:2]])
            if abs(np.linalg.det(A)) < 1e-9:
                continue                   # labelled as crossing families but parallel
            p = np.linalg.solve(A, -np.array([lx[2], ly[2]]))
            out[f"({vx}, {vy})"] = (float(p[0]), float(p[1]))
    return out


def _selfcheck() -> None:
    import lines                                            # noqa: PLC0415

    pitch = Pitch()
    # A synthetic camera, and a synthetic pitch drawn through it.
    world = np.float32([[0, 0], [pitch.length, 0], [pitch.length, pitch.width], [0, pitch.width]])
    Hm2i = cv2.getPerspectiveTransform(
        # The pitch deliberately overflows the frame, so the goal-line /
        # near-touchline crossing lands off-image and exercises the virtual
        # intersection a real broadcast frame also needs.
        world, np.float32([[-420, 1020], [2340, 1020], [1700, 320], [230, 320]]))

    # Ground texture fixed to the PITCH, not to the frame: scuffs and stud marks
    # at known metres, so they move exactly as the camera moves them. Without
    # this the synthetic scene is six white lines on flat green, SIFT finds ~37
    # features in the whole image, and the registration half of this check would
    # be testing nothing. Real footage has stands, hoardings and turf.
    _rng = np.random.default_rng(3)
    tex = _rng.uniform([-6, -6], [pitch.length + 6, pitch.width + 6], (1200, 2))
    tex_r = _rng.integers(2, 7, len(tex))
    # Strongly green, so paint_mask's saturation gate keeps the texture OUT of
    # the paint - otherwise the speckle joins the markings and refit() fits the
    # line to the turf.
    tex_v = np.column_stack([_rng.integers(20, 60, len(tex)),
                             _rng.integers(100, 210, len(tex)),
                             _rng.integers(20, 70, len(tex))])

    def draw(H):
        img = np.zeros((1080, 1920, 3), np.uint8)
        img[:] = (60, 140, 70)
        for (x, y), r, v in zip(np.int32(_warp(H, np.float32(tex))), tex_r, tex_v):
            cv2.circle(img, (int(x), int(y)), int(r), tuple(int(c) for c in v), -1)
        for ax, val in LABELS:
            a, b = ((val, -20), (val, pitch.width + 20)) if ax == "x" else \
                   ((-20, val), (pitch.length + 20, val))
            p, q = _warp(H, np.float32([a, b]))
            cv2.line(img, tuple(np.int32(p)), tuple(np.int32(q)), (245, 245, 245), 3)
        return img

    LABELS = [("x", 0.0), ("x", 16.5), ("x", 52.5), ("y", 0.0), ("y", 54.16), ("y", 68.0)]
    img = draw(Hm2i)
    paint = lines.paint_mask(img, lines.grass_mask(img))

    # Seed segments that are DELIBERATELY WRONG - short, and pushed 4 px sideways
    # and a degree or two off - because that is what a Hough endpoint is. If the
    # refit does not pull them back onto the paint, the recovered metres below
    # cannot come out right, so this is the check on refit() as much as on the
    # intersection arithmetic.
    rng = np.random.default_rng(7)
    labels = []
    for ax, val in LABELS:
        a, b = ((val, -20), (val, pitch.width + 20)) if ax == "x" else \
               ((-20, val), (pitch.length + 20, val))
        p, q = _warp(Hm2i, np.float32([a, b]))
        ok, p, q = cv2.clipLine((0, 0, 1920, 1080), tuple(np.int32(p)), tuple(np.int32(q)))
        assert ok, (ax, val)                       # every line must be in shot
        p, q = np.array(p, float), np.array(q, float)
        p, q = p + (q - p) * 0.30, p + (q - p) * 0.70          # keep only the middle
        labels.append((tuple(np.r_[p, q] + rng.normal(0, 4.0, 4)), ax, val))

    clicks = points_from_lines(paint, labels)
    assert len(clicks) == 9, f"3 x-lines by 3 y-lines is 9 points, got {len(clicks)}"
    cal = Calibration.solve(pitch, clicks)
    assert cal.rms_error_m < 1.0, cal.rms_error_m

    # Recovered geometry must match the camera it was drawn through - this is
    # the non-circular half: the clicks came from PIXELS, not from Hm2i.
    #
    # Points that are actually IN SHOT, because that is the only place a stat is
    # ever read. Extrapolation past the frame edge is much worse and is supposed
    # to be: the goal-line corner here sits 420 px off the left of the image, and
    # the same +-4 px of seed noise costs 2.5 m out there. That is the real
    # behaviour of a homography, not a defect to tune away.
    # ...and only INSIDE THE SPAN OF THE LABELLED LINES. The labels here reach
    # x = 52.5, so everything past the halfway line is extrapolation, and the
    # assert pair below states that rather than hiding it. This is the practical
    # limit of calibrating from markings: the geometry is only as wide as the
    # ink you could identify, which on a ball-following camera is one end of the
    # pitch at a time.
    truth = np.float32([[52.5, 34.0], [10.0, 10.0], [30.0, 20.0], [45.0, 60.0]])
    back = cal.to_pitch(_warp(Hm2i, truth))
    assert np.abs(back - truth).max() < 1.0, np.abs(back - truth)

    far = np.float32([[95.0, 60.0]])                    # 42 m past the last label
    assert np.abs(cal.to_pitch(_warp(Hm2i, far)) - far).max() > 1.0, \
        "extrapolation past the labelled span is supposed to be worse"

    # A VIRTUAL intersection - the two families crossing off-frame - is still a
    # legal point, and points_from_lines must not quietly drop it.
    assert any(not (0 < x < 1920 and 0 < y < 1080) for x, y in clicks.values()), \
        "this synthetic scene was supposed to exercise an off-frame crossing"

    # --- the actual subject: a moved camera --------------------------------
    # Pan + zoom the synthetic camera by a KNOWN amount and check the measured
    # drift matches what the geometry says it must be.
    pan = np.array([[0.94, 0.0, 130.0], [0.0, 0.94, 22.0], [0.0, 0.0, 1.0]])
    moved = draw(pan @ Hm2i)
    H, inl = register(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                      cv2.cvtColor(moved, cv2.COLOR_BGR2GRAY))
    assert H is not None and inl >= MIN_INLIERS, f"registration failed: {inl} inliers"
    probes = pitch_probes(cal, (1920, 1080))
    assert len(probes) > 20, len(probes)
    got = drift_m(cal, H, probes)
    want = drift_m(cal, pan, probes)          # the truth, from the known pan
    assert abs(np.median(got) - np.median(want)) < 0.5, (np.median(got), np.median(want))
    assert np.median(got) > 3.0, f"a 130 px pan must cost metres, got {np.median(got)}"

    # ...and a camera that did NOT move must cost ~nothing. Without this the
    # check above passes on anything that returns a big number.
    H0, _ = register(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                     cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    assert np.median(drift_m(cal, H0, probes)) < 0.05, "a still camera must not drift"

    # Hold.hold_s must read off the first crossing, not the last.
    h = Hold(tol_m=1.0, fps=25.0, n_probes=10,
             samples=[(0.0, 0.0, 0, 0, 0), (1.0, 0.4, 0, 0, 99), (2.0, 3.0, 0, 0, 99),
                      (3.0, 0.2, 0, 0, 99)])
    assert h.hold_s == 2.0, h.hold_s
    assert "per-frame" in h.verdict, h.verdict
    assert Hold(tol_m=1.0, fps=25.0, n_probes=10,
                samples=[(0.0, 0.0, 0, 0, 0)]).hold_s == float("inf")

    # ---- consistency(): the instrument step 1 will be judged on -------------
    # A SYNTHETIC PAN WITH A KNOWN TRUTH, because an instrument that has never
    # read zero on a case where zero is the answer cannot be trusted to read
    # small on a case where it is not.
    import os  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    def pan_at(i):
        """Image-space homography for frame i: a pan plus a slow zoom."""
        return np.array([[1.0 + 4e-5 * i, 0.0, -2.2 * i],
                         [0.0, 1.0 + 4e-5 * i, -0.35 * i],
                         [0.0, 0.0, 1.0]], float)

    N_SYN = 60
    tmp = os.path.join(tempfile.gettempdir(), "pancheck_syn.mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (1920, 1080))
    assert vw.isOpened(), "no mp4v writer; cannot run the consistency control"
    for i in range(N_SYN):
        vw.write(draw(pan_at(i) @ Hm2i))
    vw.release()
    try:
        # cal_at TRUE: the exact metres->image map of each frame, inverted.
        # A perfect per-frame calibration must agree with itself, so this reads
        # ~0 and is the negative control.
        def cal_true(i):
            return Calibration(pitch=pitch, H=np.linalg.inv(pan_at(i) @ Hm2i))

        c_true = consistency(tmp, cal_true, lags=(1, 30), samples=6)
        assert c_true.lags, "consistency produced no samples on the synthetic pan"
        for k, (med, _p90, mx, _n) in c_true.lags.items():
            # Not exactly 0: SIFT registration on synthetic speckle carries a
            # fraction of a pixel, which is metres-per-pixel of error. The bar
            # is that it is far below anything a real method would produce.
            assert med < 0.25, (k, med)
            assert mx < 2.0, (k, mx)

        # ...and a SINGLE reused calibration must NOT read zero - it must
        # reproduce drift_m, because that is literally what it reduces to. If
        # these two ever disagree, this function has a bug, not the method.
        fixed = Calibration(pitch=pitch, H=np.linalg.inv(Hm2i))
        c_fix = consistency(tmp, lambda _i: fixed, lags=(30,), samples=6)
        assert c_fix.lags[30][0] > 1.0, c_fix.lags[30]
        assert c_fix.lags[30][0] > c_true.lags[30][0] * 4, (c_fix.lags[30], c_true.lags[30])

        # The camera centre of a rigid rig cannot wander. Under a pure image
        # shear like pan_at it does, so this asserts the instrument REPORTS
        # movement rather than that the synthetic rig is physical.
        sc = centre_scatter(cal_true, range(0, N_SYN, 10), (1920, 1080))
        assert sc["n"] >= 2, sc
    finally:
        os.unlink(tmp)

    print(f"pancheck.py ok  (9 points from 6 lines, {cal.rms_error_m:.3f} m rms; "
          f"a known 130 px pan measured {np.median(got):.1f} m against {np.median(want):.1f} m true; "
          f"consistency control: true-cal median {max(v[0] for v in c_true.lags.values()):.3f} m "
          f"vs one-cal {c_fix.lags[30][0]:.2f} m at lag 30)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # pancheck.py VIDEO CALIBRATION.json [step]
        video, calfile = sys.argv[1], sys.argv[2]
        step = int(sys.argv[3]) if len(sys.argv) > 3 else 25
        c = Calibration.load(calfile)
        r = hold(video, c, step=step)
        print(f"{len(r.samples)} samples, {r.n_probes} on-pitch probes, {r.fps:.0f} fps")
        print(f"{'t(s)':>7} {'median_m':>9} {'p90_m':>8} {'max_m':>8} {'inliers':>8}")
        for t, med, p90, mx, inl in r.samples:
            print(f"{t:7.1f} {med:9.2f} {p90:8.2f} {mx:8.2f} {inl:8d}")
        print(f"\nholds {r.hold_s:.1f} s within {r.tol_m} m -> {r.verdict}")
    else:
        _selfcheck()
