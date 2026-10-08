"""Measure a calibration's accuracy on footage nobody has annotated.

THE PROBLEM
    Ground truth exists for exactly one match here. Every other format - a Veo
    export, whatever footage comes next - arrives with no
    annotations at all, and "we cannot measure this format" is the same thing as
    "we cannot claim a number on this format".

THE IDEA
    The pitch is its own ruler. Its dimensions are known exactly and are not a
    matter of opinion: 105 x 68 m, a penalty box 40.32 m wide, a centre circle
    of radius 9.15. So a calibration can be scored WITHOUT external truth by
    holding landmarks out of the fit and asking where they land.

    Solve from k landmarks, project the remaining n-k, and measure the error in
    METRES - the unit the product makes claims in. That is a real accuracy
    number on unannotated footage, and it is the number that decides whether a
    format is ready to have stats quoted off it.

MEASURED, match 118575 (remap + pitch-band crop, 65 shipped keypoints)
    k= 4   5.16 m median held out
    k= 8   2.79 m
    k=20   2.54 m          -> converging on the leave-one-out 2.50 m
    The file's own rms_error_m is 2.85 m and is IN-SAMPLE (pitch.Calibration
    scores over err[keep], the inlier fit points), so it is the optimistic
    number, not the conservative one. Held out, this calibration is a ~2.5 m
    calibration. That is team-level territory; see the verdict thresholds below.

    ...and 2.50 m is not the floor of the footage, it is the floor of the MODEL.
    The remapped panorama's touchlines are still visibly bowed, so a single
    homography cannot be right across the frame; calibrators.BowWrapper adds a
    5-parameter row-bow and the same leave-one-out drops to 0.42 m.
    `python calcheck.py --real` prints the whole ledger, every model on every
    split, and is the only place these numbers should be quoted from.

HOLD OUT BY REGION, NOT ONLY AT RANDOM
    Random hold-out asks a model to guess a point its neighbours surround, which
    any local interpolator can do without understanding the geometry at all. On
    those same 65 keypoints a thin-plate spline scores 0.64 m at random and
    23.5 m when the held-out set is "every landmark that is not on a touchline"
    - i.e. the middle of the pitch, which is the only part players stand on.
    Pass `held=` to ask the deliberate question. Both, always.

WHAT HELD-OUT ERROR CANNOT TELL YOU
    That the operator clicked the right corners. A confidently wrong click set
    is self-consistent: mistake the goal area for the penalty box and every
    held-out point still lands neatly, on a pitch scaled wrong by a constant.
    check() measures the FIT, not the CORRESPONDENCE.

    height_sanity() is the guard for that, and the thing that makes it a guard
    rather than a tautology is that its evidence comes from OUTSIDE the fit: the
    detections. A person is ~1.8 m tall. The calibration says how many pixels
    1.8 m of standing player subtends at each spot on the pitch
    (sceneprior.expected_height_px), the detector says how many pixels the
    players actually occupy, and nothing in the click set gets a vote on the
    second number. A calibration wrong by a factor of s reports players s times
    too tall, whatever its landmarks agree among themselves.

    Measured on 118575's 31k real detections: the shipped calibration implies
    1.76 m, i.e. 2% off human anatomy, which is independent evidence that the
    correspondence is right - it is NOT something the 2.5 m held-out figure
    could have told us. The same detections read through the same H with the
    world scaled 3x imply 5.31 m and fail; scaled 1/3, 0.57 m and fail.

    It is still a guard, not a proof. It catches a WRONG SCALE, which is the
    failure mode a self-consistent misclick produces. It will not catch a click
    set that is right-sized and mirrored, and 1.8 +- 40% means it needs to be
    off by more than ~1.4x before it fires (measured: a 1.5x error reads 2.65 m
    and does fire) - it is looking for the 3x case, not for the last 10%.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import sceneprior
from pitch import Calibration, Pitch, resolve_landmarks


@dataclass
class Check:
    """Held-out accuracy for one calibration."""

    n_used: int                 #: landmarks in the fit
    n_held: int                 #: landmarks scored
    median_m: float             #: median held-out error, metres
    p90_m: float
    max_m: float
    @property
    def verdict(self) -> str:
        if self.median_m <= 0.5:
            return "good"
        if self.median_m <= 1.5:
            return "usable for team-level, not per-player"
        return "too loose to quote metres from"


def _project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    q = np.c_[pts, np.ones(len(pts))] @ H.T
    w = np.where(np.abs(q[:, 2:3]) < 1e-9, 1e-9, q[:, 2:3])
    return q[:, :2] / w


@dataclass
class HeightCheck:
    """What the calibration thinks the detected players' real height is."""

    n: int                      #: boxes scored; 0 = the camera was unrecoverable
    implied_height_m: float     #: median; 1.80 is right, 5 m means wrong scale
    p10_m: float                #: spread of the same estimate, not an error bar
    p90_m: float
    why: str = ""               #: set only when the camera could not be recovered

    @property
    def ok(self) -> bool:
        # Same +-40% band sceneprior uses to reject a single box, reused here on
        # the MEDIAN over thousands of them. Per-box it is wide because a
        # crouching keeper is real; on the median it is tight, because the only
        # thing that moves a median of 20k detections by 40% is the geometry.
        return abs(self.implied_height_m - sceneprior.PLAYER_HEIGHT_M) <= \
            sceneprior.HEIGHT_TOLERANCE * sceneprior.PLAYER_HEIGHT_M


def height_sanity(cal, boxes, image_wh: tuple[int, int]) -> HeightCheck:
    """How tall the calibration implies the detected people are, in metres.

    `cal` is anything with .H and .to_pitch (pitch.Calibration or any
    calibrators.Calibrator). `boxes` are (N,4) x1,y1,x2,y2 detections IN THE
    SAME PIXEL SPACE the calibration was solved in - for a RemapCalibrator that
    means after undistort_frame, not raw panorama pixels.

    This is the check that survives the mistake the deleted scale_sanity() made.
    That one projected the clicked corners through a homography fitted to make
    those same corners correct, so it reported 105 x 68 m by construction on
    every input, including a click set 3x too small. Nothing here is fitted to
    the clicks: box heights come from the detector, 1.80 m comes from human
    anatomy, and the calibration is the only thing being asked a question.

    Why the answer moves: scaling the world by s takes the recovered camera to
    s times its true distance (sceneprior.camera_from_homography reads the pose
    off H), so a 1.8 m player is predicted at 1/s the pixels it really spans,
    and the ratio comes back as s. That is the whole mechanism.
    """
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    try:
        cam = sceneprior.camera_from_homography(cal.H, image_wh)
    except ValueError as e:
        # A calibration so wrong the camera cannot be recovered at all is the
        # STRONGEST form of the answer this function exists to give, not an
        # error to propagate. sceneprior grew a 1-200 m guard after this was
        # written, and a wrong-scale click set moves the recovered camera by
        # exactly 1/s - so the worst inputs started raising instead of being
        # reported, and the self-check below (the module's whole claim to
        # catching a self-consistent misclick) has been red ever since.
        return HeightCheck(n=0, implied_height_m=float("inf"),
                           p10_m=float("inf"), p90_m=float("inf"), why=str(e))
    feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]])
    want = sceneprior.expected_height_px(cam, cal.to_pitch(feet))
    # No on-pitch filter on purpose. A wrong-scale calibration puts every
    # detection hundreds of metres off the pitch, so filtering to "on the pitch"
    # would throw away exactly the input this is meant to fail on. The median
    # over thousands of boxes already absorbs the handful of crowd detections
    # that an on-pitch filter would have removed - measured on 118575 (31121
    # boxes), all 31121 land within 5 m of the pitch anyway and the
    # filtered and unfiltered medians are identical at 1.756 m.
    keep = np.isfinite(want) & (want > 1e-6)
    if keep.sum() < 20:
        raise SystemExit(f"only {int(keep.sum())} usable boxes; "
                         "nothing projects in front of this camera")
    r = (boxes[keep, 3] - boxes[keep, 1]) / want[keep] * sceneprior.PLAYER_HEIGHT_M
    return HeightCheck(n=int(keep.sum()), implied_height_m=float(np.median(r)),
                       p10_m=float(np.percentile(r, 10)),
                       p90_m=float(np.percentile(r, 90)))


def homography_fit(src: np.ndarray, dst: np.ndarray):
    """The default model: one homography, image px -> pitch metres.

    A `fit` is a callable (src_px, dst_m) -> callable(px) -> metres. Anything
    matching that signature can be scored by check() on exactly the same splits,
    which is the only way to compare a richer model honestly - see
    calibrators.BowWrapper, whose whole case rests on these numbers.
    """
    H = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst)) if len(src) == 4 \
        else cv2.findHomography(np.float32(src), np.float32(dst))[0]
    if H is None or not np.isfinite(H).all():
        return None
    return lambda pts: _project(H, np.asarray(pts, float).reshape(-1, 2))


def check(clicks: dict[str, tuple[float, float]], pitch: Pitch | None = None,
          k: int = 4, trials: int = 60, seed: int = 0, fit=None,
          held: list[str] | set[str] | None = None) -> Check:
    """Hold-out accuracy over many random k-subsets.

    Many subsets rather than one: a single split measures that split's luck.
    Which four points you fit from changes the answer a lot on a wide shot,
    because four points near each other constrain the far end of the pitch
    barely at all.

    `fit` swaps the model (default: one homography; see homography_fit).

    `held` names the landmarks to hold out and makes the split DELIBERATE
    instead of random, which is the only split that can catch a model that just
    interpolates between its neighbours. Random hold-out flatters those badly:
    on 118575's 65 keypoints a thin-plate spline scores 0.66 m at random and
    23.5 m when the held-out points are the ones off the two touchlines. Hold
    out by REGION as well as at random, always.
    """
    pitch = pitch or Pitch()
    # resolve_landmarks, not pitch.landmarks(): the dense keypoint sets that
    # actually calibrate a panorama are named by literal "(x, y)" metres, not by
    # marking name, and filtering to `n in pitch.landmarks()` silently dropped
    # all 65 of 118575's - i.e. scored nothing on the only calibration we ship.
    lm = resolve_landmarks(pitch, clicks)
    names = list(lm)
    fit = fit or homography_fit

    def score(fit_names, held_names):
        """Metres of error at each held-out landmark, or None if the fit failed."""
        pred = fit(np.array([clicks[n] for n in fit_names], float),
                   np.array([lm[n] for n in fit_names], float))
        if pred is None:
            return None
        got = np.asarray(pred(np.array([clicks[n] for n in held_names], float)), float)
        e = np.hypot(*(got - np.array([lm[n] for n in held_names], float)).T)
        return e if np.isfinite(e).all() else None

    if held is not None:
        held_names = [n for n in names if n in set(held)]
        fit_names = [n for n in names if n not in set(held)]
        if len(fit_names) < 4 or not held_names:
            raise SystemExit(f"split leaves {len(fit_names)} to fit from and "
                             f"{len(held_names)} to score")
        e = score(fit_names, held_names)
        if e is None:
            raise SystemExit("the model failed on this split")
        # No max_m guard here on purpose. The random path drops a degenerate
        # subset because it can draw another one; a REGION split is the single
        # question being asked, and a model that answers it with 300 m has to
        # say so. That is the number three guards in a retired panning-camera tracker were blind to.
        return Check(n_used=len(fit_names), n_held=len(held_names),
                     median_m=float(np.median(e)), p90_m=float(np.percentile(e, 90)),
                     max_m=float(e.max()))

    if len(names) < k + 2:
        raise SystemExit(f"need at least {k + 2} known landmarks, got {len(names)}")

    # Sample subsets, never enumerate them. `list(combinations(65, 8))` is 5e9
    # tuples and killed the process - and a dense keypoint set is precisely the
    # case worth scoring. Duplicate draws are harmless: at 21 landmarks and 60
    # trials that is 60 draws from 5985 subsets.
    rng = np.random.default_rng(seed)

    errs: list[float] = []
    for _ in range(trials):
        f = [names[i] for i in rng.choice(len(names), k, replace=False)]
        e = score(f, [n for n in names if n not in f])
        # A degenerate subset (four near-collinear points) produces garbage that
        # would swamp the median. Drop the trial rather than the evidence.
        if e is not None and e.max() < 500:
            errs.extend(e.tolist())
    if not errs:
        raise SystemExit("every subset was degenerate; check the click set")

    a = np.array(errs)
    return Check(n_used=k, n_held=len(names) - k, median_m=float(np.median(a)),
                 p90_m=float(np.percentile(a, 90)), max_m=float(a.max()))


def _selfcheck() -> None:
    p = Pitch()
    lm = p.landmarks()
    # A synthetic camera: perspective view of the pitch.
    src = np.array([[0, 0], [p.length, 0], [p.length, p.width], [0, p.width]], np.float32)
    dst = np.array([[120, 980], [1800, 980], [1500, 300], [420, 300]], np.float32)
    Hm2i = cv2.getPerspectiveTransform(src, dst)
    clean = {n: tuple(_project(Hm2i, np.array([v]))[0]) for n, v in lm.items()}

    c = check(clean, p, trials=40)
    assert c.median_m < 0.05, c            # exact geometry -> near-zero error
    assert c.verdict == "good", c

    # Click noise must show up as metres, not be silently absorbed.
    rng = np.random.default_rng(1)
    noisy = {n: (v[0] + rng.normal(0, 6), v[1] + rng.normal(0, 6)) for n, v in clean.items()}
    cn = check(noisy, p, trials=40)
    assert cn.median_m > c.median_m, (c.median_m, cn.median_m)

    # A uniformly-wrong click set is NOT detectable by check(), and the test
    # asserts that limitation rather than pretending otherwise: clicks taken
    # from a pitch 3x too small are perfectly self-consistent, so held-out error
    # stays near zero and check() reports "good" on a calibration that is wrong
    # by a factor of three. Only outside evidence can catch it - see below.
    small = {n: tuple(_project(Hm2i, np.array([[v[0] * 0.3, v[1] * 0.3]]))[0])
             for n, v in lm.items()}
    cs = check(small, p, trials=40)
    assert cs.median_m < 0.5, cs
    assert cs.verdict == "good", f"documents the blind spot, not an endorsement: {cs}"

    # --- height_sanity: the outside evidence, on the same wrong click set -----
    wh = (1920, 1080)
    good = Calibration.solve(p, clean)
    cam = sceneprior.camera_from_homography(good.H, wh)
    # Synthetic detections: players standing on a grid, each box exactly as tall
    # as 1.8 m really is there. This is the detector's contribution, and it is
    # produced from the TRUE geometry - no calibration under test touches it.
    gx, gy = np.meshgrid(np.linspace(3, p.length - 3, 14), np.linspace(3, p.width - 3, 8))
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    feet = _project(np.linalg.inv(good.H), xy)
    hpx = sceneprior.expected_height_px(cam, xy)
    boxes = np.column_stack([feet[:, 0] - hpx / 4, feet[:, 1] - hpx,
                             feet[:, 0] + hpx / 4, feet[:, 1]])

    # This half IS circular - the boxes were drawn at expected_height_px, so of
    # course it reads 1.80 - and it is here only to pin the units. The
    # non-circular version cannot live in a self-check: it needs 31121 real
    # detections and a 189 MB cache, and it reads 1.76 m (see the header). What
    # is NOT circular is the pair below, where the boxes are held fixed and only
    # the calibration changes.
    hc = height_sanity(good, boxes, wh)
    assert abs(hc.implied_height_m - 1.80) < 0.02, hc
    assert hc.ok, hc

    # The same boxes, read through the 3x-too-small click set that check() just
    # called "good": the calibration believes the camera is 1/0.3 times further
    # away than it is, so the people it sees must be 1/0.3 times taller than a
    # person. ~6 m. This is the assert that makes it a guard.
    bad = Calibration.solve(p, small)
    hb = height_sanity(bad, boxes, wh)
    assert not hb.ok, f"guard failed to fire on a 3.3x-wrong calibration: {hb}"
    # At 0.3x the camera lands at 263 m and sceneprior refuses to decompose it,
    # so the guard fires through the unrecoverable-camera path rather than the
    # height estimate. Both are rejections; assert which one so a change of
    # mechanism cannot pass silently.
    assert hb.n == 0 and hb.why, hb

    # A milder error still recovers a camera, so the height estimate itself is
    # exercised: 0.4x should read ~1.80/0.4 = 4.5 m. This is what keeps the
    # assert above from degenerating into "any rejection will do".
    mild = {n: tuple(_project(Hm2i, np.array([[v[0] * 0.4, v[1] * 0.4]]))[0])
            for n, v in lm.items()}
    hm = height_sanity(Calibration.solve(p, mild), boxes, wh)
    assert not hm.ok, hm
    assert hm.n > 0 and 4.0 < hm.implied_height_m < 5.0, hm
    # ...and it fires the other way too, so it is measuring scale and not just
    # rejecting anything unusual.
    big = {n: tuple(_project(Hm2i, np.array([[v[0] * 2.0, v[1] * 2.0]]))[0])
           for n, v in lm.items()}
    hbig = height_sanity(Calibration.solve(p, big), boxes, wh)
    assert not hbig.ok and hbig.implied_height_m < 1.0, hbig

    # --- the deliberate split, which is the one random hold-out never asks ---
    end = [n for n, v in lm.items() if v[0] > p.length * 0.7]
    cr = check(clean, p, held=end)
    assert cr.n_held == len(end) and cr.n_used == len(lm) - len(end), cr
    assert cr.median_m < 0.05, cr                 # exact geometry, any split
    # ...and `fit` really swaps the model, rather than being quietly ignored -
    # a constant predictor has to score the pitch-sized error it deserves.
    flat_fit = lambda s, d: (lambda q: np.repeat(  # noqa: E731
        d.mean(axis=0)[None], len(np.asarray(q, float).reshape(-1, 2)), axis=0))
    assert check(clean, p, held=end, fit=flat_fit).median_m > 10.0
    # A region split reports a blown-up model instead of discarding it. The
    # random path drops a degenerate draw because it can take another; this one
    # is the whole question, and three guards in a retired panning-camera tracker died of hiding the answer.
    assert check(small, p, held=end, fit=flat_fit).max_m > 10.0

    print(f"calcheck.py ok  (held-out {c.median_m:.3f} m; height guard reads "
          f"{hc.implied_height_m:.2f} m on the true fit, {hb.implied_height_m:.2f} m "
          f"on a 3.3x-wrong one, {hbig.implied_height_m:.2f} m on a 2x-wrong one)")


def _measure_real(job: str = "data/jobs/118575_full_1st/calibration.json") -> None:
    """Every model on every split, on the one calibration we ship. `--real`.

    Kept out of _selfcheck() because it needs the job on disk and ~30 s of
    least_squares. This is the table BowWrapper's docstring quotes; if the two
    ever disagree, this one is right.
    """
    import ast, json, os  # noqa: PLC0415

    import calibrators  # noqa: PLC0415 - only --real needs the fitters

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    d = json.load(open(os.path.join(root, job)))
    clicks = {k: tuple(v) for k, v in d["points"].items()}
    p = Pitch(**d["pitch"])
    lm = resolve_landmarks(p, clicks)
    W = {n: np.asarray(v, float) for n, v in lm.items()}

    def bow_fit(deg):
        def f(src, dst):
            try:
                cal = calibrators.BowWrapper.solve(
                    p, {f"({x},{y})": tuple(s) for s, (x, y) in zip(src, dst)},
                    deg=deg, check=False)
            except Exception:                      # noqa: BLE001
                return None
            return cal.to_pitch
        return f

    def tps_fit(src, dst):
        """Deliberately here as the CONTROL: it wins every random split and
        fails every regional one. Delete it only when nobody proposes a spline."""
        from scipy.interpolate import RBFInterpolator  # noqa: PLC0415

        mu, sd = src.mean(0), src.std(0)
        r = RBFInterpolator((src - mu) / sd, dst, smoothing=1e-3)
        return lambda q: r((np.asarray(q, float).reshape(-1, 2) - mu) / sd)

    names = list(W)
    x = np.array([W[n][0] for n in names]); y = np.array([W[n][1] for n in names])
    alt = np.zeros(len(names), bool)
    for yv in (0.0, p.width):
        i = np.flatnonzero(y == yv)
        alt[i[np.argsort(x[i])][1::2]] = True
    sel = lambda m: [n for n, ok in zip(names, m) if ok]  # noqa: E731
    splits = [
        ("leave-one-out",          [[n] for n in names]),
        ("random 1-in-5 x8",       None),
        ("touchlines -> the rest", [sel((y != 0) & (y != p.width))]),
        ("every 2nd touchline pt", [sel(alt)]),
        ("x band 35..70 m out",    [sel((x >= 35) & (x < 70))]),
        ("near half -> far half",  [sel(y <= p.width * 0.45)]),
        ("left half -> right",     [sel(x > p.length / 2)]),
    ]
    def pano_fit(src, dst):
        """The dense-keypoint path from calibrators.from_keypoints. Here for
        the same reason the spline is: it looks excellent at random."""
        try:
            cal = calibrators.PanoramicCalibrator(
                p, {f"({x},{y})": tuple(s) for s, (x, y) in zip(src, dst)},
                n_strips=12, sigma_scale=0.25)
        except Exception:                          # noqa: BLE001
            return None
        return cal.to_pitch

    models = [("single homography", None), ("+ bow deg 4", bow_fit(4)),
              ("+ bow deg 5", bow_fit(5)), ("panoramic, 12 strips", pano_fit),
              ("thin-plate spline", tps_fit)]
    print(f"held-out error, metres, {len(names)} landmarks of {os.path.basename(job)}")
    print(f"{'split':24s}" + "".join(f"|{m:^24s}" for m, _ in models))
    print(f"{'':24s}" + "".join(f"|{'med    p90     max':^24s}" for _ in models))
    rng = np.random.default_rng(0)
    folds = [sel(np.isin(np.arange(len(names)), rng.permutation(len(names))[f::5]))
             for _ in range(8) for f in range(5)]
    for label, hs in splits:
        row = ""
        for _, f in models:
            e = []
            for h in (folds if hs is None else hs):
                c = check(clicks, p, fit=f, held=h)
                e.append((c.median_m, c.p90_m, c.max_m))
            a = np.array(e)
            row += f"|{np.median(a[:, 0]):8.2f}{np.median(a[:, 1]):8.2f}{a[:, 2].max():8.2f}"
        print(f"{label:24s}{row}")


if __name__ == "__main__":
    import sys

    if "--real" in sys.argv:
        _measure_real()
    else:
        _selfcheck()
