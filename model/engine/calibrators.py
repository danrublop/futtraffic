"""Image->pitch projection, pluggable by camera PROJECTION TYPE.

A 3x3 homography is only right for a flat perspective image; a stitched
panorama needs another model, so each type is a Calibrator and calibration.json
names its `projection_type`. Pitch dimensions stay a per-venue parameter.
Which types the product accepts, who writes each, and the coordinate spaces:
docs/reference/calibration.md#contract (history: #evidence-stitched-panoramas-and-remap)
"""
from abc import ABC, abstractmethod
import ast
import json
import os
import sys

import cv2
import numpy as np

from pitch import Calibration, Pitch, resolve_landmarks


def _arrays(pitch: Pitch, clicks: dict[str, tuple[float, float]]):
    """(names, image px Nx2, pitch metres Nx2) for a click dict.

    Name resolution — landmark or literal "(x,y)" — lives in
    `pitch.resolve_landmarks`, which `Calibration.solve` uses too.
    """
    lm = resolve_landmarks(pitch, clicks)
    names = sorted(clicks)
    src = np.array([clicks[n] for n in names], dtype=np.float64)
    dst = np.array([lm[n] for n in names], dtype=np.float64)
    return names, src, dst


def _hartley(p: np.ndarray):
    """Isotropic normalisation: centroid at origin, mean radius sqrt(2).
    Without it the DLT on 1000+ px coordinates is badly conditioned."""
    c = p.mean(axis=0)
    s = np.sqrt(2) / max(np.sqrt(((p - c) ** 2).sum(axis=1)).mean(), 1e-12)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    return T, (p - c) * s


def _fit_weighted(src: np.ndarray, dst: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted normalised DLT homography.

    cv2.findHomography takes no weights, and weights are exactly what a
    piecewise fit needs: each strip uses EVERY correspondence but leans on the
    nearby ones. Selecting a local subset instead keeps landing on degenerate
    windows - the five halfway-line landmarks are adjacent in image x and
    collinear on the pitch, and findHomography returns a matrix for them that
    fits its own control points perfectly and maps everything else to nonsense.
    """
    Ts, a = _hartley(np.asarray(src, dtype=np.float64))
    Td, b = _hartley(np.asarray(dst, dtype=np.float64))
    rw = np.sqrt(np.asarray(w, dtype=np.float64))
    A = np.empty((2 * len(a), 9))
    for i, ((x, y), (u, v)) in enumerate(zip(a, b)):
        A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
    A *= np.repeat(rw, 2)[:, None]
    H = np.linalg.svd(A)[2][-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ H @ Ts
    if abs(H[2, 2]) < 1e-12:
        raise ValueError("degenerate homography - are the points collinear?")
    return H / H[2, 2]


def _float_maps(mapx, mapy):
    """Vendor remap tables -> the float32 pair cv2.remap and we both want.

    Most vendors ship OpenCV's fixed-point form (CV_16SC2 integer part +
    CV_16UC1 interpolation index). cv2.remap eats it, but you cannot do
    arithmetic on it, and inverting the table is arithmetic.
    """
    mapx, mapy = np.asarray(mapx), np.asarray(mapy)
    if mapx.dtype != np.float32 or mapx.ndim == 3:
        mapx, mapy = cv2.convertMaps(mapx, mapy, cv2.CV_32FC1)
    return mapx, mapy


def load_maps(path: str, yscale: float = 1.0, crop=None, scale: float = 1.0):
    """Remap tables from a directory of *mapx.npy/*mapy.npy, or one .npz.

    All three knobs resample the DESTINATION grid; the table VALUES are source
    coordinates and are untouched, so they change how densely and over what
    window we sample the panorama, not where. A vendor's canvas can squash the
    pitch; these win the resolution back.

    `crop` is (row0, row1) of the vendor canvas - see pitch_band().
    `scale` resamples BOTH axes by the same factor: sceneprior assumes square
    pixels, and make_prior stands the prior down when yscale != 1.
    `yscale` is the vertical-only version, kept only because the flag is
    documented; prefer crop+scale. Defaults are the vendor's canvas, unchanged.
    Measurements: calibration-evidence (not published)#evidence-stitched-panoramas-and-remap
    """
    import glob, os  # noqa: PLC0415

    if os.path.isdir(path):
        def one(tag):
            hits = sorted(glob.glob(os.path.join(path, f"*map{tag}.npy")))
            if len(hits) != 1:
                raise ValueError(f"expected one *map{tag}.npy in {path}, found {hits}")
            return np.load(hits[0])
        mx, my = _float_maps(one("x"), one("y"))
    else:
        z = np.load(path)
        mx, my = _float_maps(z["mapx"], z["mapy"])
    if crop is not None:
        y0, y1 = (int(round(v)) for v in crop)
        if not 0 <= y0 < y1 <= mx.shape[0]:
            raise ValueError(f"crop {(y0, y1)} outside the {mx.shape[0]}-row canvas")
        mx, my = mx[y0:y1], my[y0:y1]
    if (scale, yscale) != (1.0, 1.0):
        mx, my = (cv2.resize(m, None, fx=scale, fy=scale * yscale,
                             interpolation=cv2.INTER_LINEAR) for m in (mx, my))
    return mx, my


def pitch_band(cal, image_wh: tuple, margin_m: float = 3.0) -> tuple[int, int]:
    """Rows of the remapped canvas that can hold a player, from the geometry.

    The bottom is the lowest ground point of the pitch-plus-margin rectangle - a
    box bottom IS the ground point. The top is NOT the far touchline: it is the
    highest head the height gate would still accept there, i.e.
    (1 + HEIGHT_TOLERANCE) x the geometric player height, so the crop can never
    remove a box the prior would have kept. margin_m matches
    sceneprior.pitch_polygon's default on purpose.

    The band is then grown to sit SYMMETRICALLY about the canvas centre row:
    camera_from_homography assumes the principal point is the image centre, and
    an off-centre crop would move it unseen.
    Measurements: calibration-evidence (not published)#evidence-stitched-panoramas-and-remap
    """
    import sceneprior  # noqa: PLC0415 - only fitting needs it, never loading

    cam = sceneprior.camera_from_homography(cal.H, image_wh)
    L, W, m = cal.pitch.length, cal.pitch.width, margin_m
    gx, gy = np.meshgrid(np.linspace(-m, L + m, 120), np.linspace(-m, W + m, 80))
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    ground = cv2.perspectiveTransform(xy.reshape(-1, 1, 2), cam.G).reshape(-1, 2)[:, 1]
    head = ground - (1 + sceneprior.HEIGHT_TOLERANCE) * sceneprior.expected_height_px(cam, xy)
    mid = image_wh[1] / 2.0
    k = max(mid - head.min(), ground.max() - mid)
    return (max(0, int(np.floor(mid - k))), min(image_wh[1], int(np.ceil(mid + k))))


def remap_points(mapx, mapy, pts_px, tol_px: float = 0.5) -> np.ndarray:
    """Raw (pre-remap) px -> remapped px. The inverse of the remap table.

    Direction matters and is easy to get backwards: cv2.remap's tables answer
    "which SOURCE pixel feeds this DESTINATION pixel", so they already go
    remapped->raw. Everything annotated on the raw panorama - the vendor's
    keypoints, an operator's clicks on the unflattened frame - needs the other
    direction, and there is no closed form.

    Nearest table entry, then one exact Newton step through a local linear model
    of the table (the map is smooth, so one step converges), then a forward
    check: sample the table at the answer and confirm it lands back on the input
    within `tol_px`. Without that check an uncovered point silently returns the
    nearest edge pixel, which is a wrong calibration point that looks fine.
    """
    mapx, mapy = _float_maps(mapx, mapy)
    h, w = mapx.shape
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
    out = np.empty_like(pts)
    for i, (u, v) in enumerate(pts):
        y0, x0 = divmod(int(np.argmin((mapx - u) ** 2 + (mapy - v) ** 2)), w)
        ys = slice(max(0, y0 - 2), min(h, y0 + 3))
        xs = slice(max(0, x0 - 2), min(w, x0 + 3))
        gy, gx = np.mgrid[ys, xs]
        A = np.column_stack([gx.ravel() - x0, gy.ravel() - y0, np.ones(gx.size)])
        b = np.column_stack([mapx[ys, xs].ravel(), mapy[ys, xs].ravel()])
        c = np.linalg.lstsq(A, b, rcond=None)[0]
        try:
            out[i] = [x0, y0] + np.linalg.solve(c[:2].T, [u, v] - c[2])
        except np.linalg.LinAlgError:
            out[i] = [x0, y0]
    # Forward check, vectorised: cv2.remap the tables themselves at `out`.
    q = out.astype(np.float32).reshape(1, -1, 2)
    got = np.stack([cv2.remap(m, q[..., 0].copy(), q[..., 1].copy(),
                              cv2.INTER_LINEAR).ravel() for m in (mapx, mapy)], axis=1)
    bad = np.linalg.norm(got - pts, axis=1) > tol_px
    if bad.any():
        raise ValueError(f"{int(bad.sum())} point(s) not covered by the remap "
                         f"tables, first={pts[bad][0].tolist()}")
    return out


class Calibrator(ABC):
    """image px -> pitch metres, plus the fit report the operator UI shows.

    Every implementation exposes `pitch`, `points`, `rms_error_m` and
    `outliers` so nothing downstream has to branch on projection type.
    """

    projection_type = "abstract"

    @abstractmethod
    def to_pitch(self, pts_px: np.ndarray) -> np.ndarray:
        """(N,2) image px -> (N,2) pitch metres."""

    @abstractmethod
    def to_dict(self) -> dict:
        """JSON-safe dict; always carries `projection_type`."""

    def undistort_frame(self, frame):
        """Flatten a frame before detection. Identity unless there is a remap.

        Only RemapCalibrator has tables to flatten with; it overrides this and
        RAISES when they are missing, so a remap that lost its maps is caught.
        Everywhere else H applies to the raw pixels, and one identity here keeps
        consumers from branching on projection type.
        History: calibration-evidence (not published)#evidence-stitched-panoramas-and-remap
        """
        return frame

    def save(self, path: str) -> None:
        # A point-free calibration can never be scored by calcheck (nothing to
        # hold out) and nothing downstream complains. Warn rather than raise:
        # loading one is legitimate, writing one almost never is.
        # calibration-evidence (not published)#evidence-stitched-panoramas-and-remap
        if not getattr(self, "points", None):
            print(f"warning: {path} written with no landmarks; calcheck cannot "
                  f"score it and rms_error_m is unverifiable", file=sys.stderr)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class PerspectiveCalibrator(Calibrator):
    """One homography for the whole frame. Correct iff the image is a real
    pinhole render - i.e. a single un-stitched camera."""

    projection_type = "perspective"

    def __init__(self, calibration: Calibration):
        self.cal = calibration

    def __getattr__(self, name):
        # Delegate pitch/points/rms_error_m/outliers/H to the Calibration
        # rather than copying four fields that would then drift apart.
        if name == "cal":
            raise AttributeError(name)
        return getattr(self.cal, name)

    @classmethod
    def solve(cls, pitch: Pitch, clicks: dict[str, tuple[float, float]]) -> "PerspectiveCalibrator":
        return cls(Calibration.solve(pitch, clicks))

    def to_pitch(self, pts_px: np.ndarray) -> np.ndarray:
        return self.cal.to_pitch(pts_px)

    def to_dict(self) -> dict:
        return self.cal.to_dict()


class RemapCalibrator(Calibrator):
    """Dense remap to flatten the stitch, THEN one ordinary homography.

    The decomposition SoccerTrack v2 ships with its footage. It does NOT consume
    a vendor's own homography (their frame, their conventions): we remap the
    FRAME and solve our own homography on the flattened image. After
    `undistort_frame` every downstream stage is back in the perspective world.

    COORDINATES: `points` and `H` are in REMAPPED px (after crop and scale).
    Measurements: calibration-evidence (not published)#evidence-stitched-panoramas-and-remap
    """

    projection_type = "remap"

    def __init__(self, pitch: Pitch, mapx, mapy, inner: Calibrator,
                 maps: str | None = None, yscale: float = 1.0,
                 crop=None, scale: float = 1.0):
        self.pitch = pitch
        self._mapx, self._mapy = (None, None) if mapx is None else _float_maps(mapx, mapy)
        self._inner = inner
        # Provenance, not payload: the tables are ~26 MB of int16 and have no
        # business inside a calibration JSON. Saving the PATH means a reload can
        # remap frames again, and a reload WITHOUT the tables still yields a
        # working calibration - stats.py only needs H/to_pitch/pitch, and a
        # tracks.json that was produced from remapped frames is already flat.
        # crop/scale/yscale are part of the maps' IDENTITY, not a rendering
        # preference: points solved on a cropped 2.9x canvas are metres off on
        # the vendor's one, and nothing downstream can tell.
        self.maps, self.yscale = maps, float(yscale)
        self.crop = None if crop is None else tuple(int(v) for v in crop)
        self.scale = float(scale)
        self.points = getattr(inner, "points", {})
        self.rms_error_m = getattr(inner, "rms_error_m", 0.0)
        self.outliers = getattr(inner, "outliers", [])

    @classmethod
    def solve(cls, pitch: Pitch, clicks, maps=None, mapx=None, mapy=None,
              yscale: float = 1.0, crop=None, scale: float = 1.0,
              **kw) -> "RemapCalibrator":
        """`clicks` are landmarks on an ALREADY-REMAPPED frame."""
        if mapx is None and maps is not None:
            mapx, mapy = load_maps(maps, yscale, crop, scale)
        return cls(pitch, mapx, mapy, PerspectiveCalibrator.solve(pitch, clicks, **kw),
                   maps=maps if isinstance(maps, str) else None, yscale=yscale,
                   crop=crop, scale=scale)

    @property
    def H(self) -> np.ndarray:
        """Remapped px -> pitch metres. stats.kalman_ground differentiates this
        for its measurement noise, so it must be the INNER homography - after
        undistort_frame the frame really is a perspective image of the plane."""
        return self._inner.H

    def undistort_frame(self, frame):
        """Flatten a raw panorama frame. Call this BEFORE detection."""
        if self._mapx is None:
            raise ValueError(f"no remap tables loaded (maps={self.maps!r})")
        return cv2.remap(frame, self._mapx, self._mapy, cv2.INTER_LINEAR)

    def to_pitch(self, pts_px):
        # Points must already be in remapped space, i.e. measured on a frame that
        # went through undistort_frame. Feeding raw-panorama pixels here is the
        # one way to misuse this class, and it fails silently by metres - use
        # remap_points() to move raw-space annotations across.
        return self._inner.to_pitch(pts_px)

    def to_dict(self) -> dict:
        d = self._inner.to_dict()
        d["projection_type"] = self.projection_type
        d["maps"], d["maps_yscale"] = self.maps, self.yscale
        d["maps_crop"], d["maps_scale"] = self.crop, self.scale
        d["note"] = "points are in REMAPPED space; apply undistort_frame first"
        return d


class PanoramicCalibrator(Calibrator):
    """Piecewise homography: one locally-weighted homography per vertical strip,
    blended linearly between strip centres.

    Over a narrow column a cylindrical bend is close to affine, so a local
    homography absorbs it; blending (not switching) at strip boundaries keeps
    velocities continuous across a seam.

    REFUSED by jobs.load_calibration: its global H is a single-homography fit.
    # ponytail: strips in x only (seams on a horizontal rig are vertical); no
    # vertical curvature, and only as piecewise as the landmarks allow.
    Measurements: calibration-evidence (not published)#evidence-piecewise-and-dense-keypoints
    """

    projection_type = "panoramic"

    # Uniform entry point so the registry doesn't special-case constructors.
    solve = classmethod(lambda cls, pitch, clicks, **kw: cls(pitch, clicks, **kw))

    def __init__(self, pitch: Pitch, points: dict[str, tuple[float, float]],
                 n_strips: int = 5, min_pts: int = 6, sigma_scale: float = 1.0):
        if len(points) < min_pts:
            raise ValueError(
                f"panoramic needs >={min_pts} landmarks (got {len(points)}); "
                "with fewer, every strip fits the same points and this is just "
                "a single homography wearing a hat")
        if not 0.1 <= sigma_scale <= 2.0:
            raise ValueError(f"sigma_scale {sigma_scale} outside 0.1-2.0")
        self.pitch, self.points, self.n_strips, self.min_pts = pitch, dict(points), n_strips, min_pts
        self.sigma_scale = float(sigma_scale)

        names, src, dst = _arrays(pitch, points)
        xs = src[:, 0]
        self.centres = np.linspace(xs.min(), xs.max(), max(1, n_strips))
        # Gaussian falloff at one strip spacing, plus a floor so distant points
        # still constrain the fit. The floor is what keeps every strip
        # well-conditioned when the local landmarks happen to be collinear.
        self._sigma = float(np.ptp(self.centres) / max(n_strips - 1, 1)) or 1.0

        # LMEDS is no use here: on a real panorama a single homography declares
        # a whole half of the pitch an outlier, which would delete the very
        # points the strips need. Reject on the piecewise residual instead,
        # gated on median+MAD rather than a multiple of the median - residual
        # model mismatch is smooth and spreads the MAD, a misclick is isolated
        # and doesn't, which is exactly the distinction we want. Iterate,
        # because the first fit is contaminated by the misclick it's hunting
        # and drags the innocent neighbours over the line with it.
        keep = np.ones(len(names), bool)
        for _ in range(3):
            self._refit(src, dst, keep)
            err = np.linalg.norm(self.to_pitch(src) - dst, axis=1)
            med = float(np.median(err[keep]))
            mad = float(np.median(np.abs(err[keep] - med)))
            new = err <= max(med + 6.0 * mad, 2.0 * med, 0.5)
            # If a fifth of the landmarks look wrong, the MODEL is wrong, not
            # the clicks - rejecting them would delete the evidence and report
            # a lovely RMS. Keep the worst few at most and let the number be ugly.
            if (~new).sum() > max(1, len(names) // 5):
                break
            if new.sum() < 4 or (new == keep).all():
                break
            keep = new
        self.outliers = [n for n, ok in zip(names, keep) if not ok]
        self.rms_error_m = float(np.sqrt(np.mean(err[keep] ** 2)))

    def _refit(self, src: np.ndarray, dst: np.ndarray, keep: np.ndarray) -> None:
        w0 = keep.astype(float)
        # sigma_scale < 1 localises each strip: right for a dense set, wrong for
        # the clustered operator landmarks, which must borrow across the gaps.
        sigma = self._sigma * self.sigma_scale
        # The floor keeps a strip conditioned when its local landmarks are
        # collinear (operator set); on a dense set it only drags toward the
        # global fit. Both measured: calibration-evidence (not published)#evidence-piecewise-and-dense-keypoints
        floor = 0.05 if self.sigma_scale >= 1.0 else 0.0
        self.Hs = [
            _fit_weighted(src, dst, w0 * (np.exp(-0.5 * ((src[:, 0] - c) / sigma) ** 2) + floor))
            for c in self.centres
        ]
        # Plain global fit: what the UI overlay previews and what a
        # perspective-only consumer would degrade to.
        self.H = _fit_weighted(src, dst, w0)

    def to_pitch(self, pts_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        proj = np.stack([
            cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
            for H in self.Hs
        ])  # (S, N, 2)
        c = self.centres
        if len(c) == 1:
            return proj[0]
        # Clamp outside the clicked span: extrapolating the blend past the end
        # strips invents geometry we never measured.
        x = np.clip(pts[:, 0], c[0], c[-1])
        i = np.clip(np.searchsorted(c, x) - 1, 0, len(c) - 2)
        t = ((x - c[i]) / np.maximum(c[i + 1] - c[i], 1e-9))[:, None]
        n = np.arange(len(pts))
        return proj[i, n] * (1 - t) + proj[i + 1, n] * t

    def to_dict(self) -> dict:
        return {
            "projection_type": self.projection_type,
            "pitch": self.pitch.__dict__,
            "H": self.H.tolist(),  # global fit only - the strips refit from points
            "points": self.points,
            "rms_error_m": self.rms_error_m,
            "outliers": self.outliers,
            "n_strips": self.n_strips,
            "sigma_scale": self.sigma_scale,
        }


def estimate_k1(img, focal_px: float | None = None,
                k_range: tuple[float, float] = (-0.25, 0.10), n: int = 141,
                min_len: float = 600.0, min_px: int = 200) -> dict:
    """Recover one radial distortion coefficient from pitch lines. No labels.

    THE METHOD IS THE OLDEST ONE THERE IS AND IT NEEDS NO GROUND TRUTH: a
    straight line in the world images as a straight line through a pinhole
    camera. It does not through a real lens. So detect the pitch markings, take
    the paint pixels along the longest ones, and choose the k1 that makes them
    straightest. Everything here is measured off the frame itself.

    WHY: a homography carries straight lines to straight lines; radial
    distortion bows them, so a homography fitted to distorted imagery leaves a
    bias field worst at the edges, where the corners and longest baselines are.

    `focal_px` defaults to the image width, a convention: k1 and f trade off and
    only k1/f**2 is determined. Pass the SAME focal to UndistortWrapper.

    NOT scored against real player positions; not wired into the product.
    Measurements and caveat: calibration-evidence (not published)#evidence-bow-and-lens
    """
    import lines  # noqa: PLC0415 - only this path needs the detector

    img = np.asarray(img)
    h, w = img.shape[:2]
    f = float(focal_px or w)
    c = np.array([w / 2.0, h / 2.0])

    paint = lines.paint_mask(img, lines.grass_mask(img))
    ys, xs = np.nonzero(paint)
    pts = np.column_stack([xs, ys]).astype(np.float64)

    groups = []
    for s in lines.find(img):
        if s.length < min_len:
            continue
        a = np.array([s.x1, s.y1], float)
        d = np.array([s.x2 - s.x1, s.y2 - s.y1], float)
        L = float(np.hypot(*d))
        if L <= 0:
            continue
        u = d / L
        rel = pts - a
        along = rel @ u
        perp = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
        # Within the segment and close to it. The band is the paint width plus
        # a little - wider and it swallows the turf, which has no reason to be
        # straight and would drag the fit.
        g = pts[(along > 0) & (along < L) & (perp < 5.0)]
        if len(g) >= min_px:
            groups.append(g)

    if not groups:
        return {"k1": 0.0, "focal_px": f, "segments": 0, "bow_px": 0.0,
                "bow_px_corrected": 0.0,
                "why": f"no pitch line over {min_len:.0f} px carried {min_px} paint pixels"}

    def bow(g, k1):
        nz = (g - c) / f
        q = c + nz * (1.0 + k1 * (nz ** 2).sum(1, keepdims=True)) * f
        q = q - q.mean(0)
        v = np.linalg.svd(q, full_matrices=False)[2][0]
        return float(np.sqrt(((q[:, 0] * v[1] - q[:, 1] * v[0]) ** 2).mean()))

    tot = sum(len(g) for g in groups)
    def cost(k1):
        return sum(bow(g, k1) * len(g) for g in groups) / tot

    ks = np.linspace(k_range[0], k_range[1], n)
    costs = [cost(k) for k in ks]
    k1 = float(ks[int(np.argmin(costs))])
    return {"k1": k1, "focal_px": f, "segments": len(groups),
            "paint_px": tot, "bow_px": cost(0.0), "bow_px_corrected": cost(k1),
            "why": "plumb line: a perspective image of a straight line is straight"}


def undistorted(inner: Calibrator, k1: float, image_wh: tuple[int, int],
                focal_px: float | None = None) -> Calibrator:
    """Wrap `inner` so points are undistorted BEFORE it projects them.

    The ordering is the whole point and it is the one `UndistortWrapper` was
    built for: undo the lens, then apply the homography. Doing it the other way
    round asks a projective map to absorb a non-projective error, which §
    `estimate_k1` explains it cannot.
    """
    w, h = image_wh
    f = float(focal_px or w)
    K = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])
    return UndistortWrapper(inner, K, np.array([k1, 0.0, 0.0, 0.0, 0.0]))


class UndistortWrapper(Calibrator):
    """cv2.undistort the points, then project with the wrapped calibrator.

    Barrel distortion on a wide lens is worst at the frame edges, which is
    exactly where the pitch corners live - so the corners, our longest
    baseline and best constraints, are the correspondences a raw homography
    fits worst. One cv2.calibrateCamera pass per camera MODEL (not per venue)
    supplies K and the coefficients; this is a per-camera constant.
    """

    def __init__(self, inner: Calibrator, camera_matrix, dist_coeffs, points=None):
        self.inner = inner
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).ravel()
        # Keep the operator's ORIGINAL clicks: they are what was seen on the
        # distorted photo, and re-solving from them reproduces this object.
        self.points = dict(points if points is not None else inner.points)

    def __getattr__(self, name):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    @property
    def projection_type(self) -> str:
        return self.inner.projection_type

    @staticmethod
    def _undistort(pts, camera_matrix, dist_coeffs) -> np.ndarray:
        K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        D = np.asarray(dist_coeffs, dtype=np.float64).ravel()
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        # P=K keeps the result in pixels of the same virtual camera.
        return cv2.undistortPoints(pts, K, D, P=K).reshape(-1, 2)

    @classmethod
    def solve(cls, pitch: Pitch, clicks, camera_matrix, dist_coeffs,
              projection_type: str = "perspective", **kw) -> "UndistortWrapper":
        names = list(clicks)
        raw = np.array([clicks[n] for n in names], dtype=np.float64)
        fixed = cls._undistort(raw, camera_matrix, dist_coeffs)
        inner = _REGISTRY[projection_type].solve(pitch, dict(zip(names, map(tuple, fixed))), **kw)
        return cls(inner, camera_matrix, dist_coeffs, points=dict(clicks))

    def to_pitch(self, pts_px: np.ndarray) -> np.ndarray:
        return self.inner.to_pitch(self._undistort(pts_px, self.camera_matrix, self.dist_coeffs))

    def to_dict(self) -> dict:
        d = dict(self.inner.to_dict())
        d["points"] = self.points  # original, distorted clicks
        d["undistort"] = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
        }
        return d


class TpsCorrected(Calibrator):
    """The operator's perspective calibration, bent through its own clicks.

    Keeps the operator's homography and adds a thin-plate spline through its
    residuals at the clicks, in pitch metres, smoothing 0 (every click lands
    exactly), applied everywhere so no player jumps where a correction switches off.

    `H` is the operator's homography, untouched: the scene prior still reads it.
    Metres per pixel must come from `jacobian`, not from differentiating H.

    Guards: outlier clicks are not bent through, and a spline that folds inside
    the clicks' hull (or cannot return its own clicks) is switched OFF (`active`).
    NOT the product default; nothing in app/ writes it.
    Evidence: calibration-evidence (not published)#evidence-spline-and-fisheye
    """

    projection_type = "perspective_tps"

    def __init__(self, inner: PerspectiveCalibrator):
        from scipy.interpolate import RBFInterpolator  # noqa: PLC0415
        self.inner = inner
        rejected = set(getattr(inner, "outliers", None) or [])
        names = sorted(n for n in inner.points if n not in rejected)
        if len(names) < 3:
            raise ValueError("a perspective_tps calibration needs the clicks it was solved from")
        uv = np.array([inner.points[n] for n in names], dtype=np.float64)
        target = resolve_landmarks(inner.pitch, {n: inner.points[n] for n in names})
        self._p = np.asarray(inner.to_pitch(uv), dtype=np.float64).reshape(-1, 2)
        self._X = np.array([target[n] for n in names], dtype=np.float64)
        self._tps = RBFInterpolator(self._p, self._X - self._p, kernel="thin_plate_spline",
                                    smoothing=0.0)
        self.fit = self.checks()
        # Off if it folds, or if it cannot even return its own clicks - two clicks on one
        # pixel make the spline's system singular, and what comes back is not a map.
        self.active = not self.fit["folds"] and self.fit["exact_m"] <= 1e-3

    def __getattr__(self, name):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def to_pitch(self, pts_px: np.ndarray) -> np.ndarray:
        p = np.asarray(self.inner.to_pitch(pts_px), dtype=np.float64).reshape(-1, 2)
        return p + self._tps(p) if self.active else p

    def jacobian(self, pts_px: np.ndarray, eps: float = 0.5) -> np.ndarray:
        """d(pitch metres)/d(image px), (N,2,2), by central differences of to_pitch."""
        u = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        J = np.empty((len(u), 2, 2))
        for k in range(2):
            d = np.zeros(2)
            d[k] = eps
            J[:, :, k] = (self.to_pitch(u + d) - self.to_pitch(u - d)) / (2 * eps)
        return J

    def checks(self, step_m: float = 1.0) -> dict:
        """Does it return its clicks, fold anywhere inside their hull, and how far does
        it stretch there (the larger singular value of the spline's own Jacobian)."""
        from scipy.spatial import Delaunay  # noqa: PLC0415
        hull = Delaunay(self._p)
        lo, hi = self._p.min(0), self._p.max(0)
        g = np.stack(np.meshgrid(np.arange(lo[0], hi[0], step_m), np.arange(lo[1], hi[1], step_m),
                                 indexing="ij"), -1).reshape(-1, 2)
        g = g[hull.find_simplex(g) >= 0]
        e = 0.05
        jx = (g + [e, 0] + self._tps(g + [e, 0]) - (g - [e, 0]) - self._tps(g - [e, 0])) / (2 * e)
        jy = (g + [0, e] + self._tps(g + [0, e]) - (g - [0, e]) - self._tps(g - [0, e])) / (2 * e)
        det = jx[:, 0] * jy[:, 1] - jx[:, 1] * jy[:, 0]
        sv = np.linalg.svd(np.stack([jx, jy], -1), compute_uv=False)[:, 0]
        return {"clicks": len(self._p), "exact_m": float(np.abs(self._p + self._tps(self._p) - self._X).max()),
                "folds": bool((det <= 0).any()), "min_det": float(det.min()),
                "stretch_p90": float(np.percentile(sv, 90)), "stretch_max": float(sv.max())}

    def to_dict(self) -> dict:
        return {**self.inner.to_dict(), "projection_type": self.projection_type}


class FisheyeCamera(Calibrator):
    """The operator's clicks read through ONE fisheye camera over the pitch plane.

    This class does not fit. `camera` holds the 12 numbers a fit produced - fx fy cx
    cy k1 k2, Rodrigues rotation, translation, cv2.fisheye's model with k3 = k4 = 0 -
    and `to_pitch` inverts them: pixel -> ray -> the z = 0 plane. The maths is
    written twice on purpose (model/validation/camera_models.py) so a check can fail.

    `H` is the operator's homography, untouched, as in TpsCorrected; metres per
    pixel come from `jacobian`.
    ponytail: the prior's polygon is still the homography's, metres off on a panorama;
    draw it through `to_image` when the polygon is what is being measured.

    A ray that never meets the pitch (sky, far stand) reads through the
    homography, not NaN: a NaN position has taken tracking down before.
    NOT the product default; nothing in app/ writes it.
    Evidence (N30): calibration-evidence (not published)#evidence-spline-and-fisheye
    """

    projection_type = "perspective_fisheye"

    def __init__(self, inner: PerspectiveCalibrator, camera):
        self.inner = inner
        self.camera = [float(v) for v in camera]
        if len(self.camera) != 12 or not np.isfinite(self.camera).all():
            raise ValueError("a perspective_fisheye calibration needs its 12 camera numbers")
        q = np.array(self.camera)
        self._q, self._R, self._t = q, cv2.Rodrigues(q[6:9])[0], q[9:12]

    def __getattr__(self, name):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def _theta_d(self, th):
        return th * (1 + self._q[4] * th**2 + self._q[5] * th**4)

    def to_image(self, pts_m: np.ndarray) -> np.ndarray:
        """Pitch metres -> pixels. The forward model, which is what was fitted."""
        q, xy = self._q, np.asarray(pts_m, dtype=np.float64).reshape(-1, 2)
        X = np.column_stack([xy, np.zeros(len(xy))]) @ self._R.T + self._t
        rho = np.hypot(X[:, 0], X[:, 1])
        s = self._theta_d(np.arctan2(rho, X[:, 2])) / np.maximum(rho, 1e-12)
        return np.column_stack([q[2] + q[0] * s * X[:, 0], q[3] + q[1] * s * X[:, 1]])

    def to_pitch(self, pts_px: np.ndarray) -> np.ndarray:
        q, uv = self._q, np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        m = np.column_stack([(uv[:, 0] - q[2]) / q[0], (uv[:, 1] - q[3]) / q[1]])
        thd = np.hypot(m[:, 0], m[:, 1])
        th = thd.copy()
        for _ in range(30):                       # Newton on theta_d(theta) = |m|
            th = th - (self._theta_d(th) - thd) / (1 + 3 * q[4] * th**2 + 5 * q[5] * th**4)
        ray = np.column_stack([np.sin(th)[:, None] * m / np.maximum(thd, 1e-12)[:, None], np.cos(th)])
        d, C = ray @ self._R, -self._R.T @ self._t    # the ray and the camera, in pitch axes
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = -C[2] / d[:, 2]
            P = (C[None] + lam[:, None] * d)[:, :2]
        miss = ~((lam > 0) & np.isfinite(P).all(1))
        if miss.any():
            P[miss] = np.asarray(self.inner.to_pitch(uv[miss]), dtype=np.float64).reshape(-1, 2)
        return P

    def jacobian(self, pts_px: np.ndarray, eps: float = 0.5) -> np.ndarray:
        """d(pitch metres)/d(image px), (N,2,2), by central differences of to_pitch."""
        u = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        J = np.empty((len(u), 2, 2))
        for k in range(2):
            d = np.zeros(2)
            d[k] = eps
            J[:, :, k] = (self.to_pitch(u + d) - self.to_pitch(u - d)) / (2 * eps)
        return J

    def to_dict(self) -> dict:
        return {**self.inner.to_dict(), "projection_type": self.projection_type, "camera": self.camera}


class SphericalCamera(Calibrator):
    """A fixed panoramic rig as ONE spherical camera, found from the paint with no operator dots.

    This is what model/engine/autocal.py writes: projection_type "spherical", the pitch, the camera's
    numbers (fu fv cu cv | Rodrigues rotation | translation, then an optional column warp) and the
    frame size it was solved on (`image_wh`). Evidence and limits, one install only:
    calibration-evidence (not published)#evidence-zero-click

    There is no operator homography behind it, and two things in the pipeline were written for one:
      - `H` is the best single homography through this camera's own map over the pitch. It exists so
        that what only wants "some 3x3" keeps running (the degenerate-calibration check, the legacy
        overlay). It is metres wrong on a panorama, exactly as the homography it replaces was, and
        NOTHING that measures should read it: positions come from `to_pitch`, metres per pixel from
        `jacobian`, the camera's place from `camera_centre_m`, the pitch's outline in the image from
        `pitch_polygon_px`.
      - a pixel whose ray never meets the pitch (sky, the far stand) reads through `H`, not as NaN:
        a position that is not a number has taken tracking down before, and nobody stands there.
    """

    projection_type = "spherical"

    def __init__(self, pitch: Pitch, camera, image_wh, rms_error_m: float = 0.0, auto: dict | None = None):
        import autocal  # noqa: PLC0415 - scipy-heavy; only a spherical calibration needs it
        self._ac = autocal
        self.pitch = pitch
        self.camera = [float(v) for v in camera]
        if len(self.camera) < 10 or not np.isfinite(self.camera).all():
            raise ValueError("a spherical calibration needs its camera numbers")
        self.image_wh = (int(image_wh[0]), int(image_wh[1]))
        self._q = np.array(self.camera)
        self.points, self.outliers = {}, []
        self.rms_error_m = float(rms_error_m)
        self.auto = dict(auto or {})
        gx, gy = np.meshgrid(np.linspace(0.0, pitch.length, 22), np.linspace(0.0, pitch.width, 15))
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        px = self.to_image(grid)
        ok = np.isfinite(px).all(1)
        if ok.sum() < 8:
            raise ValueError("a spherical calibration that cannot see the pitch")
        self.H = cv2.findHomography(px[ok], grid[ok], 0)[0]

    @property
    def camera_centre_m(self) -> np.ndarray:
        """(x, y, height) in pitch metres; height positive whichever way the fit's z points."""
        C = -self._ac._rot(self._q[4:7]).T @ self._q[7:10]
        return np.array([C[0], C[1], abs(C[2])])

    def to_image(self, pts_m: np.ndarray) -> np.ndarray:
        return self._ac._fwd(self._q, np.asarray(pts_m, dtype=np.float64).reshape(-1, 2), float(self.image_wh[0]))

    def to_pitch(self, pts_px: np.ndarray) -> np.ndarray:
        uv = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        P = self._ac._inv(self._q, uv, float(self.image_wh[0]))
        miss = ~np.isfinite(P).all(1)
        if miss.any():
            P[miss] = cv2.perspectiveTransform(uv[miss].reshape(-1, 1, 2), self.H).reshape(-1, 2)
        return P

    def jacobian(self, pts_px: np.ndarray, eps: float = 0.5) -> np.ndarray:
        """d(pitch metres)/d(image px), (N,2,2), by central differences of to_pitch."""
        u = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        J = np.empty((len(u), 2, 2))
        for k in range(2):
            d = np.zeros(2)
            d[k] = eps
            J[:, :, k] = (self.to_pitch(u + d) - self.to_pitch(u - d)) / (2 * eps)
        return J

    def pitch_polygon_px(self, margin_m: float = 3.0, step_m: float = 2.0) -> np.ndarray:
        """The pitch plus a margin as an image polygon. On a panorama a touchline is a curve, so the
        outline is walked, not drawn between four corners."""
        L, W, m = self.pitch.length, self.pitch.width, margin_m
        c = np.array([[-m, -m], [L + m, -m], [L + m, W + m], [-m, W + m], [-m, -m]], float)
        pts = [a + t[:, None] * (b - a) for a, b in zip(c[:-1], c[1:])
               for t in [np.arange(0.0, 1.0, step_m / np.linalg.norm(b - a))]]
        px = self.to_image(np.vstack(pts))
        return px[np.isfinite(px).all(1)]

    def to_dict(self) -> dict:
        return {"projection_type": self.projection_type, "pitch": vars(self.pitch).copy(), "camera": self.camera,
                "image_wh": list(self.image_wh), "H": self.H.tolist(), "points": {}, "outliers": [],
                "rms_error_m": self.rms_error_m, "auto": self.auto}


class BowWrapper(Calibrator):
    """Pre-warp the image row by a smooth polynomial in the COLUMN, then project.

        (x, y) -> (x, y + b(x) * (1 + m * y_n))  ->  the wrapped calibrator

    DO NOT ENABLE THIS ON 118575. It is off by default (`bow_deg=None`) and
    neither jobs.py nor track.py passes it. It wins on the landmark hold-out
    and LOSES against ground-truth player positions: the landmarks sit mostly
    on the touchlines, where no outfield player stands.
    Evidence: calibration-evidence (not published)#evidence-bow-and-lens

    # ponytail: b() is clamped outside the fitted x span rather than
    # extrapolated. The bow is only known where a landmark measured it.
    # Two ceilings. (1) The y-dependence is ONE linear term, because two image
    # rows cannot constrain more; a richer y model overfits the touchlines and
    # invents the middle of the pitch. (2) It is a correction, not a camera
    # model - the real fix is the rig's own cylindrical parameters, at which
    # point deg=0 and this class becomes a no-op.
    """

    #: The bow must beat the plain homography by at least this factor, held
    #: out, or solve() refuses. See _must_beat_a_homography.
    MIN_GAIN = 0.8

    #: Fitted degree. 4 is the fewest that reaches the plateau: 3 leaves 1.19 m
    #: on the touchlines->interior split, 4-7 are all 0.90-1.04 m there.
    DEFAULT_DEG = 4

    def __init__(self, inner: Calibrator, coeffs, ymod: float, mu, sd, xspan,
                 points=None):
        self.inner = inner
        self.coeffs = np.asarray(coeffs, dtype=np.float64).ravel()
        self.ymod = float(ymod)
        self.mu = np.asarray(mu, dtype=np.float64).ravel()
        self.sd = np.asarray(sd, dtype=np.float64).ravel()
        self.xspan = (float(xspan[0]), float(xspan[1]))
        # The operator's ORIGINAL clicks, as UndistortWrapper keeps them: they
        # are what was seen on the frame, and re-solving from them reproduces
        # this object. The inner calibrator holds the warped ones.
        self.points = dict(points if points is not None else inner.points)

    def __getattr__(self, name):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    @property
    def projection_type(self) -> str:
        return self.inner.projection_type

    def _warp(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        x = np.clip(pts[:, 0], *self.xspan)
        xn = (x - self.mu[0]) / self.sd[0]
        yn = (pts[:, 1] - self.mu[1]) / self.sd[1]
        b = sum(c * xn ** (i + 1) for i, c in enumerate(self.coeffs)) if len(self.coeffs) \
            else np.zeros(len(pts))
        return np.column_stack([pts[:, 0], pts[:, 1] + b * (1.0 + self.ymod * yn)])

    @classmethod
    def solve(cls, pitch: Pitch, clicks, projection_type: str = "perspective",
              deg: int | None = None, check: bool = True, **kw) -> "BowWrapper":
        """Fit the bow and the homography TOGETHER, then re-solve the inner.

        Together because they trade off: a homography absorbs part of any bow as
        a tilt, so fitting one and then the other lands in a different place than
        fitting both. The inner is re-solved from the warped clicks afterwards so
        it carries the usual LMEDS outlier handling and rms report.
        """
        from scipy.optimize import least_squares  # noqa: PLC0415 - fitting only

        if projection_type == "panoramic":
            # Both models correct the same non-projective warp, from the same
            # points; stacking them is untested and from_dict would refit the
            # panoramic strips from UN-warped clicks. Pick one.
            raise ValueError("bow on top of panoramic is not supported - both "
                             "model the same warp; use one or the other")
        deg = cls.DEFAULT_DEG if deg is None else int(deg)
        names, src, dst = _arrays(pitch, clicks)
        if len(names) < 8 + deg + 1:
            raise ValueError(f"bow(deg={deg}) needs >={9 + deg} landmarks, got {len(names)}")
        mu, sd = src.mean(axis=0), src.std(axis=0)
        if not sd.all():
            raise ValueError("landmarks have no spread on one axis")
        xspan = (float(src[:, 0].min()), float(src[:, 0].max()))
        H0 = cv2.findHomography(src, dst, 0)[0]
        if H0 is None:
            raise ValueError("homography failed - are the points collinear?")

        def residual(p):
            H = np.append(p[:8], 1.0).reshape(3, 3)
            w = cls(None, p[8:8 + deg], p[8 + deg], mu, sd, xspan, points={})._warp(src)
            return (cv2.perspectiveTransform(w.reshape(-1, 1, 2), H).reshape(-1, 2)
                    - dst).ravel()

        p0 = np.r_[(H0 / H0[2, 2]).ravel()[:8], np.zeros(deg + 1)]
        p = least_squares(residual, p0, method="lm", max_nfev=40000).x
        self = cls(None, p[8:8 + deg], p[8 + deg], mu, sd, xspan, points=dict(clicks))
        warped = dict(zip(names, map(tuple, self._warp(src))))
        self.inner = _REGISTRY[projection_type].solve(pitch, warped, **kw)
        if check:
            self._must_beat_a_homography(pitch, clicks, deg)
        return self

    @staticmethod
    def _must_beat_a_homography(pitch: Pitch, clicks: dict, deg: int) -> None:
        """Refuse to exist unless the extra parameters pay for themselves.

        A model with more freedom always fits its own points better, and this
        area has already shipped four results that were exactly that. So the
        constructor asks the question the reviewer would: HELD OUT, five folds,
        is the bow better than the plain homography it wraps? If not, the frame
        is not bowed (or the landmarks cannot see the bow) and the honest answer
        is the simpler model.
        """
        import calcheck  # noqa: PLC0415 - only fitting needs it

        cls_margin = BowWrapper.MIN_GAIN
        names = sorted(clicks)
        rng = np.random.default_rng(0)
        order = rng.permutation(len(names))
        folds = [[names[i] for i in order[f::5]] for f in range(5)]
        bow = lambda s, d: BowWrapper.solve(  # noqa: E731
            pitch, {f"({x},{y})": tuple(q) for q, (x, y) in zip(s, d)},
            deg=deg, check=False).to_pitch
        med = [float(np.median([calcheck.check(clicks, pitch, fit=f, held=h).median_m
                                for h in folds]))
               for f in (calcheck.homography_fit, bow)]
        # A MARGIN, not just "smaller". On a genuinely flat frame both numbers
        # are float noise and the bow wins half the time by 1e-15, so a bare
        # comparison is a guard that can never fire - which is what the three
        # dead guards in a retired panning-camera tracker had in common.
        if med[1] > cls_margin * med[0]:
            raise ValueError(
                f"bow(deg={deg}) is not an improvement: 5-fold held-out median "
                f"{med[1]:.3f} m vs {med[0]:.3f} m for the plain homography, "
                f"which is not the >={100 * (1 - cls_margin):.0f}% this costs "
                f"{deg + 1} parameters to buy. Either the frame is not bowed or "
                f"these {len(names)} landmarks cannot measure it - ship the "
                f"homography.")

    def to_pitch(self, pts_px: np.ndarray) -> np.ndarray:
        return self.inner.to_pitch(self._warp(pts_px))

    def to_dict(self) -> dict:
        d = dict(self.inner.to_dict())
        d["points"] = self.points          # original, un-warped clicks
        d["bow"] = {
            "coeffs": self.coeffs.tolist(),
            "ymod": self.ymod,
            "mu": self.mu.tolist(),
            "sd": self.sd.tolist(),
            "xspan": list(self.xspan),
        }
        return d


_REGISTRY: dict[str, type] = {
    "perspective": PerspectiveCalibrator,
    "panoramic": PanoramicCalibrator,
    "remap": RemapCalibrator,
}


def get_calibrator(projection_type: str = "perspective", *, pitch: Pitch, clicks: dict,
                   camera_matrix=None, dist_coeffs=None, bow_deg=None, **kw) -> Calibrator:
    """Solve a calibration for the given projection type.

    `dist_coeffs` (with `camera_matrix`) wraps the result in an UndistortWrapper.
    `bow_deg` wraps it in a BowWrapper - use it when the frame is a remapped
    panorama whose touchlines are still bowed (BowWrapper says how to check).
    Extra kwargs go to the concrete calibrator, e.g. n_strips for panoramic.
    """
    if projection_type not in _REGISTRY:
        raise ValueError(f"unknown projection_type {projection_type!r}; have {sorted(_REGISTRY)}")
    if dist_coeffs is not None:
        return UndistortWrapper.solve(pitch, clicks, camera_matrix, dist_coeffs,
                                      projection_type, **kw)
    if bow_deg is not None:
        return BowWrapper.solve(pitch, clicks, projection_type, deg=bow_deg, **kw)
    return _REGISTRY[projection_type].solve(pitch, clicks, **kw)


def from_dict(d: dict) -> Calibrator:
    pitch = Pitch(**d["pitch"])
    ptype = d.get("projection_type", "perspective")
    clicks = {k: tuple(v) for k, v in d.get("points", {}).items()}
    kw = {k: d[k] for k in ("n_strips", "sigma_scale") if k in d}
    bow = d.get("bow")
    if bow:
        # Stored coefficients, not a refit: the bow was fitted jointly with the
        # inner homography and the file already holds both. Wrapping the loaded
        # calibrator OUTSIDE is equivalent to the inside position it was saved
        # from - every calibrator between them passes points through unchanged.
        inner = from_dict({k: v for k, v in d.items() if k != "bow"})
        return BowWrapper(inner, bow["coeffs"], bow["ymod"], bow["mu"],
                          bow["sd"], bow["xspan"], points=clicks)
    und = d.get("undistort")
    if und:
        return UndistortWrapper.solve(pitch, clicks, und["camera_matrix"],
                                      und["dist_coeffs"], ptype, **kw)
    if ptype == "perspective_tps":
        # The stored H and its clicks; the spline is refitted from them, which is
        # exact and deterministic, so nothing but the clicks has to be stored.
        return TpsCorrected(PerspectiveCalibrator(
            Calibration.from_dict({**d, "projection_type": "perspective"})))
    if ptype == "spherical":
        return SphericalCamera(pitch, d.get("camera") or [], d.get("image_wh") or (0, 0),
                               rms_error_m=float(d.get("rms_error_m") or 0.0), auto=d.get("auto"))
    if ptype == "perspective_fisheye":
        return FisheyeCamera(PerspectiveCalibrator(
            Calibration.from_dict({**d, "projection_type": "perspective"})), d.get("camera") or [])
    if ptype in ("perspective", "remap"):
        # Trust the stored H: older files (and the web export) may carry no
        # points at all, and refitting would throw away the saved fit.
        inner = PerspectiveCalibrator(Calibration.from_dict({**d, "projection_type": "perspective"}))
        if ptype == "perspective":
            return inner
        # Absent crop/scale read as the vendor canvas, so every calibration
        # written before the crop existed still loads and behaves identically.
        maps, ys = d.get("maps"), float(d.get("maps_yscale", 1.0))
        crop = d.get("maps_crop")
        crop = tuple(crop) if crop else None
        sc = float(d.get("maps_scale", 1.0))
        mx = my = None
        if maps:
            try:
                mx, my = load_maps(maps, ys, crop, sc)
            except Exception as e:                   # noqa: BLE001
                # Loadable WITHOUT the tables on purpose: stats.py runs on a
                # tracks.json that was already produced from remapped frames and
                # needs only H. Only undistort_frame needs the tables, and it
                # says so when they are missing.
                print(f"remap tables {maps!r} unavailable ({e}); "
                      f"calibration loaded, undistort_frame disabled")
        return RemapCalibrator(pitch, mx, my, inner, maps=maps, yscale=ys,
                               crop=crop, scale=sc)
    return _REGISTRY[ptype](pitch, clicks, **kw)


def load(path: str) -> Calibrator:
    with open(path) as f:
        return from_dict(json.load(f))


# Below this many landmarks, a set is the operator's clicked markings and the
# strips must borrow across the gaps between them; at or above it the set is
# dense enough that borrowing only reimposes a global fit. 21 named landmarks
# sit below, 118575's 65 touchline samples sit above. See resolve_landmarks.
DENSE_LANDMARKS = 40


def from_keypoints(path: str, pitch: Pitch, n_strips: int = 12) -> Calibrator:
    """Panoramic calibration from a dataset keypoint file: {"(x,y)": [px, py]}.

    Keys are literal pitch coordinates in metres, so a line fit can emit this
    format too. The operator flow cannot produce it: nobody can pick out
    "x = 5.5 m along the touchline" by eye.

    Its random-fold score flatters it: hold out the middle of the pitch and it
    fails. Evidence: calibration-evidence (not published)#evidence-piecewise-and-dense-keypoints
    """
    with open(path) as f:
        kp = json.load(f)
    clicks = {n: (float(v[0]), float(v[1])) for n, v in kp.items()}
    resolve_landmarks(pitch, clicks)  # fail here, not 40 lines deep
    dense = len(clicks) >= DENSE_LANDMARKS
    return get_calibrator("panoramic", pitch=pitch, clicks=clicks,
                          n_strips=n_strips if dense else 5,
                          sigma_scale=0.25 if dense else 1.0)


def _cylindrical(pts: np.ndarray, cx: float, cy: float, f: float) -> np.ndarray:
    """Perspective px -> cylindrical panorama px. Test fixture, not product code."""
    x, y = pts[:, 0] - cx, pts[:, 1] - cy
    return np.stack([f * np.arctan2(x, f) + cx, f * y / np.hypot(x, f) + cy], axis=1)


def _barrel(pts: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Ideal px -> lens-distorted px. Inverse of cv2.undistortPoints; fixture only."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x, y = (pts[:, 0] - cx) / fx, (pts[:, 1] - cy) / fy
    r2 = x * x + y * y
    s = 1 + D[0] * r2 + D[1] * r2 * r2
    return np.stack([x * s * fx + cx, y * s * fy + cy], axis=1)


def _cyl_maps(wh: tuple[int, int], cx: float, cy: float, f: float):
    """Remap tables that flatten a cylindrical panorama. Fixture, not product.

    Destination is the perspective image; for each destination pixel the table
    names the panorama pixel that feeds it - which is exactly _cylindrical.
    """
    w, h = wh
    gy, gx = np.mgrid[0:h, 0:w]
    s = _cylindrical(np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float64), cx, cy, f)
    return s[:, 0].reshape(h, w).astype(np.float32), s[:, 1].reshape(h, w).astype(np.float32)


def _demo() -> None:
    """Covers all three projection paths against synthetic ground truth."""
    pitch = Pitch.from_yards(118, 72)
    lm = pitch.landmarks()
    world = np.array(list(lm.values()), dtype=np.float64)
    true_H = np.array([[12.0, 3.0, 300.0], [0.0, 9.0, 120.0], [0.0, 0.006, 1.0]])
    img = cv2.perspectiveTransform(world.reshape(-1, 1, 2), true_H).reshape(-1, 2)
    clicks = dict(zip(lm.keys(), map(tuple, img)))

    # Dense held-out grid: fitting error on the clicked landmarks is the number
    # a calibrator can cheat on, error between them is the one that matters.
    gx, gy = np.meshgrid(np.linspace(2, pitch.length - 2, 25),
                         np.linspace(2, pitch.width - 2, 15))
    grid_w = np.stack([gx.ravel(), gy.ravel()], axis=1)
    grid_i = cv2.perspectiveTransform(grid_w.reshape(-1, 1, 2), true_H).reshape(-1, 2)
    worst = lambda cal, i, w: float(np.abs(cal.to_pitch(i) - w).max())  # noqa: E731

    # --- perspective ---
    cal = get_calibrator("perspective", pitch=pitch, clicks=clicks)
    assert isinstance(cal, PerspectiveCalibrator)
    assert cal.projection_type == "perspective"
    assert cal.rms_error_m < 0.01, cal.rms_error_m
    assert worst(cal, grid_i, grid_w) < 1e-3

    # --- perspective_tps: the operator's H bent through its own clicks ---
    # On a true pinhole image H is already exact, so the spline must be zero and
    # change nothing. Bend the image smoothly and the clicks must land exactly again.
    tps = from_dict({**cal.to_dict(), "projection_type": "perspective_tps"})
    assert isinstance(tps, TpsCorrected) and tps.projection_type == "perspective_tps"
    assert worst(tps, grid_i, grid_w) < 1e-3, "an exact homography gains nothing and loses nothing"
    bent = {n: (u + 0.004 * (v - 300.0) ** 2 / 10.0, v) for n, (u, v) in clicks.items()}
    btps = from_dict({**get_calibrator("perspective", pitch=pitch, clicks=bent).to_dict(),
                      "projection_type": "perspective_tps"})
    chk = btps.checks()
    assert chk["exact_m"] < 1e-6 and not chk["folds"], chk
    names = list(bent)
    assert np.abs(btps.to_pitch(np.array([bent[n] for n in names])) - np.array([lm[n] for n in names])).max() < 1e-6, \
        "every click lands on its landmark"
    back = from_dict(json.loads(json.dumps(btps.to_dict())))
    assert np.allclose(back.to_pitch(grid_i), btps.to_pitch(grid_i)), "it survives JSON"
    J = btps.jacobian(grid_i[:5])
    fd = (btps.to_pitch(grid_i[:5] + [0.0, 1.0]) - btps.to_pitch(grid_i[:5] - [0.0, 1.0])) / 2.0
    assert np.allclose(J[:, :, 1], fd, rtol=1e-3, atol=1e-6), "its Jacobian is its own map's"
    assert btps.H is not None and np.allclose(btps.H, get_calibrator("perspective", pitch=pitch, clicks=bent).H), \
        "H stays the operator's homography"
    assert btps.active, "a smooth bend does not fold"
    # A click moved across the pitch AFTER the solve, and not flagged, folds the spline:
    # it must switch itself off and be the homography.
    base_d = get_calibrator("perspective", pitch=pitch, clicks=bent).to_dict()
    far_px = np.array(bent["centre"]) + 0.6 * (np.array(bent["corner_tr"]) - np.array(bent["centre"]))
    moved = {**base_d, "points": {**base_d["points"], "centre": far_px.tolist()}, "outliers": []}
    fold = from_dict({**moved, "projection_type": "perspective_tps"})
    assert fold.fit["folds"] and fold.fit["exact_m"] < 1e-6 and not fold.active, fold.fit
    assert np.allclose(fold.to_pitch(grid_i), from_dict(base_d).to_pitch(grid_i)), "a folded correction is the homography"
    flagged = from_dict({**moved, "projection_type": "perspective_tps", "outliers": ["centre"]})
    assert flagged.active and flagged.fit["clicks"] == len(bent) - 1, \
        ("a click the solver rejected is not bent through", flagged.fit)

    # --- perspective_fisheye: a stored camera, inverted ---
    # A fisheye looking down at the pitch from behind the near touchline. Its own forward
    # model and its inverse must undo each other, it must survive JSON, H must stay the
    # operator's, and a pixel whose ray misses the pitch must read as the homography does.
    cam = [900.0, 900.0, 960.0, 540.0, -0.02, 0.003,
           *cv2.Rodrigues(np.array([[1.0, 0, 0], [0, -0.5, -0.866], [0, 0.866, -0.5]]))[0].ravel(),
           -pitch.length / 2, 5.0, 40.0]
    fish = from_dict({**cal.to_dict(), "projection_type": "perspective_fisheye", "camera": cam})
    assert isinstance(fish, FisheyeCamera) and fish.projection_type == "perspective_fisheye"
    seen = fish.to_image(grid_w)
    assert np.isfinite(seen).all() and np.abs(fish.to_pitch(seen) - grid_w).max() < 1e-6, \
        "inverse undoes forward on the pitch"
    back = from_dict(json.loads(json.dumps(fish.to_dict())))
    assert np.allclose(back.to_pitch(seen), fish.to_pitch(seen)), "it survives JSON"
    assert np.allclose(fish.H, cal.H), "H stays the operator's homography"
    J = fish.jacobian(seen[:5])
    fd = (fish.to_pitch(seen[:5] + [0.0, 1.0]) - fish.to_pitch(seen[:5] - [0.0, 1.0])) / 2.0
    assert np.allclose(J[:, :, 1], fd, rtol=1e-3, atol=1e-6), "its Jacobian is its own map's"
    sky = np.array([[960.0, -200.0]])                    # above this camera's horizon
    assert np.allclose(fish.to_pitch(sky), cal.to_pitch(sky)), "a ray that misses the pitch is the homography's"
    try:
        from_dict({**cal.to_dict(), "projection_type": "perspective_fisheye"})
        raise AssertionError("a fisheye file with no camera must not load")
    except ValueError:
        pass

    # --- spherical: what model/engine/autocal.py writes ---
    import autocal as _ac  # noqa: PLC0415
    P105 = Pitch(105.0, 68.0)
    mount = _ac.camera(P105, (4096, 1080), fov_deg=156.0, tilt_deg=19.0, height_m=7.5, behind_m=15.0, along_m=-1.0, pan_deg=1.0)
    sph = from_dict({"projection_type": "spherical", "pitch": vars(P105), "camera": mount.tolist(), "image_wh": [4096, 1080]})
    assert isinstance(sph, SphericalCamera) and sph.projection_type == "spherical"
    g105 = np.stack(np.meshgrid(np.linspace(2, 103, 9), np.linspace(2, 66, 7)), -1).reshape(-1, 2)
    assert np.abs(sph.to_pitch(sph.to_image(g105)) - g105).max() < 1e-6, "inverse undoes forward on the pitch"
    assert np.allclose(from_dict(json.loads(json.dumps(sph.to_dict()))).to_pitch(sph.to_image(g105)), g105, atol=1e-6), "it survives JSON"
    assert np.allclose(sph.camera_centre_m, [51.5, 83.0, 7.5], atol=1e-6), sph.camera_centre_m
    assert np.isfinite(sph.to_pitch(np.array([[2048.0, -300.0]]))).all(), "a ray that misses the pitch still reads as a number"
    assert abs(np.linalg.det(sph.H)) > 1e-12 and np.median(np.linalg.norm(
        cv2.perspectiveTransform(sph.to_image(g105).reshape(-1, 1, 2), sph.H).reshape(-1, 2) - g105, axis=1)) > 1.0, \
        "H is a stand-in: on a panorama one homography is metres out, and nothing that measures may read it"
    poly = sph.pitch_polygon_px()
    inside = lambda m: cv2.pointPolygonTest(poly.astype(np.float32), tuple(map(float, sph.to_image(np.array([m]))[0])), False) >= 0  # noqa: E731
    assert inside([52.5, 34.0]) and inside([1.0, 1.0]) and inside([104.0, 67.0]) and not inside([52.5, -12.0]), "the polygon is the pitch"
    Js = sph.jacobian(sph.to_image(g105[:4]))
    assert np.abs(np.linalg.det(Js)).min() > 0 and np.linalg.norm(Js[0]) > 0

    # Choosing panoramic on a genuinely perspective image must cost nothing.
    # Getting the projection-type question wrong stays cheap in that direction,
    # which is why panoramic is the safe default when nobody has checked.
    for n in (1, 2, 3, 5, 8):
        assert worst(get_calibrator("panoramic", pitch=pitch, clicks=clicks, n_strips=n),
                     grid_i, grid_w) < 1e-3, n

    # --- panoramic: a stitched frame a single homography cannot represent ---
    cyl = lambda p: _cylindrical(p, 960.0, 540.0, 800.0)  # noqa: E731
    cyl_clicks = dict(zip(lm.keys(), map(tuple, cyl(img))))
    flat = get_calibrator("perspective", pitch=pitch, clicks=cyl_clicks)
    pano = get_calibrator("panoramic", pitch=pitch, clicks=cyl_clicks, n_strips=6)
    e_flat, e_pano = worst(flat, cyl(grid_i), grid_w), worst(pano, cyl(grid_i), grid_w)
    # If this ever fails because e_flat got small, the fixture stopped being a
    # panorama - not because the homography learned how to bend.
    assert e_flat > 10.0, e_flat
    assert e_pano < 0.3 * e_flat, (e_pano, e_flat)
    assert pano.projection_type == "panoramic"
    # ...and the residual is still metres, not centimetres. Piecewise is a
    # mitigation, not a fix; the fix is unwarping the panorama. Tighten this
    # assert only when the model actually gets better.
    assert e_pano > 1.0, e_pano

    # The gating metric is error on a 5m displacement, not on absolute position:
    # a smooth position bias largely cancels when you difference it, and total
    # distance is a sum of differences. Measuring the wrong one would either
    # condemn a usable calibrator or bless an unusable one.
    rng = np.random.default_rng(0)
    p0 = np.stack([rng.uniform(3, pitch.length - 3, 2000),
                   rng.uniform(3, pitch.width - 3, 2000)], axis=1)
    th = rng.uniform(0, 2 * np.pi, 2000)
    p1 = np.clip(p0 + 5 * np.stack([np.cos(th), np.sin(th)], axis=1),
                 3, [pitch.length - 3, pitch.width - 3])
    truth = np.linalg.norm(p1 - p0, axis=1)

    def step_mape(cal, warp):
        to_img = lambda w: warp(cv2.perspectiveTransform(  # noqa: E731
            w.reshape(-1, 1, 2), true_H).reshape(-1, 2))
        d = np.linalg.norm(cal.to_pitch(to_img(p1)) - cal.to_pitch(to_img(p0)), axis=1)
        return 100.0 * np.abs(d - truth).mean() / truth.mean()

    assert step_mape(cal, lambda p: p) < 0.01                  # perspective, exact
    assert step_mape(flat, cyl) > 10.0, step_mape(flat, cyl)   # >10% band: premise fails
    assert step_mape(pano, cyl) < 5.0, step_mape(pano, cyl)    # inside the gate

    # A misclick must be named, not smeared into the nearest strip.
    bad = dict(clicks)
    bad["corner_bl"] = (img[0][0] + 90, img[0][1] + 90)
    bad_pano = PanoramicCalibrator(pitch, bad)
    assert bad_pano.outliers == ["corner_bl"], bad_pano.outliers
    assert bad_pano.rms_error_m < 0.01, bad_pano.rms_error_m

    try:
        PanoramicCalibrator(pitch, {k: cyl_clicks[k] for k in list(cyl_clicks)[:4]})
        raise AssertionError("should have rejected 4 points for a piecewise fit")
    except ValueError:
        pass

    # --- remap: flatten the stitch first, THEN one ordinary homography ---
    # Same cylindrical fixture the piecewise calibrator can only mitigate. The
    # remap should not mitigate it, it should remove it.
    mx, my = _cyl_maps((1920, 1080), 960.0, 540.0, 800.0)
    # remap_points must undo the warp: cylindrical clicks -> perspective pixels.
    back = remap_points(mx, my, np.array([cyl_clicks[k] for k in lm]))
    assert np.abs(back - img).max() < 0.05, np.abs(back - img).max()
    # ...and it must refuse a point the tables do not cover rather than
    # returning the nearest edge pixel, which would be a silent 10 m error.
    try:
        remap_points(mx, my, [[-5000.0, 540.0]])
        raise AssertionError("should have rejected an uncovered point")
    except ValueError:
        pass

    rem = get_calibrator("remap", pitch=pitch, clicks=dict(zip(lm, map(tuple, back))),
                         mapx=mx, mapy=my)
    assert isinstance(rem, RemapCalibrator)
    assert rem.projection_type == "remap"
    assert rem.rms_error_m < 0.01, rem.rms_error_m
    # The number that matters: held-out grid, raw panorama pixels in.
    e_rem = worst(rem, remap_points(mx, my, cyl(grid_i)), grid_w)
    assert e_rem < 1e-3, e_rem                      # vs e_pano > 1.0 m piecewise
    assert e_rem < 1e-4 * e_flat, (e_rem, e_flat)
    assert step_mape(rem, lambda p: remap_points(mx, my, cyl(p))) < 0.01

    # stats.py consumes .H (Jacobian for kalman_ground) and .pitch/.rms_error_m.
    assert rem.H.shape == (3, 3) and np.allclose(rem.H, rem._inner.H)
    assert rem.pitch.length == pitch.length

    # undistort_frame actually flattens: a straight pitch line drawn on the
    # panorama must come out straight. This is the model-free check.
    pano_img = np.zeros((1080, 1920), np.uint8)
    edge = _cylindrical(np.column_stack([np.linspace(200, 1700, 400),
                                         np.full(400, 700.0)]), 960.0, 540.0, 800.0)
    for x, y in edge.astype(int):
        cv2.circle(pano_img, (x, y), 3, 255, -1)
    bow = np.ptp(edge[:, 1])
    ys = [np.flatnonzero(col)[0] for col in rem.undistort_frame(pano_img)[:, 300:1600].T
          if col.any()]
    assert bow > 40, bow                            # the fixture really is bent
    assert np.ptp(ys) <= 4, (np.ptp(ys), bow)       # and comes out straight

    # yscale resamples the canvas without moving the geometry: the same raw
    # point must land at yscale x the row it landed on before.
    import tempfile, os  # noqa: PLC0415 - test-only

    with tempfile.TemporaryDirectory() as d:
        np.savez(os.path.join(d, "m.npz"), mapx=mx, mapy=my)
        mx3, my3 = load_maps(os.path.join(d, "m.npz"), yscale=3.0)
    assert mx3.shape == (3 * mx.shape[0], mx.shape[1]), mx3.shape
    p1 = remap_points(mx, my, [cyl_clicks["centre"]])[0]
    p3 = remap_points(mx3, my3, [cyl_clicks["centre"]])[0]
    assert abs(p3[0] - p1[0]) < 0.5 and abs(p3[1] - 3 * p1[1]) < 1.5, (p1, p3)

    # pitch_band derives the crop from the geometry. Two properties, both of
    # which a hand-typed band gets wrong sooner or later: it holds every ground
    # point of the pitch-plus-margin AND the head above the highest one, and it
    # stays centred on the row camera_from_homography assumes is the principal
    # point. (On a canvas where the pitch does NOT straddle the centre row the
    # second property makes the band wide and the crop nearly free - which is
    # this synthetic fixture, and is the right way round: it declines to crop
    # rather than quietly moving the principal point.)
    b = pitch_band(rem, (1920, 1080))
    assert abs(b[0] + b[1] - 1080) <= 2, b
    prow = cv2.perspectiveTransform(np.array([[[pitch.length / 2, 0.0]],
                                              [[pitch.length / 2, pitch.width]]]),
                                    true_H).reshape(-1, 2)[:, 1]
    assert b[0] < prow.min() and prow.max() < b[1], (b, prow)

    # crop + isotropic scale: a WINDOW on the same projection, upsampled. The
    # geometry must survive it exactly - the whole point of doing it this way
    # rather than with yscale is that nothing anisotropic happens.
    band, sc = (390, 690), 2.5          # symmetric about row 540, as pitch_band makes it
    mxc, myc = (cv2.resize(m[band[0]:band[1]], None, fx=sc, fy=sc,
                           interpolation=cv2.INTER_LINEAR) for m in (mx, my))
    with tempfile.TemporaryDirectory() as d:
        np.savez(os.path.join(d, "m.npz"), mapx=mx, mapy=my)
        mxl, myl = load_maps(os.path.join(d, "m.npz"), crop=band, scale=sc)
    assert np.allclose(mxl, mxc) and np.allclose(myl, myc)
    assert mxl.shape == (round((band[1] - band[0]) * sc), round(1920 * sc)), mxl.shape

    # Landmarks inside the band survive the round trip; the fit is as exact as
    # it was on the full canvas, and the held-out grid still lands sub-mm.
    inband = [k for k in lm if band[0] < back[list(lm).index(k)][1] < band[1]]
    crop_back = remap_points(mxl, myl, np.array([cyl_clicks[k] for k in inband]))
    aff = (np.array([back[list(lm).index(k)] for k in inband]) - [0, band[0]] + 0.5) * sc - 0.5
    assert np.abs(crop_back - aff).max() < 0.05, np.abs(crop_back - aff).max()
    rem_c = get_calibrator("remap", pitch=pitch, clicks=dict(zip(inband, map(tuple, crop_back))),
                           mapx=mxl, mapy=myl, crop=band, scale=sc)
    assert rem_c.crop == band and rem_c.scale == sc
    gi = remap_points(mxl, myl, cyl(grid_i[(grid_i[:, 1] > band[0] + 5)
                                           & (grid_i[:, 1] < band[1] - 5)]))
    gw = grid_w[(grid_i[:, 1] > band[0] + 5) & (grid_i[:, 1] < band[1] - 5)]
    assert worst(rem_c, gi, gw) < 1e-2, worst(rem_c, gi, gw)

    # Isotropy is the property that makes this safe, so assert it rather than
    # trusting the call site: a crop+scale must leave the recovered focal at the
    # same FRACTION of frame width. yscale is what does not - see make_prior.
    import sceneprior  # noqa: PLC0415

    f_full = sceneprior.camera_from_homography(rem.H, (1920, 1080))
    f_crop = sceneprior.camera_from_homography(
        rem_c.H, (mxl.shape[1], mxl.shape[0]))
    assert abs(f_crop.focal_px / mxl.shape[1] - f_full.focal_px / 1920) < 0.02, (
        f_full.focal_px, f_crop.focal_px)
    assert abs(f_crop.height_m - f_full.height_m) < 0.05 * f_full.height_m, (
        f_full.height_m, f_crop.height_m)

    # No tables (loaded from JSON without them) must still project, and must
    # refuse to remap rather than pretend.
    naked = RemapCalibrator(pitch, None, None, rem._inner, maps="/nonexistent")
    assert np.allclose(naked.to_pitch(back), rem.to_pitch(back))
    try:
        naked.undistort_frame(pano_img)
        raise AssertionError("should have refused to remap without tables")
    except ValueError:
        pass

    # --- undistort: barrel distortion at the frame edges, where corners live ---
    K = np.array([[900.0, 0, 960.0], [0, 900.0, 540.0], [0, 0, 1.0]])
    D = np.array([-0.28, 0.09, 0.0, 0.0, 0.0])
    dist_clicks = dict(zip(lm.keys(), map(tuple, _barrel(img, K, D))))
    naive = get_calibrator("perspective", pitch=pitch, clicks=dist_clicks)
    fixed = get_calibrator("perspective", pitch=pitch, clicks=dist_clicks,
                           camera_matrix=K, dist_coeffs=D)
    assert isinstance(fixed, UndistortWrapper)
    assert fixed.projection_type == "perspective"  # wrapper is transparent
    e_naive = worst(naive, _barrel(grid_i, K, D), grid_w)
    e_fixed = worst(fixed, _barrel(grid_i, K, D), grid_w)
    assert e_naive > 0.5, e_naive
    assert e_fixed < 0.01, e_fixed

    assert get_calibrator("perspective", pitch=pitch, clicks=clicks,
                          camera_matrix=K, dist_coeffs=None).projection_type == "perspective"

    # --- bow: a residual row-bow that a remap did not remove -----------------
    # Dense literal-metre landmarks, because deg+1 free parameters need more
    # than the 21 named markings. Same camera, bent by shifting the image ROW as
    # a function of the image COLUMN - which is what 118575's REMAPPED
    # touchlines actually still do; see BowWrapper for the measurement.
    bgx, bgy = np.meshgrid(np.linspace(0, pitch.length, 11), np.linspace(0, pitch.width, 5))
    bw = np.column_stack([bgx.ravel(), bgy.ravel()])
    bi = cv2.perspectiveTransform(bw.reshape(-1, 1, 2), true_H).reshape(-1, 2)
    xc = bi[:, 0].mean()
    bend = lambda q: np.column_stack(  # noqa: E731
        [q[:, 0], q[:, 1] + 25.0 * (((q[:, 0] - xc) / 700.0) ** 2 - 0.4)])
    bowcal = get_calibrator("perspective", pitch=pitch, bow_deg=4,
                            clicks={f"({x},{y})": tuple(v) for (x, y), v in zip(bw, bend(bi))})
    plain = get_calibrator("perspective", pitch=pitch,
                           clicks={f"({x},{y})": tuple(v) for (x, y), v in zip(bw, bend(bi))})
    e_plain, e_bow = worst(plain, bend(grid_i), grid_w), worst(bowcal, bend(grid_i), grid_w)
    assert e_plain > 2.0, e_plain               # the fixture really is bent
    assert e_bow < 0.05 * e_plain, (e_bow, e_plain)
    # ...and it must DECLINE on a frame that is not bent rather than spend five
    # parameters on noise. In-sample fit is not evidence, so the constructor
    # asks the held-out question itself - see _must_beat_a_homography.
    try:
        get_calibrator("perspective", pitch=pitch, bow_deg=4,
                       clicks={f"({x},{y})": tuple(v) for (x, y), v in zip(bw, bi)})
        raise AssertionError("bow was accepted on a frame that is not bowed")
    except ValueError:
        pass

    # --- round-trip every path through JSON, projection_type included ---
    for original in (cal, pano, rem, fixed, bowcal):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.json")
            original.save(p)
            with open(p) as f:
                assert json.load(f)["projection_type"] == original.projection_type
            back = load(p)
            assert type(back) is type(original), (type(back), type(original))
            src = bend(bi) if original is bowcal else img
            assert np.allclose(back.to_pitch(src), original.to_pitch(src), atol=1e-6)
            assert abs(back.pitch.length - pitch.length) < 1e-9  # never 105x68

    try:
        get_calibrator("equirectangular", pitch=pitch, clicks=clicks)
        raise AssertionError("should have rejected an unknown projection type")
    except ValueError:
        pass

    # --- literal "(x,y)" landmark names, for dense keypoint sets -------------
    # A coordinate name must land exactly where the same point named by a
    # landmark does; if these two fits disagree the parser is the reason.
    lit = {f"({x},{y})": px for (x, y), px in
           ((pitch.landmarks()[n], clicks[n]) for n in clicks)}
    cal_lit = get_calibrator("perspective", pitch=pitch, clicks=lit)
    assert cal_lit.rms_error_m < 0.01, cal_lit.rms_error_m
    assert worst(cal_lit, grid_i, grid_w) < 1e-3, worst(cal_lit, grid_i, grid_w)
    # A name that is neither a landmark nor a coordinate must still be refused,
    # because silently dropping a typo'd click is a calibration that looks fine.
    try:
        get_calibrator("perspective", pitch=pitch, clicks={**lit, "corner_xl": (1.0, 2.0)})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown landmark name was not refused")

    # --- from_keypoints on the real 118575 file, if it is checked out --------
    # The number this path exists for. If it regresses past ~1.5 m the panorama
    # is no longer measurable and PathCRF (F1 0.46 at 1.0 m) goes with it.
    kp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "data/footage/soccertrack_calib_118575/118575_keypoints.json")
    if os.path.exists(kp_path):
        cal_kp = from_keypoints(kp_path, Pitch())
        pts = np.array(list(cal_kp.points.values()), dtype=np.float64)
        want = np.array([ast.literal_eval(n) for n in cal_kp.points], dtype=np.float64)
        med = float(np.median(np.linalg.norm(cal_kp.to_pitch(pts) - want, axis=1)))
        assert med < 1.5, f"dense-keypoint calibration regressed to {med:.2f} m"
        assert np.allclose(from_dict(cal_kp.to_dict()).to_pitch(pts), cal_kp.to_pitch(pts)), \
            "sigma_scale did not survive to_dict/from_dict"

    # ---- estimate_k1: recover a distortion we PUT IN --------------------
    # The estimator's whole claim is that it can see lens distortion in a frame
    # with no labels. The only honest test of that is to bend known-straight
    # lines by a known amount and check it reads the amount back. Fitting real
    # footage and admiring the number proves nothing, because there is no truth
    # to compare against - which is exactly the trap `BowWrapper` below fell
    # into on landmarks.
    W_, H_ = 1920, 1080
    C_ = np.array([W_ / 2.0, H_ / 2.0])
    F_ = float(W_)

    def _distort(p, k1):
        """Apply barrel distortion to points - the inverse of what we correct."""
        nz = (np.asarray(p, float) - C_) / F_
        return C_ + nz * (1.0 + k1 * (nz ** 2).sum(1, keepdims=True)) * F_

    def _render(k1_true):
        img = np.zeros((H_, W_, 3), np.uint8)
        img[:] = (60, 140, 70)
        rng = np.random.default_rng(11)
        # Turf speckle so `grass_mask` sees grass rather than a flat field, and
        # kept strongly green so `paint_mask`'s saturation gate excludes it.
        for x, y in rng.uniform([0, 0], [W_, H_], (1500, 2)):
            cv2.circle(img, (int(x), int(y)), int(rng.integers(2, 6)),
                       (int(rng.integers(20, 60)), int(rng.integers(100, 210)),
                        int(rng.integers(20, 70))), -1)
        # Straight world lines, drawn as densely-sampled DISTORTED points, so
        # the bend is in the imagery rather than faked afterwards.
        for y0 in (300.0, 640.0, 900.0):
            pts = _distort(np.column_stack([np.linspace(40, W_ - 40, 900),
                                            np.full(900, y0)]), k1_true)
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), (245, 245, 245), 4)
        return img

    # No distortion in, ~no distortion out.
    r0 = estimate_k1(_render(0.0))
    assert r0["segments"] >= 2, r0
    assert abs(r0["k1"]) < 0.03, r0

    # Barrel in, correction out. NOTE THE SIGN, it is the thing most likely to
    # be got backwards: `estimate_k1` returns the coefficient that STRAIGHTENS
    # the image, so imagery bent by `_distort(p, -c)` must come back as `+c`.
    # A sign error here would not halve the correction, it would DOUBLE the
    # distortion, so it is asserted rather than assumed.
    for k_want in (0.08, 0.15):
        r = estimate_k1(_render(-k_want))
        assert r["segments"] >= 2, (k_want, r)
        assert abs(r["k1"] - k_want) < 0.05, (k_want, r["k1"])
        assert r["bow_px_corrected"] < r["bow_px"], (k_want, r)

    # `undistorted()` must compose, keep the projection type, and actually move
    # a point that is off-centre (and leave the principal point alone).
    _p = Pitch()
    _clicks = {"(0.0, 0.0)": (100.0, 900.0), "(105.0, 0.0)": (1800.0, 900.0),
               "(105.0, 68.0)": (1500.0, 400.0), "(0.0, 68.0)": (420.0, 400.0)}
    _inner = PerspectiveCalibrator.solve(_p, _clicks)
    _u = undistorted(_inner, -0.08, (W_, H_))
    assert _u.projection_type == "perspective", _u.projection_type
    _centre = np.array([[W_ / 2.0, H_ / 2.0]])
    assert np.allclose(_u.to_pitch(_centre), _inner.to_pitch(_centre), atol=1e-6)
    _edge = np.array([[80.0, 980.0]])
    assert not np.allclose(_u.to_pitch(_edge), _inner.to_pitch(_edge), atol=1e-3)

    print(f"calibrators.py ok  (k1 recovered {estimate_k1(_render(-0.08))['k1']:+.3f} "
          f"as the correction for a planted -0.080)")


from paths import FOOTAGE

CALIB_DIR = str(FOOTAGE / "soccertrack_calib_118575")
VIDEO = str(FOOTAGE / "118575_panorama_1st_half.mp4")


def _measure_real(calib_dir: str = CALIB_DIR, video: str = VIDEO,
                  evidence: str = str(FOOTAGE / "evidence_remapped.png"),
                  write_calib: str | None = None, bow: int | None = None) -> None:
    """The before/after on the real panorama, against its 65 published keypoints.

    Run: `python calibrators.py --real`. Kept out of _demo() because it needs
    ~26 MB of tables and a 4 GB video that no CI box has.
    """
    import glob, os  # noqa: PLC0415

    kp = json.load(open(glob.glob(os.path.join(calib_dir, "*keypoints.json"))[0]))
    world = np.array([[float(v) for v in k.strip("()").split(",")] for k in kp])
    raw = np.array(list(kp.values()), dtype=np.float64)
    mapx, mapy = load_maps(calib_dir)
    rem = remap_points(mapx, mapy, raw)

    # VERIFY THE INVERSION FIRST. Every metre below is downstream of it, and a
    # remap inverted in the wrong direction still returns plausible pixels.
    # (a) Self-consistency: take known destination pixels, read their source
    #     straight out of the tables, invert, and land back where we started.
    rng0 = np.random.default_rng(0)
    v = (mapx >= 0) & (mapx <= 4095) & (mapy >= 0) & (mapy <= 1079)
    yy, xx = np.where(v)
    k = rng0.choice(len(yy), 200, replace=False)
    d0 = np.column_stack([xx[k], yy[k]]).astype(float)
    e0 = np.linalg.norm(remap_points(mapx, mapy,
                                     np.column_stack([mapx[yy[k], xx[k]],
                                                      mapy[yy[k], xx[k]]])) - d0, axis=1)
    print(f"inversion self-check  dest->src->dest over 200 pixels: "
          f"median {np.median(e0):.3f} px  max {e0.max():.3f} px")
    assert e0.max() < 0.5, e0.max()
    # (b) Independent witness: the vendor's own homography is stated in THEIR
    #     remapped frame. World metres through it must land on our inverted
    #     keypoints. We do not use their H for anything - it is only a witness
    #     that our remapped space and theirs are the same space.
    vh = glob.glob(os.path.join(calib_dir, "*homography.npy"))
    if vh:
        p = cv2.perspectiveTransform(world.reshape(-1, 1, 2), np.load(vh[0])).reshape(-1, 2)
        e1 = np.linalg.norm(p - rem, axis=1)
        print(f"inversion witness     vendor H(world) vs our inverse(raw kp): "
              f"median {np.median(e1):.1f} px over a {np.ptp(rem[:, 0]):.0f}x"
              f"{np.ptp(rem[:, 1]):.0f} px span")
        assert np.median(e1) < 20, np.median(e1)
    print()

    fit = lambda s, d: cv2.findHomography(s, d, 0)[0]  # noqa: E731
    res = lambda s, d, H: np.linalg.norm(  # noqa: E731
        cv2.perspectiveTransform(s.reshape(-1, 1, 2), H).reshape(-1, 2) - d, axis=1)

    def loo(s, d):
        return np.array([res(s[i:i + 1], d[i:i + 1], fit(s[np.arange(len(s)) != i],
                                                         d[np.arange(len(s)) != i]))[0]
                         for i in range(len(s))])

    H = fit(rem, world)

    # The cost side of the ledger: the vendor's canvas undersamples the source,
    # in BOTH axes and by similar factors - which is the fact that makes an
    # isotropic fix possible at all. (A 50:1 anisotropy would have meant square
    # pixels were already gone and the scene prior with them.)
    gy, gx = np.meshgrid(np.linspace(2, 103, 60), np.linspace(2, 66, 30))
    inside = cv2.perspectiveTransform(np.column_stack([gy.ravel(), gx.ravel()])
                                      .reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2).astype(int)
    sy = np.gradient(mapy, axis=0)[inside[:, 1], inside[:, 0]]
    sx = np.gradient(mapx, axis=1)[inside[:, 1], inside[:, 0]]
    print(f"source px per remapped px over the pitch interior:\n"
          f"  vertical    min {sy.min():.2f} median {np.median(sy):.2f} max {sy.max():.2f}\n"
          f"  horizontal  min {sx.min():.2f} median {np.median(sx):.2f} max {sx.max():.2f}")

    # Crop to the band a player can occupy, then upscale ISOTROPICALLY by the
    # measured undersampling, which is what recovers the source pixels the
    # vendor canvas threw away. Both are derived here, not typed: the band from
    # the geometry (pitch_band), the factor from the squash just measured.
    vendor = RemapCalibrator(Pitch(105.0, 68.0), None, None,
                             PerspectiveCalibrator(Calibration(pitch=Pitch(105.0, 68.0), H=H)))
    band = pitch_band(vendor, (mapx.shape[1], mapx.shape[0]))
    scale = round(float(np.median(sy)), 1)
    mapx_c, mapy_c = load_maps(calib_dir, crop=band, scale=scale)
    rem_c = remap_points(mapx_c, mapy_c, raw)
    # The crop+scale is an affine window, so the keypoints must land exactly
    # where that affine puts them. Inverting the tables again is an independent
    # route to the same answer, and it is the one that catches an off-by-one row
    # or an axis swapped in load_maps - both of which look like a slightly worse
    # fit rather than like a bug.
    # +-0.5 because cv2.resize aligns pixel CENTRES: dst j samples src
    # (j+0.5)/scale - 0.5. Drop it and every landmark is 0.75 px out at 2.9x,
    # which is well inside "looks like a slightly noisier fit".
    affine = (rem - [0.0, band[0]] + 0.5) * scale - 0.5
    assert np.abs(rem_c - affine).max() < 1.0, np.abs(rem_c - affine).max()
    Hc = fit(rem_c, world)
    print(f"\npitch band rows {band[0]}..{band[1]} of {mapx.shape[0]} "
          f"({band[1] - band[0]} rows, {100 * (band[1] - band[0]) / mapx.shape[0]:.1f}% of the canvas)"
          f"  x{scale:g} isotropic -> {mapx_c.shape[1]}x{mapx_c.shape[0]}\n")

    print(f"{'position error, metres':50s} {'median':>8s} {'p95':>8s} {'max':>8s}")
    for name, src in (("single homography on RAW panorama", raw),
                      ("REMAP then homography", rem),
                      ("REMAP + pitch-band crop, then homography", rem_c)):
        for tag, e in ((" (fit residual)", res(src, world, fit(src, world))),
                       (" (leave-one-out)", loo(src, world))):
            print(f"{name + tag:50s} {np.median(e):8.2f} {np.percentile(e, 95):8.2f} "
                  f"{e.max():8.2f}")

    # The crop is an affine WINDOW on the same projection, so it must describe
    # the same physical camera: focal scales with the canvas, height and
    # position do not move. A focal that changes as a FRACTION of frame width is
    # the yscale failure signature (469 -> 182 px on a 3x taller canvas, which
    # collapsed the height gate onto 79% of genuine players), not a crop.
    import sceneprior  # noqa: PLC0415

    cropped = RemapCalibrator(Pitch(105.0, 68.0), None, None,
                              PerspectiveCalibrator(Calibration(pitch=Pitch(105.0, 68.0), H=Hc)))
    print()
    for lbl, c, wh in (("vendor ", vendor, (mapx.shape[1], mapx.shape[0])),
                       ("cropped", cropped, (mapx_c.shape[1], mapx_c.shape[0]))):
        cam = sceneprior.camera_from_homography(c.H, wh)
        print(f"  camera from the {lbl} canvas {wh[0]}x{wh[1]}: f={cam.focal_px:7.1f} px "
              f"(= {cam.focal_px / wh[0]:.4f} x frame width)  height={cam.height_m:5.2f} m  "
              f"at pitch ({cam.centre_m[0]:.1f}, {cam.centre_m[1]:.1f}) m")

    # Model-free, no homography involved: a straight pitch line must be straight
    # in a perspective image. This is what the remap is FOR, and it is the one
    # number no fitting choice can flatter.
    print("\ntouchline bow (deviation from a straight line, px):")
    for lbl, m in (("far  (pitch y=0)", world[:, 1] == 0), ("near (pitch y=68)", world[:, 1] == 68)):
        for sp, p in (("raw", raw[m]), ("remapped", rem[m])):
            q = p - p.mean(0)
            u = np.linalg.svd(q, 0)[2][0]
            d = np.abs(q @ [-u[1], u[0]])
            print(f"  {lbl} {sp:9s} rms {np.sqrt((d ** 2).mean()):6.2f}  max {d.max():6.2f}")

    # What is left after the remap is mostly the DATASET's own annotation error,
    # not our model's. The bow numbers above are a direct measurement of it once
    # the frame is flat (a straight line cannot bow), so push that same scatter
    # through the fit and see how much of the residual it accounts for. Without
    # this the 2.4 m reads as "our calibration is 2.4 m wrong", which it is not.
    #
    # Run it on BOTH canvases. The same physical annotation scatter is `scale`
    # times as many pixels on the cropped canvas, so if the two columns agree in
    # METRES the crop has changed the sampling and nothing else - which is the
    # claim, and the one an eyeballed "the numbers look similar" cannot make.
    rng = np.random.default_rng(0)
    print("\nfit error implied by pure annotation scatter (Monte Carlo, 30 draws):")
    print(f"{'sigma, remapped px':28s} {'vendor canvas':>16s} {'cropped canvas':>16s}")
    for sig in (1.5, 2.5, 3.5):
        got = []
        for Hx, s_ in ((H, sig), (Hc, sig * scale)):
            perfect = cv2.perspectiveTransform(world.reshape(-1, 1, 2),
                                               np.linalg.inv(Hx)).reshape(-1, 2)
            got.append(np.mean([np.median(res(p, world, fit(p, world)))
                                for p in (perfect + rng.normal(0, s_, perfect.shape)
                                          for _ in range(30))]))
        print(f"  {sig:.1f}{'':23s} {got[0]:13.2f} m {got[1]:13.2f} m")

    # 105 x 68 m is not an assumption here, it is what the dataset published:
    # the 65 keypoints' own world coordinates span exactly x 0..105, y 0..68,
    # and the ground truth puts 99.4% of all player positions inside that
    # rectangle. Every other venue still types its real numbers - see pitch.py.
    assert (world.min(0).tolist(), world.max(0).tolist()) == ([0.0, 0.0], [105.0, 68.0])

    # Round-trip a real remap calibration, tables and all.
    #
    # `points` is NOT decoration. This function used to hand Calibration() only
    # an H, and `points` defaults to {} - so every calibration ever written by
    # --write-calib (including the shipped data/jobs/118575_full_1st one) landed
    # on disk with "points": {}. The landmarks the fit was SOLVED from were
    # thrown away at save time, which makes the file permanently unscorable:
    # calcheck measures a calibration by holding landmarks out of the fit, and
    # there was nothing to hold out. The rms_error_m in those files is also
    # in-sample over the fit points, so the one number that DID survive is the
    # one that flatters itself. Keys are the keypoint file's own literal
    # "(x, y)" metre names, which resolve_landmarks already understands;
    # values are the remapped+cropped pixels the fit actually used.
    inner_clicks = {n: (float(p[0]), float(p[1])) for n, p in zip(kp, rem_c)}
    inner: Calibrator = PerspectiveCalibrator(Calibration(
        pitch=Pitch(105.0, 68.0), H=Hc, points=inner_clicks,
        rms_error_m=float(np.sqrt((res(rem_c, world, Hc) ** 2).mean()))))
    b = None
    if bow:
        # The remap flattens most of the stitch but not all of it - the
        # touchline bow above is what is left, and it is still metres out
        # there. BowWrapper takes it out; `python calcheck.py --real` is the
        # held-out ledger, region splits included.
        b = BowWrapper.solve(Pitch(105.0, 68.0), inner_clicks, deg=bow)
        inner = b.inner          # the bow goes back on OUTSIDE the remap, once
        e = np.linalg.norm(b.to_pitch(rem_c) - world, axis=1)
        print(f"\nbow(deg={bow}) on the cropped canvas: fit residual median "
              f"{np.median(e):.2f} m p95 {np.percentile(e, 95):.2f} max {e.max():.2f} "
              f"(vs {np.median(res(rem_c, world, Hc)):.2f} m for the plain homography)")
    cal = RemapCalibrator(Pitch(105.0, 68.0), mapx_c, mapy_c, inner,
                          maps=calib_dir, crop=band, scale=scale)
    if b is not None:
        # Outside is where from_dict puts it back, and every calibrator between
        # the two passes points through unchanged, so the position is a
        # formality - but wrapping BOTH sides applies the warp twice, silently,
        # for about 2 m. Wrap once.
        cal = BowWrapper(cal, b.coeffs, b.ymod, b.mu, b.sd, b.xspan,
                         points=inner_clicks)
    import tempfile  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.json")
        cal.save(p)
        back = load(p)
    assert back._mapx is not None and back.projection_type == "remap"
    assert back.crop == band and back.scale == scale, (back.crop, back.scale)
    assert np.allclose(back.to_pitch(rem_c), cal.to_pitch(rem_c))
    # The landmarks must survive the trip, to the pixel - this is what a
    # re-score depends on, and it is what was silently absent before.
    assert len(back.points) == len(world), (len(back.points), len(world))
    assert np.allclose(np.array([back.points[n] for n in kp]), rem_c), \
        "landmarks did not round-trip through save/load"

    # The calibration this whole function just validated, on disk. It was built
    # and thrown away before, so every run had to be re-done by hand - and a
    # hand-built one is exactly what nobody can check. yscale stays 1 and always
    # will: it resamples one axis, and sceneprior assumes square pixels (see
    # make_prior in track.py). crop+scale is the isotropic way to buy the same
    # detail. The uncropped calibration on the vendor canvas is kept as
    # model/validation/calib_118575_remap.json, as the baseline this is measured
    # against; it is a `crop=None, scale=1` build of the same H.
    if write_calib:
        cal.save(write_calib)
        print(f"\nwrote {write_calib}: remap calibration on the "
              f"{mapx_c.shape[1]}x{mapx_c.shape[0]} cropped canvas, "
              f"rms {cal.rms_error_m:.2f} m over {len(world)} shipped keypoints")

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"\ncould not read {video}; no evidence frame written")
        return
    # Raw on top, flattened below, keypoints marked on both. One image, and the
    # bowed touchline in the top half is the whole argument. The flattened half
    # is scaled back to the raw width purely to stack them - the cropped canvas
    # is 2.9x wider and a fifth as tall, which IS the second half of the point.
    out = back.undistort_frame(frame)
    for u, v in raw:
        cv2.circle(frame, (int(u), int(v)), 5, (0, 0, 255), -1)
    for u, v in rem_c:
        cv2.circle(out, (int(u), int(v)), 5 * int(round(scale)), (0, 0, 255), -1)
    out = cv2.resize(out, (frame.shape[1], round(out.shape[0] * frame.shape[1] / out.shape[1])))
    for im, txt in ((frame, "RAW panorama (bowed)"),
                    (out, f"REMAPPED + pitch-band crop x{scale:g} (flat)")):
        cv2.putText(im, txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
    cv2.imwrite(evidence, np.vstack([frame, out]))
    print(f"\nwrote {evidence}: raw above, remapped+cropped below, "
          f"65 keypoints mapped across by remap_points()")


if __name__ == "__main__":
    import sys

    if "--real" in sys.argv:
        i = sys.argv.index("--write-calib") if "--write-calib" in sys.argv else -1
        j = sys.argv.index("--bow") if "--bow" in sys.argv else -1
        _measure_real(write_calib=sys.argv[i + 1] if i >= 0 else None,
                      bow=int(sys.argv[j + 1]) if j >= 0 else None)
    else:
        _demo()
