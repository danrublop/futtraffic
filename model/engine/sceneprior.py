"""Scene priors a fixed camera gives you for free.

Two of them, both derived from the calibration you already have:

  1. Per-row expected player height in pixels. RETIRED from the product
     (track.py :: make_prior ships the polygon only); kept for experiments.
  2. A static pitch mask: crowd, dugout, bench and adjacent pitches are outside
     the pitch polygon, one projection of a rectangle of known size.

Both need the camera's focal length and height, recovered from the
ground-plane homography alone (Zhang's single-homography trick), which also
lets the calibration file check the mount height.
docs/reference/calibration.md#scene-prior (history: #evidence-scene-prior)
"""
import os
from dataclasses import dataclass

import cv2
import numpy as np

PLAYER_HEIGHT_M = 1.80
# A detection outside +-40% of the geometric expectation is not a player at that
# spot. Wide enough for a crouching keeper or a leaping header, tight enough to
# kill the crowd, half-occluded pairs, and boxes merged across two players.
HEIGHT_TOLERANCE = 0.40


@dataclass
class Camera:
    """Recovered from the homography. Everything here is a consequence of it."""

    focal_px: float
    height_m: float          # above the pitch plane
    centre_m: tuple          # camera position in pitch coordinates
    vz: np.ndarray           # vertical vanishing point, homogeneous
    G: np.ndarray            # pitch metres -> image pixels
    scale: float             # the lambda relating G to K[r1 r2 t]

    @property
    def mount_ok(self) -> bool:
        """Below ~6 m the stride-oscillation error alone breaks the 5% distance
        gate - see groundpoint.py. This is the check that catches a bad install
        before it silently poisons a season of numbers."""
        return self.height_m >= 6.0


# Loose on purpose: an existence check on the recovered geometry, not a mount
# recommendation. Real installs run 6-40 m and a drone 30-120 m; the nadir
# degeneracy this catches came back at 408.7 m.
MAX_CAMERA_H_M = 200.0


def camera_from_homography(H_img2pitch: np.ndarray, image_wh: tuple) -> Camera:
    """Recover focal length and camera pose from the ground-plane homography.

    Assumes square pixels, zero skew, and the principal point at image centre -
    all fine for a fixed broadcast-style rig, and none of them worth asking an
    operator to measure. Only the ORTHOGONALITY of the first two rotation
    columns is used, which gives f in closed form with no optimisation.
    """
    w, h = image_wh
    cx, cy = w / 2.0, h / 2.0
    G = np.linalg.inv(np.asarray(H_img2pitch, dtype=np.float64))
    G = G / G[2, 2]
    g1, g2 = G[:, 0], G[:, 1]

    # K^-1 g, without forming K^-1: strip the principal point, defer the 1/f.
    a1 = np.array([g1[0] - cx * g1[2], g1[1] - cy * g1[2]])
    a2 = np.array([g2[0] - cx * g2[2], g2[1] - cy * g2[2]])

    # Two independent constraints on f, from r1 and r2 being orthonormal:
    #   orthogonality  b1.b2 = 0     -> f^2 = -(a1.a2) / (g1z*g2z)
    #   equal norm     |b1| = |b2|   -> f^2 = (|a1|^2 - |a2|^2) / (g2z^2 - g1z^2)
    # Each degenerates in a different geometry, so take whichever has the larger
    # denominator. Orthogonality fails when the camera is square-on to the pitch
    # (one axis carries no depth, so g1z -> 0) - which is not an exotic case, it
    # is a camera mounted at the halfway line, i.e. the recommended install.
    d_orth = g1[2] * g2[2]
    d_norm = g2[2] ** 2 - g1[2] ** 2
    if abs(d_orth) >= abs(d_norm):
        f2 = -float(a1 @ a2) / float(d_orth) if d_orth else -1.0
    else:
        f2 = float(a1 @ a1 - a2 @ a2) / float(d_norm) if d_norm else -1.0
    if not np.isfinite(f2) or f2 <= 0:
        raise ValueError(
            "no real focal length from this homography - the view may be "
            "fronto-parallel, or the calibration points may be bad")
    f = float(np.sqrt(f2))

    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    B = np.linalg.inv(K) @ G
    lam = 1.0 / np.linalg.norm(B[:, 0])
    # Two signs the closed form leaves open, and both fail silently.
    # (1) lam: the pitch has to be IN FRONT of the camera. t[2] is the depth of
    #     the pitch origin, so its sign settles it.
    if B[2, 2] * lam < 0:
        lam = -lam
    r1 = B[:, 0] * lam
    r2 = B[:, 1] * lam
    r3 = np.cross(r1, r2)
    t = B[:, 2] * lam
    R = np.column_stack([r1, r2, r3])
    centre = -R.T @ t

    # (2) r1 x r2 is "up out of the pitch" only if the (x, y) axes are
    #     right-handed with up, and nothing in a homography says which way they
    #     run (on 118575 they do not). A camera below the pitch is not a thing,
    #     so flip. calibration-evidence (not published)#evidence-scene-prior
    up = -1.0 if centre[2] < 0 else 1.0
    height_m = float(up * centre[2])

    # (3) A near-nadir view lands on a FINITE but absurd camera and raises
    #     nothing, and a prior built on it rejects every real player. Raise;
    #     track.make_prior catches it and runs unfiltered, never "no tracking".
    #     The bound is an existence check, not a mount recommendation.
    #     calibration-evidence (not published)#evidence-scene-prior
    if not 1.0 <= height_m <= MAX_CAMERA_H_M:
        raise ValueError(
            f"recovered camera height {height_m:.1f} m is outside 1-{MAX_CAMERA_H_M:.0f} m; "
            "this homography is near-degenerate for a single-view decomposition "
            "(a near-nadir or fronto-parallel view), so the size prior it would "
            "build would reject every real player")

    return Camera(focal_px=f, height_m=height_m,
                  centre_m=(float(centre[0]), float(centre[1])),
                  vz=up * (K @ r3), G=G, scale=lam)


def expected_height_px(cam: Camera, pitch_xy: np.ndarray) -> np.ndarray:
    """Pixel height of a standing player at each (X, Y) ground point in metres."""
    p = np.column_stack([np.asarray(pitch_xy, float),
                         np.ones(len(pitch_xy))])            # (N,3)
    # x = lambda*G*(X,Y,1) + Z*(K r3). The lambda is what puts G into the same
    # scale as K r3; drop it and the head offset is wrong by orders of magnitude
    # while the ground point still dehomogenises correctly - so the bug is
    # invisible unless you check the HEIGHTS, which is why the monotonicity
    # assert below exists.
    ground_h = cam.scale * (cam.G @ p.T).T                   # (N,3) homogeneous
    head_h = ground_h + PLAYER_HEIGHT_M * cam.vz
    ground = ground_h[:, :2] / ground_h[:, 2:3]
    head = head_h[:, :2] / head_h[:, 2:3]
    return np.linalg.norm(ground - head, axis=1)


def plausible_height(cam: Camera, cal, boxes: np.ndarray,
                     tol: float = HEIGHT_TOLERANCE) -> np.ndarray:
    """Boolean mask: does each box's height match the geometry where it stands?

    boxes are (N,4) x1,y1,x2,y2. The ground point is the box bottom, which is
    what we have before any pose model exists.
    """
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]])
    want = expected_height_px(cam, cal.to_pitch(feet))
    got = boxes[:, 3] - boxes[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(want > 0, got / want, np.inf)
    return (ratio > 1 - tol) & (ratio < 1 + tol)


def pitch_polygon(cam: Camera, cal, margin_m: float = 3.0) -> np.ndarray:
    """The pitch rectangle plus a margin, projected into image pixels."""
    L, W = cal.pitch.length, cal.pitch.width
    m = margin_m
    corners = np.array([[-m, -m], [L + m, -m], [L + m, W + m], [-m, W + m]], float)
    p = np.column_stack([corners, np.ones(4)])
    img_h = (cam.G @ p.T).T
    return img_h[:, :2] / img_h[:, 2:3]


def _demo() -> None:
    from pitch import Calibration, Pitch

    # Synthetic rig with KNOWN focal and height, so recovery can be checked
    # against ground truth rather than against itself.
    w, h = 1920, 1080
    f_true, cam_h = 1400.0, 12.0
    pitch = Pitch(105.0, 68.0)

    # Camera behind the touchline at mid-pitch, tilted down at the centre spot.
    C = np.array([52.5, -25.0, cam_h])
    fwd = np.array([0.0, 25.0, -cam_h]); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.column_stack([right, down, fwd]).T
    K = np.array([[f_true, 0, w / 2], [0, f_true, h / 2], [0, 0, 1.0]])
    P = K @ np.column_stack([R, -R @ C])
    G_true = P[:, [0, 1, 3]]                      # z=0 plane
    cal = Calibration(pitch=pitch, H=np.linalg.inv(G_true / G_true[2, 2]))

    cam = camera_from_homography(cal.H, (w, h))
    assert abs(cam.focal_px - f_true) / f_true < 0.02, (cam.focal_px, f_true)
    assert abs(cam.height_m - cam_h) / cam_h < 0.02, (cam.height_m, cam_h)
    assert cam.mount_ok, cam.height_m

    # A 6 m mount passes, a 4 m mount does not - the check that catches a bad
    # install from the calibration file alone.
    C_low = np.array([52.5, -25.0, 4.0])
    fwd2 = np.array([0.0, 25.0, -4.0]); fwd2 /= np.linalg.norm(fwd2)
    r2_ = np.cross(fwd2, [0, 0, 1.0]); r2_ /= np.linalg.norm(r2_)
    R2 = np.column_stack([r2_, np.cross(fwd2, r2_), fwd2]).T
    P2 = K @ np.column_stack([R2, -R2 @ C_low])
    G2 = P2[:, [0, 1, 3]]
    low = camera_from_homography(np.linalg.inv(G2 / G2[2, 2]), (w, h))
    assert abs(low.height_m - 4.0) < 0.3, low.height_m
    assert not low.mount_ok

    # A NEAR-NADIR CAMERA MUST REFUSE, not return a number - and this uses the
    # REAL calibration rather than a synthetic one, deliberately. A perfect
    # nadir built by hand degenerates all the way to f2 <= 0 and the guard
    # above already catches it. What ships is NEAR nadir with a noisy fit,
    # where the height is barely constrained and the noise blows it up while
    # everything stays finite. Only the committed calibration reproduces that.
    # The .json is in git; the 4K video is not and is not needed here.
    cpath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "model", "validation",
                         "calib_soccer_top__test__videos__D_20220220_1_1680_1710.json")
    import calibrators as _c
    nadir = _c.load(cpath)
    try:
        bad = camera_from_homography(nadir.H, (3840, 2160))
    except ValueError as e:
        assert "near-degenerate" in str(e), e
    else:
        raise AssertionError(f"near-nadir view did not refuse: {bad.height_m:.1f} m")

    # And the refusal must reach the caller as "no filtering", never as an
    # exception and never as zero detections - that contract is track.py's, and
    # it is what makes raising here the safe move rather than a new failure.
    import track as _t
    assert _t.make_prior(nadir, (3840, 2160)) is None

    # Players must shrink with distance from the camera, monotonically.
    ys = np.linspace(2.0, 66.0, 12)
    hs = expected_height_px(cam, np.column_stack([np.full(12, 52.5), ys]))
    assert np.all(np.diff(hs) < 0), hs
    assert 5 < hs.min() < hs.max() < 400, (hs.min(), hs.max())

    # A correctly-sized box passes; crowd-sized and merged-pair boxes do not.
    xy = np.array([[52.5, 34.0]])
    px = float(expected_height_px(cam, xy)[0])
    uv = cv2.perspectiveTransform(
        np.array([[[52.5, 34.0]]]), G_true / G_true[2, 2]).reshape(2)
    ok = np.array([[uv[0] - 5, uv[1] - px, uv[0] + 5, uv[1]]])
    assert plausible_height(cam, cal, ok)[0]
    assert not plausible_height(cam, cal, ok * np.array([1, 1, 1, 1]) +
                                np.array([0, -px * 1.2, 0, 0]))[0]

    # The polygon must contain the centre spot and exclude the image corners.
    # Asserted on pitch_polygon, which is what track.py actually uses via
    # cv2.pointPolygonTest - not on a rasterised mask nothing calls.
    poly = pitch_polygon(cam, cal).astype(np.float32)
    assert cv2.pointPolygonTest(poly, (float(uv[0]), float(uv[1])), False) >= 0
    for corner in ((5.0, 5.0), (w - 5.0, 5.0)):
        assert cv2.pointPolygonTest(poly, corner, False) < 0, corner
    # Clip to the image before measuring coverage: the pitch plus margin
    # projects WIDER than the frame, so raw contourArea reports >100% and means
    # nothing. The rasterised mask this replaced clipped implicitly.
    rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    _, inter = cv2.intersectConvexConvex(poly, rect)
    frac = float(cv2.contourArea(inter) / (w * h)) if inter is not None else 0.0
    assert 0.05 < frac < 0.95, frac

    print(f"  recovered f={cam.focal_px:.0f}px (true {f_true:.0f}), "
          f"height={cam.height_m:.2f}m (true {cam_h:.1f}), mount_ok={cam.mount_ok}")
    print(f"  player height {hs.max():.0f}px near -> {hs.min():.0f}px far; "
          f"pitch mask covers {frac * 100:.0f}% of frame")
    print("sceneprior.py ok")


if __name__ == "__main__":
    _demo()
