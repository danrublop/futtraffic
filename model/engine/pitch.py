"""Pitch geometry + image->pitch homography.

Every dimension is a constructor arg and the operator types the real numbers:
pitches differ in size, and a hardcoded size is a systematic error on every
distance. calibration-evidence (not published)#evidence-pitch-dimensions
"""
from dataclasses import dataclass, field
import json
import re

import cv2
import numpy as np

YD = 0.9144


@dataclass
class Pitch:
    """Pitch in metres, origin at a corner, x along the length.

    y = 0 is the FAR touchline (top of a side-on frame; landmark ids `_b`), y = width the
    NEAR one (`_t`). docs/reference/calibration.md#coordinates
    """

    length: float = 105.0
    width: float = 68.0
    # These four are fixed by the laws of the game at every level worldwide,
    # which is why they make better calibration landmarks than the corners.
    penalty_depth: float = 16.5
    penalty_width: float = 40.32
    goal_box_depth: float = 5.5
    goal_box_width: float = 18.32
    centre_circle_r: float = 9.15

    @classmethod
    def from_yards(cls, length_yd: float, width_yd: float, **kw) -> "Pitch":
        return cls(length=length_yd * YD, width=width_yd * YD, **kw)

    def landmarks(self) -> dict[str, tuple[float, float]]:
        """Named click targets, ordered roughly left-to-right for the operator.

        Deliberately more than the 4 a homography needs: a fixed wide camera
        sees all of these, and over-constraining is what absorbs lens
        distortion at the frame edges where the corners live.
        """
        L, W = self.length, self.width
        pd, pw = self.penalty_depth, self.penalty_width
        gd, gw = self.goal_box_depth, self.goal_box_width
        cy = W / 2
        return {
            "corner_bl": (0.0, 0.0),
            "corner_tl": (0.0, W),
            "pen_l_bl": (0.0, cy - pw / 2),
            "pen_l_tl": (0.0, cy + pw / 2),
            "pen_l_br": (pd, cy - pw / 2),
            "pen_l_tr": (pd, cy + pw / 2),
            "goal_l_br": (gd, cy - gw / 2),
            "goal_l_tr": (gd, cy + gw / 2),
            "half_b": (L / 2, 0.0),
            "half_t": (L / 2, W),
            "centre": (L / 2, cy),
            "circle_b": (L / 2, cy - self.centre_circle_r),
            "circle_t": (L / 2, cy + self.centre_circle_r),
            "goal_r_bl": (L - gd, cy - gw / 2),
            "goal_r_tl": (L - gd, cy + gw / 2),
            "pen_r_bl": (L - pd, cy - pw / 2),
            "pen_r_tl": (L - pd, cy + pw / 2),
            "pen_r_br": (L, cy - pw / 2),
            "pen_r_tr": (L, cy + pw / 2),
            "corner_br": (L, 0.0),
            "corner_tr": (L, W),
        }


_COORD_NAME = re.compile(r"^\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)$")


def resolve_landmarks(pitch: Pitch, clicks: dict) -> dict[str, tuple[float, float]]:
    """name -> pitch metres, for every name in `clicks`. Raises on anything else.

    A name is either a landmark from `pitch.landmarks()` or a literal pitch
    coordinate `"(x,y)"` in metres.

    The literal form is for keypoints a dataset ships or a line fit produces,
    never the operator flow: the named markings cluster in three x-groups and
    cannot straighten a panorama. calibration-evidence (not published)#evidence-piecewise-and-dense-keypoints

    Both `Calibration.solve` and `calibrators._arrays` route through here, so a
    name is legal in exactly one place.
    """
    lm = pitch.landmarks()
    out = {}
    for n in clicks:
        if n in lm:
            out[n] = lm[n]
            continue
        m = _COORD_NAME.match(n)
        if m is None:
            raise ValueError(f"unknown landmarks: {sorted(set(clicks) - set(lm) - set(out))}")
        out[n] = (float(m.group(1)), float(m.group(2)))
    return out


@dataclass
class Calibration:
    """Image->pitch homography, solved once per camera install."""

    pitch: Pitch
    H: np.ndarray  # 3x3, image px -> pitch metres
    points: dict[str, tuple[float, float]] = field(default_factory=dict)
    rms_error_m: float = 0.0  # over inliers only
    outliers: list[str] = field(default_factory=list)

    @classmethod
    def solve(cls, pitch: Pitch, clicks: dict[str, tuple[float, float]]) -> "Calibration":
        """clicks maps landmark name -> (x_px, y_px)."""
        lm = resolve_landmarks(pitch, clicks)
        if len(clicks) < 4:
            raise ValueError(f"need >=4 landmarks, got {len(clicks)}")

        names = sorted(clicks)
        src = np.array([clicks[n] for n in names], dtype=np.float64)
        dst = np.array([lm[n] for n in names], dtype=np.float64)

        # LMEDS needs >=5 and tolerates a misclick; fall back for exactly 4.
        method = cv2.LMEDS if len(names) >= 5 else 0
        H, mask = cv2.findHomography(src, dst, method)
        if H is None:
            raise ValueError("homography failed - are the points collinear?")

        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = np.linalg.norm(proj - dst, axis=1)
        keep = mask.ravel().astype(bool) if mask is not None else np.ones(len(names), bool)
        # Score the fit on inliers; a misclick shouldn't masquerade as a bad
        # calibration. Name the outliers so the UI can flag the actual point.
        rms = float(np.sqrt(np.mean(err[keep] ** 2)))
        outliers = [n for n, ok in zip(names, keep) if not ok]
        return cls(pitch=pitch, H=H, points=clicks, rms_error_m=rms, outliers=outliers)

    def to_pitch(self, pts_px: np.ndarray) -> np.ndarray:
        """(N,2) image px -> (N,2) pitch metres."""
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_dict(self) -> dict:
        return {
            # A single 3x3 is only valid for a flat perspective image. Stamping
            # the projection type into every file means a panorama calibration
            # can never be silently read back as a perspective one - see
            # calibrators.py and Blocker 1.
            "projection_type": "perspective",
            "pitch": self.pitch.__dict__,
            "H": self.H.tolist(),
            "points": self.points,
            "rms_error_m": self.rms_error_m,
            "outliers": self.outliers,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        ptype = d.get("projection_type", "perspective")
        if ptype != "perspective":
            raise ValueError(f"{ptype!r} calibration - load it with calibrators.load()")
        return cls(
            pitch=Pitch(**d["pitch"]),
            H=np.array(d["H"]),
            points={k: tuple(v) for k, v in d.get("points", {}).items()},
            rms_error_m=d.get("rms_error_m", 0.0),
            outliers=d.get("outliers", []),
        )

    @classmethod
    def load(cls, path: str) -> "Calibration":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _demo() -> None:
    """Round-trip a synthetic camera: pitch -> image -> pitch."""
    pitch = Pitch.from_yards(118, 72)
    assert abs(pitch.length - 107.9) < 0.1, pitch.length
    assert abs(pitch.width - 65.8) < 0.1, pitch.width

    # Synthetic perspective: elevated camera at the halfway line.
    true_H = np.array([[12.0, 3.0, 300.0], [0.0, 9.0, 120.0], [0.0, 0.006, 1.0]])
    lm = pitch.landmarks()
    world = np.array(list(lm.values()), dtype=np.float64)
    img = cv2.perspectiveTransform(world.reshape(-1, 1, 2), true_H).reshape(-1, 2)
    clicks = dict(zip(lm.keys(), map(tuple, img)))

    cal = Calibration.solve(pitch, clicks)
    assert cal.rms_error_m < 0.01, cal.rms_error_m

    back = cal.to_pitch(img)
    assert np.allclose(back, world, atol=1e-3), np.abs(back - world).max()

    # A single misclicked corner must not wreck the fit - this is the whole
    # reason for LMEDS plus more points than the minimum.
    bad = dict(clicks)
    bad["corner_bl"] = (img[0][0] + 90, img[0][1] + 90)
    bad_cal = Calibration.solve(pitch, bad)
    assert bad_cal.rms_error_m < 0.5, bad_cal.rms_error_m
    assert bad_cal.outliers == ["corner_bl"], bad_cal.outliers

    # Four points is the floor, not a target.
    four = {k: clicks[k] for k in ["corner_bl", "corner_tl", "corner_br", "corner_tr"]}
    assert Calibration.solve(pitch, four).rms_error_m < 0.01
    try:
        Calibration.solve(pitch, {k: clicks[k] for k in ["corner_bl", "corner_tl"]})
        raise AssertionError("should have rejected 2 points")
    except ValueError:
        pass

    # A perspective file must not load as anything else, and vice versa.
    d = cal.to_dict()
    assert d["projection_type"] == "perspective"
    assert np.allclose(Calibration.from_dict(d).H, cal.H)
    try:
        Calibration.from_dict({**d, "projection_type": "panoramic"})
        raise AssertionError("should have refused a panoramic file")
    except ValueError:
        pass

    print("pitch.py ok")

    # Projection paths live in calibrators.py (imported here, not at module
    # level, because calibrators imports us). One command still checks the
    # whole geometry layer.
    from calibrators import _demo as _projection_demo  # noqa: PLC0415

    _projection_demo()


if __name__ == "__main__":
    _demo()
