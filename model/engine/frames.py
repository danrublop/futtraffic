"""Ragged per-track detections -> one (T, N, 2) position matrix per team, in pitch metres.

Every track is resampled onto one shared grid of GRID_DT from `stats.speed_series`'s
filtered positions (never a second `cal.to_pitch`), with a validity mask so an
occlusion gap stays a hole instead of becoming a straight line.
measurement-evidence (not published)#evidence-shape-noise
"""
import numpy as np

from pitch import Calibration, Pitch
from stats import MIN_MINUTES, speed_series

GRID_DT = 1.0
# A grid sample further than this from any real detection is not observed. Sized
# at the grid step: interpolating across a gap longer than the sampling interval
# is inventing the one stretch we know we did not see, which is the same rule
# path_length applies to distance.
MAX_GAP_S = 1.0


def resample(t: np.ndarray, xy: np.ndarray, grid: np.ndarray,
             max_gap_s: float = MAX_GAP_S) -> tuple[np.ndarray, np.ndarray]:
    """One track onto `grid`. Returns ((T,2) metres, (T,) validity mask).

    np.interp CLAMPS outside the track's own span, which on its own would park a
    substitute on the touchline for the entire first half at the position they
    first appeared. The mask is what stops that, not the interpolation - it is
    False wherever the nearest real detection is more than `max_gap_s` away, and
    that covers both ends of the track and every occlusion inside it.
    """
    if len(t) < 2:
        return np.zeros((len(grid), 2)), np.zeros(len(grid), bool)
    out = np.stack([np.interp(grid, t, xy[:, 0]), np.interp(grid, t, xy[:, 1])], axis=1)
    i = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    near = np.minimum(np.abs(t[i] - grid), np.abs(t[i - 1] - grid))
    return out, near <= max_gap_s


def team_frames(tracks: dict, cal: Calibration, fps: float, dt: float = GRID_DT,
                method: str = "kalman") -> tuple[np.ndarray, dict[str, tuple]]:
    """{id: detections} -> (grid_t, {team: (XY, mask, ids)}).

    XY is (T, N, 2) pitch metres, mask is (T, N) bool, ids lines up with N.

    Officials and sub-`MIN_MINUTES` fragments are dropped here for exactly the
    reason `build_match` drops them from the players table: a referee inside the
    position matrix is an eleventh outfielder who runs through both banks, and
    he would widen the hull and drag the centroid every time he did it.
    """
    per: dict[str, list] = {}
    lo, hi = np.inf, -np.inf
    for tid, tr in tracks.items():
        pts = sorted(tr, key=lambda d: d["f"])
        team = pts[0].get("team", "unknown") if pts else "unknown"
        if len(pts) < 2 or team not in ("home", "away"):
            continue
        t = np.array([p["f"] / fps for p in pts], dtype=np.float64)
        if (t[-1] - t[0]) / 60.0 < MIN_MINUTES:
            continue
        xy = speed_series(tr, cal, fps, method)[3]
        per.setdefault(team, []).append((tid, t, xy))
        lo, hi = min(lo, t[0]), max(hi, t[-1])

    if not per:
        return np.zeros(0), {}
    grid = np.arange(lo, hi + dt / 2, dt)
    out = {}
    for team, items in per.items():
        XY = np.empty((len(grid), len(items), 2))
        M = np.empty((len(grid), len(items)), bool)
        for j, (_, t, xy) in enumerate(items):
            XY[:, j], M[:, j] = resample(t, xy, grid, max_gap_s=max(dt, MAX_GAP_S))
        out[team] = (XY, M, [tid for tid, _, _ in items])
    return grid, out


def _demo() -> None:
    """A synthetic straight-line team must come back at the right coordinates."""
    cal = Calibration(pitch=Pitch(105.0, 68.0), H=np.eye(3))
    fps = 25.0
    n = int(60 * fps)
    # Eleven players walking straight down the pitch at 1 m/s, 6 m apart across.
    tracks = {}
    for k in range(11):
        tracks[k] = [{"f": i, "id": k, "team": "home", "jersey": k,
                      "x": 10.0 + i / fps, "y": 4.0 + 6.0 * k} for i in range(n)]
    grid, tm = team_frames(tracks, cal, fps)
    XY, M, ids = tm["home"]
    assert XY.shape == (len(grid), 11, 2), XY.shape
    assert M.all(), "a gapless synthetic team must be fully observed"
    assert ids == list(range(11))
    # Mid-grid sample: x = 10 + t, y fixed. 0.3 m of slack for the Kalman's
    # opening transient, which is the only thing that can move a straight line.
    k = len(grid) // 2
    want_x = 10.0 + (grid[k] - grid[0])
    assert abs(XY[k, 0, 0] - want_x) < 0.3, (XY[k, 0, 0], want_x)
    assert abs(XY[k, 5, 1] - 34.0) < 0.05, XY[k, 5, 1]

    # A hole in one track must come back as a hole, not as a straight line
    # across it - the interpolator will happily draw that line, the mask must not.
    gapped = dict(tracks)
    gapped[3] = [p for p in tracks[3] if not (20 * fps <= p["f"] < 40 * fps)]
    _, tm2 = team_frames(gapped, cal, fps)
    M2 = tm2["home"][1]
    assert not M2[:, 3].all(), "occlusion gap was silently interpolated over"
    assert M2[:, 3].sum() > 30, "the observed part of the track was thrown away too"
    assert M2[:, [0, 1, 2, 4]].all(), "one track's gap leaked into its team-mates"

    # A referee must never enter the matrix; nor must a 3-second fragment.
    noisy = dict(tracks)
    noisy["ref"] = [{"f": i, "id": "ref", "team": "official", "x": 50.0, "y": 34.0}
                    for i in range(n)]
    noisy["frag"] = [{"f": i, "id": "frag", "team": "home", "x": 50.0, "y": 34.0}
                     for i in range(int(3 * fps))]
    assert team_frames(noisy, cal, fps)[1]["home"][0].shape[1] == 11

    print("frames.py ok")


if __name__ == "__main__":
    _demo()
