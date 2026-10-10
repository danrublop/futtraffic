"""Tier 1 stats: player tracks -> physical and spatial numbers, from positions only (no ball).

Input feet positions are image pixels. Every output position is pitch metres in the
calibration's frame: origin at a corner, x along the length, y across.
docs/reference/measurement.md
"""
from dataclasses import dataclass
import functools
import gzip
import json
import math
import re
import sys

import numpy as np

from paths import VALIDATION
from pitch import Calibration, Pitch

# Clamp on the REPORTED speed only: distance never reads it, and it catches no ID switches.
# measurement-evidence (not published)#evidence-speed-clamp
MAX_SPEED = 9.5
SPRINT_SPEED = 5.5
SPRINT_MIN_S = 0.8
# Sub-10s fragments are tracker debris, not players. A module constant, not an
# inline literal, because validate.py must count the SAME players or every
# agreement statistic is computed over two different populations. The last
# time these two files each kept their own copy of a rule, switching the
# default estimator left HSR computed off the old one for the same track.
MIN_MINUTES = 10 / 60
SMOOTH_S = 0.4  # moving-average window, boxcar path only

# Distance is the chord sum over windows of this many seconds, averaged over every start
# offset. Quote the window with the number.
# measurement-evidence (not published)#evidence-distance-window
DISTANCE_WINDOW_S = 0.4

# Detector box-bottom noise in PIXELS; the Jacobian turns it into an anisotropic metre-space R.
# The shipped value is not the measured detector noise.
# measurement-evidence (not published)#evidence-filter-tuning
SIGMA_PX = 1.0
# Process noise, m/s^2: a filter tuning constant, not the athlete's acceleration. Fitted, not held out.
# measurement-evidence (not published)#evidence-filter-tuning
ACCEL_Q = 1.0


@dataclass
class PlayerStats:
    track_id: int
    team: str
    jersey: int | None
    minutes: float
    distance_m: float
    top_speed_ms: float
    sprints: int
    avg_x: float
    avg_y: float
    samples: int
    # Share of OBSERVED STEPS at MAX_SPEED; not dist_eval's per-fragment field of the same name.
    # measurement-evidence (not published)#evidence-speed-clamp
    clamped_frac: float
    # Share of distance windows measured, 0-1: a floor on the missing share, not a correction.
    observed_frac: float
    # Observed time; `minutes` is span.
    # measurement-evidence (not published)#evidence-observed-vs-span
    observed_minutes: float = 0.0
    # From `roles.classify`, on motion alone. NEVER writes `team`: build_match
    # DELETES rows where team == "official", so a second writer there would turn
    # a misclassification into a vanished player. See roles.py.
    role: str = "uncertain"
    role_confidence: float = 0.0
    # motion vs kit colour: True agree, False disagree, None neither committed.
    role_agrees: bool | None = None


def homography_jacobian(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """d(pitch metres)/d(image pixels) at each point. (N,2) -> (N,2,2). Feeds the filter's R."""
    u, v = uv[:, 0], uv[:, 1]
    w = H[2, 0] * u + H[2, 1] * v + H[2, 2]
    X = (H[0, 0] * u + H[0, 1] * v + H[0, 2]) / w
    Y = (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / w
    J = np.empty((len(u), 2, 2))
    J[:, 0, 0] = (H[0, 0] - X * H[2, 0]) / w
    J[:, 0, 1] = (H[0, 1] - X * H[2, 1]) / w
    J[:, 1, 0] = (H[1, 0] - Y * H[2, 0]) / w
    J[:, 1, 1] = (H[1, 1] - Y * H[2, 1]) / w
    return J


def kalman_ground(xy_m: np.ndarray, uv_px: np.ndarray, t: np.ndarray, H: np.ndarray,
                  sigma_px: float | None = None, q: float | None = None,
                  jac: np.ndarray | None = None):
    """Constant-velocity Kalman on the GROUND PLANE. Returns (positions, speeds).

    Measurement noise R comes from the homography Jacobian, so it is anisotropic and grows
    with range. Speeds are clamped at MAX_SPEED; the state is not. Distance is not computed
    here: `_windowed` reads the filtered positions.
    measurement-evidence (not published)#evidence-filter-tuning
    """
    # Resolved at call time, not bound as a default: these are the two knobs that
    # get tuned per camera, and a default argument would freeze them at import.
    sigma_px = SIGMA_PX if sigma_px is None else sigma_px
    q = ACCEL_Q if q is None else q
    # ponytail: one track at a time, scalar Python loop. Measured 15.8 us/step,
    # so a full match (2.97M steps across 22 players) is ~47 s - 0.4% of a job
    # whose detection stage is 92-98% of the wall clock. Optimising this further
    # would speed up the DEV LOOP, not production. Upgrade path if parameter
    # sweeps get painful: stack (players x sweep-configs) on a leading axis and
    # run one pass over the timestep grid. Tracks are ragged, so that needs
    # padding to a common grid plus a validity mask gating the update step.
    n = len(t)
    # `jac` is the calibration's own d(metres)/d(px) when the map is not just H - a
    # corrected map stretches the pitch, and R from H alone would be wrong by that.
    Jac = homography_jacobian(H, uv_px) if jac is None else jac
    var_px = sigma_px ** 2
    # Seed velocity from the opening samples, not from rest: a track usually begins mid-stride.
    k = min(4, n - 1)
    span = float(t[k] - t[0])
    v0 = (xy_m[k] - xy_m[0]) / span if span > 0 else np.zeros(2)
    x = np.array([xy_m[0, 0], xy_m[0, 1], v0[0], v0[1]])
    P = np.diag([1.0, 1.0, 25.0, 25.0])
    Hm = np.zeros((2, 4))
    Hm[0, 0] = Hm[1, 1] = 1.0
    out_xy = np.empty((n, 2))
    out_v = np.empty(n)
    out_xy[0], out_v[0] = xy_m[0], 0.0

    for i in range(1, n):
        dt = float(t[i] - t[i - 1])
        if dt <= 0:
            out_xy[i], out_v[i] = out_xy[i - 1], out_v[i - 1]
            continue
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q2, d2, d3, d4 = q * q, dt * dt, dt ** 3, dt ** 4
        Q = q2 * np.array([[d4 / 4, 0, d3 / 2, 0], [0, d4 / 4, 0, d3 / 2],
                           [d3 / 2, 0, d2, 0], [0, d3 / 2, 0, d2]])
        x = F @ x
        P = F @ P @ F.T + Q

        J = Jac[i]
        R = var_px * (J @ J.T)          # J @ (var*I) @ J.T, without building I
        S = P[:2, :2] + R               # Hm selects the top-left block
        # Closed-form 2x2 inverse. np.linalg.inv dispatches to LAPACK, and at
        # this size the dispatch costs more than the arithmetic - this loop runs
        # ~3M times per match. Same reason Hm is indexed rather than multiplied.
        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        Sinv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det
        K = P[:, :2] @ Sinv
        x = x + K @ (xy_m[i] - x[:2])
        P = P - K @ P[:2, :]

        out_xy[i] = x[:2]
        out_v[i] = min(float(np.hypot(x[2], x[3])), MAX_SPEED)
    return out_xy, out_v


# Adopted, not tuned: floodlight's `transforms/filter.py` `butterworth_lowpass` at
# its own defaults (MIT, (c) 2021 floodlight-sports). The method is taken and scipy
# does the filtering; no floodlight code is copied, and floodlight pins python < 3.14.
# Measured against `kalman_ground` by N25 (model/validation/dist_eval.py); NOT shipped.
BUTTER_ORDER = 3
BUTTER_HZ = 1.0


def _runs(t: np.ndarray, real: np.ndarray):
    """(start, stop, cadence) of each unbroken run long enough to filter.

    floodlight's rule: a gap ends the signal, and a run of at most 3 * (order + 1)
    samples is too short to filter (its minimum, set by `filtfilt`'s default
    padding). floodlight keeps such a run unfiltered by default; here it gets no
    speed (`remove_short_seqs=True`), because unfiltered positions hand raw
    differences to a speed threshold.
    """
    cuts = np.flatnonzero(~real) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(t)]):
        if b - a < 2:
            continue
        step = float(np.median(np.diff(t[a:b])))
        if round((t[b - 1] - t[a]) / step) + 1 > 3 * (BUTTER_ORDER + 1):
            yield int(a), int(b), step


def butter_support(t: np.ndarray, real: np.ndarray) -> np.ndarray:
    """The steps the adopted filter gives a speed to: `real`, minus short runs."""
    keep = np.zeros(len(real), bool)
    for a, b, _ in _runs(t, real):
        keep[a:b - 1] = True
    return real & keep


def butter_ground(xy: np.ndarray, t: np.ndarray, real: np.ndarray) -> np.ndarray:
    """Zero-phase Butterworth low-pass on pitch metres, one unbroken run at a time.

    floodlight's XY assumes one frame rate, so each run is interpolated onto its
    own median cadence, filtered there, and read back at the observed frames.
    Positions in a run too short to filter are returned unfiltered, and
    `butter_support` gives their steps no speed.

    ONE deviation from floodlight, disclosed in N25: the odd padding spans the
    whole run, not scipy's default 3 * (order + 1) samples. The default starts the
    filter in steady state for a CONSTANT, so a player already running reads fast
    at every run's edges: at 25 Hz a clean 5.2 m/s run reads past the 5.5 m/s
    sprint threshold. Asserted in dist_eval._selfcheck, both paddings.
    """
    from scipy.signal import butter, filtfilt  # noqa: PLC0415
    out = xy.copy()
    for a, b, step in _runs(t, real):
        grid = t[a] + step * np.arange(round((t[b - 1] - t[a]) / step) + 1)
        g = np.stack([np.interp(grid, t[a:b], xy[a:b, k]) for k in (0, 1)], 1)
        num, den = butter(BUTTER_ORDER, BUTTER_HZ, btype="lowpass", fs=1.0 / step)
        f = filtfilt(num, den, g, axis=0, padlen=len(grid) - 1)
        out[a:b] = np.stack([np.interp(t[a:b], grid, f[:, k]) for k in (0, 1)], 1)
    return out


def _moving_average(a: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(a) < w:
        return a
    pad = w // 2
    padded = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    kern = np.ones(w) / w
    return np.stack([np.convolve(padded[:, i], kern, "valid")[: len(a)] for i in range(a.shape[1])], axis=1)


def _speeds(xy: np.ndarray, t: np.ndarray, clamp: bool = True) -> np.ndarray:
    """Per-step speed in m/s. len == len(xy) - 1.

    `clamp=False` is for the `raw` REFERENCE only: clamping a reference makes it a
    function of the code under test. MAX_SPEED bounds the reported speed; it does not
    detect ID switches.
    measurement-evidence (not published)#evidence-speed-clamp
    """
    dt = np.diff(t)
    dist = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(dt > 0, dist / np.maximum(dt, 1e-9), 0.0)
    return np.minimum(v, MAX_SPEED) if clamp else v


def _count_sprints(v: np.ndarray, t: np.ndarray, real: np.ndarray | None = None) -> int:
    """Runs above SPRINT_SPEED sustained at least SPRINT_MIN_S.

    `real` is the gap mask. Without it a sprint could be manufactured by an
    occlusion twice over: the step ACROSS a gap is displacement/dt over a long dt
    with no observed path, and the run length was measured as t[i] - t[start]
    straight through the gap, so a player who vanished behind a scrum and
    reappeared 3 s later 20 m away scored a sustained sprint. Distance and HSR
    have always excluded these samples (`player_stats`); this makes the count
    agree with them about which samples exist.
    """
    fast = v >= SPRINT_SPEED
    if real is not None:
        fast = fast & real          # an unobserved step is not a fast step
    n, run_start = 0, None
    for i, is_fast in enumerate(fast):
        if is_fast and run_start is None:
            run_start = i
        elif not is_fast and run_start is not None:
            if t[i] - t[run_start] >= SPRINT_MIN_S:
                n += 1
            run_start = None
    if run_start is not None and t[-1] - t[run_start] >= SPRINT_MIN_S:
        n += 1
    return n


def path_length(xy: np.ndarray, t: np.ndarray, real: np.ndarray,
                window_s: float | None = None) -> float:
    """Total distance in metres, measured at a stated bandwidth. See `_windowed`."""
    return _windowed(xy, t, real, window_s)[0]


def _windowed(xy: np.ndarray, t: np.ndarray, real: np.ndarray,
              window_s: float | None = None) -> tuple[float, float]:
    """(metres, observed_frac): the windowed path length and the share of it measured.

    Sum of |x(t+T) - x(t)| over non-overlapping windows of T = `window_s`, averaged over
    all k start offsets. `real` is the per-step gap mask (len == len(xy) - 1); a window
    containing any unobserved step is dropped whole, never bridged. `observed_frac` is the
    share of windows, over all offsets, that survived: a floor on the missing distance,
    not a correction factor.
    measurement-evidence (not published)#evidence-distance-window
    """
    # Resolved at call time: tuners assign the module global, which a default would ignore.
    if window_s is None:
        window_s = DISTANCE_WINDOW_S
    n = len(xy)
    if n < 2:
        return 0.0, 0.0
    step = float(np.median(np.diff(t)))
    if not np.isfinite(step) or step <= 0:
        return 0.0, 0.0
    k = max(1, int(round(window_s / step)))
    # O(1) validity test per window: how many unobserved steps it contains.
    bad = np.concatenate([[0], np.cumsum(~real)])

    total, kept, seen = 0.0, 0, 0
    for p in range(k):
        idx = np.arange(p, n, k)
        if p:                                # keep the head on the same terms
            idx = np.concatenate([[0], idx])  # as the tail: a short window, not
        if idx[-1] != n - 1:                  # a truncation
            idx = np.append(idx, n - 1)
        ok = (bad[idx[1:]] - bad[idx[:-1]]) == 0
        seg = np.linalg.norm(np.diff(xy[idx], axis=0), axis=1)
        total += float(seg[ok].sum())
        kept += int(ok.sum())
        seen += len(ok)
    return (total / k, kept / seen) if seen else (0.0, 0.0)


def speed_series(track: list[dict], cal: Calibration, fps: float,
                 method: str = "kalman"):
    """(speeds, dt, real_mask, positions) for one track. The single source of
    velocity, and of the trajectory that velocity was derived from.

    Exposed because every downstream metric - distance, HSR, sprints, peak speed -
    must come off the SAME velocity estimate. When validate.py kept its own copy
    of this, switching the default from boxcar to Kalman silently left HSR on the
    old estimator while distance moved to the new one. One track, two answers.

    `positions` is the 4th return for the same reason: distance is now measured
    off the trajectory (`path_length`) rather than by integrating `speeds`, and
    handing back the RAW projected points while the speeds came from the filtered
    ones would recreate exactly the two-answers bug this docstring is about.
    """
    pts = sorted(track, key=lambda d: d["f"])
    t = np.array([p["f"] / fps for p in pts], dtype=np.float64)
    uv = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float64)
    xy = cal.to_pitch(uv)
    step = float(np.median(np.diff(t))) or (1.0 / fps)
    dt = np.diff(t)
    real = dt <= max(3 * step, 2.0 / fps)   # the gap rule - see the comment at the return

    if method == "kalman":
        jac = cal.jacobian(uv) if hasattr(cal, "jacobian") else None
        xy, v_all = kalman_ground(xy, uv, t, cal.H, jac=jac)
        v = v_all[1:]  # align with the diff-based arrays
    elif method == "raw":
        # No filter at all and NO CLAMP: sum-of-deltas path length, which is what
        # this branch always claimed to be and was not. Only useful as a
        # REFERENCE, and a bandwidth-dependent one - see NOISE_TOLERANCE.md.
        # Never a product path; it integrates every jitter as running.
        v = _speeds(xy, t, clamp=False)
    elif method == "butter":
        # N25's adopted estimator, not what ships: see BUTTER_ORDER. It never
        # filters across a gap, and a run too short to filter scores nothing.
        xy = butter_ground(xy, t, real)
        v = _speeds(xy, t)
        real = butter_support(t, real)
    else:
        xy = _moving_average(xy, max(1, int(round(SMOOTH_S / step))))
        v = _speeds(xy, t)

    # Gaps (occlusion, missed frames) must not be integrated as running - the
    # player did move, but we didn't see it and can't attribute the path.
    # Threshold off this track's own cadence, not fps: pipelines that sample
    # every Nth frame are normal, and assuming per-frame detections here
    # silently reports zero distance for all of them.
    #
    # KNOWN DEFECT: `real` masks the step across a gap, not the filter's wake after it, so
    # velocity consumers (peak, sprints) over-read after occlusions; distance does not.
    # measurement-evidence (not published)#evidence-known-limits
    return v, dt, real, xy


def player_stats(track: list[dict], cal: Calibration, fps: float,
                 method: str = "kalman") -> PlayerStats | None:
    """One track -> one stat line. Returns None for tracks too short to trust.

    `method` picks the position estimator: "kalman" ships; "boxcar", "butter" and "raw" are
    for comparison. Distance is `_windowed` on the resulting positions; the speed series
    drives peak speed and sprints.
    measurement-evidence (not published)#evidence-filter-tuning
    """
    pts = sorted(track, key=lambda d: d["f"])
    if len(pts) < 2:
        return None

    t = np.array([p["f"] / fps for p in pts], dtype=np.float64)

    v, dt, real, xy = speed_series(track, cal, fps, method)
    # Distance comes off the TRAJECTORY at a stated bandwidth, never from integrating v.
    distance, observed = _windowed(xy, t, real)

    return PlayerStats(
        track_id=pts[0]["id"],
        team=pts[0].get("team", "unknown"),
        jersey=pts[0].get("jersey"),
        # SPAN, gaps included; observed time is `observed_minutes`.
        # measurement-evidence (not published)#evidence-observed-vs-span
        minutes=float(t[-1] - t[0]) / 60.0,
        distance_m=distance,
        # Both now read only OBSERVED steps, like distance and HSR above. A step
        # across an occlusion gap is a fabricated velocity, and as a max over
        # ~135k samples peak speed is exactly the statistic that finds it.
        top_speed_ms=float(v[real].max()) if real.any() else 0.0,
        sprints=_count_sprints(v, t[1:], real),
        avg_x=float(xy[:, 0].mean()),
        avg_y=float(xy[:, 1].mean()),
        samples=len(pts),
        # Saturation marks filter recovery through occlusions, not ID switches.
        # measurement-evidence (not published)#evidence-speed-clamp
        clamped_frac=float(np.mean(v[real] >= MAX_SPEED - 1e-9)) if real.any() else 0.0,
        observed_frac=observed,
        # Observed wall clock, threaded from the speed_series call above (the expensive Kalman).
        observed_minutes=float(np.sum(dt[real])) / 60.0 if real.any() else 0.0,
        **_role_fields(xy, real, fps, cal, track, dt),
    )


def _role_fields(xy, real, fps: float, cal, track: list[dict], dt=None) -> dict:
    """roles.classify on the trajectory we already have. Never fatal.

    Wrapped because a role is a nice-to-have on every row and a hard dependency
    on none of them: identity.py takes the same stance for the same reason. A
    classifier that raises must not cost a coach the whole match report.
    """
    try:
        import roles  # noqa: PLC0415
        r = roles.classify(xy, real, fps, float(cal.pitch.width), dt)
        team = track[0].get("team", "unknown") if track else "unknown"
        return {"role": r.role, "role_confidence": r.confidence,
                "role_agrees": roles.cross_check(r, team)}
    except Exception as e:  # noqa: BLE001
        print(f"  roles: {type(e).__name__}: {e}; leaving role unset", flush=True)
        return {}


# ===========================================================================
# TEAM SHAPE. Shape is a difference, not an integral: noise does not accumulate as it
# does for distance.
# measurement-evidence (not published)#evidence-shape-noise
# ===========================================================================

# Below this many observed outfielders a sample is skipped; series_coverage counts the skips.
# measurement-evidence (not published)#evidence-shape-noise
SHAPE_MIN_PLAYERS = 7
# Two columns closer than SHAPE_DUP_M for SHAPE_DUP_FRAC of their shared samples are one
# player. A refusal threshold, not a discrimination one.
# measurement-evidence (not published)#evidence-coverage-and-duplicates
SHAPE_DUP_M = 1.0
SHAPE_DUP_FRAC = 0.5
# Below this share of possible samples the series is flagged `low_coverage`, not hidden. Not fitted.
# measurement-evidence (not published)#evidence-shape-noise
SHAPE_COVERAGE_MIN = 0.9
# Refusal threshold: with no keeper-shaped gap, nominate nobody rather than a centre-back.
# measurement-evidence (not published)#evidence-keeper-and-direction
GK_MARGIN_M = 5.0
# A keeper's median position lies within this of his own goal line (second half of the refusal).
# measurement-evidence (not published)#evidence-keeper-and-direction
GK_MAX_DEPTH_M = 35.0
FORMATION_WINDOW_S = 30.0
# A new template must beat the incumbent by this fraction to take over the label.
# measurement-evidence (not published)#evidence-formation
FORMATION_HYSTERESIS = 0.10
# Candidates closer than this merge into one interval: one dropped sample must not split an event.
# measurement-evidence (not published)#evidence-structure-stress
STRESS_MERGE_GAP_S = 2.0
# Shape is computed on the frames.GRID_DT grid and emitted every SERIES_EMIT_S; stress
# intervals use the full grid series.
# measurement-evidence (not published)#evidence-shape-noise
SERIES_EMIT_S = 5.0


def _hull_area(pts: np.ndarray) -> float:
    """Convex hull area in m^2. Monotone chain + shoelace, no scipy. 0 when collinear.
    measurement-evidence (not published)#evidence-shape-noise
    """
    p = np.unique(np.round(np.asarray(pts, float), 6), axis=0)  # also sorts lexicographically
    if len(p) < 3:
        return 0.0

    def half(q):
        h = []
        for r in q:
            while len(h) >= 2:
                a, b = h[-2], h[-1]
                if (b[0] - a[0]) * (r[1] - a[1]) - (b[1] - a[1]) * (r[0] - a[0]) > 0:
                    break
                h.pop()
            h.append(r)
        return h

    h = np.array(half(p)[:-1] + half(p[::-1])[:-1])
    if len(h) < 3:
        return 0.0                      # collinear: a line has no area
    return float(abs(h[:, 0] @ np.roll(h[:, 1], -1) - h[:, 1] @ np.roll(h[:, 0], -1)) / 2)


def team_lines(xs: np.ndarray, k: int = 3) -> tuple[list[float], list[float]]:
    """The team's k lines along the pitch, and the gaps between them. Metres.

    Complete-linkage clustering on distance from own goal. In 1-D the clusters are
    contiguous, so repeatedly merging the adjacent pair with the smallest diameter is
    exact. Trajectory-only: no ball, so no line-break count.
    measurement-evidence (not published)#evidence-formation
    """
    v = np.sort(np.asarray(xs, float))
    if len(v) <= k:
        return [float(x) for x in v], [float(d) for d in np.diff(v)]
    groups = [[float(x)] for x in v]
    while len(groups) > k:
        # Diameter of each candidate merge = last element of the right group
        # minus first of the left, since both groups are already sorted runs.
        d = [groups[i + 1][-1] - groups[i][0] for i in range(len(groups) - 1)]
        i = int(np.argmin(d))
        groups[i:i + 2] = [groups[i] + groups[i + 1]]
    means = [float(np.mean(g)) for g in groups]
    return means, [float(d) for d in np.diff(means)]


def _goalkeepers(X: np.ndarray, mask: np.ndarray, margin_m: float = GK_MARGIN_M,
                 max_depth_m: float = GK_MAX_DEPTH_M) -> set[int]:
    """Columns holding the keeper. X is (T, N) metres from OWN goal line.

    Positional: the deep cluster of median depth, clear of everyone else by `margin_m`
    and within `max_depth_m` of the goal; empty (refused) otherwise. A set, because one
    keeper can arrive as several fragments.
    measurement-evidence (not published)#evidence-keeper-and-direction
    """
    return _deep_cluster(_median_depth(X, mask), margin_m, max_depth_m)[0]


def _median_depth(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-column median of X over observed samples. inf where never observed."""
    return np.array([np.median(X[mask[:, j], j]) if mask[:, j].any() else np.inf
                     for j in range(X.shape[1])])


def _deep_cluster(med: np.ndarray, margin_m: float = GK_MARGIN_M,
                  max_depth_m: float = GK_MAX_DEPTH_M) -> tuple[set[int], float]:
    """(columns in the deep cluster, its separation from everyone else).

    Separation is -inf when the rule refuses. It is returned as well as the set
    because `infer_attacking_right` picks a direction by maximising it.
    """
    seen = [j for j in range(len(med)) if np.isfinite(med[j])]
    if len(seen) < 2:
        return set(), -np.inf
    lo = min(med[j] for j in seen)
    gk = {j for j in seen if med[j] - lo < margin_m}
    rest = [med[j] for j in seen if j not in gk]
    # Three refusals, each of which fires on real or synthetic data:
    #  * more than 3 columns in the deep cluster is a team standing on its own
    #    goal line, not a keeper;
    #  * a cluster deeper than `max_depth_m` from the goal is not a keeper - a
    #    4-4-2 read BACKWARDS puts its two forwards 37 m from the far goal with a
    #    23 m gap behind them, which out-scores the real keeper's 17 m separation
    #    on separation alone. A keeper lives in his own third;
    #  * a cluster that does not clear the rest by `margin_m` is a back line.
    if not rest or len(gk) > 3 or lo > max_depth_m:
        return set(), -np.inf
    sep = float(min(rest) - max(med[j] for j in gk))
    return (gk, sep) if sep >= margin_m else (set(), sep)


def repair_banks(teams: dict, pitch_length: float = 105.0) -> dict:
    """Fix two failures of kit-colour team assignment, positionally.

    1. Bank names: the bank whose centroid has the lower x is "home". A convention,
       stable across re-runs; not a fact about the fixture.
    2. Keepers (third kit): the two most extreme x columns overall go to the bank whose
       other columns' centroid is nearer.
    Returns `teams` unchanged when there is no bank to repair (see the column floor below).
    measurement-evidence (not published)#evidence-repair-banks
    """
    cols = []
    for side, (XY, M, ids) in teams.items():
        for j in range(XY.shape[1]):
            if M[:, j].any():
                cols.append((side, j, float(np.median(XY[M[:, j], j, 0])), ids[j]))
    # Below 8 columns there is no bank to repair and the extremes are not
    # keepers, they are just the two players furthest apart.
    if len(cols) < 8 or len(teams) < 2:
        return teams

    xs = np.array([c[2] for c in cols])
    keepers = {int(xs.argmin()), int(xs.argmax())}
    rest = [i for i in range(len(cols)) if i not in keepers]
    cent = {s: float(np.mean([cols[i][2] for i in rest if cols[i][0] == s]))
            for s in teams if any(cols[i][0] == s for i in rest)}
    if len(cent) < 2:
        return teams

    where = {i: cols[i][0] for i in rest}
    for i in keepers:
        where[i] = min(cent, key=lambda s: abs(cols[i][2] - cent[s]))

    groups = {s: [i for i in where if where[i] == s] for s in set(where.values())}
    if len(groups) < 2:
        return teams
    mid = {s: float(np.mean([cols[i][2] for i in g])) for s, g in groups.items()}
    lo = min(mid, key=mid.get)
    rename = {lo: "home", next(s for s in mid if s != lo): "away"}

    out = {}
    for s, g in groups.items():
        XY = np.concatenate([teams[cols[i][0]][0][:, cols[i][1]:cols[i][1] + 1, :] for i in g], axis=1)
        M = np.concatenate([teams[cols[i][0]][1][:, cols[i][1]:cols[i][1] + 1] for i in g], axis=1)
        out[rename[s]] = (XY, M, [cols[i][3] for i in g])
    return out


def _direction_evidence(X: np.ndarray, mask: np.ndarray,
                        pitch_length: float) -> tuple[bool, float]:
    """(attacks_right, strength in metres) for ONE side, from its keeper.

    Strength is how much better the winning orientation looks than the other:
    the difference of the two `_deep_cluster` separations, or - when the losing
    orientation found no deep cluster at all - the winner's own separation. It
    is 0.0 when neither orientation found anything, which is the refusal.

    It exists because the direction must be decided ONCE for a match and the
    two sides have to be comparable to do that. See `attacking_directions`.
    """
    med = _median_depth(X, mask)
    if not np.isfinite(med).any():
        return True, 0.0
    third = pitch_length / 3
    sr = _deep_cluster(med, max_depth_m=third)[1]
    sl = _deep_cluster(pitch_length - med, max_depth_m=third)[1]
    if sr == -np.inf and sl == -np.inf:
        return True, 0.0
    right = bool(sr >= sl)
    win, lose = (sr, sl) if right else (sl, sr)
    return right, float(win if lose == -np.inf else win - lose)


def infer_attacking_right(XY: np.ndarray, mask: np.ndarray, pitch_length: float = 105.0) -> bool:
    """Which way ONE team attacks, from where its keeper stands: `_deep_cluster` is run
    both ways round and the larger separation wins.

    Single-side primitive. DO NOT call it once per side and ship both answers: two
    independent calls can give both teams the same direction. Use `attacking_directions`.

    ponytail: one direction for the whole input, so a file spanning half-time is wrong for
    one half. Upgrade path: split on the largest frame gap and infer per half.
    measurement-evidence (not published)#evidence-keeper-and-direction
    """
    return _direction_evidence(XY[:, :, 0], mask, pitch_length)[0]


def attacking_directions(teams: dict, pitch_length: float = 105.0
                         ) -> tuple[dict[str, bool], tuple[str, ...]]:
    """{side: (XY, mask, ids)} -> ({side: attacks_right}, caveat flags).

    One direction per match, from the side with the stronger keeper evidence; the other
    side gets the opposite by construction. A disagreeing weaker side raises
    `attacking_direction_conflict`. No keeper either way on either side REFUSES: it falls
    back to home-attacks-right and raises `attacking_direction_refused`.
    measurement-evidence (not published)#evidence-keeper-and-direction
    """
    ev = {s: _direction_evidence(XY[:, :, 0], M, pitch_length)
          for s, (XY, M, _ids) in teams.items() if XY is not None}
    if not ev:
        return {}, ()
    best = max(ev, key=lambda s: ev[s][1])
    right, strength = ev[best]
    if strength <= 0.0:
        # No keeper-shaped gap anywhere, either way round. Refuse rather than
        # guess: the convention below is the coin flip this function replaced.
        return {s: s == "home" for s in ev}, ("attacking_direction_refused",)
    flags = ("attacking_direction_conflict",) if any(
        s != best and e[1] > 0.0 and e[0] == right for s, e in ev.items()) else ()
    return {s: (right if s == best else not right) for s in ev}, flags


def duplicate_groups(XY: np.ndarray, mask: np.ndarray, dup_m: float = SHAPE_DUP_M,
                     frac: float = SHAPE_DUP_FRAC, min_overlap: int = 5) -> list[int]:
    """Column index -> group id. Two fragments of one player share a group.

    Same player if within `dup_m` for at least `frac` of the samples both are observed
    (at least `min_overlap` of them); merged transitively. Shape does not act on it: the
    metrics and the SHAPE_MIN_PLAYERS gate use raw `n`, and only `n_dedup` is reported.
    build_match's side distance counts one track per group.
    ponytail: O(N^2) pairs vectorised over T; KD tree on per-column medians if N grows.
    measurement-evidence (not published)#evidence-coverage-and-duplicates
    """
    N = mask.shape[1]
    g = list(range(N))

    def root(a: int) -> int:
        while g[a] != a:
            g[a] = g[g[a]]
            a = g[a]
        return a

    for i in range(N):
        for j in range(i + 1, N):
            both = mask[:, i] & mask[:, j]
            if int(both.sum()) < min_overlap:
                continue        # too little shared time to judge either way
            d = np.hypot(*(XY[both, i] - XY[both, j]).T)
            if float(np.mean(d < dup_m)) >= frac:
                g[root(i)] = root(j)
    return [root(j) for j in range(N)]


def series_coverage(rows: list[dict], possible: int) -> dict:
    """How much of the match `shape_series` measured: {possible, emitted, frac,
    min_players, n_hist, n_dedup_hist}, counted on the full grid series, not the strided
    `samples`. No threshold here: build_match flags `low_coverage` off SHAPE_COVERAGE_MIN.
    measurement-evidence (not published)#evidence-coverage-and-duplicates
    """
    def hist(k: str) -> dict:
        v, c = np.unique([r[k] for r in rows], return_counts=True)
        return {str(int(a)): int(b) for a, b in zip(v, c)}

    return {"possible": int(possible), "emitted": len(rows),
            "frac": round(len(rows) / possible, 3) if possible else 0.0,
            "min_players": SHAPE_MIN_PLAYERS,
            "n_hist": hist("n") if rows else {},
            "n_dedup_hist": hist("n_dedup") if rows else {}}


def relative_shape(rows_a: list[dict], rows_b: list[dict], possible: int,
                   a: str = "home", b: str = "away") -> dict:
    """Between-team shape difference (b - a), on the INTERSECTION of emitted samples.

    Never a difference of the two per-team medians, which are over different instants.
    `n` is the paired count and governs; `frac` is against `possible`; `sd` ships beside
    each mean difference.
    measurement-evidence (not published)#evidence-coverage-and-duplicates
    """
    A = {round(r["t"], 1): r for r in rows_a}
    B = {round(r["t"], 1): r for r in rows_b}
    ts = sorted(set(A) & set(B))
    out = {"n": len(ts), "possible": int(possible),
           "frac": round(len(ts) / possible, 3) if possible else 0.0,
           "a": a, "b": b, "metrics": {}}
    for k in ("width_m", "depth_m", "rear_line_m", "hull_area_m2", "stretch_index_m"):
        if not ts or k not in A[ts[0]]:
            continue
        va = np.array([A[t][k] for t in ts], float)
        vb = np.array([B[t][k] for t in ts], float)
        d = vb - va
        out["metrics"][k] = {
            f"{a}_mean": round(float(va.mean()), 2),
            f"{b}_mean": round(float(vb.mean()), 2),
            # b minus a, so a positive number reads "b is wider/deeper than a".
            "diff": round(float(d.mean()), 2),
            "sd": round(float(d.std(ddof=1)), 2) if len(d) > 1 else None,
        }
    return out


def shape_series(XY: np.ndarray, mask: np.ndarray, attacking_right: bool,
                 pitch_length: float = 105.0, t: np.ndarray | None = None) -> list[dict]:
    """Per-sample team shape. (T,N,2) metres + (T,N) validity -> one dict per row.

    x is rewritten as metres from the team's OWN goal line. The keeper (`_goalkeepers`) is
    excluded from every metric. Per row: `t`; `width_m`, `depth_m` (10th-90th percentile
    spread, not min-max); `hull_area_m2`; `rear_line_m` (mean of the 4 deepest outfielders
    in every phase, not a defensive line); `centroid` [x, y]; `stretch_index_m` (mean
    radial distance to the centroid); `lines_m`, `line_gaps_m` (`team_lines`); `n`
    (outfield COLUMNS, not people); `n_dedup` (reported only, see `duplicate_groups`).
    Samples below SHAPE_MIN_PLAYERS are skipped silently: pair with `series_coverage`.
    measurement-evidence (not published)#evidence-shape-noise
    """
    T, N = mask.shape
    if N < SHAPE_MIN_PLAYERS + 1:
        return []
    X = XY[:, :, 0] if attacking_right else pitch_length - XY[:, :, 0]
    gk = _goalkeepers(X, mask, max_depth_m=pitch_length / 3)
    outfield = [j for j in range(N) if j not in gk]
    dup = duplicate_groups(XY, mask)
    # The grid is 1 Hz, so a bare row index already IS the second.
    tt = np.arange(T, dtype=float) if t is None else np.asarray(t, float)

    rows = []
    for i in range(T):
        ok = [j for j in outfield if mask[i, j]]
        if len(ok) < SHAPE_MIN_PLAYERS:
            continue
        x, y = X[i, ok], XY[i, ok, 1]
        lo_x, hi_x = np.percentile(x, [10, 90])
        lo_y, hi_y = np.percentile(y, [10, 90])
        cx, cy = float(x.mean()), float(y.mean())
        lines, gaps = team_lines(x)
        rows.append({
            "t": float(tt[i]),
            "width_m": float(hi_y - lo_y),
            "depth_m": float(hi_x - lo_x),
            "hull_area_m2": _hull_area(np.stack([x, y], axis=1)),
            "rear_line_m": float(np.sort(x)[:4].mean()),
            "centroid": [cx, cy],
            "stretch_index_m": float(np.hypot(x - cx, y - cy).mean()),
            "lines_m": lines,
            "line_gaps_m": gaps,
            "n": len(ok),
            "n_dedup": len({dup[j] for j in ok}),
        })
    return rows


# Ten outfielders per template in normalised coordinates: x 0 at the team's own goal line
# and 1 at the opponent's, y 0 to 1 across. Hand-drawn priors, not fitted.
# measurement-evidence (not published)#evidence-formation
FORMATIONS: dict[str, list[tuple[float, float]]] = {
    "4-4-2":   [(.15, .15), (.15, .38), (.15, .62), (.15, .85),
                (.45, .15), (.45, .38), (.45, .62), (.45, .85),
                (.78, .38), (.78, .62)],
    "4-3-3":   [(.15, .15), (.15, .38), (.15, .62), (.15, .85),
                (.45, .25), (.45, .50), (.45, .75),
                (.80, .15), (.80, .50), (.80, .85)],
    "4-2-3-1": [(.15, .15), (.15, .38), (.15, .62), (.15, .85),
                (.38, .35), (.38, .65),
                (.62, .15), (.62, .50), (.62, .85), (.85, .50)],
    "4-5-1":   [(.15, .15), (.15, .38), (.15, .62), (.15, .85),
                (.45, .05), (.45, .30), (.45, .50), (.45, .70), (.45, .95),
                (.82, .50)],
    "3-5-2":   [(.15, .28), (.15, .50), (.15, .72),
                (.45, .05), (.45, .30), (.45, .50), (.45, .70), (.45, .95),
                (.80, .38), (.80, .62)],
    "5-3-2":   [(.13, .05), (.13, .28), (.13, .50), (.13, .72), (.13, .95),
                (.45, .25), (.45, .50), (.45, .75),
                (.80, .38), (.80, .62)],
}


def _unit_box(p: np.ndarray) -> np.ndarray:
    """Rescale to its own bounding box. Both sides of the match are scaled, so
    the cost is SCALE-FREE: a team 25 m wide and one 45 m wide with the same
    relative arrangement score identically. That is deliberate. Absolute
    compactness is `shape_series`' job, and keeping it out of the formation cost
    is what stops `structure_stress` from multiplying one measurement by itself.
    """
    lo, hi = p.min(axis=0), p.max(axis=0)
    return (p - lo) / np.maximum(hi - lo, 1e-6)


def _template_cost(pos: np.ndarray, tpl: np.ndarray) -> float:
    """Mean optimal-assignment distance between a team and one template."""
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415 - optional dep
    a, b = _unit_box(pos), _unit_box(tpl)
    C = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    # Rectangular is fine and load-bearing: with a substitute on the pitch there
    # are 11 outfield COLUMNS for 10 template slots, and the solver drops the
    # worst-fitting one rather than forcing a bogus pairing.
    r, c = linear_sum_assignment(C)
    return float(C[r, c].mean())


def formation_windows(XY: np.ndarray, mask: np.ndarray, attacking_right: bool = True,
                      pitch_length: float = 105.0, t: np.ndarray | None = None,
                      window_s: float = FORMATION_WINDOW_S) -> list[dict]:
    """Formation label per `window_s` window. [{t0, t1, formation, cost}].

    EFPI (Bekkers, arXiv:2506.23843): optimal assignment of the team's mean positions to
    each template in FORMATIONS, both scaled to their own extent; the lowest mean
    assignment distance is the label and its cost. An estimate, not a measurement.
    measurement-evidence (not published)#evidence-formation
    """
    T, N = mask.shape
    if N < SHAPE_MIN_PLAYERS + 1:
        return []
    X = XY[:, :, 0] if attacking_right else pitch_length - XY[:, :, 0]
    gk = _goalkeepers(X, mask, max_depth_m=pitch_length / 3)
    outfield = [j for j in range(N) if j not in gk]
    tt = np.arange(T, dtype=float) if t is None else np.asarray(t, float)
    tpls = {k: np.array(v) for k, v in FORMATIONS.items()}

    out, prev = [], None
    for w0 in np.arange(tt[0], tt[-1], window_s):
        sel = (tt >= w0) & (tt < w0 + window_s)
        if sel.sum() < 3:
            continue
        pos = []
        for j in outfield:
            m = sel & mask[:, j]
            # Seen for under half the window: a substitute warming up, or a
            # track that died. Averaging him in puts a player where nobody was.
            if m.sum() * 2 < sel.sum():
                continue
            pos.append([X[m, j].mean(), XY[m, j, 1].mean()])
        if len(pos) < SHAPE_MIN_PLAYERS:
            continue
        pos = np.array(pos)
        cost = {k: _template_cost(pos, v) for k, v in tpls.items()}
        best = min(cost, key=cost.get)
        if prev is not None and cost[prev] <= cost[best] * (1 + FORMATION_HYSTERESIS):
            best = prev          # incumbent keeps the label, and keeps ITS cost
        prev = best
        out.append({"t0": float(w0), "t1": float(w0 + window_s),
                    "formation": best, "cost": cost[best]})
    return out


def stress_profile(intervals: list[dict]) -> dict:
    """{gini, profile} for a set of stress intervals: `peaked`, `moderate` or `flat` by the
    Gini coefficient of their severities. {} below two intervals. The cut points are a
    first fit on few team-halves; only peaked-vs-not is supported.
    measurement-evidence (not published)#evidence-structure-stress
    """
    if not intervals:
        return {}
    sev = np.sort(np.array([i["severity"] for i in intervals], dtype=float))
    n, total = len(sev), float(sev.sum())
    if n < 2 or total <= 0:
        return {}
    g = float((2 * np.arange(1, n + 1) - n - 1) @ sev / (n * total))
    return {"gini": round(g, 3),
            "profile": "peaked" if g >= 0.40 else "moderate" if g >= 0.20 else "flat"}


def formation_summary(windows: list[dict]) -> dict:
    """{formation, tied?, mode_share, windows, changes}: the only formation output fit to
    show. `formation` is the modal per-window label and `mode_share` its uncertainty; the
    per-window labels are unstable and ship under `window_fit_unstable`.
    measurement-evidence (not published)#evidence-formation
    """
    if not windows:
        return {}
    labs = [w["formation"] for w in windows]
    # A tie is SHOWN, not resolved: every label at the top count, in first-seen order (dict.fromkeys,
    # never set, so the order does not depend on PYTHONHASHSEED), joined " / " in `formation`.
    top = max(map(labs.count, labs))
    tied = [x for x in dict.fromkeys(labs) if labs.count(x) == top]
    return {"formation": " / ".join(tied),
            **({"tied": tied} if len(tied) > 1 else {}),
            "mode_share": round(top / len(labs), 3),
            "windows": len(labs),
            "changes": sum(1 for a, b in zip(labs, labs[1:]) if a != b)}


def structure_stress(shape: list[dict], formation: list[dict], top: int = 10,
                     merge_gap_s: float = STRESS_MERGE_GAP_S) -> list[dict]:
    """The team's `top` most structurally stressed intervals, ranked by severity.

    Per sample: z_cost + z_hull against the team's own match distribution. Candidates sit
    above the 90th percentile of that sum AND have z_hull > 0 (stretched, not compressed);
    candidates within `merge_gap_s` merge; severity is the summed score times the sample
    step. Returns [] below the sample floor, on empty inputs, or with zero variance in
    both. Named "structure stress", never "defensive error": there is no ball.
    measurement-evidence (not published)#evidence-structure-stress
    """
    if not shape or not formation:
        return []
    t = np.array([r["t"] for r in shape])
    hull = np.array([r["hull_area_m2"] for r in shape])
    cost = np.full(len(shape), np.nan)
    label = [None] * len(shape)
    for w in formation:
        sel = (t >= w["t0"]) & (t < w["t1"])
        cost[sel] = w["cost"]
        for i in np.flatnonzero(sel):
            label[i] = w["formation"]
    ok = np.isfinite(cost)
    if ok.sum() < 10:                   # a distribution over <10 samples is not one
        return []

    def _z(v):
        # sd 0 means the quantity never varied - a synthetic team, or one
        # formation window. Contribute nothing rather than dividing by zero.
        sd = float(v[ok].std())
        z = np.zeros(len(v))
        if sd > 0:
            z[ok] = (v[ok] - v[ok].mean()) / sd
        return z

    zh = _z(hull)
    sev = _z(cost) + zh
    # The p90 cut only segments candidates; zh > 0 keeps a compressed siege out.
    cut = float(np.percentile(sev[ok], 90))
    cand = ok & (sev > cut) & (zh > 0)

    step = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    idx = np.flatnonzero(cand)
    out, k = [], 0
    while k < len(idx):
        m = k
        # Merge across `merge_gap_s`: one dropped sample must not split one
        # counter-attack into two events. Real holes in the tracking are longer
        # than that and still break the interval, which is the point.
        while m + 1 < len(idx) and (t[idx[m + 1]] - t[idx[m]]) <= merge_gap_s:
            m += 1
        i, j = int(idx[k]), int(idx[m])
        sl = slice(i, j + 1)
        out.append({
            "t0": float(t[i]), "t1": float(t[j]),
            # Span PLUS one step: a single 1 Hz sample is one second of football,
            # not zero. The old span-only form made singletons 0 s long.
            "duration_s": float(t[j] - t[i] + step),
            "severity": float(sev[i:j + 1].sum() * step),
            "peak_z": float(sev[i:j + 1].max()),
            "formation": label[i],
            "cost": float(cost[i]), "hull_area_m2": float(hull[sl].max()),
        })
        k = m + 1
    return sorted(out, key=lambda d: -d["severity"])[:top]


def team_shape(players: list[PlayerStats], pitch: Pitch, attacking_right: bool,
               series: list[dict] | None = None) -> dict:
    """Median width, depth, rear line, hull area and stretch index, in metres.

    Medians of `series` (`shape_series` output, which build_match passes). With no series
    it falls back to one sample of whole-match average positions, a category error kept
    only for the signature. `defensive_line_m` keeps its key for
    app/frontend/src/lib/types.ts; it is the every-phase rear line, not a defensive line.
    measurement-evidence (not published)#evidence-shape-noise
    """
    if series is None:
        if len(players) < 3:
            return {}
        xy = np.array([[p.avg_x, p.avg_y] for p in players], dtype=np.float64)
        series = shape_series(xy[None, :, :], np.ones((1, len(players)), bool),
                              attacking_right, pitch.length)
    if not series:
        return {}
    med = lambda k: round(float(np.median([r[k] for r in series])), 1)  # noqa: E731
    return {
        "width_m": med("width_m"),
        "depth_m": med("depth_m"),
        "defensive_line_m": med("rear_line_m"),
        "compactness_m2": med("hull_area_m2"),
        "stretch_index_m": med("stretch_index_m"),
    }


def territory(tracks: dict[int, list[dict]], cal: Calibration) -> dict:
    """Share of player-position samples in each third of the pitch, by pitch x.

    The keys are LEFT / CENTRE / RIGHT of the calibration's x axis, whatever
    their names say: `defensive_third` is x < L/3, `middle_third` the centre,
    `attacking_third` x > 2L/3. They are the same thirds for both sides, not
    each team's own defensive and attacking thirds. The names stay for the
    match.json contract.

    `tracks` is the population counted: build_match passes the players table
    (officials, unassigned sides and sub-MIN_MINUTES fragments excluded).
    Non-finite positions are not counted.

    This is territory, NOT possession. It says where play happened, which is
    honest and computable from positions alone. Do not relabel it "possession".
    """
    L = cal.pitch.length
    counts = np.zeros(3)
    for tr in tracks.values():
        xy = cal.to_pitch(np.array([[p["x"], p["y"]] for p in tr], dtype=np.float64))
        xy = xy[np.isfinite(xy).all(axis=1)]
        idx = np.clip((xy[:, 0] / L * 3).astype(int), 0, 2)
        counts += np.bincount(idx, minlength=3)
    total = counts.sum() or 1.0
    return {"defensive_third": round(float(counts[0] / total * 100), 1),
            "middle_third": round(float(counts[1] / total * 100), 1),
            "attacking_third": round(float(counts[2] / total * 100), 1)}


#: Metres outside the pitch a heatmap point may sit - sceneprior.pitch_polygon's default margin.
HEATMAP_MARGIN_M = 3.0


def heatmap_points(track: list[dict], cal: Calibration, max_points: int = 600) -> list[list[float]]:
    """Positions in pitch metres, thinned. Fed straight to the SVG KDE.
    Points off the pitch by more than HEATMAP_MARGIN_M, or non-finite, are dropped before thinning."""
    xy = cal.to_pitch(np.array([[p["x"], p["y"]] for p in track], dtype=np.float64))
    m, L, W = HEATMAP_MARGIN_M, cal.pitch.length, cal.pitch.width
    with np.errstate(invalid="ignore"):
        xy = xy[(xy[:, 0] >= -m) & (xy[:, 0] <= L + m) & (xy[:, 1] >= -m) & (xy[:, 1] <= W + m)]
    if len(xy) > max_points:
        xy = xy[np.linspace(0, len(xy) - 1, max_points).astype(int)]
    return [[round(float(x), 2), round(float(y), 2)] for x, y in xy]


def _round_all(obj, nd: int = 2):
    """Round every float in a nested list/dict, for emit. The shape series is
    ~2,900 samples per team per half and json.dump(indent=2) would otherwise
    write 17 significant figures of each one.
    """
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, list):
        return [_round_all(v, nd) for v in obj]
    if isinstance(obj, dict):
        return {k: _round_all(v, nd) for k, v in obj.items()}
    return obj


# ===========================================================================
# THE METRIC ENVELOPE. Every published number ships with {value, units, n, confidence,
# provenance, justified_by, caveat_flags}. confidence is the min() of the class ceiling
# and each flag's ceiling, never a product or a mean: the weakest link governs.
# docs/reference/measurement.md#tiers
# measurement-evidence (not published)#evidence-caveat-reasons
# ===========================================================================

#: What `confidence` reads when nothing measured justifies a tier. Ranked BELOW
#: `prov` so the same min() that makes the weakest caveat govern also makes an
#: unjustified field govern - one rule, not a second branch.
NOT_EVALUATED = "not_evaluated"

TIER_RANK = {NOT_EVALUATED: -1, "prov": 0, "rel": 1, "ok": 2}

# flag -> (confidence ceiling, one-line reason). Only the ceiling is read by code; a ceiling
# of "ok" is a note, not a downgrade.
# measurement-evidence (not published)#evidence-caveat-reasons
CAVEATS: dict[str, tuple[str, str]] = {
    "unvalidated_end_to_end": ("prov",
        "Never compared with an independent instrument on real video; distance and peak speed carry their own measured flags instead."),
    "paired_only": ("prov",
        "Computed only on instants where BOTH teams cleared SHAPE_MIN_PLAYERS; `n` is the paired count."),
    "team_purity": ("prov",
        "A between-team difference fails chaotically under one misassigned player, and repair_banks is unverified beyond one clip."),
    "distance_bias_measured": ("prov",
        "Measured against ground truth: long on the frames observed, short over the whole half; observed_frac says which regime."),
    "measures_clamp": ("prov",
        "REFUSED where MAX_SPEED bound: there the value is the clamp constant, not the player; clamped_frac says where."),
    "extreme_value_statistic": ("prov",
        "One sample sets it, so it is exposed to the single worst tracking error of the match."),
    "peak_speed_overreads": ("prov",
        "Measured, it reads high, from filter velocity recovering through occlusion gaps; a surviving value is a max over a mixed track."),
    "count_metric_unstable": ("prov",
        "A count over a threshold; professional systems disagree widely on sprint counts."),
    "trend_only": ("rel",
        "Between-system disagreement is the size of match-to-match variation: read a trend over several matches."),
    "rank_only": ("rel",
        "A rank or a share: the calibration and filter terms every player shares drop out of the ratio."),
    "role_correlated_bias": ("rel",
        "The distance window under-reads in proportion to direction changes, so the bias tracks playing position."),
    "kit_colour_clustering": ("prov",
        "The side is unsupervised shirt-colour clustering, not a tag; a wrong side moves a whole distance total to the other squad."),
    "team_split_implausible": ("prov",
        "Sides more lopsided than TEAM_BALANCE_MAX or larger than TEAM_SIZE_MAX: the clustering is sorting fragments."),
    "unassigned_side": ("prov",
        "At least one track has no side; it reaches the player table but no team aggregate."),
    "identity_contaminated": ("prov",
        "A player row is not one player: stitching trades purity for wholeness, and the error pools within a team."),
    "attacking_direction_conflict": ("prov",
        "Both sides' keepers pointed the same way; the weaker side was overruled, so one direction is an override."),
    "attacking_direction_refused": ("prov",
        "No keeper found either way on either side; the direction fell back to home-attacks-right, a coin flip."),
    "low_coverage": ("prov",
        "Fewer than SHAPE_COVERAGE_MIN of the possible samples were computed; see shape_series[side].coverage."),
    "estimate_not_measurement": ("prov",
        "A template-fitting estimate; the per-window label is unstable."),
    "single_clip_reference": ("rel",
        "Compared against ground truth on real pipeline output, but on one clip."),
    "not_evaluated": ("prov",
        "The harness could not perturb this, so its pass is a tautology; unevaluated is provisional."),
    "track_span_not_playing_time": ("ok",
        "Track span: occlusion gaps count as played, and there is no substitution detection."),
    "role_unvalidated": ("prov",
        "Inferred from motion alone and never scored; staff and linesman only, never referee, and it excludes nobody."),
    "jersey_ocr_unvalidated": ("prov",
        "Unless an operator typed it, an unscored OCR read: a proposal for a human to confirm."),
    "minutes_includes_gaps": ("prov",
        "Distance excludes occluded windows and `minutes` does not, so per-90 is short on top and long below."),
    # @C 4.4795 :: n21-product-now (not published) (N11 block, pooled_sprint_ratio at .p21id)
    "sprints_withheld": ("prov",
        "WITHHELD: at the shipped config the sprint count read several times ground truth (N21); it returns when a registered measurement passes."),
    "distance_unbounded": ("prov",
        "The clamp bound on a covered row; distance is read off positions, so the clamp does not bound it. clamped_frac says which rows."),
    "distance_over_ceiling": ("prov",
        "REFUSED on at least one row: more ground than MAX_SPEED covers in the observed time; refused rows leave the side total."),
    "team_distance_partial": ("prov",
        "Counts players-table rows only (one track per duplicate group, none over the ceiling); distance_coverage is the share counted."),
    "population_not_filtered": ("rel",
        "Computed over every track, including referees and sub-MIN_MINUTES fragments."),
}

# ===========================================================================
# WHAT JUSTIFIES A TIER. Each METRICS entry names the api:// id behind its class ceiling,
# and metric() grants that ceiling only when the id is accepted in
# model/validation/api-golden.json: resolved against the golden, never a live run, and
# no value is copied. A registered id with no accepted value reads not_evaluated.
# measurement-evidence (not published)#evidence-caveat-reasons
# ===========================================================================


def accepted_measurements() -> frozenset[str]:
    """The `api://` ids that have an accepted value behind them.

    Empty on a missing, half-written or unreadable golden, which degrades every
    field rather than defaulting any of them. That is the safe direction and
    the same rule `api.py` applies to a failed producer: a failure is a status,
    never a number.

    Keyed on the golden's mtime, not cached for the process: resolve() runs on
    every request now, and a server that read the golden once kept citing an id
    a later snapshot had withdrawn until someone restarted it.
    """
    try:
        st = (VALIDATION / "api-golden.json").stat()
    except OSError:
        return frozenset()
    # mtime, size AND inode: _write_json renames a new file in, so two writes in
    # one mtime tick still differ by inode.
    return _accepted_at((st.st_mtime_ns, st.st_size, st.st_ino))


@functools.lru_cache(maxsize=2)
def _accepted_at(_stamp: tuple) -> frozenset[str]:
    try:
        with open(VALIDATION / "api-golden.json") as f:
            vals = json.load(f)["values"]
        return frozenset(k for k, v in vals.items() if v.get("status") == "OK")
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        return frozenset()


# field path in match.json ->
#     (units, provenance, class ceiling, base flags, justifying api:// id).
#
# `provenance`: measured = read off the data or off our own output directly;
# derived = computed through a stated model (projection, filter, window, hull);
# tagged = asserted by a human or by an upstream label we did not verify.
#
# The id is `None` where NOTHING measures this field. That is not a gap in the
# table, it is the table's most useful column: it is the list of numbers this
# product ships without evidence, and it is now visible in the artefact rather
# than in an audit.
# measurement-evidence (not published)#evidence-caveat-reasons
METRICS: dict[str, tuple[str, str, str, tuple[str, ...], str | None]] = {
    "players.minutes": ("min", "measured", "ok",
                        ("not_evaluated", "track_span_not_playing_time",
                         "identity_contaminated"), None),
    # Per player: justified by the per-player spread, not the squad-pooled ratio.
    "players.distance_km": ("km", "derived", "ok",
                            ("distance_bias_measured", "identity_contaminated"),
                            "product.player.distance.spread"),
    "players.top_speed_kmh": ("km/h", "derived", "ok",
                              ("extreme_value_statistic", "peak_speed_overreads",
                               "identity_contaminated"), "product.speed.peak_ms"),
    # Measured and far off at the shipped config (N21): withheld, unregistered, unjustified.
    "players.sprints": ("count", "derived", "prov",
                        ("sprints_withheld", "count_metric_unstable",
                         "unvalidated_end_to_end", "identity_contaminated"), None),
    "players.avg_position": ("m", "derived", "ok",
                             ("unvalidated_end_to_end", "identity_contaminated"),
                             None),
    # NOT product.speed.clamped_frac: that id is a share of FRAGMENTS, this a share of STEPS.
    "players.clamped_frac": ("frac", "measured", "ok", (), None),
    "players.observed_frac": ("frac", "measured", "ok", (), None),
    # Arithmetic on our own gap mask, a self-report; product.distance.coverage is another denominator.
    "players.observed_minutes": ("min", "measured", "ok", (), None),
    # Unscored: roles.py thresholds are geometry; model.role.* scores rolenet, not these.
    "players.role": ("label", "derived", "prov", ("role_unvalidated",), None),
    "players.role_confidence": ("frac", "derived", "prov", ("role_unvalidated",),
                                None),
    # docTR numbers ship on every job and nothing registered scores them (docs/experiments/n49-jersey-ocr.md).
    "players.jersey": ("number", "tagged", "prov", ("jersey_ocr_unvalidated",), None),
    # team.accuracy.track scores track.assign_teams at an oracle mapping; the shipped side is repair_banks'.
    "players.team": ("side", "derived", "prov", ("kit_colour_clustering",), None),
    # Derived in app/frontend/src/lib/report.ts off the players table; the per-player spread justifies both.
    "derived.share": ("%", "derived", "ok",
                      ("rank_only", "role_correlated_bias", "identity_contaminated"),
                      "product.player.distance.spread"),
    "derived.per90": ("km/90", "derived", "ok",
                      ("distance_bias_measured", "minutes_includes_gaps",
                       "identity_contaminated"), "product.player.distance.spread"),
    # The squad-pooled ratio measures a squad total: the right population here - on the
    # stride-2 .p21id tracks, not the August stride-1 tracks.split.json ratio_raw reads.
    "teams.distance_m": ("m", "derived", "ok", ("distance_bias_measured",),
                         "product.distance.ratio_raw.p21id"),
    "teams.top_speed_kmh": ("km/h", "derived", "ok",
                            ("extreme_value_statistic", "peak_speed_overreads",
                             "identity_contaminated"), "product.speed.peak_ms"),
    "teams.sprints": ("count", "derived", "prov",
                      ("sprints_withheld", "count_metric_unstable",
                       "unvalidated_end_to_end"), None),
    "teams.shape.width_m": ("m", "derived", "ok", ("unvalidated_end_to_end",), None),
    "teams.shape.depth_m": ("m", "derived", "ok", ("unvalidated_end_to_end",), None),
    "teams.shape.compactness_m2": ("m^2", "derived", "ok",
                                   ("unvalidated_end_to_end",), None),
    "teams.shape.stretch_index_m": ("m", "derived", "ok",
                                    ("unvalidated_end_to_end",), None),
    # Measured on one clip by model/engine/validate.py, unregistered; the key is the every-phase rear line.
    "teams.shape.defensive_line_m": ("m", "measured", "ok",
                                     ("single_clip_reference",), None),
    "territory": ("%", "derived", "ok", ("rank_only",), None),
    "shape_series.samples": ("m", "derived", "ok", ("unvalidated_end_to_end",),
                             None),
    # Paired on shared instants (relative_shape); team_purity because one misassigned player breaks it.
    "relative_shape": ("m", "derived", "prov",
                       ("unvalidated_end_to_end", "paired_only", "team_purity"),
                       None),
    "shape_series.formation": ("label", "derived", "prov",
                               ("estimate_not_measurement",), None),
    "shape_series.window_fit_unstable": ("label", "derived", "prov",
                                         ("estimate_not_measurement",), None),
    "structure_stress": ("rank", "derived", "ok",
                         ("rank_only", "unvalidated_end_to_end"), None),
    # This job's fit residual; nothing measures a per-job residual against truth (model-map (not published)).
    "meta.calibration_rms_m": ("m", "measured", "ok", (), None),
}


#: Above this many tracks per side the sides are fragments, not players; the ratio test cannot see it.
#: measurement-evidence (not published)#evidence-caveat-reasons
TEAM_SIZE_MAX = 30

#: How lopsided the two sides may be before the split is called implausible.
TEAM_BALANCE_MAX = 1.5


def _peak_kmh(sq: list[PlayerStats]) -> float | None:
    """Peak speed in km/h over a set of tracks, or None where MAX_SPEED bound on any of them.

    A REFUSAL, NOT A DAMPED READING: where the clamp bound, `top_speed_ms` IS MAX_SPEED and
    the km/h figure is a constant of this file. The zero threshold stays until
    `speed_series` stops handing out post-gap velocity. A true peak just under the clamp can
    print the same km/h; `clamped_frac` on the row separates the two.
    measurement-evidence (not published)#evidence-speed-clamp
    """
    if not sq or any(p.clamped_frac > 0 for p in sq):
        return None
    return round(max(p.top_speed_ms for p in sq) * 3.6, 1)


def _over_ceiling(p: PlayerStats) -> bool:
    """More metres than MAX_SPEED allows in the time this track was observed: refused, not published."""
    return p.distance_m > MAX_SPEED * p.observed_minutes * 60.0


def metric(field: str, value=None, n: int | None = None,
           flags: tuple[str, ...] = ()) -> dict:
    """One envelope. `flags` are the INSTANCE caveats - the ones that depend on
    this match rather than on the metric - and they are added to the field's
    permanent ones, never replace them.

    `value` is None for a table COLUMN: 22 copies of the same envelope, one per
    player row, would be 22 copies of the same sentence. The envelope governs
    the column; the values are in `players`.

    THE TIER IS RESOLVED, NOT ASSERTED. The field's class ceiling only reaches
    the artefact if the `api://` id beside it in METRICS has an accepted
    measurement behind it; otherwise `confidence` is `not_evaluated` and
    `justified_by` is null. See the JUSTIFICATION banner above METRICS for why
    that resolves against the golden and not against a live producer.
    """
    units, provenance, tier, base, *rest = METRICS[field]
    # An entry registered WITHOUT an id - model/engine/events.py merges five such
    # fields in at import - is a field nothing measures, which is exactly what
    # `not_evaluated` says. The missing element is the honest default here, not
    # a legacy shim to be tolerated.
    mid = rest[0] if rest else None
    justified = mid is not None and mid in accepted_measurements()
    fl = list(dict.fromkeys(base + tuple(flags)))       # ordered, deduplicated
    return {"value": value, "units": units, "n": n,
            # min(), so the weakest link governs - see the banner above. An
            # unjustified field enters that min() as NOT_EVALUATED, which
            # ranks below every tier, so it governs by the same rule rather
            # than by a special case.
            "confidence": min([tier if justified else NOT_EVALUATED]
                              + [CAVEATS[f][0] for f in fl],
                              key=TIER_RANK.__getitem__),
            "provenance": provenance,
            # The id travels WITH the number, so a reader can re-run the
            # verdict rather than trust it. Null when nothing justified the
            # tier - never the id it named and failed to resolve, because an
            # id printed beside `not_evaluated` reads as justification.
            "justified_by": f"api://{mid}" if justified else None,
            "caveat_flags": fl}


def unjustify(fields: dict) -> dict:
    """Strip every justification: for tracks we GENERATED, not measured.

    demo.py (and a former feed importer) synthesise positions and run them through
    build_match like footage, so every field resolved its tier as if our
    pipeline had measured it - 10 of the 11 match.json files on disk are
    synthetic and serve ok/rel tiers. A measurement of our tracking says
    nothing about a field computed from positions a random generator drew."""
    for env in fields.values():
        env["justified_by"] = None
        env["confidence"] = NOT_EVALUATED
    return fields


def resolve(fields: dict, meta: dict | None = None) -> dict:
    """Every envelope re-resolved against TODAY's golden. -> a new fields dict.

    A match.json is written once, when the job runs, and the tiers in it are
    the golden's opinion ON THAT DAY. The 11 served on 2026-09-10 predated tier
    checking altogether, so every one still said `players.jersey: ok` while
    metric() had long since stopped saying it. Whoever serves or stores a
    match calls this, so a tier is never older than the evidence behind it.

    A per-side path (`teams.home.shape.width_m`) is its METRICS key with the
    side removed, which is how build_match writes it. A path METRICS no longer
    knows, or a caveat flag CAVEATS no longer knows, degrades that envelope to
    not_evaluated: something was renamed or removed, and the safe direction
    for an unknown is down.
    """
    out = {}
    for path, env in (fields or {}).items():
        if not isinstance(env, dict):      # not an envelope at all: nothing vouches
            out[path] = {"value": env, "confidence": NOT_EVALUATED, "justified_by": None}
            continue
        key = re.sub(r"\.(home|away)(?=\.|$)", "", path)
        flags = tuple(env.get("caveat_flags") or ())
        if key in METRICS and all(f in CAVEATS for f in flags):
            out[path] = metric(key, env.get("value"), env.get("n"), flags)
        else:
            out[path] = dict(env, confidence=NOT_EVALUATED, justified_by=None)
    return unjustify(out) if (meta or {}).get("synthetic_tracking") else out


def resolved_match(path) -> dict:
    """A match.json with its tiers re-resolved - what every surface serves.
    One function, so the API route and the agent cannot drift apart."""
    with open(path) as f:
        m = json.load(f)
    m["fields"] = resolve(m.get("fields") or {}, m.get("meta"))
    # Sprints are withheld on every served match, including files written before the refusal.
    for row in [*(m.get("players") or []), *(m.get("teams") or {}).values()]:
        if isinstance(row, dict) and "sprints" in row:
            row["sprints"] = None
    return m


def _open(path: str):
    """Tracks files arrive both ways. Every ground-truth half under model/validation/
    is gzipped (a SoccerTrack half is ~2.7 GB raw), while jobs.py writes plain
    JSON into the job directory. A bare open() here meant build_match could not
    read the only real fixtures in the repo, which is how two planning sessions
    in a row ended up designing around a fixture that could not be loaded.

    Defined here rather than in validate.py because validate imports stats, not
    the other way round, so this is the end of the dependency that can own it.
    """
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def build_match(tracks_path: str, calib_path: str, out_path: str, meta: dict | None = None) -> dict:
    """tracks.json + calibration.json -> match.json at `out_path`; returns the same dict.

    `tracks_path` (plain or .gz) holds `fps` and `detections` [{f, id, team, x, y, jersey?}]
    with feet in image pixels; `calib_path` is loaded by `calibrators.load`; `meta` is merged
    into match.json's `meta`, and `meta["synthetic_tracking"]` strips every justification.
    Positions out are pitch metres in the calibration's frame (origin at a corner, x along
    the length, y across); shape x is metres from each side's own goal line.
    Raises ValueError for a missing, non-numeric, non-finite or non-positive `fps`, a
    non-finite player value, or any non-finite float in the result (nothing is written);
    KeyError without `detections`; whatever `calibrators.load` raises.
    docs/reference/measurement.md#contract
    """
    with _open(tracks_path) as f:
        data = json.load(f)
    # Every time base is fps: a missing, zero, negative or non-finite one is refused, never defaulted.
    fps = data.get("fps")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"build_match refused {tracks_path}: fps {fps!r} is missing or not a positive number")
    fps = float(fps)
    # calibrators.load, not Calibration.load: `remap` (stitched panorama) is accepted,
    # `panoramic` is refused. measurement-evidence (not published)#evidence-known-limits
    import calibrators  # noqa: PLC0415 - avoids a circular import at module load
    import quality  # noqa: PLC0415 - quality imports this module back
    cal = calibrators.load(calib_path)

    by_id: dict[int, list[dict]] = {}
    for d in data["detections"]:
        by_id.setdefault(d["id"], []).append(d)

    players, heat = [], {}
    for tid, tr in by_id.items():
        ps = player_stats(tr, cal, fps)
        if ps is None or ps.minutes < MIN_MINUTES:
            continue
        players.append(ps)
        heat[str(tid)] = heatmap_points(tr, cal)

    # Officials are tracked (they occlude players and matter for association) but
    # never appear as players. A referee covers more ground than most outfielders,
    # so leaving one in the table produces a phantom top-of-the-list profile that
    # reads as plausible and discredits every real number beside it.
    officials = [p for p in players if p.team == "official"]
    players = [p for p in players if p.team != "official"]
    for p in officials:
        heat.pop(str(p.track_id), None)
    # A non-finite position poisons every sum it reaches; refuse the match naming the track, never publish NaN.
    for p in players:
        if not all(math.isfinite(v) for v in (p.distance_m, p.top_speed_ms, p.avg_x, p.avg_y)):
            raise ValueError(f"build_match refused: track {p.track_id} has a non-finite position or speed")

    # frames imports this module, so the import is local for the same reason
    # calibrators' is above - it would be a cycle at module load.
    #
    # ponytail: this runs the Kalman a SECOND time over every track, because
    # player_stats already ran it and threw the trajectory away. ~47 s per match
    # duplicated, against a detection stage that is 92-98% of the job - 0.4% of
    # wall clock becomes 0.8%. Upgrade path if the dev loop starts hurting: have
    # player_stats return its `xy` and pass the cache into team_frames.
    import frames  # noqa: PLC0415
    grid, tmat = frames.team_frames(by_id, cal, fps)
    # Keepers and bank names are repaired positionally before any shape is computed.
    tmat = repair_banks(tmat, cal.pitch.length)
    # Repoint player rows at the repaired banks so both halves of teams[side] describe one
    # squad. Raw id keys, never int(): tracks files carry str ids too.
    # measurement-evidence (not published)#evidence-repair-banks
    repaired = {i: side for side, (_x, _m, ids) in tmat.items() for i in ids}
    for p in players:
        if p.track_id in repaired:
            p.team = repaired[p.track_id]
    t_rel = grid - grid[0] if len(grid) else grid   # seconds from first detection

    # Every published number gets an envelope keyed by its path in this file.
    # Column-wide ones first; the per-side instances are added in the loop, so a
    # UI resolves `teams.home.shape.width_m` and falls back to nothing rather
    # than to a confident default.
    fields = {f: metric(f, n=0 if f == "players.sprints" else len(players)) for f in (
        "players.minutes", "players.distance_km", "players.top_speed_kmh",
        "players.sprints", "players.avg_position", "players.clamped_frac",
        "players.observed_frac", "players.observed_minutes",
        "players.role", "players.role_confidence",
        "players.jersey", "players.team",
        "derived.share", "derived.per90")}
    # A lopsided or oversized split is about the kit clustering, and needs no ground truth to check.
    sides = [sum(1 for p in players if p.team == s) for s in ("home", "away")]
    side_flags: tuple[str, ...] = ()
    if (not min(sides) or max(sides) / min(sides) > TEAM_BALANCE_MAX
            or max(sides) > TEAM_SIZE_MAX):
        side_flags += ("team_split_implausible",)
    # A track with no side reaches the players table and NO team aggregate, so
    # its share is computed against the leftovers rather than against a squad.
    if any(p.team not in ("home", "away") for p in players):
        side_flags += ("unassigned_side",)
    if side_flags:
        fields["players.team"] = metric("players.team", n=len(players),
                                        flags=side_flags)

    # The clamp is a caveat per track, wherever it BOUND.
    # measurement-evidence (not published)#evidence-speed-clamp
    if any(p.clamped_frac > 0 for p in players):
        # `n` counts the rows that still HAVE a peak speed.
        fields["players.top_speed_kmh"] = metric(
            "players.top_speed_kmh", n=sum(1 for p in players if not p.clamped_frac),
            flags=("measures_clamp",))

    # Distance is flagged where the clamp bound and refused per row above the physical ceiling.
    dist_flags = ((("distance_unbounded",) if any(p.clamped_frac > 0 for p in players) else ())
                  + (("distance_over_ceiling",) if any(_over_ceiling(p) for p in players) else ()))
    n_dist = sum(1 for p in players if not _over_ceiling(p))
    for f in ("players.distance_km", "derived.per90"):
        fields[f] = metric(f, n=n_dist, flags=dist_flags)

    # The players-table population only: the same rows the table shows, sides assigned.
    terr_tracks = {p.track_id: by_id[p.track_id] for p in players if p.team in ("home", "away")}
    fields["territory"] = metric("territory", n=sum(len(t) for t in terr_tracks.values()))
    fields["meta.calibration_rms_m"] = metric("meta.calibration_rms_m",
                                              cal.rms_error_m)

    # ONE direction for the match, decided before the per-side loop: see attacking_directions.
    att_all, att_flags = attacking_directions(tmat, cal.pitch.length)

    teams, series_out, stress_out = {}, {}, {}
    raw_ser: dict[str, list[dict]] = {}       # unstrided, for the paired comparison
    for side in ("home", "away"):
        sq = [p for p in players if p.team == side]
        if not sq:
            continue
        XY, M, ids = tmat.get(side, (None, None, None))
        att, ser, forms, stress = att_all.get(side, side == "home"), [], [], []
        if XY is not None:
            ser = shape_series(XY, M, att, cal.pitch.length, t_rel)
            forms = formation_windows(XY, M, att, cal.pitch.length, t_rel)
            stress = structure_stress(ser, forms)
        cov = series_coverage(ser, len(M) if M is not None else 0)
        raw_ser[side] = ser
        # The side's distance: one track per duplicate group (the longest observed), none above the ceiling.
        by_pid = {p.track_id: p for p in sq}
        dropped: set = set()
        if XY is not None:
            groups: dict[int, list] = {}
            for j, gid in enumerate(duplicate_groups(XY, M)):
                if ids[j] in by_pid:
                    groups.setdefault(gid, []).append(by_pid[ids[j]])
            for g in groups.values():
                keep = max(g, key=lambda p: p.observed_minutes)
                dropped |= {p.track_id for p in g if p is not keep}
        counted = [p for p in sq if not _over_ceiling(p) and p.track_id not in dropped]
        side_dets = sum(len(tr) for tid, tr in by_id.items()
                        if repaired.get(tid, tr[0].get("team")) == side)
        dist_cov = sum(len(by_id[p.track_id]) for p in counted) / side_dets if side_dets else 0.0
        teams[side] = {
            "distance_m": round(sum(p.distance_m for p in counted)),
            # share of this side's detections the total counts: CAVEATS["team_distance_partial"]
            "distance_coverage": round(dist_cov, 4),
            "top_speed_kmh": _peak_kmh(sq),   # None where the clamp bound
            "sprints": None,                  # withheld: CAVEATS["sprints_withheld"]
            "shape": team_shape(sq, cal.pitch, att, ser or None),
        }
        stride = max(1, round(SERIES_EMIT_S / frames.GRID_DT))
        series_out[side] = {
            "attacking_right": att, "sample_dt_s": SERIES_EMIT_S,
            # How much of the match `samples` is, counted on the 1 Hz series it
            # was strided down from. Without this a 76%-covered series and a
            # complete one are the same object on the wire.
            "coverage": cov,
            # The only formation output fit to show anyone - see formation_summary.
            "formation": formation_summary(forms),
            # ...and the per-window series, named so it cannot be mistaken for a
            # claim. It is here for the COST, which structure_stress consumes and
            # which is well behaved; the per-window label is not.
            "window_fit_unstable": _round_all(forms),
            "samples": _round_all(ser[::stride])}
        stress_out[side] = {**stress_profile(stress), "intervals": _round_all(stress)}

        # Instance caveats: the two that depend on THIS match rather than on the
        # metric. Every shape number is a median of the same skipped-sample
        # series, so low_coverage lands on all of them together.
        thin = ("low_coverage",) if cov["frac"] < SHAPE_COVERAGE_MIN else ()
        # The clamp is a caveat only where it bound on this side; a flag on every row gets skipped.
        # measurement-evidence (not published)#evidence-speed-clamp
        clamp = ("measures_clamp",) if any(p.clamped_frac > 0 for p in sq) else ()
        # Direction sets which goal line x is measured from, and it also picks
        # which columns `shape_series` drops as keepers - so it moves every
        # shape number, not only `defensive_line_m`.
        dirn = thin + att_flags
        fields[f"teams.{side}.distance_m"] = metric(
            "teams.distance_m", teams[side]["distance_m"], n=len(counted),
            flags=("team_distance_partial",)
            + (("distance_unbounded",) if any(p.clamped_frac > 0 for p in counted) else ()))
        fields[f"teams.{side}.top_speed_kmh"] = metric(
            "teams.top_speed_kmh", teams[side]["top_speed_kmh"],
            n=sum(1 for p in sq if not p.clamped_frac), flags=clamp)
        fields[f"teams.{side}.sprints"] = metric("teams.sprints", None, n=0)
        for k, v in teams[side]["shape"].items():
            fields[f"teams.{side}.shape.{k}"] = metric(
                f"teams.shape.{k}", v, n=cov["emitted"], flags=dirn)
        fields[f"shape_series.{side}.samples"] = metric(
            "shape_series.samples", n=cov["emitted"], flags=dirn)
        fields[f"shape_series.{side}.formation"] = metric(
            "shape_series.formation", series_out[side]["formation"].get("formation"),
            n=len(forms), flags=att_flags)
        fields[f"shape_series.{side}.window_fit_unstable"] = metric(
            "shape_series.window_fit_unstable", n=len(forms), flags=att_flags)
        fields[f"structure_stress.{side}"] = metric(
            "structure_stress", stress_out[side].get("profile"), n=len(stress),
            flags=dirn)

    rel = relative_shape(raw_ser.get("home", []), raw_ser.get("away", []),
                         len(t_rel) if len(t_rel) else 0)
    if rel["n"]:
        # rear_line_m is a difference of two x-from-own-goal readings, so a
        # direction the two sides disagreed about lands on the paired
        # comparison as hard as it lands on either side's own number.
        fields["relative_shape"] = metric(
            "relative_shape", n=rel["n"], flags=att_flags + (
                () if rel["frac"] >= SHAPE_COVERAGE_MIN else ("low_coverage",)))

    if (meta or {}).get("synthetic_tracking"):
        unjustify(fields)

    match = {
        "meta": {**(meta or {}), "fps": fps,
                 "pitch": {"length": cal.pitch.length, "width": cal.pitch.width},
                 "calibration_rms_m": cal.rms_error_m,
                 "officials_excluded": len(officials),
                 # ADVISORY, never a refusal: most criteria need a ball the stats do not read.
                 # measurement-evidence (not published)#evidence-known-limits
                 "quality": quality.check(data, cal, fps)},
        "teams": teams,
        "territory": territory(terr_tracks, cal),
        "players": [
            {"id": p.track_id, "team": p.team, "jersey": p.jersey,
             "minutes": round(p.minutes, 1),
             "distance_km": None if _over_ceiling(p) else round(p.distance_m / 1000, 2),
             "top_speed_kmh": _peak_kmh([p]), "sprints": None,   # withheld: CAVEATS["sprints_withheld"]
             "avg_position": [round(p.avg_x, 1), round(p.avg_y, 1)],
             "clamped_frac": round(p.clamped_frac, 4),
             # Share of this track's distance windows that were actually
             # measured. Below 1, `distance_km` is an under-count and
             # `distance_km / minutes` is under twice - see player_stats.
             "observed_frac": round(p.observed_frac, 4),
             # Observed time beside the span-based `minutes`; report.ts per90 still divides by `minutes`.
             # measurement-evidence (not published)#evidence-observed-vs-span
             "observed_minutes": round(p.observed_minutes, 1),
             # Motion-only, and it excludes nobody - see roles.py on why this is
             # not written into `team`.
             "role": p.role, "role_confidence": p.role_confidence,
             "role_agrees": p.role_agrees}
            for p in sorted(players, key=lambda p: -p.distance_m)
        ],
        "heatmaps": heat,
        # Additive only. Every existing key keeps its meaning and its type, so
        # app/frontend/src reads this file unchanged.
        "shape_series": series_out,
        "structure_stress": stress_out,
        # The headline, and the ONLY valid way to state it: paired on the
        # instants where both teams were measurable. See relative_shape.
        "relative_shape": rel,
        # The envelope per published number, keyed by its dotted path in this dict; flat, so the
        # existing keys keep the types app/frontend/src/lib/types.ts reads.
        # docs/reference/measurement.md#tiers
        "fields": fields,
    }
    bad = _nonfinite(match)
    if bad:
        raise ValueError(f"build_match refused: non-finite value at {bad} - match.json is never written with NaN/inf")
    with open(out_path, "w") as f:
        json.dump(match, f, indent=2, allow_nan=False)
    return match


def _nonfinite(obj, path: str = "match") -> str | None:
    """Path of the first non-finite float in a JSON-able tree, or None."""
    if isinstance(obj, float):
        return None if math.isfinite(obj) else path
    items = obj.items() if isinstance(obj, dict) else enumerate(obj) if isinstance(obj, (list, tuple)) else ()
    for k, v in items:
        if (bad := _nonfinite(v, f"{path}.{k}")):
            return bad
    return None


# A 4-4-2 in real metres: keeper, back four, midfield four, two forwards.
_SQUAD_442 = [(5, 34),
              (22, 12), (22, 27), (22, 41), (22, 56),
              (45, 10), (45, 26), (45, 42), (45, 58),
              (68, 27), (68, 41)]


def _demo_shape(pitch: Pitch) -> None:
    """Team shape, lines, formation and structure stress on synthetic teams."""
    # Hull: a 10x10 square with a point in the middle is still 100 m^2, and a
    # bounding box cannot tell you that. Collinear points have no area.
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]], float)
    assert abs(_hull_area(sq) - 100.0) < 1e-6, _hull_area(sq)
    assert _hull_area(np.array([[0., 0.], [1., 1.], [2., 2.]])) == 0.0

    # Lines: three obvious banks, and the gaps between them.
    means, gaps = team_lines(np.array([20, 21, 22, 23, 45, 46, 47, 48, 70, 71.]))
    assert means == [21.5, 46.5, 70.5], means
    assert gaps == [25.0, 24.0], gaps

    T = 600
    XY = np.tile(np.array(_SQUAD_442, float), (T, 1, 1))
    M = np.ones((T, 11), bool)
    t = np.arange(T, dtype=float)

    assert infer_attacking_right(XY, M, pitch.length) is True
    flipped = XY.copy()
    flipped[:, :, 0] = pitch.length - flipped[:, :, 0]
    assert infer_attacking_right(flipped, M, pitch.length) is False

    # THE TWO SIDES MUST COME OUT OPPOSITE, and by construction rather than by
    # luck. Both banks here are the SAME 4-4-2 facing the SAME way, so the
    # per-side estimator returns True twice - the impossible answer that shipped
    # in data/jobs/118575_full_1st/match.final.json. The pair function has to
    # overrule the weaker side and say so. The strong bank keeps its keeper at
    # x=5, clear of the back four by 17 m; the weak one's keeper has advanced to
    # x=15 and clears it by 7, so home is the better-evidenced side.
    weak = XY.copy()
    weak[:, 0, 0] = 15.0
    assert infer_attacking_right(XY, M, pitch.length) is True
    assert infer_attacking_right(weak, M, pitch.length) is True, \
        "the fixture no longer reproduces the both-True contradiction"
    dirs, flags = attacking_directions({"home": (XY, M, []), "away": (weak, M, [])},
                                       pitch.length)
    assert dirs["home"] != dirs["away"], dirs      # impossible answer, made impossible
    assert dirs == {"home": True, "away": False}, dirs
    assert flags == ("attacking_direction_conflict",), flags
    # ...and a conflict that is NOT there must not be reported. Two banks facing
    # opposite ways agree with each other, so no flag - a caveat that fires
    # every time is a caveat nobody reads.
    ok_dirs, ok_flags = attacking_directions({"home": (XY, M, []), "away": (flipped, M, [])},
                                             pitch.length)
    assert ok_dirs == {"home": True, "away": False}, ok_dirs
    assert ok_flags == (), ok_flags
    # MIRROR THE PITCH AND EVERY ANSWER MUST INVERT. Without this the suite
    # cannot tell this function from a constant {home: True, away: False}: all
    # four SoccerTrack halves on disk have the GT `home` team attacking right,
    # so that constant scores 12/12 on the only ground truth this repo owns.
    # Bank NAMES are held fixed here and only the positions are reflected.
    mir = {s: (np.stack([pitch.length - P[:, :, 0], P[:, :, 1]], axis=2), Mk, [])
           for s, (P, Mk, _) in {"home": (XY, M, []), "away": (weak, M, [])}.items()}
    m_dirs, m_flags = attacking_directions(mir, pitch.length)
    assert all(m_dirs[s] != dirs[s] for s in dirs), (dirs, m_dirs)
    assert m_flags == flags, (flags, m_flags)

    # No keeper anywhere: refuse and say so, rather than return a coin flip
    # dressed as a measurement. A flat line of eleven has no deep cluster.
    flat = np.zeros((T, 11, 2)); flat[:, :, 0] = 50.0
    flat[:, :, 1] = np.arange(11) * 6.0
    ref_dirs, ref_flags = attacking_directions({"home": (flat, M, []), "away": (flat, M, [])},
                                               pitch.length)
    assert ref_flags == ("attacking_direction_refused",), ref_flags
    assert ref_dirs["home"] != ref_dirs["away"], ref_dirs

    rows = shape_series(XY, M, True, pitch.length, t)
    assert len(rows) == T
    r = rows[0]
    # The keeper is at x=5. If he were in the "4 deepest" the line reads 17.8 m
    # instead of 22 - the ~15 m error the old np.sort(xs)[:4] shipped.
    assert r["rear_line_m"] == 22.0, r["rear_line_m"]
    assert r["n"] == 10, r["n"]
    assert abs(r["depth_m"] - 46.0) < 0.01, r["depth_m"]
    assert abs(r["width_m"] - 44.4) < 0.01, r["width_m"]
    assert [round(v) for v in r["lines_m"]] == [22, 45, 68], r["lines_m"]
    # Hull of the ten outfielders, not the 46x48 bounding box (2208 m^2).
    assert 1300 < r["hull_area_m2"] < 1900, r["hull_area_m2"]

    # A keeper split into TWO track fragments must still be excluded. Regression:
    # this is 117092_2nd, where a single-index rule saw two "players" 0.9 m apart,
    # refused, and left a keeper in the outfield set - line height read 26.8 m
    # against 33-47 m for every other team in the dataset.
    split = np.concatenate([XY, XY[:, :1] + [1.0, 0.0]], axis=1)
    sm = np.ones((T, 12), bool)
    sm[:T // 2, 0] = False
    sm[T // 2:, 11] = False
    assert _goalkeepers(split[:, :, 0], sm) == {0, 11}
    assert shape_series(split, sm, True, pitch.length, t)[0]["rear_line_m"] == 22.0

    # Percentiles, not min-max: one player wandering 24 m to the touchline must
    # move team width by a few metres, not by the full 24. min-max would report
    # +8.0 m here; the whole reason for the 10th-90th is that it does not.
    wide = XY.copy()
    wide[:, 7, 1] = 66.0
    dw = shape_series(wide, M, True, pitch.length, t)[0]["width_m"] - r["width_m"]
    assert 0 < dw < 4.0, dw

    # Mirroring the team must not change a single shape number: every metric is
    # expressed from the team's OWN goal line.
    rf = shape_series(flipped, M, False, pitch.length, t)[0]
    assert abs(rf["rear_line_m"] - r["rear_line_m"]) < 1e-6
    assert abs(rf["hull_area_m2"] - r["hull_area_m2"]) < 1e-6

    forms = formation_windows(XY, M, True, pitch.length, t)
    assert len(forms) == 20, len(forms)
    assert {f["formation"] for f in forms} == {"4-4-2"}, forms[0]
    assert forms[0]["cost"] < 0.1, forms[0]
    fs = formation_summary(forms)
    assert fs == {"formation": "4-4-2", "mode_share": 1.0, "windows": 20, "changes": 0}, fs
    # A tied count is SHOWN: every tied label, first-seen order, in both orders (a set() fails one under some seeds).
    for labs in (["3-5-2", "4-3-3", "4-3-3", "3-5-2", "5-3-2"], ["4-3-3", "3-5-2", "3-5-2", "4-3-3", "5-3-2"]):
        ts = formation_summary([{"formation": x} for x in labs])
        assert ts["formation"] == f"{labs[0]} / {labs[1]}" and ts["tied"] == labs[:2], (labs, ts)
        assert ts["mode_share"] == 0.4, ts

    # Structure stress: a team that holds its 4-4-2 and then sends two players
    # to a corner for 60 s must rank THAT stretch first and cover it. The list is
    # top-N now, so "exactly one interval" is no longer the assertion - "the real
    # event outranks every filler interval, and by a wide margin" is.
    broken = XY.copy()
    broken[300:360, 9] = [100.0, 64.0]
    broken[300:360, 10] = [98.0, 66.0]
    bs = shape_series(broken, M, True, pitch.length, t)
    bf = formation_windows(broken, M, True, pitch.length, t)
    st = structure_stress(bs, bf)
    assert st[0]["t0"] == 300.0 and st[0]["t1"] == 359.0, st[0]
    assert st[0]["duration_s"] == 60.0, st[0]
    assert len(st) == 1 or st[0]["severity"] > 5 * st[1]["severity"], st[:2]
    # One interval is not a distribution, and Gini would call it "flat".
    assert stress_profile(st) == {}, stress_profile(st)
    # The cut points, on the real severity vectors they were fitted to:
    # 118575_1st away, 117092_1st home, 117092_1st away.
    band = lambda v: stress_profile([{"severity": s} for s in v])["profile"]  # noqa: E731
    assert band([166.8, 62.7, 44, 43.3, 20.8, 18.9, 18.8, 17.4, 14.5, 14.4]) == "peaked"
    assert band([73.7, 41.3, 28.1, 26.1, 24.7, 22, 21.5, 18.5, 14.2, 12.8]) == "moderate"
    assert band([57.8, 49.6, 44.6, 43.8, 42.1, 35, 30.7, 30.5, 27.8, 25.5]) == "flat"

    # A team that COMPRESSES rather than stretches must not be scored as
    # stressed, however badly it fits a template. Regression: 118575_1st home
    # 15:00, nine outfielders inside a 10x20 m box in the opponent's corner, hull
    # z -0.86, cost z +4.89 - ranked #1 on the sum alone and it is a siege, not
    # stress. Here the same ten players pack into one corner.
    squash = XY.copy()
    squash[300:360, 1:] = np.linspace(96.0, 104.0, 10)[:, None] * [1.0, 0.0] + [0.0, 62.0]
    ss = shape_series(squash, M, True, pitch.length, t)
    sf = formation_windows(squash, M, True, pitch.length, t)
    assert max([r["hull_area_m2"] for r in ss[300:360]]) < np.median(
        [r["hull_area_m2"] for r in ss]), "the squashed stretch must be COMPACT"
    assert not any(300 <= s["t0"] <= 359 for s in structure_stress(ss, sf)), \
        structure_stress(ss, sf)[:2]
    # A team that never moves has zero variance in everything, so there is no
    # "most stressed" moment to rank and the honest answer is nothing. Ranking
    # guarantees a non-empty list only when the input actually varies.
    assert structure_stress(rows, forms) == [], structure_stress(rows, forms)

    # team_shape: the median of the series, with the old keys intact.
    sh = team_shape([], pitch, True, rows)
    assert sh["defensive_line_m"] == 22.0 and abs(sh["width_m"] - 44.4) < 0.05, sh
    assert set(sh) >= {"width_m", "depth_m", "defensive_line_m", "compactness_m2"}
    # The no-series fallback keeps the old signature working.
    ps = [PlayerStats(i, "home", None, 90, 0, 0, 0, ax, ay, 100, 0.0, 1.0)
          for i, (ax, ay) in enumerate(_SQUAD_442)]
    assert team_shape(ps, pitch, True)["defensive_line_m"] == 22.0


def _demo() -> None:
    """Synthetic player on a known path; the numbers must come back out."""
    global MAX_SPEED          # E-CLAMP lifts it to prove distance cannot see it
    pitch = Pitch(105.0, 68.0)
    # Identity-ish camera so pitch metres == image "pixels", isolating the math.
    cal = Calibration(pitch=pitch, H=np.eye(3))
    fps = 25.0

    # Straight 100 m at exactly 5 m/s = 20 s.
    n = int(20 * fps)
    track = [{"f": i, "id": 1, "team": "home", "jersey": 9,
              "x": 5.0 * (i / fps), "y": 34.0} for i in range(n + 1)]
    ps = player_stats(track, cal, fps)
    assert abs(ps.distance_m - 100.0) < 1.0, ps.distance_m
    assert abs(ps.top_speed_ms - 5.0) < 0.2, ps.top_speed_ms
    assert ps.sprints == 0, ps.sprints  # 5.0 is under the 5.5 threshold

    # Same path at 7 m/s must register exactly one sprint.
    fast = [{"f": i, "id": 2, "team": "home", "x": 7.0 * (i / fps), "y": 30.0}
            for i in range(int(10 * fps) + 1)]
    assert player_stats(fast, cal, fps).sprints == 1

    # Jitter must not manufacture distance. A stationary player with 15cm of
    # per-frame noise would integrate to hundreds of fake metres unsmoothed.
    rng = np.random.default_rng(0)
    still = [{"f": i, "id": 3, "team": "away",
              "x": 50.0 + rng.normal(0, 0.15), "y": 34.0 + rng.normal(0, 0.15)}
             for i in range(int(60 * fps))]
    jitter_m = player_stats(still, cal, fps).distance_m
    assert jitter_m < 40.0, f"jitter inflated distance to {jitter_m:.0f}m over 60s"

    # A teleport must be clamped, and clamped_frac must say so.
    jump = [{"f": 0, "id": 4, "team": "home", "x": 10.0, "y": 10.0},
            {"f": 1, "id": 4, "team": "home", "x": 90.0, "y": 60.0}]
    jps = player_stats(jump, cal, fps)
    assert jps.top_speed_ms <= MAX_SPEED
    assert jps.clamped_frac == 1.0, jps.clamped_frac
    # A clean run must NOT be flagged, or the signal is noise.
    assert player_stats(track, cal, fps).clamped_frac == 0.0

    # A gap in the track must not be integrated as running.
    gapped = ([{"f": i, "id": 5, "team": "home", "x": 0.0, "y": 34.0} for i in range(5)]
              + [{"f": 500, "id": 5, "team": "home", "x": 80.0, "y": 34.0}])
    assert player_stats(gapped, cal, fps).distance_m < 5.0

    # ----------------------------------------------------------------- E-CLAMP
    # WHAT THE CLAMP ACTUALLY FIRES ON. speed-clamp (not published), in eight lines.
    # A player walking at a flat 3 m/s, occluded for 26 frames, re-acquired 15 m
    # further on - which is what a real occlusion looks like and is well inside
    # what 1.04 s of walking plus a re-detection can produce. NOTHING in the
    # input is fast: every OBSERVED step is exactly 3 m/s. The gap mask does its
    # job and excludes the one step across the hole. And the reported speed
    # still saturates MAX_SPEED for the next second and a half, because
    # `kalman_ground` propagates through the gap and takes the whole 18 m
    # residual into its velocity state. This is the mechanism behind
    # `api://product.speed.clamped_frac` = 0.4661 and behind HSR reading 1.50x.
    occluded = ([{"f": i, "id": 6, "team": "home", "x": 3.0 * (i / fps), "y": 34.0}
                 for i in range(250)]
                + [{"f": i, "id": 6, "team": "home", "x": 3.0 * (i / fps) + 15.0, "y": 34.0}
                   for i in range(275, 525)])
    v, dts, real, _xy = speed_series(occluded, cal, fps)
    step_ms = np.linalg.norm(np.diff(np.array([[p["x"], p["y"]] for p in occluded]),
                                     axis=0), axis=1) / dts
    assert step_ms[real].max() < 3.01, step_ms[real].max()   # the input is a walk
    assert (~real).sum() == 1                                # one hole, correctly masked
    hit = np.flatnonzero(real & (v >= MAX_SPEED - 1e-9))
    gap_at = int(np.flatnonzero(~real)[0])
    assert len(hit) > 0, "the post-gap transient is gone - if that was deliberate, " \
        "speed-clamp (not published) §8 is the fix and this assert is what it flips"
    # The defect must stay BOUNDED. Measured settling is 2.0-3.7 s and this
    # fixture rings for 1.48 s; if a tuning change doubles that, fail here.
    assert gap_at < hit[0] and (hit[-1] - gap_at) <= 2.0 * fps, (gap_at, hit[[0, -1]])
    # And it manufactures a sprint out of a walk - the same transient, costing
    # the count metric rather than the peak.
    assert player_stats(occluded, cal, fps).sprints == 1, "3 m/s is not a sprint"
    # THE CLAMP CANNOT REACH DISTANCE. `_windowed` reads the trajectory, never
    # the speed series, so MAX_SPEED is inert here - bit-identical, not close.
    # This is why sweeping it moved product.distance.* by exactly zero, and it
    # is the assert that fails if distance is ever wired to the speed series.
    shipped = player_stats(occluded, cal, fps).distance_m
    _keep, MAX_SPEED = MAX_SPEED, float("inf")
    try:
        lifted = player_stats(occluded, cal, fps).distance_m
    finally:
        MAX_SPEED = _keep
    assert lifted == shipped, f"distance moved with the clamp: {lifted} vs {shipped}"

    # ------------------------------------------------------------------ E2
    # DISTANCE MUST NOT DEPEND ON WINDOW PHASE. A shuttle at constant speed has
    # the same true path length in every equal-duration slice of it, so every
    # phase offset is measuring the same quantity and any spread is the window
    # boundaries landing differently against the turns. This is the regression
    # test for the phase-averaging in `_windowed`: with a single phase these ten
    # offsets ran 255.5-462.6 m, sd 71.6 on a mean of 359.0. It is exactly zero
    # here rather than merely small because k divides the shuttle's period, so
    # a non-zero sd means the phase average is not being taken at all.
    fps_s, leg_s, sp = 25.0, 0.8, 4.0
    ns = int(120 * fps_s)
    ts = np.arange(ns) / fps_s
    sign = np.where(np.floor(ts / leg_s).astype(int) % 2 == 0, 1.0, -1.0)
    sx = np.concatenate([[0.0], np.cumsum(sign[:-1] * sp / fps_s)]) + 30.0
    shuttle = np.stack([sx, np.full(ns, 34.0)], axis=1)
    kw = int(round(DISTANCE_WINDOW_S * fps_s))
    # Equal-length slices, so the true path is identical in all of them and only
    # the phase differs.
    phases = np.array([path_length(shuttle[p:ns - kw + p], ts[p:ns - kw + p],
                                   np.ones(ns - kw - 1, bool)) for p in range(kw)])
    assert phases.std(ddof=1) < 0.01, (phases.min(), phases.max(), phases.std(ddof=1))
    # ...and it is still a CHORD, so it must under-read a turning player. The
    # role-correlation this pins is the reason distance carries no "the bias is
    # common to everyone" defence - see DISTANCE_WINDOW_S.
    straight = np.stack([sp * ts + 2.0, np.full(ns, 34.0)], axis=1)
    true_m = sp * (ns - 1) / fps_s
    assert abs(path_length(straight, ts, np.ones(ns - 1, bool)) / true_m - 1) < 0.01
    assert 0.60 < phases.mean() / (sp * (ns - kw - 1) / fps_s) < 0.85, phases.mean()

    # ------------------------------------------------------------------ E3
    # A DROPPED WINDOW MUST NOT BE SILENT. Gaps delete whole windows, and the
    # only honest signal that a distance is an under-count is the share of
    # windows that survived. Clean track: everything measured.
    assert player_stats(track, cal, fps).observed_frac == 1.0
    # Punch a hole every 5 s out of a constant-speed run. Distance must fall, and
    # `1 - observed_frac` must move with it and stay BELOW it - it is a floor on
    # the missing distance, not an estimate of it, because a window is k samples
    # wide and one that straddles a gap swallows the gap's wall clock too. A UI
    # that read it as the loss would under-state a badly occluded track by 3.6x.
    cs = player_stats(track, cal, fps)
    prev = 0.0
    for hole, floor_pct in ((5, 8.2), (50, 12.9)):
        holed = [p for p in track if not (0 < (p["f"] % 125) <= hole)]
        hs = player_stats(holed, cal, fps)
        lost = 1 - hs.distance_m / cs.distance_m
        missed = 1 - hs.observed_frac
        assert abs(missed * 100 - floor_pct) < 0.5, (hole, missed)
        assert 0 < missed <= lost, (hole, missed, lost)
        assert lost > prev, (hole, lost, prev)      # more occlusion, more loss
        prev = lost
    # And the trap this field exists to expose: `minutes` is the track SPAN, so
    # it still counts the gap time that distance just dropped. km/90 is wrong in
    # both numerator and denominator, same direction. Documented in player_stats.
    assert abs(hs.minutes - cs.minutes) < 0.01, (hs.minutes, cs.minutes)

    # Sub-sampled tracks (every Nth frame) must still measure distance.
    # Regression: a frame-rate-derived gap threshold zeroed these out.
    sub = [{"f": i, "id": 6, "team": "home", "x": 5.0 * (i / fps), "y": 34.0}
           for i in range(0, n + 1, 5)]
    assert abs(player_stats(sub, cal, fps).distance_m - 100.0) < 2.0

    # Territory must sum to 100 and land in the right third.
    terr = territory({1: [{"f": 0, "id": 1, "x": 95.0, "y": 34.0}]}, cal)
    assert terr["attacking_third"] == 100.0, terr
    assert abs(sum(terr.values()) - 100.0) < 0.1

    _demo_shape(pitch)
    # Non-finite positions are not counted (they used to land in a third via astype(int)).
    nan_cal = type("NanCal", (), {"pitch": pitch, "to_pitch": lambda self, uv: np.where(uv[:, :1] > 50, np.nan, uv)})()
    assert territory({1: [{"x": 40.0, "y": 34.0}, {"x": 60.0, "y": 34.0}]}, nan_cal)["middle_third"] == 100.0
    assert _nonfinite({"a": [1.0, float("nan")]}) == "match.a.1" and _nonfinite({"a": [1.0, 2]}) is None
    # A heatmap point off the pitch beyond the margin, or non-finite, is dropped.
    hp = heatmap_points([{"x": 10.0, "y": 34.0}, {"x": 200.0, "y": 34.0}, {"x": -2.0, "y": 70.0}], cal)
    assert hp == [[10.0, 34.0], [-2.0, 70.0]], hp
    assert heatmap_points([{"x": 10.0, "y": 34.0}, {"x": 60.0, "y": 34.0}], nan_cal) == [[10.0, 34.0]]

    # An official must never reach the players table. Referees outrun most
    # outfielders, so this is the one exclusion that fails loudly if it regresses.
    import tempfile, os  # noqa: PLC0415 - self-check only
    with tempfile.TemporaryDirectory() as td:
        tp, cp, op = (os.path.join(td, n) for n in ("t.json", "c.json", "m.json"))
        dets = []
        for tid, team, y in ((1, "home", 20.0), (2, "away", 40.0), (3, "official", 60.0)):
            dets += [{"f": i, "id": tid, "team": team, "jersey": None,
                      "x": 6.0 * (i / fps), "y": y} for i in range(0, int(30 * fps), 5)]
        with open(tp, "w") as f:
            json.dump({"fps": fps, "detections": dets}, f)
        Calibration(pitch=pitch, H=np.eye(3)).save(cp)
        m = build_match(tp, cp, op)
        assert [p["id"] for p in m["players"]] == [1, 2], m["players"]
        # SYNTHETIC TRACKS JUSTIFY NOTHING - through build_match itself, so
        # deleting the call there fails here, not just unjustify() in isolation.
        syn = build_match(tp, cp, op, meta={"synthetic_tracking": True})
        assert all(e["confidence"] == NOT_EVALUATED and e["justified_by"] is None
                   for e in syn["fields"].values()), "synthetic tracks were justified"
        # observed_frac has to reach the wire, not just PlayerStats -
        # app/frontend/src/lib/report.ts reads it off the player row.
        assert all(p["observed_frac"] == 1.0 for p in m["players"]), m["players"]
        # PEAK SPEED REPORTS when the clamp never bound. These tracks run at
        # 6 m/s, under MAX_SPEED, so a null here would mean the refusal fires on
        # everything - and a mark on every row is one a reader learns to skip.
        assert all(p["top_speed_kmh"] is not None for p in m["players"]), m["players"]
        assert m["teams"]["home"]["top_speed_kmh"] is not None, m["teams"]
        assert "measures_clamp" not in m["fields"]["players.top_speed_kmh"]["caveat_flags"]
        assert "3" not in m["heatmaps"], "official leaked into heatmaps"
        assert m["meta"]["officials_excluded"] == 1, m["meta"]
        # Sprints are WITHHELD: null on every row and side, and the envelope says why.
        assert all(p["sprints"] is None for p in m["players"]), m["players"]
        assert all(t["sprints"] is None for t in m["teams"].values()), m["teams"]
        for k in ("players.sprints", "teams.home.sprints", "teams.away.sprints"):
            assert "sprints_withheld" in m["fields"][k]["caveat_flags"], (k, m["fields"][k])
        # ...and a match.json written before the refusal is served with its counts withheld too.
        with open(op) as f:
            old = json.load(f)
        old["players"][0]["sprints"], old["teams"]["home"]["sprints"] = 3, 5
        with open(op, "w") as f:
            json.dump(old, f)
        served = resolved_match(op)
        assert served["players"][0]["sprints"] is None and served["teams"]["home"]["sprints"] is None, served["teams"]
        # The quality gate is wired into every build, so this asserts it is still
        # wired - a gate that quietly stops being called is the whole failure
        # mode it exists to prevent. `ok` is asserted UNREACHABLE on purpose:
        # five of its nine criteria need a ball the stats do not read, so a
        # green verdict here would mean the ball criteria had started lying.
        q = m["meta"]["quality"]
        assert q["verdict"] in ("fail", "incomplete"), q
        assert len(q["criteria"]) == 9, q
        assert sum(r["status"] == "not_evaluated" for r in q["criteria"]) == 5, q
        # 1 v 1 is balanced, so the split caveat must stay OFF. A flag that
        # fires on everything is a flag a reader learns to skip.
        assert "team_split_implausible" not in m["fields"]["players.team"]["caveat_flags"]
        # ...and the permanent one must be on regardless, because the side is
        # clustered rather than tagged even when the split looks reasonable.
        assert "kit_colour_clustering" in m["fields"]["players.team"]["caveat_flags"]
        # The TIER is not `prov` any more, and that is this change: nothing in
        # the registry measures the side this field ships (team.accuracy.* is
        # an oracle relabel of a different function on a different file), so it
        # publishes no tier at all.
        assert m["fields"]["players.team"]["confidence"] == NOT_EVALUATED, \
            m["fields"]["players.team"]

        # 2 v 1 is past TEAM_BALANCE_MAX, so it must fire.
        dets += [{"f": i, "id": 4, "team": "home", "jersey": None,
                  "x": 6.0 * (i / fps), "y": 25.0} for i in range(0, int(30 * fps), 5)]
        with open(tp, "w") as f:
            json.dump({"fps": fps, "detections": dets}, f)
        lop = build_match(tp, cp, op)
        assert "team_split_implausible" in lop["fields"]["players.team"]["caveat_flags"], \
            lop["fields"]["players.team"]

        # ...and the case the balance test alone CANNOT see: two piles that are
        # beautifully balanced and are both far too big to be football teams.
        # The shipped report had 183 v 172 - a ratio of 1.06 - and the flag
        # stayed off, which is the whole reason TEAM_SIZE_MAX exists.
        many = []
        for tid in range(2 * TEAM_SIZE_MAX + 2):
            many += [{"f": i, "id": 1000 + tid, "team": "home" if tid % 2 else "away",
                      "jersey": None, "x": 5.0 + 0.01 * i, "y": 5.0 + tid}
                     for i in range(0, int(30 * fps), 5)]
        with open(tp, "w") as f:
            json.dump({"fps": fps, "detections": many}, f)
        big = build_match(tp, cp, op)
        n_side = [sum(1 for q in big["players"] if q["team"] == sd) for sd in ("home", "away")]
        assert max(n_side) / min(n_side) <= TEAM_BALANCE_MAX, n_side   # balanced...
        assert "team_split_implausible" in big["fields"]["players.team"]["caveat_flags"], \
            f"{n_side} tracks a side is not a football team, and balance cannot see it"

        # ...and it REFUSES where the clamp bound. One clean home track, one away
        # track that teleports 80 m every 0.2 s: the away row and the away
        # headline must both come out null rather than 34.2 km/h (= MAX_SPEED *
        # 3.6, a constant in this file), while home keeps a real number. If both
        # sides refused, the refusal would be saying nothing about the data.
        mix = [{"f": i, "id": 8, "team": "home", "jersey": None,
                "x": 6.0 * (i / fps), "y": 20.0} for i in range(0, int(30 * fps), 5)]
        mix += [{"f": i, "id": 9, "team": "away", "jersey": None,
                 "x": 5.0 + 80.0 * ((i // 5) % 2), "y": 40.0}
                for i in range(0, int(30 * fps), 5)]
        with open(tp, "w") as f:
            json.dump({"fps": fps, "detections": mix}, f)
        mm = build_match(tp, cp, op)
        peak = {q["id"]: q["top_speed_kmh"] for q in mm["players"]}
        assert peak[9] is None, f"clamped track shipped {peak[9]} km/h, not a refusal"
        assert peak[8] is not None, "clean track lost its peak speed to the refusal"
        assert mm["teams"]["away"]["top_speed_kmh"] is None, mm["teams"]["away"]
        assert mm["teams"]["home"]["top_speed_kmh"] is not None, mm["teams"]["home"]
        # The column envelope counts the rows that still HAVE a reading, not the
        # rows in the table, and it says why the others are gone.
        env = mm["fields"]["players.top_speed_kmh"]
        assert "measures_clamp" in env["caveat_flags"], env
        assert env["n"] == 1, env
        assert "measures_clamp" in mm["fields"]["teams.away.top_speed_kmh"]["caveat_flags"]
        assert "measures_clamp" not in mm["fields"]["teams.home.top_speed_kmh"]["caveat_flags"]
        # DISTANCE is flagged, not refused, where the clamp bound: same signal, on the columns and the side it touches.
        for k in ("players.distance_km", "derived.per90", "teams.away.distance_m"):
            assert "distance_unbounded" in mm["fields"][k]["caveat_flags"], (k, mm["fields"][k])
        assert "distance_unbounded" not in mm["fields"]["teams.home.distance_m"]["caveat_flags"]
        assert all(q["distance_km"] is not None for q in mm["players"]), mm["players"]

        # TERRITORY counts the players table only: an official and a 3 s fragment in the right third move nothing.
        pop = [{"f": i, "id": tid, "team": team, "jersey": None, "x": x, "y": 34.0}
               for tid, team, x, n_f in ((20, "home", 10.0, 30), (21, "away", 12.0, 30),
                                         (22, "official", 95.0, 30), (23, "home", 95.0, 3),
                                         (24, "home", 10.3, 30))
               for i in range(0, int(n_f * fps), 5)]
        with open(tp, "w") as f:
            json.dump({"fps": fps, "detections": pop}, f)
        tm = build_match(tp, cp, op)
        assert tm["territory"]["defensive_third"] == 100.0, tm["territory"]
        # TEAM DISTANCE counts one track per duplicate group (20 and 24 stand 0.3 m apart all match) and says
        # what share of the side's detections it counted: 150 of 315 (the fragment is not a players-table row).
        assert tm["teams"]["home"]["distance_coverage"] == round(150 / 315, 4), tm["teams"]["home"]
        assert "team_distance_partial" in tm["fields"]["teams.home.distance_m"]["caveat_flags"]
        assert tm["fields"]["teams.home.distance_m"]["n"] == 1, tm["fields"]["teams.home.distance_m"]

        # fps missing or <= 0 is refused, and so is a match that would carry a NaN (json allow_nan=False).
        for bad in ({"detections": pop}, {"fps": 0, "detections": pop}, {"fps": -25.0, "detections": pop},
                    {"fps": fps, "detections": [dict(d, x=float("nan")) if k == 3 else d for k, d in enumerate(pop)]}):
            with open(tp, "w") as f:
                json.dump(bad, f)
            try:
                build_match(tp, cp, op)
            except ValueError as e:
                assert "refused" in str(e), e
            else:
                raise AssertionError(f"build_match accepted {str(bad)[:60]}")

        # ...and REFUSED above the physical ceiling: 15 m/s for 30 s is more ground than MAX_SPEED allows.
        fast = [{"f": i, "id": 10, "team": "home", "jersey": None, "x": 5.0 + 15.0 * (i / fps) % 90, "y": 20.0}
                for i in range(0, int(30 * fps), 5)]
        with open(tp, "w") as f:
            json.dump({"fps": fps, "detections": mix + fast}, f)
        cm = build_match(tp, cp, op)
        dk = {q["id"]: q["distance_km"] for q in cm["players"]}
        assert dk[10] is None and dk[8] is not None, dk
        assert "distance_over_ceiling" in cm["fields"]["players.distance_km"]["caveat_flags"]
        assert cm["fields"]["players.distance_km"]["n"] == 2, cm["fields"]["players.distance_km"]
        assert cm["teams"]["home"]["distance_m"] == mm["teams"]["home"]["distance_m"], (cm["teams"], mm["teams"])

    # Every flag a metric names must exist, and the two that were retired must
    # be gone from the fields that used to carry them - distance HAS now been
    # compared end to end (model/validation/dist_eval.py), and `damped_by_filter` said
    # peak speed reads low when it reads high.
    assert all(f in CAVEATS for _u, _p, _t, fl, *_i in METRICS.values()
               for f in fl), METRICS
    assert "damped_by_filter" not in CAVEATS
    for fld in ("players.distance_km", "teams.distance_m", "derived.per90"):
        assert "distance_bias_measured" in METRICS[fld][3], fld
        assert "unvalidated_end_to_end" not in METRICS[fld][3], fld
    for fld in ("players.top_speed_kmh", "teams.top_speed_kmh"):
        assert "peak_speed_overreads" in METRICS[fld][3], fld
        assert "unvalidated_end_to_end" not in METRICS[fld][3], fld

    # ---- E-JUSTIFY: a tier is resolved, and the resolution can FAIL ---------
    # Three ways this could ship broken, so three gates. A self-check that
    # cannot fail is postmortem (not published) A, and the thing being guarded here is
    # a claim about accuracy on a coach's screen.
    real, keep_v = accepted_measurements, VALIDATION
    patch = lambda f: globals().__setitem__("accepted_measurements", f)  # noqa: E731
    try:
        # 1. AN ID NOBODY REGISTERED MUST NOT RESOLVE. Typos and wishful
        #    thinking degrade; they never quietly grant a tier.
        patch(lambda: frozenset({"product.distance.ratio_raw"}))
        METRICS["_selfcheck"] = ("m", "measured", "ok", (), "no.such.id")
        assert metric("_selfcheck")["confidence"] == NOT_EVALUATED
        assert metric("_selfcheck")["justified_by"] is None
        # 2. A RESOLVED ID PUBLISHES THE TIER AND SAYS WHICH ID.
        METRICS["_selfcheck"] = ("m", "measured", "ok", (),
                                 "product.distance.ratio_raw")
        assert metric("_selfcheck")["confidence"] == "ok"
        assert metric("_selfcheck")["justified_by"] == "api://product.distance.ratio_raw"
        # 3. RESOLVING DOES NOT OVERRIDE A CAVEAT. The weakest link still
        #    governs - justification is a gate on the ceiling, not a promotion.
        METRICS["_selfcheck"] = ("m", "measured", "ok", ("unvalidated_end_to_end",),
                                 "product.distance.ratio_raw")
        assert metric("_selfcheck")["confidence"] == "prov"
        # 4. AND WITH NO GOLDEN AT ALL, EVERYTHING DEGRADES. The direction a
        #    missing file must fail in.
        patch(frozenset)        # -> empty, however it is called
        assert metric("_selfcheck")["confidence"] == NOT_EVALUATED
        # 5. SYNTHETIC TRACKS JUSTIFY NOTHING, even where the id resolves.
        patch(lambda: frozenset({"product.distance.ratio_raw"}))
        METRICS["_selfcheck"] = ("m", "measured", "ok", (),
                                 "product.distance.ratio_raw")
        env = unjustify({"f": metric("_selfcheck")})["f"]
        assert env["confidence"] == NOT_EVALUATED and env["justified_by"] is None
        # 6. A STORED TIER IS RE-RESOLVED, not trusted. An envelope written as
        #    `ok` while its id resolved must fall to not_evaluated once the
        #    golden stops backing it - per side too - and an unknown flag or
        #    field degrades rather than raising.
        METRICS["teams._selfcheck"] = METRICS["_selfcheck"]
        stale = {"teams.home._selfcheck": dict(metric("_selfcheck"), confidence="ok"),
                 "gone.field": {"confidence": "ok", "caveat_flags": []},
                 "_selfcheck": {"confidence": "ok", "caveat_flags": ["no_such_flag"]}}
        assert metric("_selfcheck")["confidence"] == "ok"      # backed right now
        patch(frozenset)
        now = resolve(stale)
        assert all(e["confidence"] == NOT_EVALUATED and e["justified_by"] is None
                   for e in now.values()), now
        patch(lambda: frozenset({"product.distance.ratio_raw"}))
        assert resolve(stale)["teams.home._selfcheck"]["confidence"] == "ok"
        assert resolve(stale, {"synthetic_tracking": True})[
            "teams.home._selfcheck"]["confidence"] == NOT_EVALUATED
        # 7. A GOLDEN REWRITTEN UNDER A RUNNING SERVER IS SEEN. It was read
        #    once per process, so a withdrawn id stayed cited until restart.
        import os as _os  # noqa: PLC0415
        import pathlib  # noqa: PLC0415
        import tempfile as _tf  # noqa: PLC0415
        with _tf.TemporaryDirectory() as td:
            globals()["VALIDATION"] = pathlib.Path(td)
            g = pathlib.Path(td) / "api-golden.json"
            g.write_text(json.dumps({"values": {"a.b": {"status": "OK"}}}))
            assert real() == {"a.b"}
            g.write_text(json.dumps({"values": {"a.b": {"status": "FAILED"}}}))
            t = g.stat().st_mtime_ns + 10**9
            _os.utime(g, ns=(t, t))
            assert real() == frozenset(), "a withdrawn id was still accepted"
    finally:
        patch(real)
        globals()["VALIDATION"] = keep_v
        METRICS.pop("_selfcheck", None)
        METRICS.pop("teams._selfcheck", None)

    # Every id named in the table must be a measurement that actually exists -
    # "do not invent an id" enforced rather than asked for. `api.py` is the only
    # place that knows, and it lives in the dev tree, so this gate runs where
    # the dev tree does and skips where it does not. That is also the layering:
    # engine ships, validation does not, and only the self-check crosses it.
    named = {i for _u, _p, _t, _f, *rest in METRICS.values()
             for i in rest if i is not None}
    sys.path.insert(0, str(VALIDATION))
    try:
        from measurements import MEASUREMENTS
    except ImportError:
        print(f"  (no validation tree: {len(named)} ids unchecked)")
    else:
        assert named <= set(MEASUREMENTS), sorted(named - set(MEASUREMENTS))
        live = named & accepted_measurements()
        print(f"  {len(named)} ids named, all registered; {len(live)} accepted "
              f"in the golden, {len(named - live)} awaiting a snapshot")

    # OCR numbers are unscored (docs/experiments/n49-jersey-ocr.md): the column must say so.
    j = metric("players.jersey")
    assert METRICS["players.jersey"][2] == "prov" and "jersey_ocr_unvalidated" in j["caveat_flags"], j

    print("stats.py ok")


if __name__ == "__main__":
    _demo()
