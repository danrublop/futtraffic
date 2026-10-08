"""Where does a player actually touch the ground?

The homography maps the GROUND PLANE. Feed it a point that is not on the ground
and it returns a position that is wrong by an amount proportional to how far off
the ground the point is and how far the player is from the camera:

    ground error ~= d_horiz * h / H_camera

The box bottom is the lowest visible pixel, which during a run is a FOOT IN THE
AIR, not the planted one. Foot elevation oscillates at step frequency (~2-3 Hz
at running cadence), so the projected position oscillates too - and oscillation
differentiates into distance that the player never covered.

    camera H=10m, player at d=50m, foot lift h=0.3m
    -> +-1.5 m of spurious ground-plane motion at 2-3 Hz

That is the mechanism. What this module exists to answer is whether it survives
the filter, because a low-bandwidth Kalman may already reject a 2-3 Hz signal.

Estimators are pluggable because the answer decides which one we need, and the
cheap ones cost nothing if the expensive one turns out to be unnecessary.

MEASURED 2026-08-03 - and the answer changes the deployment guide, not the code.
Box bottom + the q=1.0 ground-plane Kalman, distance error vs camera height:

    cam height   d=20m   d=40m   d=60m   d=80m   d=100m
        4 m       4.9     5.1     5.5     5.7     5.9   <- breaks the 5% gate
        6 m       3.1     3.2     3.4     3.6     3.8
        8 m       2.3     2.3     2.4     2.6     2.7
       10 m       1.7     1.8     1.9     1.9     2.1
       20 m       0.8     0.8     0.8     0.8     0.8

    ankle midpoint: ~0.0% at every height and range.

Two things fall out. First, the driver is CAMERA HEIGHT, almost not range - the
error is roughly flat across the pitch and drops as 1/H. Second, the filter does
most of the work: unfiltered, this oscillation is enormous (a 1.5 m excursion at
6 Hz is ~18 m/s of phantom speed), and a 0.4 s boxcar already knocks it to +6.5%
while the Kalman takes it to +1.8%.

CONCLUSION: mount at 6 m or higher and the box bottom is inside the gate for
TOTAL DISTANCE. That is a mounting instruction, not an engineering project, and
it is why pose estimation is not on the critical path.

DO NOT extend this to peak speed. Peak speed is an extreme-value statistic - it
takes the single largest velocity of the match - so a stride spike that averages
away over 90 minutes of distance can still set the maximum. This experiment says
nothing about it, and peak speed is report-only for exactly that reason.

Caveats: synthetic, pinhole rig with no lens distortion, one running speed
(6 m/s), and idealised |sin| foot lift. Re-run on real footage with real boxes
before treating the 6 m threshold as anything more than a planning number.
"""
from dataclasses import dataclass

import numpy as np

# Running cadence is ~180 steps/min = 3 steps/s. Each step lifts one foot, so the
# LOWEST-point signal repeats at step frequency, not stride frequency.
STEP_HZ = 3.0
FOOT_LIFT_M = 0.30


# The pluggable ground-point estimators that used to live here (BoxBottom /
# AnkleMidpoint / PelvisDrop + a registry) were deleted: zero callers ever, two
# of the three needed a pose model that is not in app/server/requirements.txt, and _demo
# below is what made them unnecessary - it concluded POSE NOT NEEDED at a 6m+
# mount. The box-bottom ground point is inlined at track.py and sceneprior.py.
# Re-add an abstraction if a low mount is ever forced on us, not before.

# ---------------------------------------------------------------------------

@dataclass
class Rig:
    """Minimal side-on camera. Enough to turn foot elevation into pixel error."""

    height_m: float = 10.0
    f_px: float = 1400.0     # focal length in pixels
    cx: float = 960.0
    cy: float = 540.0

    def project(self, x_m: np.ndarray, d_m: np.ndarray, z_m: np.ndarray):
        """World (along-pitch x, horizontal range d, height z) -> pixels."""
        u = self.cx + self.f_px * x_m / d_m
        v = self.cy + self.f_px * (self.height_m - z_m) / d_m
        return u, v

    def to_ground(self, u: np.ndarray, v: np.ndarray):
        """Invert, ASSUMING z=0. That assumption is exactly the bug we measure."""
        d = self.f_px * self.height_m / (v - self.cy)
        x = (u - self.cx) * d / self.f_px
        return x, d


def _demo() -> None:
    """Does stride oscillation survive the filter? That is the whole question."""
    import stats
    from pitch import Calibration, Pitch

    rig = Rig()
    fps, secs, speed = 25.0, 20.0, 6.0
    n = int(fps * secs)
    t = np.arange(n) / fps

    # Straight run across the far side of the pitch, where the error is worst.
    x_true = speed * t
    d_true = np.full(n, 50.0)
    true_dist = float(speed * secs)

    # Lowest visible point: a foot, alternating. |sin| is never negative - the
    # lowest foot is at ground level twice per cycle and lifted in between.
    foot_z = FOOT_LIFT_M * np.abs(np.sin(2 * np.pi * STEP_HZ * t))
    u_foot, v_foot = rig.project(x_true, d_true, foot_z)
    # Ankle midpoint: one foot up, one down, so the mean is near-constant.
    u_ank, v_ank = rig.project(x_true, d_true, np.full(n, 0.5 * FOOT_LIFT_M * 0.15))

    def measure(u, v, method):
        x_m, d_m = rig.to_ground(u, v)
        track = [{"f": i, "id": 1, "team": "home", "x": float(u[i]), "y": float(v[i])}
                 for i in range(n)]
        # Identity calibration: the rig's own inverse already produced metres, so
        # feed those straight through rather than double-projecting.
        cal = Calibration(pitch=Pitch(105.0, 68.0), H=np.eye(3))
        for i, p in enumerate(track):
            p["x"], p["y"] = float(x_m[i]), float(d_m[i])
        ps = stats.player_stats(track, cal, fps, method)
        return ps.distance_m

    rows = []
    for label, (u, v) in (("box bottom", (u_foot, v_foot)),
                          ("ankle midpoint", (u_ank, v_ank))):
        for method in ("boxcar", "kalman"):
            got = measure(u, v, method)
            rows.append((label, method, got, (got - true_dist) / true_dist * 100))

    print(f"  true distance {true_dist:.0f} m  "
          f"(run at {speed} m/s, {d_true[0]:.0f} m from a {rig.height_m:.0f} m camera)")
    for label, method, got, err in rows:
        print(f"  {label:<15} {method:<8} {got:7.1f} m   {err:+6.1f}%")

    box_kal = next(e for l, m, _, e in rows if l == "box bottom" and m == "kalman")
    ank_kal = next(e for l, m, _, e in rows if l == "ankle midpoint" and m == "kalman")

    # The claim under test: stride oscillation inflates distance via the box
    # bottom. The mechanism is REAL but it is a high-frequency one, so it is only
    # visible at a high measurement bandwidth - which is what distance used to be
    # computed at, when it integrated the per-sample speed series.
    #
    # Measured here, box-bottom + boxcar, as the distance window varies:
    #     0.04 s (~= sum of deltas, the old behaviour)  +11.3%
    #     0.08 s                                         +5.8%
    #     0.20 s                                         +1.8%
    #     0.40 s (shipped, stats.DISTANCE_WINDOW_S)      +1.7%
    #
    # So the assertion is now two-sided: the artefact must still exist when you
    # look for it at stride frequency, AND the shipped window must suppress it.
    # A one-sided ">5%" test would fail simply because the estimator improved,
    # and a one-sided "<3%" test would pass even if the mechanism vanished for
    # some unrelated reason and stopped protecting us.
    box_box = next(e for l, m, _, e in rows if l == "box bottom" and m == "boxcar")
    was = stats.DISTANCE_WINDOW_S
    try:                              # one sample per window == sum of deltas
        stats.DISTANCE_WINDOW_S = 1.0 / fps
        raw_err = (measure(u_foot, v_foot, "boxcar") - true_dist) / true_dist * 100.0
    finally:
        stats.DISTANCE_WINDOW_S = was
    print(f"  stride inflation at full bandwidth {raw_err:+.1f}%  ->  "
          f"{box_box:+.1f}% at the shipped {was:g}s window")
    assert raw_err > 5.0, f"mechanism should be visible at stride freq, got {raw_err:+.1f}%"
    assert box_box < 3.0, f"shipped window should suppress it, got {box_box:+.1f}%"
    # And the ankle midpoint must beat the box bottom under the same filter,
    # or a pose model buys us nothing and we should not pay for one.
    assert abs(ank_kal) < abs(box_kal) + 1e-9, (ank_kal, box_kal)

    verdict = ("POSE NEEDED - the Kalman does not suppress it"
               if abs(box_kal) > 5.0 else
               "POSE NOT NEEDED at these parameters - the Kalman already rejects it")
    print(f"  -> {verdict} (box+kalman {box_kal:+.1f}%, ankle+kalman {ank_kal:+.1f}%)")
    print("groundpoint.py ok")


if __name__ == "__main__":
    _demo()
