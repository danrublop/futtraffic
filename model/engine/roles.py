"""Who on this pitch is not a player, from how they move: `staff`, `linesman`, else `uncertain`.

A second signal beside kit colour (`track.assign_teams`). Never `referee` (that profile
overlaps a box-to-box midfielder's); never keeper or substitute (stated by a lineups feed,
or found by `stats._goalkeepers`).

IT NEVER WRITES `team`, AND THAT IS A SAFETY PROPERTY. `stats.build_match` deletes every
row where `team == "official"`, so a second writer there would turn a classifier mistake
into a real player vanishing from the match. `role` is its own field: a wrong role costs a
wrong label and nothing else.

The thresholds below are geometry, not measurement, and have never been scored.
measurement-evidence (not published)#evidence-roles-thresholds

    python model/engine/roles.py          # self-check
"""

from __future__ import annotations

import numpy as np

# Roaming radius of a technical-area occupant; far below any outfielder's.
STAFF_RADIUS_M = 10.0

# Observed time before stillness reads as a job rather than a pause; well above stats.MIN_MINUTES.
STAFF_MIN_S = 300.0

# A linesman stays in a band beside one touchline; generous for far-side calibration error.
# measurement-evidence (not published)#evidence-roles-thresholds
LINESMAN_BAND_M = 8.0        # how close to a touchline the median must sit
LINESMAN_Y_SPAN_M = 14.0     # how little they stray across the pitch
LINESMAN_X_SPAN_M = 30.0     # how much they travel along it

ROLES = ("staff", "linesman", "uncertain")


class Role:
    __slots__ = ("role", "confidence", "why")

    def __init__(self, role: str, confidence: float, why: str):
        assert role in ROLES, role
        self.role, self.confidence, self.why = role, float(confidence), why

    def __repr__(self) -> str:
        return f"Role({self.role!r}, {self.confidence:.2f}, {self.why!r})"


def _spread(a: np.ndarray) -> float:
    """p95 - p5, which is a span that one bad projection cannot inflate."""
    return float(np.percentile(a, 95) - np.percentile(a, 5)) if len(a) else 0.0


def classify(xy: np.ndarray, real: np.ndarray, fps: float, width_m: float,
             dt: np.ndarray | None = None) -> Role:
    """(T,2) metres + the per-step observed mask -> a role.

    `xy` and `real` are exactly what `stats.speed_series` already returns, so
    this never re-runs the Kalman - see stats.player_stats, which threads them
    in. `real` has length len(xy) - 1 (it is a STEP mask, not a sample mask),
    and observed TIME is what it measures, not track span: a track that exists
    for 40 minutes and is seen for 3 of them is not a coach, it is a player we
    lost, and using span here would call it one.
    """
    xy = np.asarray(xy, float)
    if len(xy) < 3 or not np.any(real):
        return Role("uncertain", 0.0, "too few observed samples to judge")

    # `real` is a per-STEP mask and a step is not a frame: stats.speed_series is
    # built on the sampled series, so at the shipped --stride 2 each step spans
    # 2/fps seconds. Counting samples and dividing by fps therefore reported
    # exactly half the wall clock, which put STAFF_MIN_S out of reach - a man
    # standing still for 400 s scored 200 s and came back "uncertain". stats.py
    # already has the true durations for observed_minutes; take them.
    observed_s = (float(np.sum(dt[real])) if dt is not None
                  else float(np.sum(real)) / max(fps, 1e-9))
    med = np.median(xy, axis=0)
    roam = float(np.percentile(np.linalg.norm(xy - med, axis=1), 90))
    y_span, x_span = _spread(xy[:, 1]), _spread(xy[:, 0])
    to_touchline = min(float(med[1]), float(width_m - med[1]))

    # STAFF first: it is the strongest signal and the only one that cannot be
    # produced by a player who is simply having a quiet half.
    if roam < STAFF_RADIUS_M and observed_s >= STAFF_MIN_S:
        # Confidence scales on how far inside the threshold it sits, so a coach
        # pinned to one spot reads higher than one pacing the whole area.
        c = float(np.clip(1.0 - roam / STAFF_RADIUS_M, 0.0, 1.0))
        return Role("staff", round(0.5 + 0.5 * c, 3),
                    f"roamed {roam:.1f} m over {observed_s / 60:.1f} observed minutes; "
                    f"no player stays inside {STAFF_RADIUS_M:.0f} m for that long")

    if (to_touchline <= LINESMAN_BAND_M and y_span <= LINESMAN_Y_SPAN_M
            and x_span >= LINESMAN_X_SPAN_M):
        c = float(np.clip(1.0 - y_span / LINESMAN_Y_SPAN_M, 0.0, 1.0))
        return Role("linesman", round(0.4 + 0.5 * c, 3),
                    f"travelled {x_span:.0f} m along the pitch inside a {y_span:.0f} m "
                    f"band {to_touchline:.1f} m from a touchline")

    return Role("uncertain", 0.0,
                f"roamed {roam:.1f} m, {x_span:.0f}x{y_span:.0f} m span - "
                "consistent with a player, and referee is not guessed here")


def cross_check(role: Role, team: str) -> bool | None:
    """Does motion agree with the kit-colour label? None when neither committed.

    DISAGREEMENT IS REPORTED, NEVER RESOLVED. Two weak signals that disagree are
    information about confidence, and picking a winner here throws that away -
    kit colour misplaces a large share of detections (benchmark (not published) §4,
    api://team.accuracy.detection) and these thresholds have never been
    scored, so there is no basis for either to overrule the other.
    """
    kit_says_official = team == "official"
    motion_says_official = role.role in ("staff", "linesman")
    if not kit_says_official and not motion_says_official:
        return None
    return kit_says_official == motion_says_official


def _selfcheck() -> None:
    """Each branch, and the two ways this could be dangerous rather than wrong."""
    fps, W = 25.0, 68.0
    rng = np.random.default_rng(0)

    def track(n, fn):
        xy = np.array([fn(i) for i in range(n)], float)
        return xy, np.ones(len(xy) - 1, bool)

    # A coach: parked by the touchline for 20 minutes, shuffling a couple of m.
    xy, real = track(30000, lambda i: (52.0 + rng.normal(0, 1.5), 2.0 + rng.normal(0, 0.8)))
    r = classify(xy, real, fps, W)
    assert r.role == "staff", r
    assert r.confidence > 0.5, r

    # A linesman: one touchline, half the pitch, all half.
    xy, real = track(30000, lambda i: (26.0 + 25.0 * np.sin(i / 900.0), 3.0 + rng.normal(0, 1.2)))
    r = classify(xy, real, fps, W)
    assert r.role == "linesman", r

    # A BOX-TO-BOX MIDFIELDER MUST NOT BECOME A REFEREE, and must not become
    # staff either. This is the false positive that would delete a real player
    # if role ever gated exclusion, so it is asserted hardest.
    xy, real = track(30000, lambda i: (52.5 + 35.0 * np.sin(i / 700.0),
                                       34.0 + 22.0 * np.cos(i / 500.0)))
    r = classify(xy, real, fps, W)
    assert r.role == "uncertain", r

    # A WINGER hugs a touchline too. He must not read as a linesman, because he
    # crosses into the middle - which is what LINESMAN_Y_SPAN_M is for.
    xy, real = track(30000, lambda i: (52.5 + 40.0 * np.sin(i / 700.0),
                                       12.0 + 16.0 * np.cos(i / 400.0)))
    r = classify(xy, real, fps, W)
    assert r.role == "uncertain", r

    # A PLAYER WE LOST is not a coach. Same still position, but observed for
    # 20 s of a 20-minute track - span would call this staff, observed time
    # must not.
    xy = np.array([(52.0, 2.0)] * 30000, float)
    real = np.zeros(len(xy) - 1, bool)
    real[:500] = True
    assert classify(xy, real, fps, W).role == "uncertain", "span was used, not observed time"

    # Degenerate inputs never raise.
    assert classify(np.zeros((0, 2)), np.zeros(0, bool), fps, W).role == "uncertain"
    assert classify(np.zeros((2, 2)), np.zeros(1, bool), fps, W).role == "uncertain"

    # cross_check reports, and refuses to invent an opinion.
    staff = Role("staff", 0.9, "")
    unc = Role("uncertain", 0.0, "")
    assert cross_check(staff, "official") is True
    assert cross_check(staff, "home") is False
    assert cross_check(unc, "official") is False
    assert cross_check(unc, "home") is None

    print("roles.py ok")


if __name__ == "__main__":
    _selfcheck()
