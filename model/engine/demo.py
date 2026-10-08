"""Run the stats pipeline end to end on synthetic tracks.

This fabricates TRACKS and runs them through the real stats pipeline. It does
not fabricate stat values - every number in the output is computed by stats.py
from the positions, exactly as it would be for a real game. The output is
labelled synthetic so nobody mistakes it for a real result.

It used to double as the no-backend demo, writing web/match.json for the
zero-build pages to read. Those pages are gone; the UI now reads the API only,
so the output lands in data/ and this is a pipeline self-check. `synth_tracks`
is also the reference for the frame-index convention (pipeline-2026-09 (not published) §5).

Replace with: track.py on real footage -> tracks.json -> stats.build_match.
"""
import json

import numpy as np

from paths import DATA, REPO
from pitch import Calibration, Pitch
from stats import build_match

FPS = 25.0
MINUTES = 90
# A 4-4-2 per side in pitch metres, home attacking right.
FORMATION = [(8, 34), (22, 12), (20, 27), (20, 41), (22, 56),
             (45, 14), (43, 28), (43, 40), (45, 54), (68, 26), (68, 42)]


STEP = 5                 # sample every 5th frame -> 5 Hz
# Real 90-minute targets: outfielders cover ~9-11 km, keepers ~4-5 km, and peak
# speeds reach 25-32 km/h. An earlier version used a slow sinusoid that produced
# 0.5 km and a 1 km/h top speed - which looked like a pipeline bug in the demo
# rather than what it was, unrealistic input.
TARGET_KM = {"keeper": 4.5, "outfield": 10.0}


def default_squads(pitch: Pitch) -> dict:
    """The built-in 4-4-2, as slots: (x_m, y_m, jersey, is_keeper) per side."""
    return {
        team: [
            (bx if team == "home" else pitch.length - bx,
             by if team == "home" else pitch.width - by,
             i + 1, i == 0)
            for i, (bx, by) in enumerate(FORMATION)
        ]
        for team in ("home", "away")
    }


def synth_tracks(path: str, pitch: Pitch, seed: int = 0, squads: dict | None = None) -> None:
    """Ornstein-Uhlenbeck wander around a formation slot, plus sprint bursts.

    A random walk pulled toward a home position reproduces the two things that
    actually matter for these stats: players stay in their zone over 90 minutes,
    but the instantaneous speed distribution has a long tail. A sinusoid has
    neither - it never sprints and never rests.

    `squads` swaps the built-in 4-4-2 for real slots — a feed importer can pass
    the team sheets of a fixture. The MOTION MODEL stays here and is the
    only one: its constants are tuned so the output lands on real distances and
    real speed tails, and a second copy of it somewhere else drifts off those
    within a week. (It did: a first version in a former importer reported 395 km a
    team and 7380 sprints.)
    """
    rng = np.random.default_rng(seed)
    dt = STEP / FPS
    n_samples = int(MINUTES * 60 * FPS / STEP)
    dets = []
    tid = 0
    for team in ("home", "away"):
        for cx, cy, jersey, keeper in (squads or default_squads(pitch))[team]:
            tid += 1
            role = "keeper" if keeper else "outfield"

            # Mean speed that lands on the target distance over the match.
            mean_ms = TARGET_KM[role] * 1000.0 / (MINUTES * 60.0)
            pull = 0.06 if keeper else 0.02      # how hard the slot pulls back
            spread = 4.0 if keeper else 16.0     # how far they roam

            pos = np.array([cx, cy], float)
            vel = np.zeros(2)
            for k in range(n_samples):
                # Sprint bursts: rare, short, and the only source of the tail.
                burst = 4.0 if (not keeper and rng.random() < 0.004) else 1.0
                accel = rng.normal(0, mean_ms * 2.2 * burst, 2)
                home_pull = -pull * (pos - np.array([cx, cy])) / spread * 100.0
                vel = vel * 0.90 + (accel + home_pull) * dt
                sp = np.linalg.norm(vel)
                cap = 8.8 if not keeper else 5.0     # m/s, ~32 km/h
                if sp > cap:
                    vel *= cap / sp
                pos = pos + vel * dt
                pos[0] = np.clip(pos[0], 0.5, pitch.length - 0.5)
                pos[1] = np.clip(pos[1], 0.5, pitch.width - 0.5)
                dets.append({
                    "f": k * STEP, "id": tid, "team": team, "jersey": jersey,
                    # 6 cm of detector jitter. The Kalman should reject this;
                    # if a change makes demo distances jump, it stopped doing so.
                    "x": float(pos[0] + rng.normal(0, 0.06)),
                    "y": float(pos[1] + rng.normal(0, 0.06)),
                })
    with open(path, "w") as f:
        json.dump({"fps": FPS, "detections": dets}, f)


def main() -> None:
    pitch = Pitch.from_yards(118, 72)
    # Identity homography: the synthetic tracks are already in pitch metres.
    # Repo-relative, not CWD-relative: running demo.py from anywhere but the
    # repo root used to scatter these three files wherever the shell happened
    # to be, and the next run would read a stale one from the root.
    cal_p = str(REPO / "calibration.json")
    tr_p = str(REPO / "tracks.json")
    out = DATA / "demo"
    out.mkdir(parents=True, exist_ok=True)
    Calibration(pitch=pitch, H=np.eye(3)).save(cal_p)
    synth_tracks(tr_p, pitch)
    m = build_match(tr_p, cal_p, str(out / "match.json"), meta={
        "competition": "SYNTHETIC DEMO - not a real match",
        "home_name": "Home", "away_name": "Away",
        "score": "–", "status": "Demo data",
        # Generated positions: stats.build_match refuses every justification
        # on this flag, and the frontend's synthetic-data note keys on it.
        "synthetic_tracking": True, "demo": True,
    })
    print(f"{len(m['players'])} players, "
          f"home {m['teams']['home']['distance_m'] / 1000:.1f} km, "
          f"territory {m['territory']}")


if __name__ == "__main__":
    main()
