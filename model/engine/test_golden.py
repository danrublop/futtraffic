"""Golden-file regression test on real tracks.

The whole product is a number a user cannot independently check. A model
version bump, an OpenCV upgrade, a calibration tweak or a filter retune shifts
distances by several percent with no exception and no crash - the pipeline keeps
producing confident, plausible, wrong numbers. That is the failure this guards.

Run: .venv/bin/python test_golden.py          compare against golden.json
     .venv/bin/python test_golden.py --bless  rewrite golden.json (review the diff!)

Input is a fixed slice of real SoccerTrack v2 tracks (CC BY 4.0), not synthetic
data, so it exercises real player motion including the stop-start and direction
changes that synthetic sinusoids never produce.
"""
import json
import sys
from pathlib import Path

import numpy as np

import stats
from pitch import Calibration, Pitch

from paths import REPO, VALIDATION

HERE = REPO
SOURCE = VALIDATION / "soccertrack/118575_1st.tracks.json.gz"
GOLDEN = VALIDATION / "golden.json"

# Any single metric drifting more than this fails. 0.5% is far tighter than our
# 5% accuracy gate on purpose: this test is not measuring accuracy, it is
# measuring CHANGE. A legitimate improvement should be blessed deliberately,
# never absorbed silently.
TOL_PCT = 0.5

N_TRACKS = 6
N_SAMPLES = 2000
STRIDE = 5


def compute() -> dict:
    from validate import load_tracks

    if not SOURCE.exists():
        print(f"SKIP: {SOURCE} not present (run the SoccerTrack fetch first)")
        sys.exit(0)

    tracks, fps = load_tracks(str(SOURCE))
    cal = Calibration(pitch=Pitch(105.0, 68.0), H=np.eye(3))
# Frame IDs are NOT renumbered by striding - a strided list still carries the
# original f values - so the ORIGINAL fps is the correct one. Dividing it by
# STRIDE stretched the clock 5x: 33.3 reported minutes for a 6.7 minute slice,
# 1.8 m/s peak speed for a footballer, and every physics gate (9.5 m/s teleport
# clamp, 5 s gap window, sprint thresholds) relaxed by the same factor.
    fps_eff = fps

    out = {}
    for tid in sorted(tracks)[:N_TRACKS]:
        tr = tracks[tid][::STRIDE][:N_SAMPLES]
        if len(tr) < 100:
            continue
        for method in ("kalman", "boxcar"):
            ps = stats.player_stats(tr, cal, fps_eff, method)
            if ps is None:
                continue
            out[f"{tid}/{method}"] = {
                "distance_m": round(ps.distance_m, 3),
                "top_speed_ms": round(ps.top_speed_ms, 4),
                "sprints": ps.sprints,
                "minutes": round(ps.minutes, 4),
                "clamped_frac": round(ps.clamped_frac, 5),
            }
    if not out:
        print("SKIP: no usable tracks in the fixture")
        sys.exit(0)
    return out


def main() -> int:
    got = compute()

    if "--bless" in sys.argv or not GOLDEN.exists():
        why = "blessed" if GOLDEN.exists() else "created (no golden file yet)"
        GOLDEN.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n")
        print(f"golden.json {why}: {len(got)} entries. Review the diff before committing.")
        return 0

    want = json.loads(GOLDEN.read_text())
    fails = []

    for key in sorted(set(want) | set(got)):
        if key not in got:
            fails.append(f"{key}: MISSING from output (was in golden)")
            continue
        if key not in want:
            fails.append(f"{key}: NEW in output (not in golden)")
            continue
        for metric, exp in want[key].items():
            act = got[key].get(metric)
            if act is None:
                fails.append(f"{key}.{metric}: metric disappeared")
            elif isinstance(exp, int) and isinstance(act, int):
                if exp != act:
                    fails.append(f"{key}.{metric}: {exp} -> {act}")
            elif exp == 0:
                if abs(act) > 1e-9:
                    fails.append(f"{key}.{metric}: 0 -> {act}")
            else:
                drift = abs(act - exp) / abs(exp) * 100
                if drift > TOL_PCT:
                    fails.append(f"{key}.{metric}: {exp} -> {act}  ({drift:+.2f}%)")

    if fails:
        print(f"GOLDEN REGRESSION - {len(fails)} metric(s) moved beyond {TOL_PCT}%:\n")
        for f in fails[:25]:
            print(f"  {f}")
        if len(fails) > 25:
            print(f"  ... and {len(fails) - 25} more")
        print("\nIf this change is intentional and correct, re-bless:")
        print("  .venv/bin/python test_golden.py --bless")
        return 1

    print(f"test_golden.py ok ({len(want)} entries within {TOL_PCT}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
