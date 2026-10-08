"""Sweep (ACCEL_Q, SIGMA_PX) on TeamTrack, choose on one clip, report on others.

Everything here goes through `validate.main` rather than calling `stats`
directly, so the held-out / FITTED separation is enforced by the harness instead
of by my discipline. The selection is made on the FITTED split and the number
quoted afterwards comes from the held-out split; those are different clips of
the match and the script never mixes them.

Reference is `--gt-estimator raw`: unfiltered sum-of-deltas path length through
the real homography. That is bandwidth-dependent (see `--bandwidth`), and it is
the only thing an ABSOLUTE bias can be measured against - scoring the reference
with the same Kalman cancels exactly the bias under investigation.

Run: .venv/bin/python model/validation/tune.py [--noise 0,5.9,11.8] [--bandwidth]
"""
import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "engine"))
import validate                       # noqa: E402
import stats                          # noqa: E402

TUNE = ["soccer_top__train__videos__D_20220220_1_0000_0030",
        "soccer_top__train__videos__D_20220220_1_0030_0060"]
HELD = ["soccer_top__test__videos__D_20220220_1_1680_1710",
        "soccer_top__test__videos__D_20220220_1_1710_1740",
        "soccer_top__test__videos__D_20220220_1_1740_1770",
        "soccer_top__test__videos__D_20220220_1_1770_1800"]
OUT = HERE / "tune_runs"


def run(q: float, sigma_px: float, noise: float, clips: list[str]) -> dict:
    """One validate.py invocation -> its report dict."""
    argv = ["--gt"] + [str(HERE / f"{c}.tracks.json.gz") for c in clips] + [
        "--calib", str(HERE),
        "--degrade", f"noise={noise}",
        "--gt-estimator", "raw",
        "--held-out", ",".join(HELD),
        "--reference-error", "0.0",
        "--source", "TeamTrack soccer_top (image-space bbox GT, MIT)",
        "--out", str(OUT), "--label", "sweep",
        "--q", str(q), "--sigma-px", str(sigma_px),
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        validate.main(argv)
    with open(OUT / "sweep.json") as f:
        return json.load(f)


def bias(rep: dict, split: str) -> tuple[float, float, int]:
    d = (rep.get(split) or {}).get("distance_m")
    if not d:
        return float("nan"), float("nan"), 0
    return d["bias_pct"], d["mae_pct"], d["n"]


def bandwidth_report(clip: str) -> None:
    """How much the REFERENCE itself moves with sample rate.

    A total-distance claim is not falsifiable without this: sum-of-deltas path
    length is not a fixed quantity, it is a function of the rate you sample the
    trajectory at, and the reference and the product must be defined at the same
    one.
    """
    from pitch import Calibration
    tracks, fps = validate.load_tracks(str(HERE / f"{clip}.tracks.json.gz"))
    cal = Calibration.load(str(HERE / f"calib_{clip}.json"))
    print(f"\nreference bandwidth sensitivity ({clip}, raw sum-of-deltas):")
    base = None
    for stride in (1, 2, 3, 5, 10, 25, 60):
        total = 0.0
        for tr in tracks.values():
            pts = sorted(tr, key=lambda d: d["f"])[::stride]
            xy = cal.to_pitch(np.array([[p["x"], p["y"]] for p in pts], float))
            t = np.array([p["f"] / fps for p in pts], float)
            v = np.minimum(np.linalg.norm(np.diff(xy, axis=0), axis=1)
                           / np.diff(t), stats.MAX_SPEED)
            total += float(np.sum(v * np.diff(t)))
        base = base or total
        print(f"  {fps/stride:6.2f} Hz -> {total:8.0f} m over 22 players "
              f"({total/base*100:5.1f}% of the 29.97 Hz figure)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noise", default="0,5.5,11",
                    help="comma-separated PIXEL noise sigmas to inject")
    ap.add_argument("--q", default="0.25,0.5,1,2,4,8,16")
    ap.add_argument("--sigma-px", default="0.5,1,2,4,6,12,24")
    ap.add_argument("--bandwidth", action="store_true")
    a = ap.parse_args()

    if a.bandwidth:
        bandwidth_report(HELD[0])
        return 0

    noises = [float(x) for x in a.noise.split(",")]
    qs = [float(x) for x in a.q.split(",")]
    sps = [float(x) for x in a.sigma_px.split(",")]

    print(f"tune clips  {', '.join(TUNE)}")
    print(f"held out    {', '.join(HELD)}")
    print(f"reference   raw sum-of-deltas path length at 29.97 Hz through the "
          f"per-clip homography\n")

    hdr = "  ".join(f"n={n:g}px" for n in noises)
    print(f"{'q':>6} {'sigma_px':>9} | FITTED bias%% ({hdr})   | worst |")
    rows = []
    for q in qs:
        for sp in sps:
            # Sweep the TUNE clips only. Loading the held-out clips here would
            # be 3x the wall clock and, worse, would put the number I am
            # selecting on in the same process as the number I report.
            fit = [bias(run(q, sp, n, TUNE), "fitted_train_split")[0]
                   for n in noises]
            worst = max(abs(b) for b in fit)
            rows.append((worst, q, sp, fit))
            print(f"{q:6g} {sp:9g} | " + "  ".join(f"{b:+7.1f}" for b in fit)
                  + f"   | {worst:5.1f} |")

    rows.sort()
    print("\nbest 5 by worst-case |bias| on the FITTED split:")
    for worst, q, sp, fit in rows[:5]:
        print(f"  q={q:g} sigma_px={sp:g}: " + ", ".join(
            f"{n:g}px -> {b:+.1f}%" for n, b in zip(noises, fit)))

    worst, q, sp, _ = rows[0]
    print(f"\n=== HELD-OUT, q={q:g} sigma_px={sp:g} ===")
    for n in noises:
        rep = run(q, sp, n, TUNE + HELD)
        b, mae, k = bias(rep, "headline")
        fb, fmae, fk = bias(rep, "fitted_train_split")
        print(f"  noise {n:5g} px: held-out bias {b:+6.1f}%  MAE {mae:5.1f}%  "
              f"n={k}   (fitted {fb:+.1f}% / {fmae:.1f}%, n={fk})")
    return 0


def _selfcheck() -> None:
    """The harness plumbing, not the tuning: one run must produce both splits."""
    if not (HERE / f"{TUNE[0]}.tracks.json.gz").exists():
        print("SKIP: no TeamTrack tracks - run model/validation/teamtrack_tracks.py")
        return
    rep = run(1.0, 1.0, 0.0, TUNE[:1] + HELD[:1])
    assert rep["headline"] is not None, "held-out split produced no headline"
    assert rep["fitted_train_split"] is not None, "tuning split vanished"
    assert rep["held_out_matches"] == HELD[:1], rep["held_out_matches"]
    # Reference and measurement must not be the same estimator, or the bias
    # under investigation cancels and every number below is a tautological 0.
    assert any("raw" in c for c in rep["caveats"]), rep["caveats"]
    print("tune.py ok")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
