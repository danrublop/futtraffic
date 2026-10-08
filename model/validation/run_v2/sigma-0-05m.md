# Accuracy validation — sigma 0.05m

**Overall gate: PASS** (n = 22 player-matches, held-out matches: 118575_1st)

> **UNDERPOWERED: n=22 player-matches, below the n>=60 the design doc requires before any figure is published.**

Reference: **SoccerTrack v2 GSR (per-frame pitch metres, CC BY 4.0)**, own error stated at **±1.5%**. Every figure below is agreement between two imperfect instruments, not error against truth — the limits of agreement column is the honest read.

## Gating metrics (held-out)

| Metric | Target | n | MAE | Bias | 95% limits of agreement | Verdict |
|---|---|---|---|---|---|---|
| distance_m | <=5% | 22 | 3.8% | +3.7% | -2.0% to +9.4% | **PASS** |
| minutes | <=5% | 22 | 0.0% | +0.0% | +0.0% to +0.0% | **PASS** |
| ID switches (mean/player) | <2 per player | 22 | 0.0 | +0.0 | +0.0 to +0.0 | **PASS** |


## Report-only metrics (held-out)

| Metric | Target | n | MAE | Bias | 95% limits of agreement | Verdict |
|---|---|---|---|---|---|---|
| hsr_m | <=15% | 22 | 20.4% | -22.7% | -28.6% to -16.9% | **MISSED** |
| sprints | <=20% | 21 | 63.8% | -98.7% | -179.1% to -18.3% | **MISSED** |
| peak_speed_ms | <=10% | 22 | 1.4% | +1.4% | -2.4% to +5.1% | **MET** |


### Bands

- **<=5% PASS** — ship the claim.
- **5-10% MARGINAL** — one time-boxed iteration on smoothing, calibration and outlier rejection, then re-decide. Not an automatic pass and not an automatic pivot.
- **>10% FAIL** — the single-camera premise is wrong at 1080p.

## Tuning split — FITTED, NOT A CLAIM

These matches were available for smoothing and threshold tuning. Any number here is fitted to the data it is measured on. It is shown only so the gap against the held-out figures is visible; quoting it would be a measurement error, not a rounding one.

Matches: 117092_1st, 117092_2nd, 117093_1st

| Metric | Target | n | MAE | Bias | 95% limits of agreement | Verdict |
|---|---|---|---|---|---|---|
| distance_m | <=5% | 79 | 3.0% | +3.0% | -0.9% to +6.9% | **PASS** |
| minutes | <=5% | 79 | 0.0% | +0.0% | +0.0% to +0.0% | **PASS** |
| ID switches (mean/player) | <2 per player | 79 | 0.0 | +0.0 | +0.0 to +0.0 | **PASS** |

| Metric | Target | n | MAE | Bias | 95% limits of agreement | Verdict |
|---|---|---|---|---|---|---|
| hsr_m | <=15% | 79 | 21.0% | -23.5% | -30.5% to -16.4% | **MISSED** |
| sprints | <=20% | 74 | 64.1% | -99.8% | -182.0% to -17.7% | **MISSED** |
| peak_speed_ms | <=10% | 79 | 1.0% | +0.9% | -2.3% to +4.0% | **MET** |


## Caveats

- **Measured side is synthetic** (noise=0.05,drop=0.0,swap=0.0): ground truth degraded with a known error budget. This is a sensitivity study of the stats layer, NOT a measurement of the tracking pipeline, which has never been run on real footage.
- SoccerTrack v2 is 4K panoramic footage; the product camera is 1080p perspective. The domain gap is real and unquantified.
- Both sides run through the same metric estimator (stats.py smoothing, speed clamp, gap rules), so the estimator's own bias cancels. These figures are a lower bound on error. The published claim needs GPS-vest reference data, where nothing cancels.
- Player pairing is by track id (or an explicit --map). Identity is unbuilt in the pipeline, so a real run needs the mapping supplied.
- n >= 60 player-matches is required before publishing; this report has 22. **Not met.**
- Peak speed is an extreme-value statistic and is report-only by design. A 10% agreement target against a reference whose own peak-speed error is 5-10% is inside the instrument's noise and may not be evaluable at all.
