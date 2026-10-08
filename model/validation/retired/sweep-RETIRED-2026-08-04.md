# Accuracy validation — sweep

**Overall gate: FAIL** (n = 88 player-matches, held-out matches: soccer_top__test__videos__D_20220220_1_1680_1710, soccer_top__test__videos__D_20220220_1_1710_1740, soccer_top__test__videos__D_20220220_1_1740_1770, soccer_top__test__videos__D_20220220_1_1770_1800)

Reference: **TeamTrack soccer_top (image-space bbox GT, MIT)**, own error stated at **±0%**. Every figure below is agreement between two imperfect instruments, not error against truth — the limits of agreement column is the honest read.

## Gating metrics (held-out)

| Metric | Target | n | MAE | Bias | 95% limits of agreement | Verdict |
|---|---|---|---|---|---|---|
| distance_m | <=5% | 88 | 410.8% | +120.5% | +62.9% to +178.1% | **FAIL** |
| minutes | <=5% | 88 | 0.0% | +0.0% | +0.0% to +0.0% | **PASS** |
| ID switches (mean/player) | <2 per player | 88 | 0.0 | +0.0 | +0.0 to +0.0 | **PASS** |


## Report-only metrics (held-out)

| Metric | Target | n | MAE | Bias | 95% limits of agreement | Verdict |
|---|---|---|---|---|---|---|
| hsr_m | <=15% | 42 | 18709.0% | +184.0% | +152.0% to +215.9% | **MISSED** |
| sprints | <=20% | 12 | 75.0% | -150.0% | -327.3% to +27.3% | **MISSED** |
| peak_speed_ms | <=10% | 88 | 107.2% | +59.9% | -6.7% to +126.5% | **MISSED** |


### Bands

- **<=5% PASS** — ship the claim.
- **5-10% MARGINAL** — one time-boxed iteration on smoothing, calibration and outlier rejection, then re-decide. Not an automatic pass and not an automatic pivot.
- **>10% FAIL** — the single-camera premise is wrong at 1080p.

## Tuning split — FITTED, NOT A CLAIM

These matches were available for smoothing and threshold tuning. Any number here is fitted to the data it is measured on. It is shown only so the gap against the held-out figures is visible; quoting it would be a measurement error, not a rounding one.

Matches: soccer_top__train__videos__D_20220220_1_0000_0030, soccer_top__train__videos__D_20220220_1_0030_0060

| Metric | Target | n | MAE | Bias | 95% limits of agreement | Verdict |
|---|---|---|---|---|---|---|
| distance_m | <=5% | 44 | 188.1% | +94.4% | +63.2% to +125.6% | **FAIL** |
| minutes | <=5% | 44 | 0.0% | +0.0% | +0.0% to +0.0% | **PASS** |
| ID switches (mean/player) | <2 per player | 44 | 0.0 | +0.0 | +0.0 to +0.0 | **PASS** |

| Metric | Target | n | MAE | Bias | 95% limits of agreement | Verdict |
|---|---|---|---|---|---|---|
| hsr_m | <=15% | 41 | 8557.9% | +188.7% | +162.0% to +215.4% | **MISSED** |
| sprints | <=20% | 5 | 100.0% | -200.0% | -200.0% to -200.0% | **MISSED** |
| peak_speed_ms | <=10% | 44 | 19.1% | +14.5% | -28.3% to +57.3% | **MISSED** |


## Caveats

- Reference aggregates use the **raw** estimator, not the product's Kalman, so this is an ABSOLUTE error against a stated-bandwidth path length rather than agreement between two filtered copies. Sum-of-deltas path length grows with sample rate; the figure is only meaningful next to the rate it was computed at.
- **Measured side is synthetic** (noise=11.0): ground truth degraded with a known error budget. This is a sensitivity study of the stats layer, NOT a measurement of the tracking pipeline, which has never been run on real footage.
- Player pairing is by track id (or an explicit --map). Identity is unbuilt in the pipeline, so a real run needs the mapping supplied.
- n >= 60 player-matches is required before publishing; this report has 88.
- Peak speed is an extreme-value statistic and is report-only by design. A 10% agreement target against a reference whose own peak-speed error is 5-10% is inside the instrument's noise and may not be evaluable at all.
