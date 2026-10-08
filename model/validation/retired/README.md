# Retired validation reports

Reports kept as evidence, **not as results**. Nothing in this directory may be
quoted. Each one is here because it was found to be measuring something other
than what its filename implied, and deleting it would have destroyed the record
of a mistake that had already been quoted downstream.

## `sweep-RETIRED-2026-08-04` (was `tune_runs/sweep.md` / `.json`)

Reported **distance MAE 410.8%, bias +120.5%, gate FAIL**, and was read — by the
README, by the web UI, and by anyone who opened the directory — as the measured
accuracy of the shipped product. It is not.

Reproduced bit-exactly, including `hsr_m 18709.0/+184.0` and
`sprints 75.0/-150.0`, at:

    ACCEL_Q = 16, SIGMA_PX = 0.5, degrade noise = 11 px (~0.40 m)

That is **32x the shipped q/sigma ratio**, at the highest injected noise in the
list, and it is the **worst-scoring cell in `tune.py`'s own default grid**.
Re-running `tune.py`'s selection verbatim (147 `validate.main` invocations)
picks `q=1, sigma_px=0.5`, worst-case fitted bias 6.6%. So this file cannot have
come from the default sweep; it came from a restricted or manual invocation, and
**nothing in it recorded that**, because `build_report()` did not write the
configuration into the report.

Two things were fixed so this cannot recur:

1. `build_report()` now stamps `accel_q`, `sigma_px`, `distance_window_s`,
   `gt_estimator`, `degrade` and `calibration` into every report, and
   `to_markdown()` prints them in a table directly under the headline. A report
   that does not say what produced it now cannot be written.
2. MAE and bias share the Bland-Altman denominator. The two were computed with
   incompatible normalisers (MAE divided by ground truth and was unbounded;
   bias divided by the pair mean and saturated at +-200%), which is the entire
   explanation for `sprints bias -150%` — a value that is arithmetically
   impossible for a non-negative count and was pure normaliser mismatch.

The superseding measurement is `model/validation/tune_runs/holdout-windowed.md`.
Full analysis: `AUDIT_MODEL.md`.
