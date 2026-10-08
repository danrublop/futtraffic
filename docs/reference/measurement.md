# Measurement

**Summary.** `tracks.json` + `calibration.json` in; `match.json` out: per-player physical stats,
per-team totals and shape, and a confidence envelope (`fields`) on every published number.

Upstream is [the identity reference](identity.md) (`tracks.json`). Accuracy lives in the ledgers
(`ledger-output.md`, [`ledger-claims.md`](../ledgers/ledger-claims.md))
and the experiments under [Evidence](#evidence); history and reasoning on
the evidence page. `engine-2026-09 (not published)` and
`pipeline-2026-09 (not published)` point here. The page this replaces is archived as
`metrics-2026-08.md`.

<a id="entry-point"></a>
## Entry point

`model/engine/stats.py :: build_match(tracks_path: str, calib_path: str, out_path: str, meta: dict | None = None) -> dict`,
called from `app/server/jobs.py :: _pipeline` after `_derive_tracks`, with
`meta={"job_id", "source"}` and nothing else: every setting below reaches the run as a module
constant or a signature default ([Config](#config)). `model/engine/demo.py` calls it
with `meta["synthetic_tracking"] = True`.
Serving goes through `stats.py :: resolved_match(path)` (the `app/server/app.py` match route,
`app/server/agent.py`), which re-resolves every envelope and nulls `sprints` ([Tiers](#tiers)).

### Order

1. `_open` (plain or `.gz`); `fps` refused unless a positive finite number
   ([Errors](#errors-and-refusals)); `cal = calibrators.load(calib_path)`.
2. Detections grouped by `id`; `player_stats(track, cal, fps)` per track (method `kalman`);
   rows that return `None` or have `minutes < MIN_MINUTES` are dropped. `heatmap_points` per kept row.
3. Rows with `team == "official"` are removed from `players` and `heatmaps` (counted in
   `meta.officials_excluded`). A non-finite distance, peak or mean position refuses the match.
4. `frames.team_frames(by_id, cal, fps)`: `home`/`away` tracks at or above `MIN_MINUTES`,
   resampled onto one grid of `frames.py :: GRID_DT` with a validity mask (`MAX_GAP_S`).
   Re-runs `speed_series` per track.
5. `repair_banks`: keepers reassigned by position; banks renamed so the lower-x bank is `home`.
   Player rows are repointed to the repaired side.
6. `attacking_directions`: one direction per match, from keeper separation.
7. Per side: `shape_series`, `formation_windows`, `structure_stress`, `series_coverage`;
   team totals over the counted rows ([team totals](#team-totals)); `team_shape` (median of the series).
8. `relative_shape` on instants where both sides have a sample.
9. Envelopes (`metric`), `unjustify` when `meta.synthetic_tracking`, `quality.check`
   (advisory), the non-finite scan, `json.dump(..., allow_nan=False)` to `out_path`.

<a id="per-player"></a>
### Per player

`speed_series(track, cal, fps, method)`: feet `(x, y)` -> `cal.to_pitch` -> pitch metres;
`t = f / fps`; `real` masks each step whose `dt` exceeds the track's own cadence rule
(`max(3 x median step, 2 / fps)`, in the function). `kalman_ground`: constant-velocity filter on
the ground plane; measurement covariance `R = SIGMA_PX^2 * J J^T` with `J` from `cal.jacobian`
(else `homography_jacobian`), so noise is anisotropic and grows with range; process noise
`ACCEL_Q`; velocity seeded from the opening samples, not from rest; the **reported** speed is
clamped at `MAX_SPEED`, the filter state is not.

| field | formula |
|---|---|
| `distance_km` | `_windowed` on the filtered positions: chord sum over windows of `DISTANCE_WINDOW_S`, averaged over every start offset; a window containing an unobserved step is dropped whole. **Not** an integral of speed, so `MAX_SPEED` does not bound it. Null above the [ceiling](#nulls). |
| `observed_frac` | share of `_windowed`'s windows, over all offsets, that survived the gap rule |
| `top_speed_kmh` | `max(v[real]) * 3.6` over the clamped filtered speed, observed steps only; null where the clamp bound (`_peak_kmh`) |
| `clamped_frac` | `mean(v[real] >= MAX_SPEED - 1e-9)`: a share of **steps**, not of fragments |
| `sprints` | computed (`_count_sprints`: runs at or above `SPRINT_SPEED` lasting `SPRINT_MIN_S`, observed steps only) and **not published**: always null |
| `minutes` | `(t[-1] - t[0]) / 60`: track span, gaps included |
| `observed_minutes` | `sum(dt[real]) / 60` |
| `avg_position` | mean of the filtered positions, absolute pitch frame |
| `role`, `role_confidence`, `role_agrees` | `roles.py :: classify` on the same positions and mask (`_role_fields`); never writes `team` |
| `team`, `jersey` | the track's first detection, then repointed by `repair_banks` |

`heatmap_points` uses the **unfiltered** `cal.to_pitch` positions, drops points more than
`HEATMAP_MARGIN_M` off the pitch or non-finite, then thins to `max_points` by even **index**, not
time. The heat map and the distance number come from two different position series.

<a id="team-totals"></a>
### Team totals

Per side, over the players-table rows of that side:

- `distance_m`: sum of `distance_m` over the **counted** rows: rows refused by the
  [ceiling](#nulls) are out, and of each `duplicate_groups` group only the longest-observed track
  counts. `distance_coverage` is the share of the side's detections the counted rows hold.
- `top_speed_kmh`: `_peak_kmh` over every row of the side; null if the clamp bound on any of them.
- `sprints`: null.
- `shape`: `team_shape`, the median of `shape_series` (`width_m`, `depth_m`,
  `defensive_line_m` = median `rear_line_m`, `compactness_m2` = median hull area,
  `stretch_index_m`). Row formulas: the `shape_series` docstring.

`territory(tracks, cal)` bins the players-table population with a side assigned (officials,
unassigned rows and sub-`MIN_MINUTES` debris out) into thirds of the calibration's **x axis**.
The keys say defensive/middle/attacking; they are left/centre/right of the pitch, the same for
both sides, with no attacking direction.

Formation and stress: the docstrings of `formation_windows`, `formation_summary`,
`structure_stress`, `stress_profile`, `team_lines`.

## Inputs

- `tracks.json` ([identity contract](identity.md#contract)): `fps`; `detections[]` with `f`
  (original frame index), `id`, `team` (`home`/`away`/`official`/other), feet `x, y` in **image
  pixels of the detector's canvas**, optional `jersey`.
- `calibration.json`, loaded by `model/engine/calibrators.py :: load`
  ([calibration reference](calibration.md#load-gate)). Pitch frame: metres, origin at a corner,
  `x` along the length, `y` across ([coordinates](calibration.md#coordinates),
  `model/engine/pitch.py :: Pitch`).
- `meta` (dict, merged into `match.json.meta`).
- `model/validation/api-golden.json`, read by `accepted_measurements`; absent or unreadable, every
  field reads `not_evaluated`.

<a id="contract"></a>
## Outputs

`match.json`, written in place, never with NaN or infinity. Coordinates are pitch metres in the
calibration's frame unless stated.

| key | content |
|---|---|
| `meta` | caller's `meta` plus `fps`, `pitch {length, width}` (m), `calibration_rms_m` (m), `officials_excluded` (count), `quality` (`quality.py :: check`) |
| `players[]` | sorted by distance, descending: `id`, `team`, `jersey`, `minutes` (min, span), `distance_km` (km or null), `top_speed_kmh` (km/h or null), `sprints` (always null), `avg_position [x, y]` (m, absolute frame), `clamped_frac` (fraction of observed steps), `observed_frac` (fraction of distance windows kept), `observed_minutes` (min), `role`, `role_confidence`, `role_agrees` |
| `teams[side]` | `distance_m` (m), `distance_coverage` (fraction of the side's detections), `top_speed_kmh` (km/h or null), `sprints` (always null), `shape {width_m, depth_m, defensive_line_m, compactness_m2 (m^2), stretch_index_m}` (m, medians; x from own goal) |
| `territory` | `{defensive_third, middle_third, attacking_third}` (%): left/centre/right of pitch x, pooled over both sides |
| `heatmaps` | `{str track_id: [[x, y], ...]}` (m), thinned by index; officials removed |
| `shape_series[side]` | `attacking_right`, `sample_dt_s` (s), `coverage` (`series_coverage`), `formation` (`formation_summary`), `window_fit_unstable` (per-window labels), `samples` (strided rows of `shape_series`; x from own goal, `t` in s from the first grid sample) |
| `structure_stress[side]` | `gini`, `profile` (`stress_profile`) and `intervals[]` (`t0`, `t1`, `duration_s` (s), `severity`, `peak_z`, `formation`, `cost`, `hull_area_m2` (m^2)) |
| `relative_shape` | `n`, `possible`, `frac`, `a`, `b`, `metrics {k: {<a>_mean, <b>_mean, diff (b - a), sd}}` |
| `fields` | `{dotted path: envelope}` ([Tiers](#tiers)); per-side paths carry the side (`teams.home.shape.width_m`) |

Envelope: `{value, units, n, confidence, provenance, justified_by, caveat_flags}`. `value` is null
for a column envelope (`players.*`); `n` counts the rows that still hold a value. `provenance` is
`measured`, `derived` or `tagged`. Speeds inside the engine are m/s; only `*_kmh` keys are km/h.

<a id="nulls"></a>
### Which values can be null

| value | null when | flag on its envelope |
|---|---|---|
| `players[].top_speed_kmh` | the clamp bound anywhere on the row (`clamped_frac > 0`): the value would be `MAX_SPEED` itself | `measures_clamp` |
| `teams[side].top_speed_kmh` | the clamp bound on any row of the side | `measures_clamp` |
| `players[].sprints`, `teams[side].sprints` | always: withheld per N21; `resolved_match` also nulls them in files written before the refusal | `sprints_withheld` |
| `players[].distance_km` | `distance_m > MAX_SPEED x observed seconds` (`_over_ceiling`): more ground than the clamp speed covers while seen | `distance_over_ceiling`; the row is also left out of its side's total |
| `players[].jersey` | no number passed the identity stage | `jersey_ocr_unvalidated` |
| `players[].role` | `_role_fields` raised (printed, not fatal) | `role_unvalidated` |
| `shape_series[side].formation` | no windows (`formation_summary` returns `{}`) | `estimate_not_measurement` |

A non-finite value is never nulled: it refuses the match ([Errors](#errors-and-refusals)).

<a id="tiers"></a>
### Tiers and caveats

- `METRICS` maps a field path to `(units, provenance, class ceiling, base flags, api id)`.
  `model/engine/events.py :: register` adds its own entries at import.
- `CAVEATS` maps a flag to `(ceiling, reason)`. Only the ceiling is read by code; a ceiling of
  `ok` is a note, not a downgrade.
- `metric()` sets `confidence = min(class ceiling if the api id is in accepted_measurements()
  else not_evaluated, each flag's ceiling)` ranked by `TIER_RANK`: `not_evaluated` < `prov` <
  `rel` < `ok`. `justified_by` is `api://<id>` only when accepted, else null.
- Instance flags set by `build_match` for this match:

| flag | on | set when |
|---|---|---|
| `team_split_implausible` | `players.team` | side ratio over `TEAM_BALANCE_MAX`, a side over `TEAM_SIZE_MAX`, or an empty side |
| `unassigned_side` | `players.team` | a row with neither `home` nor `away` |
| `measures_clamp` | `players.top_speed_kmh`, `teams.<side>.top_speed_kmh` | the clamp bound on a row covered |
| `distance_unbounded` | `players.distance_km`, `derived.per90`, `teams.<side>.distance_m` | the clamp bound on a row covered: distance is read off positions, so the clamp does not bound it |
| `distance_over_ceiling` | `players.distance_km`, `derived.per90` | at least one row refused by `_over_ceiling` |
| `team_distance_partial` | `teams.<side>.distance_m` | always: the total counts players-table rows only ([team totals](#team-totals)) |
| `sprints_withheld` | `players.sprints`, `teams.sprints` | always (base flag) |
| `low_coverage` | shape, samples, stress, `relative_shape` | `series_coverage` below `SHAPE_COVERAGE_MIN` |
| `attacking_direction_conflict`, `attacking_direction_refused` | every x-from-own-goal field | `attacking_directions` overrode one side, or found no keeper either way |

- **Formation ties are shown, not resolved.** `formation_summary` joins every label at the top
  count, in first-seen order, as `"A / B"` in `formation` and lists them in `tied`; the order does
  not depend on `PYTHONHASHSEED`.
- `resolve()` recomputes every envelope against today's golden; an unknown path or flag degrades
  that envelope to `not_evaluated`; `synthetic_tracking` strips every justification (`unjustify`).

<a id="config"></a>
## Config

`build_match` takes no settings. Every value is a module constant or a signature default, read
at call time unless stated.

| what | where | how it reaches the run |
|---|---|---|
| speed estimator | `stats.py :: player_stats` (`method`), `frames.py :: team_frames` (`method`) | signature default, not passed |
| filter | `stats.py :: SIGMA_PX`, `ACCEL_Q`, `MAX_SPEED` | module constants, read at call time |
| distance window | `stats.py :: DISTANCE_WINDOW_S` | module constant, read at call time |
| gap rule | `stats.py :: speed_series` (the `real` expression) | code |
| distance ceiling | `stats.py :: _over_ceiling` (`MAX_SPEED`, `observed_minutes`) | code |
| sprint | `stats.py :: SPRINT_SPEED`, `SPRINT_MIN_S` | module constants (count unpublished) |
| debris floor | `stats.py :: MIN_MINUTES` (also imported by `frames.py`, `validate.py`) | module constant |
| heat map | `stats.py :: HEATMAP_MARGIN_M`; `heatmap_points` (`max_points`) | constant; signature default |
| team grid | `frames.py :: GRID_DT`, `MAX_GAP_S` | module constants |
| shape | `stats.py :: SHAPE_MIN_PLAYERS`, `SHAPE_DUP_M`, `SHAPE_DUP_FRAC`, `SHAPE_COVERAGE_MIN`, `SERIES_EMIT_S` | module constants |
| keeper and direction | `stats.py :: GK_MARGIN_M`, `GK_MAX_DEPTH_M`; `repair_banks` column floor | constants; code |
| formation | `stats.py :: FORMATIONS`, `FORMATION_WINDOW_S`, `FORMATION_HYSTERESIS` | module constants |
| stress | `stats.py :: STRESS_MERGE_GAP_S`; `structure_stress` (`top`, percentile cut); `stress_profile` bands | constant; code |
| side sanity | `stats.py :: TEAM_BALANCE_MAX`, `TEAM_SIZE_MAX` | module constants |
| tiers | `stats.py :: METRICS`, `CAVEATS`, `TIER_RANK`, `NOT_EVALUATED`; `events.py :: EVENT_METRICS`, `EVENT_CAVEATS` | module tables |
| accepted ids | `stats.py :: accepted_measurements` (`model/validation/api-golden.json`, keyed on mtime, size, inode) | file |
| roles | `roles.py :: STAFF_RADIUS_M`, `STAFF_MIN_S`, `LINESMAN_BAND_M`, `LINESMAN_Y_SPAN_M`, `LINESMAN_X_SPAN_M` | module constants |
| unshipped estimators | `stats.py :: BUTTER_ORDER`, `BUTTER_HZ` (`butter`), `SMOOTH_S` (`boxcar`) | used only when `method` is passed |

`model/validation/shipped_config.py` does not resolve this stage; `check.py measure` freezes the
accepted ids in its golden.

<a id="errors-and-refusals"></a>
## Errors and refusals

- **Fatal to the job.** `build_match` is not wrapped: its refusals and any read error propagate
  to `app/server/jobs.py :: run`, which fails the job.
  - `ValueError` when `fps` is missing, boolean, non-numeric, non-finite, zero or negative. It is
    never defaulted.
  - `ValueError` when a kept player row has a non-finite distance, peak speed or mean position
    (names the track), and when any float in the finished `match.json` is non-finite (names the
    path, `_nonfinite`). `match.json` is not written in either case.
  - `KeyError` for a missing `detections`; a calibration refused by `calibrators.load`.
- **Never fatal.** `_role_fields` catches any exception and leaves the role unset (printed).
  `accepted_measurements` returns empty on any golden read error, which degrades every tier.
- **Refused values** ([nulls](#nulls)): peak speed where the clamp bound, distance above the
  ceiling, sprints always.
- **Skipped, and counted.** Shape samples below `SHAPE_MIN_PLAYERS` are skipped and counted in
  `coverage`. `attacking_directions` falls back to home-attacks-right and flags it when no keeper
  is found either way. `stress_profile` returns `{}` below two intervals; `structure_stress`
  returns `[]` below its sample floor.
- `quality.check` is advisory: it never refuses a job.

<a id="limits"></a>
## Limits

History and measurements: Evidence: known limits.

- **The shipped numbers over-read.** At the shipped config per-player peak speed reads
  `@C 10.5794 m/s :: model/validation/api-golden.json product.speed.peak_ms player_peak_median_ms (N21's .p21id tracks, 9 STC2025 clips, n=1 configuration)`
  against ground truth's
  `@C 7.3144 m/s :: model/validation/api-golden.json product.speed.peak_ms player_gt_peak_median_ms (same run)`,
  and the pooled sprint count read
  `@C 4.4795x :: n21-product-now (not published) pooled_sprint_ratio (.p21id, 9 STC2025 clips)`
  ground truth, which is why sprints are withheld. Distance at what ships:
  N21, [N22](../experiments/n22-what-inflates-distance.md)-[N28](../experiments/n28-plausibility.md).
- `minutes` is span, not playing time; `distance_km / minutes` under-reads twice on a gapped track.
  `observed_frac` is a floor on the missing share, not a correction factor
  (evidence).
- The chord under-reads in proportion to direction changes, so the bias follows playing
  position (evidence).
- The filter's post-gap velocity is not masked; it inflates the speed consumers (peak, sprints),
  not distance (speed clamp, measured on 118575's split tracks;
  evidence).
- **The gap mask is non-monotonic.** `speed_series` thresholds a gap off the track's own median
  step, so once most steps are gaps the median is the gap and nothing is masked: a sparser track
  can read as fully observed. It does not bite at the shipped stride (`app/server/jobs.py ::
  _pipeline`, the `stride=` passed to tracking); a sparse track would trip it
  (engine notes).
  **Tested 2026-09-29, not changed.** A bursty track (seen 8 frames in 25) reads near zero for a
  different reason than the median rule: its holes are longer than the shipped gap threshold, so
  they are already gaps, and no `DISTANCE_WINDOW_S` window fits inside an 8-sample burst, so every
  window is dropped. Bridging those holes moved the shipped stride-2 fixture's team and player
  distances, and no absolute threshold both keeps stride-2 output unchanged and bridges them, so
  the attempted fix failed its no-change guard and was reverted. Changing it is a measurement-policy
  decision that needs a pre-registered test against the ST2 feed, not a refactor.
- **`_windowed` degenerates at coarse cadence.** The window is `max(1, round(DISTANCE_WINDOW_S /
  step))` samples, so at a stride of about ten frames or more it is one step and distance is an
  unfiltered sum of deltas. Not reached at the shipped stride. A 2-sample floor leaves the shipped
  fixture unchanged (checked 2026-09-29) but was not shipped: no product stride reaches it.
- **An fps fallback is invisible here.** `model/engine/track.py :: run` falls back to a default
  when the container reports no fps; since 2026-09-29 it records that as `fps_source: "fallback"`
  in `tracks.json` ([contract](detection-tracking.md#contract)), but `build_match` does not read
  it and sees a valid number it cannot refuse.
- **Season surfaces withhold sprints too (since 2026-09-29).** `jobs.season` serves `sprints` and
  `sprints_per90` as null and sums nothing; the store's readers (`db.withhold_sprints`, on every
  `/api/db/*` connection) do not see metric `sprints`, though rows stored before 2026-09-28 stay
  on disk untouched; the notepad season table, season queries, chat tile and agent's
  `squad_measured` show "withheld" with the report's reason. A query naming `main.track_metric`
  explicitly still reads the old rows. `model/engine/season.py` never carried sprints.
- A player row can hold more than one person (fragment contamination); a shape column is a
  track, not a person (`n_dedup` reported, not acted on;
  evidence).
- One attacking direction per input: a file spanning half time is wrong for one half
  (evidence).
- `heatmaps` thin by index, not time, and use unfiltered positions.
- Formation labels are template estimates; only `formation_summary` is fit to show, and a tie
  is shown as a tie (evidence).
- Roles are unscored thresholds; `referee` is never assigned
  (evidence).
- `build_match` runs the Kalman twice per track (`ponytail` in `build_match`).

<a id="check"></a>
## How to check

| command | covers | does not cover |
|---|---|---|
| `.venv/bin/python model/validation/check.py measure --fast` | identity's output plus a corrections fixture (one merge, one split) through `jobs._derive_tracks`, then `build_match` on the fixture calibration; PASS is the exact sha of `match.json` with the accepted-id set frozen in the golden; plus the `stats.py`, `frames.py`, `roles.py` self-checks and `test_golden.py` | accuracy; the sprint count (unpublished) |
| `.venv/bin/python model/validation/check.py measure --real` | `dist_eval` at the shipped config on the nine STC2025 clips against ST2 truth, frozen exactly (sprint ratio included) | `build_match`, shape, tiers |
| `.venv/bin/python model/validation/check.py measure --prove` | plants window, clamp, debris-floor, officials, dropped-caveat and heat-map-cap mutants that must FAIL, and the inert ones (a tier under a lower caveat cap, sprint thresholds) that must PASS | correctness of the golden itself |
| `.venv/bin/python model/engine/stats.py` | synthetic distance, sprints, jitter, clamp, gaps, window phase, shape/direction mirror, tier resolution, each refusal and null above; every named api id is registered | real footage |
| `.venv/bin/python model/engine/frames.py` | grid, gap mask, officials and fragments excluded | real tracks |
| `.venv/bin/python model/engine/roles.py` | each role branch, midfielder/winger refusals, observed time over span | any real official |
| `cd model/engine && ../../.venv/bin/python test_golden.py` | `player_stats` on a fixed real SoccerTrack slice against `golden.json` | `build_match`, shape, tiers |
| `model/validation/dist_eval.py` | distance and peak speed against SoccerTrack truth (registry ids) | shape, roles |
| `model/engine/validate.py` | agreement harness through `player_stats` on both sides | accuracy (both sides share the window) |

<a id="evidence"></a>
## Evidence

All on the evidence page:
known limits,
distance window,
filter tuning,
speed clamp,
observed vs span,
shape noise,
keeper and direction,
repair banks,
coverage and duplicates,
formation,
structure stress,
caveat reasons,
roles thresholds.
Experiments: speed clamp,
N21, [N22](../experiments/n22-what-inflates-distance.md),
[N23](../experiments/n23-detector-or-map.md), N24,
N25, [N26](../experiments/n26-map-oracle.md),
[N27](../experiments/n27-tps-in-product.md), [N28](../experiments/n28-plausibility.md),
[N49 jersey OCR](../experiments/n49-jersey-ocr.md). Audits: API adequacy.
Superseded: metrics, 2026-08.
