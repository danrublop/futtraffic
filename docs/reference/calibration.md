# Calibration

**Summary.** A job's video and any calibration it was given in; a validated `calibration.json`
(image pixels to pitch metres) and the scene-prior box filter out.

Accuracy lives in the calibration benchmark; history and
measurements in the evidence page. `engine-2026-09 (not published)` and
`pipeline-2026-09 (not published)` point here.

## Entry point

`app/server/jobs.py :: _pipeline(st)`, the calibration branch, per job, before tracking. The stages
below are the functions it calls, in order.

### Order

1. **Record the frame size.** `frame_shape` is written on every job with a video, before any
   branch, so every hand-calibrated job can later be an inheritance source.
2. **Zero-click, panoramas only.** `jobs.py :: auto_calibrate(st)` runs when there is no
   `calibration.json`, the video exists, `AUTOCAL` is on and the frame is at least
   `PANORAMA_ASPECT` times as wide as tall. It never fails the job; a refusal is recorded as
   `calibration_auto` and the branch continues.
3. **Inherit** a hand-clicked calibration ([Inheritance](#inheritance)).
4. **Park for the operator's dots.** With still no file the job stops at stage `calibrating`
   (`waiting_for: "calibration"`). `app.py :: set_calibration` (POST
   `/api/matches/{id}/calibration`) validates through [the load gate](#load-gate), sets
   `calibration_source = "manual"`, clears `calibration_inherited_from` and re-queues.
5. **Load gate** on the file the job will use; a failure fails the job.
6. **Scene prior** for tracking ([Scene prior](#scene-prior)); `stats.build_match` reads the same file.

Non-panorama footage skips step 2.

### Pitch size

`jobs.py :: venue_pitch(st)` returns a size somebody gave (this job's `pitch`, set by
`app.py :: set_pitch`, else the latest at the same club and ground via `_venue_pitches`), or `None`,
never a default. A length with `source` `"user"` is fitted as given (`fix_length=True`) and the paint
reading is recorded beside it (`length_given_m`, `length_measured_m`, `length_overruled`); otherwise
the length is measured from the paint (`autocal.py :: _settle`). Width is never measured: it comes
from `venue_pitch` or the `pitch.py :: Pitch` default, flagged `width_assumed`. `set_pitch` deletes
an automatic calibration and re-queues; a hand calibration is left alone.

### Install camera

`jobs.py :: _install_camera(st, shape)`: the `camera` of the latest spherical calibration from
another job with the same club, slugged ground and `frame_shape`. `autocal.from_install` registers it
to this match's paint and accepts or refuses it; on refusal the search from nothing runs. Recorded as
`calibration_auto.via == "install"` and `install_from`.

### Inheritance

`jobs.py :: inherit_calibration(st)` copies the latest `calibration.json` from another job only when
this job has no file, has a venue and a readable frame size, and the source has the same
`frame_shape`, slugged venue and team, was not itself inherited, and is not automatic. Records
`calibration_inherited_from`. Why the ground must match:
Evidence: inheritance.

### The automatic write

`jobs.py :: _auto_calibrate` writes `calibration.auto.json`, loads it through the load gate, then
creates `calibration.json` with `os.link`, which fails if a human's file landed meanwhile. Only a
successful link sets `calibration_source = "auto"`.

### Load gate

`jobs.py :: load_calibration(path)` decides validity for every caller (`set_calibration`, the worker,
and through `calibrators.load` also `track.run` and `stats.build_match`). Its refusals are listed
under [Errors and refusals](#errors-and-refusals).

### Scene prior

`model/engine/track.py :: make_prior(cal, image_wh, gates)` returns a box filter, or `None` for "run
unfiltered", never an exception. Only the pitch-polygon gate ships. Spherical files walk
`pitch_polygon_px` through the camera's own map; homography files recover a camera with
`sceneprior.camera_from_homography`. The job records `prior_rejected` and `scene_prior`.

## Inputs

- The job's video (`frame_shape` in pixels) and its `st` record (club, ground, team, `pitch`).
- Optionally the operator's clicks: landmark id or `"(x,y)"` to image pixels, POSTed to
  `set_calibration`.
- Optionally a pitch size in metres (`{"length", "width", "source"}`) from `set_pitch`.

<a id="contract"></a>
## Outputs

`calibration.json` in the job directory, the job fields `calibration_source`, `calibration_auto`,
`calibration_inherited_from`, `prior_rejected`, `scene_prior`, and the prior passed to `track.run`.

### Coordinates

- **Pitch metres.** Origin at a corner; `x` along the touchline, `0..length`, to image-right on a
  side-on camera; `y` in `0..width`. **`y = 0` is the FAR touchline**, `y = width` the near one.
  Landmark `_b` ids are far (`y = 0`), `_t` near; `corner_bl` is the far-left corner.
- **Image pixels.** The raw frame for `perspective`, `perspective_tps`, `perspective_fisheye` and
  `spherical`; the **remapped canvas** for `remap` (raw panorama pixels through `to_pitch` are wrong
  with no error; convert with `calibrators.remap_points`).
- **`H`** is image px to pitch metres (`pitch.py :: Calibration.H`). On `spherical` it is a stand-in:
  measure with `to_pitch`, `jacobian` and `pitch_polygon_px`, never `H`.
- **`image_wh`** (spherical only) is the frame size the camera was solved on.

### Files by type

Every file carries `projection_type` (absent reads as `perspective`) and `pitch` (`pitch.py :: Pitch`).

| `projection_type` | written by | extra keys | loaded as |
|---|---|---|---|
| `perspective` | `app.py :: set_calibration`; `inherit_calibration`; demo jobs (identity `H`) | `H`, `points`, `rms_error_m`, `outliers` | `PerspectiveCalibrator` |
| `remap` | nothing in `app/` | perspective keys (remapped space), `maps`, `maps_yscale`, `maps_crop`, `maps_scale`, `note` | `RemapCalibrator` |
| `perspective_tps` | validation scripts (pipeline.md §4) | perspective keys | `TpsCorrected` |
| `perspective_fisheye` | validation scripts | perspective keys, `camera` | `FisheyeCamera` |
| `spherical` | `jobs.py :: _auto_calibrate` | `camera`, `image_wh`, `H` (stand-in), `points: {}`, `outliers: []`, `rms_error_m` (line-fit residual, not accuracy), `auto` | `SphericalCamera` |

`calibrators.from_dict` also honours `bow` and `undistort` wrapper keys; no product path writes either.

## Config

Read the value at the symbol.

| what | where |
|---|---|
| panorama test | `app/server/jobs.py :: PANORAMA_ASPECT` |
| zero-click on/off | `app/server/jobs.py :: AUTOCAL` (env `FUTTRAFFIC_AUTOCAL`) |
| accepted types, collinear and singular-H thresholds | `app/server/jobs.py :: load_calibration` |
| given-vs-measured length tolerance | `app/server/jobs.py :: _auto_calibrate` |
| frames in the median image | `model/engine/autocal.py :: calibrate` (`n_frames`) |
| vertical law of the spherical camera | `model/engine/autocal.py :: VERT_CUBIC` |
| registration acceptance | `model/engine/autocal.py :: MIN_HITS`, `MAX_LINE_RMS_PX`, `accept` |
| grass share refusal | `model/engine/autocal.py :: MIN_PITCH_SHARE` |
| mount search | `model/engine/autocal.py :: GRID`, `BOUNDS`, `TOP_K`, `TOP_REG` |
| length scan | `model/engine/autocal.py :: LENGTHS_COARSE`, `LENGTH_FINE` |
| registration radii | `model/engine/autocal.py :: RADII`, `AUTO_RADII` |
| default pitch | `model/engine/pitch.py :: Pitch` |
| homography solver | `model/engine/pitch.py :: Calibration.solve` |
| scene prior gates | `model/engine/track.py :: make_prior` (`gates`) |
| polygon margin | `model/engine/sceneprior.py :: pitch_polygon`, `model/engine/calibrators.py :: SphericalCamera.pitch_polygon_px` (`margin_m`) |
| near-nadir bound, mount warning | `model/engine/sceneprior.py :: MAX_CAMERA_H_M`, `Camera.mount_ok` |
| retired height gate | `model/engine/sceneprior.py :: HEIGHT_TOLERANCE` |

## Errors and refusals

- **Load gate** (`ValueError`, phrased for the operator): a `projection_type` outside the accepted
  tuple (`panoramic` included); a probe pixel that does not project to a finite point; with
  `points`, too few inlier landmarks or landmarks on nearly one line; a near-singular `H`. In the
  worker this fails the job with `unusable calibration.json: ...`; in `set_calibration` it is the
  POST's error.
- **Zero-click** never raises into the job: exceptions and refusals are recorded in
  `calibration_auto` (`accepted`, `why`) and the job falls through to inheritance or the dots.
  `accept()` refuses a camera unless both penalty-area fronts are on paint.
- **Scene prior** degrades to `None` (unfiltered) when the gate set excludes `pitch`, `image_wh`
  differs from the video, the outline is too short, camera recovery raises (including the near-nadir
  bound), a clicked landmark lies outside the frame, or `maps_yscale` is not 1. A low mount only warns.
  Zero rejections over a match is reported as "the geometry prior may not have loaded".
- **Remap** files load without their tables; `undistort_frame` then raises.

<a id="known-limits"></a>
## Limits

Each limit's detail and tags are in Evidence: known limits.

- No product job has been calibrated by autocal yet; zero-click is validated on one install
  ([N34](../experiments/n34-zero-click.md)), and `autocal.py`'s constants were shaped on it.
- The far touchline is the weak place ([N36](../experiments/n36-far-touchline.md)).
- On an install's first match, zero-click could accept a wrong length; `accept()` now refuses that
  case, so it needs the install camera or clicks (N68).
- Width is assumed when nobody gave it; every across-the-pitch distance scales with the assumption.
- A race window remains between `os.link` and the worker's save: a job can read `auto` over a
  human's file. Not reproduced.
- Identity-`H` files with no points are accepted on purpose (demo jobs); on real video they
  would read pixels as metres.
- `autocal.median_frame` holds every sampled frame in memory at once.
- Unreachable today and wrong if reached: `UndistortWrapper` around `RemapCalibrator`; the `width`
  default of `autocal._fwd`/`_inv` under the column warp.

## How to check

| command | covers | does not cover |
|---|---|---|
| `.venv/bin/python model/validation/check.py calib --fast` | pitch, track, sceneprior and jobs self-checks; the committed golden spherical file through `from_dict` (keys and map) | real footage |
| `.venv/bin/python model/validation/check.py calib --real` | `auto_calibrate` on committed median frames; accept/refuse as frozen, including the N68 refusal | other installs |
| `.venv/bin/python app/server/jobs.py selfcheck` | load gate refusals; inheritance rules; `auto_calibrate` with a stubbed camera; a human POST racing the write; a given length winning; install-camera matching | real `autocal` on video; the post-link race window |
| `.venv/bin/python model/engine/autocal.py` | a synthetic panorama: recovery with no dots, length measured and held, install start, refusals | real footage |
| `.venv/bin/python model/engine/autocal.py VIDEO` | one real video, prints the report | nothing is asserted |
| `(cd model/engine && ../../.venv/bin/python calibrators.py)` | projection types against synthetic truth | real remap tables |
| `(cd model/engine && ../../.venv/bin/python pitch.py)` | homography round trip, outliers, JSON round trip | anything but `perspective` |
| `(cd model/engine && ../../.venv/bin/python sceneprior.py)` | camera recovery, low mount, near-nadir bound, polygon | the shipped gate choice |
| `(cd model/engine && ../../.venv/bin/python track.py)` | `make_prior`'s gates, spherical outline and frame-size refusal, degrade to `None` | a real job |

## Evidence

All on the evidence page:
known limits,
zero-click,
load gate,
inheritance,
stitched panoramas and remap,
piecewise and dense keypoints,
bow and lens,
spline and fisheye,
pitch dimensions,
scene prior.
Accuracy: `benchmark-calibration (not published)`.
