# Ball

**Summary.** One frame's ball-class boxes in; at most one ball position per detected frame out, as
`tracks.raw.json :: ball[]`. A picker, not a tracker: gates, then the most confident survivor, with
no memory of previous picks.

Detection of the candidates is [the detection reference](detection-tracking.md#detection); accuracy
lives in the experiments under [Evidence](#evidence); history and reasoning in
the evidence page.

<a id="entry-point"></a>
## Entry point

`model/engine/ball.py :: BallPicker(cal=None, scenery=None, marking_gate=False)` and
`BallPicker.update(dets, frame_h, f=0, frame=None) -> Ball`, constructed and called once per detected
frame by `model/engine/track.py :: track(...)`, which `track.py :: run` reaches from
`app/server/jobs.py :: _pipeline` (the `track_run(...)` call, `ball=True`).

Path: `jobs.py :: _pipeline` calls `track.py :: run(ball=True)`; the detector runs at `BALL_CONF` for
the ball classes; `track.py :: track` splits the ball boxes off; the candidates go to the detection
cache; `BallPicker.update` runs once per detected frame; each `seen` answer becomes a `ball[]` row.

### Order

1. The detector runs at `min(conf, BALL_CONF)`; `detectors.py :: _keep` applies the ball threshold to
   `BALL_CLASSES` and the player threshold to the rest ([detection](detection-tracking.md#detection)).
2. `track.py :: track` splits `BALL_CLASSES` boxes off before the scene prior and the tracker. The
   split is unconditional; with `ball=False` the candidates are dropped and no picker is built.
3. The candidates, unfiltered, are written to the [detection cache](detection-tracking.md#cache).
4. `BallPicker.update`, in this order:
   1. **class and shape** (`plausible`): class in `("sports ball", "ball")`, box aspect within
      `ASPECT_MIN..ASPECT_MAX`, longest side within `SIZE_MIN_FRAC..SIZE_MAX_FRAC` of frame height;
   2. **scenery** (`Scenery.flags`): rejects a candidate that some candidate sat within `radius` px of
      in at least `share` of the last `window` calls, once `min_frames` calls are held; fed on
      every call, including calls that return nothing;
   3. **on the pitch** (calibration only): the centre projected by `cal.to_pitch` lies within
      `MARGIN_M` of the pitch rectangle;
   4. **physical size** (calibration only): longest box side over `_expected_px` (pixels that
      `BALL_DIAMETER_M` spans at that ground point) within `SIZE_RATIO`;
   5. **marking** (opt-in, `marking_gate=True` and a frame): `blob_extent` above `MAX_BLOB_EXTENT`
      rejects;
   6. argmax of `conf` over the survivors; none left gives `seen=False`.
5. `track` appends `{"f", "x", "y"}` only when `seen`.

Off on the job path, reachable by flags only: the marking gate (`run(ball_markings=True)`,
`track.py --ball-markings`) and the fine-tuned ball detector (`run(ball_model=...)`,
`track.py --ball-model`), which replaces the primary detector's ball boxes with its own
(`dkw["ball"]` goes False) at `BALL_VARIANT`, `BALL_TILE`, `BALL_RESOLUTION`.

## Inputs

- `dets`: `[{"xyxy", "conf", "cls"}]`, **original-frame pixels**, or the flattened canvas on remap
  footage.
- `frame_h`: the height of the frame the boxes are in (canvas height on remap footage).
- `f`: original video frame index (`i * stride`).
- `frame`: BGR image, read only by the marking gate.
- `cal`: the job's calibrator, or None; needs `to_pitch`, `H` (image to pitch) and `pitch.length`,
  `pitch.width` in metres.

<a id="contract"></a>
## Outputs

`ball.py :: Ball`: `f`; `xy` box **centre** in the same pixels as the input, `(nan, nan)` when not
seen; `conf`; `wh` box size in pixels; `seen`.

`tracks.raw.json :: ball[]` ([file contract](detection-tracking.md#contract)):

| key | type | meaning |
|---|---|---|
| `f` | int | original video frame index |
| `x`, `y` | float | box **centre**, image pixels (flattened canvas on remap footage) |

- At most one row per detected frame; frames skipped by the stride have no row.
- A missing frame means **not seen**: nothing passed the gates, nothing was proposed, or the frame
  was not detected. Rows are never interpolated.
- The key is written only when non-empty; `track.py :: run` returns it always.
- The centre is not a ground point. Projecting it through the calibration is valid only for a ball
  on the ground.
- Pixels are the flattened canvas on remap jobs and the raw frame otherwise, the same space as
  `detections[]` ([detection contract](detection-tracking.md#contract)).

### Consumers

- `model/engine/overlay.py :: by_frame` and `render`: marker, trail (projected with `cal.to_pitch`
  for its speed test) and minimap.
- `model/engine/quality.py :: check`: the four ball criteria report `not_evaluated` on every job
  and do not read `ball[]`.
- Nothing in `model/engine/stats.py`, `possession.py`, `events.py`, the report or the UI reads
  `ball[]`; `overlay.py` is the only reader.

<a id="config"></a>
## Config

Read the value at the symbol, or run `model/validation/shipped_config.py`.

| what | where | how it reaches the run |
|---|---|---|
| ball on | `app/server/jobs.py :: _pipeline` (`track_run(..., ball=True)`) | explicit argument |
| ball confidence | `model/engine/detectors.py :: BALL_CONF` | `min(conf, BALL_CONF)`, then `_keep` |
| classes split off | `model/engine/detectors.py :: BALL_CLASSES` | `track.py :: track` |
| classes the picker accepts | `model/engine/ball.py :: BallPicker.update` (literal tuple) | |
| shape gate | `model/engine/ball.py :: ASPECT_MIN`, `ASPECT_MAX`, `SIZE_MIN_FRAC`, `SIZE_MAX_FRAC` | |
| scenery | `model/engine/ball.py :: Scenery.__init__` (`radius` px, `window`, `share`, `min_frames` in picker calls) | defaults |
| pitch margin, size ratio | `model/engine/ball.py :: BallPicker.MARGIN_M` (m), `BallPicker.SIZE_RATIO`, `BALL_DIAMETER_M` (m) | |
| marking gate | `model/engine/track.py :: run` (`ball_markings`); `ball.py :: MAX_BLOB_EXTENT` | signature default (off) |
| fine-tuned detector | `model/engine/track.py :: run` (`ball_model`); `detectors.py :: BALL_MODEL`, `BALL_MODEL_SHA256`, `BALL_VARIANT`, `BALL_TILE`, `BALL_RESOLUTION` | signature default (off) |

### What ships and what does not

| | status |
|---|---|
| stock detector's `sports ball` through the picker | ships: `jobs.py` passes `ball=True` |
| N52 arm D1 (RF-DETR fine-tuned on TeamTrack clips, MIT) | research: exists, not wired; cost is a second detection pass ([N52](../experiments/n52-clean-ball-data.md)) |
| N48 arm B (trained on DFL data) | research only by licence; `run(ball_model=...)` refuses it by `BALL_MODEL_SHA256` |
| marking gate | off; opt-in flag |

## Errors and refusals

- `run(ball_model=...)` exits when the file is missing or its SHA-256 is not `BALL_MODEL_SHA256`
  (the licence gate: only N52 arm D1 may load).
- `_pipeline` deletes a `dets.cache.json` whose header lacks `"ball": true`; `run` refuses a stamped
  cache whose `settings` (`ball` included) differ ([cache refusals](detection-tracking.md#errors-and-refusals)).
- No calibration: gates 3 and 4 are skipped, not refused.
- The marking gate without a frame is skipped. A window with no pixels passes.
- Nothing raises inside `update`; every failure is `seen=False`.

<a id="limits"></a>
## Limits

- The shipped ball is COCO `sports ball` from the player detector; its candidate list is the
  ceiling (N48,
  [N65](../experiments/n65-every-ball-number-was-measured-at-384.md)).
- A licence-clean fine-tuned detector (N52 arm D1, TeamTrack, MIT) exists and is not wired; its
  cost is a second detection pass ([N52](../experiments/n52-clean-ball-data.md),
  evidence: fine-tuned detector,
  licence route). The DFL-trained arm is research-only by
  licence and is refused by the hash. With D1 the picker, not the detector, is the loss
  ([N55](../experiments/n55-picker-is-clutter-not-signal.md)).
- The ball output is unused downstream: no stat, possession, event or quality criterion reads it.
- The registry level at the stride that ships is `api://product.ball.correct.s2`;
  `api://product.ball.correct` is stride 1. Both are nine TeamTrack clips from two matches, and the
  three side clips run without a calibration.
- Six picker changes lost to the shipped gates: path, re-fit size gate, joint, player prior, fewer
  candidates, gate deletion (N49,
  N50, N51,
  N58, [N67](../experiments/n67-fewer-candidates-does-not-help-the-picker.md),
  N59); the loss is clutter, not a weak signal
  ([N55](../experiments/n55-picker-is-clutter-not-signal.md), N57).
- A ball at rest on painted lines is not proposed (N49 paint).
- The tile merge is class-agnostic, so a ball under a player can be suppressed; the class-aware
  merge was measured NOT SUPPORTED ([detection limits](detection-tracking.md#ball-merge),
  N70).
- Scenery counts picker calls in canvas pixels: under a stride its window and `min_frames` span
  `stride` times more video, and a panning camera defeats it for a window
  (scenery).
- The size gate reads `H` for any calibrator class without `jacobian`: `UndistortWrapper` and
  `BowWrapper` delegate to their inner calibrator and miss their own pre-warp. Neither is written
  by the product (`ball.py :: BallPicker._expected_px`, ponytail note).
- The marking gate costs true balls on every clip with ball truth; its only support is a by-eye count
  (evidence: marking gate).
- The gates were fitted on the TeamTrack top-down clips; their per-gate table is in-sample.
- Pitch position is wrong for a ball in the air; nothing flags airborne frames.
- No consumer turns the ball into possession, events or a stat.

## How to check

| command | covers | does not cover |
|---|---|---|
| `.venv/bin/python model/validation/check.py ball --fast` | two replayed clip excerpts through the real `track.track()` (exact rows, correct/wrong/silent against ball truth) and the detect fixture's candidates on a spherical calibration | the detector (candidates are fixed) |
| `.venv/bin/python model/validation/check.py ball --real` | the nine-clip metric at stride 1 and 2, exact against the golden | other grounds; broadcast footage |
| `.venv/bin/python model/validation/check.py ball --prove` | planted defects (size ratio, margin, scenery `min_frames`, `cal=None`, `BALL_CONF`, the pre-fix size gate) must fail | the marking gate (inert while off) |
| `(cd model/engine && ../../.venv/bin/python ball.py)` | each gate, memorylessness, not-seen, marking gate on/off | real footage (synthetic only) |
| `(cd model/engine && ../../.venv/bin/python track.py)` | the split before association, a second detector reaching the picker, cache replay of ball rows | accuracy |
| `.venv/bin/python model/validation/check.py detect --fast` | ball candidates in the detection cache at the shipped call | the picker |
| `.venv/bin/python model/validation/check.py track --fast` | - | the ball (fixture cache is ball-free) |
| `.venv/bin/python model/validation/shipped_config.py` | `ball`, `ball_model` against every mirror | runtime |
| `model/validation/ball_cmp.py`, `ball_eval.py` | accuracy against TeamTrack ball truth (see each docstring for the rule) | other grounds; broadcast footage |

## Evidence

N48 (detector A/B, confounded with tile size),
N49 path, N49 paint,
N50, N51,
[N52](../experiments/n52-clean-ball-data.md), [N55](../experiments/n55-picker-is-clutter-not-signal.md),
N57, N58,
N59,
N64,
[N65](../experiments/n65-every-ball-number-was-measured-at-384.md),
[N67](../experiments/n67-fewer-candidates-does-not-help-the-picker.md),
N70;
detection evidence: ball threshold, classes, fine-tuned detector;
the evidence page: why a picker,
gates and the temporal terms,
scenery, marking gate,
shape gate and footage, licence route.
