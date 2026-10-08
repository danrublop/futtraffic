# Detection and tracking

**Summary.** A job's video and its calibration in; `tracks.raw.json` out: per-frame player boxes
associated into ids, stitched, team-labelled, with ball candidates.

Calibration and the scene prior are [the calibration reference](calibration.md). Accuracy lives in
`benchmark (not published)` and `ledger-vision (not published)`;
history and measurements in the evidence page.
`engine-2026-09 (not published)` and `pipeline-2026-09 (not published)` point here. The four stages
share one call chain, one configuration stamp, one detection cache and one output contract.

<a id="what-runs"></a>
## Entry point

`model/engine/track.py :: run(video, out, calib=, detector=, stride=, ball=, dets_cache=, ...)`,
called from `app/server/jobs.py :: _pipeline` (the `track_run(...)` call) after calibration.
Arguments the job does not pass take `run`'s signature default.

### Order

1. `_pipeline` picks the detector: `detectors.py :: product_detector()` returns `coreml` when both
   `COREML_MODEL` and `COREML_PYTHON` exist, else `rfdetr`; recorded in `st["detector"]` only.
2. `_pipeline` deletes a `dets.cache.json` whose header does not carry `"ball": true`.
3. `run` opens the video (`fps` from the container, else `run`'s fallback), loads the calibration
   through `calibrators.load`, flattens remap footage and builds the prior with `make_prior`.
4. `run` builds the detector kwargs and the settings stamp, and loads or validates the
   [detection cache](#cache); a complete cache means the detector is never built.
5. `"best"` resolves through `tracking.py :: best_available()`; the resolved name is stamped.
6. `track.py :: track(...)` runs the per-frame loop ([Detection](#detection),
   [Association](#association), [Teams](#teams)).
7. `run` writes the final cache, runs [stitching](#stitch) when `stitch` is on and a calibration
   exists, stamps `_perf.config` and writes the file.

Off on the job path, reachable by flags only: `rolenet` (`run(roles=True)`), the fine-tuned ball
detector (`run(ball_model=...)`), pitch-plane association (`tracking.py :: PitchTracker`), geometry
tiling (`detectors.py :: geometry_tiles`), raw tile output (`TiledDetector(raw=True)`) and
`stitch.py :: HeightScale`.

<a id="detection"></a>
### Detection

`detectors.py :: get_detector(name, **kw)` looks up `_DETECTORS`: `rfdetr`, `rfdetr-whole`,
`coreml`, `cuda`, `tensorrt` (imported only when asked for) and `stub`. RF-DETR nano to large only
(Apache-2.0); XL and 2XL (PML 1.0) are refused, and no Ultralytics, YOLO-NAS or BoxMOT weights may
appear. `TiledDetector.detect` tiles the frame (`_spans`), resizes each tile to the model's square,
maps boxes back to frame pixels, flags boxes on inner tile edges (`_clip`) and merges them with
`_merge` (intersection over the smaller box, whole before clipped, then by confidence;
**class-agnostic**). With the ball on, the model runs at `min(conf, BALL_CONF)` and `_keep` applies
the per-class thresholds, so player output equals a ball-free run. `CoreMLDetector` streams tiles to
`coreml_worker.py` under Python 3.12, which applies the same class filter before piping. Ball-class
boxes leave the player path before the prior and go to `ball.BallPicker` (the ball section).

<a id="association"></a>
### Association

`tracking.py :: get_tracker("best")` resolves to `BEST` (C-BIoU, `_RoboflowTracker(kind="cbiou")`
onto `trackers.CBIoUTracker`, **constructed with no keyword arguments**) when `trackers` imports,
else to the built-in `bytetrack` with a warning. Every frame is passed to `update`, empty ones
included. An unassociated track emits nothing for that frame. `ByteTrackTracker` and `IoUTracker`
are fallbacks only.

<a id="teams"></a>
### Teams

`track.py :: team_by_shirt(frame, box)` reads a kit colour per detection; after the loop
`assign_teams(colours, classes)` labels each track: `_by_kit` (per-detection k-means, majority vote
per track, a third "official" cluster only when small and far from both teams), then
`_officials_by_contrast`, then class-name overrides (`OFFICIAL_CLASSES`, `PLAYER_NAMES`; inert on the
COCO checkpoint). The larger cluster is named `home`; the name carries no information. No usable
colour gives `unknown`. `jersey` is always `null` here.

<a id="stitch"></a>
### Stitching

`track.py :: stitch_tracks(rows, cal, fps)`, only with a calibration: `stitch.py :: split` severs an
id above `SPLIT_SPEED`; `tracklets` projects each fragment's feet to metres; `stitch` merges
fragments cheapest junction first, refusing temporal overlap, gaps over `MAX_GAP_S`, distances over
`MAX_SPEED` times the gap plus `SLACK_M`, non-finite ends, and opposite team labels; `apply`
relabels rows (nothing added, moved or deleted) and re-votes each merged id's team.

<a id="roles"></a>
### Roles

`rolenet.py` (crop classifier: player, goalkeeper, referee) runs only with `run(roles=True)`, which
the job does not pass. A name is used only above `MIN_SHARE` with at least `MIN_VOTES` crops.
`roles.py` is a different module (measurement section).

<a id="cache"></a>
### Detection cache

`dets_cache` (`dets.cache.json` in the job directory) holds, per detected frame, the boxes after
the prior, the unfiltered ball candidates and the prior's rejections (`rejected`). A hit replays the
frame without the detector and reports the same `prior_rejected`. Checkpoints are written every
`checkpoint_every` detected frames (tmp file, then rename). Refusals are under
[Errors and refusals](#errors-and-refusals).

## Inputs

- The video file (BGR frames, image pixels).
- The job's `calibration.json` ([calibration outputs](calibration.md#files-by-type)); none means no
  prior and no stitching.
- `dets.cache.json`, when present.

<a id="contract"></a>
## Outputs

### Protocols

- **Detector**, `detectors.py :: Detector`: one BGR frame in; `[{"xyxy", "conf", "cls"}]` out in
  **original-frame pixels**. `detect_batch` is optional.
- **Tracker**, `tracking.py :: Tracker`: one frame's detections in; the associated ones out with an
  `"id"`; nothing for unassociated tracks.

### `tracks.json` / `tracks.raw.json`

Written by `track.py :: run` (the job writes `tracks.raw.json`), guarded by `track.py :: _selfcheck`.

| key | type | meaning |
|---|---|---|
| `fps` | float | source frame rate (container, or `run`'s fallback) |
| `fps_source` | str | `container`, or `fallback` when the container reported no usable fps (`run` also prints a WARNING); absent in files written before 2026-09-29 |
| `detections[]` | list | one row per tracked detection |
| `.f` | int | **original** video frame index |
| `.id` | int | track id (after stitching, when stitched) |
| `.x`, `.y` | float | feet: box bottom-centre, image pixels (the flattened canvas on remap footage) |
| `.w`, `.h` | float | box size, image pixels; display only |
| `.team` | str | `home`, `away`, `official` or `unknown` |
| `.jersey` | null | always `null` from this stage |
| `ball[]` | list, optional | when non-empty: `{"f", "x", "y"}`, box **centre**, image pixels; contract in [the ball reference](ball.md#contract) |
| `_perf` | object | `frames`, `seconds`, `throughput_fps`, `prior_rejected`, `stride`, `sampled`, and `config` when stamped |

`_perf.config` is the cache `settings` (`prior_gates`, `conf`, `detector`, `ball`, `overlap`,
`tile`, `model_sha256`) plus `stride`, `stitch` and `tracker_resolved`; absent when the cache was
unstamped (asserted in `track.py :: _selfcheck_remap`).

### Frame numbers under a stride

`f = i * stride`: a strided run emits fewer rows carrying their original indices at the original
`fps`, never renumbered ones. Downstream reads a frame number as time or video position.

<a id="config"></a>
## Config

Read the value at the symbol, or run `model/validation/shipped_config.py`, which resolves the call
statically.

| what | where | how it reaches the run |
|---|---|---|
| detector backend | `app/server/jobs.py :: _pipeline` -> `model/engine/detectors.py :: product_detector` | runtime: files present |
| stride, ball, cache path | `app/server/jobs.py :: _pipeline` (`track_run(...)`) | explicit arguments |
| scene prior gates | `model/engine/track.py :: run` (`prior_gates`) -> `make_prior` (`gates`) | [calibration](calibration.md#scene-prior) |
| player confidence | `model/engine/track.py :: run` (`conf`) | signature default |
| ball confidence | `model/engine/detectors.py :: BALL_CONF` | `min(conf, BALL_CONF)`; the picker's config is in [the ball reference](ball.md#config) |
| kept classes | `model/engine/detectors.py :: PLAYER_CLASSES`, `BALL_CLASSES` | |
| tile, batch | `model/engine/detectors.py :: _coreml`, `_rfdetr` | stamped by `tile_of` |
| tile overlap, merge threshold | `model/engine/detectors.py :: OVERLAP`, `TiledDetector.__init__` (`merge_thr`) | |
| CoreML bundle, interpreter | `model/engine/detectors.py :: COREML_MODEL`, `COREML_PYTHON` (env `FT_COREML_MODEL`, `FT_COREML_PYTHON`) | baked at export |
| RF-DETR variant, resolution, queries | `model/engine/track.py :: run`; `model/engine/detectors.py :: RFDETRDetector.__init__` | fallback backend |
| tracker | `model/engine/track.py :: run` (`tracker`) -> `model/engine/tracking.py :: best_available` | stamped `tracker_resolved` |
| tracker internals | `trackers.CBIoUTracker` defaults; version in `app/server/requirements.txt` | not stamped |
| stitch on/off | `model/engine/track.py :: run` (`stitch`) | needs a calibration |
| split and link gates | `model/engine/stitch.py :: SPLIT_SPEED`, `MAX_SPEED`, `SLACK_M`, `MAX_GAP_S`, `COAST`, `EMB_W` | |
| team clustering | `model/engine/track.py :: _by_kit`, `_officials_by_contrast` | |
| class overrides | `model/engine/track.py :: OFFICIAL_CLASSES`, `PLAYER_NAMES` | inert on COCO |
| roles | `model/engine/track.py :: run` (`roles`); `model/engine/rolenet.py :: MIN_SHARE`, `MIN_VOTES`, `WEIGHTS` | off |
| fine-tuned ball detector | `model/engine/track.py :: run` (`ball_model`); `model/engine/detectors.py :: BALL_MODEL`, `BALL_MODEL_SHA256`, `BALL_VARIANT`, `BALL_TILE`, `BALL_RESOLUTION` | off; see [the ball reference](ball.md#config) |

## Errors and refusals

- **Cache refusals** (`SystemExit` in `run`): a different video basename or start frame; a recorded
  `calib_sha` that differs from `_calib_digest` of the current calibration; recorded `settings` that
  differ (`_settings_mismatch`, strict), except `LATE_KEYS` the cache never recorded, which replay
  stamped `None`; a `complete` cache asked for frames it does not hold. A cache with no `settings`
  replays and the tracks are written **unstamped**.
- **Detector**: a missing CoreML bundle or interpreter raises (no fallback inside the detector);
  RF-DETR XL/2XL are refused; every CoreML reply's sequence number is checked.
- **Tracker**: without `trackers` the job falls back to `bytetrack` with a printed warning.
- **Stitching** is skipped, not refused, without a calibration.

<a id="limits"></a>
## Limits

Each limit's detail and tags are in Evidence: known limits.

<a id="conf"></a>
- Player confidence stays at `run`'s default; lowering it did not clear its bar
  ([N41](../experiments/n41-conf.md)).
<a id="overlap"></a>
- Tile overlap was chosen on one canvas (tile overlap); on some
  frame sizes both candidates give the same grid.
<a id="tracker"></a>
- The tracker runs on library defaults, including a frame rate the product does not feed; passing the
  real rate had no effect (N69). The package version is not stamped.
- Registry tracker ids score stored artefacts, not a fresh product run (`check.py status`).
<a id="ball-merge"></a>
- The tile merge is class-agnostic, so a ball under a player can be suppressed; a class-aware merge
  was not supported (N70).
<a id="stride"></a>
- The stride's identity cost is unmeasured; the evidence for it is UNREPRODUCIBLE
  (Evidence: stride).
- `check.py track` replays a cache detected without the ball.
<a id="stitch-trade"></a>
- Stitching trades purity for wholeness: fewer, longer tracks, more rows under another player's id
  (Evidence: stitching).
- One team label per track; colour cannot separate goalkeeper from referee.
- Uncalibrated runs are not stitched; `HeightScale` is not wired into the job path.
- `rolenet` reads dark kits as officials (Evidence: roles).
- A packaged build without Python 3.12 runs `rfdetr`, recorded only in `st["detector"]`.
- A container with no fps runs at `run`'s fallback; recorded as `fps_source: "fallback"` in
  [`tracks.json`](#contract) with a WARNING in the worker log, and nothing downstream acts on it.

<a id="check"></a>
## How to check

| command | covers | does not cover |
|---|---|---|
| `.venv/bin/python model/validation/check.py track --fast` | the stored 118575 cache replayed through `track.run` at the shipped call shape, hashed against a golden | the ball; detection |
| `.venv/bin/python model/validation/check.py detect --fast` | the product detector on a committed excerpt at the shipped stride, conf, ball and prior | other footage; accuracy |
| `check.py track --prove`, `check.py detect --prove` | each planted defect must fail | - |
| `.venv/bin/python model/validation/check.py status` | read-only: goldens, cache staleness, runs, processes | - |
| `(cd model/engine && ../../.venv/bin/python track.py)` | output contract, stride numbering, cache resume/replay, team vote, stitch gate, stamp and cache refusals | real footage |
| `(cd model/engine && ../../.venv/bin/python detectors.py)` | tile spans, merge, tile-to-frame mapping, bundle paths; CoreML worker when present | accuracy |
| `(cd model/engine && ../../.venv/bin/python tracking.py)` | IoU invariants, no teleport welding, empty-frame ageing | C-BIoU accuracy |
| `(cd model/engine && ../../.venv/bin/python stitch.py)` | temporal exclusion, team gate, split, NaN ends | real tracklets |
| `(cd model/engine && ../../.venv/bin/python rolenet.py)` | the majority-share vote | the model |
| `~/.venvs/ftcoreml/bin/python model/engine/coreml_worker.py --selfcheck` | pre/post-processing against goldens; class filter equals `_keep` | - |
| `.venv/bin/python model/validation/shipped_config.py` | product configuration against every mirror and stamped artefact | runtime-only keys |
| `.venv/bin/python model/validation/verify_claims.py --all` | re-runs every measured tag, including the evidence page's | - |

<a id="evidence"></a>
## Evidence

All on the evidence page:
known limits,
tiling and scale,
far touchline and confidence,
tile overlap,
decoder queries,
CoreML port and throughput,
stride,
ball threshold and classes,
association,
stitching,
kit colour and teams,
role classifier.
