# Glossary

> **Citation style.** `file.py :: symbol`, not line numbers — the codebase is
> under active refactoring. Verified against the tree as of 2026-08-04.

Domain terms a new engineer needs, with the code that uses each one.

---

## Geometry and projection

### Homography
A 3×3 matrix `H` mapping one **plane** to another under a projective camera.
Here: image pixels → pitch metres. Eight degrees of freedom, so four
non-collinear point correspondences determine it.

```
        image plane                    pitch plane (z = 0)
        ┌──────────────┐                 ┌──────────────┐
        │   ·  ·   ·   │   ── H ──►      │  ·   ·   ·   │
        │  ·  · ·  ·   │                 │  ·   ·   ·   │
        └──────────────┘                 └──────────────┘
   (u,v,1)ᵀ  →  H·(u,v,1)ᵀ  →  divide by the third component
```

Code: `pitch.py :: Calibration.solve` / `.to_pitch`. Solved once per camera
install from operator clicks on named landmarks (`pitch.py :: Pitch.landmarks`),
which deliberately offers more than the four a homography needs — a fixed wide
camera sees all of them, and over-constraining is what absorbs lens distortion
at the frame edges where the corners live.

**Why it can be wrong:** a homography maps a plane to a plane. It is valid only
if (a) the pitch is flat and (b) the image is a real pinhole render. A stitched
cylindrical panorama satisfies neither, and no 3×3 matrix undoes it.

### Perspective vs panorama

| | Perspective | Panorama (cylindrical / stitched) |
|---|---|---|
| Produced by | one un-warped camera | multi-camera rig — Veo, SoccerTrack v2 |
| Straight world lines | stay straight | **bow**; the near touchline bows 169 px across its length on real footage |
| Single homography | correct | **median 5.35 m error** on real footage |
| "lower in frame = closer" | holds | **fails at the panorama edges** |
| Correct handling | click-to-calibrate directly | `cv2.remap` to flatten the stitch, **then** one homography |

`pitch.py :: Calibration.from_dict` refuses to load a non-perspective file
rather than silently reading it as perspective. Every calibration file carries a
`projection_type` stamp for exactly that reason.

### Remap
A dense per-pixel lookup table (`mapx`, `mapy`) that resamples a distorted or
stitched image into a rectified one, via `cv2.remap`. OpenCV's fixed-point form
is `CV_16SC2` + `CV_16UC1`, which is what SoccerTrack ships. After a remap the
image behaves like a pinhole camera and one homography becomes correct again.

`calibrators.py :: RemapCalibrator` deliberately does **not** consume a vendor's
own homography — theirs lives in their remapped frame with their conventions and
padding, and reverse-engineering that per vendor is unbounded work for no
product value. It remaps the *frame* and runs our own click-to-calibrate on the
flattened image.

### Piecewise homography
`calibrators.py :: PanoramicCalibrator`: one locally-weighted homography per
vertical strip, **blended** linearly between strip centres rather than switched
— a hard switch would put a step in every player's velocity as they cross a
seam, and velocity is what we sell. A mitigation, not a fix; the fix is the
remap. Two ceilings, named in-code: it cannot model vertical curvature, and it
is only as piecewise as the landmarks allow (the 21 landmarks sit in three
x-clusters, so past ~6 strips the extra ones interpolate rather than measure).

### Homography Jacobian
`d(pitch metres) / d(image pixels)`, a 2×2 matrix evaluated at each image point
(`stats.py :: homography_jacobian`). It is what makes the Kalman's measurement
noise honest: one pixel of vertical box jitter is **~0.45 m at the near
touchline and ~2.07 m at the far one**, because the grazing angle foreshortens
depth.

```
  R = σ_px² · J·Jᵀ        stats.py :: kalman_ground
```

### Ground point
The pixel projected through the homography — where the code believes the player
touches the ground. The homography maps the *ground plane*, so a point that is
not on the ground returns a position wrong by roughly

```
  ground error ≈ d_horiz · h / H_camera
```

**What ships is the box bottom-centre**, hardcoded in `track.py :: track` as
`y = fy2`. Pose-based alternatives (ankle midpoint, pelvis-drop) were prototyped
and then *measured away* — the conclusion was a mounting instruction rather than
an engineering project — and the estimator classes have since been deleted.
`groundpoint.py` now holds only the synthetic rig and the measurement that
produced that decision.

### Stride oscillation
The box bottom is the **lowest visible pixel**, which during a run is an
airborne foot, alternating at step frequency (`groundpoint.STEP_HZ = 3.0`,
`FOOT_LIFT_M = 0.30`). The projected ground point therefore oscillates — and
oscillation differentiates into distance the player never covered.

```
  camera H=10 m, player at d=50 m, foot lift h=0.3 m
    →  ±1.5 m of spurious ground-plane motion at 2–3 Hz
```

Measured (synthetic rig): the q=1.0 Kalman suppresses it to **+1.8%** at a 10 m
mount vs **+6.5%** for a 0.4 s boxcar. The driver is **camera height**, almost
not range. Conclusion: mount at 6 m or higher. It explicitly does **not** apply
to peak speed, which is an extreme-value statistic.

### `mount_ok`
`sceneprior.py :: Camera.mount_ok` — true iff the camera height recovered from
the homography is ≥ 6 m. Catches a bad install from the calibration file alone,
at onboarding, instead of silently poisoning a season of numbers.

---

## Detection and tracking

### Tracklet
A contiguous run of detections the tracker believes is one player, between two
breaks. A real match produces many per player because occlusion splits tracks;
joining them back is what `identity.py :: merge_tracklets` does.

```
   true player   ────────────────────────────────────────────────►
   tracker       ──id 7──┐        ┌──id 23──┐      ┌──id 51──────►
                         └ 0.8 s ─┘         └ 1.2s ┘
                          occlusion          occlusion
   tracklets:      [7]              [23]            [51]
   merged group:   ├──────────── group 7 ─────────────┤
```

The **group id is the smallest member track id**, so it is stable across runs.

### ID switch (IDSW)
The tracker hands one player's identity to a different player. Two kinds:

- **Loud** — a teleport across the pitch. Trips the 9.5 m/s clamp and shows up
  in `clamped_frac`.
- **Quiet** — a smooth handoff between two *adjacent* players. Never trips the
  clamp. Moves distance from one profile to another and leaves both looking
  plausible. `stats.py :: player_stats` calls this "not cheaply detectable";
  **that is no longer true (2026-08-18)**. With per-detection ground truth it
  is cheap, and it has been measured: **33.2%** of detections sit under an id
  whose dominant true player is somebody else, 49.6% before the 45 m/s split
  and the team gate (**CORRECTED 2026-09-28:** one match, 118575, tracks config unrecorded; 49.6% is the ungated stitch and raw tracks before stitching score 15.1%, per `ledger-vision.md`; at what ships `@C 35.0 :: n21-product-now (not published):87` (pooled ST2 sequences, after the identity stage)). What `clamped_frac` cannot see, ground truth can.
  See **Contamination**.

Measured by `validate.py :: id_switches`, with globally optimal per-frame
assignment inside a 2 m radius (greedy nearest-neighbour invents switches that
never happened when 22 players are inside a few metres at a corner), plus
`IDSW_HOLD = 12` frames of hysteresis so a one-frame flicker is not counted.

### ByteTrack
The association idea vendored in `tracking.py :: ByteTrackTracker`: run the
normal high-confidence matching pass, then a **second pass** re-associating
*low*-confidence detections to still-unmatched tracks — because a low-confidence
box sitting on an already-tracked player is usually that player occluded, not
noise. That second pass is what keeps an id alive through a crowd.

### Expansion-IoU / Deep-EIoU
Enlarging boxes before computing IoU, because plain IoU is degenerate at 20 px —
a 2 px shift destroys it. **Named as a decision in `STACK (not published)`; not implemented.**

### SAHI / tiled inference
Slice the frame into overlapping crops, run the detector on each, NMS the union.
The point is **tile-scale alignment**: sizing tiles so players land at the
detector's trained object scale. One SoccerNet SynLoc team reported AP50:95
**0.517 → 0.878** from that alone. **Not implemented.**

### Scene prior
Two rejections a fixed camera gives you free, both derived from the homography
you already have (`sceneprior.py`). The broadcast-focused literature ignores
both, because a moving camera destroys them.

1. **Pitch polygon** — the pitch rectangle plus a 3 m margin, projected into
   image pixels. Crowd, dugout, bench, adjacent pitches and passing cars are all
   outside it. ~~Measured to exclude ~50% of the frame on a synthetic rig.~~
   UNMEASURED (marked 2026-09-28): no command in the repo reproduces that share,
   and a synthetic rig's share says nothing about a real frame; treat it as not
   measured.
2. **Per-row player size prior** — the expected pixel height of a
   `PLAYER_HEIGHT_M = 1.80` standing player at each ground point, with a
   `HEIGHT_TOLERANCE = ±40%` gate. Kills crowd boxes, half-occluded pairs and
   boxes merged across two players.

Both need the camera's focal length and height, recovered from the ground-plane
homography alone (Zhang's single-homography trick,
`sceneprior.py :: camera_from_homography`). See `pipeline-2026-09 (not published)` §2[C].
CORRECTED 2026-09-28 (was "built and not active in the service"): the prior IS
active in the service, pitch polygon only; the height gate is off by default
(`track.py :: make_prior`, default `gates=("pitch",)`).

---

## Measurement

### Clamp / `clamped_frac`
`stats.MAX_SPEED = 9.5 m/s` is applied inside the Kalman loop. `clamped_frac` is
the share of *observed* steps at which the filtered speed saturated that limit.

It is a **data-quality flag, not a metric**: a track that repeatedly saturates is
either switching identity or badly tracked, and either way its numbers should not
be quoted. Above 1% the UI flags the track wherever it appears
(`app/frontend/src/app/players/[trackId]/player-view.tsx`, `report/page.tsx`,
`checks/run-panel.tsx`). It sees loud switches only.

### HSR — high-speed running
Distance covered above a speed threshold: `Σ v·dt` over samples with
`v ≥ 5.5 m/s` **and** inside a real (non-gap) step
(`validate.py :: player_metrics`). The threshold is deliberately shared with the
sprint threshold via `validate.HSR_SPEED = stats.SPRINT_SPEED`.

Published benchmarks show **>40% deviation between commercial systems** on HSR,
most of it threshold and filtering artefact rather than tracking error. Never
present it as cross-system comparable, and always state the threshold.

**It is not shipped:** `stats.py :: build_match` does not emit it, and the UI
does not ask for it — `app/frontend/src/lib/types.ts` records the absence so no
component can read it back by accident.

### Sprint
A maximal run with `v ≥ SPRINT_SPEED (5.5 m/s)` sustained for
`≥ SPRINT_MIN_S (0.8 s)` — `stats.py :: _count_sprints`.

### Territory
The share of **player-position samples** in each longitudinal third
(`stats.py :: territory`). Explicitly **not possession** — it says where players
were, which is honest and computable from positions alone. The code names this
and forbids the relabel.

### Contamination
The share of detections sitting under one track id whose dominant TRUE player is
somebody else (`stitch.py`). Measured on 118575 against SoccerTrack: **49.6%**
with neither the teleport split nor the team-gated stitch, **33.2%** with both (**CORRECTED 2026-09-28:** one match, 118575, tracks config unrecorded; 49.6% is the ungated stitch and raw tracks before stitching score 15.1%, per `ledger-vision.md`; at what ships `@C 35.0 :: n21-product-now (not published):87` (pooled ST2 sequences, after the identity stage)).

It is the number that governs how a per-player row may be read, and it is not
IDF1: IDF1 credits a player with the frames carrying its single most common id,
so a large impure id scores WELL for whoever dominates it. Fragmentation and
contamination trade against each other and both have to be read.

### Team shape
Width, depth, defensive-line height and compactness, computed from each player's
**time-averaged** position (`stats.py :: team_shape`) — a bounding box of
averages, not an average of instantaneous shapes. At 33.2% contamination (one match; see **Contamination**) a
"player's" time-averaged position is an average over more than one person, which
is why the between-team **paired difference** is the only shape figure quoted.

### Per 90
Total scaled to a 90-minute equivalent: `distance / minutes × 90`
(`app/frontend/src/lib/report.ts`). Totals reward minutes rather than output, so per
90 is the fairer scaling — but it is **not** a number to compare across players
here. At 33.2% **contamination** (one match; see **Contamination**) a third of each row's running belongs to
somebody else, so comparing two rows compares two blends. The UI warns when it
extrapolates from under 20 minutes; that is the smaller of the two problems.

---

## Validation and statistics

### Bland–Altman limits of agreement
The standard way to compare two imperfect instruments. Plot the difference
against the mean of each pair; report the **bias** (mean difference) and the
**95% limits of agreement** = bias ± 1.96·SD (`validate.py :: bland_altman`).

Reported instead of a bare MAE because **MAE folds sign away**: a pipeline that
is 4% high on every player and one that is randomly ±4% both score 4%, and only
the first is fixable with a calibration constant.

```
  diff %
    +9.4 ┤ ─────────── upper LoA ───────────
         │        ·   ·      ·
    +3.7 ┤ ·  ·  ─── bias ───  ·   ·
         │   ·        ·    ·
    −2.0 ┤ ─────────── lower LoA ───────────
         └──────────────────────────────────► mean of (GT, measured)
```

Note the method divides by the **pair mean**, so +400 on a 10200 mean and −400
on a 9800 mean do not cancel. That asymmetry is the method, not a bug — and
`validate.py :: _demo` asserts it.

### TEE — typical error of estimate
The residual SD from regressing the criterion on the practical measure,
expressed as a coefficient of variation (`validate.py :: agreement_stats`).
Rated on Hoppe et al.'s scale: **good <5%, moderate 5–10%, poor ≥10%**.

This is the headline statistic the optical-vs-GPS validation literature uses.
`STACK (not published)` notes it is **not a named function in any R or Python package** — the
field's real toolchain is Hopkins' spreadsheets plus hand-rolled stats, which is
why `validate.py` implements it.

### Lin's CCC — concordance correlation coefficient
Correlation **and** agreement in one number. Pearson `r` cannot see a constant
offset, so a pipeline reading 10% high scores `r = 1.0` and is still wrong.

### Calibration equation
The `slope` and `intercept` from regressing criterion on measured, reported so a
**user can correct our output** rather than just being told it is off.

### Standardised bias
Mean difference divided by the pooled SD — Hopkins' scale.

### FITTED vs held-out
`validate.py` bakes the distinction into code, not discipline. Any figure
computed on the tuning split is labelled `fitted=True` and is **structurally
barred** from being the headline; a report with no held-out matches prints
`NO HEADLINE FIGURE` and a `REFUSED` status.

**The gap:** the split is enforced per *report*, not per *constant*.
`stats.ACCEL_Q` was fitted on `118575_1st`, and `model/validation/run_v2` reports its
headline on "held-out" `118575_1st`.

### HOTA — Higher Order Tracking Accuracy
A tracking metric that balances **detection** accuracy and **association**
accuracy as a geometric mean, rather than letting one mask the other (as MOTA
does). Reported as `DetA` and `AssA` sub-scores. On a fixed camera the 2025
SoccerTrack winner reports **DetA 0.76, AssA 0.47** — detection solved, identity
not.

### GS-HOTA — Game State HOTA
SoccerNet's Game State Reconstruction metric: HOTA extended to require the right
player *identity and team*, not just a consistent arbitrary id. The field moved
**63.81 (2024) → 63.90 (2025)** — +0.09 in a year, on broadcast footage with
100–300 px players, using a vision-language model. This is the single strongest
piece of evidence that identity is not about to be solved for us.

### mAP-LocSim
Average precision scored on **metric distance on the pitch** rather than image
IoU, with τ = 1 m. From Spiideo's SoccerNet SynLoc devkit `sskit` (MIT). Adopted
as the detection metric because image IoU is meaningless when the product's
error budget is in metres.

### FIFA EPTS
FIFA's Electronic Performance and Tracking Systems certification. Worth knowing
what it does **not** cover: it certifies **instantaneous position and velocity
only** — not total distance, HSR, or sprint counts — and publishes **no numeric
thresholds at all**, only a five-band colour rating against an unpublished
industry distribution. There is no external bar to hit here; the Hoppe scale is
the field's convention, not a certification.

---

## Data and process

### `tracks.json`
The contract between detection/tracking and everything downstream:
`{"fps": float, "detections": [{f, id, team, jersey, x, y}]}`, where `x,y` is
the **feet** position in **image pixels**. Guarded by an explicit self-check in
`track.py :: _selfcheck`. Full field table in `pipeline-2026-09 (not published)`.

### `match.json`
The product output: meta, per-team totals and shape, territory, per-player rows,
and heat-map point lists (`stats.py :: build_match`).

### Golden file
`golden.json` plus `test_golden.py`: a regression tripwire on real tracks with a
**0.5% tolerance**, far tighter than the 5% accuracy gate on purpose — *"this
test is not measuring accuracy, it is measuring CHANGE."* A model bump, an
OpenCV upgrade or a filter retune shifts distances several percent with no
exception and no crash; this is what catches that. It currently runs on a
defective clock — see `pipeline-2026-09 (not published)` §5a.

### `ponytail:` comment
Marks a **deliberate** simplification, and by convention names the ceiling and
the upgrade path — e.g. greedy rather than Hungarian matching in
`tracking.py :: _greedy`, the whole match held in RAM in `track.py :: run`, the
scalar Python Kalman loop in `stats.py :: kalman_ground`. Read them as intent,
not oversight.

### Blocker 0 / 1 / 2 / 4
Shorthand from the design doc, referenced throughout the code:

| | |
|---|---|
| **Blocker 0** | No real footage. *Partly resolved* — real SoccerTrack panorama frames are on disk (`data/footage/README.md`). The full 48-min video that pairs with existing ground truth is still an incomplete download. |
| **Blocker 1** | Is the source camera perspective or panorama? *Answered: panorama* (`STACK (not published)`, `data/footage/README.md` §2). |
| **Blocker 2** | The jersey **legible-frame rate** — reported by `identity.py :: scan_video` as `legible_frame_rate`. Note it is the *heuristic's acceptance rate*, not the true readable rate; only hand-labelling ~300 real frames answers that. The code's note moved to `identity-evidence.md` on 2026-09-28. |
| **Blocker 4** | **Operator minutes per match** — how much hand correction a match needs. Instrumented as `operator_seconds` in `jobs.py :: apply_corrections` (an earlier estimate from a clusters-per-player script was withdrawn). |
