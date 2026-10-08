# Assumptions

> **Citation style.** `file.py :: symbol`, not line numbers — the codebase is
> under active refactoring. Verified against the tree as of 2026-08-04.

Every assumption the system makes, tagged **VERIFIED** (evidence named) or
**UNVERIFIED**. "Verified" means someone measured it, not that someone believed
it. Where a code comment asserts something a measurement contradicts, the
measurement wins and the comment is flagged.

Evidence quality:

- **REAL** — measured on real footage or real annotations
- **SYNTH-ON-REAL** — real trajectories, artificial degradation
- **SYNTH** — invented data or an invented camera rig
- **NONE** — asserted, never tested

---

## A. Camera and scene

| # | Assumption | Status | Evidence |
|---|---|---|---|
| A1 | **The camera is rigidly fixed** — no pan, tilt, zoom, cuts. | **VERIFIED (REAL)** | Phase correlation of a static background strip between frames 0 and 740 of real SoccerTrack footage: **(−0.014, +0.017) px** over 30 s (`data/footage/README.md` §4). This is the core technical bet and it held on real hardware. |
| A2 | **The calibration does not drift over a season.** | **UNVERIFIED** | A1 covers 30 seconds, not 90 days. The archived stack doc (not published) notes *"nothing published on sports calibration drift"* and proposes ECC drift correction — not implemented. Mitigated only by a deployment instruction: *"do not move the camera mid-season."* |
| A3 | **The pitch is planar.** | **UNVERIFIED** | The whole homography model depends on it (`pitch.py :: Calibration`, `sceneprior.py :: camera_from_homography`, `stats.py :: homography_jacobian`). Real pitches are crowned for drainage; nobody has measured the crown or its effect, and no code path detects a non-planar pitch. |
| A4 | **The image is a flat perspective render.** | **FALSIFIED for the only real footage on disk.** | SoccerTrack v2 is a 3-camera cylindrical stitch. One homography on the raw panorama: **median 5.35 m, p95 15.2 m, max 17.0 m** against 65 published keypoints; equivalently mean 116 px / median 46 px / max 428 px reprojection. Straight touchlines bow **169 px** across their length (`data/footage/README.md` §2). `README.md` still frames this as "still confirm what the source camera outputs"; for the footage we have, it is settled and the answer is no. |
| A5 | **The camera is mounted ≥ 6 m above the pitch plane.** | **UNVERIFIED as a threshold; the underlying curve is SYNTH.** | `groundpoint.py` docstring: stride-oscillation distance error 4.9–5.9% @4 m, 3.1–3.8% @6 m, 1.7–2.1% @10 m, ~0.8% @20 m. Its own caveat: *"synthetic, pinhole rig with no lens distortion, one running speed (6 m/s), and idealised \|sin\| foot lift. Re-run on real footage… before treating the 6 m threshold as anything more than a planning number."* `sceneprior.py :: Camera.mount_ok` enforces it from the calibration file, which is the right place. The real footage's own camera height is *"roughly 15–20 m, and this is an estimate, not a published figure"* (`data/footage/README.md` §4). |
| A6 | **Square pixels, zero skew, principal point at image centre.** | **UNVERIFIED (SYNTH only)** | `sceneprior.py :: camera_from_homography` states them as "all fine for a fixed broadcast-style rig". Verified only against a synthetic rig built with exactly those properties — a circular check. The real footage's shipped intrinsics have `fy = 80,777` vs `fx = 1,628`, i.e. wildly non-square, because a pinhole model does not describe a cylindrical stitch. |
| A7 | **Lower in the image = further from the camera** (the per-row size prior). | **FALSIFIED for panoramas** | *"A player near the near touchline appears near the image bottom at centre-x but near the image top at the left and right ends"* (`data/footage/README.md` §2). `sceneprior`'s per-row prior must run **after** a remap, never on raw panorama frames. Nothing in the code enforces that ordering. |
| A8 | **The whole pitch is in frame.** | **VERIFIED (REAL)** for SoccerTrack | Ground-truth x range −14.7 to 110.25 m on a 105 m pitch (`data/footage/README.md` §4). |

---

## B. Players and detections

| # | Assumption | Status | Evidence |
|---|---|---|---|
| B1 | **A standing player is 1.80 m tall** (`sceneprior.PLAYER_HEIGHT_M`). | **UNVERIFIED** | Used to build the ±40% height-plausibility gate. No per-player height, no distribution. A 1.60 m and a 1.95 m player are 11% and 8% off; the ±40% tolerance absorbs that, but the *tolerance* is doing the absorbing, not the model. **CORRECTED 2026-09-28:** the ±40% height gate is retired from the product (`track.py :: make_prior` ships the polygon only), so this assumption no longer feeds any shipped filter; it survives in `sceneprior.py` for the experiments that re-open the gate. |
| B2 | **A player is 20–50 px tall in frame** (`identity.HEIGHT_PX_FAR/NEAR`). | **FALSIFIED (REAL)** | Real measurement on the 4096×1080 panorama: **8–115 px** — far side 20–30, mid 30–50, near 70–115, a ~14× range in one frame (`data/footage/README.md` §3). Still hardcoded at 20/50, and it drives every torso crop and therefore every OCR attempt and every appearance histogram. |
| B3 | **The box bottom is on the ground.** | **UNVERIFIED (SYNTH evidence only)** | It is not: the lowest visible pixel during a run is an airborne foot, oscillating at `groundpoint.STEP_HZ = 3.0` with `FOOT_LIFT_M = 0.30`. The claim is not that it's true but that the filter suppresses it — SYNTH measurement gives box-bottom + Kalman at **+1.8%** vs boxcar at **+6.5%** at a 10 m mount, 50 m range (reproduced by running `groundpoint.py`). The module explicitly refuses to extend this to peak speed. The pluggable ground-point estimators have since been deleted; `track.py :: track` hardcodes `y = fy2`. |
| B4 | **A torso is ~0.42 body-heights wide**, and the shirt band is 20–50% of body height. | **UNVERIFIED** | `identity.py :: torso_box`, `track.py :: team_by_shirt`. Chosen, not measured. |
| B5 | **Pixel height is linear in image row.** | **UNVERIFIED, and contradicted by A7** | `identity.py :: torso_box`, flagged in-code as a `ponytail:` shortcut with the upgrade path named (project a 1.8 m vertical through the homography). On a panorama the relationship is not even monotonic. |

---

## C. Teams and identity

| # | Assumption | Status | Evidence |
|---|---|---|---|
| C1 | **Kit colours separate into 2 or 3 clusters in Lab space.** | **VERIFIED ON REAL FOOTAGE, AND IT IS NOT ENOUGH (2026-08-18)** | Scored against SoccerTrack's per-detection `team` labels on 118575: **70.7% per detection, 93.0% per track**, home/away detection ratio **1.18** against a truth of 1.0000 (`model/validation/team_eval.py`, re-run 2026-08-25; the 51.6% once quoted needed a retired `track.py`, and the mean-Lab readout through the current clustering scores 56.4%). So the clusters do separate, and nearly one detection in three still lands on the wrong side. `track.py :: assign_teams`. The archived stack doc (not published) cites *"0.13 same-kit vs 0.98 cross-kit"* — unnamed data, and it gave no hint that the shipped accuracy was 51.6%. PRTreID independently reports team affiliation at 97.6% rank-1 — at crops **~2-4× larger than our near band** (SoccerNet-Tracking, ~100×50 px; corrected 2026-08-25 from "10×"). The transferable finding is the SHAPE, not the crop size: PRTreID's own per-player identity mAP is **72.59** against team **92.89**, and under a team-only loss identity collapses to **12.34 mAP**. Team is nearly free from appearance; identity is not. That gap is our residual. The self-check exercises it only on hand-built Gaussian colour blobs. |
| C2 | **A third cluster smaller than 25% of tracks, and ≥0.5× the inter-team distance from either team centre, is the officiating crew.** | **UNVERIFIED** | `track.py :: assign_teams`. The code marks the thresholds as fixed and not learned and names the ceiling: *"a heavily reduced side or a keeper kit close to a team kit."* Two goalkeepers in a third kit plus 1–3 officials is 3–5 tracks — right at the boundary. |
| C3 | **Team is constant for a whole track.** | **MEASURED, AND THE GAP IS THE COST (2026-08-18)** | Assigned once from the mean colour over the track's whole life. A track that switches players mid-way gets one label — and the 93.0% per-track vs 70.7% per-detection split is exactly the size of that: the label is right for the track and wrong for a lot of the detections inside it. |
| C4 | **Players do not exceed 9.5 m/s** (`stats.MAX_SPEED`). | **UNVERIFIED** | *"A D3 field player tops out around 9 m/s"* — no citation. It is simultaneously a physical claim (peak speed), a filter clamp, a tracklet-merge gate (`identity.py :: _pair_costs`) and a data-quality signal (`clamped_frac`). Too low and real sprints are silently truncated; too high and ID switches leak into distance. |
| C5 | **A track break longer than 5 s is a substitution or a walk off-frame, not an occlusion** (`identity.MAX_GAP_S`). | **UNVERIFIED** | No measurement of real occlusion-duration distributions. |
| C6 | **A merge is safe iff it is mutual-best with a ≥0.12 cost margin.** | **PARTIALLY VERIFIED, and internally contradicted** | Purity held (0 impure groups) across every configuration measured on real trajectories fragmented synthetically (the archived stack doc (not published) identity table) — **but under a 5× distorted clock** (`pipeline-2026-09 (not published)` §5b), which makes the 9.5 m/s gate behave like 47.5 m/s and the 5 s gap window behave like 1 s. Meanwhile a comment in `identity.py` records a *contradicting* real-tracks result: *"the merger welded 3 groups containing two different players, and tightening MAX_COST/MERGE_MARGIN did not remove a single one… motion cost genuinely cannot tell 'same player continuing' from 'different player who was in that spot'."* Both results are in the repo; **they were reconciled on 2026-08-18 and the contradicting side won**: measured against SoccerTrack per-detection ground truth on real tracker output, 33.2% of detections sit under an id whose dominant true player is somebody else (49.6% before the 45 m/s split and the team gate). The "0 impure" result is an artefact of synthetic fragmentation — pieces of ONE true player were cut and rejoined, so no join could weld two people. **Status is now FALSIFIED on real fragments.** |
| C7 | **Dense-phase refusal prevents the remaining wrong merges.** | **NOT IMPLEMENTED** | The archived stack doc (not published) describes it as shipped architecture and the identity module called it *"the only fix."* It existed as a `_contested()` helper that was never called; that helper has since been deleted. There is now no dense-phase logic in the codebase at all. |
| C8 | **Torso appearance helps re-association.** | **FALSIFIED (SYNTH and SYNTH-ON-REAL)** | **Zero** extra merges at every density tested. An *oracle* per-player embedding recovers all 22 groups, so the merger uses signal fine — *"the signal isn't there. Do not budget for this helping."* |
| C9 | **Blending appearance into the merge cost helps.** | **FALSIFIED** | 352→22 became 352→36 in a crowd, because a near-equal term shrinks the first-to-second gap the ambiguity test depends on. Composed as a **union** instead. The archived stack doc (not published) generalises this: *"compose evidence by union or hard constraint, never by weighted sum"* — including for future jersey-number cues. |
| C10 | **Jersey OCR can read shirt numbers.** | **VERIFIED — REAL, and it FAILS AT THE SHIPPED GATE** | Scored through `jobs.apply_identity` itself against SoccerTrack's `jersey` field ([`n49-jersey-ocr.md`](../experiments/n49-jersey-ocr.md), 118575 frames 15000-29999, docTR 1.1.0, min_conf 0.6, 79.1% of the half not scanned, n=1): **21 numbers written over 681 groups, group precision 0.4375, detection-weighted 0.6768, 6 of 22 players right and 8 given a wrong number**. The whole error is one-digit reads (0 of 9 right; 7 of 7 two-digit right), and they arrive at confidence 1.0, so no `min_conf` separates them. The heuristic's acceptance on real footage is 0.3233, not the 40.8% synthetic ceiling. `jersey` is NOT `None` everywhere — the product writes these numbers to users today. |

---

## D. The measurement layer

| # | Assumption | Status | Evidence |
|---|---|---|---|
| D1 | **Detector box-bottom noise is σ = 1.0 px** (`stats.SIGMA_PX`). | **UNVERIFIED** | Never measured on real detections, and it sets the entire measurement covariance. Separately, `NOISE_TOLERANCE.md` says the accuracy target *requires* **0.05–0.22 px**, which is *"not achievable per-frame by any detector at any mAP"* — so the assumed value is 5–20× larger than what the gate needs, and the filter is what has to close the gap. |
| D2 | **Process noise q = 1.0 m/s² is the right tuning** (`stats.ACCEL_Q`). | **FITTED, not verified** | Explicitly not the physical ~6 m/s²; at q=6 errors run 37–98%. Fitted on `118575_1st`, and the constant's own comment says *"FITTED VALUE — the published number must come from held-out matches."* The report that quotes 3.8% nominates `118575_1st` as its held-out match. |
| D3 | **Measurement noise is anisotropic and Jacobian-derived.** | **UNTESTED in the regime it exists for** | All benchmarking used an **identity homography**, so the Jacobian was identity: *"the anisotropic term is untested. It only starts doing work once a real calibration is loaded, and that is exactly the regime (far touchline, 2.07 m/px) where it matters most."* |
| D4 | **Integrating filtered velocity removes the noise ratchet.** | **VERIFIED (SYNTH-ON-REAL)** | The boxcar path's bias is positive and monotonic in noise; the Kalman's is not. Kalman vs boxcar at σ = 0.05/0.10/0.20 m on held-out `117093_1st`: 5.5/4.3/2.4% vs 9.5/19.7/41.6%. Regression-guarded in `validate.py :: _demo`, which explicitly tests the degrader and the filter separately so "the degrader is broken" cannot be confused with "the filter works". |
| D5 | **Lower Kalman error at higher noise is a good sign.** | **FALSIFIED by the authors** | *"Kalman error decreases as noise rises, which is backwards. That is over-smoothing loss cancelling noise inflation — the same knife-edge cancellation criticised in the boxcar path… The honest claim is 'robust across a 10× noise range', not 'accurate'."* |
| D6 | **Detections arrive at a stable cadence.** | **VERIFIED in code, by construction** | The gap threshold derives from the track's own median step, not from `fps` (`stats.py :: speed_series`), with a regression test for sub-sampled tracks. A frame-rate-derived threshold previously zeroed those out. **But** the threshold is `max(3·step, 2/fps)` off the *median* — a track whose cadence is genuinely irregular (detection bursts around occlusions) has its slow stretches silently dropped from distance. |
| D7 | **The video's declared fps is correct.** | **UNVERIFIED, silently defaulted** | `cap.get(CAP_PROP_FPS) or 25.0`. Every time, speed and distance number scales linearly with this. A variable-frame-rate source breaks `t = f/fps` entirely. Nothing warns. |
| D8 | **The frame index `f` is the original video frame index.** | **VERIFIED as the convention, VIOLATED by two consumers** | `demo.py :: synth_tracks` honours it. `test_golden.py` and `measure_identity.py` both pass `fps / STRIDE` while leaving `f` un-renumbered, stretching the clock 5×. See `pipeline-2026-09 (not published)` §5. |
| D9 | **A gap in a track means the player's motion is unattributable.** | **VERIFIED as policy** | Gaps are excluded from distance, and the tracker emits nothing rather than a predicted box — inventing positions would turn an occlusion into fake distance covered. Consequence: distance is systematically **under**-reported by however much running happened during occlusions, and nothing quantifies that. Note peak speed and sprints do **not** apply the gap mask. |
| D10 | **Boxcar smoothing over 0.4 s is a valid alternative path.** | **VERIFIED AS UNRELIABLE** | *"It carries a large positive bias that partly cancels against the smoother's own negative bias, and relying on that cancellation is a knife-edge (sweeping the window 0.8–2.0 s swings the result from +16% to −18%)."* Kept only so the two can be compared on the same input. |
| D11 | **Tracks under 10 s are debris, not players** (`stats.MIN_MINUTES`). | **UNVERIFIED** | A genuine late substitute is fine; a player whose track fragments into 9-second pieces disappears entirely. Correctly shared with `validate.py` so both sides count the same population. |
| D12 | **`clamped_frac` is a usable proxy for tracking failure.** | **NO — the failure it misses is 33.2% of detections** | An ID switch between two adjacent players is a smooth handoff that never trips the clamp. Now quantified: on the shipped 118575 report the longest tracks read `clamped_frac` under 0.002 while identity contamination is 33.2%. A near-zero clamp share is evidence of no TELEPORT, not of a clean track, and the UI's 1% threshold therefore flags almost nothing. It is 0.0 for every entry in `golden.json`, which has never exercised the clamp (see D8). |

---

## E. Pitch geometry

| # | Assumption | Status | Evidence |
|---|---|---|---|
| E1 | **The operator types the real pitch dimensions.** | **DESIGNED FOR, NOT ENFORCED** | Every dimension is a constructor arg (`pitch.py :: Pitch`) precisely because hardcoding 105×68 is a **4–6% systematic error on every distance and heat map** — larger than the entire error budget. But the default is still 105.0×68.0, and `validate.py :: METRE_CAL`, `test_golden.py` and `measure_identity.py` all hardcode `Pitch(105.0, 68.0)`. |
| E2 | **Penalty box, goal box and centre circle dimensions are fixed by the laws of the game at every level worldwide.** | **VERIFIED (rules)** | Which is exactly why `pitch.py :: Pitch.landmarks` prefers them over the corners as calibration targets. |
| E3 | **Over-constraining the homography absorbs lens distortion at the frame edges.** | **UNVERIFIED** | `pitch.py :: Calibration.solve` uses LMEDS with ≥5 points and plain DLT at exactly 4. Tested only against a synthetic pinhole camera with one deliberately misclicked corner. Barrel distortion is *modelled* by `calibrators.py :: UndistortWrapper` and demonstrated on a synthetic barrel — but that class is not produced by anything in the pipeline. |
| E4 | **The pitch coordinate origin is the bottom-left corner.** | **VERIFIED (REAL)** for SoccerTrack | 99.49% of ground-truth positions fall in `0 ≤ x ≤ 105` and 99.39% in `0 ≤ y ≤ 68`; excursions are throw-ins, keepers behind the goal line and staff on the touchline (`data/footage/README.md` §1). `validate.py :: load_soccertrack` shifts their centre-origin coordinates by +52.5/+34.0 — which hardcodes a 105×68 pitch into the loader. |
| E5 | **A ±3 m margin around the pitch rectangle is the right playable-area mask.** | **UNVERIFIED** | `sceneprior.py :: pitch_polygon`. The real ground truth ranges from x = −14.7 to 110.25 m, so a 3 m margin would clip real throw-in and goalkeeper positions. |

---

## F. Service and data handling

| # | Assumption | Status | Evidence |
|---|---|---|---|
| F1 | **One job at a time is enough** (2–4 GPU-hours per match, one GPU). | **VERIFIED as a design constraint** | `jobs.py :: pump`, marked `ponytail:` with the upgrade path named. |
| F2 | **A stage artifact on disk proves the stage finished.** | **VERIFIED (self-check)** | A worker killed mid-tracking is resumed from the last artifact and driven to done (`jobs.py :: _selfcheck`). |
| F3 | **A pid alone does not prove liveness after a reboot.** | **VERIFIED (design)** | Requires a live pid **and** an `alive` heartbeat younger than `DEAD_AFTER_S = 45`. |
| F4 | **A whole match fits in RAM and in one JSON blob.** | **UNVERIFIED at full scale** | `track.py :: run` names the ceiling: ~135k frames × ~20 players ≈ a few GB and a ~400 MB file. The `tracks.json` in the repo is already 60 MB from a 5 Hz synthetic demo. The streaming-JSONL upgrade is blocked because team assignment needs every track's mean colour first. |
| F5 | **Container type can be decided from magic bytes.** | **VERIFIED (design)** | `app.py :: sniff`; the client's `Content-Type` is never trusted, the client's filename never becomes a path, and job ids are regex-validated in `jobs.py :: job_dir` before any filesystem access. |
| F6 | **Identity failure must never fail a match.** | **VERIFIED (self-check)** | Catches `(Exception, SystemExit)` — `SystemExit` specifically, because `scan_video` raises it on an unreadable video and it is a `BaseException`. Asserted with a deliberately corrupt mp4. |
| F7 | **Corrections are replayable from raw.** | **VERIFIED (self-check)** | Raw model output is never mutated; merge → undo → the second track reappears intact. |
| F8 | **A calibration accepted by the API can be consumed by the worker.** | **FALSE — live inconsistency** | `app.py :: set_calibration` validates through `calibrators.load` so a **panoramic** calibration is accepted (its comment says the panoramic path "is the normal one" for the real footage). `stats.py :: build_match` still loads through `pitch.Calibration.load`, which **raises on any non-perspective `projection_type`**. Posting a panoramic calibration returns 200 and then fails the job hours later — the exact failure the endpoint's docstring says it exists to prevent. **REFUTED 2026-09-28 (now TRUE):** `stats.py :: build_match` and `track.py :: run` both load through `calibrators.load`, and `app.py :: set_calibration` validates by calling `jobs.py :: load_calibration`, the worker's own load, which refuses `panoramic` outright. Accepted and consumable are one gate. |

---

## G. Legal and licensing assumptions (load-bearing, not incidental)

| # | Assumption | Status | Evidence |
|---|---|---|---|
| G1 | **No face or gait re-ID.** BIPA has a private right of action at $1,000–5,000 per violation and regulates *collection*, so inference-only is not an exemption. | **ENFORCED IN CODE** | Head region never cropped, embedded or stored (`identity.py :: torso_box`); motion-and-IoU-only tracking (`tracking.py`); kit colour is the only appearance signal the pipeline ever touches. Costs almost nothing on a fixed camera — there is no camera motion for a BoT-SORT-style compensation step to fix. |
| G2 | **AGPL cannot appear anywhere, even transitively** — it triggers on network use and this ships as SaaS. | **ENFORCED BY CONVENTION ONLY** | Ultralytics YOLO (all versions), BoxMOT, YOLOv9, YOLO-NAS weights and sn-gamestate are all excluded by policy. ByteTrack is **vendored** (MIT, ~60 lines) rather than depended on, partly because the pip package `trackers` collides with our own module name. Nothing enforces this at build time. |
| G3 | **SoccerTrack v2 data is CC BY 4.0 and commercially usable.** | **VERIFIED (REAL)** | Verbatim from `LICENSE-DATA`: *"for any purpose, even commercially"*; attribution required (`data/footage/README.md` §1). |
| G4 | **No commercially usable pretrained tiny-person detector exists.** | **VERIFIED (survey)** | Every tiny-person dataset is non-commercial; Roboflow re-uploads claiming CC BY 4.0 are third-party relabels with no authority to relicense (the archived stack doc (not published), "Legal traps"). |
| G5 | **`weights ≠ code licence`.** | **VERIFIED (survey)** | RF-DETR XL/2XL are PML 1.0, not Apache. DEIMv2's backbone weights are DINOv3 under Meta's bespoke licence. SoccerNet-derived weights are NDA-gated regardless of what an HF uploader tagged them. |

---

## H. Assumptions with no owner

Things the system relies on that nothing in the code, the docs or the deployment
guide addresses:

| Assumption | Why it matters |
|---|---|
| **Both teams keep attacking the same direction all match.** | `stats.attacking_directions` infers the direction from the keepers and forces the two sides opposite, so within one half it is measured (12/12 side-halves against SoccerTrack GT for 118575) rather than assumed. There is still **no half-time switch**: one direction per input file, so a file spanning half-time gives `defensive_line_m` measured from the wrong goal line for one of the two halves. |
| **Territory is meaningful when aggregated over both teams and all tracks.** | `build_match` passes the unfiltered `by_id`, including officials and sub-10 s debris, and both squads. With mirrored formations the result tends toward 33/33/33 regardless of play. |
| **Heat-map thinning by index approximates thinning by time.** | `stats.py :: heatmap_points` samples 600 evenly-spaced *indices*, so detection-dense stretches are over-weighted. |
| **The heat map and the distance number describe the same positions.** | They do not. `heatmap_points` uses raw `cal.to_pitch`; distance uses Kalman-filtered positions. |
| **`RemapCalibrator` output can be loaded back.** | **REFUTED 2026-09-28: it can** - `calibrators.py :: from_dict` has a dedicated `remap` branch and `jobs.py :: load_calibration` accepts it (see `pipeline-2026-09 (not published)` §4, corrected 2026-08-25). Original text: It cannot. `"remap"` is not in `calibrators._REGISTRY`, so `from_dict` raises `KeyError`. Its live `projection_type` attribute also reads `"abstract"` while its file says `"remap"`. |
| **A panoramic calibration produces a usable Jacobian.** | `stats.py :: kalman_ground` needs a single `cal.H`. `PanoramicCalibrator.H` is the **global** fit, not the per-strip transform it actually projects with — so the measurement covariance would come from a matrix the positions did not. |
| **The scene prior runs in production.** | **REFUTED 2026-09-28: it does, polygon only** - `jobs.py :: _pipeline` passes `calib=` to `track_run`, and `track.py :: run` builds `make_prior(cal, wh)`, whose default is the pitch polygon alone (the height gate is retired). Original text: It does not. `jobs.py :: _pipeline` calls `track_run` without `calib`, and in the service calibration is collected *after* tracking. The archived stack doc (not published) lists it as **BUILT**, which is true, and implies it is active, which is not. |
| ~~**Tracklet merging runs in pitch metres in production.**~~ | **This entry was itself wrong, corrected 2026-08-18.** It DOES run in metres: `_try_identity(d, st, cal)` takes the calibration as a required argument, its docstring says so explicitly, and a self-check in `jobs.py` asserts the strategy is constructed with it. The pixel-space failure this row described is real history — it produced ~125 fragments for 25 people — and it is fixed. |
