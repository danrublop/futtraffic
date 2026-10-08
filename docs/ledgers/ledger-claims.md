# Claims ledger

Every factual claim on a user-facing surface (README, docs, app), quoted
verbatim, against what actually measured it. Swept 2026-08-19.

**Status vocabulary.** SUPPORTED — a named, runnable, reproduced measurement backs
it. PARTIAL — measured, but narrower/weaker than the sentence implies. UNSUPPORTED
— no measurement either way. REFUTED — a measurement in this repo says otherwise.
UNREPRODUCIBLE — the cited artefact does not produce the number as invoked.

**The one-line finding.** The repo has split into two layers that disagree.
`docs/` and the frontend carry the 2026-08-18 corrections and are, with a handful
of exceptions, honest. The September 2026 README and the archived stack doc do not, and they were the
two files a reader opened first. Neither is published in this repository: below,
`README:<line>` and `STACK:<line>` point into those two files, and the verbatim
quotes stand on their own. (The current README carries no numbers.)

---

| Claim (verbatim, file:line) | Kind | Status | What actually supports it | Verdict |
|---|---|---|---|---|
| "No wearables, no human tagging: video in, distance / minutes / speed / sprints out, per player." `README:3-4` | capability | **REFUTED** by its own file | `README:199-201` "**Identity is human-seeded.** … the operator binds tracks to names once per match"; `STACK:44` "Human seeds 22 clicks at kickoff" | soften to "no wearables; one round of human seeding per match" |
| "It finds and follows every player. Hours for a full match" `app/frontend/src/app/page.tsx:300` | capability | **REFUTED** | `model/engine/detsweep.py:26-31`, 3,000 cached frames of the shipping config: band-0 recall 0.179 in metres / 0.458 in pixels — "**46% of far-touchline players have no box at all**" | delete "every"; soften to "finds and follows the players it can see" |
| "Nothing here has been validated against an independent reference on a real match." `README:146-147` | capability | **PARTIAL** — no longer true of everything | `metrics-2026-08 (not published):14-17`: team-assign and identity output *have* been scored against SoccerTrack per-detection GT on 118575 (`model/validation/team_eval.py`, `model/validation/contam_eval.py`, both self-check clean). Nothing in the distance chain has | soften to "nothing in the distance chain has" |
| "Identity: **1.0-1.7 clusters/player**, zero welded pairs at every density \| `measure_identity.py`" `README:276` | accuracy | **REFUTED** | `model/validation/contam_eval.py`: 33.2% of detections sit under an id whose dominant true player is somebody else at the shipped setting, 49.6% without the split and team gate. Withdrawn at `pipeline-2026-09 (not published)` :: §5b, "WITHDRAWN 2026-08-18", `metrics-2026-08 (not published):35`, [`docs/ledgers/assumptions.md`](assumptions.md) C6 ("**Status is now FALSIFIED on real fragments**") — but not here | delete the row; replace with the contamination figure |
| "\| Identity \| independent, 20 events \| 132 \| 22 \| **1.0** \| 0 \|" … "correlated, 60 events \| 572 \| 38 \| **1.7** \| **0** \|" `STACK:80-83` (the "impure groups" column) | accuracy | **REFUTED** | Same. `pipeline-2026-09 (not published)` :: §5b, "WITHDRAWN 2026-08-18": "the fragmentation was synthetic: pieces of ONE true player were cut and rejoined, **so no join could ever weld two people**". Also taken under a 5× stretched clock (`pipeline-2026-09 (not published)` :: §5b) | strike the "impure groups" column, stamp the table WITHDRAWN in place |
| "**Purity holds at every density.** Zero groups welded two players together, even at 572 fragments." `STACK:91-92` | accuracy | **REFUTED** | As above | delete |
| "welding corrupts both players' distance numbers while leaving both plausible and is undetectable downstream." `STACK:93-94` | capability | **REFUTED**, and by a tool this repo now owns | `model/validation/contam_eval.py` detects exactly this; its self-check scores a fully welded input at 50.0% | rewrite: welding is now measured, at 33.2% |
| "**~22-38 operator clicks per match — one to three minutes.**" `STACK:88-89` | capability | **UNSUPPORTED** — inherits the defect | `pipeline-2026-09 (not published)` :: §5b, "WITHDRAWN 2026-08-18": the operator-click estimate and the conclusion drawn from it **are unsupported until re-measured on real fragments** | delete until re-measured on real fragments; needs measurement M1 |
| "**Operator minutes per match** … estimated as clusters-per-player by `measure_identity.py`." [`docs/guides/glossary.md:379`](../guides/glossary.md) | capability | **PARTIAL** — points at the withdrawn estimator | `jobs.py :: apply_corrections` genuinely instruments `operator_seconds`; no run has produced a number | keep the instrument, drop the `measure_identity.py` estimate |
| "18 test cases (3 positions x 3 noise x 2 occlusion): MAE **86.3% -> 0.9%**, worst case 301% -> 2.5%, inside the 5% gate **1/18 -> 18/18**." `STACK:40` | accuracy | **REFUTED** | `NOISE_TOLERANCE (not published):107-108`: "An earlier internal estimate of 86.3% -> 0.9% for this change came from synthetic trajectories and **did not reproduce**"; `metrics-2026-08 (not published):43` lists it as **withdrawn** | delete the numbers, keep the design rationale |
| "~~Ground-plane Kalman with anisotropic `R`~~ **DONE.** 4.3% vs boxcar's 19.7% at σ=0.10, held out across matches." `STACK:213-214` | accuracy | **REFUTED / UNRELIABLE** | Those figures are the table in `NOISE_TOLERANCE (not published)`, whose own header reads "**Do not use any number in this file.** The entire measurement was taken at a 5x-stretched clock and does not describe the pipeline." (`NOISE_TOLERANCE.md:120-123`) | strike the numbers; "DONE" stands, the ratio does not |
| "Measured 0.13 same-kit vs 0.98 cross-kit." `STACK:43` | accuracy | **UNREPRODUCIBLE** | [`docs/ledgers/assumptions.md`](assumptions.md) C1: "`STACK` cites *'0.13 same-kit vs 0.98 cross-kit'* — **unnamed data**, and it gave no hint that the shipped accuracy was 51.6%." The shipped layer now measures **70.7% per detection / 93.0% per track** (`model/validation/team_eval.py:36-44`) | replace with 70.7% / 93.0% and name the dataset |
| "**Kit-colour histogram** … Beats every ReID embedding legally obtainable at this scale" `STACK:43` | comparison | **PARTIAL** — true of a different question | `model/validation/reid_eval.py:50-56`: SigLIP torso scores AUC **0.6113** on the *per-player* question, where box height alone scores 0.586. That is same-team separation, not team assignment. Kit colour was never scored against a ReID embedding on the same task | qualify: "for team assignment", not "beats every ReID embedding" |
| "PRTreID independently confirms our own measurement: team affiliation 97.6% rank-1" `STACK:54-55` | comparison | **REFUTED as an endorsement** | Ours is **70.7% per detection** (`team_eval.py`), ~27 points below the number cited to endorse it, and [`docs/ledgers/assumptions.md`](assumptions.md) C1 notes PRTreID reports it "**at crops 10× larger than ours**" | rewrite: PRTreID shows the ceiling is high; we are 27 points under it |
| "`sigma_px` **measured at 5.70 px**, versus the 1.0 shipped \| `measure_sigma.py`" `README:275` | accuracy | **UNREPRODUCIBLE as cited** | Run bare, `measure_sigma.py` defaults to `--detector stub` (`measure_sigma.py:120`) and prints "recall 0.0% (0/858 GT boxes matched) … no matches - cannot estimate SIGMA_PX" — **and exits 0**. The real provenance is `model/engine/stats.py:176-188`: RF-DETR tiled against TeamTrack, **recall 60.2%**, and the comment says outright that 5.70 px "is a **FLATTERING** number, not a conservative one" and that the clip is "a hovering drone at nadir, not the side-on fixed camera this product targets" | restate with detector, dataset, recall and geometry, or delete; make the bare run exit non-zero |
| "`measure_sigma.py` \| detector positional noise vs image-space ground truth \| yes" (self-check column) `README:242` | capability | **PARTIAL** | The self-check (`measure_sigma.py:288-299`) exercises the box matcher, not the sigma path. A green self-check coexists with 0/858 matches | note that the self-check covers matching only |
| "Identity — the thing gating reels — has no number at all today." `TODOS (not published):152-156` | capability | **REFUTED** | Four numbers exist: contamination 33.2%/49.6% (`contam_eval.py`), team 70.7%/93.0% (`team_eval.py`), appearance re-ID AUC 0.611 (`reid_eval.py`), association (`assoc_eval.py`) | rewrite; the reel gate now has a threshold to argue about |
| "IDF1, MOTA, and perspective tracking-error numbers in `model/validation/`" `TODOS (not published):148-149` | roadmap | **PARTIAL / stale** | `model/validation/contam_eval.py:4-12` argues IDF1 was "caught rewarding the exact failure it was supposed to detect" and that two real improvements were "measured, rejected on IDF1, and only shipped after being re-measured here" | drop IDF1 from the TODO or state why it is wanted anyway |
| "identity reaches ~9.1% sender accuracy against a 4.5% chance floor" `TODOS (not published):128-129`; "sender accuracy falls from 100% to 9.1%, against a chance floor of 4.5%" `app/frontend/src/app/checks/page.tsx:256-259`; same in `app/frontend/src/app/players/[trackId]/player-view.tsx:313-316` | accuracy | **SUPPORTED** | Measured by charging a perfect model with this pipeline's identity error; consistent across all three surfaces, and used to *withhold* a feature rather than to sell one | keep as-is — this is the model row |
| "Sender naming \| ours 86.2% \| paper 80.44%" `app/frontend/src/app/checks/page.tsx:133` | comparison | **PARTIAL — dangerous adjacency** | Measured on clean ground-truth tracks (`checks/page.tsx:204`), not on this pipeline. The same page, 120 lines lower, says our identity takes sender accuracy to 9.1%. A reader who stops at the table reads "we beat the paper at naming players" | put "on ground-truth tracks — 9.1% with our identity" in the row, not only in the prose below it |
| "Event detection F1 \| ours 0.6007 \| paper 0.7569" `app/frontend/src/app/checks/page.tsx:132` | comparison | **SUPPORTED** | Static, one fixed experiment, explicitly framed as transfer and explicitly not a pipeline claim (`checks/page.tsx:167-173`) | keep as-is |
| "Phase 1 has never been run on real tracker output over a full half." + the three other rows `app/frontend/src/app/checks/page.tsx:279-296` | capability | **SUPPORTED** | Self-declared gaps, set in the same type size as the evidence | keep as-is |
| "share of detections sitting under an id whose dominant true player is somebody else runs 49.6% … and 33.2% with both" `app/frontend/src/lib/report.ts:69` | accuracy | **SUPPORTED** | `model/validation/contam_eval.py`, per detection on a full half against SoccerTrack GT; self-check reproduces the sentinel cases | keep as-is. **CORRECTED 2026-09-28:** status is REFUTED, per `ledger-vision.md` (its "49.6% with neither" row): 49.6% is the *ungated stitch*, not a no-stitch baseline. Raw tracks score lower than the gated stitch + split on the same match (118575, tracks config unrecorded; figures in ledger-vision), so stitching raises contamination; at what ships it is `@C 35.0 :: n21-product-now (not published):87` (pooled ST2 sequences, after the identity stage). (was: SUPPORTED) |
| "two players in the same kit at 24–50 px of torso separate at AUC 0.61, where knowing the box height alone scores 0.586" `app/frontend/src/lib/report.ts:69` | accuracy | **SUPPORTED** | `model/validation/reid_eval.py:50-56`, with the time-gap confound explicitly controlled (`reid_eval.py:30-40`) | keep as-is |
| "it scores 70.7% per detection and 93.0% per track, and the home/away detection ratio is 1.08 where the truth is 1.0000" `app/frontend/src/lib/report.ts:89` | accuracy | **SUPPORTED** | `model/validation/team_eval.py:31-44`, 300 frames spread across the half, best-of-two label mappings stated as a spent free parameter, replicated at 70.8% on an independent 211-frame sample | keep as-is |
| "this class of method is 72.4–75.33% accurate against expert labels" `app/frontend/src/lib/report.ts:63` | comparison | **PARTIAL** | External literature figure with no citation anywhere in the repo | name the source or drop the range |
| "The held-out 1.2% (bias +1.1%, n=88) is not an accuracy figure" `app/frontend/src/lib/report.ts:42` | accuracy | **SUPPORTED** | The claim is a *de-claim*, and `README:149-162` documents the correction that produced it | keep as-is — best-in-repo |
| "the same true trajectory measures 1836 m at 25 Hz and 1481 m at 1 Hz" `app/frontend/src/lib/validation.ts:112` vs "The same reference tracks measure 627 m at 30 Hz and 554 m at 3 Hz" `README:174-175` | accuracy | **PARTIAL — two figures, one point** | Both are plausible bandwidth-dependence demonstrations on different tracks/rates, but neither cites a run and a club comparing the two pages sees inconsistency | pick one, cite the run |
| "Mark where the pitch is, once. Click four corners on a still from your video — about a minute." `app/frontend/src/lib/status.ts:49-51` | capability | **UNSUPPORTED** | `operator_seconds` is instrumented (`jobs.py :: apply_corrections`) but no run has produced a distribution | drop the duration or measure it; needs measurement M1 |
| "Two sides measured by the same instrument can be compared with each other. **Neither can be compared with a GPS vest or another vendor.**" `app/frontend/src/app/report/page.tsx:214-215` | comparison | **SUPPORTED** | `STACK:128` (Linke: ">40% deviation between commercial systems"), and no reference device has ever been run (`app/frontend/src/lib/validation.ts:102-105`) | keep as-is |
| "possession inferred from player movement alone is out by about **7** points … a half is only around **73%** live" `app/frontend/src/app/report/page.tsx:1494-1500` | accuracy | **SUPPORTED** | Measured: −1.12 pp with a ball-in-play mask, +6.97 pp without (`report/page.tsx:1460-1463`); the on-screen figure is the honest no-mask one | keep as-is |
| "at 6 m+ the box bottom is inside the gate \| `groundpoint.py`" `README:274`; "**Mount at 6 m or higher.**" `README:286`, `STACK:226` | accuracy / deployment | **PARTIAL** | [`docs/ledgers/assumptions.md`](assumptions.md) A5: "**UNVERIFIED as a threshold; the underlying curve is SYNTH**" — pinhole rig, one running speed, idealised foot lift, its own docstring asking for a real re-run | keep the instruction, label the threshold a planning number |
| "assuming 105x68 m is a 4-6% systematic error, larger than the whole budget" `README:288-289`, `STACK:228-229` | accuracy | **SUPPORTED** | Arithmetic over NCAA 115–120 × 70–75 yd; and [`docs/ledgers/assumptions.md`](assumptions.md) E1 notes the default is still 105.0×68.0 in three harnesses | keep, and fix the defaults E1 names |
| "**The camera is rigidly fixed: 0.014 px drift over 740 frames.** The core technical bet, confirmed on real hardware." `STACK:198-199`, `README:273` | accuracy | **SUPPORTED** | [`docs/ledgers/assumptions.md`](assumptions.md) A1, VERIFIED (REAL): phase correlation, (−0.014, +0.017) px over 30 s of real SoccerTrack footage | keep as-is |
| "single homography on the RAW panorama \| **5.35 m** \| 15.2 m \| 17.0 m" `STACK:176`; "remap-then-homography **6.17 m -> 2.36 m**" `README:271` | accuracy | **SUPPORTED** | REAL, against the dataset's own 65 published pitch keypoints; `metrics-2026-08 (not published):40` classes it REAL; [`docs/ledgers/assumptions.md`](assumptions.md) A4 marks the perspective assumption FALSIFIED on this footage on the same evidence | keep as-is |
| "**Player heights are 8-115 px, not the 20-50 we assumed.**" `STACK:191` | accuracy | **SUPPORTED** | REAL (`metrics-2026-08 (not published):42`) — and [`docs/ledgers/assumptions.md`](assumptions.md) B2 records that 20/50 is **still hardcoded** and drives every torso crop | keep; fix the constant |
| "Touchline bow **17.4 -> 1.5 px RMS** (model-free: a perspective image of a straight line is straight)" `README:272` | accuracy | **SUPPORTED** | Model-free geometric check on real footage | keep as-is |
| "\| `coreml` \| **2** \| **3.90** \| **5.1 h** \|" and "The two knobs are independent and they multiply; that was checked end to end on real footage, not assumed." `README:94-97` | accuracy (perf) | **PARTIAL** | Asserted as measured on a base M3 against real 118575 footage; no committed artefact, log or script reproduces the table | commit the timing run, or mark it an observation |
| "Identity is flat on contamination (.294 → .301)" `README:105-106` | accuracy | ~~**SUPPORTED**~~ **UNREPRODUCIBLE** (re-graded 2026-09-28) | `model/validation/contam_eval.py:11-12`: `--stride 2` "looked like a quality improvement on IDF1 and is really a wash". RE-GRADED 2026-09-28: a docstring sentence is not a runnable reproduction; `ledger-vision.md` row "TRACKING — `--stride 2` is free" found no harness that replays the shipped path at a stride, a leftover strided track file that scores far worse, and two stride tables (`track.py`, `detectors.py`) that disagree | ~~keep as-is~~ quote as a cited historical number only, until a stride harness exists |
| "band-0 image recall 0.328 → 0.759 … those boxes land 2.30 m from truth against a 2 m gate, so the product metric moves 0.636 → 0.657 while precision pays 0.868 → 0.812" `README:115-118` | accuracy | **SUPPORTED** | `model/engine/detsweep.py`, measured in both spaces per depth band on 3,000 cached frames | keep as-is |
| "**The accuracy lives in the measurement layer, not the detector.**" `STACK:8` | capability / strategy | **PARTIAL — superseded** | True of positional noise. But the binding constraint is now identity: 33.2% contamination (`contam_eval.py`) and 46% band-0 miss (`detsweep.py:31`) are not measurement-layer problems, and `reid_eval.py` says appearance cannot fix the first | rewrite: the accuracy lived in the measurement layer; the *product* now lives or dies on identity |
| "Aligning tile scale to the detector's trained object scale took a SynLoc team from **AP50:95 0.517 -> 0.878**." `STACK:38` | comparison | **SUPPORTED as an external cite** | External result; and `README:268` measures that on *our* remapped canvas tiling gives 12.0 det/frame and 0% reaching 22 — the external number does not transfer | keep, add the "does not survive our remap" caveat |
| "Two independent studies find coordinate jitter is near-identical across architectures and **uncorrelated with mAP**." `STACK:37` | comparison | **PARTIAL** | Two unnamed studies; no citation anywhere in the repo | name them |
| "arXiv `\"fixed camera\" AND \"player detection\"` returns **zero** results. This is unclaimed." `STACK:104` | comparison | **UNREPRODUCIBLE** | A search result at an unstated date, with no query artefact. Also the strongest-form claim in the file | soften to "we found nothing", date it |
| "**≤5% on total distance is a defensible public claim.**" `STACK:133` | accuracy | **UNSUPPORTED for this product** | It is a claim about what the *literature* permits (Hoppe et al., `STACK:131`), not about what we hit; `stats.py` (the banner above `SIGMA_PX`) still forbids quoting distance | reword as a target, never as a capability; needs measurement M2 |
| "FIFA EPTS certifies **instantaneous position and velocity only** … **publishes no numeric thresholds at all**" `STACK:118-120`; the Linke/Hoppe benchmark table `STACK:124-131` | comparison | **SUPPORTED as external cites** | Named authors and years; used to set the bar, not to claim it | keep as-is |
| The legal-traps list `STACK:139-156` and "**No AGPL anywhere**, deliberately" `README:306` | licensing | **SUPPORTED** | Each licence is checkable against upstream; the exclusions match the dependency set (`roboflow/trackers`, RF-DETR N/S/M/L) | keep as-is |
| "`model/engine/track.py` on real video is **new**. Treat first runs as debugging." `README:202` | capability | **PARTIAL — understated** | `app/frontend/src/app/checks/page.tsx:367-368` and `app/frontend/src/lib/validation.ts:95-97`: "track.py has **never been run end to end on real match footage**" | "new" → "never run end to end"; make it match /checks |
| "[`docs/ledgers/assumptions.md`](assumptions.md) tags all **50** assumptions VERIFIED or UNVERIFIED." `README:204` | capability | **REFUTED** (trivially) | 53 rows in [`docs/ledgers/assumptions.md`](assumptions.md) | fix the count or drop it |
| "There is no authentication anywhere in `app/server/app.py`. `GET /api/matches` returns every job on the machine" `TODOS (not published):10-12` | capability | **SUPPORTED** | Confirmed against `app/server/app.py`; framed as a deployment gate, not a task | keep as-is |
| The provenance table `metrics-2026-08 (not published):26-43` (SYNTH / SYNTH-ON-REAL / REAL per figure) and the three provenance problems `metrics-2026-08 (not published):45-57` | capability | **SUPPORTED** | Every row names its source file and split; it is the only place in the repo where each figure carries its evidence class | keep as-is — this is the template the September README and the archived stack doc should have been rewritten against |
| "**Status is now FALSIFIED on real fragments.**" [`docs/ledgers/assumptions.md`](assumptions.md) C6; "**WITHDRAWN 2026-08-18**" `pipeline-2026-09 (not published)` :: §5b | capability | **SUPPORTED** | The withdrawals are correct, dated, and name the falsifying measurement | keep as-is |
| "Identical `fps_eff = fps / STRIDE`. … With time stretched 5×" `pipeline-2026-09 (not published)` :: §5b; "It currently runs on a defective clock" [`docs/guides/glossary.md:357-362`](../guides/glossary.md); "both pass `fps / STRIDE` while leaving `f` un-renumbered" [`docs/ledgers/assumptions.md`](assumptions.md) D8 | capability | **REFUTED — the defect is fixed** | `model/engine/measure_identity.py:104` and `model/engine/test_golden.py:55` both read `fps_eff = fps`; `AUDIT_MODEL.md:296-304` says so and greps to prove it. Three docs still describe a live defect that no longer exists | update all three; the identity table stays withdrawn on the *synthetic-fragmentation* argument alone, which does not depend on the clock |
| "distance MAE @ σ=0.05 m \| **3.8% PASS**" `metrics-2026-08 (not published):28` | accuracy | **PARTIAL — de-claimed in place** | `metrics-2026-08 (not published):52` "**The 3.8% figure is fitted, not measured**"; n=22 against `MIN_PLAYER_MATCHES = 60`, stamped UNDERPOWERED; both sides through the same estimator, "a lower bound on error" | keep only with its three caveats attached; never quote the 3.8% alone |
| "**On the shipped constants … distance agreement is `MAE 7.4%, bias −8.0%` (n=88)**" `AUDIT_MODEL.md:28-30` | accuracy | **PARTIAL** | The best distance figure in the repo — real homography, held-out TeamTrack, zero injected noise. But produced by `scratch/exp1.py`, which is not committed, so nobody else can re-run it | promote it to a committed harness, or it will be retracted like its predecessors |
| "**SoccerTrack should be retired as a distance reference outright**: 88% of its consecutive GT frames are zero-displacement … It cannot validate a 5% distance claim" `AUDIT_MODEL.md:393-397` | capability | **SUPPORTED** | `AUDIT_MODEL.md:112-123` (exp7/exp9), and it is the root cause of the retracted −26% | keep; act on it — it invalidates the reference behind several surviving rows |
| "**Your ±5 % distance gate is roughly the *inter-system* disagreement of the professional market.** … If you land at ±5 % against a reference you are *at* the state of the practice for optical …" `RESEARCH_MODEL.md:388-393` | comparison | **UNSUPPORTED — conditional, and the condition is unmet** | The comparison table (Linke 2020, Crang 2025, Makar 2023, Buchheit 2014) is properly cited; the "if you land at" has never been tested | keep the citation, keep the "if"; this sentence must never be quoted without it |
| "**TRACAB Gen5** … measures **0.08 m pRMSE**" … "Your 0.10 m budget is asking a single 1080p camera to be within 25 % of an eight-camera stadium rig's *best-case bench number*." `RESEARCH_MODEL.md:297-302` | comparison | **SUPPORTED** | Linke/Link/Lames PLOS ONE 2020, and the sentence argues *against* our own budget | keep as-is |
| Calibration accuracy against real players (`model/engine/calcheck.py`, ~1.010 m over 8,679 player samples) | accuracy | **SUPPORTED but unpublished** | Measured, and it is *better* than the in-sample RMS the UI shows. See `ledger-output (not published)` for the shipped-number defect | surface it; this is a row where the product undersells itself. **CORRECTED 2026-09-28:** the figure is scoped to 118575's vendor remap with its 65 dataset keypoints, which the product cannot produce (the operator flow offers 21 dots; auto_calibrate writes `spherical`). It is not the product's calibration, so it does not undersell it (was: "the product undersells itself"). The clicked-homography product path scores `@C 7.12 m :: docs/experiments/n31-line-registration.md §4` (pooled median at feet, column SHIP, one install) |

---

## Claims retired from a summary page, 2026-09-28

A since-unpublished summary page carries no numbers now; the reasons live here.

- **Team accuracy (per detection / per track) and IDF1** — removed. Untagged,
  one match (118575), research config, and superseded. Registry value:

      @M 0.6967 :: .venv/bin/python model/validation/api.py --get team.accuracy.detection
         118575 1st half, 300 frames, n=4101 detections, research config (conf 0.3,
         stride 1) per n8-n11-downstream (not published) N9.

  Home/away naming is at chance: n8-n11-downstream (not published) N9.
- **"a median 96 fragments" per player over a half** — removed. Untagged; no
  run named.
- **"Roughly one in three observations attached to the wrong player"** — reworded
  without a number. It is the contamination figure, whose only measurement is on
  one match's `tracks.split.json` with an unrecorded config
  (model/validation/contam_eval.py N8 docstring), not the shipped configuration.
- **Still carrying an unscoped quantity:** "At the far touchline the system
  currently misses close to half the players." Left open.
- **Ball bullet, CORRECTED 2026-09-29.** Original: "**The ball is not tracked yet.**
  Ball detection does not work today. We know the route to fixing it and it mainly
  needs a training machine." Both halves were stale: the job path has run the stock
  detector's ball since 66fba82 (`app/server/jobs.py :: _pipeline` passes `ball=True`),
  and the training is done - N52 arm D1 exists
  ([N52](../experiments/n52-clean-ball-data.md)) and is not wired because it needs a
  second detection pass ([ball reference](../reference/ball.md#limits)). Replaced with
  a reworded line, no numbers. The level that ships:

      @M 0.2623 :: .venv/bin/python model/validation/api.py --get product.ball.correct.s2
         9 TeamTrack clips, 2 matches, stride 2, shipped merge and thresholds, stock weights.

  Nothing downstream reads the ball rows ([ball reference](../reference/ball.md#consumers)).

## Four places the docs contradict each other

A reader of two of these files gets two answers. Each of these is a
one-line fix.

| Question | One file says | Another says | Truth |
|---|---|---|---|
| Is the 5× clock live? | `pipeline-2026-09 (not published)` :: §5b, [`docs/guides/glossary.md:362`](../guides/glossary.md), [`docs/ledgers/assumptions.md`](assumptions.md) D8: yes | `AUDIT_MODEL.md:296-304`: fixed | **Fixed** — verified: `measure_identity.py:104`, `test_golden.py:55` both `fps_eff = fps` |
| Does a panoramic calibration crash the worker? | [`docs/ledgers/assumptions.md`](assumptions.md) F8: "**FALSE — live inconsistency**… returns 200 and then fails the job hours later"; `pipeline-2026-09 (not published)` :: §4, `PanoramicCalibrator` row agrees | `pipeline-2026-09 (not published)` :: §4, "The live inconsistency — RESOLVED": "**RESOLVED**… no longer occurs" | unresolved in the ledger — needs one grep and one edit. **RESOLVED 2026-09-28:** it does not crash; `panoramic` is refused at POST by `jobs.py :: load_calibration`, and `track.py :: run` / `stats.py :: build_match` both load via `calibrators.load`. F8 is marked REFUTED in `docs/ledgers/assumptions.md` |
| Is the scene prior active in production? | [`pipeline-2026-09 (not published)` :: §2 [C], "Active" row](pipeline.md): "**Active** (corrected 2026-08-13)" | [`docs/ledgers/assumptions.md:135`](assumptions.md): "It does not"; [`docs/guides/glossary.md:191`](../guides/glossary.md): "built and not active" | unresolved. **RESOLVED 2026-09-28:** active, polygon only - `jobs.py :: _pipeline` passes `calib=` to `track_run`; `track.py :: make_prior` defaults to the polygon ("ONLY THE POLYGON SHIPS"). `docs/ledgers/assumptions.md` corrected; `docs/guides/glossary.md:191` still reads "not active" |
| Is tiling implemented? | [`pipeline-2026-09 (not published)` :: §2 [B], "Tiling" row](pipeline.md): shipped, with a 500-frame measurement | `pipeline-2026-09 (not published)` :: §6, "Tiled inference: SAHI" row, [`docs/guides/glossary.md:173`](../guides/glossary.md): "**Not implemented**" | shipped — the "not implemented" rows are stale |

---

## Measurements that would close the open rows

- **M1 — operator clicks and minutes on real fragments.** Re-run the click budget
  on real tracker output, not synthetic fragmentation of one player, at the true
  clock. This is the number the operator-click claim rests on
  (`STACK:88-89`), and it is currently the withdrawn one.
- **M2 — total distance against an independent instrument on one real half.**
  Until this exists, `STACK:133`'s "≤5%" is a literature target and
  `stats.py` (the banner above `SIGMA_PX`) is right to forbid quoting distance.
- **M3 — `sigma_px` on side-on fixed-camera footage at full recall.** The shipped
  5.70 px is a nadir-drone number at 60.2% recall that the code itself calls
  flattering.

---

## The five that would do the most damage if a reader checked them

1. **`STACK:88-89` — "~22-38 operator clicks per match."** Resting entirely
   on a purity column this repo has already falsified. A reader who asks "how many
   clicks, actually?" gets no answer, and `pipeline-2026-09 (not published)` :: §5b, "WITHDRAWN 2026-08-18" is on record
   saying so.
2. **`README:276` and `STACK:78-94` — "zero welded pairs at every density",
   "welding … is undetectable downstream."** Contradicted by `contam_eval.py` at
   33.2%, by four sibling files that already carry the withdrawal, and — worst —
   the frontend says 33.2% while the README said zero.
   The same app shipped both sentences.
3. **`app/frontend/src/app/page.tsx:300` — "It finds and follows every player."** The
   front door of the app, refuted by `detsweep.py`: 46% of far-touchline players
   have no box at all. This is the one a coach falsifies by watching the overlay,
   in about a minute, without asking anyone.
4. **`STACK:54-55` — "PRTreID independently confirms our own measurement:
   team affiliation 97.6% rank-1."** An external number cited as an endorsement
   of a layer that measures 70.7%. A reader who looks up PRTreID finds the gap
   and concludes the citation was chosen for its size.
5. **`README:275` — "`sigma_px` measured at 5.70 px \| `measure_sigma.py`."**
   The named script, run as named, matches 0 of 858 ground-truth boxes, prints
   "cannot estimate SIGMA_PX", and exits 0. Anyone who tries to reproduce a
   headline number and gets a silent zero stops trusting every other row in that
   table.
