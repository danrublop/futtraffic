"""DISTANCE: the headline number, against ground truth, for the first time.

WHY THIS EXISTS
    `stats.py` marks distance `unvalidated_end_to_end` and has since the day it
    was written. Every other harness here measures TRACKING - `detsweep.py` for
    detection, `assoc_eval.py` for association, `contam_eval.py` for identity,
    `team_eval.py` for sides. All of them stop at "whose box is this". None of
    them ever asked whether the METRES are right, which is the only number a
    coach reads. The one measurement that looked like an answer - "distance MAE
    1.2%, bias +1.1%" - is agreement between the estimator and ITSELF on
    perturbed input, because `validate.player_metrics` puts both sides through
    `stats.player_stats` at the same window (see the retraction at
    DISTANCE_WINDOW_S). This file uses a reference that did not come out of our
    pipeline.

WHAT IT CAN AND CANNOT ESTABLISH. Read this before quoting anything below.
    CANNOT: this is not validation against an independent instrument. There is
    no GPS, no LPS, no surveyed target anywhere in this repo. SoccerTrack GT is
    HUMAN ANNOTATION of the same footage, carrying its own error, and one error
    in particular is large and known: positions are quantised to a 100x100 grid
    over the pitch, 1.05 m in x and 0.68 m in y (verified in `_selfcheck`, not
    taken on trust). Uniform quantisation of that size is sigma 0.30 m / 0.20 m
    of pure noise on every annotated point, and raw sum-of-deltas over it
    asserts ~14 km per player per match. So GT is NOT a distance oracle; it is a
    second imperfect instrument that happens to be independent of ours.

    CAN: whether our metres are systematically high or low against that
    reference, by how much, and whether the error is a scale (one number, common
    to the squad - calibration or filter) or a scatter (per-player, which is
    what fragmentation and contamination produce). And it can settle whether the
    peak speeds pinned at MAX_SPEED are a clamp problem or teleporting tracks,
    because GT peak speed is computable and ours is not the only estimate.

THE THREE THINGS THAT WOULD MAKE THIS NUMBER A LIE, AND WHAT IS DONE ABOUT EACH

  1. FILTERING. Distance is a sum of displacements, so it AMPLIFIES noise: an
     unsmoothed trajectory reads high by an amount that depends on its noise
     level, not on the player. Integrating raw GT against our Kalman-filtered
     output would "discover" a deficit exactly the size of our own filter.
     So GT is pushed BACKWARDS through the calibration homography into
     pseudo-pixels and run through `stats.speed_series` and `stats._windowed` -
     the shipped functions, imported, not reimplemented - so both sides get the
     same Kalman, the same Jacobian-derived R, the same 0.4 s phase-averaged
     chord window. `--gt-method raw` shows what skipping this would have
     claimed; the gap between the two is reported in every run.

  2. FRAGMENTS. We emit ~361 ids for 22 players. Each id is attributed to a GT
     player with `contam_eval.owners` (nearest GT inside 2.0 m, per detection,
     majority owner per id) and our distance is summed over every fragment that
     lands on that player.

  3. COVERAGE, WHICH IS THE ONE MOST LIKELY TO PRODUCE A CONFIDENT WRONG ANSWER.
     A fragment set covering 60% of a player's frames under-reads distance by
     roughly 40% however good the tracking is, and that is a RECALL finding
     wearing an accuracy costume. Two defences. First, coverage is printed
     beside every ratio and a ratio is never quoted without it. Second, and this
     is the number to actually read: `gt_matched` re-scores GT with OUR coverage
     as its gap mask, so the same windows are dropped on both sides and the
     ratio no longer contains recall at all. `ratio_raw` (against full GT) is
     what the product would ship today; `ratio_matched` is what the estimator is
     worth where it can see.

N10 / N11 (2026-09-12): DISTANCE AND THE SPEED STATS, AT THE PRODUCT CONFIG
    `api://product.distance.ratio_raw`, `.coverage`, `.ratio_matched` and
    `api://product.player.distance.spread` all come off ONE job directory whose
    tracks file predates the current tracker (model-map (not published) §3), and
    `api://product.speed.peak_ms` is held out of the golden for the same reason.
    The sprint/HSR numbers in speed-clamp (not published) §5 were never registered at
    all, and that document says its own headline is known wrong. `--stc`
    re-measures all of it on the tracks `app/server/jobs.py` produced.

    PROTOCOL, fixed before the first run:

    population   the nine STC2025 clips with product tracks on disk; 132831 is
                 excluded because its ST2 chain cannot be clicked at all
                 (st2_product.py). Four minutes each, not a half.
    reference    ST2's OWN dense feed, `raw/<m>/<m>_tracker_box_data.xml`, in
                 corner-origin pitch metres (normalised loc x 105, 68), aligned
                 to clip frame 0 by the offset+permutation join in
                 `st2_team_gt.py` and gated by its `_ok`. VERIFIED, not assumed:
                 `_selfcheck_stc` reprojects those metres through ST2's own
                 vendor chain and requires them to land within 25 px of the MOT
                 GT feet, which pins the offset, the permutation and the
                 corner-origin convention in one assertion.
                 It is the SAME 1% grid as the SoccerTrack gz - 1.05 m in x,
                 0.68 m in y - so everything the header says about this being a
                 second imperfect instrument rather than an oracle still holds.
    held fixed   coreml r576 t1024, conf 0.5, stride 2, the clicked calibration
                 (`{seq}.p05.calibration.json`, 5.8-6.3 m RMS), scene prior,
                 metric stitch. `.p05` and not `.p05id`: N8 measured the
                 identity stage merging zero tracklets on all nine, so the two
                 files carry the same geometry.
    attribution  PIXELS. `contam_eval.owners_iou` - Hungarian on IoU >= 0.5
                 against ST2's MOT boxes - for the same reason as N8: a 2 m
                 metre gate under a 6 m calibration measures the calibration.
                 A fragment is credited to its MAJORITY owner and EVERY metre it
                 ran is credited there, so N8's 37.3% per-detection
                 contamination is the size of what this attribution is papering
                 over. Reported beside the ratio, every time.
    frames       both sides on the EVEN frames only - the ones a stride-2
                 tracker runs on. GT sampled on the same cadence, or the
                 coverage mask (`have[:-1] & have[1:]`) is False at every step
                 and `gt_matched` collapses to zero by construction.
    estimator    identical on both sides: GT metres pushed BACKWARDS through the
                 product's own H into pseudo-pixels (`to_px`, round-trip
                 asserted), then `stats.speed_series` and `stats._windowed` -
                 the shipped functions, imported. Integrating raw GT against our
                 filtered output would discover a deficit the size of our filter.
    metrics      N10: per-row (per GT player) `ratio_matched`, side-summed team
                 ratio, each beside its coverage; `ratio_raw` is never quoted
                 without `coverage` next to it. Floor: 1.000 is unbiased, and
                 the live control `--control 0` must read 1.00 at coverage 1.00
                 or nothing else in the run means anything.
                 N11: HSR metres (v >= stats.SPRINT_SPEED over observed steps,
                 `validate.py`'s definition) and sprint count
                 (`stats._count_sprints`, the shipped one) against GT re-emitted
                 on exactly the frames we held, plus the same settle-window arm
                 speed-clamp (not published) §5 used - v := 0 for `settle_s` after every
                 gap and at every fragment start, applied to BOTH sides.
    pass rule    fixed before running, and separately for each:
                 N10 is reportable iff `--control 0` reads 1.00 and the nine
                 clips agree on the SIGN of the bias; if they disagree in sign
                 there is no product distance number, only nine of them.
                 N11 PASSES - the §5 defect reproduces at the product config -
                 iff the HSR and sprint ratios fall monotonically as `settle_s`
                 rises AND the settle-1.0 s ratio is closer to 1 than the
                 shipped 0.0 s one. If they do not move, the 1.50x/2.06x is a
                 property of the old job's fragments and not of the estimator,
                 and speed-clamp (not published) §5 is wrong in a second way.
    drop share   on every row: coverage in frames, the orphan fragments whose
                 metres are emitted for nobody, and N8's contamination as the
                 attribution's own error bar.

    python model/validation/dist_eval.py --selfcheck
    python model/validation/dist_eval.py --tracks tracks.split.json
    python model/validation/dist_eval.py --grid          # every shipped config
    python model/validation/dist_eval.py --stc           # N10/N11, the nine clips

N22 (2026-09-13): WHAT INFLATES DISTANCE AND SPRINTS AT WHAT SHIPS
    N21 re-measured N10/N11 on the product as it ships (`.p21`,
    n21-product-now (not published)): each covered metre over-reads by about a third,
    sprints read 4.5x GT, and the settle window that explained N11 on the old
    tracks no longer does. Two mechanisms make a fragment run too far - its
    detections belong to more than one player (IDENTITY), or they sit in the
    wrong place (POSITION: pixel jitter through the ~8 m clicked map,
    n12-calibration (not published)) - and which one it is decides the next product
    project. This apportions the over-read by replacing one with the truth at a
    time.

    PROTOCOL, fixed before the first run:

    population   N21's nine clips, `.p21` tracks, stride-2 frames, N6's
                 calibration, the shipped settle - exactly N10/N11's inputs.
    owner map    ONE, computed on the shipped rows by `contam_eval.owners_iou`
                 and held fixed in every arm: moving a detection's position
                 moves its box, and re-attributing would move the answer.
    arms         shipped   `.p21` as N10/N11 score it.
                 drop      shipped minus every detection that matches no GT box.
                           No truth exists to substitute for those, so every
                           oracle arm drops them, and THIS is their baseline.
                 identity  drop, and each fragment split wherever its owner
                           changes: our positions, true identities.
                 position  drop, and each detection moved to its owner's ground-
                           truth position on that frame: our identities, switches
                           included, true positions.
                 both      identity and position together. What is left is
                           duplicate ids on one player and the estimator's own
                           fragment edges, which neither oracle touches.
    metric       pooled `ratio_matched` PRIMARY - metres per covered metre - with
                 the pooled sprint and HSR ratios beside it. n=5 configurations,
                 nothing tuned.
    share        per oracle, the part of the drop arm's excess over 1.0 that it
                 removes: (drop - oracle) / (drop - 1).
    rule         an over-read's driver is IDENTITY iff the identity share is at
                 least 0.5 and the position share is below 0.5; POSITION by the
                 mirror; BOTH if both reach 0.5; NEITHER ALONE if neither does.
                 Applied to all three metrics; the decision reads ratio_matched.
                 IDENTITY points the next project at association (stitch.py's
                 cost, the identity stage), not calibration; POSITION at the
                 calibration project; the other two at no single project.
    failing      NEITHER ALONE is a result, not a failed run: the over-read
                 needs both fixes, or a mechanism this does not model.
    drop share   the drop arm's own movement on every metric, beside shipped;
                 `unattributed_frac` on every arm.

    REVIEW, 2026-09-13, after the run - the rule above is left as committed:
                 the identity arm split once per (fragment, owner), not wherever
                 the owner changed, as stated; fixed and re-run. And its share
                 measures the scorer's CREDITING, not path length: a fragment is
                 credited whole to its majority owner, so its minority-owner
                 frames are covered for nobody, and the identity oracle raises
                 the covered-GT denominator while our own metres barely move.
                 docs/experiments/n22-what-inflates-distance.md carries both.

    python model/validation/dist_eval.py --n22           # N22, five arms on .p21

N23 (2026-09-13): WHICH PART OF "POSITION" - THE DETECTOR OR THE MAP
    N22 found position errors carry about three quarters of the sprint and HSR
    over-count at what ships, and could not say whose: its position oracle
    replaced the detector's foot points and the clicked map's geometry at once.
    N23 replaces them one at a time. The steps telescope, so the parts sum to
    N22's position share exactly.

    PROTOCOL, fixed before the first run:

    population   as N22: nine clips, `.p21`, stride-2 frames, N6's calibration,
                 the shipped settle, one owner map computed on the shipped rows.
    arms         drop      N22's baseline.
                 gtpix     each detection moved to its owner's MOT ground-truth
                           foot in clip pixels (`st2_team_gt.mot_feet`), then run
                           through OUR clicked map: the detector's error gone, the
                           map's kept.
                 feedpix   each detection moved to its owner's position-feed
                           metres pushed through ST2's own vendor chain into clip
                           pixels (`st2_overlay.projector`), then through our map.
                 position  N22's arm: the owner's true metres, the map's error
                           gone as well.
    shares       of the drop arm's excess over 1.0:
                   detector    (drop - gtpix)       / (drop - 1)
                   annotation  (gtpix - feedpix)    / (drop - 1)   ST2's boxes
                               against ST2's own feed through its own chain - a
                               property of the reference, not of the product
                   map         (feedpix - position) / (drop - 1)
                 which sum to N22's (drop - position) / (drop - 1).
    metric       pooled sprint ratio PRIMARY - the metric N22 found position
                 drives - with HSR and distance per covered metre beside it.
                 n=4 configurations, nothing tuned.
    rule         INCONCLUSIVE if the annotation share is more than a third of the
                 position share: the two ground truths disagree too much to split
                 what is left. Otherwise DETECTOR iff the detector share is at
                 least two thirds of the position share, MAP iff the map share
                 is, MIXED if neither.
                 DETECTOR points position work at foot points and track
                 smoothing; MAP at the calibration project; MIXED at both.
    drop share   `unattributed_frac` on every arm: a row whose owner has no MOT
                 foot, or no valid vendor pixel, on that frame is dropped - and
                 that is counted, not hidden.

    REVIEW, 2026-09-13, after the run - the rule above is left as committed:
                 the committed code reads the annotation guard as a MAGNITUDE,
                 which the text does not say; distance's INCONCLUSIVE rests on
                 that reading and would be MAP on a signed one. The detector
                 share covers detections matched at IoU >= 0.5 only. The map
                 share is our map against ST2's own projection, nearest-pixel
                 table lookup included, not against the pitch.

    python model/validation/dist_eval.py --n23           # N23, four arms on .p21

N24 (2026-09-13): DISTANCE CREDITED PER DETECTION, BESIDE PER FRAGMENT
    Review of N22 found the scorer's crediting inside what `ratio_matched`
    measures: a fragment is credited whole to its majority owner, so the frames it
    spends on anyone else count against the wrong player, and N22's identity
    oracle moved the ratio by moving its denominator while our own metres barely
    changed (docs/experiments/n22-what-inflates-distance.md §3). Two questions have been
    sharing one number:
      ROW      how close is a player row's distance to that player's - the
               shipped field's question. A row carries its identity errors, so a
               whole fragment is credited to its majority owner: N10 as it is.
      RUNNING  on metres the tracker genuinely followed one player, how far does
               the estimator over-read - each STEP credited to the player both of
               its ends belong to.
    N24 adds RUNNING and re-reads N10, N11, N22 and the registered spread with it.

    PROTOCOL, fixed before the first run:

    crediting    `credit="detection"`: each fragment goes through the product's
                 estimator ONCE, whole. For each ground-truth player, the steps
                 whose two ends are not both that player's are masked exactly as
                 `_windowed` masks an unobserved step, so a window holding one is
                 dropped for that player. A switch step and an unowned detection
                 are credited to nobody, and those metres are reported. The
                 ground-truth side is unchanged: that player's metres over the
                 frames credited to them.
    population   (a) N10/N11 on `.p21id`, the shipped settle - what ships;
                 (b) N22's five arms on `.p21`, N22's population.
    held fixed   everything N10, N11 and N22 held fixed; one owner map per clip.
    validity     VOID unless (i) the GT-as-tracks control reads 1.000 at coverage
                 1.000 under detection crediting, and (ii) under detection
                 crediting the identity oracle moves pooled `ratio_matched` by at
                 most 0.02 from the drop arm. If splitting fragments by owner still
                 moves the ratio, the crediting still depends on fragment identity.
    metrics      pooled `ratio_matched`, sprint and HSR ratios and the per-player
                 spread under BOTH creditings, and the share of our metres
                 credited to nobody. n=2 creditings per variant, nothing tuned.
    what follows at what ships: RUNNING `ratio_matched` at most 1.05 means the
                 estimator is accurate on the metres it genuinely tracks, and the
                 shipped over-read is identity bookkeeping - association work.
                 Above 1.05 there is a physical over-read, and position work (N23:
                 the map, for HSR) is the lever. Distance only.
    registry     `product.player.distance.spread` STAYS on row crediting: it
                 justifies a field that reports rows, and scoring it on metres a row
                 does not report would flatter it. RUNNING is registered as its own
                 id, `product.distance.ratio_matched.running` - a diagnostic of the
                 estimator, never a field's justification.

    python model/validation/dist_eval.py --n24           # N24, both creditings

N25 (2026-09-14): THE SPEED ESTIMATOR - FLOODLIGHT'S FILTER AGAINST OURS
    N23 found that every input read off the image through our map - our
    detections, MOT's feet, ST2's own feed - reads about the same HSR, and only
    true metres bring it down. That step holds two things N23 could not part:
    how the estimator turns noise, gaps and jumps into speed, and the map's
    geometry. The sports-science libraries surveyed on 2026-09-11 low-pass
    position with a zero-phase filter, one unbroken run at a time, before
    differencing (research-data-interchange (not published)). `kalman_ground` is causal,
    runs through gaps (stats.speed_series, KNOWN DEFECT) and assumes one pixel of
    noise. N25 swaps the estimator and nothing else.

    PROTOCOL, fixed before the first run:

    adopted      `method="butter"` (stats.butter_ground): floodlight's
                 transforms/filter.py `butterworth_lowpass` at its own defaults -
                 order 3, 1 Hz, scipy's `filtfilt` - on pitch metres. MIT, (c) 2021
                 floodlight-sports; the method is adopted, no code is copied,
                 nothing is tuned. Per unbroken run: a run ends where
                 `speed_series`' gap rule marks a step unobserved; inside a run,
                 positions are interpolated onto its median cadence and read back
                 at the observed frames; a run of at most 3 * (order + 1) samples
                 gets no speed (floodlight's `remove_short_seqs=True`, over its
                 default, because unfiltered positions hand raw differences to a
                 speed threshold). Speeds per step off the filtered positions;
                 thresholds, crediting and the ground-truth side unchanged.
    deviation    ONE, found on a synthetic constant-speed walk before any real
                 data was read: the odd padding spans the whole run, not scipy's
                 default 3 * (order + 1) samples, which starts the filter in steady
                 state for a constant and reads a running player fast at every
                 run's edges. Both paddings are asserted in _selfcheck.
    conditions   K   `kalman`, what ships.
                 Kc  `kalman` scored only on the steps B scores
                     (`common_support`), so B cannot win by leaving runs out.
                 B   `butter`.
    population   N22 and N23's: nine clips, `.p21`, stride-2 frames, N6's
                 calibration, settle 0, one owner map, fragment crediting. Inputs
                 drop, gtpix, feedpix, position and both on `.p21`, and what
                 ships on `.p21id`; every input under all three conditions.
                 n=2 estimators, the Kalman scored two ways; nothing tuned.
    metric       closure: the share of our detections' excess over true positions,
                 on the SAME frames and fragments, that the filter removes - for
                 the pooled HSR ratio and the pooled sprint ratio, two ways:
                   multiplicative  1 - (drop / position - 1)[B] / (same)[Kc]
                   additive        1 - (drop - position)[B]     / (same)[Kc]
                 A bias the filter puts on every reading cancels in the first; an
                 artefact it adds to every fragment alike - a smeared identity
                 switch, say - cancels in the second. The rule needs both.
    validity     VOID unless
                 (i)   the ruler held: every input's ground-truth metres, HSR and
                       sprints are identical under K, Kc and B;
                 (ii)  B does not flatten clean running: on ground truth's own
                       continuous tracks it reads at least 0.8 of the Kalman's HSR
                       metres and of its sprint count;
                 (iii) B scores at least 0.9 of the observed seconds K scores on
                       the drop arm.
    rule         ESTIMATOR if all four closures are at least 0.5; NOT THE
                 ESTIMATOR if all four are below 0.25; MIXED otherwise.
                 ESTIMATOR puts the filter in the product's speed series next, as
                 its own change, with its own run at what ships and a check that
                 sprints match in time and not only in count. NOT THE ESTIMATOR
                 puts position work on the map - calibration. MIXED, both.

    python model/validation/dist_eval.py --n25           # N25, six inputs x three conditions

N26 (2026-09-14): THE MAP'S OWN ORACLE - HOW MUCH OF THE SPEED OVER-READ A BETTER MAP REMOVES
    N23 put the HSR over-read on the step from image to metres; N25 found the
    estimator does not hold it - every input read off the image through our map
    stays above twice the true HSR under either estimator. N12 found the map
    itself 8.27 m off where players stand and flat to click care, because one
    homography cannot describe a stitched panorama. The answer the literature and
    the vendors give (research-farfield-engineering (not published) §8.2,
    research-competitors-vendors (not published) §8.2) is a projective map plus a smooth
    correction. N26 sizes the prize before anything is built, and asks whether the
    operator's own 21 clicks can carry it.

    REVISED 2026-09-14 BEFORE THE FIRST RUN, after an independent review of the
    protocol committed in 5e65e06, which never ran. That version's round-trip check
    voided 117092 on grid arithmetic; its reference inverse picked an arbitrary
    grid point per remap pixel and added frame-to-frame noise no other arm carries;
    its hull switch made a corrected player jump metres in one step; its ruler could
    not fail; and two of its next steps did not follow from their outcomes. Each is
    fixed below. No arm was scored before or during the revision.

    PROTOCOL:

    arms         drop      N22's baseline: our detections through the shipped map,
                           N6's UI-solved `.p05.calibration.json`.
                 tps21     the same map plus a thin-plate spline through its
                           residuals at the 21 clicks it was solved from
                           (calib_product.clicks_of: perfect clicks from ST2's own
                           chain - the operator's ceiling, not a real operator),
                           fitted in pitch metres, smoothing 0, applied everywhere
                           so no player jumps where it would switch off. scipy's
                           RBFInterpolator; one configuration, nothing tuned.
                 refmap    our detections through ST2's own chain inverted
                           SMOOTHLY (_smooth_inverse): each distinct pixel that
                           calib_product's 0.1 m reference grid projects to stands
                           for its plateau's centroid, and a clip pixel reads those
                           centroids linearly interpolated. A row farther than
                           calib_product.MAX_GRID_PX from every projected pixel is
                           unplaced and dropped. This is the vendor's map, checked
                           against player positions on two clips (N12 §4), not the
                           pitch.
                 feedref   the control: the owner's true metres pushed into pixels
                           through ST2's own chain (N23's feedpix) and read back
                           through refmap - what refmap adds when its input is truth.
                 position  N22's arm: the owner's true metres.
                 Every arm re-enters the scorer as the SHIPPED map's pixels at the
                 arm's metres; the Kalman's noise model is then evaluated at those
                 pixels, which every arm but drop shares.
    population   N22 and N23's: nine clips, `.p21`, stride-2 frames, settle 0, one
                 owner map, fragment crediting, the shipped Kalman.
                 n=3 maps against the truth, and one control; nothing tuned.
    metric       share(arm) = (drop - arm) / (drop - position) on the pooled HSR
                 ratio, PRIMARY. The pooled sprint ratio and distance per covered
                 metre are reported beside it and not ruled on: N23 left sprints
                 INCONCLUSIVE, and a count of runs past a line with no hysteresis
                 moves with position noise.
    validity     VOID unless
                 (i)   the ruler held: every arm's ground truth over the full
                       tracks is identical, and its coverage-matched metres, HSR
                       and sprints are within 0.02 of the drop arm's, relative;
                 (ii)  refmap adds little: |feedref - position| <= 0.1 *
                       (drop - position), in pooled HSR ratio;
                 (iii) refmap drops at most 0.02 of the sampled rows beyond what
                       the drop arm drops, on every clip.
                 tps21 alone is VOID, and the rest stands, unless on every clip
                 it returns the clicks to 1e-6 m and its Jacobian determinant stays
                 positive over the clicks' hull. Its stretch over the hull and the
                 share of rows outside the hull are reported, not ruled.
    rule         MAP if share(refmap) >= 2/3; NOT THE MAP ALONE if below 1/3;
                 UNCLEAR between. Under MAP: CLICKS CARRY IT if share(tps21) >=
                 share(refmap) / 2, else INTERIOR LANDMARKS NEEDED. The rule is also
                 applied leaving out each clip in turn; if any of the nine disagrees
                 with the pooled verdict, the verdict is FRAGILE and says how many.
                 MAP + CLICKS CARRY IT puts the TPS correction in the product next,
                 as its own change: scored at player feet (N12's metric), with
                 noisy clicks, and on speed at what ships. MAP + INTERIOR LANDMARKS
                 NEEDED puts line-derived landmarks next, and says it only of this
                 one configuration of the correction. NOT THE MAP ALONE and UNCLEAR
                 build nothing: the excess is located by pitch band and fragment
                 first.
    held out     nothing is fitted on the scored population: the correction sees
                 21 painted-line clicks and is scored on every player's speed.

    python model/validation/dist_eval.py --n26           # N26, five arms on .p21

N27 (2026-09-14): THE 21-CLICK CORRECTION IN THE PRODUCT - AT PLAYERS' FEET, WITH NOISY CLICKS, ON SPEED AT WHAT SHIPS
    N26 found the clicked map is the speed over-read: a thin-plate spline through the
    shipped homography's residuals at its own 21 clicks removes most of our detections'
    HSR excess over true positions even with movement scale matched to the feed
    (docs/experiments/n26-map-oracle.md, after review). Its rule sends the correction into the
    product next, as its own change, scored at players' feet, with noisy clicks and on
    speed at what ships. N27 is that change and that score.

    REVISED 2026-09-14 BEFORE THE FIRST RUN, after an independent review of the protocol
    committed in 429a0d2, which never ran. That version could not run - the tracker
    refuses to replay a detection cache under another calibration file - and would have
    voided itself on folds at 10-20 px of click noise and on a Jacobian tolerance a click
    node breaks. Its SHIP covered every perspective calibration, where a synthetic
    pinhole camera doubles its position error under the correction; its scale check
    passed a map that shrinks movement; its position check could not fail against a map
    8 m off; and it read one noisy seed and no leave-one-clip-out. Each is fixed below.
    No arm was scored.

    PROTOCOL:

    the change   model/engine/calibrators.py `TpsCorrected`, projection_type "perspective_tps":
                 the operator's perspective calibration - H and its clicks, as the UI
                 already writes them - plus a thin-plate spline through H's residuals at
                 the clicks the solver kept, in pitch metres, smoothing 0, applied
                 everywhere. If the spline folds inside the clicks' hull it switches itself
                 off and the map is H. Speed reads its own Jacobian for measurement noise;
                 app/server/jobs.py loads it. The UI still writes "perspective".
    population   the eight clips N21 calibrated whose feed join is sound: N21's nine minus
                 132877. Stride-2 frames. Each clip's feed is also read against ST2's own
                 chain at the same feet and reported, so a clip with doubtful truth shows.
    clicks       sigma 0 is N6's calibration on disk; noisy sets are N12's exactly -
                 Gaussian pixel noise of 1, 2, 5, 10, 20 px, seeds 0-2, re-solved by the
                 UI's own solver; the correction is fitted on the same noisy clicks.
    A position   MOT box feet on clip frames divisible by 20, through each map, against the
                 joined feed at that frame and player; far half = feed y < 34. Shipped
                 against corrected, per clip and pooled, at every sigma and seed; the share
                 of click sets whose correction switched off is reported.
    B scale      on the product's own tracks at sigma 0, over runs that stay on one
                 fragment and one owner with no missed frame: the least-squares slope of
                 4 s chord displacement through the map against the feed's, chords of at
                 least 5 m of true movement, per clip; the median chord length ratio beside.
    C speed      the product path end to end, at sigma 0 and at sigma 5 seeds 0-2. N21's
                 `.p21` detection cache is copied for each arm with the arm's calibration
                 digest written in, so every arm replays N21's detections as N6's pitch
                 polygon gated them - for a noisy H not its own polygon, which is what the
                 tracker's guard exists to flag, disclosed here instead - through track.run
                 (the stitch reads the map) and jobs.apply_identity. Arms: shipped sigma 0
                 is `.p21id`; corrected sigma 0 `.p27id`; shipped and corrected at sigma 5
                 `.p27h5s<seed>id` and `.p27t5s<seed>id`. Our side read through the arm's
                 calibration, the ground truth through N6's. Pooled HSR ratio PRIMARY with
                 each arm's movement scale matched to the feed (metres scaled about the
                 pitch centre by 1 / B's slope on that arm's own tracks); both sides of the
                 ratio, coverage and the own-scale ratio recorded per clip. Beside, not
                 ruled: the corrected map on `.p21id` without a restitch, and `.p27id` read
                 with H's noise model.
    validity     VOID unless every click set solved; every slope is finite; the corrected
                 Jacobian agrees with a ten-times finer difference of its own map to a
                 median of 1% and differs from H's by more; the ground truth over the full
                 tracks is identical in every arm and every replay carries exactly `.p21`'s
                 (frame, x, y) rows - both hold by construction and are kept as plumbing -
                 and no `.p27*` output existed before the run.
    rule         SHIP FOR STITCHED PANORAMAS iff all of
                 A  on every clip, corrected median below shipped at sigma 0 and at every
                    sigma-5 seed; pooled median, p90 and far-half median not above shipped
                    at any sigma and seed; no correction switched off at sigma <= 5;
                 B  the corrected slope in [0.90, 1.10] on every clip, its median |slope - 1|
                    at most 0.05 and not above the shipped map's - the truth itself reads box
                    feet a few percent short (review of N26), so a tighter bound is unreadable;
                 C  with movement scale matched, (shipped - corrected) / (shipped - 1) >= 0.5
                    on the pooled HSR ratio at sigma 0, at every sigma-5 seed, and leaving out
                    any one clip at sigma 0.
                 Otherwise DO NOT SHIP, naming each part that failed; NOTHING TO REMOVE if the
                 shipped matched ratio is at or below 1. SHIP says nothing about single
                 cameras; which footage gets the correction is a separate measurement.
    n            one correction configuration, nothing tuned; noise 6 sigmas x 3 seeds for
                 A, sigma 0 and sigma 5 x 3 seeds for C.

    REVIEW, 2026-09-14, after the run - the rule above is left as committed:
                 C's bar is out of reach for any map. True positions through the same
                 replays' fragments (N22's position oracle) read HSR 1.6634 on `.p21id`,
                 which scores 0.3553 on C, so C's failure says nothing about the correction;
                 against that floor the correction removes about half the excess on the same
                 fragments. B's "the truth itself reads box feet a few percent short" is ST2's
                 chain disagreeing with the feed, and the sigma-0 clicks are the chain's own
                 pixels. The matched share moves by about +-0.2 with the scale statistic; the
                 scaling leaves the Kalman's initial and process noise unscaled; and N26's
                 grid-staircase requirement was dropped. docs/experiments/n27-tps-in-product.md carries it.

    python model/validation/dist_eval.py --n27           # N27
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "model", "engine"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import calibrators  # noqa: E402
import stats  # noqa: E402
import contam_eval  # noqa: E402  - attribution is THEIRS, imported not reimplemented
from contam_eval import STRIDE, owners  # noqa: E402

JOB = os.path.join(ROOT, "data/jobs/118575_full_1st")
GT = os.path.join(ROOT, "model/validation/soccertrack/118575_1st.tracks.json.gz")


def to_px(cal, xy_m: np.ndarray) -> np.ndarray:
    """Pitch metres -> the pixels that project to them. The inverse of to_pitch.

    This is what lets GT go through the SAME estimator as our detections rather
    than a lookalike: `kalman_ground` needs pixels to differentiate the
    homography at, and a GT point's Jacobian is the true one for that patch of
    grass. The round-trip is ASSERTED, not assumed - `to_pitch` is only H for
    the perspective and remap calibrators, and silently returning garbage on a
    panoramic or bow-wrapped one would move every number in this file.
    """
    H = np.asarray(cal.H, float)
    p = np.c_[xy_m, np.ones(len(xy_m))] @ np.linalg.inv(H).T
    px = p[:, :2] / p[:, 2:]
    err = float(np.abs(np.asarray(cal.to_pitch(px)) - xy_m).max())
    if err > 1e-6:
        raise SystemExit(f"{type(cal).__name__}.to_pitch is not H (round-trip {err:.3g} m); "
                         "this harness cannot push GT backwards through it")
    return px


def track_metrics(track: list[dict], cal, fps: float, method: str = "kalman",
                  real_override: np.ndarray | None = None,
                  window_s: float | None = None) -> tuple[float, float, float, float]:
    """(metres, observed_frac, peak m/s UNCLAMPED, clamped share) for one track.

    Peak speed is deliberately taken with `stats.MAX_SPEED` lifted. The clamp is
    a product safety rule - a track running at 9.5 m/s is mis-tracked, not fast -
    but a harness whose job is to ask whether the clamp is right cannot read the
    clamped value, because that reads 9.5 whatever the truth is. The clamp is
    applied afterwards in numpy instead, which is exactly equivalent: the filter
    STATE is never clamped, only the reported speed (see `kalman_ground`).
    """
    t = np.array([p["f"] / fps for p in sorted(track, key=lambda d: d["f"])], float)
    keep, stats.MAX_SPEED = stats.MAX_SPEED, np.inf
    try:
        v, _dt, real, xy = stats.speed_series(track, cal, fps, method)
    finally:
        stats.MAX_SPEED = keep
    if real_override is not None:
        real = real & real_override
    dist, obs = stats._windowed(xy, t, real, window_s)
    peak = float(v[real].max()) if real.any() else 0.0
    clamped = float(np.mean(v[real] >= keep)) if real.any() else 0.0
    return dist, obs, peak, clamped


def load_gt(path: str = GT):
    """(fps, {player_id: {"f": frames, "xy": metres, "team":, "jersey":}}), 0-based."""
    d = json.load(gzip.open(path))
    rows = d["detections"]
    base = min(r["f"] for r in rows)
    per = collections.defaultdict(list)
    meta = {}
    for r in rows:
        g = str(r["id"])
        per[g].append((r["f"] - base, r["x"], r["y"]))
        meta.setdefault(g, (r.get("team", "unknown"), r.get("jersey")))
    out = {}
    for g, v in per.items():
        a = np.array(sorted(v), float)
        out[g] = {"f": a[:, 0].astype(np.int64), "xy": a[:, 1:],
                  "team": meta[g][0], "jersey": meta[g][1]}
    return float(d["fps"]), out


def gt_as_prediction(gt, cal, noise_px: float = 0.0, seed: int = 0) -> list[dict]:
    """GT re-emitted as if it were our tracker's output. THE LIVE CONTROL.

    A harness that reports ~1.19 has to be shown reporting ~1.00 on a pipeline
    that is right, or the 1.19 could be anything - plumbing, attribution, the
    window phase, the backwards projection. This feeds GT pseudo-pixels through
    the ENTIRE prediction path (attribution, per-fragment windowing, summation)
    and the answer must come out at 1.00 with coverage 1.00. With `noise_px` it
    also shows what detector jitter alone does to distance, which is the
    mechanism the real run has to be judged against.
    """
    rng = np.random.default_rng(seed)
    out = []
    for g, t in gt.items():
        px = to_px(cal, t["xy"])
        if noise_px:
            px = px + rng.normal(0, noise_px, px.shape)
        out.extend({"f": int(f), "id": int(g), "x": float(u), "y": float(v)}
                   for f, (u, v) in zip(t["f"], px))
    return out


def evaluate(tracks_file: str, job: str = JOB, gt_path: str = GT,
             stride: int = STRIDE, gt_method: str = "kalman",
             window_s: float | None = None, control_noise_px: float | None = None,
             max_frag_peak: float | None = None) -> dict:
    cal = calibrators.load(os.path.join(job, "calibration.json"))
    gt_fps, gt = load_gt(gt_path)

    if control_noise_px is not None:
        rows, fps = gt_as_prediction(gt, cal, control_noise_px), gt_fps
        tracks_file = f"CONTROL: GT as prediction, noise {control_noise_px} px"
    else:
        blob = json.load(open(os.path.join(job, tracks_file)))
        rows, fps = blob["detections"], float(blob.get("fps", gt_fps))
    if abs(fps - gt_fps) > 1e-6:
        raise SystemExit(f"fps mismatch: ours {fps}, GT {gt_fps} - frame indices do not align")

    # --- 2. fragments -> players. contam_eval's attribution, unmodified. -------
    owner = owners(rows, cal, contam_eval.load_gt(gt_path), stride)
    votes = collections.defaultdict(collections.Counter)
    for i, g in owner.items():
        votes[int(rows[i]["id"])][g] += 1
    dominant = {tid: c.most_common(1)[0][0] for tid, c in votes.items()}

    by_id = collections.defaultdict(list)
    for r in rows:
        by_id[int(r["id"])].append(r)

    # --- our distance, summed per GT player; frames covered, for the gap mask --
    ours = collections.defaultdict(lambda: {"m": 0.0, "peak": 0.0, "frag": 0,
                                            "over": 0, "n": 0})
    frames = collections.defaultdict(set)
    orphan_m, orphan_ids = 0.0, 0
    dropped_m, dropped_ids = 0.0, 0
    for tid, trk in by_id.items():
        if len(trk) < 2:
            continue
        m, _obs, peak, clamped = track_metrics(trk, cal, fps, window_s=window_s)
        # DROPPING THE TELEPORTS, and why this is a fair test rather than a
        # cherry-pick: `gt_matched` is rebuilt from whatever frames survive, so
        # removing fragments removes the SAME windows from the reference. The
        # ratio therefore stays an accuracy number while coverage falls, which
        # is the one manipulation this design makes safe.
        if max_frag_peak is not None and peak > max_frag_peak:
            dropped_m += m
            dropped_ids += 1
            continue
        g = dominant.get(tid)
        if g is None:
            orphan_m += m
            orphan_ids += 1
            continue
        o = ours[g]
        o["m"] += m
        o["peak"] = max(o["peak"], peak)
        o["frag"] += 1
        o["over"] += peak > stats.MAX_SPEED
        o["n"] += 1
        frames[g].update(int(r["f"]) for r in trk)

    # --- 3. GT twice: full, and re-scored with OUR coverage as the gap mask ----
    per_player = []
    for g, t in sorted(gt.items(), key=lambda kv: int(kv[0])):
        gpx = to_px(cal, t["xy"])
        trk = [{"f": int(f), "id": g, "x": x, "y": y} for f, (x, y) in zip(t["f"], gpx)]
        # How many metres one pixel of detector jitter is worth WHERE THIS
        # PLAYER PLAYED. The camera's grazing angle makes this vary 0.04-0.63
        # m/px across the pitch, so identical pixel noise becomes wildly
        # different metre noise per player - which is what turns a systematic
        # mechanism into a per-player spread. Reported so the two can be told
        # apart: correlated with ratio => noise; uncorrelated => contamination.
        m_per_px = float(np.linalg.svd(
            stats.homography_jacobian(np.asarray(cal.H, float), gpx[::250]),
            compute_uv=False)[:, 0].mean())
        full_m, _o, gt_peak, _c = track_metrics(trk, cal, gt_fps, gt_method,
                                                window_s=window_s)
        seen = frames.get(g, set())
        # A GT step counts only if we held that player at BOTH ends of it: the
        # same rule `_windowed` applies to our own occlusion gaps.
        have = np.fromiter((int(f) in seen for f in t["f"]), bool, len(t["f"]))
        matched_m, _o2, _p, _c2 = track_metrics(
            trk, cal, gt_fps, gt_method, real_override=have[:-1] & have[1:],
            window_s=window_s)
        o = ours.get(g, {"m": 0.0, "peak": 0.0, "frag": 0, "over": 0, "n": 0})
        cov = len(seen & set(t["f"].tolist())) / len(t["f"])
        per_player.append({
            "gt_id": g, "team": t["team"], "jersey": t["jersey"],
            "gt_m": full_m, "gt_matched_m": matched_m, "ours_m": o["m"],
            "coverage": cov, "fragments": o["frag"],
            "ratio_raw": o["m"] / full_m if full_m else 0.0,
            "ratio_matched": o["m"] / matched_m if matched_m else 0.0,
            "gt_peak_ms": gt_peak, "our_peak_ms": o["peak"], "m_per_px": m_per_px,
            # Share of THIS PLAYER'S fragments whose peak speed exceeds
            # MAX_SPEED, i.e. the share of shipped rows that would report the
            # clamp (34.2 km/h) rather than a speed.
            "clamped_frac": o["over"] / o["n"] if o["n"] else 0.0,
        })

    r = [p["ratio_matched"] for p in per_player if p["gt_matched_m"] > 0]
    sq_gt = sum(p["gt_m"] for p in per_player)
    sq_gtm = sum(p["gt_matched_m"] for p in per_player)
    sq_ours = sum(p["ours_m"] for p in per_player)
    return {
        "tracks": tracks_file, "gt_method": gt_method,
        "players": per_player,
        "squad": {
            "gt_km": sq_gt / 1000, "gt_matched_km": sq_gtm / 1000,
            "ours_km": sq_ours / 1000,
            "ratio_raw": sq_ours / sq_gt if sq_gt else 0.0,
            "ratio_matched": sq_ours / sq_gtm if sq_gtm else 0.0,
            "coverage": float(np.mean([p["coverage"] for p in per_player])),
            "ids": len(by_id), "attributed_ids": len(dominant),
            "orphan_ids": orphan_ids, "orphan_km": orphan_m / 1000,
            "dropped_ids": dropped_ids, "dropped_km": dropped_m / 1000,
            "max_frag_peak": max_frag_peak, "window_s": window_s or stats.DISTANCE_WINDOW_S,
        },
        # Systematic vs scattered. A scale error moves the median and leaves the
        # spread alone; contamination and mis-attribution move the spread and
        # leave the median alone. Both are reported so the reader picks.
        "spread": {
            "median_ratio": float(np.median(r)) if r else 0.0,
            "iqr": [float(np.percentile(r, 25)), float(np.percentile(r, 75))] if r else [0, 0],
            "min": float(min(r)) if r else 0.0, "max": float(max(r)) if r else 0.0,
            "cv": float(np.std(r) / np.mean(r)) if r else 0.0,
            # Pearson r between a player's local metres-per-pixel and how much
            # we over-read them. Detector noise predicts a strong positive
            # correlation; contamination predicts none.
            "corr_ratio_vs_m_per_px": float(np.corrcoef(
                r, [p["m_per_px"] for p in per_player if p["gt_matched_m"] > 0])[0, 1])
            if len(r) > 2 else 0.0,
        },
        "peak": {
            "gt_max_ms": max(p["gt_peak_ms"] for p in per_player),
            "gt_median_ms": float(np.median([p["gt_peak_ms"] for p in per_player])),
            "our_max_ms": max(p["our_peak_ms"] for p in per_player),
            "our_median_ms": float(np.median([p["our_peak_ms"] for p in per_player])),
            "MAX_SPEED": stats.MAX_SPEED,
            "our_frags_over_max": (over := sum(o["over"] for o in ours.values())),
            "our_frags": (nfrag := sum(o["n"] for o in ours.values())),
            # The share as well as the two counts, because a caller that needs
            # the rate should not have to divide - `report` printed the pair and
            # `api.py` wanted the quotient, and a number computed in two places
            # is a number that can disagree with itself.
            "clamped_frac": over / nfrag if nfrag else 0.0,
        },
    }


def report(r: dict) -> None:
    s, sp, pk = r["squad"], r["spread"], r["peak"]
    print(f"\n=== {r['tracks']}   (GT filtered: {r['gt_method']}, "
          f"window {r['squad']['window_s']} s) ===")
    print(f"{'gt':>3} {'team':>5} {'#':>3} | {'GT m':>7} {'GTmatch':>8} {'ours m':>7} "
          f"| {'cov':>5} {'frag':>4} | {'raw':>5} {'match':>5} | "
          f"{'GTpk':>5} {'ourpk':>6} {'clmp':>5} {'m/px':>5}")
    for p in r["players"]:
        print(f"{p['gt_id']:>3} {p['team']:>5} {str(p['jersey']):>3} | "
              f"{p['gt_m']:7.0f} {p['gt_matched_m']:8.0f} {p['ours_m']:7.0f} | "
              f"{p['coverage']:5.2f} {p['fragments']:4d} | "
              f"{p['ratio_raw']:5.2f} {p['ratio_matched']:5.2f} | "
              f"{p['gt_peak_ms']:5.2f} {p['our_peak_ms']:6.2f} {p['clamped_frac']:5.2f} "
              f"{p['m_per_px']:5.3f}")
    print(f"\nsquad  GT {s['gt_km']:.2f} km   GT-matched {s['gt_matched_km']:.2f} km   "
          f"ours {s['ours_km']:.2f} km")
    print(f"       ratio raw {s['ratio_raw']:.3f}   ratio matched {s['ratio_matched']:.3f}   "
          f"mean coverage {s['coverage']:.2f}")
    print(f"       ids {s['ids']}  attributed {s['attributed_ids']}  "
          f"orphan {s['orphan_ids']} ids / {s['orphan_km']:.2f} km emitted for nobody"
          + (f"   DROPPED {s['dropped_ids']} ids / {s['dropped_km']:.2f} km "
             f"peaking over {s['max_frag_peak']} m/s" if s["max_frag_peak"] else ""))
    print(f"per-player ratio_matched  median {sp['median_ratio']:.3f}  "
          f"IQR {sp['iqr'][0]:.3f}-{sp['iqr'][1]:.3f}  "
          f"range {sp['min']:.3f}-{sp['max']:.3f}  cv {sp['cv']:.3f}\n"
          f"           corr(ratio, local m/px) = {sp['corr_ratio_vs_m_per_px']:+.2f}  "
          "(high => detector noise; ~0 => contamination)")
    print(f"peak speed m/s   GT max {pk['gt_max_ms']:.2f} median {pk['gt_median_ms']:.2f}   "
          f"ours max {pk['our_max_ms']:.2f} median {pk['our_median_ms']:.2f}   "
          f"(MAX_SPEED {pk['MAX_SPEED']})")
    print(f"                 {pk['our_frags_over_max']}/{pk['our_frags']} of our fragments "
          f"peak above MAX_SPEED, i.e. would ship the clamp not a speed", flush=True)


# ==========================================================================
# N10 / N11. Everything above scores one job directory against the SoccerTrack
# gz and is untouched, so `api://product.distance.*` still reproduces. Protocol
# and both pass rules are in the module docstring, fixed before the run.

STC = os.path.join(ROOT, "model/validation/out/stc2025")
STC_GT_CACHE = os.path.join(ROOT, "model/validation/out/n10_stc_gt.json")
#: Both sides on the frames a stride-2 tracker runs on. See the protocol.
STC_EVERY = 2


def _stc_gt_key(j: dict, every: int) -> str:
    """What a cached stc_gt entry was computed from: the join's (period, offset), the
    stride, AND its identity permutation and team labels. Keyed on the first three
    only, a re-solved permutation (two players swapped) or a re-labelled team kept
    serving the old players' metres under the new ids."""
    import hashlib  # noqa: PLC0415
    ident = hashlib.sha256(json.dumps([j["mot_to_player"], j["team"]], sort_keys=True)
                           .encode()).hexdigest()[:16]
    return f"{j['period']}:{j['offset']}:{every}:{ident}"


def stc_gt(seq: str, every: int = STC_EVERY, cache: str = STC_GT_CACHE) -> dict:
    """{mot id: {"xy": metres on the even clip frames, "f": those frames, "team"}}.

    Scans `<seq>_tracker_box_data.xml` for the joined window ONLY. `st2_team_gt.
    dense_loc` reads the whole match and costs ~800 MB of RSS per sequence; this
    machine runs three lanes at once on 16 GiB, so the window is worth ten lines.
    The regexes and the join are that file's, imported - only the retention is
    different.

    Cached, keyed on the join's own (period, offset), its permutation and teams
    (`_stc_gt_key`). A cache that cannot tell it is stale is worse than no cache:
    if the join is ever re-solved these numbers would silently keep answering for
    the old alignment.

    READS, for api.py's keys: the join (st2_team_gt.load), `<seq>_tracker_box_data.xml`
    under st2_team_gt.RAW on a cache miss, and STC_GT_CACHE.
    """
    import st2_team_gt as G  # noqa: PLC0415

    j = G.load()[seq]
    if not G._ok(j):
        raise SystemExit(f"{seq}: the team/position join failed its own gate")
    if G.feed_mirrored(j):
        if seq not in G.MIRRORED_FEEDS_REPRODUCED:
            raise SystemExit(f"{seq}: the joined position feed is mirrored against ST2's own camera, or its "
                             "handedness was never checked - its positions are wrong (st2_team_gt, HANDEDNESS). "
                             "Rebuild the join with --build, and leave the clip out of any position result.")
        print(f"  {seq}: MIRRORED ground-truth feed, read only to reproduce results registered on it "
              "(st2-132877-feed-is-mirrored (internal note))", file=sys.stderr)
    key = _stc_gt_key(j, every)
    blob = json.load(open(cache)) if os.path.exists(cache) else {}
    if blob.get(seq, {}).get("key") == key:
        return {m: {"f": np.array(v["f"], np.int64), "xy": np.array(v["xy"], float),
                    "team": v["team"]}
                for m, v in blob[seq]["players"].items()}

    p2m = {int(v): int(k) for k, v in j["mot_to_player"].items()}
    want = {j["offset"] + f: f for f in range(0, j["clip_frames"], every)}
    period = j["period"].encode()
    per: dict[int, list] = collections.defaultdict(list)
    keep, clip_f = False, -1
    with open(f"{G.RAW}/{seq}/{seq}_tracker_box_data.xml", "rb") as fh:
        for line in fh:
            s = line.lstrip()
            if s[:6] == b"<frame":
                m = G._FRAME.match(s)
                keep = bool(m) and m.group(2) == period and int(m.group(1)) in want
                clip_f = want[int(m.group(1))] if keep else -1
            elif keep and s[:7] == b"<player":
                m = G._PLAYER.match(s)
                pid = int(m.group(1)) if m else None
                if pid in p2m:
                    per[p2m[pid]].append(
                        (clip_f, float(m.group(2)) * 105.0, float(m.group(3)) * 68.0))
    out = {}
    for mid, v in per.items():
        a = np.array(sorted(v), float)
        out[str(mid)] = {"f": a[:, 0].astype(np.int64), "xy": a[:, 1:],
                         "team": j["team"][str(mid)]}
    blob[seq] = {"key": key, "players": {m: {"f": v["f"].tolist(),
                                             "xy": v["xy"].tolist(), "team": v["team"]}
                                         for m, v in out.items()}}
    json.dump(blob, open(cache, "w"))
    return out


def _settle(t: np.ndarray, real: np.ndarray, settle_s: float) -> np.ndarray:
    """True where the velocity estimate is INSIDE a post-gap settling window.

    `kalman_ground` has no gap handling: it propagates through the hole and the
    re-acquisition update dumps the whole residual into the velocity state,
    which then bleeds into the next ~30 observed steps that `real` calls
    trustworthy (stats.speed_series, the KNOWN DEFECT comment). This marks those
    steps. It never touches `real`, so distance cannot move - and must not.
    """
    te = t[1:]
    if settle_s <= 0:
        return np.zeros(len(te), bool)
    restart = np.where(~real, te, -np.inf)
    restart[0] = te[0]          # a fragment's own start is a cold filter too
    return (te - np.maximum.accumulate(restart)) < settle_s


def frag_stats(track: list[dict], cal, fps: float, method: str = "kalman",
               real_override: np.ndarray | None = None,
               window_s: float | None = None, settle_s: float = 0.0,
               common_support: bool = False) -> dict:
    """One fragment -> distance, coverage, peak, clamp rate, HSR metres, sprints.

    `common_support` scores only the steps N25's adopted filter gives a speed to,
    whatever `method` is: the Kalman arm it is compared with scores the same steps.

    One `speed_series` call for all six, which is `stats.speed_series`'s own
    reason for existing: when validate.py kept a second copy, HSR and distance
    ended up computed off different estimates of the same track.

    Peak is read with MAX_SPEED lifted (see `track_metrics`); the clamp share is
    applied afterwards in numpy, which is exactly equivalent because the filter
    STATE is never clamped.
    """
    t = np.array([p["f"] / fps for p in sorted(track, key=lambda d: d["f"])], float)
    keep, stats.MAX_SPEED = stats.MAX_SPEED, np.inf
    try:
        v, dt, real, xy = stats.speed_series(track, cal, fps, method)
    finally:
        stats.MAX_SPEED = keep
    if real_override is not None:
        real = real & real_override
    if common_support:
        real = stats.butter_support(t, real)
    dist, obs = stats._windowed(xy, t, real, window_s)
    # v := 0 inside the settling window, `real` untouched. This is the operation
    # speed-clamp (not published) §5 prices, and it must be applied to BOTH sides or the
    # ratio measures the operation instead of the defect.
    v = np.where(_settle(t, real, settle_s), 0.0, v)
    # HSR threshold: validate.py:66 sets HSR_SPEED = stats.SPRINT_SPEED, so this
    # reads the shipped constant rather than a second copy that can drift.
    fast = real & (v >= stats.SPRINT_SPEED)
    return {"m": dist, "observed_frac": obs,
            "peak": float(v[real].max()) if real.any() else 0.0,
            "clamped": float(np.mean(v[real] >= keep)) if real.any() else 0.0,
            "hsr_m": float(np.sum(v[fast] * dt[fast])),
            "sprints": stats._count_sprints(v, t[1:], real),
            "observed_s": float(np.sum(dt[real])) if real.any() else 0.0}


def evaluate_stc(seq: str, variant: str = "p05", every: int = STC_EVERY,
                 gt_method: str = "kalman", window_s: float | None = None,
                 settle_s: float = 0.0, control: bool = False, oracle: str | None = None,
                 credit: str = "fragment", method: str = "kalman",
                 common_support: bool = False, our_cal=None) -> dict:
    """One STC2025 clip at the product config. See the protocol in the header.

    `our_cal` is N27's: our side is read through it, the ground truth still through
    N6's calibration, so an arm can change the map without moving the ruler.

    `method` and `common_support` are N25's, and move OUR side only: the ground
    truth stays on `gt_method`, and is also read through `method` beside it as
    `gt_alt_*` - the check that the estimator under test does not flatten clean
    running.

    `control=True` replaces our tracks with the GT itself, one fragment per
    player, and skips the IoU attribution. It MUST read ratio 1.000 at coverage
    1.000: if it does not, the reference loader, the backwards projection, the
    per-player summation or the coverage mask is wrong, and every number the
    real run prints is a confident lie rather than a measurement. The attribution
    it skips has its own control in `contam_eval._selfcheck_stc`.
    """
    import contam_eval as C  # noqa: PLC0415

    cal = calibrators.load(f"{STC}/{seq}.p05.calibration.json")
    blob = json.load(open(f"{STC}/{seq}.{variant}.tracks.json"))
    rows, fps = blob["detections"], float(blob["fps"])
    gt = stc_gt(seq, every)
    map_checks = None

    if control:
        rows = [{"f": int(f), "id": int(m), "x": float(u), "y": float(v)}
                for m, t in gt.items()
                for f, (u, v) in zip(t["f"], to_px(cal, t["xy"]))]
        owner = {i: str(r["id"]) for i, r in enumerate(rows)}
        counts = {"our_boxes_sampled": len(rows)}
    else:
        owner, counts = C.owners_iou(seq, rows, every)
        if oracle:
            feet = G_mot_feet(seq) if oracle == "gtpix" else None
            feedpx = _vendor_px(seq, gt) if oracle in ("feedpix", "feedref") else None
            mapper, map_checks = (_tps21(seq, cal) if oracle == "tps21" else
                                  (_refmap(seq), None) if oracle in ("refmap", "feedref") else (None, None))
            rows, owner = _oracle_rows(rows, owner, gt, cal, oracle, feet, feedpx, mapper)
    votes: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for i, g in owner.items():
        votes[int(rows[i]["id"])][g] += 1
    dominant = {tid: c.most_common(1)[0][0] for tid, c in votes.items()}
    # The attribution's own error bar: a fragment is credited whole to its
    # majority owner, so this is the share of attributed detections that are
    # credited to somebody they do not belong to. N8's number, recomputed on the
    # same owner map rather than quoted from another run.
    impure = sum(n - c.most_common(1)[0][1] for c in votes.values()
                 for n in [sum(c.values())])
    attributed = sum(sum(c.values()) for c in votes.values())

    if credit not in ("fragment", "detection"):
        raise ValueError(f"unknown crediting {credit!r}")
    if credit == "detection" and (method != "kalman" or common_support or our_cal is not None):
        raise ValueError("N25's estimators and N27's maps are scored under fragment crediting only")
    by_id: dict[int, list] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if int(r["f"]) % every == 0:
            by_id[int(r["id"])].append(i)

    ours: dict[str, dict] = collections.defaultdict(
        lambda: {"m": 0.0, "hsr_m": 0.0, "sprints": 0, "peak": 0.0, "over": 0, "n": 0, "s": 0.0})
    frames: dict[str, set] = collections.defaultdict(set)
    orphan_m, orphan_ids, uncredited_m = 0.0, 0, 0.0
    for tid, idx in by_id.items():
        if len(idx) < 2:
            continue
        trk = [rows[i] for i in idx]
        if credit == "detection":
            parts, whole = _credited(trk, [owner.get(i) for i in idx], cal, fps, window_s, settle_s)
            uncredited_m += whole - sum(p["m"] for p in parts.values())
            for g, p in parts.items():
                o = ours[g]
                o["m"] += p["m"]
                o["hsr_m"] += p["hsr_m"]
                o["sprints"] += p["sprints"]
                o["peak"] = max(o["peak"], p["peak"])
                o["over"] += p["peak"] > stats.MAX_SPEED
                o["n"] += 1
                o["s"] += p["s"]
                frames[g].update(p["frames"])
            continue
        s = frag_stats(trk, our_cal if our_cal is not None else cal, fps, method, window_s=window_s,
                       settle_s=settle_s, common_support=common_support)
        g = dominant.get(tid)
        if g is None:
            orphan_m += s["m"]
            orphan_ids += 1
            continue
        o = ours[g]
        o["m"] += s["m"]
        o["hsr_m"] += s["hsr_m"]
        o["sprints"] += s["sprints"]
        o["peak"] = max(o["peak"], s["peak"])
        o["over"] += s["peak"] > stats.MAX_SPEED
        o["n"] += 1
        o["s"] += s["observed_s"]
        frames[g].update(int(r["f"]) for r in trk)

    per_player = []
    for g, t in sorted(gt.items(), key=lambda kv: int(kv[0])):
        gpx = to_px(cal, t["xy"])
        trk = [{"f": int(f), "id": g, "x": x, "y": y} for f, (x, y) in zip(t["f"], gpx)]
        # Metres per pixel where THIS player played: the camera's grazing angle
        # makes identical pixel noise into wildly different metre noise, so this
        # is what tells a systematic mechanism from a per-player one.
        m_per_px = float(np.linalg.svd(
            stats.homography_jacobian(np.asarray(cal.H, float), gpx[::100]),
            compute_uv=False)[:, 0].mean())
        full = frag_stats(trk, cal, fps, gt_method, window_s=window_s, settle_s=settle_s)
        alt = full if method == gt_method else frag_stats(trk, cal, fps, method,
                                                          window_s=window_s, settle_s=settle_s)
        seen = frames.get(g, set())
        have = np.fromiter((int(f) in seen for f in t["f"]), bool, len(t["f"]))
        matched = frag_stats(trk, cal, fps, gt_method, real_override=have[:-1] & have[1:],
                             window_s=window_s, settle_s=settle_s)
        o = ours.get(g, {"m": 0.0, "hsr_m": 0.0, "sprints": 0, "peak": 0.0,
                         "over": 0, "n": 0, "s": 0.0})
        per_player.append({
            "gt_id": g, "team": t["team"],
            "gt_m": full["m"], "gt_matched_m": matched["m"], "ours_m": o["m"],
            "coverage": len(seen & set(t["f"].tolist())) / len(t["f"]),
            "fragments": o["n"],
            "ratio_raw": o["m"] / full["m"] if full["m"] else 0.0,
            "ratio_matched": o["m"] / matched["m"] if matched["m"] else 0.0,
            "gt_peak_ms": full["peak"], "our_peak_ms": o["peak"], "m_per_px": m_per_px,
            "gt_hsr_m": full["hsr_m"], "gt_matched_hsr_m": matched["hsr_m"],
            "our_hsr_m": o["hsr_m"],
            "gt_sprints": full["sprints"], "gt_matched_sprints": matched["sprints"],
            "our_sprints": o["sprints"],
            "clamped_frac": o["over"] / o["n"] if o["n"] else 0.0,
            "our_observed_s": o["s"],
            "gt_alt_hsr_m": alt["hsr_m"], "gt_alt_sprints": alt["sprints"],
        })

    def side(k, team=None):
        return sum(p[k] for p in per_player if team is None or p["team"] == team)

    r = [p["ratio_matched"] for p in per_player if p["gt_matched_m"] > 0]
    out = {
        "seq": seq, "variant": variant, "credit": credit, "settle_s": settle_s,
        "method": method, "common_support": common_support, "map_checks": map_checks,
        "window_s": window_s or stats.DISTANCE_WINDOW_S,
        "players": per_player,
        "squad": {
            "gt_m": side("gt_m"), "gt_matched_m": side("gt_matched_m"),
            "ours_m": side("ours_m"),
            "ratio_raw": side("ours_m") / side("gt_m") if side("gt_m") else 0.0,
            "ratio_matched": side("ours_m") / side("gt_matched_m")
            if side("gt_matched_m") else 0.0,
            "coverage": float(np.mean([p["coverage"] for p in per_player])),
            "ids": len(by_id), "attributed_ids": len(dominant),
            "orphan_ids": orphan_ids, "orphan_m": orphan_m, "uncredited_m": uncredited_m,
            "fragments": sum(p["fragments"] for p in per_player),
            # The drop shares, on the row. `unattributed_frac` is our boxes that
            # matched no GT box at all; `attribution_contaminated` is the share
            # of the ones that DID match which this whole-fragment credit puts
            # on the wrong player anyway.
            "unattributed_frac": round(1 - attributed / counts["our_boxes_sampled"], 4),
            "attribution_contaminated": round(impure / attributed, 4) if attributed else 0.0,
        },
        # Side-summed, and the label is a CLUSTER not a side: N9 measured
        # home/away naming at chance, so these two rows are "the two banks",
        # matched to GT sides by our own attribution, not by the team labeller.
        "teams": {k: {"gt_m": side("gt_m", k), "gt_matched_m": side("gt_matched_m", k),
                      "ours_m": side("ours_m", k),
                      "ratio_raw": side("ours_m", k) / side("gt_m", k)
                      if side("gt_m", k) else 0.0,
                      "ratio_matched": side("ours_m", k) / side("gt_matched_m", k)
                      if side("gt_matched_m", k) else 0.0,
                      "coverage": float(np.mean([p["coverage"] for p in per_player
                                                 if p["team"] == k]))}
                  for k in ("home", "away")},
        "spread": {
            "median_ratio": float(np.median(r)) if r else 0.0,
            "iqr": [float(np.percentile(r, 25)), float(np.percentile(r, 75))]
            if r else [0, 0],
            "min": float(min(r)) if r else 0.0, "max": float(max(r)) if r else 0.0,
            "cv": float(np.std(r) / np.mean(r)) if r else 0.0,
        },
        "speed": {
            "our_hsr_m": side("our_hsr_m"), "gt_hsr_m": side("gt_hsr_m"),
            "gt_matched_hsr_m": side("gt_matched_hsr_m"),
            "hsr_ratio": side("our_hsr_m") / side("gt_matched_hsr_m")
            if side("gt_matched_hsr_m") else 0.0,
            "our_sprints": side("our_sprints"), "gt_sprints": side("gt_sprints"),
            "our_observed_s": side("our_observed_s"),
            "gt_alt_hsr_m": side("gt_alt_hsr_m"), "gt_alt_sprints": side("gt_alt_sprints"),
            "gt_matched_sprints": side("gt_matched_sprints"),
            "sprint_ratio": side("our_sprints") / side("gt_matched_sprints")
            if side("gt_matched_sprints") else 0.0,
            "our_peak_max_ms": max(p["our_peak_ms"] for p in per_player),
            "our_peak_median_ms": float(np.median([p["our_peak_ms"] for p in per_player])),
            "gt_peak_max_ms": max(p["gt_peak_ms"] for p in per_player),
            "gt_peak_median_ms": float(np.median([p["gt_peak_ms"] for p in per_player])),
            "clamped_frac": sum(p["clamped_frac"] * p["fragments"] for p in per_player)
            / max(sum(p["fragments"] for p in per_player), 1),
        },
    }
    return out


def _ratio_or_nan(num: float, den: float) -> float:
    return round(num / den, 4) if den else float("nan")


#: evaluate_stc_all's unweighted means over clips, each also returned under its old
#: `pooled_*` name. Everything else called pooled_* there IS pooled (sums, then a ratio).
MEAN_KEYS = ("mean_coverage", "mean_unattributed_frac", "mean_attribution_contaminated",
             "mean_clamped_frac")


def evaluate_stc_all(seqs=None, variant: str = "p05", every: int = STC_EVERY,
                     gt_method: str = "kalman", window_s: float | None = None,
                     settle_s: float = 0.0, verbose: bool = True, oracle: str | None = None,
                     credit: str = "fragment", method: str = "kalman",
                     common_support: bool = False, our_cals: dict | None = None) -> dict:
    """The nine clips, per clip and pooled. Pooled = sum of metres, not mean of ratios;
    the MEAN_KEYS are the exception, and say so in their names.

    Held one clip at a time on purpose: a full-half tracks file is 594,000
    detections and three lanes share this machine.
    """
    import contam_eval as C  # noqa: PLC0415
    rows, ratios, peaks, gt_peaks, mpp = [], [], [], [], []
    for s in (seqs or C.STC_SEQS):
        r = evaluate_stc(s, variant, every, gt_method, window_s, settle_s, oracle=oracle, credit=credit,
                         method=method, common_support=common_support,
                         our_cal=(our_cals or {}).get(s))
        # Flattened before the per-player rows are dropped: `product.player.
        # distance.spread` is a spread over PLAYERS, and a median of nine
        # per-clip medians is not that. Held one clip at a time all the same.
        ratios += [p["ratio_matched"] for p in r["players"] if p["gt_matched_m"] > 0]
        mpp += [p["m_per_px"] for p in r["players"] if p["gt_matched_m"] > 0]
        peaks += [p["our_peak_ms"] for p in r["players"] if p["fragments"]]
        gt_peaks += [p["gt_peak_ms"] for p in r["players"]]
        r.pop("players")
        rows.append(r)
        if verbose:
            q, sp = r["squad"], r["speed"]
            print(f"{s}  ratio_raw {q['ratio_raw']:.3f}  matched {q['ratio_matched']:.3f}  "
                  f"cov {q['coverage']:.3f}  median {r['spread']['median_ratio']:.3f}  "
                  f"HSR {sp['hsr_ratio']:.2f}  sprints {sp['sprint_ratio']:.2f}  "
                  f"clamped {sp['clamped_frac']:.3f}", flush=True, file=sys.stderr)

    def tot(sec, k):
        return sum(r[sec][k] for r in rows)

    out = {
        "variant": variant, "oracle": oracle, "credit": credit, "settle_s": settle_s, "clips": len(rows),
        "method": method, "common_support": common_support,
        "sequences": [r["seq"] for r in rows],
        "pooled_ratio_raw": round(tot("squad", "ours_m") / tot("squad", "gt_m"), 4),
        "pooled_ratio_matched": round(
            tot("squad", "ours_m") / tot("squad", "gt_matched_m"), 4),
        # MEAN OF PER-CLIP VALUES, not pooled: every clip weighs the same whatever
        # its player count or frames. Named mean_*; the pooled_* spelling is kept
        # as an alias because check.py's REAL_KEYS, the n22-n26 result files and
        # measured-claim commands in docs/experiments read it - same value, misleading name.
        "mean_coverage": round(float(np.mean([r["squad"]["coverage"] for r in rows])), 4),
        "pooled_ours_m": tot("squad", "ours_m"), "pooled_gt_m": tot("squad", "gt_m"),
        "pooled_gt_matched_m": tot("squad", "gt_matched_m"),
        "pooled_orphan_m": tot("squad", "orphan_m"),
        "pooled_uncredited_m": tot("squad", "uncredited_m"),
        "pooled_orphan_ids": tot("squad", "orphan_ids"),
        "pooled_fragments": tot("squad", "fragments"),
        "mean_unattributed_frac": round(float(np.mean(
            [r["squad"]["unattributed_frac"] for r in rows])), 4),
        "mean_attribution_contaminated": round(float(np.mean(
            [r["squad"]["attribution_contaminated"] for r in rows])), 4),
        # THE SPREAD, over all 9 x 22 players pooled, not over nine clip medians.
        # A scale error moves the median and leaves the spread alone;
        # contamination and mis-attribution move the spread and leave the median
        # alone. Both are here so the reader picks, not the author.
        "player_ratio_n": len(ratios),
        "player_ratio_median": round(float(np.median(ratios)), 4),
        "player_ratio_iqr": [round(float(np.percentile(ratios, 25)), 4),
                             round(float(np.percentile(ratios, 75)), 4)],
        "player_ratio_min": round(float(min(ratios)), 4),
        "player_ratio_max": round(float(max(ratios)), 4),
        "player_ratio_cv": round(float(np.std(ratios) / np.mean(ratios)), 4),
        # Detector noise predicts a strong positive correlation here;
        # contamination predicts none. Pooled across clips, so a clip-to-clip
        # difference in camera geometry contributes to it as well as a
        # within-clip one - which is a real part of the question at n=9 venues.
        "player_ratio_corr_m_per_px": round(float(np.corrcoef(ratios, mpp)[0, 1]), 4),
        # Peak speed, both sides, because neither is quotable alone: ours is a
        # max over fragments that contain id switches and GT's is a max over one
        # real player - on a 1% grid, so GT's own max is quantisation through
        # the filter and not an athlete either. The medians are the readable pair.
        "player_peak_median_ms": round(float(np.median(peaks)), 4),
        "player_gt_peak_median_ms": round(float(np.median(gt_peaks)), 4),
        "player_peak_n": len(peaks),
        "player_gt_n": len(gt_peaks),
        "median_of_per_player_ratio": round(float(np.median(
            [r["spread"]["median_ratio"] for r in rows])), 4),
        "per_clip_ratio_matched_min": round(min(r["squad"]["ratio_matched"] for r in rows), 4),
        "per_clip_ratio_matched_max": round(max(r["squad"]["ratio_matched"] for r in rows), 4),
        "clips_over_one": sum(r["squad"]["ratio_matched"] > 1 for r in rows),
        "team_ratio_matched": {
            k: round(sum(r["teams"][k]["ours_m"] for r in rows) /
                     sum(r["teams"][k]["gt_matched_m"] for r in rows), 4)
            for k in ("home", "away")},
        "team_coverage": {k: round(float(np.mean([r["teams"][k]["coverage"] for r in rows])), 4)
                          for k in ("home", "away")},
        "pooled_hsr_ratio": round(tot("speed", "our_hsr_m") /
                                  tot("speed", "gt_matched_hsr_m"), 4),
        # NaN, not max(den, 1): with no matched GT sprint there is no ratio, and a
        # clamped denominator reported our raw sprint COUNT as a ratio. NaN cannot
        # reach the golden (api.py rule 2), so a consumer FAILS rather than quotes it.
        "pooled_sprint_ratio": _ratio_or_nan(tot("speed", "our_sprints"),
                                             tot("speed", "gt_matched_sprints")),
        "pooled_our_hsr_m": tot("speed", "our_hsr_m"),
        "pooled_gt_matched_hsr_m": tot("speed", "gt_matched_hsr_m"),
        "pooled_gt_hsr_m": tot("speed", "gt_hsr_m"),
        "pooled_our_sprints": tot("speed", "our_sprints"),
        "pooled_gt_matched_sprints": tot("speed", "gt_matched_sprints"),
        "pooled_gt_sprints": tot("speed", "gt_sprints"),
        "pooled_our_observed_s": tot("speed", "our_observed_s"),
        "pooled_gt_alt_hsr_m": tot("speed", "gt_alt_hsr_m"),
        "pooled_gt_alt_sprints": tot("speed", "gt_alt_sprints"),
        "mean_clamped_frac": round(float(np.mean(
            [r["speed"]["clamped_frac"] for r in rows])), 4),
        "our_peak_max_ms": max(r["speed"]["our_peak_max_ms"] for r in rows),
        "gt_peak_max_ms": max(r["speed"]["gt_peak_max_ms"] for r in rows),
        "per_clip": {r["seq"]: r for r in rows},
    }
    for k in MEAN_KEYS:
        out["pooled_" + k[len("mean_"):]] = out[k]      # the old name, same value
    return out


# ---------------------------------------------------------------------------


#: N22's arms, in the order the header lists them. None is the shipped tracks.
N22_ARMS = (None, "drop", "identity", "position", "both")


def _oracle_rows(rows: list[dict], owner: dict, gt: dict, cal, mode: str,
                 feet: dict | None = None, feedpx: dict | None = None,
                 mapper=None) -> tuple[list, dict]:
    """N22: the shipped rows with one kind of error replaced by the truth.

    `owner` is the shipped owner map and is reused, never recomputed. Every
    arm drops the rows it has no owner for, because there is no truth to put in
    their place. N23's two arms take their pixels from `feet` ({frame: {mot id:
    (u, v)}}) and `feedpx` ({gt id: {frame: (u, v)}}), passed in so the
    substitution can be checked on arithmetic. N26's two arms take a `mapper`,
    clip pixels -> (pitch metres, placed?), and re-enter as the shipped map's
    pixels at those metres. -> (rows, owner map re-indexed).
    """
    if mode not in ("drop", "identity", "position", "both", "gtpix", "feedpix", "tps21", "refmap", "feedref"):
        raise ValueError(f"unknown oracle arm {mode!r}")
    if (mode == "gtpix" and feet is None) or (mode in ("feedpix", "feedref") and feedpx is None) or \
            (mode in ("tps21", "refmap", "feedref") and mapper is None):
        raise ValueError(f"the {mode} arm needs its ground-truth pixels passed in")
    pos: dict = {}
    if mode in ("position", "both"):
        for g, t in gt.items():
            pos[g] = {int(f): (float(u), float(v))
                      for f, (u, v) in zip(t["f"], to_px(cal, np.asarray(t["xy"], float)))}
    mapped: dict = {}
    if mode in ("tps21", "refmap", "feedref"):
        idx = [i for i in range(len(rows)) if owner.get(i) is not None and
               (mode != "feedref" or int(rows[i]["f"]) in feedpx.get(owner[i], {}))]
        src = ([feedpx[owner[i]][int(rows[i]["f"])] for i in idx] if mode == "feedref" else
               [(rows[i]["x"], rows[i]["y"]) for i in idx])
        X, placed = mapper(np.array(src, float).reshape(-1, 2))
        px = to_px(cal, np.asarray(X, float)) if idx else np.zeros((0, 2))
        mapped = {i: (float(u), float(v)) for i, (u, v), k in zip(idx, px, placed) if k}
    run_id: dict[int, int] = {}
    if mode in ("identity", "both"):
        # A new id every time a fragment's owner CHANGES, walked in frame order:
        # A,B,A is three pieces. One id per (fragment, owner) - the first version -
        # joined the two A runs across B's frames, and a gap that short counts as
        # observed, so the arm kept part of the error it exists to remove.
        nxt, last = 0, {}
        for i in sorted(owner, key=lambda i: (int(rows[i]["id"]), int(rows[i]["f"]))):
            tid, g = int(rows[i]["id"]), owner[i]
            if tid not in last or last[tid][0] != g:
                last[tid] = (g, nxt)
                nxt += 1
            run_id[i] = last[tid][1]
    out, own = [], {}
    for i, r in enumerate(rows):
        g = owner.get(i)
        if g is None:
            continue
        r = dict(r)
        if mode in ("position", "both", "gtpix", "feedpix", "tps21", "refmap", "feedref"):
            f = int(r["f"])
            p = (pos.get(g, {}).get(f) if mode in ("position", "both") else
                 feet.get(f, {}).get(int(g)) if mode == "gtpix" else
                 feedpx.get(g, {}).get(f) if mode == "feedpix" else
                 mapped.get(i))
            if p is None:
                continue
            r["x"], r["y"] = p
        if mode in ("identity", "both"):
            r["id"] = run_id[i]
        own[len(out)] = g
        out.append(r)
    return out, own


def n22_verdict(res: dict) -> dict:
    """The header's rule, in code. `res` maps arm name -> evaluate_stc_all output."""
    out = {}
    for k in ("pooled_ratio_matched", "pooled_sprint_ratio", "pooled_hsr_ratio"):
        base = res["drop"][k]
        if base <= 1.0:
            out[k] = {"driver": "no over-read to apportion", "drop": base}
            continue
        ids = (base - res["identity"][k]) / (base - 1.0)
        pos = (base - res["position"][k]) / (base - 1.0)
        out[k] = {"identity_share": round(ids, 4), "position_share": round(pos, 4),
                  "driver": ("IDENTITY" if ids >= 0.5 > pos else
                             "POSITION" if pos >= 0.5 > ids else
                             "BOTH" if ids >= 0.5 and pos >= 0.5 else "NEITHER ALONE")}
    return out


def run_n22(seqs=None) -> dict:
    """N22 end to end: the five arms on `.p21`, the rule, one results file."""
    res = {("shipped" if a is None else a): evaluate_stc_all(seqs, "p21", oracle=a, verbose=False)
           for a in N22_ARMS}
    verdict = n22_verdict(res)
    keys = ("pooled_ratio_matched", "pooled_sprint_ratio", "pooled_hsr_ratio",
            "pooled_coverage", "pooled_unattributed_frac", "pooled_attribution_contaminated",
            # both sides of ratio_matched, because the identity arm moves the
            # DENOMINATOR (docstring N22, REVIEW) and a ratio alone hides which
            "pooled_ours_m", "pooled_gt_matched_m")
    table = {arm: {k: r[k] for k in keys} for arm, r in res.items()}
    with open(os.path.join(STC, "n22_results.json"), "w") as fh:
        json.dump({"arms": table, "verdict": verdict,
                   "protocol": "model/validation/dist_eval.py docstring N22"}, fh, indent=1)
    for arm, row in table.items():
        print(f"{arm:9s} " + "  ".join(f"{k.replace('pooled_', '')} {v}" for k, v in row.items()))
    for k, v in verdict.items():
        print(f"N22 {k.replace('pooled_', '')}: {v}")
    return {"arms": table, "verdict": verdict}



def G_mot_feet(seq: str) -> dict:
    """N23's detector-free pixels: the MOT boxes' feet, st2_team_gt's reading of them."""
    import st2_team_gt as G  # noqa: PLC0415
    return G.mot_feet(seq)


def _vendor_px(seq: str, gt: dict) -> dict:
    """N23: the position feed's metres through ST2's OWN vendor chain into clip
    pixels - the projection _selfcheck_stc pins to within 25 px of the MOT feet.
    -> {gt id: {frame: (u, v)}}, valid pixels only."""
    import st2_mot as T  # noqa: PLC0415
    from st2_overlay import projector  # noqa: PLC0415
    proj = projector(seq, T._wh(seq))
    out = {}
    for g, t in gt.items():
        px, ok = proj([tuple(p) for p in np.asarray(t["xy"], float)])
        out[g] = {int(f): (float(u), float(v)) for f, (u, v), k in zip(t["f"], px, ok) if k}
    return out


def _tps_correction(p: np.ndarray, X: np.ndarray):
    """N26: a thin-plate spline through the residuals X - p at the clicks, applied
    everywhere - a hull switch made a corrected player jump metres in one step (review
    of 5e65e06). -> (correct, checks): correct(q) gives corrected metres and which
    points lie inside the clicks' hull; checks say whether the spline returns the
    clicks exactly, whether the corrected map folds inside the hull, and how far it
    stretches there (the larger singular value of its Jacobian)."""
    from scipy.interpolate import RBFInterpolator  # noqa: PLC0415
    from scipy.spatial import Delaunay  # noqa: PLC0415
    tps = RBFInterpolator(p, X - p, kernel="thin_plate_spline", smoothing=0.0)
    hull = Delaunay(p)

    def correct(q):
        q = np.asarray(q, float)
        return q + tps(q), hull.find_simplex(q) >= 0

    lo, hi = p.min(0), p.max(0)
    g = np.stack(np.meshgrid(np.arange(lo[0], hi[0], 1.0), np.arange(lo[1], hi[1], 1.0),
                             indexing="ij"), -1).reshape(-1, 2)
    g = g[hull.find_simplex(g) >= 0]
    e = 0.05
    jx = (g + [e, 0] + tps(g + [e, 0]) - (g - [e, 0]) - tps(g - [e, 0])) / (2 * e)
    jy = (g + [0, e] + tps(g + [0, e]) - (g - [0, e]) - tps(g - [0, e])) / (2 * e)
    det = jx[:, 0] * jy[:, 1] - jx[:, 1] * jy[:, 0]
    sv = np.linalg.svd(np.stack([jx, jy], -1), compute_uv=False)[:, 0] if len(g) else np.zeros(1)
    return correct, {"exact_m": float(np.abs(p + tps(p) - X).max()),
                     "folds": bool((det <= 0).any()),
                     "min_det": float(det.min()) if len(det) else None,
                     "stretch_p90": float(np.percentile(sv, 90)), "stretch_max": float(sv.max())}


def _tps21(seq: str, cal):
    """N26's tps21 arm: the shipped map corrected through its residuals at the 21
    clicks it was solved from. -> (mapper, checks); checks count the rows inside the hull."""
    import calib_product as CP  # noqa: PLC0415
    P, clicks = CP.clicks_of(seq)
    names = sorted(clicks)
    p = np.asarray(cal.to_pitch(np.array([clicks[n] for n in names], float)), float)
    X = np.array([P.landmarks()[n] for n in names], float)
    correct, checks = _tps_correction(p, X)
    checks.update(clicks=len(names), inside=0, rows=0)

    def mapper(uv):
        out, inside = correct(np.asarray(cal.to_pitch(uv), float))
        checks["inside"] += int(inside.sum())
        checks["rows"] += len(inside)
        return out, np.ones(len(out), bool)
    return mapper, checks


def _smooth_inverse(G: np.ndarray, px: np.ndarray, max_px: float):
    """N26: invert a forward map that rounds to whole pixels, smoothly. `G` are grid
    metres and `px` the pixels they project to; many grid points share a pixel, so
    each distinct pixel stands for its plateau's centroid and a query pixel reads the
    centroids linearly interpolated over their triangulation. -> mapper(uv) ->
    (metres, placed), placed where a projected pixel lies within `max_px` and the
    query is inside the triangulation."""
    from scipy.interpolate import LinearNDInterpolator  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415
    upx, k, n = np.unique(np.asarray(px, float), axis=0, return_inverse=True, return_counts=True)
    k = np.asarray(k).ravel()
    G = np.asarray(G, float)
    cen = np.stack([np.bincount(k, G[:, 0]) / n, np.bincount(k, G[:, 1]) / n], 1)
    lin, tree = LinearNDInterpolator(upx, cen), cKDTree(upx)

    def mapper(uv):
        uv = np.asarray(uv, float).reshape(-1, 2)
        X = lin(uv)
        placed = (tree.query(uv)[0] <= max_px) & np.isfinite(X).all(1)
        return np.where(placed[:, None], X, 0.0), placed
    return mapper


def _refmap(seq: str):
    """N26's refmap and feedref arms: ST2's own chain on calib_product's reference
    grid, inverted by `_smooth_inverse`. calib_product.reference picks an arbitrary
    grid point of a remap pixel's plateau, which is noise no other arm carries."""
    import calib_product as CP  # noqa: PLC0415
    P = CP.pitch_of(seq)
    gx = np.arange(-CP.GRID_MARGIN_M, P.length + CP.GRID_MARGIN_M + 1e-9, CP.GRID_M)
    gy = np.arange(-CP.GRID_MARGIN_M, P.width + CP.GRID_MARGIN_M + 1e-9, CP.GRID_M)
    G = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    px, ok = CP.projector(seq, CP.T._wh(seq))(G)
    return _smooth_inverse(G[ok], np.asarray(px)[ok], CP.MAX_GRID_PX)


#: N23's arms. The steps telescope: drop -> gtpix -> feedpix -> position.
N23_ARMS = ("drop", "gtpix", "feedpix", "position")


def n23_verdict(res: dict) -> dict:
    """The header's rule for N23, in code. `res` maps arm -> evaluate_stc_all output."""
    out = {}
    for k in ("pooled_sprint_ratio", "pooled_hsr_ratio", "pooled_ratio_matched"):
        base, pos = res["drop"][k], res["position"][k]
        if base <= 1.0 or base - pos <= 0:
            out[k] = {"driver": "no position over-read to apportion"}
            continue
        ex = base - 1.0
        det = (base - res["gtpix"][k]) / ex
        ann = (res["gtpix"][k] - res["feedpix"][k]) / ex
        mp = (res["feedpix"][k] - pos) / ex
        whole = det + ann + mp
        out[k] = {"detector_share": round(det, 4), "annotation_share": round(ann, 4),
                  "map_share": round(mp, 4), "position_share": round(whole, 4),
                  "driver": ("INCONCLUSIVE" if abs(ann) > whole / 3 else
                             "DETECTOR" if det >= 2 * whole / 3 else
                             "MAP" if mp >= 2 * whole / 3 else "MIXED")}
    return out


def run_n23(seqs=None) -> dict:
    """N23 end to end: four arms on `.p21`, the rule, one results file."""
    res = {a: evaluate_stc_all(seqs, "p21", oracle=a, verbose=False) for a in N23_ARMS}
    verdict = n23_verdict(res)
    keys = ("pooled_sprint_ratio", "pooled_hsr_ratio", "pooled_ratio_matched",
            "pooled_coverage", "pooled_unattributed_frac")
    table = {arm: {k: r[k] for k in keys} for arm, r in res.items()}
    with open(os.path.join(STC, "n23_results.json"), "w") as fh:
        json.dump({"arms": table, "verdict": verdict,
                   "protocol": "model/validation/dist_eval.py docstring N23"}, fh, indent=1)
    for arm, row in table.items():
        print(f"{arm:9s} " + "  ".join(f"{k.replace('pooled_', '')} {v}" for k, v in row.items()))
    for k, v in verdict.items():
        print(f"N23 {k.replace('pooled_', '')}: {v}")
    return {"arms": table, "verdict": verdict}



def _credited(trk: list[dict], owners: list, cal, fps: float, window_s=None,
              settle_s: float = 0.0) -> tuple[dict, float]:
    """N24: one fragment through the product's estimator ONCE, each step credited to
    the ground-truth player BOTH of its ends belong to.

    A step that is not a player's is masked for that player exactly as
    `stats._windowed` masks an unobserved one - a window holding it is dropped - so
    the filter runs once over the whole fragment and only the crediting changes. A
    fragment owned throughout by one player is credited exactly what frag_stats
    gives it. -> ({gt id: {m, hsr_m, sprints, peak, frames}}, the fragment's metres)
    """
    order = sorted(range(len(trk)), key=lambda k: trk[k]["f"])
    trk, own = [trk[k] for k in order], [owners[k] for k in order]
    t = np.array([p["f"] / fps for p in trk], float)
    keep, stats.MAX_SPEED = stats.MAX_SPEED, np.inf
    try:
        v, dt, real, xy = stats.speed_series(trk, cal, fps, "kalman")
    finally:
        stats.MAX_SPEED = keep
    whole, _obs = stats._windowed(xy, t, real, window_s)
    v = np.where(_settle(t, real, settle_s), 0.0, v)
    out = {}
    for g in {o for o in own if o is not None}:
        step = np.array([own[k] == g and own[k + 1] == g for k in range(len(own) - 1)], bool)
        m = real & step
        if not m.any():
            continue
        dist, _obs = stats._windowed(xy, t, m, window_s)
        fast = m & (v >= stats.SPRINT_SPEED)
        ends = np.flatnonzero(step)
        out[g] = {"m": dist, "hsr_m": float(np.sum(v[fast] * dt[fast])), "s": float(np.sum(dt[m])),
                  "sprints": stats._count_sprints(v, t[1:], m),
                  "peak": float(v[m].max()),
                  "frames": {int(trk[k]["f"]) for k in ends} | {int(trk[k + 1]["f"]) for k in ends}}
    return out, whole


def n24_verdict(res: dict) -> dict:
    """The header's N24 rule, in code. `res` holds the control's pooled ratio and
    coverage, the N22 arms under detection crediting, and what ships under it."""
    control_ok = abs(res["control"] - 1.0) < 1e-3 and abs(res["control_cov"] - 1.0) < 1e-3
    shift = abs(res["arms_running"]["identity"]["pooled_ratio_matched"]
                - res["arms_running"]["drop"]["pooled_ratio_matched"])
    valid = control_ok and shift <= 0.02
    running = res["ships_running"]["pooled_ratio_matched"]
    follows = ("VOID" if not valid else
               "IDENTITY BOOKKEEPING - association work" if running <= 1.05 else
               "PHYSICAL OVER-READ - position work")
    return {"control_ok": control_ok, "identity_shift": round(shift, 4), "valid": valid,
            "running_ratio_matched": running, "follows": follows}


def run_n24(seqs=None) -> dict:
    """N24 end to end: the control, what ships under both creditings, N22's arms
    under detection crediting, the rule, one results file."""
    import contam_eval as C  # noqa: PLC0415
    seqs = tuple(seqs or C.STC_SEQS)
    ctl = [evaluate_stc(s, "p21id", control=True, credit="detection") for s in seqs]
    control = sum(c["squad"]["ours_m"] for c in ctl) / sum(c["squad"]["gt_matched_m"] for c in ctl)
    control_cov = float(np.mean([c["squad"]["coverage"] for c in ctl]))
    ships = {c: evaluate_stc_all(seqs, "p21id", verbose=False, credit=c) for c in ("fragment", "detection")}
    arms = {("shipped" if a is None else a): evaluate_stc_all(seqs, "p21", oracle=a, verbose=False,
                                                           credit="detection")
            for a in N22_ARMS}
    verdict = n24_verdict({"control": control, "control_cov": control_cov,
                           "arms_running": arms, "ships_running": ships["detection"]})
    keys = ("pooled_ratio_raw", "pooled_ratio_matched", "pooled_coverage", "pooled_sprint_ratio",
            "pooled_hsr_ratio", "player_ratio_median", "player_ratio_cv", "player_ratio_n",
            "pooled_ours_m", "pooled_gt_matched_m", "pooled_uncredited_m")
    table = {"ships": {c: {k: r[k] for k in keys} for c, r in ships.items()},
             "arms_running": {a: {k: r[k] for k in keys} for a, r in arms.items()}}
    with open(os.path.join(STC, "n24_results.json"), "w") as fh:
        json.dump({"control": round(control, 4), "control_coverage": round(control_cov, 4),
                   **table, "verdict": verdict,
                   "protocol": "model/validation/dist_eval.py docstring N24"}, fh, indent=1)
    print(f"control ratio {control:.4f} at coverage {control_cov:.4f}")
    for c, r in table["ships"].items():
        print(f"ships  {c:9s} " + "  ".join(f"{k.replace('pooled_', '')} {v}" for k, v in r.items()))
    for a, r in table["arms_running"].items():
        print(f"arm    {a:9s} " + "  ".join(f"{k.replace('pooled_', '')} {v}" for k, v in r.items()))
    print(f"N24 {verdict}")
    return {"table": table, "verdict": verdict}


class _Flat:
    """Identity calibration: one 'pixel' IS one metre. to_pitch and H agree."""
    H = np.eye(3)

    @staticmethod
    def to_pitch(px):
        return np.asarray(px, float)


#: N25's inputs: (name, tracks variant, N22/N23 arm). "ships" is `.p21id` as it ships.
N25_INPUTS = (("ships", "p21id", None), ("drop", "p21", "drop"), ("gtpix", "p21", "gtpix"),
              ("feedpix", "p21", "feedpix"), ("position", "p21", "position"), ("both", "p21", "both"))
#: N25's conditions: (our side's estimator, scored on the adopted filter's steps only).
N25_CONDITIONS = {"K": ("kalman", False), "Kc": ("kalman", True), "B": ("butter", False)}
#: The ground-truth side of every input, which no condition may move.
N25_RULER = ("pooled_gt_matched_m", "pooled_gt_matched_hsr_m", "pooled_gt_matched_sprints",
             "pooled_gt_hsr_m", "pooled_gt_sprints")


def n25_verdict(res: dict) -> dict:
    """The header's N25 rule, in code. `res` maps input -> condition -> evaluate_stc_all output."""
    ruler = all(res[i][c][k] == res[i]["K"][k] for i in res for c in res[i] for k in N25_RULER)
    K, B = res["drop"]["K"], res["drop"]["B"]
    level = {"hsr": B["pooled_gt_alt_hsr_m"] / K["pooled_gt_hsr_m"],
             "sprints": B["pooled_gt_alt_sprints"] / K["pooled_gt_sprints"]}
    support = B["pooled_our_observed_s"] / K["pooled_our_observed_s"]
    valid = ruler and min(level.values()) >= 0.8 and support >= 0.9
    closure = {}
    d, p = res["drop"], res["position"]
    for m, k in (("hsr", "pooled_hsr_ratio"), ("sprints", "pooled_sprint_ratio")):
        mul = [d[c][k] / p[c][k] - 1 if p[c][k] > 0 else float("inf") for c in ("Kc", "B")]
        add = [d[c][k] - p[c][k] for c in ("Kc", "B")]
        closure[m] = {w: (round(1 - e[1] / e[0], 4) if 0 < e[0] < float("inf") else None)
                      for w, e in (("multiplicative", mul), ("additive", add))}
    c = [v for m in closure.values() for v in m.values()]
    follows = ("VOID" if not valid else
               "NO EXCESS TO CLOSE" if None in c else
               "ESTIMATOR" if min(c) >= 0.5 else
               "NOT THE ESTIMATOR" if max(c) < 0.25 else "MIXED")
    return {"ruler_held": ruler, "level": {k: round(v, 4) for k, v in level.items()},
            "support": round(support, 4), "valid": valid, "closure": closure, "follows": follows}


def run_n25(seqs=None) -> dict:
    """N25 end to end: six inputs under three conditions, the rule, one results file."""
    res = {name: {c: evaluate_stc_all(seqs, variant, oracle=arm, verbose=False, method=m, common_support=cs)
                  for c, (m, cs) in N25_CONDITIONS.items()}
           for name, variant, arm in N25_INPUTS}
    verdict = n25_verdict(res)
    keys = ("pooled_hsr_ratio", "pooled_sprint_ratio", "pooled_ratio_matched", "pooled_our_hsr_m",
            "pooled_our_sprints", "pooled_our_observed_s", "player_peak_median_ms",
            "player_gt_peak_median_ms", "pooled_unattributed_frac", "pooled_gt_alt_hsr_m",
            "pooled_gt_alt_sprints", *N25_RULER)
    table = {i: {c: {k: r[k] for k in keys} for c, r in rc.items()} for i, rc in res.items()}
    with open(os.path.join(STC, "n25_results.json"), "w") as fh:
        json.dump({"inputs": table, "verdict": verdict,
                   "protocol": "model/validation/dist_eval.py docstring N25"}, fh, indent=1)
    for i, rc in table.items():
        for c, row in rc.items():
            print(f"{i:8s} {c:2s} " + "  ".join(f"{k.replace('pooled_', '')} {v}" for k, v in row.items()))
    print(f"N25 {verdict}")
    return {"inputs": table, "verdict": verdict}


#: N26's arms: the shipped map, two better maps, the control, and the truth.
N26_ARMS = ("drop", "tps21", "refmap", "feedref", "position")
#: The ground truth over its full tracks, which no arm may move.
N26_RULER = ("pooled_gt_m", "pooled_gt_hsr_m", "pooled_gt_sprints")
#: The coverage-matched ground truth, which an arm that drops rows moves a little.
N26_MATCHED = ("pooled_gt_matched_m", "pooled_gt_matched_hsr_m", "pooled_gt_matched_sprints")


def _n26_hsr(arm: dict, clips) -> float:
    """An arm's HSR ratio pooled over `clips`, from its per-clip sums."""
    c = [arm["per_clip"][s]["speed"] for s in clips]
    return sum(x["our_hsr_m"] for x in c) / sum(x["gt_matched_hsr_m"] for x in c)


def _n26_label(res: dict, clips) -> tuple:
    """The rule on one set of clips. -> (label, share(refmap), share(tps21))."""
    r = {a: _n26_hsr(res[a], clips) for a in ("drop", "tps21", "refmap", "position")}
    ex = r["drop"] - r["position"]
    if ex <= 0:
        return "NO EXCESS TO APPORTION", None, None
    ref, tps = (r["drop"] - r["refmap"]) / ex, (r["drop"] - r["tps21"]) / ex
    return ("NOT THE MAP ALONE" if ref < 1 / 3 else "UNCLEAR" if ref < 2 / 3 else
            "MAP + CLICKS CARRY IT" if tps >= ref / 2 else "MAP + INTERIOR LANDMARKS NEEDED"), ref, tps


def n26_verdict(res: dict) -> dict:
    """The header's N26 rule, in code. `res` maps arm -> evaluate_stc_all output."""
    clips = list(res["drop"]["per_clip"])
    full = all(res[a][k] == res["drop"][k] for a in res for k in N26_RULER)
    matched = all(abs(res[a][k] - res["drop"][k]) <= 0.02 * abs(res["drop"][k])
                  for a in res for k in N26_MATCHED)
    hsr = {a: _n26_hsr(res[a], clips) for a in res}
    excess = hsr["drop"] - hsr["position"]
    control = abs(hsr["feedref"] - hsr["position"]) / excess if excess > 0 else float("inf")
    dropped = max(res["refmap"]["per_clip"][c]["squad"]["unattributed_frac"]
                  - res["drop"]["per_clip"][c]["squad"]["unattributed_frac"] for c in clips)
    valid = full and matched and control <= 0.1 and dropped <= 0.02
    checks = [c["map_checks"] for c in res["tps21"]["per_clip"].values()]
    tps_ok = all(c["exact_m"] <= 1e-6 and not c["folds"] for c in checks)
    label, ref, tps = _n26_label(res, clips)
    loo = {c: _n26_label(res, [x for x in clips if x != c])[0] for c in clips}
    same = (lambda l: l.split(" +")[0]) if not tps_ok else (lambda l: l)  # noqa: E731
    differ = sum(same(l) != same(label) for l in loo.values())
    follows = ("VOID" if not valid else
               ("MAP; tps21 VOID" if label.startswith("MAP") and not tps_ok else label)
               + (f" (FRAGILE: {differ} of {len(clips)} leave-one-out verdicts differ)" if differ else ""))
    beside = {}
    for m, k in (("sprints", "pooled_sprint_ratio"), ("distance", "pooled_ratio_matched")):
        ex = res["drop"][k] - res["position"][k]
        beside[m] = {a: (round((res["drop"][k] - res[a][k]) / ex, 4) if ex > 0 else None)
                     for a in ("tps21", "refmap")}
    return {"ruler_held": full and matched, "control": round(control, 4),
            "refmap_dropped_max": round(dropped, 4), "valid": valid, "tps21_valid": tps_ok,
            "tps21_inside": round(sum(c["inside"] for c in checks) / max(sum(c["rows"] for c in checks), 1), 4),
            "hsr": {a: round(v, 4) for a, v in hsr.items()},
            "share_hsr": {"refmap": None if ref is None else round(ref, 4),
                          "tps21": None if tps is None else round(tps, 4)},
            "share_beside": beside, "leave_one_out": loo, "follows": follows}


def run_n26(seqs=None) -> dict:
    """N26 end to end: five arms on `.p21`, the rule, one results file."""
    res = {a: evaluate_stc_all(seqs, "p21", oracle=a, verbose=False) for a in N26_ARMS}
    verdict = n26_verdict(res)
    keys = ("pooled_hsr_ratio", "pooled_sprint_ratio", "pooled_ratio_matched", "pooled_our_hsr_m",
            "pooled_our_sprints", "pooled_ours_m", "pooled_unattributed_frac",
            "player_peak_median_ms", *N26_RULER, *N26_MATCHED)
    table = {a: {k: r[k] for k in keys} for a, r in res.items()}
    per_clip = {a: {c: round(r["per_clip"][c]["speed"]["hsr_ratio"], 4) for c in r["per_clip"]}
                for a, r in res.items()}
    checks = {c: r["map_checks"] for c, r in res["tps21"]["per_clip"].items()}
    with open(os.path.join(STC, "n26_results.json"), "w") as fh:
        json.dump({"arms": table, "per_clip_hsr_ratio": per_clip, "tps21_checks": checks,
                   "verdict": verdict, "protocol": "model/validation/dist_eval.py docstring N26"}, fh, indent=1)
    for a, row in table.items():
        print(f"{a:9s} " + "  ".join(f"{k.replace('pooled_', '')} {v}" for k, v in row.items()))
    print(f"per-clip HSR ratio: {per_clip}")
    print(f"tps21 checks: {checks}")
    print(f"N26 {verdict}")
    return {"arms": table, "verdict": verdict}


#: N27's population: N21's nine calibrated clips minus 132877, whose joined feed is mirrored.
N27_SEQS = ("117092", "117093", "118575", "118576", "118577", "118578", "128057", "128058")
N27_SIGMAS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)
N27_SEEDS = (0, 1, 2)


class _ScaledCal:
    """A calibration whose metres are scaled about the pitch centre by `s`: N27's movement
    scale matched to the truth. Its Jacobian scales with it, and H is the original's."""

    def __init__(self, cal, s: float, centre=(52.5, 34.0)):
        self.cal, self.s, self.c, self.H = cal, float(s), np.asarray(centre, float), cal.H

    def to_pitch(self, px):
        return self.c + self.s * (np.asarray(self.cal.to_pitch(px), float).reshape(-1, 2) - self.c)

    def jacobian(self, px):
        uv = np.asarray(px, float).reshape(-1, 2)
        J = self.cal.jacobian(uv) if hasattr(self.cal, "jacobian") else \
            stats.homography_jacobian(np.asarray(self.cal.H, float), uv)
        return self.s * J


def _chord_slope_arrays(tid, f, g, X, G, fps: float, every: int = STC_EVERY,
                        secs: float = 4.0, min_len: float = 5.0) -> tuple:
    """N27 B: least-squares slope of chord displacement through a map (X) against the
    truth's (G), over runs that stay on one fragment and one owner with no missed frame,
    chords `secs` apart and at least `min_len` metres of true movement.
    -> (slope, chords, median chord length ratio)."""
    order = np.lexsort((np.asarray(f), np.asarray(tid)))
    tid, f, g = np.asarray(tid)[order], np.asarray(f)[order], np.asarray(g)[order]
    X, G = np.asarray(X, float)[order], np.asarray(G, float)[order]
    brk = np.r_[True, (tid[1:] != tid[:-1]) | (f[1:] - f[:-1] != every) | (g[1:] != g[:-1])]
    starts = np.flatnonzero(brk)
    ends = np.r_[starts[1:], len(f)]
    n = int(round(secs * fps / every))
    parts = [np.arange(a, b - n) for a, b in zip(starts, ends) if b - a > n]
    if not parts:
        return float("nan"), 0, float("nan")
    A = np.concatenate(parts)
    dG, dX = G[A + n] - G[A], X[A + n] - X[A]
    keep = np.isfinite(dG).all(1) & np.isfinite(dX).all(1) & (np.linalg.norm(dG, axis=1) >= min_len)
    if not keep.any():
        return float("nan"), 0, float("nan")
    ratio = np.linalg.norm(dX[keep], axis=1) / np.linalg.norm(dG[keep], axis=1)
    return float(np.sum(dX[keep] * dG[keep]) / np.sum(dG[keep] ** 2)), int(keep.sum()), float(np.median(ratio))


def _chord_slope(seq: str, variant: str, cal) -> tuple:
    """N27 B on one clip's tracks through `cal`."""
    import contam_eval as C  # noqa: PLC0415
    blob = json.load(open(f"{STC}/{seq}.{variant}.tracks.json"))
    rows, fps = blob["detections"], float(blob["fps"])
    owner, _counts = C.owners_iou(seq, rows, STC_EVERY)
    look = {k: dict(zip(v["f"].tolist(), map(tuple, v["xy"]))) for k, v in stc_gt(seq, STC_EVERY).items()}
    idx = [i for i in owner if int(rows[i]["f"]) % STC_EVERY == 0]
    X = np.asarray(cal.to_pitch(np.array([[rows[i]["x"], rows[i]["y"]] for i in idx], float)), float)
    G = np.array([look.get(owner[i], {}).get(int(rows[i]["f"]), (np.nan, np.nan)) for i in idx], float)
    return _chord_slope_arrays([int(rows[i]["id"]) for i in idx], [int(rows[i]["f"]) for i in idx],
                               [owner[i] for i in idx], X, G, fps)


def _feet_pixels(seq: str, every_f: int = 20) -> tuple:
    """N27 A's population: MOT box feet on clip frames divisible by `every_f`, with the
    joined feed at the same frame and player. -> (feet px (N,2), feed metres (N,2))."""
    feet = G_mot_feet(seq)
    uv, G = [], []
    for g, t in stc_gt(seq, STC_EVERY).items():
        for fr, xy in zip(t["f"].tolist(), t["xy"]):
            p = feet.get(fr, {}).get(int(g)) if fr % every_f == 0 else None
            if p is not None:
                uv.append(p)
                G.append(xy)
    return np.array(uv, float), np.array(G, float)


def _n27_payloads(seq: str) -> dict:
    """N12's click sets re-solved by the UI's own solver: {(sigma, seed): payload or None}.
    Sigma 0 is N6's calibration on disk, which N12's --demo pins as that solve bit for bit."""
    import calib_product as CP  # noqa: PLC0415
    P, clean = CP.clicks_of(seq)
    names = list(clean)
    base = np.array([clean[n] for n in names], float)
    out = {(0.0, 0): json.load(open(f"{STC}/{seq}.p05.calibration.json"))}
    for sigma in N27_SIGMAS[1:]:
        for seed in N27_SEEDS:
            rng = np.random.default_rng([int(seq), seed, int(sigma * 10)])
            jit = base + rng.normal(0.0, sigma, base.shape)
            out[(sigma, seed)] = CP.solve(P, {n: list(map(float, q)) for n, q in zip(names, jit)})
    return out


def _as_tps(payload: dict) -> dict:
    return {**payload, "projection_type": "perspective_tps"}


class _HNoise:
    """A calibration read with its homography's noise model instead of its own Jacobian:
    N27's diagnostic that separates better positions from a different filter."""

    def __init__(self, cal):
        self.cal, self.H = cal, cal.H

    def to_pitch(self, px):
        return self.cal.to_pitch(px)


def _n27_cache(seq: str, tag: str, calib_file: str) -> None:
    """N21's detection cache copied for one N27 arm, with that arm's calibration digest
    written in. Disclosed in the protocol: the boxes stay the ones N6's pitch polygon gated."""
    import st2_product as ST  # noqa: PLC0415
    import track  # noqa: PLC0415
    raw = json.load(open(ST._p(seq, ".p21.dets.cache.json")))
    raw["calib"], raw["calib_sha"] = calib_file, track._calib_digest(calib_file)
    with open(ST._p(seq, f"{tag}.dets.cache.json"), "w") as fh:
        json.dump(raw, fh)


def _n27_share(speed: dict, ship: str, corr: str, clips) -> float | None:
    """Share of the shipped arm's pooled HSR excess over 1 the corrected arm removes, over
    `clips`; None when the shipped arm has no excess."""
    rs = sum(speed[ship][q][0] for q in clips) / sum(speed[ship][q][1] for q in clips)
    rc = sum(speed[corr][q][0] for q in clips) / sum(speed[corr][q][1] for q in clips)
    return None if rs <= 1.0 else (rs - rc) / (rs - 1.0)


def n27_verdict(res: dict) -> dict:
    """The header's N27 rule, in code. `res`: validity flags; position pooled per
    "sigma/seed", per clip, and correction switch-offs; slopes of ship0 and tps0 per clip;
    speed as {arm: {clip: (our HSR m, matched GT HSR m)}} at matched movement scale."""
    if not all(res["validity"].values()):
        return {"valid": False, "follows": "VOID"}
    pos, sl, sp = res["position"], res["slope"], res["speed"]
    clips = sorted(sl["ship0"])
    five = sorted(k for k in pos["pooled"] if k.startswith("5/"))
    small = [k for k in pos["pooled"] if float(k.split("/")[0]) <= 5]
    a = (all(pos["per_clip"][q][k]["tps"] < pos["per_clip"][q][k]["ship"] for q in clips for k in ["0/0", *five])
         and all(v["tps"] <= v["ship"] and v["tps_p90"] <= v["ship_p90"] and v["tps_far"] <= v["ship_far"]
                 for v in pos["pooled"].values())
         and not any(pos["fallback"][q].get(k, False) for q in clips for k in small))
    dev = {arm: float(np.median([abs(x - 1.0) for x in sl[arm].values()])) for arm in ("ship0", "tps0")}
    b = all(0.90 <= sl["tps0"][q] <= 1.10 for q in clips) and dev["tps0"] <= 0.05 and dev["tps0"] <= dev["ship0"]
    share0 = _n27_share(sp, "ship0", "tps0", clips)
    base = {"valid": True, "A": a, "B": b, "scale_deviation": {k: round(v, 4) for k, v in dev.items()}}
    if share0 is None:
        return {**base, "follows": "NOTHING TO REMOVE"}
    shares5 = {k.split("_")[1]: _n27_share(sp, k.replace("tps5", "ship5"), k, clips)
               for k in sorted(sp) if k.startswith("tps5_")}
    loo = {q: _n27_share(sp, "ship0", "tps0", [c for c in clips if c != q]) for q in clips}
    c = (share0 >= 0.5 and all(x is not None and x >= 0.5 for x in shares5.values())
         and all(x is not None and x >= 0.5 for x in loo.values()))
    failed = [n for n, ok in (("A position", a), ("B scale", b), ("C speed", c)) if not ok]
    rnd = lambda x: None if x is None else round(x, 4)  # noqa: E731
    return {**base, "C": c, "share_matched": {"0": rnd(share0), **{f"5/{k}": rnd(x) for k, x in shares5.items()}},
            "leave_one_out": {q: rnd(x) for q, x in loo.items()},
            "follows": "SHIP FOR STITCHED PANORAMAS" if not failed else "DO NOT SHIP (failed: " + ", ".join(failed) + ")"}


def run_n27(seqs=N27_SEQS) -> dict:
    """N27 end to end: the click sets, position at feet, the replays, scale and speed, the
    rule, one results file."""
    import glob  # noqa: PLC0415
    import calibrators as CAL  # noqa: PLC0415
    import st2_product as ST  # noqa: PLC0415
    stale = sorted(f for q in seqs for f in glob.glob(ST._p(q, ".p27*")))
    if stale:
        raise SystemExit(f"{len(stale)} N27 outputs already exist (e.g. {stale[0]}): a partial run is not "
                         "resumable, its replays would be reused silently. Remove them and re-run.")
    payloads = {q: _n27_payloads(q) for q in seqs}
    unsolved = [f"{q}/{k[0]:g}/{k[1]}" for q in seqs for k, v in payloads[q].items() if v is None]
    # --- A: position at feet -------------------------------------------------------------
    pooled = collections.defaultdict(lambda: collections.defaultdict(list))
    per_clip, fallback, chain_vs_feed, jac_ok = {}, {}, {}, True
    rel = lambda A, B: float(np.median(np.linalg.norm(A - B, axis=(1, 2)) / np.linalg.norm(B, axis=(1, 2))))  # noqa: E731
    for q in seqs:
        uv, G = _feet_pixels(q)
        far = G[:, 1] < 34.0
        Xr, placed = _refmap(q)(uv)
        chain_vs_feed[q] = round(float(np.median(np.linalg.norm(Xr[placed] - G[placed], axis=1))), 3)
        per_clip[q], fallback[q] = {}, {}
        for (sigma, seed), pay in payloads[q].items():
            if pay is None:
                continue
            key = f"{sigma:g}/{seed}"
            ship, tps = CAL.from_dict(pay), CAL.from_dict(_as_tps(pay))
            fallback[q][key] = not tps.active
            es = np.linalg.norm(np.asarray(ship.to_pitch(uv), float) - G, axis=1)
            et = np.linalg.norm(np.asarray(tps.to_pitch(uv), float) - G, axis=1)
            per_clip[q][key] = {"ship": float(np.median(es)), "tps": float(np.median(et))}
            for name, e in (("ship", es), ("tps", et)):
                pooled[key][name].append(e)
                pooled[key][f"{name}_far"].append(e[far])
            if sigma == 0 and tps.active:
                u, e = uv[::50], 0.05
                J = tps.jacobian(u)
                fd = np.stack([(tps.to_pitch(u + [e, 0]) - tps.to_pitch(u - [e, 0])) / (2 * e),
                               (tps.to_pitch(u + [0, e]) - tps.to_pitch(u - [0, e])) / (2 * e)], -1)
                JH = stats.homography_jacobian(np.asarray(tps.H, float), u)
                jac_ok = jac_ok and rel(J, fd) <= 0.01 and rel(JH, J) > 0.01
    position = {}
    for key, v in pooled.items():
        cat = {m: np.concatenate(x) for m, x in v.items()}
        position[key] = {"ship": float(np.median(cat["ship"])), "tps": float(np.median(cat["tps"])),
                         "ship_p90": float(np.percentile(cat["ship"], 90)),
                         "tps_p90": float(np.percentile(cat["tps"], 90)),
                         "ship_far": float(np.median(cat["ship_far"])), "tps_far": float(np.median(cat["tps_far"]))}
    # --- C: the product path, replayed from N21's detections ---------------------------------
    arms = {"ship0": ("p21id", None, (0.0, 0), False), "tps0": ("p27id", ".p27", (0.0, 0), True)}
    for seed in N27_SEEDS:
        arms[f"ship5_{seed}"] = (f"p27h5s{seed}id", f".p27h5s{seed}", (5.0, seed), False)
        arms[f"tps5_{seed}"] = (f"p27t5s{seed}id", f".p27t5s{seed}", (5.0, seed), True)
    key_rows = lambda path: sorted((int(r["f"]), round(float(r["x"]), 3), round(float(r["y"]), 3))  # noqa: E731
                                   for r in json.load(open(path))["detections"])
    cals, same_dets = {a: {} for a in arms}, True
    for q in seqs:
        base_rows = key_rows(ST._p(q, ".p21.tracks.json"))
        for arm, (variant, tag, key, corrected) in arms.items():
            if tag is None:
                cals[arm][q] = CAL.load(ST._p(q, ".p05.calibration.json"))
                continue
            pay = payloads[q][key]
            if pay is None:
                raise SystemExit(f"{q}: no calibration for sigma {key[0]:g} seed {key[1]} - N27 is VOID")
            cfile = ST._p(q, f"{tag}.calibration.json")
            with open(cfile, "w") as fh:
                json.dump(_as_tps(pay) if corrected else pay, fh)
            _n27_cache(q, tag, cfile)
            ST.track(q, cfile, tag=tag)
            ST.identify(q, cfile, src=tag, dst=f"{tag}id")
            same_dets = same_dets and key_rows(ST._p(q, f"{tag}.tracks.json")) == base_rows
            cals[arm][q] = CAL.load(cfile)
    # --- B and C: movement scale and speed ----------------------------------------------------
    pair = lambda r: (r["speed"]["our_hsr_m"], r["speed"]["gt_matched_hsr_m"])  # noqa: E731
    slope, speed, detail, gt_full, ruler = ({a: {} for a in arms}, {a: {} for a in arms},
                                            {a: {} for a in arms}, {}, True)
    for arm, (variant, _tag, _key, _c) in arms.items():
        for q in seqs:
            cal = cals[arm][q]
            sl, n, lr = _chord_slope(q, variant, cal)
            slope[arm][q] = sl
            own = evaluate_stc(q, variant, our_cal=cal)
            full = (own["squad"]["gt_m"], own["speed"]["gt_hsr_m"], own["speed"]["gt_sprints"])
            ruler = ruler and gt_full.setdefault(q, full) == full
            if np.isfinite(sl) and sl > 0:
                speed[arm][q] = pair(evaluate_stc(q, variant, our_cal=_ScaledCal(cal, 1.0 / sl)))
            detail[arm][q] = {"slope": sl, "chords": n, "length_ratio": lr, "hsr_own": pair(own),
                              "hsr_matched": speed[arm].get(q), "coverage": own["squad"]["coverage"],
                              "sprints_own": (own["speed"]["our_sprints"], own["speed"]["gt_matched_sprints"]),
                              "distance_own": (own["squad"]["ours_m"], own["squad"]["gt_matched_m"]),
                              "correction_active": getattr(cal, "active", None)}
    diagnostics = {}
    for name, variant, wrap in (("tps0_no_restitch", "p21id", lambda c: c), ("tps0_H_noise", "p27id", _HNoise)):
        num = den = 0.0
        for q in seqs:
            cal = wrap(cals["tps0"][q])
            sl = _chord_slope(q, variant, cal)[0]
            r = evaluate_stc(q, variant, our_cal=_ScaledCal(cal, 1.0 / sl) if np.isfinite(sl) and sl > 0 else cal)
            num, den = num + r["speed"]["our_hsr_m"], den + r["speed"]["gt_matched_hsr_m"]
        diagnostics[name] = round(num / den, 4)
    validity = {"solved_all": not unsolved,
                "slopes_finite": all(np.isfinite(v) and v > 0 for a in slope for v in slope[a].values()),
                "jacobian_is_its_own": jac_ok, "ruler": ruler, "replays_same_detections": same_dets,
                "no_stale_outputs": True}
    res = {"validity": validity, "position": {"pooled": position, "per_clip": per_clip, "fallback": fallback},
           "slope": {"ship0": slope["ship0"], "tps0": slope["tps0"]}, "speed": speed}
    verdict = n27_verdict(res)
    pooled_speed = {a: round(sum(v[0] for v in speed[a].values()) / sum(v[1] for v in speed[a].values()), 4)
                    for a in arms if speed[a]}
    with open(os.path.join(STC, "n27_results.json"), "w") as fh:
        json.dump({"position": position, "per_clip_position": per_clip, "fallback": fallback,
                   "chain_vs_feed_m": chain_vs_feed, "slope": slope, "pooled_hsr_matched": pooled_speed,
                   "detail": detail, "diagnostics": diagnostics, "validity": validity, "unsolved": unsolved,
                   "verdict": verdict, "protocol": "model/validation/dist_eval.py docstring N27"}, fh, indent=1)
    print(f"position at feet: {position}")
    print(f"feed against ST2's own chain at the same feet (median m): {chain_vs_feed}")
    print(f"movement scale (4 s chord slope): {slope}")
    print(f"pooled HSR ratio, movement scale matched: {pooled_speed}; diagnostics {diagnostics}")
    print(f"validity: {validity}")
    print(f"N27 {verdict}")
    return {"position": position, "slope": slope, "speed": pooled_speed, "verdict": verdict}


def _walk(n: int, fps: float, vx: float, noise: float = 0.0, seed: int = 0):
    """A player walking straight at vx m/s for n frames. True path = vx*(n-1)/fps."""
    t = np.arange(n) / fps
    xy = np.stack([vx * t, np.full(n, 34.0)], 1)
    if noise:
        xy = xy + np.random.default_rng(seed).normal(0, noise, xy.shape)
    return [{"f": i, "id": 1, "x": xy[i, 0], "y": xy[i, 1]} for i in range(n)]


def _selfcheck() -> None:
    """Synthetic trajectories whose answers are arithmetic, plus the trap.

    The trap is the reason this file exists, so it is asserted hardest: an
    unfiltered noisy trajectory reads HIGH, and comparing it against a filtered
    one manufactures a deficit that is a fact about the filter, not the product.
    """
    fps = 25.0

    # --- arithmetic cases -------------------------------------------------
    # 100 m walk: 5 m/s for 20.04 s. Clean input, so the chord is the only loss
    # and on a straight line the chord loses nothing.
    m, obs, peak, _ = track_metrics(_walk(502, fps, 5.0), _Flat, fps)
    assert abs(m - 100.2) < 1.0, f"straight 100 m walk read {m:.2f}"
    assert obs == 1.0 and abs(peak - 5.0) < 0.2, (obs, peak)

    # A stationary player runs nowhere. Clean input has no jitter to integrate.
    m, _o, peak, _ = track_metrics(_walk(502, fps, 0.0), _Flat, fps)
    assert m < 0.5, f"stationary player read {m:.2f} m"
    assert peak < 0.2, peak

    # A 40% observation gap: 500 frames, the middle 200 deleted. Distance falls
    # by ~40%, which is a RECALL fact and not an accuracy fact, and the whole
    # reason `coverage` is printed beside every ratio in this file.
    full = _walk(500, fps, 5.0)
    gappy = [p for p in full if not (150 <= p["f"] < 350)]
    m_full, _o, _p, _c = track_metrics(full, _Flat, fps)
    m_gap, obs, _p, _c = track_metrics(gappy, _Flat, fps)
    assert 0.55 < m_gap / m_full < 0.65, f"40% gap read {m_gap / m_full:.2f} of full"
    # AND observed_frac DOES NOT SEE IT. One long hole is one dropped window, so
    # observed_frac reads 0.97 while 40% of the running is gone. This is why
    # `stats.observed_frac` cannot be used as the coverage number here (its own
    # docstring calls it a floor) and why coverage is counted in FRAMES instead.
    assert obs > 0.9, obs
    assert 1 - obs < 0.2 * (1 - m_gap / m_full), (
        f"observed_frac {obs:.2f} must UNDER-state a {1 - m_gap / m_full:.0%} loss")

    # The fix for that, and the number this harness actually reads: re-score the
    # reference with our own coverage as its gap mask. Same windows dropped both
    # sides, so the ratio contains no recall at all.
    have = np.fromiter((not (150 <= i < 350) for i in range(500)), bool, 500)
    m_matched, _o, _p, _c = track_metrics(full, _Flat, fps,
                                          real_override=have[:-1] & have[1:])
    assert abs(m_gap / m_matched - 1.0) < 0.05, (
        f"coverage-matched reference must cancel the hole: {m_gap:.1f} vs {m_matched:.1f}")

    # THE TRAP, and the assertions this file exists for. Same trajectory plus
    # sigma 0.30 m of white noise - the size of SoccerTrack's own 1.05 m
    # x-quantisation, so this is not a hypothetical noise level, it is the
    # reference's.
    noisy = _walk(502, fps, 5.0, noise=0.30, seed=1)
    naive, _o, _p, _c = track_metrics(noisy, _Flat, fps, "raw", window_s=1 / fps)
    shipped, _o, _p, _c = track_metrics(noisy, _Flat, fps, "kalman")
    assert naive > 2.5 * shipped, f"per-frame sum-of-deltas must read high: {naive:.0f} vs {shipped:.0f}"
    assert abs(shipped - 100.2) / 100.2 < 0.15, f"shipped estimator read {shipped:.1f} vs 100.2"
    # 279 m against a 100 m truth. So integrating GT per-frame while windowing
    # ours would manufacture a ~64% deficit that is a fact about the bandwidth,
    # not about the product.
    assert shipped / naive < 0.40, "the deficit an unwindowed reference manufactures"

    # WHERE THE NOISE SUPPRESSION ACTUALLY COMES FROM, and it is not the Kalman.
    # At the shipped 0.4 s window the unfiltered trajectory reads only ~3% above
    # the filtered one, because the chord already averages the jitter away. Good
    # news for this harness - the answer barely depends on which filter the
    # reference gets - and the reason --gt-method is worth running both ways.
    win_only, _o, _p, _c = track_metrics(noisy, _Flat, fps, "raw")
    assert shipped < win_only < 1.15 * shipped, (
        f"at 0.4 s the window does the work: raw {win_only:.1f} kalman {shipped:.1f}")

    # THE CONTROL. If our distance were secretly garbage this harness has to
    # move, or it is measuring coverage and filtering rather than distance.
    # Same coverage, same fragments, positions scaled 1.5x: the ratio must go
    # to 1.5, not stay at 1.0.
    truth = _walk(502, fps, 5.0)
    fat = [dict(p, x=p["x"] * 1.5, y=p["y"] * 1.5) for p in truth]
    m_t, _o, _p, _c = track_metrics(truth, _Flat, fps)
    m_f, _o, _p, _c = track_metrics(fat, _Flat, fps)
    assert 1.45 < m_f / m_t < 1.55, f"a 1.5x scale error must read 1.5x, got {m_f / m_t:.2f}"

    # Coverage cannot rescue a wrong number and a wrong number cannot hide in
    # coverage: gt_matched must equal full GT when coverage is total.
    have = np.ones(len(truth) - 1, bool)
    m_m, _o, _p, _c = track_metrics(truth, _Flat, fps, real_override=have)
    assert abs(m_m - m_t) < 1e-9, (m_m, m_t)

    # Peak speed is read UNCLAMPED, or the answer to "is the clamp wrong" is
    # always 9.5. A 20 m/s teleport must be visible here.
    tele = _walk(100, fps, 0.0)
    for p in tele[50:]:
        p["x"] += 40.0
    _m, _o, peak, clamped = track_metrics(tele, _Flat, fps)
    assert peak > stats.MAX_SPEED, f"an unclamped teleport must exceed MAX_SPEED, got {peak:.1f}"
    assert clamped > 0.0, clamped

    # --- the reference's own defect, verified rather than quoted ------------
    if os.path.exists(GT):
        _fps, gt = load_gt(GT)
        xy = gt["1"]["xy"]
        gx = np.unique(np.abs(np.diff(np.unique(xy[:, 0]))).round(4))
        gy = np.unique(np.abs(np.diff(np.unique(xy[:, 1]))).round(4))
        assert gx.tolist() == [1.05] and gy.tolist() == [0.68], (gx, gy)
        held = float((np.linalg.norm(np.diff(xy, axis=0), axis=1) < 1e-9).mean())
        assert held > 0.9, held
        print(f"  GT grid verified: 1.05 m x 0.68 m, position held {held:.1%} of frames "
              "- this reference is quantised, not surveyed")

    _selfcheck_stc()
    # --- N22: the substitutions and the rule, on rows whose answer is known ---
    rows = [{"f": 0, "id": 7, "x": 1.0, "y": 1.0}, {"f": 2, "id": 7, "x": 2.0, "y": 1.0},
            {"f": 4, "id": 7, "x": 9.0, "y": 9.0}, {"f": 6, "id": 7, "x": 3.0, "y": 3.0}]
    owner = {0: "1", 1: "1", 2: "2"}                      # row 3 matched nobody
    gt = {"1": {"f": np.array([0, 2]), "xy": np.array([[1.5, 1.0], [2.5, 1.0]])},
          "2": {"f": np.array([4]), "xy": np.array([[8.0, 8.0]])}}
    d, _own = _oracle_rows(rows, owner, gt, _Flat, "drop")
    assert len(d) == 3 and {r["id"] for r in d} == {7}, "drop removes the unowned row and nothing else"
    ident, own = _oracle_rows(rows, owner, gt, _Flat, "identity")
    assert len({r["id"] for r in ident}) == 2, "a fragment that changes owner becomes two"
    assert [own[i] for i in range(3)] == ["1", "1", "2"] and ident[2]["x"] == 9.0, \
        "the identity arm keeps our positions"
    posn, _own = _oracle_rows(rows, owner, gt, _Flat, "position")
    assert [(r["x"], r["y"]) for r in posn] == [(1.5, 1.0), (2.5, 1.0), (8.0, 8.0)], \
        "the position arm moves each row to its owner's truth"
    assert {r["id"] for r in posn} == {7}, "and keeps our identities, switch included"
    both, _own = _oracle_rows(rows, owner, gt, _Flat, "both")
    assert len({r["id"] for r in both}) == 2 and (both[2]["x"], both[2]["y"]) == (8.0, 8.0)
    arm = lambda m: {"pooled_ratio_matched": m, "pooled_sprint_ratio": m, "pooled_hsr_ratio": m}  # noqa: E731
    rule = lambda i, p: n22_verdict({"drop": arm(1.4), "identity": arm(i), "position": arm(p)})["pooled_ratio_matched"]["driver"]  # noqa: E731
    assert rule(1.05, 1.35) == "IDENTITY" and rule(1.35, 1.05) == "POSITION", "the rule must fire both ways"
    assert rule(1.1, 1.1) == "BOTH" and rule(1.3, 1.3) == "NEITHER ALONE"
    assert n22_verdict({"drop": arm(0.9), "identity": arm(0.9), "position": arm(0.9)})[
        "pooled_ratio_matched"]["driver"] == "no over-read to apportion"
    # --- N23: the detector-free pixels, and the rule in all four directions ---
    feet = {0: {1: (1.2, 1.0)}, 2: {1: (2.2, 1.0)}, 4: {2: (8.2, 8.0)}}
    feedpx = {"1": {0: (1.4, 1.0), 2: (2.4, 1.0)}, "2": {4: (8.4, 8.0)}}
    g1, _own = _oracle_rows(rows, owner, gt, _Flat, "gtpix", feet=feet)
    assert [(r["x"], r["y"]) for r in g1] == [(1.2, 1.0), (2.2, 1.0), (8.2, 8.0)], \
        "gtpix takes each row's owner's MOT foot"
    assert {r["id"] for r in g1} == {7}, "and keeps our identities"
    f1, _own = _oracle_rows(rows, owner, gt, _Flat, "feedpix", feedpx=feedpx)
    assert [(r["x"], r["y"]) for r in f1] == [(1.4, 1.0), (2.4, 1.0), (8.4, 8.0)]
    four = lambda d, g, f, p: {"drop": arm(d), "gtpix": arm(g), "feedpix": arm(f), "position": arm(p)}  # noqa: E731
    rule23 = lambda *a: n23_verdict(four(*a))["pooled_sprint_ratio"]["driver"]  # noqa: E731
    assert rule23(3.0, 1.5, 1.45, 1.4) == "DETECTOR" and rule23(3.0, 2.9, 2.85, 1.4) == "MAP"
    assert rule23(3.0, 2.2, 2.15, 1.4) == "MIXED" and rule23(3.0, 2.6, 1.8, 1.4) == "INCONCLUSIVE"
    # The guard as COMMITTED reads the annotation share as a magnitude; the
    # protocol text says only "more than a third". Pinned here, and disclosed in
    # docs/experiments/n23-detector-or-map.md, because a negative share is where they differ.
    assert rule23(3.0, 2.6, 3.4, 1.4) == "INCONCLUSIVE", "abs(): -0.4 is past the guard"
    # N22's identity split, on the case the first version got wrong: A,A,B,A.
    aba = [{"f": f, "id": 3, "x": 0.0, "y": 0.0} for f in (0, 2, 4, 6)]
    split, _own = _oracle_rows(aba, {0: "1", 1: "1", 2: "2", 3: "1"}, gt, _Flat, "identity")
    assert len({r["id"] for r in split}) == 3, "a fragment that goes A,B,A is three pieces"
    split = n23_verdict(four(3.0, 2.2, 2.15, 1.4))["pooled_sprint_ratio"]
    assert abs(split["detector_share"] + split["annotation_share"] + split["map_share"]
               - (3.0 - 1.4) / 2.0) < 1e-9, "the three parts must sum to N22's position share"
    # --- N24: crediting per detection, on a walk whose answer is arithmetic ---
    walk = _walk(400, fps, 5.0)
    whole_m = frag_stats(walk, _Flat, fps)["m"]
    parts, whole = _credited(walk, ["1"] * 400, _Flat, fps)
    assert abs(whole - whole_m) < 1e-9 and abs(parts["1"]["m"] - whole_m) < 1e-9, \
        "a fragment owned throughout is credited exactly what frag_stats gives it"
    parts, whole = _credited(walk, ["1"] * 200 + ["2"] * 200, _Flat, fps)
    chord = stats.DISTANCE_WINDOW_S * 5.0
    assert abs(parts["1"]["m"] - 5.0 * 199 / fps) <= chord and abs(parts["2"]["m"] - 5.0 * 199 / fps) <= chord, parts
    assert 0 <= whole - parts["1"]["m"] - parts["2"]["m"] <= 2 * chord, \
        "the switch step, and the windows holding it, are credited to nobody - and only those"
    parts, _whole = _credited(walk, ["1"] * 150 + [None] * 100 + ["1"] * 150, _Flat, fps)
    assert 150 not in parts["1"]["frames"] and 149 in parts["1"]["frames"] and 250 in parts["1"]["frames"], \
        "an unowned stretch is covered for nobody, its edges for the owner"
    assert abs(parts["1"]["m"] - 5.0 * 298 / fps) <= 2 * chord, parts["1"]["m"]
    base = {"pooled_ratio_matched": 1.2}
    rule24 = lambda ident, ship, ctl=1.0: n24_verdict({"control": ctl, "control_cov": 1.0, "arms_running": {"drop": base, "identity": {"pooled_ratio_matched": ident}}, "ships_running": {"pooled_ratio_matched": ship}})["follows"]  # noqa: E731
    assert rule24(1.21, 1.03).startswith("IDENTITY") and rule24(1.21, 1.2).startswith("PHYSICAL")
    assert rule24(1.25, 1.03) == "VOID", "an identity oracle that still moves the ratio voids the fix"
    assert rule24(1.2, 1.03, ctl=1.01) == "VOID", "a control that does not read 1.000 voids the run"
    # --- N25: the adopted filter, on walks whose answers are arithmetic -------
    from scipy.signal import butter, filtfilt  # noqa: PLC0415
    walk = _walk(300, fps, 5.2)
    b = frag_stats(walk, _Flat, fps, "butter")
    assert abs(b["peak"] - 5.2) < 1e-3 and b["hsr_m"] == 0 and b["sprints"] == 0, \
        f"a clean 5.2 m/s run reads its own speed through the filter: {b}"
    num, den = butter(stats.BUTTER_ORDER, stats.BUTTER_HZ, fs=fps)
    ramp = 5.2 * np.arange(300) / fps
    assert np.diff(filtfilt(num, den, ramp)).max() * fps > stats.SPRINT_SPEED, \
        "scipy's default padding reads the same clean run past the sprint threshold - the disclosed deviation"
    # The wake: 1 m/s, a 2.44 s gap, re-acquired 20 m on. The shipped filter makes
    # a sprint of it (speed_series, KNOWN DEFECT); the adopted one ends the signal.
    gap = _walk(100, fps, 1.0) + [dict(r, f=r["f"] + 160, x=r["x"] + 160 / fps + 20.0)
                                  for r in _walk(100, fps, 1.0)]
    assert frag_stats(gap, _Flat, fps)["peak"] > stats.SPRINT_SPEED, "the Kalman's wake, seen"
    b = frag_stats(gap, _Flat, fps, "butter")
    assert b["peak"] < 1.01 and b["sprints"] == 0 and b["hsr_m"] == 0, b
    # A run too short to filter scores nothing, and common support holds the Kalman to it.
    short = _walk(12, fps, 5.2)
    assert frag_stats(short, _Flat, fps, "butter")["observed_s"] == 0.0
    assert frag_stats(short, _Flat, fps, common_support=True)["observed_s"] == 0.0, \
        "the Kalman is scored on the filter's steps, and no others"
    assert frag_stats(walk, _Flat, fps, common_support=True)["observed_s"] == frag_stats(walk, _Flat, fps)["observed_s"]

    def res25(dk, pk, db, pb, level, support, moved):
        def row(r, c):
            out = dict(dict.fromkeys(N25_RULER, 1.0), pooled_hsr_ratio=r, pooled_sprint_ratio=r,
                       pooled_our_observed_s=support if c == "B" else 1.0,
                       pooled_gt_alt_hsr_m=level if c == "B" else 1.0,
                       pooled_gt_alt_sprints=level if c == "B" else 1.0)
            if moved and c == "B":
                out["pooled_gt_matched_hsr_m"] = 2.0
            return out
        return {"drop": {"K": row(dk, "K"), "Kc": row(dk, "Kc"), "B": row(db, "B")},
                "position": {"K": row(pk, "K"), "Kc": row(pk, "Kc"), "B": row(pb, "B")}}
    rule25 = lambda *a: n25_verdict(res25(*a))["follows"]  # noqa: E731
    assert rule25(3.0, 1.5, 1.8, 1.5, 1.0, 1.0, False) == "ESTIMATOR", "excess 1.0 -> 0.2 is closure 0.8, both ways"
    assert rule25(3.0, 1.5, 2.8, 1.4, 1.0, 1.0, False) == "NOT THE ESTIMATOR", "closure 0 and 0.07"
    assert rule25(3.0, 1.5, 2.4, 1.5, 1.0, 1.0, False) == "MIXED", "closure 0.4, both ways"
    assert rule25(3.0, 1.5, 6.0, 4.5, 1.0, 1.0, False) == "MIXED", \
        "an artefact added to both arms closes the ratio (0.67) and not the difference (0) - no verdict on one"
    assert rule25(3.0, 1.5, 1.8, 1.5, 0.79, 1.0, False) == "VOID", "a filter that flattens clean running voids the run"
    assert rule25(3.0, 1.5, 1.8, 1.5, 1.0, 0.89, False) == "VOID", "so does one that scores too little of it"
    assert rule25(3.0, 1.5, 1.8, 1.5, 1.0, 1.0, True) == "VOID", "and a ruler that moved"
    # --- N26: the map arms, the correction's guards, the smooth inverse, the rule ---
    rows26 = [{"f": f, "id": 5, "x": x, "y": 1.0} for f, x in ((0, 1.0), (2, 2.0), (4, 3.0))]
    shift = lambda uv: (uv + [1.0, 0.0], np.array([True, False, True]))  # noqa: E731
    m26, _own = _oracle_rows(rows26, {0: "1", 1: "1", 2: "1"}, gt, _Flat, "refmap", mapper=shift)
    assert [(r["x"], r["y"]) for r in m26] == [(2.0, 1.0), (4.0, 1.0)], \
        "an arm's metres re-enter as the shipped map's pixels, and a row the map cannot place is dropped"
    same26 = lambda uv: (uv, np.ones(len(uv), bool))  # noqa: E731
    f26, _own = _oracle_rows(rows26, {0: "1", 1: "1", 2: "1"}, gt, _Flat, "feedref",
                             feedpx={"1": {0: (7.0, 7.0), 4: (9.0, 9.0)}}, mapper=same26)
    assert [(r["x"], r["y"]) for r in f26] == [(7.0, 7.0), (9.0, 9.0)], \
        "feedref reads the owner's feed pixels, and a frame without one is dropped"
    sq = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [5, 5]], float)
    correct, chk = _tps_correction(sq, sq + [[0, 0], [0, 0], [0, 0], [0, 0], [1.0, 0.5]])
    assert chk["exact_m"] < 1e-6 and not chk["folds"], chk
    got, inside = correct(np.array([[5.0, 5.0], [10.001, 5.0], [9.999, 5.0]]))
    assert np.allclose(got[0], [6.0, 5.5]) and inside.tolist() == [True, False, True], got
    assert np.linalg.norm(got[1] - got[2]) < 0.01, "a player leaving the clicks' hull does not jump"
    crossed = sq.copy()
    crossed[4] = [12.0, 12.0]
    assert _tps_correction(sq, crossed)[1]["folds"], "a correction that turns the map inside out is caught"
    # A forward map that rounds to whole pixels, read back two ways along a straight walk.
    g26 = np.stack(np.meshgrid(np.arange(0, 20, 0.1), np.arange(0, 10, 0.1), indexing="ij"), -1).reshape(-1, 2)
    fwd26 = np.round(g26 * [1.0, 2.0])
    walk26 = np.stack([np.linspace(2.0, 18.0, 400), np.full(400, 5.02)], 1)
    back, placed = _smooth_inverse(g26, fwd26, 8.0)(walk26 * [1.0, 2.0])
    from scipy.spatial import cKDTree  # noqa: PLC0415
    pick = g26[cKDTree(fwd26).query(walk26 * [1.0, 2.0])[1]]
    rough = lambda P: float(np.sqrt(np.mean(np.sum(np.diff(P, 2, axis=0) ** 2, 1))))  # noqa: E731
    assert placed.all() and np.abs(back - walk26).max() < 0.1, "the smooth inverse reads the map back"
    assert rough(back) < rough(pick) / 3, (rough(back), rough(pick))

    def res26(values, frac, folds, moved, matched_off):
        clips = ("a", "b", "c")
        out = {}
        for arm in N26_ARMS:
            pc = {c: {"speed": {"our_hsr_m": values[arm][j], "gt_matched_hsr_m": 1.0},
                      "squad": {"unattributed_frac": frac if arm == "refmap" else 0.0},
                      "map_checks": {"exact_m": 0.0, "folds": folds, "inside": 9, "rows": 10}}
                  for j, c in enumerate(clips)}
            out[arm] = dict(dict.fromkeys(N26_RULER + N26_MATCHED, 1.0), pooled_sprint_ratio=1.0,
                            pooled_ratio_matched=1.0, per_clip=pc)
        if moved:
            out["refmap"]["pooled_gt_hsr_m"] = 2.0
        if matched_off:
            out["feedref"]["pooled_gt_matched_hsr_m"] = 1.05
        return n26_verdict(out)["follows"]

    def arms26(tps, ref, feed):
        return {"drop": [3.0] * 3, "tps21": tps, "refmap": ref, "feedref": feed, "position": [1.5] * 3}
    base = arms26([2.2] * 3, [1.8] * 3, [1.55] * 3)
    assert res26(base, 0.0, False, False, False) == "MAP + CLICKS CARRY IT", "refmap 0.8, tps21 0.53"
    assert res26(arms26([2.8] * 3, [1.8] * 3, [1.55] * 3), 0.0, False, False, False) == "MAP + INTERIOR LANDMARKS NEEDED"
    assert res26(arms26([2.8] * 3, [2.7] * 3, [1.55] * 3), 0.0, False, False, False) == "NOT THE MAP ALONE"
    assert res26(arms26([2.8] * 3, [2.2] * 3, [1.55] * 3), 0.0, False, False, False) == "UNCLEAR"
    assert res26(base, 0.0, True, False, False) == "MAP; tps21 VOID", "a folded correction voids its own arm only"
    assert res26(arms26([2.2] * 3, [1.8] * 3, [1.8] * 3), 0.0, False, False, False) == "VOID", \
        "a refmap that adds 0.2 of the excess to the truth voids the run"
    assert res26(base, 0.03, False, False, False) == "VOID", "so does one that drops too many rows"
    assert res26(base, 0.0, False, True, False) == "VOID", "and a ruler that moved"
    assert res26(base, 0.0, False, False, True) == "VOID", "and matched ground truth that moved past 0.02"
    assert res26(arms26([2.2] * 3, [1.2, 1.2, 3.0], [1.55] * 3), 0.0, False, False, False).startswith(
        "MAP + CLICKS CARRY IT (FRAGILE: 3 of 3"), "a verdict one clip can overturn says so"
    # --- N27: movement scale, the scaled map, the noise models, and the rule -----------
    fr27 = np.arange(0, 402, 2)
    G27 = np.stack([5.0 * fr27 / fps, np.full(len(fr27), 30.0)], 1)
    slope27, n27, lr27 = _chord_slope_arrays(np.ones(len(fr27), int), fr27, np.array(["a"] * len(fr27)),
                                             1.2 * G27, G27, fps)
    assert abs(slope27 - 1.2) < 1e-9 and n27 > 0 and abs(lr27 - 1.2) < 1e-9, (slope27, n27, lr27)
    owners27 = np.array(["a"] * 100 + ["b"] * 101)
    X27 = np.where((owners27 == "b")[:, None], 1.2 * G27 + [20.0, 0.0], 1.2 * G27)  # a 20 m jump at the change
    assert abs(_chord_slope_arrays(np.ones(len(fr27), int), fr27, owners27, X27, G27, fps)[0] - 1.2) < 1e-9, \
        "a chord never spans a change of owner"
    sc = _ScaledCal(_Flat, 2.0)
    assert np.allclose(sc.to_pitch(np.array([[62.5, 34.0]])), [[72.5, 34.0]])
    assert np.allclose(sc.jacobian(np.array([[1.0, 1.0]])), 2.0 * np.eye(2)), "the noise model scales with the map"

    class _Heavy(_Flat):
        """_Flat whose calibration says each pixel is 50 m: the filter must trust it less."""
        @staticmethod
        def jacobian(px):
            return np.tile(50.0 * np.eye(2), (len(np.asarray(px).reshape(-1, 2)), 1, 1))
    noisy27 = _walk(500, fps, 3.0, noise=0.3, seed=3)
    assert frag_stats(noisy27, _Heavy, fps)["peak"] < frag_stats(noisy27, _Flat, fps)["peak"], \
        "speed reads the calibration's own Jacobian for measurement noise"
    assert frag_stats(noisy27, _HNoise(_Heavy), fps)["peak"] == frag_stats(noisy27, _Flat, fps)["peak"], \
        "the diagnostic reads H's noise model instead"

    def rule27(tps_med, far_t, fb, t_slope, h_ship, h_tps, h_tps5, per_clip_h, valid):
        clips, keys = ("a", "b"), ("0/0", "5/0", "5/1", "20/0")
        pooled = {k: {"ship": 8.0, "tps": tps_med, "ship_p90": 9.0, "tps_p90": 3.0, "ship_far": 8.0,
                      "tps_far": far_t} for k in keys}
        sp = {"ship0": {q: (h_ship, 1.0) for q in clips},
              "tps0": {q: (per_clip_h[j], 1.0) for j, q in enumerate(clips)}}
        for sd in ("0", "1"):
            sp[f"ship5_{sd}"], sp[f"tps5_{sd}"] = {q: (h_ship, 1.0) for q in clips}, {q: (h_tps5, 1.0) for q in clips}
        return n27_verdict({"validity": {"all": valid},
                            "position": {"pooled": pooled,
                                         "per_clip": {q: {k: {"ship": 8.0, "tps": tps_med} for k in keys} for q in clips},
                                         "fallback": {q: {k: fb and k == "5/1" for k in keys} for q in clips}},
                            "slope": {"ship0": {q: 1.2 for q in clips}, "tps0": {q: t_slope for q in clips}},
                            "speed": sp})["follows"]
    assert rule27(2.0, 2.0, False, 0.97, 3.6, 1.8, 1.9, (1.8, 1.8), True) == "SHIP FOR STITCHED PANORAMAS"
    assert rule27(9.0, 2.0, False, 0.97, 3.6, 1.8, 1.9, (1.8, 1.8), True) == "DO NOT SHIP (failed: A position)"
    assert rule27(2.0, 9.0, False, 0.97, 3.6, 1.8, 1.9, (1.8, 1.8), True) == "DO NOT SHIP (failed: A position)", "far half"
    assert rule27(2.0, 2.0, True, 0.97, 3.6, 1.8, 1.9, (1.8, 1.8), True) == "DO NOT SHIP (failed: A position)", \
        "a correction that switches off under 5 px of click noise fails A"
    assert rule27(2.0, 2.0, False, 0.85, 3.6, 1.8, 1.9, (1.8, 1.8), True) == "DO NOT SHIP (failed: B scale)", \
        "a map that shrinks movement fails B even against one that stretches it"
    assert rule27(2.0, 2.0, False, 0.97, 3.6, 3.0, 1.9, (3.0, 3.0), True) == "DO NOT SHIP (failed: C speed)"
    assert rule27(2.0, 2.0, False, 0.97, 3.6, 1.8, 3.0, (1.8, 1.8), True) == "DO NOT SHIP (failed: C speed)", "sigma 5"
    assert rule27(2.0, 2.0, False, 0.97, 3.6, 1.8, 1.9, (1.0, 3.5), True) == "DO NOT SHIP (failed: C speed)", \
        "a pooled pass one clip carries fails leave-one-out"
    assert rule27(2.0, 2.0, False, 0.97, 0.9, 0.8, 0.8, (0.8, 0.8), True) == "NOTHING TO REMOVE"
    assert rule27(2.0, 2.0, False, 0.97, 3.6, 1.8, 1.9, (1.8, 1.8), False) == "VOID"

    # stc_gt's cache key must move when the join's identities move, not only its
    # clock: two players swapped, or a team relabelled, served the old metres.
    j = {"period": "1", "offset": 5, "mot_to_player": {"0": "10", "1": "11"},
         "team": {"0": "home", "1": "away"}}
    k0 = _stc_gt_key(j, 2)
    assert k0 != _stc_gt_key(dict(j, mot_to_player={"0": "11", "1": "10"}), 2), \
        "a swapped identity permutation kept the stc_gt cache key"
    assert k0 != _stc_gt_key(dict(j, team={"0": "away", "1": "home"}), 2), \
        "a relabelled team kept the stc_gt cache key"
    # No matched GT sprint is no ratio: NaN (which api.py FAILS), never our count / 1.
    assert np.isnan(_ratio_or_nan(7, 0)), "a zero sprint denominator returned a number"
    assert _ratio_or_nan(3, 2) == 1.5
    print("dist_eval.py ok  (100 m walk 100.2, stationary 0, 40% gap -40% with "
          "observed_frac blind to it, per-frame integration reads 2.8x, 1.5x scale "
          "reads 1.5x, teleport visible unclamped)")


def _selfcheck_stc() -> None:
    """The N10/N11 reference, checked against a SECOND source, and the settle rule.

    A wrong frame offset or a wrong pitch convention does not crash: it produces
    a plausible distance for the wrong player at the wrong time, and every ratio
    below it is then a confident lie. The check is independent of how the join
    was solved - take the GT metres, push them through ST2's OWN vendor chain
    (`st2_overlay.projector`, median 7.8 px against 65 hand-annotated keypoints)
    and require them to land on the MOT ground-truth feet. That single assertion
    pins the offset, the permutation AND the corner-origin convention.
    """
    if not os.path.exists(f"{STC}/118575.p05.tracks.json"):
        return
    import st2_mot as T  # noqa: PLC0415
    from st2_overlay import projector  # noqa: PLC0415
    import st2_team_gt as G  # noqa: PLC0415

    seq = "118575"
    gt = stc_gt(seq)
    assert len(gt) == 22, f"22 joined players expected, got {len(gt)}"
    # A mirrored feed joins every id as well as a true one; only its handedness says
    # its positions are wrong, and the loader must refuse it (st2_team_gt, HANDEDNESS).
    try:
        stc_gt("132831", cache=os.path.join(os.environ.get("TMPDIR", "/tmp"), "mirrored_feed_check.json"))
    except SystemExit:
        pass
    else:
        raise AssertionError("132831's mirrored ground-truth feed was served as positions")
    feet = G.mot_feet(seq)
    proj = projector(seq, T._wh(seq))
    err = []
    for cf in (0, 1000, 3000):
        pts = [tuple(gt[m]["xy"][list(gt[m]["f"]).index(cf)]) for m in gt
               if cf in gt[m]["f"] and int(m) in feet[cf]]
        got = [feet[cf][int(m)] for m in gt if cf in gt[m]["f"] and int(m) in feet[cf]]
        px, ok = proj(pts)
        err.append(float(np.median(np.linalg.norm(
            np.array(px)[ok] - np.array(got)[ok], axis=1))))
    assert max(err) < 25.0, (
        f"GT metres reproject {err} px from the MOT feet - the join, the offset or "
        "the corner-origin convention is wrong and every N10 number is void")

    # The settle mask, on arithmetic. 25 fps, a gap at step 10, 1 s of blanking.
    t = np.arange(0, 2.0, 0.04)
    real = np.ones(len(t) - 1, bool)
    real[10] = False
    m = _settle(t, real, 1.0)
    assert m[0] and m[9] and m[10], "start and gap must open a settling window"
    assert m[34] and not m[35], f"1 s after the gap at t={t[11]:.2f}: {np.where(m)[0]}"
    assert not _settle(t, real, 0.0).any(), "settle 0 is the shipped path: blank nothing"

    # And the operation must cost DISTANCE nothing - `real` is untouched. This is
    # the claim speed-clamp (not published) §5 rests on, asserted rather than quoted.
    cal = calibrators.load(f"{STC}/{seq}.p05.calibration.json")
    blob = json.load(open(f"{STC}/{seq}.p05.tracks.json"))
    trk = [r for r in blob["detections"] if int(r["id"]) == 0][:400]
    a = frag_stats(trk, cal, float(blob["fps"]), settle_s=0.0)
    b = frag_stats(trk, cal, float(blob["fps"]), settle_s=1.0)
    assert a["m"] == b["m"], f"blanking moved distance {a['m']} -> {b['m']}"
    assert b["hsr_m"] <= a["hsr_m"] and b["sprints"] <= a["sprints"], (a, b)

    # THE LIVE CONTROL, and N10's pass rule will not be read without it: GT as
    # its own prediction must come out at 1.000 with coverage 1.000 through the
    # whole STC path.
    c = evaluate_stc(seq, control=True)["squad"]
    assert abs(c["ratio_raw"] - 1.0) < 1e-9 and abs(c["ratio_matched"] - 1.0) < 1e-9, c
    assert abs(c["coverage"] - 1.0) < 1e-9 and c["orphan_ids"] == 0, c


GRID = ("tracks.cbiou.json", "tracks.final.json", "tracks.split.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="tracks.split.json")
    ap.add_argument("--job", default=JOB)
    ap.add_argument("--gt", default=GT)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--gt-method", default="kalman", choices=("kalman", "raw", "boxcar"))
    ap.add_argument("--window", type=float, default=None,
                    help="distance window in seconds; default stats.DISTANCE_WINDOW_S")
    ap.add_argument("--control", type=float, nargs="?", const=0.0, default=None,
                    metavar="NOISE_PX",
                    help="ignore --tracks; score GT-as-prediction with this much "
                         "pixel noise. 0 must read ratio 1.00.")
    ap.add_argument("--max-frag-peak", type=float, default=None, metavar="MS",
                    help="drop fragments whose unclamped peak exceeds this; the "
                         "reference re-matches to what is left, so the ratio stays honest")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--stc", action="store_true",
                    help="N10/N11: the nine STC2025 clips at the product config")
    ap.add_argument("--stc-variant", default="p05")
    ap.add_argument("--n22", action="store_true", help="N22: five arms on .p21, see the header")
    ap.add_argument("--n23", action="store_true", help="N23: four arms on .p21, see the header")
    ap.add_argument("--n24", action="store_true", help="N24: both creditings, see the header")
    ap.add_argument("--n25", action="store_true", help="N25: the speed estimator, see the header")
    ap.add_argument("--n26", action="store_true", help="N26: the map's own oracle, see the header")
    ap.add_argument("--n27", action="store_true", help="N27: the 21-click correction in the product")
    ap.add_argument("--seqs", help="comma-separated clip list for --stc")
    ap.add_argument("--settle", type=float, default=0.0, metavar="S",
                    help="N11: blank the velocity for S seconds after every gap and "
                         "at every fragment start, on BOTH sides. 0.0 is what ships.")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
    elif a.n22:
        run_n22()
    elif a.n23:
        run_n23()
    elif a.n24:
        run_n24()
    elif a.n25:
        run_n25()
    elif a.n26:
        run_n26()
    elif a.n27:
        run_n27()
    elif a.stc:
        print(json.dumps(evaluate_stc_all(
            tuple(a.seqs.split(",")) if a.seqs else None, a.stc_variant,
            gt_method=a.gt_method, window_s=a.window, settle_s=a.settle), indent=2))
    elif a.control is not None:
        r = evaluate("", a.job, a.gt, a.stride, a.gt_method, a.window, a.control,
                     a.max_frag_peak)
        print(json.dumps(r, indent=2)) if a.json else report(r)
    else:
        for f in (GRID if a.grid else (a.tracks,)):
            if not os.path.exists(os.path.join(a.job, f)):
                print(f"(missing {f})")
                continue
            r = evaluate(f, a.job, a.gt, a.stride, a.gt_method, a.window, None,
                         a.max_frag_peak)
            print(json.dumps(r, indent=2)) if a.json else report(r)
