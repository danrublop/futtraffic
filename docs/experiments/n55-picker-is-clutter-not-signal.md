# N55: the picker problem is dense look-alike clutter, not a weak signal

2026-09-23. A literature review outside sports CV, after the fourth picker change
lost (`n49-ball-path.md` §5b). Nothing here is measured; this
page is a reading of other people's results against ours, and every number it
quotes carries the sigil it already has elsewhere in this repo.

## The reframe, which is the whole finding

Track-before-detect — radar, infrared search-and-track, astronomy — exists for a
target that is **below the per-frame detection threshold** and only appears when
energy is integrated along a hypothesised path. It needs three things: a
sub-threshold target, clutter that does not look like the target, and a motion
model the target obeys.

Our population violates all three, in the direction that matters:

- the ball is already in the candidate list on **0.8822** of frames
  ([`n52-clean-ball-data.md`](n52-clean-ball-data.md) §7b, arm D `oracle`). There is
  no sub-threshold energy left to integrate.
- the clutter is target-like in appearance *and* in motion: N49 §5 measured the
  path's own movement at p50 1.85-2.49 px against the ball's 3.0-4.1, i.e. it
  locked onto things that move **less** than a ball.
- a third of the test population is a ball that does not move at all (`travel_px`
  0.9 over 300 frames, N49 §2).

So the analogue is **data association in dense, heterogeneous, target-like
clutter**, where the published remedy is a better per-candidate likelihood, not
longer temporal integration. That retires, before they are run: DP-TBD/Viterbi
variants (already lost twice here), IMM "move-stop-move" models (a zero-velocity
mode makes the motion prior MORE hospitable to static decoys, which is the losing
direction N49 §5 already measured), shift-and-stack, and event cameras (which emit
nothing at all for a ball at rest, and need hardware we do not have:
the target setup has one panoramic camera).

## What survives, ranked by gain over cost to test

**R1 — a two-horizon background, replacing the 60-frame scenery deque.** The
abandoned-object literature runs two background models at different learning
rates: a stationary object stays foreground in the slow one and is absorbed by the
fast one, so "arrived and stopped" is separable from "permanent structure". Our
gate asks whether a neighbour was within 8 px on >=25 of the last 60 frames (N49
§1) - a single rate, one horizon - and paint and a dead ball both pass it. This is
the 295-frame population, and the repo already has a number pointing the same way:
`path_nosc` scores 0.4856 against shipped's 0.4706 with the gate off (N49 §5b,
EXPLORATORY, no bar). **It is a gate change**, so CLAUDE.md's filter rule applies:
the drop share and the non-GT box rate travel with any result, as they did in N43.

**R2a — is confidence evidence at all?** The Viterbi's emission term was
`log(conf)`, which assumes confidence argues FOR a candidate. On the losing frames
N49 §2 records the true candidate at conf p50 0.078-0.162 against winners at
0.272-0.494. The radar answer to "amplitude is the wrong ranking signal" is not to
drop it but to use the likelihood ratio `p(a | target) / p(a | clutter)`. If that
ratio is flat or inverted over our range, the path picker was scored with an
emission term of the wrong sign - which would reframe the four losses rather than
add a fifth. Pure arithmetic over committed caches.

**R3 — the player-conditioned prior.** Two papers are this exact problem: Wei et
al. (CVIU 2013) track players first and infer which player is in possession; Maksai
et al. (CVPR 2016) model the ball's STATE - in-possession or free flight - so a
disappearance is explained by the state rather than charged as a miss. We hold
reliable player tracks, a calibration, and PathCRF already scores possession
ball-free (`pathcrf-works-ball-free-passing` (internal note)).
N49 §6 records that a non-kinematic emission term inside the same Viterbi is
untested; this is one. It cannot help the dead-ball frames - a resting ball is near
nobody - so the two populations must be pre-registered and scored apart.

**R4a — frames-in-channels for the detector.** Corsel et al. (WACVW 2023) feed
three timesteps into the three colour channels of an unmodified detector on a
static camera, and report tiny moving objects improving without degrading
stationary ones. Data-loader change only, same architecture, same D1 tiles. It is
the only item here that attacks the 11.8% the detector never proposes - but 72% of
those are balls AT REST, where a temporal-context detector's gain is concentrated
in the wrong place, so the at-rest and in-play subsets are separate populations
before the run, not after.

## Where the literature has nothing

A ball **at rest for tens of seconds, fused with a painted line, in a fixed wide
shot**. Sports ball tracking is about fast balls: TrackNet, WASB, TOTNet, DeepBall
and FootAndBall all lean on motion or temporal continuity, and none of their
benchmarks contains a long dead-ball stretch. The closest transferable work is not
sports at all - it is abandoned-object surveillance, which is R1.

Also absent: any benchmark for "which of N detector candidates is the ball". The
detection-calibration line (VarifocalNet, GFLv2) fixes class-versus-quality
misalignment, not target-versus-distractor discrimination.

## What this page does NOT say

- Nothing was run. No number here is a measurement of this session's.
- No method here is supported. Each needs its own pre-registered bar, and R1 and
  R3 each need their dead-ball and in-play populations split BEFORE the run - N49
  §7 asked for that split and it is still undone.
