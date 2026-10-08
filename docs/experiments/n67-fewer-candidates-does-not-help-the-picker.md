# N64 E2: the true negatives cut the decoys 20% and the picker gained nothing

> **Pre-registration not provable (2026-09-29):** an E2 VOID list was added to the rule paragraph of `model/validation/n64_ball_negatives.py` between the rule commit e9eb4e3 and the result commit 5eefa62; the verdict stands as weaker evidence. Recorded in `model/validation/protocol.py` NOT_PROVABLE.

2026-09-27. Rule V2 committed in 121c512 and its VOID list restored in the commit that added
VOID (f), both before either arm trained. **The first arm in this programme whose mechanism
worked**, and it did not help.

## The result

    @M 0.9811,54.2,43.4,0.5872,0.5811,-0.0061 :: .venv/bin/python model/validation/n64_ball_negatives.py --pair
       ball_cmp.CLIPS, 1800 gt-present frames, gate 20 px, both arms' caches built at the
       640 they were TRAINED at (VOID (f) refuses 384). In order: the matched oracle recall
       both arms were solved to, D1's false candidates per frame there, E2's, then pooled
       shipped for each and the delta. n=1 per arm, seed 0 pinned.

| | D1′ | E2 | verdict |
|---|---|---|---|
| false candidates/frame at matched recall 0.9811 | 54.2 | **43.4** | **−20.0%, MECHANISM PASS** (gate ≤ 48.8) |
| pooled `shipped` | 0.5872 | 0.5811 | **−0.0061, OUTCOME FAIL** (bar +0.05) |
| paired cluster bootstrap 95% CI over 6 clips | | | **[−0.0311, +0.0222]**, spans zero |
| `oracle` | 0.9817 | 0.9811 | floor 0.9717, GUARD PASS |
| raw candidates/frame | 55.7 | 44.8 | the number V1 would have gated on |

Leave-one-clip-out on the paired delta: −0.0087, −0.0173, −0.0020, +0.0020, −0.0013, −0.0093.
No clip is carrying it; the null is the whole corpus.

**VOID (c) passed outright** — both arms recorded `device cuda, amp_dtype fp16, seed 0`, and
both checkpoints' sha256 are printed beside the number. Unlike E1, **this pair is not
confounded**: same card, same precision, same seed, same epochs, same base weights, and the
arms differ by 53 tiles and nothing else.

## What it closes, which is larger than E2

[`n55-picker-is-clutter-not-signal.md`](n55-picker-is-clutter-not-signal.md) named exactly
two exits from data association in target-like clutter:

1. **A better per-candidate likelihood.** Closed by N58: the best unused signal separates
   ball from decoy at AUC **0.5466** against confidence's 0.8893, and it failed even with
   ground-truth players (`n58-player-prior.md`).
2. **Fewer candidates.** Closed here: **20% fewer false candidates at matched recall moves
   the picker −0.0061, with an interval spanning zero.**

Both exits are now measured shut. The picker's headroom over this detector — 0.5811 against
`oracle` 0.9811, about **0.40** — is not reachable by having fewer things to choose between,
and not by a better per-candidate score. Seven pre-registered picker changes have lost, and
now the two remedies the framing itself proposed have lost too.

What that leaves, from the literature rather than from us: **temporal input to the network**
(TrackNet reports precision 95.7% → 99.8% purely from 3-frame input) and **multi-crop
ensembling** (BallSeg, ~+10% at low FPR, because false positives are not spatially consistent
and the ball is). Both act before the picker exists, which is the one place neither closed
exit applies. Neither is a picker change, and that is the point.

## The backend measurement, which is why D1′ was run at all

D1 and D1′ are the same 916 tiles, same hyperparameters, same epoch count. D1 trained on
Apple MPS; D1′ on CUDA with fp16 and seed 0 pinned.

| | D1 (MPS) | D1′ (CUDA/fp16) | delta |
|---|---|---|---|
| pooled `shipped` | 0.5850 | 0.5872 | **+0.0022** |
| raw candidates/frame | 71.7 | 55.7 | **−16.0 (−22%)** |

**The backend barely moves accuracy and moves the candidate count a great deal.** That is a
correction this repo needs on the record: a candidate count is not comparable across
backends, so a decoy-count comparison between arms trained on different machines measures
the machine as much as the arm.

It applies to this repo's own writing.
`n64-e1-passed-the-bar-on-a-different-experiment.md`
compared E1 (CUDA/bf16) at 106.5 candidates against D1 (MPS) at 71.7 and called the mechanism
"refuted harder". The same-backend comparison is E1 106.5 against **D1′ 55.7**, still very
nearly 2x, so the conclusion stands — but it stood on a comparison that was partly measuring
the card, and the honest baseline for it is D1′.

It also puts a size on the confound E1 was voided for: **+0.0022 of accuracy**. So E1's
device-and-precision confound was small, and its COMPOSITION confound — 865 of its 916 added
tiles held a ball — was the one that mattered.

## Limits

- **n=1 per arm.** Seed 0 is pinned so a re-run lands in the same place, but one draw is not
  a noise floor. The cluster bootstrap bounds the CORPUS variance, not the seed's.
- Six clips from **two matches**; `ball_clean.py` states that every number this data can
  produce is same-match.
- The mechanism gate is at **matched oracle recall 0.9800**, which is near this corpus's
  ceiling (pooled oracle 0.9811-0.9817, and exactly 1.0000 on the dead clips). Matching two
  arms at a recall that high forces both thresholds very low — 0.050 and 0.055 — so the
  false-candidate counts are near-worst-case for both. A lower matched recall would give
  smaller counts and possibly a different ratio; that is one sweep and it has no held-out
  story, so it is not run.
- A null from six clips reads "no evidence", never "evidence of no effect". The pre-registered
  SESOI was +0.05 and the interval's upper bound is +0.0222, so an effect that large is
  excluded; an effect of +0.01 is not.

## Corrected 2026-09-28: the mechanism number was measured on one clip of six

A review found that `pair()` pooled its per-frame data with `{frame: value}` keys. But
`ball_eval.build_cache` numbers every clip's frames `for f in range(want)`, so all six clips
share the keys 0–299 and the flat dict kept only **the last clip written**. `matched()` —
the whole MECHANISM gate — therefore solved its thresholds and counted false candidates on
**one clip of six**, while every label on it said "pooled 1800".

| | target recall | D1′ fc/frame | E2 fc/frame | cut |
|---|---|---|---|---|
| as first published (keys collided, n=300) | 0.9800 | 48.8 | 28.4 | −41.8% |
| **corrected, keyed by (clip, frame), n=1800** | **0.9811** | **54.2** | **43.4** | **−20.0%** |

**The verdict does not move** — 20% still clears the ≤ 0.90× gate, and `shipped`, `oracle`
and the bootstrap were all computed per clip and were never affected, so "the picker gains
nothing" stands on every number that mattered to it. What was wrong is the *magnitude* of the
mechanism and the *population* it was measured over.

Two things worth keeping from how this was found. The bug was **deterministic**, so
`verify_claims` re-confirmed 28.4 on every run — a green measured tag proves a command
reproduces its number, not that the number covers the population its label claims. And the fix is now
guarded: `pair()` asserts the pooled key count equals the sum of per-clip frame counts, so a
collision fails loudly instead of quietly shrinking the population by six.
