# N65: every ball number in this repo was measured at 384 on a detector trained at 640

2026-09-27. Rule v1 in `model/validation/n64_ball_negatives.py` at 2cbb5b1, **withdrawn as
vacuous**; rule v2 at 59249a2, both committed before the caches existed. Found by an
adversarial code review of N64, not by any gate.

## The result

    @M 0.4706,0.5167,0.8822,86.2,0.5850,0.5678,0.9783,71.7 :: .venv/bin/python model/validation/n64_ball_negatives.py --res
       ball_cmp.CLIPS, all six, 1800 gt-present frames, gate 20 px, D1's checkpoint
       (run_d1/checkpoint_best_ema.pth), tile 640 on both sides. The ONLY thing that
       differs between the rows is the model's input square. n=1 configuration.

| input square | shipped | conf | oracle | candidates/frame |
|---|---|---|---|---|
| **384** — as every ball number was measured | 0.4706 | 0.5167 | 0.8822 | 86.2 |
| **640** — as the checkpoint was trained | **0.5850** | **0.5678** | **0.9783** | **71.7** |
| delta | **+0.1144** | **+0.0511** | **+0.0961** | **−14.5 (−16.8%)** |

The 384 row reproduces D1's published figures exactly — 0.4706, 0.5167, 0.8822, 86.2 —
so this is a clean A/B on resolution and nothing else.

## Why it happened

`RFDETRNanoConfig().resolution` is **384**. `run_d1/training_config.json` records
`model_config.resolution: 640`. **No ball call site has ever passed `resolution`:**
`ball_eval.build_cache` built `kw = {conf, ball, weights, variant}` and gave `tile` to the
tiler, and `n64.mine()` did the same. `pretrain_weights` does not restore it — resolution
is an explicit ModelConfig override only.

So the pipeline cut a 640 px tile and then resized it to 384, shrinking a ball already
only a few pixels across by a further 1.67x. Two comments in this repo describe the
opposite: `ball_cmp.py`'s "tile 640 because that is what it was trained at… behind the
default 1024 the ball shrinks by a third", and `build_cache`'s "saw the ball at NATIVE
scale inside 640 px tiles". Both were reasoning about the tile and neither noticed the
model's own input.

Training was always correct — `ball_train.train` passes `RFDETRNano(resolution=T)` with
T = 640. **The checkpoints are fine. Only the measuring was wrong.**

## The part that stings

**Candidates per frame fall 86.2 → 71.7 at the correct resolution.** Cutting the decoy
count is precisely what N64's entire programme existed to do: 12,001 tiles mined, a 606 MB
kit shipped to a training PC, 168 minutes of GPU, an arm that cleared its bar by 0.0016,
and two withdrawn rule versions — all chasing a 17% candidate reduction that a missing
keyword argument was causing.

And `shipped` moves **+0.1144**, more than twice the +0.05 bar every ball arm has been
measured against, without training anything.

## What this invalidates, and what it does not

**Every ball number in this repo is at the wrong input resolution.** That includes D1's
0.4706 / 0.8822 / 86.2, E1's 0.5222 / 0.9456 / 105.8, N50's 0.3772, N51's 0.3344 and
0.3856, N57's AUCs, N58's whole table and its 0.5466-vs-0.8893 comparison, N59's eight
subsets, and the memory notes quoting any of them. None may be cited as a property of the
detector until re-measured at 640.

**What survives — and this paragraph over-claimed when first written.** It said "every arm
within an experiment was measured at the same (wrong) resolution", which holds only for
comparisons *between nano arms*. `ball_cmp.ARMS` makes arms **A and C `coreml`
rfdetr-medium-576** — a CoreML export at the input it was exported for, correctly configured —
while **B, D and E1 are `rfdetr` nano at the 384 default against a 640 checkpoint**. So:

- **Same-arm comparisons are safe.** N49, N50, N51, N58 and N59 each vary a picker over one
  arm's candidates, and their relative verdicts stand — N58 and N59 were re-measured at 640 and
  both held.
- **Cross-arm deltas subtract a correctly-configured arm from a misconfigured one**, and are
  NOT internally consistent: N48's **+0.1972** (B−C, "the fine-tune at a fixed tile size") and
  **+0.1339** (C−A, "tiling alone"), `benchmark-status.md`'s `@D 0.4044 :: 0.1339 / 0.3311`,
  N49's B−C **+0.1300**, and `clean-ball-detector-exists` (internal note)'s D−C **+0.3644** — which is the basis
  of the "licence question is closed" verdict. **Corrected 2026-09-28.** None of those should be
  quoted until both sides are measured at the input their own checkpoint was trained for.

What also changes is any conclusion resting on a level rather than a difference: the gates were
judged against a picker at 0.4706 with 86 candidates, and at 640 it starts from 0.5850 with 72.

**The product is NOT affected.** `app/server/jobs.py` runs CoreML when bundled and
`rfdetr` medium otherwise, and takes the ball from that same detector's `sports ball`
class. Nothing under `app/` or `model/engine/` loads a nano checkpoint — `ball_train` is
wired into nothing, and `shipped_config.py` reports 0 disagreements. This is a research
measurement defect, not a shipped one.

## v1 of this rule was vacuous, and that is recorded rather than quietly replaced

V1 tested `oracle` on **one** clip, `ball_cmp.CLIPS[0]`, which is the dead-ball clip: the
ball is stationary and visible, some candidate is within 20 px on every frame, and
`oracle` read **1.0000 at both resolutions**. A statistic at its ceiling cannot move, so
"EXISTING NUMBERS STAND" measured nothing. The same failure as N61's first bucket, which
used a clip-wide min/max band and so could not put anything outside it.

The numbers that could move on that clip did — `conf` 0.4467 → 0.6767 — which is what
prompted v2. V2 pools all six clips, puts `conf` in the bar beside `oracle` because `conf`
is what the picker consumes, and adds a VOID that refuses the design outright if pooled
`oracle` at 384 is above 0.99.

## What has to happen before any ball number is quoted again

1. **Re-cache all six clips at 640 for every arm that is still live** — D1 is done
   (`.cmpDr`), E1 is not, and D1′/E2 must be built at 640 from the start.
2. **Re-run N58 and N59 at 640.** Their verdicts are probably unchanged, since both are
   relative, but "probably" is not this repo's standard and both are cheap replays now
   (33.6 s and 16.5 s after the gzip fix).
3. **Re-read the E1 comparison.** E1 was scored at 384 against D1 at 384. That pairing is
   internally consistent, but a +0.0016 margin sits inside a configuration change worth
   +0.1144, so nothing about E1 should be carried forward without re-measurement.
4. **Decide whether E2 is still worth the GPU.** Its hypothesis was fewer candidates; a
   configuration fix already delivers −16.8%. E2 may still add something on top, and it
   may not — that is now a question about the *residual*, not about the original 86.2.

## Limits

- **One checkpoint** (D1, arm D's `checkpoint_best_ema.pth`) and one tile size (640). This
  does not say what resolution is *best*, only that 640 is what this checkpoint was trained
  at and that the gap to 384 is large.
- Six clips from **two matches** — `ball_clean.py`: "every number this data can ever
  produce is same-match". The direction is unlikely to be an artefact of that; the
  magnitude may be.
- `oracle` is monotone in candidate-list length, and here the list got **shorter** (86.2 →
  71.7) while oracle **rose** — so the +0.0961 cannot be the list-length artefact that
  confounds E1's oracle gain. That is the one thing this comparison is cleaner about than
  N64's.
- n=1: one resolution pair, nothing swept. A sweep over resolutions would need its own
  held-out story, which six clips from two matches cannot support well (N59 measured that).
