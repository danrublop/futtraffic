# N45: our association on the STC2025 winner's boxes — 75% of the gap is the detector

**2026-09-18, branch `n30-operator-clicks`.** Protocol [`model/validation/n45_winner_boxes.py`](../../model/validation/n45_winner_boxes.py),
committed before running; n = 1 configuration per arm, nothing tuned, the challenge pair as fixed by the organisers.
Follows `research-stc2025-winners.md`, which found that GTATrack's repository ships its
final track files and its raw detector output for the pair. **Their detections are a research measurement only: the
model is AGPL and its weights unpublished; nothing from that arm can ship.**

## The three arms on the same frames (TrackEval, ground truth at every 2nd frame - the product's stride; their
answer files score the same at every frame, 0.5584 vs 0.5583 on 128057)

| 128057 (has a zero-click calibration: the metric stitch as the product runs it; the scene prior was NOT applied to the replayed boxes - it would have removed 0.16% of them, found by the regression review) | HOTA | DetA | DetRe | DetPr | AssA | IDF1 | IDSW | ids |
|---|---|---|---|---|---|---|---|---|
| our product on our boxes (`.p35`) | 0.3836 | 0.6573 | 0.7426 | 0.7888 | 0.2242 | 0.4022 | 264 | 58 |
| **our tracker + stitch on the winner's boxes** | **0.5149** | 0.7459 | 0.7853 | 0.8243 | 0.3558 | 0.5767 | 239 | 51 |
| the winner's pipeline (their answer file) | 0.5583 | 0.7458 | 0.7816 | 0.8288 | 0.4181 | 0.6999 | 252 | |

| 132831 (no calibration for this 3840x1504 canvas: our tracker alone, no prior, no stitch) | HOTA | DetA | AssA | IDF1 | ids |
|---|---|---|---|---|---|
| our tracker on the winner's boxes | 0.5273 | 0.7781 | 0.3580 | 0.5819 | 122 |
| the winner's pipeline | 0.6164 | 0.7803 | 0.4873 | 0.7292 | |

    @M 0.5584,0.5583,0.3836,0.6573,0.7426,0.7888,0.2242,0.4022,264,264 58,0.5149,0.7459,0.7853,0.8243,0.3558,0.5767,239,239 51,0.7458,0.7816,0.8288,0.4181,0.6999,252,0.5273,0.7781,0.3580,0.5819,122,0.6164,0.7803,0.4873,0.7292 :: .venv/bin/python model/validation/n45_winner_boxes.py --show

**128057: 0.3836 -> 0.5149 -> 0.5583. The detector accounts for +0.1314 of the +0.1748 gap, 75%; their tracker and
offline linker for the remaining 25%.** The pre-registered reading for "HOTA >= 0.50 on their boxes" applies: the
detector is most of the gap, the fine-tune plan (`plan-detector-finetune.md`) is the
first move, and the association question shrinks to the 0.06 AssA their Deep-EIoU + GTA-Link add on equal boxes.

    @D 0.0623 :: 0.4181 - 0.3558   [AssA, their pipeline minus our association, both on their boxes, 128057]
    @D 0.0434 :: 0.5583 - 0.5149   [HOTA, same]

## What it corrects

- The "ByteTrack 0.42 floor" this repo measured itself against was ByteTrack on the winner's fine-tuned detector
  (their own Table 3); on that detector ByteTrack's AssA was 0.26 and ours on their boxes is 0.36. Our association is
  not below the floor; our detection was.
- `association-not-detection-clears-the-leaderboard (internal note)`
  stands in its own terms (an oracle merge of our tracklets reaches AssA 0.53) and is now bounded: the realisable
  association gain on equal boxes is about 0.06 AssA / 0.04 HOTA, and it costs an appearance model that
  `research-gait-detection-legal.md` §C says the product must not build from a
  bodily template. Kit colour, position, persistence and the team-gate lead (12.7% of true joins blocked) are what is left.
- Their detector's non-GT boxes are 0.2-0.6 per frame against our 2.2 because it learned that officials are
  background. For HOTA that is worth about half of the DetA gap. The fine-tune (`plan-detector-finetune.md`
  §7) follows the winner and trains officials as background; an officials class was considered and not built.

## What follows

1. The fine-tune, when data and a GPU are in hand: RF-DETR (Apache) on SoccerTrack v2's boxes, officials as background,
   at 1280-class resolution or the product's tiles, three seeds, match-split, one held-out run - the plan page, now
   with a known target: DetA 0.75, occluded recall 0.6-0.9, non-GT boxes under 1 per frame.
2. Until then, the two N43b follow-ups (paint-row off-pitch test, persistence) and the team-gate lead, all CPU.
3. Not on the list: a bodily-template ReID. The 0.04 HOTA it would buy is priced by the legal page.
