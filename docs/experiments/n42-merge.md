# N42: the box merge is not what loses occluded players

**2026-09-18, branch `n30-operator-clicks`.** Protocol [`model/validation/n42_merge.py`](../../model/validation/n42_merge.py), committed
at `41484e6` before any pass; two harness fixes after (the reproduction check and the candidate caches both need the
product's pitch prior, which the product applies before writing its cache and a replay does not re-apply). Follows
[`n40-association.md`](n40-association.md) §2 and `research-gait-detection-legal.md` §B.3,
which ranked this first. **Three dev clips; the held-out four were NOT run - see the deviation below.**

## What was tested

`model/engine/detectors.py` `_merge` de-duplicates every box in a frame at intersection-over-smaller 0.55, and its own comment
says it drops a far player standing inside a near player's box. One unmerged detector pass per clip (every tile's
boxes kept, with tile identity), then three merges applied offline to the same boxes: M0 the shipped rule, M1 the
file's own upgrade path (IoS across tiles, IoU 0.55 within a tile), M2 the same with IoU 0.70 within a tile.
Reproduction: M0 after the prior equals the product's cache box for box on 0 differing frames of 3000 / 2988 / 2988.

## Recall by class, three dev clips (IoU >= 0.5 against the MOT boxes)

| merge | occluded far | occluded mid | occluded near | isolated far | isolated mid | isolated near | non-GT boxes / frame |
|---|---|---|---|---|---|---|---|
| M0 shipped | 0.4285 | 0.5614 | 0.7085 | 0.8146 | 0.9621 | 0.966 | 2.651 |
| M1 upgrade path | 0.4316 | 0.5667 | 0.7334 | 0.8146 | 0.9623 | 0.9661 | 2.655 |
| M2 within-tile 0.70 | 0.4318 | 0.5669 | 0.7336 | 0.8146 | 0.9623 | 0.9661 | 2.655 |

    @M 0.4285,0.5614,0.7085,0.8146,0.9621,0.966,2.651,0.4316,0.5667,0.7334,0.9623,0.9661,2.655,0.4318,0.5669,0.7336 :: .venv/bin/python model/validation/n42_merge.py --show

## The product path on M2, three dev clips (N41's scorer)

HOTA 0.3615 -> 0.3609, DetA 0.5659 -> 0.5671, DetRe 0.6370 -> 0.6382, DetPr 0.7618 -> 0.7622, AssA 0.2330 -> 0.2313,
IDF1 0.3877 -> 0.3935, IDSW 291.0 -> 290.3.

    @M 0.3615,0.3609,0.5659,0.5671,0.6370,0.6382,0.7618,0.7622,0.2330,0.2313,0.3877,0.3935 :: .venv/bin/python model/validation/n42_merge.py --show

**Verdict: NOT SUPPORTED, and the hypothesis is refuted.** The merge deletes a third of a point of far-occluded
recall and 2.5 points near; the detector does not produce the second box in the first place. **Deviation from the
protocol:** the held-out clips were not run. The rule asks for +0.010 HOTA and the candidate moves detection recall by
0.001; four detector passes to confirm a null would be waste, and the deviation is recorded here rather than hidden.

**A side finding that matters for N41 and anything after it.** Two runs whose boxes differ by a fraction of a percent
moved distance per covered metre raw from 1.2177 to 1.1633 and HSR raw from 1.4760 to 1.3740 on the dev mean, in
opposite directions on different clips. On four-minute clips those two figures are noisier than N41's +-0.01 rule
assumed; a future rule should set its metre thresholds from a measured run-to-run spread, not from round numbers.

    @M 1.2177,1.1633,1.4760,1.3740 :: .venv/bin/python model/validation/n42_merge.py --show

## What is left on the detection ladder

Confidence (N41) and the merge (N42) are both out. What remains is making the detector SEE occluded and far-third
players: per-tile resolution on the far band (a measurement under the polygon-only prior; the file's earlier far-band
reading was made with the retired height gate on), then a fine-tune of RF-DETR Medium on SoccerTrack v2's own boxes
with occluded and far-third crops oversampled - match-split, one held-out run, a rented GPU, a licence notice.
