# Benchmarks

Every number below carries a tag (see [How numbers are kept honest](README.md#numbers)).
The tag lines under each table are the evidence: the command that re-measures the value, or the
source it was read from, and what was held fixed. The SoccerTrack v2 footage is real match video,
but all of it comes from **one panoramic camera at one ground**. Treat every figure as a property of
that install.

### Calibration and measurement: zero clicks vs. a hand-clicked homography

| | hand-clicked homography | zero-click |
|---|---|---|
| median player-position error, clip 118575 | 8.0 m | **2.33 m** |
| distance covered vs. the dataset's feed, median over seven clips (range) | 1.226 to 1.558 | **1.183** (1.120 to 1.221) |
| high-speed running vs. the feed, median over seven clips (range) | 1.935 to 4.213 | **1.199** (1.110 to 2.026) |

    @M 8.0,2.33 :: .venv/bin/python -c "import json;r=json.load(open('model/validation/out/stc2025/n35_e2e.json'));print(*[r[a][k] for k in ('pos_median_m',) for a in ('SHIP','AUTO')],r['SHIP']['slope']['all'],r['AUTO']['slope']['all'],r['SHIP']['distance_over_floor'],r['AUTO']['distance_over_floor'],r['SHIP']['hsr_over_floor'],r['AUTO']['hsr_over_floor'])"
       held fixed: clip 118575 (ST2 train split), the product config, stride-2 frames; n=1 per arm.
       Median metres from our own boxes' feet to the dataset feed. Rows with no IoU owner are
       excluded: 9.5% (hand-clicked) and 10.9% (zero-click). One clip, one install.
    @M 1.120,1.221,1.183,1.110,2.026,1.199,1.226,1.558,1.935,4.213 :: .venv/bin/python model/validation/n35_raw.py
       held fixed: the seven same-install ST2 clips, the product config, nothing tuned; n=1 per arm.
       Raw ratios against the feed, not over the tracking floor: zero-click distance min, max,
       median, then HSR min, max, median; then the hand-clicked distance and HSR ranges.

Caveats: the seven clips are one camera on one pitch, so they are seven samples of the players, not
seven tests of the calibration. The feed itself reads a few percent long, so "vs. the feed" is not
"vs. the truth". Distance and high-speed running still over-read, and what remains comes from
fragmented tracks, not from the map. Source: [n34, sections 7 and 8](experiments/n34-zero-click.md).

### Detection recall (stock detector, as shipped)

Share of ground-truth players detected at IoU of at least 0.5, by depth third of the image. A player
counts as partly occluded when another player covers at least 35% of their box.

| | near third | middle third | far third |
|---|---|---|---|
| isolated | 0.9703 | 0.9716 | 0.8775 |
| partly occluded | 0.6846 | 0.6140 | 0.4928 |

    @M 0.9703,0.9716,0.8775,0.6846,0.6140,0.4928 :: .venv/bin/python model/validation/n40_occlusion_recall.py
       held fixed: the product's CoreML RF-DETR detections, conf 0.5, stride 2, the cached frames
       of the seven ST2 clips (one install), IoU 0.5; n=1 configuration.

Occluded players are where tracks end. Source: [n40, section 2](experiments/n40-association.md).

### Tracking on SoccerTrack v2, match 128057

TrackEval, ground truth at every second frame (the product's stride). The middle row replays the
challenge winner's published detections through our unchanged tracker and stitcher. It locates the
gap; it is not something that can ship (their detector is AGPL and its weights are unpublished).

| | HOTA | DetA | AssA | IDF1 |
|---|---|---|---|---|
| this project, stock detector (as shipped) | 0.3836 | 0.6573 | 0.2242 | 0.4022 |
| our tracker on the winner's detections | 0.5149 | | | |
| the winner's full pipeline, same frames | 0.5583 | | | |

    @M 0.3836,0.6573,0.2242,0.4022,0.5149,0.5583 :: .venv/bin/python model/validation/n45_winner_boxes.py --show
       held fixed: ST2 match 128057 (one sequence of the challenge pair), the product's
       zero-click calibration and metric stitch, GT at every 2nd frame; n=1 configuration per arm,
       nothing tuned.
    @D 0.7516 :: (0.5149 - 0.3836) / (0.5583 - 0.3836)   [share of the HOTA gap closed by the winner's detections, inputs from the tag above]

On this match the detector accounts for about three quarters of the gap to the winner. One match
only. Source: [n45](experiments/n45-winner-boxes.md).

### SoccerTrack Challenge 2025 leaderboard (cited)

HOTA on the challenge pair (matches 128057 and 132831). The leading entries use detectors fine-tuned
on this footage; ours is a stock detector with no dataset-specific training.

| entry | HOTA |
|---|---|
| GTATrack (wccjs, winner) | 0.60 |
| mdk-tdu | 0.59 |
| SShota | 0.59 |
| Nagoya U. (takashun, YMori) | 0.59 |
| Deep-EIoU (baseline in the winner's paper) | 0.54 |
| ByteTrack (baseline in the winner's paper) | 0.42 |
| this project, stock detector | 0.369 |

    @C 0.60,0.59 :: organisers' STC2025 leaderboard sheet, https://docs.google.com/spreadsheets/d/1ArnzG1jMJFPYElxmfwp3p1DEODKu7sIagrKgQtalz6U (gid 1845616445), rows wccjs, mdk-tdu, SShota, takashun, YMori
    @C 0.54,0.42 :: arXiv:2602.00484 (GTATrack), Table 3, Deep-EIoU and ByteTrack rows, both on the authors' fine-tuned YOLOv11x detector
    @C 0.369 :: docs/experiments/st2-input-ledger.md, STC2025 table (model/validation/st2_mot.py run of 2026-09-08, mean of 128057 and 132831, conf 0.5, no remap, no calibration, n=1)

The ByteTrack baseline is ByteTrack on the winner's fine-tuned detector, not an independent floor.
Our 0.369 is a recorded run whose tracks predate later changes, so it is cited rather than re-run;
the re-measured single-match figure is the 0.3836 above. Under our harness the winner's own answer
files score slightly below their published 0.60 ([n45](experiments/n45-winner-boxes.md)).

### Ball: the new clean detector (not in the app)

Scored on six TeamTrack clips from two matches, not on SoccerTrack v2. "Ball among candidates" asks
whether any candidate in the frame is the ball; "pick is the ball" asks whether the picker chose it.
The only difference between the rows is the model's input size: earlier ball numbers in this repo
were measured at 384 px on a detector trained at 640.

| model input | ball among candidates | pick is the ball | candidates per frame |
|---|---|---|---|
| 384 px | 0.8822 | 0.4706 | 86.2 |
| 640 px (as trained) | 0.9783 | 0.5850 | 71.7 |

    @M 0.8822,0.9783,0.4706,0.5850,86.2,71.7 :: .venv/bin/python model/validation/n64_ball_negatives.py --res
       held fixed: ball_cmp.CLIPS, all six TeamTrack clips, 1800 gt-present frames, gate 20 px,
       N52 arm D1's checkpoint, tile 640 on both sides; n=1 configuration.

The ball is almost always among the candidates; choosing it is the weak step. Sources:
[n52, section 7b](experiments/n52-clean-ball-data.md), [n65](experiments/n65-every-ball-number-was-measured-at-384.md).

### Jersey numbers (OCR)

| | value |
|---|---|
| group precision at the shipped threshold (share of written numbers that are right) | 0.4375 |

    @M 0.4375 :: .venv/bin/python model/validation/jersey_eval.py --score
       held fixed: 118575 1st half, frames 15000-29999, jobs.apply_identity/assisted, docTR 1.1.0,
       min_conf 0.6, the job's remap calibration; n=1 configuration. Drop: 79.1% of the half's
       frames not scanned; 5 of 21 written groups have no GT player and leave precision.

This fails the project's own bar for showing a number to a user. One window of one match. Source:
[n49](experiments/n49-jersey-ocr.md).
