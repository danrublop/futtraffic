# N47: the far-side bench, cut by the paint and by persistence - precision back at shipped; identity switches now block the far pass

**2026-09-18, branch `n30-operator-clicks`.** Protocol [`model/validation/n47_bench.py`](../../model/validation/n47_bench.py),
committed before any run (679df0a). Follows [`n43b-far-fixes.md`](n43b-far-fixes.md), which found that the far tile
pass's extra boxes are the bench behind the far touchline and named the two tests that do not go through the map.
Dev clips only; nothing ran on held-out clips.

## What was tried

On N43b's F12 boxes (the far pass with the strip-boundary fix and the 0.5 m margin):

- **B1 paint** - a box whose foot sits more than TOL px above the far touchline's paint row (found on the clip's
  players-removed median by autocal's grass-flanked rule) stands beyond the line. TOL was set from the players' own
  ground-truth feet, not from the bench: 2.37 px, the 99.5th percentile over 41,471 GT boxes near the far paint.
- **B2 static** - a box within 20 px of the far paint whose 15x8 px foot cell is occupied in at least half the frames
  of the surrounding minute.
- **B12** both.

    @M 2.37,41471 :: .venv/bin/python -c "import json;t=json.load(open('model/validation/out/stc2025/n47_tol.json'));print(round(t['tol_px'],2),t['gt_boxes_used'])"

## Result, dev (three clips), N40's recall classes at IoU 0.5

| arm | occluded far | occluded mid | occluded near | isolated far | isolated mid | isolated near | non-GT boxes / frame |
|---|---|---|---|---|---|---|---|
| T0 shipped | 0.4285 | 0.5614 | 0.7085 | 0.8146 | 0.9621 | 0.966 | 2.651 |
| F12 far pass | 0.7165 | 0.5701 | 0.6972 | 0.9814 | 0.9614 | 0.9627 | 8.125 |
| B1 paint | 0.7165 | 0.5701 | 0.6972 | 0.9809 | 0.9614 | 0.9627 | 5.519 |
| B2 static | 0.7165 | 0.5701 | 0.6972 | 0.9814 | 0.9614 | 0.9627 | 5.679 |
| B12 both | 0.7165 | 0.5701 | 0.6972 | 0.9809 | 0.9614 | 0.9627 | 4.433 |

    @M 5.519,5.679,4.433,0.9809 :: .venv/bin/python model/validation/n47_bench.py --show

**All three stop at the gate** (non-GT boxes per frame at most T0 + 1.0 = 3.651). They do what they were built for:
B12 removes two-thirds of the far pass's EXTRA non-GT boxes over shipped (8.125 -> 4.433 against 2.651) and takes
almost no players with it - of F12's boxes that matched a ground-truth box, B12 dropped about 0.02% on 118575, about
0.03% on 118576 and none on 117093 (that count matches any GT box at IoU 0.5, not one-to-one) - and far recall moves
by at most 0.001 (isolated far 0.9814 -> 0.9809). On 117093, the clip with the bench in view, it removes 7.0 boxes per
frame. It is not enough. TOL was set on the same dev clips it is scored on, so its "about 0.5% of players at the line"
is an in-sample figure.

    @D 0.674 :: (8.125 - 4.433) / (8.125 - 2.651) [the table: the share of F12's extra non-GT boxes over T0 that B12 removes]

    @M 0.0002,0.0003,7.0 :: .venv/bin/python model/validation/n47_bench.py --show   B12 per clip

## What is left, and why neither test reaches it

`--why` on the boxes B12 keeps that match neither a player nor a shipped box:

    @M 1.71,0.32,3.07,0.39,1.89,0.96 :: for q in 118575 118576 117093; do .venv/bin/python model/validation/n47_bench.py --why $q; done

Per clip (extra boxes per frame; share with paint found at their column; paint row minus foot row p10 / p50 / p90 px;
foot row p10 / p50 / p90):

| clip | per frame | paint found | paint - foot, px | foot row |
|---|---|---|---|---|
| 118575 | 1.71 | 0.32 | -26.0 / -0.9 / 2.0 | 152 / 156 / 189 |
| 118576 | 3.07 | 0.39 | -0.9 / 0.9 / 2.1 | 154 / 156 / 176 |
| 117093 | 1.89 | 0.96 | -1.3 / 0.6 / 2.2 | 154 / 168 / 208 |

    @M -26.0,-0.9,2.0,152.0,156.0,189.0,-1.3,0.6,2.2,154.0,168.0,208.0 :: for q in 118575 117093; do .venv/bin/python model/validation/n47_bench.py --why $q; done

    @M 0.297,0.385,0.424 :: .venv/bin/python -c "import json;print(json.load(open('model/validation/out/stc2025/n47_tol.json'))['far_paint_width_cover'])"

1. **The paint was not found where most of them stand, on 118575 and 118576.** Only 32% and 39% of them are in a
   column where the far paint was found (the paint search covers 0.297 and 0.385 of the image width on those two
   clips, 0.424 on 117093), so B1 had no evidence and kept them, as it was told to. Their median foot row is 156, the
   far line's.
2. **Where the paint was found, the middle of them stands on it.** On 118576 and 117093 the p10 to p90 of paint row
   minus foot row is about -1 to +2 px, inside TOL 2.37 - that is, on the line to within the paint test's own
   resolution. 118575 has a tail well inside the pitch (p10 -26 px), and 117093's foot rows run deeper (median 168),
   so not all of the residual is at the line. Whether they move is NOT measured: B2 misses a person who stands still
   but whose box jitters across 15 px cells as surely as it misses a mover. Who they are is an interpretation (the
   assistant referee, players warming up); checking it needs the video.

## What follows, pre-registered before either runs

The two below are named, not yet pre-registered: each gets its constants and gate in its protocol file, committed
before it runs.

- **B3, paint gaps filled by the camera.** Where the paint is not found, use the zero-click camera's far line shifted
  by the found paint's residual, interpolated along the line - the camera exists at every column and its far-line error
  varies slowly. Targets leftover (1); n = 2 for the paint rule. Gate: this page's L1 gate, unchanged.
- **B4, a line-hugger rule at track level.** A track that stays within a few pixels of the far touchline for its whole
  life and never goes deeper into the pitch is not a player in play; a player who reaches the line leaves it. Needs
  the tracker, so it is judged on the product path (N43's rule), not here. Targets leftover (2). First, look at the
  video to see who the leftover boxes are.
- Only if B3 + B4 clear the gate: the held-out far pass (about 10 h of detection) and N43's product rule.

## Round 2: B3 and B4 (protocol 0c2ecd0, committed before either ran)

A look at the leftover boxes first (magnified crops, 24 per clip on 118576 and 117093): most are plainly not players
in play - the assistant referee with his flag, substitutes in bibs warming up and stretching, staff and spectators on
the steps and under the tent just behind the line, many in loose groups that shift across cells. Seen, not counted.

**B3, the far paint filled from the camera's line** (the found paint's smoothed residual interpolated across the gaps):

| arm | occluded far | occluded mid | occluded near | isolated far | isolated mid | isolated near | non-GT boxes / frame |
|---|---|---|---|---|---|---|---|
| B3 | 0.7163 | 0.5701 | 0.6972 | 0.9809 | 0.9614 | 0.9627 | 3.524 |

    @M 0.7163,0.6972,0.9809,3.524 :: .venv/bin/python -c "import json;print(json.load(open('model/validation/out/stc2025/n47_l1.json'))['B3'])"

It does what it was for: non-GT boxes 4.433 -> 3.524, under the gate's 3.651, with far recall held against F12. **It
still stops at L1**, on a line it does not touch: occluded near 0.6972 against T0's 0.7085 is 0.0113 below, and the
gate allows 0.01. That loss is F12's own (the same 0.6972 in the table above) - N43b's strip-boundary fix - not the
bench filter's.

**B4, the line-hugger rule, on B3's tracks through the product path** (dev, exploratory): it dropped 3 of 62, 11 of
74 and 4 of 74 raw tracks on 118575, 118576 and 117093.

| mean of three dev clips | HOTA | DetRe | DetPr | AssA | IDSW | no-owner |
|---|---|---|---|---|---|---|
| T0 shipped | 0.3615 | 0.6370 | 0.7618 | 0.2330 | 291.0 | 0.1079 |
| B3 | 0.3752 | 0.7471 | 0.7516 | 0.2230 | 368.3 | 0.1388 |
| B34 | 0.3784 | 0.7471 | 0.7673 | 0.2230 | 367.0 | 0.1208 |

    @M 0.3615,0.3752,0.3784,0.7471,0.7673,0.2230,291.0000,367.0000,0.1208 :: .venv/bin/python model/validation/n47_bench.py --show4
    @M 0.0170,-0.0099,1.261,0.0129,-0.0001 :: .venv/bin/python model/validation/n47_bench.py --show4
       the gate's parts

Against the rule: HOTA +0.0170 (needs +0.015) passes; worst-clip DetPr -0.0099 (needs >= -0.020) passes; no-owner
+0.0129 (<= 0.02) passes; DetRe unchanged from B3 (worst clip -0.0001), so the rule removed no players; **IDSW 1.261x
T0 (<= 1.15) fails.** B34 does not pass.

What it means: **the bench is solved at the product level on these clips** - N43b's far pass lost 0.13-0.17 DetPr on
the two clips with the bench in view; with B3 + B4 precision is at or near shipped on every clip. What still blocks the
far pass is ASSOCIATION on the extra far boxes: identity switches up by a quarter, and on 118576 AssA 0.209 -> 0.179.
More far players found means more far players to keep apart, at a resolution where they are 20-30 px tall.

## What follows

- The far pass is now held back by two things, neither of them the bench: N43b's strip-boundary fix costs occluded-near
  recall 0.011, and the far boxes cost association (IDSW x1.26). Both are about the far pass's own boxes, and the
  detector fine-tune (N46) changes those boxes - so the next honest step is to re-run B3 + B4 on the fine-tuned
  detector's far pass once N46 returns, not to tune these constants against the current boxes.
- Held-out: nothing here has touched the four held-out clips; it needs the far pass run on them (about 10 h).

**What this page establishes:** the bench behind the far touchline is removable without removing players - by the
paint (filled from the camera where it is not found) and by whole-track persistence along the line - and with it gone
the far pass's precision is back at shipped (dev, exploratory). The far pass still does not ship: identity switches
rise by a quarter and N43b's boundary fix costs near-side occluded recall. Nothing ships from this page.

    @D 3.69 :: 8.125 - 4.433 [the table: non-GT boxes per frame B12 removes from F12]
