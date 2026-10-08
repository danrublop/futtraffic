# N43b: the far pass finds the bench — the two fixes, and why one of them cannot work at half a metre per pixel

**2026-09-18, branch `n30-operator-clicks`.** Protocol [`model/validation/n43b_far_fixes.py`](../../model/validation/n43b_far_fixes.py),
committed before any run. Follows [`n43-far-tiles.md`](n43-far-tiles.md). Three dev clips; no held-out run (the gate
was not cleared); the product-path numbers below are EXPLORATORY and say so.

## The two fixes, offline on N43's own boxes

F1 drops far-tile boxes whose foot maps more than 0.5 m beyond the far touchline; F2 replaces the far pass's boxes with
the shipped grid's in a 100 px band about the far strip's bottom edge; F12 both.

| dev, 3 clips | occluded far | occluded mid | occluded near | isolated far | isolated mid | isolated near | non-GT boxes / frame |
|---|---|---|---|---|---|---|---|
| T0 shipped | 0.4285 | 0.5614 | 0.7085 | 0.8146 | 0.9621 | 0.966 | 2.651 |
| T1 far pass | 0.771 | 0.7282 | 0.6338 | 0.9923 | 0.9554 | 0.9183 | 12.793 |
| F1 margin | 0.77 | 0.7282 | 0.6338 | 0.9911 | 0.9554 | 0.9183 | 8.78 |
| F2 boundary | 0.7175 | 0.5701 | 0.6972 | 0.9826 | 0.9614 | 0.9627 | 12.139 |
| F12 both | 0.7165 | 0.5701 | 0.6972 | 0.9814 | 0.9614 | 0.9627 | 8.125 |

    @M 0.77,0.7282,0.6338,0.9911,0.9554,0.9183,8.78,0.7175,0.5701,0.6972,0.9826,0.9614,0.9627,12.139,0.7165,0.9814,8.125 :: .venv/bin/python model/validation/n43b_far_fixes.py --show

- **F2 does what it was meant to:** the near and mid recall the far pass appeared to lose comes back (isolated near 0.9183
  -> 0.9627, against 0.966 shipped). The strip-boundary merge artefact is real and this is its fix. It costs some of the
  far gain (0.771 -> 0.7175 occluded far) because the band also held real far-tile finds.
- **F1 removes only 4 of the 10 extra boxes per frame.** All three variants stop at the gate (non-GT within +1.0).

## What the remaining extra boxes are

`--why`: on 117093, 8.03 extra non-GT boxes per frame whose feet map to 0.3 m from the far touchline at the median
(p10 -0.3, p90 1.7 m), 28 px tall, and the five busiest 15x8 px spots are occupied in 2703, 1845, 1608, 1433 and 1223
of 2988 frames; on 118576, 6.65 per frame, the same 0.3 m median, spots occupied in 2025 to 1745 frames of 2988.

    @M 8.03,-0.3,0.3,1.7,median 28,2703,1845,1608,1433,1223,2988 :: .venv/bin/python model/validation/n43b_far_fixes.py --why 117093
    @M 6.65,2.3,median 27,2025,1937,1870,1784,1745 :: .venv/bin/python model/validation/n43b_far_fixes.py --why 118576

They do not move and they stand at the far line: the **bench** - substitutes and staff sitting and standing on the
steps just behind the far touchline, which on this ground is where the dug-outs are
([`img/n43b_117093_far_bench.png`](../img/n43b_117093_far_bench.png), the busiest spot of 117093, magnified 3x). The
shipped grid could not see them; the far pass can. **A metre margin through the map cannot separate them from
players on the line, because one far pixel is about half a metre**: their feet map to 0.3 m INSIDE the pitch. The
ground truth holds the 22 players only, so every one of them counts as a non-GT box.

## Exploratory: the product path on F12, dev clips (NOT the rule's run - the gate was not cleared)

| clip | HOTA | DetRe | DetPr | AssA | IDF1 | IDSW | no-owner rows | track ids |
|---|---|---|---|---|---|---|---|---|
| 118575 | 0.382 -> 0.422 | 0.615 -> 0.787 | 0.742 -> 0.735 | 0.270 -> 0.270 | 0.410 -> 0.428 | 191 -> 348 | 0.109 -> 0.138 | 44 -> 59 |
| 118576 | 0.344 -> 0.302 | 0.637 -> 0.698 | 0.767 -> 0.641 | 0.209 -> 0.174 | 0.372 -> 0.311 | 378 -> 418 | 0.105 -> 0.266 | 51 -> 74 |
| 117093 | 0.358 -> 0.359 | 0.659 -> 0.756 | 0.776 -> 0.608 | 0.220 -> 0.244 | 0.381 -> 0.355 | 304 -> 335 | 0.110 -> 0.318 | 61 -> 87 |
| mean | 0.3615 -> 0.3612 | 0.6370 -> 0.7469 | 0.7618 -> 0.6615 | 0.2330 -> 0.2296 | 0.3877 -> 0.3644 | 291 -> 367 | 0.1079 -> 0.2406 | 52 -> 73 |

    @M 0.3615,0.3612,0.6370,0.7469,0.7618,0.6615,0.2330,0.2296,0.3877,0.3644,291.0000,367.0000,0.1079,0.2406 :: .venv/bin/python model/validation/n43b_far_fixes.py --show
    @M 0.382,0.422,0.615,0.787,0.742,0.735,0.270,0.410,0.428,191,348,0.109,0.138,ids 44,ids 44.000-> 59,0.344,0.302,0.637,0.698,0.767,0.641,0.209,0.174,0.372,0.311,378,418,0.105,0.266,ids 51,ids 51.000-> 74,0.358,0.359,0.659,0.756,0.776,0.608,0.220,0.244,0.381,0.355,304,335,0.110,0.318,ids 61,ids 61.000-> 87 :: .venv/bin/python model/validation/n43b_far_fixes.py --show

On the clip with no bench in view (118575) the far pass is a clean +0.04 HOTA with precision held. On the two with the
bench, the bench becomes 20-30 extra tracks and a third of all track rows have no owner: precision falls 0.13-0.17 and
HOTA does not move or falls. The measurement is right and the product would be wrong to ship it as is.

## What follows, each pre-registered before it runs

1. **An off-pitch test that does not go through the map.** The far touchline's paint row is known to a fraction of a
   pixel per column (N36's grass-flanked ridge, `model/engine/autocal.search(native=)`); a box whose bottom sits above that
   row by more than the box's own noise is behind the line, whatever the map says. Players on the line during play
   are brief; the bench is not.
2. **Persistence.** A person occupying one spot for minutes is not a player in play; a tracker-level rule can hold such
   tracks out of the pitch statistics (and out of HOTA's false positives) without touching detection.
3. Only then the held-out far pass (four clips, about 10 h of detection on this machine), and the product rule.

**Followed by [`n47-bench.md`](n47-bench.md):** the paint and static tests, run - two-thirds of the extra non-GT boxes go, at about 0.03% of player matches or less; the gate still fails.

**What this page establishes:** resolution is the far-third detection problem and the far pass solves it
(0.43 -> 0.77 occluded, 0.81 -> 0.99 isolated); the strip-boundary fix works; the bench behind the far touchline is
the obstacle, and it is a geometry-and-persistence problem, not a detector one. Nothing ships from this page.
