# N41: a 0.3 confidence floor is NOT SUPPORTED as a product setting

**2026-09-18, branch `n30-operator-clicks`.** Protocol [`model/validation/n41_conf.py`](../../model/validation/n41_conf.py), committed at
`f02a558` before any pass; harness fix `db0a866` (ground truth staged at the tracks' stride, metres through the
zero-click map - both arms, found on the dev run, no threshold moved). Follows [`n40-association.md`](n40-association.md)
§2 (occluded players are detected weakly, not missed). **Seven same-install clips; DEV three, HELD four run once.**

The product's own path (detector -> scene prior -> tracker -> split + stitch -> identity) at conf 0.3 against what ships
at 0.5. TrackEval on the stitched tracks; distance and HSR from the identified tracks against ST2's feed through the
frozen N34 map, raw and over the fragments' floor.

| HELD, mean of four clips | 0.5 (ships) | 0.3 | rule |
|---|---|---|---|
| HOTA | 0.3639 | 0.3711 | >= +0.010: **FAIL** (+0.007) |
| DetA | 0.6186 | 0.6344 | |
| DetRe / DetPr | 0.6994 / 0.7695 | 0.7225 / 0.7642 | DetPr >= -0.020 every clip: ok |
| AssA | 0.2157 | 0.2184 | |
| IDF1 | 0.3872 | 0.3982 | |
| IDSW | 315.0 | 337.75 | <= 1.15x: ok |
| distance per covered metre, raw | 1.1444 | 1.1530 | <= +0.010: ok |
| HSR ratio, raw | 1.1778 | 1.2067 | must not rise: **FAIL** |
| no-owner share | 0.0999 | 0.1006 | ok |

    @M 0.3639,0.3711,0.6186,0.6344,0.6994,0.7225,0.7695,0.7642,0.2157,0.2184,0.3872,0.3982,315.0000,337.7500,1.1444,1.1530,1.1778,1.2067,0.0999,0.1006 :: .venv/bin/python model/validation/n41_conf.py --show

DEV (three clips, looked at first): HOTA 0.3615 -> 0.3705, IDSW 291 -> 339, distance raw 1.2177 -> 1.1852, HSR raw
1.4760 -> 1.3373 - the same borderline HOTA, and there the metre figures went the other way.

    @M 0.3615,0.3705,291,339,1.2177,1.1852,1.4760,1.3373 :: .venv/bin/python model/validation/n41_conf.py --show

**Reading.** The boxes are real: recall +0.023 for 0.005 of precision, on every held-out clip. Through the tracker they
buy 0.007 HOTA and, on three of four held-out clips, a higher sprint over-read - weak far-side boxes forming short fast
fragments. A confidence floor makes the detector fire more often on occluded players; it does not make it find them
well. 0.5 stays. n = 1 candidate value.

**What follows** (`research-gait-detection-legal.md` §B.3): the product's own
frame-wide box merge at 55% intersection-over-smaller (`model/engine/detectors.py` `_merge`), which by its own comment deletes a
far player standing in front of a nearer one, tested next on the cached frames; then per-tile resolution on the far
band; then a fine-tune on SoccerTrack v2's occluded and far-third boxes, match-split, one held-out run.
