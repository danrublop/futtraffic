# N37: feet on the paint — a map-free depth reference, first pass

**2026-09-17, branch `n30-operator-clicks`.** Protocol and code: [`model/validation/n37_feet_on_lines.py`](../../model/validation/n37_feet_on_lines.py)
(committed at `436056d` before any position was read; revision 1 at the next commit, recorded in the docstring). Follows
[`n36-far-touchline.md`](n36-far-touchline.md) section 4 (the ~4% along-ray gap and its two explanations) and
`n32-depth-reference.md`. **Development, ONE install, seven four-minute clips, human-drawn
boxes. Twelve events. UNRESOLVED by its own rule (needs 30 at the far touchline); the direction is reported because
it was predicted first.**

## The idea, the author's

The target setup has one panoramic camera, so depth truth has to come from the picture itself. A player standing with a foot ON a painted line, in the image, is on
that line in the world, to the line's width - and neither our map nor ST2's feed had any say in it. Six lines parallel
to the touchlines give depth at six known ranges. Events are found with their own logic: stands (a box foot still for
0.8 s or more), labelled by the nearest drawing line, LOCATED by the paint the line search finds along that line, the
foot within a pixel (or the line's half-width) of that paint.

## What the seven clips hold

> **CORRECTED 2026-09-28.** `model/validation/out/stc2025/n34_autocal.json` is NOT the camera the product runs today: it predates the length fit (`autocal._settle`), and the shipped code run on the same median frame gives a different camera. The SHIP arm here is N34's camera, kept as the record; it was not re-run. The current product camera for 118575 is frozen in [`model/validation/fixtures/calib/golden.json`](../../model/validation/fixtures/calib/golden.json), checked by `model/validation/check.py calib --real`.

| line | events | clips | SHIP (zero-click, ships) | KAPPA (N36 candidate) | FEED (ST2) |
|---|---|---|---|---|---|
| far touchline (y = 0) | 8 | 4 | +1.55 m | +0.09 m | -2.38 m |
| far penalty-area edges (y = 13.84) | 4 | 2 | +1.18 m | +0.80 m | -1.94 m |
| the four other depth lines | 0 | | | | |

Median signed error (+ = inside the pitch from that line, i.e. nearer the camera). Of the 8 far events, three are one
118575 player standing in one spot, split into three stands: about six independent spots.

    @M touch_far 8,4,1.55,0.09,-2.38,4,pen_far 4 2,1.18,0.8,-1.94 :: .venv/bin/python model/validation/n37_feet_on_lines.py --score

- **The rule says UNRESOLVED** (8 of the 30 far events it asks for). Nothing here is a verdict.
- **The direction was predicted.** The protocol wrote both hypotheses down first: a feed scaled 4% about the camera's
  foot would put far-touchline standers about 3.3 m OUTSIDE the line with the map near zero; a map with a height or
  vertical-law error would put them 3 m INSIDE with the feed near zero. Twelve events read: FEED outside by 2-2.4 m,
  the shipping map inside by 1.2-1.6 m, the KAPPA candidate within 0.1-0.8 m. Both of section 4's suspects are
  present, and the feed's share is the larger.
- **The far-touchline feed check:** 75% of far events have feed y <= 0, so the feed is not clipped to the pitch.
- **Why so few events.** Four-minute clips; nobody stands still on the near lines for 0.8 s in them (the near lines are
  5 px wide and the locator's grass flanks fall on paint there, which also needs fixing); the strict locator dropped 19
  of the first pass's 31 "events" that were not on paint at all (revision 1).

## What it takes to make this decisive

The instrument works; the sample is the problem. Full matches at this install carry the feed for 90 minutes and would
give tens of far-touchline stands (throw-ins alone are 20-40 a match) - with our own detector's boxes instead of the
human-drawn ones, whose bottom-edge accuracy then becomes part of the test. Fix the near-line locator (flanks scaled to
the line's width) so the near touchline and the near box edges join in; they are where 4% is 0.6 m and a pixel is
3 cm, so they say whether the map's error is a constant fraction of range or not. Then the rule as written decides.
