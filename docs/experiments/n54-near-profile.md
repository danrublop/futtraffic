# N54: the near touchline held out - neither far-side term predicts it better than no term

> **Pre-registration not provable (2026-09-29):** the rule and the result landed in the same commit, 363e88a, on the day the registry's cut-off took effect; the verdict stands as weaker evidence. Recorded in `model/validation/protocol.py` NOT_PROVABLE.

**2026-09-23, branch `main`. Status: UNRESOLVED by its own rule. Nothing ships; `model/engine/autocal.py` is untouched.**
Protocol and code: [`model/validation/n54_near_profile.py`](../../model/validation/n54_near_profile.py), whose docstring
is the pre-registration - written, self-checked and fingerprinted (sha256 `b7fdeb8c...5f47`, 17:35) before any clip
was fitted by it. This is test (a) of `n39-column-term.md` §5, run held out instead of
in-sample. Follows `n38-kappa-free.md` (kappa is not one number; -0.34 is what the clips say)
and [`n36-far-touchline.md`](n36-far-touchline.md). **One install, the same seven four-minute clips every
calibration page since N30 has read.** It selects; it confirms nothing.

## The question

N39 left two descriptions of the far-touchline effect: a **cubic in the row** (kappa) and a **two-lens column
term** (beta, d). Both were fitted to every line, so their agreement with the near touchline was in-sample, and the
term with more parameters wins that by construction. Here each arm is fitted **without** the near touchline and the
two goal lines (`leave_out`), and the question is which camera lands on that paint.

Arms: **rigid** (control), **kappa -0.34 fixed**, **column** (beta, d free). n = 3, on top of the >= 34
configurations N36 and N39 count on these clips. Rule, fixed first: per clip, the lower near-touchline bin RMS
wins if lower by >= 0.25 px; CUBIC or COLUMN needs 5 of 7 wins; otherwise UNRESOLVED.

## The result

| arm | clips scored | near-touchline bin RMS, mean px | beats rigid by >= 0.25 px |
|---|---|---|---|
| rigid | 7 of 7 | 1.555 | - |
| kappa -0.34 | 7 of 7 | 1.947 | 0 of 7 |
| column | 7 of 7 | 1.666 | 1 of 7 |

    @M 1.555,1.947,1.666 :: .venv/bin/python model/validation/n54_near_profile.py --show   "scored 7 of 7" lines
    @M px on 0,px on 1 :: .venv/bin/python model/validation/n54_near_profile.py --show   "vs rigid (reported, not ruling)" lines
    @M kappa wins 1,column wins 4,ties 2,"ties 2, of" 7 :: .venv/bin/python model/validation/n54_near_profile.py --show   RULE line: kappa wins, column wins, ties, clips

**Kappa wins 1, column wins 4, ties 2: UNRESOLVED.** Column comes closer to a verdict than kappa does, but four is
not five.

**The line that matters more than the verdict (reported, not ruling):** the held-out near touchline is predicted
**best by the rigid camera**. Kappa makes it worse on every clip, and the column term beats rigid by the 0.25 px margin on
only 1 of 7. Neither term the far touchline wants helps the paint it was not fitted to. That fits N38's leave-out, where kappa shrank (but did not vanish) when the far
touchline was not fitted: whatever the far row asks for is not a property the rest of the drawing shares.

Nothing was dropped: no fit was refused, all 21 arm-clips scored. Near-touchline hits were 195-209 of 210 on six
clips, and 160-173 of 209 on 118578 (the ruling bins still filled).

    @M 195,209,210,160,173 :: .venv/bin/python model/validation/n54_near_profile.py --show   hits columns; 118578 rigid and kappa

## Post hoc, read after the rule, not ruling

- **The goal lines go the other way.** Median |residual| on the two held-out goal lines: rigid 1.29-1.738 px,
  kappa 0.917-1.226, column 0.558-0.943, column lowest on 7 of 7. The goal lines sit at the frame edges, where the
  column term is largest.

      @M 1.29,1.738,0.917,1.226,0.558,0.943 :: .venv/bin/python model/validation/n54_near_profile.py --show   goal median abs px per row

- **The column term's lens axes land outside the frame again,** d from 2855 to 6069 px against the 2048 the physics
  needs, on 7 of 7, against 6 of 7 in N39 (§2 there: d inside the frame on 1 of 7). Without the near line pinning it, the term becomes a smooth
  edge-bending term, not a two-lens stitch. That is how it helps the goal lines without helping the near touchline.

      @M 2855,6069 :: .venv/bin/python model/validation/n54_near_profile.py --show   column rows, d

- **A shared residual no arm models:** the last seventh of the near touchline reads -1.5 to -3.2 px (paint outside
  the model line) on every arm and every clip. It is a corner effect common to all three, not a discriminator.

      @M -1.52,-3.20 :: .venv/bin/python model/validation/n54_near_profile.py --show   last bin, smallest and largest magnitude


## What this changes

- **Do not build either term into `autocal` on this evidence.** The far-touchline fix (kappa -0.4, N36) passes its
  own rule by fitting the far row and costs the held-out near touchline. N39's in-sample advantage for the cubic does
  not survive holding lines out.
- **The goal-line gain is the one lead,** but it came from a term the physics cannot explain (lens axes outside the frame), so it
  is a description, not a model. Test it with its own rule, not by reading this table again.
- **The next discriminator is a second install** (N39 §5 (b)), as the rule said for UNRESOLVED, and not a fourth arm
  on these seven clips.
- Position error at the far touchline stays where `benchmark-status.md` §A puts it.
