# N28: a professional league's top speed cannot flag the product on four-minute clips

> **Corrected 2026-09-14 after independent review of the result.** The verdict stands. Its reading
> does not:
> - **The over-read is not "too small to catch".** The product's tracks mix short and long spans, and
>   PSV-99 rises with span. Over matched spans the product reads well above the truth, and its long
>   tracks sit above the league's maximum (§1, §4).
> - **Most of the gap between N21's 10.58 m/s and 8.70 is a change of population,** not a maximum set
>   against a percentile. The app does show a single-sample top speed wherever the cap never binds (§4).
> - **The prior is a per-match mean** of each player's PSV-99, per position group, with no goalkeepers,
>   and its maximum comes from a two-match row. The verdict holds under stricter maxima (§1).
> - **The GNSS "agreement" was a coincidence of summaries,** and the GNSS failed its own check (§3).
> - **Not stated until review:** the 1.0 s floor is this protocol's own addition, and it dropped
>   tracks and players (§1).

**2026-09-14, branch `full-pipeline`.** Protocol in the header of
[`model/validation/plausibility.py`](../../model/validation/plausibility.py), committed in `a4aae51`
before any arm was computed (a wording fix in `a1656e9` left the rule unchanged). The one
exception is ST2's whole matches, which an exploratory check had already read, so that arm is
reported and not ruled. Results in `model/validation/out/n28_results.json` and `n28.log`, gitignored.

`n21-product-now.md` knows the product over-reads speed only because
ST2 has ground truth, and footage a user uploads has none. Top speed has a published range:
SkillCorner's open A-League aggregates give every player a PSV-99, the 99th percentile of
their speed samples above 4 m/s
(`research-physical-metrics-and-joins.md` §1, §6).
N28 asked whether "the typical player outruns the fastest player in a professional league"
catches the product without flagging the truth. The two sources, scored the same way:
- **truth:** ST2's own per-row speed field.
- **product:** what ships, `.p21id` through N6's calibration, per track as the app reports
  it, with the 9.5 m/s cap lifted.

## 1. The result

    @M 8.9722,6.9007,8.6992,8.654,0.0115,0.4377,174,313,False,BLIND TO THE PRODUCT :: .venv/bin/python -c "import json;r=json.load(open('model/validation/out/n28_results.json'));v=r['verdict'];print(round(r['prior']['max'],4),round(v['truth']['median'],4),round(v['product']['median'],4),round(v['shipped']['median'],4),round(v['truth']['above_max'],4),round(v['product']['above_max'],4),v['truth']['players'],v['product']['players'],v['fragile'],v['follows'])"
       held fixed: N27's eight clips (N22's nine without 132877), each clip's 4 minutes, `.p21id`
       tracks, N6's calibration, build_match's population, SkillCorner's A-League 2024/25 aggregates
       as the prior; n=1 rule, nothing tuned. Printed: the prior's maximum PSV-99 in m/s; the median
       PSV-99 of the truth, the product and the product under its cap; the share of truth players and
       of product tracks above the prior's maximum; players and tracks; fragile; the verdict.

    @M 406,BLIND TO THE PRODUCT :: .venv/bin/python model/validation/plausibility.py --n28  #slow

- **Valid.** The truth is not flagged: its typical player's PSV-99 is well under the league's
  fastest, and only this share of its players are above it:
  `@D 0.0115 :: 2 / 174`
- **BLIND TO THE PRODUCT.** The product's typical track reads faster than the truth on the same
  clips, by this factor:
  `@D 1.2606 :: 8.6992 / 6.9007`
  and still under the league's maximum. Leaving out any one clip gives the same verdict. The cap
  barely matters here: the capped median is 8.654. **Corrected after review:** that factor divides a
  median over tracks of every span by a median over 4-minute truth windows, and PSV-99 rises with
  span. Scored over matched spans, in three span bands, the product reads this many times the truth:
  `@C 1.30x, 1.41x, 1.35x :: independent review 2026-09-14, truth_fragmented.py outside the repo, re-run in session`
  Its tracks longer than two minutes have a median PSV-99 of
  `@C 9.24 m/s :: same`
  which is above the league's maximum. Review check, not registered.
- **Dropped, and not stated until review.** The 1.0 s floor removed these from build_match's
  population, of which the product arm kept 313 and the truth 174:
  `@C 40 tracks, 2 players :: independent review 2026-09-14, re-counted in session`
  Keeping them could only lower the product's median. The floor is this protocol's own addition: the
  definition it quotes has none.
- **The prior, read closer after review.** Printed:
  - rows with at most five matches where `psv99` equals `psv99_top5`, and all such rows;
  - rows with more than five matches where the two differ, and all such rows;
  - the match count of the row holding the maximum;
  - the maximum over rows with at least five matches, then the maximum of `psv99_top5`, in m/s.

      @M 218,218,188,188,max_psv99_matches 2,8.83,9.09 :: .venv/bin/python -c "import csv;r=list(csv.DictReader(open(__import__('os').path.expanduser('~/Downloads/skillcorner-opendata/data/aggregates/aus1league_physicalaggregates_20242025.csv'))));c=lambda x:int(float(x['count_match']));e=lambda x:float(x['psv99'])==float(x['psv99_top5']);print(sum(e(x) for x in r if c(x)<=5),sum(c(x)<=5 for x in r),sum(not e(x) for x in r if c(x)>5),sum(c(x)>5 for x in r),'max_psv99_matches',c(max(r,key=lambda x:float(x['psv99']))),round(max(float(x['psv99']) for x in r if c(x)>=5)/3.6,2),round(max(float(x['psv99_top5']) for x in r)/3.6,2))"

  So `psv99` is a mean of per-match values, rows are per player and position group, and no
  goalkeeper appears. The ruled maximum comes from a two-match row. The product's median sits under
  either stricter maximum, so the verdict holds, but the share of the margin it uses rises.
- **Beside the rule, not ruled:** the product puts 0.4377 of its tracks above the league's
  maximum, against 0.0115 of truth players. That separates the two sources far more clearly than
  the median does. A rule built on it now would be chosen after seeing this result, so it needs its
  own protocol and data this run did not touch.

## 2. Per clip

    @M truth 22,6.76,product 35,10.85,0.77,truth 22,6.97,product 50,8.64,0.42,truth 21,7.44,product 29,8.46,0.34,truth 21,6.53,product 31,7.76,0.29,truth 22,7.26,product 34,8.65,0.47,truth 22,7.51,product 40,8.87,0.47,truth 22,6.9,product 47,8.7,0.36,truth 22,6.86,product 47,8.21,0.38 :: .venv/bin/python -c "import json;v=json.load(open('model/validation/out/n28_results.json'))['verdict']['per_clip'];print(*[f\"truth {v[c]['truth']['players']},{round(v[c]['truth']['median'],2)} product {v[c]['product']['players']},{round(v[c]['product']['median'],2)},{round(v[c]['product']['above_max'],2)}\" for c in sorted(v)])"

| clip | truth players | truth median PSV-99, m/s | product tracks | product median PSV-99, m/s | product tracks above the league's max |
|---|---|---|---|---|---|
| 117092 | 22 | 6.76 | 35 | 10.85 | 0.77 |
| 117093 | 22 | 6.97 | 50 | 8.64 | 0.42 |
| 118575 | 21 | 7.44 | 29 | 8.46 | 0.34 |
| 118576 | 21 | 6.53 | 31 | 7.76 | 0.29 |
| 118577 | 22 | 7.26 | 34 | 8.65 | 0.47 |
| 118578 | 22 | 7.51 | 40 | 8.87 | 0.47 |
| 128057 | 22 | 6.9 | 47 | 8.7 | 0.36 |
| 128058 | 22 | 6.86 | 47 | 8.21 | 0.38 |

Only 117092, the clip with N26's largest HSR over-read, flags on its own.

## 3. Reported, not ruled

- **ST2's whole matches**, per match in clip-id order, then how many are flagged:

      @M 7.37,7.86,7.43,7.53,7.79,7.79,7.55,7.71,9.93,7.21,implausible 1 :: .venv/bin/python -c "import json;m=json.load(open('model/validation/out/n28_results.json'))['match'];print(*[round(m[s]['median'],2) for s in sorted(m)], 'implausible', sum(m[s]['implausible'] for s in m))"

  Nine sit where a professional league's players do. 132831's feed is flagged, and its speed
  field says its typical player runs faster than any footballer
  (`st2-132877-feed-is-mirrored` (internal note)). That was seen before the protocol, so it shows
  the check can catch a broken ruler; it cannot test the rule.
- **SoccerTrack v1's GNSS**, a positioning system with no camera in it
  (`soccertrack-v1-gnss.md`): players, median PSV-99, share above the
  league's maximum, and its georeference's RMS and scale:

      @M players 22,6.82,0.0,0.134,1.0037,0.9983 :: .venv/bin/python -c "import json;g=json.load(open('model/validation/out/n28_results.json'))['gnss'];print('players',g['players'],round(g['median'],2),g['above_max'],round(g['georef_rms_m'],3),*[round(s,4) for s in g['georef_scale']])"

  **Corrected after review: this is a coincidence of summaries, not a validation.** It is a
  different session with different players. The gap reverses when ST2's truth is smoothed as the
  GNSS appears to be. And the GNSS failed its own check
  (`soccertrack-v1-gnss.md`).

## 4. What it shows

- **Withdrawn after review: "too small for the league's range to catch."** On the ruled medians the
  margin between the truth's typical player and the league's fastest is
  `@D 2.0715 :: 8.9722 - 6.9007`
  m/s, and the product's median uses
  `@D 0.8682 :: (8.6992 - 6.9007) / (8.9722 - 6.9007)`
  of it. But the product's median mixes spans: `@C 107 of 313 :: independent review 2026-09-14, re-run in session`
  of its tracks last under a minute, and PSV-99 rises with span in both sources. The median rule is blind here because the product's
  population is mostly short tracks, not because the over-read is small.
- **Corrected twice: N21's 10.58 m/s against the league.** This page first said that flag fails
  because 10.58 is a maximum set against a percentile, and that the app shows neither. Review found
  both wrong:
  - On N28's own tracks the median cap-lifted maximum per track is
    `@C 9.3347 m/s :: independent review 2026-09-14, re-computed in session`
    So most of the drop from 10.58 to 8.70 is a change of population: N21's figure is each player's
    maximum over every credited fragment, on nine clips.
  - A per-track maximum at what ships is still above the league's fastest.
  - The app does show a single-sample top speed on every row where the cap never binds
    (`stats._peak_kmh` refuses only where it binds):
    `@C 202 of 353 rows :: independent review 2026-09-14, re-counted in session`

  Review checks, not registered.
- **The share above the maximum is the candidate**, and N28 does not rule on it.

## 5. What it changes

- **No plausibility gate ships on the median rule.** On these clips it would pass the product as
  it ships.
- **A rule on the share of tracks above the league's maximum** needs its own protocol, on data this
  run did not read (whole-match product runs, or other footage), and it should compare spans that
  match.
- **The research page's §6 flag is corrected** to this result.
- **SoccerTrack v1's GNSS does not validate ST2's speeds.** That reading is withdrawn (§3).
