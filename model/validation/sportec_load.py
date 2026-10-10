"""The Sportec/DFL open set: a second venue, and the one number it may not score.

WHY A SECOND VENUE AT ALL
    `model-map (not published)` §3 ends: *"Every downstream number is on 118575 half 1,
    and that match sits in the ST2 train split. Nothing downstream is held out."*
    The Sportec/DFL open set is seven matches of elite tracking and events under
    CC BY 4.0 (`research-data-interchange (not published)` §4), on the Hugging Face Hub
    as `pysport/idsse-data`, and it is the only candidate second venue on the
    table that is annotated and commercially licensed.

THE TRAP, FIRST, BECAUSE IT DISQUALIFIES THE OBVIOUS EXPERIMENT
    Our vendored PathCRF checkpoints were TRAINED ON THIS DATASET
    (`licence-audit` (not published) §2.4: upstream's own README, "Data Preparation",
    Sportec Open DFL, figshare 28196177). **Any PathCRF number taken here is
    void** and would look like a large improvement over
    `api://product.possession.f1.contact`. Do not score `model/engine/possession.py`
    on this data. This module deliberately does not import it.

    AND THAT VOIDS N14 AS WRITTEN, which is the correction this file exists to
    make. `research-data-interchange (not published)` §4 and §5.3 nominate N14, team
    possession share, as the number to re-measure here "because it needs
    positions and events and nothing else". It does not. `model-map (not published)`
    §5's N14 row names its system under test in its own last column:
    **"PathCRF's real product output"** - and `model/engine/possession.py::share` is
    the repo's ONLY producer of team possession share. `model/engine/stats.py:1692`
    refuses on purpose to relabel `territory()` as possession. So N14 is a
    PathCRF measurement end to end, and on this dataset it is in-sample.

    N14 IS NOT REACHABLE IN THIS VENUE. What is reachable, and what this file
    produces, is N14's REFERENCE AND ITS FLOOR: what the truth is, what the
    live/dead mask is worth, and what score a constant 50/50 already gets. Those
    are properties of the DATA, not of a model, so training on it cannot
    contaminate them.

THE PROTOCOL, fixed before anything ran
    Population   one match, `DFL-MAT-J03WMX` (DFL-COM-000001), both halves,
                 EVERY ball frame in DFL_04_03_positions_raw_observed at 25 Hz.
                 One match and not seven: the positions XML is 418 MB and this
                 machine is at 16 GiB with four agents live (§ MEMORY below).
    Split        none to state, and that is the point - this is a measurement OF
                 the reference data, not a model score, so there is nothing to
                 hold out. What must be stated instead: this match is not in ST2
                 and not in the product's eval set, and IS in PathCRF's training
                 set. n=1 configuration, no sweep.
    Held fixed   the file as published, no resampling, no smoothing, no
                 interpolation. `BallStatus` and `BallPossession` as the DFL
                 wrote them.
    Metrics      alive_share          in-play frames / all ball frames
                 share_home_alive     home-possession frames / in-play frames
                 share_home_all       home-possession frames / all frames
                 mask_bias_pp         |share_home_all - share_home_alive|, i.e.
                                      what NOT having a live/dead signal costs a
                                      share estimator. `model/engine/possession.py`
                                      GATES its share output on exactly this and
                                      quotes -1.12 pp with a mask / +6.97 pp
                                      without, measured on 118575.
    THE FLOOR    a constant 50/50 predictor, which needs no model, no tracking
                 and no ball:  floor_pp = |share_home_alive - 50|. A team-share
                 number without this beside it is `model-map (not published)` §0.4 again
                 - the possession headline that lost to a metronome. The
                 metronome cannot score a share; this can.
    Drop share   nothing is filtered. The one exclusion is `BallStatus != 1` in
                 `share_home_alive`, and its size is `1 - alive_share`, printed.
    Pass rule    this run's job is to establish that the data is FIT to be N14's
                 reference. It FAILS if any of:
                   a) a ball frame is missing `BallStatus`
                   b) `BallPossession` takes any value outside {1, 2} - see the
                      kloppy note below, where that silently reads as "away"
                   c) a half's ball track has a HOLE - last - first + 1 is not
                      the frame count. A gap there silently shrinks the
                      denominator of every share below it. (This replaces a
                      pass rule drafted as "the spans disagree with the
                      match-info periods": DFL_02_01_matchinformation carries no
                      period frames at all, so there was nothing to disagree
                      with. kloppy derives its periods FROM these same FrameSets,
                      so checking against it would be checking the file against
                      itself.)
                   d) alive_share falls outside [0.50, 0.95], which would mean
                      the field means something other than ball-in-play
                 A failing result is reported as a failing result. It does not
                 become a smaller population.

A DEFAULT TO KNOW ABOUT IN KLOPPY, NOT A BUG WE HIT
    `sportec/deserializer.py:396` reads
    `home_team if int(ball_data.get(BALL_POSSESSION, 0)) == 1 else away_team`.
    A ball frame with no `BallPossession`, or with any third value, is therefore
    handed to the AWAY team with no warning - the shape of `lint_truth` check 1.
    Pass rule (b) above exists to find out whether it bites on this file, rather
    than to assume it does not.

MEMORY, because the answer decided the design
    kloppy's Sportec tracking deserializer materialises the whole match -
    `frames[period][frame_id][object_id]` for every object of every frame, then
    a `Frame` with 22 `PlayerData` for each - and `limit` is the only bound on
    it. 135k frames x 23 objects is not safe next to three other lanes. So the
    ball track, which is all the possession reference needs, is read here by a
    bounded `iterparse` that holds one `Frame` element at a time; kloppy is used
    for the metadata, the events (776 KB) and any BOUNDED tracking window, which
    is what it is good at.

        @M 200,players per frame 22 :: .venv/bin/python model/validation/sportec_load.py --kloppy-window 200
           held fixed: DFL-MAT-J03WMX positions through
           `kloppy.sportec.load_tracking(limit=200, coordinates='sportec')`.
           Frames returned and players per frame - 22, flat, every frame, which
           is what elite tracking actually looks like and is the contrast case
           for `model/validation/kloppy_export.py`'s presence flag. Needs
           ~/Downloads/idsse-data (see DATA below); same precedent as the ST2
           tags, which need ~/Downloads/st2.

THE RESULT, AND IT MOVES TWO THINGS IN `model/engine/possession.py`

        @M 0.5849,0.6679,0.5068,45.81,5.27,4.19 :: .venv/bin/python model/validation/sportec_load.py
           held fixed: DFL-MAT-J03WMX, both halves, all 145,967 ball frames at
           25 Hz, file as published, nothing filtered. n=1 configuration.
           Values: ball-in-play share over the match, then first half, then
           second half; home possession share of LIVE frames in pp; the mask
           bias in pp; and the constant-50/50 floor in pp. All four pass rules
           are computed by the same command and printed as `fails`, which is
           `[]` on this match. Needs ~/Downloads/idsse-data - `--fetch` pulls
           it, ~419 MB; same precedent as the ST2 tags needing ~/Downloads/st2.

    1. **"A half is only ~73% ball-in-play" is not true here, and it carries no
       sigil.** `model/engine/possession.py:21` and `:307` state it twice with no
       source; on an elite Bundesliga match it is **0.585**, and the two halves
       of that one match are 0.668 and 0.507. A 16 pp swing between halves means
       the mask is not a constant that can be assumed at all, in either
       direction. The claim needs a tag or deletion (docs/README.md, Numbers), and
       whoever owns `model/engine/` should do that - this lane does not write there.

    2. **The gate's premise survives, measured somewhere else.** `share`'s
       docstring quotes +6.97 pp of bias from dead time on 118575 GT. On this
       match, a different venue, a different vendor and outside our eval set,
       the same quantity is **5.27 pp** over the match and 6.38 pp over the first
       half. An independently measured bias of the same size and sign is the
       best evidence the refusal in `share` is right.

        @D 5.27 > 4.19 :: mask_bias_pp vs floor_5050_pp, both from the @M above
           And this is the sentence N14 has to live with: on this match the bias
           from having NO live/dead signal is LARGER than the entire error of
           predicting a flat 50/50 and going home. An unmasked share estimator,
           however good its model, cannot beat the floor. It does not hold
           everywhere - in the first half the floor is 8.87 pp against a 6.38 pp
           bias - which is itself the point: whether a possession model beats
           doing nothing depends on the half, so the floor travels with the
           number or the number is not reportable.

WHAT THIS VENUE CAN AND CANNOT SCORE
    CANNOT   anything PathCRF (in-sample); anything needing video - there is no
             footage, so no detector, tracker, calibration or team-assignment
             number can be taken here at all.
    CAN      distance definitions, formation, event timing against a real event
             feed, the ball-in-play mask, and the reference + floor below.

DATA
    ~/Downloads/idsse-data, three XML files per match, CC BY 4.0, from
    https://huggingface.co/datasets/pysport/idsse-data - Bassek et al.,
    *Scientific Data* (2025), doi 10.1038/s41597-025-04505-y. Never inside the
    repo; same convention as ~/Downloads/st2 and ~/Downloads/skillcorner-opendata.

        .venv/bin/python model/validation/sportec_load.py --fetch      # ~419 MB, one match
        .venv/bin/python model/validation/sportec_load.py              # the reference
        .venv/bin/python model/validation/sportec_load.py --selfcheck
"""

import argparse
import io
import json
import os
import subprocess
import sys

DATA = os.path.expanduser("~/Downloads/idsse-data")
HF = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main"
# One match, the DFL-COM-000001 pair. The other six are listed in the HF sibling
# file list; pulling them is 6 x 419 MB and nothing yet needs them.
MATCH = "DFL-COM-000001_DFL-MAT-J03WMX"

KINDS = {"info": "DFL_02_01_matchinformation",
         "events": "DFL_03_02_events_raw",
         "positions": "DFL_04_03_positions_raw_observed"}

# The DFL's own encoding, from a ball Frame element:
#   BallStatus="1"      in play
#   BallPossession="1"  home    "2"  away
ALIVE, HOME, AWAY = "1", "1", "2"


def path(kind: str, match: str = MATCH) -> str:
    return os.path.join(DATA, f"{KINDS[kind]}_{match}.xml")


def fetch(match: str = MATCH) -> None:
    """One match, ~419 MB. Curl rather than the hub client: no new dependency,
    and `-C -` resumes rather than restarting a 400 MB download."""
    os.makedirs(DATA, exist_ok=True)
    for kind in KINDS:
        p = path(kind, match)
        print(f"-> {os.path.basename(p)}", flush=True)
        subprocess.run(["curl", "-sL", "-C", "-", "-o", p,
                        f"{HF}/{os.path.basename(p)}"], check=True)
    for kind in KINDS:
        print(f"{os.path.getsize(path(kind, match)) / 1e6:9.1f} MB  {KINDS[kind]}")


def ball_track(positions_path: str):
    """Stream the BALL FrameSets only. Yields (period, frame_id, status, possession).

    Bounded memory by construction: one `Frame` element is live at a time and
    its parent is truncated behind it. The whole 418 MB file is still WALKED -
    lxml has to reach every FrameSet - but nothing accumulates.
    """
    from lxml import etree
    ctx = etree.iterparse(positions_path, events=("start", "end"), huge_tree=True)
    period, in_ball = None, False
    for ev, el in ctx:
        if ev == "start" and el.tag == "FrameSet":
            in_ball = el.get("TeamId") == "BALL"
            period = el.get("GameSection") if in_ball else None
        elif ev == "start" and el.tag == "Frame" and in_ball:
            yield (period, int(el.get("N")), el.get("BallStatus"),
                   el.get("BallPossession"))
        elif ev == "end" and el.tag in ("Frame", "FrameSet"):
            el.clear()
            while el.getprevious() is not None:
                del el.getparent()[0]


def possession_reference(rows=None) -> dict:
    """N14's reference and its floor. See THE PROTOCOL in the module docstring.

    `rows` is anything yielding `ball_track`'s tuples; it defaults to the real
    match. Taking the stream rather than the path is what lets `_selfcheck` run
    THIS counter over a synthetic file instead of a second copy of it -
    lint_truth check 2 exists because a check that recomputes its subject is two
    implementations agreeing with each other.

    Returns per-half and overall rows. Raises nothing - a pass-rule violation
    comes back in `fails`, because a reference that fails its own fitness test
    is a finding and not an exception.
    """
    rows = ball_track(path("positions")) if rows is None else rows
    halves: dict[str, dict] = {}
    bad_possession: dict[str, int] = {}
    missing_status = 0
    for period, fid, status, poss in rows:
        h = halves.setdefault(period, {"frames": 0, "alive": 0, "home_alive": 0,
                                       "home_all": 0, "first": fid, "last": fid})
        h["frames"] += 1
        h["last"] = fid
        if status is None:
            missing_status += 1
        if poss not in (HOME, AWAY):
            bad_possession[str(poss)] = bad_possession.get(str(poss), 0) + 1
        h["home_all"] += poss == HOME
        if status == ALIVE:
            h["alive"] += 1
            h["home_alive"] += poss == HOME
    return _summarise(halves, missing_status, bad_possession)


def _summarise(halves: dict, missing_status: int, bad_possession: dict) -> dict:
    out, tot = {}, {"frames": 0, "alive": 0, "home_alive": 0, "home_all": 0}
    for name, h in halves.items():
        for k in tot:
            tot[k] += h[k]
        out[name] = _row(h)
    out["match"] = _row(tot)
    out["match"]["fails"] = f = []
    if missing_status:
        f.append(f"(a) {missing_status} ball frames with no BallStatus")
    if bad_possession:
        f.append(f"(b) BallPossession outside {{1,2}}: {bad_possession} "
                 f"- kloppy reads every one of these as AWAY")
    for name, h in halves.items():
        span = h["last"] - h["first"] + 1
        if span != h["frames"]:
            f.append(f"(c) {name} ball track has a hole: frames {h['frames']} "
                     f"over a span of {span}")
    if not 0.50 <= out["match"]["alive_share"] <= 0.95:
        f.append(f"(d) alive_share {out['match']['alive_share']} outside [0.50, 0.95]")
    return out


def _row(h: dict) -> dict:
    n, a = h["frames"], h["alive"]
    r = {"frames": n, "alive_frames": a,
         "alive_share": round(a / n, 4) if n else None,
         "dead_share": round(1 - a / n, 4) if n else None,
         "share_home_alive_pp": round(100 * h["home_alive"] / a, 2) if a else None,
         "share_home_all_pp": round(100 * h["home_all"] / n, 2) if n else None}
    if r["share_home_alive_pp"] is not None:
        r["mask_bias_pp"] = round(abs(r["share_home_all_pp"] - r["share_home_alive_pp"]), 2)
        # THE FLOOR. What a constant 50/50 already scores, in the same units any
        # future N14 would report its error in.
        r["floor_5050_pp"] = round(abs(r["share_home_alive_pp"] - 50.0), 2)
    if "first" in h:
        r["first_frame"], r["last_frame"] = h["first"], h["last"]
    return r


def load_events(match: str = MATCH, **kw):
    """kloppy's own Sportec event reader. 776 KB, so no memory question."""
    from kloppy import sportec
    return sportec.load_event(event_data=path("events", match),
                              meta_data=path("info", match), **kw)


def load_tracking(match: str = MATCH, limit: int = 2000, **kw):
    """kloppy's Sportec tracking reader, BOUNDED. `limit` is not a convenience
    here, it is the memory bound - see MEMORY in the module docstring. Passing
    limit=None on a 418 MB file is the thing this signature exists to discourage.
    """
    if not limit:
        raise ValueError(
            "limit=None materialises the whole match; see MEMORY in this "
            "module's docstring. Pass an explicit frame budget.")
    from kloppy import sportec
    return sportec.load_tracking(meta_data=path("info", match),
                                 raw_data=path("positions", match),
                                 limit=limit, coordinates="sportec", **kw)


def _selfcheck() -> None:
    """Synthetic ball XML with the answer written down here, not read off a run.

    Three things are asserted, and the middle one is the point: a BallPossession
    value kloppy would silently fold into AWAY must come back as a NAMED failure
    rather than as a slightly different share.
    """
    from lxml import etree

    def xml(frames):
        return ('<PutDataRequest><Positions>'
                '<FrameSet GameSection="firstHalf" TeamId="referee" PersonId="r1">'
                '<Frame N="1" X="0" Y="0"/></FrameSet>'
                '<FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="b1">'
                + "".join(frames) + '</FrameSet></Positions></PutDataRequest>')

    def fr(n, status="1", poss="1"):
        p = f' BallPossession="{poss}"' if poss is not None else ""
        return f'<Frame N="{n}" X="0" Y="0" BallStatus="{status}"{p}/>'

    # 10 frames: 8 alive (6 home, 2 away), 2 dead and BOTH home. So
    #   alive_share 0.8, share_home_alive 75.00 pp, share_home_all 80.00 pp,
    #   mask_bias 5.00 pp, floor 25.00 pp.  Written before running.
    frames = ([fr(i, "1", "1") for i in range(6)]
              + [fr(i, "1", "2") for i in range(6, 8)]
              + [fr(i, "0", "1") for i in range(8, 10)])
    buf = io.BytesIO(xml(frames).encode())
    got = list(ball_track(buf))
    assert len(got) == 10 and got[0] == ("firstHalf", 0, "1", "1"), got[:2]
    assert all(g[0] == "firstHalf" for g in got), "a referee frameset leaked in"

    r = possession_reference(got)["match"]
    assert r["alive_share"] == 0.8, r
    assert r["share_home_alive_pp"] == 75.0, r
    assert r["share_home_all_pp"] == 80.0, r
    assert r["mask_bias_pp"] == 5.0, r
    assert r["floor_5050_pp"] == 25.0, r
    assert r["fails"] == [], r

    # Pass rule (b): a third possession value, and a missing one. Both are the
    # kloppy default's blind spot, so both must be NAMED, not absorbed.
    buf = io.BytesIO(xml(frames[:8] + [fr(8, "0", "0"), fr(9, "0", None)]).encode())
    r = possession_reference(ball_track(buf))["match"]
    assert len(r["fails"]) == 1 and r["fails"][0].startswith("(b)"), r["fails"]
    assert "AWAY" in r["fails"][0]

    # Pass rule (c): a hole in the ball track shrinks every denominator under
    # it, so it must be named rather than divided by.
    buf = io.BytesIO(xml([fr(i) for i in range(6)]
                         + [fr(i) for i in (10, 11, 12)]).encode())
    r = possession_reference(ball_track(buf))["match"]
    assert any(f.startswith("(c)") for f in r["fails"]), r["fails"]
    assert "9 over a span of 13" in " ".join(r["fails"]), r["fails"]

    # Pass rule (d): an implausible alive_share is a failure, not a footnote.
    buf = io.BytesIO(xml([fr(i, "0", "1") for i in range(10)]).encode())
    r = possession_reference(ball_track(buf))["match"]
    assert any(f.startswith("(d)") for f in r["fails"]), r["fails"]

    # And the guard that keeps the memory decision honest.
    try:
        load_tracking(limit=0)
    except ValueError as e:
        assert "materialises" in str(e)
    else:
        raise AssertionError("load_tracking must refuse an unbounded load")

    print("sportec_load selfcheck OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fetch", action="store_true", help="pull one match, ~419 MB")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--kloppy-window", type=int, default=0,
                    help="frames to read through kloppy's own reader, then report "
                         "the shape it returns")
    a = ap.parse_args()
    if a.selfcheck:
        return _selfcheck()
    if a.fetch:
        return fetch()
    if not os.path.exists(path("positions")):
        sys.exit(f"no {path('positions')} - run with --fetch first")
    if a.kloppy_window:
        ds = load_tracking(limit=a.kloppy_window)
        per = {len(f.players_data) for f in ds.records}
        print(f"kloppy frames {len(ds.records)}  players per frame "
              f"{min(per)}..{max(per)}  periods "
              f"{sorted({f.period.id for f in ds.records})}")
        return
    print(json.dumps(possession_reference(), indent=2))


if __name__ == "__main__":
    main()
