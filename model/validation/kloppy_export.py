"""Our tracks -> kloppy, carrying the one flag kloppy's own model throws away.

WHY THIS EXISTS
    `research-data-interchange (not published)` §1 measured what kloppy does to a
    dataset full of holes: `PlayerData` exposes `coordinates`, `distance`,
    `speed`, `other_data` and nothing else, so SkillCorner's `is_detected` -
    the flag separating a measured position from an extrapolated one - is gone
    the moment the file is read. 19.8 raw player rows a frame, 56.7% of them
    detected, come back as a flat 22.0 players a frame. A coverage number
    computed through kloppy reads 100%.

    Our output has the same property and worse. `api://product.distance.coverage`
    is 0.70, a track id is a fragment and not a player, and a shipped tracks file
    holds a detection for about a fifth of its (id, frame) grid. Serialising that
    into any dense interchange format means INVENTING the other four fifths.
    Doing it without a per-player-frame flag is `on_pitch` again
    (`model/validation/verify_claims.py`, WORKED EXAMPLE): the hard rows disappear and
    the output looks better than the input.

    So the flag is the deliverable and the serialisation is plumbing.

    What that costs on a real file, which is the reason to bother:

        @M 32464,16395,162000,0.0729 :: .venv/bin/python model/validation/kloppy_export.py model/validation/out/stc2025/118575.p05.tracks.json model/validation/out/stc2025/118575.p05.calibration.json model/validation/out/l5_epts_118575_p05
           held fixed: N6 arm A - the product tracker, conf 0.5, stride 2, the
           UI's clicked calibration - on 118575 half 1, 3,000 stride-2 frames,
           54 home/away track ids. Observed player-frames, exporter-FILLED
           player-frames, the size of the dense grid, and the share of source
           rows DROPPED because kloppy's EPTS metadata cannot hold a third team
           (officials). Needs model/validation/out/, which is gitignored and is
           rebuilt by `st2_product.py --track`; same precedent as the ST2 tags.

        @D 0.336 :: 16395 / (32464 + 16395)   [the @M above]
           Of every player-frame this file offers a position for, a THIRD is
           interpolation. Read that next to §1's SkillCorner figure: their file
           is 56.7% detected and kloppy returns a flat 22.0 players a frame. We
           are worse and the flag is the only thing that says so.

WHAT IT WRITES, AND WHY EPTS
    kloppy has no tracking-data *writer* at all - `SportsCodeSerializer` (code
    data) is the only serializer in the package. So "loadable by kloppy" means
    writing a format one of its ten tracking deserializers reads. Of those ten,
    `metrica_epts` is THE ONLY ONE that populates `PlayerData.other_data`; the
    other nine hard-code `other_data={}`, skillcorner and sportec included. That
    is not a preference, it is the only door:

        @M fill_other_data 1,providers 10 :: .venv/bin/python -c "import pathlib,re; p=pathlib.Path('.venv/lib/python3.14/site-packages/kloppy/infra/serializers/tracking'); f=[x for x in p.rglob('*.py') if 'other_data' in x.read_text()]; print('fill_other_data',sum(1 for x in f if re.search(r'other_data.update|other_data=other_data', x.read_text())), 'providers', len([d for d in p.iterdir() if d.name not in ('__init__.py','__pycache__','deserializer.py')]))"
           held fixed: kloppy 3.19.0 as installed in .venv on 2026-09-12.
           Deserializer modules that ever put a value into `other_data`, and
           tracking provider modules in the package. One of ten. Needs kloppy
           installed - it is validation-only and therefore sits in
           app/server/requirements.txt's "Deliberately NOT here" block, so this tag does
           not reproduce from a bare `pip install -r app/server/requirements.txt`.

    So: Metrica EPTS. Two files, `metadata.xml` + `raw.txt`, and a third sensor
    beside `position` whose channel is `presence`. kloppy's EPTS deserializer
    copies any sensor that is not position/distance/speed straight into
    `other_data` keyed by sensor id, so `player_data.other_data["presence"]`
    comes back on every player of every frame.

THE PROTOCOL FOR THE SELF-CHECK (`--selfcheck`), fixed before it ran
    Population   one synthetic 2-team, 4-track, 11-frame tracks dict with three
                 gaps built by hand: an interior 1-frame hole, an interior
                 3-frame hole, and a track that starts late and ends early.
                 Synthetic ON PURPOSE - the assert is about the SHAPE of the
                 round trip, and a real file cannot state its own answer.
    Held fixed   kloppy 3.19.0, an identity-ish homography, 25 fps, stride 1.
    Metric       set equality, not a score. Three sets must match exactly:
                 observed (id, frame) pairs in, presence==1 pairs out, and the
                 inverse.
    Floor        the trap arm. Filled rows must carry FINITE coordinates, i.e.
                 be indistinguishable from observed ones on `coordinates` alone.
                 If a reader could recover the gaps without the flag the assert
                 proves nothing, so the check asserts it CANNOT.
    Pass rule    all four asserts. A FAILING result is: any observed pair coming
                 back without presence==1, any filled pair coming back with it,
                 `other_data` arriving empty (the §1 failure, reproduced by us),
                 or a filled row arriving as NaN (which would make the flag
                 decoration).
    n=1          one configuration, no sweep. Nothing here is tuned.

    `to_epts` re-reads its own output through kloppy on every export and raises
    if the flag did not survive. A promised guard absent from the code has cost
    this repo its headline twice (docs-guard-must-have-a-runnable-check (internal note)),
    so the writer refuses rather than the docstring promising.

WHAT THIS EXPORT IS NOT
    * Not 22 players. It writes one kloppy `Player` per TRACK ID. 118575.p05
      has 56 of them for a 22-player half, 54 after the officials are cut. A
      consumer that counts players gets 54.
      `ProviderPlayerParameters/fut_track_id` says which fragment each is.
    * Not refereed. kloppy's EPTS metadata reader builds its team map from the
      `Score` element's two ids, so a third team is silently dropped - `Ground.REFEREE`
      exists in the domain model but cannot be reached through this file format.
      Tracks labelled `official` or `unknown` are therefore DROPPED, and the
      drop share is returned and printed. On 118575.p05 it is 7.3% of rows.
    * Not calibrated away. Coordinates are our own homography's metres, divided
      by pitch length/width into EPTS normalised 0..1. Every metre inherits
      `api://calib.product.rms` and everything in docs/ledgers/ledger-claims.md about it.
    * Not Metrica. `ProviderName` is `fut-traffic`, so kloppy warns "The Provider
      is not known to Kloppy" and applies no coordinate transform. That warning
      is correct and we are not going to silence it by claiming to be Metrica.

USAGE
    .venv/bin/python model/validation/kloppy_export.py --selfcheck
    .venv/bin/python model/validation/kloppy_export.py \
        model/validation/out/stc2025/118575.p05.tracks.json \
        model/validation/out/stc2025/118575.p05.calibration.json \
        model/validation/out/l5_epts_118575_p05 --limit 200
"""

import argparse
import json
import os
import sys
from xml.sax.saxutils import escape

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "model", "engine"))

# Only these two reach kloppy. `official` and `unknown` cannot be represented
# (see WHAT THIS EXPORT IS NOT) and are counted out loud instead.
EXPORTED_TEAMS = ("home", "away")


def _grid(dets: list[dict]) -> tuple[list[int], dict[int, dict[int, tuple]]]:
    """Sampled frames, and {track id: {frame: (x_px, y_px)}}."""
    frames = sorted({int(d["f"]) for d in dets})
    by_id: dict[int, dict[int, tuple]] = {}
    for d in dets:
        by_id.setdefault(int(d["id"]), {})[int(d["f"])] = (float(d["x"]), float(d["y"]))
    return frames, by_id


def _fill(frames: list[int], obs: dict[int, tuple]) -> dict[int, tuple]:
    """Linear fill of INTERIOR gaps only. Returns {frame: (x, y)} for the filled
    frames alone - never for an observed one, and never outside the track's own
    first..last span, because extrapolating a track that has not started is not
    inference, it is fabrication.

    ponytail: a straight line across a 40 s gap is garbage. It is emitted anyway
    and flagged presence=0; a consumer that wants only real rows filters on the
    flag, which is the whole point of shipping it. Add a max-gap cut here only
    if someone measures that filtering is not enough.
    """
    if len(obs) < 2:
        return {}
    ks = sorted(obs)
    lo, hi = ks[0], ks[-1]
    out, j = {}, 0
    for f in frames:
        if f in obs or not (lo < f < hi):
            continue
        while ks[j + 1] < f:
            j += 1
        a, b = ks[j], ks[j + 1]
        t = (f - a) / (b - a)
        out[f] = tuple(obs[a][k] + t * (obs[b][k] - obs[a][k]) for k in (0, 1))
    return out


def _metadata_xml(players: list[dict], frames: list[int], fps: float,
                  length: float, width: float) -> str:
    """EPTS metadata. Three channels a player: x, y, presence.

    `presence` is its own Sensor because kloppy keys `other_data` by SENSOR id
    and reads the value through that sensor's FIRST channel - see
    metrica_epts/deserializer.py:55-59. A fourth channel on the position sensor
    would be parsed and thrown away.
    """
    def pp(p):
        return (f'<ProviderPlayerParameters>'
                f'<ProviderParameter><Name>fut_track_id</Name><Value>{p["tid"]}</Value></ProviderParameter>'
                f'<ProviderParameter><Name>fut_jersey_known</Name><Value>{int(p["jersey"] is not None)}</Value></ProviderParameter>'
                f'</ProviderPlayerParameters>')

    teams = "".join(
        f'<Team id="{t}"><Name>{t}</Name>'
        f'<ProviderTeamsParameters><ProviderParameter>'
        f'<Name>attack_direction_first_half</Name><Value>left_to_right</Value>'
        f'</ProviderParameter></ProviderTeamsParameters></Team>'
        for t in EXPORTED_TEAMS)
    player_elms = "".join(
        f'<Player id="{p["pid"]}" teamId="{p["team"]}">'
        f'<Name>{escape(p["name"])}</Name><ShirtNumber>{p["shirt"]}</ShirtNumber>'
        f'{pp(p)}</Player>' for p in players)
    chans = "".join(
        f'<PlayerChannel id="{p["pid"]}_{c}" playerId="{p["pid"]}" channelId="{c}"/>'
        for p in players for c in ("x", "y", "presence"))
    refs = "".join(
        f'<SplitRegister separator=",">'
        + "".join(f'<PlayerChannelRef playerChannelId="{p["pid"]}_{c}"/>'
                  for c in ("x", "y", "presence"))
        + '</SplitRegister>' for p in players)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<EPTS>\n'
        '<Metadata>'
        '<GlobalConfig>'
        '<FileFormat>1.4</FileFormat>'
        '<ProviderName>fut-traffic</ProviderName>'
        f'<FrameRate>{int(round(fps))}</FrameRate>'
        '<ProviderGlobalParameters>'
        f'<ProviderParameter><Name>first_half_start</Name><Value>{frames[0]}</Value></ProviderParameter>'
        f'<ProviderParameter><Name>first_half_end</Name><Value>{frames[-1]}</Value></ProviderParameter>'
        '</ProviderGlobalParameters>'
        '</GlobalConfig>'
        '<Sessions><Session id="s1">'
        '<MatchParameters><Score idLocalTeam="home" idVisitingTeam="away">'
        '<LocalTeamScore>0</LocalTeamScore><VisitingTeamScore>0</VisitingTeamScore>'
        '</Score></MatchParameters>'
        f'<FieldSize><Width>{int(round(length))}</Width><Height>{int(round(width))}</Height></FieldSize>'
        '</Session></Sessions>'
        f'<Teams>{teams}</Teams>'
        f'<Players>{player_elms}</Players>'
        '<Devices><Device id="d1"><Name>fut-traffic</Name><Sensors>'
        '<Sensor id="position"><Name>position</Name><Channels>'
        '<Channel id="x"><Name>x</Name><Unit>normalized</Unit></Channel>'
        '<Channel id="y"><Name>y</Name><Unit>normalized</Unit></Channel>'
        '</Channels></Sensor>'
        '<Sensor id="presence"><Name>presence</Name><Channels>'
        '<Channel id="presence"><Name>presence</Name><Unit>boolean</Unit></Channel>'
        '</Channels></Sensor>'
        '</Sensors></Device></Devices>'
        f'<PlayerChannels>{chans}</PlayerChannels>'
        '</Metadata>\n'
        f'<DataFormatSpecifications><DataFormatSpecification '
        f'startFrame="{frames[0]}" endFrame="{frames[-1]}" separator=":">'
        '<StringRegister name="frameCount"/>'
        f'<SplitRegister separator=";">{refs}</SplitRegister>'
        '<SplitRegister separator=","><BallChannelRef channelId="x"/>'
        '<BallChannelRef channelId="y"/></SplitRegister>'
        '</DataFormatSpecification></DataFormatSpecifications>\n'
        '</EPTS>\n')


def to_epts(tracks: dict, cal, out_dir: str, limit: int | None = None) -> dict:
    """Write `out_dir/metadata.xml` + `out_dir/raw.txt`, then PROVE the flag
    survived by reading them back through kloppy. Returns the shares.

    Raises if a single observed (id, frame) comes back without presence==1, or a
    single filled one comes back with it. That is not belt-and-braces: this
    function's only reason to exist is that flag, and a serializer that quietly
    stops carrying it is worth less than no serializer at all.
    """
    dets = [d for d in tracks["detections"] if d.get("team") in EXPORTED_TEAMS]
    dropped = len(tracks["detections"]) - len(dets)
    if not dets:
        raise ValueError("no home/away detections to export")

    frames, by_id = _grid(dets)
    if limit:
        frames = frames[:limit]
        keep = set(frames)
        by_id = {t: {f: v for f, v in o.items() if f in keep} for t, o in by_id.items()}
        by_id = {t: o for t, o in by_id.items() if o}

    team_of, jersey_of = {}, {}
    for d in dets:
        team_of.setdefault(int(d["id"]), d["team"])
        if d.get("jersey") is not None:
            jersey_of.setdefault(int(d["id"]), d["jersey"])

    players = [{"tid": t, "pid": f"t{t}", "team": team_of[t],
                "name": f"track {t}", "jersey": jersey_of.get(t),
                "shirt": int(jersey_of.get(t, t))}
               for t in sorted(by_id) if team_of[t] in EXPORTED_TEAMS]

    # Pixels -> metres in ONE homography call per track, not one per row.
    obs_m: dict[int, dict[int, tuple]] = {}
    for p in players:
        o = by_id[p["tid"]]
        fs = sorted(o)
        m = cal.to_pitch(np.array([o[f] for f in fs], dtype=np.float64))
        obs_m[p["tid"]] = {f: (float(m[i][0]), float(m[i][1])) for i, f in enumerate(fs)}

    filled = {p["tid"]: _fill(frames, obs_m[p["tid"]]) for p in players}
    L, W = float(cal.pitch.length), float(cal.pitch.width)

    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "metadata.xml")
    raw_path = os.path.join(out_dir, "raw.txt")
    with open(meta_path, "w") as fh:
        fh.write(_metadata_xml(players, frames, float(tracks["fps"]), L, W))

    n_obs = n_fill = 0
    with open(raw_path, "w") as fh:
        for f in frames:
            cells = []
            for p in players:
                t = p["tid"]
                xy, pres = obs_m[t].get(f), 1
                if xy is None:
                    xy, pres = filled[t].get(f), 0
                if xy is None:
                    cells.append(",,0")          # not tracked at all -> NaN, 0
                else:
                    cells.append(f"{xy[0] / L:.6f},{xy[1] / W:.6f},{pres}")
                    n_obs += pres
                    n_fill += 1 - pres
            fh.write(f"{f}:" + ";".join(cells) + ":,\n")

    grid = len(frames) * len(players)
    stats = {"frames": len(frames), "players": len(players),
             "observed": n_obs, "filled": n_fill, "grid": grid,
             "observed_share": round(n_obs / grid, 4),
             "filled_share": round(n_fill / grid, 4),
             "dropped_rows": dropped,
             "dropped_row_share": round(dropped / len(tracks["detections"]), 4),
             "meta": meta_path, "raw": raw_path}
    _assert_flag_survives(out_dir, obs_m, filled, frames)
    return stats


def load_epts(out_dir: str, **kw):
    """Read an export back. Thin on purpose: it is kloppy's own reader, so a
    round trip that passes here is a round trip anyone else's tooling gets."""
    from kloppy import metrica
    return metrica.load_tracking_epts(
        meta_data=os.path.join(out_dir, "metadata.xml"),
        raw_data=os.path.join(out_dir, "raw.txt"), **kw)


def presence_sets(ds) -> tuple[set, set]:
    """(observed, not-observed) as {(track_id, frame_id)} read back out of kloppy.

    The flag is read from `other_data`, which is the only place kloppy's model
    has room for it. If a future kloppy drops `other_data` from `PlayerData`
    this raises rather than returning two plausible sets.
    """
    seen, unseen = set(), set()
    for fr in ds.records:
        for pl, pd in fr.players_data.items():
            if "presence" not in pd.other_data:
                raise RuntimeError(
                    f"presence flag gone for {pl.player_id} at frame {fr.frame_id}: "
                    f"other_data={pd.other_data!r}. This is exactly the failure "
                    "docs/research/research-data-interchange.md §1 measured on SkillCorner.")
            key = (int(pl.attributes["fut_track_id"]), int(fr.frame_id))
            (seen if pd.other_data["presence"] == 1 else unseen).add(key)
    return seen, unseen


def _assert_flag_survives(out_dir, obs_m, filled, frames) -> None:
    ds = load_epts(out_dir)
    seen, unseen = presence_sets(ds)
    want_seen = {(t, f) for t, o in obs_m.items() for f in o if f in set(frames)}
    want_fill = {(t, o) for t, d in filled.items() for o in d}
    assert seen == want_seen, (
        f"presence==1 set differs from the detections we wrote: "
        f"{len(seen)} vs {len(want_seen)}, "
        f"+{len(seen - want_seen)} / -{len(want_seen - seen)}")
    assert want_fill <= unseen, (
        f"{len(want_fill - unseen)} interpolated rows came back marked observed")


def _selfcheck() -> None:
    """See THE PROTOCOL FOR THE SELF-CHECK in the module docstring."""
    import tempfile
    import warnings
    from pitch import Calibration, Pitch

    # A camera looking straight down, 10 px per metre: pixels/10 are metres.
    cal = Calibration(pitch=Pitch(), H=np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1.0]]))

    # Four tracks over eleven frames, with the holes written down here so the
    # assert has something to be right ABOUT.
    frames = list(range(11))
    holes = {1: {5}, 2: {4, 5, 6}, 3: set()}          # interior gaps, by id
    late = {3: range(3, 8)}                            # id 3 starts late, ends early
    dets = []
    for tid, team in ((0, "home"), (1, "home"), (2, "away"), (3, "away"),
                      (9, "official")):
        span = late.get(tid, frames)
        for f in span:
            if f in holes.get(tid, set()):
                continue
            dets.append({"f": f, "id": tid, "x": 100.0 + 5 * tid + f,
                         "y": 200.0 + f, "w": 10.0, "h": 30.0,
                         "team": team, "jersey": None})
    tracks = {"fps": 25.0, "detections": dets}

    with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
        warnings.simplefilter("ignore")               # "Provider is not known to Kloppy"
        st = to_epts(tracks, cal, d)                  # raises if the flag is lost
        ds = load_epts(d)
        seen, unseen = presence_sets(ds)

        # 1. the flag is exactly our detections, both directions.
        want = {(int(x["id"]), int(x["f"])) for x in dets if x["team"] != "official"}
        assert seen == want, f"{len(seen)} observed vs {len(want)} written"
        assert not (seen & unseen)

        # 2. the holes we built are present and marked not-observed.
        assert (1, 5) in unseen and {(2, f) for f in (4, 5, 6)} <= unseen
        assert (3, 0) in unseen and (3, 9) in unseen, "off-span frames must appear"

        # 3. THE TRAP ARM. An interior filled row must be FINITE - i.e. it is
        #    indistinguishable from an observed one on coordinates alone, so the
        #    flag is the only thing carrying the distinction. If this ever fails
        #    the assert above has become free and proves nothing.
        by_f = {fr.frame_id: fr for fr in ds.records}
        pmap = {int(p.attributes["fut_track_id"]): p
                for p in ds.metadata.teams[0].players + ds.metadata.teams[1].players}
        c = by_f[5].players_data[pmap[1]].coordinates
        assert c is not None and np.isfinite(c.x) and np.isfinite(c.y), \
            "a filled row must look real, or the flag is decoration"
        # ...and it must be the right fill: id 1 is linear in f, so frame 5 is
        # the midpoint of frames 4 and 6. Checked against the SOURCE pixels
        # through the same homography, not against a second copy of _fill.
        want_x = cal.to_pitch(np.array([[105.0 + 5, 205.0]]))[0][0] / cal.pitch.length
        assert abs(c.x - want_x) < 1e-6, (c.x, want_x)

        # 4. off-span rows are NaN, so "never tracked" and "tracked but filled"
        #    are still separable downstream.
        c0 = by_f[0].players_data[pmap[3]].coordinates
        assert c0 is None or not np.isfinite(c0.x), "off-span must not be invented"

        # 5. the plumbing: ids, frame count, and an observed coordinate survive.
        assert len(ds.records) == 11, len(ds.records)
        assert set(pmap) == {0, 1, 2, 3}, set(pmap)
        assert st["dropped_rows"] == 11 and st["dropped_row_share"] > 0, st
        c1 = by_f[0].players_data[pmap[0]].coordinates
        assert abs(c1.x - 10.0 / cal.pitch.length) < 1e-6, c1

    print("kloppy_export selfcheck OK  "
          f"observed={st['observed']} filled={st['filled']} "
          f"grid={st['grid']} dropped_rows={st['dropped_rows']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tracks", nargs="?")
    ap.add_argument("calibration", nargs="?")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N sampled frames only")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck or not a.tracks:
        return _selfcheck()

    from pitch import Calibration
    with open(a.tracks) as fh:
        tracks = json.load(fh)
    st = to_epts(tracks, Calibration.load(a.calibration), a.out_dir, limit=a.limit)
    print(json.dumps(st, indent=2))


if __name__ == "__main__":
    main()
