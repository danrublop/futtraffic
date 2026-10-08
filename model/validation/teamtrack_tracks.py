"""TeamTrack annotation CSV -> tracks.json.gz in IMAGE PIXELS, identity carried.

`measure_sigma.load_teamtrack` returns boxes per frame and throws the player
away, which is fine for a recall/error count and useless for a trajectory. This
keeps (TeamID, PlayerID) as the track id all the way through.

Carried, never inferred. A previous measurement rebuilt identity by looking up
(frame, x, y) and two players collided on one quantised key, inventing errors in
code that was working. The identity is in the CSV header; it comes from there.

Output is the shape `validate.load_tracks` already reads, with x/y as image
pixels (box bottom-centre), so it must be paired with the matching
`model/validation/calib_<clip>.json`.

Run: .venv/bin/python model/validation/teamtrack_tracks.py
"""
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model" / "engine"))
from paths import FOOTAGE2, VALIDATION      # noqa: E402

FPS = 30000 / 1001.0        # 29.97, the container rate of every soccer_top clip


def load(csv_path: Path, want_ball: bool = False) -> dict[str, list[dict]]:
    """{track_id: [{f, id, x, y, w, h, team, jersey}]}, x/y = box bottom-centre px.

    `want_ball` swaps which columns are read: the CSV carries the ball as a
    22nd "player" under TeamID/PlayerID = BALL, and the default path skips it
    because a ball is not a body and every downstream harness here counts
    bodies. It is the ONLY ball ground truth in the repo, so it is selectable
    rather than discarded - see model/validation/ball_eval.py.

    Ball rows differ in two ways, both deliberate. y is the box CENTRE, not the
    bottom edge, because a ball in flight has no ground point (`ball.Ball.xy`
    is a centre for the same reason). And w/h are carried, because the gate a
    ball evaluation needs is sized from the ball's own pixels.
    """
    rows = list(csv.reader(open(csv_path)))
    teams, players, attrs = rows[0][1:], rows[1][1:], rows[2][1:]
    cols: dict[tuple, dict] = {}
    for i, (t, p, at) in enumerate(zip(teams, players, attrs)):
        if not at or (p == "BALL") != want_ball:
            continue
        cols.setdefault((t, p), {})[at] = i

    tracks: dict[str, list[dict]] = {}
    for r in rows[3:]:
        if not r or not r[0].strip():
            continue
        try:
            fi = int(float(r[0]))
        except ValueError:
            continue
        vals = r[1:]
        for (team, player), c in cols.items():
            try:
                l = float(vals[c["bb_left"]]); t_ = float(vals[c["bb_top"]])
                w = float(vals[c["bb_width"]]); h = float(vals[c["bb_height"]])
            except (KeyError, ValueError, IndexError):
                continue                    # player off-frame: blank cells
            if w <= 0 or h <= 0:
                continue
            tid = f"{team}_{player}"
            row = {
                "f": fi, "id": tid,
                "team": {"0": "home", "1": "away"}.get(team, "unknown"),
                "jersey": int(player) if player.isdigit() else None,
                "x": l + w / 2.0, "y": t_ + h,
            }
            # w/h ON BOTH BRANCHES. They used to be ball-only, so a player row
            # carried no size at all and `r.get("h", 0.0)` read 0.0 for every
            # body in the corpus - which silently emptied N58's whole fit set
            # before it ran (review 2026-09-25). A box the loader HAS should not
            # be withheld because one caller happened not to need it.
            row["w"], row["h"] = w, h
            if want_ball:
                row["y"] = t_ + h / 2.0
            tracks.setdefault(tid, []).append(row)
    return tracks


def write(tracks: dict, out: Path) -> None:
    dets = [p for tr in tracks.values() for p in tr]
    with gzip.open(out, "wt") as f:
        json.dump({"fps": FPS, "detections": dets}, f)


def main() -> int:
    for ann in sorted(FOOTAGE2.glob("soccer_top__*__annotations__*.csv")):
        clip = ann.name.replace("annotations", "videos").replace(".csv", "")
        tr = load(ann)
        n = sum(len(v) for v in tr.values())
        frames = sorted({p["f"] for v in tr.values() for p in v})
        out = VALIDATION / f"{clip}.tracks.json.gz"
        write(tr, out)
        print(f"{clip}: {len(tr)} players, {n} boxes, frames "
              f"{frames[0]}-{frames[-1]} -> {out.name}")
    return 0


def _selfcheck() -> None:
    ann = sorted(FOOTAGE2.glob("soccer_top__*__annotations__*.csv"))
    if not ann:
        print("SKIP: no TeamTrack annotations")
        return
    tr = load(ann[0])
    assert 20 <= len(tr) <= 22, f"{len(tr)} players is not a football match"
    # Identity must be unique per frame: if two players ever share a track id on
    # one frame the ids were inferred, not carried.
    for tid, pts in tr.items():
        fs = [p["f"] for p in pts]
        assert len(fs) == len(set(fs)), f"{tid} has duplicate frames"
    # Ground points must sit inside a 4K frame and move like a footballer.
    xy = np.array([[p["x"], p["y"]] for p in next(iter(tr.values()))])
    assert xy[:, 0].min() > 0 and xy[:, 0].max() < 3840, xy[:, 0].min()
    assert xy[:, 1].min() > 0 and xy[:, 1].max() < 2160, xy[:, 1].max()
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    assert np.median(step) < 10, f"{np.median(step)} px/frame is not a player"
    print("teamtrack_tracks.py ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
