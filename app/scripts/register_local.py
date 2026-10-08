#!/usr/bin/env python3
"""Register a local video file as a match, without uploading it.

WHY THIS EXISTS. Every match in the store arrived from a feed scrape and carries
`"video": null`, so `/api/matches/{id}/video` 404s for all of them and the film
editor has never played a frame. The HTTP API (`POST /api/matches` + `/upload`)
would work, but it COPIES the bytes into the job dir, and the one real match on
this box is a 3.3 GB panorama. This symlinks instead.

    python app/scripts/register_local.py data/footage/118575_panorama_1st_half.mp4 \
        --events data/soccertrack_v2/bas__118575__118575_12_class_events.json \
        --tracks data/jobs/118575_full_1st/tracks.final.json \
        --name "SoccerTrack 118575, 1st half" --half 1

    python app/scripts/register_local.py path/to/panorama.mp4 \
        --name "My panorama"

    python app/scripts/register_local.py --selfcheck

The events file is SoccerTrack v2's BAS 12-class action list. `position` is
milliseconds from kickoff and lines up with the video: for 118575 half 1 the
actions span 0.64 s to 2871.28 s against a 2874.0 s file.

`--tracks` WARNING, learned the hard way on 118575. The overlay draws image
pixels of the video you register, so the tracks must be in THAT video's space.
`data/jobs/118575_full_1st/tracks.{raw,final}.json` are NOT: they run x to 9873
on a 4096-wide file, because the tracker ran on the undistorted panorama.
`calibration.json` says so itself - `projection_type: remap`, `maps_scale: 2.9`,
`note: "points are in REMAPPED space; apply undistort_frame first"`. Point those
at this script and the film editor correctly refuses to draw and says the
overlay is not registered to the video. To exercise the overlay for real,
undistort a short slice of footage first and register that.

ponytail: a symlink, not a copy. The cost is that the job is only as portable as
the target - `data/footage/118575_panorama_1st_half.mp4` is itself a symlink into
~/Downloads, so a registered match does not survive being moved to another
machine. Upgrade path if that matters: copy the bytes, or put the footage on a
shared volume.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "server"))

import jobs  # noqa: E402

# BAS gives `team: "left" | "right"`. Which end a side attacks from is not which
# side is at home, and nothing in the file says. So the mapping is positional and
# the names say so - inventing "Northbridge FC" here would be a fabricated fixture.
SIDE = {"left": "home", "right": "away"}


def bas_to_plays(actions: list[dict], half: int) -> list[dict]:
    """SoccerTrack BAS actions -> the `plays` shape `lib/moments.ts` reads.

        {"gameTime": "1 - 12:04", "label": "PASS",       ──▶  {"t": 724.0,
         "position": "724000", "team": "left",                 "type": "pass",
         "player_id": "476362"}                                "side": "home",
                                                               "players": []}

    NO SHIRT NUMBERS. `Play.players` is `number[]` and the UI renders each one
    as `#N`, which reads as a shirt. SoccerTrack gives opaque player ids, so
    minting ordinals here would put `#3` on a row where nobody knows the shirt.
    The id goes in `note` instead, where it cannot be mistaken for one.
    """
    out = []
    prefix = f"{half} -"
    for i, a in enumerate(actions):
        if not str(a.get("gameTime", "")).startswith(prefix):
            continue
        try:
            t = int(a["position"]) / 1000.0
        except (KeyError, TypeError, ValueError):
            continue  # a row with no usable timecode is not placeable
        label = str(a.get("label") or "").strip()
        if not label:
            continue
        pid = a.get("player_id")
        out.append({
            "id": f"bas{i:05d}",
            "type": label.lower().replace(" ", "_"),
            "side": SIDE.get(str(a.get("team")), "home"),
            "period": half,
            "minute": int(t // 60),
            "clock": f"{int(t // 60)}:{int(t % 60):02d}",
            "zone": "",           # BAS carries no zone and none is invented
            "outcome": "",        # nor an outcome
            "players": [],
            "board": None,
            "t": round(t, 2),
            "note": f"soccertrack player {pid}" if pid else "",
            "source": "soccertrack-bas",
        })
    out.sort(key=lambda p: p["t"])
    return out


def already_registered(target: Path) -> str | None:
    """The job id already pointing at this file, if any.

    This is the real "ran it twice" case: `new_id()` is random, so a colliding
    job id is not the risk - a second match for the same footage is.
    """
    if not jobs.JOBS_DIR.exists():
        return None
    for jf in sorted(jobs.JOBS_DIR.glob("*/job.json")):
        try:
            st = json.loads(jf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        name = st.get("video")
        if not name:
            continue
        p = jf.parent / name
        try:
            if p.exists() and p.resolve() == target:
                return st.get("id")
        except OSError:
            continue
    return None


def register(video: Path, events: Path | None, tracks: Path | None, name: str,
             team: str, half: int, force: bool) -> str:
    target = video.resolve(strict=True)
    dupe = already_registered(target)
    if dupe and not force:
        raise SystemExit(
            f"already registered as {dupe} — pass --force to add a second copy")

    jid = jobs.new_id()
    d = jobs.JOBS_DIR / jid
    d.mkdir(parents=True)
    ext = target.suffix or ".mp4"
    os.symlink(target, d / f"video{ext}")

    # Symlinked too, and for the same reason: a pipeline run's tracks.json is
    # tens to hundreds of MB. The route serves the file unmodified and
    # public/tracks-worker.js streams exactly this shape, so the CLI's own
    # output can be pointed at directly with no conversion.
    if tracks:
        os.symlink(tracks.resolve(strict=True), d / "tracks.json")

    st = {
        "id": jid,
        "status": "done",
        "error": None,
        "created": time.time(),
        "updated": time.time(),
        "pid": None,
        "attempts": 0,
        "video": f"video{ext}",
        "display_name": name or target.stem,
        "team": team or jobs.DEFAULT_TEAM,
        "venue": "",
        "bytes": target.stat().st_size,
        "total_bytes": target.stat().st_size,
        "upload_complete": True,
        "operator_seconds": 0.0,
        "corrections": 0,
    }
    jobs.save(st)

    if events:
        actions = (json.loads(events.read_text()) or {}).get("actions") or []
        plays = bas_to_plays(actions, half)
        (d / "match.json").write_text(json.dumps({
            "meta": {
                # Positional, because BAS says left/right and nothing names the
                # clubs. A fabricated fixture would be worse than a plain one.
                "home_name": "Left",
                "away_name": "Right",
                "score": None,
                "status": "finished",
                "fps": 25.0,
                "source": str(events),
                "provenance": "SoccerTrack v2 BAS 12-class actions, CC BY 4.0",
                "facts": {"plays": plays},
            },
        }, indent=1))
        print(f"{len(plays)} plays from half {half}")

    return jid


def selfcheck() -> None:
    """python app/scripts/register_local.py --selfcheck — asserts, no framework.

    Matches the convention `python app/server/jobs.py selfcheck` already uses.
    """
    a = [
        {"gameTime": "1 - 0:00", "position": "640", "label": "PASS",
         "team": "left", "player_id": "476362"},
        {"gameTime": "1 - 0:05", "position": "5720", "label": "HIGH PASS",
         "team": "right", "player_id": "476356"},
        {"gameTime": "2 - 93:50", "position": "5630920", "label": "PASS",
         "team": "right", "player_id": "476355"},
        {"gameTime": "1 - 0:02", "position": "nonsense", "label": "SHOT",
         "team": "left"},
        {"gameTime": "1 - 0:03", "position": "3000", "label": "",
         "team": "left"},
    ]

    p = bas_to_plays(a, half=1)
    assert len(p) == 2, f"half filter + bad rows dropped: {len(p)}"
    assert [x["t"] for x in p] == [0.64, 5.72], p
    assert p[0]["side"] == "home" and p[1]["side"] == "away"
    assert p[1]["type"] == "high_pass", p[1]["type"]      # UI turns _ back into a space
    assert p[0]["players"] == [], "never mint a shirt number"
    assert "476362" in p[0]["note"]
    assert p[0]["clock"] == "0:00" and p[1]["clock"] == "0:05"

    h2 = bas_to_plays(a, half=2)
    assert len(h2) == 1 and h2[0]["period"] == 2, h2
    assert h2[0]["minute"] == 93, h2[0]["minute"]

    assert bas_to_plays([], half=1) == []
    # Sorted by time even when the source is not. Ids are NOT compared: they
    # carry the source row index, so they are stable for a given file and
    # deliberately different for a reordered one.
    assert [x["t"] for x in bas_to_plays(list(reversed(a)), half=1)] == [0.64, 5.72]

    print("selfcheck ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", type=Path)
    ap.add_argument("--events", type=Path, help="SoccerTrack BAS actions json")
    ap.add_argument("--tracks", type=Path,
                    help="a pipeline tracks.json, so the overlay has positions to draw")
    ap.add_argument("--name", default="")
    ap.add_argument("--team", default="")
    ap.add_argument("--half", type=int, default=1, choices=(1, 2))
    ap.add_argument("--force", action="store_true",
                    help="register again even if this file is already a match")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()
    if not args.video:
        ap.error("a video path is required (or --selfcheck)")

    jid = register(args.video, args.events, args.tracks, args.name, args.team,
                   args.half, args.force)
    print(f"registered {jid}")
    print(f"  http://localhost:3000/workspace?v=film&m={jid}")


if __name__ == "__main__":
    main()
