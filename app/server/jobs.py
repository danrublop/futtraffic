"""File-backed job queue for the video pipeline.

No Redis, no Celery, no database. A match job is hours long and there is at
most one GPU, so the queue depth that matters is 1 and the only coordination
problem worth solving is "the box rebooted halfway through".

Restart safety comes from the stage artifacts, not from the state file:
`tracks.raw.json` / `match.json` existing on disk IS the record that a stage
finished. A resumed job re-checks what is already there and skips it. The
JSON state file is a cache of that plus the things the filesystem cannot tell
us - error text, attempt count, operator minutes.

Liveness: the worker touches `alive` every few seconds. A job is running only
if its pid answers AND that file is fresh. Pid alone is not enough after a
reboot, when the number has been handed to something unrelated.

    python jobs.py selfcheck     # asserts, no test framework
    python jobs.py run <job_id>  # worker entry point (used by the parent)
"""
from __future__ import annotations

import _path  # noqa: F401  - puts model/engine/ on sys.path

import datetime as _dt
import fcntl
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from html import unescape
from pathlib import Path

# Outside the web root on purpose: nothing here is ever served as a STATIC file.
# A mount over this directory would bypass job_dir()'s id check, which is the
# only thing standing between a client-supplied id and path traversal.
#
# NARROWED 2026-08-04: the video IS now reachable, at GET /api/matches/{jid}/video,
# because a review page that cannot play the uploaded match is not a product.
# That route goes through job_dir() like everything else. The invariant that
# actually matters is unchanged - no static mount, every path id-validated -
# but "never reachable by URL" was the old wording and is no longer true.
#
# Worth knowing: a 32-hex uuid4 id is the ONLY thing protecting a match, there
# is no auth anywhere in app.py, and GET /api/matches lists every id on the box.
# Single-tenant on localhost that is fine. Hosted for more than one club it is
# not, and this is the line to revisit first.
from paths import DATA as _DEFAULT_DATA

DATA_DIR = Path(os.environ.get("FUTTRAFFIC_DATA", _DEFAULT_DATA))
JOBS_DIR = DATA_DIR / "jobs"
ROSTER_PATH = DATA_DIR / "roster.json"

ID_RE = re.compile(r"^[0-9a-f]{32}$")
# Jobs written before teams existed carry no team. They are defaulted on READ
# rather than migrated: rewriting every job.json on the box to add one string is
# a batch job that can half-run, and a torn job.json loses a match.
DEFAULT_TEAM = "Default"
HEARTBEAT_S = 5
DEAD_AFTER_S = 45
MAX_ATTEMPTS = 2          # one retry, then it fails loudly instead of looping
STALE_UPLOAD_S = 3600     # an upload nobody finished is a failure, not a queue entry

RUNNING = {"tracking", "identifying"}
TERMINAL = {"done", "failed"}
# "calibrating" is deliberately none of the above: it means the job is blocked
# on a human clicking pitch landmarks. That is a legitimate parked state, and
# the status payload says so out loud so it can never be mistaken for a hang.
#
# It is now the FIRST gate, not the last. The calibration is an input to
# tracking (track.py's pitch mask + player-height prior throw out ~half the
# frame before the tracker sees it) and to identity (identity.py merges in
# metres; in pixels MAX_SPEED rejects every merge). Tracking first and asking
# for landmarks afterwards spent 2-4 GPU-hours producing tracks that neither
# stage could use.

# Progress is stage-based. track.py owns the frame loop and has no callback,
# so a real percentage would mean editing a file this module does not own.
# ponytail: coarse progress; wire a callback into track.run when someone
# actually wants a moving bar rather than a stage name.
# calibrating is 0.02, not 0.9: it is now the FIRST gate, so a job parked
# there has run nothing. Reporting 90% for "waiting on a human before any
# GPU work" is the kind of progress bar that destroys trust in every other
# number on the page.
PROGRESS = {"queued": 0.0, "calibrating": 0.02, "tracking": 0.05,
            "identifying": 0.75, "done": 1.0, "failed": 0.0}


# ---------------------------------------------------------------- state I/O

def new_id() -> str:
    return uuid.uuid4().hex


def job_dir(jid: str) -> Path:
    """Every path in this module routes through here, which is the only place
    a job id gets validated. A client-supplied id is otherwise a path
    traversal waiting to happen."""
    if not ID_RE.match(jid or ""):
        raise ValueError(f"bad job id: {jid!r}")
    return JOBS_DIR / jid


def _slug(s: str | None) -> str:
    """The comparison key for a team or a venue. `roster._slug`, capped.

    Two people typing "Westfield Park" and "westfield park" mean one ground, and
    since venue equality is now a SAFETY condition on inherited calibrations,
    comparing the raw strings would silently re-open the bug this closes.

    This used to be its own regex, WITHOUT the NFKD fold, which is the exact
    implementation roster._slug's docstring exists to warn against: it gave
    "Jose Park" -> jose-park but "José Park" -> jos-park, so an operator who
    typed the accent one week and not the next had two venues and no inherited
    calibration. The same file was already inconsistent with itself about it -
    the roster FILENAME is written through the folding slug (`R._slug`, below)
    and the DB team id through `O._slug`, while the team filter and this safety
    gate used the stripping one. Accented club names are the norm in the market
    this is aimed at.
    """
    import roster as R  # noqa: PLC0415 - lazy, as everywhere else in this file

    return R._slug(s or "")[:60]


def _defaults(st: dict) -> dict:
    """Fields added after jobs already existed on disk. Read-time, not a
    migration - see DEFAULT_TEAM."""
    st.setdefault("team", DEFAULT_TEAM)
    st.setdefault("venue", "")
    return st


def load(jid: str) -> dict:
    with open(job_dir(jid) / "job.json") as f:
        return _defaults(json.load(f))


# Keys a HUMAN writes through the API while a worker may hold a stale copy of the job for hours.
# save() writes the caller's whole dict, so without this a worker's routine save reverted the
# operator's "manual" calibration flag and dropped the typed pitch size (review, 2026-09-17).
OPERATOR_KEYS = ("pitch", "calibration_source")


def own(st: dict, key: str, value) -> None:
    """Set an operator-owned key so that the next save() writes it. Anyone else's save() keeps the
    value on disk for these keys, whatever their in-memory copy says."""
    if value is None:
        st.pop(key, None)
    else:
        st[key] = value
    st.setdefault("_owned", []).append(key)


def save(st: dict) -> None:
    d = job_dir(st["id"])
    owned = set(st.pop("_owned", ()))
    with open(d / ".job.lock", "w") as lk:                      # the read-merge-write below must not interleave with another writer
        fcntl.flock(lk, fcntl.LOCK_EX)
        _save_locked(st, d, owned)


def _save_locked(st: dict, d, owned: set) -> None:
    if (d / "job.json").exists():
        try:
            with open(d / "job.json") as f:
                disk = json.load(f)
        except Exception:  # noqa: BLE001 - a torn file is replaced, not merged
            disk = {}
        for k in OPERATOR_KEYS:
            if k in owned:
                continue
            if k in disk:
                st[k] = disk[k]
            else:
                st.pop(k, None)
    st["updated"] = time.time()
    tmp = d / "job.json.tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, d / "job.json")  # atomic: a torn state file loses a match


def _f(v) -> float | None:
    """float(v), preserving None. A refused measurement is not a zero."""
    return None if v is None else float(v)


@functools.lru_cache(maxsize=256)
def _season_rows_cached(path: str, mtime: float) -> tuple:
    """(id, team, jersey, minutes, distance_km, top_speed) per player.

    No sprints: they are WITHHELD (stats.CAVEATS["sprints_withheld"]), and a
    match.json written before 2026-09-28 still carries a count that must not be summed.

    Keyed on mtime so a re-derive invalidates it. Cached for the same reason
    `_fixture_cached` is, only more so: `season()` reads EVERY finished match,
    and match.json carries the heatmaps - a 30-match season is tens of MB of
    JSON re-parsed on every request, to pull seven numbers per player out of it.
    """
    try:
        players = json.loads(Path(path).read_text()).get("players") or []
    except (json.JSONDecodeError, OSError):
        return ()
    out = []
    for p in players:
        try:
            tid = int(p.get("id"))
        except (TypeError, ValueError):
            continue  # a row with no usable track id can never be bound
        out.append((tid, p.get("team"), p.get("jersey"),
                    float(p.get("minutes") or 0), float(p.get("distance_km") or 0),
                    # NOT `or 0`. stats._peak_kmh returns None where the clamp
                    # bound, and its docstring is explicit that this is a
                    # REFUSAL, not a damped reading - 34.2 km/h reads as an
                    # elite sprint while meaning "this track teleported", so it
                    # publishes nothing instead. Coercing it here turned 168 of
                    # 361 rows on the one real match into a confident 0.0 km/h
                    # by the time they reached the season table. db.py:490 gets
                    # this right on the same file ("if p.get(k) is not None"),
                    # which is what made the two ingest paths disagree.
                    _f(p.get("top_speed_kmh"))))
    return tuple(out)


@functools.lru_cache(maxsize=256)
def _fixture_cached(path: str, mtime: float) -> dict | None:
    """meta -> {home, away, score}. Keyed on mtime so a rewrite invalidates.

    ponytail: cached because the matches list polls and match.json carries the
    heatmaps - parsing a few hundred KB per job per poll to read three strings
    is the kind of thing nobody notices until there are fifty matches. Drop the
    cache if the pipeline ever writes the fixture into job.json instead.
    """
    try:
        meta = json.loads(Path(path).read_text()).get("meta") or {}
    except (json.JSONDecodeError, OSError):
        return None
    home, away = meta.get("home_name"), meta.get("away_name")
    if not home and not away:
        return None
    return {"home": home, "away": away, "score": meta.get("score") or None}


def _fixture(path) -> dict | None:
    try:
        return _fixture_cached(str(path), path.stat().st_mtime)
    except OSError:
        return None


def all_jobs(team: str | None = None) -> list[dict]:
    """Every job, or only one team's.

    A team is a FIELD, not a directory. Nesting jobs under a team would rewrite
    every job_dir() call site in this module and in app.py plus need a migration
    of the jobs already on disk, for behaviour nobody can see from the outside.
    Flat and slug-keyed is how rosters already work (_roster_path).
    """
    want = _slug(team) if team else None
    out = []
    for p in sorted(JOBS_DIR.glob("*/job.json")) if JOBS_DIR.exists() else []:
        try:
            st = _defaults(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            continue  # half-written or hand-deleted; the pump ignores it
        if want is not None and _slug(st["team"]) != want:
            continue
        out.append(st)
    return out


def create(display_name: str, total_bytes: int | None, ext: str,
           venue: str = "", team: str = "") -> dict:
    jid = new_id()
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    (JOBS_DIR / jid).mkdir()
    st = {
        "id": jid,
        "status": "queued",
        "error": None,
        "created": time.time(),
        "updated": time.time(),
        "pid": None,
        "attempts": 0,
        "video": f"video{ext}",
        "display_name": display_name,       # for the UI only - never a path
        # Which club this match belongs to, and which ground it was filmed at.
        # The venue is not decoration: it is half of the condition under which
        # another match's homography may be reused here.
        "team": str(team or "").strip()[:60] or DEFAULT_TEAM,
        "venue": str(venue or "").strip()[:80],
        "bytes": 0,
        "total_bytes": total_bytes,
        "upload_complete": False,
        "operator_seconds": 0.0,            # accumulates across correction sessions
        "corrections": 0,
    }
    save(st)
    return st


# ------------------------------------------------------ calibration inherited

def _video_shape(path: Path) -> tuple[int, int] | None:
    """(width, height) of a video, or None if it cannot be opened."""
    try:
        import cv2  # noqa: PLC0415 - heavy, and only this path needs it
    except Exception:  # noqa: BLE001
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None
    wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return wh if wh[0] and wh[1] else None


# -------------------------------------------------------------- video stills
# Shared by app.py's /frame.jpg route and the agent's get_frame tool. The cv2
# code lived only inside the route, so the agent would have had to duplicate a
# decoder path to show the model a frame - two copies of one decoder is how
# they drift. Raises ValueError with an operator-readable message; the callers
# decide what status code or tool_error that becomes.


def frame_jpeg(jid: str, f: int = 0, w: int = 0) -> bytes:
    """One JPEG still from a job's video. `f` is a frame number, `w` an optional
    downscale width."""
    import cv2  # noqa: PLC0415 - heavy, and only this path needs it

    st = load(jid)
    # A job can carry NO video at all - a report imported straight from a feed
    # never had one - and `job_dir(jid) / None` is a TypeError, not the
    # ValueError the route knows how to turn into a 404. So every match card
    # for such a job answered its thumbnail with a 500 and a traceback.
    if not st.get("video"):
        raise ValueError("no video uploaded yet")
    path = job_dir(jid) / st["video"]
    if not path.exists():
        raise ValueError("no video uploaded yet")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("video is not decodable - the upload may be incomplete")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if f > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(f, max(total - 1, 0))))
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"could not read frame {f} (video has {total})")
    if w and 0 < w < img.shape[1]:
        img = cv2.resize(img, (w, round(img.shape[0] * w / img.shape[1])))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("jpeg encode failed")
    return buf.tobytes()


THUMB_W_MAX = 480
# ~1 minute in at 25fps. Frame 0 on match footage is routinely a black lead-in
# or a static empty pitch; a minute in is play. frame_jpeg clamps to the last
# frame, so a 30-second clip lands on its own end rather than failing.
THUMB_F = 1500


def thumb_path(jid: str, f: int, w: int) -> Path:
    """Where a match card's still is cached, keyed on the VIDEO's mtime.

    THE MTIME IS THE INVALIDATION. A re-uploaded video is a new mtime, so it is
    a new key, so the stale jpeg is simply never asked for again. No sweep, no
    `thumb_mtime` field in clips.json to keep in step with anything.

        jobs/<id>/thumbs/<video mtime>_<frame>_<width>.jpg

    Raises ValueError (which the route maps to 404) for a job that carries no
    video at all - a report imported straight from a feed never had one. Same
    guard as frame_jpeg, and for the same reason: `job_dir(jid) / None` is a
    TypeError and a 500, not a 404.
    """
    st = load(jid)
    if not st.get("video"):
        raise ValueError("no video uploaded yet")
    v = job_dir(jid) / st["video"]
    if not v.exists():
        raise ValueError("no video uploaded yet")
    return job_dir(jid) / "thumbs" / f"{int(v.stat().st_mtime)}_{f}_{w}.jpg"


def video_seconds(jid: str) -> float | None:
    """Duration of a job's video in seconds, or None when it cannot be opened.
    The agent's create_clip clamps its out-point against this; None means
    "unknown", not "zero"."""
    import cv2  # noqa: PLC0415 - heavy

    try:
        st = load(jid)
    except (FileNotFoundError, ValueError):
        return None
    path = job_dir(jid) / st["video"]
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return (n / fps) if fps > 0 and n > 0 else None


def inherit_calibration(st: dict) -> bool:
    """Adopt the most recent hand-clicked calibration from the same rig, if one fits.

    SAFETY: frame size, ground AND club must all match. Frame size alone matched
    an away ground (most clubs film at one resolution) and silently rescaled every
    distance; a job with no venue never inherits. The copy is recorded as
    `calibration_inherited_from` so the UI can show it and offer a re-do.
    docs/reference/calibration.md#inheritance

    Returns True if one was adopted.
    """
    d = job_dir(st["id"])
    if (d / "calibration.json").exists():
        return False
    venue, team = _slug(st.get("venue")), _slug(st.get("team") or DEFAULT_TEAM)
    if not venue:
        return False        # cannot prove the ground matches, so do not guess
    shape = _video_shape(d / st["video"])
    if shape is None:
        return False        # cannot prove the geometry matches, so do not guess

    best: tuple[float, Path, str] | None = None
    for other in JOBS_DIR.iterdir():
        if not other.is_dir() or other.name == st["id"]:
            continue
        cal = other / "calibration.json"
        if not cal.exists():
            continue
        try:
            ost = _defaults(json.loads((other / "job.json").read_text()))
        except Exception:  # noqa: BLE001
            continue
        if ost.get("frame_shape") != list(shape):
            continue
        if _slug(ost["venue"]) != venue or _slug(ost["team"]) != team:
            continue
        # Never inherit an inherited one: chains let a single bad calibration
        # propagate across a whole season with nothing recording where it began.
        if ost.get("calibration_inherited_from"):
            continue
        # Never adopt another match's AUTOMATIC camera: when AUTOCAL is on, auto_calibrate has just tested
        # that very camera against this footage and refused it (or found no pitch at all); when
        # FUTTRAFFIC_AUTOCAL=0 no automatic camera is adopted at all, by design. A hand-clicked one is a human's.
        if ost.get("calibration_source") == "auto":
            continue
        ts = float(ost.get("updated") or 0.0)
        if best is None or ts > best[0]:
            best = (ts, cal, ost.get("display_name") or other.name[:8])
    if best is None:
        return False

    shutil.copyfile(best[1], d / "calibration.json")
    st["calibration_inherited_from"] = best[2]
    save(st)
    return True


# ------------------------------------------------------ calibration, no clicks

# FUTTRAFFIC_AUTOCAL=0 turns it off: the operator-click instance N30 froze its dots on must never be
# calibrated behind the operator's back, and neither should a test that wants a job to park.
# Order of the calibration paths: docs/reference/calibration.md#order
AUTOCAL = os.environ.get("FUTTRAFFIC_AUTOCAL", "1") != "0"
# ponytail: width/height is how a PANORAMA is recognised. autocal models one spherical camera and its
# search (autocal.BOUNDS) does not cover a single-lens 16:9 pinhole, which is left to the operator's dots.
# Upgrade: fit both families, keep the one whose lines close.
PANORAMA_ASPECT = 1.9


def _install_camera(st: dict, shape) -> tuple[list | None, str | None]:
    """The camera of the most recent match from the SAME rig that was calibrated with no clicks -
    same club, same ground, same frame: inherit_calibration's safety conditions, for the same reasons.
    It is only ever a START: model/engine/autocal.py registers it to THIS match's paint and accepts or
    refuses it on this footage, so a rig that was nudged is caught instead of inherited.
    docs/reference/calibration.md#install-camera"""
    venue, team = _slug(st.get("venue")), _slug(st.get("team") or DEFAULT_TEAM)
    if not venue:
        return None, None
    best = None
    for other in JOBS_DIR.iterdir():
        cal = other / "calibration.json"
        if not other.is_dir() or other.name == st["id"] or not cal.exists():
            continue
        try:
            ost = _defaults(json.loads((other / "job.json").read_text()))
            c = json.loads(cal.read_text())
        except Exception:  # noqa: BLE001
            continue
        if (c.get("projection_type") != "spherical" or ost.get("frame_shape") != list(shape)
                or _slug(ost["venue"]) != venue or _slug(ost["team"]) != team):
            continue
        ts = float(ost.get("updated") or 0.0)
        if best is None or ts > best[0]:
            best = (ts, c["camera"], ost.get("display_name") or other.name[:8])
    return (best[1], best[2]) if best else (None, None)


def venue_pitch(st: dict) -> dict | None:
    """The pitch size somebody GAVE for this ground: this job's own, else the most recent job's at the
    same club and ground. {"length": m or None, "width": m or None, "source": ...}. Never a default -
    a caller that gets None knows it is assuming."""
    own_ = st.get("pitch")
    if own_ and (own_.get("width") or own_.get("length")):
        return own_
    venue, team = _slug(st.get("venue")), _slug(st.get("team") or DEFAULT_TEAM)
    if not venue:
        return None
    best = _venue_pitches(_jobs_stamp()).get((venue, team))      # this job has no size of its own, so it is never `best`
    return dict(best[1], source=f"venue:{best[1].get('source', 'user')}") if best else None


def _jobs_stamp() -> tuple:
    """(name, mtime of job.json) for every job: the key under which the venue table is cached. status_view
    calls venue_pitch once per ROW of the match list; parsing every job.json per row made it quadratic.
    Stats are cheap; parsing is not. calibration-evidence (not published)#evidence-inheritance"""
    out = []
    for other in JOBS_DIR.iterdir():
        try:
            out.append((other.name, (other / "job.json").stat().st_mtime))
        except OSError:
            continue
    return tuple(sorted(out))


@functools.lru_cache(maxsize=4)
def _venue_pitches(stamp: tuple) -> dict:
    """{(venue, team): (updated, pitch)} - the most recent given pitch size per ground."""
    table = {}
    for name, _mt in stamp:
        try:
            ost = _defaults(json.loads((JOBS_DIR / name / "job.json").read_text()))
        except Exception:  # noqa: BLE001
            continue
        op = ost.get("pitch")
        if not op or not (op.get("width") or op.get("length")):
            continue
        key = (_slug(ost["venue"]), _slug(ost["team"]))
        ts = float(ost.get("updated") or 0.0)
        if key not in table or ts > table[key][0]:
            table[key] = (ts, op)
    return table


def auto_calibrate(st: dict) -> bool:
    """Find the camera from the video's own painted lines and write calibration.json. No clicks.

    model/engine/autocal.py: a player-free median frame, paint found by shape, a search over physical
    mounts, registration to the paint, and an acceptance test that uses no ground truth and REFUSES
    rather than guesses (docs/experiments/n34-zero-click.md). A refusal is recorded on the job with its reason
    and the job parks for the operator's dots exactly as it always has. Returns True if it calibrated.

    THE PITCH'S SIZE. A GIVEN length (venue_pitch) wins; otherwise it is measured from the paint.
    Width is never measured: venue_pitch, else the Pitch default, flagged `width_assumed`.
    docs/reference/calibration.md#pitch-size
    """
    d = job_dir(st["id"])
    video = d / st["video"]
    shape = _video_shape(video)
    if not AUTOCAL or shape is None or shape[0] < PANORAMA_ASPECT * shape[1]:
        return False
    import autocal  # noqa: PLC0415 - scipy-heavy
    import calibrators  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from pitch import Pitch  # noqa: PLC0415

    import dataclasses  # noqa: PLC0415
    given = venue_pitch(st) or {}
    pitch = Pitch(length=float(given.get("length") or Pitch().length), width=float(given.get("width") or Pitch().width))
    start, start_from = _install_camera(st, shape)
    try:
        return _auto_calibrate(st, d, video, pitch, given, start, start_from, autocal, calibrators, np, dataclasses)
    except Exception as e:  # noqa: BLE001 - calibration by hand is always still possible; never fail the job
        st["calibration_auto"] = {"accepted": False, "why": f"{type(e).__name__}: {e}"}
        save(st)
        return False


def _auto_calibrate(st, d, video, pitch, given, start, start_from, autocal, calibrators, np, dataclasses) -> bool:
    # A length a HUMAN typed wins (author's decision, 2026-09-28): fit at it, record the paint's. An OSM length
    # ("osm" / "venue:osm") only seeds the search. No source = typed: it predates the OSM lookup.
    fixed = bool(given.get("length")) and "osm" not in str(given.get("source") or "user")
    cam, rep = autocal.calibrate(str(video), pitch, install_camera=start, fix_length=fixed)
    summary = {k: rep.get(k) for k in ("accepted", "via", "why", "hits", "samples", "line_rms_px", "camera_centre_m", "f_px_per_rad")}
    if fixed:
        meas, sd = rep.get("length_measured_m"), rep.get("length_measured_sd_m")
        # ponytail: 0.5 m tolerance = autocal's fine length step (LENGTH_FINE); the fit's own sd if larger
        summary.update(length_given_m=pitch.length, length_measured_m=meas,
                       length_overruled=meas is not None and abs(meas - pitch.length) > max(sd or 0.0, 0.5))
    if start_from and rep.get("via") == "install":
        summary["install_from"] = start_from
    st["calibration_auto"] = summary
    if cam is None:
        save(st)
        return False
    pitch = dataclasses.replace(pitch, length=float(rep.get("length_m") or pitch.length))   # the camera is in THIS pitch's frame
    summary.update(length_m=pitch.length, length_sd_m=rep.get("length_sd_m"), width_m=pitch.width,
                   width_assumed=not given.get("width"), width_source=given.get("source") if given.get("width") else None)
    st["calibration_auto"] = summary
    cal = calibrators.SphericalCamera(pitch, cam, rep["image_wh"])
    # The fit's residual in metres at a typical place on the pitch: the line RMS through the map's
    # own metres-per-pixel. It is how well the drawing sits on the paint, NOT an accuracy at players.
    gx, gy = np.meshgrid(np.linspace(5, pitch.length - 5, 9), np.linspace(5, pitch.width - 5, 6))
    J = cal.jacobian(cal.to_image(np.column_stack([gx.ravel(), gy.ravel()])))
    cal.rms_error_m = float(rep["line_rms_px"] * np.median(np.linalg.svd(J, compute_uv=False)[:, 0]))
    X, _T, grp = autocal.model_samples(pitch, 1.0)
    px = cal.to_image(X)
    cal.auto = {**summary,
                "overlay_px": {g: np.round(px[grp == g], 1).tolist() for g in sorted(set(grp))}}
    tmp = d / "calibration.auto.json"          # not the API's tmp name: an operator may be posting right now
    tmp.write_text(json.dumps(cal.to_dict()))
    load_calibration(tmp)                      # the worker's own load: what is written is what is read
    # Create-if-absent in one step: a human who posted while this ran keeps theirs, and a POST that
    # lands between a check and a replace can no longer be overwritten. Not .job.lock: save() takes it.
    # ponytail: os.link needs a filesystem with hard links (APFS/ext4 do); on one without, it raises
    # and auto_calibrate records the refusal - the operator's clicks still work.
    try:
        os.link(tmp, d / "calibration.json")
    except FileExistsError:
        return True
    finally:
        tmp.unlink(missing_ok=True)
    own(st, "calibration_source", "auto")
    st.pop("calibration_inherited_from", None)
    save(st)
    return True


# ------------------------------------------------------- calibration contract

def load_calibration(path: str | Path):
    """THE load the pipeline performs. Nothing else may decide what is valid.

    app.py validates an uploaded calibration by calling this, so "accepted at
    POST" and "loadable by the worker" are one statement. track.run and
    stats.build_match load through calibrators.load too; keep them there.
    `panoramic` stays refused: its global H would give plausible wrong numbers.
    docs/reference/calibration.md#load-gate (history: #evidence-load-gate)
    """
    import math  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    from pitch import Calibration  # noqa: PLC0415 - pulls cv2/numpy

    with open(path) as f:
        d = json.load(f)
    ptype = d.get("projection_type", "perspective")
    # The accepted types and who writes each: docs/reference/calibration.md#files-by-type
    if ptype not in ("perspective", "remap", "perspective_tps", "perspective_fisheye", "spherical"):
        # pitch.Calibration.from_dict raises here too; this is the same refusal
        # phrased for the operator who just clicked the landmarks.
        raise ValueError(
            f"{ptype!r} calibration: this pipeline consumes perspective, "
            f"remap, perspective_tps, perspective_fisheye or spherical "
            f"projections. A stitched panorama must be exported as "
            f"'remap' (dense flatten, then one homography) - 'panoramic' is a "
            f"single global fit with ~6 m median position error, which would "
            f"produce plausible wrong distances instead of a loud failure.")
    # The SAME call track.py and stats.build_match make; Calibration.from_dict
    # raises on every type but perspective.
    import calibrators  # noqa: PLC0415
    cal = calibrators.load(str(path))
    # Check the capability, not the representation: what has to work is that it
    # projects an image point onto the pitch and lands somewhere real.
    xy = cal.to_pitch([[100.0, 100.0]])
    if len(xy) != 1 or not all(math.isfinite(float(v)) for v in xy[0]):
        raise ValueError("calibration does not project image points onto the pitch")
    # A degenerate fit is the one bad calibration that does not raise anywhere: findHomography
    # returns SOME H from collinear clicks, with a tiny rms, and every distance is wrong.
    # Check the clicks, not det(H) - det scales with pixel units and missed both selfcheck cases.
    # Inliers only: LMEDS can drop the one off-line click and leave a line behind.
    from pitch import resolve_landmarks  # noqa: PLC0415
    pts = d.get("points") or {}
    if pts:   # spherical/autocal and legacy files have none; the H test below still runs
        bad = set(d.get("outliers") or [])
        X = np.array(list(resolve_landmarks(cal.pitch, {n: 0 for n in pts if n not in bad}).values()), float).reshape(-1, 2)
        if len(X) < 4:
            raise ValueError(f"need at least 4 landmarks that fit, got {len(X)} - click more of the pitch")
        s = np.linalg.svd(X - X.mean(0), compute_uv=False)
        # ponytail: 0.05 = the click spread's thin axis must be >= 1/20 of its long axis. Goal line
        # plus one goal-box corner, the thinnest honest named set, passes; a line scores 0.
        if s[1] < 0.05 * s[0]:
            raise ValueError("degenerate calibration - the landmarks that fit lie on (nearly) one line; "
                             "click points off that line, e.g. a box corner")
    # Point-free files: only H to look at. Scale-free (Frobenius-normalised) so pixel units don't move it.
    s = np.linalg.svd(np.asarray(cal.H, float) / np.linalg.norm(cal.H), compute_uv=False)
    if s[-1] < 1e-10 * s[0]:
        raise ValueError("degenerate homography - the landmarks are collinear")
    return cal


def status_view(st: dict) -> dict:
    """What GET /api/matches/{id} returns."""
    d = job_dir(st["id"])
    waiting = "calibration" if st["status"] == "calibrating" else None
    return {
        "id": st["id"],
        "status": st["status"],
        "progress": PROGRESS.get(st["status"], 0.0),
        "error": st.get("error"),
        "waiting_for": waiting,
        "name": st.get("display_name"),
        "team": st.get("team") or DEFAULT_TEAM,
        # Shown next to the inherited-calibration note: "reused from last week"
        # only means anything if you can see it was the same ground.
        "venue": st.get("venue") or "",
        "uploaded_bytes": st.get("bytes", 0),
        "total_bytes": st.get("total_bytes"),
        "upload_complete": st.get("upload_complete", False),
        "attempts": st.get("attempts", 0),
        "created": st.get("created"),
        "updated": st.get("updated"),
        "elapsed_s": round(time.time() - st.get("stage_started", st["updated"]), 1)
        if st["status"] in RUNNING else None,
        "operator_seconds": round(st.get("operator_seconds", 0.0), 1),
        "corrections": st.get("corrections", 0),
        "has_calibration": (d / "calibration.json").exists(),
        "has_tracks": (d / "tracks.json").exists(),
        "has_match": (d / "match.json").exists(),
        # Who played and what it finished, for the matches list. Lives in
        # match.json rather than the job file, so it only exists once the
        # pipeline has run - None until then, and the list shows the upload name.
        "fixture": _fixture(d / "match.json"),
        # Set when this match adopted another match's calibration instead of
        # asking a human. Surfaced deliberately: an inherited homography that is
        # subtly wrong (the camera got nudged) shifts every distance on the
        # page, and the only way anyone suspects that is by being told the
        # calibration was not solved for this footage.
        "calibration_inherited_from": st.get("calibration_inherited_from"),
        # "auto" when the camera was found from the paint with no clicks; `calibration_auto` is that
        # attempt's own report, present on a REFUSAL too so the setup screen can say why it is asking.
        "calibration_source": st.get("calibration_source"),
        "calibration_auto": st.get("calibration_auto"),
        # The pitch size somebody GAVE for this match's ground (its own or the venue's); None when assumed.
        "pitch": venue_pitch(st),
        "frame_shape": st.get("frame_shape"),
        # Both of these used to live only in the job file, where nothing read
        # them. A stage that quietly degraded (identity crashed, geometry prior
        # never loaded) has to reach the results screen, or the product's
        # failure mode is a confident wrong number with no marking on it.
        "identity_note": st.get("identity_note"),
        # The match finished but the query store refused it: the report page
        # works (it reads match.json), season totals and the DB tab do not.
        "ingest_error": st.get("ingest_error"),
        "scene_prior": st.get("scene_prior"),
        "prior_rejected": st.get("prior_rejected"),
        # Which detector actually ran - CoreML when bundled, else PyTorch.
        "detector": st.get("detector"),
    }


# ------------------------------------------------------------------- liveness

def _alive(st: dict) -> bool:
    pid = st.get("pid")
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    hb = job_dir(st["id"]) / "alive"
    last = hb.stat().st_mtime if hb.exists() else 0
    # `updated` covers the window between spawn and the worker's first beat.
    return time.time() - max(last, st.get("updated", 0)) < DEAD_AFTER_S


def _heartbeat(d: Path, stop: threading.Event) -> None:
    while not stop.wait(HEARTBEAT_S):
        try:
            (d / "alive").touch()
        except OSError:
            return


# ----------------------------------------------------------------- the pump

def _fail(st: dict, msg: str) -> None:
    st["status"] = "failed"
    st["error"] = msg
    st["pid"] = None
    save(st)


def _spawn(st: dict) -> None:
    st["status"] = "tracking"          # the worker corrects this if it resumes later
    st["stage_started"] = time.time()
    st["error"] = None
    save(st)
    try:
        if getattr(sys, "frozen", False):
            # PyInstaller bundle: there is no jobs.py on disk. The frozen
            # binary (futtraffic-api, entry app.py) dispatches "run <id>" to
            # jobs.run() itself.
            cmd = [sys.executable, "run", st["id"]]
            cwd = None
        else:
            cmd = [sys.executable, str(Path(__file__).resolve()), "run", st["id"]]
            cwd = str(Path(__file__).resolve().parent)
        p = subprocess.Popen(
            cmd,
            cwd=cwd,
            env={**os.environ, "FUTTRAFFIC_DATA": str(DATA_DIR)},
            # Worker output goes to a file, NOT /dev/null. track.py prints the
            # scene-prior-unavailable warning and the sub-6m camera-height
            # warning that groundpoint.py shows can break the 5% distance gate.
            # Discarding those means a geometry prior can silently fail to load
            # and nothing anywhere records it - for a product whose failure mode
            # is a confident wrong number, that is the worst possible default.
            stdout=open(job_dir(st["id"]) / "worker.log", "ab"),
            stderr=subprocess.STDOUT,
            start_new_session=True,   # survives the server being ctrl-C'd
        )
    except OSError as e:
        _fail(st, f"could not start worker: {e}")
        return
    st["pid"] = p.pid
    save(st)


def pump() -> None:
    """Reap dead workers, then start at most one job. Idempotent; call it on
    a timer and at startup. This is the whole scheduler."""
    running, ready = 0, []
    for st in all_jobs():
        s = st.get("status")
        if s in TERMINAL:
            continue
        if s in RUNNING:
            if _alive(st):
                running += 1
                continue
            # The worker vanished without writing a terminal state: SIGKILL,
            # OOM, or a reboot. Retry from the last finished artifact.
            st["attempts"] = st.get("attempts", 0) + 1
            st["pid"] = None
            if st["attempts"] > MAX_ATTEMPTS:
                _fail(st, f"worker died {st['attempts']}x during {s}; giving up")
            else:
                st["status"] = "queued"
                st["error"] = f"worker died during {s}; resuming from last finished stage"
                save(st)
            continue
        if s == "queued":
            if not st.get("upload_complete"):
                if time.time() - st.get("updated", 0) > STALE_UPLOAD_S:
                    _fail(st, "upload never completed")
                continue
            ready.append(st)
    # ponytail: one job at a time - a match is 2-4 GPU-hours and there is one
    # GPU. Add a worker count here if that ever stops being true.
    if running == 0 and ready:
        _spawn(min(ready, key=lambda s: s.get("created", 0)))


# ------------------------------------------------------------------ pipeline

def _stage(st: dict, name: str) -> None:
    st["status"] = name
    st["stage_started"] = time.time()
    save(st)


def _derive_tracks(d: Path) -> None:
    """tracks.json = base (tracks.id.json, else tracks.raw.json) + corrections.json, replayed
    from scratch. The base is never mutated. Per detection: split -> merge -> exclude -> assign.
    Writes tracks.json atomically and deletes tracks_summary.json.
    docs/reference/identity.md#corrections
    """
    data = json.loads(_base_tracks(d).read_text())
    dets = data["detections"]

    corr = {}
    if (d / "corrections.json").exists():
        corr = json.loads((d / "corrections.json").read_text())

    next_id = max((int(x["id"]) for x in dets), default=0) + 1

    # Splits first: merges and assigns may name their ids. Indexed by track id (this runs inside
    # the POST). The stored new_id is used; a legacy entry without one gets max+1+index.
    # `members` = the ids merged into the track when it was split. docs/reference/identity.md#corrections
    splits: dict[int, list[tuple[int, int]]] = {}
    for i, sp in enumerate(corr.get("split", [])):
        new = int(sp["new_id"]) if "new_id" in sp else next_id + i
        for t in sp.get("members") or [sp["track"]]:
            splits.setdefault(int(t), []).append((int(sp["at_frame"]), new))
    for rules in splits.values():
        rules.sort(reverse=True)            # latest split at or before f wins
    hops = len(corr.get("split", [])) + 1   # a split of a split can chain

    remap = _merge_map(corr.get("merge", []))

    assign = {int(k): v for k, v in corr.get("assign", {}).items() if v}
    # excluded ids (referees, ball kids, phantom boxes) are dropped, not relabelled
    drop = {int(t) for t in corr.get("exclude", [])}

    # One pass: split -> merge -> exclude -> assign, in that order per detection.
    out = []
    for x in dets:
        tid, f = int(x["id"]), int(x["f"])
        for _ in range(hops):
            rules = splits.get(tid)
            if not rules:
                break
            for at, new in rules:
                if f >= at:
                    tid = new
                    break
            else:
                break
        tid = remap.get(tid, tid)
        if tid in drop:
            continue
        x["id"] = tid
        a = assign.get(tid)
        if a:
            if "jersey" in a:
                # a typed number: its source says so, and the OCR confidence no longer describes it
                x["jersey"] = a["jersey"]
                x["jersey_source"] = "operator"
                x.pop("jersey_conf", None)
            if a.get("team"):  # a null team means "keep what the model guessed"
                x["team"] = a["team"]
        out.append(x)

    data["detections"] = out
    tmp = d / "tracks.json.tmp"
    tmp.write_text(json.dumps(data))
    os.replace(tmp, d / "tracks.json")
    (d / "tracks_summary.json").unlink(missing_ok=True)


def _base_tracks(d: Path) -> Path:
    """The model output corrections replay on: identity's if it ran, else the tracker's."""
    base = d / "tracks.id.json"
    return base if base.exists() else d / "tracks.raw.json"


def _merge_map(merges: list) -> dict[int, int]:
    """[[keep, drop], ...] -> {drop: final keep}. Merges as drop -> keep, chains flattened. A merge whose keep already resolves to its drop (a
    self-merge, or the edge closing a cycle) is skipped: it would loop the resolve forever.
    apply_corrections refuses both; this covers a corrections.json saved before it did."""
    remap: dict[int, int] = {}
    for m in merges:
        keep, drop = int(m[0]), int(m[1])
        while keep in remap:
            keep = remap[keep]
        if keep != drop:
            remap[drop] = keep
    for k in list(remap):
        v = remap[k]
        while v in remap:
            v = remap[v]
        remap[k] = v
    return remap


def _try_identity(d: Path, st: dict, cal) -> None:
    """Best effort: apply_identity on tracks.raw.json -> tracks.id.json and st["identity_report"].

    Never fails the job: an import error, any exception, or SystemExit (an unreadable video)
    goes to st["identity_note"], as do a degraded run's `warnings`. `cal` (the loaded
    calibration) is required on purpose: without it merges run in pixel space and every one is
    refused. docs/reference/identity.md#errors-and-refusals
    """
    try:
        import identity  # noqa: PLC0415 - optional, heavy (OCR backends)
    except Exception as e:  # noqa: BLE001
        st["identity_note"] = f"identity unavailable: {type(e).__name__}: {e}"
        save(st)
        return
    try:
        tracks = json.loads((d / "tracks.raw.json").read_text())
        video = d / st["video"]
        tracks, report = apply_identity(identity, tracks,
                                        str(video) if video.exists() else None, cal)
        (d / "tracks.id.json").write_text(json.dumps(tracks))
        st["identity_report"] = report
        # a degraded run (no OCR backend, no facing gate) finishes; it must not read as a clean one
        if report.get("warnings"):
            st["identity_note"] = "; ".join(report["warnings"])
        save(st)
    # SystemExit too: scan_video raises it on an unreadable video, and it is a BaseException
    except (Exception, SystemExit) as e:  # noqa: BLE001 - deliberately non-fatal
        st["identity_note"] = f"{type(e).__name__}: {e}"
        save(st)


def apply_identity(identity, tracks: dict, video: str | None, cal) -> tuple[dict, dict]:
    """The identity stage with no job around it: -> (tracks, strategy report).

    Strategy from env FUTTRAFFIC_IDENTITY (default in the call), built with `cal` only.
    Mutates and returns `tracks`: per detection `raw_id` (pre-merge id), `id` = group id, and
    when a number survived `jersey`, `jersey_conf` (0-1, 3 dp), `jersey_source`; `team`.
    model/validation/st2_product.py scores this function. docs/reference/identity.md#contract
    """
    strat = identity.get_identity_strategy(
        os.environ.get("FUTTRAFFIC_IDENTITY", "assisted"), cal=cal)
    ids = strat.assign(tracks, video)
    for x in tracks["detections"]:
        p = ids.get(int(x["id"]))
        if p is None:
            continue
        x["raw_id"] = int(x["id"])   # keep the pre-merge id for traceability
        # the group id replaces the tracklet id; a wrong merge is undone by a split. After stitching the
        # merge is nearly a no-op on the nine ST2 sequences:
        # @C 512,511 :: appearance-term-decision (not published), pooled row (tracks in; groups, appearance OFF)
        if p.track_group is not None:
            x["id"] = int(p.track_group)
        if p.number is not None:
            x["jersey"] = p.number
            x["jersey_conf"] = round(float(p.confidence), 3)
            x["jersey_source"] = p.source      # "ocr" on this path; the operator's is "operator"
        if p.team:
            x["team"] = p.team
    return tracks, getattr(strat, "report", {})


def _pipeline(st: dict) -> None:
    d = job_dir(st["id"])
    video = d / st["video"]
    calib = d / "calibration.json"

    # A missing upload can never become runnable, so it fails here rather than
    # parking. Everything after this point is recoverable by a human.
    if not video.exists() and not (d / "tracks.raw.json").exists():
        raise FileNotFoundError(f"upload missing: {video.name}")

    # Recorded for EVERY job, before the calibration branch, because a job that
    # was calibrated by hand is exactly the one later uploads want to inherit
    # from. Doing it only on the inherit path would mean nothing was ever
    # inheritable.
    if not st.get("frame_shape") and video.exists():
        shape = _video_shape(video)
        if shape:
            st["frame_shape"] = list(shape)
            save(st)

    # No clicks first: the camera from this match's own paint, started from the rig's last known camera
    # when there is one. Then the older path - a hand-clicked calibration copied from the same rig.
    # Then a human. docs/reference/calibration.md#order
    if not calib.exists() and video.exists() and auto_calibrate(st):
        calib = d / "calibration.json"
    if not calib.exists() and inherit_calibration(st):
        calib = d / "calibration.json"        # adopted from an earlier match

    if not calib.exists():
        # Parked, not stuck, and parked BEFORE the expensive stage: the venue
        # calibration is an input to tracking and to identity, not a reporting
        # detail. POST one and the pump picks the job straight back up.
        _stage(st, "calibrating")
        st["pid"] = None
        save(st)
        return

    # Loaded once, here, so a calibration the pipeline cannot consume fails the
    # job in seconds with a clear message instead of after the tracking stage.
    # app.py validates uploads with this same function, so reaching this line
    # with a bad file means somebody wrote into the job directory by hand.
    try:
        cal = load_calibration(calib)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"unusable calibration.json: {e}") from e

    if not (d / "tracks.raw.json").exists():
        _stage(st, "tracking")
        import detectors  # noqa: PLC0415 - cheap; the weights are lazy
        from track import CACHE_VERSION, run as track_run  # noqa: PLC0415 - needs detector weights

        # THE JOB PATH RUNS THE FAST CONFIG, because nobody submitting through
        # the app gets to pass flags and track.py's defaults are tuned for a
        # bare `python track.py video.mp4` that must work anywhere. Measured on
        # this machine (M3, 10 GPU cores), 118575 at 11878x870, in-loop:
        #
        #     rfdetr  stride 1   1.06 fps      <- track.py's defaults, was here
        #     coreml  stride 1   1.98 fps         1.87x
        #     coreml  stride 2   3.90 fps         3.68x
        #
        # which is 18.8 h -> 5.1 h for one 71,850-frame half. The two knobs are
        # independent and they multiply, checked end to end rather than assumed.
        #
        # stride 2 is safe in every consumer, which is the part worth checking
        # rather than the speed: rows keep their ORIGINAL frame numbers, stats.py
        # reads each track's cadence off median(diff(t)), frames.py resamples to
        # a 1 Hz grid with a 1 s gap tolerance, and the video overlay holds the
        # previous frame's rings for STALE_S = 0.5 s (video-panel.tsx) against a
        # gap of 0.04 s. Identity is flat, on contamination and not on IDF1 -
        # see track.py's module docstring on why IDF1 is the wrong judge here.
        #
        # conf STAYS 0.5, and NOT because the gain is fake - it is real and it
        # survives the scene prior (detectors.py has the reproduction; a 40-frame
        # end-to-end run cannot see it because a tracker needs more samples than
        # that to confirm a new track). It stays because of what the gain costs:
        # at 0.25 the far-side players it finds land 2.30 m from truth against a
        # 2 m gate, up from 2.05 m, so overall recall in METRES - the product's
        # own metric - buys only 0.636 -> 0.657 while image precision pays 0.868
        # -> 0.812. More marginal boxes into an online tracker is the exact input
        # that welds two players into one id, and the metric that would show that
        # - contamination through the shipping path - has no runnable
        # implementation in this repo. Write that scorer, then move this.
        fast = detectors.product_detector() == "coreml"      # the one rule; check.py detect uses it too
        # NOT a silent fallback of the kind detectors.py refuses: that rule is
        # about a human typing --detector coreml and getting PyTorch. Here
        # nobody asked, and the job records what actually ran so the clock is
        # never unexplained.
        #
        # IT USED TO BE ALWAYS FALSE IN THE SHIPPED PRODUCT, because
        # app/desktop/resources/backend/_internal carries a Python 3.14 framework and
        # no 3.12, and `git ls-files model/models/` returns nothing (/models/ is
        # gitignored - the .mlpackage is 58 MB). Every packaged install took this
        # branch to "rfdetr" and ran a half in ~16.5 h rather than ~3.9 h.
        #
        # Now build-backend.sh freezes coreml_worker.py with a 3.12 interpreter
        # into resources/backend/coreml/ (59 MB - the worker's runtime closure is
        # coremltools + numpy, nothing else) and exports the .mlpackage beside it
        # (58 MB), and detectors._bundled() looks there when sys.frozen. So this
        # is True in a packaged install THAT WAS BUILT ON A MACHINE WITH A 3.12;
        # on one without, the build says so loudly and this stays False. Either
        # way st["detector"] below is the answer, and it is the only place the
        # difference is recorded - it is on the job JSON and on no screen.
        st["detector"] = "coreml" if fast else "rfdetr"
        save(st)
        # calib=: this is the scene prior. Without it the detector's boxes on the
        # crowd, dugout and next pitch can steal associations from real players.
        # docs/reference/calibration.md#scene-prior
        #
        # dets_cache=: pump() retries a dead worker "from the last finished
        # artifact", and tracking's only artifact is tracks.raw.json at the very
        # end - so before this line a SIGKILL at frame 60,000 of 71,850 threw
        # away the whole run and burned one of MAX_ATTEMPTS = 2. That is not
        # hypothetical; track.py:track() cites the 2026-08-04 job that lost nine
        # hours at frame 6,000. The cache checkpoints every 2,000 detected
        # frames and resumes per frame, and it refuses itself if the
        # calibration changed underneath (track._calib_digest), which matters
        # here because POST /calibration overwrites calibration.json in place.
        # ball=True costs nothing extra: the detector already runs at
        # min(conf, BALL_CONF) and the ball boxes come out of the SAME pass,
        # player output byte-identical (detectors.py :: BALL_CONF). What it buys is the stock
        # detector's candidates through ball.BallPicker: api://product.ball.correct.s2 at this
        # stride. The clean fine-tune (N52 arm D1) scores higher but needs a SECOND detection
        # pass, so it is a costed change and not this one; its shipped-picker level at 640:
        # @C 0.5850 :: every-ball-number-was-measured-at-384 (internal note)   arm D1, six TeamTrack clips
        # docs/reference/ball.md#limits
        # A cache detected WITHOUT the ball cannot be reused, and the two ways it
        # fails are both silent-to-the-user. Stamped (since 8028cc6, 2026-09-13):
        # _settings_mismatch raises SystemExit, the job retries once, hits the same
        # wall and fails permanently with hours of detection stranded in a file only
        # a shell can delete. Unstamped (written before that): _settings_mismatch
        # takes None to mean "no disagreement", the ball-free boxes replay, and the
        # job completes with zero ball rows - exactly the state ball=True exists to
        # end. Dropping it costs a re-detect, which is what the SystemExit asks for.
        # Same for a ball=True cache written before ball candidates were stored
        # (track.CACHE_VERSION): track.run refuses it, which here would be the permanent
        # failure again, so it is dropped too.
        cache_p = d / "dets.cache.json"
        head = cache_p.open().read(4096) if cache_p.exists() else ""
        if cache_p.exists() and ('"ball": true' not in head
                                 or f'"cache_version": {CACHE_VERSION}' not in head):
            # settings sit in the first object, ahead of "frames"; 4 KB is ~13x the
            # longest prefix observed. A cache whose stamp we cannot see is dropped
            # rather than trusted - the failure of the guard must be the safe way.
            print(f"  detection cache {cache_p} predates ball=True or stored ball candidates; "
                  "dropping it and re-detecting", flush=True)   # stdout is worker.log
            cache_p.unlink()
        res = track_run(str(video), str(d / "tracks.raw.json"), calib=str(calib),
                        detector=st["detector"], stride=2, ball=True,
                        dets_cache=str(d / "dets.cache.json"))
        rej = int((res or {}).get("_perf", {}).get("prior_rejected", 0))
        st["prior_rejected"] = rej
        # make_prior() returns None rather than raising when the geometry is
        # unusable, so "filter working" and "filter silently absent" look
        # identical from out here. Zero rejections over a whole match means the
        # second one; say so on the job instead of shipping unmarked numbers.
        st["scene_prior"] = "active" if rej else (
            "no detections rejected - the geometry prior may not have loaded; "
            "see worker.log")
        # The cache exists to protect an UNFINISHED run, and tracks.raw.json
        # now exists, so it is 190 MB a half of dead weight next to a 158 MB
        # tracks.raw.json in a directory nothing ever prunes. Re-running the
        # tracker off cached detections is a development workflow - use
        # track.py --dets-cache for it - not something the product does.
        (d / "dets.cache.json").unlink(missing_ok=True)
        save(st)

    if not (d / "tracks.id.json").exists() and not st.get("identity_done"):
        _stage(st, "identifying")
        _try_identity(d, st, cal)
        st["identity_done"] = True
        save(st)

    # Replaying corrections is a full parse+rewrite of a file that is hundreds
    # of MB on a real match, and apply_corrections() already did it inline so
    # the correction UI could re-read immediately. Only redo it if an input is
    # actually newer than the output.
    tj = d / "tracks.json"
    newest = max((p.stat().st_mtime for p in
                  (d / "tracks.id.json", d / "tracks.raw.json", d / "corrections.json")
                  if p.exists()), default=0.0)
    if not tj.exists() or tj.stat().st_mtime < newest:
        _derive_tracks(d)

    from stats import build_match  # noqa: PLC0415 - pulls cv2/numpy
    build_match(str(tj), str(calib), str(d / "match.json"),
                meta={"job_id": st["id"], "source": st.get("display_name")})
    # Never fatal (see ingest) but never silent either: this return value was
    # discarded, so a store that refused every new match read as `done`.
    res = ingest(st["id"])
    st["ingest_error"] = res.get("error") if isinstance(res, dict) else None
    st["status"] = "done"
    st["error"] = None
    st["pid"] = None
    save(st)


def run(jid: str) -> None:
    """Worker entry point. Always leaves a terminal or parked state behind."""
    st = load(jid)
    d = job_dir(jid)
    st["pid"] = os.getpid()
    save(st)
    stop = threading.Event()
    threading.Thread(target=_heartbeat, args=(d, stop), daemon=True).start()
    try:
        _pipeline(st)
    except BaseException as e:  # noqa: BLE001 - the point is that nothing escapes
        st = load(jid)
        _fail(st, f"{type(e).__name__}: {e}"[:2000])
        raise
    finally:
        stop.set()
        (d / "alive").unlink(missing_ok=True)


# -------------------------------------------------------------- corrections

def apply_corrections(jid: str, payload: dict) -> dict:
    """Merge one correction session into corrections.json and re-derive. -> status_view.

    payload: {"assign": {tid: {jersey?, team?, player?} | null}, "merge": [[keep, drop]],
    "split": [{"track", "at_frame"}], "exclude": [tid], "undo": [tid], "operator_seconds": s}.
    Accumulates: assign by key, merge/split de-duplicated, exclude as a set; undo removes what
    names the id, including a split whose track or new_id it is. Raises ValueError for a self or cyclic merge, or a split before any tracks
    exist; KeyError/TypeError for a malformed entry (app.py answers 400).
    docs/reference/identity.md#corrections
    """
    d = job_dir(jid)
    st = load(jid)
    cur = {}
    if (d / "corrections.json").exists():
        cur = json.loads((d / "corrections.json").read_text())

    cur.setdefault("assign", {})
    for k, v in (payload.get("assign") or {}).items():
        cur["assign"][str(k)] = v
    cur.setdefault("merge", [])
    cur.setdefault("split", [])
    for m in payload.get("merge") or []:
        keep, drop = int(m[0]), int(m[1])
        if [keep, drop] in [[int(o[0]), int(o[1])] for o in cur["merge"]]:
            continue
        # A self-merge or one closing a cycle hung _derive_tracks forever (inside this POST).
        if _merge_map(cur["merge"]).get(keep, keep) == drop:
            raise ValueError(f"merge [{keep}, {drop}] would merge track {drop} into itself")
        cur["merge"].append([keep, drop])
    new_splits = [s for s in payload.get("split") or []
                  if (int(s["track"]), int(s["at_frame"])) not in
                  {(int(o["track"]), int(o["at_frame"])) for o in cur["split"]}]
    if new_splits:
        # new_id minted here and stored, above every id the base or any split can hold
        # ponytail: a second parse of the base on a split POST; carry max id in job state if it bites.
        base = _base_tracks(d)
        if not base.exists():
            raise ValueError("no tracks to split yet")
        top = max((int(x["id"]) for x in json.loads(base.read_text())["detections"]), default=0)
        top = max([top + len(cur["split"])] + [int(s["new_id"]) for s in cur["split"] if "new_id" in s])
        remap = _merge_map(cur["merge"])
        for s in new_splits:
            t = int(s["track"])
            top += 1
            cur["split"].append({"track": t, "at_frame": int(s["at_frame"]), "new_id": top,
                                 # what the operator saw as track t: t plus every id merged into it so far
                                 "members": [t] + sorted(k for k, v in remap.items() if v == t)})
    cur["exclude"] = sorted({*(cur.get("exclude") or []), *(payload.get("exclude") or [])})

    # undo removes a split named by its `track` or by the new_id it minted (docs/reference/identity.md#corrections)
    for tid in payload.get("undo") or []:
        tid = int(tid)
        cur["assign"].pop(str(tid), None)
        cur["exclude"] = [t for t in cur["exclude"] if int(t) != tid]
        cur["merge"] = [m for m in cur["merge"] if tid not in (int(m[0]), int(m[1]))]
        cur["split"] = [s for s in cur["split"] if tid not in (int(s["track"]), int(s.get("new_id", -1)))]
    cur["updated"] = time.time()

    tmp = d / "corrections.json.tmp"
    tmp.write_text(json.dumps(cur, indent=2))
    os.replace(tmp, d / "corrections.json")
    _rebind(jid, payload.get("assign") or {}, payload.get("undo") or [])

    # operator time, cumulative seconds (identity-evidence (not published)#evidence-design-stance)
    st["operator_seconds"] = round(
        st.get("operator_seconds", 0.0) + float(payload.get("operator_seconds") or 0.0), 1)
    st["corrections"] = len(cur["assign"]) + len(cur["merge"]) + len(cur["split"])

    if (d / "tracks.raw.json").exists():
        _derive_tracks(d)
        (d / "match.json").unlink(missing_ok=True)
        # re-queue: the match rebuild runs in the worker, not in this request
        st["status"] = "queued" if (d / "calibration.json").exists() else "calibrating"
        st["attempts"] = 0
    save(st)
    return status_view(st)


def read_corrections(jid: str) -> dict:
    """corrections.json, or the empty schema when there is none."""
    p = job_dir(jid) / "corrections.json"
    return json.loads(p.read_text()) if p.exists() else {"assign": {}, "merge": [], "split": [], "exclude": []}


# --------------------------------------------------------------------- clips
# A coach's clip list is the one artefact on this system that is NOT derived
# from the pipeline: it is a human saying "watch this". It therefore lives in
# its own file next to the job and is never touched by _derive_tracks, a
# re-track or a re-calibration - reprocessing a match must not delete the notes
# somebody spent an evening making. Times are seconds into the video, so they
# survive a re-track that renumbers every track id.

# Fixed, small, and closed on purpose. A free tag list becomes 40 spellings of
# "turnover" by the third match and nothing can be filtered on afterwards; the
# free-text note is where anything else goes.
CLIP_TAGS = ("Goal", "Chance", "Turnover", "Press", "Set piece", "Transition", "Error")
MAX_CLIPS = 500


def _clean_clip(c: dict) -> dict:
    """Normalise one clip. Everything a client sends is bounded here: this is
    the only writer of clips.json and app.py's size check is not enough on its
    own (4 MB of JSON is still 4 MB of tag names)."""
    if not isinstance(c, dict):
        raise ValueError("a clip must be an object")
    cid = str(c.get("id") or "")
    if not ID_RE.match(cid):
        cid = new_id()          # a client-supplied id is a label, never a path
    try:
        start, end = float(c.get("start", 0)), float(c.get("end", 0))
    except (TypeError, ValueError):
        raise ValueError("start and end must be seconds into the video")
    # Written as one comparison chain so NaN and inf fail it too - both compare
    # False against everything, and a NaN start would sort and seek nowhere.
    if not (0 <= start < end < 1e9):
        raise ValueError("a clip needs an out-point after its in-point")
    return {
        "id": cid,
        "name": (str(c.get("name") or "").strip() or "Clip")[:120],
        "start": round(start, 3),
        "end": round(end, 3),
        # Unknown tags are dropped rather than rejected: a page from an older
        # deploy sending a retired tag should still save the clip.
        "tags": [t for t in CLIP_TAGS if t in (c.get("tags") or [])],
        "note": str(c.get("note") or "")[:2000],
        "created": float(c.get("created") or time.time()),
    }


def read_clips(jid: str) -> list[dict]:
    p = job_dir(jid) / "clips.json"
    if not p.exists():
        return []
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return []               # half-written; an empty list beats a 500
    return d.get("clips", []) if isinstance(d, dict) else []


def _write_clips(jid: str, clips: list[dict]) -> list[dict]:
    d = job_dir(jid)
    tmp = d / "clips.json.tmp"
    tmp.write_text(json.dumps({"clips": clips, "updated": time.time()}, indent=2))
    os.replace(tmp, d / "clips.json")   # atomic, same as save()
    return clips


def save_clip(jid: str, clip: dict) -> list[dict]:
    """Upsert one clip by id and return the whole list, in playing order."""
    c = _clean_clip(clip)
    clips = [x for x in read_clips(jid) if x.get("id") != c["id"]]
    if len(clips) >= MAX_CLIPS:
        raise ValueError(f"at most {MAX_CLIPS} clips per match")
    clips.append(c)
    clips.sort(key=lambda x: (x.get("start", 0), x.get("created", 0)))
    return _write_clips(jid, clips)


def delete_clip(jid: str, clip_id: str) -> list[dict]:
    """Idempotent: deleting a clip that is already gone is a success, or two
    tabs open on the same match turn one delete into a 404."""
    return _write_clips(jid, [x for x in read_clips(jid) if x.get("id") != str(clip_id)])


# ------------------------------------------------------------------- roster
# Backed by roster.Roster, one file per team: one store, so the season join joins.
# docs/reference/identity.md#roster

def _roster_dir() -> Path:
    d = DATA_DIR / "rosters"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _roster_files() -> list[Path]:
    d = DATA_DIR / "rosters"
    return sorted(d.glob("*.json")) if d.exists() else []


def _roster_path(team: str) -> Path:
    """DATA_DIR/rosters/<team-slug>.json. ValueError for a team with an empty slug."""
    import roster as R  # noqa: PLC0415

    slug = re.sub(r"[^a-z0-9-]", "", R._slug(team))[:60]
    if not slug:
        raise ValueError("team name required")
    return _roster_dir() / f"{slug}.json"


PRESETS_PATH = DATA_DIR / "presets.json"

# A preset is small and there will never be many. The cap is here because this
# file is written from a browser and "never many" is a habit, not a guarantee.
PRESETS_MAX = 200


def presets() -> list[dict]:
    """The saved stat selections, shared by every note in the project.

    Beside the roster rather than inside a document, because the whole point is
    that a comparison built once is offered in the next note and the next match.
    A note carries the ANSWER; this carries the question somebody keeps asking.
    """
    try:
        raw = json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [p for p in raw if isinstance(p, dict)] if isinstance(raw, list) else []


def save_presets(items: list[dict]) -> list[dict]:
    """Whole-list replace. A preset has no server-minted identity to preserve —
    unlike a roster player, where the id is the season join — so the simplest
    honest contract is: send the list you want, get the list back."""
    clean = []
    for p in items[:PRESETS_MAX]:
        if not isinstance(p, dict):
            continue
        name = str(p.get("name", "")).strip()[:60]
        cols = [str(c)[:40] for c in (p.get("cols") or []) if isinstance(c, (str, int))][:24]
        if name and cols:
            clean.append({"name": name, "cols": cols, "sort": str(p.get("sort", ""))[:40]})
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PRESETS_PATH.write_text(json.dumps(clean, indent=1), encoding="utf-8")
    return clean


# ------------------------------------------------------------------- notes
# THE NOTE LIBRARY. One folder tree of HTML documents for the whole club.
#
# Notes used to be one `doc.html` per job, filed under the match the way a
# report is. That was backwards, and it is the reason this store exists: an
# automated match report IS about one match, but a note is the thing a coach
# writes ACROSS matches - a scouting document on an opponent seen three times,
# a development file on a player who climbed U15 -> U17, a season review that
# spans every team on the ladder. Filed under a match, none of those had a home;
# the notepad's own stat blocks could already reach the whole library while the
# document containing them could not.
#
# So: no match id, no team id, no owner - a path. The FILESYSTEM is the folder
# tree, which is the whole reason there is no `note` table here. A folder is a
# directory, a rename is a rename, and the tree a coach organises survives being
# copied to another machine, opened in Finder, or put in git. A database would
# buy ordering and indexes for a few hundred documents that are read whole.
#
# The client supplies the path, so `note_path` is the only place a string
# becomes one - the same contract job_dir() has for ids, for the same reason.
def notes_dir() -> Path:
    """A FUNCTION of DATA_DIR, not a constant. `_selfcheck` rebinds DATA_DIR to
    a temp directory, and a constant computed at import ignores that - which is
    how a self-check run once wrote into the developer's real store."""
    return DATA_DIR / "notes"


# A FULL SCOUTING REPORT CAN BE 953 KB. That is the artefact this product
# exists to make - eighteen pages, fifty boards, thirty-six headshots - and it
# sat at 93% of a 1 MB cap, which meant the first paragraph a coach added to it
# would have been refused. A limit the flagship document nearly fails is not a
# limit, it is a bug waiting for a demo.
#
# ponytail: four megabytes, because the photographs are inlined as data URIs so
# the note works offline. If reports start failing this, the fix is not a bigger
# number - it is storing the photos once per note and referencing them, which is
# a real change to what "a note is one file" means.
NOTE_MAX = 4 * 1024 * 1024  # an illustrated report, not a video
NOTE_DEPTH = 6          # folders inside folders; past this it is a filing system
NOTE_SEG = 80           # per path segment, well inside every filesystem's limit
# Reserved on Windows and awkward everywhere; a leading dot hides the file.
_BAD_SEG = re.compile(r'[\x00-\x1f/\\:*?"<>|]')


def note_path(rel: str, *, folder: bool = False) -> Path:
    """Turn a client string into a path inside the note library, or raise.

    Every note route goes through here. The checks are structural (segment by
    segment) rather than a scan for "..", because a scan is the version of this
    that gets bypassed: `%2e%2e`, a backslash on the wrong platform, a NUL that
    truncates the name in the syscall. The resolve() at the end is the backstop
    for the one thing segments cannot see - a symlinked folder inside the
    library pointing out of it."""
    parts = (rel or "").split("/")
    if not 1 <= len(parts) <= NOTE_DEPTH:
        raise ValueError(f"bad note path: {rel!r}")
    for seg in parts:
        if not seg or seg.startswith(".") or len(seg) > NOTE_SEG or _BAD_SEG.search(seg):
            raise ValueError(f"bad note path: {rel!r}")
    if not folder and not parts[-1].endswith(".html"):
        raise ValueError(f"a note is an .html file: {rel!r}")
    root = notes_dir()
    path = root.joinpath(*parts)
    if root.resolve() not in path.resolve().parents:
        raise ValueError(f"note path escapes the library: {rel!r}")
    return path


def _note_name(s: str) -> str:
    """A display name as a filename. Not reversible and not meant to be - the
    path is the identity, the name is what it happens to read as."""
    clean = _BAD_SEG.sub(" ", s or "").strip(" .")
    return " ".join(clean.split())[:NOTE_SEG - 5] or "Note"


_TAG_RE = re.compile(r"<[^>]*>")
_SVG_RE = re.compile(r"<svg\b|</svg>", re.I)

# How much of a note is read to build its card. Enough to reach past a board's
# serialised discs to the figure after it, and far short of a heat map's 96 bins
# of embedded numbers.
PREVIEW_READ = 256 * 1024
# The top of the note rides in the listing, once per note, so a card can show
# the document as it actually looks rather than a paraphrase of it. Enough for
# the first screenful; far short of a heat map's 96 bins of embedded numbers.
HEAD_MAX = 24 * 1024


def _head(head: str) -> str:
    """The opening markup of the note, cut where a browser can still parse it.

    Unclosed <p>s and <li>s the parser closes by itself, so the cut only has to
    land outside a TAG - hence the trim back past a `<` with no `>` after it,
    and only then: a cut that always retreated to the last `<` would throw away
    the whole of a note that opens with one long paragraph. The one thing it
    must not leave behind is half a figure: an <svg> whose close was past the
    cut draws as a chart with its right-hand side missing, on exactly the notes
    that open with the biggest ones. The tags are counted rather than matched
    because an svg can nest one inside another."""
    if len(head) <= HEAD_MAX:
        return head
    cut = head[:HEAD_MAX]
    if cut.rfind("<") > cut.rfind(">"):
        cut = cut[:cut.rfind("<")]
    depth, opened = 0, -1
    for m in _SVG_RE.finditer(cut):
        if m.group(0).lower() == "<svg":
            if depth == 0:
                opened = m.start()
            depth += 1
        else:
            depth -= 1
    return cut[:opened] if depth > 0 else cut


@functools.lru_cache(maxsize=256)
def _preview(path: str, mtime: float) -> tuple[str, str]:
    """(snippet, opening markup) for the library page's cards, off ONE read of
    the front of the file. The card renders the markup in a sandboxed frame, at
    the size of a thumbnail: what makes a note recognisable at a glance is the
    shape of its own first page - its heading, its first figure, its columns -
    and no summary of ours reproduces that. The snippet is what is left to say
    when there is no markup at all.

    ponytail: mtime-keyed like `_season_rows_cached`, and for the same reason -
    the library page lists every note on every load, and this reads the first
    256 KB of each one to produce a thumbnail that only changes when the note
    does. The caller already has the stat, so the key is free."""
    try:
        with Path(path).open(encoding="utf-8", errors="replace") as f:
            head = f.read(PREVIEW_READ)
    except OSError:
        return "", ""
    return " ".join(unescape(_TAG_RE.sub(" ", head)).split())[:200], _head(head)


def notes() -> list[dict]:
    """Every note and every folder, path-sorted. Folders are listed in their own
    right so an empty one survives - a coach who makes `Scouting/` before
    writing the first note should find it there tomorrow."""
    root = notes_dir()
    if not root.exists():
        _seed_notes(root)
    _ladder_folders(root)
    out: list[dict] = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        is_dir = p.is_dir()
        try:
            note_path(rel, folder=is_dir)
        except ValueError:
            continue  # placed by hand, too deep, or not a note; not addressable
        if is_dir:
            out.append({"path": rel, "dir": True})
        elif p.is_file():
            st = p.stat()
            snippet, head = _preview(str(p), st.st_mtime)
            out.append({"path": rel, "dir": False, "updated": st.st_mtime,
                        "bytes": st.st_size, "snippet": snippet, "head": head})
    return out


def _ladder_folders(root: Path) -> None:
    """A folder for the club and one per team, kept in step with the ladder.

    The tree MIRRORS the organisation, so a club that adds a U19 in August finds
    somewhere to file the U19 notes without making it first. That is also why
    this runs on every listing rather than once at seed time: a folder that only
    ever appeared for the teams that existed on install would be wrong by the
    end of the first season.

    The consequence, stated rather than hidden: deleting a team's folder brings
    it back empty on the next listing. To rename it, rename the TEAM - the
    folder follows. A folder you want to keep to yourself lives under any name
    that is not a team's.
    """
    try:
        club = org()
    except (OSError, ValueError, json.JSONDecodeError):
        return  # no ladder yet is not a reason to fail the listing
    names = [str(club.get("club", {}).get("name") or "")]
    names += [str(t.get("name") or "") for t in club.get("teams", [])]
    for name in names:
        clean = _note_name(name)
        if clean and clean != "Note":  # _note_name's fallback, not a real name
            (root / clean).mkdir(parents=True, exist_ok=True)


def read_note(rel: str) -> str:
    return note_path(rel).read_text(encoding="utf-8")


def note_sha(html: str) -> str:
    """WHICH VERSION OF A NOTE THIS IS. A pure function of the content, so the
    reader and the writer derive it the same way without either re-reading the
    file. Short because it is an equality check between two strings that were
    both produced here, not a signature."""
    return hashlib.sha256(html.encode("utf-8")).hexdigest()[:16]


class NoteConflict(Exception):
    """The note changed under the writer.

    Carries what it changed TO, so the caller can hand the loser its own text
    back rather than dropping it on the floor. That is the whole point: a
    guard whose rejections are invisible is worse than no guard, because it
    turns a loud clobber into a quiet deletion."""

    def __init__(self, current: str) -> None:
        super().__init__("the note changed since you read it")
        self.current = current


def write_note(rel: str, html: str, base: str | None = None) -> int:
    """Write the note. With `base`, write it ONLY if that is still its version.

    Two writers reach this file now - a coach with the editor open, and a
    report script - and until this existed the second silently overwrote the
    first. `base` is opt-in because the two want opposite things: the editor
    quotes the version it loaded and wants to be told when it is stale, while a
    report script owns its output path and re-running it is meant to replace
    what is there. Absent `base` therefore behaves exactly as before.

    The read-compare-write is not atomic and cannot be made so against a plain
    file without a lock. It closes the window from seconds (an editor's
    debounce) to microseconds, which is the whole gap that was being hit."""
    p = note_path(rel)
    if base is not None:
        current = note_sha(p.read_text(encoding="utf-8")) if p.exists() else ""
        if current != base:
            raise NoteConflict(current)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return len(html.encode())


def make_folder(rel: str) -> None:
    note_path(rel, folder=True).mkdir(parents=True, exist_ok=True)


def delete_note(rel: str) -> None:
    """A folder goes with everything in it - the tree shows what that is before
    it asks. Deleting nothing is not an error: the UI's job is to end up with
    the note gone, and a 404 on the second click of a double-click is noise."""
    p = note_path(rel, folder=True)
    if p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()


def move_note(src: str, dst: str) -> None:
    """Rename, and drag-into-a-folder, which are the same operation. Refuses to
    land on something that already exists rather than silently replacing a note
    somebody else wrote - the one destructive thing a drag could do."""
    a = note_path(src, folder=True)
    b = note_path(dst, folder=a.is_dir())
    if not a.exists():
        raise ValueError(f"no such note: {src!r}")
    if b.exists():
        raise ValueError(f"already there: {dst!r}")
    if a.is_dir() and a.resolve() in b.resolve().parents:
        raise ValueError("a folder cannot be moved inside itself")
    b.parent.mkdir(parents=True, exist_ok=True)
    os.replace(a, b)


def _seed_notes(root: Path) -> None:
    """First open: lift the old per-match notepads into the library.

    Seeded on READ, like the club ladder in org.py, so no install needs a
    migration step that can half-run. The job's own doc.html is COPIED, not
    moved: a seed that got a name wrong costs nothing to redo by hand.
    """
    root.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    for d in sorted(JOBS_DIR.iterdir()) if JOBS_DIR.exists() else []:
        src = d / "doc.html"
        if not src.is_file():
            continue
        try:
            name = _note_name(load(d.name).get("display_name") or d.name)
        except (OSError, ValueError, json.JSONDecodeError, KeyError):
            name = d.name
        # Two matches can carry one name; the second must not overwrite the first.
        rel = f"Matches/{name}.html"
        while rel in used:
            name = f"{name} (2)"[:NOTE_SEG - 5]
            rel = f"Matches/{name}.html"
        used.add(rel)
        try:
            write_note(rel, src.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # one unreadable note must not stop the seed


def roster() -> list[dict]:
    """One row per (player, TEAM). The correction screen's shape: it is fixing
    one match, which belongs to one team, so it wants that team's squad.

    A player who is registered with two teams appears twice here, once per
    squad, with that squad's shirt number. `players()` is the other question -
    one row per HUMAN - and the two must not be conflated.
    """
    try:
        import roster as R  # noqa: PLC0415
    except Exception:
        return []
    out = []
    for p in _roster_files():
        r = R.Roster.load(str(p))
        out += [{"id": pl.player_id, "name": pl.name, "number": pl.number,
                 "position": pl.position, "team": r.team, "team_id": p.stem}
                for pl in r.players.values()]
    return sorted(out, key=lambda p: (p["team"] or "", p["number"] if p["number"] is not None else 999))


def save_roster(players: list[dict]) -> list[dict]:
    """Upsert. Player ids are minted by roster.py (name-slugged, so a
    mid-season shirt change doesn't split one player into two profiles), which
    is why the canonical list is returned rather than echoing the input."""
    import roster as R  # noqa: PLC0415

    by_team: dict[str, list[dict]] = {}
    for p in players[:400]:
        if not isinstance(p, dict):
            continue
        team = str(p.get("team") or "").strip()[:60]
        if team:
            by_team.setdefault(team, []).append(p)

    for team, ps in by_team.items():
        path = _roster_path(team)
        r = R.Roster.load(str(path), team=team)
        r.team = team
        for p in ps:
            num = p.get("number")
            name = str(p.get("name"))[:60] if p.get("name") else None
            pid = str(p.get("id"))[:60] if p.get("id") else None
            pos = str(p.get("position"))[:12] if p.get("position") else None
            r.add(number=num if isinstance(num, int) else None, name=name,
                  player_id=pid if pid in r.players else None, position=pos)
        r.save(str(path))
    refresh_club()
    return roster()


def players() -> list[dict]:
    """THE PLAYER REGISTRY - one row per HUMAN, across every team in the club.

    This is the thing an academy buys: a player climbs U13 -> U15 -> U17 and
    stays one profile the whole way, so three years of a career is one query
    rather than three unrelated squad lists.

    It is a MERGE, not a store. roster.py mints `player_id` from the name slug,
    so the same human already carries the same id in every team file they
    appear in; joining on it here is therefore guaranteed consistent with the
    per-team squads by construction. A second player table would be a second
    source of truth about who somebody is, and the week the two disagreed one
    human would become two half-careers - which is the exact failure roster.py
    was written to prevent.

    Inherited limit, stated rather than hidden: two DIFFERENT humans with the
    same name share an id, and merging across teams makes that more likely, not
    less. `teams` on each row is what makes it visible - a "player" carrying
    both the U9 and the first team is the shape that bug takes. The fix is the
    same as always: give one of them an explicit player_id.
    """
    try:
        import roster as R  # noqa: PLC0415
    except Exception:
        return []
    out: dict[str, dict] = {}
    for p in _roster_files():
        try:
            r = R.Roster.load(str(p))
        except Exception:  # noqa: BLE001 - a corrupt roster must not empty the club
            continue
        # Matches this team's file has a HUMAN-CONFIRMED binding for. A machine
        # guess is not a season profile; see season() for the whole argument.
        played: dict[str, set] = {}
        for mid, tb in r.bindings.items():
            for b in tb.values():
                if b.get("source") == "manual":
                    played.setdefault(b["player_id"], set()).add(mid)
        for pl in r.players.values():
            row = out.setdefault(pl.player_id, {
                "id": pl.player_id, "name": pl.name, "position": pl.position,
                "teams": [], "matches": 0,
            })
            if pl.name and not row["name"]:
                row["name"] = pl.name
            if pl.position and not row["position"]:
                row["position"] = pl.position
            row["teams"].append({"team_id": p.stem, "team": r.team,
                                 "number": pl.number, "position": pl.position})
            row["matches"] += len(played.get(pl.player_id, ()))
    for row in out.values():
        row["photo"] = has_photo(row["id"])
    return sorted(out.values(), key=lambda p: (p["name"] or "").lower())


# -------------------------------------------------------------------- photos
# HEADSHOTS. A scouting report is headed by faces, and a coach reading a squad
# he does not know yet learns them off those pages before he learns the numbers.
#
# A FILE PER PLAYER, keyed by the same `player_id` the roster mints - the join
# is the one roster.py already guarantees, so a photo follows a player from U15
# to the first team for free. No table: it is a few hundred small files that are
# read whole, which is the same argument the note library makes.
#
# The bytes are NOT trusted. The type is decided by the file's own magic number
# and never by a header or a filename, and only the three formats every browser
# draws are accepted - which also means no SVG, because an SVG is a document
# that can carry script and this one would be served from our own origin.
def photos_dir() -> Path:
    """A FUNCTION of DATA_DIR, for the reason notes_dir() is one."""
    return DATA_DIR / "photos"


PHOTO_MAX = 4 * 1024 * 1024        # a headshot, not a photograph library
# \Z, not $: Python's `$` also matches immediately BEFORE a trailing newline,
# so "abc\n" passed this guard and wrote data/bios/abc\n.json. Reachable as
# PUT /api/players/abc%0A/bio, because Starlette percent-decodes the path
# param. Not traversal - "/", "." and "\\" are outside the class - but a file
# whose stem carries a newline is then served back under that stem.
_PID = re.compile(r"^[A-Za-z0-9_-]{1,64}\Z")
# (magic prefix, extension, media type). WEBP is RIFF....WEBP, so the check
# skips the four length bytes in the middle.
_PHOTO_KINDS = (
    (b"\x89PNG\r\n\x1a\n", "png", "image/png"),
    (b"\xff\xd8\xff", "jpg", "image/jpeg"),
)


def photo_kind(data: bytes) -> tuple[str, str] | None:
    """(extension, media type) for bytes we will serve, or None. The file says
    what it is; nothing the client claimed is consulted."""
    for magic, ext, mime in _PHOTO_KINDS:
        if data.startswith(magic):
            return ext, mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp", "image/webp"
    return None


def photo_path(pid: str, ext: str) -> Path:
    """The one place a player id becomes a path - note_path's contract, for the
    same reason. The id is a slug roster.py minted, so anything outside that
    alphabet is not an id at all."""
    if not _PID.match(pid or ""):
        raise ValueError(f"bad player id: {pid!r}")
    return photos_dir() / f"{pid}.{ext}"


def find_photo(pid: str) -> tuple[Path, str] | None:
    """(path, media type) of whatever this player has, or None."""
    for _magic, ext, mime in (*_PHOTO_KINDS, (b"", "webp", "image/webp")):
        p = photo_path(pid, ext)
        if p.is_file():
            return p, mime
    return None


def has_photo(pid: str) -> bool:
    try:
        return find_photo(pid) is not None
    except ValueError:
        return False


def write_photo(pid: str, data: bytes) -> str:
    """Store one headshot and return its media type. Replaces whatever was
    there in any format - a player has one face, not one per file type."""
    kind = photo_kind(data)
    if kind is None:
        raise ValueError("not a PNG, JPEG or WebP")
    ext, mime = kind
    delete_photo(pid)
    p = photo_path(pid, ext)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return mime


def bios_dir() -> Path:
    """A FUNCTION of DATA_DIR, for the reason notes_dir() is one."""
    return DATA_DIR / "bios"


def bios() -> dict:
    """Every player bio written by an importer (none ships in this repository), keyed on the roster's own id.

    Read whole and served whole, like photo_credits: the registry shows one
    profile at a time but it is one small file per player over a squad, and 23
    reads of a few hundred bytes beat 23 round trips.

    A missing directory is {}, not an error. Bios only exist for players who
    came from a feed - a club that entered its squad by hand has none, and
    that is a card with fewer fields on it rather than a failure.
    """
    d = bios_dir()
    if not d.exists():
        return {}
    out = {}
    for f in sorted(d.glob("*.json")):
        try:
            b = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # one unreadable bio must not cost the other twenty-two
        if isinstance(b, dict):
            out[f.stem] = b
    return out


#: What an operator is allowed to write onto a bio. Everything else in the
#: file - traits, heatmap, shots, stat_groups, market value - is the FEED's
#: account of a player and is not editable by hand: those are measurements
#: somebody else made, and a form that lets you type over them would turn a
#: cited number into an anonymous one.
BIO_FIELDS = {
    "name": str,
    "team": str,
    "on_loan": bool,
    "birth_date": str,
    "height_cm": int,
    "shirt": int,
    "foot": str,
    "country": str,
    "country_code": str,
    "primary_position": str,
    "other_positions": list,
}


def save_bio(pid: str, patch: dict) -> dict:
    """MERGE the operator's fields onto a bio and return the whole stored bio.

    A merge and not a replace, because the two halves of this file have
    different authors. The feed wrote the traits, the heatmap and the shots;
    the operator is editing a height. A PUT that replaced the document would
    silently delete the measured half every time somebody fixed a shirt
    number - and the loss would only show up in the one panel that draws it.

    `None` DELETES a field rather than storing a null, so clearing a box in
    the form leaves the bio the way a player who never had a height looks -
    which is the state every reader here already handles.

    Fields outside BIO_FIELDS are dropped, not rejected: the client sends the
    form it has, and a new field on the client must not 400 the old server.
    """
    if not _PID.match(pid or ""):
        raise ValueError(f"bad player id: {pid!r}")
    if not isinstance(patch, dict):
        raise ValueError("bio must be an object")

    d = bios_dir()
    p = d / f"{pid}.json"
    try:
        bio = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        bio = {}
    if not isinstance(bio, dict):
        bio = {}

    for k, kind in BIO_FIELDS.items():
        if k not in patch:
            continue
        v = patch[k]
        if v is None or v == "":
            bio.pop(k, None)
            continue
        if kind is int:
            try:
                v = int(v)
            except (TypeError, ValueError):
                continue
        elif kind is bool:
            v = bool(v)
        elif kind is list:
            if not isinstance(v, list):
                continue
            v = [str(x)[:24] for x in v[:12] if str(x).strip()]
        else:
            v = str(v)[:80]
        bio[k] = v

    # The age the panel prints is DERIVED, not entered: two fields that can
    # disagree about the same fact is how a card ends up saying 24 under a
    # birth date in 2001. Recomputed here so it cannot drift from birth_date.
    bio.pop("age", None)
    born = str(bio.get("birth_date") or "")[:10]
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", born)
    if m:
        y, mo, day = (int(x) for x in m.groups())
        today = _dt.date.today()
        try:
            age = today.year - y - ((today.month, today.day) < (mo, day))
        except ValueError:
            age = None
        if age is not None and 0 <= age < 120:
            bio["age"] = age

    d.mkdir(parents=True, exist_ok=True)
    # COMPACT, not indent=2. The feed's half of this file is heatmap and shot
    # arrays - the largest in the store is 136 KB - and re-dumping one at
    # indent=2 grew it by a quarter, permanently, on the first save of a single
    # typed field. Nothing reads these by eye; every reader is json.loads.
    p.write_text(json.dumps(bio, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return bio


def photo_credits() -> dict:
    """The attribution written beside the photos by headshots.py, or {}.

    Kept as one JSON file rather than one per photo: it is read whole, every
    time, by whatever is about to print a face."""
    try:
        raw = json.loads((photos_dir() / "credits.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    # Only what there is actually a photo FOR. A seeding run that resolved a
    # name and then failed to download it leaves a credit behind, and a report
    # that printed it would be crediting a photographer for a face nobody is
    # looking at.
    return {k: v for k, v in raw.items() if has_photo(k)}


def delete_photo(pid: str) -> None:
    """Deleting nothing is not an error - the caller's job is to end up with no
    photo, which is the same rule delete_note follows."""
    for _magic, ext, _mime in (*_PHOTO_KINDS, (b"", "webp", "")):
        p = photo_path(pid, ext)
        if p.is_file():
            p.unlink()


# ----------------------------------------------------------------------- club
# The academy: teams, age levels, coaches. org.py owns the model and the file;
# this is the view the API hands the browser, which is the club joined to what
# is actually on disk - how many players each squad holds and how many matches
# it has - because a team with no squad and no footage is the one a coach needs
# to see is empty.

# FUNCTIONS, not module constants, and that distinction has already cost
# something: `_selfcheck` points DATA_DIR at a temp directory to run the whole
# pipeline in isolation, and a constant computed at import time does not follow
# that rebind. So a test run wrote its throwaway match into the REAL
# data/match.db and re-ingested an EMPTY club over the live one, which silently
# emptied `squad` - every player's club membership - on the developer's own
# machine. ROSTER_PATH avoids this only because the self-check remembers to
# rebind it by hand, which is a rule nobody can be expected to keep.
def org_path() -> Path:
    return DATA_DIR / "org.json"


def db_path() -> Path:
    return DATA_DIR / "match.db"


def ingest(jid: str) -> dict:
    """Land one finished match in the query store, and refresh the ladder.

    Called at the end of the pipeline, which is what makes the store a live
    surface rather than something somebody remembers to rebuild: a match that
    finishes processing is queryable and IN the season totals immediately.

    NEVER FATAL. match.json is the source of truth and the store is a query
    layer over it (see db.py) - so a database that is locked, corrupt or on a
    full disk must not turn a finished match into a failed job. It returns the
    error instead, and `python db.py build` rebuilds the whole store from the
    files whenever someone wants it back.
    """
    try:
        import db  # noqa: PLC0415 - only the ingest path pays for this
        import org as O  # noqa: PLC0415

        st = load(jid)
        path = job_dir(jid) / "match.json"
        conn = db.connect(str(db_path()))
        try:
            db.ingest_org(conn, org(), roster())
            out = db.ingest_match(conn, jid, json.loads(path.read_text()))
            db.set_match_team(conn, jid, O._slug(st.get("team") or "") or None,
                              st.get("season") or "")
        finally:
            conn.close()
        return out
    except Exception as e:  # noqa: BLE001 - see the docstring
        return {"error": f"{type(e).__name__}: {e}"}


def refresh_club() -> dict:
    """The ladder and squads into the store after a club or roster edit.

    Without it the store only caught up when the next match finished, so the
    Data pages showed a renamed team under its old name and a new squad not at
    all. Never fatal, for the same reason as ingest: the files already saved.
    """
    try:
        import db  # noqa: PLC0415

        conn = db.connect(str(db_path()))
        try:
            return db.ingest_org(conn, org(), roster())
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001 - see ingest's docstring
        return {"error": f"{type(e).__name__}: {e}"}


def _club():
    """The club, with any team that exists in the data but not yet in the club
    registered on the way past.

    Seeded on READ, never migrated: an install that predates org.py has its
    team strings on the jobs and on the roster filenames, and rewriting those
    to introduce a club is a batch job that can half-run. Same reasoning as
    _defaults. The seed is persisted only when it actually added something.
    """
    import org as O  # noqa: PLC0415

    c = O.Club.load(str(org_path()))
    # Minted once and never re-derived, for the same reason a team keeps its id
    # through a rename: it is the foreign key every team, coach and squad row
    # in the store hangs off, so re-slugging it when the club is finally given
    # a name would orphan the whole ladder.
    dirty = not c.club_id
    c.club_id = c.club_id or "our-club"
    names = [st["team"] for st in all_jobs()]
    for p in _roster_files():
        try:
            names.append(json.loads(p.read_text()).get("team") or p.stem)
        except (OSError, json.JSONDecodeError):
            names.append(p.stem)
    if c.seed(names) or dirty:
        c.save(str(org_path()))
    return c


def org() -> dict:
    """The club as the UI reads it: the ladder, the staff, and the size of each."""
    c = _club()
    squad: dict[str, int] = {}
    for p in _roster_files():
        try:
            squad[p.stem] = len(json.loads(p.read_text()).get("players") or {})
        except (OSError, json.JSONDecodeError):
            squad[p.stem] = 0
    played: dict[str, int] = {}
    for st in all_jobs():
        played[_slug(st["team"])] = played.get(_slug(st["team"]), 0) + 1
    return {
        "club": {"club_id": c.club_id, "name": c.name, "season": c.season},
        "teams": [{**t.__dict__, "players": squad.get(t.team_id, 0),
                   "matches": played.get(t.team_id, 0)} for t in c.ladder()],
        "coaches": [{**co.__dict__,
                     "team_ids": [t.team_id for t in c.teams_of(co.coach_id)]}
                    for co in sorted(c.coaches.values(), key=lambda x: x.name.lower())],
    }


def save_org(payload: dict) -> dict:
    """Upsert the club from what the browser sends, and return the canonical
    view - team ids are server-minted, like roster player ids, because the id
    is the join to the squad file and a client-invented one would orphan it.

    A team missing from `teams` is REMOVED FROM THE LADDER but its
    `rosters/<team_id>.json` is deliberately left on disk. Deleting a squad and
    a season of track bindings because someone removed a row is not undoable
    from the UI, and re-adding the team by the same name picks the file back up.
    """
    import org as O  # noqa: PLC0415

    c = _club()
    club = payload.get("club") or {}
    if isinstance(club, dict):
        if club.get("name") is not None:
            # The name is display only. `club_id` is minted once in _club and
            # never follows a rename - see the comment there.
            c.name = str(club["name"])[:80].strip()
        if club.get("season") is not None:
            c.season = str(club["season"])[:20].strip()

    coaches = payload.get("coaches")
    if isinstance(coaches, list):
        c.coaches = {}
        for x in coaches[:200]:
            if isinstance(x, dict) and str(x.get("name") or "").strip():
                c.add_coach(str(x["name"])[:60].strip(),
                            coach_id=str(x.get("coach_id") or "")[:60] or None,
                            role=str(x.get("role") or "")[:40],
                            email=str(x.get("email") or "")[:120])

    teams = payload.get("teams")
    if isinstance(teams, list):
        keep = set()
        for x in teams[:200]:
            if not isinstance(x, dict) or not str(x.get("name") or "").strip():
                continue
            t = c.add_team(
                str(x["name"])[:60].strip(),
                team_id=str(x.get("team_id") or "")[:60] or None,
                age_band=(str(x["age_band"])[:12] if x.get("age_band") is not None else None),
                gender=str(x.get("gender") or "")[:10],
                level=str(x.get("level") or "")[:8],
                season=(str(x["season"])[:20] if x.get("season") is not None else None),
                coach_ids=[str(i)[:60] for i in (x.get("coach_ids") or [])
                           if str(i) in c.coaches][:12],
            )
            keep.add(t.team_id)
        c.teams = {k: v for k, v in c.teams.items() if k in keep}

    c.save(str(org_path()))
    refresh_club()
    return org()


_FEED_CACHE: tuple | None = None


def season_feed() -> dict:
    """The whole library's entered feed as one fact table - season.py's build.

    Reading every match.json is ~19 MB of parse, so it is cached on the set of
    (job id, mtime): a new match or a re-run invalidates it, and nothing else
    can. No TTL - the input is files on this machine, so staleness is
    observable rather than guessed at. It lived as a global in app.py, where
    the agent's squad tools could not reach it and would have re-parsed the
    library on every call.
    """
    global _FEED_CACHE
    import season as S  # noqa: PLC0415

    stamp = tuple(sorted(
        (d.name, (d / "match.json").stat().st_mtime)
        for d in (JOBS_DIR.iterdir() if JOBS_DIR.exists() else [])
        if (d / "match.json").is_file()
    ))
    if _FEED_CACHE is not None and _FEED_CACHE[0] == stamp:
        return _FEED_CACHE[1]
    out = S.build(JOBS_DIR)
    _FEED_CACHE = (stamp, out)
    return out


def season(team: str | None = None) -> dict:
    """Every finished match for a team, joined into per-player season rows.

    This is the season join `roster.tracks_for` describes, executed. It exists
    server-side rather than in the browser for one reason that is not
    performance: the browser cannot do it at all. The bindings that say which
    TRACK in match 12 is which HUMAN live in the roster file and are not on any
    other route, so a frontend that fetched thirty match.json files could only
    join on (team, shirt number) - which merges two players the week a shirt
    changes and splits one the week identity missed the number.

    ONLY `source == "manual"` bindings count. A machine guess is not a season
    profile: sender accuracy is 9.1% against a 4.5% chance floor (see
    lib/report.ts), so an auto-bound row would put one player's running under
    another player's name, in a document that gets sent to that player. Rows
    with no confirmed binding are counted in `coverage` and dropped, so the
    caller can say what share of the match this rests on instead of presenting
    a partial total as a whole one.

    Everything summed here is an ABSOLUTE distance/speed figure, so the
    aggregate carries the weakest tier of its parts - provisional. Nothing in
    this function upgrades a number by adding it to another one.
    """
    try:
        import roster as R  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - a season without a roster is empty, not an error
        return {"team": team or "", "matches": [], "players": [],
                "coverage": {"bound": 0, "total": 0}}

    want = _slug(team) if team else None
    squad: dict[str, dict] = {}
    bindings: dict[str, dict[int, str]] = {}
    for path in _roster_files():
        try:
            r = R.Roster.load(str(path))
        except Exception:  # noqa: BLE001 - a corrupt roster must not kill the page
            continue
        if want is not None and _slug(r.team) != want:
            continue
        for pl in r.players.values():
            squad[pl.player_id] = {"id": pl.player_id, "name": pl.name,
                                   "number": pl.number, "position": pl.position,
                                   "team": r.team}
        for mid, tb in r.bindings.items():
            for tid, b in tb.items():
                if b.get("source") == "manual":
                    bindings.setdefault(mid, {})[int(tid)] = b["player_id"]

    matches: list[dict] = []
    rows: dict[str, dict] = {}
    bound = total = 0

    for st in all_jobs(team):
        mid = st["id"]
        mpath = job_dir(mid) / "match.json"
        try:
            mtime = mpath.stat().st_mtime
        except OSError:
            continue  # no report yet: the match has not finished
        fix = _fixture(mpath) or {}
        matches.append({
            "id": mid,
            "name": st.get("display_name") or mid,
            "created": st.get("created"),
            "home_name": fix.get("home"),
            "away_name": fix.get("away"),
            "score": fix.get("score"),
        })
        by_track = bindings.get(mid, {})
        for tid, side, jersey, mins, km, peak in _season_rows_cached(str(mpath), mtime):
            total += 1
            pid = by_track.get(tid)
            if not pid or pid not in squad:
                continue
            bound += 1
            row = rows.setdefault(pid, {**squad[pid], "matches": 0, "minutes": 0.0,
                                        "distance_km": 0.0,
                                        # WITHHELD, never summed - null, not 0 (see the cache above).
                                        "sprints": None,
                                        "top_speed_kmh": None, "by_match": {}})
            row["matches"] += 1
            row["minutes"] += mins
            row["distance_km"] += km
            # A season peak is the best single reading, never a sum or a mean -
            # and a refused match is not a candidate for it. All-refused stays
            # None, which the UI renders as "refused" rather than 0.0.
            if peak is not None:
                row["top_speed_kmh"] = (peak if row["top_speed_kmh"] is None
                                        else max(row["top_speed_kmh"], peak))
            row["by_match"][mid] = {"minutes": round(mins, 1),
                                    "distance_km": round(km, 2),
                                    "sprints": None, "top_speed_kmh": peak,
                                    "side": side, "jersey": jersey}

    out = []
    for row in rows.values():
        mins = row["minutes"]
        out.append({**row,
                    "minutes": round(mins, 1),
                    "distance_km": round(row["distance_km"], 2),
                    # Per 90 and not a total, for the reason report.ts already
                    # gives: a total ranks whoever played longest.
                    "km_per90": round(row["distance_km"] / mins * 90, 2) if mins else 0.0,
                    "sprints_per90": None})
    out.sort(key=lambda r: (-r["matches"], -r["km_per90"]))
    matches.sort(key=lambda m: m.get("created") or 0)
    return {"team": team or "", "matches": matches, "players": out,
            "coverage": {"bound": bound, "total": total}}


def _rebind(jid: str, assign: dict, undo: list) -> None:
    """Mirror assignments into roster bindings (source manual, confidence 1.0).

    Binds an assign entry's `player` only in the job's own team roster; unbinds ids in `undo`
    and assignments without a player in every roster file. Unknown players and unreadable
    roster files are skipped silently. docs/reference/identity.md#roster
    """
    try:
        import roster as R  # noqa: PLC0415
    except Exception:
        return
    # An assignment without a player (cleared, or a number only) unbinds, like an undo: the binding
    # mirrors the current assignment. Binds go only into the job's OWN team roster - player ids are
    # name slugs and collide across teams (season() filters the same way); unbinds go everywhere.
    drop = {int(t) for t in undo or []} | {int(t) for t, a in (assign or {}).items() if not (a or {}).get("player")}
    team = _slug(load(jid).get("team") or DEFAULT_TEAM)
    for path in _roster_files():
        try:
            r = R.Roster.load(str(path))
        except Exception:  # noqa: BLE001 - a corrupt roster must not fail a match
            continue
        changed = False
        for tid in drop:
            if r.bindings.get(jid, {}).pop(tid, None) is not None:
                changed = True
        for tid, a in (assign or {}).items():
            pid = (a or {}).get("player")
            if pid and pid in r.players and _slug(r.team) == team:
                r.bind(jid, int(tid), pid, 1.0, "manual")
                changed = True
        if changed:
            r.save(str(path))


# ------------------------------------------------------------------ selfcheck

def _api_selfcheck(cal_doc: dict) -> None:
    """The routes added on 2026-08-04 - /video, /tracks.json, /retry - plus the
    conditional-GET helper.

    They had no in-repo test: they were verified once by a throwaway script that
    no longer exists. Range serving, the 409 on a half-uploaded file and the id
    validation are all things that break silently under a dependency bump, and
    the whole review page is dead the moment they do.
    """
    from fastapi.testclient import TestClient  # noqa: PLC0415

    import app as _app  # noqa: PLC0415 - pulls fastapi/starlette

    # app.py imports jobs a second time (this module is __main__), so it holds a
    # different module object. FUTTRAFFIC_DATA is set before that import, so the
    # two agree on where the jobs live - pinned rather than assumed.
    assert _app.jobs.JOBS_DIR == JOBS_DIR, "app.py is reading a different data dir"
    # NOT `with TestClient(...)`: entering it runs the lifespan, which starts the
    # pump loop and would spawn real workers against these fixtures.
    c = TestClient(_app.app)

    j = create("range.mp4", None, ".mp4", venue="Westfield Park", team="Westfield")
    jid, d = j["id"], job_dir(j["id"])
    body = bytes(range(256)) * 8          # 2 KB, every offset distinguishable
    (d / "video.mp4").write_bytes(body)
    (d / "tracks.json").write_text(json.dumps({"fps": 25.0, "detections": []}))

    # --- a partial upload is a truncated video, and a browser handed one plays
    # a prefix and then stalls with no error. 409 says so instead.
    assert c.get(f"/api/matches/{jid}/video").status_code == 409
    j["upload_complete"] = True
    save(j)

    # --- RANGE. Seeking is dead without it, and "it 200s the whole file" looks
    # fine in a browser right up to the first drag of the scrub bar.
    r = c.get(f"/api/matches/{jid}/video", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206, r.status_code
    assert r.content == body[10:20], r.content            # exact bytes, not just a length
    assert r.headers["content-range"] == f"bytes 10-19/{len(body)}", r.headers
    r = c.get(f"/api/matches/{jid}/video", headers={"Range": "bytes=2040-"})
    assert r.status_code == 206 and r.content == body[2040:], r.status_code
    r = c.get(f"/api/matches/{jid}/video", headers={"Range": f"bytes={len(body)}-{len(body) + 9}"})
    assert r.status_code == 416, f"past EOF must be 416, got {r.status_code}"

    # --- id validation. job_dir() is the ONLY thing between a client-supplied
    # id and the filesystem, and every one of these routes goes through it.
    for evil in ("..", "....", "%2e%2e", "/etc/passwd", "0" * 31, "g" * 32, ""):
        for route in ("video", "tracks.json", "match.json"):
            got = c.get(f"/api/matches/{evil}/{route}").status_code
            assert got in (400, 404, 405), f"{evil!r}/{route} -> {got}"
    gone = new_id()
    assert c.get(f"/api/matches/{gone}/video").status_code == 404
    assert c.get(f"/api/matches/{gone}/tracks.json").status_code == 404
    assert c.post(f"/api/matches/{gone}/retry").status_code == 404

    # --- CONDITIONAL GET. Starlette's FileResponse sets an ETag and then never
    # looks at If-None-Match, so /review re-downloaded a 60 MB tracks.json on
    # every load. Checked on all three files the helper is wired to.
    (d / "match.json").write_text(json.dumps({"players": []}))
    (d / "calibration.json").write_text(json.dumps(cal_doc))
    for route in ("tracks.json", "match.json", "calibration"):
        url = f"/api/matches/{jid}/{route}"
        r = c.get(url)
        tag = r.headers.get("etag")
        assert r.status_code == 200 and tag, (route, r.status_code, r.headers)
        r2 = c.get(url, headers={"If-None-Match": tag})
        assert r2.status_code == 304, f"{route}: re-sent the whole body"
        assert not r2.content and r2.headers["etag"] == tag, r2.headers
        # a tag from some other version must still get the file
        assert c.get(url, headers={"If-None-Match": '"stale"'}).status_code == 200
    # ...and the tag moves when the file does, or 304 becomes a lie. These three
    # are rewritten, never mutated, so mtime+size always changes with them.
    old = c.get(f"/api/matches/{jid}/match.json").headers["etag"]
    time.sleep(0.01)
    (d / "match.json").write_text(json.dumps({"players": [{"id": 1}]}))
    assert c.get(f"/api/matches/{jid}/match.json").headers["etag"] != old
    assert c.get(f"/api/matches/{jid}/match.json",
                 headers={"If-None-Match": old}).status_code == 200

    # --- RETRY. MAX_ATTEMPTS is what stops a deterministic failure looping on
    # the GPU forever; without a retry the only recovery is re-uploading a file
    # already on disk. A human asking again resets the counter.
    (d / "calibration.json").unlink()
    j = load(jid)
    j.update(status="failed", error="worker died 3x during tracking", attempts=3, identity_done=True)
    save(j)
    r = c.post(f"/api/matches/{jid}/retry")
    assert r.status_code == 200, r.text
    # identity_done without tracks.id.json = identity failed; a retry must re-attempt it
    assert "identity_done" not in load(jid), "retry left identity_done set: identity never re-runs"
    # no calibration on disk -> back to the human gate, not the GPU queue
    assert r.json()["status"] == "calibrating", r.json()
    assert r.json()["attempts"] == 0 and r.json()["error"] is None, r.json()
    (d / "calibration.json").write_text(json.dumps(cal_doc))
    assert c.post(f"/api/matches/{jid}/retry").json()["status"] == "queued"

    # ...and it refuses to touch a job a worker is currently inside, which would
    # give two processes one job directory.
    j = load(jid)
    j["status"] = "tracking"
    save(j)
    assert c.post(f"/api/matches/{jid}/retry").status_code == 409

    # ...or one whose upload never finished: there is nothing to run.
    half = create("half.mp4", 10 ** 9, ".mp4", venue="Westfield Park")
    assert c.post(f"/api/matches/{half['id']}/retry").status_code == 409

    # --- the team filter is the same list, scoped.
    ids = {m["id"] for m in c.get("/api/matches").json()}
    mine = {m["id"] for m in c.get("/api/matches", params={"team": "westfield"}).json()}
    assert jid in mine and mine < ids, (len(mine), len(ids))
    assert c.get("/api/matches", params={"team": "no such club"}).json() == []


#: Self-check cases that FAIL today and are reported rather than fixed here; each one
#: prints on the "jobs.py ok" line every run. A listed case that starts passing FAILS
#: the self-check until it is removed, so the list cannot go stale.
SELFCHECK_KNOWN_BUGS: dict[str, str] = {}   # derive_tracks_two_splits fixed 2026-09-28


def _selfcheck() -> None:
    """Two things must hold: a job survives the machine dying under it, and a
    job that cannot succeed says so instead of sitting in a running state
    forever. Everything else in this file is bookkeeping."""
    global DATA_DIR, JOBS_DIR, ROSTER_PATH
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="futjobs-"))
    known_bugs: list[str] = []
    DATA_DIR, JOBS_DIR, ROSTER_PATH = tmp, tmp / "jobs", tmp / "roster.json"
    # The store and the club follow DATA_DIR by being FUNCTIONS of it, so this
    # rebind is all they need. It was not always so: they were constants
    # computed at import, so a self-check run wrote its throwaway match into the
    # developer's real data/match.db and re-ingested an empty club over the live
    # one, emptying `squad` without a word. Assert it rather than trust it.
    assert db_path().parent == tmp, db_path()
    assert org_path().parent == tmp, org_path()
    os.environ["FUTTRAFFIC_DATA"] = str(tmp)

    # --- THUMBNAIL CACHE KEYS. The key carries the video's mtime, which is the
    # only thing making a re-uploaded video self-heal. If that ever stops being
    # true, every match card shows a still of footage that is no longer there.
    tj = create("thumbs", None, ".mp4")
    tv = job_dir(tj["id"]) / tj["video"]
    tv.write_bytes(b"not really a video, but it has an mtime")
    k1 = thumb_path(tj["id"], 1500, 240)
    assert k1.parent.name == "thumbs" and k1.suffix == ".jpg", k1
    assert k1 == thumb_path(tj["id"], 1500, 240), "same video, same key"
    assert k1 != thumb_path(tj["id"], 1500, 480), "width is part of the key"
    assert k1 != thumb_path(tj["id"], 0, 240), "frame is part of the key"
    os.utime(tv, (0, 12345))
    assert k1 != thumb_path(tj["id"], 1500, 240), "a re-upload must miss the cache"
    # A feed-imported match has no video at all: ValueError, which the route
    # turns into a 404. Before this guard it was `job_dir(jid) / None`, a
    # TypeError, and every such card answered its thumbnail with a 500.
    nv = create("no video", None, ".mp4")
    st_nv = load(nv["id"])
    st_nv["video"] = None
    save(st_nv)
    try:
        thumb_path(nv["id"], 1500, 240)
        raise AssertionError("a job with no video must refuse, not TypeError")
    except ValueError:
        pass

    def wait(jid, want, limit=60):
        for _ in range(limit * 10):
            pump()
            st = load(jid)
            if st["status"] in want:
                return st
            time.sleep(0.1)
        raise AssertionError(f"job stuck in {load(jid)['status']}, wanted {want}")

    # --- THE ACCEPT/CONSUME CONTRACT. Whatever the API says yes to, the worker
    # must be able to load. Validating with anything weaker than the worker's
    # own load turns a 400 into a ValueError hours into a GPU job.
    cal_doc = {"pitch": {"length": 105.0, "width": 68.0},
               "H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
               "points": {}, "rms_error_m": 0.0, "outliers": []}
    probe = tmp / "cal.json"
    probe.write_text(json.dumps(cal_doc))
    assert load_calibration(probe) is not None
    # `remap` is ACCEPTED: a stitched panorama flattened by a dense remap and then
    # projected by one homography is a projection the whole pipeline consumes, and
    # stitched rigs (Veo, multi-camera) are likely most real-world footage.
    # `panoramic` stays refused - its global H is a single fit with ~6 m median
    # position error, so accepting it swaps a loud failure for a quiet wrong number.
    probe.write_text(json.dumps({**cal_doc, "projection_type": "remap"}))
    assert load_calibration(probe) is not None, "remap must be accepted"
    for bad_doc, why in (({**cal_doc, "projection_type": "panoramic"}, "panoramic"),
                         ({**cal_doc, "H": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]}, "degenerate H")):
        probe.write_text(json.dumps(bad_doc))
        try:
            load_calibration(probe)
            raise AssertionError(f"accepted a calibration the worker cannot use: {why}")
        except AssertionError:
            raise
        except Exception:
            pass
    # Collinear clicks that det(H) let through (review of the calibration section): 3 of 4 on
    # the halfway line; and 5 on it plus one box corner that LMEDS throws away as the outlier.
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from pitch import Calibration, Pitch  # noqa: PLC0415
    true_H = np.array([[12.0, 3.0, 300.0], [0.0, 9.0, 120.0], [0.0, 0.006, 1.0]])
    lm = Pitch().landmarks()
    rng = np.random.default_rng(0)

    def clicked(names, noise=0.5):
        px = cv2.perspectiveTransform(np.array([[lm[n] for n in names]], float), true_H)[0]
        return {n: tuple(p + rng.normal(0, noise, 2)) for n, p in zip(names, px)}
    for names in (["half_b", "centre", "half_t", "corner_bl"],
                  ["half_b", "circle_b", "centre", "circle_t", "half_t", "pen_l_br"]):
        probe.write_text(json.dumps(Calibration.solve(Pitch(), clicked(names)).to_dict(), default=float))
        try:
            load_calibration(probe)
            raise AssertionError(f"accepted collinear clicks {names}")
        except ValueError:
            pass
    probe.write_text(json.dumps(Calibration.solve(Pitch(), clicked(list(lm))).to_dict(), default=float))
    assert load_calibration(probe) is not None, "a full, well-spread click set must be accepted"

    # --- a job whose tracking stage is already on disk runs stats and finishes.
    st = create("demo.mp4", None, ".mp4")
    d = job_dir(st["id"])
    fps = 25.0
    dets = [{"f": i, "id": 1, "team": "home", "jersey": None,
             "x": 5.0 * (i / fps), "y": 34.0} for i in range(int(20 * fps) + 1)]
    dets += [{"f": i, "id": 2, "team": "away", "jersey": None,
              "x": 40.0, "y": 20.0} for i in range(int(20 * fps) + 1)]
    (d / "tracks.raw.json").write_text(json.dumps({"fps": fps, "detections": dets}))
    (d / "calibration.json").write_text(json.dumps(
        {"pitch": {"length": 105.0, "width": 68.0}, "H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
         "points": {}, "rms_error_m": 0.0, "outliers": []}))
    st["upload_complete"] = True
    save(st)
    st = wait(st["id"], {"done", "failed"})
    assert st["status"] == "done", st
    m = json.loads((d / "match.json").read_text())
    assert len(m["players"]) == 2, m["players"]

    # --- SURVIVES A RESTART. Fake a worker killed mid-tracking: state says
    # tracking, the pid is gone, no heartbeat. The pump must resume it from
    # the artifact already on disk and drive it to done.
    st["status"] = "tracking"
    st["pid"] = 999_999            # never a live pid
    st["updated"] = time.time() - DEAD_AFTER_S - 5
    st["attempts"] = 0
    (d / "match.json").unlink()
    save(st)
    pump()
    assert load(st["id"])["status"] == "queued", load(st["id"])
    st = wait(st["id"], {"done", "failed"})
    assert st["status"] == "done", st
    assert (d / "match.json").exists()

    # --- corrections replay: merge two tracks, and the rebuilt match must
    # show one player, not two.
    apply_corrections(st["id"], {"merge": [[1, 2]], "operator_seconds": 42.0})
    st = wait(st["id"], {"done", "failed"})
    assert st["status"] == "done", st
    m = json.loads((d / "match.json").read_text())
    assert len(m["players"]) == 1, m["players"]
    assert load(st["id"])["operator_seconds"] == 42.0

    # ...and it is a merge, not a delete: the raw model output is untouched.
    raw = json.loads((d / "tracks.raw.json").read_text())
    assert len({x["id"] for x in raw["detections"]}) == 2

    # --- undo puts a wrong merge back. Corrections replay from raw, so the
    # second track has to reappear intact.
    apply_corrections(st["id"], {"undo": [2]})
    st = wait(st["id"], {"done", "failed"})
    m = json.loads((d / "match.json").read_text())
    assert len(m["players"]) == 2, m["players"]

    # --- split: one bad tracklet becomes two ids at the given frame, and only
    # from that frame on. The replay is a single pass now, so this also guards
    # the split -> merge -> assign ordering that pass has to preserve.
    apply_corrections(st["id"], {"split": [{"track": 1, "at_frame": 250}]})
    st = wait(st["id"], {"done", "failed"})
    tj = json.loads((d / "tracks.json").read_text())
    home = {(int(x["f"]) >= 250, int(x["id"])) for x in tj["detections"] if x["team"] == "home"}
    before = {i for late, i in home if not late}
    after = {i for late, i in home if late}
    assert before == {1} and after and after != before, sorted(home)
    assert len({x["id"] for x in tj["detections"]}) == 3, "split did not mint a new id"
    apply_corrections(st["id"], {"undo": [1]})
    st = wait(st["id"], {"done", "failed"})
    tj = json.loads((d / "tracks.json").read_text())
    assert len({x["id"] for x in tj["detections"]}) == 2, "undo did not put the split back"

    # --- TWO splits of one track, at 100 then 250: three ids, and frames >= 250 carry
    # the SECOND minted id (legacy entries, no stored new_id). _derive_tracks called directly.
    two = tmp / "two-splits"
    two.mkdir()
    (two / "tracks.raw.json").write_text(json.dumps({"fps": fps, "detections": [
        {"f": f, "id": 7, "team": "home", "x": 1.0, "y": 1.0} for f in range(0, 400, 10)]}))
    (two / "corrections.json").write_text(json.dumps({"split": [{"track": 7, "at_frame": 100},
                                                                {"track": 7, "at_frame": 250}]}))
    _derive_tracks(two)
    got = {int(x["f"]): int(x["id"]) for x in json.loads((two / "tracks.json").read_text())["detections"]}
    ok = ({got[f] for f in got if f < 100} == {7} and {got[f] for f in got if 100 <= f < 250} == {8}
          and {got[f] for f in got if f >= 250} == {9})     # 8 and 9: minted in split order
    assert ok, f"two splits of one track: frame -> id {sorted(got.items())[::5]}"

    # --- A SELF OR CYCLIC MERGE hung the replay forever. A saved one must terminate; a new one is refused.
    import signal  # noqa: PLC0415

    def _hung(*_):
        raise AssertionError("_derive_tracks hung on a self/cyclic merge")
    for bad in ([[7, 7]], [[7, 8], [8, 7]]):
        (two / "corrections.json").write_text(json.dumps({"merge": bad}))
        old = signal.signal(signal.SIGALRM, _hung)
        signal.alarm(5)
        try:
            _derive_tracks(two)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    cj = create("corr.mp4", None, ".mp4")       # no calibration: apply_corrections parks it, the pump never runs it
    cd = job_dir(cj["id"])
    raw = [{"f": f, "id": t, "team": "home", "x": 1.0, "y": 1.0, "jersey": 4, "jersey_conf": 0.9}
           for t in (1, 2, 5, 7) for f in range(0, 400, 10)]
    (cd / "tracks.raw.json").write_text(json.dumps({"fps": fps, "detections": raw}))
    apply_corrections(cj["id"], {"merge": [[1, 2]]})
    for bad in ([[5, 5]], [[2, 1]]):            # self; and 1 already holds 2, so 2 <- 1 closes a cycle
        try:
            apply_corrections(cj["id"], {"merge": bad})
            raise AssertionError(f"merge {bad} accepted")
        except ValueError:
            pass
    assert read_corrections(cj["id"])["merge"] == [[1, 2]], read_corrections(cj["id"])

    def ids_of(t, lo=0):
        tj = json.loads((cd / "tracks.json").read_text())["detections"]
        return {int(x["id"]) for x in tj if x["_raw"] == t and x["f"] >= lo}
    for x in raw:
        x["_raw"] = x["id"]                     # traceability for ids_of only
    (cd / "tracks.raw.json").write_text(json.dumps({"fps": fps, "detections": raw}))

    # --- SPLIT AFTER MERGE: 2 is shown inside 1, so splitting 1 at 150 moves 2's frames >= 150 too.
    apply_corrections(cj["id"], {"split": [{"track": 1, "at_frame": 150}]})
    s1 = read_corrections(cj["id"])["split"][0]["new_id"]
    assert ids_of(1, 150) == ids_of(2, 150) == {s1} and ids_of(2) - {s1} == {1}, (ids_of(1), ids_of(2))

    # --- SPLIT IDS ARE STORED: undoing an earlier split leaves a later one's id and its assignment.
    apply_corrections(cj["id"], {"split": [{"track": 7, "at_frame": 100}]})
    apply_corrections(cj["id"], {"split": [{"track": 5, "at_frame": 200}]})
    s5 = read_corrections(cj["id"])["split"][-1]["new_id"]
    apply_corrections(cj["id"], {"assign": {str(s5): {"jersey": 11}}})
    apply_corrections(cj["id"], {"undo": [7]})
    tj = json.loads((cd / "tracks.json").read_text())["detections"]
    late5 = [x for x in tj if x["_raw"] == 5 and x["f"] >= 200]
    assert {x["id"] for x in late5} == {s5}, "undoing split 7 renumbered split 5"
    # an operator's number says so, and the OCR confidence no longer rides on it
    assert all(x["jersey"] == 11 and x["jersey_source"] == "operator" and "jersey_conf" not in x
               for x in late5), late5[0]

    # --- UNDO ON A SPLIT'S new_id removes the split: frames go back to the original id, assignment gone.
    apply_corrections(cj["id"], {"undo": [s5]})
    c = read_corrections(cj["id"])
    assert all(int(sp.get("new_id", -1)) != s5 for sp in c["split"]), c["split"]
    assert str(s5) not in c["assign"], c["assign"]
    assert ids_of(5) == {5}, f"undo of new_id {s5} left track 5 as {ids_of(5)}"

    # --- identity is best effort and must never fail the match. An unreadable
    # video makes identity.scan_video raise SystemExit; the job still finishes,
    # it just hands the operator unmerged tracks instead of proposals.
    idj = create("corrupt.mp4", None, ".mp4")
    dj = job_dir(idj["id"])
    (dj / "video.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 512)
    (dj / "tracks.raw.json").write_text(json.dumps({"fps": fps, "detections": dets}))
    (dj / "calibration.json").write_text((d / "calibration.json").read_text())
    idj["upload_complete"] = True
    save(idj)
    idj = wait(idj["id"], {"done", "failed"})
    assert idj["status"] == "done", idj
    assert idj.get("identity_note"), "identity failure should be recorded, not silent"

    # --- IDENTITY MERGES IN METRES. This is the regression that made the whole
    # assisted-identity stage a no-op in production: the strategy was built with
    # no calibration, identity._positions used raw pixels, and MAX_SPEED (9.5
    # m/s) then rejected every candidate merge because 2 metres of movement is
    # ~78 pixels. Two asserts on purpose - the spy fails loudly on the exact
    # mistake, the merge assert fails on any other way of reintroducing it.
    import identity as _id  # noqa: PLC0415

    px_m = 39.0                       # measured scale of the real rig
    seen: dict = {}
    real_get = _id.get_identity_strategy

    def _spy(name, **kw):
        seen.update(kw)
        return real_get(name, **kw)

    def _px(tid, f0, f1, x0_m, vx):
        return [{"f": f, "id": tid, "team": "home", "jersey": None,
                 "x": (x0_m + vx * (f - f0) / fps) * px_m, "y": 30.0 * px_m}
                for f in range(f0, f1 + 1, 5)]

    mj = create("pixels.mp4", None, ".mp4")
    md = job_dir(mj["id"])
    # One player, 1 s occlusion at 2 m/s: 2 m apart in metres (mergeable),
    # 78 px apart in pixels (rejected as a teleport).
    (md / "tracks.raw.json").write_text(json.dumps(
        # track 1 runs 20 m -> 44 m over 12 s; track 2 picks it up 2 m further
        # on after a 1 s gap. 2 m/s in metres, 78 px/s in pixels.
        {"fps": fps, "detections": _px(1, 0, 300, 20.0, 2.0) + _px(2, 325, 625, 46.0, 2.0)}))
    (md / "calibration.json").write_text(json.dumps(
        {**cal_doc, "H": [[1 / px_m, 0, 0], [0, 1 / px_m, 0], [0, 0, 1]]}))
    mcal = load_calibration(md / "calibration.json")
    _id.get_identity_strategy = _spy
    try:
        _try_identity(md, mj, mcal)
    finally:
        _id.get_identity_strategy = real_get
    assert seen.get("cal") is not None, (
        "identity strategy built without cal=: merges run in pixel space and "
        "MAX_SPEED silently rejects all of them (identity._positions)")
    # A degraded run (no OCR backend, facing gate off) finishes, and its warnings reach identity_note.
    class _Degraded:
        report = {"warnings": ["no OCR backend loaded ('auto'): no numbers read"]}

        def assign(self, tracks, video):
            return {}
    _id.get_identity_strategy = lambda name, **kw: _Degraded()
    try:
        wj = {**mj, "identity_note": None}
        _try_identity(md, wj, mcal)
    finally:
        _id.get_identity_strategy = real_get
    assert "no OCR backend" in (wj.get("identity_note") or ""), wj.get("identity_note")
    (md / "tracks.id.json").unlink()
    _id.get_identity_strategy = _spy
    try:
        _try_identity(md, mj, mcal)
    finally:
        _id.get_identity_strategy = real_get
    merged = json.loads((md / "tracks.id.json").read_text())
    assert len({x["id"] for x in merged["detections"]}) == 1, (
        f"pixel-space merge: {sorted({x['id'] for x in merged['detections']})} "
        f"- one player's two tracklets stayed apart")

    # ...and the job that owns them finishes as ONE player, not two fragments
    # that MIN_MINUTES would have deleted along with their distance.
    mj["upload_complete"] = True
    save(mj)
    mj = wait(mj["id"], {"done", "failed"})
    assert mj["status"] == "done", mj
    assert len(json.loads((md / "match.json").read_text())["players"]) == 1

    # --- THE SCENE PRIOR IS REACHABLE. track.run() only builds the pitch mask
    # and player-height prior when it is handed a calibration; the old flow
    # called it with none and asked for landmarks afterwards, so the prior was
    # structurally unreachable in the product (prior_rejected was 0 on every
    # real run). Spy on the call rather than on a real video: what regresses is
    # the argument, not the tracker.
    import track as _tr  # noqa: PLC0415

    calls: dict = {}

    def _fake_run(video, out, **kw):
        calls.update(kw)
        Path(out).write_text(json.dumps(
            {"fps": fps, "detections": _px(7, 0, 300, 20.0, 2.0),
             "_perf": {"prior_rejected": 0}}))
        return {"_perf": {"prior_rejected": 0}}

    pj = create("prior.mp4", None, ".mp4")
    pd = job_dir(pj["id"])
    (pd / "video.mp4").write_bytes(b"\x00" * 32)
    (pd / "calibration.json").write_text(json.dumps(
        {**cal_doc, "H": [[1 / px_m, 0, 0], [0, 1 / px_m, 0], [0, 0, 1]]}))
    real_run, _tr.run = _tr.run, _fake_run
    try:
        _pipeline(pj)
    finally:
        _tr.run = real_run
    assert calls.get("calib") == str(pd / "calibration.json"), (
        f"track.run called without calib=: the pitch mask and player-height "
        f"prior never run, and ~50% of the frame stays in play ({calls})")
    # ...and the rest of the config the Mac path is documented at. Nobody
    # submitting through the app can pass a flag, so every argument NOT asserted
    # here silently reverts to track.py's portable defaults - which for this
    # machine is a 3.7x slower run that cannot survive being killed.
    assert calls.get("stride") == 2, f"the job path lost --stride ({calls})"
    assert calls.get("dets_cache") == str(pd / "dets.cache.json"), (
        f"no detection cache: pump() retries a dead worker from the last "
        f"finished artifact, and tracking's only artifact is the file it writes "
        f"at the very end - so a SIGKILL at frame 60,000 restarts at 0 ({calls})")
    assert calls.get("ball") is True, (
        f"the job path lost ball=: no ball is detected on any job, which is "
        f"what it did until 2026-09-23 ({calls})")
    assert calls.get("detector") in ("coreml", "rfdetr"), calls

    # A BALL-FREE CACHE MUST NOT REACH track.run. Stamped, it raises SystemExit
    # and the job burns both attempts and fails for good; unstamped, it replays
    # ball-free and the job completes having produced nothing. Both were live
    # the moment ball=True landed, so the drop is asserted, not assumed.
    # A FRESH JOB PER STAMP. A resumed job skips the stages it has already
    # finished, so re-running _pipeline on the job above never reaches the
    # tracking call and the assertion would pass against a no-op - which it did,
    # first time of asking.
    import track as _trk  # noqa: PLC0415
    keep = f'{{"settings": {{"ball": true, "cache_version": {_trk.CACHE_VERSION}}}, "frames": {{}}}}'
    for stamp in ('{"settings": {"ball": false}, "frames": {}}',
                  '{"frames": {}}',                       # predates settings entirely
                  '{"settings": {"ball": true}, "frames": {}}',   # ball, but before candidates were stored
                  keep):                                  # this one must SURVIVE
        bj = create("ballcache.mp4", None, ".mp4")
        bd = job_dir(bj["id"])
        (bd / "video.mp4").write_bytes(b"\x00" * 32)
        (bd / "calibration.json").write_text(json.dumps(
            {**cal_doc, "H": [[1 / px_m, 0, 0], [0, 1 / px_m, 0], [0, 0, 1]]}))
        (bd / "dets.cache.json").write_text(stamp)
        # What matters is what track.run SEES, not what is left afterwards: a
        # successful run unlinks the cache itself (the `dets.cache.json` unlink
        # below), so "still there at the end" would be false for every arm.
        seen = {}

        def _fake_run_seeing(*a, **kw):
            seen["cache"] = os.path.exists(kw.get("dets_cache", ""))
            return _fake_run(*a, **kw)

        real_run, _tr.run = _tr.run, _fake_run_seeing
        try:
            _pipeline(bj)
        finally:
            _tr.run = real_run
        if stamp == keep:
            assert seen.get("cache") is True, \
                "a cache detected WITH the ball was dropped - a re-detect nobody asked for"
        else:
            assert seen.get("cache") is False, \
                f"a ball-free detection cache reached track.run ({stamp})"
    assert pj.get("detector") == calls["detector"], (
        f"the detector that actually ran must be recorded on the job, or a run "
        f"that fell back to PyTorch has an unexplained clock ({pj.get('detector')})")
    # ...and a prior that rejected nothing is recorded, not swallowed: make_prior
    # returns None instead of raising, so this is the only trace it left.
    assert "may not have loaded" in (pj.get("scene_prior") or ""), pj.get("scene_prior")
    assert status_view(pj)["scene_prior"] == pj["scene_prior"]
    assert status_view(pj)["detector"] == pj["detector"]

    # --- CALIBRATION IS AN INPUT, NOT A REPORTING DETAIL. With no calibration
    # the job must park BEFORE tracking: tracking without the scene prior burns
    # GPU-hours on the crowd and hands identity pixels it cannot merge.
    gate = create("uncalibrated.mp4", None, ".mp4")
    gd = job_dir(gate["id"])
    (gd / "video.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 512)
    gate["upload_complete"] = True
    save(gate)
    gate = wait(gate["id"], {"calibrating", "failed", "done"})
    assert gate["status"] == "calibrating", gate
    assert not (gd / "tracks.raw.json").exists(), "tracked before it was calibrated"
    assert status_view(gate)["waiting_for"] == "calibration", status_view(gate)
    # and the gate opens the moment one arrives (this video is 512 unreadable
    # bytes, so it fails IN tracking - the point is that it got there at all).
    (gd / "calibration.json").write_text(json.dumps(cal_doc))
    gate["status"] = "queued"
    save(gate)
    gate = wait(gate["id"], {"failed", "done"}, limit=90)
    assert gate["status"] != "calibrating", gate

    # --- FAILS RATHER THAN HANGING. No video, no tracks: the worker cannot
    # do anything. It must end in failed with a message, and stay there.
    bad = create("nope.mp4", None, ".mp4")
    bad["upload_complete"] = True
    save(bad)
    bad = wait(bad["id"], {"failed"}, limit=90)
    assert bad["status"] == "failed", bad
    assert bad["error"] and "video" in bad["error"], bad["error"]
    for _ in range(3):
        pump()
    assert load(bad["id"])["status"] == "failed"  # terminal means terminal

    # --- roster persists across matches and the binding joins back. Without
    # this the operator redoes identity every week and season profiles never
    # join, which is the product.
    save_roster([{"name": "Ana Ruiz", "number": 9, "team": "Westfield"}])
    ana = roster()[0]
    assert ana["name"] == "Ana Ruiz" and ana["team"] == "Westfield", ana
    # _rebind binds into the JOB's team roster only; a same-named player in another club's file
    # carries the same slug id and must not collect this match's tracks.
    save_roster([{"name": "Ana Ruiz", "number": 4, "team": "Eastfield"}])
    import roster as R
    assert ana["id"] in R.Roster.load(str(_roster_path("Eastfield"))).players, "no slug collision to test"
    st = load(st["id"])
    st["team"] = "Westfield"
    save(st)
    apply_corrections(st["id"], {"assign": {"1": {"player": ana["id"], "jersey": 9}}})
    r = R.Roster.load(str(_roster_path("Westfield")))
    assert r.tracks_for(ana["id"]) == [(st["id"], 1)], r.bindings
    assert R.Roster.load(str(_roster_path("Eastfield"))).tracks_for(ana["id"]) == [], "bound into another team"
    # clearing the assignment unbinds it
    apply_corrections(st["id"], {"assign": {"1": None}})
    assert R.Roster.load(str(_roster_path("Westfield"))).tracks_for(ana["id"]) == [], "a cleared assignment stayed bound"
    apply_corrections(st["id"], {"assign": {"1": {"player": ana["id"], "jersey": 9}}})
    _roster_path("Eastfield").unlink()      # the rest of this check is one club
    refresh_club()
    # a shirt-number change must update the player, not create a second one
    save_roster([{"id": ana["id"], "name": "Ana Ruiz", "number": 21, "team": "Westfield"}])
    assert len(roster()) == 1 and roster()[0]["number"] == 21, roster()
    # ...and the store follows the edit without waiting for a match to finish
    import db as D
    conn = D.connect(str(db_path()))
    got = conn.execute("SELECT number FROM squad WHERE player_id = ?", (ana["id"],)).fetchall()
    conn.close()
    assert [g["number"] for g in got] == [21], got

    # --- THE SEASON JOIN. The one thing the browser cannot do for itself: the
    # bindings that name a track live in the roster file, so a season row only
    # exists if `season()` reads them. And it must count ONLY the rows it could
    # actually attribute - a partial total presented as a whole one is the
    # failure this whole module is written around.
    save_roster([{"id": ana["id"], "name": "Ana Ruiz", "number": 21,
                  "team": "Westfield", "position": "CM"}])
    # A season is one club's matches against that club's squad, so the match has
    # to belong to the club whose roster holds the binding (`_rebind` binds only there).
    st = load(st["id"])
    st["team"] = "Westfield"
    save(st)
    apply_corrections(st["id"], {"assign": {"1": {"player": ana["id"], "jersey": 21}}})
    # apply_corrections deletes match.json and re-queues, so the season has
    # nothing to read until the pump has rebuilt the report.
    assert wait(st["id"], {"done", "failed"})["status"] == "done"
    sea = season("Westfield")
    tracked = len(json.loads((job_dir(st["id"]) / "match.json").read_text())["players"])
    assert sea["coverage"] == {"bound": 1, "total": tracked}, sea["coverage"]
    row = next(p for p in sea["players"] if p["id"] == ana["id"])
    assert row["matches"] == 1 and row["position"] == "CM", row
    assert st["id"] in row["by_match"], row
    # per 90, never a raw total - a total ranks whoever played longest
    assert row["km_per90"] == (round(row["distance_km"] / row["minutes"] * 90, 2)
                               if row["minutes"] else 0.0), row
    # Sprints are WITHHELD at season level too: a match.json written before the
    # refusal still carries a count, and the season must not surface or sum it.
    mp = job_dir(st["id"]) / "match.json"
    planted = json.loads(mp.read_text())
    for pl in planted["players"]:
        pl["sprints"] = 7
    mp.write_text(json.dumps(planted))
    os.utime(mp, (time.time() + 5, time.time() + 5))   # the season cache keys on mtime
    row = next(p for p in season("Westfield")["players"] if p["id"] == ana["id"])
    assert row["sprints"] is None and row["sprints_per90"] is None, row
    assert all(m["sprints"] is None for m in row["by_match"].values()), row

    apply_corrections(st["id"], {"undo": [1]})
    r = R.Roster.load(str(_roster_path("Westfield")))
    assert r.tracks_for(ana["id"]) == [], r.bindings
    # an unbound track is not a season row, and is not silently missing either
    assert wait(st["id"], {"done", "failed"})["status"] == "done"
    sea = season("Westfield")
    assert sea["players"] == [] and sea["coverage"]["bound"] == 0, sea["coverage"]
    assert sea["coverage"]["total"] == tracked, sea["coverage"]

    # --- CLIPS SURVIVE REPROCESSING. A coach's clip list is the only thing on
    # this system a human authored, and the whole pipeline re-runs on every
    # correction. If clips lived anywhere the derive step touches, an identity
    # fix would silently delete an evening of tagging.
    cl = save_clip(st["id"], {"name": "Counter", "start": 12.0, "end": 20.0,
                              "tags": ["Transition", "Nope"], "note": "left channel"})
    assert len(cl) == 1 and cl[0]["tags"] == ["Transition"], cl   # unknown tag dropped
    cid = cl[0]["id"]
    save_clip(st["id"], {"name": "Early press", "start": 3.0, "end": 9.0, "tags": ["Press"]})
    assert [c["start"] for c in read_clips(st["id"])] == [3.0, 12.0], read_clips(st["id"])
    # an edit is an upsert on the same id, not a second clip
    save_clip(st["id"], {"id": cid, "name": "Counter (2nd ball)", "start": 12.5, "end": 21.0})
    got = [c for c in read_clips(st["id"]) if c["id"] == cid]
    assert len(got) == 1 and got[0]["name"] == "Counter (2nd ball)", read_clips(st["id"])
    for bad, why in (({"start": 5, "end": 5}, "zero-length"),
                     ({"start": 9, "end": 2}, "out before in"),
                     ({"start": "x", "end": 2}, "not a number")):
        try:
            save_clip(st["id"], bad)
            raise AssertionError(f"accepted a clip that cannot play: {why}")
        except AssertionError:
            raise
        except ValueError:
            pass
    apply_corrections(st["id"], {"exclude": [2]})
    st = wait(st["id"], {"done", "failed"})
    assert len(read_clips(st["id"])) == 2, "reprocessing ate the clips"
    assert delete_clip(st["id"], cid) == read_clips(st["id"])
    assert len(read_clips(st["id"])) == 1
    delete_clip(st["id"], cid)          # idempotent: two tabs, one clip
    assert len(read_clips(st["id"])) == 1

    # --- calibration inheritance. The rules are safety rules, so they are
    # pinned rather than trusted: same ground AND same frame geometry, never
    # chained, and always recorded so the UI can say the homography was not
    # solved here.
    HOME, AWAY, CLUB = "Westfield Park", "Rovers Ground", "Westfield"
    src = create("week 1", None, ".mp4", venue=HOME, team=CLUB)
    src["frame_shape"] = [1920, 1080]
    src["upload_complete"] = True
    save(src)
    (job_dir(src["id"]) / "calibration.json").write_text(json.dumps(cal_doc))

    same = create("week 2", None, ".mp4", venue=HOME.lower(), team=CLUB)
    same["frame_shape"] = [1920, 1080]
    save(same)
    assert inherit_calibration(same) is False, "inherited with no video to measure"
    # _video_shape needs a real file; the recorded shape is what a real job
    # carries by the time this runs, so drive the comparison directly.
    import unittest.mock as _mock
    with _mock.patch(f"{__name__}._video_shape", lambda p: (1920, 1080)):
        # same ground, typed in lower case: one venue, not two.
        assert inherit_calibration(same) is True, "did not inherit a matching calibration"
        assert (job_dir(same["id"]) / "calibration.json").exists()
        assert load(same["id"])["calibration_inherited_from"] == "week 1"
        assert status_view(load(same["id"]))["calibration_inherited_from"] == "week 1"

        # already has one -> no-op, and never overwrites
        assert inherit_calibration(load(same["id"])) is False

        # --- THE REGRESSION. Every club films in 1080p, so frame_shape alone
        # matches an away fixture at a completely different ground. Adopting the
        # home pitch's homography there rescales every distance on the report
        # with no error and nothing looking wrong - the exact silent wrong
        # number this product exists to prevent. Same resolution, same camera,
        # same club, DIFFERENT ground: must not inherit.
        away = create("away day", None, ".mp4", venue=AWAY, team=CLUB)
        save(away)
        assert inherit_calibration(away) is False, (
            "inherited across venues: a 1080p match at another ground took this "
            "pitch's homography and every distance on it is now wrong")
        assert not (job_dir(away["id"]) / "calibration.json").exists()

        # --- THE ACCEPTED-CAMERA PATH. auto_calibrate's whole body sits inside a
        # "never fail the job" wrapper, so a bug on the accepted path (a NameError
        # once did exactly this - review, 2026-09-18) records as a polite refusal
        # and parks every panorama for clicks with nothing red anywhere. Drive it
        # with a stubbed camera: the file must land, the flag must say "auto",
        # and the refusal text must not be an exception's name.
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model" / "engine"))
        import autocal as _ac
        from pitch import Pitch as _Pitch
        pano = create("pano", None, ".mp4", venue=HOME.lower(), team=CLUB)
        save(pano)
        _q = _ac.camera(_Pitch(105.0, 68.0), (4096, 1080), fov_deg=156.0, tilt_deg=19.5, height_m=7.3,
                        behind_m=16.0, along_m=-2.5, pan_deg=1.7)
        _rep = {"accepted": True, "via": "search", "why": None, "hits": 900, "samples": 1351, "line_rms_px": 0.7,
                "camera_centre_m": [51.0, 83.0, 7.5], "f_px_per_rad": 1513.0, "length_m": 105.1, "image_wh": [4096, 1080],
                "hits_by_group": {"touch_far": [100, 210]}}
        with _mock.patch(f"{__name__}._video_shape", lambda p: (4096, 1080)), \
                _mock.patch.object(_ac, "calibrate", lambda *a, **k: (_q, _rep)):
            ok = auto_calibrate(load(pano["id"]))
        st_p = load(pano["id"])
        assert ok is True, st_p.get("calibration_auto")
        assert (job_dir(pano["id"]) / "calibration.json").exists()
        assert st_p.get("calibration_source") == "auto", st_p.get("calibration_source")
        assert st_p["calibration_auto"]["accepted"] and "Error" not in str(st_p["calibration_auto"].get("why")), st_p["calibration_auto"]
        assert st_p["calibration_auto"]["width_assumed"] is True and abs(st_p["calibration_auto"]["length_m"] - 105.1) < 1e-6

        # A human POST landing after any check and just before the worker's write: theirs must stand,
        # and the source must not be stamped "auto" over it. The post is injected into the write itself.
        race = create("race", None, ".mp4", venue="race ground", team=CLUB)
        save(race)
        human = '{"human": true}'

        def _post_first(real):
            def f(src, dst, *a, **k):
                if Path(dst).name == "calibration.json":
                    Path(dst).write_text(human)
                return real(src, dst, *a, **k)
            return f
        with _mock.patch(f"{__name__}._video_shape", lambda p: (4096, 1080)), \
                _mock.patch.object(_ac, "calibrate", lambda *a, **k: (_q, _rep)), \
                _mock.patch("os.replace", _post_first(os.replace)), _mock.patch("os.link", _post_first(os.link)):
            auto_calibrate(load(race["id"]))
        assert (job_dir(race["id"]) / "calibration.json").read_text() == human, "auto overwrote a human's calibration"
        assert load(race["id"]).get("calibration_source") != "auto"
        assert not (job_dir(race["id"]) / "calibration.auto.json").exists()

        # A GIVEN length wins over the paint's (author's decision, 2026-09-28): autocal is asked to hold it fixed,
        # the file is drawn at it, and the disagreement is recorded.
        typed = create("typed length", None, ".mp4", venue="typed ground", team=CLUB)
        own(typed, "pitch", {"length": 100.0, "width": 64.0, "source": "user"})
        save(typed)

        def _cal(*a, **k):
            return _q, {**_rep, "length_m": 100.0 if k.get("fix_length") else 105.1,
                        "length_measured_m": 105.1, "length_measured_sd_m": 0.02}
        with _mock.patch(f"{__name__}._video_shape", lambda p: (4096, 1080)), \
                _mock.patch.object(_ac, "calibrate", _cal):
            assert auto_calibrate(load(typed["id"])) is True
        ca = load(typed["id"])["calibration_auto"]
        assert ca.get("length_given_m") == 100.0 and ca["length_m"] == 100.0 and ca.get("length_overruled") is True, ca
        assert json.loads((job_dir(typed["id"]) / "calibration.json").read_text())["pitch"]["length"] == 100.0
        # ...but an OSM length only seeds the search: the paint's length stands and nothing is "overruled".
        osm = create("osm length", None, ".mp4", venue="osm ground", team=CLUB)
        own(osm, "pitch", {"length": 100.0, "width": 64.0, "source": "osm", "osm_way": 1})
        save(osm)
        with _mock.patch(f"{__name__}._video_shape", lambda p: (4096, 1080)), \
                _mock.patch.object(_ac, "calibrate", _cal):
            assert auto_calibrate(load(osm["id"])) is True
        ca = load(osm["id"])["calibration_auto"]
        assert ca["length_m"] == 105.1 and "length_given_m" not in ca, ca

        # The install camera (docs/reference/calibration.md#install-camera): a spherical camera from an
        # earlier match on the same rig is autocal's START. Another ground or frame size must not supply one.
        rig = create("rig week 1", None, ".mp4", venue="Rig Ground", team=CLUB)
        rig["frame_shape"] = [4096, 1080]
        save(rig)
        rig_cam = [float(v) for v in _q]
        (job_dir(rig["id"]) / "calibration.json").write_text(json.dumps({"projection_type": "spherical", "camera": rig_cam}))
        rig2 = create("rig week 2", None, ".mp4", venue="rig ground", team=CLUB)
        save(rig2)
        assert _install_camera(rig2, (4096, 1080)) == (rig_cam, "rig week 1"), _install_camera(rig2, (4096, 1080))
        assert _install_camera(rig2, (1920, 1080)) == (None, None), "install camera crossed frame sizes"
        elsewhere = create("rig elsewhere", None, ".mp4", venue="another ground", team=CLUB)
        save(elsewhere)
        assert _install_camera(elsewhere, (4096, 1080)) == (None, None), "install camera crossed grounds"

        # ...and a job with no venue at all cannot prove the ground, so it does
        # not guess. That is every job created before venues existed.
        legacy = create("no venue recorded", None, ".mp4", team=CLUB)
        save(legacy)
        assert inherit_calibration(legacy) is False, "inherited with no venue"

        # ...nor across clubs. Two teams can each call their ground "The Rec".
        guest = create("other club, same ground name", None, ".mp4",
                       venue=HOME, team="Rovers")
        save(guest)
        assert inherit_calibration(guest) is False, "inherited across teams"

        # a DIFFERENT resolution must not inherit either: the homography maps
        # pixels, so it is only valid at the framing it was solved at.
        other = create("different camera", None, ".mp4", venue=HOME, team=CLUB)
        save(other)
    with _mock.patch(f"{__name__}._video_shape", lambda p: (1280, 720)):
        assert inherit_calibration(other) is False, "inherited across resolutions"

    # never chain: `same` is itself inherited, so it cannot be a source.
    with _mock.patch(f"{__name__}._video_shape", lambda p: (1920, 1080)):
        (job_dir(src["id"]) / "calibration.json").unlink()   # leave only the inherited one
        third = create("week 3", None, ".mp4", venue=HOME, team=CLUB)
        save(third)
        assert inherit_calibration(third) is False, "chained an inherited calibration"

    # --- a team is a field, not a directory: all_jobs() filters, nothing moves.
    assert {j["id"] for j in all_jobs("rovers")} == {guest["id"]}, all_jobs("rovers")
    assert {j["id"] for j in all_jobs(CLUB)} >= {src["id"], same["id"], away["id"]}
    assert len(all_jobs()) > len(all_jobs(CLUB)), "the team filter matched everything"
    # jobs written before teams existed read back under DEFAULT_TEAM rather than
    # vanishing from every listing. Written without the key on purpose: that is
    # what is on disk, and defaulting on read is what replaces a migration.
    old_job = create("last season", None, ".mp4")
    (job_dir(old_job["id"]) / "job.json").write_text(json.dumps(
        {k: v for k, v in old_job.items() if k not in ("team", "venue")}))
    assert load(old_job["id"])["team"] == DEFAULT_TEAM
    assert old_job["id"] in {j["id"] for j in all_jobs(DEFAULT_TEAM)}
    assert status_view(load(old_job["id"]))["team"] == DEFAULT_TEAM

    # --- an upload nobody finished is failed, not queued forever.
    stale = create("half.mp4", 10 ** 9, ".mp4")
    stale["updated"] = time.time() - STALE_UPLOAD_S - 1
    # written raw: save() stamps `updated`, which is exactly the field under test
    (job_dir(stale["id"]) / "job.json").write_text(json.dumps(stale))
    pump()
    assert load(stale["id"])["status"] == "failed", load(stale["id"])

    # --- THE NOTE LIBRARY. A client hands it the path, so the only thing
    # standing between a note name and the rest of the disk is note_path().
    for bad in ("../x.html", "a/../../x.html", "/etc/x.html", "", "x.html/..",
                "a\\b.html", ".hidden.html", "a/b/c/d/e/f/g.html",
                "x" * 200 + ".html", "notes", "a\x00.html"):
        try:
            note_path(bad)
        except ValueError:
            continue
        raise AssertionError(f"note_path accepted {bad!r}")
    assert write_note("Scouting/Sept/Rivals.html", "<h2>Rivals</h2><p>Deep block.</p>")
    assert read_note("Scouting/Sept/Rivals.html").startswith("<h2>")
    listed = {n["path"]: n for n in notes()}
    assert listed["Scouting/Sept/Rivals.html"]["snippet"] == "Rivals Deep block."
    assert listed["Scouting"]["dir"] and listed["Scouting/Sept"]["dir"]
    # The ladder gets a folder each, and keeps getting one as it grows. The
    # teams here were seeded from the jobs created above, so this also pins the
    # part that matters: a team that appears later still gets its folder.
    ladder = org()
    wanted = [ladder["club"]["name"]] + [t["name"] for t in ladder["teams"]]
    assert len(wanted) > 1, wanted
    for name in wanted:
        clean = _note_name(name)
        if clean and clean != "Note":
            assert listed[clean]["dir"], (name, sorted(listed))
    # The card shows the note's own markup, so the listing carries it verbatim
    # while it fits.
    assert listed["Scouting/Sept/Rivals.html"]["head"] == "<h2>Rivals</h2><p>Deep block.</p>"
    # Past the cap it is cut outside a tag, and never inside a figure: half an
    # svg draws as a chart with its right-hand side missing.
    write_note("Fig.html", "<p>" + "a" * HEAD_MAX + "</p><svg><g><svg/></g></svg>")
    head = {n["path"]: n for n in notes()}["Fig.html"]["head"]
    assert head.startswith("<p>aaa") and "<svg" not in head and "<" not in head[-1:], head[-40:]
    write_note("Fig.html", "<svg>" + "a" * HEAD_MAX + "</svg><p>after</p>")
    assert {n["path"]: n for n in notes()}["Fig.html"]["head"] == "", "half a figure survived"
    delete_note("Fig.html")
    move_note("Scouting/Sept/Rivals.html", "Scouting/Rivals FC.html")
    assert "Scouting/Rivals FC.html" in {n["path"] for n in notes()}
    make_folder("Players")
    assert {"path": "Players", "dir": True} in notes()  # empty folders survive
    # A move must never eat the note it lands on.
    write_note("Players/Ten.html", "<p>ten</p>")
    try:
        move_note("Scouting/Rivals FC.html", "Players/Ten.html")
        raise AssertionError("move overwrote an existing note")
    except ValueError:
        pass
    assert read_note("Players/Ten.html") == "<p>ten</p>"
    delete_note("Scouting")
    assert not any(n["path"].startswith("Scouting") for n in notes())

    # ---------------------------------------------- two writers, one note
    # A coach with the editor open and a report script now both write here.
    # Without a version the second silently overwrote the first.
    write_note("Race.html", "<p>one</p>")
    v1 = note_sha("<p>one</p>")
    assert note_sha(read_note("Race.html")) == v1, "the version must follow the content"

    # The script writes without quoting a version: it owns its output path and
    # re-running it is meant to replace what is there. Unchanged behaviour.
    write_note("Race.html", "<p>script</p>")
    assert read_note("Race.html") == "<p>script</p>"

    # The editor quotes the version it loaded, which is now stale.
    try:
        write_note("Race.html", "<p>coach</p>", v1)
        raise AssertionError("a stale write landed")
    except NoteConflict as e:
        assert e.current == note_sha("<p>script</p>"), "409 must say what it actually is"
    assert read_note("Race.html") == "<p>script</p>", "a refused write still wrote"

    # Quoting the current version works, and hands back the next one.
    write_note("Race.html", "<p>coach</p>", note_sha("<p>script</p>"))
    assert read_note("Race.html") == "<p>coach</p>"

    # The empty version means "there must be no file here", so the first write
    # of a seeded document is guarded too - and refuses once one exists.
    try:
        write_note("Race.html", "<p>new</p>", "")
        raise AssertionError("a create-only write landed on an existing note")
    except NoteConflict:
        pass
    write_note("Fresh.html", "<p>fresh</p>", "")
    assert read_note("Fresh.html") == "<p>fresh</p>"
    delete_note("Race.html")
    delete_note("Fresh.html")

    # ------------------------------------------------------- headshots
    # The type comes off the BYTES. A file that claims to be a photo and is
    # not gets refused; the smallest real PNG gets stored and found again.
    png = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    assert photo_kind(b"<svg onload=alert(1)>") is None, "an SVG is not a headshot"
    assert photo_kind(png) == ("png", "image/png")
    assert not has_photo("varga")
    assert write_photo("varga", png) == "image/png"
    assert has_photo("varga")
    found = find_photo("varga")
    assert found and found[0].name == "varga.png"
    # A second upload in another format REPLACES the first - a player has one
    # face, not one per file type.
    write_photo("varga", b"\xff\xd8\xff" + b"\x00" * 8)
    assert find_photo("varga")[0].name == "varga.jpg", "the old png survived"
    for bad in ("", "../etc", "a/b", "x" * 65):
        try:
            photo_path(bad, "png")
        except ValueError:
            pass
        else:
            raise AssertionError(f"photo_path took {bad!r}")
    delete_photo("varga")
    assert not has_photo("varga")
    delete_photo("varga")  # deleting nothing is not an error

    _api_selfcheck(cal_doc)

    shutil.rmtree(tmp, ignore_errors=True)
    print("jobs.py ok" + (f" - KNOWN BUG(S) still present: {', '.join(known_bugs)}" if known_bugs else ""))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        _selfcheck()
