"""Upload + processing service.

    .venv/bin/pip install fastapi uvicorn
    .venv/bin/python app.py            # http://localhost:8000, /docs for the API

API only - there is no static mount. The UI is `app/frontend/` (`npm run dev`),
which serves itself and rewrites /api here, so the browser sees one origin.

Upload is a raw streamed body, not multipart:

    curl -X POST --data-binary @match.mp4 \
         -H 'Content-Type: video/mp4' \
         'http://localhost:8000/api/matches?name=match.mp4&total=12345678'

A multi-GB file must never be buffered, and multipart would spool a second
full copy to disk on the way past. If the connection drops, GET the job,
read `uploaded_bytes`, and POST the remainder to /upload with X-Upload-Offset.
# ponytail: raw body only. Add a multipart branch when a no-JS <form> needs one.

Trust boundary, all of it in this file:
  - file type is decided by magic bytes, never by the client's Content-Type
  - the client's filename is stored for display and never used as a path
  - job ids are validated by jobs.job_dir() before touching the filesystem
  - uploads land in DATA_DIR, which is outside the served web root
"""
from __future__ import annotations

import _path  # noqa: F401  - puts model/engine/ on sys.path

import asyncio
import hashlib
import json
import mimetypes
import os
import re
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

# Response: /frame.jpg returns raw bytes. It was used and never imported, so
# that endpoint - the one the whole calibration flow starts from - raised
# NameError on its last line after decoding the frame.
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

import agent
import chats
import db
import jobs

MAX_UPLOAD = int(os.environ.get("FUTTRAFFIC_MAX_UPLOAD", 32 * 1024 ** 3))  # 32 GB
MAX_JSON = 4 * 1024 ** 2  # calibration / corrections / roster are all tiny


async def _json(request: Request) -> dict:
    """A small JSON object body, refused rather than parsed if it is neither."""
    body = await request.body()
    if len(body) > MAX_JSON:
        raise HTTPException(413, "body too large")
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"bad JSON: {e}")
    if not isinstance(payload, dict):
        raise HTTPException(400, "expected an object")
    return payload
CHUNK = 1024 * 1024
PUMP_EVERY = 2.0

# Container magic. A 90-minute match is worth 30 seconds of sniffing to avoid
# handing an arbitrary upload to a decoder. Extension comes from here, not
# from anything the client said.
MAGIC = [
    (lambda b: b[4:8] == b"ftyp", ".mp4"),                     # ISO-BMFF: mp4/mov/m4v
    (lambda b: b[:4] == b"\x1a\x45\xdf\xa3", ".mkv"),           # Matroska / WebM
    (lambda b: b[:4] == b"RIFF" and b[8:12] == b"AVI ", ".avi"),
    # Two sync bytes one packet apart. One 0x47 alone matches every file that
    # happens to start with "G", which is not a check.
    (lambda b: b[:1] == b"\x47" and b[188:189] == b"\x47", ".ts"),
    (lambda b: b[:4] in (b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3"), ".mpg"),  # MPEG-PS
]
SNIFF_BYTES = 192  # enough for the second TS sync byte


def sniff(head: bytes) -> str | None:
    for test, ext in MAGIC:
        try:
            if test(head):
                return ext
        except IndexError:
            continue
    return None


def safe_name(raw: str) -> str:
    """Display only. Never joined to a path - but strip it anyway, because
    'display only' is one refactor away from being false."""
    raw = os.path.basename(raw or "")[-120:]
    return re.sub(r"[^A-Za-z0-9._ -]", "_", raw) or "upload"


def get_job(jid: str) -> dict:
    try:
        return jobs.load(jid)
    except ValueError:
        raise HTTPException(400, "bad match id")
    except (FileNotFoundError, json.JSONDecodeError):
        raise HTTPException(404, "no such match")


async def _pump_forever() -> None:
    while True:
        try:
            await asyncio.to_thread(jobs.pump)
        except Exception as e:  # noqa: BLE001 - the scheduler must not die
            print(f"pump error: {type(e).__name__}: {e}")
        await asyncio.sleep(PUMP_EVERY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    jobs.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    # Startup pump is the resume: any job left mid-stage by a crash gets
    # reaped and re-queued here, before the first request arrives.
    await asyncio.to_thread(jobs.pump)
    task = asyncio.create_task(_pump_forever())
    yield
    task.cancel()


app = FastAPI(title="fut traffic", lifespan=lifespan)


# ------------------------------------------------------------------ origin
# The API has no auth: it trusts whoever can reach it, which by default is this
# machine only. A browser breaks that - any page the user visits can POST to
# localhost:8000 (a text/plain POST needs no preflight), and a DNS-rebound name
# can read GET responses. So: a request that changes state must come from our
# own UI (the Next dev/prod server on localhost, or the desktop shell's
# futtraffic:// scheme) or from no browser at all (no Origin header: curl, the
# note-api script); and while bound to loopback the Host must be loopback too.
# A deploy that serves the UI from elsewhere lists its origins and hosts in
# FUTTRAFFIC_ALLOWED_ORIGINS / FUTTRAFFIC_ALLOWED_HOSTS (comma-separated).
_LOOPBACK = {"localhost", "127.0.0.1", "[::1]", "::1"}
_EXTRA_ORIGINS = {o.strip().rstrip("/") for o in os.environ.get("FUTTRAFFIC_ALLOWED_ORIGINS", "").split(",") if o.strip()}
_EXTRA_HOSTS = {h.strip().lower() for h in os.environ.get("FUTTRAFFIC_ALLOWED_HOSTS", "").split(",") if h.strip()}


def _hostname(netloc: str) -> str:
    """'127.0.0.1:8000' -> '127.0.0.1', '[::1]:3000' -> '[::1]'."""
    netloc = netloc.lower()
    if netloc.startswith("["):
        return netloc.split("]")[0] + "]"
    return netloc.rsplit(":", 1)[0] if netloc.count(":") == 1 else netloc


def origin_allowed(origin: str | None) -> bool:
    if not origin:                      # not a browser
        return True
    origin = origin.rstrip("/")
    if origin in _EXTRA_ORIGINS or origin == "futtraffic://app":
        return True
    scheme, _, rest = origin.partition("://")
    return scheme in ("http", "https") and _hostname(rest) in _LOOPBACK


def host_allowed(host: str | None) -> bool:
    if os.environ.get("HOST", "127.0.0.1") not in ("127.0.0.1", "localhost", "::1"):
        return True                     # bound beyond loopback: a deploy decides its own hosts
    return not host or _hostname(host) in _LOOPBACK or host.lower() in _EXTRA_HOSTS \
        or _hostname(host) in _EXTRA_HOSTS


@app.middleware("http")
async def _same_origin_only(request: Request, call_next):
    if not host_allowed(request.headers.get("host")):
        return JSONResponse({"detail": "host not allowed"}, status_code=403)
    if request.method not in ("GET", "HEAD", "OPTIONS") and not origin_allowed(request.headers.get("origin")):
        return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
    return await call_next(request)


# ------------------------------------------------------------------- upload

async def _write_stream(request: Request, path: Path, start: int, limit: int) -> int:
    """Stream the body to disk. Returns total bytes on disk. Never buffers."""
    written = start
    with open(path, "ab" if start else "wb") as f:
        async for chunk in request.stream():
            written += len(chunk)
            if written > limit:
                f.close()
                path.unlink(missing_ok=True)
                raise HTTPException(413, f"upload exceeds {limit} bytes")
            f.write(chunk)
    return written


@app.post("/api/matches")
async def create_match(request: Request, name: str = "", total: int = 0,
                       venue: str = "", team: str = ""):
    """Stream a match video in. `total` (bytes) enables resume; without it a
    dropped connection means starting over.

    `venue` and `team` are their own query params rather than more text packed
    into `name` (which is where the date rides). The venue decides whether
    another match's homography may be reused here, so it is a machine-compared
    value; parsing a safety condition back out of a free-text display string is
    how it would quietly start matching the wrong ground again. A job posted
    without a venue still works - it just never inherits a calibration.
    """
    if total and total > MAX_UPLOAD:
        raise HTTPException(413, f"declared size {total} exceeds {MAX_UPLOAD} bytes")

    # Sniff before committing to a job directory: a text file posted here
    # should cost nothing.
    stream = request.stream()
    head = b""
    async for chunk in stream:
        head += chunk
        if len(head) >= SNIFF_BYTES:
            break
    ext = sniff(head)
    if ext is None:
        raise HTTPException(415, "not a recognised video container (mp4/mov/mkv/webm/avi/ts)")

    st = jobs.create(safe_name(name), total or None, ext, venue=venue, team=team)
    path = jobs.job_dir(st["id"]) / st["video"]
    written = len(head)
    try:
        with open(path, "wb") as f:
            f.write(head)
            async for chunk in stream:
                written += len(chunk)
                if written > MAX_UPLOAD:
                    f.close()
                    path.unlink(missing_ok=True)
                    jobs._fail(st, "upload exceeded size limit")
                    raise HTTPException(413, f"upload exceeds {MAX_UPLOAD} bytes")
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:  # client vanished mid-upload: keep the partial for resume
        st["bytes"] = path.stat().st_size if path.exists() else 0
        jobs.save(st)
        raise HTTPException(400, f"upload interrupted at {st['bytes']} bytes: {e}")

    st["bytes"] = written
    st["upload_complete"] = (not total) or written >= total
    jobs.save(st)
    return jobs.status_view(st)


@app.post("/api/matches/{jid}/upload")
async def resume_upload(jid: str, request: Request):
    """Append to an interrupted upload.

    X-Upload-Offset must equal the bytes already on disk. That alone proves the
    client knows the byte COUNT, not that it is resuming the same FILE - the tail
    of a different video appended at the right offset used to be accepted, and
    the result is a container that only fails when something tries to decode it,
    which is after the operator has calibrated and the tracker has burned GPU
    hours. So X-Upload-Total is checked against the size declared at creation.

    That closes wrong-file-different-size, which is the accident that actually
    happens (picking the wrong file from a folder). Two DIFFERENT videos of
    byte-identical length would still slip through; catching that needs a
    content hash and a client that sends one. Not worth it until it happens.
    """
    st = get_job(jid)
    if st.get("upload_complete"):
        raise HTTPException(409, "upload already complete")
    path = jobs.job_dir(jid) / st["video"]
    have = path.stat().st_size if path.exists() else 0
    try:
        offset = int(request.headers.get("x-upload-offset", ""))
    except ValueError:
        raise HTTPException(400, f"X-Upload-Offset required; server has {have} bytes")
    if offset != have:
        raise HTTPException(409, f"offset {offset} != {have} bytes on disk")

    # REQUIRED, not optional, whenever the job declared a size: a check a client
    # can skip by omitting a header protects nothing. Jobs created without
    # `total` have no size to compare against and are exempt - they also cannot
    # be resumed meaningfully, since nothing knows when they are finished.
    total = st.get("total_bytes")
    if total:
        declared = request.headers.get("x-upload-total")
        if declared is None:
            raise HTTPException(400,
                f"X-Upload-Total required to resume; this match was started as "
                f"{total} bytes and the server cannot otherwise tell your file "
                f"from a different one of the same length.")
        try:
            declared = int(declared)
        except ValueError:
            raise HTTPException(400, f"X-Upload-Total must be an integer, got {declared!r}")
        if declared != total:
            raise HTTPException(409,
                f"this is a different file: {declared} bytes, but this match was "
                f"started as {total}. Resume the original file, or upload this one "
                f"as a new match.")

    written = await _write_stream(request, path, have, MAX_UPLOAD)
    st["bytes"] = written
    st["upload_complete"] = (not total) or written >= total
    if st["upload_complete"]:
        st["status"] = "queued"
    jobs.save(st)
    return jobs.status_view(st)


# ------------------------------------------------------------------- status

@app.get("/api/matches")
def list_matches(team: str = ""):
    """Every match, or one team's. Matched on the team slug, so the caller does
    not have to reproduce the club's exact capitalisation."""
    return [jobs.status_view(s) for s in sorted(
        jobs.all_jobs(team or None), key=lambda s: -s.get("created", 0))]


@app.get("/api/matches/{jid}")
def match_status(jid: str):
    return jobs.status_view(get_job(jid))


@app.get("/api/season")
def season():
    """The whole library as one fact table - see season.py.

    Every finished match's entered feed, flattened to one row per (match, club)
    and one per (match, player). Clubs, players and opponents are three
    GROUPINGS of this, done in the browser, because they are three cuts of one
    fact rather than three questions worth three endpoints.

    Cached in jobs.season_feed, which the agent's squad tools read too - see
    there for why the cache is keyed on mtimes.
    """
    return jobs.season_feed()


@app.post("/api/matches/{jid}/calibration")
async def set_calibration(jid: str, request: Request):
    """Attach the click-to-calibrate output.

    Validated by performing the worker's own load - jobs.load_calibration is
    literally the call _pipeline makes - so "accepted here" and "loadable
    there" cannot drift apart. They had: this endpoint was widened to
    calibrators.load so panoramic exports would be accepted, while track.run()
    and stats.build_match() then loaded through pitch.Calibration, which raises
    on any non-perspective projection_type. The result was 200 OK followed by a
    ValueError hours into the job. CORRECTED 2026-09-28: both now load through
    calibrators.load (track.py :: run, stats.py :: build_match), and the types
    accepted are the ones jobs.load_calibration lists - never accept a file
    nothing downstream can read.
    """
    st = get_job(jid)
    body = await request.body()
    if len(body) > MAX_JSON:
        raise HTTPException(413, "calibration too large")
    d = jobs.job_dir(jid)
    tmp = d / "calibration.check.json"
    try:
        tmp.write_bytes(body)
        jobs.load_calibration(tmp)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, f"invalid calibration: {type(e).__name__}: {e}")
    os.replace(tmp, d / "calibration.json")

    # A job parked on calibrating is now runnable, and so is a finished one
    # whose numbers were computed against the old homography. "failed" is in
    # here because the calibration is now an input to tracking: a job that died
    # on an unusable one has to be recoverable by posting a good one, or the
    # only fix is a re-upload.
    #
    # tracks.raw.json is deliberately NOT invalidated. It was produced with the
    # old calibration's scene prior, so strictly it is stale - but re-tracking
    # is 2-4 GPU-hours and throws away the frame the operator's corrections are
    # anchored to. The prior only ever removed non-player boxes; re-solving the
    # positions through the new homography is what actually moves the numbers.
    # A human solved this one against this footage, so it is no longer somebody
    # else's homography. Clearing the flag also stops it being skipped as an
    # inheritance SOURCE - a hand-calibrated match is the best source there is.
    st.pop("calibration_inherited_from", None)
    jobs.own(st, "calibration_source", "manual")   # a human's calibration replaces an automatic one, and says so

    if st["status"] in ("calibrating", "done", "failed"):
        (d / "match.json").unlink(missing_ok=True)
        st["status"] = "queued"
        st["attempts"] = 0
        st["error"] = None
    jobs.save(st)
    return jobs.status_view(st)


@app.post("/api/matches/{jid}/retry")
def retry(jid: str):
    """Put a failed or parked job back in the queue.

    The worker gives up after MAX_ATTEMPTS so a job that fails deterministically
    does not loop forever burning GPU. That is right, but it left the only
    recovery being a re-upload of a file already on disk. Resets the counter
    because a human asking again is new information - usually they fixed the
    thing that broke it.
    """
    st = get_job(jid)
    if st["status"] in jobs.RUNNING:
        raise HTTPException(409, f"already {st['status']}")
    if not st.get("upload_complete"):
        raise HTTPException(409, "upload never finished; re-upload this match")
    d = jobs.job_dir(jid)
    st["status"] = "queued" if (d / "calibration.json").exists() else "calibrating"
    st["attempts"] = 0
    st["error"] = None
    st.pop("identity_done", None)   # a failed identity is re-attempted; a finished one left tracks.id.json
    jobs.save(st)
    return jobs.status_view(st)


def _client_has(request: Request, etag: str) -> bool:
    """If-None-Match holds this tag (weak or strong) or `*`."""
    have = [t.strip().removeprefix("W/") for t in
            request.headers.get("if-none-match", "").split(",")]
    return etag in have or "*" in have


def json_file(request: Request, path: Path) -> Response:
    """A JSON artifact, answering 304 when the client already has this version.

    FileResponse sets ETag and Last-Modified but Starlette 1.3.1 implements
    NEITHER If-None-Match nor If-Modified-Since - read the class, there is no
    conditional branch in it - so a re-request comes back 200 with the whole
    body. The reference tracks.json is 60 MB and /review re-fetches it on every
    load.

    Safe for exactly these three files because all three are REWRITTEN
    (write-tmp-then-os.replace), never mutated in place, so the mtime+size the
    tag is derived from always moves when the content does.

    The tag is Starlette's own, taken off the response rather than recomputed
    here: two implementations of one etag formula is how a 304 starts lying.
    # ponytail: If-None-Match only. If-Modified-Since is the weaker, lower-
    # resolution half of the same question and every client sends both.
    """
    r = FileResponse(path, media_type="application/json", stat_result=path.stat())
    etag = r.headers.get("etag", "")
    if etag and _client_has(request, etag):
        # No body, and no content-length: Response special-cases 304.
        return Response(status_code=304, headers={
            "etag": etag, "last-modified": r.headers["last-modified"]})
    return r


@app.get("/api/matches/{jid}/match.json")
def match_json(jid: str, request: Request):
    st = get_job(jid)
    p = jobs.job_dir(jid) / "match.json"
    if not p.exists():
        raise HTTPException(404, f"not ready: {st['status']}")
    # NOT json_file: the tiers inside were the golden's opinion the day the job
    # ran, and 11 of 11 files on disk served `players.jersey: ok` long after
    # that stopped being true. Re-resolved per request (stats.resolve), so the
    # file's bytes are not the response and a stat-based etag would lie across
    # a golden change. The tag is a hash of the bytes actually sent instead:
    # exact by construction, and the 304 json_file exists for still holds.
    from stats import resolved_match  # noqa: PLC0415 - pulls numpy
    body = json.dumps(resolved_match(p)).encode()
    etag = f'"{hashlib.sha1(body).hexdigest()}"'
    if _client_has(request, etag):
        return Response(status_code=304, headers={"etag": etag})
    return Response(body, media_type="application/json", headers={"etag": etag})


@app.get("/api/matches/{jid}/pitch/osm")
def osm_pitch(jid: str, lat: float, lon: float):
    """Pitch outlines OpenStreetMap has within 300 m of a point, each checked against THIS match's painted
    length: about half of OSM's pitches are traced round the grass, and only the length gives that away
    (model/engine/pitchsize.py). Sends the coordinates and nothing else; never chooses for the user."""
    import pitchsize  # noqa: PLC0415
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(400, "latitude, longitude in degrees")
    painted = (get_job(jid).get("calibration_auto") or {}).get("length_m")
    found = pitchsize.osm_pitches(lat, lon, paint_length=painted)
    if found is None:
        raise HTTPException(503, "OpenStreetMap's lookup service did not answer - it is free and often busy; try again in a minute")
    return {"painted_length_m": painted, "pitches": found, "attribution": pitchsize.ATTRIBUTION}


@app.post("/api/matches/{jid}/pitch")
async def set_pitch(jid: str, request: Request):
    """The pitch's size, from somebody who knows it: {"width": m, "length": m (optional)}.

    The automatic calibration measures LENGTH from the paint and cannot measure WIDTH (it lies along
    the camera's line of sight), so width is the one number a venue owes us. It is kept on the job and
    every later match at the same club and ground takes it from there. An automatic calibration made
    under the old size is dropped and the match re-queued, so the camera is found again on a pitch of
    the right size; a hand calibration carries its own size and is left alone."""
    st = get_job(jid)
    try:
        body = json.loads(await request.body() or b"{}")
        width = float(body["width"])
        length = float(body["length"]) if body.get("length") not in (None, "") else None
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"pitch size: {type(e).__name__}: {e}")
    if not (45.0 <= width <= 90.0) or (length is not None and not (90.0 <= length <= 120.0)):
        raise HTTPException(400, "a football pitch is 45-90 m wide and 90-120 m long (IFAB Law 1)")
    # an OpenStreetMap-derived width is kept AS one (its way id beside it): ODbL attribution, and a re-check later
    source, way = body.get("source") or "user", body.get("osm_way")
    if source == "osm" and (not isinstance(way, int) or isinstance(way, bool) or not 0 < way < 2 ** 53):
        raise HTTPException(400, "an OpenStreetMap width needs the outline's way id")
    if source not in ("user", "osm"):
        raise HTTPException(400, "pitch size source: user or osm")
    if st["status"] in jobs.RUNNING:
        # the worker holds calibration.json and is metres into the match under the old size
        raise HTTPException(409, "this match is being processed; set the pitch size when it has finished")
    jobs.own(st, "pitch", {"width": width, "length": length, "source": source, **({"osm_way": way} if source == "osm" else {})})
    d = jobs.job_dir(jid)
    if st.get("calibration_source") == "auto" and (d / "calibration.json").exists():
        (d / "calibration.json").unlink()
        (d / "match.json").unlink(missing_ok=True)
        jobs.own(st, "calibration_source", None)
        if st["status"] in ("calibrating", "done", "failed"):
            st["status"], st["attempts"], st["error"] = "queued", 0, None
    jobs.save(st)
    return jobs.status_view(st)


@app.get("/api/matches/{jid}/calibration")
def get_calibration(jid: str, request: Request):
    get_job(jid)
    p = jobs.job_dir(jid) / "calibration.json"
    if not p.exists():
        raise HTTPException(404, "no calibration")
    return json_file(request, p)


# ------------------------------------------------------------------ identity

def _summarise(jid: str) -> dict:
    """Per-track summary for the correction UI. tracks.json is tens of MB, so
    the digest is cached next to it and rebuilt only when it goes stale."""
    d = jobs.job_dir(jid)
    src, cache = d / "tracks.json", d / "tracks_summary.json"
    if not src.exists():
        raise HTTPException(404, "tracks not ready")
    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        return json.loads(cache.read_text())

    data = json.loads(src.read_text())
    fps = float(data.get("fps") or 25.0)
    by: dict[int, dict] = {}
    for x in data["detections"]:
        tid = int(x["id"])
        t = by.get(tid)
        if t is None:
            t = by[tid] = {"id": tid, "first_f": x["f"], "last_f": x["f"], "n": 0,
                           "team": x.get("team"), "jersey": x.get("jersey"),
                           "confidence": x.get("jersey_conf"), "mid_f": x["f"]}
        t["n"] += 1
        t["first_f"] = min(t["first_f"], x["f"])
        t["last_f"] = max(t["last_f"], x["f"])
        if x.get("jersey") is not None and t["jersey"] is None:
            t["jersey"] = x["jersey"]
        if x.get("jersey_conf") is not None:
            t["confidence"] = max(t["confidence"] or 0.0, x["jersey_conf"])
    for t in by.values():
        t["mid_f"] = (t["first_f"] + t["last_f"]) // 2
        t["seconds"] = round((t["last_f"] - t["first_f"]) / fps, 1)
    # Second pass over the already-parsed list to pin the feet position nearest
    # each track's mid frame. Costs one in-memory sweep here and saves /crop a
    # full re-parse of tracks.json per thumbnail - 1.4 s on a 60 MB synthetic,
    # and the correction UI opens with one thumbnail per track.
    for x in data["detections"]:
        t = by[int(x["id"])]
        dist = abs(x["f"] - t["mid_f"])
        if dist < t.get("_d", 1 << 30):
            t["_d"], t["f"], t["x"], t["y"] = dist, x["f"], x["x"], x["y"]
    for t in by.values():
        t.pop("_d", None)
    out = {"fps": fps,
           "tracks": sorted(by.values(), key=lambda t: -t["seconds"])}
    cache.write_text(json.dumps(out))
    return out


@app.get("/api/matches/{jid}/tracks")
def track_summary(jid: str):
    get_job(jid)
    out = _summarise(jid)
    out["corrections"] = jobs.read_corrections(jid)
    return out


@app.get("/api/matches/{jid}/tracks.json")
def tracks_full(jid: str, request: Request):
    """The per-frame detections, which is what the video overlay draws.

    `/tracks` (no extension) is the per-TRACK digest the correction UI wants;
    this is the whole file. Two routes because they are two different sizes -
    the digest is a few KB and this is tens of MB.

    Served as the file, unmodified, because the review page already parses
    exactly this shape and `app/frontend/public/tracks-worker.js` already streams it
    off the main thread. Inventing a leaner wire format here would mean a second
    parser on both sides to maintain.
    """
    get_job(jid)
    p = jobs.job_dir(jid) / "tracks.json"
    if not p.exists():
        raise HTTPException(404, "tracks not ready")
    return json_file(request, p)


@app.post("/api/matches/{jid}/identity")
async def post_identity(jid: str, request: Request):
    """Human identity corrections. Also the only place operator time is
    recorded, which is the number that decides whether this is a software
    product or a services business."""
    get_job(jid)
    body = await request.body()
    if len(body) > MAX_JSON:
        raise HTTPException(413, "corrections too large")
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"bad JSON: {e}")
    if not isinstance(payload, dict):
        raise HTTPException(400, "expected an object")
    for key, kind in (("assign", dict), ("merge", list), ("split", list),
                      ("exclude", list), ("undo", list)):
        if key in payload and payload[key] is not None and not isinstance(payload[key], kind):
            raise HTTPException(400, f"{key} must be a {kind.__name__}")
    try:
        return jobs.apply_corrections(jid, payload)
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(400, f"could not apply corrections: {e}")


@app.get("/api/matches/{jid}/video")
def video(jid: str):
    """The uploaded video itself, so video.html can play the real match.

    This is the endpoint that makes an uploaded match watchable; without it the
    review page could only ever show the synthetic fixture. It is deliberately a
    route and not a static mount - `jobs.job_dir` is the one place a job id is
    validated, and a StaticFiles mount over DATA_DIR would skip it and hand out
    path traversal. See the invariant note in jobs.py, which this narrows.

    Range comes free: Starlette's FileResponse implements 206, 416 and
    if-range itself. Seeking is dead without it - `video.seekable` comes back
    empty and the review page silently ignores every jump. Do not swap this for
    a plain streaming response.
    """
    st = get_job(jid)
    # A feed-imported match carries no video at all, and `job_dir(jid) / None`
    # is a TypeError - a 500 with a traceback where a 404 belongs. Same guard
    # jobs.frame_jpeg already carries, for the same reason.
    if not st.get("video"):
        raise HTTPException(404, "no video uploaded yet")
    path = jobs.job_dir(jid) / st["video"]
    if not path.exists():
        raise HTTPException(404, "no video uploaded yet")
    # A partial upload is a truncated file, and a browser handed one plays a
    # prefix and then stalls with no error. Saying so is better than a hang.
    if not st.get("upload_complete"):
        raise HTTPException(409, "upload has not finished")
    return FileResponse(path, media_type=(
        mimetypes.guess_type(path.name)[0] or "application/octet-stream"))


@app.get("/api/matches/{jid}/frame.jpg")
def frame(jid: str, f: int = 0, w: int = 0):
    """One still from the uploaded video, for clicking pitch landmarks on.

    Calibration is the first gate, so this has to work BEFORE tracking has run -
    it reads the video directly and depends on nothing the pipeline produces.
    The decode itself is jobs.frame_jpeg, shared with the agent's get_frame
    tool; this route only maps the failure onto a status code.
    """
    get_job(jid)
    try:
        data = jobs.frame_jpeg(jid, f, w)
    except ValueError as e:
        msg = str(e)
        code = 404 if "no video" in msg else (500 if "jpeg" in msg else 422)
        raise HTTPException(code, msg)
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/matches/{jid}/thumb.jpg")
def thumb(jid: str, f: int = jobs.THUMB_F, w: int = 240):
    """A cached still for a match card. NOT frame.jpg, deliberately.

    frame.jpg is `no-store` because calibration must always see the real frame
    it is about to be measured against. A gallery is the opposite case: the same
    still, over and over, for every card on every visit. Served through that
    route, thumbnails cost a cv2 seek per card per paint, which is exactly why
    the match library dropped them once already (hooks/use-matches.ts) and why
    the film gallery's comment says the same rule holds.

    So: decode once, keep it on disk under a key that carries the video's mtime
    (jobs.thumb_path), and let the browser hold it for a week. Second visit is
    free; a re-uploaded video changes the key and re-decodes once.

    ponytail: no server-side concurrency cap. `loading="lazy"` on the card means
    off-screen cards never ask, and the browser's own ~6-per-origin limit caps
    the rest. Add a semaphore only if a cold gallery on a big library measurably
    stalls the API.
    """
    get_job(jid)
    w = max(80, min(int(w), jobs.THUMB_W_MAX))
    try:
        cache = jobs.thumb_path(jid, f, w)
    except ValueError as e:
        raise HTTPException(404, str(e))
    if not cache.exists():
        try:
            data = jobs.frame_jpeg(jid, f, w)
        except ValueError as e:
            msg = str(e)
            raise HTTPException(404 if "no video" in msg else 422, msg)
        cache.parent.mkdir(exist_ok=True)
        # tmp + replace: two cards racing for the same key must not leave a
        # half-written jpeg for the third one to read.
        tmp = cache.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, cache)
    return FileResponse(cache, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=604800"})


@app.get("/api/matches/{jid}/crop/{track_id}.jpg")
def crop(jid: str, track_id: int, f: int = -1, h: int = 0):
    """One thumbnail of a track, so the operator can see who they are naming.

    The tracks contract carries a feet position and no box, so the crop is a
    fixed window anchored at the feet. CROP_H is a knob for a reason: player
    pixel height varies with camera height and how far up the pitch they are,
    and no single number is right at both touchlines.
    # ponytail: one seek per track, cached to disk. Batch into a single
    # sequential pass if opening the UI ever feels slow.
    """
    st = get_job(jid)
    d = jobs.job_dir(jid)
    cache = d / "crops" / f"{track_id}_{f}_{h}.jpg"
    if cache.exists():
        return FileResponse(cache, media_type="image/jpeg")

    summary = _summarise(jid)
    tr = next((t for t in summary["tracks"] if t["id"] == track_id), None)
    if tr is None:
        raise HTTPException(404, "no such track")

    video = d / st["video"]
    if not video.exists():
        raise HTTPException(404, "video not on disk")

    import cv2  # noqa: PLC0415 - heavy
    import numpy as np  # noqa: PLC0415

    if f < 0 and tr.get("x") is not None:
        pt = tr                       # the digest already carries the mid-frame feet
    else:
        # Explicit frame (the split scrubber) still needs the full detection
        # list. ponytail: one re-parse per scrub step; write a per-track
        # {f,x,y} index next to the crops if splitting ever feels slow.
        frame_no = tr["mid_f"] if f < 0 else max(tr["first_f"], min(f, tr["last_f"]))
        data = json.loads((d / "tracks.json").read_text())
        pt = min((x for x in data["detections"] if int(x["id"]) == track_id),
                 key=lambda x: abs(x["f"] - frame_no), default=None)
        if pt is None:
            raise HTTPException(404, "no detection for that frame")

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(pt["f"]))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(500, f"could not read frame {pt['f']}")

    fh, fw = frame.shape[:2]
    ch = h or int(os.environ.get("FUTTRAFFIC_CROP_H", 0)) or max(48, int(fh * 0.10))
    cw = int(ch * 0.55)
    # x,y is the feet point: the box sits above it, half a width either side.
    x0 = int(np.clip(pt["x"] - cw / 2, 0, max(0, fw - cw)))
    y0 = int(np.clip(pt["y"] - ch * 0.95, 0, max(0, fh - ch)))
    patch = frame[y0:y0 + ch, x0:x0 + cw]
    if patch.size == 0:
        raise HTTPException(500, "crop fell outside the frame")
    patch = cv2.resize(patch, (cw * 2, ch * 2), interpolation=cv2.INTER_CUBIC)

    cache.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(cache), patch, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return FileResponse(cache, media_type="image/jpeg")


# --------------------------------------------------------------------- clips
# The coach-authored layer: "watch this bit". Deliberately independent of the
# pipeline - clips are keyed on seconds into the video, not on track ids or
# frame numbers, so a re-track that renumbers every track leaves them intact.

def _clips(jid: str) -> dict:
    # The tag list ships with the response so the page never hard-codes a set
    # that can drift from the one the server will accept.
    return {"tags": list(jobs.CLIP_TAGS), "clips": jobs.read_clips(jid)}


@app.get("/api/matches/{jid}/clips")
def get_clips(jid: str):
    get_job(jid)
    return _clips(jid)


@app.post("/api/matches/{jid}/clips")
async def post_clip(jid: str, request: Request):
    """Upsert one clip and return the whole list. One clip per request, not a
    whole-list PUT: two coaches on one match must not overwrite each other's
    work with a stale array."""
    get_job(jid)
    body = await request.body()
    if len(body) > MAX_JSON:
        raise HTTPException(413, "clip too large")
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"bad JSON: {e}")
    try:
        jobs.save_clip(jid, payload)
    except (TypeError, ValueError) as e:
        raise HTTPException(400, str(e))
    return _clips(jid)


@app.delete("/api/matches/{jid}/clips/{clip_id}")
def del_clip(jid: str, clip_id: str):
    get_job(jid)
    jobs.delete_clip(jid, clip_id)
    return _clips(jid)


# -------------------------------------------------------------------- notes
# THE NOTE LIBRARY: one folder tree of HTML documents for the whole club, not
# one document per match. jobs.py owns the store and the docstring on why.
#
# A note is NOT assigned to a match or a team, and that is the point rather
# than an omission. The automated match report is the thing that belongs to one
# match; a note is what a coach writes across several - an opponent seen three
# times, a player who climbed two age groups, a season review over every team on
# the ladder. Its stat blocks could already reach the whole library while the
# document holding them was filed under a single job.
#
# The server stores bytes and never interprets them. `rel` is client-supplied,
# so every route hands it to jobs.note_path(), which is the only place a string
# becomes a path - the contract job_dir() has for match ids, for the same
# reason. A ValueError out of the store is a bad request, never a 500.


def _note(rel: str):
    """The store's refusals are ValueErrors; `_bad_id` turns those into 400s
    for every route at once, so this is just the lookup now."""
    return jobs.note_path(rel, folder=True)


@app.get("/api/notes")
def list_notes():
    """Every note and folder in the library, path-sorted, with a snippet for
    the library page's cards. Seeds itself from the per-match notepads the
    first time it is read, so an install that predates the library comes up
    with its notes in it."""
    return jobs.notes()


@app.post("/api/notes/folder")
async def new_folder(request: Request):
    payload = await _json(request)
    jobs.make_folder(str(payload.get("path", "")))
    return jobs.notes()


@app.post("/api/notes/move")
async def move_note(request: Request):
    """Rename and drag-into-a-folder, which are one operation. Refuses to land
    on an existing path rather than replacing a note nobody meant to lose."""
    payload = await _json(request)
    jobs.move_note(str(payload.get("from", "")), str(payload.get("to", "")))
    return jobs.notes()


@app.get("/api/notes/{rel:path}")
def get_note(rel: str):
    _note(rel)
    try:
        html = jobs.read_note(rel)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except OSError:
        raise HTTPException(404, "no such note")
    # The version is what a writer quotes back to prove it is not overwriting
    # something it never saw. See jobs.write_note.
    return {"html": html, "version": jobs.note_sha(html)}


@app.put("/api/notes/{rel:path}")
async def put_note(rel: str, request: Request):
    body = await request.body()
    if len(body) > jobs.NOTE_MAX:
        raise HTTPException(413, "note too large")
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"bad JSON: {e}")
    html = payload.get("html") if isinstance(payload, dict) else None
    if not isinstance(html, str):
        raise HTTPException(400, "expected {html: string}")
    # Optional: the version the writer loaded. Sent, it makes this a
    # write-if-unchanged; omitted, it overwrites as it always did.
    base = payload.get("base") if isinstance(payload, dict) else None
    if base is not None and not isinstance(base, str):
        raise HTTPException(400, "base must be a version string")
    try:
        n = jobs.write_note(rel, html, base)
    except jobs.NoteConflict as e:
        # 409 carries the version it actually is, so the loser can reconcile
        # rather than guess - and so a rejected write is never silent.
        raise HTTPException(409, {"error": "the note changed since you read it",
                                  "version": e.current})
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "bytes": n, "version": jobs.note_sha(html)}


@app.delete("/api/notes/{rel:path}")
def del_note(rel: str):
    _note(rel)
    jobs.delete_note(rel)
    return jobs.notes()


# ---------------------------------------------------------------------- club
# The academy: its teams, its age levels, its staff. org.py owns the model.
# There is no auth on any of this and a coach is a record, not an account -
# see org.py's docstring and TODOS (not published).

@app.get("/api/org")
def get_org():
    """The club ladder, youngest team first, with each squad's size and match
    count. Seeds itself from the team strings already on the jobs, so an
    install that predates the ladder comes up with its teams on it."""
    return jobs.org()


@app.put("/api/org")
async def put_org(request: Request):
    """Upsert the whole club. Team ids are server-minted, like roster player
    ids, because the id is the join to the squad file."""
    body = await request.body()
    if len(body) > MAX_JSON:
        raise HTTPException(413, "club too large")
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"bad JSON: {e}")
    if not isinstance(payload, dict):
        raise HTTPException(400, "expected an object")
    return jobs.save_org(payload)


# -------------------------------------------------------------------- roster

@app.get("/api/roster")
def get_roster():
    """One row per (player, TEAM) - the correction screen's shape. For one row
    per HUMAN across the whole club, which is the career view, use /api/players."""
    return jobs.roster()


@app.get("/api/players")
def get_players():
    """THE PLAYER REGISTRY: one row per human, with every team they belong to.

    The academy's actual product - a player climbing U13 -> U15 -> U17 stays
    one profile - and a merge rather than a store, so it cannot disagree with
    the per-team squads. See jobs.players for the identity limit that rides
    along with it."""
    return jobs.players()


# ------------------------------------------------------------------ headshots
# A face per player, for the notepad's report blocks. jobs.py owns the store and
# the argument for a file per player.
#
# The photo is served from OUR origin, so what is stored has to be an image and
# nothing else: jobs.write_photo decides the type from the file's own magic
# number and refuses everything but PNG, JPEG and WebP. There is no auth on any
# of this, like the rest of the app - see org.py and TODOS (not published).

@app.get("/api/players/photo-credits")
def photo_credits():
    """Who took each photograph and under what licence.

    Not a nicety: the headshots seeded from Wikimedia are CC-BY-SA, and the
    condition of using them is that the credit travels with the picture. A
    document that prints the faces has to be able to print the line, so the
    line has to be fetchable. Photos a club uploaded itself carry no entry,
    which is correct - they are the club's own.

    Declared BEFORE /api/players/{pid}/photo: FastAPI matches in order, and
    "photo-credits" is a perfectly good {pid} to a router that sees it first.
    """
    return jobs.photo_credits()


@app.get("/api/players/bios")
def player_bios():
    """Who each player IS - height, foot, country, contract, positions.

    ENTERED, all of it, and from the feed rather than from the pipeline. It is
    served on its own route rather than folded into /api/players because the
    two answer different questions: that one is the club's registry, this is a
    profile card, and a squad list has no business carrying a market value.

    Declared BEFORE /api/players/{pid}/photo for the same reason
    photo-credits is: FastAPI matches in order and "bios" is a perfectly good
    {pid} to a router that sees it first.
    """
    return jobs.bios()


@app.get("/api/players/{pid}/photo")
def get_photo(pid: str):
    found = jobs.find_photo(pid)
    if not found:
        raise HTTPException(404, "no photo")
    path, mime = found
    return FileResponse(path, media_type=mime)


@app.put("/api/players/{pid}/photo")
async def put_photo(pid: str, request: Request):
    """The raw image as the body - no multipart, because this is one file and a
    form envelope around it buys nothing. Returns the type actually stored,
    which is read off the bytes and not off any header the client sent."""
    body = await request.body()
    if len(body) > jobs.PHOTO_MAX:
        raise HTTPException(413, "photo too large")
    return {"ok": True, "type": jobs.write_photo(pid, body), "bytes": len(body)}


@app.delete("/api/players/{pid}/photo")
def del_photo(pid: str):
    jobs.delete_photo(pid)
    return {"ok": True}


@app.put("/api/players/{pid}/bio")
async def put_bio(pid: str, request: Request):
    """The entered half of a profile - height, foot, country, positions.

    A PATCH in spirit even though it is a PUT: jobs.save_bio MERGES, because
    the feed owns the other half of this file (traits, heatmap, shots) and a
    replace would drop it. The whole stored bio comes back so the client
    renders what is on disk rather than what it hoped it sent - `age` in
    particular is derived from birth_date on the way in.
    """
    payload = await _json(request)
    try:
        # OFF THE EVENT LOOP. save_bio reads, parses, re-serialises and writes
        # the whole bio file, and the feed's half of one is large - the biggest
        # in the store is 136 KB of heatmap and shot arrays. Doing that inline
        # in an `async def` stalls every other request, the agent's SSE stream
        # included. get_photo and del_photo beside it are plain `def` for the
        # same reason; this route needs the async body read, so it hands the
        # blocking half to the threadpool itself.
        return await run_in_threadpool(jobs.save_bio, pid, payload)
    except ValueError as e:
        raise HTTPException(400, str(e))


# ------------------------------------------------------------------- the store
# The database tab. `data/match.db` is a QUERY LAYER over match.json and
# org.json (see db.py) - it is rebuildable, so serving it read-only costs
# nothing and losing it costs nothing.
#
# WHY A SQL ENDPOINT AND NOT TWENTY REPORT ROUTES. The ask is "everything
# queryable": compare any players over any factors across any matches. That is
# a different question every time, and the alternative is a new endpoint per
# question forever, each with its own aggregation bug. One read-only connection
# answers all of them.
#
# HOW IT IS READ-ONLY, and it matters that this is not a regex over the SQL:
#   - the connection is opened `mode=ro`, so SQLite itself refuses every write,
#     including one hidden inside a CTE or a trigger.
#   - `query_only` on top of it.
#   - an AUTHORIZER that denies ATTACH. mode=ro does not cover this: ATTACH
#     names a path, and SQLite CREATES that file if it is missing, so a
#     "read-only" endpoint could drop zero-byte files anywhere the server user
#     can write. Verified, not assumed - `ATTACH '/tmp/evil.db'` did exactly
#     that before this callback existed.
#   - Connection.execute runs exactly ONE statement; a trailing `; DROP ...` is
#     a syntax error, not a second command.
# All four are the database's own mechanisms. A blocklist of keywords would be
# the version of this that gets bypassed, and it would also refuse a legitimate
# query containing the word "delete" inside a string literal.

DB_ROWS = 5000            # a table view, not an export; ask for less with LIMIT


def _no_attach(action, _a1, _a2, _db, _source):
    if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH):
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _store():
    if not jobs.db_path().exists():
        raise HTTPException(
            404, "no store yet - it is written when a match finishes processing, "
                 "or rebuild it with `python db.py build`")
    conn = sqlite3.connect(f"file:{jobs.db_path()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    db.withhold_sprints(conn)   # before query_only: old stored sprint counts are not served
    conn.execute("PRAGMA query_only = ON")
    conn.set_authorizer(_no_attach)
    return conn


@app.get("/api/db/schema")
def db_schema():
    """Every table and view in the store, its columns and its row count. What
    the database tab draws its sidebar from, and the only honest way to write a
    query against a schema that grows."""
    conn = _store()
    try:
        out = []
        for r in conn.execute(
                "SELECT name, type FROM sqlite_master WHERE type IN ('table','view')"
                " AND name NOT LIKE 'sqlite_%' ORDER BY type DESC, name"):
            cols = [{"name": c["name"], "type": c["type"], "pk": bool(c["pk"])}
                    for c in conn.execute(f"PRAGMA table_info({r['name']})")]
            try:
                n = conn.execute(f"SELECT count(*) c FROM \"{r['name']}\"").fetchone()["c"]
            except sqlite3.Error:
                n = None   # a view over a table that is empty this rebuild
            out.append({"name": r["name"], "kind": r["type"], "columns": cols, "rows": n})
        return out
    finally:
        conn.close()


@app.post("/api/db/query")
async def db_query(request: Request):
    """Run one read-only statement. `{sql, params?}` in, `{columns, rows}` out."""
    body = await request.body()
    if len(body) > MAX_JSON:
        raise HTTPException(413, "query too large")
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"bad JSON: {e}")
    sql = str((payload or {}).get("sql") or "").strip()
    if not sql:
        raise HTTPException(400, "expected {sql: string}")
    params = (payload or {}).get("params") or []
    conn = _store()
    try:
        cur = conn.execute(sql, params if isinstance(params, (list, tuple)) else [])
        rows = cur.fetchmany(DB_ROWS)
        cols = [d[0] for d in (cur.description or [])]
        return {"columns": cols, "rows": [list(r) for r in rows],
                # So the tab can say "first 5000 of more" instead of presenting
                # a truncated answer as the whole one.
                "truncated": len(rows) == DB_ROWS and cur.fetchone() is not None}
    except sqlite3.Error as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")
    finally:
        conn.close()


@app.post("/api/db/rebuild")
def db_rebuild():
    """Re-ingest every match.json and the club into a fresh store. The escape
    hatch db.py's docstring promises: delete the database and re-ingest."""
    res = db.build(str(jobs.db_path()))
    # Each job's ingest_error follows the store it describes: set by a failed
    # ingest, it was never cleared by a rebuild that then succeeded.
    for jid, r in (res.get("jobs") or {}).items():
        try:
            st = jobs.load(jid)
        except (ValueError, FileNotFoundError, json.JSONDecodeError):
            continue
        err = r.get("error") if isinstance(r, dict) else None
        # A running job's worker saves job.json itself, unlocked; leave it be
        # rather than race it - its own ingest will set this field.
        if st.get("status") in jobs.RUNNING:
            continue
        if st.get("ingest_error") != err:
            st["ingest_error"] = err
            jobs.save(st)
    return res


@app.get("/api/presets")
def get_presets():
    """Saved stat selections, shared by every note. See jobs.presets."""
    return jobs.presets()


@app.put("/api/presets")
async def put_presets(request: Request):
    body = await request.body()
    if len(body) > MAX_JSON:
        raise HTTPException(413, "presets too large")
    try:
        items = json.loads(body or b"[]")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"bad JSON: {e}")
    if not isinstance(items, list):
        raise HTTPException(400, "expected a list")
    return jobs.save_presets(items)


@app.get("/api/season/measured")
def get_season_measured(team: str = ""):
    """Per-player totals across every finished match, joined on the roster's
    human-confirmed track bindings. See jobs.season for why only those count.

    The OTHER season route, `/api/season`, is a different table and the two must
    not be confused. That one is the entered feed - goals, xG, possession -
    which a human attributed, so it can be summed per club and per named player.
    This one is what the PIPELINE measured, which is why it can only be reported
    for tracks a human confirmed: identity reaches chance a few minutes in, so
    an unconfirmed row would put one player's running under another's name."""
    return jobs.season(team or None)


# -------------------------------------------------------------------- agent
# The coach-workspace chat. Transport is SSE: one compact JSON object per
# `data:` line, the exact shapes documented at the top of agent.py - the
# frontend lane builds against that contract, so this file only serialises
# what the run emits and never invents an event of its own.
#
# Every refusal (400 bad input or model or no key, 404 no chat, 409 busy) is
# decided BEFORE the stream starts: once StreamingResponse is underway the
# status line is gone and an error can only be an `error` event.
#
# The run lives in agent.start()'s thread, which owns the chat's lock and
# releases it when the run ends. It used to live in a sync generator that
# released the lock in its `finally` - and a response dropped before its first
# chunk never runs that `finally`, so the chat answered 409 until restart.

@app.post("/api/agent/run")
async def agent_run(request: Request):
    body = await request.body()
    if len(body) > MAX_JSON:
        raise HTTPException(413, "message too large")
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"bad JSON: {e}")
    if not isinstance(payload, dict):
        raise HTTPException(400, "expected an object")
    message = str(payload.get("message") or "").strip()[:4000]
    if not message:
        raise HTTPException(400, "message required")
    model = str(payload.get("model") or "") or agent.DEFAULT_MODEL
    if model not in agent.MODELS and model != agent.DEFAULT_MODEL:
        raise HTTPException(400, f"unknown model {model!r}; one of {', '.join(agent.MODELS)}")

    cid = str(payload.get("chat_id") or "")
    # The match travels with every message: a chat can move between matches.
    mid = str(payload.get("match_id") or "")
    # The squad is the chat's, set when it is created and moved with PATCH -
    # so it only counts here for a new chat.
    team_id = str(payload.get("team_id") or "")
    if team_id and not cid and not _squad_exists(team_id):
        raise HTTPException(400, f"no squad {team_id!r} in this club")
    if cid:
        try:
            chats.get_chat(cid)
        except ValueError:
            raise HTTPException(400, "bad chat id")
        except (FileNotFoundError, json.JSONDecodeError):
            raise HTTPException(404, "no such chat")
    if mid:
        try:
            jobs.job_dir(mid)       # validation only; the id is model-facing
        except ValueError:
            raise HTTPException(400, "bad match id")

    # No key is a 400 with the fix in the string, not a 500 mid-stream. The
    # fake provider needs no key (it is how the dev box and the self-checks
    # drive this route end to end).
    if not agent.fake_forced():
        try:
            agent.resolve_key()
        except agent.MissingKeyError as e:
            raise HTTPException(400, str(e))

    # Created only once every refusal above has had its say: a chat created
    # first and refused after was an empty row in the sidebar for every "hi"
    # sent before a key existed. The first message doubles as the title until
    # someone renames it.
    if not cid:
        cid = chats.create_chat(match_id=mid, title=message[:60], team_id=team_id)["id"]

    # Waits (off the event loop) for a run that was just stopped and is still
    # finishing its current delta or tool - Stop then ask again must not 409.
    lock = await run_in_threadpool(chats.try_lock, cid, agent.RUN_LOCK_WAIT_S)
    if lock is None:
        raise HTTPException(409, "a run is already active on this chat")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    cancel, _thread = agent.start(
        cid, message, lock, lambda ev: loop.call_soon_threadsafe(queue.put_nowait, ev),
        match_id=mid, model=model)

    async def stream():
        try:
            while (ev := await queue.get()) is not None:
                yield f"data: {json.dumps(ev, separators=(',', ':'))}\n\n"
        finally:
            # Reached on a normal end too, where it is a no-op. On a client that
            # went away (closed tab, browser Stop) it winds the run down. The
            # desktop proxy may not forward the abort, which is why Stop also
            # has its own route below.
            cancel.set()

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store"})


@app.post("/api/agent/chats/{cid}/stop")
def agent_stop(cid: str):
    """Stop the chat's live run. The answer so far is kept, marked stopped."""
    try:
        chats.chat_path(cid)
    except ValueError:
        raise HTTPException(400, "bad chat id")
    return {"stopped": agent.stop(cid)}


def _squad_exists(team_id: str) -> bool:
    return any(t.get("team_id") == team_id for t in jobs.org().get("teams") or [])


@app.patch("/api/agent/chats/{cid}")
async def agent_chat_update(cid: str, request: Request):
    """Rename a chat, or move it to another squad ("" files it under none)."""
    payload = await _json(request)
    title = payload.get("title")
    team_id = payload.get("team_id")
    if title is not None and not isinstance(title, str):
        raise HTTPException(400, "title must be a string")
    if team_id is not None:
        if not isinstance(team_id, str):
            raise HTTPException(400, "team_id must be a string")
        if team_id and not _squad_exists(team_id):
            raise HTTPException(400, f"no squad {team_id!r} in this club")
    try:
        return chats.update_chat(cid, title=title, team_id=team_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except (FileNotFoundError, json.JSONDecodeError):
        raise HTTPException(404, "no such chat")


@app.delete("/api/agent/chats/{cid}")
def agent_chat_delete(cid: str):
    try:
        chats.delete_chat(cid)
    except ValueError:
        raise HTTPException(400, "bad chat id")
    except chats.RunConflict as e:
        raise HTTPException(409, str(e))
    except FileNotFoundError:
        raise HTTPException(404, "no such chat")
    return {"deleted": cid}


@app.get("/api/agent/models")
def agent_models():
    """The composer's model picker. agent.MODELS is the only copy of the list."""
    return {"default": agent.DEFAULT_MODEL,
            "models": [{"id": k, "label": v} for k, v in agent.MODELS.items()]}


@app.get("/api/agent/chats")
def agent_chats():
    return chats.list_chats()


@app.get("/api/agent/chats/{cid}")
def agent_chat(cid: str):
    try:
        return chats.get_chat(cid)
    except ValueError:
        raise HTTPException(400, "bad chat id")
    except (FileNotFoundError, json.JSONDecodeError):
        raise HTTPException(404, "no such chat")


# ----------------------------------------------------------------- settings
# The one setting is a credential, so the route is deliberately narrow: PUT
# accepts exactly {"anthropic_api_key": string}, GET answers whether a key
# exists and where it comes from - never the key itself.

@app.get("/api/settings")
def get_settings():
    return agent.settings_view()


@app.put("/api/settings")
async def put_settings(request: Request):
    body = await request.body()
    if len(body) > MAX_JSON:
        raise HTTPException(413, "settings too large")
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"bad JSON: {e}")
    if not isinstance(payload, dict) or set(payload) != {"anthropic_api_key"} \
            or not isinstance(payload["anthropic_api_key"], str):
        raise HTTPException(400, 'expected exactly {"anthropic_api_key": "..."}')
    agent.save_key(payload["anthropic_api_key"].strip())
    return agent.settings_view()


@app.put("/api/roster")
async def put_roster(request: Request):
    body = await request.body()
    if len(body) > MAX_JSON:
        raise HTTPException(413, "roster too large")
    try:
        players = json.loads(body or b"[]")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"bad JSON: {e}")
    if not isinstance(players, list):
        raise HTTPException(400, "expected a list")
    return jobs.save_roster(players)


@app.exception_handler(ValueError)
def _bad_id(request: Request, exc: ValueError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


if __name__ == "__main__":
    import sys

    # Frozen (PyInstaller) builds reuse this binary as the tracking worker:
    # jobs._spawn calls `[sys.executable, "run", <job id>]` because there is no
    # jobs.py file to pass once bundled. Unfrozen, `python jobs.py run <id>`
    # stays the path and this branch never fires.
    if getattr(sys, "frozen", False) and len(sys.argv) > 2 and sys.argv[1] == "run":
        jobs.run(sys.argv[2])
    else:
        import uvicorn

        uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                    port=int(os.environ.get("PORT", 8000)))
