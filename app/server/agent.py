"""The coach-workspace agent: an LLM tool-use loop over provenance-contracted
match digests, streamed as event dicts.

    python agent.py                     # self-checks + golden eval, no network
    FUTTRAFFIC_EVAL_LIVE=1 python agent.py   # plus one real API call (needs a key)

Architecture, in one paragraph: run() is a generator of SSE-shaped event dicts
(the contract the frontend lane builds against - see the EVENTS block below).
Each turn it asks a provider for text and/or tool calls, executes the tools
against DIGESTS (never raw match.json/tracks - a 60 MB tracks.json in a
context window is a latency and cost accident), and feeds the digests back.
The Anthropic SDK is lazy-imported: it lives in app/server/requirements-build.txt only,
so every other module's self-check runs without it.

THE PROVENANCE CONTRACT is the product. Every number a tool returns carries
the tier the API serves for its field - the match's `fields`, re-resolved
against today's golden by stats.resolve - never one written here:
    ok            "sturdiest"     - a measurement backs it and no caveat binds
    rel           "relative"      - ranks/shares: biases cancel out of a ratio
    prov          "provisional"   - an absolute figure this repo does not stand behind
    not_evaluated "not evaluated" - nothing measures it (every field on a
                                    synthetic match, whatever its kind)
Sprints are WITHHELD (per match since 2026-09-28, per season since 2026-09-29):
the count read several times ground truth (N21), so the system prompt forbids
quoting one and the squad tool carries none, and any distance
with observed_frac < 1 is a known under-count. The golden eval at the bottom
pins one case on the answer actually marking a provisional number.

Guards (a coach on match day will hammer this):
  MAX_TURNS        tool turns per run, then the run ends with a note
  FRAME_CAP        get_frame calls per run, cached on (match_id, frame)
  DAILY_TOOL_CAP   tool calls per day across the whole install (agent_usage.json)
Tool exceptions become tool_error results and the run continues; LLM API
errors end the run with an error event. Stopping (stop(), or the client going
away) ends the run at the next stream delta or between tools, and what was
already said is saved with `stopped: true`.

EVENTS (one JSON object per SSE `data:` line, do not drift from the frontend):
  {"type":"start","chat_id":"..."}         always first
  {"type":"token","text":"..."}
  {"type":"tool_start","name":"...","input":{...}}
  {"type":"tool_done","name":"...","summary":"...","duration_ms":N}
  {"type":"ui_action","action":"open_tab","tab":"match|stats|lineup|squad|film|clips|data|docs",
   "match_id"?: "...", "note"?: "folder/name.html"}   ids checked before the frame goes out
  {"type":"ui_action","action":"seek_film","t":SECONDS, "match_id"?: "..."}
  {"type":"done","chat_id":"..."}
  {"type":"error","message":"..."}

FUTTRAFFIC_AGENT_FAKE=1 forces the rule-based fake provider (no network, no
key) - this is how the dev box and this file's self-check drive the real API
route end to end.
"""
from __future__ import annotations

import _path  # noqa: F401  - puts model/engine/ on sys.path

import base64
import json
import os
import re
import threading
import time
from pathlib import Path

import chats
import jobs
from paths import DATA as _DEFAULT_DATA

DATA_DIR = Path(os.environ.get("FUTTRAFFIC_DATA", _DEFAULT_DATA))

MAX_TURNS = 12                # tool turns per run
FRAME_CAP = 4                 # get_frame calls per run
DAILY_TOOL_CAP = 500          # tool calls per day, whole install
TOOL_OUTPUT_CAP = 2000        # chars of JSON per tool result, then a marker
# Streamed, so a high ceiling costs nothing unless used; thinking counts
# against it on the models that think by default, which is why 1024 cut
# answers off.
MAX_TOKENS = 32000
RUN_LOCK_WAIT_S = 10.0        # how long a new run waits for a stopping one
HISTORY_BLOCK = 3             # old tool results collapse in steps of this many runs

# THE MODEL LIST, the one copy of it: the composer's picker reads this from
# GET /api/agent/models and a run naming anything else is a 400.
MODELS = {
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-opus-5": "Claude Opus 5",
    "claude-haiku-4-5": "Claude Haiku 4.5",
}
DEFAULT_MODEL = os.environ.get("FUTTRAFFIC_AGENT_MODEL", "claude-sonnet-5")
FALLBACK_MODELS = {"claude-opus-5"}          # refusal fallbacks recommended here
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# The demo fixture every self-check and the dev smoke run against:
# Westport FC v Harbour City FC, status done, match.json + meta.facts present.
DEMO_MATCH = "786827bf0cb540079e7ae350a99124c2"

TABS = ("match", "stats", "lineup", "squad", "film", "clips", "data", "docs")


class AgentError(Exception):
    """Anything the run should surface as an `error` event rather than raise."""


class MissingKeyError(AgentError):
    """No Anthropic key anywhere. The API layer turns this into a 400 with the
    fix in the detail string - see resolve_key's message."""


# ------------------------------------------------------------------ the key

def settings_path() -> Path:
    return DATA_DIR / "settings.json"


def resolve_key() -> str:
    """ANTHROPIC_API_KEY wins; else the settings file. Never logs the key."""
    env = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if env:
        return env
    try:
        d = json.loads(settings_path().read_text())
    except (OSError, json.JSONDecodeError):
        d = {}
    key = str(d.get("anthropic_api_key") or "").strip()
    if key:
        return key
    raise MissingKeyError(
        "No Anthropic API key configured. Set the ANTHROPIC_API_KEY environment "
        "variable, or save a key with PUT /api/settings "
        '{"anthropic_api_key": "sk-ant-..."}.')


def settings_view() -> dict:
    """What GET /api/settings returns. The key itself never leaves the box."""
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return {"has_key": True, "source": "env"}
    try:
        d = json.loads(settings_path().read_text())
    except (OSError, json.JSONDecodeError):
        d = {}
    if str(d.get("anthropic_api_key") or "").strip():
        return {"has_key": True, "source": "settings"}
    return {"has_key": False, "source": "none"}


def save_key(key: str) -> None:
    """Write settings.json 0600 - it holds a credential. The only accepted
    shape is enforced by the API route; this only writes."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = settings_path().with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump({"anthropic_api_key": key}, f, indent=2)
    os.chmod(tmp, 0o600)
    os.replace(tmp, settings_path())


# ------------------------------------------------------------- provenance
# The tier travels WITH the number, so the model can quote it instead of the
# number acquiring confidence on the way into prose. It is read from the
# match's own `fields`, re-resolved (stats.resolve) exactly as the API serves
# them. This used to be a hand-kept table here that said `minutes: ok` and
# `calibration: ok` on every match, synthetic ones included, whatever the
# measurements behind them said.

TIER_LABEL = {"ok": "sturdiest", "rel": "relative", "prov": "provisional",
              "not_evaluated": "not evaluated"}

_PROV_NOTE = ("provisional: an absolute figure this repo does not stand behind "
              "- read the shape, not the value")
_DIST_NOTE = ("distance was compared with ground truth on 2026-08-17: +19% on "
              "the frames a track was observed, ~20% LOW over a whole half at "
              "0.70 coverage. Same cause as the identity contamination, so it "
              "does not cancel between players")
_PEAK_NOTE = ("peak speed is REFUSED (null) on any track where the 9.5 m/s "
              "clamp bound, because there the number was the clamp and not the "
              "player. NEVER fill a null in, and never call it slow")
# The report's own reason (stats.CAVEATS["sprints_withheld"], report.ts SPRINTS_WITHHELD_WHY).
_SPRINT_NOTE = ("WITHHELD, not reported: at the shipped config the sprint count read "
                "several times ground truth (N21), so the product publishes no sprint "
                "count - per match or per season. Never quote, estimate or sum one")


_NE_NOTE = ("not evaluated: nothing in this repo measures this figure's accuracy "
            "- never state it as accurate")


def _num(value, tier: str, note: str | None = None, **extra) -> dict:
    out = {"value": value, "tier": tier, "tier_label": TIER_LABEL[tier]}
    # The tier's own note first and a field's note after it: a field note used
    # to REPLACE "not evaluated", so a sprint count lost the warning that
    # nothing measures it.
    notes = [n for n in ({"prov": _PROV_NOTE, "not_evaluated": _NE_NOTE}.get(tier), note) if n]
    if notes:
        out["note"] = "; ".join(notes)
    out.update(extra)
    return out


def _tier(m: dict, path: str) -> str:
    """The served tier of one field; absent means nothing vouched for it."""
    return ((m.get("fields") or {}).get(path) or {}).get("confidence", "not_evaluated")


def _cap(payload) -> str:
    """One tool result as JSON, hard-capped. A tool that returns more would
    spend the context window on rows nobody reads."""
    s = json.dumps(payload, separators=(",", ":"), default=str)
    if len(s) > TOOL_OUTPUT_CAP:
        s = s[:TOOL_OUTPUT_CAP] + "...[truncated]"
    return s


# --------------------------------------------------------------- the tools

def _match_json(jid: str) -> dict:
    jobs.job_dir(jid)           # validation only; the agent's match ids are model-supplied
    p = jobs.job_dir(jid) / "match.json"
    if not p.exists():
        raise AgentError(f"match {jid[:8]} has no report yet (pipeline not done)")
    from stats import resolved_match  # noqa: PLC0415 - pulls numpy
    return resolved_match(p)


def _names(jid: str) -> dict[int, str]:
    """track_id -> roster name, best effort. Most matches have no roster
    bindings, and the agent must work there too."""
    out: dict[int, str] = {}
    try:
        import roster as R  # noqa: PLC0415

        for f in jobs._roster_files():
            try:
                r = R.Roster.load(str(f))
            except Exception:  # noqa: BLE001 - a corrupt roster costs names, not the tool
                continue
            for tid, b in (r.bindings.get(jid) or {}).items():
                pl = r.players.get((b or {}).get("player_id"))
                if pl and pl.name:
                    out[int(tid)] = pl.name
    except Exception:  # noqa: BLE001
        pass
    return out


def _player_row(p: dict, names: dict[int, str], m: dict) -> dict:
    obs = p.get("observed_frac")
    dist = _num(p.get("distance_km"), _tier(m, "players.distance_km"))
    if obs is not None:
        # Below 1.0 the distance is a known under-count (dropped occlusion
        # windows) - a floor on the damage, never a correction factor.
        dist["observed_frac"] = obs
        if obs < 1.0:
            dist["under_count"] = True
    row = {
        "track_id": p.get("id"),
        "team": p.get("team"),
        "jersey": p.get("jersey"),
        "name": names.get(int(p.get("id") or -1)),
        "minutes": _num(p.get("minutes"), _tier(m, "players.minutes")),
        "distance_km": dist,
        "top_speed_kmh": _num(p.get("top_speed_kmh"), _tier(m, "players.top_speed_kmh"),
                              note=_PEAK_NOTE),
        "sprints": _num(p.get("sprints"), _tier(m, "players.sprints"), note=_SPRINT_NOTE),
    }
    return row


def tool_list_matches(inp: dict, chat: dict, ctx: dict) -> dict:
    out = [{"id": s["id"], "name": s.get("display_name"), "status": s.get("status"),
            "date": time.strftime("%Y-%m-%d", time.localtime(s.get("created") or 0))}
           for s in jobs.all_jobs()]
    return {"matches": out, "n": len(out)}


def tool_get_report(inp: dict, chat: dict, ctx: dict) -> dict:
    m = _match_json(inp["match_id"])
    meta = m.get("meta") or {}
    teams = {}
    for side, t in (m.get("teams") or {}).items():
        # note= through _num, so the field's note is APPENDED to the tier's own
        # warning - assigned afterwards it replaced "not evaluated" on all six.
        notes = {"distance_m": _DIST_NOTE, "top_speed_kmh": _PEAK_NOTE,
                 "sprints": _SPRINT_NOTE}
        teams[side] = {k: _num(t.get(k), _tier(m, f"teams.{side}.{k}"), note=notes[k])
                       for k in ("distance_m", "top_speed_kmh", "sprints") if k in t}
    return {
        "match_id": inp["match_id"],
        "home": meta.get("home_name"), "away": meta.get("away_name"),
        "score": meta.get("score"),
        "status": meta.get("status"),
        "calibration_rms_m": _num(meta.get("calibration_rms_m"),
                                  _tier(m, "meta.calibration_rms_m")),
        # Said to the model in so many words: a synthetic match's numbers were
        # generated, and every tier on it is not_evaluated for that reason.
        "synthetic_tracking": bool(meta.get("synthetic_tracking")),
        "teams": teams,
        "territory": _num(m.get("territory"), _tier(m, "territory")),
    }


def tool_player_stats(inp: dict, chat: dict, ctx: dict) -> dict:
    m = _match_json(inp["match_id"])
    names = _names(inp["match_id"])
    players = m.get("players") or []
    tid = inp.get("track_id")
    if tid is not None:
        players = [p for p in players if int(p.get("id") or -1) == int(tid)]
        if not players:
            raise AgentError(f"no player with track_id {tid} in this match")
    rows = [_player_row(p, names, m) for p in players]
    return {"match_id": inp["match_id"], "n": len(rows), "players": rows,
            "synthetic_tracking": bool((m.get("meta") or {}).get("synthetic_tracking"))}


def tool_compare_players(inp: dict, chat: dict, ctx: dict) -> dict:
    m = _match_json(inp["match_id"])
    names = _names(inp["match_id"])
    want = {int(t) for t in inp.get("ids") or []}
    rows = [_player_row(p, names, m) for p in (m.get("players") or [])
            if int(p.get("id") or -1) in want]
    missing = sorted(want - {r["track_id"] for r in rows})
    return {"match_id": inp["match_id"], "players": rows,
            "not_found": missing,
            "synthetic_tracking": bool((m.get("meta") or {}).get("synthetic_tracking")),
            # Ranking inside one match is the safer reading. The tier on each
            # figure is the authority - it used to say "provisional" here
            # whatever the served tier was.
            "reading": "compare by rank within this match; quote each figure "
                       "with its own tier"}


def tool_list_events(inp: dict, chat: dict, ctx: dict) -> dict:
    m = _match_json(inp["match_id"])
    ev = ((m.get("meta") or {}).get("facts") or {}).get("events") or []
    keep = ("minute", "side", "type", "player", "score", "assist")
    out = [{k: e[k] for k in keep if e.get(k) is not None} for e in ev]
    return {"match_id": inp["match_id"], "n": len(out), "events": out[:40],
            "truncated": len(out) > 40}


def tool_get_frame(inp: dict, chat: dict, ctx: dict) -> dict:
    """A JPEG still for vision input. Capped per run and cached on
    (match_id, frame): identical asks are free, new ones spend the budget."""
    jid = inp["match_id"]
    t = float(inp.get("t") or 0.0)
    try:
        meta = json.loads((jobs.job_dir(jid) / "match.json").read_text()).get("meta") or {}
        fps = float(meta.get("fps") or 25.0)
    except (OSError, json.JSONDecodeError, ValueError):
        fps = 25.0
    f = max(0, int(t * fps))
    key = (jid, f)
    if key in ctx["frame_cache"]:
        return {"frame": f, "t": t, "cached": True}
    if ctx["frames"] >= FRAME_CAP:
        raise AgentError(f"frame budget reached ({FRAME_CAP}/run) - answer with "
                         f"the frames you already have")
    data = jobs.frame_jpeg(jid, f, w=960)   # ValueError becomes a tool_error
    ctx["frames"] += 1
    ctx["frame_cache"][key] = True
    # The image rides the tool_result as a base64 block for the real provider;
    # the digest is what gets persisted in the chat.
    return {"frame": f, "t": t, "jpeg_bytes": len(data),
            "_image_b64": base64.b64encode(data).decode()}


def tool_create_clip(inp: dict, chat: dict, ctx: dict) -> dict:
    """Times are seconds into the video. Clamps are stated, not silent: a clip
    the model asked for past the end of the video must come back saying what
    it actually saved."""
    jid = inp["match_id"]
    jobs.job_dir(jid)                       # validate before any clamp maths
    t0, t1 = float(inp.get("t0") or 0.0), float(inp.get("t1") or 0.0)
    clamped = []
    dur = jobs.video_seconds(jid)
    if t0 < 0:
        t0, clamped = 0.0, clamped + ["start clamped to 0 (was negative)"]
    if dur is not None and t1 > dur:
        t1 = dur
        clamped.append(f"end clamped to video end {round(dur, 1)}s")
    if dur is None:
        clamped.append("no video on disk - end not checked against duration")
    label = str(inp.get("label") or "Clip")[:120]
    clips = jobs.save_clip(jid, {"name": label, "start": t0, "end": t1})
    return {"saved": {"name": label, "start": round(t0, 3), "end": round(t1, 3)},
            "clamped": clamped, "n_clips": len(clips)}


# The note is rendered by the editor, the gallery preview and a PDF browser.
# The model's text is shaped by whatever it read, so nothing executable goes in.
_NOTE_UNSAFE = re.compile(r"<\s*(script|iframe|object|embed)|javascript:|\son\w+\s*=", re.I)


def tool_write_note(inp: dict, chat: dict, ctx: dict) -> dict:
    """A NEW note in the club's library. Create-only: base "" makes write_note
    refuse any path that already exists, so the agent can never overwrite
    something a coach wrote."""
    path = str(inp.get("path") or "").strip().strip("/")
    if not path.endswith(".html"):
        path += ".html"
    html = str(inp.get("html") or "")
    if not html.strip():
        return {"error": "empty note"}
    if _NOTE_UNSAFE.search(html):
        return {"error": "scripts, frames and event handlers are not allowed in a note"}
    try:
        n = jobs.write_note(path, html, base="")
    except jobs.NoteConflict:
        return {"error": f"a note already exists at {path!r} - pick another name"}
    except ValueError as e:
        return {"error": str(e)}
    return {"saved": path, "bytes": n}


def tool_list_clips(inp: dict, chat: dict, ctx: dict) -> dict:
    jobs.job_dir(inp["match_id"])
    clips = jobs.read_clips(inp["match_id"])
    return {"match_id": inp["match_id"], "n": len(clips),
            "clips": [{"id": c["id"], "name": c["name"], "start": c["start"],
                       "end": c["end"], "tags": c.get("tags") or []}
                      for c in clips]}


# ------------------------------------------------------------------ squads
# A chat can be filed under a squad (chats.py team_id). The agent then gets a
# short squad summary in its system prompt - who is in it and which matches it
# has - and two season tools. The team id is the slug jobs.all_jobs and
# jobs.season already filter on, so both are reused as they are.

SQUAD_DIGEST_CAP = 2000        # bytes of squad summary in the system prompt
SQUAD_ROWS = 15                # rows per season table, so a result fits TOOL_OUTPUT_CAP


def _squad(team_id: str) -> dict:
    """The club's row for a squad, or AgentError. A model-supplied id is
    checked against the club before anything is read with it."""
    for t in jobs.org().get("teams") or []:
        if t.get("team_id") == team_id:
            return t
    raise AgentError(f"no squad {team_id!r} in this club")


def _finished(team_id: str) -> list[dict]:
    """The squad's matches that have a report, oldest first."""
    out = [st for st in jobs.all_jobs(team_id)
           if (jobs.job_dir(st["id"]) / "match.json").exists()]
    return sorted(out, key=lambda st: st.get("created") or 0)


def _day(ts) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts or 0))


def squad_digest(team_id: str) -> str | None:
    """The squad summary the system prompt carries, capped. None with no squad
    or a squad the club no longer has - that chat gets the plain prompt."""
    if not team_id:
        return None
    try:
        t = _squad(team_id)
    except AgentError:
        return None
    matches = _finished(team_id)
    players = [p for p in jobs.roster() if p.get("team_id") == team_id]
    head = [
        f"SQUAD: {t.get('name')} (team_id {team_id})"
        + (f", age band {t['age_band']}" if t.get("age_band") else "")
        + (f", season {t['season']}" if t.get("season") else ""),
        "This conversation is about this squad: \"we\", \"our players\" and \"the "
        "season\" mean this squad unless the coach says otherwise. For season "
        "questions use squad_overview (results, entered feed) and squad_measured "
        "(what the pipeline measured, confirmed players only); the per-match tools "
        "take the match ids below.",
        f"MATCHES ({len(matches)} with a report; date, name, match id):",
    ]
    rows = [f"{_day(st.get('created'))}  {st.get('display_name') or st['id']}  {st['id']}"
            for st in matches]
    rows.append(f"PLAYERS ({len(players)}; shirt, name, position):")
    rows += [f"#{p['number'] if p.get('number') is not None else '-'} {p.get('name') or '?'}"
             + (f" {p['position']}" if p.get("position") else "") for p in players]
    text = "\n".join(head)
    for i, r in enumerate(rows):
        if len(text) + len(r) + 1 > SQUAD_DIGEST_CAP:
            text += f"\n(+{len(rows) - i} more lines - call squad_overview)"
            break
        text += "\n" + r
    return text


def _fit(out: dict, tables: list[str]) -> dict:
    """Drop rows from the named tables, last table first, until the result
    fits TOOL_OUTPUT_CAP - and keep each table's `shown` honest. _cap would
    otherwise cut the JSON mid-row, and a model handed half a table reads the
    half as the whole."""
    def size():
        return len(json.dumps(out, separators=(",", ":"), default=str))
    for name in reversed(tables):
        t = out.get(name)
        while t and t["rows"] and size() > TOOL_OUTPUT_CAP:
            t["rows"].pop()
            t["shown"] = len(t["rows"])
    return out


def _our_club(fixtures: list[dict], club_name: str) -> str | None:
    """Which side of the feed is us: the club in EVERY one of the squad's
    fixtures. With one fixture both clubs qualify, and the club's own name
    decides; if nothing decides, None - the tool says so rather than guessing
    a side and handing the coach the opponent's record."""
    common: set[str] | None = None
    for f in fixtures:
        pair = {f.get("home"), f.get("away")} - {None}
        common = pair if common is None else common & pair
    if not common:
        return None
    if len(common) == 1:
        return next(iter(common))
    return club_name if club_name in common else None


def tool_squad_overview(inp: dict, chat: dict, ctx: dict) -> dict:
    team_id = inp["team_id"]
    t = _squad(team_id)
    ids = {st["id"] for st in _finished(team_id)}
    feed = jobs.season_feed()
    fixtures = sorted((f for f in feed["fixtures"] if f["id"] in ids),
                      key=lambda f: f.get("date") or "")
    ours = _our_club(fixtures, (jobs.org().get("club") or {}).get("name") or "")
    out: dict = {
        "squad": t.get("name"), "team_id": team_id, "our_club": ours,
        "source": "entered match feed: human-attributed facts, NOT measured by the "
                  "pipeline; no provenance tier applies",
        "matches_with_report": len(ids), "matches_in_feed": len(fixtures),
        "demo_fixtures": sum(1 for f in fixtures if f.get("demo")),
    }
    if ours is None:
        out["error"] = ("cannot tell which side is this squad in the feed; "
                        "answer per match with get_report instead")
        return out
    clubs = {(c["match"], c["club"]): c for c in feed["clubs"]}
    rec = {"W": 0, "D": 0, "L": 0, "gf": 0, "ga": 0}
    rows = []
    for f in fixtures:
        c = clubs.get((f["id"], ours)) or {}
        res = None
        if c.get("gf") is not None and c.get("ga") is not None:
            res = "W" if c["gf"] > c["ga"] else "L" if c["gf"] < c["ga"] else "D"
            rec[res] += 1
            rec["gf"] += c["gf"]
            rec["ga"] += c["ga"]
        rows.append([(f.get("date") or "")[:10], c.get("opponent"),
                     "H" if c.get("home") else "A", f.get("score"), res, f["id"]])
    agg: dict[str, dict] = {}
    for p in feed["players"]:
        if p["match"] not in ids or p["club"] != ours:
            continue
        a = agg.setdefault(p["name"], {"apps": 0, "minutes": 0.0, "goals": 0.0,
                                       "assists": 0.0, "xg": 0.0})
        st = p.get("stats") or {}
        a["apps"] += 1
        for k in ("minutes", "goals", "assists", "xg"):
            a[k] += st.get(k) or 0
    top = sorted(agg.items(), key=lambda kv: (-kv[1]["goals"], -kv[1]["assists"],
                                              -kv[1]["minutes"]))[:SQUAD_ROWS]
    out.update({
        "record": rec,
        "matches": {"columns": ["date", "opponent", "venue", "score", "result", "match_id"],
                    "rows": rows[-SQUAD_ROWS:], "shown": min(len(rows), SQUAD_ROWS),
                    "of": len(rows)},
        "players": {"columns": ["name", "apps", "minutes", "goals", "assists", "xg"],
                    "rows": [[n, a["apps"], round(a["minutes"]), round(a["goals"]),
                              round(a["assists"]), round(a["xg"], 1)] for n, a in top],
                    "shown": len(top), "of": len(agg)},
    })
    return _fit(out, ["matches", "players"])


def tool_squad_measured(inp: dict, chat: dict, ctx: dict) -> dict:
    team_id = inp["team_id"]
    t = _squad(team_id)
    s = jobs.season(team_id)
    return _fit(_measured(t, team_id, s, inp), ["table"])


def _measured(t: dict, team_id: str, s: dict, inp: dict) -> dict:
    rows = sorted(s["players"], key=lambda r: -(r.get("km_per90") or 0))
    want = str(inp.get("player") or "").strip().lower()
    if want:
        rows = [r for r in rows if want in (r.get("name") or "").lower()]
    cov = s.get("coverage") or {}
    return {
        "squad": t.get("name"), "team_id": team_id,
        "matches": len(s.get("matches") or []),
        "coverage": {"bound": cov.get("bound", 0), "total": cov.get("total", 0),
                     "note": "only tracks a human confirmed as a named player are "
                             "counted: bound/total is the share of all tracked rows "
                             "these totals rest on. Say it whenever you quote one."},
        "tiers": {"minutes": "ok", "km_per90": "prov", "top_speed_kmh": "prov"},
        # No sprints column: jobs.season withholds them (null), and a withheld
        # number is said, not tabulated.
        "withheld": {"sprints": _SPRINT_NOTE, "sprints_per90": _SPRINT_NOTE},
        "table": {"columns": ["name", "shirt", "matches", "minutes", "km_per90",
                              "top_speed_kmh"],
                  "rows": [[r.get("name"), r.get("number"), r.get("matches"), r.get("minutes"),
                            r.get("km_per90"), r.get("top_speed_kmh")]
                           for r in rows[:SQUAD_ROWS]],
                  "shown": min(len(rows), SQUAD_ROWS), "of": len(rows)},
    }


# UI actions are NOT executed server-side: the loop emits them as ui_action
# events (fire-and-forget, the frontend owns the tabs) and hands the model a
# confirmation so the conversation can move on.
UI_ACTIONS = {"open_tab", "seek_film"}

TOOLS = {
    "list_matches": (tool_list_matches,
        "List every match on this box: id, name, status, date.", {
        "type": "object", "properties": {}}),
    "get_report": (tool_get_report,
        "Score, teams and top-level team stats for a match. Every number "
        "carries its provenance tier. NOT the full match.json.", {
        "type": "object", "required": ["match_id"],
        "properties": {"match_id": {"type": "string"}}}),
    "player_stats": (tool_player_stats,
        "Per-player rows: jersey/name, minutes, distance (with observed_frac), "
        "top speed, sprints - each number with its tier. Sprints are "
        "WITHHELD (null): never quote one.", {
        "type": "object", "required": ["match_id"],
        "properties": {"match_id": {"type": "string"},
                       "track_id": {"type": "integer"}}}),
    "compare_players": (tool_compare_players,
        "Side-by-side rows for the given track ids, same provenance as "
        "player_stats.", {
        "type": "object", "required": ["match_id", "ids"],
        "properties": {"match_id": {"type": "string"},
                       "ids": {"type": "array", "items": {"type": "integer"}}}}),
    "list_events": (tool_list_events,
        "Goals, cards and substitutions from the match facts, when present.", {
        "type": "object", "required": ["match_id"],
        "properties": {"match_id": {"type": "string"}}}),
    "get_frame": (tool_get_frame,
        "One JPEG still at t seconds into the video, for looking at. Capped "
        f"at {FRAME_CAP} per run.", {
        "type": "object", "required": ["match_id", "t"],
        "properties": {"match_id": {"type": "string"},
                       "t": {"type": "number", "description": "seconds"}}}),
    "create_clip": (tool_create_clip,
        "Save a coach clip from t0 to t1 (seconds) with a label. Times are "
        "clamped to the video and the clamp is reported.", {
        "type": "object", "required": ["match_id", "t0", "t1", "label"],
        "properties": {"match_id": {"type": "string"},
                       "t0": {"type": "number"}, "t1": {"type": "number"},
                       "label": {"type": "string"}}}),
    "write_note": (tool_write_note,
        "Save a NEW note in the club's note library. `path` is 'folder/name.html' "
        "(folders are created); an existing path is refused, never overwritten. "
        "`html` is plain HTML: h1-h3, p, ul/ol/li, strong/em, blockquote, hr, and "
        "table/tr/th/td for tables. Follow with open_tab tab=docs to show it.", {
        "type": "object", "required": ["path", "html"],
        "properties": {"path": {"type": "string"}, "html": {"type": "string"}}}),
    "list_clips": (tool_list_clips,
        "The coach-authored clips already saved for a match.", {
        "type": "object", "required": ["match_id"],
        "properties": {"match_id": {"type": "string"}}}),
    "squad_overview": (tool_squad_overview,
        "The squad's season from the entered match feed: record, results by "
        "match, and top scorers/assisters. Human-attributed facts, not pipeline "
        "measurements. team_id defaults to the conversation's squad.", {
        "type": "object",
        "properties": {"team_id": {"type": "string"}}}),
    "squad_measured": (tool_squad_measured,
        "Season totals the PIPELINE measured for the squad (minutes, km per 90, "
        "top speed; sprints are WITHHELD), counting only human-confirmed players, with "
        "the coverage they rest on. All but minutes are provisional. Optional "
        "`player` filters by name. team_id defaults to the conversation's squad.", {
        "type": "object",
        "properties": {"team_id": {"type": "string"},
                       "player": {"type": "string"}}}),
    "open_tab": (None,
        "Open a surface beside the chat, where the coach can see it while "
        "talking: film/clips a match's video, data/match/lineup/stats/squad the "
        "tables, docs a note. match_id defaults to the current match; note is a "
        "note path like 'folder/name.html'. Fire-and-forget.", {
        "type": "object", "required": ["tab"],
        "properties": {"tab": {"type": "string", "enum": list(TABS)},
                       "match_id": {"type": "string"},
                       "note": {"type": "string"}}}),
    "seek_film": (None,
        "Open the match's film beside the chat at t seconds. match_id defaults "
        "to the current match. Fire-and-forget.", {
        "type": "object", "required": ["t"],
        "properties": {"t": {"type": "number", "description": "seconds"},
                       "match_id": {"type": "string"}}}),
}


def _ui_refs(inp: dict) -> tuple[dict, str | None]:
    """The match and note a UI action names, checked: a match must be a job on
    this box, a note an existing file inside the note library. Returns the
    refs to put on the frame, or an error for the model."""
    refs: dict = {}
    mid = str(inp.get("match_id") or "")
    if mid:
        try:
            if not (jobs.job_dir(mid) / "job.json").exists():
                return {}, f"no match {mid[:8]} on this box"
        except ValueError:
            return {}, f"bad match id {mid!r}"
        refs["match_id"] = mid
    note = str(inp.get("note") or "")
    if note:
        try:
            if not jobs.note_path(note).is_file():
                return {}, f"no note {note!r}"
        except ValueError as e:
            return {}, str(e)
        refs["note"] = note
    return refs, None


def _anthropic_tools() -> list[dict]:
    return [{"name": n, "description": d, "input_schema": s}
            for n, (_f, d, s) in TOOLS.items()]


def _withheld_check() -> None:
    """A season row carrying an old sprint count (jobs.season before 2026-09-29)
    never reaches the model through squad_measured."""
    s = {"players": [{"name": "Ana", "number": 9, "matches": 1, "minutes": 90.0,
                      "km_per90": 9.87, "sprints": 7, "sprints_per90": 2100.0,
                      "top_speed_kmh": 30.1}], "matches": [{}], "coverage": {"bound": 1, "total": 1}}
    out = json.dumps(_measured({"name": "Sq"}, "sq", s, {}))
    assert "2100" not in out and '"sprints": 7' not in out, out
    assert "sprints_per90" not in _measured({"name": "Sq"}, "sq", s, {})["table"]["columns"]
    assert "WITHHELD" in _SPRINT_NOTE and "63.8" not in SYSTEM


# ------------------------------------------------------------- system prompt

SYSTEM = """You are the fut traffic match analyst, chatting with a coach about matches on this box.

RULES THAT ARE THE PRODUCT (provenance):
- Every number from a tool carries a tier: "ok" (sturdiest), "rel" (relative), "prov" (provisional), "not_evaluated" (nothing measures it). When you state a number marked "prov" you MUST say it is provisional in the same sentence; when you state one marked "not_evaluated" you MUST say its accuracy has not been measured. The tier on the number is the authority - not these rules and not your expectations.
- If get_report says synthetic_tracking is true, the positions were GENERATED, not filmed: say so before any number.
- distance_km and top_speed_kmh are never better than provisional. Distance HAS been compared with ground truth (2026-08-17): it reads ~19% high on the frames a track was observed and ~20% low over a whole half, because only 70% of the half is observed. Say which one you mean. Read the shape, not the value.
- top_speed_kmh is null on any track where the 9.5 m/s clamp bound - the pipeline refuses it rather than publishing the clamp. A null is "we will not say", NOT zero and NOT slow.
- Sprints are WITHHELD, per match and per season: at the shipped config the count read several times ground truth. NEVER quote, estimate or sum a sprint count - say it is withheld and why.
- A distance with observed_frac < 1.0 is an under-count (occlusion windows were dropped). Say so. NEVER divide by observed_frac to "correct" it.
- minutes counts occlusion gaps and track loss as played time.
- Prefer ranks within one match (relative) over absolute values; every player went through the same camera and filter, so orderings survive where absolutes do not.
- If a tool returns an error, tell the coach plainly and move on - do not invent the number.

Style: short, concrete, coach-facing. Use the tools; never guess a figure. When the coach asks to see something, use get_frame / open_tab / seek_film rather than describing from memory.
A question that starts with [match:<id>] was asked with that match selected - it is what "this match" means for that question. A conversation can move between matches; a question with no tag has no match selected.
"""


def system_blocks(digest: str | None) -> list[dict]:
    """The system prompt as blocks: the fixed rules, then the squad summary
    when there is one. The cache breakpoint sits on the last block, so tools +
    rules + squad are one cached prefix; nothing in it changes within a chat."""
    blocks = [{"type": "text", "text": SYSTEM}]
    if digest:
        blocks.append({"type": "text", "text": digest})
    blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks


# ---------------------------------------------------------------- providers
# A provider's stream() is a generator of ("text", delta) pairs followed by
# exactly one ("final", reply), where reply = {text, tool_uses, stop_reason}
# plus, optionally, `content`: the raw assistant blocks to send back within
# the run (thinking and fallback blocks must go back unchanged). `cancel` is
# checked between deltas; a set cancel ends the stream with stop_reason
# "cancelled" and whatever text already went out stays said.

def _wire(messages: list[dict]) -> list[dict]:
    """The loop tags tool_result blocks with private `_name` keys the fake
    reads. The API rejects unknown fields, so they never leave this box."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = [{k: v for k, v in b.items() if not k.startswith("_")}
                 if isinstance(b, dict) else b for b in c]
        out.append({**m, "content": c})
    return out


class AnthropicProvider:
    """The real thing. Imported lazily: anthropic is a build-only dep, and a
    box without it must still boot the API and pass every other self-check."""

    def __init__(self, model: str | None = None):
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as e:
            raise AgentError(
                "the anthropic package is not installed in this venv "
                "(.venv/bin/pip install anthropic; it ships in "
                "app/server/requirements-build.txt)") from e
        self.client = anthropic.Anthropic(api_key=resolve_key())
        self.model = model or DEFAULT_MODEL

    def stream(self, system: str, tools: list[dict], messages: list[dict],
               cancel: threading.Event | None = None):
        # Top-level cache_control caches up to the last message too, so a
        # multi-turn tool loop re-reads its own growing history from cache.
        kw = dict(model=self.model, max_tokens=MAX_TOKENS, system=system,
                  tools=tools, messages=_wire(messages),
                  cache_control={"type": "ephemeral"})
        if self.model in FALLBACK_MODELS:
            # Server-side refusal fallback: a declined request is re-run on a
            # substitute model inside the same call rather than ending the
            # answer. Opt-in and beta; only where Anthropic recommends it.
            opener = self.client.beta.messages.stream(
                betas=[FALLBACK_BETA], fallbacks="default", **kw)
        else:
            opener = self.client.messages.stream(**kw)
        with opener as s:
            for event in s:
                if cancel is not None and cancel.is_set():
                    # Leaving the `with` closes the HTTP stream, which is what
                    # stops the model spending tokens on an answer nobody reads.
                    yield "final", {"text": "", "tool_uses": [], "stop_reason": "cancelled"}
                    return
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    yield "text", event.delta.text
            final = s.get_final_message()
        yield "final", {
            "text": "".join(b.text for b in final.content if b.type == "text"),
            "tool_uses": [{"id": b.id, "name": b.name, "input": b.input}
                          for b in final.content if b.type == "tool_use"],
            "stop_reason": final.stop_reason,
            "content": final.content,
        }


class FakeProvider:
    """No network, no key. Two modes:
      - scripted: `script` is a list of replies returned verbatim, one per turn
        (the guard self-checks drive exact loops with this);
      - rule-based (script exhausted or absent): keyword rules pick plausible
        tools for the question, then a canned answer that marks provisional
        numbers. This is what FUTTRAFFIC_AGENT_FAKE=1 wires into the API, and
        what the golden eval runs against - the eval asserts tool-set
        containment, so the rules only have to choose a SUPERSET of the
        expected tools, never an exact sequence.
    """

    model = "fake"

    def __init__(self, script: list[dict] | None = None):
        self.script = list(script or [])

    # question -> tools the question obviously wants. Keep the mapping dumb:
    # this provider exists so the plumbing and the contract can be tested, not
    # to be smart.
    _RULES = (
        (("report", "score", "how did we do", "team run", "how far did"),
         [("get_report", {})]),
        (("compare",), [("compare_players", {"ids": [7, 11]})]),
        (("striker", "who ran", "player", "sprints", "number", "#", "fastest",
          "minutes", "distance"),
         [("player_stats", {})]),
        (("happened", "goal", "card", "events", "subs"), [("list_events", {})]),
        (("frame", "show me at", "look at"), [("get_frame", {"t": 600.0})]),
        (("clip",), [("create_clip", {"t0": 720.0, "t1": 750.0, "label": "Coach clip"})]),
        (("clips",), [("list_clips", {})]),
        (("open", "tab"), [("open_tab", {"tab": "film"})]),
        (("jump", "seek"), [("seek_film", {"t": 1800.0})]),
        (("matches", "list"), [("list_matches", {})]),
        (("season", "record", "top scorer"), [("squad_overview", {})]),
        (("confirmed", "per 90"), [("squad_measured", {})]),
    )

    def stream(self, system: str, tools: list[dict], messages: list[dict],
               cancel: threading.Event | None = None):
        reply = self.complete(system, tools, messages)
        # Slices, so the SSE contract (many token events) is exercised even
        # though the fake has its whole answer up front.
        for chunk in _chunks(reply.get("text") or ""):
            if cancel is not None and cancel.is_set():
                yield "final", {"text": "", "tool_uses": [], "stop_reason": "cancelled"}
                return
            if chunk:
                yield "text", chunk
        yield "final", reply

    def complete(self, system: str, tools: list[dict], messages: list[dict]) -> dict:
        if self.script:
            return self.script.pop(0)
        last = messages[-1]
        content = last.get("content")
        # A user message carrying tool_results means the tools ran; answer now.
        if isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            names = [b.get("_name", "?") for b in content if isinstance(b, dict)]
            parts = [f"Based on {', '.join(names)}:"]
            joined = " ".join(str(b.get("content")) for b in content
                              if isinstance(b, dict))
            if "create_clip" in names:
                parts.append("Clip saved.")
            if '"tier":"prov"' in joined or '"tier":"not_evaluated"' in joined:
                parts.append(
                    "The absolute figures here (distance, top speed) "
                    "are provisional or not evaluated at all - never validated "
                    "end to end, and sprints are withheld: the count read "
                    "several times ground truth. Read the shape and the "
                    "rank, not the value.")
            return {"text": " ".join(parts), "tool_uses": [],
                    "stop_reason": "end_turn"}
        q = str(content if isinstance(content, str) else
                next((b.get("text", "") for b in content if isinstance(b, dict)
                      and b.get("type") == "text"), "")).lower()
        mid = _chat_match(messages)
        calls: list[tuple[str, dict]] = []
        for words, tcs in self._RULES:
            if any(w in q for w in words):
                for name, inp in tcs:
                    if (name, inp) not in calls:
                        calls.append((name, dict(inp)))
        if not calls:
            return {"text": "I can look at the report, players, events, frames "
                            "and clips - what would you like to know?",
                    "tool_uses": [], "stop_reason": "end_turn"}
        uses = [{"id": f"fake-{i}", "name": n,
                 "input": {**inp, **({"match_id": mid} if "match_id" in
                                      TOOLS[n][2].get("properties", {}) else {})}}
                for i, (n, inp) in enumerate(calls)]
        return {"text": "", "tool_uses": uses, "stop_reason": "tool_use"}


def _stamp(text: str, match_id: str) -> str:
    """The match a question was asked about, as a prefix on that question.
    In the user turn and not the system prompt, so the system block stays
    byte-identical across a chat that moves between matches (a changing system
    prompt re-bills the whole prefix every time the match changes)."""
    return f"[match:{match_id}] {text}" if match_id else text


def _chat_match(messages: list[dict]) -> str:
    """The match stamped on the latest question - what the fake auto-fills."""
    for m in reversed(messages):
        c = m.get("content")
        if m.get("role") == "user" and isinstance(c, str):
            return c[7:c.index("]")] if c.startswith("[match:") else ""
    return ""


def fake_forced() -> bool:
    return os.environ.get("FUTTRAFFIC_AGENT_FAKE", "").strip() not in ("", "0")


def default_provider(model: str | None = None):
    return FakeProvider() if fake_forced() else AnthropicProvider(model)


# ------------------------------------------------------------ install budget
# Tool calls per day across the whole install. It was per chat, which a new
# chat reset, so it limited nothing and only stopped long squad reviews.
# Read once when a run starts, written once when it ends.

_USAGE_LOCK = threading.Lock()


def usage_path() -> Path:
    return DATA_DIR / "agent_usage.json"


def tool_calls_today() -> int:
    try:
        d = json.loads(usage_path().read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    return int(d.get("count", 0)) if d.get("day") == time.strftime("%Y-%m-%d") else 0


def charge_tool_calls(n: int) -> int:
    """Add n to today's install-wide count; returns the new total. Two runs
    ending together serialise on the lock, so neither overwrites the other."""
    with _USAGE_LOCK:
        total = tool_calls_today() + n
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = usage_path().with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"day": time.strftime("%Y-%m-%d"), "count": total}))
        os.replace(tmp, usage_path())
        return total


# ------------------------------------------------------------------ the loop

def _history(chat: dict) -> list[dict]:
    """Rebuild provider messages from the persisted chat. Each question
    carries the match it was asked about; assistant turns replay their tool
    calls so the model sees its own earlier work; images are NOT replayed (the
    digest stands in - a session of replays would re-spend the frame budget in
    tokens). An answer with no text and no tools is skipped: the API refuses
    an empty text block."""
    out: list[dict] = []
    runs = sum(1 for m in chat.get("messages") or [] if m["role"] == "assistant" and m.get("events"))
    # Tool results from runs before `cut` replay as their one-line summary. The
    # cut moves in steps of HISTORY_BLOCK, never every turn, so the replayed
    # prefix - and the prompt cache on it - stays the same for a block of runs,
    # and at least one full block of recent results is always there verbatim.
    cut = max(0, (runs // HISTORY_BLOCK - 1) * HISTORY_BLOCK)
    run_no = 0
    for m in chat.get("messages") or []:
        if m["role"] == "user":
            out.append({"role": "user", "content": _stamp(m["content"], m.get("match_id", ""))})
            continue
        old = bool(m.get("events")) and run_no < cut
        if m.get("events"):
            run_no += 1
        blocks: list[dict] = []
        if m.get("content"):
            blocks.append({"type": "text", "text": m["content"]})
        results: list[dict] = []
        for i, ev in enumerate(m.get("events") or []):
            tid = ev.get("id") or f"h{len(out)}-{i}"
            blocks.append({"type": "tool_use", "id": tid,
                           "name": ev["name"], "input": ev.get("input") or {}})
            content = (_cap({"summary": ev.get("summary") or "",
                             "note": "earlier result, not replayed in full; call "
                                     "the tool again if you need its numbers"})
                       if old else _cap(ev.get("result") or {}))
            results.append({"type": "tool_result", "tool_use_id": tid,
                            "content": content, "is_error": not ev.get("ok", True)})
        if not blocks:
            continue
        out.append({"role": "assistant", "content": blocks})
        if results:
            out.append({"role": "user", "content": results})
    return out


def _chunks(text: str, n: int = 240) -> list[str]:
    return [text[i:i + n] for i in range(0, len(text), n)] or [""]


def _summary(name: str, result: dict) -> str:
    """One line for the tool row in the UI. Short on purpose."""
    if "error" in result:
        return f"error: {result['error'][:120]}"
    if name == "list_matches":
        return f"{result.get('n', 0)} matches"
    if name == "get_report":
        return f"{result.get('home')} v {result.get('away')}, {result.get('score')}"
    if name in ("player_stats",):
        return f"{result.get('n', 0)} players"
    if name == "compare_players":
        return f"{len(result.get('players') or [])} players compared"
    if name == "list_events":
        return f"{result.get('n', 0)} events"
    if name == "get_frame":
        return f"frame {result.get('frame')} ({result.get('jpeg_bytes', 0)} B)" + \
               (" [cached]" if result.get("cached") else "")
    if name == "create_clip":
        s = result.get("saved") or {}
        note = f" ({'; '.join(result['clamped'])})" if result.get("clamped") else ""
        return f"clip {s.get('start')}s-{s.get('end')}s saved{note}"
    if name == "list_clips":
        return f"{result.get('n', 0)} clips"
    if name == "write_note":
        return f"note {result.get('saved')} saved"
    if name in UI_ACTIONS:
        return "sent to the UI"
    return "ok"


# Stop handles for the runs start() has going, by chat. POST .../stop sets one.
_CANCELS: dict[str, threading.Event] = {}
_CANCELS_GUARD = threading.Lock()


def run(chat_id: str, message: str, provider=None, match_id: str = "",
        model: str | None = None, cancel: threading.Event | None = None):
    """One agent run as a generator the CALLER consumes to the end - the
    self-checks and scripts. It takes and releases the chat's run lock itself;
    a busy chat yields a single conflict error.

    The API does not use this. A response can be dropped before its generator
    ever starts, a generator that never started never runs its `finally`, and
    that is exactly how the lock used to leak and 409 a chat until restart.
    The API uses start(), whose thread always runs to its end."""
    lock = chats.try_lock(chat_id)
    if lock is None:
        yield {"type": "error", "message": "a run is already active on this chat"}
        return
    try:
        yield from _run_locked(chat_id, message, provider, match_id, model,
                               cancel or threading.Event())
    finally:
        lock.release()


def start(chat_id: str, message: str, lock: threading.Lock, emit, provider=None,
          match_id: str = "", model: str | None = None):
    """Run in a thread that owns `lock` (already acquired) and releases it when
    the run ends - however it ends, and whether or not anyone is reading.
    `emit(event)` receives every event and then None. Returns (cancel, thread):
    setting `cancel` (or stop(chat_id)) ends the run at the next delta or
    between tools, and what was already said is saved as a stopped answer.

        route ── try_lock(wait) ──▶ start() ──▶ thread: _run_locked ──emit──▶ queue ──▶ SSE
                                      │                    │ finally: release lock, emit(None)
        POST /stop ── stop(cid) ──────┴──▶ cancel.set() ───┘   (disconnect sets it too)
    """
    cancel = threading.Event()
    with _CANCELS_GUARD:
        _CANCELS[chat_id] = cancel

    def say(ev):
        try:
            emit(ev)
        except Exception:  # noqa: BLE001 - a reader that went away must not stop the run saving
            pass

    def work():
        try:
            for ev in _run_locked(chat_id, message, provider, match_id, model, cancel):
                say(ev)
        except Exception as e:  # noqa: BLE001 - the stream's last word, not a dead thread
            say({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            with _CANCELS_GUARD:
                if _CANCELS.get(chat_id) is cancel:
                    del _CANCELS[chat_id]
            lock.release()
            say(None)

    t = threading.Thread(target=work, name=f"agent-run-{chat_id[:8]}", daemon=True)
    t.start()
    return cancel, t


def stop(chat_id: str) -> bool:
    """Ask the chat's live run to stop. False when nothing is running."""
    with _CANCELS_GUARD:
        cancel = _CANCELS.get(chat_id)
    if cancel is None:
        return False
    cancel.set()
    return True


def _run_locked(chat_id: str, message: str, provider, match_id: str,
                model: str | None, cancel: threading.Event):
    chats.append_message(chat_id, "user", message, match_id=match_id)
    # Reload AFTER the append: the history the provider sees must include the
    # message just written, and append_message returns the message, not the chat.
    chat = chats.get_chat(chat_id)
    # First, so a client that sent no chat_id can Stop a run it cannot yet name.
    yield {"type": "start", "chat_id": chat_id}
    history = _history(chat)
    team_id = chat.get("team_id") or ""
    system = system_blocks(squad_digest(team_id))

    if provider is None:
        try:
            provider = default_provider(model)
        except AgentError as e:
            yield {"type": "error", "message": str(e)}
            return

    ctx = {"frames": 0, "frame_cache": {}}
    tool_events: list[dict] = []
    said: list[str] = []
    turns = 0
    spent, budget = 0, DAILY_TOOL_CAP - tool_calls_today()
    stopped = False
    error: str | None = None

    try:
        while not stopped:
            if cancel.is_set():
                stopped = True
                break
            reply, parts = None, []
            try:
                for kind, payload in provider.stream(system, _anthropic_tools(), history, cancel):
                    if kind == "text":
                        parts.append(payload)
                        yield {"type": "token", "text": payload}
                    else:
                        reply = payload
            except AgentError as e:
                error = str(e)
                break
            except Exception as e:  # noqa: BLE001 - LLM API error: end the run, say why
                error = f"LLM call failed: {type(e).__name__}: {e}"
                break
            text = "".join(parts)
            if text:
                said.append(text)
            reason = (reply or {}).get("stop_reason")
            if reply is None or reason == "cancelled":
                stopped = True
                break
            note = {"refusal": "(the model declined to answer this)",
                    "max_tokens": "(the answer was cut off at the length limit)"}.get(reason)
            if note:
                said.append(note)
                yield {"type": "token", "text": note}

            uses = reply.get("tool_uses") or []
            if not uses:
                break
            turns += 1
            if turns > MAX_TURNS:
                note = f"(stopped after {MAX_TURNS} tool turns)"
                said.append(note)
                yield {"type": "token", "text": note}
                break

            history.append({"role": "assistant", "content": reply.get("content") or (
                ([{"type": "text", "text": text}] if text else []) +
                [{"type": "tool_use", "id": u["id"], "name": u["name"],
                  "input": u["input"]} for u in uses])})
            results: list[dict] = []
            for u in uses:
                name, inp = u["name"], u.get("input") or {}
                props = TOOLS.get(name, (None, "", {}))[2].get("properties", {})
                if "match_id" in props and not inp.get("match_id") and match_id:
                    inp = {**inp, "match_id": match_id}
                if "team_id" in props and not inp.get("team_id"):
                    inp = {**inp, "team_id": team_id}
                yield {"type": "tool_start", "name": name, "input": inp}
                t0 = time.time()
                if spent >= budget:
                    result = {"error": f"daily tool budget reached ({DAILY_TOOL_CAP} "
                                       f"tool calls a day across all chats)"}
                elif name in UI_ACTIONS:
                    # The one path where model output goes to the browser without
                    # a tool reading it first - so every id is checked here,
                    # before the frame is emitted, the way the data tools check
                    # theirs by using them.
                    refs, bad = _ui_refs(inp)
                    if bad:
                        result = {"error": bad}
                    elif name == "open_tab":
                        tab = str(inp.get("tab") or "")
                        if tab not in TABS:
                            result = {"error": f"unknown tab {tab!r}; one of {', '.join(TABS)}"}
                        else:
                            yield {"type": "ui_action", "action": "open_tab", "tab": tab, **refs}
                            result = {"ok": True, "note": "the UI was asked to open the tab"}
                    else:
                        try:
                            t = float(inp.get("t"))
                            yield {"type": "ui_action", "action": "seek_film", "t": t,
                                   **{k: v for k, v in refs.items() if k == "match_id"}}
                            result = {"ok": True, "note": f"the UI was asked to seek to {t}s"}
                        except (TypeError, ValueError):
                            result = {"error": f"bad seek time {inp.get('t')!r}"}
                elif "team_id" in inp and not inp["team_id"]:
                    result = {"error": "this conversation has no squad; pass team_id "
                                       "(list_matches shows each match's team)"}
                else:
                    fn = TOOLS.get(name, (None,))[0]
                    if fn is None:
                        result = {"error": f"unknown tool {name!r}"}
                    else:
                        try:
                            result = fn(inp, chat, ctx)
                        except Exception as e:  # noqa: BLE001 - tool error: tell the model, run on
                            result = {"error": f"{type(e).__name__}: {e}"}
                spent += 1
                ok = "error" not in result
                summary = _summary(name, result)
                duration_ms = round((time.time() - t0) * 1000)
                yield {"type": "tool_done", "name": name,
                       "summary": summary, "duration_ms": duration_ms}
                persisted = {k: v for k, v in result.items() if k != "_image_b64"}
                # summary + duration_ms are saved so a reopened chat can redraw the
                # same tool row the stream drew. The saved shape is a contract with
                # app/frontend/src/lib/agent.ts, pinned by the golden fixture that
                # _api_selfcheck compares against.
                tool_events.append({"id": u["id"], "name": name, "input": inp,
                                    "result": persisted, "ok": ok,
                                    "summary": summary, "duration_ms": duration_ms})
                block: dict = {"type": "tool_result", "tool_use_id": u["id"],
                               "_name": name, "is_error": not ok}
                if result.get("_image_b64"):
                    block["content"] = [
                        {"type": "image", "source": {"type": "base64",
                                                     "media_type": "image/jpeg",
                                                     "data": result["_image_b64"]}},
                        {"type": "text", "text": _cap(persisted)}]
                else:
                    block["content"] = _cap(result)
                results.append(block)
                # Between tools, not mid-tool: a clip half-saved is worse than
                # a stop that lands one tool late.
                if cancel.is_set():
                    stopped = True
                    break
            history.append({"role": "user", "content": results})
    finally:
        # Saved however the run ends - finished, stopped, failed, or the
        # caller walking away - so the history never holds an answer the coach
        # saw but the file does not.
        if spent:
            charge_tool_calls(spent)
        if said or tool_events or stopped:
            chats.append_message(chat_id, "assistant", "\n".join(s for s in said if s),
                                 events=tool_events, model=getattr(provider, "model", None),
                                 stopped=stopped)

    if error:
        yield {"type": "error", "message": error}
        return
    yield {"type": "done", "chat_id": chat_id}


# ------------------------------------------------------------------ selfcheck

def _fixture_into(tmp: Path) -> str:
    """Copy the demo job's job.json + match.json into the temp jobs dir. The
    self-check runs the tools against REAL pipeline output (the provenance
    asserts are meaningless against a fixture I authored), but never writes
    into the repo's data dir."""
    # The data dir first, then the PACKAGED copy under app/desktop/resources. This
    # is the same fix db.py::_demo already carries, for the same reason: the
    # fixture used to be found only because a copy happened to be sitting in
    # the developer's library, so deleting that match - an ordinary thing to do
    # - broke this self-check with a bare FileNotFoundError, and it is broken
    # on every fresh clone.
    packaged = Path(__file__).resolve().parents[2] / "app" / "desktop" / "resources" / "demo-job"
    for src in (Path(_DEFAULT_DATA) / "jobs" / DEMO_MATCH, packaged / DEMO_MATCH):
        if all((src / n).exists() for n in ("job.json", "match.json")):
            break
    else:
        raise SystemExit(f"demo job {DEMO_MATCH} not found under "
                         f"{Path(_DEFAULT_DATA) / 'jobs'} or {packaged}")
    dst = tmp / "jobs" / DEMO_MATCH
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("job.json", "match.json"):
        (dst / name).write_bytes((src / name).read_bytes())
    return DEMO_MATCH


def _selfcheck() -> None:
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="futagent-"))
    old_env = {k: os.environ.get(k) for k in
               ("FUTTRAFFIC_DATA", "ANTHROPIC_API_KEY", "FUTTRAFFIC_AGENT_FAKE")}
    try:
        os.environ["FUTTRAFFIC_DATA"] = str(tmp)
        # Repoint the already-imported modules at the temp dir - jobs.py's
        # self-check established this pattern (module globals, not re-import).
        jobs.DATA_DIR, jobs.JOBS_DIR = tmp, tmp / "jobs"
        jobs.ROSTER_PATH = tmp / "roster.json"
        chats.DATA_DIR, chats.CHATS_DIR = tmp, tmp / "chats"
        global DATA_DIR
        DATA_DIR = tmp
        mid = _fixture_into(tmp)

        # --- MISSING KEY. Neither env nor settings file: resolve_key must
        # refuse with the actionable message the API turns into a 400.
        os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            resolve_key()
            raise AssertionError("resolve_key succeeded with no key anywhere")
        except MissingKeyError as e:
            assert "PUT /api/settings" in str(e)
        os.environ["ANTHROPIC_API_KEY"] = "sk-env-wins"
        assert resolve_key() == "sk-env-wins"          # env beats the file
        save_key("sk-from-settings")
        assert settings_view()["source"] == "env"
        os.environ.pop("ANTHROPIC_API_KEY", None)
        assert resolve_key() == "sk-from-settings"
        assert oct(settings_path().stat().st_mode & 0o777) == "0o600", \
            "settings.json holds a credential and must be 0600"
        assert settings_view() == {"has_key": True, "source": "settings"}
        settings_path().unlink()
        assert settings_view() == {"has_key": False, "source": "none"}

        chat = chats.create_chat(match_id=mid, title="self-check")
        cid = chat["id"]

        # --- FULL LOOP, scripted fake: one tool turn, then the answer.
        full = list(run(cid, "show me the report", provider=FakeProvider(script=[
            {"text": "", "stop_reason": "tool_use", "tool_uses": [
                {"id": "t1", "name": "get_report", "input": {"match_id": mid}}]},
            {"text": "Westport FC won 3 - 1.", "tool_uses": [],
             "stop_reason": "end_turn"},
        ])))
        kinds = [e["type"] for e in full]
        assert kinds[:2] == ["start", "tool_start"] and "tool_done" in kinds, kinds
        assert full[0]["chat_id"] == cid
        assert kinds[-1] == "done" and full[-1]["chat_id"] == cid
        assert any(e["type"] == "token" and "3 - 1" in e["text"] for e in full)
        done = chats.get_chat(cid)
        assert [m["role"] for m in done["messages"]] == ["user", "assistant"]
        ev = done["messages"][1]["events"][0]
        assert ev["name"] == "get_report" and ev["ok"] and ev["result"]["score"]

        # --- the report digest carries tiers and NOT the full match.json.
        rep = tool_get_report({"match_id": mid}, chat, {})
        assert rep["score"] == "3 - 1" and "heatmaps" not in rep
        # The tier the agent quotes IS the tier the API serves, field by field.
        served = _match_json(mid)
        assert rep["teams"]["home"]["distance_m"]["tier"] == _tier(served, "teams.home.distance_m")
        assert rep["calibration_rms_m"]["tier"] == _tier(served, "meta.calibration_rms_m")
        assert rep["calibration_rms_m"]["tier"] != "ok", \
            "calibration quoted as ok: nothing measures a job's own fit residual"

        # --- PROVENANCE in player_stats: every number carries its tier, and
        # distance carries observed_frac. This is the contract the whole
        # feature exists for.
        ps = tool_player_stats({"match_id": mid}, chat, {})
        row = ps["players"][0]
        for k in ("minutes", "distance_km", "top_speed_kmh", "sprints"):
            assert row[k]["tier"] == _tier(served, f"players.{k}"), (k, row[k])
            assert "note" in row[k] or row[k]["tier"] in ("ok", "rel"), row[k]
        assert "observed_frac" in row["distance_km"]
        assert "WITHHELD" in row["sprints"]["note"] and row["sprints"]["value"] is None
        # The fixture is the packaged demo, which is SYNTHETIC: every payload
        # says so, and a field note never replaces the tier's own warning.
        assert ps["synthetic_tracking"] is True and rep["synthetic_tracking"] is True
        assert row["sprints"]["tier"] == "not_evaluated", row["sprints"]
        assert "not evaluated" in row["sprints"]["note"], row["sprints"]["note"]
        for side in rep["teams"].values():        # team rows too, not only players
            for k, v in side.items():
                assert v["tier"] != "not_evaluated" or "not evaluated" in v["note"], (k, v)
        cmp_ = tool_compare_players({"match_id": mid, "ids": [row["track_id"], 999]},
                                    chat, {})
        assert cmp_["not_found"] == [999] and len(cmp_["players"]) == 1

        # --- MAX-TURNS guard: a provider that always asks for another tool
        # must be stopped, and the run still ends with done.
        loop = FakeProvider(script=[
            {"text": "", "stop_reason": "tool_use", "tool_uses": [
                {"id": f"x{i}", "name": "list_matches", "input": {}}]}
            for i in range(MAX_TURNS + 5)])
        kinds = [e["type"] for e in run(cid, "loop forever", provider=loop)]
        assert kinds.count("tool_start") == MAX_TURNS, kinds
        assert kinds[-1] == "done", kinds

        # --- TOOL ERROR path: a tool raising (bad match id) becomes a
        # tool_error result, the model is told, and the run CONTINUES.
        err = list(run(cid, "bad match", provider=FakeProvider(script=[
            {"text": "", "stop_reason": "tool_use", "tool_uses": [
                {"id": "e1", "name": "get_report", "input": {"match_id": "0" * 32}}]},
            {"text": "That match has no report.", "tool_uses": [],
             "stop_reason": "end_turn"},
        ])))
        td = next(e for e in err if e["type"] == "tool_done")
        assert td["summary"].startswith("error:"), td
        assert err[-1]["type"] == "done"
        # a malformed id (traversal-shaped) is a tool error too, not a crash
        pth = list(run(cid, "evil", provider=FakeProvider(script=[
            {"text": "", "stop_reason": "tool_use", "tool_uses": [
                {"id": "e2", "name": "get_report", "input": {"match_id": "../.."}}]},
            {"text": "No.", "tool_uses": [], "stop_reason": "end_turn"},
        ])))
        assert any(e["type"] == "tool_done" and "bad job id" in e["summary"]
                   for e in pth), pth

        # --- CLIP CLAMPING: negative in-point clamps to 0 and SAYS so; with
        # no video on disk the out-point is reported as unchecked. Then with a
        # duration the out-point clamps too.
        r = tool_create_clip({"match_id": mid, "t0": -5.0, "t1": 42.0,
                              "label": "first press"}, chat, {})
        assert r["saved"]["start"] == 0.0, r
        assert any("clamped" in c for c in r["clamped"]), r
        assert any("no video" in c for c in r["clamped"]), r
        real_vs = jobs.video_seconds
        jobs.video_seconds = lambda j: 30.0
        try:
            r2 = tool_create_clip({"match_id": mid, "t0": 10.0, "t1": 99.0,
                                   "label": "over the end"}, chat, {})
        finally:
            jobs.video_seconds = real_vs
        assert r2["saved"]["end"] == 30.0 and any("clamped" in c for c in r2["clamped"])
        assert len(jobs.read_clips(mid)) == 2
        assert tool_list_clips({"match_id": mid}, chat, {})["n"] == 2

        # --- WRITE_NOTE: creates, never overwrites, refuses script.
        r = tool_write_note({"path": "agent/Press", "html": "<h1>Press</h1>"}, chat, {})
        assert r == {"saved": "agent/Press.html", "bytes": 14}, r
        r = tool_write_note({"path": "agent/Press.html", "html": "<p>x</p>"}, chat, {})
        assert "already exists" in r["error"], r
        assert jobs.note_path("agent/Press.html").read_text() == "<h1>Press</h1>"
        for bad in ("<script>x</script>", '<img src=a onerror="x">', '<a href="javascript:x">'):
            assert "not allowed" in tool_write_note({"path": "b.html", "html": bad}, chat, {})["error"]
        assert "bad note path" in tool_write_note({"path": "../x.html", "html": "<p/>"}, chat, {})["error"]

        # --- GET_FRAME cap + cache: 4 fresh frames a run, identical asks free.
        real_fj = jobs.frame_jpeg
        jobs.frame_jpeg = lambda j, f=0, w=0: b"\xff\xd8fake-jpeg"
        try:
            fctx = {"frames": 0, "frame_cache": {}}
            for i in range(FRAME_CAP):
                tool_get_frame({"match_id": mid, "t": float(i * 10)}, chat, fctx)
            again = tool_get_frame({"match_id": mid, "t": 0.0}, chat, fctx)
            assert again.get("cached"), "identical frame asks must hit the cache"
            try:
                tool_get_frame({"match_id": mid, "t": 999.0}, chat, fctx)
                raise AssertionError("5th distinct frame was not capped")
            except AgentError as e:
                assert "budget" in str(e)
        finally:
            jobs.frame_jpeg = real_fj

        # --- UI ACTIONS: emitted fire-and-forget, confirmed to the model.
        ui = list(run(cid, "open the film tab", provider=FakeProvider(script=[
            {"text": "", "stop_reason": "tool_use", "tool_uses": [
                {"id": "u1", "name": "open_tab", "input": {"tab": "film"}},
                {"id": "u2", "name": "seek_film", "input": {"t": 95.5}}]},
            {"text": "Done.", "tool_uses": [], "stop_reason": "end_turn"},
        ])))
        acts = [e for e in ui if e["type"] == "ui_action"]
        assert acts[0] == {"type": "ui_action", "action": "open_tab", "tab": "film"}
        assert acts[1] == {"type": "ui_action", "action": "seek_film", "t": 95.5}
        assert sum(1 for e in ui if e["type"] == "tool_done") == 2

        # --- UI ACTIONS NAME THINGS NOW, and every name is checked before the
        # frame reaches the browser: the match of the question fills in, a real
        # note goes through, and a traversal or a missing file never does.
        (jobs.notes_dir() / "club").mkdir(parents=True, exist_ok=True)
        (jobs.notes_dir() / "club" / "press.html").write_text("<p>press</p>")
        named = list(run(cid, "open it", match_id=mid, provider=FakeProvider(script=[
            {"text": "", "stop_reason": "tool_use", "tool_uses": [
                {"id": "n1", "name": "open_tab", "input": {"tab": "docs", "note": "club/press.html"}},
                {"id": "n2", "name": "seek_film", "input": {"t": 12.0}},
                {"id": "n3", "name": "open_tab", "input": {"tab": "docs", "note": "../../settings.json"}},
                {"id": "n4", "name": "open_tab", "input": {"tab": "film", "match_id": "0" * 32}},
                {"id": "n5", "name": "open_tab", "input": {"tab": "docs", "note": "club/nope.html"}},
                {"id": "n6", "name": "open_tab", "input": {"tab": "data"}}]},
            {"text": "Done.", "tool_uses": [], "stop_reason": "end_turn"},
        ])))
        acts = [e for e in named if e["type"] == "ui_action"]
        assert acts == [
            {"type": "ui_action", "action": "open_tab", "tab": "docs", "note": "club/press.html",
             "match_id": mid},
            {"type": "ui_action", "action": "seek_film", "t": 12.0, "match_id": mid},
            {"type": "ui_action", "action": "open_tab", "tab": "data", "match_id": mid},
        ], acts
        errs = [e["summary"] for e in named if e["type"] == "tool_done" and e["summary"].startswith("error")]
        assert len(errs) == 3 and "no match" in errs[1] and "no note" in errs[2], errs

        # --- RUN LOCK conflict: a second run on a busy chat yields one error.
        held = chats.try_lock(cid)
        assert held is not None
        conflict = list(run(cid, "concurrent", provider=FakeProvider()))
        assert conflict == [{"type": "error",
                             "message": "a run is already active on this chat"}]
        held.release()

        # --- MISSING KEY surfaces as an error event from run() itself: no env
        # key, no settings file (both cleared above), fake not forced.
        mk = list(run(cid, "no key", provider=None))
        assert mk[-1]["type"] == "error" and "API key" in mk[-1]["message"], mk

        # --- PER-MESSAGE MATCH. A chat that moves to another match answers
        # about the one the question was asked on, not the one it started on:
        # the auto-filled match_id, the stamp on the question, and the saved
        # message all name match B. The system prompt does not change with it.
        other = "b" * 32
        moved = chats.create_chat(match_id=mid, title="moves")
        list(run(moved["id"], "how did we do?", provider=FakeProvider(), match_id=mid))
        seen: dict = {}

        class _Spy(FakeProvider):
            def complete(self, system, tools, messages):
                seen.setdefault("system", system)
                seen.setdefault("history", list(messages))  # the loop appends to it
                return super().complete(system, tools, messages)
        evs = list(run(moved["id"], "and the score in this match?", provider=_Spy(), match_id=other))
        starts = [e for e in evs if e["type"] == "tool_start"]
        assert starts and all(e["input"].get("match_id") == other for e in starts), starts
        assert seen["history"][-1]["content"] == f"[match:{other}] and the score in this match?"
        assert seen["history"][0]["content"].startswith(f"[match:{mid}]"), seen["history"][0]
        assert seen["system"] == system_blocks(None) and other not in str(seen["system"])
        saved = chats.get_chat(moved["id"])
        assert saved["match_id"] == other
        assert saved["messages"][-2]["match_id"] == other

        # --- STOP. start() runs in a thread that owns the lock; stop() ends the
        # run at the next delta, the partial answer is saved as stopped, and the
        # lock is free again - with nobody reading the events at all.
        class _Blocking(FakeProvider):
            def __init__(self):
                super().__init__()
                self.streaming = threading.Event()

            def stream(self, system, tools, messages, cancel=None):
                yield "text", "Half an ans"
                self.streaming.set()
                cancel.wait(5)
                yield "final", {"text": "", "tool_uses": [], "stop_reason": "cancelled"}

        sc = chats.create_chat(match_id=mid, title="stop")
        blocking = _Blocking()
        lock = chats.try_lock(sc["id"])
        got: list = []
        cancel, th = start(sc["id"], "long question", lock, got.append, provider=blocking)
        assert blocking.streaming.wait(5), "the blocking provider never streamed"
        assert chats.run_active(sc["id"])
        t0 = time.time()
        assert stop(sc["id"]) is True
        th.join(5)
        assert not th.is_alive() and time.time() - t0 < 2, "stop did not end the run"
        assert not chats.run_active(sc["id"]), "a stopped run kept the lock"
        assert stop(sc["id"]) is False, "nothing is running any more"
        last = chats.get_chat(sc["id"])["messages"][-1]
        assert last["stopped"] is True and last["content"] == "Half an ans", last
        assert got[-1] is None and got[-2]["type"] == "done", got[-3:]

        # --- THE LEAK THIS REPLACES: a run whose events nobody ever reads still
        # releases the lock. (On 51e77a3 the lock was released only in a sync
        # generator's finally, and closing that generator before its first
        # next() left the chat locked - reproduced before this change.)
        unread = chats.try_lock(sc["id"])
        _c, th = start(sc["id"], "nobody listens", unread, lambda ev: None,
                       provider=FakeProvider())
        th.join(5)
        assert not chats.run_active(sc["id"]), "an unread run leaked the lock"

        # --- THE REAL PROVIDER'S STREAM, against a stub client: text deltas go
        # out in order, tool_use blocks come back in the final reply with the
        # raw content to echo, private `_name` keys never reach the wire, the
        # beta fallback path is taken only for the models it is meant for, and
        # a cancel leaves the SDK stream (closing the HTTP request).
        from types import SimpleNamespace as NS
        calls: list = []

        class _Stream:
            def __init__(self, kw):
                self.kw, self.closed = kw, False
                calls.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                self.closed = True

            def __iter__(self):
                yield NS(type="message_start")
                for t in ("Two ", "players"):
                    yield NS(type="content_block_delta", delta=NS(type="text_delta", text=t))
                yield NS(type="content_block_delta", delta=NS(type="input_json_delta"))

            def get_final_message(self):
                return NS(stop_reason="tool_use", content=[
                    NS(type="text", text="Two players"),
                    NS(type="tool_use", id="tu1", name="player_stats", input={"match_id": mid})])

        prov = AnthropicProvider.__new__(AnthropicProvider)
        prov.client = NS(messages=NS(stream=lambda **kw: _Stream(kw)),
                         beta=NS(messages=NS(stream=lambda **kw: _Stream(kw))))
        prov.model = "claude-sonnet-5"
        wire_msgs = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x",
                                                   "_name": "list_matches", "content": "{}"}]}]
        out = list(prov.stream(SYSTEM, [], wire_msgs, threading.Event()))
        assert [p for k, p in out if k == "text"] == ["Two ", "players"], out
        fin = out[-1][1]
        assert fin["tool_uses"] == [{"id": "tu1", "name": "player_stats", "input": {"match_id": mid}}]
        assert fin["content"] and fin["stop_reason"] == "tool_use"
        sent = calls[-1].kw
        assert "_name" not in sent["messages"][0]["content"][0], "private key reached the API"
        assert "fallbacks" not in sent and sent["max_tokens"] == MAX_TOKENS
        prov.model = "claude-opus-5"
        list(prov.stream(SYSTEM, [], [], threading.Event()))
        assert calls[-1].kw["fallbacks"] == "default" and calls[-1].kw["betas"] == [FALLBACK_BETA]
        stopping = threading.Event()
        stopping.set()
        out = list(prov.stream(SYSTEM, [], [], stopping))
        assert out == [("final", {"text": "", "tool_uses": [], "stop_reason": "cancelled"})], out
        assert calls[-1].closed, "a cancelled stream was left open"

        # --- SQUADS. The demo job's team is "Default", so the club seeds a
        # `default` squad from it. Give the club a name and the squad a player.
        jobs.save_org({"club": {"name": "Westport FC"}})
        jobs.save_roster([{"team": "Default", "name": "Test Keeper", "number": 1,
                           "position": "GK"}])
        assert squad_digest("") is None and squad_digest("no-such-squad") is None
        dig = squad_digest("default")
        assert "SQUAD: Default (team_id default)" in dig and mid in dig and "#1 Test Keeper GK" in dig, dig
        assert len(dig) <= SQUAD_DIGEST_CAP
        real_roster = jobs.roster
        jobs.roster = lambda: [{"team_id": "default", "name": f"Player {i}", "number": i}
                               for i in range(400)]
        try:
            big = squad_digest("default")
            assert len(big) <= SQUAD_DIGEST_CAP + 60 and "more lines" in big, big[-80:]
        finally:
            jobs.roster = real_roster
        blocks = system_blocks(dig)
        assert [b["text"] for b in blocks] == [SYSTEM, dig]
        assert "cache_control" in blocks[-1] and "cache_control" not in blocks[0]

        # our side: the club in every fixture; ambiguous with one fixture unless
        # the club's own name settles it; never a guess.
        fx = [{"home": "Westport FC", "away": "Harbour City FC"}]
        assert _our_club(fx, "Westport FC") == "Westport FC"
        assert _our_club(fx, "") is None
        assert _our_club(fx + [{"home": "LA", "away": "Westport FC"}], "") == "Westport FC"

        ov = tool_squad_overview({"team_id": "default"}, chat, {})
        assert ov["our_club"] == "Westport FC" and ov["record"]["W"] == 1, ov
        assert ov["matches"]["rows"][0][1] == "Harbour City FC" and ov["matches"]["rows"][0][4] == "W"
        assert ov["players"]["rows"] and "NOT measured" in ov["source"]
        assert len(_cap(ov)) <= TOOL_OUTPUT_CAP and not _cap(ov).endswith("[truncated]"), len(json.dumps(ov))
        # a result too big for the cap loses whole rows, never half a row
        long = {"a": "x" * 100, "t": {"rows": [["n" * 80] for _ in range(40)], "shown": 40, "of": 40}}
        fitted = _fit(long, ["t"])
        assert len(json.dumps(fitted, separators=(",", ":"))) <= TOOL_OUTPUT_CAP
        assert fitted["t"]["shown"] == len(fitted["t"]["rows"]) < 40 and fitted["t"]["of"] == 40
        me = tool_squad_measured({"team_id": "default"}, chat, {})
        assert me["tiers"]["km_per90"] == "prov" and me["tiers"]["minutes"] == "ok"
        assert "sprints_per90" not in me["table"]["columns"] and "WITHHELD" in me["withheld"]["sprints"]
        assert "confirmed" in me["coverage"]["note"] and me["coverage"]["total"] > 0, me
        try:
            tool_squad_overview({"team_id": "../../etc"}, chat, {})
            raise AssertionError("an unknown squad id was read")
        except AgentError:
            pass

        # the loop fills the chat's squad in, and refuses a squad tool when the
        # chat has none rather than reading the whole club.
        sqc = chats.create_chat(team_id="default", title="squad")
        evs = list(run(sqc["id"], "what is our season record?", provider=FakeProvider()))
        st_ = next(e for e in evs if e["type"] == "tool_start")
        assert st_["name"] == "squad_overview" and st_["input"]["team_id"] == "default", st_
        assert "error" not in chats.get_chat(sqc["id"])["messages"][-1]["events"][0]["result"]
        nosq = chats.create_chat(title="no squad")
        list(run(nosq["id"], "what is our season record?", provider=FakeProvider()))
        assert "no squad" in chats.get_chat(nosq["id"])["messages"][-1]["events"][0]["result"]["error"]

        # history: old tool results collapse to their summary, in blocks, so the
        # replayed prefix only changes every HISTORY_BLOCK runs.
        def _hist_chat(n_runs):
            msgs = []
            for i in range(n_runs):
                msgs += [{"role": "user", "content": f"q{i}"},
                         {"role": "assistant", "content": f"a{i}", "events": [
                             {"id": f"t{i}", "name": "list_matches", "input": {},
                              "result": {"n": i, "matches": ["x" * 50]}, "ok": True,
                              "summary": f"{i} matches"}]}]
            return {"messages": msgs}

        def _collapsed(h):
            return [json.loads(b["content"]).get("summary") for m in h if m["role"] == "user"
                    and isinstance(m["content"], list) for b in m["content"]
                    if "summary" in json.loads(b["content"])]
        assert _collapsed(_history(_hist_chat(5))) == []
        six = _history(_hist_chat(6))
        assert _collapsed(six) == ["0 matches", "1 matches", "2 matches"], _collapsed(six)
        seven, eight = _history(_hist_chat(7)), _history(_hist_chat(8))
        assert seven[:len(six)] == six and eight[:len(six)] == six, "the prefix moved mid-block"
        assert len(_collapsed(_history(_hist_chat(9)))) == 6

        # --- DAILY CAP, install-wide: at the budget, tools refuse in EVERY
        # chat - a fresh chat used to reset it - and the next day resets it.
        charge_tool_calls(DAILY_TOOL_CAP - tool_calls_today())
        for c_id in (cid, chats.create_chat(match_id=mid)["id"]):
            capd = list(run(c_id, "over budget", provider=FakeProvider(script=[
                {"text": "", "stop_reason": "tool_use", "tool_uses": [
                    {"id": "b1", "name": "list_matches", "input": {}}]},
                {"text": "Budget gone.", "tool_uses": [], "stop_reason": "end_turn"},
            ])))
            assert any(e["type"] == "tool_done" and "budget" in e["summary"]
                       for e in capd), capd
        usage_path().write_text(json.dumps({"day": "1999-01-01", "count": DAILY_TOOL_CAP}))
        assert tool_calls_today() == 0, "yesterday's spend must not carry over"

        _golden_eval(mid)
        _api_selfcheck(mid)
        _live_smoke(mid)
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmp, ignore_errors=True)
    print("agent.py ok")


# ------------------------------------------------------------- golden eval
# ~15 realistic coach questions against the rule-based fake. The assertions
# are TOOL-SET CONTAINMENT plus terminal answer properties - never exact
# sequences - so a prompt or rule edit that changes the order or adds a
# sensible extra tool does not flake. (Q, expected tools, required substring
# in the final answer.)

_EVAL = (
    ("show me the report", {"get_report"}, None),
    ("what was the score?", {"get_report"}, None),
    ("how far did the team run?", {"get_report"}, "provisional"),
    ("how did the striker do?", {"player_stats"}, None),
    ("who ran the most?", {"player_stats"}, None),
    ("how many sprints did number 10 make?", {"player_stats"}, "provisional"),
    ("who played the most minutes?", {"player_stats"}, None),
    ("compare 7 and 11", {"compare_players"}, None),
    ("what happened at 63'?", {"list_events"}, None),
    ("any goals or cards?", {"list_events"}, None),
    ("clip 12:00 to 12:30", {"create_clip"}, "clip"),
    ("what clips do we have?", {"list_clips"}, None),
    ("show me the frame at 10:00", {"get_frame"}, None),
    ("list my matches", {"list_matches"}, None),
    ("open the film tab and jump to 30:00", {"open_tab", "seek_film"}, None),
)


def _golden_eval(mid: str) -> None:
    for q, want, answer_has in _EVAL:
        chat = chats.create_chat(match_id=mid, title=q[:60])
        events = list(run(chat["id"], q, provider=FakeProvider(), match_id=mid))
        assert events[-1]["type"] == "done", (q, events[-1])
        called = {e["name"] for e in events if e["type"] == "tool_start"}
        assert want <= called, f"{q!r}: wanted {want} within {called}"
        # no tool run may die: errors go back to the model, the run finishes
        assert not any(e["type"] == "error" for e in events), q
        final = "".join(e["text"] for e in events if e["type"] == "token")
        assert final.strip(), f"{q!r}: no terminal answer"
        if answer_has:
            assert answer_has in final.lower(), (
                f"{q!r}: answer must mark provisional numbers, got: {final!r}")
    print(f"golden eval ok ({len(_EVAL)} questions)")


def _api_selfcheck(mid: str) -> None:
    """The /api/agent and /api/settings routes, the same way jobs.py exercises
    its own routes: TestClient, no lifespan (the pump must not spawn)."""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    import app as _app  # noqa: PLC0415 - pulls fastapi/starlette

    assert _app.jobs.JOBS_DIR == jobs.JOBS_DIR, "app.py is reading another data dir"
    assert _app.chats.CHATS_DIR == chats.CHATS_DIR
    c = TestClient(_app.app)

    os.environ["FUTTRAFFIC_AGENT_FAKE"] = "1"
    os.environ.pop("ANTHROPIC_API_KEY", None)

    # --- POST a run with no chat_id: a chat is created, titled from the first
    # message, and the SSE stream carries the contract's event shapes.
    r = c.post("/api/agent/run", json={"match_id": mid,
                                       "message": "show me the report"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-store"
    lines = [ln[6:] for ln in r.text.splitlines() if ln.startswith("data: ")]
    evs = [json.loads(ln) for ln in lines]
    kinds = [e["type"] for e in evs]
    assert "tool_start" in kinds and "tool_done" in kinds and "token" in kinds
    assert kinds[0] == "start" and kinds[-1] == "done", kinds
    cid = evs[-1]["chat_id"]
    assert evs[0]["chat_id"] == cid

    # --- the chat endpoints: list digest, full chat with messages + events.
    lst = c.get("/api/agent/chats")
    assert lst.status_code == 200 and cid in {x["id"] for x in lst.json()}
    full = c.get(f"/api/agent/chats/{cid}").json()
    assert full["messages"][0]["content"] == "show me the report"
    assert full["messages"][1]["events"][0]["name"] == "get_report"

    # --- THE SAVED-CHAT CONTRACT. The frontend reads this exact shape back
    # when a chat is reopened (lib/agent.ts HistoryMessage, store.ts loadChat),
    # and it once read `text` where this writes `content`, so every reopened
    # chat was blank while both sides' tests passed against their own idea of
    # the shape. The fixture is the one copy both suites read: a key added or
    # renamed here fails until the fixture - and so the frontend test - moves.
    golden = json.loads((Path(__file__).resolve().parents[2] / "app" / "frontend" / "src"
                         / "test" / "fixtures" / "chat.golden.json").read_text())
    assert set(full) == set(golden), (sorted(full), sorted(golden))
    g_user, g_agent = golden["messages"]
    assert set(full["messages"][0]) == set(g_user), full["messages"][0]
    assert set(full["messages"][1]) == set(g_agent), full["messages"][1]
    assert set(full["messages"][1]["events"][0]) == set(g_agent["events"][0]), \
        full["messages"][1]["events"][0]

    assert c.get("/api/agent/chats/" + "../..").status_code in (400, 404, 405)
    assert c.get(f"/api/agent/chats/{chats.new_id()}").status_code == 404

    # --- a follow-up run on the SAME chat keeps the history growing.
    r2 = c.post("/api/agent/run", json={"chat_id": cid,
                                        "message": "any goals or cards?"})
    assert r2.status_code == 200, r2.text
    assert len(c.get(f"/api/agent/chats/{cid}").json()["messages"]) == 4

    # --- CONCURRENCY: one active run per chat. A new run waits for one that
    # is winding down (Stop-then-ask), and gets a 409 only if it never does.
    # _app.agent, not this module's global: run as a script this file is
    # __main__, and app.py imported its own `agent`.
    saved_wait, _app.agent.RUN_LOCK_WAIT_S = _app.agent.RUN_LOCK_WAIT_S, 0.3
    try:
        held = chats.try_lock(cid)
        assert held is not None
        r3 = c.post("/api/agent/run", json={"chat_id": cid, "message": "again"})
        assert r3.status_code == 409, r3.status_code
        threading.Timer(0.1, held.release).start()
        r3b = c.post("/api/agent/run", json={"chat_id": cid, "message": "after the stop"})
        assert r3b.status_code == 200, (r3b.status_code, r3b.text)
    finally:
        _app.agent.RUN_LOCK_WAIT_S = saved_wait

    # --- STOP route: idle is a no-op, a malformed id is a 400.
    assert c.post(f"/api/agent/chats/{cid}/stop").json() == {"stopped": False}
    assert c.post("/api/agent/chats/nope/stop").status_code == 400

    # --- MODELS: one list, served; a run naming anything else is refused.
    ms = c.get("/api/agent/models").json()
    assert ms["default"] == DEFAULT_MODEL and {m["id"] for m in ms["models"]} == set(MODELS)
    assert c.post("/api/agent/run", json={"message": "x", "model": "gpt-5"}).status_code == 400
    ok = c.post("/api/agent/run", json={"chat_id": cid, "message": "on opus",
                                        "model": "claude-opus-5"})
    assert ok.status_code == 200, ok.text

    # --- /api/season reads the same cached feed the squad tools do, now that
    # the cache lives in jobs.py rather than inside the route.
    feed = c.get("/api/season")
    assert feed.status_code == 200 and mid in {f["id"] for f in feed.json()["fixtures"]}
    assert jobs.season_feed() is jobs.season_feed(), "the feed was rebuilt with nothing changed"

    # --- SQUAD CHATS over the API: created in a squad, renamed, moved, deleted.
    assert c.post("/api/agent/run", json={"message": "x", "team_id": "nope"}).status_code == 400
    rs = c.post("/api/agent/run", json={"message": "our season record", "team_id": "default"})
    assert rs.status_code == 200, rs.text
    scid = json.loads(rs.text.splitlines()[0][6:])["chat_id"]
    assert next(x for x in c.get("/api/agent/chats").json() if x["id"] == scid)["team_id"] == "default"
    pr = c.patch(f"/api/agent/chats/{scid}", json={"title": "Season so far", "team_id": ""})
    assert pr.status_code == 200 and pr.json()["team_id"] == "" and pr.json()["title"] == "Season so far"
    assert c.patch(f"/api/agent/chats/{scid}", json={"team_id": "nope"}).status_code == 400
    assert c.patch(f"/api/agent/chats/{scid}", json={"title": "  "}).status_code == 400
    assert c.patch(f"/api/agent/chats/{chats.new_id()}", json={"title": "x"}).status_code == 404
    assert c.patch("/api/agent/chats/nope", json={"title": "x"}).status_code == 400
    busy = chats.try_lock(scid)
    assert c.delete(f"/api/agent/chats/{scid}").status_code == 409
    busy.release()
    assert c.delete(f"/api/agent/chats/{scid}").status_code == 200
    assert c.get(f"/api/agent/chats/{scid}").status_code == 404
    assert c.delete(f"/api/agent/chats/{scid}").status_code == 404
    assert c.delete("/api/agent/chats/nope").status_code == 400

    # --- a bad match id on an EXISTING chat is refused too, now that the
    # match travels with every message.
    assert c.post("/api/agent/run", json={"chat_id": cid, "message": "x",
                                          "match_id": "../.."}).status_code == 400

    # --- MISSING KEY is a 400 with the fix in the detail string.
    os.environ.pop("FUTTRAFFIC_AGENT_FAKE", None)
    before = len(chats.list_chats())
    r4 = c.post("/api/agent/run", json={"match_id": mid, "message": "hi"})
    assert r4.status_code == 400 and "PUT /api/settings" in r4.json()["detail"], r4.text
    # A refused run must not leave an empty chat behind: most chats on the dev
    # box were exactly that, a "hi" sent before a key was set.
    assert len(chats.list_chats()) == before, "a refused run created a chat"
    os.environ["FUTTRAFFIC_AGENT_FAKE"] = "1"

    # --- bad inputs are 400s, not 500s.
    assert c.post("/api/agent/run", json={"message": ""}).status_code == 400
    assert c.post("/api/agent/run",
                  json={"chat_id": "..", "message": "x"}).status_code == 400
    assert c.post("/api/agent/run",
                  json={"chat_id": chats.new_id(), "message": "x"}).status_code == 404

    # --- SETTINGS. PUT writes 0600, GET never returns the key, and a key set
    # here is the one resolve_key finds when the env is empty.
    assert c.get("/api/settings").json() == {"has_key": False, "source": "none"}
    r5 = c.put("/api/settings", json={"anthropic_api_key": "sk-test-secret"})
    assert r5.status_code == 200 and r5.json() == {"has_key": True,
                                                   "source": "settings"}
    assert oct((DATA_DIR / "settings.json").stat().st_mode & 0o777) == "0o600"
    body = c.get("/api/settings").text
    assert "sk-test-secret" not in body, "GET /api/settings leaked the key"
    assert resolve_key() == "sk-test-secret"
    assert c.put("/api/settings", json={"other": "x"}).status_code == 400
    assert c.put("/api/settings", json={"anthropic_api_key": 42}).status_code == 400

    # --- the T1 transport spike is gone; the real transport above replaces it.
    assert c.get("/api/dev/sse-spike").status_code == 404
    os.environ.pop("FUTTRAFFIC_AGENT_FAKE", None)


def _live_smoke(mid: str) -> None:
    """One real API call, opt-in only: FUTTRAFFIC_EVAL_LIVE=1 plus a key."""
    if os.environ.get("FUTTRAFFIC_EVAL_LIVE") != "1":
        return
    try:
        resolve_key()
    except MissingKeyError:
        print("live smoke skipped: FUTTRAFFIC_EVAL_LIVE=1 but no key")
        return
    chat = chats.create_chat(match_id=mid, title="live smoke")
    evs = list(run(chat["id"], "show me the report", provider=AnthropicProvider()))
    assert evs[-1]["type"] == "done", evs[-1]
    assert any(e["type"] == "tool_done" and e["name"] == "get_report"
               for e in evs), evs
    print("live smoke ok")


# ------------------------------------------------------------- squad eval
# Does the squad summary help, measured rather than hoped: the same questions,
# on the real model and the real library, with the squad's summary in the
# prompt and without it. Opt-in and paid (real API calls):
#
#     FUTTRAFFIC_EVAL_LIVE=1 .venv/bin/python app/server/agent.py --squad-eval <team_id> [model]
#
# Held fixed per run: the library under FUTTRAFFIC_DATA, the team, the model,
# the questions below. Chats go to a throwaway directory; tool calls DO count
# against today's install budget. What it checks per answer: the expected
# tools were called (containment, never order), a provisional figure is marked
# as provisional where the question asks for one, and the run did not error.
# It does NOT check that the answer is right - that needs a human reading it.

_SQUAD_EVAL = (
    ("what is our record this season?", {"squad_overview"}, None),
    ("who is our top scorer?", {"squad_overview"}, None),
    ("which of our players covers the most ground per 90 this season?",
     {"squad_measured"}, "provisional"),
    ("what share of our tracking is confirmed to named players?", {"squad_measured"}, None),
    ("which matches have we played?", set(), None),
    ("how did we do in our last match?", {"get_report"}, None),
    ("who played the most minutes for us this season?", set(), None),
    ("what is our top speed this season, and can I trust it?", {"squad_measured"}, "provisional"),
    ("how many sprints per 90 does our busiest player make?", {"squad_measured"}, "withheld"),
    ("list our goals and cards in the last match", {"list_events"}, None),
    ("how many goals have we conceded?", {"squad_overview"}, None),
    ("who assists the most in this squad?", {"squad_overview"}, None),
)


def squad_eval(team_id: str, model: str | None = None) -> None:
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    if os.environ.get("FUTTRAFFIC_EVAL_LIVE") != "1":
        raise SystemExit("squad eval makes real API calls: set FUTTRAFFIC_EVAL_LIVE=1")
    resolve_key()
    _squad(team_id)                                    # a typo fails before any spend
    model = model or DEFAULT_MODEL
    real_dir = chats.CHATS_DIR
    tmp = Path(tempfile.mkdtemp(prefix="futsquadeval-"))
    chats.CHATS_DIR = tmp
    arms = {"with summary": team_id, "without": ""}
    score = {arm: {"pass": 0, "tools": 0} for arm in arms}
    try:
        for q, want, answer_has in _SQUAD_EVAL:
            for arm, tid in arms.items():
                c = chats.create_chat(team_id=tid, title=q[:60])
                evs = list(run(c["id"], q, provider=AnthropicProvider(model)))
                called = {e["name"] for e in evs if e["type"] == "tool_start"}
                final = "".join(e["text"] for e in evs if e["type"] == "token")
                ok = (evs[-1]["type"] == "done" and want <= called
                      and (answer_has is None or answer_has in final.lower()))
                score[arm]["pass"] += ok
                score[arm]["tools"] += sum(1 for e in evs if e["type"] == "tool_start")
                print(f"[{'PASS' if ok else 'FAIL'}] {arm:13} {q!r} tools={sorted(called)}")
    finally:
        chats.CHATS_DIR = real_dir
        shutil.rmtree(tmp, ignore_errors=True)
    n = len(_SQUAD_EVAL)
    cmd = f"FUTTRAFFIC_EVAL_LIVE=1 .venv/bin/python app/server/agent.py --squad-eval {team_id} {model}"
    for arm, sc in score.items():
        # prints `value :: command` - prefix the sigil when you quote it
        print(f"{sc['pass']}/{n} pass, {sc['tools']} tool calls :: {cmd}   "
              f"{arm}; team {team_id}; model {model}; n=1 configuration")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "--squad-eval":
        squad_eval(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        _withheld_check()   # first: _selfcheck needs the packaged demo job
        print("agent.py withheld sprints ok")
        _selfcheck()
