"""File-backed chat persistence for the coach-workspace agent.

    python chats.py     # self-check, no test framework

One JSON file per chat under data/chats/, the same idiom as jobs.py: the id is
validated before ANY path is built from it (a client-supplied id is otherwise
a path traversal), and writes are tmp-then-rename so a torn write loses a
message, never the whole history.

A chat is a conversation that can move between matches: each user message
records the match it was asked about, and the chat's own `match_id` is only
the last one, for display. Messages carry role + text plus the run's tool
events, so the UI can re-render tool rows on reload without re-running
anything. The daily tool budget is NOT here - it was per chat, so opening a
new chat reset it; it is install-wide in agent.py now.

Run exclusion: at most one agent run per chat, enforced by an in-process lock.
The deployment is one uvicorn process, so a threading.Lock is the whole
mechanism - no lock files to go stale across a restart.
"""
from __future__ import annotations

import _path  # noqa: F401  - puts model/engine/ on sys.path

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

from paths import DATA as _DEFAULT_DATA

DATA_DIR = Path(os.environ.get("FUTTRAFFIC_DATA", _DEFAULT_DATA))
CHATS_DIR = DATA_DIR / "chats"

# Same shape as a job id, and for the same reason: chat_path() is the only
# place a chat id gets validated, and everything routes through it.
ID_RE = re.compile(r"^[0-9a-f]{32}$")

# One active run per chat. A second POST while a run is streaming would
# interleave two assistant turns into one history, and the persisted events
# would no longer match either stream.
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def new_id() -> str:
    return uuid.uuid4().hex


def chat_path(cid: str) -> Path:
    """Every path in this module routes through here, which is the only place
    a chat id gets validated. Copied from jobs.job_dir's invariant."""
    if not ID_RE.match(cid or ""):
        raise ValueError(f"bad chat id: {cid!r}")
    return CHATS_DIR / f"{cid}.json"


def _save(chat: dict) -> None:
    chat["updated"] = time.time()
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    p = chat_path(chat["id"])
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(chat, f, indent=2)
    os.replace(tmp, p)          # atomic: a torn state file loses a chat


def load(cid: str) -> dict:
    with open(chat_path(cid)) as f:
        return json.load(f)


def create_chat(match_id: str = "", title: str = "", team_id: str = "") -> dict:
    chat = {
        "id": new_id(),
        # The squad this conversation is about, "" for none - the sidebar
        # folder it files under and the squad digest the agent is given.
        # Validated against the club at the API boundary, never here: a chat
        # whose squad was later removed must still load (it files as no squad).
        "team_id": str(team_id or ""),
        # A match id is recorded, never validated against jobs here - a chat
        # about a deleted match must still load. The match id is validated at
        # the API boundary before this is called.
        "match_id": str(match_id or ""),
        # From the first user message; editable later if anyone needs it.
        "title": (title or "New chat")[:80],
        "created": time.time(),
        "updated": time.time(),
        "messages": [],
    }
    _save(chat)
    return chat


def append_message(cid: str, role: str, content: str,
                   events: list[dict] | None = None, **fields) -> dict:
    """Append one message and return it. `events` are the run's tool rows
    (name, input, summary, duration_ms, ok) so the UI can re-render them.
    `fields` are the per-message facts the run knows: `match_id` on a user
    message (which also becomes the chat's last match), `model` and `stopped`
    on an answer. Falsy ones are left off rather than written as nulls."""
    if role not in ("user", "assistant"):
        raise ValueError(f"bad role: {role!r}")
    chat = load(cid)
    msg = {"role": role, "content": str(content), "ts": time.time()}
    if events:
        msg["events"] = events
    msg.update({k: v for k, v in fields.items() if v})
    if role == "user" and fields.get("match_id"):
        chat["match_id"] = str(fields["match_id"])
    chat["messages"].append(msg)
    _save(chat)
    return msg


def get_chat(cid: str) -> dict:
    return load(cid)


def update_chat(cid: str, title: str | None = None, team_id: str | None = None) -> dict:
    """Rename, or move to another squad ("" = no squad). None leaves a field
    as it is. Returns the list digest of the chat."""
    chat = load(cid)
    if title is not None:
        title = str(title).strip()[:80]
        if not title:
            raise ValueError("title cannot be empty")
        chat["title"] = title
    if team_id is not None:
        chat["team_id"] = str(team_id)
    _save(chat)
    return _digest(chat)


class RunConflict(Exception):
    """A run is still going on this chat."""


def delete_chat(cid: str) -> None:
    """Remove the chat file. Refused while a run is live: the run saves its
    answer at the end, and would write it into a file that no longer exists."""
    p = chat_path(cid)
    if run_active(cid):
        raise RunConflict("a run is active on this chat; stop it first")
    p.unlink()                  # FileNotFoundError for a well-formed absent id


def _digest(c: dict) -> dict:
    return {"id": c["id"], "title": c.get("title") or "New chat",
            "match_id": c.get("match_id") or "",
            # Files written before squads existed read as no squad.
            "team_id": c.get("team_id") or "",
            "created": c.get("created"), "updated": c.get("updated"),
            "n_messages": len(c.get("messages") or [])}


# name -> (mtime_ns, size, digest). The sidebar re-lists after every answer,
# and a chat file carries every tool result it ever saw, so re-parsing all of
# them each time grows with use. An unchanged file is not read again.
_LIST_CACHE: dict[str, tuple[int, int, dict]] = {}


def list_chats() -> list[dict]:
    """Newest first. The messages stay on disk; the list is a digest."""
    out = []
    paths = sorted(CHATS_DIR.glob("*.json")) if CHATS_DIR.exists() else []
    seen = set()
    for p in paths:
        try:
            st = p.stat()
        except OSError:
            continue
        seen.add(p.name)
        hit = _LIST_CACHE.get(p.name)
        if hit and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
            out.append(hit[2])
            continue
        try:
            d = _digest(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError, KeyError):
            continue            # half-written; skip rather than 500 the list
        _LIST_CACHE[p.name] = (st.st_mtime_ns, st.st_size, d)
        out.append(d)
    for gone in set(_LIST_CACHE) - seen:
        del _LIST_CACHE[gone]
    return sorted(out, key=lambda c: -(c.get("updated") or 0))


# ---------------------------------------------------------------- run lock

def try_lock(cid: str, wait_s: float = 0.0) -> threading.Lock | None:
    """Acquire this chat's run lock, waiting up to `wait_s`, or return None
    when a run is still active. The caller owns release().

    The wait exists for Stop-then-ask: a stopped run notices the stop between
    stream deltas or after the tool it is running, so the next question can
    arrive a moment before the lock is free."""
    chat_path(cid)              # validate the id before it keys anything
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(cid, threading.Lock())
    got = lock.acquire(timeout=wait_s) if wait_s > 0 else lock.acquire(blocking=False)
    return lock if got else None


def run_active(cid: str) -> bool:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(cid)
    return bool(lock and lock.locked())


# ------------------------------------------------------------------ selfcheck

def _selfcheck() -> None:
    """Three things must hold: a chat survives a reload, a client-supplied id
    can never reach the filesystem, and one chat can never run two agent loops
    at once."""
    global DATA_DIR, CHATS_DIR
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="futchats-"))
    DATA_DIR, CHATS_DIR = tmp, tmp / "chats"

    # --- roundtrip: create, append (with tool events), reload from disk.
    c = create_chat(match_id="0" * 32, title="how did the striker do?")
    cid = c["id"]
    append_message(cid, "user", "how did the striker do?")
    append_message(cid, "assistant", "Provisional numbers only.",
                   events=[{"name": "player_stats", "input": {"match_id": "0" * 32},
                            "summary": "22 players", "duration_ms": 3, "ok": True}])
    got = get_chat(cid)
    assert [m["role"] for m in got["messages"]] == ["user", "assistant"]
    assert got["messages"][1]["events"][0]["name"] == "player_stats"
    assert got["match_id"] == "0" * 32
    mine = list_chats()
    assert len(mine) == 1 and mine[0]["n_messages"] == 2, mine
    assert "messages" not in mine[0]        # the list is a digest, not the file

    # --- id traversal rejection, the jobs.job_dir invariant copied. Nothing
    # here may touch the filesystem for an id that is not 32 lowercase hex.
    for evil in ("..", "../chats", "%2e%2e", "/etc/passwd", "0" * 31,
                 "g" * 32, "", cid.upper(), cid + "x"):
        for fn in (chat_path, load, try_lock):
            try:
                fn(evil)
                raise AssertionError(f"{fn.__name__} accepted {evil!r}")
            except AssertionError:
                raise
            except (ValueError, FileNotFoundError):
                pass
        # FileNotFoundError is only acceptable for well-formed-but-absent ids;
        # every id above is malformed, so it must be ValueError specifically.
        try:
            chat_path(evil)
            raise AssertionError(f"chat_path accepted {evil!r}")
        except AssertionError:
            raise
        except ValueError:
            pass
    gone = new_id()
    try:
        load(gone)
        raise AssertionError("loaded a chat that does not exist")
    except FileNotFoundError:
        pass

    # --- run lock: one active run per chat; the second caller gets nothing.
    lock = try_lock(cid)
    assert lock is not None
    assert run_active(cid)
    assert try_lock(cid) is None, "second run on one chat must conflict"
    lock.release()
    assert not run_active(cid)
    again = try_lock(cid)
    assert again is not None, "a finished run must release the chat"
    again.release()
    # locks are per chat, not global
    other = create_chat()
    assert try_lock(other["id"]) is not None

    # --- a waiting acquire gets the lock the moment a run releases it, and
    # still gives up when the run does not.
    held = try_lock(cid)
    threading.Timer(0.1, held.release).start()
    t0 = time.time()
    waited = try_lock(cid, wait_s=2.0)
    assert waited is not None and time.time() - t0 < 1.5, "wait did not see the release"
    t0 = time.time()
    assert try_lock(cid, wait_s=0.2) is None and time.time() - t0 >= 0.19
    waited.release()

    # --- per-message match: the message keeps its own, the chat keeps the last.
    append_message(cid, "user", "and the second game?", match_id="1" * 32)
    got = get_chat(cid)
    assert got["messages"][-1]["match_id"] == "1" * 32 and got["match_id"] == "1" * 32
    append_message(cid, "assistant", "partial", stopped=True, model="claude-sonnet-5")
    last = get_chat(cid)["messages"][-1]
    assert last["stopped"] is True and last["model"] == "claude-sonnet-5"
    append_message(cid, "assistant", "whole", stopped=False)
    assert "stopped" not in get_chat(cid)["messages"][-1], "falsy fields are left off"

    # --- squads: a chat files under a squad, can move and be renamed, and a
    # file from before squads lists as no squad.
    sq = create_chat(title="pressing", team_id="u15-boys")
    assert next(c for c in list_chats() if c["id"] == sq["id"])["team_id"] == "u15-boys"
    moved = update_chat(sq["id"], title="  pressing v Brighton ", team_id="first-team")
    assert moved["team_id"] == "first-team" and moved["title"] == "pressing v Brighton"
    assert update_chat(sq["id"], team_id="")["team_id"] == ""
    try:
        update_chat(sq["id"], title="   ")
        raise AssertionError("an empty title was accepted")
    except ValueError:
        pass
    legacy = load(sq["id"])
    legacy.pop("team_id")
    _save(legacy)
    assert next(c for c in list_chats() if c["id"] == sq["id"])["team_id"] == ""

    # --- the list does not re-read a file that has not changed, and does see
    # one that has.
    list_chats()
    real_loads, reads = json.loads, []
    json.loads = lambda t, *a, **k: (reads.append(1), real_loads(t, *a, **k))[1]
    try:
        list_chats()
        assert not reads, f"re-parsed {len(reads)} unchanged chat files"
        update_chat(sq["id"], title="renamed")
        reads.clear()           # update_chat's own load is not the list's read
        assert next(c for c in list_chats() if c["id"] == sq["id"])["title"] == "renamed"
        assert len(reads) == 1, reads
    finally:
        json.loads = real_loads

    # --- delete: refused while a run is live, then gone - from the list too.
    busy = try_lock(sq["id"])
    try:
        delete_chat(sq["id"])
        raise AssertionError("deleted a chat mid-run")
    except RunConflict:
        pass
    busy.release()
    delete_chat(sq["id"])
    assert sq["id"] not in {c["id"] for c in list_chats()}
    try:
        delete_chat(sq["id"])
        raise AssertionError("deleted a chat twice")
    except FileNotFoundError:
        pass

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("chats.py ok")


if __name__ == "__main__":
    _selfcheck()
