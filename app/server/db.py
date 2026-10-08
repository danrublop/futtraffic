"""The match store: schema.sql, plus the one function that fills it.

`ingest_match` takes a match.json exactly as jobs.py writes it today and lands
every field in the tables schema.sql describes. That direction matters - the
pipeline keeps writing JSON and this reads it - because the store is a query
layer, not a second source of truth. Delete the database and re-ingest.

Read schema.sql first. It carries the model; this file is the plumbing.

WHAT INGEST DOES THAT A DUMP WOULD NOT, and why each is here rather than in
whoever renders the number:

  * decomposes "618 (88%)" into 618 / 88, keeping the string. The frontend
    already shipped a bug reading "4-2-3-1" as the number 4.
  * point-reflects the home side's event coordinates into absolute pitch
    metres. The feed gives every side its own attacking frame; joining that to
    a pitch without reflecting one of them puts a team's shots in its own net,
    and it hides behind the median because the two errors cancel in aggregate.
  * normalises 6'5" to centimetres, and "58'" to 58.
  * separates measured from entered. Track physicals go to track_metric and
    stay anonymous; the feed's per-player numbers go to player_metric. Nothing
    joins them but a binding.
"""
import _path  # noqa: F401  - puts model/engine/ on sys.path

import json
import os
import re
import sqlite3
from pathlib import Path

import roster

SCHEMA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")

# The demo carries the 2D map as 600 points per track instead of every frame,
# because the raw tracks file does not survive into the job directory. Ingest
# still lands them in `position` - a sparse map is a map - but flags the track
# so a dwell or density query can refuse. See track.position_source.
HEATMAP_POINTS = 600


# Columns added to tables that already exist on someone's disk. `CREATE TABLE
# IF NOT EXISTS` is a no-op against an older store, so a new column in
# schema.sql would be silently absent and every query naming it would fail with
# "no such column" on exactly the machines that have data.
#
# This is not a migration framework and must not become one. It handles the one
# case that is safe and lossless - ADD COLUMN with a default - and nothing else.
# Anything structural takes the route the module docstring already promises:
# delete the file and re-ingest, because the store is a query layer.
_ADDED = (
    ("club", "is_ours", "INTEGER NOT NULL DEFAULT 0"),
    ("match", "team_id", "TEXT"),
    ("match", "season", "TEXT"),
)


def _migrate(conn: sqlite3.Connection) -> None:
    for table, col, decl in _ADDED:
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if cols and col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
    # THE ONE STRUCTURAL EXCEPTION, and why it is not the start of a framework.
    # field_meta's CHECK rejected `not_evaluated`, so every match built since
    # tiers began resolving could not be stored - and SQLite cannot alter a
    # CHECK. The table is pure derived data, rewritten on every ingest, so it
    # is copied into the new shape rather than making every existing store
    # fail until someone deletes it. Every row whose match exists is copied;
    # an orphan (possible only from before foreign keys were switched on) is
    # invisible to every join and would fail the new table's foreign key.
    # Copied as not_evaluated: the old rows carry tiers from before resolution
    # (players.jersey `ok` on every match), and the store must not serve those
    # until a re-ingest resolves them (db.build).
    #
    # One transaction, the check repeated inside it: two processes connecting
    # at once both saw the old table and raced the rename.
    stale = "SELECT sql FROM sqlite_master WHERE type='table' AND name='field_meta'"
    old = conn.execute(stale).fetchone()
    if old and "not_evaluated" not in old["sql"]:
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        try:
            old = conn.execute(stale).fetchone()
            if old and "not_evaluated" not in old["sql"]:
                with open(SCHEMA) as f:
                    ddl = re.search(r"CREATE TABLE IF NOT EXISTS field_meta \(.*?\) STRICT;",
                                    f.read(), re.S).group(0)
                cols = "match_id, field_path, units, n, provenance, caveat_flags"
                conn.execute("ALTER TABLE field_meta RENAME TO field_meta_old")
                conn.execute(ddl)
                conn.execute(f"INSERT INTO field_meta ({cols}, confidence) "
                             f"SELECT {cols}, 'not_evaluated' FROM field_meta_old "
                             "WHERE match_id IN (SELECT match_id FROM match)")
                conn.execute("DROP TABLE field_meta_old")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    conn.commit()


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # Per-connection, not per-database: SQLite defaults this OFF and a schema
    # full of foreign keys that are never enforced is decoration.
    conn.execute("PRAGMA foreign_keys = ON")
    with open(SCHEMA) as f:
        conn.executescript(f.read())
    _migrate(conn)
    return conn


# Sprints are WITHHELD (stats.CAVEATS["sprints_withheld"], author's decision 2026-09-29 for
# season surfaces). Rows ingested before 2026-09-28 still hold counts and stay untouched;
# what READS the store sees them removed. The pipeline's count is the only 'sprints'
# metric - a feed's "Number of sprints" is entered data under its own id.
_WITHHELD_METRICS = ("track_metric", "team_metric")


def withhold_sprints(conn: sqlite3.Connection) -> None:
    """Shadow the metric tables, and every view over them, with TEMP views that
    drop metric 'sprints'. Unqualified names resolve temp-first in SQLite, so every
    query on this connection reads the filtered copy; the rows on disk do not change.
    Call before `PRAGMA query_only`.

    ponytail: `main.track_metric` spelled out bypasses it; add an authorizer on
    SQLITE_READ of main.*_metric if an explicit schema ever shows up in a caller.
    """
    for t in _WITHHELD_METRICS:
        conn.execute(f"CREATE TEMP VIEW {t} AS SELECT * FROM main.{t} WHERE metric <> 'sprints'")
    # Every main view, recreated in creation order so dependents see the shadows.
    for (sql,) in conn.execute("SELECT sql FROM main.sqlite_master WHERE type = 'view'"
                               " ORDER BY rowid").fetchall():
        conn.execute(re.sub(r"^CREATE VIEW", "CREATE TEMP VIEW", sql, flags=re.I))


def _withheld_demo() -> None:
    """An old stored sprint count never reaches a reader of the store."""
    import tempfile  # noqa: PLC0415 - self-check only
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.db")
        conn = connect(path)
        conn.execute("INSERT INTO match (match_id, fps, pitch_length_m, pitch_width_m)"
                     " VALUES ('old', 25, 105, 68)")
        conn.execute("INSERT INTO club (club_id, name) VALUES ('c', 'C')")
        conn.execute("INSERT INTO player (player_id, club_id, name) VALUES ('p', 'c', 'P')")
        conn.execute("INSERT INTO track (match_id, track_id, kind, player_id, bind_source)"
                     " VALUES ('old', 1, 'player', 'p', 'manual')")
        for metric, v in (("sprints", 7.0), ("distance_km", 5.0)):
            conn.execute("INSERT INTO track_metric VALUES ('old', 1, ?, ?)", (metric, v))
        conn.execute("INSERT INTO team_metric (match_id, side, metric, value)"
                     " VALUES ('old', 'home', 'sprints', 7)")
        conn.commit()
        conn.close()

        ro = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        withhold_sprints(ro)
        ro.execute("PRAGMA query_only = ON")
        for sql in ("SELECT count(*) FROM track_metric WHERE metric = 'sprints'",
                    "SELECT count(*) FROM team_metric WHERE metric = 'sprints'",
                    "SELECT count(*) FROM player_track_metric WHERE metric = 'sprints'",
                    "SELECT count(*) FROM season_track_metric WHERE metric = 'sprints'"):
            assert ro.execute(sql).fetchone()[0] == 0, sql
        assert ro.execute("SELECT value FROM season_track_metric").fetchall() == [(5.0,)]
        # ...and the stored row is still there, untouched.
        assert ro.execute("SELECT value FROM main.track_metric WHERE metric = 'sprints'"
                          ).fetchall() == [(7.0,)]
        ro.close()


# ---------------------------------------------------------------------------
# Parsing the feed's shapes. Every one of these exists because the value
# arrives as display text and something has to turn it into a number ONCE.

_QTY = re.compile(r"^\s*(-?[\d.]+)(?:\s*/\s*(-?[\d.]+))?\s*(?:\((-?[\d.]+)%\))?\s*%?\s*$")


def quantity(raw) -> tuple[float | None, float | None, float | None, str | None]:
    """Feed value -> (value, attempted, pct, raw text).

        700           -> (700, None, None, None)
        "72%"         -> (72,  None, None, "72%")
        "618 (88%)"   -> (618, None, 88,   "618 (88%)")
        "6/12 (50%)"  -> (6,   12,   50,   "6/12 (50%)")
        "4-2-3-1"     -> (None, None, None, "4-2-3-1")     <- NOT the number 4

    The last line is the whole point. A regex anchored at both ends refuses
    anything it does not fully understand, instead of reading a leading token
    off a formation and drawing a bar chart of it.
    """
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw), None, None, None
    text = str(raw or "").strip()
    m = _QTY.match(text)
    if not m:
        return None, None, None, text or None
    value = float(m.group(1))
    attempted = float(m.group(2)) if m.group(2) else None
    pct = float(m.group(3)) if m.group(3) else (value if text.endswith("%") else None)
    return value, attempted, pct, text


def height_cm(v) -> int | None:
    """`6'5"` to centimetres. Height is a real thing to sort a squad by, but
    not in the unit it arrives in - and feet sort as strings."""
    m = re.match(r"^(\d+)'\s*(\d+)?", str(v or ""))
    return round((int(m.group(1)) * 12 + int(m.group(2) or 0)) * 2.54) if m else None


def minute(v) -> int | None:
    """`58'` and `90+3'` to 58 and 90. Stoppage time is dropped deliberately:
    the pitch clock it would need does not exist in the feed, and 93 would sort
    into the second half of a 90-minute axis as if it were extra data."""
    m = re.match(r"^\s*(\d+)", str(v or ""))
    return int(m.group(1)) if m else None


def metric_id(label: str) -> str:
    """A metric's display label as a stable id. Same rule as the frontend's
    metricId, so a chart saved against 'accurate-passes' still resolves."""
    return re.sub(r"^-|-$", "", re.sub(r"[^a-z0-9]+", "-", str(label or "").lower()))


def _pitch_xy(x: float, y: float, length: float, width: float,
              flip: bool) -> tuple[float, float]:
    """Feed fraction -> THE MAP'S frame, which is the calibration's frame.

    There is only one pitch coordinate system in this database and `position`
    defines it. An event stored in any other frame is an event that cannot be
    joined to the players who made it - and the failure is silent, because a
    reflected point is still a point on a pitch.

    `flip` is a point reflection about the centre spot, which is the only
    transform any of these feeds needs. Working out WHICH rows to flip is the
    caller's job, because the answer is a property of the feed, not of the
    geometry, and the feeds disagree:

      * match.json's shots are ONE SHARED frame - home at low x, away at high -
        so the whole set flips or none of it does, depending on which way
        stats.infer_attacking_right says home was going.
      * the real event CSV is ATTACK-NORMALISED: every row sits in the acting
        team's own attacking frame, so only one side's rows flip. Getting that
        backwards moves the median event-to-actor distance 1.15 m -> 2.04 m and
        HIDES behind the median while it does it (it is visible only as a
        bimodal histogram, or at p75). model/engine/events.to_pitch already knows
        this rule; route any new feed through it rather than growing a third
        copy of it here.
    """
    px, py = ((1 - x) * length, (1 - y) * width) if flip else (x * length, y * width)
    # Clamped to the pitch. A feed's fraction is not guaranteed to be 0-1 —
    # a blocked shot or one struck from behind the goal line comes through at
    # 1.03, and 109 m on a 105 m pitch is a coordinate no join can use.
    return (min(max(px, 0.0), length), min(max(py, 0.0), width))


# ---------------------------------------------------------------------------


def ingest_match(conn: sqlite3.Connection, match_id: str, report: dict) -> dict:
    """One match.json into the store. Idempotent - re-ingesting replaces."""
    meta = report.get("meta") or {}
    facts = meta.get("facts") or {}
    pitch = meta.get("pitch") or {}
    length = float(pitch.get("length") or 105.0)
    width = float(pitch.get("width") or 68.0)
    fps = float(meta.get("fps") or 25.0)

    cur = conn.cursor()
    cur.execute("DELETE FROM match WHERE match_id = ?", (match_id,))
    cur.execute("DELETE FROM event WHERE match_id = ?", (match_id,))

    comp = meta.get("competition")
    comp_id = roster._slug(comp) if comp else None
    if comp_id:
        cur.execute("INSERT OR IGNORE INTO competition VALUES (?, ?)", (comp_id, comp))

    clubs = {}
    for side in ("home", "away"):
        name = meta.get(f"{side}_name") or side.title()
        cid = roster._slug(name) or side
        clubs[side] = cid
        # Named columns and an upsert that touches only the name, because
        # `is_ours` is set by ingest_org and a REPLACE would silently reset our
        # own academy to "an opponent" every time one of its matches re-ingests.
        cur.execute("INSERT INTO club (club_id, name) VALUES (?, ?)"
                    " ON CONFLICT(club_id) DO UPDATE SET name = excluded.name",
                    (cid, name))

    cur.execute(
        "INSERT INTO match (match_id, competition_id, status, fps, pitch_length_m,"
        " pitch_width_m, calibration_rms_m, officials_excluded)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (match_id, comp_id, meta.get("status"), fps, length, width,
         meta.get("calibration_rms_m"), int(meta.get("officials_excluded") or 0)),
    )

    lineups = facts.get("lineups") or {}
    shape = report.get("shape_series") or {}
    stress = report.get("structure_stress") or {}
    for side in ("home", "away"):
        lu = lineups.get(side) or {}
        sh = shape.get(side) or {}
        cov = sh.get("coverage") or {}
        st = stress.get(side) or {}
        cur.execute(
            "INSERT INTO match_side VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (match_id, side, clubs[side], lu.get("formation"), lu.get("team_rating"),
             None if sh.get("attacking_right") is None else int(sh["attacking_right"]),
             cov.get("possible"), cov.get("emitted"), cov.get("min_players"),
             st.get("gini"), st.get("profile")),
        )

    players = _ingest_squad(cur, match_id, clubs, facts)
    _ingest_player_metrics(cur, match_id, players, facts)
    _ingest_team_metrics(cur, match_id, report, facts)
    _ingest_tracks(cur, match_id, report, fps)
    # Which way the shot points have to be turned to land in the map's frame.
    # The demo draws home attacking -x while its own shape series says home
    # attacks +x, so the two do not agree and something has to reconcile them
    # before a shot and a player can be on the same pitch.
    home_right = ((report.get("shape_series") or {}).get("home") or {}).get("attacking_right")
    _ingest_events(cur, match_id, players, facts, length, width, home_right is not False)
    _ingest_series(cur, match_id, report, facts)
    _ingest_plays(cur, match_id, facts)
    # Re-resolved, not copied: the store answers with the same tier the API
    # serves (app.match_json), not the one frozen into the file.
    from stats import resolve  # noqa: PLC0415 - pulls numpy
    _ingest_field_meta(cur, match_id, resolve(report.get("fields") or {}, meta))

    conn.commit()
    return {
        "players": len(players),
        "tracks": cur.execute(
            "SELECT count(*) c FROM track WHERE match_id=?", (match_id,)).fetchone()["c"],
        "positions": cur.execute(
            "SELECT count(*) c FROM position WHERE match_id=?", (match_id,)).fetchone()["c"],
        "events": cur.execute(
            "SELECT count(*) c FROM event WHERE match_id=?", (match_id,)).fetchone()["c"],
    }


def _prune(cur, table: str, key: str, club_id: str, keep: list[str]) -> None:
    """Delete this club's rows in `table` except the ones named in `keep`.

    WAS A BUG, and a subtle one: the two call sites wrote
    `key NOT IN ({placeholders or 'NULL'})`, which is correct for a non-empty
    list and silently a no-op for an empty one — `x NOT IN (NULL)` is NULL, not
    true, so SQLite deletes nothing. Removing the LAST coach from a club left
    every coach in the store forever, and the club screen and the store then
    disagreed about who works there with nothing raising.

    An empty keep-list is the whole point of a prune, so it takes the branch
    without the clause rather than a placeholder that reads like one.
    """
    if keep:
        cur.execute(
            f"DELETE FROM {table} WHERE club_id = ?"
            f" AND {key} NOT IN ({','.join('?' * len(keep))})",
            (club_id, *keep))
    else:
        cur.execute(f"DELETE FROM {table} WHERE club_id = ?", (club_id,))


def ingest_org(conn: sqlite3.Connection, view: dict, roster_rows: list) -> dict:
    """The club ladder into the store: teams, staff, and squad membership.

    `view` is jobs.org()'s shape and `roster_rows` is jobs.roster()'s - plain
    dicts rather than org.Club objects, so this is testable without a data
    directory and so db.py keeps not importing jobs at module scope.

    Idempotent and authoritative: the ladder is REPLACED wholesale, because
    org.json is the source of truth and a team deleted there must not survive
    here as a ghost row that season queries still group by.

    WHAT IT DELIBERATELY DOES NOT TOUCH is `player.club_id`. That column means
    "the club whose team sheet this name appeared on", and ingest_match owns
    it. Our own academy players get their club membership from `squad` instead
    - which is the honest answer anyway, because a player belongs to a TEAM in
    a SEASON, not to a club in the abstract. Two writers on one column is how
    a re-ingest of last week's match would quietly reassign this year's squad.
    """
    club = view.get("club") or {}
    club_id = club.get("club_id") or "our-club"
    cur = conn.cursor()

    cur.execute("INSERT INTO club (club_id, name, is_ours) VALUES (?,?,1)"
                " ON CONFLICT(club_id) DO UPDATE SET name = excluded.name, is_ours = 1",
                (club_id, club.get("name") or "Our club"))

    # UPSERT THEN PRUNE, never delete-then-reinsert. `match.team_id` is
    # ON DELETE SET NULL, so deleting a team the club still has - even for a
    # microsecond, even to write the identical row back - unattributes every
    # match that pointed at it. This runs on every single-match ingest, so the
    # naive version blanked the whole library's team column each upload.
    # Only a team genuinely gone from org.json is deleted, and losing its
    # matches' attribution is then correct.
    import org as O  # noqa: PLC0415 - only for the ladder key

    coaches = view.get("coaches") or []
    for co in coaches:
        cur.execute(
            "INSERT INTO coach VALUES (?,?,?,?,?) ON CONFLICT(coach_id) DO UPDATE SET"
            " club_id = excluded.club_id, name = excluded.name,"
            " role = excluded.role, email = excluded.email",
            (co["coach_id"], club_id, co.get("name") or "",
             co.get("role") or "", co.get("email") or ""))
    _prune(cur, "coach", "coach_id", club_id, [c["coach_id"] for c in coaches])

    seasons = {}
    for t in view.get("teams") or []:
        band = t.get("age_band") or ""
        season = t.get("season") or club.get("season") or ""
        seasons[t["team_id"]] = season
        cur.execute(
            "INSERT INTO team VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(team_id) DO UPDATE SET"
            " club_id = excluded.club_id, name = excluded.name,"
            " age_band = excluded.age_band, age_key = excluded.age_key,"
            " gender = excluded.gender, level = excluded.level, season = excluded.season",
            (t["team_id"], club_id, t.get("name") or t["team_id"], band,
             O.age_key(band or t.get("name") or ""), t.get("gender") or "",
             t.get("level") or "", season))
        # Staff assignment is replaced per team, which is the only way a coach
        # REMOVED from a squad actually leaves it.
        cur.execute("DELETE FROM team_coach WHERE team_id = ?", (t["team_id"],))
        for cid in t.get("coach_ids") or []:
            cur.execute("INSERT OR IGNORE INTO team_coach VALUES (?,?)", (t["team_id"], cid))
    _prune(cur, "team", "team_id", club_id, list(seasons))

    # Squad rows ARE replaced wholesale per team - a player dropped from a
    # roster file has to leave the squad, and nothing hangs off a squad row, so
    # unlike `team` above there is no cascade to be careful of.
    if seasons:
        cur.execute(
            f"DELETE FROM squad WHERE team_id IN ({','.join('?' * len(seasons))})",
            tuple(seasons))

    for r in roster_rows or []:
        tid = r.get("team_id")
        if tid not in seasons:
            continue  # a roster file for a team no longer on the ladder
        cur.execute(
            "INSERT INTO player (player_id, club_id, name, number, position)"
            " VALUES (?,?,?,?,?) ON CONFLICT(player_id) DO UPDATE SET"
            " name = coalesce(excluded.name, player.name)",
            (r["id"], club_id, r.get("name"), r.get("number"), r.get("position")))
        cur.execute("INSERT OR REPLACE INTO squad VALUES (?,?,?,?,?)",
                    (tid, r["id"], seasons[tid], r.get("number"), r.get("position")))

    conn.commit()
    q = lambda s: cur.execute(s).fetchone()["c"]  # noqa: E731
    return {"club": club_id, "teams": q("SELECT count(*) c FROM team"),
            "coaches": q("SELECT count(*) c FROM coach"),
            "squad": q("SELECT count(*) c FROM squad")}


def set_match_team(conn: sqlite3.Connection, match_id: str, team_id: str | None,
                   season: str | None = None) -> None:
    """Whose match this is in our club. Kept out of ingest_match because it
    comes off the JOB, not off match.json - the report knows the two sides,
    not which of our squads filmed it.

    An unknown team lands as NULL rather than a foreign-key error: a job can
    carry a team string that was later removed from the ladder, and the match
    is still a match. Unattributed is the honest state and the UI can say so."""
    if team_id and not conn.execute("SELECT 1 FROM team WHERE team_id = ?",
                                    (team_id,)).fetchone():
        team_id = None
    conn.execute("UPDATE match SET team_id = ?, season = ? WHERE match_id = ?",
                 (team_id or None, season or None, match_id))
    conn.commit()


def _ingest_squad(cur, match_id, clubs, facts) -> dict[str, str]:
    """Team sheet + subs -> player and appearance. Returns name -> player_id.

    The name is the only key the feed has, so it is the only key an appearance
    can be minted from. That is why these players are ENTERED identities and
    have no relationship to a track until someone binds one.
    """
    names: dict[str, str] = {}
    lineups = facts.get("lineups") or {}
    subs = facts.get("subs") or {}
    for side in ("home", "away"):
        entries = [(p, True) for p in (lineups.get(side) or {}).get("xi") or []]
        entries += [(p, False) for p in subs.get(side) or []]
        for p, started in entries:
            name = p.get("name")
            if not name:
                continue
            pid = roster._slug(name) or f"{clubs[side]}-{p.get('n')}"
            names[name] = pid
            # UPSERT, never INSERT OR REPLACE. A REPLACE is a DELETE plus an
            # INSERT, so it fires every ON DELETE CASCADE hanging off this row:
            # re-ingesting one match used to silently empty `squad` - a
            # player's whole club membership - because their name appeared on a
            # team sheet. `track.player_id` is the same exposure.
            cur.execute("INSERT INTO player (player_id, club_id, name, number, position)"
                        " VALUES (?,?,?,?,?) ON CONFLICT(player_id) DO UPDATE SET"
                        " club_id = excluded.club_id, name = excluded.name,"
                        " number = excluded.number, position = excluded.position",
                        (pid, clubs[side], name, p.get("n"), p.get("pos")))
            cur.execute(
                "INSERT OR REPLACE INTO appearance VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (match_id, pid, side, p.get("n"), p.get("pos"), int(started),
                 None if started else minute(p.get("on")), minute(p.get("off")),
                 None, p.get("rating"), p.get("age"), height_cm(p.get("height")),
                 p.get("x"), p.get("y")),
            )
    return names


def _ingest_player_metrics(cur, match_id, names, facts) -> None:
    resolve = _name_resolver(names)
    for row in facts.get("player_stats") or []:
        pid = resolve(row.get("name"))
        if not pid:
            continue  # a stat line for nobody on the team sheet is not a player
        for label, raw in (row.get("stats") or {}).items():
            value, attempted, pct, text = quantity(raw)
            cur.execute("INSERT OR REPLACE INTO player_metric VALUES (?,?,?,?,?,?,?)",
                        (match_id, pid, metric_id(label), value, attempted, pct, text))
        if (mins := (row.get("stats") or {}).get("minutes")) is not None:
            cur.execute("UPDATE appearance SET minutes=? WHERE match_id=? AND player_id=?",
                        (float(mins), match_id, pid))


def _ingest_team_metrics(cur, match_id, report, facts) -> None:
    def put(side, metric, raw, section=None, ord_=0, featured=0):
        value, attempted, pct, text = quantity(raw)
        cur.execute(
            "INSERT INTO team_metric VALUES (?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(match_id, side, metric) DO UPDATE SET"
            " featured = max(featured, excluded.featured)",
            (match_id, side, metric, value, attempted, pct, text, section, ord_, featured))

    for i, sec in enumerate(facts.get("sections") or []):
        for j, row in enumerate(sec.get("rows") or []):
            for side in ("home", "away"):
                put(side, metric_id(row.get("label")), row.get(side), sec.get("title"), i * 100 + j)

    # top_stats is the same metrics the sections carry, promoted. A flag on the
    # existing row, so the two can never disagree about a value.
    for t in facts.get("top_stats") or []:
        for side in ("home", "away"):
            put(side, metric_id(t.get("label")), t.get(side), "Top stats", 0, 1)

    for side in ("home", "away"):
        team = (report.get("teams") or {}).get(side) or {}
        for k in ("distance_m", "top_speed_kmh", "sprints"):
            if team.get(k) is not None:
                put(side, k, team[k], "Physical")
        for k, v in (team.get("shape") or {}).items():
            put(side, k, v, "Shape")
        zones = (facts.get("attacking_zones") or {}).get(side) or []
        for name, v in zip(("left", "centre", "right"), zones):
            put(side, f"attacking-zone-{name}", v, "Attacking zones")

    # Match-level, so side 'both': these belong to the fixture, not a team.
    for k, v in (report.get("territory") or {}).items():
        put("both", f"territory-{metric_id(k)}", v, "Territory")
    for k, m in ((report.get("relative_shape") or {}).get("metrics") or {}).items():
        put("home", f"rel-{k}", m.get("home_mean"), "Relative shape")
        put("away", f"rel-{k}", m.get("away_mean"), "Relative shape")
        put("both", f"rel-{k}-sd", m.get("sd"), "Relative shape")


def _ingest_tracks(cur, match_id, report, fps) -> None:
    heat = report.get("heatmaps") or {}
    for p in report.get("players") or []:
        tid = int(p["id"])
        # 'unknown' is what teamembed emits when the classifier ABSTAINS, and it
        # is a real answer, not a missing one. NULL rather than a third side
        # value: every side-keyed query should drop these tracks, and a NULL
        # does that automatically where a string 'unknown' would quietly form a
        # third team in every GROUP BY.
        side = p.get("team") if p.get("team") in ("home", "away") else None
        cur.execute(
            "INSERT OR REPLACE INTO track (match_id, track_id, kind, side, jersey,"
            " position_source) VALUES (?,?,?,?,?,?)",
            (match_id, tid, "player", side, p.get("jersey"),
             "heatmap_sample" if str(tid) in heat else "frames"),
        )
        for k in ("minutes", "distance_km", "top_speed_kmh", "sprints",
                  "clamped_frac", "observed_frac"):
            if p.get(k) is not None:
                cur.execute("INSERT OR REPLACE INTO track_metric VALUES (?,?,?,?)",
                            (match_id, tid, k, float(p[k])))
        if (avg := p.get("avg_position")):
            cur.execute("INSERT OR REPLACE INTO track_metric VALUES (?,?,?,?)",
                        (match_id, tid, "avg_x_m", float(avg[0])))
            cur.execute("INSERT OR REPLACE INTO track_metric VALUES (?,?,?,?)",
                        (match_id, tid, "avg_y_m", float(avg[1])))

        pts = heat.get(str(tid)) or []
        if not pts:
            continue
        # Spread the downsample evenly across the time the track was on the
        # pitch. The frames are RECONSTRUCTED, not recorded - which is exactly
        # what position_source='heatmap_sample' is there to declare.
        span = float(p.get("minutes") or 0) * 60.0 * fps
        step = span / len(pts) if span else 1.0
        cur.executemany(
            "INSERT OR REPLACE INTO position (match_id, frame, track_id, x_m, y_m)"
            " VALUES (?,?,?,?,?)",
            [(match_id, int(i * step), tid, float(xy[0]), float(xy[1]))
             for i, xy in enumerate(pts)],
        )


def _name_resolver(names: dict[str, str]):
    """Feed name -> player_id, across the feed's TWO naming conventions.

    The team sheet says "Brandt" and the timeline says "Leo Brandt" for the
    same human, so an exact lookup drops every assist on the floor - which is
    how this was found. Resolution is: exact slug, else a unique surname match,
    else nothing. Never a fuzzy best guess: two brothers on one team sheet is
    not rare, and an event credited to the wrong sibling is worse than an event
    credited to nobody, because nobody is visible and wrong is not.
    """
    exact = {roster._slug(n): pid for n, pid in names.items()}
    by_last: dict[str, set] = {}
    for n, pid in names.items():
        last = roster._slug((n or "").split()[-1] if n else "")
        if last:
            by_last.setdefault(last, set()).add(pid)

    def resolve(name):
        slug = roster._slug(name or "")
        if not slug:
            return None
        if slug in exact:
            return exact[slug]
        last = roster._slug(name.split()[-1])
        hit = by_last.get(last) or set()
        return next(iter(hit)) if len(hit) == 1 else None

    return resolve


def _ingest_events(cur, match_id, names, facts, length, width, shot_flip) -> None:
    resolve = _name_resolver(names)

    def add(**kw):
        cols = ("match_id", "period", "minute", "t_s", "side", "type", "player_id",
                "x_m", "y_m", "outcome", "body", "situation", "xg", "xgot", "detail")
        cur.execute(
            f"INSERT INTO event ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            tuple(kw.get(c) for c in cols))
        return cur.lastrowid

    for e in facts.get("events") or []:
        # The feed spells a substitution with in/out and everything else with
        # player/assist, and it calls it 'substitution' here while filing the
        # same event again under facts.subs. One type name, one row.
        kind = "sub" if e.get("type") == "substitution" else e.get("type")
        actor = e.get("in") if kind == "sub" else e.get("player")
        pid = resolve(actor)
        roles = ((("on", e.get("in")), ("off", e.get("out"))) if kind == "sub"
                 else (("actor", e.get("player")), ("assist", e.get("assist"))))
        # A name that does not resolve is kept verbatim rather than dropped. The
        # demo needs this on its own data: the timeline substitutes a player off
        # who is on neither the XI nor the bench. Somebody has to see that, and
        # a silently player-less substitution is nobody seeing it.
        detail = {k: v for k, v in (("score", e.get("score")),) if v}
        detail.update({f"{role}_name": name for role, name in roles
                       if name and not resolve(name)})
        eid = add(match_id=match_id, minute=minute(e.get("minute")), side=e.get("side"),
                  # 'HT' has no minute and never will; period places it instead.
                  period=1 if e.get("minute") == "HT" else None,
                  type=kind, player_id=pid,
                  detail=json.dumps(detail) if detail else None)
        for role, name in roles:
            if (rid := resolve(name)):
                cur.execute("INSERT OR IGNORE INTO event_player VALUES (?,?,?)",
                            (eid, rid, role))
                # The team sheet knows who came ON and when; only the timeline
                # knows who went OFF, so this is the one place off_minute can
                # come from.
                if role in ("on", "off"):
                    col = "on_minute" if role == "on" else "off_minute"
                    cur.execute(
                        f"UPDATE appearance SET {col}=? WHERE match_id=? AND player_id=?",
                        (minute(e.get("minute")), match_id, rid))

    shots = facts.get("shots") or {}
    detail = shots.get("detail") or {}
    for pt in shots.get("points") or []:
        side = pt.get("side")
        x_m, y_m = _pitch_xy(float(pt.get("x") or 0), float(pt.get("y") or 0),
                             length, width, shot_flip)
        add(match_id=match_id, period=pt.get("period"), side=side, type="shot",
            x_m=x_m, y_m=y_m, outcome="Goal" if pt.get("goal") else "No goal")
    # The demo carries ONE worked shot detail as an example rather than one per
    # point, so it cannot be joined to a point and is stored as its own event.
    # Real feed data attaches these per shot and this branch goes away.
    if detail:
        add(match_id=match_id, minute=minute(detail.get("minute")), type="shot",
            player_id=resolve(detail.get("player")), outcome=detail.get("result"),
            body=detail.get("shot_type"), situation=detail.get("situation"),
            xg=detail.get("xg"), xgot=detail.get("xgot"))


def _ingest_series(cur, match_id, report, facts) -> None:
    for side, s in (report.get("shape_series") or {}).items():
        for r in s.get("samples") or []:
            lines = (r.get("lines_m") or []) + [None, None, None]
            gaps = (r.get("line_gaps_m") or []) + [None, None]
            centroid = r.get("centroid") or [None, None]
            cur.execute(
                "INSERT OR REPLACE INTO shape_sample VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (match_id, side, r.get("t"), r.get("width_m"), r.get("depth_m"),
                 r.get("hull_area_m2"), r.get("rear_line_m"), r.get("stretch_index_m"),
                 centroid[0], centroid[1], lines[0], lines[1], lines[2], gaps[0], gaps[1],
                 r.get("n"), r.get("n_dedup")))
        for w in s.get("window_fit_unstable") or []:
            cur.execute("INSERT OR REPLACE INTO formation_window VALUES (?,?,?,?,?,?)",
                        (match_id, side, w.get("t0"), w.get("t1"),
                         w.get("formation") or "", w.get("cost")))

    for side, s in (report.get("structure_stress") or {}).items():
        for i in s.get("intervals") or []:
            cur.execute("INSERT OR REPLACE INTO stress_interval VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (match_id, side, i.get("t0"), i.get("t1"), i.get("duration_s"),
                         i.get("severity"), i.get("peak_z"), i.get("formation"),
                         i.get("cost"), i.get("hull_area_m2")))

    for i, v in enumerate((facts.get("momentum") or {}).get("values") or []):
        cur.execute("INSERT OR REPLACE INTO momentum_sample VALUES (?,?,?)",
                    (match_id, i + 1, float(v)))


def _ingest_plays(cur, match_id, facts) -> None:
    for p in facts.get("plays") or []:
        board = p.get("board") or {}
        cur.execute(
            "INSERT OR REPLACE INTO play (play_id, match_id, type, side, period, minute,"
            " clock, t0_s, t1_s, zone, outcome, note, source, board)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (p.get("id"), match_id, p.get("type"), p.get("side"), p.get("period"),
             p.get("minute"), p.get("clock"), float(p.get("t") or 0), None,
             p.get("zone"), p.get("outcome"), p.get("note"),
             p.get("source") if p.get("source") in ("model", "manual", "demo") else "demo",
             json.dumps(board)))
        # From the board's OWN discs, never the play's players list: filtering
        # for "19 involved" must not return a board with no 19 drawn on it.
        for mag in board.get("magnets") or []:
            label = str(mag.get("label") or "")
            if label.isdigit():
                cur.execute("INSERT OR IGNORE INTO play_player VALUES (?,?,?,?)",
                            (match_id, p.get("id"), int(label), None))


def _ingest_field_meta(cur, match_id, fields, prefix="") -> None:
    """The fields tree is nested by the shape of the report; flatten it to
    dotted paths so a provenance lookup is one indexed read."""
    for key, node in (fields or {}).items():
        path = f"{prefix}{key}"
        if not isinstance(node, dict):
            continue
        if "provenance" in node or "confidence" in node:
            cur.execute("INSERT OR REPLACE INTO field_meta (match_id, field_path, "
                        "units, n, confidence, provenance, caveat_flags, "
                        "justified_by) VALUES (?,?,?,?,?,?,?,?)",
                        (match_id, path, node.get("units"), node.get("n"),
                         node.get("confidence"), node.get("provenance"),
                         json.dumps(node.get("caveat_flags") or []),
                         node.get("justified_by")))
        else:
            _ingest_field_meta(cur, match_id, node, f"{path}.")


def bind_track(conn, match_id: str, track_id: int, player_id: str,
               confidence: float = 1.0, source: str = "manual") -> None:
    """Attach a tracklet to a human. The ONLY way a measured number becomes a
    player's. Kept as its own function so the write is greppable."""
    conn.execute(
        "UPDATE track SET player_id=?, bind_confidence=?, bind_source=?"
        " WHERE match_id=? AND track_id=?",
        (player_id, confidence, source, match_id, track_id))
    conn.commit()


def _demo() -> None:
    """Ingest the real demo report and assert the mapping holds."""
    assert quantity("618 (88%)") == (618.0, None, 88.0, "618 (88%)")
    assert quantity("6/12 (50%)") == (6.0, 12.0, 50.0, "6/12 (50%)")
    assert quantity("72%") == (72.0, None, 72.0, "72%")
    assert quantity(700) == (700.0, None, None, None)
    # The bug this parser exists for: a formation is not the number 4.
    assert quantity("4-2-3-1") == (None, None, None, "4-2-3-1")
    assert height_cm("6'5\"") == 196 and height_cm("5'7\"") == 170
    assert minute("90+3'") == 90 and minute(None) is None

    # The flip is a point reflection about the centre spot: BOTH axes, or a
    # left-wing move comes out as a right-wing one.
    px, py = _pitch_xy(0.12, 0.325, 100.0, 60.0, False)
    fx, fy = _pitch_xy(0.12, 0.325, 100.0, 60.0, True)
    assert (px, fx) == (12.0, 88.0), (px, fx)
    assert round(py + fy, 6) == 60.0, "reflected about the axis, not the centre"

    # jobs.JOBS_DIR rather than a local __file__ walk: it already honours
    # FUTTRAFFIC_DATA, and paths.py exists because eleven modules each
    # computing the repo root is how fixtures go "missing" and tests skip green.
    from jobs import JOBS_DIR  # noqa: PLC0415 - heavy import, self-check only
    # The PACKAGED demo job as well as the data directory. The hand-made demo
    # below is a fixture this check pins numbers to, and it used to be found
    # only because a copy happened to be sitting in the developer's data dir -
    # so deleting that match from the library, which is an ordinary thing to
    # do, broke this module's self-check with a bare StopIteration. The shipped
    # copy under app/desktop/resources is the one that is guaranteed to exist.
    packaged = Path(__file__).resolve().parents[2] / "app" / "desktop" / "resources" / "demo-job"
    files = sorted(JOBS_DIR.glob("*/match.json")) + sorted(packaged.glob("*/match.json"))
    reports = [json.loads(f.read_text()) for f in files]
    assert reports, f"no match.json under {JOBS_DIR} or {packaged}"
    # Two shapes have to survive ingest and they are not variations of each
    # other: the demo carries a full feed and no real tracks, a pipeline run
    # carries real tracks and no feed at all. Assert on both or the store only
    # works for whichever one happened to sort first.
    # The HAND-MADE demo, not just the first report with a feed. Once the
    # third-party feed importer (not in this repository) landed, "first with facts" started picking whichever real
    # fixture sorted first, and this self-check began asserting one match's
    # possession against another's. A fixture a test pins to has to be named.
    # Named means the PACKAGED one: "first with a feed and no fotmob_id" began
    # picking data/jobs/0beb202f..., which has facts and zero players, and this
    # check failed on it before a single field was ingested.
    report = json.loads(next(packaged.glob("*/match.json")).read_text())
    raw = next((r for r in reports
                if not (r.get("meta") or {}).get("facts")
                and any(p.get("team") == "unknown" for p in r.get("players") or [])), None)

    conn = connect(":memory:")
    got = ingest_match(conn, "m1", report)
    q = lambda sql, *a: conn.execute(sql, a).fetchall()  # noqa: E731

    assert got["players"] >= 22 and got["tracks"] == 22, got
    assert got["positions"] == got["tracks"] * HEATMAP_POINTS, got

    # Every position is on the pitch it claims to be on.
    row = q("SELECT min(x_m) a, max(x_m) b, min(y_m) c, max(y_m) d FROM position")[0]
    m = q("SELECT * FROM match")[0]
    assert 0 <= row["a"] and row["b"] <= m["pitch_length_m"], dict(row)
    assert 0 <= row["c"] and row["d"] <= m["pitch_width_m"], dict(row)

    # THE IDENTITY RULE. Nobody has bound anything, so no measured number has
    # a player - even though 22 tracks and 31 named players both exist.
    assert q("SELECT count(*) c FROM track WHERE player_id IS NOT NULL")[0]["c"] == 0
    assert q("SELECT count(*) c FROM player_track_metric")[0]["c"] == 0
    assert q("SELECT count(*) c FROM track_metric WHERE metric='distance_km'")[0]["c"] == 22

    pid = q("SELECT player_id FROM appearance WHERE started=1 LIMIT 1")[0]["player_id"]
    tid = q("SELECT track_id FROM track LIMIT 1")[0]["track_id"]
    bind_track(conn, "m1", tid, pid)
    bound = q("SELECT * FROM player_track_metric WHERE metric='distance_km'")
    assert len(bound) == 1 and bound[0]["bind_source"] == "manual", bound

    # Entered and measured stay apart: the feed's per-player rows are keyed on
    # names and never landed in track_metric.
    assert q("SELECT count(*) c FROM player_metric WHERE metric='rating'")[0]["c"] >= 22
    assert q("SELECT count(*) c FROM track_metric WHERE metric='rating'")[0]["c"] == 0

    # "618 (88%)" survived as both a count and a percentage.
    ap = q("SELECT * FROM player_metric WHERE metric='acc-passes' AND raw LIKE '%(%' LIMIT 1")[0]
    assert ap["value"] and ap["pct"] and ap["raw"], dict(ap)

    # Four featured stats, and the featured row carries the section's value -
    # the two representations cannot disagree because there is one row.
    feat = q("SELECT * FROM team_metric WHERE featured=1 AND side='home'")
    assert len(feat) == 4, [r["metric"] for r in feat]
    poss = q("SELECT * FROM team_metric WHERE metric='ball-possession' AND side='home'")[0]
    assert poss["value"] == 72 and poss["featured"] == 1, dict(poss)

    # Territory is the fixture's, not a team's.
    assert q("SELECT count(*) c FROM team_metric WHERE side='both'")[0]["c"] > 0

    # Goals, cards and substitutions are one timeline now.
    types = {r["type"] for r in q("SELECT DISTINCT type FROM event")}
    assert {"goal", "yellow_card", "sub", "shot"} <= types, types
    assert "substitution" not in types, "the feed's two spellings both survived"
    # Nine substitutions in the feed and nine rows - not eighteen, which is what
    # storing facts.subs and facts.events as separate events gives you.
    assert q("SELECT count(*) c FROM event WHERE type='sub'")[0]["c"] == 9
    # Eight of the nine, because the demo's own timeline takes off a player who
    # is on neither its XI nor its bench. The database is supposed to make that
    # visible: the name survives in detail even though the link cannot.
    assert q("SELECT count(*) c FROM event_player WHERE role='off'")[0]["c"] == 8
    orphan = q("SELECT detail FROM event WHERE type='sub' AND detail LIKE '%off_name%'")
    assert len(orphan) == 1 and "caro" in orphan[0]["detail"], orphan
    # Who came off is knowable only from the timeline; who came on, from either.
    # Nine, not eight: the team sheet also carries an `off` minute, and for the
    # 59th minute the sheet and the timeline name DIFFERENT players. Both are
    # kept - the sheet's on appearance, the timeline's in event.detail - because
    # picking a winner here would be inventing data, and a coach looking at a
    # substitution that disagrees with itself is the correct outcome.
    assert q("SELECT count(*) c FROM appearance WHERE off_minute IS NOT NULL")[0]["c"] == 9
    assert q("SELECT count(*) c FROM appearance WHERE started=0 AND on_minute IS NULL")[0]["c"] == 0
    assert q("SELECT count(*) c FROM event_player WHERE role='assist'")[0]["c"] >= 1
    # The two naming conventions joined: the timeline's "Leo Brandt" and the
    # team sheet's "Brandt" are one player row, not two.
    assert q("SELECT count(*) c FROM event WHERE type='goal' AND player_id IS NULL")[0]["c"] == 0

    # Home shots land in the half a home side attacks.
    hs = q("SELECT avg(x_m) a FROM event WHERE type='shot' AND side='home' AND x_m IS NOT NULL")
    as_ = q("SELECT avg(x_m) a FROM event WHERE type='shot' AND side='away' AND x_m IS NOT NULL")
    assert hs[0]["a"] > m["pitch_length_m"] / 2 > as_[0]["a"], (hs[0]["a"], as_[0]["a"])

    # Series are rows, so a window is a WHERE rather than a slice in Python.
    first15 = q("SELECT avg(width_m) w FROM shape_sample WHERE side='home' AND t_s < 900")
    assert 20 < first15[0]["w"] < 60, first15[0]["w"]
    assert q("SELECT count(*) c FROM formation_window")[0]["c"] == 360
    assert q("SELECT count(*) c FROM momentum_sample")[0]["c"] == 96

    # Every play's shirt list came off its own board.
    for p in q("SELECT match_id, play_id, board FROM play"):
        drawn = {int(mg["label"]) for mg in json.loads(p["board"])["magnets"]
                 if str(mg.get("label", "")).isdigit()}
        stored = {r["shirt"] for r in q(
            "SELECT shirt FROM play_player WHERE play_id=?", p["play_id"])}
        assert drawn == stored, (p["play_id"], drawn, stored)

    # Provenance came across whole, and nothing lost its tier.
    fm = q("SELECT * FROM field_meta")
    assert len(fm) == 38, len(fm)
    assert all(r["provenance"] for r in fm)
    # The tier stored is the one resolved TODAY, not the one frozen in the
    # file - the same answer app.match_json serves - and `not_evaluated` is a
    # value the store accepts, which its CHECK once refused.
    from stats import resolve  # noqa: PLC0415
    now = resolve(report["fields"], report.get("meta"))
    # The packaged demo is generated positions: nothing on it may cite a
    # measurement. It shipped without the flag and served justified tiers.
    assert (report.get("meta") or {}).get("synthetic_tracking"), "demo not marked synthetic"
    assert not any(e.get("justified_by") for e in now.values()), \
        "the shipped synthetic demo cites a measurement"
    dk = q("SELECT * FROM field_meta WHERE field_path='players.distance_km'")[0]
    assert dk["confidence"] == now["players.distance_km"]["confidence"], dict(dk)
    assert dk["justified_by"] == now["players.distance_km"]["justified_by"], dict(dk)
    assert dk["provenance"] == "derived", dict(dk)
    conn.execute("INSERT OR REPLACE INTO field_meta (match_id, field_path, confidence) "
                 "VALUES ('m1', '_probe', 'not_evaluated')")

    # AN OLD STORE MIGRATES WITHOUT LOSING A ROW. Planted: field_meta in the
    # shape every existing store has, with a row in it, then a reconnect.
    import tempfile  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "old.db")
        connect(path).close()
        c = sqlite3.connect(path)
        c.executescript(
            "PRAGMA foreign_keys=OFF; DROP TABLE field_meta;"
            "CREATE TABLE field_meta (match_id TEXT NOT NULL, field_path TEXT NOT NULL,"
            " units TEXT, n INTEGER, confidence TEXT CHECK (confidence IN ('ok','rel','prov')),"
            " provenance TEXT, caveat_flags TEXT, PRIMARY KEY (match_id, field_path)) STRICT;"
            "INSERT INTO match (match_id, fps, pitch_length_m, pitch_width_m)"
            " VALUES ('x', 25, 105, 68);"
            "INSERT INTO field_meta VALUES ('x','kept','m',1,'rel','derived','[]');"
            "INSERT INTO field_meta VALUES ('orphan','gone','m',1,'rel','derived','[]');")
        c.close()
        c = connect(path)
        assert [tuple(r) for r in c.execute(
            "SELECT field_path, confidence FROM field_meta")] == [("kept", "not_evaluated")], \
            "a migrated row kept its pre-resolution tier"
        c.execute("INSERT INTO field_meta (match_id, field_path, confidence) "
                  "VALUES ('x', 'new', 'not_evaluated')")   # the CHECK moved
        c.close()

    # Re-ingest is a replace, not a double.
    again = ingest_match(conn, "m1", report)
    assert again == got, (again, got)

    # A real pipeline run: no feed at all, and tracks the classifier abstained
    # on. Both are the normal state today, and neither may raise.
    assert raw, "no feed-free pipeline output to ingest"
    got2 = ingest_match(conn, "m2", raw)
    assert got2["players"] == 0 and got2["events"] == 0, got2
    assert got2["tracks"] > 0 and got2["positions"] > 0, got2
    unknown = q("SELECT count(*) c FROM track WHERE match_id='m2' AND side IS NULL")[0]["c"]
    assert unknown > 0, "the abstaining classifier's tracks lost their NULL"
    # ...and they drop out of every side-keyed question rather than forming a
    # third team in the GROUP BY.
    sides = {r["side"] for r in q("SELECT DISTINCT side FROM track WHERE match_id='m2'")}
    assert sides == {None, "home", "away"}, sides
    assert q("SELECT count(*) c FROM match")[0]["c"] == 2

    # --- the club ladder ---------------------------------------------------
    # An academy is a ladder of squads and ONE player who climbs it. Both
    # failures below are silent: a wrong ladder order just looks odd, and a
    # squad row that disappears takes a player's club membership with it while
    # every other number on the page stays right.
    view = {
        "club": {"club_id": "westfield", "name": "Westfield FC", "season": "2026/27"},
        "teams": [
            {"team_id": "u15-boys", "name": "U15 Boys", "age_band": "U15",
             "level": "A", "gender": "boys", "season": "2026/27",
             "coach_ids": ["ana-ruiz"]},
            {"team_id": "first-team", "name": "First team", "age_band": "",
             "coach_ids": ["ana-ruiz", "bo-chen"]},
        ],
        "coaches": [{"coach_id": "ana-ruiz", "name": "Ana Ruiz", "role": "Head coach",
                     "email": ""},
                    {"coach_id": "bo-chen", "name": "Bo Chen", "role": "", "email": ""}],
    }
    # One human in BOTH squads, with a different shirt in each. That is the
    # whole point of squad membership being its own table: a number on `player`
    # would have to pick one of the two and be wrong about the other.
    rows = [
        {"id": "ana-ruiz", "name": "Ana Ruiz", "number": 7, "position": "RW",
         "team_id": "u15-boys"},
        {"id": "ana-ruiz", "name": "Ana Ruiz", "number": 24, "position": "RW",
         "team_id": "first-team"},
        {"id": "bo-chen", "name": "Bo Chen", "number": 4, "position": "CB",
         "team_id": "u15-boys"},
        {"id": "ghost", "name": "Ghost", "number": 1, "position": "GK",
         "team_id": "no-such-team"},   # a roster file for a team off the ladder
    ]
    org = ingest_org(conn, view, rows)
    assert org == {"club": "westfield", "teams": 2, "coaches": 2, "squad": 3}, org
    assert q("SELECT is_ours FROM club WHERE club_id='westfield'")[0]["is_ours"] == 1
    assert q("SELECT is_ours FROM club WHERE club_id='home'")[0]["is_ours"] == 0, \
        "a club off a team sheet is not ours"

    # The ladder order, which every naive sort gets wrong.
    assert [r["name"] for r in q("SELECT name FROM team ORDER BY age_key, level")] == \
        ["U15 Boys", "First team"]

    # The career: one human, two squads, two shirts.
    career = q("SELECT team_id, number FROM squad WHERE player_id='ana-ruiz'"
               " ORDER BY team_id")
    assert [(r["team_id"], r["number"]) for r in career] == \
        [("first-team", 24), ("u15-boys", 7)], career

    # WAS A BUG, and the reason ingest_match upserts instead of INSERT OR
    # REPLACE: a REPLACE is a DELETE plus an INSERT, so re-ingesting a match
    # fired squad's ON DELETE CASCADE and emptied every membership of anyone
    # whose name appeared on a team sheet. Nothing raised; the club just lost
    # its squads. `report` names Ana Ruiz, so this re-ingest reproduces it.
    report["meta"]["facts"]["lineups"]["home"]["xi"].append(
        {"name": "Ana Ruiz", "n": 7, "pos": "RW"})
    ingest_match(conn, "m1", report)
    assert q("SELECT count(*) c FROM squad")[0]["c"] == 3, \
        "re-ingesting a match cascaded the club's squad away again"

    # match.team_id: whose match it is in OUR club, which the report cannot say.
    set_match_team(conn, "m1", "u15-boys", "2026/27")
    assert q("SELECT team_id FROM match WHERE match_id='m1'")[0]["team_id"] == "u15-boys"
    # An unknown team lands as NULL rather than a foreign-key error - a job can
    # name a team that was removed from the ladder, and it is still a match.
    set_match_team(conn, "m2", "deleted-team")
    assert q("SELECT team_id FROM match WHERE match_id='m2'")[0]["team_id"] is None

    # Re-ingesting the club REPLACES the ladder, so a team deleted in org.json
    # cannot survive here as a ghost row that season queries still group by -
    # and the match that pointed at it is set back to unattributed rather than
    # blocking the delete.
    view["teams"] = [view["teams"][0]]
    ingest_org(conn, view, rows)
    assert q("SELECT count(*) c FROM team")[0]["c"] == 1
    assert q("SELECT count(*) c FROM squad")[0]["c"] == 2, "cascade should drop first-team"
    # ...and the SURVIVING team's matches keep their attribution. This is the
    # reason ingest_org upserts rather than deleting the ladder and writing it
    # back: match.team_id is ON DELETE SET NULL, so a rewrite-in-place used to
    # unattribute the entire library on every single-match ingest.
    assert q("SELECT team_id FROM match WHERE match_id='m1'")[0]["team_id"] == "u15-boys"

    # A coach removed from the club leaves, and takes their assignments with
    # them rather than lingering as a name on a squad.
    view["coaches"] = [view["coaches"][0]]
    view["teams"][0]["coach_ids"] = []
    ingest_org(conn, view, rows)
    assert q("SELECT count(*) c FROM coach")[0]["c"] == 1
    assert q("SELECT count(*) c FROM team_coach")[0]["c"] == 0

    # ...and removing the LAST one actually empties it. `NOT IN ()` rendered as
    # `NOT IN (NULL)` is NULL, never true, so this prune used to delete nothing
    # at all and a club that let its staff go kept them in the store forever.
    view["coaches"] = []
    ingest_org(conn, view, rows)
    assert q("SELECT count(*) c FROM coach")[0]["c"] == 0, "empty prune deleted nothing"
    # The same shape of hole on the ladder: no teams means no teams, and the
    # matches that pointed at one go back to unattributed rather than blocking.
    view["teams"] = []
    ingest_org(conn, view, rows)
    assert q("SELECT count(*) c FROM team")[0]["c"] == 0, "empty prune deleted nothing"
    assert q("SELECT count(*) c FROM squad")[0]["c"] == 0, "team delete cascades squad"
    assert q("SELECT team_id FROM match WHERE match_id='m1'")[0]["team_id"] is None

    print("db ok:", got, org)


def build(db_path: str | None = None) -> dict:
    """Every job's match.json plus the club ladder into one store. The whole
    point of the exercise: thirty match.json files cannot answer "this player's
    last six games", and one database can."""
    import jobs  # noqa: PLC0415 - heavy import, CLI only

    path = db_path or str(jobs.DATA_DIR / "match.db")
    conn = connect(path)
    # The ladder FIRST: match.team_id references team, so attributing a match
    # to a squad that has not been ingested yet is a foreign-key error.
    org = ingest_org(conn, jobs.org(), jobs.roster())
    out = {}
    teams = {st["id"]: (_slug_team(st.get("team")), st.get("season") or "")
             for st in jobs.all_jobs()}
    for f in sorted(jobs.JOBS_DIR.glob("*/match.json")):
        jid = f.parent.name
        try:
            out[jid] = ingest_match(conn, jid, json.loads(f.read_text()))
            tid, season = teams.get(jid, (None, ""))
            set_match_team(conn, jid, tid, season)
        except Exception as e:  # one malformed job must not cost the other thirty
            # ...and must not leave its half-written rows pending either: the
            # NEXT match's commit would write them.
            conn.rollback()
            out[jid] = {"error": f"{type(e).__name__}: {e}"}

    # PRUNE. Ingest only ever inserts and replaces, so a match whose job has
    # been deleted - or which was never a job at all - survives every rebuild
    # and keeps appearing in season totals. That is not hypothetical: it is how
    # a test run's throwaway match ended up in a real store. "Delete the
    # database and re-ingest" is the promise this module's docstring makes, and
    # this is the half of it build() was missing.
    live = list(out)
    gone = conn.execute(
        f"DELETE FROM match WHERE match_id NOT IN ({','.join('?' * len(live)) or 'NULL'})",
        live).rowcount
    conn.commit()

    # THE BINDINGS, replayed. ingest_match writes the tracks; WHICH PERSON a
    # track is lives in the roster file, because it is a human act and nothing
    # in match.json knows it. build() never replayed them, so every rebuilt
    # store had track.player_id NULL on every row - and the whole point of the
    # store is per-player questions across matches.
    #
    # It was silent because two code paths answer the same question from
    # different places: /api/season/measured reads the roster JSON directly and
    # reported 110 confirmed bindings, while the store's own Named column, over
    # the same club, read 0. Two numbers, one fact, and the disagreement only
    # showed as an amber zero on a table nobody had built yet.
    #
    # ONLY `source == "manual"`. A machine guess is not a binding - identity
    # reaches chance a few minutes into a match - and this is the same rule
    # jobs.season() applies, deliberately spelled out in both places rather than
    # left to be inferred from one.
    bound = 0
    for rp in jobs._roster_files():
        try:
            r = roster.Roster.load(str(rp))
        except Exception:  # noqa: BLE001 - a corrupt roster must not empty the store
            continue
        for mid, tb in r.bindings.items():
            if mid not in out:
                continue  # a binding for a match whose job is gone
            for tid, b in tb.items():
                if b.get("source") != "manual":
                    continue
                bind_track(conn, mid, int(tid), b["player_id"],
                           b.get("confidence", 1.0), "manual")
                bound += 1

    conn.close()
    return {"db": path, "org": org, "jobs": out, "pruned": gone, "bound": bound}


def _slug_team(name) -> str | None:
    """A job's team string -> the team_id org.py minted for it. Same slug on
    both sides by construction (org.Club.add_team slugs the name), which is why
    the job never had to store an id."""
    return roster._slug(name or "") or None


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["build"]:
        print(json.dumps(build(*sys.argv[2:3]), indent=1))
    else:
        _withheld_demo()   # first: _demo fails independently at HEAD 49cc12c ('players: 0')
        print("db.py withheld sprints ok")
        _demo()
