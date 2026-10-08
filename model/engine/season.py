"""The season fact table: every match's entered feed, flattened into rows.

WHY A TABLE AND NOT THREE ENDPOINTS. The Data surface wants clubs, players and
opponents. Those are three GROUPINGS of one fact, not three questions: a club
row IS the sum of its per-match rows, and an opponent row is the same sum cut a
different way. So this returns the atoms - one row per (match, club) and one per
(match, player) - and the client groups them. 11 matches is 22 club rows and
~344 player rows, tens of KB, against 19 MB of match.json that never has to
leave the machine.

WHAT IS IN HERE AND WHAT IS NOT. Everything below comes from `meta.facts`, the
human-entered / vendor feed: names, goals, xG, cards. It is ATTRIBUTABLE because
a human attributed it. The pipeline's own measurements - distance, sprints, top
speed - are deliberately absent: identity reaches chance a few minutes into a
match, so a per-player season total of a MEASURED quantity would be a confident
sum of the wrong people. `teams.distance_m` is honest at team level and still
carries a 19% over-read; it is left to the per-match report where its caveat is
already written, rather than being multiplied by eleven here.

COVERAGE IS A COLUMN, NOT A FOOTNOTE. The feeds disagree about which stats they
carry: `minutes` is on all 344 player lines in this dataset, `sot` on 31 - one
match. Summing both and printing two numbers side by side says "season total"
about one of them and lies. So every aggregate ships `n`, the number of rows
that actually carried it, and the client shows it.
"""
from __future__ import annotations

import json
from pathlib import Path


def _num(v) -> float | None:
    """Feed values arrive as 23, "3.63" or "32%". Return a number or None.

    A percentage returns its face value (32%, not 0.32) because that is what
    every label carrying one is named for - "Ball possession" reads 32, and
    a mean of face values is the mean of the percentages.
    """
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, str):
        return None
    s = v.strip().rstrip("%").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None  # "5 (3)" and friends: a shape this table does not model


def _score(s) -> tuple[int, int] | None:
    """"2 - 1" -> (2, 1). Anything else -> None, and the row carries no result."""
    if not isinstance(s, str):
        return None
    parts = s.replace("–", "-").split("-")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except ValueError:
        return None


def _stats_for_side(facts: dict, side: str) -> dict[str, float]:
    """Every section row, flattened to {label: number} for one side.

    Sections repeat labels across themselves - "Total shots" is in both Top
    stats and Shots - which is harmless: the same label carries the same number,
    so last-write-wins is a no-op rather than a choice.
    """
    out: dict[str, float] = {}
    for section in facts.get("sections") or []:
        for row in section.get("rows") or []:
            label = row.get("label")
            n = _num(row.get(side))
            if isinstance(label, str) and n is not None:
                out[label] = n
    # top_stats carries `kind` and sometimes a label the sections do not.
    for row in facts.get("top_stats") or []:
        label = row.get("label")
        n = _num(row.get(side))
        if isinstance(label, str) and n is not None:
            out.setdefault(label, n)
    return out


def rows(match: dict, jid: str) -> tuple[dict | None, list[dict], list[dict]]:
    """One match.json -> (fixture, club rows, player rows).

    Returns (None, [], []) for a match with no entered feed - a real upload that
    the pipeline tracked but nobody has entered a teamsheet for. It is not an
    error and it is not a zero; it simply has no rows in this table.
    """
    meta = match.get("meta") or {}
    facts = meta.get("facts") or {}
    home, away = meta.get("home_name"), meta.get("away_name")
    if not facts or not home or not away:
        return None, [], []

    sc = _score(meta.get("score"))
    fixture = {
        "id": jid,
        "home": home,
        "away": away,
        "score": meta.get("score"),
        "date": meta.get("kickoff_utc"),
        "competition": meta.get("competition"),
        "round": meta.get("round"),
        "demo": bool(meta.get("demo")),
    }

    clubs = []
    for side, club, opp in (("home", home, away), ("away", away, home)):
        gf = ga = None
        if sc is not None:
            gf, ga = (sc[0], sc[1]) if side == "home" else (sc[1], sc[0])
        clubs.append({
            "match": jid,
            "club": club,
            "opponent": opp,
            "home": side == "home",
            "gf": gf,
            "ga": ga,
            "stats": _stats_for_side(facts, side),
        })

    players = []
    for p in facts.get("player_stats") or []:
        side = p.get("side")
        name = p.get("name")
        if side not in ("home", "away") or not isinstance(name, str):
            continue
        stats = {k: n for k, n in
                 ((k, _num(v)) for k, v in (p.get("stats") or {}).items())
                 if n is not None}
        players.append({
            "match": jid,
            "club": home if side == "home" else away,
            "opponent": away if side == "home" else home,
            "name": name,
            "pos": p.get("pos"),
            "stats": stats,
        })

    return fixture, clubs, players


def build(jobs_dir: Path, ids: list[str] | None = None) -> dict:
    """Read every finished match under `jobs_dir` and return the fact table.

    A match.json that is missing, half-written or not JSON is skipped rather
    than raising: this endpoint is a view over whatever has finished, and one
    corrupt job must not blank the whole database.
    """
    fixtures: list[dict] = []
    clubs: list[dict] = []
    players: list[dict] = []

    for d in sorted(jobs_dir.iterdir()) if jobs_dir.exists() else []:
        if ids is not None and d.name not in ids:
            continue
        p = d / "match.json"
        if not p.is_file():
            continue
        try:
            match = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError, ValueError):
            continue
        fx, cs, ps = rows(match, d.name)
        if fx is None:
            continue
        fixtures.append(fx)
        clubs += cs
        players += ps

    # Newest first, and a match with no kickoff sorts last rather than crashing
    # the comparison against the ones that have one.
    fixtures.sort(key=lambda f: (f["date"] or ""), reverse=True)
    return {"fixtures": fixtures, "clubs": clubs, "players": players}


def _demo() -> None:
    """Self-check. The parsing is where this is wrong if it is wrong."""
    assert _num(23) == 23.0
    assert _num("3.63") == 3.63
    assert _num("32%") == 32.0, "a percentage keeps its face value"
    assert _num("5 (3)") is None, "an unmodelled shape is absent, not zero"
    assert _num(None) is None and _num(True) is None

    assert _score("2 - 1") == (2, 1)
    assert _score("0-0") == (0, 0)
    assert _score("2 – 1") == (2, 1), "en dash, which is what feeds emit"
    assert _score("abandoned") is None and _score(None) is None

    match = {"meta": {
        "home_name": "Northbridge FC", "away_name": "Eastvale FC", "score": "2 - 1",
        "kickoff_utc": "2025-10-26T15:15:00.000Z", "competition": "Example League",
        "facts": {
            "sections": [{"title": "Top stats", "rows": [
                {"label": "Ball possession", "home": "32%", "away": "68%"},
                {"label": "Total shots", "home": 23, "away": 15}]}],
            "top_stats": [{"label": "Big chances", "home": 4, "away": 2}],
            "player_stats": [
                {"name": "Tomas Varga", "side": "home", "pos": "Attacker",
                 "stats": {"goals": 1, "rating": "8.4", "minutes": 90}},
                {"name": "Piotr Nowel", "side": "away", "pos": "Goalkeeper",
                 "stats": {"saves": 5, "minutes": 90}}],
        }}}
    fx, cs, ps = rows(match, "abc")
    assert fx["home"] == "Northbridge FC" and fx["id"] == "abc"
    assert len(cs) == 2
    home = next(c for c in cs if c["home"])
    away = next(c for c in cs if not c["home"])
    assert (home["gf"], home["ga"]) == (2, 1)
    assert (away["gf"], away["ga"]) == (1, 2), "the away row is not the home row"
    assert home["opponent"] == "Eastvale FC" and away["club"] == "Eastvale FC"
    assert home["stats"]["Ball possession"] == 32.0
    assert home["stats"]["Big chances"] == 4.0, "top_stats fills gaps in sections"
    assert len(ps) == 2
    striker = next(p for p in ps if p["name"].startswith("Tomas"))
    assert striker["club"] == "Northbridge FC" and striker["opponent"] == "Eastvale FC"
    assert striker["stats"] == {"goals": 1.0, "rating": 8.4, "minutes": 90.0}

    # No feed is no rows, not empty rows.
    assert rows({"meta": {"home_name": "A", "away_name": "B"}}, "x") == (None, [], [])
    assert rows({}, "x") == (None, [], [])

    print("season.py self-check OK")


if __name__ == "__main__":
    _demo()
