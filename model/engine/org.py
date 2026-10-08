"""The club: its teams, its age levels, its coaches.

WHY THIS FILE EXISTS. Until now a "team" was a free-text string on a job and a
file called `rosters/<slug>.json`. That is enough for one squad and it is the
wrong shape for an academy, where the unit of the product is a CLUB with a
ladder of teams under it - U11, U13, U15, U17, reserves, first team - each with
its own coaches and its own squad, and one player who climbs that ladder.

WHAT IS AND IS NOT HERE.

  * HERE: the club, its teams (age band, level, gender, season), its coaches,
    and which coach works with which team. All of it entered by a human; none
    of it is derivable from footage, which is why it needs a store of its own.

  * NOT HERE: players. A team's squad and its track bindings stay in
    roster.py's per-team file, untouched. That is not laziness, it is the
    identity model: roster.py mints `player_id` from the NAME SLUG, so a player
    who appears in `u15.json` and in `u17.json` already carries ONE id in both.
    The club-level player registry is therefore a MERGE over the team rosters
    on a key that is already stable - see jobs.roster - and not a second store
    that could disagree with them. A second store is exactly how one human
    becomes two half-seasons, which is the failure roster.py exists to prevent.

    The known limit rides along with it: two different humans with the same
    name share an id, in one team or across two (roster.py `_demo`, BUG 1). The
    escape hatch is the same one - give the second player an explicit
    `player_id`.

  * NOT HERE EITHER: accounts. A coach is a record, not a login. There is no
    auth anywhere in this product yet and TODOS (not published) gates it on the first hosted
    deployment. Nothing in this file is a security boundary and nothing should
    start treating it as one.

ONE CLUB PER INSTALL. The install is the academy. Opponents are still stored as
clubs in the match store, because a season history has to join on who you
played - but there is exactly one "us", and it lives here.
"""
from dataclasses import asdict, dataclass, field
import json
import os
import re

from roster import _slug

# The academy ladder, oldest last. Anything that is not an age band sorts after
# every band that is one, because "First team" and "Reserves" sit above U19 and
# a plain alphabetical sort puts them between U19 and U21.
SENIOR_KEY = 99


def age_key(band: str) -> int:
    """Sort key for an age band. U11 < U15 < U19 < anything else.

    A club's team list is read as a LADDER - a coach scanning it expects U11 at
    the top and the first team at the bottom - and every naive sort gets this
    wrong: alphabetical gives U11, U13, U15, U9 (U9 last, because '9' > '1'),
    and a plain int parse crashes on 'First team'.
    """
    m = re.search(r"\bu-?(\d{1,2})\b", (band or "").lower())
    return int(m.group(1)) if m else SENIOR_KEY


def band_of(name: str) -> str:
    """Pull 'U15' out of 'U15 Boys A'. Returns '' when there is no band in the
    name, which is the honest answer for 'First team' - the caller shows the
    team without a band rather than inventing one."""
    m = re.search(r"\bu-?(\d{1,2})\b", (name or "").lower())
    return f"U{m.group(1)}" if m else ""


@dataclass
class Coach:
    """One human on the staff. Not an account: see the module docstring."""

    coach_id: str
    name: str
    role: str = ""                       # 'Head coach', 'Assistant', 'GK', ...
    email: str = ""


@dataclass
class Team:
    """One squad in the club.

    `team_id` is also the roster file name, so a team and its squad cannot
    drift apart: `rosters/<team_id>.json` is that team's players and bindings,
    and it is the same slug jobs._roster_path already computes from the team
    name. Renaming a team therefore keeps its id - the display name changes and
    the squad follows it, rather than the rename orphaning a roster file.
    """

    team_id: str
    name: str                            # 'U15 Boys', 'First team'
    # Parsed from the name on create, but STORED, because a club that calls a
    # squad 'Infantil A' still wants it on the U13 rung and only a human knows.
    age_band: str = ""
    gender: str = ""                     # 'boys' | 'girls' | 'mixed' | ''
    level: str = ""                      # 'A' | 'B' - the second axis under a band
    season: str = ""                     # '2026/27'; '' means the club's current
    coach_ids: list[str] = field(default_factory=list)


@dataclass
class Club:
    """The academy. One per install."""

    club_id: str = ""
    name: str = ""
    season: str = ""                     # the current season, the default for new teams
    teams: dict[str, Team] = field(default_factory=dict)
    coaches: dict[str, Coach] = field(default_factory=dict)

    # --- teams -------------------------------------------------------------

    def add_team(self, name: str, team_id: str | None = None, age_band: str | None = None,
                 gender: str = "", level: str = "", season: str | None = None,
                 coach_ids: list[str] | None = None) -> Team:
        """Upsert a team. Passing an existing `team_id` edits it in place and
        NEVER re-slugs from the new name - a rename must not orphan the roster
        file the id points at."""
        tid = team_id or _slug(name)
        if not tid:
            raise ValueError("team name required")
        t = self.teams.get(tid)
        if t is None:
            t = Team(team_id=tid, name=name,
                     age_band=age_band if age_band is not None else band_of(name),
                     gender=gender, level=level,
                     season=season if season is not None else self.season,
                     coach_ids=list(coach_ids or []))
            self.teams[tid] = t
            return t
        if name:
            t.name = name
        if age_band is not None:
            t.age_band = age_band
        if gender:
            t.gender = gender
        if level:
            t.level = level
        if season is not None:
            t.season = season
        if coach_ids is not None:
            t.coach_ids = list(coach_ids)
        return t

    def ladder(self) -> list[Team]:
        """Every team, youngest first, seniors last. The order the club is read
        in, so every surface that lists teams agrees on it."""
        return sorted(self.teams.values(),
                      key=lambda t: (age_key(t.age_band or t.name), t.level, t.name))

    # --- coaches -----------------------------------------------------------

    def add_coach(self, name: str, coach_id: str | None = None, role: str = "",
                  email: str = "") -> Coach:
        cid = coach_id or _slug(name)
        if not cid:
            # A name with no Latin letters folds to "", and "" would collide
            # every such coach onto one record - roster.py's BUG 2, same shape.
            cid = f"coach-{len(self.coaches) + 1}"
        c = self.coaches.get(cid)
        if c is None:
            c = Coach(coach_id=cid, name=name, role=role, email=email)
            self.coaches[cid] = c
            return c
        if name:
            c.name = name
        if role:
            c.role = role
        if email:
            c.email = email
        return c

    def teams_of(self, coach_id: str) -> list[Team]:
        return [t for t in self.ladder() if coach_id in t.coach_ids]

    def coaches_of(self, team_id: str) -> list[Coach]:
        t = self.teams.get(team_id)
        return [self.coaches[c] for c in (t.coach_ids if t else []) if c in self.coaches]

    # --- migration ---------------------------------------------------------

    def seed(self, names) -> int:
        """Register any team name that exists in the data but not in the club.

        Read-time, not a batch migration. An install that has been running as
        "one team is a string on a job" comes up with those strings already on
        the ladder, so nothing has to be re-entered and no job.json is rewritten
        - the same reason jobs._defaults exists rather than a migration script.

        Returns how many were new.
        """
        n = 0
        for name in names:
            name = (name or "").strip()
            if not name or _slug(name) in self.teams:
                continue
            self.add_team(name)
            n += 1
        return n

    # --- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(tmp, "w") as f:
            json.dump({
                "club_id": self.club_id, "name": self.name, "season": self.season,
                "teams": {k: asdict(v) for k, v in self.teams.items()},
                "coaches": {k: asdict(v) for k, v in self.coaches.items()},
            }, f, indent=2)
        os.replace(tmp, path)  # a half-written club is worse than a stale one

    @classmethod
    def load(cls, path: str) -> "Club":
        """Load, or an empty club if nothing has been entered yet."""
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            d = json.load(f)
        return cls(
            club_id=d.get("club_id", ""), name=d.get("name", ""),
            season=d.get("season", ""),
            teams={k: Team(**v) for k, v in d.get("teams", {}).items()},
            coaches={k: Coach(**v) for k, v in d.get("coaches", {}).items()},
        )


# ---------------------------------------------------------------------------

def _demo() -> None:
    """The ladder order and the rename-keeps-its-id rule. Both are silent when
    they break: a mis-sorted ladder just looks odd, and a re-slugged team id
    orphans a roster file - the squad and its season vanish with no error."""
    import tempfile  # noqa: PLC0415 - self-check only

    # 1. The ladder. Every naive sort gets U9 wrong.
    assert age_key("U9") == 9 and age_key("u-9") == 9 and age_key("U15") == 15
    assert age_key("First team") == age_key("") == SENIOR_KEY
    assert band_of("U15 Boys A") == "U15" and band_of("First team") == ""

    c = Club(club_id="westfield", name="Westfield FC", season="2026/27")
    for n in ("U15 Boys", "First team", "U9 Boys", "U13 Girls", "Reserves"):
        c.add_team(n)
    assert [t.name for t in c.ladder()] == [
        "U9 Boys", "U13 Girls", "U15 Boys", "First team", "Reserves"], \
        [t.name for t in c.ladder()]
    # Alphabetical would have put U9 last and the seniors mid-ladder.
    assert c.teams["u15-boys"].age_band == "U15"
    assert c.teams["first-team"].age_band == "", "a senior team has no band"
    # New teams inherit the club's current season so a coach types it once.
    assert c.teams["u9-boys"].season == "2026/27"

    # The second axis: two squads on the same rung sort A before B.
    c.add_team("U15 Boys B", level="B")
    c.add_team("U15 Boys", team_id="u15-boys", level="A")
    assert [t.team_id for t in c.ladder()][:3] == ["u9-boys", "u13-girls", "u15-boys"]
    assert [t.team_id for t in c.ladder()][3] == "u15-boys-b", c.ladder()[3]

    # 2. A RENAME KEEPS THE ID. The id is the roster file name, so re-slugging
    # on rename would leave the squad behind in a file nothing looks at.
    c.add_team("U16 Boys", team_id="u15-boys", age_band="U16")
    assert "u15-boys" in c.teams and "u16-boys" not in c.teams
    assert c.teams["u15-boys"].name == "U16 Boys"
    assert c.teams["u15-boys"].age_band == "U16", "an explicit band must win"
    assert c.teams["u15-boys"].level == "A", "a partial edit must not clear the rest"
    c.add_team("U15 Boys", team_id="u15-boys", age_band="U15")  # put it back

    # 3. Coaches, and the many-to-many. One coach can run two age groups and one
    # team can have three staff - which is why this is a list, not a column.
    ana = c.add_coach("Ana Ruiz", role="Head coach")
    bo = c.add_coach("Bo Chen", role="Assistant")
    assert ana.coach_id == "ana-ruiz"
    c.add_team("U15 Boys", team_id="u15-boys", coach_ids=[ana.coach_id, bo.coach_id])
    c.add_team("U9 Boys", team_id="u9-boys", coach_ids=[ana.coach_id])
    assert [t.team_id for t in c.teams_of("ana-ruiz")] == ["u9-boys", "u15-boys"]
    assert [t.team_id for t in c.teams_of("bo-chen")] == ["u15-boys"]
    assert [x.name for x in c.coaches_of("u15-boys")] == ["Ana Ruiz", "Bo Chen"]
    assert c.coaches_of("nope") == [], "an unknown team is empty, not a crash"
    # A coach dropped from a team leaves no dangling reference behind.
    c.add_team("U15 Boys", team_id="u15-boys", coach_ids=[ana.coach_id])
    assert c.teams_of("bo-chen") == []
    # A name with no Latin letters must not collide every coach onto one record.
    z1, z2 = c.add_coach("李明"), c.add_coach("王芳")
    assert z1.coach_id != z2.coach_id and z1.name == "李明"

    # 4. Seeding an install that predates this file. The team strings already on
    # the jobs and the roster files become the ladder, without a migration.
    fresh = Club(name="Westfield FC")
    assert fresh.seed(["Northbridge FC", "Default", "", None]) == 2
    assert set(fresh.teams) == {"northbridge-fc", "default"}
    assert fresh.seed(["Northbridge FC"]) == 0, "seeding twice must not duplicate"

    # 5. Persistence, including an empty club and unicode.
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "sub", "org.json")   # parent dir is created for us
        c.save(p)
        c2 = Club.load(p)
        assert (c2.club_id, c2.name, c2.season) == ("westfield", "Westfield FC", "2026/27")
        assert c2.teams == c.teams and c2.coaches == c.coaches
        assert [t.team_id for t in c2.ladder()] == [t.team_id for t in c.ladder()]
        assert not os.path.exists(p + ".tmp"), "atomic save left its temp file"

        empty = os.path.join(td, "empty.json")
        Club(name="Nobody FC").save(empty)
        e = Club.load(empty)
        assert e.name == "Nobody FC" and e.teams == {} and e.ladder() == []

        # Never created reads as empty rather than exploding - every caller in
        # jobs.py relies on this for a club's first run.
        assert Club.load(os.path.join(td, "nope.json")).teams == {}

    print("org.py ok")


if __name__ == "__main__":
    _demo()
