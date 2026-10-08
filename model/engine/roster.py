"""Per-team squad and match_id -> track_id -> binding, persisted as one JSON file per team.

Contract: docs/reference/identity.md#roster. Rationale: identity-evidence (not published)#evidence-design-stance
"""
from dataclasses import asdict, dataclass, field
import json
import os
import re
import unicodedata


@dataclass
class Player:
    """One human, stable across the season."""

    player_id: str
    team: str
    number: int | None = None
    name: str | None = None
    # free text a coach types; never derived from tracked position
    # (identity-evidence (not published)#evidence-design-stance)
    position: str | None = None


def _slug(s: str) -> str:
    """Name -> ASCII id fragment: NFKD accent-FOLDING (José and Jose agree), lowercased,
    non-alphanumeric runs to "-". "" for a name with no Latin letters; add() then falls back
    to team-number. identity-evidence (not published)#evidence-roster-ids
    """
    folded = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")


@dataclass
class Roster:
    """One team's squad plus every match's track bindings.

    bindings: match_id -> track_id -> {player_id, confidence, source}. Keeping
    the confidence and the source means the correction UI can show which
    bindings a human confirmed and which the pipeline guessed, and the next
    match's identity pass can trust the former and re-derive the latter.
    """

    team: str
    season: str = ""
    players: dict[str, Player] = field(default_factory=dict)
    bindings: dict[str, dict[int, dict]] = field(default_factory=dict)

    # --- squad -------------------------------------------------------------

    def add(self, number: int | None = None, name: str | None = None,
            player_id: str | None = None, position: str | None = None) -> Player:
        """Add or update a player -> Player.

        Id: `player_id` if given, else _slug(name), else "<team-slug>-<number>" (the team slug
        alone with no number). An existing id is updated (number, name, position when given),
        never re-keyed, so a shirt change stays one player and two humans with one name share
        one id unless the caller passes `player_id`. docs/reference/identity.md#limits
        """
        pid = player_id
        if not pid:
            pid = _slug(name) if name else ""
            if not pid:
                pid = f"{_slug(self.team)}-{number}" if number is not None else _slug(self.team)
        p = self.players.get(pid)
        if p is None:
            p = Player(player_id=pid, team=self.team, number=number, name=name,
                       position=position)
            self.players[pid] = p
        else:  # a re-add updates the mutable facts, never the id
            if number is not None:
                p.number = number
            if name:
                p.name = name
            if position:
                p.position = position
        return p

    def by_number(self, number: int) -> Player | None:
        """First player holding `number`, in insertion order, or None."""
        return next((p for p in self.players.values() if p.number == number), None)

    # --- bindings ----------------------------------------------------------

    def bind(self, match_id: str, track_id: int, player_id: str,
             confidence: float = 1.0, source: str = "manual") -> None:
        """Bind a track to a player; confidence stored to 3 dp. KeyError for a player_id not in
        the squad. No guard against overwriting a manual binding: callers check confirmed()."""
        if player_id not in self.players:
            raise KeyError(f"unknown player {player_id!r}; add() it first")
        self.bindings.setdefault(match_id, {})[int(track_id)] = {
            "player_id": player_id, "confidence": round(float(confidence), 3),
            "source": source,
        }

    def player_for(self, match_id: str, track_id: int) -> Player | None:
        """The bound Player, or None; track_id may be int or str."""
        b = self.bindings.get(match_id, {}).get(int(track_id))
        return self.players.get(b["player_id"]) if b else None

    def confirmed(self, match_id: str) -> dict[int, str]:
        """Human-verified bindings for a match. These override any machine guess."""
        return {t: b["player_id"] for t, b in self.bindings.get(match_id, {}).items()
                if b.get("source") == "manual"}

    def tracks_for(self, player_id: str) -> list[tuple[str, int]]:
        """Every (match_id, track_id) this player owns. The season join."""
        return sorted((m, t) for m, tb in self.bindings.items()
                      for t, b in tb.items() if b["player_id"] == player_id)

    # --- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        """Atomic write (temp file + os.replace); track ids become JSON string keys."""
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({
                "team": self.team, "season": self.season,
                "players": {k: asdict(v) for k, v in self.players.items()},
                # JSON object keys are strings; track ids come back as ints on load.
                "bindings": {m: {str(t): b for t, b in tb.items()}
                             for m, tb in self.bindings.items()},
            }, f, indent=2)
        os.replace(tmp, path)  # a half-written roster is worse than a stale one

    @classmethod
    def load(cls, path: str, team: str = "") -> "Roster":
        """Load, or return an empty roster for `team` if the file doesn't exist. Track ids come back
        as int. Raises on unreadable JSON."""
        if not os.path.exists(path):
            return cls(team=team)
        with open(path) as f:
            d = json.load(f)
        return cls(
            team=d.get("team", team), season=d.get("season", ""),
            players={k: Player(**v) for k, v in d.get("players", {}).items()},
            bindings={m: {int(t): b for t, b in tb.items()}
                      for m, tb in d.get("bindings", {}).items()},
        )


# ---------------------------------------------------------------------------

def _demo() -> None:
    """The two ways a season profile dies quietly: two people merged into one
    id, or one person split across two. Both leave plausible numbers behind, so
    nothing downstream will ever notice. Everything here asserts CURRENT
    behaviour - the KNOWN BUG blocks pin bugs, they do not excuse them.
    """
    import tempfile  # noqa: PLC0415 - self-check only

    # 1. Slug normalisation. Punctuation and case fold away, which is wanted.
    assert _slug("Ana Ruiz") == _slug("ana  ruiz") == "ana-ruiz"
    assert _slug("O'Brien") == _slug("O Brien") == "o-brien"

    # --- id-scheme bugs: identity-evidence (not published)#evidence-roster-ids
    # BUG 1 (same name = same id) STANDS: the caller disambiguates with player_id.
    r = Roster(team="Westfield")
    a = r.add(number=9, name="Jon Smith")
    b = r.add(number=14, name="Jon Smith")
    assert a is b and len(r.players) == 1, "BUG 1 changed; re-read this block"
    assert a.number == 14, "the second add takes over the number"
    # The escape hatch, and the reason this is a limitation rather than a bug:
    c = r.add(number=14, name="Jon Smith", player_id="jon-smith-2")
    assert c is not a and len(r.players) == 2 and c.player_id == "jon-smith-2"

    # WAS BUG 2: a name with no Latin letters slugged to "" and a whole squad shared one id.
    assert _slug("李明") == _slug("王芳") == "", "fold changed; re-read _slug"
    u = Roster(team="Westfield")
    li, wang = u.add(number=7, name="李明"), u.add(number=8, name="王芳")
    assert len(u.players) == 2, "non-Latin squad collapsed again"
    assert li.player_id == "westfield-7" and wang.player_id == "westfield-8"
    assert all(p.player_id for p in u.players.values()), "empty id is falsy downstream"
    assert li.name == "李明", "the NAME must survive verbatim; only the id is ASCII"

    # WAS BUG 3: accents were stripped, not folded, so one player typed two ways split in two.
    assert _slug("José Ruiz") == _slug("Jose Ruiz") == "jose-ruiz"
    assert _slug("Muñoz") == _slug("Munoz") == "munoz"
    assert _slug("Håland") == _slug("Haland") == "haland"
    acc = Roster(team="Westfield")
    assert acc.add(number=7, name="José Ruiz") is acc.add(number=7, name="Jose Ruiz")
    assert len(acc.players) == 1, "one player, typed two ways, split again"

    # 2. THE reason ids are name-slugged: a mid-season shirt change must not
    # split one player in two. This is the behaviour the docstring promises.
    r = Roster(team="Westfield", season="2026")
    ana = r.add(number=9, name="Ana Ruiz")
    again = r.add(number=21, name="Ana Ruiz")           # January, new shirt
    assert again.player_id == ana.player_id, (again, ana)
    assert len(r.players) == 1 and ana.number == 21, r.players
    # A number-keyed id would have produced two. Prove the id ignores the number.
    assert ana.player_id == "ana-ruiz", ana.player_id

    # 3. Bindings across three matches, all joining back to one player_id.
    # tracks_for IS the season join; if it drops a match, a transfer profile
    # quietly reports two thirds of a season as if it were the whole thing.
    for m, t in (("match-01", 4), ("match-02", 11), ("match-03", 2)):
        r.bind(m, t, ana.player_id, 1.0, "manual")
    assert r.tracks_for(ana.player_id) == [
        ("match-01", 4), ("match-02", 11), ("match-03", 2)], r.tracks_for(ana.player_id)
    assert r.player_for("match-02", 11).player_id == ana.player_id
    assert r.player_for("match-02", "11") is not None, "str track id must resolve"
    assert r.player_for("match-09", 4) is None, "unknown match must not resolve"

    # 4. Human-confirmed vs pipeline-guessed. The next identity pass trusts the
    # first and re-derives the second, so the distinction must survive a file.
    bo = r.add(number=6, name="Bo Chen")
    r.bind("match-03", 8, bo.player_id, 0.42, "ocr")
    assert r.confirmed("match-03") == {2: ana.player_id}, r.confirmed("match-03")
    assert r.bindings["match-03"][8]["source"] == "ocr"

    # Roster.bind has NO guard against overwriting a human's answer; pinned so it stays visible.
    r.bind("match-03", 2, bo.player_id, 0.1, "ocr")
    assert r.confirmed("match-03") == {}, "bind() silently downgraded a human binding"
    r.bind("match-03", 2, ana.player_id, 1.0, "manual")  # put it back

    # 5. Persistence. Empty roster, unicode, and every field intact.
    with tempfile.TemporaryDirectory() as td:  # cleans up even if an assert fires
        path = os.path.join(td, "westfield.json")
        r.save(path)
        r2 = Roster.load(path)
        assert (r2.team, r2.season) == ("Westfield", "2026"), (r2.team, r2.season)
        assert r2.players == r.players and r2.bindings == r.bindings
        # JSON object keys are strings; track ids MUST come back as ints or every
        # lookup keyed on an int misses and the season join silently empties.
        assert all(isinstance(t, int) for tb in r2.bindings.values() for t in tb)
        assert r2.tracks_for(ana.player_id) == r.tracks_for(ana.player_id)
        assert r2.confirmed("match-03") == {2: ana.player_id}
        assert r2.bindings["match-03"][8] == {"player_id": bo.player_id,
                                              "confidence": 0.42, "source": "ocr"}
        assert not os.path.exists(path + ".tmp"), "atomic save left its temp file"

        # An empty roster round-trips as empty, not as a crash and not as None.
        empty = os.path.join(td, "empty.json")
        Roster(team="Nobody FC").save(empty)
        e = Roster.load(empty)
        assert e.team == "Nobody FC" and e.players == {} and e.bindings == {}

        # Unicode names survive verbatim. The id is still ASCII (see BUG 2), but
        # the display name must not come back mangled or the operator can't tell
        # who a profile belongs to.
        uni = os.path.join(td, "uni.json")
        ur = Roster(team="Beşiktaş")
        n = ur.add(number=10, name="Emre Kaya")
        ur.bind("match-01", 3, n.player_id)
        ur.save(uni)
        u2 = Roster.load(uni)
        assert u2.team == "Beşiktaş" and u2.players[n.player_id].name == "Emre Kaya"
        assert u2.tracks_for(n.player_id) == [("match-01", 3)]

        # A roster that was never created reads as empty rather than exploding -
        # every caller in jobs.py relies on this for a team's first match.
        fresh = Roster.load(os.path.join(td, "nope.json"), team="New FC")
        assert fresh.team == "New FC" and fresh.players == {}

    # 6. Bindings must never point at a player who isn't in the squad, or
    # player_for returns None for a track that looks bound and the minutes
    # vanish with no error anywhere.
    try:
        r.bind("match-04", 1, "ghost-player")
        raise AssertionError("bind accepted an unknown player_id")
    except KeyError:
        pass

    # by_number is first-match-wins over insertion order; pinned so it stays visible.
    dup = Roster(team="Westfield")
    old = dup.add(number=17, name="Old Guy")
    new = dup.add(number=3, name="New Guy")
    dup.add(player_id=new.player_id, number=17)
    assert dup.by_number(17) is old, "by_number tie-break changed"
    assert dup.by_number(99) is None

    # A nameless player still gets a usable id, and confidence is stored rounded
    # so a re-save can't drift the value that decides trust.
    numbered = Roster(team="Westfield").add(number=9)
    assert numbered.player_id == "westfield-9", numbered.player_id
    r.bind("match-01", 5, bo.player_id, 0.123456, "ocr")
    assert r.bindings["match-01"][5]["confidence"] == 0.123

    print("roster.py ok")


if __name__ == "__main__":
    _demo()
