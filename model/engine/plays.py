"""Plays: one moment of a match, drawn on the same board a coach draws on.

A PLAY IS A BOARD PLUS METADATA. The diagram is exactly the structure the
notepad's whiteboard already stores - magnets, shapes, crop, orientation - so a
play dropped into a note is not a screenshot or a special widget, it is a board
that happens to have been filled in by the machine instead of by hand. A coach
can then move a disc on it, which is the whole point: the analysis arrives as
something editable rather than as a picture of somebody else's analysis.

This is the shape of InStat's Pic. 1-28: a cropped pitch, discs for both sides,
solid arrows for the ball, dashed for runs, and a caption naming the fixture and
the clock. Ours carry more - type, zone, outcome, who was involved - because the
point is not to print 28 fixed pictures but to let a coach ask for "set pieces
in the final third" and scroll what comes back.

EVERYTHING HERE IS SYNTHETIC. The pipeline follows bodies and never sees the
ball, so it cannot know that a moment was a corner or that a pass was intended.
These plays are written into `meta.facts`, which the app already treats as
entered-not-measured, and every block that renders one carries that tier. The
generator is deterministic so the demo is stable, and it hangs its minutes,
sides and outcomes off the feed that is already there - so the plays agree with
the shot map and the timeline rather than telling a third story.

The real version of this file computes plays from tracks + ball. That is the
2D-map work; this is the schema it has to produce, written down early so
everything downstream can be built against it.
"""
from dataclasses import dataclass, field
import json
import random
import sys

# Board coordinates are PERCENT of the pitch, 0-100 on both axes, and home
# attacks to the RIGHT. Same space as app/frontend/src/components/notepad/pitch-data.ts
# - if that file's Magnet/Shape/Board change, this has to change with it, because
# nothing checks it across the language boundary.

TYPES = {
    "positional_attack": "Positional attack",
    "counter_attack": "Counter-attack",
    "high_press": "High press",
    "position_defence": "Position defence",
    "corner_attack": "Corner, attacking",
    "corner_defence": "Corner, defending",
    "free_kick_attack": "Free kick, attacking",
    "throw_in": "Throw-in",
}

ZONES = ("own_third", "middle_third", "final_third", "box")
OUTCOMES = ("goal", "shot", "chance", "cleared", "lost")


@dataclass
class Play:
    id: str
    type: str
    side: str
    period: int
    minute: int
    clock: str
    zone: str
    outcome: str
    players: list[int]
    board: dict
    # Seconds into the video. The film tab already speaks this unit (clips are
    # seconds, never frames, so a re-track cannot orphan them) and a play that
    # cannot be watched back is half a play.
    t: float = 0.0
    note: str = ""
    source: str = "demo"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _board(magnets, shapes, view="full", orient="h") -> dict:
    """A board in the notepad's own shape. The styling keys are included rather
    than left out: a play that renders with a different grass to the board
    beside it looks like a different kind of object, which it is not."""
    return {
        "magnets": magnets,
        "shapes": shapes,
        "view": view,
        "orient": orient,
        "lines": "standard",
        "grass": "green",
        "stripes": True,
    }


def _m(x, y, label="", side="home"):
    return {"x": round(x, 1), "y": round(y, 1), "label": str(label), "side": side}


def _ball(x, y):
    return {"x": round(x, 1), "y": round(y, 1), "label": "", "side": "ball"}


def _s(kind, pts, color="white", label=None):
    d = {"kind": kind, "pts": [round(p, 1) for p in pts], "color": color}
    if label is not None:
        d["label"] = label
    return d


def _mirror(board: dict) -> dict:
    """Flip a board left-to-right, so one generator can write every play as if
    it attacks right and the away version falls out. Sides swap with it, or the
    colours would say the wrong team."""
    swap = {"home": "away", "away": "home", "ball": "ball"}
    return {
        **board,
        "magnets": [{**m, "x": round(100 - m["x"], 1), "side": swap[m["side"]]} for m in board["magnets"]],
        "shapes": [
            {**s, "pts": [round(100 - p, 1) if i % 2 == 0 else p for i, p in enumerate(s["pts"])]}
            for s in board["shapes"]
        ],
    }


# --------------------------------------------------------------------- shapes

def _defenders(rng, xs, n=4, spread=(18, 82)):
    """A defensive line, jittered. A perfectly even line is the one thing a real
    one never is, and a diagram of it reads as a formation chart rather than a
    moment."""
    lo, hi = spread
    step = (hi - lo) / max(1, n - 1)
    return [_m(xs + rng.uniform(-2.5, 2.5), lo + i * step + rng.uniform(-3, 3), "", "away") for i in range(n)]


def _positional(rng) -> dict:
    """Build-up through the middle: the ball circulated, one line broken."""
    home = [
        _m(14, 50, "1"), _m(28, 20, "2"), _m(26, 42, "5"), _m(26, 58, "6"), _m(28, 80, "3"),
        _m(42, 34, "8"), _m(40, 50, "4"), _m(44, 66, "7"),
        _m(58, 26, "11"), _m(62, 50, "9"), _m(58, 74, "10"),
    ]
    away = _defenders(rng, 72, 4) + _defenders(rng, 56, 3, (28, 72))
    x0, y0 = 26, 42
    return _board(
        home + away + [_ball(x0, y0)],
        [
            _s("pass", [x0, y0, 40, 50]),
            _s("pass", [40, 50, 58, 26], "yellow"),
            _s("run", [58, 26, 74, 32], "yellow"),
            _s("run", [62, 50, 76, 46]),
            _s("text", [38, 12], "white", "switch, then in behind"),
        ],
        view="full",
    )


def _counter(rng) -> dict:
    """Won deep, moved to the free side in two passes."""
    home = [
        _m(20, 50, "1"), _m(30, 30, "5"), _m(30, 70, "6"),
        _m(38, 18, "3"), _m(44, 52, "8"), _m(40, 82, "2"),
        _m(64, 30, "11"), _m(70, 52, "9"), _m(66, 74, "10"),
    ]
    away = [_m(48 + rng.uniform(-4, 4), 20 + i * 20, "", "away") for i in range(4)] + [
        _m(78, 50, "", "away"), _m(84, 40, "", "away")
    ]
    return _board(
        home + away + [_ball(44, 52)],
        [
            _s("pass", [44, 52, 66, 74], "red"),
            _s("dribble", [66, 74, 82, 66], "red"),
            _s("run", [70, 52, 88, 48], "yellow"),
            _s("run", [64, 30, 84, 30], "yellow"),
            _s("text", [46, 88], "red", "3 v 2 the moment it turns"),
        ],
        view="full",
    )


def _press(rng) -> dict:
    """Six players onto the ball inside three seconds."""
    away = [_m(22, 50, "", "away"), _m(30, 26, "", "away"), _m(30, 74, "", "away"),
            _m(42, 40, "", "away"), _m(44, 62, "", "away")]
    home = [
        _m(34, 30, "11"), _m(36, 52, "9"), _m(34, 70, "10"),
        _m(48, 38, "8"), _m(50, 60, "7"), _m(58, 50, "4"),
    ]
    return _board(
        home + away + [_ball(30, 26)],
        [
            _s("zone", [18, 12, 48, 88], "yellow"),
            _s("run", [34, 30, 31, 27], "yellow"),
            _s("run", [36, 52, 32, 40], "yellow"),
            _s("run", [48, 38, 38, 32], "yellow"),
            _s("text", [50, 7], "yellow", "trap the touchline"),
        ],
        view="half",
    )


def _low_block(rng) -> dict:
    """Two banks, and the pass that has to be allowed."""
    home = [_m(52 + rng.uniform(-1, 1), 14 + i * 18, "", "home") for i in range(5)] + [
        _m(64 + rng.uniform(-1, 1), 24 + i * 17, "", "home") for i in range(4)
    ]
    away = [_m(40, 50, "6", "away"), _m(46, 22, "7", "away"), _m(46, 78, "11", "away")]
    return _board(
        home + away + [_ball(40, 50)],
        [
            _s("zone", [48, 10, 70, 90], "blue"),
            _s("pass", [40, 50, 46, 22], "blue"),
            _s("text", [50, 7], "blue", "shoulder is free"),
        ],
        view="half",
    )


def _corner(rng, defending=False) -> dict:
    """The set-piece view: cropped to the box, stood upright, goal at the top.
    Exactly how pages 10 and 11 of a scouting report are drawn."""
    near = rng.random() < 0.5
    y0 = 2 if near else 98
    att = "away" if defending else "home"
    dfd = "home" if defending else "away"
    box = [
        _m(88, 38, "9", att), _m(90, 52, "19", att), _m(86, 62, "2", att),
        _m(93, 46, "15", att), _m(84, 48, "8", att),
    ]
    keeper = [_m(97, 50, "1", dfd)]
    marks = [_m(m["x"] - 2.5, m["y"] + rng.uniform(-3, 3), "", dfd) for m in box[:4]]
    edge = [_m(76, 50, "5", att)]
    target = (89, 48 if near else 54)
    return _board(
        [_m(99, y0, "10", att)] + box + keeper + marks + edge + [_ball(99, y0)],
        [
            _s("pass", [99, y0, target[0], target[1]], "yellow"),
            _s("run", [88, 38, 92, 44], "white"),
            _s("run", [86, 62, 90, 56], "white"),
            _s("run", [76, 50, 84, 50], "white"),
            # No words on a box crop: a sentence needs more pitch than a 24%
            # window has, and the caption under the diagram already carries it.
        ],
        view="box",
        orient="v",
    )


def _free_kick(rng) -> dict:
    wide = rng.random() < 0.5
    y0 = 16 if wide else 50
    x0 = 70 if wide else 76
    return _board(
        [
            _m(x0, y0, "10"), _m(88, 40, "9"), _m(90, 54, "19"), _m(86, 62, "2"), _m(93, 47, "15"),
            _m(97, 50, "1", "away"),
            _m(90, 44, "", "away"), _m(88, 56, "", "away"), _m(92, 60, "", "away"),
            _ball(x0, y0),
        ],
        [
            _s("pass", [x0, y0, 90, 50], "yellow"),
            _s("run", [88, 40, 92, 46]),
            _s("run", [86, 62, 89, 56]),
        ],
        view="box",
        orient="v",
    )


def _throw_in(rng) -> dict:
    y0 = 2 if rng.random() < 0.5 else 98
    return _board(
        [
            _m(62, y0, "3"), _m(58, y0 + (12 if y0 < 50 else -12), "8"),
            _m(70, y0 + (16 if y0 < 50 else -16), "11"), _m(66, 50, "9"),
            _m(60, y0 + (8 if y0 < 50 else -8), "", "away"),
            _m(72, y0 + (20 if y0 < 50 else -20), "", "away"),
            _ball(62, y0),
        ],
        [
            _s("pass", [62, y0, 58, y0 + (12 if y0 < 50 else -12)]),
            _s("run", [70, y0 + (16 if y0 < 50 else -16), 82, y0 + (10 if y0 < 50 else -10)], "yellow"),
        ],
        view="third",
    )


BUILDERS = {
    "positional_attack": lambda r: _positional(r),
    "counter_attack": lambda r: _counter(r),
    "high_press": lambda r: _press(r),
    "position_defence": lambda r: _low_block(r),
    "corner_attack": lambda r: _corner(r, False),
    "corner_defence": lambda r: _corner(r, True),
    "free_kick_attack": lambda r: _free_kick(r),
    "throw_in": lambda r: _throw_in(r),
}

# The crops, as PERCENT of the pitch - the same windows VIEWS defines in
# app/frontend/src/components/notepad/pitch-data.ts, converted from its metres. A
# mark outside its own board's crop is invisible, or worse half-visible: the
# first version of this file put "switch, then in behind" at x=20 on a board
# cropped to the final third, and it rendered as "hind".
CROPS = {
    "full": (0.0, 0.0, 100.0, 100.0),
    "half": (47.6, 0.0, 100.0, 100.0),
    "third": (64.8, 0.0, 100.0, 100.0),
    "box": (76.2, 7.4, 100.0, 92.6),
}

# Roughly how wide a character is, as a percent of pitch length, at the size
# shapeSvg draws labels. Used only to keep a label from running off the right of
# its crop - it does not need to be exact, it needs to be cautious.
CHAR_W = 1.35


ZONE_OF = {
    "positional_attack": "final_third",
    "counter_attack": "middle_third",
    "high_press": "middle_third",
    "position_defence": "own_third",
    "corner_attack": "box",
    "corner_defence": "box",
    "free_kick_attack": "box",
    "throw_in": "final_third",
}


def _clock(minute: int) -> str:
    return f"{minute}:{(minute * 37) % 60:02d}"


def generate(report: dict, n: int = 28, seed: int = 7) -> list[dict]:
    """A season's worth of one match's plays, hung off the feed already there.

    Minutes, sides and outcomes come from `facts.events` and `facts.shots` where
    those exist, so a play agrees with the timeline and the shot map instead of
    inventing a third account of the same match. Everything the feed cannot
    supply - the diagram, the type, who was involved - is synthesised.
    """
    rng = random.Random(seed)
    meta = report.get("meta") or {}
    facts = meta.get("facts") or {}
    events = facts.get("events") or []
    shots = (facts.get("shots") or {}).get("points") or []
    squad = {
        "home": [p["n"] for p in ((facts.get("lineups") or {}).get("home") or {}).get("xi", [])] or [7, 8, 9, 10, 11],
        "away": [p["n"] for p in ((facts.get("lineups") or {}).get("away") or {}).get("xi", [])] or [7, 8, 9, 10, 11],
    }

    # Anchors: a real moment for each shot and each goal, then filler across the
    # ninety so a filter on minute has something to bite on.
    anchors: list[tuple[int, str, str]] = []
    for e in events:
        if e.get("type") == "goal":
            m = int("".join(ch for ch in e.get("minute", "0") if ch.isdigit())[:2] or 0)
            anchors.append((m, e.get("side") or "home", "goal"))
    for s in shots[:10]:
        anchors.append((rng.randint(3, 88), s.get("side") or "home", "shot"))
    while len(anchors) < n:
        anchors.append((rng.randint(2, 90), rng.choice(["home", "away"]), rng.choice(OUTCOMES)))

    kinds = list(BUILDERS)
    out: list[dict] = []
    for i, (minute, side, outcome) in enumerate(sorted(anchors)[:n]):
        kind = kinds[i % len(kinds)]
        board = BUILDERS[kind](rng)
        if side == "away":
            board = _mirror(board)
        # Who was involved comes off the BOARD, not out of the squad list. A
        # coach who filters for "#19 involved" and gets a diagram with no 19 on
        # it has been told something false about the one thing they asked.
        # `pool` is only the fallback for a board drawn with unnumbered discs.
        pool = squad.get(side) or [7, 8, 9, 10]
        drawn = sorted(
            {
                int(m["label"])
                for m in board["magnets"]
                if m["side"] == ("away" if side == "away" else "home") and m["label"].isdigit()
            }
        )
        involved = drawn[:4] if drawn else sorted(rng.sample(pool, min(3, len(pool))))
        out.append(
            Play(
                id=f"play-{i + 1:02d}",
                type=kind,
                side=side,
                period=1 if minute <= 45 else 2,
                minute=minute,
                clock=_clock(minute),
                zone=ZONE_OF[kind],
                outcome=outcome,
                players=involved,
                board=board,
                t=float(minute * 60 + (minute * 37) % 60),
                note="",
                source="demo",
            ).to_dict()
        )
    return out


def write_into(path: str, n: int = 28, seed: int = 7) -> int:
    """Put generated plays into a match.json's `meta.facts.plays`.

    Under `facts` on purpose: that key is already the app's word for entered
    rather than measured, so a play inherits the tier and every block that
    renders one says so without needing a second vocabulary.
    """
    with open(path) as f:
        report = json.load(f)
    plays = generate(report, n=n, seed=seed)
    report.setdefault("meta", {}).setdefault("facts", {})["plays"] = plays
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f)
    import os
    os.replace(tmp, path)  # a half-written report is worse than a stale one
    return len(plays)


# ---------------------------------------------------------------------------

def _selfcheck() -> None:
    """The board a play carries has to be one the notepad can actually render.
    Everything asserted here is a shape the TypeScript side will parse, and a
    play that fails it renders as an empty pitch with no error anywhere."""
    demo = {
        "meta": {
            "facts": {
                "events": [{"minute": "11'", "side": "home", "type": "goal"}],
                "shots": {"points": [{"x": 0.1, "y": 0.3, "side": "home", "goal": True, "period": 1}]},
                "lineups": {"home": {"xi": [{"n": i} for i in range(1, 12)]},
                            "away": {"xi": [{"n": i} for i in range(1, 12)]}},
            }
        }
    }
    plays = generate(demo, n=28)
    assert len(plays) == 28, len(plays)

    # Deterministic: a demo that reshuffles on every run is one nobody can
    # write a test or a screenshot against.
    assert generate(demo, n=28) == plays, "generator is not deterministic"

    kinds = {"pass", "run", "dribble", "pen", "zone", "text"}
    sides = {"home", "away", "ball"}
    views = {"full", "half", "third", "box"}
    for p in plays:
        assert p["type"] in TYPES, p["type"]
        assert p["zone"] in ZONES and p["outcome"] in OUTCOMES, p
        assert p["side"] in ("home", "away")
        assert 1 <= p["minute"] <= 90 and p["period"] in (1, 2)
        assert p["period"] == (1 if p["minute"] <= 45 else 2)
        b = p["board"]
        assert b["view"] in views and b["orient"] in ("h", "v"), b["view"]
        assert b["magnets"], "a play with no discs is a blank pitch"
        # Every coordinate inside the pitch: the TS parser clamps, so an out-of
        # range disc would silently pile up on a touchline instead of erroring.
        for m in b["magnets"]:
            assert m["side"] in sides, m
            assert 0 <= m["x"] <= 100 and 0 <= m["y"] <= 100, m
        assert any(m["side"] == "ball" for m in b["magnets"]), "no ball in the play"
        for s in b["shapes"]:
            assert s["kind"] in kinds, s
            assert len(s["pts"]) >= (2 if s["kind"] == "text" else 4), s
            assert all(0 <= v <= 100 for v in s["pts"]), s
            if s["kind"] == "text":
                assert s.get("label"), "a text shape with no words is an invisible mark"
                # A label outside its own crop is not a cosmetic problem: it
                # renders as the tail of a sentence with no beginning.
                x0, y0, x1, y1 = CROPS[b["view"]]
                tx, ty = s["pts"][0], s["pts"][1]
                assert x0 <= tx <= x1 and y0 <= ty <= y1, (
                    f"{p['id']}: label starts outside the {b['view']} crop")
                run = tx + len(s["label"]) * CHAR_W
                assert run <= x1 + 2, (
                    f"{p['id']}: label runs {run:.0f}% past the {b['view']} crop")

    # Mirroring puts an away play in the away half and swaps the colours, or
    # every away diagram would show the wrong team attacking the wrong goal.
    away = [p for p in plays if p["side"] == "away"]
    assert away, "no away plays generated"
    for p in away:
        assert any(m["side"] == "away" for m in p["board"]["magnets"]), p["id"]

    # Every player the filter can return has to be ON the diagram it returns.
    # Filtering for a shirt number and getting a board without it is the one
    # failure that makes the browser worse than scrolling everything.
    for p in plays:
        side = p["side"]
        drawn = {
            int(m["label"])
            for m in p["board"]["magnets"]
            if m["side"] == side and m["label"].isdigit()
        }
        if drawn:
            missing = set(p["players"]) - drawn
            assert not missing, f"{p['id']}: filterable {sorted(missing)} not on the board"

    # A filter is the whole point, so every axis one can filter on has to have
    # more than one value in the demo.
    for axis in ("type", "zone", "outcome", "side"):
        assert len({p[axis] for p in plays}) > 1, f"{axis} has one value; nothing to filter"
    assert len({p["period"] for p in plays}) == 2, "all plays in one half"

    print("plays.py ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "write":
        print(f"wrote {write_into(sys.argv[2])} plays into {sys.argv[2]}")
    else:
        _selfcheck()
