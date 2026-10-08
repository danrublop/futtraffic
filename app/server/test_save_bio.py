"""The one check save_bio has to pass: it MERGES.

Run it: `.venv/bin/python app/server/test_save_bio.py`

Not a framework, and not a suite - the thing worth a test here is the single
behaviour the whole feature rests on. The bio file has two authors: the feed
wrote the traits and the heatmap, the operator types a height over the top. If
the write ever turns into a replace, the measured half disappears the first
time somebody fixes a shirt number, and it disappears QUIETLY - the panel just
stops drawing a hexagon. That is the failure this asserts against.

DATA_DIR is rebound to a temp dir, which works only because bios_dir() is a
function of it (see data-dir-rebind-needs-functions (internal note)).
"""

from __future__ import annotations

import datetime as _dt
import json
import tempfile
from pathlib import Path

import _path  # noqa: F401  - puts model/engine/ on sys.path
import jobs


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        jobs.DATA_DIR = Path(tmp)
        d = jobs.bios_dir()
        d.mkdir(parents=True)
        (d / "ida-hellstrom.json").write_text(
            json.dumps({"name": "Ada", "height_cm": 170, "traits": [{"key": "goals"}]})
        )

        # 1. THE MERGE. A typed height must not cost the feed's traits.
        out = jobs.save_bio("ida-hellstrom", {"height_cm": 173})
        assert out["height_cm"] == 173, out
        assert out["traits"] == [{"key": "goals"}], "the feed's half was dropped"
        assert out["name"] == "Ada", out

        # 2. CLEARING a box removes the field - it does not store a null, which
        #    every reader here treats as "present but unknown".
        out = jobs.save_bio("ida-hellstrom", {"height_cm": ""})
        assert "height_cm" not in out, out

        # 3. AGE IS DERIVED. Entering a birth date must set it, and an age sent
        #    by a client must never win over the date.
        out = jobs.save_bio("ida-hellstrom", {"birth_date": "1995-07-10", "age": 3})
        assert out["age"] >= 29, out
        assert out["birth_date"] == "1995-07-10", out

        # 4. FIELDS OUTSIDE THE FORM are dropped, not written: an operator does
        #    not get to type over a measurement.
        out = jobs.save_bio("ida-hellstrom", {"traits": [], "shots": [1, 2]})
        assert out["traits"] == [{"key": "goals"}], out
        assert "shots" not in out, out

        # 5. A NEW PLAYER has no file yet; the first save creates one.
        out = jobs.save_bio("new-signing", {"name": "New", "other_positions": ["CM", "AM"]})
        assert out == json.loads((d / "new-signing.json").read_text()), out
        assert out["other_positions"] == ["CM", "AM"], out

        # 6. THE ID IS A SLUG, and a path is not one.
        # "abc\n" is the one that got through: Python's `$` matches before a
        # trailing newline, and the id arrives percent-decoded from the path.
        for bad in ("../../etc/passwd", "", "a/b", "abc\n", "ida-hellstrom\n"):
            try:
                jobs.save_bio(bad, {"name": "x"})
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted a bad id: {bad!r}")

        # 7. A BIO FILE THAT WILL NOT PARSE is treated as an empty one, not as
        #    a 500. This file is written by a feed job and read by a form; a
        #    half-written or hand-edited one must still let the operator save,
        #    because the alternative is a profile nobody can ever fix again.
        (d / "torn.json").write_text('{"name": "half a wr')
        out = jobs.save_bio("torn", {"height_cm": 181})
        assert out == {"height_cm": 181}, out

        # 7b. And a file whose top level is not an object. json.loads succeeds
        #     here, so the ValueError guard above never fires - it is the
        #     isinstance check that catches this one.
        (d / "listy.json").write_text("[1, 2, 3]")
        out = jobs.save_bio("listy", {"name": "L"})
        assert out == {"name": "L"}, out

        # 8. A BODY THAT IS NOT AN OBJECT is a 400, not a crash: app.put_bio
        #    turns exactly this ValueError into the status the client reads.
        for bad in ([], "x", None, 3):
            try:
                jobs.save_bio("ida-hellstrom", bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted a non-object body: {bad!r}")

        # 9. AGE CANNOT OUTLIVE ITS DATE. Clearing the birth date has to take
        #    the derived age with it, or the card goes on printing an age with
        #    nothing under it - the exact drift the derivation exists to stop.
        out = jobs.save_bio("ida-hellstrom", {"birth_date": "1995-07-10"})
        assert "age" in out, out
        out = jobs.save_bio("ida-hellstrom", {"birth_date": ""})
        assert "age" not in out and "birth_date" not in out, out

        # 10. A DATE IN THE FUTURE derives no age rather than a negative one.
        #     A mistyped year is the ordinary way this arrives (2019 for 1991).
        out = jobs.save_bio("ida-hellstrom", {"birth_date": f"{_dt.date.today().year + 2}-01-01"})
        assert "age" not in out, out
        assert out["birth_date"].endswith("-01-01"), out

        # 11. A VALUE OF THE WRONG TYPE is skipped, and skipping means the
        #     STORED value survives. A dropdown cannot send "tall", but a
        #     client that changes shape can, and losing a real height to it
        #     would look exactly like the operator having cleared the box.
        jobs.save_bio("kinds", {"height_cm": 180, "other_positions": ["CM"]})
        out = jobs.save_bio("kinds", {"height_cm": "tall", "other_positions": "CM"})
        assert out["height_cm"] == 180, out
        assert out["other_positions"] == ["CM"], out
        # A numeric string IS a height - that is what a form sends.
        assert jobs.save_bio("kinds", {"height_cm": "183"})["height_cm"] == 183, out

        # 12. THE CAPS ARE REAL, on every kind: a paste into a text box must
        #     not be able to write an arbitrarily large file into the store.
        out = jobs.save_bio(
            "caps",
            {
                "name": "n" * 500,
                "other_positions": ["p" * 50] + [f"P{i}" for i in range(30)] + ["", "  "],
                "on_loan": 1,
            },
        )
        assert len(out["name"]) == 80, len(out["name"])
        assert len(out["other_positions"]) == 12, out["other_positions"]
        assert len(out["other_positions"][0]) == 24, out["other_positions"][0]
        assert out["on_loan"] is True, out  # bool, not the 1 that was sent

    print("save_bio: 12 checks passed")


if __name__ == "__main__":
    main()
