"""A ratchet on bare sigils: a claim dressed as evidence may not be added, only removed.

WHY
    `verify_claims.py` counts them itself - "3029 BARE markers ... carry no
    `value :: command` and verify nothing - prefix decoration, not claims" - and
    then does nothing about them. A bare `@M` reads to a reader as *measured,
    re-runnable* and has nothing behind it; 632 of the 3,029 are exactly that.
    A claim wearing the costume of evidence is worse than an unlabelled one,
    because the costume is what earns the trust. harness-design (not published) §1.1.

WHY A RATCHET AND NOT A REWRITE
    3,029 markers across dozens of documents cannot be converted to real tags
    mechanically - each one needs an evidence handle only its author knows - and
    stripping them blind would delete an author's signal along with the
    decoration. So the rule is the one `lint_truth` already lives by: the count
    per file may go DOWN and may not go UP. New decoration is refused today; the
    existing stock is paid down one honest conversion at a time.

WHAT IT REFUSES TO BE
    A second opinion on what a bare marker is. The regexes are imported from
    verify_claims, never copied, and `demo()` asserts this file's total equals
    verify_claims' own census - so the two tools cannot quietly disagree about
    the number this whole argument rests on.

AND A SECOND RATCHET, ON `#example`
    A tag marked `#example` is a specimen: verify_claims never runs it and
    derived.py never evaluates it. That makes the marker an off switch for both
    checkers, one word long. verify_claims.py's own specimens are the allowlist;
    everywhere else the count may not rise past EXAMPLE_BASELINE.

    .venv/bin/python model/validation/bare_markers.py                 # fail on any increase
    .venv/bin/python model/validation/bare_markers.py --rebaseline    # accept a DECREASE only
    .venv/bin/python model/validation/bare_markers.py --demo
"""
import collections
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model" / "validation"))
import verify_claims as V  # noqa: E402 - MARKER, TAG and TEXT, reused and never copied

BASELINE = ROOT / "model" / "validation" / "bare_markers.baseline.json"
#: `#example` tags outside verify_claims.py on 2026-09-29: one in
#: audit-claims (not published), one in derived.py. Lower it when one goes.
EXAMPLE_BASELINE = 2


def examples(claims=None) -> list[str]:
    """Every `#example` tag outside verify_claims.py (its specimens are the allowlist)."""
    claims = V.scan() if claims is None else claims
    return [c.key for c in claims if c.example and not c.mine]


def example_rise(keys: list[str], baseline: int = EXAMPLE_BASELINE) -> str | None:
    if len(keys) <= baseline:
        return None
    return (f"{len(keys)} #example tags outside {V.ME}, baseline {baseline}: a specimen is never "
            f"run or evaluated, so the marker switches both checkers off - " + ", ".join(keys))


def count_bare(text: str) -> int:
    """Sigils on a line, minus the one a well-formed tag consumes.

    The same rule verify_claims.scan's census applies, line for line - and demo()
    asserts the totals match rather than trusting that this sentence is true.
    """
    n = 0
    for line in text.splitlines():
        n += len(V.MARKER.findall(line)) - (1 if V.TAG.search(line) else 0)
    return n


def _files():
    def ls(*a):
        # env=V.git_env(): GIT_DIR beats cwd=, and a gate run from inside a
        # hook would otherwise list the wrong tree's index. Imported, never
        # copied - the same rule this file follows for MARKER and TAG.
        return subprocess.run(["git", "ls-files", *a], cwd=ROOT, env=V.git_env(),
                              capture_output=True, text=True).stdout.splitlines()
    for rel in ls() + ls("--others", "--exclude-standard"):
        p = ROOT / rel
        if p.suffix in V.TEXT and p.is_file():
            yield rel, p


def counts(files=None) -> dict:
    out = {}
    for rel, path in (files if files is not None else _files()):
        try:
            c = count_bare(path.read_text(errors="replace"))
        except OSError:
            continue
        if c > 0:
            out[rel] = c
    return out


def check(current: dict, baseline: dict) -> list[str]:
    """Every file whose bare-marker count rose. A new file's baseline is zero."""
    return [f"{f}: {n} bare marker(s), baseline {baseline.get(f, 0)} (+{n - baseline.get(f, 0)})"
            for f, n in sorted(current.items()) if n > baseline.get(f, 0)]


def rebaseline(current: dict, path: Path = BASELINE) -> list[str]:
    """Write the new baseline - refused if it would RAISE any file's count."""
    if path.exists():
        rises = check(current, json.loads(path.read_text()))
        if rises:
            return rises
    path.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
    return []


def demo():
    import tempfile

    # 1. what counts as bare
    # Specimens are ASSEMBLED, never spelled contiguously. Written out, these are
    # real-looking tags in this file's own source, and verify_claims --all tried
    # to run `x.py` and `cmd` as claims - the same bug derived.py's specimens had,
    # repeated here, and caught by the pre-push gate this file is part of.
    at = "@"
    assert count_bare(at + "M 0.5 :: .venv/bin/python x.py") == 0, "a real tag is not bare"
    assert count_bare("`@M` the headroom was quoted as 0.52") == 1
    assert count_bare("@M and @C together") == 2
    assert count_bare(at + "M 0.5 :: cmd   and a stray " + at + "V") == 1, \
        "a tag consumes one sigil, not all"

    # 2. the ratchet refuses a rise and a new file, and allows a fall
    assert check({"a.md": 3}, {"a.md": 2}) == ["a.md: 3 bare marker(s), baseline 2 (+1)"]
    assert check({"new.md": 1}, {}) != [], "a new file's baseline is zero"
    assert check({"a.md": 1}, {"a.md": 5}) == [], "paying the stock down is always allowed"

    # 3. rebaseline is a ratchet too - it cannot be used to launder an increase
    with tempfile.TemporaryDirectory() as t:
        base = Path(t) / "b.json"
        assert rebaseline({"a.md": 4}, base) == []           # first write
        assert rebaseline({"a.md": 2}, base) == []           # a decrease is accepted
        assert rebaseline({"a.md": 9}, base) != []           # an increase is refused
        assert json.loads(base.read_text()) == {"a.md": 2}, "a refused rebaseline wrote nothing"

    # 3b. the #example ratchet: at baseline passes, one more fails, and the
    #     checker's own specimens are not counted
    spec = V.Claim("M", "1", "x  #example", "x/a.md", 1, [])
    own = V.Claim("M", "1", "x  #example", V.ME, 1, [])
    assert examples([spec, own]) == ["x/a.md:1"], "the allowlist is wrong"
    assert example_rise(["x/a.md:1"], baseline=1) is None
    assert example_rise(["x/a.md:1", "x/b.md:2"], baseline=1), \
        "a new #example tag passed - one word switches a claim out of both checkers"

    # 4. RELATIONAL: this file and verify_claims agree on the number the argument
    #    rests on. If either tool's definition of "bare" drifts, this fires.
    census = collections.Counter()
    V.scan(census)
    assert sum(counts().values()) == sum(census.values()), \
        (sum(counts().values()), sum(census.values()))
    print(f"bare_markers.py ok  ({sum(census.values())} bare markers, agreeing with verify_claims)")


def main() -> int:
    if "--demo" in sys.argv:
        demo()
        return 0
    current = counts()
    if "--rebaseline" in sys.argv:
        refused = rebaseline(current)
        for r in refused:
            print(f"REFUSED  {r}")
        print(f"{'refused' if refused else 'wrote'} {BASELINE.relative_to(ROOT)} "
              f"({sum(current.values())} bare markers in {len(current)} files)")
        return 1 if refused else 0
    if not BASELINE.exists():
        print(f"no baseline at {BASELINE.relative_to(ROOT)} - run --rebaseline once to create it")
        return 1
    rises = check(current, json.loads(BASELINE.read_text()))
    for r in rises:
        print(f"RISE  {r}")
    ex = examples()
    if (why := example_rise(ex)):
        print(f"RISE  {why}")
    print(f"\n{len(rises)} file(s) added bare markers; {sum(current.values())} in the repo; "
          f"{len(ex)} #example tag(s) outside {V.ME} (baseline {EXAMPLE_BASELINE})")
    return 1 if (rises or why) else 0


if __name__ == "__main__":
    raise SystemExit(main())
