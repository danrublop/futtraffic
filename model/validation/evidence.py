"""Reports carry evidence handles, findings are typed records, and lanes stay in their lane.

WHY - THREE ENTRY POINTS FROM harness-design (not published) §2 AND §F
    RELAY. A claim moved from one agent to another without being re-checked. On
    2026-09-12 every lane report was verified against git before being relayed,
    but only because the supervisor remembered to; and the N10 identity was
    relayed into the shared map unchecked anyway. An agent's report is a claim,
    not evidence. So a report carries HANDLES - a command and what it must print -
    and `verify()` re-runs them. Prose summarises; handles prove.

    TRIAGE. Findings arrived as ranked prose and were triaged by hand. A finding
    is now a typed record, and a CONFIRMED one must carry evidence - a reviewer
    distinguishing "I ran it" from "it reads wrong" was the most useful habit of
    the day, and this makes it a rule instead of a habit.

    CONCURRENCY. A supervisor ran `git stash` in a tree with two live agents.
    Nothing was lost and nothing prevented it. `ownership_violations()` refuses a
    commit that stages files outside its lane, and refuses one that stages across
    lanes - which is the signature `git add -A` leaves in a shared tree.

    .venv/bin/python model/validation/evidence.py --verify handles.json
    .venv/bin/python model/validation/evidence.py --demo
"""
import fnmatch
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

VERDICTS = {"CONFIRMED", "SUSPECTED"}
SEVERITIES = {"BLOCKER", "SHOULD-FIX", "NOTE"}


def _run(cmd: str, cwd=ROOT) -> tuple[int, str]:
    # A shell is required: handles are the commands reviewers already write.
    # They are run from a report the harness itself received, never from
    # untrusted input, which is the same trust verify_claims places in @M.
    p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout


def handle(cmd: str, expect: list[str] | None = None, cwd=ROOT) -> dict:
    """Run a command once and record what a second run must reproduce.

    With `expect`, the handle pins those tokens - robust to a producer printing
    timings. Without it, the whole stdout is hashed, which is strict and is only
    right for a command with no clock in its output.
    """
    code, out = _run(cmd, cwd)
    h = {"cmd": cmd, "exit": code}
    if expect:
        missing = [e for e in expect if e not in out]
        if missing:
            raise ValueError(f"cannot mint a handle: {missing} not in the output of {cmd!r}")
        h["expect"] = list(expect)
    else:
        # An empty stdout hashes to a constant, so a command that never ran at
        # all "reproduces" perfectly on every re-run. The first version of this
        # demo proved exactly that against itself: a mis-quoted path failed twice,
        # identically, and verified as REPRODUCED.
        if not out.strip():
            raise ValueError(f"cannot mint a hash handle from empty output: {cmd!r} "
                             f"(exit {code}) - there is nothing to vouch for")
        h["out_sha"] = hashlib.sha256(out.encode()).hexdigest()[:16]
    return h


def verify(h: dict, cwd=ROOT) -> tuple[str, str]:
    """('REPRODUCED' | 'DIFFERS', why). The re-run, not the report, decides."""
    code, out = _run(h["cmd"], cwd)
    if code != h["exit"]:
        return "DIFFERS", f"exit {code}, handle recorded {h['exit']}"
    if "expect" in h:
        missing = [e for e in h["expect"] if e not in out]
        return ("DIFFERS", f"no longer prints {missing}") if missing else ("REPRODUCED", "")
    now = hashlib.sha256(out.encode()).hexdigest()[:16]
    return ("REPRODUCED", "") if now == h["out_sha"] else ("DIFFERS", "output changed")


def validate_finding(f: dict) -> list[str]:
    """Problems with a finding record; [] if it may be triaged."""
    bad = [f"missing {k}" for k in ("claim", "verdict", "severity", "owner") if not f.get(k)]
    if f.get("verdict") and f["verdict"] not in VERDICTS:
        bad.append(f"verdict {f['verdict']!r} is not one of {sorted(VERDICTS)}")
    if f.get("severity") and f["severity"] not in SEVERITIES:
        bad.append(f"severity {f['severity']!r} is not one of {sorted(SEVERITIES)}")
    # The rule that matters: CONFIRMED means someone ran it. Say what.
    if f.get("verdict") == "CONFIRMED" and not (f.get("handle") or f.get("location")):
        bad.append("CONFIRMED without a handle or file:line - that is a SUSPECTED finding")
    return bad


def ownership_violations(staged: list[str], manifest: dict, lane: str | None) -> list[str]:
    """Refuse staged paths outside `lane`, and any commit that spans lanes."""
    def owner_of(path):
        # A docs page keeps its owner when it moves into a subfolder:
        # `docs/n12-*.md` also matches the same page one folder down.
        flat = "docs/" + path.rsplit("/", 1)[-1] if path.startswith("docs/") else path
        return {name for name, globs in manifest.items()
                if any(fnmatch.fnmatch(path, g) or fnmatch.fnmatch(flat, g) for g in globs)}
    bad = []
    if lane is not None:
        if lane not in manifest:
            return [f"lane {lane!r} is not in the manifest"]
        for p in staged:
            if lane not in owner_of(p):
                bad.append(f"{p} is not owned by {lane}")
    touched = set().union(*[owner_of(p) for p in staged]) if staged else set()
    if len(touched) > 1:
        bad.append(f"one commit stages files owned by {sorted(touched)} - "
                   f"the signature of `git add -A` in a shared tree")
    return bad


def demo():
    # Quoted: this repo lives under "fut traffic", and an unquoted interpreter
    # path splits in a shell. That is how the first run of this demo failed.
    py = f'"{sys.executable}"'
    # 1. a handle reproduces, and a tampered one does not
    h = handle(f'{py} -c "print(42)"')
    assert h["exit"] == 0, "the command must actually have run, or nothing was reproduced"
    assert verify(h)[0] == "REPRODUCED"
    assert verify(dict(h, out_sha="0" * 16))[0] == "DIFFERS"
    assert verify(dict(h, exit=1))[0] == "DIFFERS", "a changed exit code is a changed result"

    # 1b. THE BUG THIS DEMO FIRST HAD: a command that prints nothing cannot vouch.
    try:
        handle(f'{py} -c "pass"')
        raise AssertionError("an empty output must not mint a hash handle")
    except ValueError:
        pass

    # 2. expected tokens survive a clock in the output; a wrong token does not
    clocked = f'{py} -c "import time; print(\'hota 0.3711\', time.time())"'
    h2 = handle(clocked, expect=["0.3711"])
    assert verify(h2)[0] == "REPRODUCED", "timings must not break a token handle"
    assert verify(dict(h2, expect=["0.2356"]))[0] == "DIFFERS"
    try:
        handle(clocked, expect=["9.9999"])
        raise AssertionError("a handle must refuse to vouch for a token it never saw")
    except ValueError:
        pass

    # 3. typed findings: CONFIRMED needs evidence, SUSPECTED does not
    base = {"claim": "the gate passes an empty file", "severity": "BLOCKER", "owner": "L1"}
    assert validate_finding(dict(base, verdict="SUSPECTED")) == []
    assert validate_finding(dict(base, verdict="CONFIRMED")) != []
    assert validate_finding(dict(base, verdict="CONFIRMED", location="model/engine/quality.py:299")) == []
    assert validate_finding(dict(base, verdict="PROBABLY")) != []
    assert validate_finding(dict(base, verdict="SUSPECTED", severity="URGENT")) != []

    # 4. ownership: in-lane passes, out-of-lane and cross-lane are refused
    manifest = {"L1": ["model/validation/calib_*.py", "docs/n12-*.md"],
                "L2": ["model/validation/dist_eval.py", "docs/n8-n11-*.md"]}
    assert ownership_violations(["model/validation/calib_product.py"], manifest, "L1") == []
    assert ownership_violations(["model/validation/dist_eval.py"], manifest, "L1") != []
    moved = "/".join(("docs", "experiments", "n12-x.md"))   # spelt apart: doclinks scans this file
    assert ownership_violations([moved], manifest, "L1") == [], \
        "a docs page lost its lane by moving into a subfolder"
    spans = ownership_violations(["model/validation/calib_product.py", "model/validation/dist_eval.py"],
                                 manifest, None)
    assert spans and "git add -A" in spans[0], spans
    assert ownership_violations(["x.py"], manifest, "L9") == ["lane 'L9' is not in the manifest"]

    # 5. the hook entry point: no manifest means no refusal, ever; a manifest arms it
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        absent = Path(t) / "none.json"
        assert check_staged_ownership(absent, "L1", ["anything.py"]) == []
        live = Path(t) / "lanes.json"
        live.write_text(json.dumps(manifest))
        assert check_staged_ownership(live, "L1", ["model/validation/calib_product.py"]) == []
        assert check_staged_ownership(live, "L1", ["model/validation/dist_eval.py"]) != []
    print("evidence.py ok")


LANES = ROOT / ".lanes.json"


def check_staged_ownership(manifest_path: Path = LANES, lane=None, staged=None) -> list[str]:
    """What the pre-commit hook runs. With no manifest there is no multi-lane work
    in progress, and an ordinary commit is never refused."""
    import os
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text())
    if staged is None:
        # THE ONE GIT CALL IN THIS HARNESS THAT MUST INHERIT ITS ENVIRONMENT,
        # for the opposite of verify_claims.git_env's reason. This runs from
        # pre-commit, and for `git commit <paths>` or `--amend` git builds the
        # intended commit in a TEMPORARY index and names it in GIT_INDEX_FILE.
        # Stripping it would read .git/index instead - a different set of files
        # than the one being committed, which is the check not running.
        # The scanners strip GIT_*; this one is being TOLD which index to read.
        staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT,
                                capture_output=True, text=True).stdout.split()
    return ownership_violations(staged, manifest, lane if lane is not None else os.environ.get("LANE"))


def main() -> int:
    if "--demo" in sys.argv:
        demo()
        return 0
    if "--ownership" in sys.argv:
        bad = check_staged_ownership()
        for b in bad:
            print(f"commit refused: {b}")
        if bad:
            print("Stage only the paths your lane owns (.lanes.json), and set LANE=<name>.")
        return 1 if bad else 0
    if "--verify" in sys.argv:
        handles = json.loads(Path(sys.argv[sys.argv.index("--verify") + 1]).read_text())
        bad = 0
        for h in handles:
            status, why = verify(h)
            bad += status != "REPRODUCED"
            print(f"{status:11s} {h['cmd'][:90]}" + (f"\n            {why}" if why else ""))
        return 1 if bad else 0
    print(__doc__.strip().splitlines()[-2].strip())
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
