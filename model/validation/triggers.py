"""Which `check.py` sections a change reaches, derived from the code - not typed in.

WHY
    `.githooks/pre-push` decided which `check.py <section> --fast` to run with six
    hand-written regexes, one per section. A regex is a copy of the import graph,
    and it went stale the way copies do: `pitch.py`, `rolenet.py`, `paths.py`,
    `assoc_eval.py` are imported by sections whose regex never named them, so a
    push changing them ran none of the sections they feed. "One fact, several
    places, updated in some of them" (harness-design (not published) §2), in the
    gate itself.

HOW
    For each entry in `check.SECTIONS`, read by `ast` (check.py is never imported):
      1. the top-level functions and globals the entry reaches by name;
      2. every module those import (lazily or not), every repo path they build
         as `ROOT / "a" / "b"`, and every string literal that is exactly a repo
         path (`SELFCHECKS = ("model/engine/pitch.py", ...)`);
      3. then the import closure of every repo module reached, over check.py's
         own search path, following string literals naming a sibling .py
         (`"coreml_worker.py"`, launched as a subprocess) and ROOT-built paths.
    A fixture directory reached in step 2 triggers on any file under it.
    ALWAYS (check.py, this file, requirements) triggers every section.

    Conservative on purpose: a lazy import anywhere in a reached module counts.
    Running a section that did not need it costs seconds; skipping one that did
    is the hole this closes.

    .venv/bin/python model/validation/triggers.py --triggers calib     # its trigger set
    git diff --name-only A B | .venv/bin/python model/validation/triggers.py --sections
    .venv/bin/python model/validation/triggers.py --selfcheck
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "model" / "validation" / "check.py"
#: check.py's own sys.path, in its order, plus its own directory.
SEARCH = ("model/engine", "app/server", "model/validation")
ALWAYS = ("model/validation/check.py", "model/validation/triggers.py", "app/server/requirements.txt")


def _docstrings(tree) -> set[int]:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body \
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant):
            out.add(id(n.body[0].value))
    return out


ROOT_NAMES = {"ROOT", "REPO"}   # check.py's and model/engine/paths.py's name for the repo root


def _chain(node, env) -> str | None:
    """`ROOT / "a" / "b"` (or a name already resolved to a path) -> "a/b"."""
    parts = []
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        if not (isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)):
            return None
        parts.insert(0, node.right.value)
        node = node.left
    if not parts or not isinstance(node, ast.Name):
        return None
    base = "" if node.id in ROOT_NAMES else env.get(node.id)
    if base is None:
        return None
    return "/".join(p for x in [base, *parts] for p in x.split("/") if p)


def _module_file(name: str) -> str | None:
    head = name.split(".")[0]
    for d in SEARCH:
        if (ROOT / d / f"{head}.py").is_file():
            return f"{d}/{head}.py"
    return None


#: Written by a gate on every run and read by nobody a section depends on.
NOT_INPUTS = {"model/validation/claims.log"}


def _refs(nodes, env, here: str, docs: set[int], runs_paths: bool = False) -> tuple[set[str], set[str]]:
    """(code, data) named in `nodes`.

    code  - modules imported, a sibling `.py` named by basename (a script
            launched by path, "coreml_worker.py"), and - in check.py only,
            `runs_paths` - a repo `.py` literal, which there is a SELFCHECKS
            entry it executes: followed for their imports. Elsewhere a repo
            `.py` literal is a file READ (shipped_config.CONSTANTS): data.
    data  - repo files built as `ROOT / ...` or written as a path literal: a
            trigger, NOT followed. shipped_config READS jobs.py for its call
            site; jobs.py's own imports are not what that reading depends on.
    """
    code, data = set(), set()
    # a string inside `ROOT / "a" / "b.py"` is that path's piece, not a literal
    pieces = {id(n.right) for top in nodes for n in ast.walk(top)
              if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)}
    for top in nodes:
        for n in ast.walk(top):
            if isinstance(n, ast.Import):
                code.update(f for a in n.names if (f := _module_file(a.name)))
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                code.update(f for f in [_module_file(n.module)] if f)
            elif isinstance(n, ast.BinOp) and (p := _chain(n, env)):
                if (ROOT / p).is_file() or _fixture_dir(p):
                    data.add(p)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) \
                    and id(n) not in docs and id(n) not in pieces:
                s = n.value.strip()
                if " " in s:
                    continue
                sib = f"{Path(here).parent.as_posix()}/{s}"
                if s.endswith(".py") and "/" not in s and (ROOT / sib).is_file():
                    code.add(sib)
                elif "/" in s and s.split("/")[0] in ("model", "app") and (ROOT / s).is_file():
                    (code if runs_paths and s.endswith(".py") else data).add(s)
    return code, data - NOT_INPUTS


def _fixture_dir(p: str) -> bool:
    """model/validation/fixtures/<one section's dir> - never the fixtures root."""
    parts = p.strip("/").split("/")
    return len(parts) == 4 and parts[:3] == ["model", "validation", "fixtures"] and (ROOT / p).is_dir()


def _parse(f: str):
    try:
        return ast.parse((ROOT / f).read_text())
    except SyntaxError as e:
        # fail CLOSED: an unparseable module could import anything
        raise SystemExit(f"triggers: {f} does not parse ({e.msg}) - cannot derive triggers")


def _closure(code: set[str], data: set[str]) -> set[str]:
    """Every code file's imports, transitively; data files ride along as leaves."""
    seen, leaves, todo = set(), set(data), list(code)
    while todo:
        f = todo.pop()
        if f in seen or not (ROOT / f).is_file():
            continue
        seen.add(f)
        tree = _parse(f)
        c, d = _refs([tree], _globals(tree), f, _docstrings(tree))
        todo.extend(c - seen)
        leaves |= d
    return seen | leaves


def _globals(tree) -> dict[str, str]:
    """Top-level NAME = ROOT / ... paths, including names imported from a repo
    module that defines them (`from paths import VALIDATION`)."""
    env = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and not node.level \
                and (f := _module_file(node.module)) and f.startswith("model/engine/paths"):
            theirs = _globals(_parse(f))
            env.update({a.asname or a.name: theirs[a.name] for a in node.names if a.name in theirs})
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            tuples = [t for t in node.targets if isinstance(t, ast.Tuple)]
            if names and (p := _chain(node.value, env)):
                env.update({n: p for n in names})
            for t in tuples:
                if isinstance(node.value, ast.Tuple):
                    for tn, tv in zip(t.elts, node.value.elts):
                        if isinstance(tn, ast.Name) and (p := _chain(tv, env)):
                            env[tn.id] = p
    return env


def sections(check: Path = CHECK) -> dict[str, set[str]]:
    """{section: trigger set} - files, and directories (ending "/") whose contents trigger."""
    tree = ast.parse(check.read_text())
    env = _globals(tree)
    docs = _docstrings(tree)
    funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assigns = {}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                for nm in ([t] if isinstance(t, ast.Name) else getattr(t, "elts", [])):
                    if isinstance(nm, ast.Name):
                        assigns.setdefault(nm.id, []).append(n)
    table = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", "") == "SECTIONS" for t in n.targets))
    here = check.relative_to(ROOT).as_posix()
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    out = {}
    for k, v in zip(table.keys, table.values):
        reached, todo = set(), [v.id]
        nodes = list(top)
        while todo:
            name = todo.pop()
            if name in reached:
                continue
            reached.add(name)
            body = [funcs[name]] if name in funcs else assigns.get(name, [])
            nodes.extend(body)
            for b in body:
                todo.extend(n.id for n in ast.walk(b) if isinstance(n, ast.Name)
                            and (n.id in funcs or n.id in assigns) and n.id not in reached)
        code, data = _refs(nodes, env, here, docs, runs_paths=True)
        # only a FIXTURE directory triggers on its contents; `ROOT / "model"`
        # would otherwise make every file in the repo a trigger
        dirs = {p.rstrip("/") + "/" for p in data if _fixture_dir(p)}
        data = {p for p in data if (ROOT / p).is_file()}
        # a fixture FILE's directory is the fixture: a new file beside it counts
        dirs |= {Path(p).parent.as_posix() + "/" for p in data if _fixture_dir(Path(p).parent.as_posix())}
        # check.py is a trigger for every section but NOT a closure seed: its
        # other sections' imports are not this section's.
        out[k.value] = _closure(code, data) | set(ALWAYS) | dirs
    return out


def hit(changed: list[str], triggers: set[str]) -> bool:
    return any(c in triggers or any(c.startswith(d) for d in triggers if d.endswith("/"))
               for c in changed)


def selfcheck():
    s = sections()
    assert set(s) == {"calib", "track", "detect", "identity", "measure", "ball"}, sorted(s)
    # every file the old hand regexes named is still reached - nothing narrowed
    old = {
        "calib": ["model/engine/autocal.py", "model/engine/calibrators.py", "model/engine/pitch.py",
                  "model/engine/sceneprior.py", "app/server/jobs.py", "model/validation/fixtures/calib/golden.json"],
        "track": ["model/engine/track.py", "model/engine/tracking.py", "model/engine/stitch.py",
                  "model/validation/fixtures/track/golden.json"],
        "detect": ["model/engine/detectors.py", "model/engine/coreml_worker.py", "app/server/requirements.txt",
                   "model/validation/fixtures/detect/golden.json"],
        "identity": ["model/engine/identity.py", "model/engine/roster.py", "model/engine/track.py",
                     "app/server/jobs.py", "model/validation/fixtures/identity/x", "model/validation/fixtures/track/x"],
        "measure": ["model/engine/stats.py", "model/engine/frames.py", "model/engine/roles.py",
                    "model/engine/quality.py", "model/engine/test_golden.py", "app/server/jobs.py",
                    "model/validation/golden.json", "model/validation/fixtures/measure/x"],
        "ball": ["model/engine/ball.py", "model/engine/track.py", "model/engine/detectors.py",
                 "model/validation/fixtures/ball/x"],
    }
    missed = {k: [f for f in fs if not hit([f], s[k])] for k, fs in old.items()}
    missed = {k: v for k, v in missed.items() if v}
    # the old regexes are the FLOOR, not the spec: a miss is reported, and only
    # golden.json (a data file read by path in a helper) is allowed to need ALWAYS
    assert not missed, f"derived triggers dropped files the hand regexes had: {missed}"
    # the files the review named reach the sections that import them
    for f, sec in (("model/engine/pitch.py", "calib"), ("model/engine/paths.py", "measure")):
        if (ROOT / f).is_file():
            assert hit([f], s[sec]), f"{f} does not trigger {sec}"
    assert all(hit(["model/validation/check.py"], t) for t in s.values()), "check.py must trigger all"
    assert not any(hit(["docs/README.md"], t) for t in s.values()), "a doc triggered a section"
    # a planted module: imported by one section's helper, reached transitively
    import tempfile
    with tempfile.TemporaryDirectory(dir=ROOT / "model" / "validation") as t:
        rel = Path(t).relative_to(ROOT).as_posix()
        (Path(t) / "check.py").write_text(
            "from pathlib import Path\nROOT = Path(__file__).resolve().parents[2]\n"
            "def a(argv):\n    return b()\n"
            "def b():\n    import stats\n    return 0\n"
            "def c(argv):\n    return 1\n"
            "SECTIONS = {'one': a, 'two': c}\n")
        got = sections(Path(t) / "check.py")
        assert "model/engine/stats.py" in got["one"], "a lazy import two calls deep was not followed"
        assert "model/engine/stats.py" not in got["two"], "a section reached a module it never calls"
        # and stats' own imports come with it
        if (ROOT / "model/engine/frames.py").is_file() and "frames" in (ROOT / "model/engine/stats.py").read_text():
            assert "model/engine/frames.py" in got["one"], "the closure stopped at the first module"
    print("triggers.py self-check OK (" + ", ".join(f"{k} {len(v)}" for k, v in s.items()) + " paths)")


def main(argv) -> int:
    if "--selfcheck" in argv:
        selfcheck()
        return 0
    if "--triggers" in argv:
        for p in sorted(sections()[argv[argv.index("--triggers") + 1]]):
            print(p)
        return 0
    if "--sections" in argv:
        changed = [ln.strip() for ln in sys.stdin if ln.strip()]
        print(" ".join(k for k, t in sections().items() if hit(changed, t)) or "NONE")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
