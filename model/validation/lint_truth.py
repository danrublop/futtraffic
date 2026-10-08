"""Eight mechanical checks for failure modes that had no mechanism.

Each one exists because it would have caught something real on 2026-09-08/09.
None of them is clever; that is the point. A check nobody runs is worth nothing,
so all eight are pure stdlib + ast and the whole file runs in under a second.

  1 DEFAULT TRAP     a harness default that contradicts the shipped constant
  2 SELF-CHECK ECHO  a self-check that recomputes the thing it is checking
  3 SPLIT VALUES     one metric, several live values, no scope saying which
  4 ODOMETER         how many times a variant has been scored (from the ledger)
  5 FLAG DRIFT       a CLI default that disagrees with the function it feeds
  6 INERT REBIND     a constant used as a function default, reassigned later
  7 SELF-CHECK DRIFT a self-check passing a value the default does not use
  8 TRACKED SYMLINK  a pointer to one machine, committed as source (blocks without --strict)

WHAT EACH ONE WOULD HAVE CAUGHT

  1. `assoc_eval.evaluate()` defaults `tracker_name="bytetrack"`. The shipped
     tracker is `tracking.BEST = "cbiou"`. Every figure in the input ledger - the
     document defining what everything else is conditioned on - was the fallback:
     IDF1 0.155 not 0.212, tracks 796 not 294, far band 0.148 not 0.054. And
     `nvidia.py`'s `_cuda`/`_tensorrt` defaulted `overlap=0.25` under a docstring
     reading "Same tiling as detectors._rfdetr", which ships 0.10. Two detector
     paths, two tilings, one comment claiming they matched.

  2. `st2_align.demo` asserted the ball's y orientation and carried its OWN copy
     of `by + 34.0`. Mutating the real conversion left the check green. A check
     that recomputes its subject is a second implementation agreeing with itself.
     Extracting `ball_to_pitch` was the fix - the assertion was already there.

  3. Four live answers to "our recall" across four files (0.878 / 0.652 / 0.628 /
     0.624), none carrying which tracker, clip or metric it belongs to. Also five
     values for one CoreML speed-up ratio, and 0.3548 vs 0.3522 for the stitch
     baseline after `_vel` was clamped.

  4. A configuration was scored ~24 times against the test pair and the best
     reported. Nothing recorded that. `status.py` is the only writer of the
     ledger, so counting its rows per variant IS the odometer - exact, not
     estimated.

WHY 5 AND 6 EXIST — this lint printed "(none)" while the bug was live

  `@M` 2026-09-10. `model/validation/st2_metres.py` shipped `tps_field(..., 
  smoothing=100.0)` while every documented result used 10000.0, so `--tps` made
  the product WORSE (overall median 1.36 -> 2.11 m) and no command line could
  reproduce the numbers in the docs. Check 1 did not see it, and could not:
  `defaults()` only fires for parameter names listed in SHIPPED, and nobody had
  added `smoothing`. **A registry-based check only catches the traps someone
  already thought of.** Check 5 needs no registry: it compares an argparse
  `default=` against the default of the function that consumes the same name,
  which is the shape of the bug without needing to know the knob exists.

  The FIRST attempt at fixing it was the same bug again: `TPS_SMOOTHING =
  a.tps_smooth` inside `if __name__ == "__main__"`. Python evaluates defaults at
  `def` time, so rebinding the module constant afterwards cannot change them -
  `--tps-smooth 100` and `--tps-smooth 10000` produced identical output. Check 6
  catches that: a name used as a function default that is assigned more than
  once means someone believes the reassignment does something. It does not.

`ponytail: heuristics, deliberately. 3 will have false positives; a scope note
silences one and takes ten seconds to write, which is the behaviour we want.`
"""
import argparse
import ast
import collections
import contextlib
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Check 1. The shipped value for a knob, and every alias a signature might use.
# Adding a row here is how you say "this decision has one answer".
SHIPPED = {
    # 0.10 from 7c2cd35 (2026-08-19) until it was scored at the SHIPPED conf
    # rather than at 0.2 and lost the far two bands - tile-overlap (not published).
    "overlap":     (0.25,      ("overlap", "overlap_ratio")),
    # "best" is an alias resolved by tracking.best_available() -> cbiou, so it is
    # not a trap. A literal fallback name in a default IS.
    "tracker":     ("cbiou",   ("tracker", "tracker_name")),
}

# Check 2. If a self-check contains these AND the module does, it may be
# recomputing rather than checking.
ARITH = ("np.hypot", "np.linalg.norm", "math.sqrt", "** 0.5", "np.sqrt")

# Check 3. Metrics whose value should not silently differ between files.
METRIC = re.compile(
    r"\b(HOTA|DetA|AssA|LocA|IDF1|recall|precision|mAP@50|AP_S)\b[\s:=]*([01]\.\d{3})",
    re.I)
SCOPED = re.compile(r"(cbiou|bytetrack|conf|remap|panorama|clip|tracker|held|tuned|"
                    r"post-track|detector|IoU|shipped|fallback)", re.I)


def defaults(paths):
    """Function defaults that contradict SHIPPED."""
    out = []
    for p in paths:
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError:
            continue
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            args = fn.args.args + fn.args.kwonlyargs
            defs = ([None] * (len(fn.args.args) - len(fn.args.defaults))
                    + list(fn.args.defaults) + list(fn.args.kw_defaults))
            for a, d in zip(args, defs):
                if not isinstance(d, ast.Constant):
                    continue
                # `None` is not a contradiction, it is the sentinel for "take
                # whatever ships" - the one spelling that CANNOT go stale.
                if d.value is None:
                    continue
                for want, (val, names) in SHIPPED.items():
                    if a.arg in names and d.value != val and d.value != "best":
                        out.append((p, fn.lineno, fn.name, a.arg, d.value, val))
    return out


def selfcheck_drift(paths):
    """A self-check that passes a literal where the default is something else.

    `@M` 2026-09-10, the third default trap of the week and the one checks 1 and
    5 both missed. `model/validation/calib_hypothesis.py` defaulted `smoothing` to 1.0
    in the CLI **and** in both function signatures - so FLAG DRIFT saw no
    disagreement, because they agreed with each other and disagreed with reality.
    `selfcheck()` passed `smoothing=10000.0`, and at 1.0 not one of the file's
    published numbers reproduced.

    **A self-check that exercises a configuration nobody runs is not a check.**
    Whichever value is right, the two must agree, and that is decidable by
    reading the file.
    """
    # A self-check SHOULD shrink the work: fewer frames, fewer samples, a smaller
    # n. That is not drift, it is a fast test, and flagging it is how a lint gets
    # turned off. Only parameters whose value changes the ANSWER are interesting.
    SIZE = ("n", "samples", "frames", "n_frames", "limit", "count", "iters",
            "rounds", "every", "first", "max_frames", "steps", "seed", "k",
            "n_seeds", "stride", "width", "height", "size", "workers", "batch",
            "trials", "y", "x")
    out = []
    for p in paths:
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError:
            continue
        defaults = {}
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            args = fn.args.args + fn.args.kwonlyargs
            defs = ([None] * (len(fn.args.args) - len(fn.args.defaults))
                    + list(fn.args.defaults) + list(fn.args.kw_defaults))
            for a, d in zip(args, defs):
                if isinstance(d, ast.Constant) and isinstance(d.value, (int, float)) \
                        and not isinstance(d.value, bool):
                    defaults.setdefault(a.arg, (d.value, fn.name, fn.lineno))
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name in ("demo", "_selfcheck", "selfcheck")]:
            for c in [n for n in ast.walk(fn) if isinstance(n, ast.Call)]:
                for kw in c.keywords:
                    if kw.arg in SIZE:
                        continue
                    if kw.arg in defaults and isinstance(kw.value, ast.Constant) \
                            and isinstance(kw.value.value, (int, float)) \
                            and not isinstance(kw.value.value, bool):
                        val, owner, line = defaults[kw.arg]
                        if kw.value.value != val:
                            out.append((p, c.lineno, fn.name, kw.arg,
                                        kw.value.value, owner, line, val))
    return out


def flag_drift(paths):
    """argparse default vs the default of the function that consumes the name.

    Registry-free on purpose. Check 1 needs SHIPPED to already know a knob
    exists; this one derives the pair from the file itself, which is why it
    would have caught `smoothing` and check 1 could not.
    """
    out = []
    for p in paths:
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError:
            continue
        fn_defaults = {}
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            args = fn.args.args + fn.args.kwonlyargs
            defs = ([None] * (len(fn.args.args) - len(fn.args.defaults))
                    + list(fn.args.defaults) + list(fn.args.kw_defaults))
            for a, d in zip(args, defs):
                if isinstance(d, ast.Constant) and isinstance(d.value, (int, float)):
                    fn_defaults.setdefault(a.arg, (d.value, fn.name, fn.lineno))
        for c in [n for n in ast.walk(tree) if isinstance(n, ast.Call)]:
            f = c.func
            if not (isinstance(f, ast.Attribute) and f.attr == "add_argument"):
                continue
            if not c.args or not isinstance(c.args[0], ast.Constant):
                continue
            name = str(c.args[0].value).lstrip("-").replace("-", "_")
            dflt = next((k.value for k in c.keywords if k.arg == "default"), None)
            if not isinstance(dflt, ast.Constant):
                continue
            # Match on a PREFIX, not equality: the real bug was `--tps-smooth`
            # against `smoothing=`, and exact matching missed it. Minimum four
            # characters, because short names collide across unrelated functions
            # (`-s` matched `jit(s=3.0)` and that was noise).
            if not isinstance(dflt.value, (int, float)):
                continue
            for key in (name, name.split("_")[-1]):
                if len(key) < 4:
                    continue
                hit = next((k for k in fn_defaults
                            if k == key or k.startswith(key) or key.startswith(k)),
                           None)
                if hit:
                    val, fname, fline = fn_defaults[hit]
                    if dflt.value != val:
                        out.append((p, c.lineno, name, dflt.value, fname, fline, val))
                    break
    return out


def inert_rebind(paths):
    """A module constant used as a function default, then reassigned.

    Defaults are evaluated at `def` time. Reassigning the name afterwards - in
    `__main__`, in a setup function, anywhere - cannot change the default, so the
    reassignment is a no-op that reads like a knob.
    """
    out = []
    for p in paths:
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError:
            continue
        used = {}
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            for d in list(fn.args.defaults) + list(fn.args.kw_defaults):
                if isinstance(d, ast.Name):
                    used.setdefault(d.id, (fn.name, fn.lineno))
        assigns = collections.defaultdict(list)
        for n in ast.walk(tree):
            if isinstance(n, (ast.Assign, ast.AugAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        assigns[t.id].append(n.lineno)
        for name, (fname, fline) in used.items():
            if len(assigns.get(name, [])) > 1:
                out.append((p, assigns[name][-1], name, fname, fline,
                            len(assigns[name])))
    return out


def _selfcheck_5_and_6():
    """Prove 5 and 6 catch the two bugs that shipped past this lint on 09-10."""
    import tempfile
    d = pathlib.Path(tempfile.mkdtemp())

    # Bug 1: the function default and the CLI default disagree. This is exactly
    # st2_metres.tps_field(smoothing=100.0) against --tps-smooth default 10000.
    a = d / "drift.py"
    a.write_text("import argparse\n"
                 "def tps_field(seq, smoothing=100.0):\n    return smoothing\n"
                 "ap = argparse.ArgumentParser()\n"
                 "ap.add_argument('--tps-smooth', type=float, default=10000.0)\n")
    hits = flag_drift([a])
    assert hits and hits[0][2] == "tps_smooth" and hits[0][3] == 10000.0, hits

    # Bug 2: a constant used as a default, rebound in __main__. Python captured
    # the default at def time, so the rebind cannot reach it.
    b = d / "inert.py"
    b.write_text("TPS_SMOOTHING = 10000.0\n"
                 "def tps_field(seq, smoothing=TPS_SMOOTHING):\n"
                 "    return smoothing\n"
                 "if __name__ == '__main__':\n"
                 "    TPS_SMOOTHING = 100.0\n")
    hits2 = inert_rebind([b])
    assert hits2 and hits2[0][2] == "TPS_SMOOTHING", hits2

    # And neither may fire on the corrected shape: one constant, one default,
    # threaded through as a parameter rather than rebound.
    c = d / "fixed.py"
    c.write_text("import argparse\n"
                 "TPS_SMOOTHING = 10000.0\n"
                 "def tps_field(seq, smoothing=TPS_SMOOTHING):\n"
                 "    return smoothing\n"
                 "ap = argparse.ArgumentParser()\n"
                 "ap.add_argument('--tps-smooth', type=float, "
                 "default=TPS_SMOOTHING)\n")
    assert not flag_drift([c]), flag_drift([c])
    assert not inert_rebind([c]), inert_rebind([c])
    return "5 catches the 100-vs-10000 flag drift, 6 catches the __main__ rebind"


# Check 8. A tracked symlink. Empty on purpose and it should stay empty: a
# symlink in the index is a pointer to a machine, and git stores the TARGET
# STRING, not the file. Add a path here only with the reason it must be tracked.
ALLOWED_SYMLINKS = ()

# The mode git gives a symlink blob. Hoisted out of the function so mutate.py
# can blind the check to it - a guard nothing can prove still bites is a guard
# on trust, which is the argument mutate.py exists to make.
SYMLINK_MODE = "120000"


def tracked_symlinks(rows=None):
    """Symlinks in the git index. Mode 120000 is the whole test.

    WHAT IT WOULD HAVE CAUGHT, twice: `model/models`,
    `model/validation/out` and `model/validation/role_data` were committed as
    ABSOLUTE-PATH symlinks by `git add -A` in a worktree where they pointed at a
    checkout somewhere else. Each .gitignore entry ended in a slash, and a
    trailing slash matches a DIRECTORY, not a symlink standing at the same path,
    so nothing refused them. A human reviewer found them both times, and one of
    the three pointed at the DFL export that had just been deleted for licence
    reasons - a pointer that outlived the data it names.

    The ignore file now carries both spellings for those three, which fixes
    three paths and not the rule. This is the rule: git stores a symlink as its
    target string, so a tracked one is a fact about one machine's filesystem
    committed as if it were source, and it is unresolvable for everybody else.
    `.venv` is the live instance today - `.venv/` in .gitignore does not match
    the symlink a worktree puts there, so `git add -A` would take it.

    Takes `git ls-files -s` rows so the check itself is testable without a repo.
    """
    if rows is None:
        import subprocess  # noqa: PLC0415 - only this check shells out
        sys.path.insert(0, str(ROOT / "model" / "validation"))
        import verify_claims as V  # noqa: PLC0415 - git_env, reused and never copied
        rows = subprocess.run(["git", "ls-files", "-s"], cwd=ROOT, env=V.git_env(),
                              capture_output=True, text=True).stdout.splitlines()
    out = []
    for row in rows:
        mode, _, rest = row.partition(" ")
        if mode != SYMLINK_MODE:
            continue
        path = rest.split("\t", 1)[-1].strip()
        if path in ALLOWED_SYMLINKS:
            continue
        target = ""
        with contextlib.suppress(OSError):
            target = os.readlink(ROOT / path)
        out.append((path, target))
    return out


def echoes(paths):
    """Self-checks that recompute their subject."""
    out = []
    for p in paths:
        src = p.read_text()
        for name in ("def demo(", "def _selfcheck("):
            i = src.find(name)
            if i < 0:
                continue
            body = src[i:]
            nxt = re.search(r"\ndef (?!demo|_selfcheck)", body)
            body = body[:nxt.start()] if nxt else body
            head = src[:i]
            hits = [a for a in ARITH if a in body and a in head]
            if hits:
                out.append((p, src[:i].count("\n") + 1, name.strip("def ("), hits))
    return out


def split_values(paths):
    """One metric, several live values, no scope qualifier on the line."""
    seen = collections.defaultdict(list)
    for p in paths:
        for n, line in enumerate(p.read_text().splitlines(), 1):
            for m in METRIC.finditer(line):
                if not SCOPED.search(line):
                    seen[m.group(1).upper()].append((float(m.group(2)), p, n, line.strip()[:70]))
    return {k: v for k, v in seen.items() if len({x[0] for x in v}) > 1}


def odometer():
    """How many times each variant has been scored. status.py is the only writer."""
    led = ROOT / "docs/ledgers/status-ledger.md"
    c = collections.Counter()
    if not led.exists():
        return c
    for line in led.read_text().splitlines():
        cells = [x.strip() for x in line.split("|")]
        if len(cells) > 4 and cells[1].startswith("20"):
            c[cells[3]] += 1
    return c


def doc_pages(root=ROOT):
    """Every page under docs/ at any depth except the memory store. Recursive: a
    top-level glob went silently green on every page moved into a subfolder."""
    return sorted(p for p in (root / "docs").rglob("*.md")
                  if p.relative_to(root / "docs").parts[0] != "memory")


def main(strict=False):
    py = [p for d in ("model/engine", "model/validation", "app/server")
          for p in (ROOT / d).glob("*.py") if p.name != "lint_truth.py"]
    md = doc_pages()
    bad = 0

    print("1. DEFAULT TRAP — a harness default contradicting the shipped value")
    for p, ln, fn, arg, got, want in defaults(py):
        print(f"   {p.relative_to(ROOT)}:{ln}  {fn}({arg}={got!r})  shipped is {want!r}")
        bad += 1
    print("   (none)" if not bad else "")

    n0 = bad
    print("2. SELF-CHECK ECHO — a check recomputing what it checks")
    for p, ln, fn, hits in echoes(py):
        print(f"   {p.relative_to(ROOT)}:{ln}  {fn} shares {', '.join(hits)} with its module")
        bad += 1
    print("   (none)" if bad == n0 else "")

    n1 = bad
    print("3. SPLIT VALUES — one metric, several values, no scope on the line")
    for metric, rows in sorted(split_values(md).items()):
        vals = sorted({r[0] for r in rows})
        print(f"   {metric}: {vals}")
        for v, p, ln, txt in rows[:4]:
            print(f"      {p.relative_to(ROOT)}:{ln}  {txt}")
        bad += 1
    print("   (none)" if bad == n1 else "")

    n3 = bad
    print("5. FLAG DRIFT — a CLI default disagreeing with the function it feeds")
    for p, ln, name, cli, fname, fline, fval in flag_drift(py):
        print(f"   {p.relative_to(ROOT)}:{ln}  --{name} default={cli!r} but "
              f"{fname}() at :{fline} defaults {fval!r}")
        bad += 1
    print("   (none)" if bad == n3 else "")

    n4 = bad
    print("6. INERT REBIND — a constant used as a default, reassigned later")
    for p, ln, name, fname, fline, k in inert_rebind(py):
        print(f"   {p.relative_to(ROOT)}:{ln}  {name} is {fname}()'s default at "
              f":{fline} and is assigned {k}x — the later ones do nothing")
        bad += 1
    print("   (none)" if bad == n4 else "")

    n5 = bad
    print("7. SELF-CHECK DRIFT — a check passing a value the default does not use")
    for p, ln, chk, arg, got, owner, oline, want in selfcheck_drift(py):
        print(f"   {p.relative_to(ROOT)}:{ln}  {chk}() passes {arg}={got!r} but "
              f"{owner}() at :{oline} defaults {want!r}")
        bad += 1
    print("   (none)" if bad == n5 else "")

    n6 = bad
    print("8. TRACKED SYMLINK — a pointer to one machine, committed as source")
    hard = tracked_symlinks()
    for path, target in hard:
        print(f"   {path} -> {target or '(unreadable)'}")
        bad += 1
    print("   (none)" if bad == n6 else "")

    print("4. ODOMETER — times each variant has been scored")
    od = odometer()
    for k, v in od.most_common():
        flag = "   <- a best-of-N is not a result" if v >= 5 else ""
        print(f"   {k or '(none)'}: {v}{flag}")
    print("   (ledger empty)" if not od else "")

    print(f"\n{bad} finding(s).")
    # Check 8 blocks WHETHER OR NOT --strict. The other seven are heuristics
    # with false positives, which is why pre-commit runs this advisory; a
    # tracked symlink is not a heuristic, it is a mode bit, and it has been
    # committed twice. Exiting here is what makes .githooks/pre-commit refuse
    # it at the moment it is added rather than at review, and it costs the
    # other checks nothing because their count is reported either way.
    if hard:
        print("a tracked symlink is a machine path in the index - `git rm --cached` "
              "it, and add BOTH spellings to .gitignore (`x` and `x/`: a trailing "
              "slash matches a directory, never a symlink standing at that path)")
        return 1
    return 1 if (strict and bad) else 0


def demo():
    """Self-check: each check must FIRE on a planted instance of its own bug."""
    import tempfile
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "x.py").write_text(
        "import numpy as np\nOVERLAP = 0.25\n"
        # The retired value, which is what a stale default looks like now -
        # and `overlap=None` below must NOT fire: it is the "take whatever
        # ships" sentinel, the one spelling that cannot go stale.
        "def f(overlap: float = 0.10):\n    return np.hypot(1, 2)\n"
        "def g(overlap: float | None = None):\n    return overlap\n"
        "def demo():\n    assert np.hypot(3, 4) == 5\n")
    hits = defaults([d / "x.py"])
    assert hits, "default trap did not fire on overlap=0.10"
    assert all(h[4] is not None for h in hits), f"fired on the None sentinel: {hits}"
    assert echoes([d / "x.py"]), "echo check did not fire on a duplicated np.hypot"
    (d / "y.md").write_text("recall 0.878 is fine\nrecall 0.652 is also fine\n")
    assert "RECALL" in split_values([d / "y.md"]), "split-values did not fire"
    (d / "z.md").write_text("recall 0.878 with cbiou\nrecall 0.652 with bytetrack\n")
    assert not split_values([d / "z.md"]), "split-values fired on scoped lines"
    # 7 must fire on a check exercising a config the default does not use.
    d7 = d / "sc.py"
    d7.write_text("def build(x, smoothing=1.0):\n    return smoothing\n"
                  "def selfcheck():\n    build(0, smoothing=10000.0)\n")
    assert selfcheck_drift([d7]), "7 did not fire on a self-check passing 10000"
    d7.write_text("def build(x, smoothing=10000.0):\n    return smoothing\n"
                  "def selfcheck():\n    build(0, smoothing=10000.0)\n")
    assert not selfcheck_drift([d7]), "7 fired when they agree"
    # 8 must fire on a symlink row and stay quiet on an ordinary blob - the
    # rows are exactly `git ls-files -s` output, absolute targets and all.
    sym = "120000 8b1a9953c4611296a827abf8c47804d7 0\tmodel/models"
    blob = "100644 4f2b1c0d3e5a6f7890abcdef12345678 0\tmodel/engine/track.py"
    assert tracked_symlinks([sym, blob]), "8 did not fire on a tracked symlink"
    assert [p for p, _ in tracked_symlinks([sym, blob])] == ["model/models"], \
        "8 fired on an ordinary blob"
    assert not tracked_symlinks([blob]), "8 fired with no symlink in the index"
    # and the whole index right now: this repo may not carry one at all.
    assert not tracked_symlinks(), f"tracked symlinks in HEAD: {tracked_symlinks()}"

    # The doc scan must see a page in a subfolder, and not the memory store.
    for parts in (("docs", "top.md"), ("docs", "audits", "a.md"), ("docs", "memory", "m.md")):
        d.joinpath(*parts).parent.mkdir(parents=True, exist_ok=True)
        d.joinpath(*parts).write_text("# T\n")
    assert [p.relative_to(d).parts for p in doc_pages(d)] == \
        [("docs", "audits", "a.md"), ("docs", "top.md")], "the doc scan missed a subfolder"

    extra = _selfcheck_5_and_6()
    print("demo ok: all five static checks fire on planted bugs and stay quiet "
          f"when scoped; {extra}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true", help="exit 1 on any finding")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    sys.exit(demo() if a.demo else main(a.strict))
