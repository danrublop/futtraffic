"""What the product actually runs, resolved the way Python resolves it - and every
place that claims to mirror it, checked against it.

WHY THIS EXISTS
    0a5a7cd retired the scene prior's height gate by changing `make_prior`'s
    default. `track.run` carried its own copy of the old gate set and passed it
    explicitly, so `app/server/jobs.py` never reached `make_prior`'s default and the
    product kept both gates for a day. `model/engine/detsweep.py` called `make_prior`
    with no gates and so measured the NEW default. Product and measurement had
    drifted in opposite directions, each internally consistent, and every check
    in the repo passed - because every check looked at one place.

    That is this repo's most-recurring failure: seven memory notes describe it,
    and 13 of the ~27 mistakes surfaced on 2026-09-12 were one fact held in
    several places and updated in some of them. The instruments were all local.
    This one looks at the PAIRS.

WHAT IT RESOLVES - AND WHAT IT REFUSES TO TYPE IN
    `product_config()` reads `app/server/jobs.py`'s single `track_run(...)` call with
    `ast`, takes the keyword arguments it passes, falls back to `track.run`'s
    signature default for everything else, follows `prior_gates=None` through to
    `make_prior`'s own default, and reads tile and overlap off the detector
    constructor that actually runs. **No configuration value is written in this
    file.** A value copied in here would be exactly the defect it exists to
    catch, and would go stale the same way.

    A key the call site fills at run time (`detector=st["detector"]`) cannot be
    resolved statically; it is recorded as `<runtime:...>` and excluded from
    every comparison, and the output says so rather than guessing.

    `**kwargs` (or `*args`) at the call site can fill ANY parameter, so once one
    is present every key the call does not pass by name becomes
    `<runtime:**...>` - never the signature default, which is what it silently
    read before. A default the call might override is not what ships.

WHAT IT CHECKS
    1. HARNESS AGREEMENT. A measurement harness that claims to reproduce the
       product must resolve the same prior gates, stride, conf and overlap.
       Read by `ast`, not by importing, so a heavy harness is never executed.
    2. CONSTANT AGREEMENT. A config constant assigned a literal OUTSIDE its
       defining module - `TILE, OVERLAP = 1024, 0.10` - must equal the
       definition. `lint_truth` cannot see this: its DEFAULT TRAP reads function
       arguments and `ast.Constant`, and this is a module-level tuple assignment.
    3. ARTEFACT STAMPS. `model/engine/track.py` writes the configuration a run actually
       used into `_perf.config`, and the gates come off the filter object itself
       (`make_prior` attaches them), so the stamp cannot disagree with what ran.
       `config_of(path)` diffs a stamp against the product. An artefact made
       before stamps existed reads UNSTAMPED - never on-config by default.

    .venv/bin/python model/validation/shipped_config.py                    # checks 1 and 2
    .venv/bin/python model/validation/shipped_config.py --artefact X.json  # check 3
    .venv/bin/python model/validation/shipped_config.py --demo             # self-check
"""
import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model" / "engine"))
sys.path.insert(0, str(ROOT / "model" / "validation"))
import verify_claims as V  # noqa: E402 - git_env, reused and never copied

JOBS = ROOT / "app" / "server" / "jobs.py"
# The keys a measurement can disagree with the product on. Deliberately the
# ones that have already moved under a measurement at least once.
# "ball" is in here because it was NOT, and that blindness cost a real guard: on
# 2026-09-23 jobs.py started passing ball=True and this checker - built to catch
# exactly "one fact, several places, updated in some of them" - reported 0
# disagreements, because it never looked at the ball.
#: `ball_model` is here so that "the fine-tuned ball detector is OFF by default" is a
#: value the gate CHECKS rather than a sentence somebody wrote. It is a second per-frame
#: pass and a visible product change (docs/experiments/n65-every-ball-number-was-measured-at-384.md),
#: so it should not be able to arrive by accident.
#: `tracker_resolved` is what "best" became (track.run stamps it since 2026-09-28).
KEYS = ("prior_gates", "stride", "conf", "overlap", "tile", "stitch", "ball",
        "ball_model", "tracker_resolved")
# Where each config constant is DEFINED. Any other module-level assignment of
# the same name is a copy, and a copy must agree.
# The autocal four decide whether a real panorama is calibrated or silently parked for clicks
# (model/validation/check.py calib); n31_linereg.py carries copies of two of them.
CONSTANTS = {"OVERLAP": "model/engine/detectors.py", "VERT_CUBIC": "model/engine/autocal.py",
             "MIN_HITS": "model/engine/autocal.py", "MAX_LINE_RMS_PX": "model/engine/autocal.py",
             "PANORAMA_ASPECT": "app/server/jobs.py"}


def check_ball_model() -> list[str]:
    """The product must ship WITHOUT the fine-tuned ball detector, and that must be CHECKED.

    Adding "ball_model" to KEYS resolved it into the printed table and compared it against
    nothing: check_harnesses iterates the HARNESS's keys and no harness declares one. That is
    the same half-fix this file already diagnoses for `ball` - a comment claiming a whole one.
    """
    cfg = product_config()
    if cfg.get("ball_model") is not None:
        return [f"the product resolves ball_model={cfg['ball_model']!r}, but it is a SECOND "
                f"per-frame pass measured at 3.82x detection cost and is not approved to "
                f"ship (model/engine/track.py's ball_model comment)"]
    return []


def _runtime(v) -> bool:
    return isinstance(v, str) and v.startswith("<runtime:")


def _default(func, name):
    return inspect.signature(func).parameters[name].default


def _ast_default(path: Path, func: str, param: str):
    """A function parameter's literal default, read without importing the file."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            a = node.args
            pos = a.posonlyargs + a.args
            for arg, d in zip(pos[len(pos) - len(a.defaults):], a.defaults):
                if arg.arg == param:
                    return ast.literal_eval(d)
            for arg, d in zip(a.kwonlyargs, a.kw_defaults):
                if arg.arg == param and d is not None:
                    return ast.literal_eval(d)
    raise KeyError(f"{path.name}::{func}({param}=...) not found")


def _ast_constant(path: Path, name: str):
    """Module-level literal assignments to `name`, including tuple unpacking and
    annotated assignment (`OVERLAP: float = 0.1`, which is an ast.AnnAssign and
    was invisible). A file that does not parse RAISES: skipping it would put its
    constants beyond the check in green."""
    out = []
    try:
        body = ast.parse(path.read_text()).body
    except SyntaxError as e:
        raise SystemExit(f"shipped_config: {path} does not parse ({e.msg}, line {e.lineno}) - "
                         f"its constants cannot be checked, so nothing is") from e
    for node in body:
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            node = ast.Assign(targets=[node.target], value=node.value, lineno=node.lineno)
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                try:
                    out.append((node.lineno, ast.literal_eval(node.value)))
                except ValueError:
                    pass
            elif isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                for t, v in zip(target.elts, node.value.elts):
                    if isinstance(t, ast.Name) and t.id == name:
                        try:
                            out.append((node.lineno, ast.literal_eval(v)))
                        except ValueError:
                            pass
    return out


def _spread(call: ast.Call) -> list[str]:
    """`**opts` / `*args` at a call site: either may fill any parameter not
    passed by name, so what it fills is only known at run time."""
    return [ast.unparse(k.value) for k in call.keywords if k.arg is None] + \
        [ast.unparse(a.value) for a in call.args if isinstance(a, ast.Starred)]


def product_config(jobs_path: Path = JOBS) -> dict:
    """What a job runs, resolved from the call site outward."""
    import detectors
    import track
    import tracking
    calls = [n for n in ast.walk(ast.parse(jobs_path.read_text()))
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "track_run"]
    if len(calls) != 1:
        # Two call sites could run two configurations; the product would no
        # longer be one thing, and this resolver would have to be told which.
        raise SystemExit(f"{jobs_path}: expected one track_run(...) call, found {len(calls)}")
    passed = {k.arg: k.value for k in calls[0].keywords if k.arg is not None}
    spread = _spread(calls[0])

    def val(name):
        if name in passed:
            node = passed[name]
            try:
                return ast.literal_eval(node)
            except ValueError:
                return f"<runtime:{ast.unparse(node)}>"
        if spread:
            return f"<runtime:**{','.join(spread)}>"
        return _default(track.run, name)

    gates = val("prior_gates")
    if gates is None:
        gates = _default(track.make_prior, "gates")
    return {"prior_gates": gates if _runtime(gates) else tuple(gates),
            "stride": val("stride"), "conf": val("conf"),
            # tile and overlap off the constructor the shipped detector uses:
            # that is what runs, and it is bound at definition time.
            "overlap": _default(detectors._coreml, "overlap"),
            "tile": _default(detectors._coreml, "tile"),
            "detector": val("detector"), "tracker": val("tracker"), "stitch": val("stitch"),
            "tracker_resolved": tracking.best_available(quiet=True) if val("tracker") == "best"
            else val("tracker"),
            "ball": val("ball"),
            # The fine-tuned ball detector. jobs.py does not pass it, so this resolves to
            # run()'s default and the gate asserts the product ships WITHOUT it. If it
            # ever arrives, that is a second per-frame pass and a changed product number,
            # and it should have to be argued for rather than appear.
            "ball_model": val("ball_model")}


def harness_configs() -> dict:
    """Each harness that claims to reproduce the product, resolved the same way."""
    import detectors
    import track
    mp = _default(track.make_prior, "gates")
    tr = inspect.signature(track.run).parameters

    st2 = ROOT / "model" / "validation" / "st2_product.py"
    g = _ast_default(st2, "track", "gates")
    c = _ast_default(st2, "track", "conf")
    hits = _ast_constant(st2, "STRIDE")
    st2_cfg = {"prior_gates": tuple(mp if g is None else g),
               "stride": hits[-1][1] if hits else None,
               "conf": tr["conf"].default if c is None else c}
    st = _ast_default(st2, "track", "stitch")
    st2_cfg["stitch"] = tr["stitch"].default if st is None else st
    # BALL. Adding it to KEYS was half the fix and the comment above claimed the
    # whole one: with no harness resolving it, this checker still could not see
    # the disagreement it was widened for. st2_product.track() has no ball
    # parameter, so it takes track.run's default - which is now the OPPOSITE of
    # what the product passes, and that is a disagreement the pair-checker exists
    # to print rather than to take on trust.
    # A MISSING PARAMETER IS AN ANSWER, not an error: a harness that does not
    # pass `ball` runs track.run's default, and saying so is the whole point of
    # resolving "from the call site outward". _ast_default raises when the name
    # is absent, which is right for a param that should be there and wrong here.
    try:
        b = _ast_default(st2, "track", "ball")
    except KeyError:
        b = None
    st2_cfg["ball"] = tr["ball"].default if b is None else b

    ds = ROOT / "model" / "engine" / "detsweep.py"
    mp_calls = [n for n in ast.walk(ast.parse(ds.read_text()))
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "make_prior"]
    ds_gates = [("gates" in [k.arg for k in n.keywords] or len(n.args) > 2) for n in mp_calls]
    ds_cfg = {"prior_gates": None if any(ds_gates) else tuple(mp)}
    # The overlap decision's SHIPPED arm. 8472cef named it by position in a
    # literal `arms` default and it was the retired 0.10. It is `ship=None` now,
    # deferring to detectors.OVERLAP the way prior_gates=None defers to
    # make_prior, so a literal put back in that default is compared again.
    ship = _ast_default(ds, "overlap_decision", "ship")
    ds_cfg["overlap"] = detectors.OVERLAP if ship is None else ship
    return {"model/validation/st2_product.py::track": st2_cfg,
            "model/engine/detsweep.py (score, overlap_decision)": ds_cfg}


#: Sentences that assert a shipped flag is OFF, and the key each one is about.
#: A number has three checkers here; a SENTENCE about the product has none, and
#: on 2026-09-24 a review found fourteen surfaces still saying the product runs
#: no ball a day after it started running one - README, the ledger the project
#: instructions tell you to quote INSTEAD of README, the model map, the user-facing
#: script, and four pages that say both things at once. "One fact, several
#: places, updated in some of them" is what this file's own docstring calls the
#: repo's most-recurring failure; it applies to prose too.
#: A hit needs BOTH a negation and a subject on the same line. "no ball at all"
#: alone catches "PathCRF needs no ball at all", which is a true sentence about a
#: model; the subject test is what makes the phrase about the PRODUCT.
PROSE = {
    "ball": {
        "deny": ("never passes ball", "never passes `ball", "no ball at all",
                 "passes no ball", "runs no ball", "detects no ball",
                 "never passes it", "ball=false"),
        "subject": ("product", "jobs.py", "job path", "pipeline", "the server",
                    "shipped job", "shipping path"),
        # PAST TENSE IS NOT STALE. A page recording what was true before 66fba82
        # is the record working, and a checker that cannot tell "it does not" from
        # "it did not" teaches people to delete history to get green.
        "past": ("until 66fba82", "before 66fba82", "since 66fba82", "predates",
                 "passed no ball", "used to", "no longer", "was false", "stale"),
    },
}
#: Where a stale sentence is allowed to survive: a page whose subject IS the
#: history. Each entry names the file and why, and it is a LIST so that adding
#: one is a diff somebody reviews.
PROSE_OK = (
    "dfl-dataset (not published)",            # the record of a deletion, written in past tense
    "docs/audits/postmortem.md",             # incidents, quoted verbatim on purpose
    "docs/audits/audit-claims.md",           # quotes the sentences it is auditing
    "model/validation/shipped_config.py",   # this list
)


def check_prose(product=None, root: Path = ROOT) -> list[str]:
    """Tracked text asserting a flag is off while the product has it on.

    Grep-level on purpose. It cannot understand a sentence, so it looks only for
    the literal forms this repo actually wrote, and a false positive is fixed by
    rewording the sentence - which is the outcome wanted anyway.
    """
    product = product_config() if product is None else product
    out = []
    # env: GIT_DIR beats cwd= (verify_claims.git_env). splitlines: a path with a
    # space in it is one path, and .split() made it two that both fail to open.
    files = subprocess.run(["git", "ls-files", "*.md", "*.py"], cwd=root, env=V.git_env(),
                           capture_output=True, text=True).stdout.splitlines()
    for rel in files:
        if any(rel.startswith(ok) for ok in PROSE_OK):
            continue
        try:
            text = (Path(root) / rel).read_text(errors="replace")
        except OSError:
            continue
        for key, rule in PROSE.items():
            if product.get(key) is not True:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                low = line.lower()
                if (any(d in low for d in rule["deny"])
                        and any(sub in low for sub in rule["subject"])
                        and not any(pt in low for pt in rule["past"])):
                    out.append(f"{rel}:{i}  says {key} is OFF, product runs it ON: {line.strip()[:78]}")
    return out


def check_harnesses(product=None, harnesses=None) -> list[str]:
    product = product or product_config()
    harnesses = harnesses if harnesses is not None else harness_configs()
    bad = []
    for site, cfg in harnesses.items():
        for k, v in cfg.items():
            if k not in KEYS or v is None or _runtime(product.get(k)):
                continue
            if v != product.get(k):
                bad.append(f"{site}: {k} = {v!r}, but the product runs {product[k]!r}")
    return bad


def check_constants(files=None) -> list[str]:
    # read off the defining file, never imported: jobs.py and autocal.py are heavy
    truth = {name: _ast_constant(ROOT / home, name)[0][1] for name, home in CONSTANTS.items()}
    bad = []
    if files is None:
        # GIT_* dropped: they beat cwd=, and inside a hook this would read
        # another tree's index. See verify_claims.git_env.
        out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, env=V.git_env(),
                             capture_output=True, text=True).stdout.splitlines()
        files = [ROOT / f for f in out]
    for path in files:
        rel = str(path.relative_to(ROOT)) if path.is_absolute() and ROOT in path.parents \
            else str(path)
        for name, home in CONSTANTS.items():
            if rel == home:
                continue
            try:
                hits = _ast_constant(path, name)
            except (UnicodeDecodeError, OSError):
                continue
            for line, v in hits:
                if v != truth[name]:
                    bad.append(f"{rel}:{line}: {name} = {v!r}, but {home} defines {truth[name]!r}")
    return bad


def config_of(tracks_path) -> dict | None:
    """The configuration an artefact records, or None if it predates stamps."""
    with open(tracks_path) as fh:
        return (json.load(fh).get("_perf") or {}).get("config")


def diff_artefact(stamp: dict | None, product=None) -> tuple[str, list[str]]:
    """('ON' | 'OFF' | 'UNSTAMPED', [differences])."""
    if stamp is None:
        return "UNSTAMPED", []
    product = product or product_config()
    out = []
    for k in KEYS:
        if k not in stamp or _runtime(product.get(k)):
            continue
        s = stamp[k]
        s = tuple(s) if isinstance(s, list) else s
        if s != product[k]:
            out.append(f"{k}: artefact {s!r}, product {product[k]!r}")
    return ("OFF" if out else "ON"), out


def _norm(v):
    return tuple(v) if isinstance(v, list) else v


def compare_guard(stamp_a: dict | None, stamp_b: dict | None, varied: set) -> tuple[str, list[str]]:
    """May two artefacts be compared as differing ONLY in `varied`?

    ('OK' | 'CONFOUNDED' | 'NO_DIFFERENCE' | 'UNSTAMPED', [why]).

    N17 compared conf 0.3 against conf 0.5, and the overlap reverted between its
    two detection runs, so it compared conf-and-overlap while believing it compared
    conf. Nothing refused it, because nothing knew what either arm had been run
    under. The stamps now say; this reads them. (CORRECTED 2026-09-13: this used to
    say a quarter to a third of N17's effect was tiling. N17's clips are 4096x1080,
    where overlap 0.10 and 0.25 give the same tile grid - n21-product-now (not published)
    §1 - so on those clips the confound was real in the stamps and empty in the
    boxes. The guard is right to refuse it anyway: it cannot see frame sizes.)

    NO_DIFFERENCE is the mirror failure: an A/B whose declared variable is equal
    in both arms measures nothing, and would report its noise as an effect.
    """
    if stamp_a is None or stamp_b is None:
        return "UNSTAMPED", ["at least one arm predates configuration stamps"]
    undeclared = [f"{k}: {_norm(stamp_a.get(k))!r} vs {_norm(stamp_b.get(k))!r}"
                  for k in KEYS if k not in varied
                  and _norm(stamp_a.get(k)) != _norm(stamp_b.get(k))]
    if undeclared:
        return "CONFOUNDED", undeclared
    if all(_norm(stamp_a.get(k)) == _norm(stamp_b.get(k)) for k in varied):
        return "NO_DIFFERENCE", [f"declared variable(s) {sorted(varied)} are equal in both arms"]
    return "OK", []


def demo():
    """Every check is shown to FAIL on the defect it exists for."""
    import tempfile

    import track
    real = product_config()
    assert real["prior_gates"] == tuple(_default(track.make_prior, "gates")), real
    assert not _runtime(real["stride"]) and _runtime(real["detector"]), real

    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        # 1. THE 0a5a7cd BUG, replayed: a call site whose resolved gates differ.
        jobs = t / "jobs.py"
        jobs.write_text("def go():\n    track_run(v, o, calib=c, stride=2, "
                        "prior_gates=('pitch', 'height'))\n")
        replay = product_config(jobs)
        assert replay["prior_gates"] == ("pitch", "height"), replay
        bad = check_harnesses(replay, {"harness": {"prior_gates": ("pitch",)}})
        assert len(bad) == 1 and "prior_gates" in bad[0], bad

        # 2. two call sites is two products, and is refused rather than guessed
        two = t / "two.py"
        two.write_text("track_run(a)\ntrack_run(b)\n")
        try:
            product_config(two)
            raise AssertionError("two track_run call sites must be refused")
        except SystemExit:
            pass

        # 2b. `**opts` at the call site can fill ANY key, so an unnamed key is
        #     runtime - reading the signature default here is a guess at what ships
        kw = t / "kw.py"
        kw.write_text("track_run(v, o, stride=2, **opts)\n")
        spread = product_config(kw)
        assert spread["stride"] == 2, spread
        assert _runtime(spread["conf"]) and _runtime(spread["prior_gates"]), \
            f"a **kwargs call site resolved to the defaults: {spread}"
        assert check_harnesses(spread, {"h": {"conf": 0.3}}) == [], "a runtime key was compared"

        # 3. a runtime key is never compared - it would always "differ"
        assert check_harnesses(real, {"h": {"detector": "coreml"}}) == []

        # 4. CONSTANT AGREEMENT - the st2_tiles.py shape, tuple-unpacked
        copy = t / "copy.py"
        copy.write_text("TILE, OVERLAP = 1024, 0.10\n")
        import detectors
        found = check_constants([copy])
        if detectors.OVERLAP != 0.10:
            assert len(found) == 1 and "OVERLAP" in found[0], found
        agrees = t / "agrees.py"
        agrees.write_text(f"OVERLAP = {detectors.OVERLAP!r}\n")
        assert check_constants([agrees]) == []
        # an ANNOTATED copy is a copy too
        ann = t / "ann.py"
        ann.write_text("MIN_HITS: int = 900\n")
        found = check_constants([ann])
        assert len(found) == 1 and "MIN_HITS" in found[0], f"an AnnAssign copy was invisible: {found}"
        # a scanned file that does not parse is refused, never skipped in green
        broken = t / "broken.py"
        broken.write_text("MIN_HITS = 900\ndef (:\n")
        try:
            check_constants([broken])
            raise AssertionError("a file with a SyntaxError was skipped - its constants go unchecked")
        except SystemExit:
            pass
        # check_prose: a stale sentence in a tracked file WITH A SPACE IN ITS
        # PATH, listed under a hostile ambient GIT_DIR (as inside a hook)
        import os
        pr = t / "prose"
        pr.mkdir()
        (pr / "a page.md").write_text("The product runs no ball.\n")
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        subprocess.run(["git", "init", "-q"], cwd=pr, env=env, check=True)
        subprocess.run(["git", "add", "-A"], cwd=pr, env=env, check=True)
        saved = os.environ.get("GIT_DIR")
        os.environ["GIT_DIR"] = "/nonexistent/.git"
        try:
            stale = check_prose({"ball": True}, root=pr)
        finally:
            os.environ.pop("GIT_DIR") if saved is None else os.environ.update(GIT_DIR=saved)
        assert len(stale) == 1 and stale[0].startswith("a page.md:1"), \
            f"check_prose missed a stale sentence (path with a space, or GIT_DIR leaked): {stale}"
        # the autocal constants are policed the same way: a stale MIN_HITS copy is named
        stale = t / "stale.py"
        stale.write_text("MIN_HITS = 900\n")
        found = check_constants([stale])
        assert len(found) == 1 and "MIN_HITS" in found[0], found

        # 5. ARTEFACT STAMPS - unstamped is never ON, and a wrong stamp is OFF
        assert diff_artefact(None, real)[0] == "UNSTAMPED"
        on = {k: (list(v) if isinstance(v, tuple) else v) for k, v in real.items()}
        assert diff_artefact(on, real) == ("ON", []), diff_artefact(on, real)
        off = dict(on, prior_gates=["pitch", "height"])
        status, why = diff_artefact(off, real)
        assert status == "OFF" and "prior_gates" in why[0], (status, why)
        # a harness that ran another tracker than "best" resolves to is OFF too
        status, why = diff_artefact(dict(on, tracker_resolved="bytetrack"), real)
        assert status == "OFF" and "tracker_resolved" in why[0], (status, why)

        # 6. THE N17 CONFOUND, replayed: conf declared, overlap also moved.
        arm05 = dict(on, conf=0.5, overlap=0.10)
        arm03 = dict(on, conf=0.3, overlap=0.25)
        verdict, why = compare_guard(arm05, arm03, {"conf"})
        assert verdict == "CONFOUNDED" and "overlap" in why[0], (verdict, why)
        # the honest version of the same comparison passes
        assert compare_guard(dict(on, conf=0.5), dict(on, conf=0.3), {"conf"})[0] == "OK"
        # an A/B whose variable never moved measures nothing
        assert compare_guard(on, dict(on), {"conf"})[0] == "NO_DIFFERENCE"
        # and an unstamped arm is never silently OK
        assert compare_guard(None, on, {"conf"})[0] == "UNSTAMPED"
    print("shipped_config.py ok")


def main() -> int:
    if "--demo" in sys.argv:
        demo()
        return 0
    if "--compare" in sys.argv:
        i = sys.argv.index("--compare")
        a, b = config_of(sys.argv[i + 1]), config_of(sys.argv[i + 2])
        varied = set(sys.argv[sys.argv.index("--varied") + 1].split(",")) \
            if "--varied" in sys.argv else set()
        verdict, why = compare_guard(a, b, varied)
        print(f"{verdict}  {sys.argv[i + 1]}  vs  {sys.argv[i + 2]}  (varied: {sorted(varied)})")
        for w in why:
            print(f"   {w}")
        return 0 if verdict == "OK" else 1
    if "--artefact" in sys.argv:
        path = sys.argv[sys.argv.index("--artefact") + 1]
        status, why = diff_artefact(config_of(path))
        print(f"{status}  {path}")
        for w in why:
            print(f"   {w}")
        return 0 if status == "ON" else 1
    product = product_config()
    print("product config, resolved from app/server/jobs.py outward:")
    for k, v in product.items():
        print(f"   {k:12s} {v!r}" + ("   (filled at run time - not compared)" if _runtime(v) else ""))
    bad = check_harnesses(product) + check_constants() + check_ball_model()
    prose = check_prose(product)
    print()
    for b in bad:
        print(f"DISAGREES  {b}")
    for b in prose:
        print(f"STALE      {b}")
    print(f"\n{len(bad)} disagreement(s) with the shipped configuration, "
          f"{len(prose)} stale sentence(s) about it")
    bad += prose
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
