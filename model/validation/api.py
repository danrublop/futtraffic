"""The only sanctioned source of a number in this repo.

WHY THIS EXISTS
    `postmortem (not published)` §B is six incidents of one failure: a number was written
    into prose, and prose cannot be re-run. `+0.024` survived in SEVEN documents
    after the experiment producing it was withdrawn. Mean DetA was wrong at two
    different sequence counts and nobody had challenged it. `team_eval` drifted
    from 70.7% to 69.7% while every doc kept quoting 70.7%.

    Correcting those by hand is not a fix, it is the same debt paid again. The
    fix has to remove the COPY STEP: a number stops being a literal you can
    paste and becomes an identifier you resolve.

        docs say:   the gate is worth `api://ab.op.hota.delta`
        not:        the gate is worth +0.0122

    Then a document cannot go stale, because it holds no number to go stale.

THE RULES THIS ENFORCES MECHANICALLY

    1. **A measurement carries its scope or it does not exist.** `n`, the
       sequence set, and whether the split is tuned or held out are REQUIRED
       fields. `postmortem (not published)` B2 was a subpopulation rate compared to a
       full-population rate; that comparison is unstateable here, because both
       carry their denominator. `verify()` diffs `n` and `scope` as well as the
       value: a rate that silently changed denominator is B2 all over again.

    2. **A failed producer returns FAILED, never a default.** Every silent
       fallback in this repo's history turned a failure into a plausible number.
       `get()` raises or returns `status="FAILED"` with the error; it will not
       hand back a stale value dressed as fresh. A value that cannot round-trip
       through the golden - NaN, inf, None, anything JSON cannot write - is a
       failure too, because `abs(was - nan) > 1e-9` is False and a NaN walked
       straight past the guard as "unchanged".

    3. **A cached value knows when it is stale.** The cache key is a content
       hash of the inputs - ground truth, weights and installed packages, not
       just code - plus every module that produces the value, plus the id's own
       slice of the registry. Change the scorer, the tracks or labels it reads,
       or the `n` on a Spec, and the value is marked STALE and re-run. This is
       the `+0.024` failure made impossible, and it is what lets the gate
       resolve from the cache instead of re-running two hours of producers.

    4. **A number in prose is a copy, and a copy goes stale.** §E promised "a
       lint fails on a bare decimal that has no id" and it was never written;
       `check_docs()` only validated the citations already present, which is the
       exact hole `+0.024` fit through for seven documents. It fails on a bare
       decimal now, and when `verify()` finds a value has moved it names every
       document still quoting the old one.

WHAT IT IS NOT
    Not a new measurement layer. Every producer is a thin call into the scorer
    that already exists - `st2_ab`, `st2_mot`, `st2_metres`, `team_eval`. If the
    number is wrong, it is wrong in the same place it was always wrong, and there
    is exactly one place to fix it.

`ponytail: the cache is a JSON file and the registry is a dict. A database here
would be a second thing to be wrong.`
"""
import argparse
import ast
import collections
import functools
import glob
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "model", "validation", "out", "api-cache.json")
GOLDEN = os.path.join(ROOT, "model", "validation", "api-golden.json")
# The registry produces every value, so it belongs in every key - its shared
# producers plus the id's own entry, see `_key`. Declared here rather than in a
# Spec's `code` because a Spec that forgot it would serve its own old `n` - and
# forgetting was exactly how four of the first twelve shipped with no inputs.
REGISTRY = "model/validation/measurements.py"
#: Above this a file is keyed on its stat (see _hash), not content. The panorama video a
#: detector id reads is 3.5 GB; reading it on every resolve made the KEY cost
#: more than some producers, and a key that slow is a key nobody checks.
BIG = 64 << 20
#: When this process started. A dep file CHANGED after it may have been loaded
#: BEFORE the change, so a hash of its current bytes would describe code that
#: never ran - see get(). Compared against ctime, not mtime: `cp -p` and
#: `rsync -a` put an old mtime back on new bytes, and nothing can set ctime.
STARTED_NS = time.time_ns()


@dataclass
class Measurement:
    id: str
    value: object
    unit: str
    scope: str
    n: int
    desc: str
    status: str = "OK"
    commit: str = ""
    inputs_hash: str = ""
    at: str = ""
    error: str = ""

    def line(self):
        v = f"{self.value:.4f}" if isinstance(self.value, float) else self.value
        return (f"{self.id:<34s} {str(v):>12s} {self.unit:<7s} n={self.n:<3d} "
                f"{self.scope:<26s} {self.status} {self.error}").rstrip()


def _git_env():
    """Every GIT_* variable dropped. See verify_claims.git_env for the failure:
    they beat cwd=, so a stamp taken inside a hook names another tree's HEAD."""
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              env=_git_env(), capture_output=True, text=True,
                              timeout=30).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _hash(paths):
    """Content hash of the inputs a measurement depends on.

    Path AND content, each separated. Hashing contents alone let two different
    input sets whose bytes concatenate the same produce one key, and a directory
    input raised IsADirectoryError, was swallowed as OSError and hashed as the
    constant `MISSING:` - a key that can never change, which is rule 3 defeated
    the same way `postmortem (not published)` B4 defeated it.

    `ponytail: files over BIG are keyed on stat, not content: size, mtime,
    ctime and inode. `cp -p` / `rsync -a` restore the mtime but not the ctime
    or inode, and utime cannot set ctime, so a rewrite always moves the key.
    tracks.split.json (145 MB) is on this path, not only the video. os.walk
    does not follow symlinked subdirectories; pass followlinks=True with a
    visited set if a symlinked tree is ever declared.`
    """
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(b"\0" + str(p).encode() + b"\0")
        full = p if os.path.isabs(p) else os.path.join(ROOT, p)
        files = [(b"", full)]
        if os.path.isdir(full):
            files = sorted((os.path.relpath(os.path.join(d, f), full).encode(),
                            os.path.join(d, f))
                           for d, _, fs in os.walk(full) for f in fs)
        for name, f in files:
            h.update(name + b"\0")
            try:
                st = os.stat(f)
                if st.st_size > BIG:
                    h.update(f"stat:{st.st_size}:{st.st_mtime_ns}:"
                             f"{st.st_ctime_ns}:{st.st_ino}".encode())
                    continue
                with open(f, "rb") as fh:
                    h.update(hashlib.file_digest(fh, "sha256").digest())
            except OSError:
                h.update(b"MISSING")
    return h.hexdigest()[:16]


@functools.lru_cache(maxsize=4)
def _registry_slices(src):
    """measurements.py -> (shared text, {id: that id's own entry text}).

    `shared` is everything outside the MEASUREMENTS literal: the producers and
    constants any entry may call. Comments between entries are in neither, so
    editing prose re-runs nothing - prose does not change a value."""
    for node in ast.parse(src).body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and any(getattr(t, "id", None) == "MEASUREMENTS"
                        for t in node.targets)):
            lines = src.splitlines(keepends=True)
            shared = "".join(lines[:node.lineno - 1] + lines[node.end_lineno:])
            return shared, {k.value: ast.get_source_segment(src, v)
                            for k, v in zip(node.value.keys, node.value.values)
                            if isinstance(k, ast.Constant)}
    return None, {}


def _key(mid, spec):
    """Declared inputs + declared code + this id's slice of the registry.

    The slice, not the whole file: with the whole registry in every key, one
    edited desc string invalidated all producers at once. Shared producer code
    still invalidates everything, which is the safe direction. Code a producer
    loads WITHOUT declaring it is covered separately, by `_deps` at resolve."""
    with open(os.path.join(ROOT, REGISTRY), encoding="utf-8") as fh:
        shared, entries = _registry_slices(fh.read())
    if mid not in entries:
        # Not a literal entry (a planted or dynamically added Spec): key on
        # the whole registry file, as every id was before.
        return _hash(tuple(spec.inputs) + tuple(spec.code) + (REGISTRY,))
    base = _hash(tuple(spec.inputs) + tuple(spec.code))
    return hashlib.sha256(
        (base + "\0" + shared + "\0" + entries[mid]).encode()).hexdigest()[:16]


@functools.cache
def _dists_of():
    return importlib.metadata.packages_distributions()


@functools.cache
def _version(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"


def _deps():
    """Repo files and package versions loaded in this process right now.

    Recorded after a producer runs, because a hand-declared `code` tuple is a
    claim and the claim was wrong: model/engine/pitch.py applies the homography
    under every metre-space id and was in none of their keys. It is the whole
    process's sys.modules rather than what this producer alone imported, so
    it over-includes - that costs a re-run; the other direction serves a stale
    number as fresh. A producer run in a SUBPROCESS is invisible here, which
    is why none are.
    """
    real, here = os.path.realpath(ROOT) + os.sep, os.path.abspath(ROOT) + os.sep
    venv = os.sep + ".venv" + os.sep
    files, dists = set(), {"python": sys.version.split()[0]}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        # BOTH the path as imported and its realpath are tested against ROOT,
        # and the UNRESOLVED one wins. realpath alone silently drops any repo
        # file reached through a symlinked directory - `model/validation/out`
        # and `model/models` are symlinks in every worktree - and dropping a
        # dep is the direction that serves a stale number as fresh. It also
        # broke demo()'s planted-dependency assertion, which is how it was
        # found: an environment shape read as a measurement failure.
        a, r = os.path.abspath(f), os.path.realpath(f)
        if venv in a or venv in r:
            continue
        if a.startswith(here):
            files.add(os.path.relpath(a, here))
        elif r.startswith(real):
            files.add(os.path.relpath(r, real))
        for d in _dists_of().get(name.partition(".")[0], ()):
            dists[d] = _version(d)
    # The mechanism and the registry are keyed elsewhere (see _key), and in
    # here they would put every id back behind every edit to either.
    files -= {"model/validation/api.py", REGISTRY}
    return {"files": sorted(files), "dists": dists}


def _fresh_deps(entry):
    """True iff a cache entry's recorded dependencies still hash the same.
    A missing, malformed or null record is stale, never a crash: one bad entry
    must cost a re-run, not the whole verify."""
    if not isinstance(entry, dict):
        return False
    deps, h = entry.get("deps"), entry.get("deps_hash")
    if not (h and isinstance(deps, dict) and isinstance(deps.get("files"), list)
            and isinstance(deps.get("dists"), dict)):
        return False
    return h == _deps_hash(deps)


def _deps_hash(deps):
    """What the recorded dependencies ARE now: file contents and versions."""
    now = {d: (sys.version.split()[0] if d == "python" else _version(d))
           for d in deps["dists"]}
    return hashlib.sha256((_hash(deps["files"]) + json.dumps(now, sort_keys=True))
                          .encode()).hexdigest()[:16]


def _read_json(path):
    """-> the object, or None if it is absent OR unreadable. The caller must be
    able to tell "no golden yet" from "the golden is half-written"."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path, obj):
    """Write via a temp file and rename.

    A `json.dump` straight onto the golden is a truncating open: interrupted, it
    leaves a half-written file AND destroys the history it was appending to. The
    rename is atomic, so the golden is either the old one or the new one.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, allow_nan=False)
    os.replace(tmp, path)


def keys(mids=None):
    """Every key, computed BEFORE any producer runs. Pass the result to get().

    The producers memoise (`_dist`, `_calib`, `_mot` ...), so within one run a
    sibling id can return a value computed before an input was rewritten. Keyed
    at its own get() time, that old value was stored under the NEW input's key
    and served as fresh ever after. Keyed up front, it is stored under the key
    of the inputs it was actually computed from, and the next run re-runs it."""
    unknown = [m for m in (mids or ()) if m not in MEASUREMENTS]
    if unknown:
        raise KeyError(f"unknown measurement {unknown[0]!r}; --list shows all")
    return {m: _key(m, MEASUREMENTS[m]) for m in (mids or MEASUREMENTS)}


def get(mid, refresh=False, key=None):
    """Resolve one measurement. Never returns a stale or defaulted value.

    `key` from `keys()` whenever more than one id is resolved in a process."""
    if mid not in MEASUREMENTS:
        raise KeyError(f"unknown measurement {mid!r}; --list shows all")
    spec = MEASUREMENTS[mid]
    # The key is inputs AND every module that produces the value.
    ih = key or _key(mid, spec)
    c = _read_json(CACHE)
    c = (c.get(mid) if isinstance(c, dict) else None) or {}
    if (not refresh and c.get("inputs_hash") == ih
            and c.get("status") == "OK" and "value" in c
            and _fresh_deps(c)):
        # Only the VALUE comes from the cache. Scope, n, unit and desc are read
        # from the Spec every time, so a cached record can never hand back a
        # denominator the registry has since corrected.
        return Measurement(mid, c["value"], spec.unit, spec.scope, spec.n,
                           spec.desc, "OK", c.get("commit", ""), ih,
                           c.get("at", ""))
    at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())   # Z means UTC
    try:
        val = spec.fn()
        # A producer may return (value, n_seen): the denominator it ACTUALLY scored.
        # The Spec's `n` is a literal someone typed; a population that shrank under
        # it (a clip dropped, a player lost) kept printing the old n beside a value
        # computed on the new one - rule 1 broken by a stale constant.
        if isinstance(val, tuple) and len(val) == 2 and type(val[1]) is int:
            val, seen = val
            if seen != spec.n:
                raise ValueError(f"the producer scored n={seen}, the Spec says n={spec.n}: "
                                 f"the population moved under the registered denominator")
        if val is None:
            raise ValueError("producer returned None, which is not a measurement")
        json.dumps(val, allow_nan=False)   # rule 2: NaN and inf are failures
        m = Measurement(mid, val, spec.unit, spec.scope, spec.n, spec.desc,
                        "OK", _commit(), ih, at)
    # SystemExit too: st2_mot.stage_pred and ball_eval.load_ball refuse with
    # sys.exit, which is not an Exception, so a missing input killed the whole
    # verify run instead of recording one FAILED row.
    except (Exception, SystemExit) as e:   # rule 2: a failure is a status, never a number
        m = Measurement(mid, None, spec.unit, spec.scope, spec.n, spec.desc,
                        "FAILED", _commit(), ih, at, f"{type(e).__name__}: {e}")
    deps = _deps()
    # A dep file edited while this process ran may have been imported before
    # the edit: its bytes now are not the code that produced the value. No
    # hash, so the entry can never be served, and the next run re-runs it.
    root = os.path.realpath(ROOT)
    edited = any(os.stat(os.path.join(root, f)).st_ctime_ns > STARTED_NS
                 for f in deps["files"] if os.path.exists(os.path.join(root, f)))
    d = _read_json(CACHE)
    d = d if isinstance(d, dict) else {}       # a cache that is not a map is no cache
    d[mid] = dict(asdict(m), deps=deps,
                  deps_hash=None if edited else _deps_hash(deps))
    _write_json(CACHE, d)
    return m


def snapshot(reason, only=None):
    """Freeze measurements as the accepted truth. Requires a REASON.

    A snapshot without a reason is how a number changes silently, so the reason
    is a required argument rather than a nicety - it is the only record of WHY
    a value was allowed to move. Always re-runs: freezing a cached value would
    freeze whatever the cache happened to hold.

    `only` freezes just those ids and leaves every other golden entry as it
    was. Without it, accepting ONE new id re-blessed all of them under that one
    reason - the next blanket snapshot would have accepted anything that had
    drifted in the meantime, and nobody would have been asked. An id in `only`
    that is in the golden but no longer registered is DELETED from it - the
    only other way to accept a deletion was a blanket snapshot. Each history
    entry is the snapshot that wrote it: its reason, its ids, and the values
    it replaced, so any accepted value can be traced to why it moved.
    """
    if not str(reason or "").strip():
        raise SystemExit("refusing to snapshot without a reason: the reason is "
                         "the only record of why a value was allowed to move")
    if only is not None and not only:
        raise ValueError("an empty `only` would re-freeze every id; name them")
    prev = _read_json(GOLDEN) or {}
    old = prev.get("values", {})
    unknown = [m for m in (only or ()) if m not in MEASUREMENTS and m not in old]
    if unknown:
        raise KeyError(f"neither registered nor in the golden: {unknown}")
    ids = list(dict.fromkeys(only or MEASUREMENTS))
    vals = dict(old) if only else {}
    ks = keys([m for m in ids if m in MEASUREMENTS])
    for mid in ids:
        if mid not in MEASUREMENTS:
            vals.pop(mid)           # accepting a deletion, by name
            continue
        m = get(mid, refresh=True, key=ks[mid])
        # A KNOWN FAILURE IS A LEGITIMATE THING TO FREEZE. Rule 2 says a failed
        # producer never returns a NUMBER; it does not say the failure cannot be
        # recorded. Refusing to snapshot one sounded strict and was worse: a
        # measurement deliberately registered as a live work item - possession
        # through our own tracker, the unbuilt event model - would have made the
        # golden unmovable, and the pressure would have gone on deleting the
        # registration rather than fixing the gap. So the failure is frozen WITH
        # its blocker, and verify() treats a change of status in EITHER direction
        # as a finding: OK -> FAILED is a regression, FAILED -> OK is a gap
        # closing that someone should notice and re-snapshot.
        vals[mid] = {"value": m.value, "n": m.n, "scope": m.scope,
                     "status": m.status, "error": m.error}
    now, commit = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), _commit()
    history = list(prev.get("history", []))
    # Goldens written before this change record only the SUPERSEDED snapshot
    # in history; carry the current one over once so no reason is lost.
    top = {"at": prev.get("updated"), "commit": prev.get("commit"),
           "reason": prev.get("reason")}
    if prev.get("reason") and not (history and all(
            history[-1].get(k) == v for k, v in top.items())):
        history.append(top)
    history.append({"at": now, "commit": commit, "reason": reason,
                    "ids": sorted(ids) if only else "all",
                    "replaced": {k: v for k, v in old.items() if vals.get(k) != v}})
    _write_json(GOLDEN, {"updated": now, "commit": commit, "reason": reason,
                         "history": history, "values": vals})
    return vals


def id_drift():
    """-> (registered but not in the golden, in the golden but not registered).

    Seconds, no producer runs, so it can sit in pre-commit. The registry once
    ran ahead of the golden for twenty commits: every new id rendered
    `not_evaluated` in the product and the push gate was red, and nothing said
    so until someone ran a full verify (model-map (not published) §0.1)."""
    g = (_read_json(GOLDEN) or {}).get("values", {})
    return sorted(set(MEASUREMENTS) - set(g)), sorted(set(g) - set(MEASUREMENTS))


def verify(refresh=True):
    """Resolve everything and diff against the golden. -> list of differences.

    This is the answer to "how do we know an API change did not move something
    it should not have". Any movement is a finding: either it was intended, and
    a snapshot with a reason records that, or it was not, and it is a bug.

    WHAT refresh=False CANNOT SEE. It re-runs only ids whose key moved - the
    declared inputs, declared code, registry slice, and every repo module and
    package version the producer loaded (`_deps`). It does NOT catch a
    nondeterministic producer, or an input read from somewhere no key covers
    (an env var, a file outside every declaration). refresh=True does, at the
    cost of every producer; `snapshot()` always re-runs what it freezes.
    """
    g = _read_json(GOLDEN)
    if g is None:
        return [("(golden)", None, None,
                 "missing or unreadable - run --snapshot first")]
    out, vals = [], g.get("values", {})
    ks = keys()
    for mid in MEASUREMENTS:
        m = get(mid, refresh, key=ks[mid])
        known = vals.get(mid)
        was_status = (known or {}).get("status", "OK")
        if m.status != "OK":
            # A failure the golden ALREADY RECORDS, with the same blocker, is a
            # known gap and verifies clean - otherwise registering a gap as a
            # live work item makes the golden permanently unmovable, and the
            # pressure falls on deleting the registration instead of fixing the
            # gap. Everything else about a failure is still a finding: a new
            # one, one the golden thought was OK, or the same failure with a
            # DIFFERENT blocker, which means the work item moved.
            same = (known is not None and was_status == "FAILED"
                    and (known.get("error") or "")[:120] == (m.error or "")[:120])
            if not same:
                why = ("REGRESSION, was OK" if known is not None and was_status == "OK"
                       else "same failure, NEW blocker (was FAILED)" if was_status == "FAILED"
                       else "NEW measurement, and it is FAILED")
                out.append((mid, (known or {}).get("value", "(absent)"), None,
                            f"{why}: {m.error}"))
            continue
        if was_status == "FAILED":
            out.append((mid, "(failed)", m.value,
                        "a GAP CLOSED - it produces a value now, re-snapshot"))
            continue
        if known is None:
            out.append((mid, "(absent)", m.value, "NEW measurement, not in golden"))
            continue
        was, now = known.get("value"), m.value
        if isinstance(was, float) and isinstance(now, float):
            # NOT `abs(was - now) > 1e-9`: that is False when either side is
            # NaN, so a producer that started returning NaN read as unchanged.
            if not abs(was - now) <= 1e-9:
                out.append((mid, was, now, f"moved {now - was:+.6f}"))
        elif json.dumps(was, sort_keys=True) != json.dumps(now, sort_keys=True):
            out.append((mid, was, now, "changed"))
        # Rule 1: the same value over a different population is a different
        # claim. The golden records both and both are diffed.
        if known.get("n") != m.n:
            out.append((mid, known.get("n"), m.n, "DENOMINATOR changed"))
        if known.get("scope") != m.scope:
            out.append((mid, known.get("scope"), m.scope, "SCOPE changed"))
    for mid in vals:
        if mid not in MEASUREMENTS:
            out.append((mid, vals[mid].get("value"), "(removed)",
                        "measurement DELETED - is anything still citing it?"))
    return out


CITE = re.compile(r"api://([a-z0-9_.]+)")
# A decimal, but not `v1.4.0` or `§1.3`: a version and a section reference are
# not measurements, and a lint that cries about them is a lint that gets muted.
NUM = re.compile(r"(?<![\w.§])(\d+\.\d+)(%?)")
STRICT, MARK = "<!-- api:strict -->", "<!-- api:ok"


def doc_pages(root=ROOT):
    """Every page under docs/ at any depth except the memory store. Recursive: a
    top-level glob went silently green on every page moved into a subfolder."""
    docs = os.path.join(root, "docs")
    return sorted(p for p in glob.glob(os.path.join(docs, "**", "*.md"), recursive=True)
                  if os.path.relpath(p, docs).split(os.sep)[0] != "memory")


def _scan(paths=None):
    """Every prose line of every doc. -> [(file, lineno, line, strict)]

    Fenced code blocks are skipped: those are commands and their output, and
    output is not the copy step - prose is.

    A list, not a generator, so the "scanned nothing" guard below raises when it
    is CALLED. As a generator it raised only if someone iterated, which is the
    A7 shape it was written to prevent - and demo() caught exactly that.
    """
    if paths is None:
        paths = doc_pages()
    if not paths:
        raise RuntimeError("scanned NO documents. A lint that inspects nothing "
                           "reports clean, which is postmortem A7 exactly")
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            text = fh.read()
        prose, fence = [], False
        for i, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("```"):
                fence = not fence
            elif not fence:
                prose.append((os.path.relpath(p, ROOT), i, line))
        # The opt-in marker counts as PROSE only. Matching it anywhere in the
        # file made a document that merely SHOWS the marker in a code fence opt
        # itself in - 195 findings against the doc explaining the lint. That is
        # verify_claims.py checking its own seed block (postmortem A7), rebuilt
        # inside the fix for it, and it is why `fence` is computed first.
        strict = any(STRICT in line for _, _, line in prose)
        out += [(f, i, line, strict) for f, i, line in prose]
    return out


def _cited_floats(mids, golden):
    """The golden values of the ids a line cites - the ONLY decimals it may carry bare."""
    vals = (golden or {}).get("values", {})
    return [x for m in mids for x in _floats((vals.get(m) or {}).get("value"))]


def check_docs(paths=None):
    """The §E lint. -> [(file, lineno, kind, detail, blocking)]

    Two findings. An `api://` that resolves to nothing is always blocking. A
    decimal with no id on its line is a COPY - `postmortem (not published)` B1 is one
    such copy in seven documents - and it blocks in any file that has opted in
    with `<!-- api:strict -->`.

    A legitimate bare number - a literature baseline, a date, a figure quoted
    precisely because it was WRONG - is marked `<!-- api:ok why -->` on its line.

    `ponytail: opt-in per file, because ~11,000 decimals predate the rule and a
    lint with 11,000 findings is a lint everyone turns off. Everything not opted
    in is counted as backlog and printed, never silently dropped. Upgrade path:
    mark documents strict as they are rewritten, and flip the default when the
    backlog reaches zero.`
    """
    out, golden = [], _read_json(GOLDEN)
    for f, i, line, strict in _scan(paths):
        cited = CITE.findall(line)
        for mid in cited:
            if mid not in MEASUREMENTS:
                out.append((f, i, "unknown-id", f"api://{mid}", True))
        if MARK in line:
            continue
        # A citation used to exempt its WHOLE line, so `api://x` let any other
        # decimal ride beside it unlinted. Now only the cited value itself - a
        # rendering of the golden value of an id the line cites - is exempt.
        ok = _cited_floats(cited, golden)
        for lit, pct in NUM.findall(CITE.sub("", line)):
            dp = len(lit.split(".")[1]) + (2 if pct else 0)
            x = float(lit) / (100 if pct else 1)
            if any(round(v, dp) == round(x, dp) for v in ok):
                continue
            out.append((f, i, "bare-number", lit + pct, strict))
    return out


def _floats(v):
    """Every finite float inside a measurement value - `ab.*` values are dicts.
    Ints are skipped on purpose: `k=4` and `n=10` match half the prose in the
    repo, and a count that changed shows up as DENOMINATOR changed anyway."""
    if isinstance(v, dict):
        return [x for w in v.values() for x in _floats(w)]
    return [v] if isinstance(v, float) and math.isfinite(v) else []


def stale_copies(diffs, paths=None):
    """Documents still quoting a number that just moved. -> [(file, line, lit, id)]

    `+0.024` lived in SEVEN documents because nothing connected "the experiment
    was withdrawn" to "the prose still says it". A moved value and a document
    holding the old one IS that connection, and this is the only moment it is
    catchable. Cited lines are searched too: `api://x` next to a pasted 0.363 is
    still a paste.
    """
    old = {mid: _floats(was) for mid, was, _now, _why in diffs}
    old = {k: v for k, v in old.items() if v}
    if not old:
        return []
    hits = []
    for f, i, line, _strict in _scan(paths):
        for lit, pct in NUM.findall(line):
            dp = len(lit.split(".")[1]) + (2 if pct else 0)
            if dp < 2:      # one decimal place matches everything
                continue
            x = float(lit) / (100 if pct else 1)
            hits += [(f, i, lit + pct, mid) for mid, vs in old.items()
                     if any(round(v, dp) == round(x, dp) for v in vs)]
    return hits


def demo():
    """The rules must be enforceable, not aspirational.

    `postmortem (not published)` §A is five self-checks that could not fail, so every
    assertion here is written against a PLANTED break and carries the incident
    it re-runs. The registry below is planted too: the guard has to be testable
    without the five-minute pipeline, and without touching the production cache
    or golden - the first version wrote its planted failure into both.
    """
    global CACHE, GOLDEN, MEASUREMENTS, BIG, STARTED_NS
    keep = CACHE, GOLDEN, MEASUREMENTS
    tmp = tempfile.mkdtemp()
    CACHE, GOLDEN = os.path.join(tmp, "cache.json"), os.path.join(tmp, "g.json")
    try:
        # Rule 1, on the REAL registry: scope, denominator, inputs, code.
        for mid, spec in MEASUREMENTS.items():
            assert spec.scope and spec.desc, mid
            assert spec.n > 0, f"{mid} has no denominator - rule 1"
            assert spec.inputs, (f"{mid} declares no inputs, so it hashes to "
                                 f"sha256 of nothing and can NEVER invalidate")
            assert spec.code, f"{mid} declares no producing code - rule 3"
        ab = MEASUREMENTS["ab.offpitch.hota"]
        assert "model/validation/st2_ab.py" in ab.code, \
            "an A/B measurement must invalidate when st2_ab.py changes"
        assert _hash(tuple(ab.inputs) + tuple(ab.code)) != _hash(tuple(ab.inputs)), \
            "the producing code is not in the cache key"
        # Rule 3 for DATA: an id whose inputs are only .py files has declared
        # its code twice and its data never, so new ground truth or new weights
        # serve the old number as fresh - team.* and calib.metres.* did this.
        for mid, spec in MEASUREMENTS.items():
            # A bare string where a tuple belongs - `("a.py")` - is hashed as
            # one "path" per CHARACTER: a key that looks fine and tracks nothing.
            assert isinstance(spec.inputs, tuple) and isinstance(spec.code, tuple), \
                f"{mid}: inputs/code must be tuples, got a string"
            assert any(not str(p).endswith(".py") for p in spec.inputs), \
                f"{mid} declares only code as inputs - its data is not in the key"

        # The registry slice. Every literal entry must have one, an edit to one
        # entry must move only that entry, and an edit to the shared producers
        # must move every entry.
        with open(os.path.join(ROOT, REGISTRY), encoding="utf-8") as fh:
            real = fh.read()
        shared, entries = _registry_slices(real)
        assert set(entries) == set(MEASUREMENTS), \
            "an id has no entry in the MEASUREMENTS literal, so its key ignores " \
            "its own definition"
        a_, b_ = list(MEASUREMENTS)[0], list(MEASUREMENTS)[-1]
        one = real.replace(entries[a_], entries[a_].replace("Spec(", "Spec( ", 1), 1)
        sh1, en1 = _registry_slices(one)
        assert en1[a_] != entries[a_] and sh1 == shared, \
            "editing an entry did not move its own slice"
        assert en1[b_] == entries[b_], \
            "editing one entry moved another's slice - every edit re-runs everything"
        assert _registry_slices("#\n" + real)[0] != shared, \
            "editing shared producer code did not move the shared slice"

        # _hash, against every way it used to lie. Each of these is a property
        # that CAN fail, not a restatement of the implementation.
        d = os.path.join(tmp, "inputs")
        os.makedirs(d)
        one, two = os.path.join(d, "a"), os.path.join(d, "b")
        for p_, body in ((one, b"ab"), (two, b"c")):
            with open(p_, "wb") as fh:
                fh.write(body)
        split, folder = _hash((one, two)), _hash((d,))
        for p_, body in ((one, b"a"), (two, b"bc")):
            with open(p_, "wb") as fh:
                fh.write(body)
        assert split != _hash((one, two)), \
            "contents were concatenated with no separator, so (ab)(c) and " \
            "(a)(bc) hashed alike"
        assert folder != _hash((d,)), \
            "a directory input hashed as MISSING - a constant, so never stale"
        folder = _hash((d,))
        os.rename(two, os.path.join(d, "renamed"))
        assert folder != _hash((d,)), \
            "a file renamed inside a directory input did not move the hash"
        present = _hash((one,))
        os.remove(one)
        assert present != _hash((one,)), \
            "an input that vanished did not move the hash"
        # Two declarations with identical content are still two declarations.
        # Over-invalidating costs a re-run; under-invalidating serves a stale
        # number as fresh, which is the failure this module exists to stop.
        for p_ in (one, two):
            with open(p_, "wb") as fh:
                fh.write(b"same")
        assert _hash((one,)) != _hash((two,)), \
            "the declared path is not in the key, so renaming an input is invisible"
        # A file over BIG is keyed on its stat. A same-size rewrite whose mtime
        # is put back afterwards (what `cp -p` and `rsync -a` do) must still
        # move the key.
        big_keep, BIG = BIG, 1
        try:
            st = os.stat(one)
            before = _hash((one,))
            with open(one, "wb") as fh:
                fh.write(b"SAME")               # same size, new bytes
            os.utime(one, ns=(st.st_atime_ns, st.st_mtime_ns))
            assert before != _hash((one,)), \
                "a large input rewritten with its mtime restored kept its key"
        finally:
            BIG = big_keep

        # An unknown id must raise, not return something plausible.
        try:
            get("no.such.measurement")
            raise AssertionError("unknown id must raise")
        except KeyError:
            pass

        # Everything below runs on a PLANTED registry.
        val = [0.5]
        src = ("model/validation/api.py",)
        def planted(fn, n=3, scope="planted scope"):
            return Spec("planted", "u", scope, n, fn, src, src)
        MEASUREMENTS = {"p.ok": planted(lambda: val[0]),
                        "p.new": planted(lambda: 0.42),
                        "p.boom": planted(lambda: 1 / 0)}

        # Rule 2: a failing producer becomes FAILED and carries NO value.
        m = get("p.boom", refresh=True)
        assert m.status == "FAILED" and m.value is None \
            and "ZeroDivisionError" in m.error, m
        # ...and so does one that refuses with sys.exit, which is not an
        # Exception: uncaught, it ended the whole verify on the first missing file.
        MEASUREMENTS["p.exit"] = planted(lambda: sys.exit("missing tracks"))
        e = get("p.exit", refresh=True)
        assert e.status == "FAILED" and "missing tracks" in e.error, e
        del MEASUREMENTS["p.exit"]
        # ...and so does anything that cannot survive the round trip to JSON.
        # A NaN reaching the golden reads as "unchanged" forever after.
        for junk in (float("nan"), float("inf"), None, {1, 2}):
            MEASUREMENTS["p.junk"] = planted(lambda v=junk: v)
            j = get("p.junk", refresh=True)
            assert j.status == "FAILED" and j.value is None, (junk, j)
        del MEASUREMENTS["p.junk"]
        # A producer that reports the n it scored must match the Spec's n: a
        # population that shrank under a typed denominator is a failure.
        MEASUREMENTS["p.seen"] = planted(lambda: (0.5, 2), n=3)
        bad = get("p.seen", refresh=True)
        assert bad.status == "FAILED" and "n=2" in bad.error, \
            f"a producer that scored n=2 was published under n=3: {bad}"
        MEASUREMENTS["p.seen"] = planted(lambda: (0.5, 3), n=3)
        assert get("p.seen", refresh=True).value == 0.5
        del MEASUREMENTS["p.seen"]
        # A cached FAILURE is never served either. It has a value field (None)
        # and a matching key, so only the status check stands between a failure
        # and a caller reading it as an OK measurement with no number.
        broken = [True]
        MEASUREMENTS["p.flaky"] = planted(lambda: 1 / 0 if broken[0] else 0.25)
        assert get("p.flaky").status == "FAILED"
        broken[0] = False
        fixed = get("p.flaky")
        assert fixed.status == "OK" and fixed.value == 0.25, \
            f"a cached FAILED entry was served instead of re-run: {fixed}"
        del MEASUREMENTS["p.flaky"]

        # Rule 3: a cache hit is served only while the key holds, and the key
        # covers the registry, where n and scope live.
        assert get("p.ok").value == 0.5
        val[0] = 0.9
        assert get("p.ok").value == 0.5, "a valid cache entry must not re-run"
        assert get("p.ok", refresh=True).value == 0.9, "--refresh must re-run"
        assert get("p.ok").inputs_hash == _hash(src + src + (REGISTRY,)), \
            "measurements.py is not in the key, so an n correction there would " \
            "be served with the old n from cache"
        # ...and even a cache entry whose key still matches cannot hand back
        # metadata the registry has since corrected: only the VALUE is cached.
        cached = _read_json(CACHE)
        cached["p.ok"] = dict(cached["p.ok"], scope="a stale scope", n=999)
        _write_json(CACHE, cached)
        hit = get("p.ok")
        assert (hit.value == 0.9 and hit.scope == "planted scope"
                and hit.n == 3), f"scope and n came from the cache, not the Spec: {hit}"
        # A cache entry whose key does NOT match must be re-run, not served.
        # This is rule 3 itself, and it was true only by accident before: every
        # entry was written by the running code, so no test ever saw a bad key.
        val[0] = 0.7
        cached["p.ok"] = dict(cached["p.ok"], inputs_hash="not-the-key")
        _write_json(CACHE, cached)
        assert get("p.ok").value == 0.7, \
            "a cache entry whose key no longer matches was served as fresh"

        # UNDECLARED CODE. A producer that loads a repo module nobody declared
        # must still go stale when that module changes - model/engine/pitch.py sat
        # under every metre-space id this way. Planted in model/validation/out (git-
        # ignored), because only files under the repo root are recorded.
        import importlib.util as ilu  # noqa: PLC0415
        dep = os.path.join(ROOT, "model", "validation", "out", f"_demo_dep_{os.getpid()}.py")
        with open(dep, "w") as fh:
            fh.write("X = 1\n")
        mod = ilu.module_from_spec(ilu.spec_from_file_location("_demo_dep", dep))
        mod.__spec__.loader.exec_module(mod)
        sys.modules["_demo_dep"] = mod
        try:
            MEASUREMENTS["p.dep"] = planted(lambda: float(sys.modules["_demo_dep"].X))
            started_keep = STARTED_NS
            # (1) Written AFTER this process started: its bytes may not be what
            # was imported, so the entry gets no hash and is never served.
            assert get("p.dep").value == 1.0
            rel = os.path.relpath(dep, ROOT)
            entry = _read_json(CACHE)["p.dep"]
            assert rel in entry["deps"]["files"], \
                "a module the producer loaded was not recorded"
            assert entry["deps_hash"] is None, \
                "a dep changed mid-run was hashed as if it were the code that ran"
            # (2) The real property, with the mid-run guard out of the way: a
            # fresh process (STARTED_NS moved past the write) hashes the dep,
            # serves it while it holds, and re-runs once it changes. With the
            # guard alone this passed even if _deps_hash were a constant.
            STARTED_NS = time.time_ns() + 1
            assert get("p.dep", refresh=True).value == 1.0
            assert _read_json(CACHE)["p.dep"]["deps_hash"], "an unchanged dep was not hashed"
            mod.X = 5                  # value moves, file does not: must be SERVED
            assert get("p.dep").value == 1.0, "a hashed, unchanged dep was not served"
            with open(dep, "w") as fh:
                fh.write("X = 22\n")
            mod.X = 22
            STARTED_NS = time.time_ns() + 1      # a later process
            assert get("p.dep").value == 22.0, \
                "an undeclared module changed and the cached value was served"
            STARTED_NS = started_keep
            for bad in ({}, {"deps": None, "deps_hash": "x"},
                        {"deps": {"files": "x"}, "deps_hash": "x"}, "not a dict", None):
                assert _fresh_deps(bad) is False, bad   # stale, not a crash
        finally:
            sys.modules.pop("_demo_dep", None)
            MEASUREMENTS.pop("p.dep", None)
            os.remove(dep)

        # THE SAME DEP, REACHED THROUGH A SYMLINKED DIRECTORY, and it must still
        # be recorded. `model/validation/out` and `model/models` are symlinks in
        # every worktree; _deps() resolved each module with realpath alone, so
        # anything under one of them landed OUTSIDE ROOT and was dropped - which
        # is the stale-served-as-fresh direction, and which failed this very
        # assertion on 2026-09-19 and was read for two cycles as a measurement
        # problem rather than an environment shape.
        link = os.path.join(ROOT, "model", "validation", "out", f"_demo_ln_{os.getpid()}")
        away = os.path.join(tmp, "outside")
        os.makedirs(away, exist_ok=True)
        os.symlink(away, link)
        dep2 = os.path.join(link, "_demo_dep2.py")
        with open(dep2, "w") as fh:
            fh.write("X = 3\n")
        m2 = ilu.module_from_spec(ilu.spec_from_file_location("_demo_dep2", dep2))
        m2.__spec__.loader.exec_module(m2)
        sys.modules["_demo_dep2"] = m2
        try:
            MEASUREMENTS["p.dep2"] = planted(lambda: float(sys.modules["_demo_dep2"].X))
            assert get("p.dep2").value == 3.0
            rel2 = os.path.relpath(dep2, ROOT)
            assert rel2 in _read_json(CACHE)["p.dep2"]["deps"]["files"], \
                "a dep behind a symlinked directory was not recorded - realpath " \
                "put it outside ROOT and the whole dependency went unwatched"
            # and the recorded path must still HASH, or recording it bought nothing.
            assert _hash([rel2]) != _hash(["nope/missing.py"]), \
                "the recorded path does not resolve to the file that was imported"
        finally:
            sys.modules.pop("_demo_dep2", None)
            MEASUREMENTS.pop("p.dep2", None)
            os.remove(dep2)
            os.unlink(link)

        # A value is filed under the key its inputs had BEFORE the producer ran:
        # a memoised value computed from an input that was then rewritten must
        # not be served under the new input's key.
        inp = os.path.join(tmp, "inp")
        with open(inp, "w") as fh:
            fh.write("1")
        MEASUREMENTS["p.memo"] = Spec("planted", "u", "s", 1, lambda: 11.0, (inp,), src)
        k_old = keys(["p.memo"])["p.memo"]
        with open(inp, "w") as fh:
            fh.write("2")                       # rewritten while the run was live
        get("p.memo", key=k_old)                # the memo still hands back 11.0
        MEASUREMENTS["p.memo"] = Spec("planted", "u", "s", 1, lambda: 12.0, (inp,), src)
        assert get("p.memo").value == 12.0, \
            "a value computed from the old input was filed under the new input's key"
        del MEASUREMENTS["p.memo"]

        # The guard must catch a moved number, or it is decoration.
        base = {"value": 0.1, "n": 3, "scope": "planted scope"}
        _write_json(GOLDEN, {"values": {"p.ok": base}})
        diffs = verify(refresh=False)
        assert any(x[0] == "p.ok" and "moved" in x[3] for x in diffs), diffs
        assert any(x[0] == "p.new" and "NEW measurement" in x[3] for x in diffs), diffs
        assert any(x[0] == "p.boom" and "FAILED" in x[3] for x in diffs), \
            "a NEW measurement that FAILED read as merely new, hiding its blocker"
        # A NaN already in the golden must not read as unchanged.
        with open(GOLDEN, "w") as fh:
            json.dump({"values": {"p.ok": dict(base, value=float("nan"))}}, fh)
        assert any(x[0] == "p.ok" and "moved" in x[3]
                   for x in verify(refresh=False)), \
            "NaN in the golden read as unchanged: abs(was-now) > 1e-9 is False"
        # A denominator or a scope that moved under an identical value is a
        # different claim (postmortem B2) and must be caught.
        now = get("p.ok", refresh=False).value
        for k, wrong, msg in (("n", 999, "DENOMINATOR changed"),
                              ("scope", "another population", "SCOPE changed")):
            _write_json(GOLDEN, {"values": {"p.ok": dict(base, value=now, **{k: wrong})}})
            assert any(msg in x[3] for x in verify(refresh=False)), (k, msg)
        _write_json(GOLDEN, {"values": {"p.gone": base}})
        assert any("DELETED" in x[3] for x in verify(refresh=False)), \
            "a measurement removed while the golden still lists it must be flagged"
        # A golden caught half-written must report, never read as clean.
        with open(GOLDEN, "w") as fh:
            fh.write('{"values": {"p.ok": {"val')
        assert any("unreadable" in x[3] for x in verify(refresh=False)), \
            "a half-written golden read as clean"

        # A KNOWN FAILURE IS FROZEN AS A FAILURE, not refused. This assertion
        # used to demand the opposite; it was changed deliberately, because
        # refusing made the golden unmovable whenever a gap was registered as a
        # live work item, and the pressure then falls on deleting the
        # registration rather than fixing the gap.
        snapshot("p.boom is a known gap")
        frozen = _read_json(GOLDEN)["values"]["p.boom"]
        assert frozen["status"] == "FAILED" and frozen["value"] is None, frozen
        assert "ZeroDivisionError" in (frozen["error"] or ""), \
            f"the blocker must be frozen too, got {frozen['error']!r}"
        assert not verify(refresh=True), \
            "a failure frozen as a failure must verify clean"
        # ...and the status moving in EITHER direction is a finding.
        MEASUREMENTS["p.boom"] = Spec("recovered", "x", "none", 1,
                                      lambda: 1.0, ("model/validation/api.py",),
                                      ("model/validation/api.py",))
        d = verify(refresh=True)
        assert any("GAP CLOSED" in x[3] for x in d), d
        del MEASUREMENTS["p.boom"]
        try:
            snapshot("   ")
            raise AssertionError("snapshot accepted a blank reason")
        except SystemExit:
            pass
        val[0] = 0.55       # the cache still holds 0.7 under a matching key
        snapshot("first")
        assert _read_json(GOLDEN)["values"]["p.ok"]["value"] == 0.55, \
            "snapshot froze the cached value instead of re-running the producer"
        snapshot("second")
        assert [h["reason"] for h in _read_json(GOLDEN)["history"]][-2:] == \
            ["first", "second"], "the history lost a snapshot or its reason"
        assert not verify(refresh=False), "a fresh snapshot must verify clean"

        # Per-id snapshot: accepting one id must not re-bless another, and the
        # history must keep the value it replaced.
        MEASUREMENTS["p.later"] = planted(lambda: 0.33)
        assert id_drift() == (["p.later"], []), id_drift()
        val[0] = 0.66           # p.ok drifted; an --only p.later must not accept it
        snapshot("accept p.later", only=["p.later"])
        g = _read_json(GOLDEN)
        assert g["values"]["p.ok"]["value"] == 0.55, \
            "a per-id snapshot re-froze an id it was not given"
        assert g["values"]["p.later"]["value"] == 0.33 and id_drift() == ([], [])
        snapshot("accept the drift", only=["p.ok"])
        assert _read_json(GOLDEN)["history"][-1]["replaced"]["p.ok"]["value"] \
            == 0.55, "the history does not keep the value a snapshot replaced"
        assert _read_json(GOLDEN)["history"][-1]["reason"] == "accept the drift", \
            "a history entry is filed under another snapshot's reason"
        try:
            snapshot("nothing named", only=[])
            raise AssertionError("an empty --only re-froze every id")
        except ValueError:
            pass
        del MEASUREMENTS["p.later"]
        assert id_drift() == ([], ["p.later"])
        snapshot("p.later was deleted", only=["p.later"])
        assert id_drift() == ([], []), "a deletion could not be accepted by name"

        # Rule 4: the lint must FAIL on a bare decimal, and the marker must be
        # the only thing that silences it.
        docs = os.path.join(tmp, "docs")
        os.makedirs(docs)
        plant = os.path.join(docs, "planted.md")
        def lint(text):
            with open(plant, "w") as fh:
                fh.write(text)
            return [x for x in check_docs([plant]) if x[4]]
        assert lint(f"{STRICT}\nthe gate is worth 0.0122\n"), \
            "the bare-decimal lint does not fail on a bare decimal"
        assert not lint("the gate is worth 0.0122\n"), \
            "a document that has not opted in must not block the gate"
        assert not lint(f"{STRICT}\nByteTrack scores 0.42 {MARK} published -->\n"), \
            "the marker does not exempt a legitimate bare number"
        assert not lint(f"{STRICT}\nthe gate is api://p.ok\n"), \
            "a cited number must not be flagged as bare"
        # ...but a citation exempts only ITS value, not the rest of the line.
        _write_json(GOLDEN, {"values": {"p.ok": {"value": 0.55}}})
        assert lint(f"{STRICT}\nthe gate is api://p.ok and costs 0.0122\n"), \
            "an uncited decimal rode unlinted beside an api:// citation"
        assert not lint(f"{STRICT}\nthe gate is api://p.ok (0.55)\n"), \
            "the cited id's own value was flagged as a bare copy"
        assert lint("see api://no.such.thing\n"), \
            "an unresolvable citation must block"
        assert not lint(f"{STRICT}\n```\n0.0122\n```\n"), \
            "a fenced block is command output, not prose"
        assert not lint(f"```\n{STRICT}\n```\nthe gate is worth 0.0122\n"), \
            "the opt-in marker inside a code fence opted the whole file in - a " \
            "document explaining the lint flagged itself, 195 times"
        try:
            _scan([])
            raise AssertionError("a lint that scans nothing must say so, not "
                                 "report clean (postmortem A7)")
        except RuntimeError:
            pass
        # The default scan must see a page in a subfolder, and not the memory store.
        for rel in ("audits/a.md", "memory/m.md"):
            os.makedirs(os.path.dirname(os.path.join(docs, rel)), exist_ok=True)
            open(os.path.join(docs, rel), "w").close()
        assert [os.path.relpath(p, docs) for p in doc_pages(tmp)] == \
            ["audits/a.md", "planted.md"], "the doc scan missed a subfolder"
        # ...and a value that moved must name the documents still quoting it,
        # in either form. B4 was a fraction in the code (0.707) and a percentage
        # in seven documents (70.7%), which is why the percent form is matched.
        lint("worth 0.0122, team was 70.7%, and the tracker scores 0.3634\n")
        hits = stale_copies(
            [("ab.offpitch.hota",     # an ab.* value is a DICT, not a scalar
              {"mean": 0.012179416562299891, "k": 7, "p": 0.0078125}, 0.0, "moved"),
             ("team.accuracy.detection", 0.7071, 0.6967, "moved")], [plant])
        assert sorted(h[2] for h in hits) == ["0.0122", "70.7%"], hits
    finally:
        CACHE, GOLDEN, MEASUREMENTS = keep
    print(f"ok  ({len(MEASUREMENTS)} measurements, all scoped, all with data "
          f"inputs; a failing, exiting, NaN or None producer returns FAILED not "
          f"a number; the hash discriminates inputs, directories, missing and "
          f"rewritten large files; one registry entry's edit moves only its own "
          f"slice; an undeclared module that changes re-runs its producer; a "
          f"value is filed under its pre-run key; the guard catches a moved "
          f"value, a NaN, a new id, a changed denominator, a changed scope and a "
          f"deletion; a per-id snapshot leaves the rest alone, keeps what it "
          f"replaced, refuses an empty list and accepts a deletion by name; the "
          f"bare-decimal lint fails and its marker exempts)")


# The registry, imported LAST so everything it pulls from this module exists.
# A lazy-proxy dict was tried first and recursed; this is the boring version and
# it is the one that works.
from measurements import MEASUREMENTS, Spec  # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--get", metavar="ID")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--check-docs", action="store_true")
    ap.add_argument("--snapshot", metavar="REASON",
                    help="freeze current values as accepted truth; REASON required")
    ap.add_argument("--only", metavar="ID", nargs="+",
                    help="with --snapshot: freeze just these ids, leave the rest")
    ap.add_argument("--verify", action="store_true",
                    help="the whole gate: demo, then resolve all (re-running "
                         "only ids whose key moved; --refresh re-runs every one) "
                         "and fail on any unexplained movement or broken citation")
    ap.add_argument("--ids", action="store_true",
                    help="seconds: fail if registry and golden hold different ids")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        demo()
    elif a.snapshot is not None:
        v = snapshot(a.snapshot, a.only)
        n = len(a.only) if a.only else len(v)
        print(f"snapshot: {n} measurement(s) frozen — {a.snapshot}")
    elif a.ids:
        new, gone = id_drift()
        for mid in new:
            print(f"  registered, not in the golden:  {mid}")
        for mid in gone:
            print(f"  in the golden, not registered:  {mid}")
        if new or gone:
            print(f"{len(new) + len(gone)} id(s) out of step - the push gate is "
                  f"red and any field citing a new id renders not_evaluated. "
                  f"Accept by name: api.py --snapshot \"why\" --only <id>...")
        sys.exit(1 if new or gone else 0)
    elif a.verify:
        # demo() first, always. It runs in a second against a planted registry,
        # and a guard that has not checked itself before checking everything
        # else is postmortem §A: five self-checks that could not fail.
        demo()
        # From the cache, re-running only what went stale. This used to be a
        # full re-run - 2 h 07 min on 2026-09-10 - which only became safe to
        # drop once the key covered ground truth, weights and packages; see
        # `_hash`, `_key` and the IN_*/PKG_* inputs in measurements.py.
        diffs = verify(refresh=a.refresh)
        for mid, was, now, why in diffs:
            # Fixed widths so the REASON stays in one column however wide the
            # values are; `--get <id>` prints the value in full.
            print(f"  {mid:<34s} {str(was)[:22]:>22s} -> {str(now)[:22]:<22s} {why}")
        for f, ln, lit, mid in stale_copies(diffs):
            print(f"  {f}:{ln}  still says {lit} — {mid} MOVED")
        # --verify used to PRINT "no broken citations" while the check was
        # conditional on --ci and had not run. A guard reporting on a check it
        # skipped is postmortem A7, so it always runs now.
        bad = [x for x in check_docs() if x[4]]
        for f, ln, kind, detail, _ in bad:
            print(f"  {f}:{ln}  {kind}: {detail}")
        if not diffs and not bad:
            print(f"verified: all {len(MEASUREMENTS)} measurements match the "
                  f"golden; no broken citations")
        else:
            print(f"\n{len(diffs)} difference(s) against the golden, {len(bad)} "
                  f"doc finding(s). Either the change was intended - then "
                  f"re-snapshot WITH A REASON - or it is a bug.")
        # One line per run, so model/validation/dashboard.py can say when the golden
        # last reproduced and at which commit. The exit code alone is gone the
        # moment the shell moves on. model/validation/out/ is gitignored: local state.
        dirty = bool(subprocess.run(["git", "diff", "HEAD", "--quiet"], cwd=ROOT,
                                    env=_git_env()).returncode)
        os.makedirs(os.path.join(ROOT, "model", "validation", "out"), exist_ok=True)
        with open(os.path.join(ROOT, "model", "validation", "out", "api-verify.log"), "a") as fh:
            fh.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": _commit(),
                                 "dirty": dirty, "ok": not (diffs or bad),
                                 "diffs": len(diffs), "doc_findings": len(bad)}) + "\n")
        sys.exit(1 if (diffs or bad) else 0)
    elif a.list:
        for mid, s in MEASUREMENTS.items():
            print(f"{mid:<34s} {s.unit:<8s} n={s.n:<5d} {s.desc}")
    elif a.check_docs:
        found = check_docs()
        bad = [x for x in found if x[4]]
        for f, ln, kind, detail, _ in bad:
            print(f"{f}:{ln}  {kind}: {detail}")
        backlog = collections.Counter(x[0] for x in found if not x[4])
        worst = ", ".join(f"{f} {n}" for f, n in backlog.most_common(3))
        print(f"{len(bad)} blocking; {sum(backlog.values())} uncited decimals "
              f"across {len(backlog)} documents not yet marked {STRICT} "
              f"(worst: {worst})")
        sys.exit(1 if bad else 0)
    elif a.get or a.all:
        ids = [a.get] if a.get else list(MEASUREMENTS)
        ks = keys(ids)
        for mid in ids:
            print(get(mid, a.refresh, key=ks[mid]).line())
    else:
        ap.print_help()
