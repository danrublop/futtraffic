"""A check is not trusted until it has been seen to fail.

WHY
    On 2026-09-12 the only self-check a reviewer could fully trust was the one it
    broke three ways and watched catch all three. `model/engine/quality.py`'s empty-file
    assert passed for every possible input and nobody had tried to make it fail.
    A check never observed failing is a hypothesis about a check.
    harness-design (not published) §D.

HOW - AND WHY NOT BY EDITING FILES
    Each mutation swaps ONE module attribute for a broken version at runtime,
    runs that module's self-check, and restores it. No source file is touched, so
    this is safe in a tree where other agents are working - the alternative,
    mutating source in place, is `git stash` with extra steps, and that
    already happened once today.

    Before any mutation, the UNMUTATED self-check must pass. If it already fails,
    "it failed on the mutation" proves nothing, and the row reads BASELINE-FAILED.

EXPECTATIONS ARE DECLARED, INCLUDING THE GAPS
    A mutation declares `killed` (the check must catch it) or `survives` (a KNOWN
    gap, written down with the finding that established it). The run fails on any
    mismatch in either direction: a `killed` mutation that survives is a check
    that stopped working; a `survives` mutation that gets killed is a gap someone
    closed, and the declaration must be updated so this file stays true.

    .venv/bin/python model/validation/mutate.py          # run every mutation
    .venv/bin/python model/validation/mutate.py --only "verify_claims:"   # rows whose label contains it
    .venv/bin/python model/validation/mutate.py --demo   # the runner's own self-check
"""
import contextlib
import importlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
for sub in ("model/validation", "model/engine"):
    sys.path.insert(0, str(ROOT / sub))


@dataclass
class Mutation:
    label: str
    module: str
    attr: str
    broken: Callable      # original -> replacement
    selfcheck: str        # the module's self-check function
    expect: str           # "killed" | "survives"
    why: str = ""         # for a declared gap: the finding that established it


def _raise_on(func):
    """Run a self-check quietly; True if it raised."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            func()
        except BaseException:  # noqa: BLE001 - AssertionError, SystemExit, anything
            return True
    return False


def _labels_only_filter(original):
    """n66: filter `labels` and leave boxes/area misaligned - the failure the sentinel-class
    design exists to prevent, which demo() survived until it recorded more than labels."""
    return ("labels",)


def _off_pitch_bucket(original):
    """n67: draw ignore regions from the OFF-PITCH bucket - the bench, which N43b and N47
    spent the far-tile programme removing and which the rule explicitly forbids."""
    return "off-pitch"


def _no_region_floor(original):
    """n67: disarm VOID (f), so an arm with no regions reads as the arm."""
    return 0


def _negatives_unfiltered(original):
    """N64's bug, re-planted: select mined tiles WITHOUT requiring them to hold no ball.

    This is what shipped. The rule called them hard negatives, `negatives()` ranked the
    pool by false-positive confidence, and 865 of 916 contained the ball - so the arm
    tested twice the data rather than suppression, and the file's own VOID (e) passed
    because it counted firing tiles (12,001) not negative ones (1,762).
    """
    def patched(n=None):
        import json  # noqa: PLC0415
        import n64_ball_negatives as N  # noqa: PLC0415
        n = N.N_NEG if n is None else n
        hits = [h for h in json.load(open(N.MINED))["hits"] if h["conf"] >= N.MINE_CONF]
        return [h["file_name"] for h in hits][:n]
    return patched


def _loo_leaks(original):
    """N59's leave-one-clip-out, selecting on ALL clips including the held-out one.

    Worth +0.0883 on N59's own table (in-sample +0.0627 against held-out -0.0256), and
    the in-line `assert out not in rest` could not see it because it restated the line
    above it.
    """
    def patched(tab, clips, arms=None):
        import n59_gates as N  # noqa: PLC0415
        arms = arms or sorted(N.subsets(), key=lambda s: (len(s), sorted(s)))
        best = max(arms, key=lambda s: N._pooled(tab, clips, s)[0])
        return [(h, best, clips) for h in clips]
    return patched


def _verdict_always_present(original):
    """N63's verdict, stuck on PRESENT. demo() used to hold its own copy of the
    expression, so flipping the real one changed nothing it could see."""
    return lambda a, b, n: "PRESENT"


def _empty_reads_fail(original):
    """The FIX for R1's finding, applied as a mutation. The empty-file assert
    accepts it and accepts the bug, which is exactly what makes it vacuous."""
    def patched(data, cal, fps=None):
        out = original(data, cal, fps)
        if not (data.get("detections") or []):
            out = dict(out, verdict="fail")
        return out
    return patched


def _nearest_pick(original):
    """Re-introduce the first N26 reference inverse: any grid point of a pixel's plateau."""
    import numpy as np
    from scipy.spatial import cKDTree

    def patched(G, px, max_px):
        tree = cKDTree(np.asarray(px, float))

        def mapper(uv):
            d, i = tree.query(np.asarray(uv, float).reshape(-1, 2))
            return np.asarray(G, float)[i], d <= max_px
        return mapper
    return patched


def _hull_switch(original):
    """Re-introduce the first N26 tps21: the correction switched off outside the clicks' hull."""
    import numpy as np

    def patched(p, X):
        correct, checks = original(p, X)

        def switched(q):
            out, inside = correct(q)
            return np.where(inside[:, None], out, np.asarray(q, float)), inside
        return switched, checks
    return patched


def _tps_off(original):
    """The corrected calibration that corrects nothing: its homography alone."""
    import numpy as np

    def patched(self, pts_px):
        return np.asarray(self.inner.to_pitch(pts_px), dtype=np.float64).reshape(-1, 2)
    return patched


def _mirror_blind(original):
    """The ground-truth loader with its handedness gate blind: every feed reads as true."""
    import st2_team_gt as G  # noqa: PLC0415

    def f(*a, **k):
        keep, G.feed_mirrored = G.feed_mirrored, (lambda d: False)
        try:
            return original(*a, **k)
        finally:
            G.feed_mirrored = keep
    return f


def _defaults_with_copy(original):
    """Re-introduce 0a5a7cd: give track.run a COPY of the gate set again.

    BY NAME, not by position. This used to set `original[-1]` - the LAST default - which
    was prior_gates until 2026-09-27, when a new trailing parameter (`ball_model`) made
    "last" the wrong one. The mutation then changed something track._selfcheck does not
    read and SURVIVED, so a row declared `killed` was silently measuring nothing. The gate
    caught it, which is the point of declaring the expected outcome; the lesson is that a
    guard indexing a signature positionally rots the moment the signature grows.
    """
    import inspect  # noqa: PLC0415

    import track  # noqa: PLC0415
    withdef = [p for p in inspect.signature(track.run).parameters.values()
               if p.default is not inspect.Parameter.empty]
    assert len(withdef) == len(original), (len(withdef), len(original))
    i = [n for n, p in enumerate(withdef) if p.name == "prior_gates"]
    assert i, "track.run has no prior_gates parameter - this mutation targets nothing"
    d = list(original)
    d[i[0]] = ("pitch", "height")
    return tuple(d)


def _tagline_only(original):
    """Re-introduce the hook's old trigger: a tag LINE, blind to its block.

    The grep `.githooks/pre-push` used to carry. It is the whole hole: rule 5's
    continuation - clip, split, conf, n= - changes no line carrying a sigil,
    so `conf 0.05` -> `conf 0.5` under an unchanged tag read as prose and
    skipped every heavy gate.
    """
    import re  # noqa: PLC0415
    import verify_claims as V  # noqa: PLC0415

    def patched(diff):
        return bool(re.search(rf"^[-+].*@({V.SIGILS})\b\s", diff, re.M))
    return patched


def _git_vars_through(original):
    """Let GIT_* reach git again. An exported GIT_DIR BEATS cwd=, so every
    scanner reads the OTHER tree's index and prints a green count over a file
    set that is not the one being pushed - 731 of 765, measured 2026-09-20."""
    import os  # noqa: PLC0415

    def patched(base=None):
        return dict(base or os.environ)
    return patched


def _realpath_only_deps(original):
    """Re-introduce api._deps' realpath-only resolution.

    Any repo file reached through a symlinked directory - model/validation/out
    and model/models are symlinks in every worktree - resolves outside ROOT and
    is dropped from the recorded dependencies. Dropping an input is the
    direction that serves a stale number as fresh, and it announces nothing.
    """
    import os  # noqa: PLC0415

    def patched():
        d = original()
        real = os.path.realpath(api.ROOT) + os.sep
        return {"files": [f for f in d["files"]
                          if os.path.realpath(os.path.join(api.ROOT, f)).startswith(real)],
                "dists": d["dists"]}
    return patched


def _old_token_present(original):
    """The matcher before 2026-09-29: no sign, exponent or anchoring rule."""
    import re  # noqa: PLC0415
    return lambda value, text: re.search(r"(?<![\d.])" + re.escape(value) + r"(?![\d])", text) is not None


def _disk_is_tracked(original):
    """doclinks before 2026-09-29: a target resolves if it is on THIS disk."""
    import os  # noqa: PLC0415

    def patched(root):
        out = set()
        for base, _, files in os.walk(root):
            rel = os.path.relpath(base, root)
            rel = "" if rel == "." else rel
            out.add(rel)
            out.update(os.path.join(rel, f) if rel else f for f in files)
        return out
    return patched


def _annassign_blind(original):
    """shipped_config before 2026-09-29: `NAME: T = v` is not an assignment."""
    import re  # noqa: PLC0415
    return lambda path, name: [] if re.search(rf"^{name}\s*:", path.read_text(), re.M) else original(path, name)


def _syntax_skipped(original):
    """shipped_config before 2026-09-29: a scanned file that does not parse is skipped."""
    def patched(path, name):
        try:
            return original(path, name)
        except SystemExit:
            return []
    return patched


import api  # noqa: E402 - after the sys.path setup above

MUTATIONS = [
    # 2026-09-20, the four guards written during the gate audit. Each mutation
    # breaks the PROPERTY, not the code: the old tag-line-only trigger, GIT_*
    # reaching git, realpath-only dependency resolution, and a check blind to
    # the mode bit that makes a symlink a symlink.
    Mutation("verify_claims: the heavy-gate trigger sees the tag line only",
             "verify_claims", "touches_tag", _tagline_only, "_selfcheck", "killed"),
    Mutation("verify_claims: GIT_* reaches git again, so a scan reads another tree",
             "verify_claims", "git_env", _git_vars_through, "_selfcheck", "killed"),
    # 2026-09-26, the three that SURVIVED a review of that day's experiments.
    # mutate.py covered the scanners and no experiment module at all, so "every VOID gets
    # an assertion that it fires" was enforced for the gate and for nothing being gated.
    # 2026-09-29, the checker holes a review confirmed. Each row re-plants the
    # pre-fix behaviour and must be killed by the file's own self-check.
    Mutation("verify_claims: a non-zero exit verifies a value it printed", "verify_claims", "judge",
             lambda o: (lambda c, blob, rc: o(c, blob, 0 if rc > 0 else rc)), "_selfcheck", "killed"),
    Mutation("verify_claims: prose after the command is part of the command", "verify_claims", "cut_cmd",
             lambda o: (lambda source: source.strip()), "_selfcheck", "killed"),
    Mutation("verify_claims: a command that fails sh -n is run anyway", "verify_claims", "syntax_error",
             lambda o: (lambda cmd: False), "_selfcheck", "killed"),
    Mutation("verify_claims: the matcher ignores sign, exponent and small-integer anchoring",
             "verify_claims", "token_present", _old_token_present, "_selfcheck", "killed"),
    Mutation("verify_claims: a range hyphen (1.7-5.0) is read as a sign", "verify_claims", "LEFT",
             lambda o: r"(?<![\d.\-])", "_selfcheck", "killed"),
    Mutation("verify_claims: an integer claim misses its own N.0000 spelling", "verify_claims", "INT_ZEROS",
             lambda o: "", "_selfcheck", "killed"),
    Mutation("verify_claims: only name=N declares a label (no `of 7`, no quoted label)",
             "verify_claims", "split_label",
             lambda o: (lambda v: (lambda m: (m[1], r"""(?:['"]?\s*[=:]\s*|\s+)""", m[2]) if m
                                   else (None, "", v))(__import__("verify_claims").DECLARED.fullmatch(v))),
             "_selfcheck", "killed"),
    Mutation("bare_markers: #example tags are unlimited", "bare_markers", "example_rise",
             lambda o: (lambda *a, **k: None), "demo", "killed"),
    Mutation("derived: UNEVALUABLE is counted and never fails", "derived", "failures",
             lambda o: (lambda rows, baseline=0: o(rows, 10 ** 9)), "demo", "killed"),
    Mutation("shipped_config: a **kwargs call site resolves to the defaults", "shipped_config", "_spread",
             lambda o: (lambda call: []), "demo", "killed"),
    Mutation("shipped_config: an annotated constant copy is invisible", "shipped_config", "_ast_constant",
             _annassign_blind, "demo", "killed"),
    Mutation("shipped_config: a scanned file with a SyntaxError is skipped", "shipped_config", "_ast_constant",
             _syntax_skipped, "demo", "killed"),
    Mutation("shipped_config: check_prose lists files under the ambient GIT_DIR", "shipped_config",
             "V.git_env", _git_vars_through, "demo", "killed"),
    Mutation("protocol: a rule loosened before its result still reads PROVEN", "protocol", "rule_text",
             lambda o: (lambda *a, **k: []), "demo", "killed"),
    Mutation("protocol: SAME_COMMIT never blocks, however new the page", "protocol", "blocks",
             lambda o: (lambda status, *a, **k: status in ("AFTER", "MISSING", "RULE_CHANGED")),
             "demo", "killed"),
    Mutation("protocol: NOT_PROVABLE excuses any case, listed or stale", "protocol", "recorded",
             lambda o: (lambda *a, **k: "RECORDED"), "demo", "killed"),
    Mutation("doclinks: a link resolves if its target is on this disk", "doclinks", "tracked",
             _disk_is_tracked, "selfcheck", "killed"),
    Mutation("triggers: a section's imports are not followed past the first module", "triggers",
             "_closure", lambda o: (lambda code, data: set(code) | set(data)), "selfcheck", "killed"),
    Mutation("n64: mined negatives are not required to hold no ball",
             "n64_ball_negatives", "negatives", _negatives_unfiltered, "demo", "killed"),
    Mutation("n59: leave-one-clip-out selects on the held-out clip too",
             "n59_gates", "_loo", _loo_leaks, "demo", "killed"),
    Mutation("n63: the keeper-composition verdict is always PRESENT",
             "n63_train_composition", "_verdict", _verdict_always_present, "demo", "killed"),
    # 2026-09-28: the two newest experiment modules, added after a review found n67's demo()
    # survived five planted defects and n66's survived the misalignment its own comment names.
    Mutation("n66: forward() filters labels only, leaving boxes misaligned",
             "n66_ignore", "PER_INSTANCE", _labels_only_filter, "demo", "killed"),
    Mutation("n67: the ignore set draws from the off-pitch bucket",
             "n67_ignore_arm", "BUCKET", _off_pitch_bucket, "demo", "killed"),
    Mutation("n67: the region floor accepts an empty arm",
             "n67_ignore_arm", "MIN_REGIONS", _no_region_floor, "demo", "killed"),
    Mutation("api: a dependency behind a symlinked directory goes unrecorded",
             "api", "_deps", _realpath_only_deps, "demo", "killed"),
    Mutation("lint_truth: the tracked-symlink check is blind to the mode bit",
             "lint_truth", "SYMLINK_MODE", lambda o: "100644", "demo", "killed"),
    Mutation("derived: every @D holds", "derived", "holds",
             lambda o: (lambda *a, **k: True), "demo", "killed"),
    Mutation("derived: no approximate identity is ever false", "derived", "APPROX_REL",
             lambda o: 1e9, "demo", "killed"),
    Mutation("shipped_config: harnesses always agree", "shipped_config", "check_harnesses",
             lambda o: (lambda *a, **k: []), "demo", "killed"),
    Mutation("shipped_config: copied constants always agree", "shipped_config", "check_constants",
             lambda o: (lambda *a, **k: []), "demo", "killed"),
    Mutation("shipped_config: every comparison is clean", "shipped_config", "compare_guard",
             lambda o: (lambda *a, **k: ("OK", [])), "demo", "killed"),
    Mutation("bare_markers: nothing is ever bare", "bare_markers", "count_bare",
             lambda o: (lambda text: 0), "demo", "killed"),
    Mutation("track: run() carries its own copy of the gate set again", "track", "run.__defaults__",
             _defaults_with_copy, "_selfcheck", "killed"),
    Mutation("track: a cache detected under other settings replays anyway", "track",
             "_settings_mismatch", lambda o: (lambda *a, **k: []), "_selfcheck", "killed"),
    Mutation("dist_eval: N22's rule always names identity", "dist_eval", "n22_verdict",
             lambda o: (lambda res: {k: {"driver": "IDENTITY"} for k in
                                     ("pooled_ratio_matched", "pooled_sprint_ratio", "pooled_hsr_ratio")}),
             "_selfcheck", "killed"),
    Mutation("dist_eval: detection crediting hands a whole fragment to its majority owner", "dist_eval",
             "_credited",
             lambda o: (lambda trk, owners, *a, **k: o(trk, [max(set(x for x in owners if x), key=owners.count)
                                                             if any(owners) else None] * len(owners), *a, **k)),
             "_selfcheck", "killed"),
    Mutation("dist_eval: N24's rule never voids", "dist_eval", "n24_verdict",
             lambda o: (lambda res: dict(o(res), follows="PHYSICAL OVER-READ - position work")),
             "_selfcheck", "killed"),
    Mutation("dist_eval: N25's rule never voids", "dist_eval", "n25_verdict",
             lambda o: (lambda res: dict(o(res), follows="ESTIMATOR")),
             "_selfcheck", "killed"),
    Mutation("stats: the adopted filter runs across a gap", "dist_eval", "stats._runs",
             lambda o: (lambda t, real: o(t, real | True)),
             "_selfcheck", "killed"),
    Mutation("stats: the Kalman is scored beyond the filter's support", "dist_eval", "stats.butter_support",
             lambda o: (lambda t, real: real),
             "_selfcheck", "killed"),
    Mutation("dist_eval: N26's rule never voids", "dist_eval", "n26_verdict",
             lambda o: (lambda res: dict(o(res), follows="MAP + CLICKS CARRY IT")),
             "_selfcheck", "killed"),
    Mutation("dist_eval: N26 ignores what leaving out a clip does", "dist_eval", "n26_verdict",
             lambda o: (lambda res: (lambda v: dict(v, follows=v["follows"].split(" (FRAGILE")[0]))(o(res))),
             "_selfcheck", "killed"),
    Mutation("dist_eval: the reference inverse picks a plateau member again", "dist_eval", "_smooth_inverse",
             _nearest_pick, "_selfcheck", "killed"),
    Mutation("dist_eval: the map correction switches off at the hull again", "dist_eval", "_tps_correction",
             _hull_switch, "_selfcheck", "killed"),
    Mutation("dist_eval: the map correction never reports a fold", "dist_eval", "_tps_correction",
             lambda o: (lambda p, X: (lambda c, k: (c, dict(k, folds=False)))(*o(p, X))),
             "_selfcheck", "killed"),
    Mutation("dist_eval: a map arm keeps rows its map cannot place", "dist_eval", "_oracle_rows",
             lambda o: (lambda rows, owner, gt, cal, mode, feet=None, feedpx=None, mapper=None: o(
                 rows, owner, gt, cal, mode, feet, feedpx,
                 (lambda uv: (mapper(uv)[0], [True] * len(uv))) if mapper else None)),
             "_selfcheck", "killed"),
    Mutation("calibrators: the 21-click correction bends nothing", "calibrators", "TpsCorrected.to_pitch",
             _tps_off, "_demo", "killed"),
    Mutation("stats: speed ignores the calibration's own Jacobian", "dist_eval", "stats.kalman_ground",
             lambda o: (lambda xy, uv, t, H, sigma_px=None, q=None, jac=None: o(xy, uv, t, H, sigma_px, q)),
             "_selfcheck", "killed"),
    Mutation("dist_eval: N27 ships whatever it measures", "dist_eval", "n27_verdict",
             lambda o: (lambda res: dict(o(res), follows="SHIP")),
             "_selfcheck", "killed"),
    Mutation("dist_eval: N27's movement scale is never read", "dist_eval", "_chord_slope_arrays",
             lambda o: (lambda *a, **k: (1.0, 1, 1.0)),
             "_selfcheck", "killed"),
    Mutation("dist_eval: N27 ignores what leaving out a clip does", "dist_eval", "_n27_share",
             lambda o: (lambda speed, ship, corr, clips: o(speed, ship, corr, sorted(speed[ship]))),
             "_selfcheck", "killed"),
    Mutation("st2_team_gt: a mirrored feed reads as true", "st2_team_gt", "feed_mirrored",
             lambda o: (lambda d: False), "_selfcheck", "killed"),
    Mutation("dist_eval: the loader serves a mirrored feed as positions", "dist_eval", "stc_gt",
             _mirror_blind, "_selfcheck", "killed"),
    Mutation("plausibility: N28 catches whatever it measures", "plausibility", "n28_verdict",
             lambda o: (lambda res: dict(o(res), follows="CATCHES THE PRODUCT")), "_selfcheck", "killed"),
    Mutation("plausibility: a PSV-99 needs no second above 4 m/s", "plausibility", "PSV_MIN_S",
             lambda o: 0.0, "_selfcheck", "killed"),
    Mutation("plausibility: the product is read under the speed cap", "plausibility", "_track_speeds",
             lambda o: (lambda tr, cal, fps: (lambda r: r and dict(r, product=r["shipped"]))(o(tr, cal, fps))),
             "_selfcheck", "killed"),
    Mutation("st1_gnss: the georeference cannot see the pitch's scale", "st1_gnss", "_affine",
             lambda o: (lambda pitch, lonlat: dict(o(pitch, lonlat), scale=[1.0, 1.0])), "_selfcheck", "killed"),
    Mutation("st1_gnss: the time offset is never scanned", "st1_gnss", "_scan",
             lambda o: (lambda grid, ft, feet, fn, offsets: o(grid, ft, feet, fn, [0.0] * len(offsets))),
             "_selfcheck", "killed"),
    Mutation("calibrators: a folded correction stays switched on", "calibrators", "TpsCorrected.checks",
             lambda o: (lambda self, step_m=1.0: dict(o(self, step_m), folds=False)),
             "_demo", "killed"),
    Mutation("dist_eval: N23's rule always blames the map", "dist_eval", "n23_verdict",
             lambda o: (lambda res: {k: {"driver": "MAP"} for k in
                                     ("pooled_sprint_ratio", "pooled_hsr_ratio", "pooled_ratio_matched")}),
             "_selfcheck", "killed"),
    Mutation("dist_eval: the identity oracle never splits a fragment", "dist_eval", "_oracle_rows",
             lambda o: (lambda rows, owner, gt, cal, mode, *a, **k: o(rows, owner, gt, cal,
                                                                     "drop" if mode == "identity" else mode, *a, **k)),
             "_selfcheck", "killed"),
    Mutation("dashboard: every id reads as on-config", "dashboard", "badge",
             lambda o: (lambda states: "ON" if states else None), "demo", "killed"),
    Mutation("dashboard: no tracks file carries a stamp", "dashboard", "stamp_of",
             lambda o: (lambda path: None), "demo", "killed"),
    Mutation("protocol: every rule is proven pre-registered", "protocol", "preregistered",
             lambda o: (lambda *a, **k: ("PROVEN", "x", "y")), "demo", "killed"),
    Mutation("quality: an empty tracks file reads FAIL instead of incomplete", "quality", "check",
             _empty_reads_fail, "demo", "survives",
             why="R1, 2026-09-12: `assert empty['verdict'] in ('fail','incomplete')` "
                 "accepts both, so the empty-file case is not pinned"),
]


def _setattr_path(mod, path, value):
    head, _, tail = path.rpartition(".")
    setattr(getattr(mod, head) if head else mod, tail, value)


def _getattr_path(mod, path):
    obj = mod
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def run_mutation(m: Mutation) -> str:
    """One mutation. Dotted attributes such as `run.__defaults__` are supported."""
    mod = importlib.import_module(m.module)
    check = getattr(mod, m.selfcheck)
    if _raise_on(check):
        return "BASELINE-FAILED"
    original = _getattr_path(mod, m.attr)
    _setattr_path(mod, m.attr, m.broken(original))
    try:
        return "KILLED" if _raise_on(check) else "SURVIVED"
    finally:
        _setattr_path(mod, m.attr, original)


def run_all(mutations=MUTATIONS) -> list[tuple[Mutation, str, bool]]:
    out = []
    for m in mutations:
        got = run_mutation(m)
        ok = (got == "KILLED" and m.expect == "killed") or \
             (got == "SURVIVED" and m.expect == "survives")
        out.append((m, got, ok))
    return out


def demo():
    """The runner itself: a real check is killed, a vacuous one survives, and a
    check that already fails is refused rather than counted."""
    import types

    fake = types.ModuleType("fake_mut")
    fake.double = lambda x: 2 * x

    def real_check():
        assert fake.double(3) == 6

    def vacuous_check():
        assert fake.double(3) in (6, 7, 8, 0)   # accepts the bug and the fix

    def broken_check():
        raise AssertionError("already failing")

    fake.real_check, fake.vacuous_check, fake.broken_check = real_check, vacuous_check, broken_check
    sys.modules["fake_mut"] = fake
    try:
        kill = Mutation("x", "fake_mut", "double", lambda o: (lambda x: 0), "real_check", "killed")
        vac = Mutation("x", "fake_mut", "double", lambda o: (lambda x: 0), "vacuous_check", "survives")
        base = Mutation("x", "fake_mut", "double", lambda o: (lambda x: 0), "broken_check", "killed")
        assert run_mutation(kill) == "KILLED"
        assert run_mutation(vac) == "SURVIVED", "a check that accepts the bug must survive the mutation"
        assert run_mutation(base) == "BASELINE-FAILED", "a failing check cannot vouch for anything"
        assert fake.double(3) == 6, "every mutation is restored"
        # a declared expectation that is wrong is itself a failure
        (_, got, ok), = run_all([Mutation("x", "fake_mut", "double", lambda o: (lambda x: 0),
                                          "vacuous_check", "killed")])
        assert got == "SURVIVED" and not ok
    finally:
        del sys.modules["fake_mut"]
    print("mutate.py ok")


def main() -> int:
    if "--demo" in sys.argv:
        demo()
        return 0
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    rows = run_all([m for m in MUTATIONS if only is None or only in m.label])
    bad = 0
    for m, got, ok in rows:
        mark = "  " if ok else "!!"
        print(f"{mark} {got:15s} expect {m.expect:8s} {m.label}")
        if m.why and (got == "SURVIVED"):
            print(f"      known gap: {m.why}")
        bad += not ok
    killed = sum(g == "KILLED" for _, g, _ in rows)
    print(f"\n{killed}/{len(rows)} killed; {bad} outcome(s) differ from their declaration")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
