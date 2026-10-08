"""`check.py status`: what is frozen, what is stale, what is running. READ-ONLY, no producer runs.

    .venv/bin/python model/validation/check.py status [--section PREFIX] [--json]

Every value printed comes from a file named in the footer (with its mtime) or from `ps`/`lsof`;
a missing source prints "unavailable (<path>)", never a blank or a zero.

RULES - each one mechanical, so a reader can check it against the code below
    last frozen  the most recent api-golden.json history entry whose `ids` list names the id, or
                 else the most recent BLANKET entry (`ids` == "all" or absent): api.snapshot() without
                 --only rewrites every registered id, so a blanket entry froze every id it did not name.
    state        api.py's own freshness test, imported not copied: api._key() (the keys api.keys()
                 computes, threaded here only for speed) vs the cache's inputs_hash, then
                 api._fresh_deps() on the cache entry. Nothing is run, so:
                   OK      cache fresh, status OK, value == golden
                   MOVED   cache fresh, status OK, value != golden (api.py --verify would flag it)
                   STALE   no cache entry, key moved, or a loaded module/package changed since
                   FAILED  cache fresh and FAILED, golden OK
                   GAP     golden froze it as FAILED (a registered known gap)
                   UNFROZEN registered but absent from the golden
    shipped?     STORED  a declared input is a stored prediction (model/validation/out/, data/jobs/,
                         model/validation/fixtures/, a ball_cands_ cache), WHATEVER the code tuple
                         says: declaring a producer module (detectors, detsweep) proves the key
                         imports it, not that the value was produced by it - that was the loophole
                         the ball ids read YES through. The key cannot see the engine, so it is NOT
                         a live regression guard. `:on/:off/:unst/:miss` is the worst shipped_config
                         stamp over its inputs named *tracks*.json (dashboard.badge) - tracks.split.json
                         included, which `.tracks.json` alone missed.
                 NO      the scope states conf/stride/overlap/tile and one differs from
                         shipped_config.product_config(), or it says "not shippable"/"not the product".
                 YES     not STORED, >= 2 of those keys stated, all equal to product_config().
                 ?       anything else. Never guessed.
"""
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "model" / "validation"
sys.path[:0] = [str(VAL), str(ROOT / "model" / "engine")]
RUNLOG = VAL / "out" / "check-runs.jsonl"
PRODUCERS = ("model/engine/track.py", "model/engine/tracking.py", "model/engine/stitch.py",
             "model/engine/detectors.py", "model/engine/detsweep.py")
ACTIVE = ("tracking", "identifying", "calibrating", "queued")
PROC_PAT = ("model/validation/", "model/engine/", "app/server/jobs.py", "api.py", "train")
SOURCES = {}                        # path -> mtime string, or None if unavailable


def _src(p):
    p = Path(p)
    SOURCES[str(p)] = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)) if p.exists() else None
    return p.exists()


def _json(p):
    if not _src(p):
        return None
    try:
        return json.loads(Path(p).read_text())
    except ValueError:
        SOURCES[str(p)] = None
        return None


def last_frozen(mid, golden):
    for h in reversed(golden.get("history", [])):
        ids = h.get("ids", "all")
        if ids == "all" or (isinstance(ids, list) and mid in ids):
            return h
    return {}


def _stored(p):
    return (p.startswith(("model/validation/out/", "data/jobs/", "model/validation/fixtures/"))
            or Path(p).name.startswith("ball_cands_"))


def _is_tracks(p):
    return fnmatch.fnmatch(Path(p).name, "*tracks*.json")


def shipped(spec, product, stamp):
    ins = [os.path.relpath(p, ROOT) if os.path.isabs(p) else str(p) for p in spec.inputs]
    if any(_stored(p) for p in ins):
        tracks = [stamp(p) for p in ins if _is_tracks(p)]
        import dashboard
        b = dashboard.badge(tracks)
        return "STORED" + (f":{b.lower()[:4]}" if b else "")
    s = spec.scope
    said = {"conf": re.search(r"\bconf (\d*\.\d+)", s), "stride": re.search(r"\bstride (\d+)", s),
            "overlap": re.search(r"\boverlap (\d*\.\d+)", s), "tile": re.search(r"\bt(?:ile )?(\d{3,4})\b", s)}
    said = {k: float(m.group(1)) for k, m in said.items() if m}
    if any(v != product.get(k) for k, v in said.items()) or re.search(r"not shippable|not the product", s, re.I):
        return "NO"
    return "YES" if len(said) >= 2 else "?"


def benchmarks(section=None):
    """-> rows, or a one-line reason the table is unavailable."""
    import api
    import shipped_config as SC
    from dashboard import _fmt, _state_of
    golden = _json(api.GOLDEN)
    cache = _json(api.CACHE)
    _src(ROOT / api.REGISTRY)
    if golden is None:
        return f"unavailable ({os.path.relpath(api.GOLDEN, ROOT)})"
    reg = {m: s for m, s in api.MEASUREMENTS.items() if not section or m.split(".")[0] == section}
    ids = sorted(set(reg) | {m for m in golden.get("values", {}) if not section or m.split(".")[0] == section})
    product, stamps = SC.product_config(), {}

    def stamp(p):
        if p not in stamps:
            stamps[p] = "MISSING" if not (ROOT / p).exists() else _state_of(ROOT / p, product)
        return stamps[p]

    tracks = {os.path.relpath(p, ROOT) if os.path.isabs(p) else str(p)
              for s in reg.values() for p in s.inputs if _is_tracks(p)}
    with ThreadPoolExecutor(8) as ex:            # api._key per id == api.keys(); threads only for speed
        keys = dict(zip(reg, ex.map(lambda m: api._key(m, reg[m]), reg)))
        list(ex.map(stamp, sorted(tracks)))
    rows = []
    for mid in ids:
        g, spec = golden.get("values", {}).get(mid), reg.get(mid)
        c = (cache or {}).get(mid) or {}
        fresh = spec is not None and c.get("inputs_hash") == keys[mid] and api._fresh_deps(c)
        if spec is None:
            state = "DELETED"                    # in the golden, no longer registered
        elif g is None:
            state = "UNFROZEN"
        elif g.get("status") == "FAILED":
            state = "GAP"
        elif cache is None:
            state = "UNKNOWN"                    # no cache file to compare against
        elif not fresh:
            state = "STALE"
        elif c.get("status") != "OK":
            state = "FAILED"
        else:
            state = "OK" if json.dumps(c.get("value"), sort_keys=True) == json.dumps(g.get("value"), sort_keys=True) else "MOVED"
        h = last_frozen(mid, golden) if g is not None else {}
        rows.append({"id": mid, "section": mid.split(".")[0],
                     "value": None if g is None else (g.get("value") if g.get("status") == "OK" else "FAILED"),
                     "shown": "-" if g is None else (_fmt(g.get("value")) if g.get("status") == "OK" else "(gap)"),
                     "n": (g or {}).get("n"), "scope": (g or {}).get("scope") or (spec.scope if spec else ""),
                     "shipped": shipped(spec, product, stamp) if spec else "?",
                     "frozen": str(h.get("at") or "")[:10], "commit": h.get("commit") or "",
                     "reason": h.get("reason") or "", "state": state})
    SOURCES[f"{len(stamps)} tracks.json stamps (_perf.config)"] = "read now"
    return rows


def sections(names, goldens, section=None):
    runs = {}
    if _src(RUNLOG):
        for line in RUNLOG.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            runs[(r.get("section"), r.get("mode"))] = r
    out = []
    for name in names:
        if section and name != section:
            continue
        g = _json(goldens[name]) if name in goldens else None
        row = {"section": name, "golden": os.path.relpath(goldens[name], ROOT) if name in goldens else None,
               "frozen": g and g.get("date"), "commit": g and (g.get("git_head") or "")[:7],
               "uncommitted": g and len(g.get("uncommitted_engine_files") or []),
               "reason": g and g.get("reason"),
               "runs": [r for (s, _), r in sorted(runs.items(), key=lambda kv: kv[1].get("t", "")) if s == name]}
        out.append(row)
    return out


def _sh(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _tail(pathlike):
    """Last non-empty line of the newest log/csv/jsonl a process names, or None."""
    p = Path(pathlike)
    if p.is_dir():
        c = [f for pat in ("*.log", "*.csv", "*.jsonl", "*/*.log", "*/*.csv") for f in p.glob(pat)]
        p = max(c, key=lambda f: f.stat().st_mtime) if c else None
    if not p or not p.is_file() or p.suffix not in (".log", ".csv", ".jsonl", ".txt"):
        return None
    with open(p, "rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 8192))
        lines = [ln for ln in fh.read().decode("utf-8", "replace").splitlines() if ln.strip()]
    return (str(p.relative_to(ROOT)) if ROOT in p.resolve().parents else str(p), lines[-1][:160]) if lines else None


def processes():
    ps = _sh(["ps", "-Ao", "pid=,ppid=,etime=,ucomm=,command="])
    SOURCES["ps -Ao pid,ppid,etime,ucomm,command"] = "live" if ps else None
    rows, byid = [], {}
    for line in ps.splitlines():
        p = line.split(None, 4)
        if len(p) == 5:
            byid[p[0]] = p
    for pid, ppid, et, exe, cmd in byid.values():
        if "python" not in exe.lower() or int(pid) == os.getpid() or not any(k in cmd for k in PROC_PAT):
            continue
        cwd = next((ln[1:] for ln in _sh(["lsof", "-a", "-d", "cwd", "-p", pid, "-Fn"]).splitlines()
                    if ln.startswith("n")), "")
        if str(ROOT) not in cwd and str(ROOT) not in cmd:
            continue
        args = cmd.split()
        script = next((a for a in args if a.endswith(".py")), args[1] if len(args) > 1 else cmd)
        base = Path(cwd) if cwd else ROOT
        tail = next((t for a in args[1:] for t in [_tail(base / a.split("=")[-1])] if t), None)
        rows.append({"pid": int(pid), "elapsed": et, "script": script,
                     "args": " ".join(args[args.index(script) + 1:] if script in args else args[1:])[:120],
                     # macOS `caffeinate CMD` execs CMD and leaves a caffeinate CHILD holding the assertion
                     "caffeinate": "caffeinate" in byid.get(ppid, ["", "", "", ""])[3]
                     or any(q[1] == pid and q[3] == "caffeinate" for q in byid.values()), "cwd": cwd,
                     "log": tail})
    return rows


def jobs():
    from paths import DATA
    d = Path(os.environ.get("FUTTRAFFIC_DATA", DATA)) / "jobs"
    SOURCES[f"{d}/*/job.json"] = "read now" if d.exists() else None
    out = []
    for f in sorted(d.glob("*/job.json")) if d.exists() else ():
        try:
            st = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if st.get("status") in ACTIVE:
            t0 = st.get("stage_started") or st.get("updated")
            out.append({"id": str(st.get("id", f.parent.name))[:8], "name": st.get("display_name"),
                        "status": st.get("status"), "detector": st.get("detector"),
                        "attempts": st.get("attempts"),
                        "stage_elapsed_s": round(time.time() - t0) if isinstance(t0, (int, float)) else None})
    return out


def _cut(s, w):
    s = "-" if s is None or s == "" else str(s)
    return s if len(s) <= w else s[:w - 1] + "~"


def main(argv, names, goldens):
    """names: check.SECTIONS; goldens: {section: its golden path}."""
    section = argv[argv.index("--section") + 1] if "--section" in argv and argv.index("--section") + 1 < len(argv) else None
    bench, secs, procs, js = benchmarks(section), sections(names, goldens, section), processes(), jobs()
    if "--json" in argv:
        print(json.dumps({"benchmarks": bench, "sections": secs, "processes": procs, "jobs": js,
                          "sources": SOURCES}, indent=1, default=str))
        return 0
    print("BENCHMARKS  (values from the golden; state from api.py's cache keys - nothing was re-run)")
    if isinstance(bench, str):
        print(f"  {bench}")
    else:
        cols = (("id", 36), ("value", 22), ("n", 6), ("scope", 26), ("shipped?", 11), ("frozen", 10),
                ("commit", 8), ("reason", 34), ("state", 8))
        fmt = lambda vals: "  " + " ".join(_cut(v, w).ljust(w) for v, (_, w) in zip(vals, cols))  # noqa: E731
        sec = None
        for r in bench:
            if r["section"] != sec:
                sec = r["section"]
                print(f"\n  [{sec}]")
                print(fmt([c for c, _ in cols]))
            print(fmt([r["id"], r["shown"], r["n"], r["scope"], r["shipped"], r["frozen"], r["commit"],
                       r["reason"], r["state"]]))
        if not bench:
            print(f"  no id with section prefix {section!r}")
    print("\nSECTION CHECKS  (check.py goldens; runs from " + os.path.relpath(RUNLOG, ROOT) + ")")
    for s in secs:
        if s["frozen"] is None:
            print(f"  {s['section']}: golden unavailable ({s['golden']})")
        else:
            dirty = f" + {s['uncommitted']} uncommitted engine files" if s["uncommitted"] else ""
            print(f"  {s['section']}: golden frozen {s['frozen']} @ {s['commit']}{dirty} - {_cut(s['reason'], 90)}")
        for r in s["runs"] or [None]:
            print("    last run: unavailable (" + os.path.relpath(RUNLOG, ROOT) + ")" if r is None else
                  f"    last {r.get('mode') or '(none)':10s} {r.get('t')}  exit {r.get('exit')}  @ {r.get('git')}")
    print("\nRUNNING WORK  (ps, read-only)")
    for p in procs:
        print(f"  pid {p['pid']:<7d} {p['elapsed']:>11s}  {'caffeinate ' if p['caffeinate'] else ''}{p['script']} {p['args']}")
        if p["log"]:
            print(f"      {p['log'][0]}: {p['log'][1]}")
    if not procs:
        print("  no repo python process matched")
    print("  app jobs in " + "/".join(ACTIVE) + ":")
    for j in js:
        el = "?" if j["stage_elapsed_s"] is None else f"{j['stage_elapsed_s'] // 60}m{j['stage_elapsed_s'] % 60:02d}s"
        print(f"    {j['id']}  {_cut(j['name'], 36):36s} {j['status']:<12s} {el:>9s}  detector={j['detector']}  attempts={j['attempts']}")
    if not js:
        print("    none")
    print("\nsources: " + "; ".join(f"{os.path.relpath(k, ROOT) if k.startswith(str(ROOT)) else k} "
                                    f"({v})" if v else f"unavailable ({k})" for k, v in SOURCES.items()))
    return 0


def demo():
    g = {"history": [{"ids": ["a"], "commit": "1"}, {"commit": "2"}, {"ids": ["b"], "commit": "3"}]}
    assert last_frozen("a", g)["commit"] == "2", "a blanket entry after a per-id one froze it later"
    assert last_frozen("b", g)["commit"] == "3" and last_frozen("c", g)["commit"] == "2"

    class S:
        def __init__(s, scope, inputs=("x.csv",), code=()):
            s.scope, s.inputs, s.code = scope, inputs, code
    prod = {"conf": 0.5, "stride": 2, "overlap": 0.25, "tile": 1024}
    assert shipped(S("coreml, conf 0.3, stride 1"), prod, None) == "NO"
    assert shipped(S("conf 0.5, stride 2"), prod, None) == "YES"
    assert shipped(S("conf 0.5"), prod, None) == "?", "one stated key is not enough for YES"
    assert shipped(S("conf 0.5, stride 2", ("data/jobs/x/calibration.json",)), prod, None) == "STORED"
    seen = []
    def stamp(p):
        seen.append(p)
        return "UNSTAMPED"
    assert shipped(S("conf 0.5, stride 2", ("data/jobs/x/tracks.split.json",), ("model/engine/detectors.py",)),
                   prod, stamp) == "STORED:unst", \
        "a stored input next to a producer module must read STORED, stamped from its tracks.split.json"
    assert seen == ["data/jobs/x/tracks.split.json"], "tracks.split.json was not stamped"
    assert shipped(S("conf 0.5, stride 2", ("model/validation/fixtures/ball/m0.json.gz",)),
                   prod, None) == "STORED", "a committed fixture is a stored prediction"
    assert shipped(S("ORACLE MAP - not shippable, conf 0.5, stride 2"), prod, None) == "NO"
    print("check_status ok")


if __name__ == "__main__":
    demo()
