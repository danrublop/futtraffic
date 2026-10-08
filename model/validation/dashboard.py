"""A results page nobody can type into: its measurements and verdicts come from files and exit codes.

WHY
    docs/INDEX.md is generated because a hand-written index is an unverified claim
    about the directory. This is the same rule applied to results. An agent's
    summary of where the model stands is a claim; the golden, the gates' own
    output, the stamps on the artefacts and the process table are not.

WHAT IT READS - AND THE ONLY THINGS IT READS
    model/validation/api-golden.json         every registered value: n, status, scope, the snapshot that froze it
    model/validation/measurements.py         each id's input files, for the on-config badge
    model/validation/claims.log              verify_claims' own run log
    model/validation/out/api-verify.log      api.py --verify's own run log
    each gate's exit and last line    shipped_config, derived, bare_markers, protocol, mutate, lint_truth
    _perf.config on tracks files       through shipped_config.diff_artefact: ON / OFF / UNSTAMPED
    ps, git, and *.log under model/validation/out touched in the last hour

    It never opens docs/ (memory notes included), commit messages or chat, and it shows a
    gate's LAST LINE only - lint_truth and derived quote document lines in their
    bodies. demo() plants a number in a doc, in a memory note and in a gate's body,
    and asserts none of them reaches the page. Running commands and log tails ARE
    shown verbatim, labelled as what is running, not what was measured.

WHAT IS STILL PROSE
    The scope, description and snapshot reason beside a value were written into
    the registry by whoever registered it. They are committed and change only
    through a snapshot with a reason - but they are text, not measurement, and the
    page labels them as registered text.

    .venv/bin/python model/validation/dashboard.py            # write model/validation/out/dashboard.html once
    .venv/bin/python model/validation/dashboard.py --watch    # a fresh render process every 30 s
    .venv/bin/python model/validation/dashboard.py --demo
"""
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model" / "validation"))
import shipped_config as SC  # noqa: E402 - diff_artefact and config_of, reused and never copied

OUT = ROOT / "model" / "validation" / "out" / "dashboard.html"
GATES = ("shipped_config", "derived", "bare_markers", "protocol",
         "mutate", "lint_truth")
RANK = {"ON": 1, "UNSTAMPED": 2, "OFF": 3, "MISSING": 4, "UNREADABLE": 5}
PROCS = ("model/validation/", "model/engine/", "app/server/", "import track")


def _sh(args, cwd, timeout=900) -> tuple[int, str]:
    try:
        p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, f"{type(e).__name__}: {e}"


def git_state(root) -> dict:
    head = _sh(["git", "rev-parse", "--short", "HEAD"], root)[1].strip()
    diff = _sh(["git", "diff", "HEAD"], root)[1]
    new = _sh(["git", "ls-files", "--others", "--exclude-standard"], root)[1]
    return {"head": head, "dirty": bool(diff.strip()),
            "branch": _sh(["git", "branch", "--show-current"], root)[1].strip(),
            # What the gates' verdicts depend on: the commit, every uncommitted
            # edit, and every new file (bare_markers scans those too).
            "key": _key(root, head + diff + new, new.split())}


def _key(root, text: str, untracked: list[str]) -> str:
    """The gates' cache key: commit, uncommitted diff, and the CONTENTS of new
    files - bare_markers scans those, so an edit to one must re-run the gates."""
    h = hashlib.sha1(text.encode())
    for f in untracked:
        p = Path(root) / f
        if p.is_file() and p.stat().st_size < 2_000_000:
            h.update(p.read_bytes())
    return h.hexdigest()


def stamp_of(path):
    """A tracks file's _perf.config, without parsing its detections.

    model/engine/track.py writes _perf after the detections, so it sits in the tail.
    ponytail: a 64 KB tail read; a file with a larger key after _perf (the
    identity stage's report) falls back to shipped_config's full parse.
    """
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 65536))
        tail = fh.read().decode("utf-8", "replace")
    i = tail.rfind('"_perf": ')
    if i < 0:
        return SC.config_of(path)
    perf, _ = json.JSONDecoder().raw_decode(tail, i + len('"_perf": '))
    return (perf or {}).get("config")


def badge(states):
    """The worst state over an id's tracks inputs; None when it reads none."""
    return max(states, key=RANK.get) if states else None


def snapshot_of(mid, golden) -> dict:
    """The last snapshot that wrote `mid`: per-id entries name it, blanket ones say 'all'."""
    for h in reversed(golden.get("history", [])):
        ids = h.get("ids")
        if ids == "all" or (isinstance(ids, list) and mid in ids):
            return h
    return {"at": golden.get("updated"), "commit": golden.get("commit"),
            "reason": golden.get("reason")}


def last_run(path):
    """verify_claims logs one row per claim with the run's start time as `t`:
    the last run is the last block of rows sharing one."""
    if not path.exists():
        return None
    rows, t = [], None
    for line in path.read_text(errors="replace").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("t") != t:
            rows, t = [], r.get("t")
        rows.append(r)
    if not rows:
        return None
    return {"t": t, "git": rows[0].get("git"), "dirty": rows[0].get("dirty"),
            "counts": dict(Counter(r.get("status") for r in rows))}


def last_line_json(path):
    if not path.exists():
        return None
    lines = [ln for ln in path.read_text(errors="replace").splitlines() if ln.strip()]
    try:
        return json.loads(lines[-1]) if lines else None
    except ValueError:
        return None


def gates(root, key, names=GATES) -> dict:
    """Each gate's exit code and LAST LINE, re-run only when the commit or the tree
    moved. Not the full output: lint_truth and derived quote the lines of the
    documents they check, so a gate's body is where typed numbers reach a page."""
    cache = root / "model" / "validation" / "out" / "dashboard.gates.json"
    if cache.exists():
        old = json.loads(cache.read_text())
        if old.get("key") == key:
            return old
    rows = []
    for g in names:
        t0 = time.time()
        code, out = _sh([sys.executable, f"model/validation/{g}.py"], root)
        lines = [ln for ln in out.splitlines() if ln.strip()]
        rows.append({"gate": g, "exit": code, "summary": lines[-1] if lines else "(no output)",
                     "secs": round(time.time() - t0, 1)})
    res = {"key": key, "ran_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows}
    cache.write_text(json.dumps(res))
    return res


def running() -> list[dict]:
    # ucomm is the executable's own name, one token: a shell whose command line
    # merely MENTIONS python is not a python process.
    code, out = _sh(["ps", "-eo", "pid,etime,rss,ucomm,command"], ROOT, timeout=30)
    rows = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        pid, et, rss, exe, cmd = parts
        if "python" in exe.lower() and any(p in cmd for p in PROCS) and "dashboard.py" not in cmd:
            rows.append({"pid": pid, "elapsed": et, "rss_mb": int(rss) // 1024, "cmd": cmd[-220:]})
    return rows


def recent_logs(root, window_s=3600) -> list[dict]:
    """*.log one or two levels under model/validation/out, touched within the window."""
    base = root / "model" / "validation" / "out"
    out = []
    for p in sorted(list(base.glob("*.log")) + list(base.glob("*/*.log"))):
        age = time.time() - p.stat().st_mtime
        if age <= window_s:
            with open(p, "rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - 8192))
                lines = [ln for ln in fh.read().decode("utf-8", "replace").splitlines() if ln.strip()]
            out.append({"path": str(p.relative_to(root)), "age_s": int(age), "tail": lines[-4:]})
    return out


def _state_of(path, product) -> str:
    """ON/OFF/UNSTAMPED for one tracks file - UNREADABLE when it cannot be parsed,
    which is what a file track.run is still writing looks like."""
    try:
        return SC.diff_artefact(stamp_of(path), product)[0]
    except (ValueError, OSError):
        return "UNREADABLE"


def artefacts(root, product) -> list[dict]:
    """Every tracks file under model/validation/out/stc2025, grouped by variant, stamp states counted."""
    d = root / "model" / "validation" / "out" / "stc2025"
    groups: dict = {}
    for p in sorted(d.glob("*.tracks.json")) if d.exists() else ():
        # {seq}{variant}.tracks.json - the sequence is the LEADING digits only;
        # a variant such as .p21 has digits of its own.
        variant = re.match(r"\d*(.*)", p.name[:-len(".tracks.json")]).group(1) or "(none)"
        state = _state_of(p, product)
        g = groups.setdefault(variant, {"variant": variant, "files": 0, "states": Counter()})
        g["files"] += 1
        g["states"][state] += 1
    return sorted(groups.values(), key=lambda g: g["variant"])


def _e(x) -> str:
    return html.escape(str(x))


def _fmt(v) -> str:
    """For reading. The exact golden value rides along in the cell's title."""
    if isinstance(v, dict):
        return ", ".join(f"{k}={_fmt(x)}" for k, x in v.items())
    return f"{v:.4g}" if isinstance(v, float) else str(v)


#: The app's tier language (app/frontend/src/components/tier.tsx) mapped onto what
#: this page says, under the same rule: blue is an instrument reading and never
#: approval, green only for a genuine pass, amber the one thing worth an eye.
TONE = {"ON": "ok", "OK": "ok", "OFF": "warn", "UNSTAMPED": "none", "MISSING": "bad",
        "UNREADABLE": "bad", None: "none"}


def _badge(text, tone) -> str:
    return f'<span class="badge t-{tone}">{_e(text)}</span>'


#: The app's own font files, copied beside the page by main(): a file:// page
#: cannot reach up into app/frontend/ in every browser, and a missing one only
#: falls back to the system face.
FONTS = {"inter-latin-wght-normal.woff2":
         "app/frontend/node_modules/@fontsource-variable/inter/files/inter-latin-wght-normal.woff2",
         "geist-mono-latin.woff2":
         "app/frontend/node_modules/next/dist/next-devtools/server/font/geist-mono-latin.woff2"}

#: app/frontend/src/app/globals.css's dark ladder - the app pins <html class="dark">
#: with no theme toggle, so this page is dark only too. One deviation, said:
#: --grid-line mixes into --card, because these tables sit on cards, not panels.
CSS = """
@font-face{font-family:"Inter Variable";src:url("fonts/inter-latin-wght-normal.woff2") format("woff2");font-weight:100 900;font-display:swap}
@font-face{font-family:"Geist Mono";src:url("fonts/geist-mono-latin.woff2") format("woff2");font-weight:100 900;font-display:swap}
:root{color-scheme:dark;--ink:#f4f4f4;--ink-soft:#d6d6d6;--slate:#8a8a8a;--mute:#6e6e6e;
--bg:#121212;--panel:#1a1a1a;--panel-2:#0f0f0f;--card:#232323;--card-hover:#2c2c2c;
--hairline:rgba(255,255,255,.08);--hairline-strong:rgba(255,255,255,.14);
--grid-line:color-mix(in oklab,var(--ink) 8%,var(--card));
--accent:#4a9eff;--success:#34d399;--danger:#f87171;--warn:#fbbf24;--ok:#60a5fa;
--font-ui:"Inter Variable","Inter",ui-sans-serif,system-ui,-apple-system,sans-serif;
--font-mono:"Geist Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
*{box-sizing:border-box;scrollbar-width:thin;scrollbar-color:var(--hairline-strong) transparent}
html,body{margin:0;background:var(--bg);color:var(--ink)}
body{font:14px/1.45 var(--font-ui);font-feature-settings:"cv05" 1,"ss01" 1;-webkit-font-smoothing:antialiased}
.bar{position:sticky;top:0;z-index:1;height:44px;display:flex;align-items:center;gap:6px;padding:0 16px;background:var(--bg);border-bottom:1px solid var(--hairline)}
.brand{font-size:13px;font-weight:500;color:var(--ink-soft)}.crumb{font-size:13px;color:var(--slate)}
.bar .meta{margin-left:auto;font:12px var(--font-mono);color:var(--slate);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
main{background:var(--panel);min-height:calc(100vh - 44px);padding:24px 16px 48px}
.page{max-width:64rem;margin:0 auto;display:grid;gap:16px;min-width:0}
h1,h2{font-weight:600;letter-spacing:-.01em;margin:0}h1{font-size:18px;letter-spacing:-.2px}
.sub{font-size:14px;color:var(--slate);margin:2px 0 0}
.card{background:var(--card);border:1px solid var(--hairline);border-radius:12px;padding:16px 0;display:grid;gap:12px;min-width:0;overflow:hidden}
.card>*{padding:0 16px;min-width:0}.card>.wrap{padding:0 8px}
.card h2{font-size:14px;font-weight:500;letter-spacing:0;display:flex;align-items:center;gap:8px}
.desc{font-size:13px;color:var(--slate);margin:2px 0 0}
.eyebrow{text-transform:uppercase;letter-spacing:.08em;font-size:13px;font-weight:500;color:var(--slate)}
.readout,code,pre{font-family:var(--font-mono);font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.readout{font-size:13px}code{font-size:12px;background:var(--card-hover);border-radius:4px;padding:1px 4px}
.muted{color:var(--slate)}.soft{color:var(--ink-soft)}
.wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:14px}
th{height:40px;padding:0 8px;text-align:left;vertical-align:middle;font-size:12px;font-weight:500;white-space:nowrap;color:var(--slate);border-bottom:1px solid var(--grid-line)}
td{padding:8px;vertical-align:middle;white-space:nowrap;border-bottom:1px solid var(--grid-line)}
tbody tr:last-child td{border-bottom:0}
tbody tr:not(.group):hover td{background:color-mix(in srgb,var(--card-hover) 50%,transparent)}
th.num,td.num{text-align:right}td.cmd{font:12px var(--font-mono);color:var(--ink-soft);word-break:break-all;white-space:normal}td.wraps{white-space:normal;min-width:14rem}td.wraps pre{max-width:36rem}td.wraps>.readout{white-space:nowrap}
tr.group td{padding:18px 8px 6px}
.badge{display:inline-flex;align-items:center;height:16px;padding:0 6px;border-radius:9999px;border:1px solid var(--hairline-strong);font-size:10px;font-weight:500;letter-spacing:.025em;text-transform:uppercase;white-space:nowrap;vertical-align:middle}
.t-ok{color:var(--ok);border-color:color-mix(in srgb,var(--ok) 40%,transparent)}
.t-good{color:var(--success);border-color:color-mix(in srgb,var(--success) 40%,transparent)}
.t-warn{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 40%,transparent)}
.t-bad{color:var(--danger);background:color-mix(in srgb,var(--danger) 12%,transparent);border-color:transparent}
.t-none{color:var(--slate);border-style:dashed}
.dot{width:6px;height:6px;border-radius:50%;background:var(--accent);display:inline-block}
details>summary{cursor:pointer;color:var(--slate);font-size:13px;list-style:none}
details>summary::-webkit-details-marker{display:none}
details>summary::before{content:"\\203A";display:inline-block;width:12px;transition:transform 150ms cubic-bezier(.22,1,.36,1)}
details[open]>summary::before{transform:rotate(90deg)}
pre{font-size:12px;line-height:1.5;color:var(--ink-soft);background:var(--panel-2);border:1px solid var(--hairline);border-radius:8px;padding:10px 12px;margin:8px 0 0;white-space:pre-wrap;word-break:break-word}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--slate)}.legend .badge{margin-right:6px}
.log+.log{margin-top:12px}
"""


def _card(title, desc, body, live=False) -> str:
    dot = '<span class="dot"></span>' if live else ""
    return (f'<section class="card"><div><h2>{dot}{title}</h2><p class="desc">{desc}</p></div>'
            f"{body}</section>")


def _table(head, rows) -> str:
    th = "".join(f'<th class="num">{_e(c[1:])}</th>' if c.startswith(">") else f"<th>{_e(c)}</th>"
                 for c in head)
    return f'<div class="wrap"><table><thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def render(root, refresh=None, run_gates=True, registry=None, gate_names=GATES) -> str:
    root = Path(root)
    golden = json.loads((root / "model" / "validation" / "api-golden.json").read_text())
    if registry is None:
        import measurements  # noqa: PLC0415
        registry = measurements.MEASUREMENTS
    git = git_state(root)
    product = None

    def state(p):
        nonlocal product
        full = Path(p) if os.path.isabs(p) else root / p
        if not full.exists():
            return "MISSING"
        product = product or SC.product_config()
        return _state_of(full, product)

    h = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         '<meta name="theme-color" content="#121212"><title>Harness · FutTraffic</title>']
    if refresh:
        h.append(f'<meta http-equiv="refresh" content="{int(refresh)}">')
    h.append(f"<style>{CSS}</style></head><body>")
    dirty = " + uncommitted" if git["dirty"] else ""
    h.append(f'<header class="bar"><span class="brand">FutTraffic</span><span class="crumb">/ Harness</span>'
             f'<span class="meta">{_e(git["branch"])} @ {_e(git["head"])}{dirty} · '
             f'{_e(time.strftime("%H:%M:%S"))}</span></header><main><div class="page">')
    h.append('<div><h1>Harness</h1><p class="sub">Measurements, gate verdicts and stamps are read from '
             "files and exit codes, never from documents or memory. Registry text is labelled; running "
             "commands and log tails are shown verbatim.</p></div>")

    # 1. running now - Signal Blue is machine state, which is exactly what this is
    procs, logs = running(), recent_logs(root)
    rows = [f'<tr><td class="readout">{_e(p["pid"])}</td><td class="readout">{_e(p["elapsed"])}</td>'
            f'<td class="num readout">{p["rss_mb"]} MB</td><td class="cmd">{_e(p["cmd"])}</td></tr>'
            for p in procs]
    body = _table(("PID", "Elapsed", ">RSS", "Command"), rows) if procs else ""
    body += "".join(f'<div class="log"><div class="readout soft">{_e(lg["path"])} '
                    f'<span class="muted">· written {lg["age_s"]} s ago</span></div>'
                    f'<pre>{_e(chr(10).join(lg["tail"]))}</pre></div>' for lg in logs)
    n = len(procs)
    h.append(_card("Running now", f"{n} repo python process{'es' if n != 1 else ''}, and logs under "
                   "<code>model/validation/out</code> written in the last hour - command lines and log "
                   "tails shown VERBATIM. They say what is running, not what was measured.",
                   body, live=bool(procs)))

    # 2. verification freshness
    rows = []
    for label, run in (("verify_claims", last_run(root / "model" / "validation" / "claims.log")),
                       ("api.py --verify", last_line_json(root / "model" / "validation" / "out" / "api-verify.log"))):
        if run is None:
            rows.append(f'<tr><td class="readout">{label}</td><td>{_badge("no run logged", "none")}</td>'
                        '<td class="muted">—</td><td class="muted">—</td><td class="muted">—</td></tr>')
            continue
        at_head = bool(run.get("git")) and git["head"].startswith(str(run["git"])[:7])
        tag = (("at HEAD", "good") if at_head and not run.get("dirty") else
               ("at HEAD, uncommitted", "warn") if at_head else ("stale", "warn"))
        res = (" · ".join(f"{k} {v}" for k, v in run["counts"].items()) if "counts" in run else
               ("reproduced" if run.get("ok") else f"{run.get('diffs')} diff(s), {run.get('doc_findings')} doc finding(s)"))
        rows.append(f'<tr><td class="readout">{label}</td><td>{_badge(*tag)}</td>'
                    f'<td class="readout muted">{_e(str(run.get("t")).replace("T", " "))}</td>'
                    f'<td class="readout">{_e(run.get("git"))}</td><td class="readout soft">{_e(res)}</td></tr>')
    h.append(_card("Does the golden still reproduce", "The last run of each check, and whether it ran "
                   "against the commit you are on.", _table(("Check", "State", "Ran", "Commit", "Result"), rows)))

    # 3. gates
    if run_gates:
        gr = gates(root, git["key"], gate_names)
        rows = [f'<tr><td class="readout">{_e(r["gate"])}</td>'
                f'<td>{_badge("pass", "good") if r["exit"] == 0 else _badge('fail · exit ' + str(r['exit']), 'bad')}</td>'
                f'<td class="readout soft">{_e(r["summary"][:140])}</td>'
                f'<td class="num readout muted">{r["secs"]} s</td></tr>' for r in gr["rows"]]
        h.append(_card("Gates", f"Each gate's exit code and its own last line, last run {_e(gr['ran_at'])} - "
                       "run one to see its full output. They re-run whenever the commit, an "
                       "uncommitted edit or a new file changes.",
                       _table(("Gate", "Result", "Its last line", ">Time"), rows)))

    # 4. artefacts
    arts = artefacts(root, product or (SC.product_config() if (root / "model/validation/out/stc2025").exists() else None))
    if arts:
        rows = [f'<tr><td class="readout">{_e(a["variant"])}</td><td class="num readout">{a["files"]}</td><td>'
                + " &nbsp; ".join(f'{_badge(s, TONE[s])} <span class="readout soft">{c}</span>'
                                  for s, c in sorted(a["states"].items(), key=lambda kv: RANK[kv[0]]))
                + "</td></tr>" for a in arts]
        h.append(_card("Tracks on disk, against what ships", "<code>model/validation/out/stc2025/*.tracks.json</code> "
                       "by variant: each file's own stamp, diffed against what <code>app/server/jobs.py</code> ships.",
                       _table(("Variant", ">Files", "Stamps"), rows)))

    # 5. registered measurements, grouped by family like a team sheet
    vals = golden.get("values", {})
    legend = ('<div class="legend">'
              f'<span>{_badge("on", "ok")}every input tracks file is stamped as what ships</span>'
              f'<span>{_badge("off", "warn")}one is stamped as something else</span>'
              f'<span>{_badge("unstamped", "none")}written before stamps, so unknown</span>'
              f'<span>{_badge("missing", "bad")}an input file is gone</span>'
              f'<span>{_badge("n/a", "none")}reads no tracks file</span></div>')
    rows, family = [], None
    for mid in sorted(vals):
        if mid.split(".")[0] != family:
            family = mid.split(".")[0]
            rows.append(f'<tr class="group"><td colspan="6" class="eyebrow">{_e(family)}</td></tr>')
        g, spec = vals[mid], registry.get(mid)
        states = [state(str(p)) for p in (spec.inputs if spec else ()) if str(p).endswith(".tracks.json")]
        b = badge(states)
        snap = snapshot_of(mid, golden)
        ok = g.get("status") == "OK"
        shown = _fmt(g.get("value")) if ok else str(g.get("error"))
        unit = f' <span class="muted">{_e(spec.unit)}</span>' if spec and spec.unit else ""
        rows.append(
            f'<tr><td class="wraps"><span class="readout">{_e(mid)}</span>'
            f'<details><summary>registered text</summary><pre>scope: {_e(g.get("scope"))}\n\n'
            f'description: {_e(spec.desc if spec else "")}\n\nsnapshot reason: {_e(snap.get("reason"))}'
            "</pre></details></td>"
            f'<td class="readout soft wraps" title="{_e(json.dumps(g.get("value")))}">{_e(shown)}{unit}</td>'
            f'<td class="num readout">{_e(g.get("n"))}</td>'
            f'<td>{_badge(g.get("status"), "ok" if ok else "bad")}</td>'
            f'<td>{_badge(b or "n/a", TONE[b])}</td>'
            f'<td class="readout">{_e(snap.get("commit"))} <span class="muted">{_e(str(snap.get("at"))[:10])}</span></td></tr>')
    h.append(_card(f'Registered measurements <span class="muted">{len(vals)}</span>',
                   "Value, n and status from <code>model/validation/api-golden.json</code>; hover a value for the "
                   "exact figure. The vs-shipped badge is computed from the stamps on the id's input files.",
                   legend + _table(("Measurement", "Value", ">n", "Status", "Vs shipped", "Frozen by"), rows)))
    h.append("</div></main></body></html>")
    return "\n".join(h)


def demo():
    import tempfile

    # 1. REACH. A number planted in a doc and in a memory note never reaches the
    #    page, while the golden's own value does.
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        (root / "model" / "validation").mkdir(parents=True)
        (root / "docs" / "memory").mkdir(parents=True)
        (root / "docs" / "model-map.md").write_text("HOTA is 987.654 on everything\n")
        (root / "docs" / "memory" / "note.md").write_text("the tracker scores 555.555\n")
        (root / "model" / "validation" / "api-golden.json").write_text(json.dumps({
            "commit": "ccc3333", "updated": "2026-09-13T00:00:00Z", "reason": "top",
            "history": [{"at": "2026-09-01", "commit": "aaa1111", "ids": ["x.one"], "reason": "first"},
                        {"at": "2026-09-02", "commit": "bbb2222", "ids": "all", "reason": "blanket"},
                        {"at": "2026-09-03", "commit": "ddd4444", "ids": ["x.two"], "reason": "second"}],
            "values": {"x.one": {"value": 123.456, "n": 9, "scope": "s", "status": "OK", "error": None},
                       "x.two": {"value": {"hota": 0.4321}, "n": 2, "scope": "s2", "status": "OK", "error": None}}}))
        page = render(root, run_gates=False, registry={})
        assert "123.456" in page and "hota=0.4321" in page, "the golden's values must reach the page"
        assert "987.654" not in page, "a number written in a doc reached the page"
        assert "555.555" not in page, "a number written in a memory note reached the page"
        # ...and with the gates ON: a gate that quotes a document line in its body,
        # as lint_truth does, must not carry the number through. The first
        # version of this check ran with gates off and could not see this path.
        (root / "model" / "validation" / "out").mkdir()
        (root / "model" / "validation" / "quoting_gate.py").write_text(
            "print('docs/ledgers/model-map.md:3  HOTA is 777.777 on everything')\nprint('1 finding(s).')\n")
        page = render(root, run_gates=True, registry={}, gate_names=("quoting_gate",))
        assert "1 finding(s)." in page, "the gate's own summary line belongs on the page"
        assert "777.777" not in page, "a gate carried a quoted document number onto the page"

    # 2. which snapshot froze an id: the latest that names it, a blanket one counting
    g = json.loads(json.dumps({"history": [
        {"commit": "aaa1111", "ids": ["x.one"]}, {"commit": "bbb2222", "ids": "all"},
        {"commit": "ddd4444", "ids": ["x.two"]}]}))
    assert snapshot_of("x.one", g)["commit"] == "bbb2222", "a blanket snapshot rewrote it later"
    assert snapshot_of("x.two", g)["commit"] == "ddd4444"

    # 3. the badge is the WORST input, so one off-config file cannot hide behind six good ones
    assert badge(["ON", "ON", "UNSTAMPED"]) == "UNSTAMPED"
    assert badge(["ON", "OFF", "UNSTAMPED"]) == "OFF"
    assert badge(["ON", "MISSING"]) == "MISSING"
    assert badge(["ON"]) == "ON" and badge([]) is None

    # 4. RELATIONAL: the fast tail reader agrees with shipped_config's full parse -
    #    stamped, unstamped, and a report after _perf too big for the tail.
    with tempfile.TemporaryDirectory() as t:
        dets = [{"f": i, "id": i % 7, "xyxy": [1, 2, 3, 4]} for i in range(3000)]
        cases = {"stamped": {"fps": 25, "detections": dets, "_perf": {"config": {"overlap": 0.25}}},
                 "unstamped": {"fps": 25, "detections": dets, "_perf": {"frames": 9}},
                 "big_tail": {"fps": 25, "detections": dets, "_perf": {"config": {"overlap": 0.1}},
                              "_identity_report": {"pad": "x" * 100000}}}
        for name, blob in cases.items():
            p = Path(t) / f"{name}.tracks.json"
            p.write_text(json.dumps(blob))
            assert stamp_of(p) == SC.config_of(p), (name, stamp_of(p), SC.config_of(p))

    # 5. the last verify_claims run is the last block sharing a start time
    with tempfile.TemporaryDirectory() as t:
        log = Path(t) / "claims.log"
        log.write_text("\n".join(json.dumps(r) for r in (
            {"t": "T1", "git": "old", "status": "OK"}, {"t": "T1", "git": "old", "status": "OK"},
            {"t": "T2", "git": "new", "status": "OK"}, {"t": "T2", "git": "new", "status": "ERROR"})) + "\n")
        assert last_run(log) == {"t": "T2", "git": "new", "dirty": None, "counts": {"OK": 1, "ERROR": 1}}
    print("dashboard.py ok")


def main() -> int:
    if "--demo" in sys.argv:
        demo()
        return 0
    if "--watch" in sys.argv:
        # Each render in a FRESH process. A watcher that imports once renders with
        # the shipped_config and registry it loaded at start, and its header still
        # says HEAD: the first one did, for seven hours, and showed an unstitched
        # arm as on-config. A crashed render is one missed frame, not a dead page.
        while True:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--refresh"], cwd=ROOT)
            time.sleep(30)
    every = 30 if "--refresh" in sys.argv else None
    fonts = OUT.parent / "fonts"
    fonts.mkdir(parents=True, exist_ok=True)
    for name, src in FONTS.items():
        if (ROOT / src).exists() and not (fonts / name).exists():
            shutil.copy(ROOT / src, fonts / name)
    page = render(ROOT, refresh=every)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(page)
    os.replace(tmp, OUT)
    print(f"wrote {OUT.relative_to(ROOT)} at {time.strftime('%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
