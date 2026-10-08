"""N68 fixes F1-F3, each as a switchable ARM over the unedited engine, scored by the pass rule pre-registered in
n68-128057-length (not published) section 3 (commit 861b335). Nothing here edits model/engine.

    .venv/bin/python model/validation/n68_fixes.py                 # every arm (base, F1, F2, F3)  #slow
    .venv/bin/python model/validation/n68_fixes.py --arms F1       # one arm (base always runs: clause 2's "today's")

Arms (monkeypatched into autocal in a fresh process per run):
    base  autocal before the fix: accept()'s pooled "pen" family (re-planted once F1 shipped) - the reference for
          clause 2's map displacement. Not a fix variant.
    F1    accept() per side: pen_l_front AND pen_r_front each >= 50% of their samples on paint, replacing "any pen
          group at 25%"; every other family clause unchanged.
    F2    _settle: if the registration at the given/nominal length (P.length) fails accept(), return it refused - no scan.
    F3    a measured length outside nominal +- 2 m is a FLAG: autocal prefers an accepted candidate within +- 2 m of
          nominal when one exists; otherwise the out-of-band result stands, accepted and flagged.

Population (the pass rule): the 8 medians model/validation/out/stc2025/*.median40.png from nothing, Pitch(), no
install camera; 128057 via from_install from the N34 install camera; autocal._demo() (the 110.5 m synthetic pitch).
PASS iff (1) 128057 from nothing refused or |L - 105| <= 0.5; (2) 117093 118575 118576 118577 118578 128058 accepted,
|L - 105| <= 0.5 and corner-origin map vs base within median 0.02 / max 0.1 m on check.py's 12x8 grid;
(3) 117092 refused; (4) 128057 from_install accepted, |L - 105| <= 0.5; (5) the self-check passes.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent)]
from n68_128057_length import GT_L, MEDIAN, N34, ROOT  # noqa: E402

CLIPS = ("117092", "117093", "118575", "118576", "118577", "118578", "128057", "128058")
KEEP = ("117093", "118575", "118576", "118577", "118578", "128058")
ARMS = ("base", "F1", "F2", "F3")
F1_SHARE, F3_BAND = 0.5, 2.0
OUT = ROOT / "model" / "validation" / "out" / "n68_fixes"


def _patch(A, arm):
    """Switch `arm` on in the imported autocal module A. Module globals, so every internal caller sees it."""
    if arm == "base" and hasattr(A, "_missing"):   # F1 shipped (N68 section 5): base is the rule it replaced
        def _missing(rep):
            fam = {k.split("_")[0] for k, (h, n) in rep["hits_by_group"].items() if h >= 0.25 * n}
            return sorted({"touch", "goal", "halfway", "pen", "circle"} - fam)
        A._missing = _missing
    elif arm == "F1":
        base_accept = A.accept

        def accept(cam, rep):
            ok, C = base_accept(cam, rep)
            g = rep["hits_by_group"]
            fam = {k.split("_")[0] for k, (h, n) in g.items() if h >= 0.25 * n} - {"pen"}
            fronts = all(g[k][0] >= F1_SHARE * g[k][1] for k in ("pen_l_front", "pen_r_front"))
            ok = bool(rep["refused"] is None and 2.0 <= abs(C[2]) <= 20.0 and {"touch", "goal", "halfway", "circle"} <= fam
                      and fronts)
            return ok, C
        A.accept = accept
    elif arm == "F2":
        settle = A._settle

        def _settle(ridge, grass, start, P, native=None, fix_length=False):
            # _settle's own first step, copied: go(start, P) at the given length, up to 4 passes
            empty, cam, rep, radii = np.zeros((0, 2)), np.asarray(start, float), None, A.AUTO_RADII
            for _ in range(4):
                nxt, nrep = A.register(ridge, cam, P, empty, empty, grass=grass, square=1.0, radii=radii, native=native)
                if rep is not None and nrep["hits"] <= 1.02 * rep["hits"]:
                    break
                cam, rep, radii = nxt, nrep, A.AUTO_RADII[1:]
            if not A.accept(cam, rep)[0]:
                return cam, rep, P                      # refused at the given length: no scan
            return settle(ridge, grass, start, P, native=native, fix_length=fix_length)   # recomputes that step, deterministic
        A._settle = _settle
    elif arm == "F3":
        settle, autocal = A._settle, A.autocal

        def wrapped(bgr, P, verbose=False, fix_length=False):
            seen = []

            def rec(*a, **k):
                out = settle(*a, **k)
                seen.append(out)
                return out
            A._settle = rec
            try:
                cam, rep = autocal(bgr, P, verbose=verbose, fix_length=fix_length)
            finally:
                A._settle = settle
            flag = rep.get("length_m") is not None and abs(rep["length_m"] - P.length) > F3_BAND
            if flag:
                near = [(A.accept(c, r)[0], r["hits"] // 50, -r["line_rms_px"], i) for i, (c, r, Pf) in enumerate(seen)
                        if abs(Pf.length - P.length) <= F3_BAND]
                near = [k for k in near if k[0]]
                if near:
                    c, r, Pf = seen[max(near)[3]]
                    ok, C = A.accept(c, r)
                    return c, {**rep, "accepted": True, "hits": r["hits"], "line_rms_px": r["line_rms_px"],
                               "length_m": Pf.length, "length_sd_m": r.get("length_sd_m"), "hits_by_group": r["hits_by_group"],
                               "camera_centre_m": [float(v) for v in C], "why": None, "f3": "preferred nominal"}
            return cam, {**rep, "f3": "flagged" if flag else None}
        A.autocal = wrapped


def _job(arm, what):
    """what: a clip id (from nothing), 'install' (128057 from_install), or 'demo' (the self-check).
    Cached in out/n68_fixes/ keyed by the sha256 of autocal.py and of this file: a crash does not redo finished runs."""
    import hashlib  # noqa: PLC0415
    key = hashlib.sha256((ROOT / "model/engine/autocal.py").read_bytes() + Path(__file__).read_bytes()).hexdigest()[:12]
    cache = OUT / f"{arm}_{what}_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    r = _job_run(arm, what)
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(r))
    print(f"  done {arm} {what}", flush=True)
    return r


def _job_run(arm, what):
    import cv2  # noqa: PLC0415
    sys.path[:0] = [str(ROOT / "model" / "engine")]
    import autocal as A  # noqa: PLC0415
    from pitch import Pitch  # noqa: PLC0415
    _patch(A, arm)
    if what == "demo":
        try:
            import contextlib  # noqa: PLC0415
            import io  # noqa: PLC0415
            with contextlib.redirect_stdout(io.StringIO()) as out:
                A._demo()
            return {"ok": True, "out": out.getvalue().strip()}
        except AssertionError as e:
            return {"ok": False, "out": f"AssertionError {str(e)[:300]}"}
    seq = "128057" if what == "install" else what
    img, P = cv2.imread(str(MEDIAN).format(seq)), Pitch()
    if what == "install":
        n34 = json.load(open(N34))
        start = np.median([v["camera"] for q, v in n34.items()
                           if v["camera"] is not None and v.get("via") != "install" and q != seq], axis=0)
        cam, rep = A.from_install(img, P, start)
    else:
        cam, rep = A.autocal(img, P)
    g = rep.get("hits_by_group") or {}
    return {"accepted": bool(rep["accepted"]), "why": rep.get("why"), "hits": rep["hits"], "length_m": rep.get("length_m"),
            "f3": rep.get("f3"), "width": img.shape[1], "camera": None if cam is None else np.asarray(cam).tolist(),
            "pen_fronts": {k: g[k] for k in ("pen_l_front", "pen_r_front") if k in g}}


def _disp(a, b, width):
    """Corner-origin metres: the pixel base camera `a` draws each grid point at, read back through arm camera `b`."""
    sys.path[:0] = [str(ROOT / "model" / "engine"), str(ROOT / "model" / "validation")]
    import autocal as A  # noqa: PLC0415
    from check import _grid  # noqa: PLC0415
    d = np.linalg.norm(A._inv(np.array(b), A._fwd(np.array(a), _grid(), float(width)), float(width)) - _grid(), axis=1)
    return float(np.median(d)), float(d.max())


def verdict(arm, r, base):
    ok105 = lambda x: x["accepted"] and x["length_m"] is not None and abs(x["length_m"] - GT_L) <= 0.5  # noqa: E731
    fails = []
    c = r["128057"]
    if not (not c["accepted"] or ok105(c)):
        fails.append(f"(1) 128057 from nothing accepted at {c['length_m']} m")
    for q in KEEP:
        if not ok105(r[q]):
            fails.append(f"(2) {q} accepted={r[q]['accepted']} length {r[q]['length_m']}")
        elif base[q]["camera"] is not None:
            med, mx = _disp(base[q]["camera"], r[q]["camera"], r[q]["width"])
            r[q]["map_vs_base"] = (round(med, 4), round(mx, 4))
            if med > 0.02 or mx > 0.1:
                fails.append(f"(2) {q} map vs base median {med:.4f} / max {mx:.4f} m")
    if r["117092"]["accepted"]:
        fails.append("(3) 117092 accepted")
    if not ok105(r["install"]):
        fails.append(f"(4) 128057 from_install accepted={r['install']['accepted']} length {r['install']['length_m']}")
    if not r["demo"]["ok"]:
        fails.append(f"(5) self-check: {r['demo']['out']}")
    return fails


def main(arms):
    arms = ["base"] + [a for a in arms if a != "base"]
    jobs = [(a, w) for a in arms for w in CLIPS + ("install", "demo")]
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() // 2), max_tasks_per_child=1) as ex:
        futs = {j: ex.submit(_job, *j) for j in jobs}
        res = {j: f.result() for j, f in futs.items()}
    base = {w: res[("base", w)] for w in CLIPS + ("install",)}
    for a in arms:
        r = {w: res[(a, w)] for w in CLIPS + ("install", "demo")}
        fails = verdict(a, r, base)
        print(f"== {a}: {'PASS' if not fails else 'FAIL'}")
        for w in CLIPS + ("install",):
            x = r[w]
            fr = " ".join(f"{k} {h}/{n}" for k, (h, n) in x["pen_fronts"].items())
            print(f"  {w:8s} accepted={x['accepted']} hits={x['hits']} length={x['length_m']} f3={x['f3']}"
                  f" map_vs_base={x.get('map_vs_base')} {fr} why={x['why']!r}")
        print(f"  demo     ok={r['demo']['ok']} {r['demo']['out'][:200]}")
        for f in fails:
            print(f"  FAILS {f}")
    print(f"n={len([a for a in arms if a != 'base'])} fix arms tried: {', '.join(a for a in arms if a != 'base')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=ARMS)
    main(ap.parse_args().arms)
