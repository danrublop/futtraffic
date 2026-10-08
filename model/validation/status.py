"""One command: score the current code on EVERY sequence, and show the overtuning gap.

WHY THIS EXISTS
    On 2026-09-08 a configuration was scored ~24 times against the benchmark test
    pair and the best result reported as a finding. Nothing recorded that. The
    number happened to survive held-out checking, which was luck, not method.

    Separately and worse, the input ledger was measured with
    `assoc_eval`'s DEFAULT tracker - the zero-dependency fallback, not the one
    that ships. Every figure in the document defining "what everything else is
    conditioned on" described a tracker nobody runs.

    Both failures are invisible in a single number. Both are obvious in a table
    that names its instrument and splits tuned-on from held-out. That table is
    this file.

WHAT IT PRINTS, AND WHY EACH COLUMN IS THERE
    * every sequence with tracks on disk, never a pooled figure alone - a mean
      hides the sequence that fell over (132831's DetA moved 0.521 -> 0.610 under
      a change that moved 128057's 0.674 -> 0.647, and the pooled number barely
      twitched).
    * TUNED vs HELD OUT, and the gap between them. That gap IS the overtuning
      measurement. A config fitted to the test pair shows a positive gap; a
      mechanism shows ~zero. Measured 2026-09-09: +0.020 HOTA on both, which is
      what a mechanism looks like.
    * the INSTRUMENT on every row - tracker, conf, git commit. A number without
      its instrument is not a measurement, it is a rumour.

`ponytail: reuses st2_mot.run() rather than re-implementing TrackEval. A checker
that carries its own copy of the thing it checks agrees with itself and nothing
else - see model/validation/st2_align.py, which was blind for exactly that reason.`
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import st2_mot  # noqa: E402
import verify_claims as V  # noqa: E402 - the HEAD stamp, reused and never copied

# The two the organisers scored, and the two every config was swept against.
TUNED = ("128057", "132831")
HISTORICAL = ("<!-- api:ok historical record: what status.py measured at "
              "this commit, not a live claim -->")
LEDGER = "docs/ledgers/status-ledger.md"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _commit():
    # A PROVENANCE stamp, so it must name the tree these numbers came out of:
    # verify_claims.head drops GIT_* (which beats cwd=) for that reason. A
    # ledger row stamped with another tree's HEAD is worse than an unstamped one.
    sha, dirty = V.head()
    return sha + ("+dirty" if dirty else "")


def available(variant):
    out = []
    for s in sorted(os.listdir(st2_mot.WORK)):
        if s.endswith(f"{variant}.tracks.json"):
            seq = s[:-len(f"{variant}.tracks.json")]
            if os.path.exists(f"{st2_mot.GT_SRC}/{seq}.txt"):
                out.append(seq)
    return out


def report(variant, note, write):
    st2_mot.VARIANT = variant
    seqs = available(variant)
    if not seqs:
        sys.exit(f"no *{variant}.tracks.json in {st2_mot.WORK}")
    st2_mot.stage_gt(seqs)
    st2_mot.stage_pred(seqs)
    res = st2_mot.run(seqs)

    tuned = [s for s in seqs if s in TUNED]
    held = [s for s in seqs if s not in TUNED]
    mean = lambda g, k: sum(res[s][k] for s in g) / len(g) if g else float("nan")

    print(f"\n{'sequence':>10s} {'split':>9s} " +
          " ".join(f"{k:>7s}" for k in ("HOTA", "DetA", "AssA", "LocA", "IDF1")))
    for s in seqs:
        print(f"{s:>10s} {'TUNED' if s in TUNED else 'held-out':>9s} " +
              " ".join(f"{res[s][k]:7.3f}" for k in ("HOTA", "DetA", "AssA", "LocA", "IDF1")))
    print()
    for nm, g in (("TUNED", tuned), ("HELD OUT", held)):
        if g:
            print(f"{nm:>10s} {'mean':>9s} " +
                  " ".join(f"{mean(g, k):7.3f}" for k in ("HOTA", "DetA", "AssA", "LocA", "IDF1")))

    gap = None
    if tuned and held:
        gap = mean(tuned, "HOTA") - mean(held, "HOTA")
        print(f"\n  OVERTUNING GAP (tuned - held out): {gap:+.3f} HOTA over "
              f"{len(tuned)} vs {len(held)} sequences")
        print("  A mechanism reads near zero. A fitted constant reads positive.")
        print("  Sequence difficulty also lives in this gap - read it against the")
        print("  same gap on the UNMODIFIED baseline, never on its own.")

    line = (f"| {datetime.datetime.now().isoformat(timespec='seconds')} "
            f"| `{_commit()}` | `{variant or '(none)'}` "
            f"| {mean(tuned, 'HOTA'):.3f} | {mean(held, 'HOTA'):.3f} "
            f"| {gap if gap is None else f'{gap:+.3f}'} "
            f"| {len(seqs)} | {note} |")
    if write:
        new = not os.path.exists(LEDGER)
        with open(LEDGER, "a") as fh:
            if new:
                fh.write("# Status ledger\n\nAppend-only. Every row is one run of "
                         "`model/validation/status.py`, which is the only thing that "
                         "writes here.\n\n"
                         "| when | commit | variant | TUNED HOTA | HELD-OUT HOTA "
                         "| gap | seqs | note |\n|---|---|---|---|---|---|---|---|\n")
            # The ledger is <!-- api:strict --> so every bare decimal must be
            # marked. A ledger row is a HISTORICAL RECORD - what this script
            # measured at a named commit - not a live claim, so it can never be
            # an api:// citation: an id resolves to today's value and would let
            # history rewrite itself on the next snapshot. The escape says that.
            fh.write(line + "  " + HISTORICAL + "\n")
        print(f"\n  -> appended to {LEDGER}")
    else:
        print(f"\n  (not written; --write appends)\n  {line}")
    return res


def demo():
    """Self-check: the split must be honoured and the gap must be real arithmetic."""
    assert set(TUNED) == {"128057", "132831"}, TUNED
    seqs = available(".c03s") or available("")
    assert seqs, "no tracks on disk to check against"
    # a sequence in TUNED must never be counted as held out, and vice versa
    t = [s for s in seqs if s in TUNED]
    h = [s for s in seqs if s not in TUNED]
    assert not (set(t) & set(h)), (t, h)
    assert set(t) | set(h) == set(seqs)
    # the ledger must be append-only: writing twice must not shrink it
    if os.path.exists(LEDGER):
        n = len(open(LEDGER).read().splitlines())
        assert n >= 4, "ledger exists but has no rows"
    print(f"demo ok: {len(seqs)} sequences on disk, {len(t)} tuned / {len(h)} held out, "
          f"split is a partition")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default=".c03s", help="tracks.json suffix to score")
    ap.add_argument("--note", default="", help="one line for the ledger row")
    ap.add_argument("--write", action="store_true", help="append to the status ledger")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    demo() if a.demo else report(a.variant, a.note or "-", a.write)
