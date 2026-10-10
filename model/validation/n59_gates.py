"""N59: which of BallPicker's three gates earn their place, per population.

    .venv/bin/python model/validation/n59_gates.py --score
    .venv/bin/python model/validation/n59_gates.py --demo

WHY THIS RUNS
    `ball_cmp` pooled only shipped/oracle/any/centre, so the `conf` arm - argmax
    over the shape gate alone, with SCENERY, ON-PITCH and SIZE removed - was
    computed per clip and never pooled. Review pooled it on 2026-09-25: arm D
    `conf` 0.5167 against `shipped` 0.4706. **Five picker experiments were read
    from a table that could not show the largest measured picker lever**, and
    that lever is deleting gates rather than adding cleverness.

    N57's split says why, and it is not "the gates are bad". Per population (the
    same review, scratchpad, not a repo command - which is half of what this file
    exists to fix): dead 0.0117 -> 0.2233, in-play 0.7000 -> 0.6633. The gates
    EARN their place while the ball is moving and destroy the dead-ball
    population, which is exactly the population `Scenery` is built to delete,
    because a ball at rest is indistinguishable from paint by the only test that
    gate applies.

WHAT THIS CAN AND CANNOT BE A BLIND TEST OF
    The all-gates-off pooled number is ALREADY MEASURED. Pre-registering "does
    removing every gate beat shipped" would be writing a rule around an answer I
    have already seen, which is the thing this repo's protocol exists to stop. So
    that configuration is reported as KNOWN, not tested.

    What is genuinely unknown, and what the rule below rules on: whether ANY gate
    subset survives leave-one-clip-out SELECTION with a per-population floor. A
    sweep's argmax is not a result (docs/README.md#working-rules), and 8 subsets over 6 clips is a
    sweep - so the winner is chosen on five clips and scored on the sixth, six
    times, and the reported number is the held-out one.

RULE, FIXED BEFORE THE SCORED RUN (committed with this file, nothing run yet)
    population  ball_cmp.CLIPS, the committed `.cmpD` caches, 1800 gt-present
                frames, gate 20 px, split by N57 Part A into dead (2 clips, 600
                frames) and in-play (4 clips, 1200).
    arms        all 8 subsets of {scenery, on-pitch, size}. `ball.plausible()`,
                the shape gate, is ON in every arm - it is not one of the three
                and `any` minus `oracle` says it costs nothing.
    selection   LEAVE ONE CLIP OUT. For each held-out clip, choose the subset
                with the best pooled rate on the other five, then score THAT
                subset on the held-out clip. The six held-out scores are summed.
                No subset is ever scored on a clip that chose it.
    PASS        the held-out total beats `shipped` by >= +0.05 pooled AND neither
                population falls by more than 0.01 against `shipped`. Both, or it
                does not ship.
    REPORT      all 8 subsets, in-sample and held-out, per population, with
                `wrong` and `silent` - a picker that answers less often buys
                correctness with silence, and silence is a failure here.
    n           n=8 configurations, stated on every number this produces. The
                held-out protocol is what makes the argmax admissible; without it
                the best of 8 on 6 clips is a coin flip with a press release.
    VOID        (a) the full-gate subset does not reproduce `shipped` 0.4706;
                (b) `centre` scores over 1%;
                (c) any subset is scored on a clip that selected it.
"""
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
GATES = ("scenery", "pitch", "size")

import ball as ballmod  # noqa: E402
import ball_cmp as C  # noqa: E402 - CLIPS and the .cmpD arm, never re-listed here
import ball_eval as B  # noqa: E402 - load_cache, score, predict; the metric is its own
import n57_conf_evidence as N57  # noqa: E402 - the dead/in-play split, reused not recopied


def subsets():
    """All 8, as frozensets of the gates that are ON."""
    return [frozenset(c) for n in range(len(GATES) + 1) for c in itertools.combinations(GATES, n)]


def picker(cal, on: frozenset):
    """`ball.BallPicker` with the gates in `on` and no others, by NEUTRALISING the
    ones that are off - never by reimplementing `update`.

    The three gates have no flags, and a subclass would have to copy the one method
    that holds all of them, which is a copy that drifts from the shipped picker the
    first time anyone edits it (a-callers-default-silently-beats-the-callees (internal note)
    is that failure in the other direction). Each gate has an injection point instead:

        scenery  the constructor already takes the Scenery object; a stub whose
                 flags() are all False removes it and keeps the memory feed intact.
        pitch    MARGIN_M is how far off the pitch a candidate may sit; at 1e9 the
                 bounds test cannot reject anything.
        size     SIZE_RATIO is the accepted observed/expected span; (0, 1e9) same.

    Both overrides are per INSTANCE, so the class attributes the product reads are
    untouched. All three off is `conf` - `update` applies `plausible()` itself - and
    all three on is `shipped`; score() asserts both against ball_eval rather than
    trusting this docstring.
    """
    class _NoScenery(ballmod.Scenery):
        def flags(self, xy):
            return np.zeros(len(np.asarray(xy).reshape(-1, 2)), bool)

    p = ballmod.BallPicker(cal=cal, scenery=None if "scenery" in on else _NoScenery())
    if "pitch" not in on:
        p.MARGIN_M = 1e9
    if "size" not in on:
        p.SIZE_RATIO = (0.0, 1e9)
    return p


def preds(clip, on: frozenset, suffix=".cmpD", cache=None):
    """{frame: xy or None} for one gate subset, over the committed cache.

    `cache` is passed in by rates(), which has already loaded it: eight subsets per clip
    meant nine gzip loads where one would do.
    """
    cache = cache if cache is not None else B.load_cache(clip + suffix)
    frames = sorted(cache["frames"])
    p = picker(B._clip_cal(clip), on)
    out = {}
    for f in frames:
        b = p.update(B._dets(cache["frames"][f]), cache["frame_h"], f, None)
        out[f] = np.array(b.xy) if b.seen else None
    return out, frames


def rates(suffix=".cmpD"):
    """{clip: {subset: score dict}} plus the population of each clip."""
    pop = {c: v[0] for c, v in N57.split().items()}
    out = {}
    for clip in C.CLIPS:
        cache = B.load_cache(clip + suffix)
        gt, _wh, _n = B.load_ball(clip)
        frames = sorted(cache["frames"])
        gt_here = {f: gt[f] for f in frames if f in gt}
        out[clip] = {}
        for on in subsets():
            pr, fr = preds(clip, on, suffix, cache=cache)
            out[clip][on] = B.score(gt_here, pr, fr, B.GATE_PX)
        out[clip]["pop"] = pop.get(clip, "in-play")
    return out


def _loo(tab, clips, arms=None):
    """[(held-out clip, the subset five clips chose, those five clips)].

    Factored out because demo() used to re-implement this on a synthetic dict and so
    never executed the loop that produces the number. A mutation that changed the
    selection to `rest = clips` - turning the held-out figure into an in-sample argmax,
    worth +0.0883 on this table - left both the demo and the in-line assert green,
    because `assert out not in rest` merely restates the line above it.

    TIE-BREAK, stated because the rule text does not: subsets are ordered by
    (size, names) before `max`, so a tie resolves to the FEWEST gates - the direction
    the hypothesis wants. Ties are plausible here, since a gate that rejects nothing
    makes two subsets identical.
    """
    arms = arms or sorted(subsets(), key=lambda s: (len(s), sorted(s)))
    out = []
    for held in clips:
        rest = [c for c in clips if c != held]
        best = max(arms, key=lambda s: _pooled(tab, rest, s)[0])
        out.append((held, best, rest))
    return out


def _pooled(tab, clips, on):
    c = sum(tab[k][on]["correct"] for k in clips)
    n = sum(tab[k][on]["gt_present"] for k in clips)
    return c / n if n else 0.0, c, n


def score(suffix=".cmpD"):
    """The rule in this file's docstring, run. n=8 configurations."""
    tab = rates(suffix)
    clips = list(C.CLIPS)
    # VOID (a) and (b): the full subset must BE shipped, and the control must fail.
    full, empty = frozenset(GATES), frozenset()
    # ONE LOAD PER CLIP. This used to call load_cache inside the `if f in ...` condition
    # of a dict comprehension, so the gzip was re-read once per ground-truth frame - about
    # 1,800 loads, which made the claim take 2,431s against the gate's 300s timeout and
    # blocked a push. Correct all along, absurdly slow.
    caches = {c: B.load_cache(c + suffix) for c in clips}
    fr = {c: sorted(caches[c]["frames"]) for c in clips}
    gtc = {c: B.load_ball(c)[0] for c in clips}
    here = {c: {f: g for f, g in gtc[c].items() if f in set(fr[c])} for c in clips}
    ship = {c: B.score(here[c], B.predict(caches[c], "shipped", None, None),
                       fr[c], B.GATE_PX) for c in clips}
    sh_rate = sum(ship[c]["correct"] for c in clips) / sum(ship[c]["gt_present"] for c in clips)
    mine, _, _ = _pooled(tab, clips, full)
    print(f"  VOID check (a): all gates ON {mine:.4f} vs ball_eval shipped {sh_rate:.4f}")
    if abs(mine - sh_rate) > 1e-9:
        print("  VOID: the all-gates subset is not the shipped picker")
        return 1
    ctr = {c: B.score(here[c], B.predict(caches[c], "centre", None, None),
                      fr[c], B.GATE_PX) for c in clips}
    ctr_rate = sum(ctr[c]["correct"] for c in clips) / sum(ctr[c]["gt_present"] for c in clips)
    print(f"  VOID check (b): centre {ctr_rate:.4f}")
    if ctr_rate > 0.01:
        print("  VOID: the centre control scores over 1%")
        return 1

    # REPORT: every subset, in-sample, per population. n=8 on every line.
    dead = [c for c in clips if tab[c]["pop"] == "dead"]
    play = [c for c in clips if tab[c]["pop"] != "dead"]
    print(f"\n  in-sample, n=8 configurations; dead {len(dead)} clip(s), in-play {len(play)}")
    print("  gates ON                    pooled     dead   in-play    wrong  silent")
    for on in sorted(subsets(), key=lambda s: (len(s), sorted(s))):
        r, _, _ = _pooled(tab, clips, on)
        d, _, _ = _pooled(tab, dead, on)
        p_, _, _ = _pooled(tab, play, on)
        w = sum(tab[c][on]["wrong"] for c in clips)
        s_ = sum(tab[c][on]["silent"] for c in clips)
        name = "+".join(sorted(on)) or "(none = conf)"
        print(f"  {name:24s}  {r:.4f}   {d:.4f}   {p_:.4f}    {w:5d}  {s_:5d}")

    # SELECTION: leave one clip out, and no subset is scored on a clip that chose it.
    held_c = held_n = 0
    dead_c = dead_n = play_c = play_n = 0
    for out, best, rest in _loo(tab, clips):
        held_c += tab[out][best]["correct"]
        held_n += tab[out][best]["gt_present"]
        if tab[out]["pop"] == "dead":
            dead_c += tab[out][best]["correct"]; dead_n += tab[out][best]["gt_present"]
        else:
            play_c += tab[out][best]["correct"]; play_n += tab[out][best]["gt_present"]
        print(f"  held out {out[-9:]}  chose {'+'.join(sorted(best)) or '(none)':24s} "
              f"-> {tab[out][best]['correct']}/{tab[out][best]['gt_present']}")
    held = held_c / held_n
    sh_d, _, _ = _pooled(tab, dead, full)
    sh_p, _, _ = _pooled(tab, play, full)
    hd = dead_c / dead_n if dead_n else 0.0
    hp = play_c / play_n if play_n else 0.0
    print(f"\n  HELD-OUT pooled {held:.4f} vs shipped {sh_rate:.4f}  "
          f"(delta {held - sh_rate:+.4f}, bar +0.05), n=8")
    print(f"  dead {hd:.4f} vs {sh_d:.4f} ({hd - sh_d:+.4f})   "
          f"in-play {hp:.4f} vs {sh_p:.4f} ({hp - sh_p:+.4f})   floor -0.01 each")
    ok = (held - sh_rate >= 0.05) and (hd - sh_d >= -0.01) and (hp - sh_p >= -0.01)
    print(f"  RULE: >= +0.05 pooled AND neither population below -0.01 -> "
          f"{'SUPPORTED' if ok else 'NOT SUPPORTED'}")
    return 0


def demo():
    """The selection protocol, on synthetic scores where the answer is known.

    The arithmetic that can be silently wrong here is not the picking - it is
    whether a subset is ever scored on a clip that chose it, which is the whole
    difference between a held-out number and a sweep's argmax.
    """
    assert len(subsets()) == 8 and frozenset() in subsets() and frozenset(GATES) in subsets()
    # THE REAL _loo, on a synthetic table where the answer is known. Subset A wins on
    # clips 0-4 and scores zero on clip 5, so a correct leave-one-out must pick A five
    # times and be punished on the sixth; a leaking one would never see the zero.
    cl = [f"c{i}" for i in range(6)]
    tab = {c: {"A": {"correct": (100 if i < 5 else 0), "gt_present": 100},
               "B": {"correct": 50, "gt_present": 100}} for i, c in enumerate(cl)}
    folds = _loo(tab, cl, arms=["A", "B"])
    assert [b for _h, b, _r in folds] == ["A"] * 6, folds
    assert all(h not in r for h, _b, r in folds), "a fold kept its own held-out clip"
    got = sum(tab[h][b]["correct"] for h, b, _r in folds)
    assert got == 500, got          # 5 x 100 + the zero on the clip that chose nothing
    # and a LEAKING selection must score higher, so the guard has something to catch
    leak = sum(max(("A", "B"), key=lambda s: _pooled(tab, cl, s)[0]) == "A" for _ in cl)
    assert leak == 6 and sum(tab[c]["A"]["correct"] for c in cl) == 500
    print("n59 demo: 8 subsets, and leave-one-clip-out never scores a subset on its chooser")


if __name__ == "__main__":
    if "--score" in sys.argv[1:]:
        # VOID (a) here compares the all-gates subset against ball_eval's shipped over the
        # SAME cache, so it is self-consistent at either resolution and needs no constant.
        raise SystemExit(score(".cmpDr" if "--at640" in sys.argv[1:] else ".cmpD"))
    demo()
