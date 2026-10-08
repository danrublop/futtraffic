"""W5: drop tracks whose feet are not on the pitch. The gate, from geometry.

WHY
    `@M` Across the ten ST2 sequences, tracks that never match a single GT
    player at IoU 0.5 hold 24.2% of all boxes on 117092 and 22.2% on 132831 -
    people on an adjacent training pitch, a footpath, a bank. They are 71.8% and
    78.7% of those sequences' false positives.

    `@M` They are also why team assignment collapses there: 72.2% of 117092's
    `away` boxes and 34.5% of 132831's `home` boxes belong to them, giving
    home/away ratios of 1.77 and 1.74 against a truth of 1.00 on every sequence.
    The k=2 fallback in `track.assign_teams` is the SYMPTOM; non-players
    reaching the kit clustering at all is the cause. Fix this and that follows.

WHY NOT A ROW CONSTANT
    The measured version of this gate used `0.55 * H`, chosen after seeing the
    answer, on two sequences. `@M` It cannot generalise: 117092 is 3840x1906 and
    132831 is 3840x1504 while the other eight are 4096x1080, and those two ARE
    the contaminated ones. A fraction of frame height means a different thing on
    each. So the boundary comes from the pitch model projected through the
    sequence's own calibration, densified because a straight touchline images as
    a curve on a stitched panorama (see st2_overlay's chord bug).

THIS USES CALIBRATION, AND THAT IS A DELIBERATE DEPARTURE
    `st2_mot.py` runs the tracker with no --remap and no --calib on purpose, to
    report what the shipped stack does with no dataset-specific help. This gate
    needs geometry: "off the pitch" is a statement about the world, not about
    pixels. The product HAS a calibration - the operator clicks 21 landmarks -
    so using one here is honest about the product even though it is more than
    the tracker benchmark previously allowed. Scored both ways; the variant is
    separate so the ungated number stays quotable.

IT REFUSES A DEGENERATE CALIBRATION RATHER THAN DELETING THE MATCH
    `@M` On 132831 the first version of this gate cut 100% of boxes. That is not
    an off-pitch finding, it is the known degenerate homography - condition
    number 1.47e13 against ~1.8e6 for the others, and 100% of its projected
    pitch vertices land on no paint (context-pitch-mapping-pivot (not published) 3.2).
    A gate that silently empties a match because its calibration is bad is worse
    than no gate. So: if fewer than MIN_INSIDE of the boxes fall inside the
    polygon, the calibration is unusable for this purpose and the sequence
    passes through untouched, loudly. Degrade honestly - process the footage and
    report what was not done.

A TRACK IS DROPPED, NOT A DETECTION
    A person is on the pitch or is not. Dropping individual boxes would punch
    holes in a real player's track every time his feet are mis-localised, which
    is a fragmentation source; dropping the track keeps the decision at the unit
    the question is actually about. Threshold is a simple majority of the
    track's own foot points.
"""
import argparse
import json
import os
import sys

import numpy as np
from matplotlib.path import Path as MplPath

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "engine"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overlay import densify  # noqa: E402
from st2_mot import WORK, _wh  # noqa: E402 - WORK is root-relative there
from st2_overlay import pitch_of, projector  # noqa: E402
MARGIN_M = 3.0
INSIDE_FRAC = 0.5
# A pitch holding 22 players should contain most of what was detected. Below
# this the polygon is wrong, not the footage.
MIN_INSIDE = 0.4


def polygon(seq, margin_m=MARGIN_M, step_m=0.5):
    """The pitch-plus-margin rectangle, in this sequence's panorama pixels."""
    P = pitch_of(seq)
    fn = projector(seq, _wh(seq))
    lo_x, hi_x = -margin_m, P.length + margin_m
    lo_y, hi_y = -margin_m, P.width + margin_m
    ring = [(lo_x, lo_y), (hi_x, lo_y), (hi_x, hi_y), (lo_x, hi_y), (lo_x, lo_y)]
    pts = []
    for a, b in zip(ring, ring[1:]):
        pts += densify([a, b], step_m=step_m)
    px, ok = fn(np.array(pts, float))
    if ok.sum() < 8:
        return None
    return px[ok]


def gate(seq, variant=".c03s", out=None, margin_m=MARGIN_M):
    poly = polygon(seq, margin_m)
    src = f"{WORK}/{seq}{variant}.tracks.json"
    d = json.load(open(src))
    rows = d["detections"]
    if poly is None:
        print(f"  {seq}: no usable projection, gate skipped")
        keep_rows = rows
        dropped = 0
    else:
        path = MplPath(poly)
        # tracks.json x,y IS the foot point (bottom-centre), which is the whole
        # reason this test is cheap: no ground-point estimation needed.
        feet = np.array([(r["x"], r["y"]) for r in rows], float)
        inside = path.contains_points(feet)
        if inside.mean() < MIN_INSIDE:
            print(f"  {seq}: REFUSED - only {100 * inside.mean():.1f}% of boxes "
                  f"inside the projected pitch; calibration unusable for this "
                  f"gate, passing through untouched")
            keep_rows, dropped = rows, 0
        else:
            by = {}
            for r, ins in zip(rows, inside):
                a, b = by.setdefault(r["id"], [0, 0])
                by[r["id"]] = [a + bool(ins), b + 1]
            keep = {t for t, (i, n) in by.items() if i >= INSIDE_FRAC * n}
            keep_rows = [r for r in rows if r["id"] in keep]
            dropped = len(by) - len(keep)
    out = out or f"{WORK}/{seq}{variant}_op.tracks.json"
    json.dump({**d, "detections": keep_rows,
               "_gate": {"margin_m": margin_m, "inside_frac": INSIDE_FRAC,
                         "tracks_dropped": dropped,
                         "boxes_dropped": len(rows) - len(keep_rows)}},
              open(out, "w"))
    print(f"  {seq}: {len(rows)} -> {len(keep_rows)} boxes "
          f"({100 * (len(rows) - len(keep_rows)) / max(len(rows), 1):4.1f}% cut), "
          f"tracks {len(set(r['id'] for r in rows))} -> "
          f"{len(set(r['id'] for r in keep_rows))}")
    return out


def demo():
    """The polygon must contain the centre spot and exclude the far stand."""
    seq = "128057"
    poly = polygon(seq)
    assert poly is not None and len(poly) > 100, poly
    fn = projector(seq, _wh(seq))
    P = pitch_of(seq)
    path = MplPath(poly)
    centre, _ = fn(np.array([[P.length / 2, P.width / 2]]))
    assert path.contains_point(centre[0]), centre
    # a point 40 m beyond the far touchline is off the pitch by any reading
    far, ok = fn(np.array([[P.length / 2, -40.0]]))
    if ok[0]:
        assert not path.contains_point(far[0]), far
    # and the corners of the pitch itself must be inside a 3 m margin
    for xy in ((0.0, 0.0), (P.length, 0.0), (0.0, P.width), (P.length, P.width)):
        c, o = fn(np.array([xy]))
        if o[0]:
            assert path.contains_point(c[0]), xy
    # And the refusal must fire on the sequence with the degenerate homography
    # rather than emptying it.
    out = gate("132831", ".c03s", out=f"{WORK}/_demo_op.tracks.json")
    n = len(json.load(open(out))["detections"])
    assert n > 100000, f"132831 must pass through untouched, got {n}"
    os.remove(out)
    print(f"ok  ({len(poly)} polygon vertices; centre in, far stand out, "
          f"all four corners in; 132831 refused not emptied)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="*")
    ap.add_argument("--variant", default=".c03s")
    ap.add_argument("--margin", type=float, default=MARGIN_M)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        demo()
    else:
        seqs = a.seqs or sorted({s.split(".")[0] for s in os.listdir(WORK)
                                 if s.endswith(f"{a.variant}.tracks.json")})
        for s in seqs:
            gate(s, a.variant, margin_m=a.margin)
