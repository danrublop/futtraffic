"""How many metres does one pixel of box wobble cost, and where.

WHY THIS FILE EXISTS
    stitch.py cannot repair its 91.1% tracklet-purity ceiling because splitting
    impure tracklets on intra-tracklet teleports fires 1682 times on 189
    tracklets: "frame-to-frame speed in the pitch plane is not a measurement of
    the player here, it is a measurement of the calibration - a far-side box
    wobbling one pixel moves metres". It concludes the repair "will stay closed
    until calibration improves. Line-based calibration is the standing upgrade."

    That conclusion rests on a number nobody currently has. NOISE_TOLERANCE.md
    carried a metres-per-pixel table, but that file ends in a blanket retraction
    - "Do not use any number in this file", a 5x-stretched clock - and while
    this particular quantity is pure geometry and clock-independent, a retracted
    file is not a citable source. So it is re-derived here from the calibration
    the pipeline actually ships, and this file is the citable one.

WHAT IT MEASURES
    The Jacobian of the image -> pitch map, as metres per pixel, separately for
    horizontal and VERTICAL image motion, binned by how far across the pitch the
    point lands. Vertical is the one that matters: a detector's box bottom is
    the position estimate, and a grazing camera foreshortens vertical image
    motion enormously at the far side.

    Then the inverse question, which is the one that decides the repair: how
    many pixels of wobble does the stats layer's positional budget allow?

WHAT IT DOES NOT MEASURE
    The detector's actual per-frame box jitter in pixels. That needs cached
    detections with identities and is a separate exercise. This file gives the
    AMPLIFIER; multiply by a real pixel jitter to get metres.

Run:  python model/validation/jitter_eval.py [path/to/calibration.json]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model", "engine"))

import calibrators  # noqa: E402

#: The stats layer's positional budget. NOISE_TOLERANCE.md's sigma->distance
#: table is retracted along with the rest of that file, so this is carried here
#: as the DESIGN TARGET it has always been quoted as, not as a measured result:
#: below ~0.10 m the distance error was inside the <=5% band, above it was not.
#: Re-measuring it is a separate job; it does not change the amplifier below.
SIGMA_TARGET_M = 0.10


def jacobian_m_per_px(cal, pts_px: np.ndarray, eps: float = 0.5):
    """Central difference. Returns (horizontal, vertical) metres per pixel.

    Central rather than forward because the remap and strip calibrators are
    piecewise, and a one-sided step at a strip boundary measures the seam
    instead of the gradient.
    """
    dx = np.array([eps, 0.0])
    dy = np.array([0.0, eps])
    h = np.linalg.norm(cal.to_pitch(pts_px + dx) - cal.to_pitch(pts_px - dx), axis=1) / (2 * eps)
    v = np.linalg.norm(cal.to_pitch(pts_px + dy) - cal.to_pitch(pts_px - dy), axis=1) / (2 * eps)
    return h, v


def main(path: str) -> int:
    d = json.load(open(path))
    cal = calibrators.from_dict(d)
    L = d["pitch"]["length"]
    W = d["pitch"]["width"]

    # Grid over the image region the calibration was actually fitted in.
    # Extrapolating outside the clicked hull measures invented geometry.
    pts = np.array(list(d["points"].values()), dtype=np.float64)
    if len(pts) < 4:
        print(f"{path}: {len(pts)} clicked points, nothing to grid over")
        return 1
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    gx, gy = np.meshgrid(np.linspace(x0, x1, 160), np.linspace(y0, y1, 120))
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)

    world = cal.to_pitch(grid)
    # Keep what lands on the pitch. A point that maps off it is a point no
    # player stands on, and its gradient is not a cost anyone pays.
    on = (world[:, 0] >= 0) & (world[:, 0] <= L) & (world[:, 1] >= 0) & (world[:, 1] <= W)
    grid, world = grid[on], world[on]
    if not len(grid):
        print(f"{path}: no grid point lands on the pitch")
        return 1

    h, v = jacobian_m_per_px(cal, grid)

    print(f"{os.path.basename(os.path.dirname(path))}  "
          f"{d.get('projection_type')}  {len(d['points'])} points  "
          f"rms_error_m {d.get('rms_error_m', 0):.2f}  "
          f"({len(grid)} grid points on a {L:.0f}x{W:.0f} m pitch)\n")

    # Binned by y, which is across the pitch: the camera looks along it, so y is
    # the axis the grazing angle acts on. Near/far is what the whole question
    # is about.
    print(f"{'across-pitch band':<22}{'n':>7}{'m/px horiz':>13}{'m/px vert':>12}"
          f"{'px for σ=' + format(SIGMA_TARGET_M, '.2f') + 'm':>18}")
    edges = np.linspace(0, W, 5)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (world[:, 1] >= lo) & (world[:, 1] < hi)
        if not m.any():
            continue
        hh, vv = float(np.median(h[m])), float(np.median(v[m]))
        print(f"{f'{lo:.0f}-{hi:.0f} m':<22}{int(m.sum()):>7}"
              f"{hh:>13.3f}{vv:>12.3f}{SIGMA_TARGET_M / vv:>18.3f}")

    # DELIBERATELY NO POOLED MEDIAN. The grid is uniform in IMAGE space, and a
    # grazing camera spends most of its pixels on the near half - look at the
    # n column, four fifths of the samples land in one band. Pooling those
    # gives a pitch-wide figure that is really a near-side figure wearing a
    # pitch-wide label, which flatters the number by roughly the amount this
    # whole file exists to measure. Bands are comparable to each other because
    # each is a median within itself; only the pool is meaningless.
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (world[:, 1] >= lo) & (world[:, 1] < hi)
        if m.any():
            bands.append((float(np.median(v[m])), lo, hi, int(m.sum())))
    worst, wlo, whi, wn = max(bands)
    budget = SIGMA_TARGET_M / worst

    print(f"\nWorst band is {wlo:.0f}-{whi:.0f} m at {worst:.3f} m/px vertical "
          f"({wn} samples).")
    print(f"One pixel of vertical box wobble costs {worst:.2f} m there, so holding "
          f"σ = {SIGMA_TARGET_M:.2f} m\nneeds {budget:.3f} px of vertical stability "
          f"on the worst part of the pitch.")
    print("\nA box detector does not resolve its own box bottom to a fraction of a "
          "pixel, so\nper-frame accuracy at the far side is not reachable by a better "
          "detector. Anything\nthat closes this is temporal (a filter over many frames) "
          "or optical (more pixels on\nthe far side), NOT a better calibration model: the "
          "amplifier is set by where the\ncamera is, and refitting the same view does not "
          "move it.")
    return 0


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default = os.path.join(root, "data/jobs/118575_full_1st/calibration.json")
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else default))
