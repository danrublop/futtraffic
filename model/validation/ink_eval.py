"""How much of a printed page the document actually uses.

WHY THIS EXISTS
    "The spacing is poor and the pages do not fill" is a real complaint and an
    unfalsifiable one, so every attempt to fix it argues about taste. This turns
    it into three numbers a change either moves or does not:

        ink %         share of the sheet that is not paper
        cols used %   share of the page WIDTH between the leftmost and
                      rightmost ink - the figure caps show up here
        bottom gap %  share of the page HEIGHT below the last ink - the missing
                      vertical justification shows up here

    Baseline, block gallery, 2026-09-05, before any layout work:

        mean ink 15.7%, cols used 50-62%, bottom gap 36-47%

    Read them together and against the same document. Ink alone is not a
    target: a page can be 70% white and read as dense if every gap is a
    deliberate gutter, and 40% white and read as cramped if the gaps are
    letterboxing. What these catch is the second kind.

    Ink is also not information. A pitch figure scores ~24% because the grass is
    green, and a page of nothing but pitches will look "full" here while saying
    very little. Compare like with like.

Run:  python model/validation/ink_eval.py report.pdf [--dpi 100]
Needs pdftoppm (poppler) on PATH, plus pillow and numpy.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

#: Anything below this on a 0-255 grey is ink. Deliberately generous: the point
#: is to find EMPTY page, and a very pale tint is not empty.
PAPER = 245


def measure(path: str, dpi: int = 100) -> list[dict]:
    if not shutil.which("pdftoppm"):
        raise SystemExit("pdftoppm not on PATH — brew install poppler")
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), path, os.path.join(tmp, "p")],
            check=True, capture_output=True,
        )
        for f in sorted(glob.glob(os.path.join(tmp, "p-*.png"))):
            im = np.asarray(Image.open(f).convert("L"))
            ink = im < PAPER
            rows = ink.any(axis=1)
            cols = ink.any(axis=0)
            last = int(np.nonzero(rows)[0].max()) if rows.any() else 0
            out.append({
                "page": len(out) + 1,
                "h": im.shape[0], "w": im.shape[1],
                "ink": float(ink.mean() * 100),
                # Span between first and last ink, not the count of inked
                # columns: a figure that is half the measure leaves a margin
                # here whether or not its own interior is dense.
                "cols": float(cols.mean() * 100),
                "gap": float((1 - (last + 1) / im.shape[0]) * 100),
            })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--dpi", type=int, default=100)
    a = ap.parse_args(argv)

    pages = measure(a.pdf, a.dpi)
    if not pages:
        print("no pages")
        return 1
    print(f"{'page':>5}{'ink %':>9}{'cols used %':>14}{'bottom gap %':>15}")
    for p in pages:
        print(f"{p['page']:>5}{p['ink']:>9.1f}{p['cols']:>14.1f}{p['gap']:>15.1f}")
    ink = [p["ink"] for p in pages]
    print(f"\nmean ink {np.mean(ink):.1f}%   worst page {min(ink):.1f}%   "
          f"mean cols {np.mean([p['cols'] for p in pages]):.1f}%   "
          f"mean bottom gap {np.mean([p['gap'] for p in pages]):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
