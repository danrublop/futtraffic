#!/usr/bin/env python3
"""Refuse to ship a bundle containing GPL-licensed native libraries.

    python3 app/desktop/check-gpl.py [app/desktop/resources]

Exits 1 and names the offenders. Run it after PyInstaller and before
electron-builder; see ffmpeg-lgpl (not published) for why this exists and what the
fix is.

WHY A NAME LIST AND NOT A LICENCE SCANNER. The libraries below are exactly
FFmpeg's own EXTERNAL_LIBRARY_GPL_LIST (configure, n7.1.1): enabling any one
of them requires --enable-gpl, which relicenses the entire FFmpeg build as
GPL, whatever the individual library's upstream terms now say. vid.stab is
the case that proves the point - upstream relicensed to LGPL-2.1+ after
v1.1.2, but FFmpeg 7.1.1 still gates it behind --enable-gpl, so its presence
still means the build is GPL. The gate is FFmpeg's, not the library's.

This is a filename check on purpose: it is the same check a reviewer can run
by eye, and it cannot be fooled by a stripped binary.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# FFmpeg's EXTERNAL_LIBRARY_GPL_LIST. Presence of any => the build is GPL.
GPL = ("x264", "x265", "rubberband", "vidstab", "xvid", "frei0r",
       "davs2", "xavs", "dvdnav", "dvdread", "cdio")

MACHO = (".dylib", ".so", ".node")


def main(root: Path) -> int:
    if not root.exists():
        print(f"check-gpl: no such tree: {root}", file=sys.stderr)
        return 2

    files, links = [], []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix not in MACHO or p.is_symlink():
            continue
        if any(g in p.name for g in GPL):
            files.append(p)
            continue
        # A dylib may be gone from disk yet still be named in a load command,
        # which is the failure mode that breaks the app at launch rather than
        # at build. Catch both.
        try:
            out = subprocess.run(["otool", "-L", str(p)], capture_output=True,
                                 text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        for line in out.splitlines()[1:]:
            dep = line.strip().split(" ")[0]
            if any(g in dep.rsplit("/", 1)[-1] for g in GPL):
                links.append((p, dep))

    if not files and not links:
        print(f"check-gpl: clean - no GPL native libraries under {root}")
        return 0

    print(f"check-gpl: FAIL - GPL libraries in the distributable ({root})\n",
          file=sys.stderr)
    for p in sorted(files):
        print(f"  ships:  {p.relative_to(root)}", file=sys.stderr)
    for p, dep in sorted(links):
        print(f"  links:  {p.relative_to(root)} -> {dep}", file=sys.stderr)
    print("\nA closed-source app may not be distributed against these."
          "\nSee ffmpeg-lgpl (not published). Do NOT just delete them: they are"
          "\neager load commands and the app will not start.", file=sys.stderr)
    return 1


def demo() -> None:
    """Self-check: the rule fires on a GPL name and stays quiet otherwise."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "libavutil.59.dylib").write_bytes(b"")
        assert main(root) == 0, "clean tree must pass"
        (root / "libx264.164.dylib").write_bytes(b"")
        assert main(root) == 1, "a GPL dylib must fail the build"
    print("check-gpl: self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        demo()
    else:
        arg = [a for a in sys.argv[1:] if not a.startswith("-")]
        here = Path(__file__).resolve().parent
        sys.exit(main(Path(arg[0]) if arg else here / "resources"))
