"""Fetch the large result files some measured claims read, from the GitHub release.

    .venv/bin/python model/validation/fetch_results.py                # the GitHub release: download, verify, unpack
    .venv/bin/python model/validation/fetch_results.py URL            # the same bundle from another URL
    .venv/bin/python model/validation/fetch_results.py PATH.tar.gz    # a bundle already on disk
    .venv/bin/python model/validation/fetch_results.py --selfcheck

WHY A RELEASE. Eighteen files under model/validation/out/ (tracker outputs, detection caches
and a ground-truth export, ~147 MB) are inputs to measured claims in the README and docs/experiments/ (n28, n36,
n40, n47, n49 and others). They are pipeline outputs on SoccerTrack v2 (CC BY 4.0), too large for git,
and regenerable from the dataset by their producer scripts. The release bundle is the shortcut.
The default URL is the release asset on github.com/danrublop/futtraffic (override with an
argument or FUTTRAFFIC_RESULTS_URL).

The bundle is checked against SHA256 before anything is written, and only members under
model/validation/out/ are extracted - a member that would land anywhere else is refused.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = "fut-traffic-results-v1.tar.gz"
SHA256 = "6c44c9252c8e8501241801b85890d5def16fa7d517266937a08dbae55e7a2c5a"
PREFIX = "model/validation/out/"
URL = f"https://github.com/danrublop/futtraffic/releases/download/results-v1/{BUNDLE}"


def safe_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    """Every member, refusing any that is not a plain file or directory under PREFIX."""
    out = []
    for m in tar.getmembers():
        name = os.path.normpath(m.name)
        if m.isdir() and PREFIX.startswith(name.rstrip("/") + "/"):
            continue                                   # the parent folders on the way down
        if not (m.isfile() or m.isdir()) or os.path.isabs(name) or ".." in Path(name).parts \
                or not (name + "/").startswith(PREFIX):
            raise SystemExit(f"refusing bundle member {m.name!r}: outside {PREFIX}")
        out.append(m)
    return out


def unpack(data: bytes, expected: str = SHA256, root: Path = ROOT) -> list[str]:
    got = hashlib.sha256(data).hexdigest()
    if got != expected:
        raise SystemExit(f"checksum mismatch: got {got}, expected {expected}")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = safe_members(tar)
        tar.extractall(root, members=members)
    return [m.name for m in members if m.isfile()]


def _selfcheck() -> None:
    def bundle(files: dict[str, bytes]) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for name, body in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(body)
                tar.addfile(info, io.BytesIO(body))
        return buf.getvalue()

    with tempfile.TemporaryDirectory() as t:
        good = bundle({PREFIX + "stc2025/x.json": b"{}"})
        assert unpack(good, hashlib.sha256(good).hexdigest(), Path(t)) == [PREFIX + "stc2025/x.json"]
        assert (Path(t) / PREFIX / "stc2025/x.json").read_bytes() == b"{}"
        for bad_name in ("../escape.json", "/etc/x", "model/engine/track.py", PREFIX + "../../x.py"):
            bad = bundle({bad_name: b"x"})
            try:
                unpack(bad, hashlib.sha256(bad).hexdigest(), Path(t))
            except SystemExit:
                continue
            raise AssertionError(f"accepted {bad_name}")
        try:
            unpack(good, "0" * 64, Path(t))
        except SystemExit:
            pass
        else:
            raise AssertionError("accepted a checksum mismatch")
    print("fetch_results ok")


def main() -> None:
    args = sys.argv[1:]
    if args == ["--selfcheck"]:
        return _selfcheck()
    src = args[0] if args else os.environ.get("FUTTRAFFIC_RESULTS_URL", URL)
    if os.path.exists(src):
        data = Path(src).read_bytes()
    else:
        print(f"downloading {src}")
        with urllib.request.urlopen(src) as r:   # noqa: S310 - a URL the user passed
            data = r.read()
    for name in unpack(data):
        print("  ", name)


if __name__ == "__main__":
    main()
