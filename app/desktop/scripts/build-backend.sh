#!/bin/bash
# Freeze app/server/app.py into app/desktop/resources/backend/ with PyInstaller.
# The same binary serves the API (no args) and runs tracking jobs
# (`futtraffic-api run <job id>`).
set -euo pipefail

DESKTOP="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(dirname "$(dirname "$DESKTOP")")"
VENV="$REPO/.venv"

if [ ! -x "$VENV/bin/python" ]; then
  echo "no venv at $VENV - run the README quick start first" >&2
  exit 1
fi

if ! "$VENV/bin/python" -c "import PyInstaller" 2>/dev/null; then
  echo "installing build dependencies (app/server/requirements-build.txt: rfdetr + pyinstaller)"
  "$VENV/bin/pip" install -r "$REPO/app/server/requirements-build.txt"
fi

# The venv's OpenCV decides what FFmpeg lands in the bundle, so it is checked
# BEFORE the hour of freezing rather than after: the PyPI macOS wheel is built
# --enable-gpl and a closed-source app may not be distributed against it.
# app/desktop/scripts/build-lgpl-opencv.sh builds the replacement.
CVLIB="$("$VENV/bin/python" -c 'import cv2,os;print(os.path.dirname(cv2.__file__))')"
# Not `| grep -q`: it exits at the first match, strings takes SIGPIPE, and
# pipefail reads that as a GPL verdict on an LGPL build.
CVLICENCE="$(strings -a "$CVLIB"/.dylibs/libavcodec.*.dylib 2>/dev/null || true)"
if [[ "$CVLICENCE" != *"license: LGPL"* ]]; then
  echo "the venv's OpenCV carries a GPL FFmpeg - that cannot be distributed." >&2
  echo "  app/desktop/scripts/build-lgpl-opencv.sh" >&2
  echo "  $VENV/bin/pip install --force-reinstall --no-deps \\" >&2
  echo "      $DESKTOP/build/wheels/opencv_python_headless-*.whl" >&2
  echo "  See docs/audits/ffmpeg-lgpl.md." >&2
  exit 1
fi

"$VENV/bin/pyinstaller" "$DESKTOP/backend.spec" --clean --noconfirm \
  --distpath "$DESKTOP/resources" --workpath "$DESKTOP/build"

# What actually shipped, by name and by load command. The check above is about
# the venv; this one is about the tree electron-builder will pick up.
python3 "$DESKTOP/check-gpl.py" "$DESKTOP/resources"

# ---------------------------------------------------------------- the ANE half
# AFTER the backend build, never before: PyInstaller's COLLECT wipes its output
# directory, and this one writes INSIDE it (resources/backend/coreml).
#
# WHY IT IS A SECOND BUILD. coremltools has no working native extension for the
# repo's Python 3.14 - it installs and dies at runtime with "BlobWriter not
# loaded" - so the CoreML detector runs in a 3.12 subprocess. Frozen, that
# subprocess is 59 MB: the worker imports coremltools and numpy and nothing
# else (the torch/rfdetr decode it used to borrow is inlined in numpy, and
# coreml_worker._selfcheck diffs the inline against the originals). Plus the
# 58 MB .mlpackage, 117 MB total, against a 971 MB backend.
#
# WHAT IT BUYS: one half goes from ~16.5 h to ~3.9 h. Without it, jobs.py sees
# no interpreter and no model, records st["detector"] = "rfdetr" and runs the
# slow path - correct, honest, and four times too slow.
#
# OPTIONAL BY DESIGN. This needs a 3.12 with coremltools (and, only to export
# the model, rfdetr). A machine without one still produces a working app; it
# produces a SLOW one, and says so loudly rather than at run time.
CORE_PY="${FT_COREML_BUILD_PYTHON:-$HOME/.venvs/ftcoreml/bin/python}"
MLPKG="$REPO/model/models/rfdetr-medium-576.mlpackage"

if [ ! -x "$CORE_PY" ]; then
  echo "" >&2
  echo "WARNING: no Python 3.12 at $CORE_PY - shipping WITHOUT the Neural Engine" >&2
  echo "  path. Every job in this build runs PyTorch: ~16.5 h a half, not ~3.9 h." >&2
  echo "  Fix: python3.12 -m venv ~/.venvs/ftcoreml && ~/.venvs/ftcoreml/bin/pip \\" >&2
  echo "         install coremltools numpy pyinstaller rfdetr" >&2
  echo "  or set FT_COREML_BUILD_PYTHON." >&2
else
  "$CORE_PY" -c "import PyInstaller" 2>/dev/null || \
    "$(dirname "$CORE_PY")/pip" install pyinstaller

  # The .mlpackage is 58 MB and /models/ is gitignored, so a clean checkout has
  # no model. Exporting one takes ~10 s and needs rfdetr in the 3.12 env; that
  # is cheaper than putting 58 MB of fp16 weights in git, and it is the same
  # export the measurements were taken against (medium, fp16, 576, batch 1 -
  # batching on the ANE is a LOSS, see detectors.CoreMLDetector).
  if [ ! -d "$MLPKG" ]; then
    echo "exporting $MLPKG (rfdetr, ~10 s)"
    MLPKG="$MLPKG" "$CORE_PY" - <<'PYEOF' || { echo "export failed - is rfdetr installed in the 3.12 env?" >&2; exit 1; }
import os
import shutil
from pathlib import Path

from rfdetr import RFDETRMedium

want = Path(os.environ["MLPKG"])
out = want.parent / "export"
# export() returns the path it chose; rfdetr names the file, not us, so move it
# to the one name detectors.COREML_MODEL resolves rather than guess the name.
got = Path(RFDETRMedium().export(format="coreml", coreml_precision="float16",
                                 shape=(576, 576), batch_size=1, output_dir=str(out)))
assert got.suffix == ".mlpackage" and got.exists(), got
want.parent.mkdir(parents=True, exist_ok=True)
shutil.move(str(got), str(want))
shutil.rmtree(out, ignore_errors=True)
print(f"exported {want}")
PYEOF
  fi

  "$CORE_PY" -m PyInstaller "$DESKTOP/coreml_worker.spec" --clean --noconfirm \
    --distpath "$DESKTOP/resources/backend" --workpath "$DESKTOP/build/coreml"
  cp -R "$MLPKG" "$DESKTOP/resources/backend/coreml/"

  # Smoke test: the frozen worker must load the .mlpackage and decode 300
  # queries. It is the only thing that proves the excludes in coreml_worker.spec
  # did not cut something the runtime needs - a bundle that merely EXISTS is
  # what makes a packaged install silently slow in the first place.
  "$DESKTOP/resources/backend/coreml/coreml-worker" --selfcheck \
    "$DESKTOP/resources/backend/coreml/rfdetr-medium-576.mlpackage" || {
      echo "frozen CoreML worker failed its self-check - removing it so the app" >&2
      echo "falls back to PyTorch honestly rather than dying mid-job" >&2
      rm -rf "$DESKTOP/resources/backend/coreml"
      exit 1
    }
fi

# Smoke test: boot the frozen API on a throwaway port and data dir, probe it,
# kill it. (`futtraffic-api --help` would start the server - there is no CLI.)
TMP="$(mktemp -d)"
PORT=8123 FUTTRAFFIC_DATA="$TMP" "$DESKTOP/resources/backend/futtraffic-api" \
  >"$TMP/boot.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; rm -rf "$TMP"' EXIT
for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:8123/docs >/dev/null 2>&1 && break
  sleep 1
done
curl -sf http://127.0.0.1:8123/docs >/dev/null || {
  echo "frozen backend failed to serve /docs - boot log:" >&2
  tail -20 "$TMP/boot.log" >&2
  exit 1
}
# What this line may and may not claim. The frozen worker's OWN self-check ran
# further up as a build gate - it loaded the real .mlpackage and decoded 300
# queries - so "the worker works" is earned. What is NOT tested anywhere is the
# composition: a frozen PARENT spawning this frozen CHILD. The parent resolves
# the path through detectors._bundled (verified against a real bundle) and the
# child is verified standalone, but the two have never run together, and the
# first real job is not where that should be discovered.
#
# This used to print "ANE worker OK" off a bare -x test, which is a check that
# cannot fail in the way that matters: an executable bit says nothing about
# whether the parent can find or spawn it.
if [ -x "$DESKTOP/resources/backend/coreml/coreml-worker" ]; then
  echo "backend frozen -> $DESKTOP/resources/backend/ (boots, serves /docs;"
  echo "  ANE worker present and its own self-check passed; frozen parent -> frozen"
  echo "  child spawn is NOT covered here - run one real job to close it)"
else
  echo "backend frozen -> $DESKTOP/resources/backend/ (boots and serves /docs; NO ANE path)"
fi
