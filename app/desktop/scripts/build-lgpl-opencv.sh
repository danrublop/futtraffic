#!/bin/bash
# Build an opencv-python-headless wheel whose FFmpeg is LGPL, not GPL.
#
# The PyPI macOS wheel bundles Homebrew's FFmpeg, which is built --enable-gpl
# (x264, x265, rubberband, vidstab), and a closed-source app cannot ship that.
# See ffmpeg-lgpl (not published). This builds FFmpeg 7.1.1 - the version the PyPI
# wheel carries, so the sonames and the decoders match - with GPL off and no
# external libraries but dav1d (BSD, for AV1), then OpenCV against it, then
# vendors the dylibs into the wheel with delocate.
#
# Not the conda LGPL FFmpeg ffmpeg-lgpl (not published) §6 names: that one links 18
# further @rpath libraries (cairo, glib, rsvg...) that would all need vendoring.
#
# Verified 2026-09-19: decoded frames bit-identical to the PyPI wheel on mp4
# (h264, hevc), mov, mkv, webm (vp9, av1), avi, ts and mpg, and on 100 frames
# of a real 4096x1080 panorama.
#
#   app/desktop/scripts/build-lgpl-opencv.sh      # -> app/desktop/build/wheels/*.whl (work in ~/.cache)
#   .venv/bin/pip install --force-reinstall --no-deps app/desktop/build/wheels/opencv_python_headless-*.whl
#
# Needs: Homebrew python@3.14, cmake, pkg-config, dav1d. ~10 min on 8 cores.
set -euo pipefail

DESKTOP="$(cd "$(dirname "$0")/.." && pwd)"
# Outside the repo: FFmpeg's build breaks on a space in the path ("fut traffic").
WORK="${LGPL_WORK:-$HOME/.cache/futtraffic/lgpl-opencv}"
OUT="$DESKTOP/build/wheels"
FFV=7.1.1
CVV=5.0.0.93
PY="${PYTHON:-/opt/homebrew/bin/python3.14}"
mkdir -p "$WORK" "$OUT"
cd "$WORK"

if [ ! -f ff/lib/libavcodec.dylib ]; then
  curl -sSL -o ffmpeg.tar.xz "https://ffmpeg.org/releases/ffmpeg-$FFV.tar.xz"
  tar -xf ffmpeg.tar.xz
  (cd "ffmpeg-$FFV" &&
    PKG_CONFIG_PATH="$(brew --prefix dav1d)/lib/pkgconfig" ./configure --prefix="$WORK/ff" \
      --enable-shared --disable-static --enable-pthreads --enable-version3 \
      --disable-gpl --disable-nonfree --disable-programs --disable-doc \
      --disable-avdevice --disable-avfilter --disable-network \
      --enable-libdav1d --enable-videotoolbox --enable-audiotoolbox --enable-neon &&
    make -j"$(sysctl -n hw.ncpu)" && make install)
fi
# Not `| grep -q`: it exits at the first match, strings takes SIGPIPE, and
# pipefail reads that as a failure.
[[ "$(strings -a ff/lib/libavcodec.dylib)" == *"libavcodec license: LGPL"* ]] ||
  { echo "built FFmpeg is not LGPL" >&2; exit 1; }

[ -x venv/bin/python ] || "$PY" -m venv venv
venv/bin/pip -q install -U pip delocate
rm -rf dist fixed
ENABLE_HEADLESS=1 PKG_CONFIG_LIBDIR="$WORK/ff/lib/pkgconfig" \
  CMAKE_BUILD_PARALLEL_LEVEL="$(sysctl -n hw.ncpu)" \
  CMAKE_ARGS="-DWITH_FFMPEG=ON -DWITH_AVFOUNDATION=ON -DOPENCV_FFMPEG_ENABLE_LIBAVDEVICE=OFF" \
  venv/bin/pip wheel --no-deps --no-binary opencv-python-headless \
    "opencv-python-headless==$CVV" -w dist/
venv/bin/delocate-wheel -w fixed dist/opencv_python_headless-*.whl

if unzip -l fixed/opencv_python_headless-*.whl | grep -E "x264|x265|rubberband|vidstab"; then
  echo "GPL library in the wheel" >&2
  exit 1
fi
cp fixed/opencv_python_headless-*.whl "$OUT/"
ls "$OUT"/opencv_python_headless-*.whl
