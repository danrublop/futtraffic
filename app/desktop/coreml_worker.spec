# PyInstaller spec for the CoreML/ANE worker - the Python 3.12 half.
#
# BUILT BY A DIFFERENT INTERPRETER THAN THE REST OF THE APP, which is the whole
# reason this file exists. backend.spec is frozen with the repo's 3.14 venv;
# coremltools has no working native extension for 3.14 (it installs, then dies
# at runtime with "BlobWriter not loaded"), so this one is frozen with a 3.12:
#
#   ~/.venvs/ftcoreml/bin/pyinstaller app/desktop/coreml_worker.spec --clean --noconfirm \
#       --distpath app/desktop/resources/backend --workpath app/desktop/build/coreml
#
# Output: app/desktop/resources/backend/coreml/coreml-worker. It lands INSIDE the
# backend directory on purpose - electron-builder already copies that whole
# tree to Contents/Resources/backend, so nothing in electron-builder.yml has to
# know about it, and detectors._bundled() finds it next to the frozen API
# binary. build-backend.sh drives both builds and drops the .mlpackage beside
# this one.
#
# THE EXCLUDES ARE THE POINT. coreml_worker.py's _selfcheck imports torch,
# PIL and rfdetr to diff itself against the real references, inside a try -
# PyInstaller's static analysis cannot see that it is optional and would pull
# 1.1 GB in for a check that never runs here. Excluding them also makes the
# build a proof: if the frozen worker runs a tile, the runtime closure really
# is coremltools + numpy.
import os

from PyInstaller.utils.hooks import collect_all

REPO = os.path.dirname(os.path.dirname(os.path.abspath(SPECPATH)))

# coremltools loads its MIL passes by module path and ships .proto descriptors
# and a native blob writer; static analysis finds none of it.
datas, binaries, hiddenimports = collect_all("coremltools")

a = Analysis(
    [os.path.join(REPO, "model", "engine", "coreml_worker.py")],
    pathex=[os.path.join(REPO, "model", "engine")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["torch", "torchvision", "rfdetr", "PIL", "matplotlib", "scipy",
              "transformers", "cv2", "pandas"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="coreml-worker",   # detectors.COREML_WORKER_EXE - it is matched by name
    console=True,           # its traceback is the only diagnosis the parent gets
)
coll = COLLECT(exe, a.binaries, a.datas, name="coreml")
