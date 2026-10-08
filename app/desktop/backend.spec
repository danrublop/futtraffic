# PyInstaller spec for the FastAPI backend that ships inside FutTraffic.app.
#
#   .venv/bin/pip install -r app/server/requirements-build.txt
#   .venv/bin/pyinstaller app/desktop/backend.spec --clean --noconfirm \
#       --distpath app/desktop/resources --workpath app/desktop/build
#
# Output: app/desktop/resources/backend/futtraffic-api (one-dir). electron-builder
# picks the whole directory up via extraResources.
#
# The same binary is both the API server (no args) and the tracking worker
# (`futtraffic-api run <job id>`) - see the __main__ guard in app/server/app.py.
import os

from PyInstaller.utils.hooks import collect_all

SPECPATH_DIR = os.path.abspath(SPECPATH)  # directory holding this spec
REPO = os.path.dirname(os.path.dirname(SPECPATH_DIR))

datas, binaries, hiddenimports = [], [], []
# rfdetr pulls pydantic config models and registers detectors by module path;
# uvicorn loads loop/protocol implementations by string. Neither survives
# static analysis - collect them whole.
for pkg in ("rfdetr", "uvicorn", "torch", "torchvision"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += ["multipart", "cv2", "scipy.optimize", "scipy.ndimage"]

# app/server/schema.sql is the ONLY non-.py file either script dir needs, and it is
# not something collect_all() can find: db.py resolves it as a sibling of
# itself, which frozen means <_MEIPASS>/schema.sql. Without it every
# db.connect() raises FileNotFoundError - and jobs.ingest() catches Exception
# deliberately ("never fatal"), so a packaged build finished every match,
# reported done, and silently never wrote a row. The Data tab and the season
# table stayed empty with nothing anywhere saying why. It has never shipped:
# the last build's Analysis TOC has zero hits for it.
datas += [(os.path.join(REPO, "app", "server", "schema.sql"), ".")]


a = Analysis(
    [os.path.join(REPO, "app", "server", "app.py")],
    # Two script dirs, not packages. app/server/ imports model/engine/ by bare name
    # (app/server/_path.py does the same at runtime); PyInstaller needs both on
    # pathex or every engine import resolves to nothing and the bundle
    # builds clean, then dies on the first job.
    pathex=[os.path.join(REPO, "app", "server"), os.path.join(REPO, "model", "engine")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="futtraffic-api",
    console=True,  # worker.log captures stdout; do not silence it
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="backend",
)
