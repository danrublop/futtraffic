"""Where things live on disk. One definition, imported everywhere.

Before this existed, eleven modules each computed the repo root from their own
`__file__`. That is fine until the files move - and when they moved into this
package every one of those anchors silently pointed one directory too deep,
which does not raise, it just makes fixtures "missing" and tests SKIP green.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

DOCS = REPO / "docs"
VALIDATION = REPO / "model" / "validation"
FOOTAGE = REPO / "data" / "footage"
# The perspective TeamTrack clips live under data/footage/, not beside it: two
# top-level dirs called footage and footage2 tell a reader nothing.
FOOTAGE2 = FOOTAGE / "soccertrack"
# Weights. Not in git (see jobs.py's note on /models/ being ignored), so
# anything here is either a developer's own file or one a module fetches on
# first use - which is what headshots.py does with the face detector.
MODELS = REPO / "model" / "models"
# Runtime state: uploads, per-job tracks, roster. Outside the web root on
# purpose - an uploaded match video must never be reachable by URL.
DATA = REPO / "data"
