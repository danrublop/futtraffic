"""Put model/engine/ on sys.path. Import this first, before any engine module.

The two layers are flat script directories, not packages: every module still
runs directly and self-checks itself (`python model/engine/stats.py`), which is this
repo's entire test suite. Python only puts the *running script's* directory on
sys.path, so a server module invoked as `python app/server/app.py` can see its own
siblings but not engine's.

Deliberately a sibling of model/engine/ rather than a nested package. Eleven modules
derive the repo root as `__file__ -> parent.parent`, and paths.py's docstring
records what happened last time that depth changed: the anchors pointed one
directory too deep, which does not raise - it makes fixtures "missing" and
tests SKIP green. Same depth, same anchors, nothing to re-derive.
"""
import os
import sys

_ENGINE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model", "engine")
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)
