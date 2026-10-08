"""Fail if model/engine/ imports app/server/. Run it: `python app/server/layercheck.py`.

The split is only worth having if something enforces it. model/engine/ is the model -
video in, numbers out - and it must run with no FastAPI, no SQLite store and no
job queue on the path, because that is what makes it testable on footage alone
and what lets the worker subprocess stay a plain `python` invocation.

Nothing structural stops a future `import jobs` at the top of a vision module,
and the failure is quiet: it works fine on a dev box where both directories are
importable, then dies in the frozen bundle or drags FastAPI into a validation
harness. headshots.py did exactly this - it lived in the model layer and reached
for jobs.write_photo - which is why it now sits in app/server/ instead.

AST, not grep: an import inside a function body is still a dependency, and this
codebase uses lazy imports heavily to break real cycles (pitch <-> calibrators,
track <-> sceneprior), so they cannot be waved through.
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ENGINE = os.path.join(REPO, "model", "engine")


def _imported_names(path: str) -> list[tuple[str, int]]:
    """Every module imported by `path`, top-level or inside a function."""
    with open(path) as fh:
        tree = ast.parse(fh.read(), path)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(a.name.split(".")[0], node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.module.split(".")[0], node.lineno))
    return out


def violations() -> list[str]:
    server = {f[:-3] for f in os.listdir(HERE) if f.endswith(".py")}
    bad = []
    for f in sorted(os.listdir(ENGINE)):
        if not f.endswith(".py"):
            continue
        for name, line in _imported_names(os.path.join(ENGINE, f)):
            if name in server:
                bad.append(f"model/engine/{f}:{line} imports server module '{name}'")
    return bad


if __name__ == "__main__":
    bad = violations()
    for b in bad:
        print(b)
    n = len([f for f in os.listdir(ENGINE) if f.endswith(".py")])
    print(f"layercheck.py {'FAILED' if bad else 'ok'} ({n} engine modules scanned)")
    sys.exit(1 if bad else 0)
