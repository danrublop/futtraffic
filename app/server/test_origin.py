"""The one check the origin guard has to pass: another website cannot drive the API.

Run it: `.venv/bin/python app/server/test_origin.py`

Any page the user visits can send a no-preflight POST to localhost:8000, and
the API has no auth - so a refused cross-origin POST is the whole defence.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import _path  # noqa: F401  - puts model/engine/ on sys.path
import app as server


def main() -> None:
    c = TestClient(server.app, base_url="http://127.0.0.1:8000")
    url = "/api/db/rebuild"   # any state-changing route; the guard runs before it

    # 1. Another site is refused, whatever the body type.
    for origin in ("https://evil.example", "http://localhost.evil.example", "null"):
        r = c.post(url, content="{}", headers={"origin": origin, "content-type": "text/plain"})
        assert r.status_code == 403, (origin, r.status_code)

    # 2. Our own UI and non-browser clients get through the guard (whatever the route then says).
    for origin in (None, "http://localhost:3000", "http://127.0.0.1:3000", "futtraffic://app"):
        h = {"origin": origin} if origin else {}
        r = c.post(url, json={}, headers=h)
        assert r.status_code != 403, (origin, r.status_code, r.text)

    # 3. A DNS-rebound name is refused even for a read.
    r = c.get("/api/matches", headers={"host": "rebind.evil.example:8000"})
    assert r.status_code == 403, r.status_code
    assert c.get("/api/matches").status_code != 403

    print("origin guard ok")


if __name__ == "__main__":
    main()
