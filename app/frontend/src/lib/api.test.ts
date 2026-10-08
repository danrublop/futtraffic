import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api, clearStoreCache, column, whole } from "./api";

/** `detail()` is private, so it is exercised the way the app hits it: through a
 *  failing request. Every case here is a body a real server has sent us —
 *  FastAPI's JSON, a proxy's HTML error page, a truncated reply, nothing. */
const fetchSpy = (...replies: Response[]) => {
  const mock = vi.fn<(path: string, init?: RequestInit) => Promise<Response>>(
    async () => (replies.length > 1 ? (replies.shift() as Response) : replies[0]),
  );
  vi.stubGlobal("fetch", mock);
  return mock;
};

const answer = (body: string, init: ResponseInit) =>
  fetchSpy(new Response(body, init));

const failure = async (): Promise<ApiError> => {
  try {
    await api.listMatches();
  } catch (e) {
    return e as ApiError;
  }
  throw new Error("expected the request to reject");
};

afterEach(() => vi.unstubAllGlobals());

describe("error detail", () => {
  it("shows the sentence FastAPI wrote for the operator", async () => {
    const sentence =
      "The four points you marked are nearly in a line, so the pitch cannot be " +
      "solved from them. Click corners that are further apart.";
    answer(JSON.stringify({ detail: sentence }), {
      status: 400,
      statusText: "Bad Request",
    });

    const err = await failure();
    expect(err).toBeInstanceOf(ApiError);
    expect(err.name).toBe("ApiError");
    expect(err.status).toBe(400);
    expect(err.message).toBe(sentence);
  });

  it("falls back to the status line for an HTML error page", async () => {
    answer("\n  <!doctype html><html><body><h1>502 Bad Gateway</h1></body></html>", {
      status: 502,
      statusText: "Bad Gateway",
    });

    expect((await failure()).message).toBe("502 Bad Gateway");
  });

  it("falls back to the status line for an empty body", async () => {
    answer("", { status: 500, statusText: "Internal Server Error" });

    const err = await failure();
    expect(err.status).toBe(500);
    expect(err.message).toBe("500 Internal Server Error");
  });

  it("shows plain text that is not JSON and not a page", async () => {
    answer("worker died: out of memory", { status: 500, statusText: "Internal Server Error" });

    expect((await failure()).message).toBe("worker died: out of memory");
  });

  it("shows the raw body when the JSON is truncated mid-flight", async () => {
    answer('{"detail": "the pitch could not be', {
      status: 400,
      statusText: "Bad Request",
    });

    expect((await failure()).message).toBe('{"detail": "the pitch could not be');
  });

  it("keeps something readable when detail is not a string", async () => {
    // FastAPI's 422 puts a LIST in `detail`, one entry per invalid field. There
    // is no sentence to show, so the body itself is what reaches the user.
    // BUG (api.ts, not ours): this surfaces raw JSON on a validation error.
    const body = JSON.stringify({ detail: [{ loc: ["body", "total"], msg: "field required" }] });
    answer(body, { status: 422, statusText: "Unprocessable Entity" });

    const err = await failure();
    expect(err.status).toBe(422);
    expect(err.message).toBe(body);
  });
});

describe("req", () => {
  it("returns the parsed body on success", async () => {
    answer(JSON.stringify([{ id: "m1" }]), { status: 200, statusText: "OK" });

    await expect(api.listMatches()).resolves.toEqual([{ id: "m1" }]);
  });

  it("does not try to parse a 204", async () => {
    // A 204 body is empty; JSON.parse would throw where nothing is wrong.
    fetchSpy(new Response(null, { status: 204 }));

    await expect(api.saveIdentity("m1", {})).resolves.toBeUndefined();
  });

  it("never caches: a poll must see the new status, not the old one", async () => {
    const fetchMock = fetchSpy(new Response("[]", { status: 200 }));

    await api.listMatches();
    expect(fetchMock).toHaveBeenCalledWith("/api/matches", { cache: "no-store" });
  });

  it("escapes the match id in the path", async () => {
    const fetchMock = fetchSpy(new Response("{}", { status: 200 }));

    await api.getMatch("a b/c");
    expect(fetchMock.mock.calls[0][0]).toBe("/api/matches/a%20b%2Fc");
  });

  it("falls back to the match itself when retry is refused", async () => {
    // The job may already have been requeued by another tab; a failed retry
    // must not leave the card showing a stale state.
    const fetchMock = fetchSpy(
      new Response("nope", { status: 409, statusText: "Conflict" }),
      new Response(JSON.stringify({ id: "m1" }), { status: 200 }),
    );

    await expect(api.retry("m1")).resolves.toEqual({ id: "m1" });
    expect(fetchMock.mock.calls[0][0]).toBe("/api/matches/m1/retry");
    expect(fetchMock.mock.calls[1][0]).toBe("/api/matches/m1");
  });
});

describe("roster", () => {
  it("PUTs the list and returns the server's canonical ids, not the input", async () => {
    // The whole reason saveRoster's reply is used rather than its argument:
    // roster.py mints player_id by accent-folding the name. A caller that
    // guessed "jose-ruiz" from "José Ruiz" locally would agree here by luck and
    // disagree on the next accent, binding the track to nothing in silence.
    const fetchMock = fetchSpy(
      new Response(
        JSON.stringify([
          { id: "jose-ruiz", name: "José Ruiz", number: 9, team: "Westfield" },
        ]),
        { status: 200 },
      ),
    );

    const out = await api.saveRoster([
      { name: "José Ruiz", number: 9, team: "Westfield" },
    ]);

    expect(fetchMock.mock.calls[0][0]).toBe("/api/roster");
    expect(fetchMock.mock.calls[0][1]?.method).toBe("PUT");
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual([
      { name: "José Ruiz", number: 9, team: "Westfield" },
    ]);
    // No id went up; one came back, and it is the only thing that may be sent
    // as an assignment's `player`.
    expect(out[0].id).toBe("jose-ruiz");
  });
});

/** THE TWO WRITES BEHIND THE PROFILE FORM. Both address a player by id in the
 *  PATH, which is the part of a URL that has no escaping of its own - so the
 *  encoding here is the only thing between a player id and a different route.
 *  The server's own guard (jobs._PID) is the second line, not the first. */
describe("player profile writes", () => {
  it("PUTs the bio as JSON and renders the REPLY, which carries the derived age", async () => {
    // The point of returning the stored bio: `age` is recomputed from
    // birth_date by jobs.save_bio and was never sent. A caller that rendered
    // its own payload would print no age until the next full reload.
    const fetchMock = fetchSpy(
      new Response(JSON.stringify({ name: "Ada", birth_date: "1995-07-10", age: 31 }), {
        status: 200,
      }),
    );

    const out = await api.saveBio("ida-hellstrom", { name: "Ada", birth_date: "1995-07-10" });

    expect(fetchMock.mock.calls[0][0]).toBe("/api/players/ida-hellstrom/bio");
    expect(fetchMock.mock.calls[0][1]?.method).toBe("PUT");
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({
      name: "Ada",
      birth_date: "1995-07-10",
    });
    expect(out.age).toBe(31);
  });

  it("sends the content-type FastAPI needs to read the body as JSON", async () => {
    const fetchMock = answer(JSON.stringify({}), { status: 200 });
    await api.saveBio("a", { shirt: 9 });
    expect(fetchMock.mock.calls[0][1]?.headers).toEqual({ "content-type": "application/json" });
  });

  it("keeps a punctuated id inside its own path segment", async () => {
    // A "/" in an id would otherwise address /api/players/a/b/bio - a route
    // that does not exist - and a trailing newline is the exact shape
    // jobs._PID was changed from `$` to `\\Z` to refuse. Encoding it here is
    // what puts the string in front of that guard rather than in a filename.
    const fetchMock = fetchSpy(
      new Response("{}", { status: 200 }),
      new Response("{}", { status: 200 }),
    );

    await api.saveBio("a/b", {});
    expect(fetchMock.mock.calls[0][0]).toBe("/api/players/a%2Fb/bio");

    await api.saveBio("abc\n", {});
    expect(fetchMock.mock.calls[1][0]).toBe("/api/players/abc%0A/bio");
  });

  it("PUTs the photo as the raw file, with no content-type of its own", async () => {
    // The server sniffs the magic prefix off the bytes. Naming a type here
    // would let the browser's guess (or a renamed .png) outrank the file.
    const fetchMock = answer(JSON.stringify({ ok: true, type: "png", bytes: 12 }), { status: 200 });
    const file = new File([new Uint8Array([137, 80, 78, 71])], "face.png");

    const out = await api.savePhoto("ida-hellstrom", file);

    expect(fetchMock.mock.calls[0][0]).toBe("/api/players/ida-hellstrom/photo");
    expect(fetchMock.mock.calls[0][1]?.method).toBe("PUT");
    expect(fetchMock.mock.calls[0][1]?.body).toBe(file);
    expect(fetchMock.mock.calls[0][1]?.headers).toBeUndefined();
    expect(out.ok).toBe(true);
  });

  it("raises the server's sentence when the photo is refused, so the form can say why", async () => {
    answer(JSON.stringify({ detail: "that file is 14 MB" }), { status: 400 });
    await expect(api.savePhoto("a", new Blob([]))).rejects.toThrow("that file is 14 MB");
  });
});

/** THE STORE CACHE, which is invisible when it works and invisible when it
 *  breaks: a stale answer and a refetched one look identical on screen. These
 *  are the three rules the Data page depends on. */
describe("store cache", () => {
  afterEach(() => clearStoreCache());

  it("asks the server once per key, however many callers there are", async () => {
    const fetchMock = answer(JSON.stringify({}), { status: 200 });
    clearStoreCache();

    await Promise.all([api.getBios(), api.getBios()]);
    await api.getBios();

    expect(fetchMock.mock.calls.length).toBe(1);
  });

  it("does not cache a failure - the next caller tries again", async () => {
    const fetchMock = fetchSpy(
      new Response("", { status: 500 }),
      new Response(JSON.stringify({}), { status: 200 }),
    );
    clearStoreCache();

    await expect(api.getBios()).rejects.toBeInstanceOf(ApiError);
    await expect(api.getBios()).resolves.toEqual({});
    expect(fetchMock.mock.calls.length).toBe(2);
  });

  it("drops everything when the store is rebuilt", async () => {
    // Two Responses, not one replayed: a body can only be read once.
    const fetchMock = fetchSpy(
      new Response(JSON.stringify({}), { status: 200 }),
      new Response(JSON.stringify({}), { status: 200 }),
    );
    clearStoreCache();

    await api.getBios();
    clearStoreCache();
    await api.getBios();

    expect(fetchMock.mock.calls.length).toBe(2);
  });
});

/** THE TWO BOUNDARY GUARDS, each written because its absence was invisible:
 *  a store read cut off at the row cap looks like a smaller club, not an
 *  error, and a renamed column reads as "no data" in every field. Both must
 *  throw, or they are comments. */
describe("store boundary guards", () => {
  const result = (truncated: boolean) => ({ columns: ["player_id", "goals"], rows: [], truncated });

  it("whole() passes a complete result through untouched", () => {
    const r = result(false);
    expect(whole(r, "appearances")).toBe(r);
  });

  it("whole() refuses a result that hit the row cap, naming the read", () => {
    expect(() => whole(result(true), "team stats")).toThrow(/team stats/);
  });

  it("column() finds a column by name", () => {
    const i = column(result(false));
    expect(i("player_id")).toBe(0);
    expect(i("goals")).toBe(1);
  });

  it("column() throws on a missing column instead of answering -1", () => {
    // row[-1] is undefined, which every mapper would have read as "no goals".
    expect(() => column(result(false))("assists")).toThrow(/assists/);
  });
});

describe("cropUrl", () => {
  it("asks the server for a frame only when the scrubber names one", () => {
    // No `f` means the track's middle frame, which is what every gallery wants;
    // the split dialog is the one caller that picks the moment itself.
    expect(api.cropUrl("m1", 7, 220)).not.toMatch(/[&?]f=/);
    expect(api.cropUrl("m1", 7, 220, 0)).toMatch(/[&?]f=0(&|$)/);
    expect(api.cropUrl("m1", 7, 220, 1500)).toMatch(/[&?]f=1500(&|$)/);
  });
});
