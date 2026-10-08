import { afterEach, describe, expect, it, vi } from "vitest";

import { dispatchEvent, parseEventLine, postRun, stopRun, type RunHandlers } from "./agent";

/** Every test goes through postRun the way the app does — the parser is
 *  exported, but the behaviour that matters is what comes out the far side of
 *  the stream. */
const streamOf = (...chunks: string[]) =>
  new Response(
    new ReadableStream<Uint8Array>({
      start(c) {
        const enc = new TextEncoder();
        for (const ch of chunks) c.enqueue(enc.encode(ch));
        c.close();
      },
    }),
    { status: 200, headers: { "content-type": "text/event-stream" } },
  );

const fetchSpy = (res: Response) => {
  const mock = vi.fn<(url: string, init?: RequestInit) => Promise<Response>>(async () => res);
  vi.stubGlobal("fetch", mock);
  return mock;
};

/** Collects every handler call as a plain tuple, so assertions read as the
 *  list of frames the UI would have seen. */
const recorder = () => {
  const seen: unknown[][] = [];
  const handlers: RunHandlers = {
    onStart: (chatId) => seen.push(["start", chatId]),
    onToken: (text) => seen.push(["token", text]),
    onToolStart: (name, input) => seen.push(["tool_start", name, input]),
    onToolDone: (name, summary, durationMs) => seen.push(["tool_done", name, summary, durationMs]),
    onUiAction: (action) => seen.push(["ui_action", action]),
    onDone: (chatId) => seen.push(["done", chatId]),
    onError: (message) => seen.push(["error", message]),
  };
  return { seen, handlers };
};

const line = (obj: unknown) => `data: ${JSON.stringify(obj)}\n`;

afterEach(() => vi.unstubAllGlobals());

describe("parseEventLine", () => {
  it("parses a data line", () => {
    expect(parseEventLine('data: {"type":"token","text":"hi"}')).toEqual({
      type: "token",
      text: "hi",
    });
  });

  it("skips blank keep-alives, comments, and malformed JSON", () => {
    expect(parseEventLine("")).toBeNull();
    expect(parseEventLine(": heartbeat")).toBeNull();
    expect(parseEventLine("event: message")).toBeNull();
    expect(parseEventLine('data: {"type":"token","text":"unterminated')).toBeNull();
    expect(parseEventLine("data: not json at all")).toBeNull();
    expect(parseEventLine("data: [1,2,3]")).toBeNull();
  });
});

describe("postRun", () => {
  it("posts the run body to /api/agent/run", async () => {
    const mock = fetchSpy(streamOf(line({ type: "done", chat_id: "c1" })));

    await postRun({ chat_id: "c0", match_id: "m1", message: "stats?" }, {});

    expect(mock.mock.calls[0][0]).toBe("/api/agent/run");
    const init = mock.mock.calls[0][1] as RequestInit;
    expect(init.method).toBe("POST");
    expect(JSON.parse(String(init.body))).toEqual({
      chat_id: "c0",
      match_id: "m1",
      model: null,
      team_id: null,
      message: "stats?",
    });
  });

  it("carries the picked model", async () => {
    const mock = fetchSpy(streamOf(line({ type: "done", chat_id: "c1" })));
    await postRun({ message: "hard one", model: "claude-opus-5" }, {});
    const init = mock.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(String(init.body)).model).toBe("claude-opus-5");
  });

  it("dispatches every event type in order", async () => {
    fetchSpy(
      streamOf(
        line({ type: "token", text: "The " }) +
          line({ type: "tool_start", name: "get_report", input: { match_id: "m1" } }) +
          line({ type: "tool_done", name: "get_report", summary: "report loaded", duration_ms: 42 }) +
          line({ type: "ui_action", action: "open_tab", tab: "stats" }) +
          line({ type: "ui_action", action: "seek_film", t: 63.5 }) +
          line({ type: "token", text: "report." }) +
          line({ type: "done", chat_id: "c1" }),
      ),
    );
    const { seen, handlers } = recorder();

    await postRun({ message: "go" }, handlers);

    expect(seen).toEqual([
      ["token", "The "],
      ["tool_start", "get_report", { match_id: "m1" }],
      ["tool_done", "get_report", "report loaded", 42],
      ["ui_action", { action: "open_tab", tab: "stats" }],
      ["ui_action", { action: "seek_film", t: 63.5 }],
      ["token", "report."],
      ["done", "c1"],
    ]);
  });

  it("surfaces an error frame to onError", async () => {
    fetchSpy(streamOf(line({ type: "error", message: "the model is offline" })));
    const { seen, handlers } = recorder();

    await postRun({ message: "go" }, handlers);

    expect(seen).toEqual([["error", "the model is offline"]]);
  });

  it("reassembles a JSON frame split across read boundaries", async () => {
    const full = line({ type: "token", text: "split me" });
    fetchSpy(streamOf(full.slice(0, 9), full.slice(9, 23), full.slice(23)));
    const { seen, handlers } = recorder();

    await postRun({ message: "go" }, handlers);

    expect(seen).toEqual([["token", "split me"]]);
  });

  it("skips a malformed line and keeps the run alive", async () => {
    fetchSpy(
      streamOf(
        'data: {"type":"token","text":"a"}\n' +
          "data: {garbage\n" +
          ": heartbeat\n" +
          "\n" +
          'data: {"type":"token","text":"b"}\n',
      ),
    );
    const { seen, handlers } = recorder();

    await postRun({ message: "go" }, handlers);

    expect(seen).toEqual([
      ["token", "a"],
      ["token", "b"],
    ]);
  });

  it("dispatches a final frame that has no trailing newline", async () => {
    fetchSpy(streamOf('data: {"type":"done","chat_id":"c9"}'));
    const { seen, handlers } = recorder();

    await postRun({ message: "go" }, handlers);

    expect(seen).toEqual([["done", "c9"]]);
  });

  it("rejects on a non-2xx response instead of parsing the body as SSE", async () => {
    fetchSpy(new Response("upstream down", { status: 502, statusText: "Bad Gateway" }));

    await expect(postRun({ message: "go" }, {})).rejects.toMatchObject({
      name: "ApiError",
      status: 502,
      message: "upstream down",
    });
  });

  it("passes the abort signal to fetch, so a cancelled run rejects", async () => {
    // The mock honours the signal the way real fetch does: an already-aborted
    // signal rejects before a byte is read.
    vi.stubGlobal(
      "fetch",
      vi.fn((_url: string, init?: RequestInit) => {
        if (init?.signal?.aborted) {
          return Promise.reject(new DOMException("The operation was aborted.", "AbortError"));
        }
        return Promise.resolve(streamOf(line({ type: "done", chat_id: "c1" })));
      }),
    );

    const ctrl = new AbortController();
    ctrl.abort();
    await expect(postRun({ message: "go" }, {}, ctrl.signal)).rejects.toMatchObject({
      name: "AbortError",
    });

    // And unaborted, the same call streams normally.
    const { seen, handlers } = recorder();
    await postRun({ message: "go" }, handlers, new AbortController().signal);
    expect(seen).toEqual([["done", "c1"]]);
  });
});

describe("ui_action with names", () => {
  it("carries the match and note the server checked", () => {
    const { seen, handlers } = recorder();
    dispatchEvent({ type: "ui_action", action: "open_tab", tab: "docs", note: "club/a.html", match_id: "m1" }, handlers);
    dispatchEvent({ type: "ui_action", action: "seek_film", t: 12, match_id: "m1" }, handlers);
    dispatchEvent({ type: "ui_action", action: "open_tab", tab: "film" }, handlers);
    expect(seen).toEqual([
      ["ui_action", { action: "open_tab", tab: "docs", match_id: "m1", note: "club/a.html" }],
      ["ui_action", { action: "seek_film", t: 12, match_id: "m1" }],
      ["ui_action", { action: "open_tab", tab: "film" }],
    ]);
  });
});

describe("stopRun", () => {
  it("posts to the chat's stop route and reports whether a run was live", async () => {
    const mock = fetchSpy(new Response(JSON.stringify({ stopped: true }), { status: 200 }));
    expect(await stopRun("abc")).toBe(true);
    expect(mock.mock.calls[0][0]).toBe("/api/agent/chats/abc/stop");
    expect((mock.mock.calls[0][1] as RequestInit).method).toBe("POST");
  });
});

describe("dispatchEvent", () => {
  it("routes the start frame, which names the chat before anything else", () => {
    const { seen, handlers } = recorder();
    dispatchEvent({ type: "start", chat_id: "c9" }, handlers);
    expect(seen).toEqual([["start", "c9"]]);
  });

  it("ignores an event type it does not speak", () => {
    const { seen, handlers } = recorder();
    // The backend may ship a frame before the frontend learns it; a run must
    // not die of one.
    dispatchEvent({ type: "progress", pct: 50 } as never, handlers);
    expect(seen).toEqual([]);
  });
});
