import { beforeEach, describe, expect, it, vi } from "vitest";

import golden from "@/test/fixtures/chat.golden.json";

import { getChat, postRun, stopRun, type ChatHistory } from "./agent";
import { useChatStore } from "./store";

/** The store is exercised against a mocked transport: these tests are about
 *  what the reducer does with frames, not about SSE (agent.test.ts owns that). */
vi.mock("./agent", () => ({
  postRun: vi.fn(),
  getChat: vi.fn(),
  stopRun: vi.fn(async () => true),
}));

const postRunMock = vi.mocked(postRun);
const getChatMock = vi.mocked(getChat);
const stopRunMock = vi.mocked(stopRun);

type Handlers = Parameters<typeof postRun>[1];

/** Runs postRun's side effects synchronously from a script of frames. */
const script =
  (fn: (h: Handlers) => void) =>
  async (_req: unknown, h: Handlers) => {
    fn(h);
  };

const state = () => useChatStore.getState();

beforeEach(() => {
  postRunMock.mockReset();
  getChatMock.mockReset();
  state().newChat();
  stopRunMock.mockClear();
});

describe("send", () => {
  it("appends the user message and streams tokens into one agent bubble", async () => {
    postRunMock.mockImplementation(
      script((h) => {
        h.onToken?.("Shots: ");
        h.onToken?.("12–7");
        h.onDone?.("c1");
      }),
    );

    await state().send("  how did it go?  ");

    expect(state().messages).toEqual([
      { role: "user", text: "how did it go?", tools: [] },
      { role: "agent", text: "Shots: 12–7", tools: [] },
    ]);
    expect(state().chatId).toBe("c1");
    expect(state().streaming).toBe(false);
    expect(state().error).toBeNull();
  });

  it("sends the stored chat_id on the next turn", async () => {
    postRunMock.mockImplementation(script((h) => h.onDone?.("c1")));
    await state().send("first");
    await state().send("second", "m1");

    expect(postRunMock.mock.calls[1][0]).toEqual({
      chat_id: "c1",
      match_id: "m1",
      model: null,
      team_id: null,
      message: "second",
    });
  });

  it("pairs tool_done with its tool_start and records summary + duration", async () => {
    postRunMock.mockImplementation(
      script((h) => {
        h.onToolStart?.("get_report", { match_id: "m1" });
        h.onToolStart?.("get_tracks", undefined);
        h.onToolDone?.("get_report", "report loaded", 42);
        h.onToolDone?.("get_tracks", "16k detections", 310);
      }),
    );

    await state().send("break it down");

    expect(state().messages[1].tools).toEqual([
      { name: "get_report", running: false, input: { match_id: "m1" }, summary: "report loaded", durationMs: 42 },
      { name: "get_tracks", running: false, summary: "16k detections", durationMs: 310 },
    ]);
  });

  it("shows an error frame as the banner and stops streaming", async () => {
    postRunMock.mockImplementation(script((h) => h.onError?.("the model is offline")));

    await state().send("anything");

    expect(state().error).toBe("the model is offline");
    expect(state().streaming).toBe(false);
    state().clearError();
    expect(state().error).toBeNull();
  });

  it("shows a transport failure as the banner", async () => {
    postRunMock.mockRejectedValue(new Error("connection lost"));

    await state().send("anything");

    expect(state().error).toBe("connection lost");
    expect(state().streaming).toBe(false);
  });

  it("does nothing for whitespace", async () => {
    await state().send("   ");
    expect(postRunMock).not.toHaveBeenCalled();
    expect(state().messages).toEqual([]);
  });

  it("aborts the live run when a new message arrives mid-stream", async () => {
    const signals: AbortSignal[] = [];
    // The first run never finishes on its own — it is the still-streaming
    // answer the new message has to kill.
    postRunMock.mockImplementation(async (_req, _h, signal) => {
      signals.push(signal as AbortSignal);
      if (signals.length === 1) {
        await new Promise((_, reject) => {
          signal?.addEventListener("abort", () =>
            reject(new DOMException("The operation was aborted.", "AbortError")),
          );
        });
      }
    });

    const first = state().send("slow question");
    expect(state().streaming).toBe(true);
    await state().send("faster question");
    await first;

    expect(signals[0].aborted).toBe(true);
    expect(signals[1].aborted).toBe(false);
    // The aborted run's failure must not surface as an error banner…
    expect(state().error).toBeNull();
    // …and its finally must not have cleared the SECOND run's flag; the flag
    // is only down because run two also settled.
    expect(state().streaming).toBe(false);
    expect(state().messages[0].text).toBe("slow question");
    expect(state().messages[2].text).toBe("faster question");
  });

  it("abort() cancels the run and drops the streaming flag", async () => {
    let signal: AbortSignal | undefined;
    postRunMock.mockImplementation(async (_req, _h, sig) => {
      signal = sig as AbortSignal;
      await new Promise((_, reject) => {
        sig?.addEventListener("abort", () =>
          reject(new DOMException("The operation was aborted.", "AbortError")),
        );
      });
    });

    const run = state().send("slow");
    state().abort();
    await run;

    expect(signal?.aborted).toBe(true);
    expect(state().streaming).toBe(false);
    expect(state().error).toBeNull();
  });

  it("Stop tells the server too, naming the chat from the start frame, and marks the answer", async () => {
    // The desktop proxy may not forward an aborted fetch, so an abort alone
    // left the run spending on the server.
    postRunMock.mockImplementation(async (_req, h, sig) => {
      h.onStart?.("c7");
      h.onToken?.("Half an ans");
      await new Promise((_, reject) => {
        sig?.addEventListener("abort", () =>
          reject(new DOMException("The operation was aborted.", "AbortError")),
        );
      });
    });

    const run = state().send("long question");
    expect(state().chatId).toBe("c7");
    state().abort();
    await run;

    expect(stopRunMock).toHaveBeenCalledWith("c7");
    expect(state().messages[1]).toEqual({ role: "agent", text: "Half an ans", tools: [], stopped: true });
  });

  it("abort() with nothing running does not call the server", () => {
    state().abort();
    expect(stopRunMock).not.toHaveBeenCalled();
  });

  it("a new chat is created in the picked squad, and an existing one never re-filed by a send", async () => {
    postRunMock.mockImplementation(script((h) => h.onDone?.("c1")));
    state().newChat("u15");
    await state().send("first");
    expect(postRunMock.mock.calls[0][0]).toMatchObject({ chat_id: null, team_id: "u15" });
    await state().send("second");
    expect(postRunMock.mock.calls[1][0]).toMatchObject({ chat_id: "c1", team_id: null });
  });

  it("reopening a chat takes its squad", async () => {
    getChatMock.mockResolvedValue({ id: "c2", team_id: "first", messages: [] });
    state().newChat("u15");
    await state().loadChat("c2");
    expect(state().teamId).toBe("first");
  });

  it("titles a new chat with its first message, keeps it on later turns, and takes a reopened chat's", async () => {
    postRunMock.mockImplementation(script((h) => h.onDone?.("c1")));
    expect(state().title).toBeNull();
    await state().send("x".repeat(70));
    expect(state().title).toBe("x".repeat(60));
    await state().send("second");
    expect(state().title).toBe("x".repeat(60));
    getChatMock.mockResolvedValue({ id: "c2", title: "Pressing", messages: [] });
    await state().loadChat("c2");
    expect(state().title).toBe("Pressing");
    state().newChat();
    expect(state().title).toBeNull();
  });

  it("passes the picked model", async () => {
    postRunMock.mockImplementation(script((h) => h.onDone?.("c1")));
    await state().send("hard one", null, "claude-opus-5");
    expect(postRunMock.mock.calls[0][0]).toMatchObject({ model: "claude-opus-5" });
  });
});

describe("ui_action", () => {
  it("parks the action until the workspace consumes it — once", async () => {
    postRunMock.mockImplementation(
      script((h) => h.onUiAction?.({ action: "open_tab", tab: "stats" })),
    );

    await state().send("show the stats");

    expect(state().pendingUiAction).toEqual({ action: "open_tab", tab: "stats" });
    expect(state().consumeUiAction()).toEqual({ action: "open_tab", tab: "stats" });
    expect(state().pendingUiAction).toBeNull();
    // Consumed is consumed: a second read must not re-open the tab.
    expect(state().consumeUiAction()).toBeNull();
  });

  it("the latest action wins while one is still pending", async () => {
    postRunMock.mockImplementation(
      script((h) => {
        h.onUiAction?.({ action: "open_tab", tab: "film" });
        h.onUiAction?.({ action: "seek_film", t: 63.5 });
      }),
    );

    await state().send("the goal");

    expect(state().pendingUiAction).toEqual({ action: "seek_film", t: 63.5 });
  });
});

describe("newChat", () => {
  it("resets the conversation and any pending action", async () => {
    postRunMock.mockImplementation(
      script((h) => {
        h.onUiAction?.({ action: "open_tab", tab: "film" });
        h.onDone?.("c1");
      }),
    );
    await state().send("hi");

    state().newChat();

    expect(state().chatId).toBeNull();
    expect(state().messages).toEqual([]);
    expect(state().pendingUiAction).toBeNull();
    expect(state().streaming).toBe(false);
  });
});

describe("loadChat", () => {
  // The fixture is the shape the SERVER writes, and app/server/agent.py's
  // self-check fails if its real output drifts from it. This test used to mock
  // the frontend's own idea of the shape (`text`, SSE frames), so it passed
  // while every reopened chat rendered blank.
  it("renders a saved chat's text and tool rows, from the server's own shape", async () => {
    getChatMock.mockResolvedValue(golden as ChatHistory);

    await state().loadChat(golden.id);

    expect(state().chatId).toBe(golden.id);
    expect(state().messages).toEqual([
      { role: "user", text: "how did the striker do?", tools: [] },
      {
        role: "agent",
        text: golden.messages[1].content,
        tools: [
          {
            name: "player_stats",
            running: false,
            input: { match_id: "786827bf0cb540079e7ae350a99124c2" },
            summary: "22 players",
            durationMs: 41,
          },
          // Saved before summaries were: the row still draws, just without one.
          { name: "get_report", running: false, input: { match_id: "" }, summary: undefined, durationMs: undefined },
        ],
      },
    ]);
    expect(state().messages.every((m) => m.text.length > 0)).toBe(true);
  });

  it("keeps a stopped answer marked as stopped", async () => {
    getChatMock.mockResolvedValue({
      id: "c1",
      messages: [
        { role: "user", content: "long question" },
        { role: "assistant", content: "Half an ans", stopped: true },
      ],
    });
    await state().loadChat("c1");
    expect(state().messages[1]).toEqual({ role: "agent", text: "Half an ans", tools: [], stopped: true });
  });
});
