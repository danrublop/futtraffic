import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";

import { getChat, postRun, stopRun, type ChatHistory, type SavedToolEvent, type UiAction } from "./agent";

/** The one conversation.
 *
 *  Every rendering of the conversation reads THIS store — there is no second
 *  agent and no per-page message list. A run
 *  streams tokens into the last (agent) message, tool calls collect on it as
 *  collapsible rows, and ui_action frames park in `pendingUiAction` until the
 *  workspace consumes them (it is the only surface with artifact tabs).
 *
 *  Sending while a run is live stops that run first: the backend answers one
 *  run per chat, and a stale stream appending tokens after the new question
 *  would interleave two answers into one bubble. Stopping is a request to the
 *  server as well as an aborted fetch — the desktop proxy may not carry the
 *  abort through, and the server waits a moment for a stopped run to let go.
 */

export interface ToolCall {
  name: string;
  running: boolean;
  /** The arguments the tool was called with, from its tool_start frame — the
   *  provenance the expanded row shows as input JSON. */
  input?: Record<string, unknown>;
  summary?: string;
  durationMs?: number;
}

export interface ChatMessage {
  role: "user" | "agent";
  text: string;
  tools: ToolCall[];
  /** The coach stopped this answer; `text` is what had arrived by then. */
  stopped?: boolean;
}

interface ChatState {
  chatId: string | null;
  /** The open session's title; null before the first message. Mirrors what
   *  the server stores (the first message's first 60 characters, or a rename)
   *  so the chat's top row can name it without re-listing every chat. */
  title: string | null;
  /** The squad this conversation is about ("" = none): what a new chat is
   *  created under, and what the composer's chip names. */
  teamId: string;
  /** Bumped whenever the chat LIST changed without a run finishing (rename,
   *  move, delete), so the sidebar knows to re-list. */
  chatsVersion: number;
  messages: ChatMessage[];
  streaming: boolean;
  /** The last {"type":"error"} frame or transport failure, shown as a banner. */
  error: string | null;
  /** Set by ui_action frames; consumed (read-and-cleared) by the workspace. */
  pendingUiAction: UiAction | null;

  /** Drops the conversation and any live run — the sidebar's "New Task". A
   *  squad files the chat-to-be under that squad. */
  newChat: (teamId?: string) => void;
  /** Change the squad: before the first message it only picks the folder the
   *  chat will be created in; after, the caller has already PATCHed it. */
  setTeamId: (teamId: string) => void;
  setTitle: (title: string) => void;
  touchChats: () => void;
  /** Replaces the conversation with a saved one (GET /api/agent/chats/{id}). */
  loadChat: (id: string) => Promise<void>;
  send: (text: string, matchId?: string | null, model?: string | null) => Promise<void>;
  abort: () => void;
  clearError: () => void;
  consumeUiAction: () => UiAction | null;
}

// Module-level, not in state: an AbortController is a handle, not data, and
// putting it in the store would re-render on every replacement.
let active: AbortController | null = null;

/** Ends the live run on both ends: the server is told (fire and forget — a
 *  stop that fails leaves a run that finishes on its own), then the stream is
 *  dropped. A run with no chat id yet cannot be named, and the abort alone has
 *  to do. */
function stopActive(chatId: string | null) {
  if (!active) return;
  if (chatId) void stopRun(chatId).catch(() => {});
  active.abort();
  active = null;
}

/** A saved chat as the thread renders it. The server saves each tool call as
 *  ONE finished record, so there is nothing to pair — this used to fold SSE
 *  start/done frames that a saved chat never contains, and returned no rows. */
export function fromHistory(history: ChatHistory): ChatMessage[] {
  return history.messages.map((m) => ({
    role: m.role === "user" ? "user" : "agent",
    text: m.content ?? "",
    ...(m.stopped ? { stopped: true } : {}),
    tools: (m.events ?? []).map(
      (e: SavedToolEvent): ToolCall => ({
        name: e.name,
        running: false,
        input: e.input,
        summary: e.summary,
        durationMs: e.duration_ms,
      }),
    ),
  }));
}

// subscribeWithSelector so the workspace can subscribe to pendingUiAction
// changes directly — an external-system callback, not a render-derived effect.
export const useChatStore = create<ChatState>()(subscribeWithSelector((set, get) => ({
  chatId: null,
  title: null,
  teamId: "",
  chatsVersion: 0,
  messages: [],
  streaming: false,
  error: null,
  pendingUiAction: null,

  clearError: () => set({ error: null }),

  newChat: (teamId = "") => {
    stopActive(get().chatId);
    set({ chatId: null, title: null, teamId, messages: [], streaming: false, error: null, pendingUiAction: null });
  },

  setTeamId: (teamId) => set({ teamId }),

  setTitle: (title) => set({ title }),

  touchChats: () => set((s) => ({ chatsVersion: s.chatsVersion + 1 })),

  loadChat: async (id) => {
    stopActive(get().chatId);
    const history = await getChat(id);
    set({
      chatId: history.id,
      title: history.title || "New chat",
      teamId: history.team_id ?? "",
      streaming: false,
      error: null,
      pendingUiAction: null,
      messages: fromHistory(history),
    });
  },

  send: async (text, matchId, model) => {
    const message = text.trim();
    if (!message) return;

    stopActive(get().chatId);
    const ctrl = new AbortController();
    active = ctrl;

    set((s) => ({
      // The server titles a new chat with message[:60]; say the same here.
      title: s.title ?? message.slice(0, 60),
      streaming: true,
      error: null,
      messages: [
        ...s.messages,
        { role: "user", text: message, tools: [] },
        { role: "agent", text: "", tools: [] },
      ],
    }));

    /** Rewrites the trailing agent message in place — every token, tool start
     *  and tool done lands on the same bubble the run opened with. */
    const patchAgent = (fn: (m: ChatMessage) => ChatMessage) =>
      set((s) => {
        const i = s.messages.length - 1;
        if (i < 0 || s.messages[i].role !== "agent") return s;
        const messages = s.messages.slice();
        messages[i] = fn(messages[i]);
        return { messages };
      });

    try {
      await postRun(
        {
          chat_id: get().chatId,
          match_id: matchId ?? null,
          model: model ?? null,
          team_id: get().chatId ? null : get().teamId || null,
          message,
        },
        {
          onStart: (chatId) => set({ chatId }),
          onToken: (t) => patchAgent((m) => ({ ...m, text: m.text + t })),
          onToolStart: (name, input) =>
            patchAgent((m) => ({ ...m, tools: [...m.tools, { name, running: true, input }] })),
          onToolDone: (name, summary, durationMs) =>
            patchAgent((m) => {
              const tools = m.tools.slice();
              // Pair with the latest still-running call of the same name — the
              // agent may run the same tool twice in one answer.
              for (let i = tools.length - 1; i >= 0; i--) {
                if (tools[i].running && tools[i].name === name) {
                  tools[i] = { ...tools[i], running: false, summary, durationMs };
                  break;
                }
              }
              return { ...m, tools };
            }),
          onUiAction: (action) => set({ pendingUiAction: action }),
          onDone: (chatId) => set({ chatId }),
          onError: (message) => set({ error: message }),
        },
        ctrl.signal,
      );
    } catch (e) {
      // An abort is the next message taking over, not a failure to show.
      if (!ctrl.signal.aborted) {
        set({ error: e instanceof Error ? e.message : String(e) });
      }
    } finally {
      // Only the run that is still current may turn the spinner off — an
      // aborted run's finally must not clear the new run's streaming flag.
      if (active === ctrl) {
        active = null;
        set({ streaming: false });
      }
    }
  },

  abort: () => {
    if (!active) return;
    stopActive(get().chatId);
    // Marked here rather than waiting for a reload: the server saves the same
    // thing, and the thread should say so the moment Stop is pressed.
    set((s) => {
      const i = s.messages.length - 1;
      const messages = s.messages.slice();
      if (i >= 0 && messages[i].role === "agent") messages[i] = { ...messages[i], stopped: true };
      return { messages, streaming: false };
    });
  },

  consumeUiAction: () => {
    const a = get().pendingUiAction;
    if (a) set({ pendingUiAction: null });
    return a;
  },
})));
