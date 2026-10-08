import { ApiError, detail } from "./api";

/** The coach-agent contract.
 *
 *  Same-origin `/api/agent`, rewritten to the FastAPI service like everything
 *  else in lib/api.ts. The run endpoint is SSE over POST — EventSource cannot
 *  POST, so this reads the stream off fetch() and cuts it into `data:` lines
 *  itself. Each line is one JSON object, the shapes below; anything that does
 *  not parse is skipped, never thrown: a half-written line at a chunk boundary
 *  must not kill the run.
 */

export const AGENT_BASE = "/api/agent";

/** The artifact tabs the agent may ask the workspace to open. The backend's
 *  open_tab enum is match|stats|lineup|squad|film|clips (app/server/agent.py);
 *  "docs" is the workspace's own notepad, wired so the backend can learn it
 *  without a frontend change. The workspace maps these onto its editor tabs
 *  in lib/tabs.ts — the two must not drift. */
export type AgentTab = "match" | "stats" | "lineup" | "squad" | "film" | "clips" | "data" | "docs";

/** `match_id` and `note` are checked by the server before the frame is sent -
 *  a match that is a job on the box, a note that exists in the library. */
export type UiAction =
  | { action: "open_tab"; tab: AgentTab; match_id?: string; note?: string }
  | { action: "seek_film"; t: number; match_id?: string };

/** One parsed `data:` line, exactly as the backend writes it. */
export type AgentEvent =
  | { type: "start"; chat_id: string }
  | { type: "token"; text: string }
  | { type: "tool_start"; name: string; input?: Record<string, unknown> }
  | { type: "tool_done"; name: string; summary?: string; duration_ms?: number }
  | { type: "ui_action"; action: "open_tab"; tab: AgentTab; match_id?: string; note?: string }
  | { type: "ui_action"; action: "seek_film"; t: number; match_id?: string }
  | { type: "done"; chat_id: string }
  | { type: "error"; message: string };

export interface RunHandlers {
  /** Always the first frame: names the chat, so a run started without a
   *  chat_id can still be stopped before it finishes. */
  onStart?: (chatId: string) => void;
  onToken?: (text: string) => void;
  onToolStart?: (name: string, input: Record<string, unknown> | undefined) => void;
  onToolDone?: (name: string, summary: string | undefined, durationMs: number | undefined) => void;
  onUiAction?: (action: UiAction) => void;
  onDone?: (chatId: string) => void;
  /** An {"type":"error"} frame — the run failed server-side. */
  onError?: (message: string) => void;
}

/** Parses one raw SSE line into an event, or returns null for anything that
 *  is not a well-formed `data:` frame (blank keep-alives, comments, split JSON
 *  that survived reassembly, a shape we do not speak). */
export function parseEventLine(line: string): AgentEvent | null {
  const trimmed = line.trim();
  if (!trimmed.startsWith("data:")) return null;
  let ev: unknown;
  try {
    ev = JSON.parse(trimmed.slice(5).trim());
  } catch {
    return null;
  }
  if (typeof ev !== "object" || ev === null || typeof (ev as { type?: unknown }).type !== "string") {
    return null;
  }
  return ev as AgentEvent;
}

/** Routes one parsed event to its handler. Unknown types fall through — the
 *  backend may add frames before the frontend learns them, and a run must not
 *  die of one. */
export function dispatchEvent(ev: AgentEvent, h: RunHandlers): void {
  switch (ev.type) {
    case "start":
      h.onStart?.(ev.chat_id);
      break;
    case "token":
      h.onToken?.(ev.text);
      break;
    case "tool_start":
      h.onToolStart?.(ev.name, ev.input);
      break;
    case "tool_done":
      h.onToolDone?.(ev.name, ev.summary, ev.duration_ms);
      break;
    case "ui_action":
      h.onUiAction?.(
        ev.action === "seek_film"
          ? { action: "seek_film", t: ev.t, ...(ev.match_id ? { match_id: ev.match_id } : {}) }
          : {
              action: "open_tab",
              tab: ev.tab,
              ...(ev.match_id ? { match_id: ev.match_id } : {}),
              ...(ev.note ? { note: ev.note } : {}),
            },
      );
      break;
    case "done":
      h.onDone?.(ev.chat_id);
      break;
    case "error":
      h.onError?.(ev.message);
      break;
  }
}

export interface RunRequest {
  chat_id?: string | null;
  /** The match THIS message is about. A chat can move between matches, so it
   *  travels with every message, not once with the chat. */
  match_id?: string | null;
  /** One of GET /api/agent/models; omitted means the server's default. */
  model?: string | null;
  /** The squad a NEW chat files under. Ignored once the chat exists — moving
   *  it is PATCH, so a stale value here cannot re-file a conversation. */
  team_id?: string | null;
  message: string;
}

/** POST /api/agent/run and stream the reply.
 *
 *  Reassembles `data:` lines across read boundaries — a chunk may end in the
 *  middle of a JSON object, so only newline-terminated text is dispatched and
 *  the tail is carried into the next chunk. Resolves when the stream closes;
 *  rejects on a non-2xx response, a network failure, or an abort (an
 *  AbortError the caller is expected to recognise, not report).
 */
export async function postRun(
  req: RunRequest,
  handlers: RunHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${AGENT_BASE}/run`, {
    method: "POST",
    cache: "no-store",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify({
      chat_id: req.chat_id ?? null,
      match_id: req.match_id ?? null,
      model: req.model ?? null,
      team_id: req.team_id ?? null,
      message: req.message,
    }),
    signal,
  });
  // detail(), not res.text(): the run endpoint refuses with FastAPI's
  // {"detail": "..."} and a raw dump put the whole JSON envelope, escapes and
  // all, into the coach's error banner. The missing-API-key refusal is the one
  // that actually happens, and it is a sentence worth reading.
  if (!res.ok) throw new ApiError(res.status, await detail(res));
  if (!res.body) throw new ApiError(res.status, "the run returned no stream");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const ev = parseEventLine(buf.slice(0, nl));
      buf = buf.slice(nl + 1);
      if (ev) dispatchEvent(ev, handlers);
    }
  }
  // A final frame with no trailing newline still counts.
  buf += decoder.decode();
  const tail = parseEventLine(buf);
  if (tail) dispatchEvent(tail, handlers);
}

/* ------------------------------------------------------- history (REST) --- */

export interface ChatSummary {
  id: string;
  title: string;
  updated: string;
  /** The squad folder it files under; "" (or absent on an old server) is none. */
  team_id?: string;
  match_id?: string;
  n_messages?: number;
}

/** One tool call as a SAVED chat records it (app/server/agent.py `tool_events`).
 *  Not the SSE frames: a saved call is one record, not a start/done pair.
 *  `summary` and `duration_ms` are missing on chats saved before 2026-09-13. */
export interface SavedToolEvent {
  id?: string;
  name: string;
  input?: Record<string, unknown>;
  result?: unknown;
  ok?: boolean;
  summary?: string;
  duration_ms?: number;
}

/** One message exactly as GET /api/agent/chats/{id} returns it (app/server/chats.py
 *  append_message). This said `text` and `AgentEvent[]` while the server wrote
 *  `content` and tool records, so every reopened chat rendered blank — the
 *  shape is pinned by test/fixtures/chat.golden.json, which the backend's
 *  self-check compares its own output against. */
export interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
  ts?: number;
  events?: SavedToolEvent[];
  /** On a question: the match it was asked about. */
  match_id?: string;
  /** On an answer: which model wrote it. */
  model?: string;
  /** On an answer: the coach stopped it, so the text is what came before. */
  stopped?: boolean;
}

export interface ChatHistory {
  id: string;
  title?: string;
  team_id?: string;
  match_id?: string;
  messages: HistoryMessage[];
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${AGENT_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new ApiError(res.status, await detail(res));
  return (await res.json()) as T;
}

export interface ModelOption {
  id: string;
  label: string;
}

export interface ModelList {
  default: string;
  models: ModelOption[];
}

export const listChats = () => get<ChatSummary[]>("/chats");
export const listModels = () => get<ModelList>("/models");

/** Rename a chat, or move it to another squad ("" = no squad). */
export async function patchChat(id: string, patch: { title?: string; team_id?: string }): Promise<ChatSummary> {
  const res = await fetch(`${AGENT_BASE}/chats/${encodeURIComponent(id)}`, {
    method: "PATCH",
    cache: "no-store",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) throw new ApiError(res.status, await detail(res));
  return (await res.json()) as ChatSummary;
}

/** Delete a chat. The server refuses (409) while it is still answering. */
export async function deleteChat(id: string): Promise<void> {
  const res = await fetch(`${AGENT_BASE}/chats/${encodeURIComponent(id)}`, {
    method: "DELETE",
    cache: "no-store",
  });
  if (!res.ok) throw new ApiError(res.status, await detail(res));
}

/** Stop the chat's live run. Its own route, not just an aborted fetch: the
 *  desktop app's proxy may not carry an abort through to the server, and the
 *  run would keep spending. Resolves false when nothing was running. */
export async function stopRun(chatId: string): Promise<boolean> {
  const res = await fetch(`${AGENT_BASE}/chats/${encodeURIComponent(chatId)}/stop`, {
    method: "POST",
    cache: "no-store",
  });
  if (!res.ok) throw new ApiError(res.status, await detail(res));
  return ((await res.json()) as { stopped: boolean }).stopped;
}
export const getChat = (id: string) => get<ChatHistory>(`/chats/${encodeURIComponent(id)}`);
