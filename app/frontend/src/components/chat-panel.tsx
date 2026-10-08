"use client";

import { useEffect, useRef, useState, type ComponentProps } from "react";
import Link from "next/link";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { toast } from "sonner";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CornerDownLeft,
  X,
} from "lucide-react";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useMatch, useMatches, useSeason } from "@/hooks/use-matches";
import { listModels, patchChat, type ModelList } from "@/lib/agent";
import { useDock } from "@/lib/dock";
import { useChatStore, type ChatMessage, type ToolCall } from "@/lib/store";
import { teamLabel, useOrg } from "@/lib/club";
import { withParam } from "@/lib/data-nav";
import { SPRINTS_WITHHELD_WHY } from "@/lib/report";
import { matchDate, matchTitle, progressOf, statusOf, TONE_CLASS } from "@/lib/status";
import { cn } from "@/lib/utils";

/** Which model answers. The list is the server's (GET /api/agent/models,
 *  agent.MODELS) — this used to be five labels written here, two of them for a
 *  provider the app does not have, and picking one changed nothing. The last
 *  pick is remembered per browser; one the server no longer offers falls back
 *  to its default rather than sending a model it would refuse. */
const MODEL_KEY = "futtraffic.model";

function useModelChoice(): [ModelList | null, string | null, (id: string) => void] {
  const [list, setList] = useState<ModelList | null>(null);
  const [model, setModel] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    listModels()
      .then((m) => {
        if (!live) return;
        let saved: string | null = null;
        try {
          saved = window.localStorage.getItem(MODEL_KEY);
        } catch {
          /* private mode: nothing remembered */
        }
        setList(m);
        setModel(m.models.some((x) => x.id === saved) ? saved : m.default);
      })
      // No backend: no picker, and the run falls back to the server default.
      .catch(() => {});
    return () => {
      live = false;
    };
  }, []);

  const pick = (id: string) => {
    setModel(id);
    try {
      window.localStorage.setItem(MODEL_KEY, id);
    } catch {
      /* still applied for this session */
    }
  };
  return [list, model, pick];
}

/** Markdown answers. Claude writes bold, lists and — for any comparison —
 *  tables, and plain text showed the coach the asterisks and pipes.
 *
 *  No rehype-raw, deliberately: the reply is model output, so raw HTML in it is
 *  shown as text rather than rendered. Tailwind's preflight strips list and
 *  table styling, which is why every element is named here. */
/** One styled element. react-markdown hands every component its hast `node`,
 *  and spreading that onto a DOM element wrote node="[object Object]" on all
 *  of them — so it is taken out here, once. */
function withoutNode<P extends object>(props: P): Omit<P, "node"> {
  const rest = { ...props } as P & { node?: unknown };
  delete rest.node;
  return rest;
}

function tag<T extends keyof React.JSX.IntrinsicElements>(Tag: T, className: string) {
  return function Styled(props: ComponentProps<T> & { node?: unknown }) {
    const El = Tag as React.ElementType;
    return <El className={className} {...withoutNode(props)} />;
  };
}

const MD: Components = {
  p: tag("p", "mb-3 last:mb-0"),
  ul: tag("ul", "mb-3 list-disc pl-5 last:mb-0"),
  ol: tag("ol", "mb-3 list-decimal pl-5 last:mb-0"),
  li: tag("li", "mb-1"),
  strong: tag("strong", "font-semibold"),
  h1: tag("h3", "mt-4 mb-2 text-[16px] font-semibold"),
  h2: tag("h3", "mt-4 mb-2 text-[15px] font-semibold"),
  h3: tag("h4", "mt-3 mb-1.5 text-[15px] font-semibold"),
  blockquote: tag("blockquote", "mb-3 border-l-2 border-[var(--hairline-strong)] pl-3 text-[var(--ink-soft)]"),
  code: tag("code", "rounded-[var(--radius-xs)] bg-[var(--card)] px-1 font-mono text-[13px]"),
  pre: tag("pre", "mb-3 overflow-x-auto rounded-[var(--radius-sm)] bg-[var(--card)] p-2.5 text-[13px]"),
  th: tag("th", "border-b border-[var(--hairline-strong)] px-2 py-1 text-left font-medium"),
  td: tag("td", "readout border-b border-[var(--hairline)] px-2 py-1"),
  a: (props) => (
    <a className="underline underline-offset-2" target="_blank" rel="noreferrer" {...withoutNode(props)} />
  ),
  // A wide table scrolls inside itself, never the thread.
  table: (props) => (
    <div className="mb-3 overflow-x-auto">
      <table className="w-full border-collapse text-[13px]" {...withoutNode(props)} />
    </div>
  ),
};

export function AnswerText({ text }: { text: string }) {
  return (
    <div className="text-[15px] leading-[1.7] text-[var(--ink)]">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD}>
        {text}
      </ReactMarkdown>
    </div>
  );
}

/** The conversation — thread, tool rows and composer.
 *
 *  There used to be a second export here, `ChatPanel`: a docked 360px column
 *  the shell mounted on every page. It is gone. The conversation is a TAB now,
 *  opened by the header's Chat tool like every other surface, so a second
 *  rendering of the same thread — with its own chrome, its own composer and its
 *  own empty state — had nothing left to do. The store's `chatOpen` slice went
 *  with it. This file is presentation only; the SSE transport and the store
 *  contract are untouched.
 *
 *  The idiom:
 *   - composer: a --card rounded field with a focus-accent border, 15px
 *     textarea, and one 26px circular send button in muted monochrome
 *     (ChatComposer / composer-shared);
 *   - user messages are right-aligned pills; AGENT messages are plain text,
 *     no bubble (ChatMessageList);
 *   - tool calls are collapsible 12.5px rows — status icon, mono duration,
 *     one-line summary, input/output behind the click (ToolCallItem);
 *   - a run of consecutive calls collapses into one "N tool calls" row
 *     (ToolCluster);
 *   - while streaming: the committed text, then ONE status line
 *     ("Running <tool>…" / "Thinking…") — never a bubble per token
 *     (StreamingMessage).
 */

/** Which squad the conversation is about, under the field.
 *
 *  Before the first message it only decides the folder the chat will be
 *  created in. After, picking another squad MOVES the chat — the same PATCH as
 *  the sidebar's "Move to squad" — so the chip and the folder cannot disagree.
 *  A club with no squads has nothing to pick, and the chip is not drawn. */
function SquadChip() {
  const { org } = useOrg();
  const teamId = useChatStore((s) => s.teamId);
  const chatId = useChatStore((s) => s.chatId);
  const setTeamId = useChatStore((s) => s.setTeamId);
  const touchChats = useChatStore((s) => s.touchChats);
  const teams = org?.teams ?? [];
  if (!teams.length) return null;
  const current = teams.find((t) => t.team_id === teamId);

  const pick = (id: string) => {
    if (id === teamId) return;
    if (!chatId) return setTeamId(id);
    patchChat(chatId, { team_id: id }).then(
      () => {
        setTeamId(id);
        touchChats();
      },
      // The chip keeps its old squad, because nothing moved - and says so.
      (e: unknown) =>
        toast.error("Could not move the chat", { description: e instanceof Error ? e.message : String(e) }),
    );
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className="flex h-6 items-center gap-1.5 rounded-[var(--radius-sm)] px-1.5 transition-colors hover:bg-[var(--card)] hover:text-[var(--ink)]"
          title={chatId ? "The squad this session is about — pick another to move it" : "The squad this session is about"}
        >
          {current ? teamLabel(current) : "Add to a squad"}
          <ChevronDown className="size-3" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-48">
        {teams.map((t) => (
          <DropdownMenuItem key={t.team_id} onSelect={() => pick(t.team_id)} className="gap-2">
            <Check className={cn("size-4", t.team_id === teamId ? "opacity-100" : "opacity-0")} />
            {teamLabel(t)}
          </DropdownMenuItem>
        ))}
        {current && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={() => pick("")}>Remove from squad</DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

/** The mark that closes a finished answer — the one place the accent colour
 *  appears in the thread, and the thing that stops a long conversation reading
 *  as one undifferentiated wall of prose. */
function Spark() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" aria-hidden className="text-[var(--accent)]">
      <g stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
        <path d="M12 3v18M3 12h18M5.6 5.6l12.8 12.8M18.4 5.6L5.6 18.4" />
      </g>
    </svg>
  );
}

/** The chat's top row, left of the dock buttons: the session's title, once
 *  the first message has made one. Nothing before that. */
export function ChatTitle() {
  const title = useChatStore((s) => s.title);
  return (
    <h1 className="min-w-0 truncate text-[15px] font-medium tracking-[-0.2px] text-[var(--ink)]">
      {title}
    </h1>
  );
}

/** THE START OF A TASK: a dashboard of the library, not a menu of tabs.
 *
 *  The empty state was a headline ("What are we breaking down?") over chips
 *  named after tabs — Teams, All players, Opponents, Database, This match,
 *  Film, Notes. Every one of those is already a click away in the sidebar and
 *  the header, so the conversation's own first screen spent itself on
 *  navigation that exists twice over, while the two things you actually open
 *  this app to know — what is in the library, and what is processing right
 *  now — were on neither.
 *
 *  So it is a panel: the club's totals as tiles, then the matches, with a
 *  second tab for whatever the pipeline is chewing on. Clicking a match grounds
 *  the conversation in it and stays here; the line under the composer names
 *  what is grounded, so the click has a visible effect without moving anyone
 *  off the screen they just chose something on.
 *
 *  NOTHING HERE FETCHES ANYTHING NEW. Matches, the club and the measured season
 *  are all already loaded and cached for the sidebar, the header and the data
 *  surface — a dashboard that opened three requests to say what the app already
 *  knew would make the chat the slowest screen in the product.
 */
function RecentMatches({ current }: { current: string | null }) {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const { matches } = useMatches();
  const { org } = useOrg();
  // The whole club, not the selected team: this is the library, and the header
  // scope is about what you are reading, not about what exists.
  const { season } = useSeason(undefined, true);
  const [tab, setTab] = useState<"recent" | "running">("recent");

  /** Set the match and keep everything else in the query. The header's switcher
   *  replaces the whole query string, which drops `?v=` and quietly bounces the
   *  workspace back to its default destination; picking a match to talk about
   *  must not move you off the chat you picked it on. */
  const ground = (id: string) =>
    router.push(`${pathname}?${withParam(params.toString(), "m", id)}`);

  const all = matches ?? [];
  // `live` is statusOf's own answer to "is the pipeline working on this", so
  // the Running tab cannot drift from what every other screen calls running.
  const running = all.filter((m) => statusOf(m).live);
  // Five, not more: this is a glance before you type, not the Matches screen.
  // At eight the panel was 547px tall and pushed the composer down a laptop.
  const recent = [...all].sort((a, b) => (b.created ?? 0) - (a.created ?? 0)).slice(0, 5);

  const players = season?.players ?? [];
  const km = players.reduce((n, p) => n + p.distance_km, 0);
  const top = players.reduce((n, p) => Math.max(n, p.top_speed_kmh ?? 0), 0);
  const cover = season?.coverage;

  const shown = tab === "recent" ? recent : running;

  return (
    <div className="grid gap-4 py-4">
      <h1 className="text-[19px] tracking-[-0.3px] text-[var(--ink)]">
        What are we breaking down{org?.club.name ? `, ${org.club.name}` : ""}?
      </h1>

      <div className="rounded-[var(--radius-lg)] border border-[var(--hairline)] p-2.5">
        <div className="flex items-center gap-1">
          <Pill on={tab === "recent"} onClick={() => setTab("recent")}>
            Recent
          </Pill>
          <Pill on={tab === "running"} onClick={() => setTab("running")}>
            Running
            {running.length > 0 && (
              <span className="ml-1.5 text-[var(--accent)]">{running.length}</span>
            )}
          </Pill>
          <span className="ml-auto pr-1 text-[11px] text-[var(--mute)]">
            {matches === null ? "" : `${all.length} in the library`}
          </span>
        </div>

        <div className="mt-2 grid grid-cols-2 gap-1.5 sm:grid-cols-4">
          <Tile label="Matches" v={String(all.length)} />
          <Tile label="Teams" v={String(org?.teams.length ?? 0)} />
          <Tile label="Players" v={String(players.length)} />
          {/* Bound of total tracks, not a count of people: it is the share of
              the tracking any name can honestly be put on, and it is the number
              every measured tile below is standing on. */}
          <Tile
            label="Confirmed"
            v={cover ? `${cover.bound}/${cover.total}` : "—"}
            title="Tracks a human confirmed as a named player, of all tracks"
          />
          <Tile label="Distance" v={km ? `${km.toFixed(0)} km` : "—"} />
          <Tile label="Sprints" v="withheld" title={SPRINTS_WITHHELD_WHY} />
          <Tile label="Top km/h" v={top ? top.toFixed(1) : "—"} />
          <Tile label="Processing" v={String(running.length)} />
        </div>

        <div className="mt-2 flex flex-col gap-px">
          {matches === null ? (
            <Line>Loading…</Line>
          ) : shown.length === 0 ? (
            <Line>
              {tab === "running"
                ? "Nothing is processing. Uploads and tracking runs show up here while they work."
                : "No matches yet. Upload one from the Matches screen — the chat can talk about the whole library, but it is far more use pointed at a game."}
            </Line>
          ) : (
            shown.map((m) => {
              const view = statusOf(m);
              const bar = progressOf(m);
              const on = m.id === current;
              return (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => ground(m.id)}
                  className={cn(
                    "flex items-center gap-3 rounded-[var(--radius-sm)] px-2.5 py-1.5 text-left transition-colors",
                    on ? "bg-[var(--card)]" : "hover:bg-[var(--card)]",
                  )}
                >
                  <span className="min-w-0 flex-1 truncate text-[13px] text-[var(--ink)]">
                    {matchTitle(m)}
                  </span>
                  {/* A running job says how far along it is; a finished one says
                      what it is. Never both — a progress bar at 100% beside the
                      word "Ready" is the same fact twice. */}
                  {bar ? (
                    <span className="flex shrink-0 items-center gap-2">
                      <span className="h-1 w-24 overflow-hidden rounded-full bg-[var(--card)]">
                        <span
                          className="block h-full rounded-full bg-[var(--accent)] transition-[width] duration-500"
                          style={{ width: `${bar.value}%` }}
                        />
                      </span>
                      <span className="readout w-10 shrink-0 text-right text-[11px] text-[var(--mute)]">
                        {bar.text}
                      </span>
                    </span>
                  ) : (
                    <span className="readout shrink-0 text-[11px] text-[var(--mute)]">
                      {matchDate(m) ?? ""}
                    </span>
                  )}
                  <span
                    className={cn("w-24 shrink-0 text-right text-[11px]", TONE_CLASS[view.tone])}
                  >
                    {view.label}
                  </span>
                </button>
              );
            })
          )}
        </div>

      </div>
    </div>
  );
}

function Pill({
  on,
  onClick,
  children,
}: {
  on: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "h-7 rounded-[var(--radius-md)] px-3 text-[13px] transition-colors",
        on
          ? "bg-[var(--card)] text-[var(--ink)]"
          : "text-[var(--mute)] hover:text-[var(--ink-soft)]",
      )}
    >
      {children}
    </button>
  );
}

function Tile({ label, v, title }: { label: string; v: string; title?: string }) {
  return (
    <div
      title={title}
      className="rounded-[var(--radius-md)] bg-[var(--card)] px-2.5 py-1.5"
    >
      <p className="truncate text-[11px] text-[var(--mute)]">{label}</p>
      <p className="readout truncate text-[15px] text-[var(--ink)]">{v}</p>
    </div>
  );
}

function Line({ children }: { children: React.ReactNode }) {
  return (
    <p className="max-w-[52ch] px-2.5 py-1.5 text-[12px] leading-relaxed text-[var(--mute)]">
      {children}
    </p>
  );
}

const clockMs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`);

/** The composer control language (composer-shared.tsx): the circular buttons
 *  take their fill from a light mix of the muted text colour, their glyph is
 *  the muted colour itself — monochrome; Signal Blue never appears here. */
const CTRL_ICON = "var(--mute)";
const CTRL_BG = "color-mix(in srgb, var(--mute) 28%, var(--bg))";

/** The thread, tool rows, error banner and composer — everything between the
 *  panel header and the page. Owns its scroll and its auto-scroll. */
export function ChatConversation() {
  const messages = useChatStore((s) => s.messages);
  const streaming = useChatStore((s) => s.streaming);
  const error = useChatStore((s) => s.error);
  const clearError = useChatStore((s) => s.clearError);
  const send = useChatStore((s) => s.send);
  const abort = useChatStore((s) => s.abort);
  const { id: pageMatch } = useMatch();
  const { matches } = useMatches();
  /** A film showing in the dock is what "this" means while you watch it: the
   *  question goes with THAT match, and the line under the field says so. */
  const dockFilm = useDock((s) => {
    const t = s.open ? s.tabs.find((x) => x.id === s.active) : undefined;
    return t?.kind === "film" ? (t.params.m ?? null) : null;
  });
  const matchId = dockFilm ?? pageMatch;
  const dockMatch = dockFilm ? matches?.find((m) => m.id === dockFilm) : undefined;

  const [draft, setDraft] = useState("");
  const [models, model, pickModel] = useModelChoice();
  const teamId = useChatStore((s) => s.teamId);
  const { org } = useOrg();
  const squad = org?.teams.find((t) => t.team_id === teamId);
  const placeholder = squad ? `Ask about ${squad.name}…` : "Ask about the match…";
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Follow the stream down, but only the stream — re-renders from elsewhere
  // (tab switches) must not yank a reader back to the bottom.
  useEffect(() => {
    scrollRef.current?.scrollTo(0, scrollRef.current.scrollHeight);
  }, [messages]);

  // The textarea auto-grows on input; a programmatic clear (send) does not
  // fire onChange, so the box would keep its tall height — reset it here.
  useEffect(() => {
    if (!draft && textareaRef.current) textareaRef.current.style.height = "auto";
  }, [draft]);

  const submit = (raw: string) => {
    const text = raw.trim();
    if (!text || streaming) return;
    setDraft("");
    void send(text, matchId, model);
  };

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      {error && (
        <div
          role="alert"
          className="mx-3 mt-3 flex shrink-0 items-start gap-2 rounded-[var(--radius-md)] border border-[var(--danger)]/30 bg-[var(--danger)]/10 px-3 py-2 text-[13px] text-[var(--danger)]"
        >
          {/* The no-key case is the first thing anyone hits, and the backend
              answers it with a sentence about exporting a shell variable. Say
              it in the app's words and point at the field that fixes it. */}
          <span className="min-w-0 flex-1">
            {/api key/i.test(error) ? (
              <>
                The chat needs an Anthropic API key.{" "}
                <Link href="/settings" className="underline underline-offset-2">
                  Add one in Settings
                </Link>
                .
              </>
            ) : (
              error
            )}
          </span>
          <button
            onClick={clearError}
            aria-label="Dismiss error"
            className="grid size-5 shrink-0 place-items-center rounded text-[var(--danger)]/70 hover:text-[var(--danger)]"
          >
            <X className="size-3.5" />
          </button>
        </div>
      )}

      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
        {/* One axis: max-w-2xl here and on the composer, 16px between blocks —
            a centred Cursor-style column, not iMessage lanes. */}
        <div className="mx-auto flex w-full max-w-2xl flex-col gap-4 px-4 py-4">
          {messages.length === 0 && <RecentMatches current={pageMatch} />}
          {messages.map((m, i) => (
            <Message key={i} msg={m} streaming={streaming && i === messages.length - 1} />
          ))}
        </div>
      </div>

      {/* The composer. The field is the only thing with a border; the controls
       *  sit BELOW it on the page, which is what keeps a 15px input from
       *  turning into a toolbar with a text box in it.
       *
       *  The wrapper's `pt-8` IS the fade: 32px of gradient to transparent, so
       *  the thread slides under the field instead of stopping dead against it.
       */}
      <div className="shrink-0 bg-gradient-to-t from-[var(--panel)] via-[var(--panel)] to-transparent px-4 pt-8 pb-3">
        <div className="mx-auto w-full max-w-2xl">
          <div className="flex items-end gap-2 rounded-2xl border border-[var(--hairline)] bg-[var(--card)]/40 px-4 py-3 transition-colors focus-within:border-[var(--hairline-strong)]">
            <textarea
              ref={textareaRef}
              value={draft}
              rows={1}
              onChange={(e) => {
                setDraft(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`;
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit(draft);
                }
              }}
              placeholder={placeholder}
              aria-label="Message"
              className="max-h-[200px] min-h-0 w-full flex-1 resize-none bg-transparent text-[15px] leading-relaxed text-[var(--ink)] outline-none placeholder:text-[var(--mute)]"
            />
            {streaming ? (
              <button
                onClick={abort}
                aria-label="Stop"
                title="Stop"
                className="grid size-[26px] shrink-0 place-items-center rounded-full transition-all"
                style={{ background: CTRL_BG }}
              >
                <span
                  aria-hidden
                  className="block size-2.5 animate-pulse rounded-[2px]"
                  style={{ background: CTRL_ICON, opacity: 0.85 }}
                />
              </button>
            ) : (
              <button
                onClick={() => submit(draft)}
                disabled={!draft.trim()}
                aria-label="Send"
                title="Send"
                className="grid size-[26px] shrink-0 place-items-center rounded-full transition-opacity duration-[var(--dur-fast)] disabled:opacity-30"
                style={{ background: CTRL_BG, color: CTRL_ICON }}
              >
                <CornerDownLeft className="size-3.5" />
              </button>
            )}
          </div>

          {/* Under the field: what this conversation is ABOUT on the left, what
              it is doing on the right. No attach or dictate buttons — nothing
              behind them yet, and a control that does nothing is worse than a
              gap. */}
          <div className="mt-2 flex items-center gap-2 px-1 text-[12px] text-[var(--mute)]">
            <SquadChip />
            {models && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button
                    type="button"
                    className="flex h-6 items-center gap-1.5 rounded-[var(--radius-sm)] px-1.5 transition-colors hover:bg-[var(--card)] hover:text-[var(--ink)]"
                    title="Which model answers"
                  >
                    {models.models.find((m) => m.id === model)?.label ?? model}
                    <ChevronDown className="size-3" />
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="w-48">
                  {models.models.map((m) => (
                    <DropdownMenuItem key={m.id} onSelect={() => pickModel(m.id)} className="gap-2">
                      <Check className={cn("size-4", m.id === model ? "opacity-100" : "opacity-0")} />
                      {m.label}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            )}
            <span className="flex-1" />
            {dockMatch && !streaming && (
              <span className="truncate" title="Questions go with the film open in the dock">
                About {matchTitle(dockMatch)}
              </span>
            )}
            {streaming && (
              <span aria-live="polite" className="flex items-center gap-1.5">
                <span className="size-1.5 animate-pulse rounded-full bg-[var(--accent)]" />
                Working
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Message({ msg, streaming }: { msg: ChatMessage; streaming: boolean }) {
  if (msg.role === "user") {
    // What you asked, set to the right on --card, capped well short of the
    // column so it reads as an aside against the answer below it. The agent's
    // reply is the document; this is the margin note that prompted it.
    return (
      <div className="flex w-full justify-end">
        <p className="max-w-[78%] rounded-2xl bg-[var(--card)] px-4 py-2.5 text-[15px] leading-relaxed whitespace-pre-wrap text-[var(--ink)]">
          {msg.text}
        </p>
      </div>
    );
  }

  // Agent: plain text, no bubble. Tools sit above the answer they produced;
  // the single status line says what is happening RIGHT NOW and nothing more.
  const running = streaming ? msg.tools.find((t) => t.running) : undefined;
  return (
    <div className="flex flex-col gap-2 py-2">
      {msg.tools.length > 0 && <ToolCluster tools={msg.tools} />}
      {msg.text && <AnswerText text={msg.text} />}
      {/* The mark closes a finished answer, so a long thread reads as a series
          of replies rather than one wall. Nothing while it is still writing —
          the status line below is doing that job. A stopped answer says so
          instead: what is above it is where it was cut, not a conclusion. */}
      {msg.stopped ? (
        <p className="text-[12px] text-[var(--mute)]">Stopped</p>
      ) : (
        msg.text && !streaming && <Spark />
      )}
      {streaming &&
        (running ? (
          <p className="text-[13px] text-[var(--mute)] italic" aria-live="polite">
            Running {running.name}…
          </p>
        ) : (
          !msg.text && (
            <p className="text-[13px] text-[var(--mute)] italic" aria-live="polite">
              Thinking…
            </p>
          )
        ))}
    </div>
  );
}

/** Consecutive tool calls collapse into one row: a
 *  single call renders bare, a run of them hides behind "N tool calls ·
 *  <last one>" until clicked. Our store collects a run's calls on the one
 *  agent message, so the whole list is consecutive by construction. */
function ToolCluster({ tools }: { tools: ToolCall[] }) {
  const [open, setOpen] = useState(false);
  if (tools.length === 1)
    return (
      <div className="max-w-[85%]">
        <ToolRow tool={tools[0]} />
      </div>
    );

  const running = tools.filter((t) => t.running).length;
  const last = tools[tools.length - 1].name;

  return (
    <div className="max-w-[85%] overflow-hidden rounded-[var(--radius-md)] border border-[var(--hairline)] text-[12.5px]">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left transition-colors select-none hover:bg-[var(--card)]/50"
      >
        <ChevronRight
          className={cn(
            "size-3 shrink-0 text-[var(--mute)] transition-transform duration-[var(--dur-fast)]",
            open && "rotate-90",
          )}
        />
        <span className="min-w-0 flex-1 truncate font-[450] text-[var(--ink-soft)]">
          {tools.length} tool calls
          {!open && <span className="text-[var(--mute)]"> · {last}</span>}
        </span>
        {running > 0 && <span className="shrink-0 text-[11px] text-[var(--accent)]">{running} running</span>}
      </button>
      {open && (
        <div className="space-y-1 px-2 pb-2">
          {tools.map((t, i) => (
            <ToolRow key={`${t.name}-${i}`} tool={t} />
          ))}
        </div>
      )}
    </div>
  );
}

/** One tool call: name, status icon, mono duration,
 *  a one-line summary while collapsed, and the input JSON + output behind
 *  the click — the call is provenance for the answer, not the headline. */
function ToolRow({ tool }: { tool: ToolCall }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="overflow-hidden rounded-[var(--radius-md)] border border-[var(--hairline)] bg-[var(--bg)] text-[12.5px] font-[450]">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-2.5 py-2 text-left transition-colors select-none hover:bg-[var(--card)]/50"
      >
        <span className="min-w-0 flex-1 truncate text-[var(--ink)]">{tool.name}</span>
        {tool.running ? (
          <span className="shrink-0 text-[11px] text-[var(--accent)]">running…</span>
        ) : (
          <span className="flex shrink-0 items-center gap-1 text-[var(--success)]">
            <CheckCircle2 className="size-3" />
            <span className="text-[11px]">Done</span>
          </span>
        )}
        {tool.durationMs != null && (
          <span className="readout shrink-0 text-[11px] text-[var(--mute)]">
            {clockMs(tool.durationMs)}
          </span>
        )}
        <ChevronRight
          className={cn(
            "size-2.5 shrink-0 text-[var(--mute)] transition-transform duration-[var(--dur-fast)]",
            open && "rotate-90",
          )}
        />
      </button>
      {!open && tool.summary && (
        <p className="-mt-0.5 truncate px-2.5 pb-1.5 text-[11px] text-[var(--mute)]">{tool.summary}</p>
      )}
      {open && (
        <div className="space-y-1.5 border-t border-[var(--hairline)] px-2.5 pb-2.5">
          <div>
            <div className="mt-1.5 mb-0.5 text-[11px] tracking-wide text-[var(--mute)] uppercase">
              Input
            </div>
            <pre className="max-h-32 overflow-x-auto rounded-[var(--radius-sm)] bg-[var(--panel)] p-1.5 font-mono text-[11px] whitespace-pre-wrap text-[var(--mute)]">
              {JSON.stringify(tool.input ?? {}, null, 2)}
            </pre>
          </div>
          <div>
            <div className="mb-0.5 text-[11px] tracking-wide text-[var(--mute)] uppercase">
              Output
            </div>
            <pre className="max-h-32 overflow-x-auto rounded-[var(--radius-sm)] bg-[var(--panel)] p-1.5 font-mono text-[11px] whitespace-pre-wrap text-[var(--slate)]">
              {tool.summary ?? (tool.running ? "Still running." : "No summary returned.")}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
