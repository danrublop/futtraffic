"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState, useSyncExternalStore, type CSSProperties } from "react";
import {
  Check,
  ChevronRight,
  Film,
  Grid3x3,
  MoreHorizontal,
  NotebookPen,
  PanelLeft,
  Plus,
  Settings,
} from "lucide-react";
import { toast } from "sonner";

import { MatchSwitcher } from "@/components/match-switcher";
import { JobBanner } from "@/components/job-banner";
import { useMatches } from "@/hooks/use-matches";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { deleteChat, listChats, patchChat, type ChatSummary } from "@/lib/agent";
import { sessionTree } from "@/lib/chat-folders";
import { useOrg, useTeam } from "@/lib/club";
import { useChatStore } from "@/lib/store";
import { TAB_TYPES, type TabType } from "@/lib/tabs";
import { LibraryActions } from "@/components/library-actions";
import { cn } from "@/lib/utils";

/** The header icon-button idiom, shared by every tool in the 40px bar —
 *  the editor header tools class verbatim. */
const iconBtn =
  "grid h-8 w-8 place-items-center rounded-[var(--radius-md)] text-[var(--graphite)] transition-colors duration-[var(--dur-fast)] hover:bg-[var(--card)] hover:text-[var(--ink)]";

/** Nav in the order the work happens: the matches you have, the film you watch
 *  them on, and the data they produced. `Data` is ONE surface holding teams,
 *  players and opponents on its own strip — it was two entries (Squad, Stats)
 *  and a header menu to pick between them, which meant three affordances for
 *  one question and no single place that was "all the numbers".
 *
 *  Everything in the nav is drawn at a size the row picks, so className is the
 *  whole contract — which is what lets the lucide icons and the hand-rolled
 *  the film icon sit in the same list. */
type NavIcon = (props: { className?: string }) => React.ReactNode;

interface NavEntry {
  href: string;
  label: string;
  icon: NavIcon;
}

/** Every sidebar entry is a SURFACE: clicking it sets `?v=` and the workspace
 *  renders that one thing. There is nothing to split off it and nothing to
 *  remember — the panes, their sizes and `editorStorageKey` are all gone; see
 *  `database-is-the-backbone (not published)`.
 *
 *  Matches used to be the exception here, a route of its own. It is gone too:
 *  the Data page's All matches table is the list you pick a match from.
 *
 *  No `needsMatch` any more either: every screen works with no match selected,
 *  each one says what it needs and offers the way to get it. A match makes
 *  these ABOUT something, it does not make them exist. */
const NAV: NavEntry[] = [
  // `lib=1` — the sidebar asks for the film LIBRARY, not the last match's
  // player. `withMatch` appends `?m=` to every row here, so without this the
  // one row a coach clicks to go and find some film would drop them straight
  // back into whatever they last watched. The header's film tool has no `lib`
  // and still opens the picked match, which is what "put the film beside this"
  // means; the gallery clears the flag as soon as you choose something.
  { href: "/workspace?v=film&lib=1", label: "Film", icon: Film },
  { href: "/workspace?v=data", label: "Data", icon: Grid3x3 },
  // Notes is a DESTINATION, not a list in this column. It opens its own page —
  // folder tree on the left, gallery of notes in the body — the same way Data
  // opens the tables. It was a section of rows down here while a note belonged
  // to a match and the sidebar was the only index there was; the library is
  // that index now, and doing it twice in a 252px column is the same widget
  // worse.
  { href: "/workspace?v=docs", label: "Notes", icon: NotebookPen },
];

/** The sidebar row's look, in ONE place, so the nav rows cannot drift apart.
 *
 *  border-transparent, not no border: the New Task CTA above has a real one,
 *  and without this the rows sit 1px narrower and the column edges disagree by
 *  a pixel down the whole sidebar. */
function navRowClass(active: boolean) {
  return cn(
    "flex h-9 w-full items-center gap-2.5 rounded-[var(--radius-md)] border border-transparent px-3 text-[14px] transition-colors duration-[var(--dur-fast)]",
    active
      ? "bg-[var(--card)] text-[var(--ink)]"
      : "text-[var(--ink-soft)] hover:bg-[var(--card)] hover:text-[var(--ink)]",
  );
}

/** One 36px ghost row, the sidebar idiom: 14px text, a 16px
 *  icon in --graphite, hover fills --card, the active row sits on --card in
 *  --ink. Disabled rows go --stone with no pointer. */
function NavRow({
  entry,
  active,
  withMatch,
  onNavigate,
}: {
  entry: NavEntry;
  active: boolean;
  withMatch: (href: string) => string;
  onNavigate?: () => void;
}) {
  const { href, label, icon: Icon } = entry;
  const cls = navRowClass(active);

  return (
    <Link
      href={withMatch(href)}
      aria-current={active ? "page" : undefined}
      onClick={onNavigate}
      className={cls}
    >
      <Icon className="size-4 shrink-0 text-[var(--graphite)]" />
      <span className="truncate">{label}</span>
    </Link>
  );
}

/** One row in a sidebar list — a saved chat or a match report. The small ring
 *  is the list glyph; it fills in on the active row. */
function ListRow({
  label,
  active,
  href,
  onClick,
}: {
  label: string;
  active: boolean;
  /** A link (reports) or a button (chats, which load into the store). */
  href?: string;
  onClick?: () => void;
}) {
  const cls = cn(
    "flex h-8 w-full items-center gap-2.5 rounded-[var(--radius-md)] px-3 text-left text-[13px] transition-colors duration-[var(--dur-fast)]",
    active
      ? "bg-[var(--card)] text-[var(--ink)]"
      : "text-[var(--ink-soft)] hover:bg-[var(--card)] hover:text-[var(--ink)]",
  );
  const inner = (
    <>
      <span
        aria-hidden
        className={cn(
          "size-[7px] flex-none rounded-full border transition-colors",
          active ? "border-[var(--ink-soft)]" : "border-[var(--stone)]",
        )}
      />
      <span className="min-w-0 flex-1 truncate">{label}</span>
    </>
  );
  return href ? (
    <Link href={href} onClick={onClick} title={label} className={cls} aria-current={active ? "page" : undefined}>
      {inner}
    </Link>
  ) : (
    <button type="button" onClick={onClick} title={label} className={cls}>
      {inner}
    </button>
  );
}

/** The saved sessions, as Claude Code lists them: sessions in no squad first,
 *  as plain rows, then a folder per squad in the club's ladder order. There
 *  is no "No squad" folder - a session without one is just a session.
 *
 *  Re-listed when a run finishes (that is when a new session gets its id and
 *  title) and whenever a rename, move or delete bumps `chatsVersion`. */
export function ChatList({
  onNavigate,
  withMatch,
}: {
  onNavigate?: () => void;
  withMatch: (href: string) => string;
}) {
  const router = useRouter();
  const { org } = useOrg();
  const [chats, setChats] = useState<ChatSummary[] | null>(null);
  /** Folders the coach folded away, by team id. Session only. */
  const [folded, setFolded] = useState<Set<string>>(() => new Set());
  const [renaming, setRenaming] = useState<string | null>(null);
  const chatId = useChatStore((s) => s.chatId);
  const streaming = useChatStore((s) => s.streaming);
  const version = useChatStore((s) => s.chatsVersion);
  const loadChat = useChatStore((s) => s.loadChat);
  const newChat = useChatStore((s) => s.newChat);
  const setTeamId = useChatStore((s) => s.setTeamId);
  const touchChats = useChatStore((s) => s.touchChats);
  const setTitle = useChatStore((s) => s.setTitle);

  useEffect(() => {
    if (streaming) return; // refetch on the falling edge, once the title exists
    let live = true;
    listChats()
      .then((c) => live && setChats(c))
      .catch(() => live && setChats([])); // no backend, or no key: an empty list
    return () => {
      live = false;
    };
  }, [streaming, version]);

  if (!chats?.length) {
    return (
      <p className="px-3 py-1 text-[12px] text-[var(--mute)]">
        {chats === null ? "" : "Your sessions will appear here"}
      </p>
    );
  }

  const teams = org?.teams ?? [];
  const toChat = () => {
    router.push(withMatch("/workspace?v=chat"));
    onNavigate?.();
  };
  const fail = (what: string) => (e: unknown) =>
    toast.error(what, { description: e instanceof Error ? e.message : String(e) });

  const actions: ChatActions = {
    teams,
    open: (c) => {
      loadChat(c.id).catch(fail("Could not open that session"));
      toChat();
    },
    rename: (c, title) => {
      setRenaming(null);
      if (!title.trim() || title.trim() === c.title) return;
      patchChat(c.id, { title }).then((saved) => {
        if (c.id === chatId) setTitle(saved.title);
        touchChats();
      }, fail("Could not rename the session"));
    },
    move: (c, teamId) =>
      patchChat(c.id, { team_id: teamId }).then(() => {
        if (c.id === chatId) setTeamId(teamId);
        touchChats();
      }, fail("Could not move the session")),
    remove: (c) => {
      if (!window.confirm(`Delete “${c.title}”? This cannot be undone.`)) return;
      deleteChat(c.id).then(() => {
        if (c.id === chatId) newChat(c.team_id ?? "");
        touchChats();
      }, fail("Could not delete the session"));
    },
  };

  const row = (c: ChatSummary) => (
    <ChatRow
      key={c.id}
      chat={c}
      active={c.id === chatId}
      renaming={renaming === c.id}
      onStartRename={() => setRenaming(c.id)}
      actions={actions}
    />
  );

  const { loose, folders } = sessionTree(chats, teams);

  return (
    <div className="flex flex-col gap-1">
      {loose.length > 0 && <div className="flex flex-col">{loose.map(row)}</div>}
      {folders.map((f) => {
        const shut = folded.has(f.teamId);
        return (
          <div key={f.teamId} className="flex flex-col">
            <div className="group/folder flex h-7 items-center gap-1 pr-1.5 pl-1.5">
              <button
                type="button"
                aria-expanded={!shut}
                onClick={() =>
                  setFolded((old) => {
                    const next = new Set(old);
                    if (shut) next.delete(f.teamId);
                    else next.add(f.teamId);
                    return next;
                  })
                }
                className="flex min-w-0 flex-1 items-center gap-1 rounded-[var(--radius-xs)] px-1 text-left text-[12px] text-[var(--slate)] hover:text-[var(--ink-soft)]"
              >
                <ChevronRight
                  className={cn("size-3 shrink-0 transition-transform duration-[var(--dur-fast)]", !shut && "rotate-90")}
                />
                <span className="truncate">{f.label}</span>
                <span className="readout shrink-0 text-[11px] text-[var(--mute)]">{f.chats.length}</span>
              </button>
              <button
                type="button"
                title={`New session in ${f.label}`}
                aria-label={`New session in ${f.label}`}
                onClick={() => {
                  newChat(f.teamId);
                  toChat();
                }}
                className="grid size-5 flex-none place-items-center rounded-[var(--radius-xs)] text-[var(--graphite)] opacity-0 transition-opacity group-hover/folder:opacity-100 hover:bg-[var(--card)] hover:text-[var(--ink)] focus-visible:opacity-100"
              >
                <Plus size={13} strokeWidth={1.75} />
              </button>
            </div>
            {!shut && f.chats.map(row)}
          </div>
        );
      })}
    </div>
  );
}

interface ChatActions {
  teams: { team_id: string; name: string }[];
  open: (c: ChatSummary) => void;
  rename: (c: ChatSummary, title: string) => void;
  move: (c: ChatSummary, teamId: string) => void;
  remove: (c: ChatSummary) => void;
}

/** One session: the row that opens it, and a ⋯ menu beside it — a sibling, not a
 *  child, because a menu button inside the row's own button is not valid HTML
 *  and swallows the click. Renaming swaps the label for a field in place. */
function ChatRow({
  chat,
  active,
  renaming,
  onStartRename,
  actions,
}: {
  chat: ChatSummary;
  active: boolean;
  renaming: boolean;
  onStartRename: () => void;
  actions: ChatActions;
}) {
  if (renaming) {
    return (
      <input
        autoFocus
        defaultValue={chat.title}
        aria-label="Chat name"
        onFocus={(e) => e.currentTarget.select()}
        onBlur={(e) => actions.rename(chat, e.currentTarget.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") e.currentTarget.blur();
          if (e.key === "Escape") actions.rename(chat, chat.title);
        }}
        // The border is the same in every state, so focus gets the accent -
        // exactly the machine state globals.css reserves it for. A review also
        // flagged 13px as an iOS zoom trigger; mobile is not a supported
        // surface yet, so the size stays as designed.
        className="mx-1 h-8 rounded-[var(--radius-md)] border border-[var(--hairline-strong)] bg-[var(--card)] px-2 text-[13px] text-[var(--ink)] focus-visible:border-[var(--accent)] focus-visible:outline-none"
      />
    );
  }
  return (
    <div className="group/row relative">
      <ListRow label={chat.title} active={active} onClick={() => actions.open(chat)} />
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            type="button"
            aria-label={`Actions for ${chat.title}`}
            className="absolute top-1.5 right-1.5 grid size-5 place-items-center rounded-[var(--radius-xs)] text-[var(--graphite)] opacity-0 group-hover/row:opacity-100 hover:bg-[var(--card)] hover:text-[var(--ink)] focus-visible:opacity-100 data-[state=open]:opacity-100"
          >
            <MoreHorizontal className="size-3.5" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-44">
          <DropdownMenuItem onSelect={onStartRename}>Rename</DropdownMenuItem>
          {actions.teams.length > 0 && (
            <DropdownMenuSub>
              <DropdownMenuSubTrigger>Move to squad</DropdownMenuSubTrigger>
              <DropdownMenuSubContent className="w-44">
                {actions.teams.map((t) => (
                  <DropdownMenuItem
                    key={t.team_id}
                    onSelect={() => actions.move(chat, t.team_id)}
                    className="gap-2"
                  >
                    <Check
                      className={cn("size-4", (chat.team_id ?? "") === t.team_id ? "opacity-100" : "opacity-0")}
                    />
                    {t.name}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuSubContent>
            </DropdownMenuSub>
          )}
          {chat.team_id && actions.teams.some((t) => t.team_id === chat.team_id) && (
            <DropdownMenuItem onSelect={() => actions.move(chat, "")}>Remove from squad</DropdownMenuItem>
          )}
          <DropdownMenuSeparator />
          <DropdownMenuItem variant="destructive" onSelect={() => actions.remove(chat)}>
            Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}

/** The Sessions section's +: a plain new session, or one in a squad. The
 *  squad the app is already scoped to (the switcher's lens) is ticked, because
 *  that is almost always the one you mean - and a club with no squads skips
 *  the menu. */
function NewChatButton({ onNew }: { onNew: (teamId: string) => void }) {
  const { org } = useOrg();
  const [lens] = useTeam();
  const teams = org?.teams ?? [];
  const cls =
    "grid size-5 flex-none place-items-center rounded-[var(--radius-xs)] text-[var(--graphite)] opacity-0 transition-[opacity,background-color,color] duration-[var(--dur-fast)] group-hover/sec:opacity-100 hover:bg-[var(--card)] hover:text-[var(--ink)] focus-visible:opacity-100 focus-visible:ring-2 focus-visible:ring-[var(--accent)] data-[state=open]:opacity-100";
  if (!teams.length) {
    return (
      <button type="button" aria-label="New session" title="New session" onClick={() => onNew("")} className={cls}>
        <Plus size={14} strokeWidth={1.75} />
      </button>
    );
  }
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button type="button" aria-label="New session" title="New session" className={cls}>
          <Plus size={14} strokeWidth={1.75} />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-48">
        <DropdownMenuItem onSelect={() => onNew("")}>New session</DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuLabel>In a squad</DropdownMenuLabel>
        {teams.map((t) => (
          <DropdownMenuItem key={t.team_id} onSelect={() => onNew(t.team_id)} className="gap-2">
            <Check className={cn("size-4", lens === t.team_id ? "opacity-100" : "opacity-0")} />
            {t.name}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

/** The full sidebar contents, shared by the docked desktop column and the
 *  mobile slide-over: the window controls row, New Task, the four nav rows,
 *  the Chats section, and the Guest account row with the settings gear. */
function SidebarBody({
  pathname,
  view,
  withMatch,
  onNavigate,
  onToggle,
  topPad,
  drag,
  noDrag,
}: {
  pathname: string;
  /** Which sidebar destination the workspace is showing. */
  view: TabType;
  withMatch: (href: string) => string;
  onNavigate?: () => void;
  onToggle: () => void;
  topPad: string;
  drag?: CSSProperties;
  noDrag?: CSSProperties;
}) {
  const router = useRouter();
  const newChat = useChatStore((s) => s.newChat);

  return (
    <>
      {/* The window controls row. This is the header now: it lives INSIDE the
          sidebar, level with the content pane's own header, so neither one
          spans the window and the two columns are independent. Under Electron
          it clears the traffic lights and drags the window. */}
      {/* -mx-2.5 so the left pad is measured from the WINDOW edge, not from
          inside the column's own padding — the traffic lights don't know about
          our gutter. */}
      <div
        className={cn("-mx-2.5 flex h-11 shrink-0 items-center gap-0.5 pr-2.5", topPad)}
        style={drag}
      >
        <button
          onClick={onToggle}
          aria-label="Hide sidebar"
          title="Hide sidebar"
          style={noDrag}
          className={iconBtn}
        >
          <PanelLeft className="size-[17px]" />
        </button>
        <span className="select-none text-[13px] font-medium text-[var(--ink-soft)]">FutTraffic</span>
      </div>

      {/* The conversation is the workspace; a new task is a new chat. It reads
          as one of the nav rows, not as a raised CTA above them — it goes to a
          screen like every other row here, so it should not be shouting. */}
      <nav aria-label="Sections" className="mt-1 flex flex-col">
        <NavRow
          entry={{ href: "/workspace?v=chat", label: "New Task", icon: Plus }}
          active={pathname === "/workspace" && view === "chat"}
          withMatch={withMatch}
          onNavigate={onNavigate}
        />
        {NAV.map((entry) => (
          <NavRow
            key={entry.href}
            entry={entry}
            active={
              entry.href.startsWith("/workspace")
                ? // READ the `v`, do not match the end of the string: Film's
                  // href carries `&lib=1` after it, and endsWith left the row
                  // unlit on the one surface it had opened.
                  pathname === "/workspace" &&
                  new URLSearchParams(entry.href.split("?")[1] ?? "").get("v") === view
                : pathname === entry.href
            }
            withMatch={withMatch}
            onNavigate={onNavigate}
          />
        ))}
      </nav>

      {/* The two lists: what has been analysed, and what has been asked. One
          scroll area so a long list of either cannot push the other off. */}
      <div className="mt-3 min-h-0 flex-1 overflow-y-auto">
        <div className="group/sec flex h-7 items-center gap-2 px-3">
          <span className="min-w-0 flex-1 truncate text-[12px] text-[var(--slate)]">Sessions</span>
          <NewChatButton
            onNew={(teamId) => {
              newChat(teamId);
              router.push(withMatch("/workspace?v=chat"));
              onNavigate?.();
            }}
          />
        </div>
        <ChatList onNavigate={onNavigate} withMatch={withMatch} />
      </div>

      {/* The account row. No accounts exist yet, so this is a placeholder:
          a fixed "Guest" identity and a gear that opens /settings, which is
          where Pitch setup and Accuracy live now. */}
      {/* The sidebar user button: ONE row that hovers as a unit, not an
          avatar plus a name plus a separate icon button with its own hover.
          26px avatar filled --card, 13px medium name, a 15px gear in
          --graphite, and no divider above it — just mt-1.5. */}
      {/* Upload, at the foot and over the account row: it puts a game in
          rather than taking you somewhere, so it is not one of the nav rows. */}
      <LibraryActions className="mt-2 w-full" />
      <Link
        href={withMatch("/settings")}
        onClick={onNavigate}
        aria-label="Settings"
        className="mt-1.5 flex w-full items-center gap-2.5 rounded-[var(--radius-sm)] px-3 py-2 text-left transition-colors duration-[var(--dur-fast)] hover:bg-[var(--card)]"
      >
        <span
          aria-hidden
          className="grid size-[26px] flex-none place-items-center rounded-full bg-[var(--card)] text-[12px] font-semibold text-[var(--ink)]"
        >
          G
        </span>
        <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-[var(--ink)]">
          Guest <span className="font-normal text-[var(--mute)]">· Local</span>
        </span>
        <Settings size={15} className="flex-none text-[var(--graphite)]" />
      </Link>
    </>
  );
}

/** The shell geometry: a 40px transparent header on --bg (draggable
 *  under Electron, clearing the macOS traffic lights), and below it a 252px
 *  sidebar column sitting directly on --bg with the page content rendered as
 *  a --panel card — rounded top-left corner, 1px hairline top/left border —
 *  that fills the rest and owns its own scroll. */
export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const params = useSearchParams();
  const match = params.get("m");
  /** Which sidebar destination the workspace is showing. Every entry is a page
   *  with its own remembered layout; `chat` is the landing one ("New Task"). */
  const raw = params.get("v");
  const view: TabType = (TAB_TYPES as readonly string[]).includes(raw ?? "")
    ? (raw as TabType)
    : "chat";

  const withMatch = (href: string) =>
    match ? `${href}${href.includes("?") ? "&" : "?"}m=${encodeURIComponent(match)}` : href;

  // Two independent states: the docked desktop column starts open; the mobile
  // slide-over starts closed and only opens from the header toggle.
  const [desktopOpen, setDesktopOpen] = useState(true);
  const [mobileOpen, setMobileOpen] = useState(false);
  const closeMobile = () => setMobileOpen(false);

  const toggleSidebar = () => {
    if (window.matchMedia("(min-width: 768px)").matches) {
      setDesktopOpen((o) => !o);
    } else {
      setMobileOpen((o) => !o);
    }
  };

  // Electron detection must wait for the client — the preload bridge does not
  // exist during SSR / static export, and reading it there would also tear
  // hydration. useSyncExternalStore with a false server snapshot is exactly
  // that: first client render still reads false, then it re-reads on mount.
  const isElectron = useSyncExternalStore(
    () => () => {},
    () => window.futtrafficDesktop?.isElectron === true,
    () => false,
  );

  const drag = isElectron ? ({ WebkitAppRegion: "drag" } as CSSProperties) : undefined;
  const noDrag = isElectron ? ({ WebkitAppRegion: "no-drag" } as CSSProperties) : undefined;

  const router = useRouter();

  // The workspace is the only route that renders a surface, so it is the only
  // place a lit sidebar row means anything.
  const isWorkspace = pathname === "/workspace";

  // LAND ON A MATCH, DON'T ASK FOR ONE.
  //
  // Every workspace surface is about one match and reads it off `?m=`, but
  // nothing ever put one there: withMatch() only forwards a match that is
  // already in the URL, so arriving fresh — a sidebar click, a header tool, a
  // pasted link — meant the data surface opened on "Choose a match" with a
  // library of finished matches sitting behind it. A gate is the right answer
  // to "which of these did you mean"; it is the wrong answer to "show me my
  // data", which is what clicking Data is.
  //
  // So the newest match with numbers becomes the default, written into the URL
  // rather than held beside it — the switcher, the film tab and a copied link
  // all read the same `?m=` and cannot disagree about which match is open.
  // `replace`, so Back does not bounce through the match-less URL.
  //
  // Nothing is invented when there is nothing to pick: no matches, or none
  // finished, still falls through to the gate, which says which it is.
  const { matches } = useMatches();
  useEffect(() => {
    if (match || !isWorkspace || !matches) return;
    const newest = matches
      .filter((m) => m.has_match)
      .sort((a, b) => (b.created ?? 0) - (a.created ?? 0))[0];
    if (!newest) return;
    const q = new URLSearchParams(params.toString());
    q.set("m", newest.id);
    router.replace(`${pathname}?${q.toString()}`);
  }, [match, isWorkspace, matches, params, pathname, router]);
  // One surface at a time, and `?v=` names it. This was the union of every
  // pane's active tab; with the panes gone it is a set of one, kept as a set
  // so the reads below did not all have to change.
  const showing = new Set<TabType>(isWorkspace ? [view] : []);

  /** The film's match switcher, and the one condition for it. Named once
   *  because `bare` and the header JSX both need it, and two copies of a
   *  three-term condition drift: the header hides while the switcher wants in. */
  const filmSwitcher = showing.has("film") && !!match && !params.get("lib");
  /** Does the header have anything on it? Only the film's match switcher and
   *  the sidebar toggle are left, and both are conditional. */
  const bare = desktopOpen && !isElectron && !filmSwitcher;

  // The traffic lights sit at the window's top-left, so whichever column is
  // there has to clear them: the sidebar when it is open, the content pane
  // when it is collapsed.
  const lightsPad = (owns: boolean) => (isElectron && owns ? "pl-20" : "pl-2");

  return (
    <div className="relative flex h-dvh overflow-hidden bg-[var(--bg)] text-[var(--ink)]">
      {/* Docked sidebar — desktop only; mobile gets the slide-over below. */}
      {desktopOpen && (
        <aside className="hidden h-full w-[252px] shrink-0 flex-col bg-[var(--bg)] px-2.5 pb-2.5 md:flex">
          <SidebarBody
            pathname={pathname}
            view={view}
            withMatch={withMatch}
            onToggle={toggleSidebar}
            topPad={lightsPad(true)}
            drag={drag}
            noDrag={noDrag}
          />
        </aside>
      )}

      {/* Mobile slide-over: the same 252px sidebar as a floating panel over
          a scrim. The BottomNav covers primary navigation on phones; this
          keeps the header toggle meaningful there too. */}
      {mobileOpen && (
        <div className="absolute inset-0 z-40 md:hidden">
          <div
            role="presentation"
            className="absolute inset-0 bg-black/60"
            onClick={closeMobile}
          />
          <aside className="absolute inset-y-0 left-0 flex w-[252px] flex-col bg-[var(--bg)] px-2.5 pb-2.5">
            <SidebarBody
              pathname={pathname}
              view={view}
                withMatch={withMatch}
              onNavigate={closeMobile}
              onToggle={closeMobile}
              topPad="pl-2"
            />
          </aside>
        </div>
      )}

      {/* Content pane: its own header, then the route. One hairline separates
          it from the sidebar; nothing spans the window. */}
      <main className="flex min-w-0 flex-1 flex-col overflow-hidden border-l border-[var(--hairline)] bg-[var(--panel)]">
        <header
          className={cn(
            "flex h-11 shrink-0 items-center gap-0.5 pr-2",
            lightsPad(!desktopOpen),
            // NOTHING LEFT TO SAY, SO NO ROW. The four surface tools used to
            // live here; with them gone this bar is empty on a desktop whose
            // sidebar is open and whose surface is not the film, and 44px of
            // empty chrome above the page reads as a gap rather than as a
            // header. It still renders wherever it has a job: the phone (the
            // toggle is the only way to the slide-over), a collapsed sidebar
            // (same), the film (the match switcher), and Electron (the drag
            // region and the traffic lights).
            bare && "md:hidden",
          )}
          style={drag}
        >
          {/* The only way back once the sidebar is hidden. */}
          {!desktopOpen && (
            <button
              onClick={toggleSidebar}
              aria-label="Show sidebar"
              title="Show sidebar"
              style={noDrag}
              className={cn(iconBtn, "mr-1")}
            >
              <PanelLeft className="size-[17px]" />
            </button>
          )}
          {/* On a phone there is no docked column at all, so the toggle is
              always the way to the slide-over. */}
          <button
            onClick={toggleSidebar}
            aria-label="Show sidebar"
            title="Show sidebar"
            className={cn(iconBtn, "mr-1 md:hidden")}
          >
            <PanelLeft className="size-[17px]" />
          </button>
          {/* The match selector is ONLY here while the film is open: everywhere
              else a match is picked where matches are listed — the Data page's
              All matches, the chat's recent list — and a second way to change
              it meant the same choice lived in two places. The film is the
              exception because it is the one surface that IS one match, watched
              for an hour, and going back to a list to change reels is the trip
              this header exists to save. */}
          <div style={noDrag} className="flex min-w-0 flex-1 items-center gap-1">
            {/* …and only with a film actually OPEN. The gallery is a list you
                pick a match FROM, so a second picker in the header there is
                two answers to the same question — the same reason the switcher
                is absent from every other screen. */}
            {filmSwitcher && <MatchSwitcher />}
          </div>
        </header>

        <JobBanner />
        <div
          className={cn(
            "flex min-h-0 min-w-0 flex-1 flex-col",
            isWorkspace
              ? "overflow-hidden"
              // p-4, the same inset a workspace pane gives its content. It used
              // to grow to p-6 at md, so Matches sat further from the edge than
              // Notes did in a pane beside it and the two surfaces read as two
              // different apps.
              : "overflow-y-auto p-4 pb-[calc(4.25rem+env(safe-area-inset-bottom))]",
          )}
        >
          {children}
        </div>
      </main>

      {/* No docked chat panel any more. The conversation is a surface like
          the others and the sidebar's New Task row opens it, so a second
          rendering of the same thread — with its own chrome, its own composer
          and its own empty state — had nothing left to do. */}

      <BottomNav pathname={pathname} view={view} withMatch={withMatch} />
    </div>
  );
}

/** Thumb-reachable nav for the four screens people move between.
 *
 *  The slide-over sidebar already works on mobile and still holds the full
 *  nav behind the header toggle — this exists because reaching it costs two
 *  taps on every navigation, and this product is used one-handed on a
 *  touchline. Secondary items (Pitch setup, Accuracy) live behind the gear
 *  on /settings: they are visited once per match, not during one.
 *
 *  ponytail: the four primary items only. A five-item bar with an overflow
 *  "More" is the next step if the nav ever grows, not before.
 */
function BottomNav({
  pathname,
  view,
  withMatch,
}: {
  pathname: string;
  /** Which surface the workspace is showing, so a row can light up. */
  view: TabType;
  withMatch: (href: string) => string;
}) {
  return (
    <nav
      aria-label="Primary"
      // Columns come from NAV, not a literal: this said grid-cols-4 while NAV
      // held three entries, so the row sat off-centre with a quarter empty.
      style={{ gridTemplateColumns: `repeat(${NAV.length}, minmax(0,1fr))` }}
      className="fixed inset-x-0 bottom-0 z-30 grid border-t border-[var(--hairline)] bg-[var(--panel)]/95 pb-[env(safe-area-inset-bottom)] backdrop-blur md:hidden"
    >
      {NAV.map(({ href, label, icon: Icon }) => {
        // `pathname === href` could never be true: every entry is
        // /workspace?v=… and pathname is /workspace, so no row ever lit and
        // none carried aria-current. Compare the surface, as NavRow does.
        const active =
          pathname === "/workspace" &&
          new URLSearchParams(href.split("?")[1] ?? "").get("v") === view;
        // min-h-14 keeps every target at/over the 44px floor.
        const cls =
          "relative flex min-h-14 flex-col items-center justify-center gap-1 text-[0.6875rem] font-medium tracking-tight transition-colors";

        return (
          <Link
            key={href}
            href={withMatch(href)}
            aria-current={active ? "page" : undefined}
            className={`${cls} ${
              active ? "text-[var(--ink)]" : "text-[var(--mute)] hover:text-[var(--ink)]"
            }`}
          >
            {/* The active mark is a chalk line at the top edge — the same
                language the sidebar and the pitch itself use. */}
            <span
              aria-hidden
              className={`absolute top-0 left-1/2 h-0.5 w-10 -translate-x-1/2 rounded-full transition-opacity ${
                active ? "bg-[var(--action)] opacity-100" : "opacity-0"
              }`}
            />
            <Icon className="size-5" />
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
