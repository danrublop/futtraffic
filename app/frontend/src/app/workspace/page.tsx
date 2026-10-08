"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { ChatConversation, ChatTitle } from "@/components/chat-panel";
import { DocPanel } from "@/components/doc-panel";
import { VideoPanel, type FilmSeek } from "@/components/video-panel";
import { DataBrowser } from "@/components/data/data-browser";
import { Dock, DockButtons } from "@/components/dock";
import { useDock } from "@/lib/dock";
import { useChatStore } from "@/lib/store";
import { dockTarget, fromAgentTab, TAB_TYPES, type TabType } from "@/lib/tabs";
import { cn } from "@/lib/utils";

/** THE WORKSPACE — one surface, chosen by the sidebar; on the chat, a dock of
 *  more surfaces beside it (components/dock.tsx).
 *
 *  This was an editor: panes on a drag divider, a tab strip per pane, a
 *  reducer, and a layout persisted per destination and per match. All of it is
 *  gone. `?v=` names the surface, the sidebar sets it, and that is the whole
 *  model — so a surface cannot be open twice, there is no arrangement to
 *  restore, and nothing on screen is chrome for managing windows.
 *
 *  The agent drives this page through ui_action frames: the chat store parks
 *  them in `pendingUiAction` and the subscription below consumes them. On the
 *  chat, an action opens (or focuses) a dock tab, so the answer that asked for
 *  the film stays on screen beside it. Anywhere else it navigates, as it did
 *  when there was only ever one surface. An action parked while another route
 *  was showing replays on mount (fireImmediately).
 */
export default function WorkspacePage() {
  const params = useSearchParams();
  const router = useRouter();
  const [seekReq, setSeekReq] = useState<FilmSeek | null>(null);

  const raw = params.get("v");
  const view: TabType = (TAB_TYPES as readonly string[]).includes(raw ?? "")
    ? (raw as TabType)
    : "chat";

  // The subscription must be set up ONCE. Reading the router and the query
  // through a ref keeps it that way: both change identity on every navigation,
  // and re-subscribing on each one would drop a frame that arrived in between.
  const nav = useRef({ params, router });
  useEffect(() => {
    nav.current = { params, router };
  }, [params, router]);

  useEffect(() => {
    return useChatStore.subscribe(
      (s) => s.pendingUiAction,
      (action) => {
        if (!action) return;
        useChatStore.getState().consumeUiAction();

        const here = nav.current.params;
        if ((here.get("v") ?? "chat") === "chat") {
          const { kind, params } = dockTarget(action, here.get("m"));
          const id = useDock.getState().openTab(kind, params);
          if (action.action === "seek_film") useDock.getState().seekFilm(id, action.t);
          return;
        }

        const tab =
          action.action === "open_tab" ? fromAgentTab(action.tab) : ({ type: "film" } as const);

        const q = new URLSearchParams(nav.current.params.toString());
        q.set("v", tab.type);
        if ("view" in tab && tab.view) q.set("tab", tab.view);
        else q.delete("tab");
        // `lib=1` asks the film surface for the GALLERY. Carrying it through an
        // agent action would land on the list with no player mounted, and the
        // stamped seek below would be dropped on the floor.
        if (tab.type === "film") q.delete("lib");
        nav.current.router.replace(`/workspace?${q.toString()}`);

        // Stamped and handed to VideoPanel as a prop rather than called
        // imperatively: the film surface may not be mounted yet, and the panel
        // only exists after the navigation above lands.
        if (action.action === "seek_film") {
          setSeekReq((p) => ({ t: action.t, n: (p?.n ?? 0) + 1 }));
        }
      },
      { fireImmediately: true },
    );
  }, []);

  return (
    <div
      className={cn(
        // @container: the surfaces lay out against THIS width, not the
        // window's, because the same surface also runs in a dock tab beside the
        // chat. Their thresholds are the old window breakpoints less the
        // sidebar (and, for the documents, this p-4).
        "@container flex h-full min-h-0 w-full flex-col",
        // Film owns its own scrolling and runs to the surface's edges: the
        // events column is flush right with nothing padding it off the page.
        // Narrow, the two columns stack and the page scrolls them as one.
        // Every other surface is a document and gets the page gutter.
        view === "film" && "overflow-y-auto xl:overflow-hidden",
        view !== "chat" && view !== "film" && "overflow-y-auto p-4",
      )}
    >
      <Surface view={view} tab={params.get("tab")} seekRequest={seekReq} />
    </div>
  );
}

/** The surfaces render the same components the routes do, so the workspace can
 *  never drift from the pages.
 *
 *  Data is the last branch because it is the only remaining TabType, not
 *  because it is a catch-all: an unrecognised `?v=` is narrowed against
 *  TAB_TYPES above and lands on chat, which is where someone arriving with no
 *  destination should land. */
function Surface({
  view,
  tab,
  seekRequest,
}: {
  view: TabType;
  tab: string | null;
  seekRequest: FilmSeek | null;
}) {
  if (view === "chat") {
    return (
      <div className="flex h-full min-h-0">
        <div className="flex min-w-0 flex-1 flex-col">
          {/* The dock's buttons stay with the chat, outside the dock, on the
              line of its tab row: the dock's my-2 and 1px border, then an h-11
              row with a 1px border-b, mirrored here so the centres match. */}
          <div className="mt-2 flex h-[calc(2.75rem+1px)] shrink-0 items-center justify-between gap-2 border-y border-transparent pr-2 pl-4">
            <ChatTitle />
            <DockButtons />
          </div>
          <ChatConversation />
        </div>
        <Dock />
      </div>
    );
  }
  if (view === "film") {
    return <VideoPanel seekRequest={seekRequest} />;
  }
  if (view === "docs") {
    return <DocPanel />;
  }
  return <DataBrowser initialTab={tab ?? undefined} />;
}
