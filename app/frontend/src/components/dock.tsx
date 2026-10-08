"use client";

import {
  useMemo,
  useRef,
  useSyncExternalStore,
  type CSSProperties,
} from "react";
import { useSearchParams } from "next/navigation";
import { Film, Grid3x3, NotebookPen, X } from "lucide-react";

import { DataBrowser } from "@/components/data/data-browser";
import { DocPanel } from "@/components/doc-panel";
import {
  SurfaceParamsProvider,
  type SurfaceParams,
} from "@/components/surface-params";
import { VideoPanel } from "@/components/video-panel";
import { useMatches } from "@/hooks/use-matches";
import { DOCK_MIN, useDock, type DockKind, type DockTab } from "@/lib/dock";
import { noteName } from "@/lib/notes";
import { matchTitle } from "@/lib/status";
import { cn } from "@/lib/utils";

const ICON: Record<DockKind, typeof Film> = {
  note: NotebookPen,
  film: Film,
  data: Grid3x3,
};

/** The dock: surfaces open beside the chat, as tabs, one visible.
 *
 *  The state and its two rules (only the active tab mounted; one tab per note)
 *  live in lib/dock.ts. This is the strip, the resize edge, and the one mounted
 *  surface, wrapped so everything under it reads the TAB's match and note
 *  rather than the page's.
 *
 *  Not drawn until the client has mounted: the tabs come from localStorage,
 *  which the prerendered page cannot know, and drawing them in the first paint
 *  would not match the HTML it hydrates.
 */
export function Dock() {
  const mounted = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false,
  );
  const { tabs, active, open, width, seek, close, focus, setOpen, setWidth } =
    useDock();
  const drag = useRef<{ x: number; w: number } | null>(null);

  if (!mounted || !open || tabs.length === 0) return null;
  const tab = tabs.find((t) => t.id === active) ?? tabs[0];

  return (
    <aside
      aria-label="Dock"
      style={{ width }}
      // A container beside the chat, not a column of it: inset on three sides,
      // its own border and radius, so it reads as a window the chat opened.
      className="relative my-2 mr-2 hidden max-w-[70%] shrink-0 flex-col md:flex"
    >
      {/* The resize edge. Pointer capture, so a fast drag that leaves the
          4px strip keeps resizing. */}
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize the dock"
        className="absolute inset-y-0 -left-1 z-10 w-2 cursor-col-resize"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          drag.current = { x: e.clientX, w: width };
        }}
        onPointerMove={(e) => {
          if (!drag.current) return;
          const max = window.innerWidth * 0.7;
          setWidth(
            Math.min(
              max,
              Math.max(DOCK_MIN, drag.current.w + drag.current.x - e.clientX),
            ),
          );
        }}
        onPointerUp={() => {
          drag.current = null;
        }}
      />

      <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-[var(--radius-xl)] border border-[var(--hairline)] bg-[var(--bg)]">
        <div className="flex h-11 shrink-0 items-center gap-1 border-b border-[var(--hairline)] px-1.5">
          <div
            role="tablist"
            className="flex min-w-0 flex-1 items-center gap-0.5 overflow-x-auto"
          >
            {tabs.map((t) => (
              <TabButton
                key={t.id}
                tab={t}
                on={t.id === tab.id}
                onFocus={() => focus(t.id)}
                onClose={() => close(t.id)}
              />
            ))}
          </div>
          {/* Hide sits at the far right, outside the strip: the strip scrolls,
            and three tabs used to push a scrolling hide button off the edge. */}
          <button
            type="button"
            onClick={() => setOpen(false)}
            aria-label="Hide the dock"
            title="Hide the dock — the tabs stay"
            className={BTN}
          >
            <X className="size-[17px]" />
          </button>
        </div>

        <div role="tabpanel" className="min-h-0 flex-1">
          {/* key = the tab: switching tabs unmounts the last surface, which is
            the memory rule and the stale-note rule in one line. */}
          <TabSurface
            key={tab.id}
            tab={tab}
            seek={seek?.tabId === tab.id ? { t: seek.t, n: seek.n } : null}
          />
        </div>
      </div>
    </aside>
  );
}

const BTN =
  "grid size-8 shrink-0 place-items-center rounded-[var(--radius-md)] text-[var(--graphite)] transition-colors hover:bg-[var(--card)] hover:text-[var(--ink)]";

/** Notes, film, data: each opens a tab in the dock beside the chat — the
 *  page's match's film or data when there is one, and focuses the tab that
 *  already shows it rather than opening it twice. They live in the chat's own
 *  top row, outside the dock, open or not. Desktop only: a phone has no room
 *  beside the chat, and the bottom nav opens each one full screen. */
export function DockButtons({ style }: { style?: CSSProperties }) {
  const openTab = useDock((s) => s.openTab);
  const match = useSearchParams().get("m");
  return (
    <div style={style} className="hidden shrink-0 items-center gap-0.5 md:flex">
      {(["note", "film", "data"] as const).map((kind) => {
        const Icon = ICON[kind];
        const label = kind === "note" ? "notes" : kind;
        return (
          <button
            key={kind}
            type="button"
            title={`Open ${label} beside the chat`}
            aria-label={`Open ${label} beside the chat`}
            onClick={() =>
              openTab(kind, kind === "note" || !match ? {} : { m: match })
            }
            className={BTN}
          >
            <Icon className="size-[17px]" />
          </button>
        );
      })}
    </div>
  );
}

function TabButton({
  tab,
  on,
  onFocus,
  onClose,
}: {
  tab: DockTab;
  on: boolean;
  onFocus: () => void;
  onClose: () => void;
}) {
  const { matches } = useMatches();
  const Icon = ICON[tab.kind];
  const m = tab.params.m
    ? matches?.find((x) => x.id === tab.params.m)
    : undefined;
  const label =
    tab.kind === "note"
      ? tab.params.n
        ? noteName(tab.params.n)
        : "Notes"
      : tab.kind === "film"
        ? m
          ? matchTitle(m)
          : "Film"
        : m
          ? `Data · ${matchTitle(m)}`
          : "Data";
  return (
    <div
      role="tab"
      aria-selected={on}
      className={cn(
        "group/tab flex h-7 max-w-[200px] shrink-0 items-center gap-1.5 rounded-[var(--radius-md)] pr-1 pl-2 text-[12.5px] transition-colors",
        on
          ? "bg-[var(--card)] text-[var(--ink)]"
          : "text-[var(--ink-soft)] hover:bg-[var(--card)]/60",
      )}
    >
      <button
        type="button"
        onClick={onFocus}
        className="flex min-w-0 items-center gap-1.5"
        title={label}
      >
        <Icon className="size-3.5 shrink-0 text-[var(--graphite)]" />
        <span className="truncate">{label}</span>
      </button>
      <button
        type="button"
        onClick={onClose}
        aria-label={`Close ${label}`}
        className={cn(
          "grid size-4 shrink-0 place-items-center rounded-[var(--radius-xs)] text-[var(--graphite)] hover:text-[var(--ink)]",
          on ? "opacity-100" : "opacity-0 group-hover/tab:opacity-100",
        )}
      >
        <X className="size-3" />
      </button>
    </div>
  );
}

function TabSurface({
  tab,
  seek,
}: {
  tab: DockTab;
  seek: { t: number; n: number } | null;
}) {
  const setParams = useDock((s) => s.setParams);
  const value = useMemo<SurfaceParams>(
    () => ({
      docked: true,
      get: (key) => tab.params[key] ?? null,
      set: (patch) => setParams(tab.id, patch),
    }),
    [tab, setParams],
  );
  return (
    <SurfaceParamsProvider value={value}>
      {tab.kind === "film" ? (
        <div className="@container h-full overflow-y-auto">
          <VideoPanel seekRequest={seek} />
        </div>
      ) : (
        <div className="@container h-full overflow-y-auto p-3">
          {tab.kind === "note" ? (
            <DocPanel />
          ) : (
            <DataBrowser initialTab={tab.params.tab} />
          )}
        </div>
      )}
    </SurfaceParamsProvider>
  );
}
