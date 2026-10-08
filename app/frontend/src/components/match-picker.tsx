"use client";

import { useMemo, useState } from "react";
import { Check, Plus, Search } from "lucide-react";

import { useSurfaceParams } from "@/components/surface-params";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { UploadCard } from "@/components/upload-card";
import { useMatches } from "@/hooks/use-matches";
import { matchDate, matchTitle, statusOf, TONE_CLASS } from "@/lib/status";
import { cn } from "@/lib/utils";

/** Choose a match, from wherever you are.
 *
 *  Every screen in this app used to dead-end the same way: "No match selected —
 *  pick one from the switcher above, or upload one", with a link back to the
 *  match list. That answer is worse than the question. It throws away the page
 *  you were on, and it tells someone to go and do a thing rather than letting
 *  them do it.
 *
 *  So the gates open this instead. It picks WITHIN the current route — the
 *  pathname is kept and only `?m=` changes — so choosing a match on the film
 *  screen leaves you on the film screen, now with a match in it. Uploading is
 *  here too, because "I do not have one yet" is the other honest answer to
 *  "which match?".
 */
export function MatchPicker({
  trigger,
  /** Overrides the default navigation — the notepad's `/stat`, say, wants to
   *  set the match without leaving the block the caret is in. */
  onPick,
}: {
  trigger: React.ReactNode;
  onPick?: (id: string) => void;
}) {
  const params = useSurfaceParams();
  const current = params.get("m");
  const { matches } = useMatches();
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const [adding, setAdding] = useState(false);

  const shown = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const list = (matches ?? []).filter(
      (m) => !needle || matchTitle(m).toLowerCase().includes(needle),
    );
    return [...list].sort((a, b) => (b.created ?? 0) - (a.created ?? 0));
  }, [matches, q]);

  const choose = (id: string) => {
    setOpen(false);
    if (onPick) return onPick(id);
    // Only `m` changes. This used to push `?m=` as the whole query, dropping
    // `?v=` and bouncing the workspace to its default - and inside a dock tab
    // it navigated the page the chat is on.
    params.set({ m: id }, { push: true });
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{adding ? "Add a match" : "Choose a match"}</DialogTitle>
        </DialogHeader>

        {adding ? (
          <UploadCard
            onDone={() => {
              setAdding(false);
              setOpen(false);
            }}
          />
        ) : (
          <div className="grid gap-3">
            {(matches?.length ?? 0) > 5 && (
              <div className="relative">
                <Search className="text-muted-foreground pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2" />
                <Input
                  value={q}
                  onChange={(e) => setQ(e.target.value)}
                  placeholder="Search"
                  autoFocus
                  className="h-9 pl-8"
                />
              </div>
            )}

            <div className="-mx-1 max-h-80 overflow-y-auto px-1">
              {matches === null ? (
                <p className="text-muted-foreground py-6 text-center text-sm">Loading…</p>
              ) : shown.length === 0 ? (
                <p className="text-muted-foreground py-6 text-center text-sm">
                  {q ? "No match by that name." : "No matches yet."}
                </p>
              ) : (
                <div className="grid gap-0.5">
                  {shown.map((m) => {
                    const view = statusOf(m);
                    return (
                      <button
                        key={m.id}
                        type="button"
                        onClick={() => choose(m.id)}
                        className={cn(
                          "flex items-center gap-2 rounded-md px-2 py-2 text-left text-sm transition-colors",
                          m.id === current ? "bg-muted" : "hover:bg-muted/60",
                        )}
                      >
                        <Check className={cn("size-4 shrink-0", m.id === current ? "opacity-100" : "opacity-0")} />
                        <span className="min-w-0 flex-1 truncate">{matchTitle(m)}</span>
                        <span className="text-muted-foreground shrink-0 text-xs">
                          {matchDate(m) ?? ""}
                        </span>
                        {/* The status, so nobody picks a match that cannot yet
                            answer the question they opened this to ask. */}
                        <span className={cn("shrink-0 text-xs", TONE_CLASS[view.tone])}>
                          {view.label}
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            <Button variant="outline" size="sm" className="w-full" onClick={() => setAdding(true)}>
              <Plus className="size-4" /> Add a match
            </Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

/** The gate itself: a line saying what is missing, and the button that fixes
 *  it without leaving the page. */
export function PickMatchGate({
  title = "No match selected",
  detail = "Choose one to fill this screen in.",
  label = "Choose a match",
}: {
  title?: string;
  detail?: string;
  label?: string;
}) {
  return (
    <div className="grid place-items-center px-6 py-12">
      <div className="grid max-w-sm justify-items-center gap-3 text-center">
        <h2 className="text-sm font-medium">{title}</h2>
        <p className="text-muted-foreground text-sm">{detail}</p>
        <MatchPicker trigger={<Button size="sm">{label}</Button>} />
      </div>
    </div>
  );
}
