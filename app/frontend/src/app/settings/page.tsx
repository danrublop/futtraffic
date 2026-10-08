"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { ChevronRight, ShieldCheck, SlidersHorizontal } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { MatchPicker } from "@/components/match-picker";
import { Input } from "@/components/ui/input";
import { api } from "@/lib/api";

/** Where the once-per-match screens live. The sidebar's Guest row gear lands
 *  here; the items keep the same rules they had in the nav — Pitch setup is
 *  meaningless without a match selected, so it greys out instead of hiding. */
const ITEMS = [
  {
    href: "/setup",
    label: "Pitch setup",
    detail: "Mark the pitch corners and name the players.",
    icon: SlidersHorizontal,
    needsMatch: true,
  },
  {
    href: "/checks",
    label: "Accuracy",
    detail: "Whether to believe the numbers this match produces.",
    icon: ShieldCheck,
    needsMatch: false,
  },
] as const;

function SettingsList() {
  const match = useSearchParams().get("m");
  const withMatch = (href: string) =>
    match ? `${href}?m=${encodeURIComponent(match)}` : href;

  return (
    <div className="mx-auto grid w-full max-w-2xl min-w-0 gap-6">
      <div>
        <h1 className="text-[18px] font-semibold tracking-[-0.2px]">Settings</h1>
        <p className="text-muted-foreground text-sm">Signed in as Guest.</p>
      </div>

      <Card className="py-0">
        <CardContent className="divide-y p-0">
          {ITEMS.map(({ href, label, detail, icon: Icon, needsMatch }) => {
            const disabled = needsMatch && !match;
            const inner = (
              <>
                <Icon className="text-muted-foreground size-4 shrink-0" />
                <span className="min-w-0 flex-1">
                  <span className="block text-sm font-medium">{label}</span>
                  <span className="text-muted-foreground block truncate text-xs">
                    {disabled ? "Choose a match to set this up" : detail}
                  </span>
                </span>
                <ChevronRight className="text-muted-foreground size-4 shrink-0" />
              </>
            );
            // A row that needs a match opens the picker instead of greying
            // out — "pick a match first" is an instruction, and this is the
            // thing it was instructing you to go and do.
            return disabled ? (
              <MatchPicker
                key={href}
                trigger={
                  <button type="button" className="hover:bg-muted/50 flex w-full items-center gap-3 p-4 text-left transition-colors">
                    {inner}
                  </button>
                }
              />
            ) : (
              <Link
                key={href}
                href={withMatch(href)}
                className="hover:bg-muted/50 flex items-center gap-3 p-4 transition-colors"
              >
                {inner}
              </Link>
            );
          })}
        </CardContent>
      </Card>

      <ApiKeyCard />
    </div>
  );
}

/** The agent's API key.
 *
 *  The backend has had GET/PUT /api/settings since the agent shipped and
 *  nothing ever called them, so the only way to make the chat work was an
 *  environment variable set before launch — which, in a double-clicked desktop
 *  app, means there was no way at all. Every conversation failed with the
 *  backend's own developer-facing sentence about exporting a shell variable.
 */
function ApiKeyCard() {
  const [state, setState] = useState<{ has_key: boolean; source: string } | null>(null);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    api
      .getSettings()
      .then((s) => live && setState(s))
      .catch(() => live && setState({ has_key: false, source: "none" }));
    return () => {
      live = false;
    };
  }, []);

  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      setState(await api.putSettings(draft.trim()));
      setDraft("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  // An environment key wins on the server, so offering a field that cannot
  // take effect would be a lie.
  const fromEnv = state?.source === "env";

  return (
    <Card>
      <CardContent className="grid gap-3">
        <div>
          <h2 className="text-sm font-medium">Agent key</h2>
          <p className="text-muted-foreground text-xs">
            {state === null
              ? "Checking…"
              : fromEnv
                ? "Set by ANTHROPIC_API_KEY in this machine's environment."
                : state.has_key
                  ? "A key is saved on this machine. Enter a new one to replace it."
                  : "The chat needs an Anthropic API key. It is stored on this machine only."}
          </p>
        </div>

        {!fromEnv && (
          <div className="flex flex-col gap-2 sm:flex-row">
            <Input
              type="password"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="sk-ant-…"
              autoComplete="off"
              spellCheck={false}
              aria-label="Anthropic API key"
              className="h-11 font-mono text-xs sm:h-9"
            />
            <Button
              onClick={save}
              disabled={busy || !draft.trim()}
              className="h-11 shrink-0 sm:h-9"
            >
              {busy ? "Saving…" : "Save"}
            </Button>
          </div>
        )}

        {error && <p className="text-bad text-xs">{error}</p>}
        {state?.has_key && !error && (
          <p className="text-muted-foreground text-xs">Chat is ready.</p>
        )}
      </CardContent>
    </Card>
  );
}

export default function SettingsPage() {
  return (
    // useSearchParams must sit inside Suspense or the route opts into dynamic
    // rendering — same reason AppShell is wrapped in layout.tsx.
    <Suspense fallback={null}>
      <SettingsList />
    </Suspense>
  );
}
