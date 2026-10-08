"use client";

import { createContext, useContext, useMemo } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

/** Where a surface reads and writes ITS parameters — the match (`m`), the
 *  note (`n`), the data table (`tab`), the film library flag (`lib`), a track
 *  or a clip range.
 *
 *  On a page, that is the URL, exactly as before. Inside a dock tab it is the
 *  tab: two open notes cannot both be `?n=`, and a film opened beside the chat
 *  must not change the match the chat is about. So every surface and every
 *  hook under one (useMatch above all) asks this instead of next/navigation,
 *  and a dock tab wraps its surface in a provider that answers from its own
 *  state. ESLint refuses a `next/navigation` import under the surfaces, which
 *  is what keeps the next picker from quietly navigating the whole page.
 *
 *      page ────────────▶ (no provider) ─▶ URL: useSearchParams / router
 *      dock tab ────────▶ <SurfaceParamsProvider value={tab}> ─▶ dock store
 */
export interface SurfaceParams {
  get: (key: string) => string | null;
  /** Merge `patch` in; null removes a key. `push` records a history entry on
   *  a page (picking a film); the default replaces (browsing a list). */
  set: (patch: Record<string, string | null>, opts?: { push?: boolean }) => void;
  /** True inside a dock tab. A surface uses it only where "leave this page"
   *  makes no sense in a tab. */
  docked: boolean;
}

const Ctx = createContext<SurfaceParams | null>(null);

export function SurfaceParamsProvider({
  value,
  children,
}: {
  value: SurfaceParams;
  children: React.ReactNode;
}) {
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useSurfaceParams(): SurfaceParams {
  const own = useContext(Ctx);
  // Called unconditionally - hooks cannot be skipped - and ignored inside a tab.
  const search = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const url = useMemo<SurfaceParams>(
    () => ({
      docked: false,
      get: (key) => search.get(key),
      set: (patch, opts) => {
        const q = new URLSearchParams(search.toString());
        for (const [k, v] of Object.entries(patch)) {
          if (v == null || v === "") q.delete(k);
          else q.set(k, v);
        }
        const href = `${pathname}?${q.toString()}`;
        if (opts?.push) router.push(href);
        else router.replace(href, { scroll: false });
      },
    }),
    [search, router, pathname],
  );
  return own ?? url;
}
