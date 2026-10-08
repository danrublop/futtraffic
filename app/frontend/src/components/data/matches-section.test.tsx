import { render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { SeasonTable } from "@/lib/season";
import type { Match } from "@/lib/types";

/** THE MATCHES SECTION, rendered.
 *
 *  The guarantee this file exists for cannot be proved by a pure test: that the
 *  section is POPULATED for a club with no entered feed. The old match table
 *  read `SeasonTable.fixtures`, which is feed-derived, and an academy filming
 *  its own games has none — so the section would have been permanently blank
 *  for the primary use case while looking perfect against the demo library,
 *  which is imported professional fixtures. The pure `matchRows` test covers
 *  the join; this covers the screen.
 *
 *  Everything the component reaches for is mocked at the module edge: the two
 *  hooks and next/navigation. Nothing here touches a network or a real router.
 */

const push = vi.fn();
let params = new URLSearchParams("v=data&m=old");

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push }),
  usePathname: () => "/workspace",
  useSearchParams: () => params,
}));

let hook: { matches: Match[] | null; error: string | null; refresh: () => void } = {
  matches: [],
  error: null,
  refresh: vi.fn(),
};
vi.mock("@/hooks/use-matches", () => ({ useMatches: () => hook }));

const { MatchesSection } = await import("./matches-section");

const job = (id: string, name: string, created: number): Match =>
  ({
    id,
    name,
    created,
    status: "done",
    progress: 1,
    error: null,
    waiting_for: null,
    upload_complete: true,
    has_match: true,
  }) as unknown as Match;

/** A library with no entered feed at all — the academy case. */
const NO_FEED: SeasonTable = { fixtures: [], clubs: [], players: [] };

const WITH_FEED: SeasonTable = {
  fixtures: [
    {
      id: "a",
      home: "Northbridge FC",
      away: "Eastvale FC",
      score: "2 - 1",
      date: "2025-10-26",
      competition: "LaLiga",
      round: null,
      demo: false,
    },
  ],
  clubs: [],
  players: [],
};

beforeEach(() => {
  push.mockClear();
  params = new URLSearchParams("v=data&m=old");
  hook = { matches: [], error: null, refresh: vi.fn() };
});

describe("with no entered feed at all", () => {
  it("CRITICAL still lists every match", () => {
    // The whole reason this section reads the job list. A fixture-derived
    // version renders nothing here, and nothing in the demo library would ever
    // have shown it.
    hook.matches = [job("a", "U15 v Rivals 2026-03-01", 300), job("b", "U15 v Town 2026-02-01", 200)];
    render(<MatchesSection table={NO_FEED} />);
    expect(screen.getByText("U15 v Rivals")).toBeTruthy();
    expect(screen.getByText("U15 v Town")).toBeTruthy();
  });

  it("shows an em dash for the feed columns rather than a gap", () => {
    hook.matches = [job("a", "U15 v Rivals 2026-03-01", 300)];
    render(<MatchesSection table={NO_FEED} />);
    const row = screen.getByText("U15 v Rivals").closest("tr")!;
    // Competition and Score both absent, and absent is a club without a feed,
    // which is the normal case rather than missing data.
    expect(within(row).getAllByText("—").length).toBeGreaterThanOrEqual(2);
  });

  it("offers no competition filter when nothing has a competition", () => {
    hook.matches = [job("a", "U15 v Rivals 2026-03-01", 300)];
    render(<MatchesSection table={NO_FEED} />);
    expect(screen.queryByText("Competition", { selector: "span" })).toBeNull();
  });
});

describe("with a feed", () => {
  it("enriches the row it has one for", () => {
    hook.matches = [job("a", "Northbridge FC v Eastvale FC 2025-10-26", 300)];
    render(<MatchesSection table={WITH_FEED} />);
    const row = screen.getByText("Northbridge FC v Eastvale FC").closest("tr")!;
    expect(within(row).getByText("LaLiga")).toBeTruthy();
    expect(within(row).getByText("2 - 1")).toBeTruthy();
  });

  it("does not invent a row for a fixture whose job is gone", () => {
    // The job list is the spine. A fixture with no job must not resurrect the
    // match. One real job is present so this is not passing merely because the
    // table rendered its empty state.
    hook.matches = [job("z", "U15 v Town 2026-02-01", 200)];
    render(<MatchesSection table={WITH_FEED} />);
    expect(screen.getByText("U15 v Town")).toBeTruthy();
    expect(screen.queryByText("Northbridge FC v Eastvale FC")).toBeNull();
  });
});

describe("clicking a match", () => {
  it("CRITICAL sets the match and keeps every other parameter", () => {
    hook.matches = [job("a", "U15 v Rivals 2026-03-01", 300)];
    render(<MatchesSection table={NO_FEED} />);
    screen.getByText("U15 v Rivals").closest("tr")!.click();

    expect(push).toHaveBeenCalledOnce();
    const q = new URLSearchParams(String(push.mock.calls[0][0]).split("?")[1]);
    expect(q.get("m")).toBe("a");
    // The bug this guards: replacing the query drops ?v= and bounces the
    // workspace to its default destination. It has shipped twice already.
    // The destination is Data now — the match surface it used to name is gone,
    // and `match` would fall through to chat.
    expect(q.get("v")).toBe("data");
  });
});

describe("empty and failed states", () => {
  it("says the library is empty, not that a filter matched nothing", () => {
    hook.matches = [];
    render(<MatchesSection table={NO_FEED} />);
    expect(screen.getByText(/no matches yet/i)).toBeTruthy();
  });

  it("does not show the empty-library copy when the library is not empty", () => {
    // Narrow: the two states have DIFFERENT copy, and this asserts only that
    // the empty-library sentence is absent when rows exist. Driving the Radix
    // select to reach the filtered-to-nothing sentence is not covered here.
    hook.matches = [job("a", "U15 v Rivals 2026-03-01", 300)];
    render(<MatchesSection table={NO_FEED} />);
    expect(screen.queryByText(/no matches yet/i)).toBeNull();
    expect(screen.getByText("U15 v Rivals")).toBeTruthy();
  });

  it("reports a failed match list instead of rendering an empty one", () => {
    hook = { matches: null, error: "Failed to fetch", refresh: vi.fn() };
    render(<MatchesSection table={NO_FEED} />);
    const alert = screen.getByRole("alert");
    expect(alert.textContent).toContain("the match list");
    expect(alert.textContent).toMatch(/not a statement about your squad/i);
  });
});
