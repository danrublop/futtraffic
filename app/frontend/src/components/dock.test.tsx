import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DOCK_DEFAULT, useDock } from "@/lib/dock";

import { Dock } from "./dock";
import { useSurfaceParams } from "./surface-params";

const router = { push: vi.fn(), replace: vi.fn() };
vi.mock("next/navigation", () => ({
  useRouter: () => router,
  usePathname: () => "/workspace",
  useSearchParams: () => new URLSearchParams("v=chat&m=page-match"),
}));
vi.mock("@/hooks/use-matches", () => ({ useMatches: () => ({ matches: [] }) }));

/** Stand-ins for the three surfaces: each shows what it READ, and offers the
 *  write a real surface makes (picking a match, picking a note). */
function Probe({ name }: { name: string }) {
  const p = useSurfaceParams();
  return (
    <div data-testid={`surface-${name}`}>
      {name} m={p.get("m") ?? "-"} n={p.get("n") ?? "-"}
      <button onClick={() => p.set({ m: "picked" }, { push: true })}>pick match</button>
      <button onClick={() => p.set({ n: "club/press.html" })}>pick note</button>
    </div>
  );
}
vi.mock("@/components/video-panel", () => ({ VideoPanel: () => <Probe name="film" /> }));
vi.mock("@/components/doc-panel", () => ({ DocPanel: () => <Probe name="note" /> }));
vi.mock("@/components/data/data-browser", () => ({ DataBrowser: () => <Probe name="data" /> }));

const dock = () => useDock.getState();

beforeEach(() => {
  router.push.mockReset();
  router.replace.mockReset();
  useDock.setState({ tabs: [], active: null, open: false, width: DOCK_DEFAULT, seek: null });
});

describe("Dock", () => {
  it("draws nothing with no tabs", () => {
    const { container } = render(<Dock />);
    expect(container.innerHTML).toBe("");
  });

  it("mounts ONLY the active tab, and each reads its own params, not the page's", () => {
    act(() => {
      dock().openTab("film", { m: "m1" });
      dock().openTab("note", { n: "club/a.html" });
    });
    render(<Dock />);
    expect(screen.queryByTestId("surface-film")).toBeNull();
    expect(screen.getByTestId("surface-note").textContent).toContain("n=club/a.html");
    expect(screen.getByTestId("surface-note").textContent).toContain("m=-");

    fireEvent.click(screen.getByTitle("Film"));
    expect(screen.queryByTestId("surface-note")).toBeNull();
    expect(screen.getByTestId("surface-film").textContent).toContain("m=m1");
  });

  it("a surface's write lands on its tab and never navigates the page", () => {
    act(() => {
      dock().openTab("film");
    });
    render(<Dock />);
    fireEvent.click(screen.getByText("pick match"));
    expect(router.push).not.toHaveBeenCalled();
    expect(router.replace).not.toHaveBeenCalled();
    expect(dock().tabs[0].params).toEqual({ m: "picked" });
    expect(screen.getByTestId("surface-film").textContent).toContain("m=picked");
  });

  it("picking a note that is open in another tab goes to that tab", () => {
    act(() => {
      dock().openTab("note", { n: "club/press.html" });
      dock().openTab("note");
    });
    render(<Dock />);
    fireEvent.click(screen.getByText("pick note"));
    expect(dock().tabs).toHaveLength(1);
    expect(screen.getAllByTestId("surface-note")).toHaveLength(1);
  });

  it("closing and hiding", () => {
    act(() => {
      dock().openTab("data");
      dock().openTab("film", { m: "m1" });
    });
    render(<Dock />);
    fireEvent.click(screen.getByLabelText("Close Film"));
    expect(screen.getByTestId("surface-data")).toBeTruthy();
    fireEvent.click(screen.getByLabelText("Hide the dock"));
    expect(screen.queryByTestId("surface-data")).toBeNull();
    expect(dock().tabs).toHaveLength(1);
  });
});

describe("useSurfaceParams on a page", () => {
  function PageProbe() {
    const p = useSurfaceParams();
    return (
      <>
        <span>m={p.get("m")}</span>
        <button onClick={() => p.set({ m: "m2", lib: null }, { push: true })}>go</button>
        <button onClick={() => p.set({ n: "x.html" })}>browse</button>
      </>
    );
  }

  it("reads the URL and MERGES into it - the whole query is never replaced", () => {
    render(<PageProbe />);
    expect(screen.getByText("m=page-match")).toBeTruthy();
    fireEvent.click(screen.getByText("go"));
    expect(router.push).toHaveBeenCalledWith("/workspace?v=chat&m=m2");
    fireEvent.click(screen.getByText("browse"));
    expect(router.replace).toHaveBeenCalledWith("/workspace?v=chat&m=page-match&n=x.html", { scroll: false });
  });
});
