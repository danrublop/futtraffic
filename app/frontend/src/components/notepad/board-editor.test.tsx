import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

// The editor reads the open match to offer "average positions"; none of that
// is what this file is about.
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  usePathname: () => "/workspace",
  useSearchParams: () => new URLSearchParams(),
}));

import { BoardEditor } from "./board-editor";
import { DEFAULT_MAGNETS } from "./pitch-data";

/** WHERE THE EDITOR OPENS. It used to swallow the window; it now fills the
 *  note's pane so the app's header and rail stay on screen. Two things carry
 *  that — the portal's target and `absolute` vs `fixed` — and they have to
 *  agree: an `absolute` box in <body> is a board pinned to the top of a page
 *  that scrolls, and a `fixed` one inside the pane covers the app anyway. */
const board = { magnets: DEFAULT_MAGNETS, shapes: [], view: "full", orient: "h", lines: "standard", grass: "green", stripes: false } as never;

// The portal root: the only `z-50` box, found from a control inside it.
const shell = () =>
  screen.getByRole("navigation", { name: "Tools" }).closest("div.z-50") as HTMLElement;

describe("BoardEditor placement", () => {
  it("fills the host it is given", () => {
    const host = document.createElement("div");
    host.className = "ft-note-pane";
    document.body.appendChild(host);

    render(<BoardEditor board={board} caption="" host={host} onSave={() => {}} onCancel={() => {}} />);

    expect(shell().parentElement).toBe(host);
    expect(shell().className).toContain("absolute");
    expect(shell().className).not.toContain("fixed");
  });

  it("falls back to the window when there is no pane", () => {
    render(<BoardEditor board={board} caption="" onSave={() => {}} onCancel={() => {}} />);

    expect(shell().parentElement).toBe(document.body);
    expect(shell().className).toContain("fixed");
  });
});
