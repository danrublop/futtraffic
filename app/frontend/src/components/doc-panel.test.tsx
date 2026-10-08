import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api";

import { DocPanel } from "./doc-panel";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  usePathname: () => "/workspace",
  useSearchParams: () => new URLSearchParams("v=docs&n=a.html"),
}));
vi.mock("@/hooks/use-matches", () => ({ useReport: () => ({ report: null }) }));
// The editor reads `html` once, like the real one; a remount is the only way
// a new text reaches it.
vi.mock("@/components/notepad/notepad", async () => {
  const { useState } = await import("react");
  return {
    Notepad: ({ html, onChange, trail }: { html: string; onChange: (h: string) => void; trail: React.ReactNode }) => {
      const [first] = useState(html);
      return (
        <div>
          <div data-testid="doc">{first}</div>
          <button onClick={() => onChange("<p>mine</p>")}>type</button>
          {trail}
        </div>
      );
    },
  };
});

const api = vi.hoisted(() => ({
  notes: vi.fn(),
  getNote: vi.fn(),
  putNote: vi.fn(),
}));
vi.mock("@/lib/api", async (orig) => ({ ...(await orig<object>()), api }));

beforeEach(() => {
  api.notes.mockResolvedValue([{ path: "a.html", kind: "note" }]);
  api.getNote.mockResolvedValue({ html: "<p>old</p>", version: "v1" });
  api.putNote.mockReset();
});

describe("a save that loses to another writer", () => {
  it("stops autosaving and lets the reader load theirs", async () => {
    api.putNote.mockRejectedValue(new ApiError(409, "changed"));
    render(<DocPanel />);
    fireEvent.click(await screen.findByText("type"));
    await screen.findByText("Load theirs", {}, { timeout: 3000 });
    expect(api.putNote).toHaveBeenCalledTimes(1);

    api.getNote.mockResolvedValue({ html: "<p>theirs</p>", version: "v2" });
    fireEvent.click(screen.getByText("Load theirs"));
    await waitFor(() => expect(screen.getByTestId("doc").textContent).toBe("<p>theirs</p>"));
    expect(screen.queryByText("Load theirs")).toBeNull();
  });

  it("keep mine overwrites without the stale version", async () => {
    api.putNote.mockRejectedValueOnce(new ApiError(409, "changed"));
    api.putNote.mockResolvedValue({ ok: true, bytes: 1, version: "v3" });
    render(<DocPanel />);
    fireEvent.click(await screen.findByText("type"));
    fireEvent.click(await screen.findByText("Keep mine", {}, { timeout: 3000 }));
    await waitFor(() => expect(screen.queryByText("Keep mine")).toBeNull());
    expect(api.putNote).toHaveBeenLastCalledWith("a.html", "<p>mine</p>");
  });
});
