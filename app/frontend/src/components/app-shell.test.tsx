import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { listChats } from "@/lib/agent";
import { useChatStore } from "@/lib/store";

import { ChatList } from "./app-shell";

const push = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push, replace: vi.fn() }),
  usePathname: () => "/workspace",
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("@/lib/agent", () => ({
  listChats: vi.fn(),
  patchChat: vi.fn(async () => ({})),
  deleteChat: vi.fn(async () => {}),
  getChat: vi.fn(async (id: string) => ({ id, team_id: "u15", messages: [] })),
  postRun: vi.fn(),
  stopRun: vi.fn(),
}));

const TEAMS = [
  { team_id: "u15", name: "U15 Boys" },
  { team_id: "first", name: "First team" },
];
let teams = TEAMS;
vi.mock("@/lib/club", () => ({
  useOrg: () => ({ org: { club: { name: "Club" }, teams, coaches: [] }, error: null }),
  useTeam: () => ["", vi.fn()],
  teamLabel: (t: { name: string }) => t.name,
}));

const chats = [
  { id: "a", title: "Pressing v Brighton", updated: "", team_id: "first" },
  { id: "b", title: "Loose chat", updated: "", team_id: "" },
  { id: "c", title: "U15 shape", updated: "", team_id: "u15" },
];

beforeEach(() => {
  teams = TEAMS;
  push.mockReset();
  vi.mocked(listChats).mockResolvedValue(chats);
  useChatStore.getState().newChat();
});

/** The sidebar files chats by squad. The grouping rule has its own test
 *  (lib/chat-folders.test.ts); this is what the coach clicks. */
describe("ChatList", () => {
  it("lists loose sessions first as plain rows, then squad folders in ladder order", async () => {
    const { container } = render(<ChatList withMatch={(h) => h} />);
    await screen.findByText("Pressing v Brighton");
    const labels = screen.getAllByRole("button", { expanded: true }).map((b) => b.textContent);
    expect(labels).toEqual(["U15 Boys1", "First team1"]);
    expect(screen.queryByText("No squad")).toBeNull();
    const text = container.textContent ?? "";
    expect(text.indexOf("Loose chat")).toBeLessThan(text.indexOf("U15 Boys"));
  });

  it("folding a squad hides its chats and nothing else", async () => {
    render(<ChatList withMatch={(h) => h} />);
    await screen.findByText("U15 shape");
    fireEvent.click(screen.getByRole("button", { name: "U15 Boys1", expanded: true }));
    expect(screen.queryByText("U15 shape")).toBeNull();
    expect(screen.getByText("Loose chat")).toBeTruthy();
  });

  it("the folder's + starts a new chat in that squad", async () => {
    render(<ChatList withMatch={(h) => h} />);
    await screen.findByText("U15 shape");
    fireEvent.click(screen.getByRole("button", { name: "New session in First team" }));
    expect(useChatStore.getState().teamId).toBe("first");
    expect(useChatStore.getState().chatId).toBeNull();
    expect(push).toHaveBeenCalledWith("/workspace?v=chat");
  });

  it("opening a chat loads it with its squad", async () => {
    render(<ChatList withMatch={(h) => h} />);
    fireEvent.click(await screen.findByText("U15 shape"));
    await waitFor(() => expect(useChatStore.getState().chatId).toBe("c"));
    expect(useChatStore.getState().teamId).toBe("u15");
  });

  it("a club with no squads keeps the flat list", async () => {
    teams = [];
    render(<ChatList withMatch={(h) => h} />);
    await screen.findByText("Loose chat");
    expect(screen.queryByText("No squad")).toBeNull();
    expect(screen.getAllByText(/Pressing|Loose|U15 shape/)).toHaveLength(3);
  });
});
