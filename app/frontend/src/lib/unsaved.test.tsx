import { afterEach, describe, expect, it, vi } from "vitest";

import { park, unpark } from "./unsaved";

afterEach(() => {
  sessionStorage.clear();
  vi.restoreAllMocks();
});

describe("text a write could not land", () => {
  it("comes back on the next open of the same note", () => {
    park("Scouting/Rivals.html", "<p>the last thing typed</p>");
    expect(unpark("Scouting/Rivals.html")).toBe("<p>the last thing typed</p>");
  });

  it("comes back once, not every time", () => {
    // Otherwise a note that recovered and then saved normally would keep
    // offering the same rescue, and the coach learns to dismiss it.
    park("A.html", "<p>x</p>");
    expect(unpark("A.html")).toBe("<p>x</p>");
    expect(unpark("A.html")).toBeNull();
  });

  it("does not leak between notes", () => {
    park("A.html", "<p>a</p>");
    expect(unpark("B.html")).toBeNull();
    expect(unpark("A.html")).toBe("<p>a</p>");
  });

  it("is null when nothing was ever parked", () => {
    expect(unpark("Never.html")).toBeNull();
  });

  // Storage throws outright in private mode and when the quota is gone. Losing
  // the crash mat is bad; taking the note switch down with it is worse.
  it("survives a storage that throws on write", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("QuotaExceededError");
    });
    expect(() => park("A.html", "<p>a</p>")).not.toThrow();
  });

  it("survives a storage that throws on read", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("SecurityError");
    });
    expect(unpark("A.html")).toBeNull();
  });

  it("keeps an empty document distinct from nothing parked", () => {
    // "" is a real document someone emptied. Returning null for it would be a
    // silent refusal to restore exactly the edit most likely to be a mistake.
    park("A.html", "");
    expect(unpark("A.html")).toBe("");
  });
});
