import { describe, expect, it } from "vitest";

import { fitCount } from "./toolbar";

/** Four 30px controls, and the `…` button costs 32. */
const W = [30, 30, 30, 30];

describe("fitCount", () => {
  it("keeps everything when the row is wide enough", () => {
    expect(fitCount(W, 120)).toBe(4);
    expect(fitCount(W, 500)).toBe(4);
  });

  it("leaves room for the overflow button once anything is dropped", () => {
    // 119px holds three at 90 — but not once 32 goes to the `…`.
    expect(fitCount(W, 119)).toBe(2);
    expect(fitCount(W, 152)).toBe(4); // exactly four, so no button is needed
  });

  it("drops to none rather than overflowing", () => {
    expect(fitCount(W, 20)).toBe(0);
    expect(fitCount([], 100)).toBe(0);
  });
});
