/** The ruler's tick arithmetic. */
import { describe, expect, it } from "vitest";

import { formatTimecode, getTickInterval } from "./time-ruler";

// The ruler's contract: at every zoom level, adjacent major labels stay at
// least ~60px apart (so they never overlap) and minor ticks at least ~8px
// apart (so they read as ticks, not a solid line). Premiere's behaviour: the
// pixel spacing stays constant; only the *interval* each tick represents
// changes with zoom.
//
// Without these guarantees the ruler renders crammed-together labels at
// moderate zoom — the bug that triggered the rewrite.
const MIN_MAJOR_PX = 60;
const MIN_MINOR_PX = 8;

describe("getTickInterval", () => {
  // A match is 45-120 minutes in a 300-1200px lane, so pps runs from about
  // 0.2 (fit a whole half) to 400 (a second across the panel).
  const PPS = [0.2, 0.5, 1, 2, 5, 10, 20, 30, 50, 80, 120, 200, 400];

  for (const pps of PPS) {
    it(`at pps=${pps}, majors stay >= ${MIN_MAJOR_PX}px apart`, () => {
      expect(getTickInterval(pps).major * pps).toBeGreaterThanOrEqual(MIN_MAJOR_PX);
    });

    it(`at pps=${pps}, minors stay >= ${MIN_MINOR_PX}px apart`, () => {
      expect(getTickInterval(pps).minor * pps).toBeGreaterThanOrEqual(MIN_MINOR_PX);
    });

    it(`at pps=${pps}, major is a clean multiple of minor`, () => {
      const { major, minor } = getTickInterval(pps);
      // Index-based ratio, matching the renderer, so sub-second intervals pass.
      const ratio = Math.round(major / minor);
      expect(Math.abs(major - ratio * minor)).toBeLessThan(1e-6);
      expect(ratio).toBeGreaterThanOrEqual(1);
    });
  }

  it("falls back to the coarsest interval when fully zoomed out", () => {
    // 0.005 pps x 3600s = 18px, below MIN_MAJOR_PX, so the picker settles on
    // the largest candidate rather than crashing or returning nothing.
    expect(getTickInterval(0.005).major).toBeGreaterThanOrEqual(3600);
  });

  it("reaches half-second ticks when a second spans the panel", () => {
    expect(getTickInterval(400).major).toBeLessThanOrEqual(0.5);
  });
});

describe("formatTimecode", () => {
  it("drops the hour until there is one, and never rounds a second up", () => {
    expect(formatTimecode(0)).toBe("0:00");
    expect(formatTimecode(573)).toBe("9:33");
    expect(formatTimecode(599.9)).toBe("9:59");
    expect(formatTimecode(3600)).toBe("1:00:00");
    expect(formatTimecode(3723)).toBe("1:02:03");
  });
});
