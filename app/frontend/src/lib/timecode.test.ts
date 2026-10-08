import { describe, expect, it } from "vitest";

import { clock, clockTenths } from "./timecode";

describe("clock", () => {
  it("pads the seconds and does not pad the minutes", () => {
    expect(clock(0)).toBe("0:00");
    expect(clock(9)).toBe("0:09");
    expect(clock(64)).toBe("1:04");
    expect(clock(2871.28)).toBe("47:51");
  });

  it("runs past an hour without an hours column", () => {
    // Deliberate: a football clock is 90 minutes and counts up. The ruler's
    // formatTimecode is the one that grows an hours field.
    expect(clock(3600)).toBe("60:00");
    expect(clock(5430)).toBe("90:30");
  });

  it("keeps a negative time readable, which the copy it replaced did not", () => {
    expect(clock(-90)).toBe("-1:30");
    expect(clock(-9)).toBe("-0:09");
  });
});

describe("clockTenths", () => {
  it("adds the tenth", () => {
    expect(clockTenths(0)).toBe("0:00.0");
    expect(clockTenths(64.35)).toBe("1:04.3");
  });

  it("truncates rather than rounds, so the clock never shows a time not reached", () => {
    expect(clockTenths(1.99)).toBe("0:01.9");
  });

  it("carries the sign onto the whole reading", () => {
    expect(clockTenths(-1.5)).toBe("-0:01.5");
  });
});
