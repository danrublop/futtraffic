import { act, renderHook } from "@testing-library/react";
import { beforeAll, describe, expect, it } from "vitest";

import { useTimelineZoom } from "./use-timeline-zoom";

/** jsdom has no ResizeObserver, and the hook measures its container in a layout
 *  effect. The stub never fires — the hook's synchronous `el.clientWidth` read
 *  is what these tests drive. */
beforeAll(() => {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
});

const DUR = 3600;
const GUTTER = 64;
const BOX = 764; // usable = 700 after the lane-name gutter

/** A container the hook can measure and attach a wheel listener to, without a
 *  real DOM: it only ever reads clientWidth and the bounding rect. */
const fakeBox = () =>
  ({
    current: {
      clientWidth: BOX,
      getBoundingClientRect: () => ({ left: 0, width: BOX }),
      addEventListener() {},
      removeEventListener() {},
    },
  }) as unknown as React.RefObject<HTMLDivElement | null>;

const setup = () => renderHook(() => useTimelineZoom(fakeBox(), DUR, GUTTER));

describe("useTimelineZoom", () => {
  it("opens fitted: the whole match across the usable width", () => {
    const { result } = setup();
    expect(result.current.usableWidth).toBe(BOX - GUTTER);
    expect(result.current.pps).toBeCloseTo(700 / DUR, 6);
    expect(result.current.fitted).toBe(true);
    // Fitted means nothing is off screen, so there is nowhere to scroll.
    expect(result.current.maxScrollX).toBe(0);
  });

  it("holds the time under the cursor fixed across a zoom", () => {
    const { result } = setup();
    const before = result.current.pps;
    // The centre of the view is 1800s at fit; it must still be 1800s after.
    const timeAtCentre = (result.current.scrollX + 350) / before;
    act(() => result.current.zoomIn());
    const after = result.current;
    expect(after.pps).toBeCloseTo(before * 1.5, 6);
    expect((after.scrollX + 350) / after.pps).toBeCloseTo(timeAtCentre, 4);
    // Zoomed in past fit, most of the match is now off screen.
    expect(after.maxScrollX).toBeCloseTo(DUR * after.pps - 700, 4);
  });

  it("refuses to zoom out past fit", () => {
    const { result } = setup();
    const fit = result.current.pps;
    act(() => result.current.zoomOut());
    expect(result.current.pps).toBeCloseTo(fit, 6);
  });

  it("clamps a scroll to the end of the timeline, and back to zero", () => {
    const { result } = setup();
    act(() => result.current.zoomIn());
    const max = result.current.maxScrollX;
    act(() => result.current.scrollTo(99999));
    expect(result.current.scrollX).toBeCloseTo(max, 4);
    act(() => result.current.scrollTo(-500));
    expect(result.current.scrollX).toBe(0);
  });

  /** The bug this guards: scrollX was corrected in an effect, so zooming back
   *  out rendered one frame scrolled past the end of a now-shorter timeline. */
  it("clamps scroll on read when a zoom-out shrinks the scrollable range", () => {
    const { result } = setup();
    act(() => result.current.zoomIn());
    act(() => result.current.zoomIn());
    act(() => result.current.scrollTo(result.current.maxScrollX));
    expect(result.current.scrollX).toBeGreaterThan(0);
    act(() => result.current.fitAll());
    // Fitted again: maxScrollX is 0, and scrollX must already agree — not on
    // the next render, now.
    expect(result.current.maxScrollX).toBe(0);
    expect(result.current.scrollX).toBe(0);
    expect(result.current.fitted).toBe(true);
  });

  it("pauses and resumes the playhead follow", () => {
    const { result } = setup();
    expect(result.current.followPaused).toBe(false);
    act(() => result.current.setFollowPaused(true));
    expect(result.current.followPaused).toBe(true);
    // Fitting is a deliberate "show me everything", so it takes the follow back.
    act(() => result.current.fitAll());
    expect(result.current.followPaused).toBe(false);
  });
});
