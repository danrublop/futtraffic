import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { SNAP_THRESHOLD } from "./constants";
import { usePlayheadDrag } from "./use-playhead-drag";

const GUTTER = 64;
/** 1 pixel per second, so a pixel offset reads directly as a timecode. */
const PPS = 1;

const fakeBox = () =>
  ({
    current: { getBoundingClientRect: () => ({ left: 0, width: 764 }) },
  }) as unknown as React.RefObject<HTMLDivElement | null>;

const setup = (opts: { scrollX?: number; snapTimes?: number[] } = {}) => {
  const onSeek = vi.fn();
  const { result } = renderHook(() =>
    usePlayheadDrag({
      pps: PPS,
      scrollX: opts.scrollX ?? 0,
      containerRef: fakeBox(),
      snapTimes: opts.snapTimes ?? [],
      onSeek,
      leftOffset: GUTTER,
    }),
  );
  /** A pointerdown at a given clientX, with the fields the hook actually uses. */
  const press = (clientX: number) =>
    act(() => {
      result.current.onPointerDown({
        clientX,
        preventDefault() {},
      } as unknown as React.PointerEvent);
    });
  return { result, onSeek, press };
};

describe("usePlayheadDrag", () => {
  it("reads a press as the time under the pointer, past the lane gutter", () => {
    const { onSeek, press } = setup();
    // 64px of gutter is not timeline, so x=364 is 300s in.
    press(GUTTER + 300);
    expect(onSeek).toHaveBeenCalledWith(300);
  });

  it("adds the scroll offset, so a scrolled timeline still seeks where you point", () => {
    const { onSeek, press } = setup({ scrollX: 1200 });
    press(GUTTER + 300);
    expect(onSeek).toHaveBeenCalledWith(1500);
  });

  it("never seeks before the start of the match", () => {
    const { onSeek, press } = setup();
    press(0);
    expect(onSeek).toHaveBeenCalledWith(0);
  });

  it("snaps onto a moment you land near, and reports it exactly", () => {
    const { onSeek, press } = setup({ snapTimes: [573] });
    // 3px off the play at 9:33 — inside the threshold, so it lands on 573.000,
    // not 576. Aiming finer than a pixel is not a thing anyone can do.
    press(GUTTER + 573 + 3);
    expect(onSeek).toHaveBeenCalledWith(573);
  });

  it("leaves a press outside the snap threshold where the pointer put it", () => {
    const { onSeek, press } = setup({ snapTimes: [573] });
    const far = SNAP_THRESHOLD + 3;
    press(GUTTER + 573 + far);
    expect(onSeek).toHaveBeenCalledWith(573 + far);
  });

  it("reports that a drag is in progress, and stops on pointerup", () => {
    const { result, press } = setup();
    expect(result.current.isDragging).toBe(false);
    press(GUTTER + 100);
    expect(result.current.isDragging).toBe(true);
    // The hook listens on window, not the element: a scrub that leaves the
    // timeline still has to end when the button comes up anywhere.
    act(() => {
      window.dispatchEvent(new Event("pointerup"));
    });
    expect(result.current.isDragging).toBe(false);
  });
});
