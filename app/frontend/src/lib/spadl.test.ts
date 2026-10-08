import { describe, expect, it } from "vitest";

import type { Pt } from "./pitch";
import {
  SPADL_LENGTH,
  SPADL_WIDTH,
  displayToMetres,
  flip,
  flipDisplay,
  fromSpadl,
  metresToDisplay,
  spadlToBoard,
  toSpadl,
} from "./spadl";

/** A real pitch that is NOT SPADL's, which is the whole point: 118 x 72 yd is
 *  what lib/pitch.ts defaults to and what the 4-6% warning is about. */
const REAL = { length: 118 * 0.9144, width: 72 * 0.9144 };
const SPADL = { length: SPADL_LENGTH, width: SPADL_WIDTH };

const close = (a: Pt, b: Pt, eps = 1e-9) => {
  expect(a[0]).toBeCloseTo(b[0], 9);
  expect(a[1]).toBeCloseTo(b[1], 9);
  void eps;
};

const SAMPLES: Pt[] = [
  [0, 0],
  [50, 50],
  [100, 100],
  [12.5, 87.5],
  [99.9, 0.1],
  [33.3, 66.6],
];

describe("the board's space and real metres", () => {
  // The y flip is the bug this module exists to stop being rediscovered:
  // display y counts DOWN from the top touchline (it is an SVG coordinate),
  // metres count UP from the bottom one.
  it("puts the top of the board at the far touchline, not the near one", () => {
    const [, y] = displayToMetres([50, 0], REAL);
    expect(y).toBeCloseTo(REAL.width, 9);
    expect(displayToMetres([50, 100], REAL)[1]).toBeCloseTo(0, 9);
  });

  it("round-trips every point", () => {
    for (const p of SAMPLES) close(metresToDisplay(displayToMetres(p, REAL), REAL), p);
  });

  it("maps the corners where they belong", () => {
    close(displayToMetres([0, 100], REAL), [0, 0]); // bottom-left
    close(displayToMetres([100, 0], REAL), [REAL.length, REAL.width]); // top-right
  });
});

describe("normalising to SPADL", () => {
  it("round-trips a real pitch through the fixed one", () => {
    for (const p of SAMPLES) {
      const metres = displayToMetres(p, REAL);
      close(fromSpadl(toSpadl(metres, REAL), REAL), metres);
    }
  });

  it("puts the centre spot on the centre spot", () => {
    close(toSpadl([REAL.length / 2, REAL.width / 2], REAL), [SPADL_LENGTH / 2, SPADL_WIDTH / 2]);
  });

  // The lossy part, stated as a test so nobody is surprised by it later: a
  // pitch that is not 105 x 68 has its distances rescaled by normalising, which
  // is SPADL's contract and the reason toSpadl has no default dimensions.
  it("rescales a pitch that is not 105 x 68, which is what the spec asks for", () => {
    expect(REAL.length).toBeGreaterThan(SPADL_LENGTH);
    const [x] = toSpadl([REAL.length, 0], REAL);
    expect(x).toBeCloseTo(SPADL_LENGTH, 9);
    // A 10 m run near one goal is not 10 SPADL metres on this pitch.
    const a = toSpadl([10, 0], REAL)[0];
    const b = toSpadl([20, 0], REAL)[0];
    expect(b - a).toBeLessThan(10);
  });

  it("is exact when the pitch already is SPADL's", () => {
    for (const p of SAMPLES) close(toSpadl(p, SPADL), p);
  });
});

describe("the other direction of play", () => {
  it("reflects through the centre spot rather than mirroring", () => {
    // A mirror would keep y. Turning a pitch round swaps left and right too.
    close(flip([0, 0], SPADL), [SPADL_LENGTH, SPADL_WIDTH]);
    close(flip([SPADL_LENGTH / 2, SPADL_WIDTH / 2], SPADL), [SPADL_LENGTH / 2, SPADL_WIDTH / 2]);
  });

  // Involutive on purpose: a double correction has to cancel, not compound.
  // Getting this wrong puts every mark in the mirror position, which does not
  // look broken - it looks like a team that attacked the wrong way.
  it("applied twice is identity", () => {
    for (const p of SAMPLES) {
      close(flip(flip(p, SPADL), SPADL), p);
      close(flipDisplay(flipDisplay(p)), p);
    }
  });
});

describe("an acting-team point onto the board", () => {
  it("leaves a home action attacking right", () => {
    // SPADL states everything left-to-right for whoever is acting, and the
    // board draws home to the right, so home needs no turn.
    const [x] = spadlToBoard([SPADL_LENGTH * 0.9, SPADL_WIDTH / 2], "home", SPADL);
    expect(x).toBeCloseTo(90, 9);
  });

  it("turns an away action round", () => {
    // The away team's own attacking third is the board's left.
    const [x] = spadlToBoard([SPADL_LENGTH * 0.9, SPADL_WIDTH / 2], "away", SPADL);
    expect(x).toBeCloseTo(10, 9);
  });

  it("puts both teams' halfway-line actions on the halfway line", () => {
    const mid: Pt = [SPADL_LENGTH / 2, SPADL_WIDTH / 2];
    close(spadlToBoard(mid, "home", SPADL), [50, 50]);
    close(spadlToBoard(mid, "away", SPADL), [50, 50]);
  });
});
