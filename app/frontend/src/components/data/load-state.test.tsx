import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { LoadEmpty, LoadFailure, mayExplainAbsence } from "./load-state";

/** THE HONESTY GATE.
 *
 *  This is the most load-bearing test on the Data surface, and it exists
 *  because of a real defect that shipped: the players table drops an em dash
 *  for any player with no confirmed tracking, and a caption underneath told the
 *  coach exactly what that meant — "an em dash means no track has been
 *  confirmed yet, not a zero". The code loading those numbers destructured
 *  `{ season }` off `useSeason` and discarded the `error`.
 *
 *  So a failed request produced a table of em dashes under a footnote making a
 *  specific, confident, WRONG claim about the squad. Worse than a blank screen:
 *  a blank screen tells you nothing, this told you something false and then
 *  explained it.
 *
 *  What is asserted here is the rule, not the wording: a sentence that
 *  INTERPRETS an absent number may only be shown when the source of those
 *  numbers actually arrived.
 */
describe("mayExplainAbsence", () => {
  it("permits the explanation when the source loaded", () => {
    expect(mayExplainAbsence(null)).toBe(true);
    expect(mayExplainAbsence(undefined)).toBe(true);
  });

  it("CRITICAL forbids it the moment the source failed", () => {
    expect(mayExplainAbsence("Failed to fetch")).toBe(false);
    expect(mayExplainAbsence("500 Internal Server Error")).toBe(false);
  });

  it("treats any non-empty error as a failure, whatever it says", () => {
    // The gate must not try to classify errors. Every one of them means the
    // same thing here: we do not know, so we do not get to explain.
    for (const e of ["x", "NetworkError", "aborted", "  "]) {
      expect(mayExplainAbsence(e)).toBe(false);
    }
  });
});

describe("LoadFailure", () => {
  it("names what failed and says the absence is a connection problem", () => {
    render(<LoadFailure what="the measured season" error="Failed to fetch" />);
    expect(screen.getByRole("alert")).toBeTruthy();
    const text = screen.getByRole("alert").textContent ?? "";
    expect(text).toContain("the measured season");
    // The load-bearing half: it must say the missing numbers are NOT a
    // statement about the squad.
    expect(text).toMatch(/not a statement about your squad/i);
    expect(text).toContain("Failed to fetch");
  });

  it("offers a retry only when there is something to retry", () => {
    const onRetry = vi.fn();
    const { rerender } = render(
      <LoadFailure what="the club" error="boom" onRetry={onRetry} />,
    );
    screen.getByRole("button", { name: /retry/i }).click();
    expect(onRetry).toHaveBeenCalledOnce();

    rerender(<LoadFailure what="the club" error="boom" />);
    expect(screen.queryByRole("button", { name: /retry/i })).toBeNull();
  });
});

describe("LoadEmpty", () => {
  it("says what could not load, and shows the reason when there is one", () => {
    render(<LoadEmpty what="the match list" error="503" />);
    expect(screen.getByText(/could not load the match list/i)).toBeTruthy();
    expect(screen.getByText("503")).toBeTruthy();
  });

  it("renders without an error string", () => {
    render(<LoadEmpty what="the club" />);
    expect(screen.getByText(/could not load the club/i)).toBeTruthy();
  });
});
