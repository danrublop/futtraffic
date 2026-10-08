import { describe, expect, it } from "vitest";

import { matchDate, matchTitle, progressOf, statusOf, matchSubtitle } from "./status";
import type { JobStatus, Match } from "./types";

/** A finished upload of a queued job. Every test starts here and changes the
 *  one field it is about, so a failure names the field. */
const match = (over: Partial<Match> = {}): Match => ({
  id: "m1",
  name: "Derby 2026-08-04",
  status: "queued",
  progress: 0,
  error: null,
  waiting_for: null,
  created: 0,
  updated: 0,
  bytes: 100,
  total_bytes: 100,
  uploaded_bytes: 100,
  upload_complete: true,
  has_match: false,
  has_tracks: false,
  has_calibration: false,
  calibration_inherited_from: null,
  frame_shape: null,
  ...over,
});

describe("statusOf", () => {
  // Label / tone / live for every JobStatus. The action is asserted per case
  // below, because "exactly one action" is the property that matters.
  const cases: [JobStatus, string, string, boolean][] = [
    ["calibrating", "Needs setup", "blocked", false],
    ["queued", "Queued", "idle", true],
    ["tracking", "Finding players", "busy", true],
    ["identifying", "Matching players", "busy", true],
    ["done", "Ready", "ready", false],
    ["failed", "Stopped", "failed", false],
  ];

  it.each(cases)("%s reads as %s", (status, label, tone, live) => {
    const v = statusOf(match({ status }));
    expect(v.label).toBe(label);
    expect(v.tone).toBe(tone);
    expect(v.live).toBe(live);
    expect(v.detail.length).toBeGreaterThan(0);
  });

  it("offers no action while the pipeline is working", () => {
    expect(statusOf(match({ status: "queued" })).action).toBeNull();
    expect(statusOf(match({ status: "tracking" })).action).toBeNull();
    expect(statusOf(match({ status: "identifying" })).action).toBeNull();
  });

  it("sends an uncalibrated match to setup, a done one to review", () => {
    expect(statusOf(match({ status: "calibrating" })).action).toEqual({
      label: "Mark the pitch",
      href: "/setup?m=m1",
    });
    expect(statusOf(match({ status: "done" })).action).toEqual({
      label: "Watch",
      href: "/review?m=m1",
    });
  });

  it("escapes the match id in the action href", () => {
    const v = statusOf(match({ id: "a b/c", status: "done" }));
    expect(v.action?.href).toBe("/review?m=a%20b%2Fc");
  });

  it("offers a retry, not a link, when the job failed", () => {
    const v = statusOf(match({ status: "failed", error: "codec unsupported" }));
    expect(v.action).toEqual({ label: "Try again", kind: "retry" });
    expect(v.detail).toBe("codec unsupported");
  });

  it("falls back to a plain sentence when a failure carries no error", () => {
    const v = statusOf(match({ status: "failed", error: null }));
    expect(v.detail).toBe("Something went wrong and it stopped.");
    // An empty string from the backend must not read as an empty card either.
    expect(statusOf(match({ status: "failed", error: "" })).detail).toBe(v.detail);
  });

  describe("incomplete upload", () => {
    it("wins over every status, including done", () => {
      for (const status of cases.map(([s]) => s)) {
        const v = statusOf(match({ status, upload_complete: false, uploaded_bytes: 5e8 }));
        expect(v.label).toBe("Upload unfinished");
        expect(v.tone).toBe("blocked");
        expect(v.live).toBe(false);
        expect(v.action).toEqual({ label: "Resume upload", href: "/" });
      }
    });

    it("names the bytes that arrived when the total is known", () => {
      const v = statusOf(
        match({ upload_complete: false, uploaded_bytes: 1.5e9, total_bytes: 3e9 }),
      );
      expect(v.detail).toBe(
        "1.50 of 3.00 GB arrived. Pick the same file again to carry on.",
      );
    });

    it("stays vague when the total was never recorded", () => {
      const v = statusOf(
        match({ upload_complete: false, uploaded_bytes: 1e9, total_bytes: null }),
      );
      expect(v.detail).toBe("The upload stopped before the whole file arrived.");
    });
  });
});

describe("progressOf", () => {
  it("reports upload progress while the file is still arriving", () => {
    expect(
      progressOf(match({ upload_complete: false, uploaded_bytes: 25, total_bytes: 100 })),
    ).toEqual({ value: 25, text: "25% uploaded" });
  });

  it("reports tracking progress once the upload is in", () => {
    expect(progressOf(match({ status: "tracking", progress: 0.42 }))).toEqual({
      value: 42,
      text: "42%",
    });
    expect(progressOf(match({ status: "identifying", progress: 1 }))).toEqual({
      value: 100,
      text: "100%",
    });
  });

  it("has nothing to show for queued, done, failed or calibrating", () => {
    for (const status of ["queued", "calibrating", "done", "failed"] as JobStatus[]) {
      expect(progressOf(match({ status, progress: 0.5 }))).toBeNull();
    }
  });

  it("shows no bar for a stalled upload of unknown size", () => {
    // No total means no denominator; a bar would have to invent one.
    expect(
      progressOf(match({ upload_complete: false, uploaded_bytes: 1e9, total_bytes: null })),
    ).toBeNull();
  });
});

describe("matchTitle", () => {
  it("strips the trailing date the uploader appends", () => {
    expect(matchTitle(match({ name: "Derby 2026-08-04" }))).toBe("Derby");
  });

  it("keeps a name that carries no date", () => {
    expect(matchTitle(match({ name: "Derby" }))).toBe("Derby");
  });

  it("keeps a date that sits in the middle of the name", () => {
    expect(matchTitle(match({ name: "Derby 2026-08-04 second half" }))).toBe(
      "Derby 2026-08-04 second half",
    );
  });

  it("falls back when the name is only a date, empty or null", () => {
    expect(matchTitle(match({ name: "2026-08-04" }))).toBe("Untitled match");
    expect(matchTitle(match({ name: "" }))).toBe("Untitled match");
    expect(matchTitle(match({ name: null }))).toBe("Untitled match");
  });
});

describe("matchDate", () => {
  it("reads the trailing date", () => {
    expect(matchDate(match({ name: "Derby 2026-08-04" }))).toBe("2026-08-04");
    expect(matchDate(match({ name: "2026-08-04" }))).toBe("2026-08-04");
  });

  it("ignores a date-like string in the middle of the name", () => {
    expect(matchDate(match({ name: "Derby 2026-08-04 second half" }))).toBeNull();
  });

  it("returns null when there is no date, or no name", () => {
    expect(matchDate(match({ name: "Derby" }))).toBeNull();
    expect(matchDate(match({ name: null }))).toBeNull();
  });

  it("agrees with matchTitle: what one strips, the other reports", () => {
    const m = match({ name: "Sunday league final 2026-08-04" });
    expect(matchTitle(m)).toBe("Sunday league final");
    expect(matchDate(m)).toBe("2026-08-04");
  });
});

/** The row shows a fixture headline AND a meta line. The meta line used to
 *  render match.name verbatim, which repeated the fixture above it and the date
 *  beside it: "Westport FC 3 - 1 Harbour City FC" over
 *  "2026-08-06 · Westport FC v Harbour City FC 2026-08-06". */
describe("matchSubtitle", () => {
  const m = (name: string) => ({ id: "x", name }) as never;

  it("drops a name that is only the fixture the headline already shows", () => {
    expect(matchSubtitle(m("Westport FC v Harbour City FC 2026-08-06"), "Westport FC", "Harbour City FC"))
      .toBeNull();
  });

  it("drops it whichever way the fixture is written", () => {
    for (const joiner of ["v", "vs", "versus", "-", "|"]) {
      expect(matchSubtitle(m(`Westport FC ${joiner} Harbour City FC`), "Westport FC", "Harbour City FC"))
        .toBeNull();
    }
  });

  it("keeps a name that carries something extra, minus the date", () => {
    expect(matchSubtitle(m("Westport FC v Harbour City FC Leagues Cup 2026-08-06"), "Westport FC", "Harbour City FC"))
      .toBe("Westport FC v Harbour City FC Leagues Cup");
  });

  it("keeps the name when there is no fixture to compare it against", () => {
    expect(matchSubtitle(m("Tuesday scrimmage 2026-08-06"))).toBe("Tuesday scrimmage");
  });

  it("is null for an unnamed upload rather than the placeholder", () => {
    expect(matchSubtitle(m(""))).toBeNull();
  });
});
