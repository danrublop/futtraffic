import { describe, expect, it } from "vitest";

import { percentileOf, tierForMetric, toChartRows, weakestTier } from "./query-rows";

describe("where a value sits in a population", () => {
  const pop = [1, 2, 3, 4, 5];

  it("counts the share at or below, so the best is 100", () => {
    expect(percentileOf(pop, 5)).toBe(100);
    expect(percentileOf(pop, 3)).toBe(60);
    expect(percentileOf(pop, 1)).toBe(20);
  });

  it("puts a value below everyone at zero", () => {
    expect(percentileOf(pop, 0)).toBe(0);
  });

  // Splitting equal values by row order would invent a ranking out of whatever
  // order SQLite happened to return.
  it("gives tied values the same place", () => {
    expect(percentileOf([10, 10, 10], 10)).toBe(100);
    expect(percentileOf([5, 10, 10, 10], 10)).toBe(100);
    expect(percentileOf([5, 10, 10, 20], 10)).toBe(75);
  });

  it("is 100 against a population of one, not undefined", () => {
    expect(percentileOf([7], 7)).toBe(100);
  });

  it("ignores nulls rather than counting them as zero", () => {
    // A metric a player has no row for must not drag everyone else up.
    expect(percentileOf([1, NaN, 3], 3)).toBe(100);
    expect(percentileOf([], 3)).toBe(0);
  });
});

describe("SQL results into a figure's rows", () => {
  it("takes the first column as the label and the rest as numbers", () => {
    const r = toChartRows(
      ["zone", "shots", "xg"],
      [
        ["Six-yard", 4, 1.9],
        ["Outside", 7, 0.4],
      ],
    );
    expect(r.rows).toEqual([
      { label: "Six-yard", values: [4, 1.9] },
      { label: "Outside", values: [7, 0.4] },
    ]);
    expect(r.series).toEqual(["shots", "xg"]);
    expect(r.max).toBe(7);
  });

  it("refuses a result that has no number column", () => {
    // Better here than three layers down as a chart with no height anywhere.
    expect(() => toChartRows(["player"], [["Varga"]])).toThrow(/label column/);
  });

  // A SUM over no matching rows is NULL, not 0, and Number(null) is 0 but
  // Number("n/a") is NaN. NaN propagates silently through a d3 scale and puts a
  // mark somewhere arbitrary, so a chart that is quietly wrong comes out. Zero
  // is visibly nothing, which is the honest failure.
  it("turns a null or non-numeric cell into zero rather than NaN", () => {
    const r = toChartRows(["m", "v"], [["a", null], ["b", "n/a"], ["c", undefined]]);
    expect(r.rows.map((x) => x.values[0])).toEqual([0, 0, 0]);
    expect(r.max).toBe(0);
  });

  it("keeps a real zero distinct from a missing label", () => {
    const r = toChartRows(["m", "v"], [[null, 0]]);
    expect(r.rows[0]).toEqual({ label: "", values: [0] });
  });

  it("survives an empty result", () => {
    const r = toChartRows(["m", "v"], []);
    expect(r.rows).toEqual([]);
    expect(r.max).toBe(0);
  });
});

describe("the honesty tier a figure inherits", () => {
  it("reads minutes as not evaluated, because nothing has evaluated it", () => {
    // This test used to assert "ok". report.ts retracted that on purpose: the
    // four `ok` rows were stale and read ok while the backend shipped
    // not_evaluated. A test pinning the old answer would pin a false claim.
    expect(tierForMetric("minutes")).toBe("not_evaluated");
  });

  it("marks distance provisional and sprints withheld, because they are", () => {
    expect(tierForMetric("distance_km")).toBe("prov");
    expect(tierForMetric("sprints")).toBe("withheld");
  });

  it("falls back to not_evaluated for a metric nobody has reasoned about", () => {
    // Not provisional: provisional is itself a judgement someone made, and an
    // invented metric has had none.
    expect(tierForMetric("some_metric_invented_today")).toBe("not_evaluated");
  });

  it("lets the weakest number govern a figure that mixes them", () => {
    expect(weakestTier(["ok", "prov"])).toBe("prov");
    expect(weakestTier(["ok", "rel"])).toBe("rel");
    expect(weakestTier(["ok", "ok"])).toBe("ok");
    expect(weakestTier([])).toBe("not_evaluated");
  });

  it("puts not_evaluated below everything, in either order", () => {
    // The bug this pins: a local three-tier table with no not_evaluated key
    // made the answer depend on argument order - [not_evaluated, prov] gave
    // not_evaluated and [prov, not_evaluated] gave prov.
    expect(weakestTier(["prov", "not_evaluated"])).toBe("not_evaluated");
    expect(weakestTier(["not_evaluated", "prov"])).toBe("not_evaluated");
    expect(weakestTier(["ok", "not_evaluated", "rel"])).toBe("not_evaluated");
  });
});
