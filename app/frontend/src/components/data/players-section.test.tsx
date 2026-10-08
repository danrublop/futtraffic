import { describe, expect, it } from "vitest";

import { longDate } from "./player-fact";
import { enteredOf, group, humanise, inBox, resultOf } from "./players-section";
import type { Appearance, PlayerBio } from "@/lib/types";

/** A BIRTHDAY IS THE ONE FACT ON THE PLAYER CARD that is nothing but a date,
 *  so both ways of getting it wrong are worth pinning: the timezone shift this
 *  function exists to avoid, and a feed date that is not a real day. */
describe("longDate", () => {
  it("does not shift the day west of Greenwich", () => {
    // `new Date("2002-05-06")` is midnight UTC and prints as the 5th here.
    expect(longDate("2002-05-06")).toBe("May 6, 2002");
    expect(longDate("1999-01-01")).toBe("Jan 1, 1999");
    expect(longDate("1999-12-31")).toBe("Dec 31, 1999");
  });

  it("omits a date it cannot read rather than printing part of one", () => {
    for (const iso of [null, undefined, "", "6 May 2002", "2002-5-6"])
      expect(longDate(iso)).toBeNull();
  });

  it("refuses a month outside 1-12", () => {
    // These used to render the literal word "undefined" onto the card.
    expect(longDate("2002-13-40")).toBeNull();
    expect(longDate("2002-00-06")).toBeNull();
  });
});

/** FOUR RAILS OUT OF FREE TEXT. `roster.position` is typed by a coach and the
 *  feed writes its own words beside it, so this is a stem match with a
 *  fall-through - and the fall-through is the part worth pinning, because a
 *  wrong bucket is not an error anywhere: it is a badge in the wrong colour
 *  and a player sorted into the wrong block of the list. */
describe("group", () => {
  it("reads the feed's long words and the coach's short ones alike", () => {
    expect(group("Goalkeeper")).toBe("GK");
    expect(group("GK")).toBe("GK");
    expect(group("Defender")).toBe("DEF");
    expect(group("CB")).toBe("DEF");
    expect(group("Attacker")).toBe("ATT");
    expect(group("Forward")).toBe("ATT");
    expect(group("Striker")).toBe("ATT");
    expect(group("ST")).toBe("ATT");
  });

  it("is case-insensitive, because nothing upstream agrees on case", () => {
    expect(group("cb")).toBe("DEF");
    expect(group("lwb")).toBe("DEF");
    expect(group("RWB")).toBe("DEF");
    expect(group("lw")).toBe("ATT");
    expect(group("cf")).toBe("ATT");
  });

  it("falls through to MID rather than inventing a group for a typo", () => {
    // Midfield has no stem test at all - it is what is left. That is the
    // design, so an unknown word lands in the largest, least wrong bucket.
    expect(group("CM")).toBe("MID");
    expect(group("Midfielder")).toBe("MID");
    expect(group("Sweeper")).toBe("MID");
    expect(group("Utility")).toBe("MID");
    expect(group(null)).toBe("MID");
    expect(group(undefined)).toBe("MID");
    expect(group("")).toBe("MID");
  });

  it("matches a PREFIX, which is what a typo survives and a compound does not", () => {
    // The stem is forgiving at the end - "goalkeepr" is still a keeper - and
    // unforgiving at the front. "Defensive Midfield" is the price: it reads
    // DEF because it starts "def", though the player is a midfielder. Pinned
    // as the known cost of the rule, not endorsed; a fifth check for it would
    // have to come before the DEF one.
    expect(group("goalkeepr")).toBe("GK");
    expect(group("Defensive Midfield")).toBe("DEF");
  });
});

/** THE SEASON ROW'S ARITHMETIC. Every field here is summed over a run of
 *  matches and printed as one number, which is the shape of mistake this repo
 *  keeps making: a total that quietly dropped its hard rows reads exactly like
 *  a total that did not. The null rules are the ones with teeth. */
describe("enteredOf", () => {
  const app = (over: Partial<Appearance> = {}): Appearance => ({
    player_id: "p", match_id: "m", side: "home", name: "P",
    minutes: 90, rating: null, age: null, shirt: null, goals: null, assists: null,
    saves: null, conceded: null, competition: null, yellow: 0, red: 0,
    ...over,
  });

  it("counts an empty run as zeros and nulls, never NaN", () => {
    expect(enteredOf([])).toEqual({
      app: 0, mins: 0, goals: 0, assists: 0, rating: null,
      yellow: 0, red: 0, age: null, shirt: null, saves: null, cleanSheets: null,
    });
  });

  it("sums the counting columns and treats a missing one as zero", () => {
    const e = enteredOf([
      app({ minutes: 90, goals: 2, assists: 1, yellow: 1, red: 0 }),
      app({ minutes: null, goals: null, assists: 3, yellow: 0, red: 1 }),
    ]);
    expect(e.app).toBe(2);
    expect(e.mins).toBe(90);
    expect(e.goals).toBe(2);
    expect(e.assists).toBe(4);
    expect(e.yellow).toBe(1);
    expect(e.red).toBe(1);
  });

  it("averages ONLY the rated matches - an unrated one is not a 0", () => {
    // A zero-filled mean is the bug this guards: three good games and one the
    // provider never scored would read as a 5.0 season.
    expect(enteredOf([app({ rating: 8 }), app({ rating: 6 }), app({ rating: null })]).rating).toBe(7);
    expect(enteredOf([app({ rating: null })]).rating).toBeNull();
  });

  it("tells an outfielder from a keeper who made no saves", () => {
    // The documented null-vs-0 rule. `!r.saves` would file both as null.
    expect(enteredOf([app(), app()]).saves).toBeNull();
    expect(enteredOf([app({ saves: 0 }), app({ saves: 0 })]).saves).toBe(0);
    expect(enteredOf([app({ saves: 3 }), app({ saves: null })]).saves).toBe(3);
  });

  it("needs 60 minutes for a clean sheet, not just a zero in the conceded column", () => {
    // A real keeper's run, verbatim: the five-minute cameo in a game already lost
    // is the appearance that made this read five where the club kept four.
    const run = [
      app({ minutes: 90, conceded: 0 }),
      app({ minutes: 90, conceded: 0 }),
      app({ minutes: 5, conceded: 0 }),   // came on at the death: not his
      app({ minutes: 90, conceded: 2 }),
    ];
    expect(enteredOf(run).cleanSheets).toBe(2);
  });

  it("puts the threshold ON 60, and a null-minute row below it", () => {
    expect(enteredOf([app({ minutes: 60, conceded: 0 })]).cleanSheets).toBe(1);
    expect(enteredOf([app({ minutes: 59, conceded: 0 })]).cleanSheets).toBe(0);
    expect(enteredOf([app({ minutes: null, conceded: 0 })]).cleanSheets).toBe(0);
  });

  it("reports no clean-sheet column at all for a run with no conceded figures", () => {
    // null is "the feed filed no keeper metrics", which the panel draws as a
    // blank. 0 would claim he kept goal four times and let one in each.
    expect(enteredOf([app(), app()]).cleanSheets).toBeNull();
  });

  it("takes the HIGHEST age, because the rows are not in date order", () => {
    // Grouped by match id, so "the last one seen" is not "the latest". A
    // birthday mid-season is the case that separates the two.
    expect(enteredOf([app({ age: 24 }), app({ age: 23 }), app({ age: null })]).age).toBe(24);
    expect(enteredOf([app({ age: null })]).age).toBeNull();
  });

  it("takes the LAST shirt seen, which is a different rule from the age", () => {
    // Deliberately not a max: shirt 7 after shirt 23 is a demotion, not an
    // error, and the max would keep the number he no longer wears.
    expect(enteredOf([app({ shirt: 23 }), app({ shirt: 7 })]).shirt).toBe(7);
    expect(enteredOf([app({ shirt: 7 }), app({ shirt: null })]).shirt).toBe(7);
  });
});

/** The feed's wire words on a card. An unmapped value still has to read like
 *  English, because the next one the provider adds will not be in any list. */
describe("humanise", () => {
  it("splits the case boundary and lowers everything after the first word", () => {
    expect(humanise("RightFoot")).toBe("Right foot");
    expect(humanise("SetPiece")).toBe("Set piece");
    expect(humanise("OpenPlay")).toBe("Open play");
  });

  it("leaves a single word and a digit boundary alone-ish", () => {
    expect(humanise("Header")).toBe("Header");
    expect(humanise("Box12Yards")).toBe("Box12 yards");
  });

  it("is null for nothing, so the card omits the row instead of printing it empty", () => {
    expect(humanise(null)).toBeNull();
    expect(humanise(undefined)).toBeNull();
    expect(humanise("")).toBeNull();
  });
});

type Shot = NonNullable<PlayerBio["shots"]>[number];
const shot = (over: Partial<Shot>): Shot => ({ x: 100, y: 34, ...over } as Shot);

/** THE SHOT MAP'S TWO CLASSIFIERS, both of which print a total under the map.
 *  `resultOf` is an ordered fall-through and the order is the whole rule. */
describe("resultOf", () => {
  it("ranks goal over on_target - a goal is on target and must not read 'Saved'", () => {
    expect(resultOf(shot({ goal: true, on_target: true }))).toBe("Goal");
    expect(resultOf(shot({ goal: true, on_target: true, blocked: true }))).toBe("Goal");
  });

  it("names the three misses apart", () => {
    expect(resultOf(shot({ on_target: true }))).toBe("Saved");
    expect(resultOf(shot({ blocked: true }))).toBe("Blocked");
    expect(resultOf(shot({}))).toBe("Off target");
  });

  it("puts a blocked shot that was also on target under Saved, as the order says", () => {
    expect(resultOf(shot({ on_target: true, blocked: true }))).toBe("Saved");
  });
});

describe("inBox", () => {
  // 16.5 m out and 20.16 m either side of y=34: the 18-yard box in metres.
  it("holds a shot on the penalty spot and one in the six-yard box", () => {
    expect(inBox(shot({ x: 94, y: 34 }))).toBe(true);
    expect(inBox(shot({ x: 104, y: 30 }))).toBe(true);
  });

  it("is inclusive on both edges, which is where a box total is won or lost", () => {
    expect(inBox(shot({ x: 88.5, y: 34 }))).toBe(true);       // exactly 16.5 out
    expect(inBox(shot({ x: 88.49, y: 34 }))).toBe(false);
    expect(inBox(shot({ x: 100, y: 34 + 20.16 }))).toBe(true); // the corner
    expect(inBox(shot({ x: 100, y: 34 - 20.16 }))).toBe(true);
    expect(inBox(shot({ x: 100, y: 54.17 }))).toBe(false);
  });

  it("is symmetric about the centre, so neither wing is counted twice", () => {
    expect(inBox(shot({ x: 95, y: 20 }))).toBe(inBox(shot({ x: 95, y: 48 })));
  });

  it("refuses a shot from outside the area however central", () => {
    expect(inBox(shot({ x: 80, y: 34 }))).toBe(false);
  });
});
