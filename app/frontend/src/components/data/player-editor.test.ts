import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ageOf, canSave, draftOf, savePlayer } from "@/components/data/player-editor";
import { NATIONS, POSITIONS, nationCode, nationFlag } from "@/lib/player-facts";

const mock = vi.hoisted(() => ({
  saveRoster: vi.fn(),
  saveBio: vi.fn(),
  savePhoto: vi.fn(),
  clearStoreCache: vi.fn(),
  error: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  api: { saveRoster: mock.saveRoster, saveBio: mock.saveBio, savePhoto: mock.savePhoto },
  clearStoreCache: mock.clearStoreCache,
}));
vi.mock("sonner", () => ({ toast: { error: mock.error } }));

/** The form reads from TWO stores and writes back to both, so the thing worth
 *  a test is the join between them - not the inputs, which React owns. */
describe("draftOf", () => {
  it("prefers the bio, falls back to the registry row", () => {
    const d = draftOf(
      { id: "x", name: "Registry Name", position: "CB", teams: [{ team_id: "u15", team: "U15", number: 4, position: "CB" }], matches: 2, photo: false },
      { name: "Bio Name", height_cm: 180, shirt: 9 },
    );
    expect(d.name).toBe("Bio Name");
    expect(d.height_cm).toBe("180");
    expect(d.shirt).toBe("9");
    // Nothing in the bio, so the squad and position come off the roster row.
    expect(d.team).toBe("U15");
    // TRANSLATED IN. The roster stores the short code, the form and the bio
    // speak the long label - so the dropdown can select it. Before this, a
    // registry-only player opened with nothing ticked and the first save wrote
    // the label back to the roster, where group() files "Centre Back" as MID.
    expect(d.primary_position).toBe("Centre Back");
  });

  it("takes the SQUAD from the registry, never the bio's club", () => {
    // bio.team is what the FEED imported - the player's professional club -
    // and only counts as a squad here if this club has one by that name.
    // Seeding the form from it re-homed a player into a roster file for a
    // squad that does not exist, on a save of an unrelated field.
    const reg = { id: "x", name: "A", position: null, teams: [{ team_id: "u15", team: "U15", number: 4, position: null }], matches: 0, photo: false };
    expect(draftOf(reg, { team: "Northbridge FC" }).team).toBe("U15");
    // No registry row and the bio's club is not one of ours: blank, so the
    // operator has to choose. canSave() refuses to save until they do.
    expect(draftOf(null, { team: "Northbridge FC" }, ["U15", "U17"]).team).toBe("");
    // ...but a club that really does have a squad by that name keeps it.
    expect(draftOf(null, { team: "U17" }, ["U15", "U17"]).team).toBe("U17");
  });

  it("takes the registry's shirt when the bio has none", () => {
    const d = draftOf(
      { id: "x", name: "A", position: null, teams: [{ team_id: "u15", team: "U15", number: 4, position: null }], matches: 0, photo: false },
      {},
    );
    expect(d.shirt).toBe("4");
  });

  it("is all empty strings for a player who does not exist yet", () => {
    const d = draftOf(null, null);
    expect(d).toEqual({
      name: "",
      team: "",
      shirt: "",
      primary_position: "",
      other_positions: [],
      country: "",
      foot: "",
      height_cm: "",
      birth_date: "",
    });
  });

  it("lets the OPERATOR's other positions beat the feed's dots", () => {
    // This field is what the operator typed; `positions` is what the feed
    // observed. The other way round made the field write-only: the save
    // stored other_positions and every reader went on preferring the dots, so
    // the edit came back undone with no error anywhere.
    const d = draftOf(null, {
      positions: [
        { short: "DM", main: true },
        { short: "CM" },
        { short: "AM" },
      ],
      other_positions: ["LB"],
    });
    expect(d.other_positions).toEqual(["LB"]);
  });

  it("falls back to the feed's dots when the operator has typed none", () => {
    const d = draftOf(null, {
      positions: [{ short: "DM", main: true }, { short: "CM" }, { short: "AM" }],
    });
    expect(d.other_positions).toEqual(["CM", "AM"]);
  });

  it("normalises the feed's long labels to short codes", () => {
    // A former feed importer (not in this repository) wrote other_positions as LABELS
    // ([x.get("label") for x in nonPrimaryPositions]) while the checkboxes key
    // on shorts - so a feed player showed "Right Midfielder" in the cell and
    // ticked nothing, and the first save silently rewrote the alphabet.
    const d = draftOf(null, { other_positions: ["Right Midfielder", "Left Back"] });
    expect(d.other_positions).toEqual(["RM", "LB"]);
  });

  it("keeps free text it does not recognise rather than dropping it", () => {
    expect(draftOf(null, { other_positions: ["Libero"] }).other_positions).toEqual(["Libero"]);
  });

  // A 0 height is not a height, but `??` would have kept it and `||` would
  // have dropped a legitimate shirt number 0. Both edges, once.
  it("keeps shirt 0 and drops nothing else silently", () => {
    expect(draftOf(null, { shirt: 0 }).shirt).toBe("0");
  });
});

describe("nationCode", () => {
  it("matches on the name the panel shows, case and space insensitive", () => {
    expect(nationCode("Ecuador")).toBe("ECU");
    expect(nationCode("  netherlands ")).toBe("NED");
  });

  it("is FIFA, not ISO - the alphabet the feed writes", () => {
    expect(nationCode("Netherlands")).toBe("NED"); // ISO would be NLD
    expect(nationCode("Portugal")).toBe("POR"); // PRT
    expect(nationCode("England")).toBe("ENG"); // no ISO country at all
  });

  it("is null for something nobody picked", () => {
    expect(nationCode("")).toBeNull();
    expect(nationCode(null)).toBeNull();
    expect(nationCode("Wakanda")).toBeNull();
  });

  it("has no malformed rows: every entry is a 3-letter code, a flag key and a name", () => {
    for (const n of NATIONS) {
      expect(n.code).toMatch(/^[A-Z]{3}$/);
      // flag-icons files its SVGs under ISO alpha-2, plus four gb-* subdivisions.
      expect(n.iso).toMatch(/^([a-z]{2}|gb-(eng|sct|wls|nir))$/);
      expect(n.name.length).toBeGreaterThan(2);
    }
    expect(new Set(NATIONS.map((n) => n.code)).size).toBe(NATIONS.length);
    expect(new Set(NATIONS.map((n) => n.iso)).size).toBe(NATIONS.length);
  });
});

describe("nationFlag", () => {
  // The whole reason the table carries two alphabets. Getting this wrong puts
  // the Union Jack against an England international, or Ecuador's flag on a
  // player from a country the table does not know.
  it("gives each home nation its own flag, never GB", () => {
    expect(nationFlag("England")).toBe("gb-eng");
    expect(nationFlag("Scotland")).toBe("gb-sct");
    expect(nationFlag("Wales")).toBe("gb-wls");
    expect(nationFlag("Northern Ireland")).toBe("gb-nir");
  });

  it("maps the FIFA name to the ISO key where they differ", () => {
    expect(nationFlag("Netherlands")).toBe("nl"); // NED -> nl
    expect(nationFlag("South Korea")).toBe("kr"); // KOR -> kr
    expect(nationFlag("China PR")).toBe("cn");
    expect(nationFlag("Kosovo")).toBe("xk"); // not an ISO country; flag-icons has it
  });

  it("is null for an unknown country - draw nothing rather than the wrong flag", () => {
    expect(nationFlag("Wakanda")).toBeNull();
    expect(nationFlag(null)).toBeNull();
  });
});

/** A NAME AND A SQUAD, because the id is minted from the one and the roster
 *  file is keyed on the other. Everything else is allowed to be blank. */
describe("canSave", () => {
  const d = (name: string, team: string) => ({ ...draftOf(null, null), name, team });

  it("wants both, and whitespace is neither", () => {
    expect(canSave(d("Ada", "U15"))).toBe(true);
    expect(canSave(d("", "U15"))).toBe(false);
    expect(canSave(d("Ada", ""))).toBe(false);
    expect(canSave(d("   ", "U15"))).toBe(false);
    expect(canSave(d("\n\t", "  "))).toBe(false);
  });
});

/** THE THREE-STORE WRITE. This is the function the memory note is about
 *  (roster-player-id-is-the-season-join (internal note)): the player id is
 *  SERVER-MINTED, so the registry has to go first and its reply is the only
 *  thing allowed to name the player the other two writes are about. Guessing
 *  the slug locally forks one career into two profiles, silently. */
describe("savePlayer", () => {
  const draft = {
    ...draftOf(null, null),
    name: "  Ida Hellstrom  ",
    team: "U15",
    shirt: "9",
    height_cm: "176",
    country: "Norway",
    primary_position: "Striker",
    other_positions: ["CM"],
    birth_date: "1995-07-10",
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mock.saveBio.mockResolvedValue({});
    mock.savePhoto.mockResolvedValue({ ok: true });
  });

  it("omits the id for a new player and writes the other two under the server's", async () => {
    mock.saveRoster.mockResolvedValue([{ id: "ida-hellstrom", name: "Ida Hellstrom", team: "U15" }]);

    const id = await savePlayer(null, draft, null);

    expect(id).toBe("ida-hellstrom");
    // No `id` key at all - present-and-null would have the server mint nothing.
    expect(mock.saveRoster.mock.calls[0][0][0]).not.toHaveProperty("id");
    expect(mock.saveBio.mock.calls[0][0]).toBe("ida-hellstrom");
  });

  it("sends the roster the SHORT code, and the bio the label", async () => {
    // The bug this pins: the form holds "Striker", and writing that to the
    // roster made group() (players-section.tsx) file the player by its own
    // fall-through - "Centre Back" and "Left Winger" land as MID, "Defensive
    // Midfielder" as DEF - and jobs.save_roster truncates past 12 characters,
    // so "Attacking Midfielder" was stored as "Attacking Mi".
    mock.saveRoster.mockResolvedValue([{ id: "ida-hellstrom", name: "Ida Hellstrom", team: "U15" }]);

    await savePlayer(null, draft, null);

    expect(mock.saveRoster.mock.calls[0][0][0].position).toBe("ST");
    expect(mock.saveBio.mock.calls[0][1].primary_position).toBe("Striker");
  });

  it("passes a position nobody recognises through untouched", async () => {
    // A coach's free text is not ours to overwrite or to drop.
    mock.saveRoster.mockResolvedValue([{ id: "a", name: "A", team: "U15" }]);
    await savePlayer(null, { ...draft, name: "A", primary_position: "Libero" }, null);
    expect(mock.saveRoster.mock.calls[0][0][0].position).toBe("Libero");
  });

  it("sends the id for an existing player, so a rename keeps the profile", async () => {
    mock.saveRoster.mockResolvedValue([{ id: "ida-hellstrom", name: "Ada H", team: "U15" }]);

    const id = await savePlayer("ida-hellstrom", { ...draft, name: "Ada H" }, null);

    expect(mock.saveRoster.mock.calls[0][0][0].id).toBe("ida-hellstrom");
    expect(id).toBe("ida-hellstrom");
  });

  it("follows the server when it mints a DIFFERENT id than the one we sent", async () => {
    // The id-match misses, so the name-and-squad match is what finds the row.
    // Taking `pid` here instead would write the bio onto the old profile.
    mock.saveRoster.mockResolvedValue([{ id: "ada-h", name: "Ida Hellstrom", team: "U15" }]);

    expect(await savePlayer("stale-id", draft, null)).toBe("ada-h");
    expect(mock.saveBio.mock.calls[0][0]).toBe("ada-h");
  });

  it("matches on the squad too - two clubs' Ada are two players", async () => {
    mock.saveRoster.mockResolvedValue([
      { id: "ida-hellstrom-u13", name: "Ida Hellstrom", team: "U13" },
      { id: "ida-hellstrom-u15", name: "Ida Hellstrom", team: "U15" },
    ]);

    expect(await savePlayer(null, draft, null)).toBe("ida-hellstrom-u15");
  });

  it("throws rather than writing a bio nobody can find, when no row comes back", async () => {
    mock.saveRoster.mockResolvedValue([]);

    await expect(savePlayer(null, draft, null)).rejects.toThrow(
      "the server did not return the saved player",
    );
    expect(mock.saveBio).not.toHaveBeenCalled();
    expect(mock.savePhoto).not.toHaveBeenCalled();
  });

  it("trims the name and derives the country code the feed's way", async () => {
    mock.saveRoster.mockResolvedValue([{ id: "a", name: "Ida Hellstrom", team: "U15" }]);

    await savePlayer(null, draft, null);

    expect(mock.saveRoster.mock.calls[0][0][0].name).toBe("Ida Hellstrom");
    const bio = mock.saveBio.mock.calls[0][1];
    expect(bio.name).toBe("Ida Hellstrom");
    expect(bio.country_code).toBe("NOR");
    expect(bio.height_cm).toBe(176);
    expect(bio.shirt).toBe(9);
  });

  it("sends null for an empty number, never 0 - Number('') is a shirt nobody wears", async () => {
    mock.saveRoster.mockResolvedValue([{ id: "a", name: "Ida Hellstrom", team: "U15" }]);

    await savePlayer(null, { ...draft, shirt: "  ", height_cm: "" }, null);

    expect(mock.saveRoster.mock.calls[0][0][0].number).toBeNull();
    expect(mock.saveBio.mock.calls[0][1].shirt).toBeNull();
    expect(mock.saveBio.mock.calls[0][1].height_cm).toBeNull();
  });

  it("keeps shirt 0 - a legitimate number that an || would have dropped", async () => {
    mock.saveRoster.mockResolvedValue([{ id: "a", name: "Ida Hellstrom", team: "U15" }]);

    await savePlayer(null, { ...draft, shirt: "0" }, null);

    expect(mock.saveRoster.mock.calls[0][0][0].number).toBe(0);
    expect(mock.saveBio.mock.calls[0][1].shirt).toBe(0);
  });

  it("nulls an unpicked country and foot rather than storing an empty string", async () => {
    mock.saveRoster.mockResolvedValue([{ id: "a", name: "Ida Hellstrom", team: "U15" }]);

    await savePlayer(null, { ...draft, country: "", foot: "", birth_date: "" }, null);

    const bio = mock.saveBio.mock.calls[0][1];
    expect(bio.country).toBeNull();
    expect(bio.country_code).toBeNull(); // nationCode("") - not "" and not undefined
    expect(bio.foot).toBeNull();
    expect(bio.birth_date).toBeNull();
  });

  it("does not take the profile down with the photo - the face fails on its own", async () => {
    mock.saveRoster.mockResolvedValue([{ id: "a", name: "Ida Hellstrom", team: "U15" }]);
    mock.savePhoto.mockRejectedValue(new Error("that file is 14 MB"));

    // The registry and the bio are already written at this point; throwing here
    // would tell the operator nothing saved, and they would type it all again.
    expect(await savePlayer(null, draft, new File([], "a.png"))).toBe("a");
    expect(mock.error).toHaveBeenCalledWith(
      "Saved, but the photo did not upload",
      expect.objectContaining({ description: "that file is 14 MB" }),
    );
    expect(mock.clearStoreCache).toHaveBeenCalled();
  });

  it("drops the memo so the panel does not redraw the bio it just replaced", async () => {
    mock.saveRoster.mockResolvedValue([{ id: "a", name: "Ida Hellstrom", team: "U15" }]);

    await savePlayer(null, draft, null);

    expect(mock.savePhoto).not.toHaveBeenCalled(); // no file, no request
    expect(mock.clearStoreCache).toHaveBeenCalledTimes(1);
  });

  it("leaves the cache alone when the registry refuses, so nothing refetches a lie", async () => {
    mock.saveRoster.mockRejectedValue(new Error("squad U15 does not exist"));

    await expect(savePlayer(null, draft, null)).rejects.toThrow("squad U15 does not exist");
    expect(mock.saveBio).not.toHaveBeenCalled();
    expect(mock.clearStoreCache).not.toHaveBeenCalled();
  });
});

/** THE AGE CELL IS AN ECHO, not a second source. The stored age is derived on
 *  the server (jobs.save_bio recomputes it from birth_date and pops any age
 *  that was sent), and this function is what the cell shows the moment a date
 *  is picked - before any save. If the two rules disagree, the card says one
 *  number while you type and a different one after Save, which is exactly the
 *  "two fields that can disagree about one fact" the panel's own comment says
 *  it is avoiding.
 *
 *  The server's rule, verbatim from app/server/jobs.py:
 *      today.year - y - ((today.month, today.day) < (mo, day)),  kept if 0 <= age < 120
 *  Every case below is that rule's answer. The clock is frozen, because an
 *  age test that reads the real date passes today and fails on a birthday. */
describe("ageOf", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-09-16T12:00:00"));
  });
  afterEach(() => vi.useRealTimers());

  it("counts the birthday as passed only once it has been", () => {
    expect(ageOf("1995-07-10")).toBe(31); // July is behind us
    expect(ageOf("1995-12-10")).toBe(30); // December is not
  });

  it("turns you a year older ON the day, not the day after", () => {
    expect(ageOf("2000-09-15")).toBe(26); // yesterday
    expect(ageOf("2000-09-16")).toBe(26); // today: the strict < is what makes this right
    expect(ageOf("2000-09-17")).toBe(25); // tomorrow
  });

  it("shows nothing rather than a number for a date it cannot read", () => {
    // The `—` in the cell. Same shapes longDate refuses, and for the same
    // reason: half a date on a card is worse than no date.
    for (const iso of ["", "10/07/1995", "1995-7-10", "1995-07", "not a date"])
      expect(ageOf(iso)).toBeNull();
  });

  it("refuses an age outside 0-120, which is the server's own range", () => {
    expect(ageOf("2027-01-01")).toBeNull(); // a future birth date: -1
    expect(ageOf("1800-01-01")).toBeNull(); // 226
    expect(ageOf("1906-09-16")).toBeNull(); // exactly 120: the bound is exclusive
    expect(ageOf("1907-09-16")).toBe(119); // and 119 is still a person
  });

  it("agrees with the server on a date only the feed could have written", () => {
    // "2002-13-40" passes the SHAPE check on both sides and neither rejects
    // it: the server's tuple compare reads (9,16) < (13,40) and subtracts a
    // year, and so does this. Pinned because the two rules are written in
    // different languages and the only thing keeping them together is that
    // somebody checks. The form itself cannot produce this - the date comes
    // from <input type="date"> - but a bio the feed wrote can.
    expect(ageOf("2002-13-40")).toBe(23);
  });
});

describe("POSITIONS", () => {
  /** The two position fields store DIFFERENT halves of this table - primary
   *  stores the label, other_positions stores the short - and the editor's
   *  "already the primary" guard compares a label to a label. Both columns
   *  have to stay unique or that guard disables the wrong row. */
  it("is unique in both columns, which the editor's disable rule depends on", () => {
    expect(new Set(POSITIONS.map((p) => p.short)).size).toBe(POSITIONS.length);
    expect(new Set(POSITIONS.map((p) => p.label)).size).toBe(POSITIONS.length);
    for (const p of POSITIONS) expect(p.short).toMatch(/^[A-Z]{2,3}$/);
  });
});
