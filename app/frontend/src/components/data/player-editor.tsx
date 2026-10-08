"use client";

// THE FLAG STYLESHEET, imported HERE and not in globals.css: this file holds
// the only `fi fi-` in the app, and in the global sheet it was render-blocking
// weight on every route before first paint.
//
// @C 35999,7652,187417,151418,33044,24189 :: .next/dev/static/chunks/src_app_globals_css_*.single.css flag block lines 5629-7627, flag-icons 7.5.0, read 2026-09-16
// Flag block bytes raw and gz, then the global sheet raw and gz with and
// without it.
//
// Cited, not measured: that chunk is a gitignored, hash-named dev-build artefact, so
// the right-hand side is an artefact somebody opened, not a command anyone can
// re-run. The comment this replaces said "60 KB" with no run at all.
//
// flag-icons 7.5.0, MIT, (c) 2013 Panayiotis Lipiridis. The notice is
// reproduced in docs/THIRD_PARTY_NOTICES.md, as the licence
// requires.
import "flag-icons/css/flag-icons.min.css";

import { useRef } from "react";
import { CalendarDays } from "lucide-react";
import { toast } from "sonner";

import {
  DropdownMenu,
  DropdownMenuCheckboxItem as CheckItem,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Fact, PLAIN, longDate } from "@/components/data/player-fact";
import { api, clearStoreCache } from "@/lib/api";
import { FEET, NATIONS, POSITIONS, nationCode, positionLabel, positionShort } from "@/lib/player-facts";
import type { ClubPlayer, PlayerBio } from "@/lib/types";
import { cn } from "@/lib/utils";

/** THE ENTERED HALF OF A PROFILE, EDITED WHERE IT IS READ.
 *
 *  There is no edit screen. The facts grid does not become a form: each value
 *  becomes typeable in the cell it already occupies, at the same size, in the
 *  same place, with its label still under it. Switching into edit mode moves
 *  nothing on screen - which is the point, because the thing you are checking
 *  while you type is the row next to the one you are typing in.
 *
 *  AT REST IT IS THE READ VIEW, pixel for pixel: no boxes, no borders, no
 *  chevrons, nothing moved and nothing added. You click the value itself and
 *  it types or it opens. The only affordance is a dotted underline under the
 *  pointer, which is painted over the text and so costs no height - and the
 *  only chrome that changes is the bolt in the band, which becomes Save and
 *  Cancel (see BandEdit in players-section.tsx).
 *
 *  Both views render the SAME Fact component (player-fact.tsx). That is what
 *  makes "nothing moves" a property of the code rather than a thing to keep
 *  re-checking by eye.
 *
 *  WHO OWNS WHAT, because this writes to THREE stores and getting it backwards
 *  is how a player forks into two people:
 *
 *    name, shirt number, position, squad   ->  PUT /api/roster   (the registry:
 *                                              who exists, in which squad)
 *    height, birth date, foot, country,    ->  PUT /api/players/{id}/bio
 *    primary and other positions               (the profile card)
 *    the face                              ->  PUT /api/players/{id}/photo
 *
 *  The registry goes FIRST and its reply is authoritative, because the player
 *  id is server-minted from the name. A new player has no id until the server
 *  says so, and a renamed player keeps the old one only because we send it -
 *  guessing the slug locally is what forks a career into two profiles
 *  (roster-player-id-is-the-season-join (internal note)).
 */

export interface PlayerDraft {
  name: string;
  team: string;
  shirt: string;
  primary_position: string;
  other_positions: string[];
  country: string;
  foot: string;
  height_cm: string;
  birth_date: string;
}

/** The form's own shape from whatever the two stores happen to hold. Strings
 *  throughout: an empty box is "" and means "not entered", which is the state
 *  every reader in the panel already handles. A number would have to be null
 *  and then 0 would be a valid height. */
export function draftOf(
  player: ClubPlayer | null,
  bio: PlayerBio | null,
  /** This club's squads. The bio's `team` is the FEED's club - e.g. "Northbridge FC" on a
   *  player it imported - and is only a squad here if the club happens to have
   *  one by that name, so it is accepted only when it is in this list. */
  teams: string[] = [],
): PlayerDraft {
  // ONE ALPHABET IN THE FORM: the short code. The feed writes the other
  // positions two ways - `positions` dots carry shorts, `other_positions`
  // carries the long labels (as a former feed importer wrote them) - and the checkboxes key on
  // shorts, so a label list ticked nothing and the first save rewrote the
  // field from labels to shorts behind the operator's back.
  const short = (p: { short?: string | null; label?: string | null }) =>
    positionShort(p.short ?? p.label) ?? p.short ?? p.label ?? "";
  const dots = (bio?.positions ?? []).filter((p) => !p.main).map(short).filter(Boolean);
  // THE OPERATOR'S LIST WINS over the feed's dots. It used to be the other way
  // round, which made this field write-only: the save stored `other_positions`
  // and both readers went on preferring `positions`, so the edit came back
  // undone with no error anywhere.
  const typed = (bio?.other_positions ?? []).map((x) => positionShort(x) ?? x).filter(Boolean);
  return {
    name: bio?.name ?? player?.name ?? "",
    team: player?.teams?.[0]?.team ?? (teams.includes(bio?.team ?? "") ? bio!.team! : ""),
    shirt: String(bio?.shirt ?? player?.teams?.[0]?.number ?? ""),
    // The form speaks the bio's alphabet (labels) for the PRIMARY position,
    // because that is what the profile prints. A registry-only player arrives
    // with the roster's short code, so it is translated in.
    primary_position:
      bio?.primary_position ?? positionLabel(player?.position) ?? player?.position ?? "",
    other_positions: typed.length ? typed : dots,
    country: bio?.country ?? "",
    foot: bio?.foot ?? "",
    height_cm: bio?.height_cm != null ? String(bio.height_cm) : "",
    birth_date: bio?.birth_date ?? "",
  };
}

/** A name and a squad, and nothing else. The id is minted from the name and
 *  the roster file is keyed on the squad, so those two are what the registry
 *  cannot do without; everything else is allowed to be blank, which is the
 *  ordinary state of a squad entered on a Tuesday night. */
export function canSave(d: PlayerDraft): boolean {
  return d.name.trim().length > 0 && d.team.trim().length > 0;
}

/** Write the draft to all three stores and return the id the server decided
 *  on. Throws with the server's own message; the caller shows it.
 *
 *  NOT OPTIMISTIC, the shape club-editor.tsx uses: PUT, take the canonical
 *  reply, drop the memo, let the caller refetch. A row that looks saved and is
 *  not is worse here than a spinner - this is the screen where somebody enters
 *  a squad once and trusts it afterwards.
 */
export async function savePlayer(
  pid: string | null,
  draft: PlayerDraft,
  photo: File | null,
): Promise<string> {
  // 1. THE REGISTRY, and its reply decides the id. `id` is sent for an
  //    existing player so a rename keeps the profile; omitted for a new one so
  //    the server mints the slug.
  const roster = await api.saveRoster([
    {
      ...(pid ? { id: pid } : {}),
      name: draft.name.trim(),
      team: draft.team,
      number: draft.shirt.trim() === "" ? null : Number(draft.shirt),
      // THE ROSTER GETS THE SHORT CODE. group() in players-section parses this
      // to pick the badge, the sort rail and the squad filters, and falls
      // through to MID for anything it does not recognise - so writing the
      // label filed a centre back as a midfielder and a defensive midfielder
      // as a defender. jobs.save_roster also truncates it to 12 characters,
      // which turned "Attacking Midfielder" into "Attacking Mi". Free text
      // nobody recognises passes through untouched rather than being guessed.
      position: positionShort(draft.primary_position) ?? draft.primary_position,
    },
  ]);
  const row =
    (pid && roster.find((r) => r.id === pid)) ||
    roster.find((r) => r.team === draft.team && (r.name ?? "").trim() === draft.name.trim());
  if (!row?.id) throw new Error("the server did not return the saved player");

  // 2. THE PROFILE CARD, merged onto whatever the feed wrote.
  await api.saveBio(row.id, {
    name: draft.name.trim(),
    team: draft.team,
    shirt: draft.shirt.trim() === "" ? null : Number(draft.shirt),
    primary_position: draft.primary_position || null,
    other_positions: draft.other_positions,
    country: draft.country || null,
    country_code: nationCode(draft.country),
    foot: draft.foot || null,
    height_cm: draft.height_cm.trim() === "" ? null : Number(draft.height_cm),
    birth_date: draft.birth_date || null,
  });

  // 3. THE FACE, last and separately: it is the one part that can fail on its
  //    own terms (too large, not an image) without the profile being wrong, so
  //    it must not take the rest of the save down with it.
  if (photo) {
    try {
      await api.savePhoto(row.id, photo);
    } catch (e) {
      toast.error("Saved, but the photo did not upload", {
        description: e instanceof Error ? e.message : String(e),
      });
    }
  }

  clearStoreCache(); // getBios is memoized; the panel would show the old one
  return row.id;
}


/* ------------------------------------------------------------ the controls */

/** 0 to 99, the whole domain of a shirt. A list rather than a typed box
 *  because that is the field most often copied off a printed team sheet, and
 *  because it is the only number here with a fixed range. */
const SHIRTS = Array.from({ length: 100 }, (_, i) => String(i));

/** 120 to 215 cm. A list for the same reason the shirt is one - it is picked,
 *  not computed - and this range because the squads here start at U9: 120 is
 *  a small nine-year-old and 215 clears the tallest professional on record. */
const HEIGHTS = Array.from({ length: 96 }, (_, i) => String(120 + i));

/** One flag, from flag-icons (MIT, (c) 2013 Panayiotis Lipiridis; the notice
 *  is in docs/THIRD_PARTY_NOTICES.md and the stylesheet is imported once in
 *  globals.css). `fis` squares it, so a row of them lines up whatever the
 *  flag's own aspect is. */
function Flag({ iso }: { iso: string }) {
  return <span className={`fi fi-${iso} fis shrink-0 rounded-[2px]`} aria-hidden />;
}

/** Age from the DRAFT's birth date, so the cell above the calendar answers the
 *  moment you pick one. The server derives the stored age the same way; this
 *  is the local echo of it, not a second source. */
export function ageOf(iso: string): number | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
  if (!m) return null;
  const [y, mo, d] = m.slice(1).map(Number);
  const now = new Date();
  const age = now.getFullYear() - y - (now.getMonth() + 1 < mo || (now.getMonth() + 1 === mo && now.getDate() < d) ? 1 : 0);
  return age >= 0 && age < 120 ? age : null;
}

/** A value you click to pick from a list. It IS the value line - same type,
 *  same place - and the menu opens under it. A Select's trigger would bring a
 *  box and a chevron with it, which is exactly the change this must not make. */
function Pick({
  value,
  onChange,
  options,
  placeholder,
  label,
  after,
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string; mark?: React.ReactNode }[];
  placeholder: string;
  /** REQUIRED. The cell's label lives in a sibling <p> inside Fact, which
   *  names nothing as far as assistive tech is concerned - a screen reader
   *  read this trigger as "—, button" eight times down the form. */
  label: string;
  /** The dim half the read view prints after the value, e.g. a country code. */
  after?: React.ReactNode;
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger aria-label={label} className={cn(PLAIN, "truncate text-left")}>
        {value || <span className="font-normal text-[var(--slate)]">{placeholder}</span>}
      </DropdownMenuTrigger>
      {after}
      <DropdownMenuContent align="start" className="max-h-72 w-56 overflow-y-auto">
        <DropdownMenuRadioGroup value={value} onValueChange={onChange}>
          {options.map((o) => (
            <DropdownMenuRadioItem key={o.value} value={o.value}>
              {o.mark}
              <span className="min-w-0 truncate">{o.label}</span>
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

/** THE EIGHT FACTS, editable where they are read.
 *
 *  Order, pairing and wording are BioFacts' own - height beside age, primary
 *  position beside the others, "178 cm" with its unit - because the two views
 *  are the same grid and a coach who has read it once knows where to look.
 *
 *  Every cell is rendered ALWAYS here, where the read view drops the ones with
 *  no value: an empty cell is the only place you can type the missing fact in.
 *  That is the one difference between the two, and it only shows on a profile
 *  that is incomplete.
 */
export function EditableFacts({
  draft,
  setDraft,
  teams,
}: {
  draft: PlayerDraft;
  setDraft: (d: PlayerDraft) => void;
  /** The club's squads. The one list here that is NOT static - a squad is this
   *  club's own fact, not a fact about football. */
  teams: string[];
}) {
  const set = (k: keyof PlayerDraft, v: string | string[]) => setDraft({ ...draft, [k]: v });
  const date = useRef<HTMLInputElement>(null);
  const age = ageOf(draft.birth_date);
  const others = draft.other_positions;

  return (
    <div className="grid min-w-0 flex-1 grid-cols-2 content-start gap-x-4">
      <Fact
        label="Height"
        value={
          <>
            <Pick
              value={draft.height_cm}
              onChange={(v) => set("height_cm", v)}
              placeholder="—"
              label="Height"
              options={HEIGHTS.map((h) => ({ value: h, label: `${h} cm` }))}
            />
            {draft.height_cm && <span className="ml-1">cm</span>}
          </>
        }
      />

      {/* THE AGE CELL. The age is not typeable - it is derived from the date,
          and two boxes that can disagree about one fact is how a card ends up
          saying 24 under a birth date in 2001. The date comes from the
          calendar, which is the one control this grid adds: a bare icon after
          the value, opening the browser's own picker. */}
      <Fact
        label="Age"
        sub={draft.birth_date ? longDate(draft.birth_date) : null}
        value={
          <>
            <span className={cn(!age && "font-normal text-[var(--slate)]")}>
              {age != null ? `${age} years` : "—"}
            </span>
            <button
              type="button"
              aria-label="Pick date of birth"
              title="Date of birth"
              onClick={() => date.current?.showPicker?.()}
              // -m-1.5 keeps the row at h-5 while the hit area grows to 24px,
              // which is WCAG 2.2 2.5.8's floor. It was 14x14, and it is the
              // ONLY way in: the real input below is aria-hidden and
              // unfocusable, so a keyboard user could not set a birth date.
              className="text-muted-foreground -m-1.5 grid size-6 shrink-0 place-items-center rounded-[var(--radius-xs)] p-1.5 transition-colors hover:text-[var(--ink)]"
            >
              <CalendarDays className="size-3.5" />
            </button>
            {/* Zero-sized and unfocusable: it exists to own showPicker() and
                the value, never to be typed into. */}
            <input
              ref={date}
              type="date"
              tabIndex={-1}
              aria-hidden
              value={draft.birth_date}
              onChange={(e) => set("birth_date", e.target.value)}
              className="pointer-events-none absolute size-0 opacity-0"
            />
          </>
        }
      />

      <Fact
        label="Preferred foot"
        value={
          <Pick
            value={draft.foot}
            onChange={(v) => set("foot", v)}
            placeholder="—"
            label="Preferred foot"
              options={FEET.map((f) => ({ value: f, label: f }))}
          />
        }
      />

      <Fact
        label="Country"
        sub={nationCode(draft.country) ?? "Country"}
        value={
          <Pick
            value={draft.country}
            onChange={(v) => set("country", v)}
            placeholder="—"
            label="Country"
              options={NATIONS.map((n) => ({
              value: n.name,
              label: n.name,
              mark: <Flag iso={n.iso} />,
            }))}
          />
        }
      />

      <Fact
        label="Primary position"
        value={
          <Pick
            value={draft.primary_position}
            onChange={(v) => set("primary_position", v)}
            placeholder="—"
            label="Primary position"
              options={POSITIONS.map((p) => ({ value: p.label, label: p.label }))}
          />
        }
      />

      {/* OTHER POSITIONS stays the comma list it is in the read view, and the
          list itself is what you click. Chips here would be a different shape
          from every cell beside it - and this answer is two or three codes,
          not a row of buttons. */}
      <Fact
        label="Other positions"
        value={
          <DropdownMenu>
            <DropdownMenuTrigger
              aria-label="Other positions"
              className={cn(PLAIN, "truncate text-left")}
            >
              {others.length ? (
                others.join(", ")
              ) : (
                <span className="font-normal text-[var(--slate)]">—</span>
              )}
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="max-h-72 w-56 overflow-y-auto">
              {POSITIONS.map((p) => (
                <CheckItem
                  key={p.short}
                  checked={others.includes(p.short)}
                  disabled={draft.primary_position === p.label}
                  // The menu stays open: this cell is usually two or three
                  // answers, and closing after each one costs a click apiece.
                  onSelect={(e) => e.preventDefault()}
                  onCheckedChange={(on) =>
                    set(
                      "other_positions",
                      on ? [...others, p.short] : others.filter((x) => x !== p.short),
                    )
                  }
                >
                  <span className="min-w-0 truncate">{p.label}</span>
                </CheckItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        }
      />

      <Fact
        label="Shirt number"
        value={
          <>
            <span className={cn(!draft.shirt && "hidden")}>#</span>
            <Pick
              value={draft.shirt}
              onChange={(v) => set("shirt", v)}
              placeholder="—"
              label="Shirt number"
              options={SHIRTS.map((n) => ({ value: n, label: `#${n}` }))}
            />
          </>
        }
      />

      <Fact
        label="Squad"
        value={
          <Pick
            value={draft.team}
            onChange={(v) => set("team", v)}
            placeholder="—"
            label="Squad"
              options={teams.map((t) => ({ value: t, label: t }))}
          />
        }
      />
    </div>
  );
}
