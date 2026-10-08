/** "2002-05-06" -> "May 6, 2002". Parsed as UTC parts rather than handed to
 *  `new Date("2002-05-06")`, which is midnight UTC and prints as the 5th to
 *  anyone west of Greenwich - a birthday off by one, on the one screen where
 *  a date is the fact. */
export function longDate(iso?: string | null): string | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso ?? "");
  if (!m) return null;
  const MONTH = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  // The regex checks the SHAPE, not the range: "2002-13-40" is four digits,
  // two, two, and printed "undefined 40, 2002" onto the player card. A date
  // the feed got wrong is better absent than shown as the word undefined.
  const month = MONTH[Number(m[2]) - 1];
  return month ? `${month} ${Number(m[3])}, ${m[1]}` : null;
}

/** ONE FACT IN THE PROFILE'S GRID - the value large, its label under it.
 *
 *  Its own file because BOTH views render it: the read view fills `value` with
 *  a string, the editor fills it with a control. That is the whole mechanism
 *  behind "editing changes nothing on screen" - the two are not two layouts
 *  that were matched by eye and will drift apart on the next change, they are
 *  the same component with different children.
 *
 *  The reference's own order - the number first, what it means second - which
 *  reads faster because the numbers are what you are scanning for.
 */
export function Fact({
  value,
  label,
  sub,
}: {
  value: React.ReactNode;
  label: string;
  sub?: string | null;
}) {
  return (
    <div className="min-w-0 py-1.5">
      {/* THE VALUE IS ONE LINE, ALWAYS. "Other positions" is a list - "DM, CM,
          LB, RB" - and in a half-width cell it used to wrap to two, which
          pushed the fact beside it out of line and made the block a different
          height for every player. Too long now scrolls sideways instead, with
          the right edge faded so the cut says there is more rather than
          looking like a clipped word. The fade is a mask on the box, so it
          only touches ink when ink actually reaches the edge - a short value
          leaves that strip empty and nothing dims. */}
      <div className="flex h-5 items-center overflow-x-auto text-sm font-medium whitespace-nowrap text-[var(--ink)] [mask-image:linear-gradient(to_right,#000_calc(100%-1.25rem),transparent)] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
        {value}
      </div>
      <p className="text-muted-foreground truncate text-xs">{sub ?? label}</p>
    </div>
  );
}

/** What an editable value wears so that it is INVISIBLE until you touch it:
 *  no border, no background, no padding, the grid's own type. The only thing
 *  that ever changes is on hover and on focus, and neither moves anything -
 *  an underline is drawn over the box, not added to it.
 *
 *  DOTTED IN BOTH STATES, hover and caret. A solid rule on focus said "this
 *  is a text box now", which is the one thing this whole approach is avoiding;
 *  the dots say editable and stay out of the way of the value.
 *
 *  Shared by every control in the editor so there is one place this is true. */
export const PLAIN =
  // min-w-[3ch], not min-w-0: an EMPTY field renders a bare em dash, which made
  // it a target about ten pixels wide - on the form a new player is typed into,
  // where every field starts empty. Three characters is wide enough to hit and
  // narrower than any real value, so a filled field is unaffected.
  //
  // WIDTH ONLY, deliberately. The height stays 20px because this control sits
  // inside Fact's one-line value box, which scrolls sideways - and in CSS a box
  // that scrolls on one axis clips the other, so a taller control would have
  // its extra height cut off and a cut-off region takes no clicks. Growing the
  // hit area vertically means growing the row itself, in both views.
  "h-5 min-w-[3ch] cursor-pointer appearance-none border-0 bg-transparent p-0 text-sm font-medium " +
  "text-[var(--ink)] underline-offset-4 outline-none hover:underline hover:decoration-dotted " +
  "focus:underline focus:decoration-dotted placeholder:font-normal placeholder:text-[var(--slate)]";
