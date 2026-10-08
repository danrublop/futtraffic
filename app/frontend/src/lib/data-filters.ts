import type { ClubRow, Fixture, PlayerRow } from "./season";
import type { ClubPlayer, Match, Season, SeasonPlayer, Team } from "./types";

/** NARROWING, as pure functions.
 *
 *  Every Data section filters, and until now each one did it with an inline
 *  `.filter()` inside a `useMemo`, which is why no two of them narrowed the
 *  same way and none of them were tested. These are the same predicates,
 *  lifted out where the tests can reach them.
 *
 *  WHAT IS NOT HERE, deliberately: the team -> matches join. A job stores its
 *  team as a NAME string and the server turns it into a team id with
 *  `jobs._slug` (a lowercase, non-alphanumeric-to-dash fold). Mirroring that
 *  function in TypeScript would be a second hand-maintained copy of a rule the
 *  two sides only agree on by manual sync, which this codebase has already been
 *  bitten by once (see the tier-registry duplication). `api.listMatches(teamId)`
 *  does the join on the side that owns the slug, and `api.ts` says so.
 *
 *  It matters here and not in theory: this club's squad is named "First team"
 *  while its jobs still say "Northbridge FC", because a team keeps its id through a
 *  rename. Matching on the display name would return nothing at all.
 */

/* ------------------------------------------------------- the feed's players */

/** Feed player rows for one club. `""` means every club, not none. */
export const byClub = (rows: PlayerRow[], club: string): PlayerRow[] =>
  club ? rows.filter((r) => r.club === club) : rows;

/** Feed player rows from one match. */
export const byMatch = (rows: PlayerRow[], matchId: string): PlayerRow[] =>
  matchId ? rows.filter((r) => r.match === matchId) : rows;

/* -------------------------------------------------------------- the fixtures */

/** Fixtures for a set of match ids. An EMPTY set is not "no filter" — it is a
 *  selection that matched nothing, and returning everything for it would answer
 *  a different question from the one on screen. `null` is no filter. */
export const fixturesIn = (fixtures: Fixture[], ids: ReadonlySet<string> | null): Fixture[] =>
  ids ? fixtures.filter((f) => ids.has(f.id)) : fixtures;

/** Every fixture a club appears in, either side of it. */
export const fixturesInvolving = (fixtures: Fixture[], club: string): Fixture[] =>
  club ? fixtures.filter((f) => f.home === club || f.away === club) : fixtures;

/* ----------------------------------------------------------- the feed's clubs */

/** Club rows, narrowed to a set of matches and optionally to one club. Same
 *  empty-set rule as `fixturesIn`. */
export function clubRowsIn(
  rows: ClubRow[],
  ids: ReadonlySet<string> | null,
  club = "",
): ClubRow[] {
  return rows.filter((r) => (ids ? ids.has(r.match) : true) && (club ? r.club === club : true));
}

/* --------------------------------------------------------------- our registry */

/** Our players, narrowed to the ones with a CONFIRMED track in one match.
 *
 *  The registry itself does not know which matches a player appeared in — that
 *  is a human act recorded in the roster file, and it reaches the browser as
 *  the measured season's `by_match`. So the match filter is answered by the
 *  same data the numbers come from, which is the only way the filter and the
 *  columns can agree.
 *
 *  A player with no measured row is filtered OUT by a match filter, because
 *  "was this person confirmed in this match" is false for them. That is not the
 *  same as the em dash in an unfiltered table, which means "no confirmation
 *  anywhere yet".
 */
export function confirmedIn(
  players: ClubPlayer[],
  measured: ReadonlyMap<string, SeasonPlayer>,
  matchId: string,
): ClubPlayer[] {
  if (!matchId) return players;
  return players.filter((p) => measured.get(p.id)?.by_match?.[matchId] != null);
}

/** The measured season as a lookup by roster id — the join every player-facing
 *  section makes. Built once here rather than in four `useMemo`s. */
export const measuredById = (season: Season | null): Map<string, SeasonPlayer> =>
  new Map((season?.players ?? []).map((p) => [p.id, p]));

/* ------------------------------------------------------------------- squads */

/** Teams in one season. The FIRST reader of the season field — it has been
 *  written on every squad since the club model shipped and nothing has ever
 *  filtered by it, so a player who moved up shows both memberships at once. */
export const teamsInSeason = (teams: Team[], season: string): Team[] =>
  season ? teams.filter((t) => (t.season || "") === season) : teams;

/** Every distinct season the ladder mentions, newest-looking first. Sorted as
 *  strings descending, which is right for "2026/27" and "2025/26" and is the
 *  only ordering available for a field a human types freely. */
export function seasonsOf(teams: Team[]): string[] {
  return [...new Set(teams.map((t) => t.season || "").filter(Boolean))].sort((a, b) =>
    b.localeCompare(a),
  );
}

/* ------------------------------------------------------------------ matches */

/** One row of the Matches section: the JOB, plus the feed's account of it when
 *  the feed has one. */
export interface MatchRow {
  match: Match;
  fixture: Fixture | null;
}

/** THE MATCHES SECTION'S JOIN, and the reason it is this way round.
 *
 *  The obvious source is `SeasonTable.fixtures`, which is what the old match
 *  table rendered. It is the wrong one: fixtures come from the ENTERED feed,
 *  and an academy filming its own matches has no feed at all — `season-tables`
 *  computes `noFeed` from exactly this array. A fixture-derived Matches section
 *  is permanently empty for the club this product is for.
 *
 *  A JOB exists for every upload, feed or not. So the job list is the spine and
 *  the fixture is an enrichment: competition, score and date where a feed
 *  supplied them, absent where it did not. It also carries something a fixture
 *  cannot — the job's processing status, so a match still being tracked appears
 *  here instead of arriving only once it finishes.
 */
export function matchRows(matches: Match[], fixtures: Fixture[]): MatchRow[] {
  const feed = new Map(fixtures.map((f) => [f.id, f]));
  return matches.map((m) => ({ match: m, fixture: feed.get(m.id) ?? null }));
}

/** Newest first, on the job's own `created` stamp rather than the feed's date,
 *  because every row has one. A row with no stamp sorts last rather than first,
 *  which is what an undefined would do through a subtraction. */
export const byNewest = (rows: MatchRow[]): MatchRow[] =>
  [...rows].sort((a, b) => (b.match.created ?? 0) - (a.match.created ?? 0));

/* ------------------------------------------------- our squad, in feed terms */

/** WHICH CLUB THE FEED CALLS THIS SQUAD, worked out from the matches rather
 *  than from the name.
 *
 *  The old join compared our squad's NAME to the feed's club name. It broke the
 *  moment a squad was renamed: a team keeps its id through a rename, so this
 *  club's squad is "First team" while every job still says "Northbridge FC" and
 *  every feed row still says "Northbridge FC". The Record and Opponents tables
 *  silently emptied — and "absent because the feed does not name this squad" is
 *  a legitimate state, so nothing looked broken.
 *
 *  The honest join is behavioural, not lexical: of the clubs that appear in
 *  THIS SQUAD'S matches, ours is the one that appears in all of them. Two clubs
 *  play each match and only one of them is us, so the club present in every
 *  match is the club whose matches these are.
 *
 *  It refuses rather than guesses. One match is not evidence — both clubs
 *  appear in all of it — so a single-match squad returns "" and the feed blocks
 *  stay absent. So does a tie. Getting this wrong would attribute an opponent's
 *  season to our squad, which is worse than showing nothing.
 */
export function feedClubFor(rows: ClubRow[], ids: ReadonlySet<string>): string {
  if (ids.size < 2) return "";
  const seen = new Map<string, Set<string>>();
  for (const r of rows) {
    if (!ids.has(r.match)) continue;
    (seen.get(r.club) ?? seen.set(r.club, new Set()).get(r.club)!).add(r.match);
  }
  const inAll = [...seen].filter(([, m]) => m.size === ids.size).map(([club]) => club);
  return inAll.length === 1 ? inAll[0] : "";
}
