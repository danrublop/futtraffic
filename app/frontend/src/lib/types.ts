/** Shapes returned by the FastAPI service. Mirrors app/server/jobs.py's
 *  `status_view` and model/engine/stats.py's `build_match` — if you change one of
 *  those, change this, because nothing checks it across the language boundary. */

export type JobStatus =
  | "queued"
  | "calibrating"
  | "tracking"
  | "identifying"
  | "done"
  | "failed";

export interface Match {
  id: string;
  name: string | null;
  /** The club this match belongs to. status_view always sends it — jobs written
   *  before teams existed read back as "Default" rather than being migrated —
   *  so it is optional here only so a hand-built fixture need not carry it. */
  team?: string;
  /** The ground it was filmed at. Half of the condition under which another
   *  match's pitch setup may be reused here, which is why it is its own field
   *  and not more text inside `name`. "" when nothing was recorded. */
  venue?: string;
  status: JobStatus;
  progress: number;
  error: string | null;
  /** What a human has to do before this can move. null = nothing. */
  waiting_for: "calibration" | null;
  created: number;
  updated: number;
  bytes: number;
  total_bytes: number | null;
  uploaded_bytes: number;
  upload_complete: boolean;
  has_match: boolean;
  /** Who played and what it finished. Comes out of match.json's meta, so it is
   *  null until the pipeline has run and the list falls back to the upload
   *  name. `score` is null when nobody recorded one — the pipeline cannot
   *  detect a goal, so a score is always something a human typed. */
  fixture?: { home: string | null; away: string | null; score: string | null } | null;
  has_tracks: boolean;
  has_calibration: boolean;
  /** Display name of the match this one's pitch setup was copied from, when it
   *  was reused instead of solved here. Only ever another match at the SAME
   *  venue, same team and same frame_shape. null when a human calibrated it. */
  calibration_inherited_from: string | null;
  /** "auto": the camera was found from the painted lines with no clicks
   *  (model/engine/autocal.py). "manual": an operator's dots. Absent on older jobs. */
  calibration_source?: "auto" | "manual" | null;
  /** The automatic attempt's own report. Present on a REFUSAL too, so the setup
   *  screen can say why it is asking for dots. */
  calibration_auto?: {
    accepted: boolean;
    via?: "search" | "install" | null;
    why?: string | null;
    install_from?: string | null;
    hits?: number | null;
    samples?: number | null;
    line_rms_px?: number | null;
    /** Measured from the paint. */
    length_m?: number | null;
    length_sd_m?: number | null;
    /** NOT measurable from this camera: given for the venue, else assumed 68 m. */
    width_m?: number | null;
    width_assumed?: boolean | null;
    width_source?: "user" | "osm" | null;
  } | null;
  /** The pitch size somebody gave for this ground; null when it is being assumed. */
  pitch?: { width: number | null; length: number | null; source: string } | null;
  /** [width, height] of the uploaded video. What makes a calibration reusable. */
  frame_shape: [number, number] | null;
  stage_started?: number | null;
  attempts?: number;
  elapsed_s?: number | null;
  operator_seconds?: number;
  corrections?: number;
  identity_note?: string | null;
  /** "active", or a sentence saying the off-pitch filter may not have run. */
  scene_prior?: string | null;
  /** Detections the off-pitch filter removed over the whole match. */
  prior_rejected?: number | null;
  /** The match finished but the query store refused it. */
  ingest_error?: string | null;
  detector?: "coreml" | "rfdetr" | null;
}

export type Side = "home" | "away";

/** One row of match.json's `players`, exactly as stats.py::build_match writes
 *  it. It is NOT the PlayerStats dataclass — build_match renames and rounds on
 *  the way out (distance_km not distance_m, avg_position not avg_x/avg_y) and
 *  drops samples entirely. There is no hsr_km: the pipeline never emits one.
 *  Officials are filtered out before this list is built. */
export interface PlayerStats {
  id: number;
  team: Side | "unknown";
  jersey: number | null;
  minutes: number;
  /** null where stats.py REFUSED it: more metres than MAX_SPEED allows in the
   *  time the track was observed (caveat `distance_over_ceiling`). */
  distance_km: number | null;
  /** null where stats.py REFUSED it — MAX_SPEED bound on this track, so the
   *  only figure it could have published was the clamp (34.2 km/h). A refusal,
   *  not a zero. */
  top_speed_kmh: number | null;
  /** null: WITHHELD by stats.py (caveat `sprints_withheld`). Older match.json files carry a count. */
  sprints: number | null;
  /** [x, y] in metres, x along the pitch, y from the near touchline. */
  avg_position: [number, number];
  /** Share of steps that hit the 9.5 m/s clamp — the one cheap correlate of an
   *  identity switch. */
  clamped_frac: number;
  /** Share of distance windows actually measured, 0-1. */
  observed_frac?: number;
  /** Minutes this track was SEEN, as against `minutes`, which is span and counts
   *  occlusion gaps as played. Divide by this for km/90 wherever the two have
   *  parted company — 58 of 361 rows on 118575. */
  observed_minutes?: number;
  /** Motion-only role. `staff` is currently unreachable in the shipped config:
   *  the scene prior keeps detections inside the pitch plus 3 m, so anyone in a
   *  technical area is filtered before tracking. Never gates exclusion. */
  role?: "staff" | "linesman" | "uncertain";
  role_confidence?: number;
  /** motion vs kit colour: true agree, false disagree, null neither committed. */
  role_agrees?: boolean | null;
}

export interface TeamShape {
  width_m: number;
  depth_m: number;
  defensive_line_m: number;
  compactness_m2: number;
}

export interface TeamTotals {
  distance_m: number;
  /** null where stats.py REFUSED it — MAX_SPEED bound on this track, so the
   *  only figure it could have published was the clamp (34.2 km/h). A refusal,
   *  not a zero. */
  top_speed_kmh: number | null;
  /** null: WITHHELD by stats.py (caveat `sprints_withheld`). */
  sprints: number | null;
  /** Share of this side's detections `distance_m` counts (caveat `team_distance_partial`). */
  distance_coverage?: number;
  shape: TeamShape;
}

/** One line of the Facts feed: a goal, a card, a substitution, or a period
 *  marker. Human/demo-entered — the pipeline's ball rows are pixel centres,
 *  not ground positions, and are not evaluated, so none of this can come out
 *  of the video. `in`/`out` are the
 *  two halves of a substitution; `score` is the running scoreline. */
export interface MatchEvent {
  minute: string;
  side?: Side;
  type: "goal" | "yellow_card" | "substitution" | "period";
  player?: string;
  score?: string;
  assist?: string;
  in?: string;
  out?: string;
}

/** One starter in a formation. x/y are fractions inside the team's OWN half:
 *  x left to right as the team attacks, y 0 at the own goal line to 1 at the
 *  halfway line. `off` is the subbed-off minute ("73'") when it happened.
 *  Age and height come from the same entered feed — the camera measures
 *  neither. */
export interface LineupPlayer {
  n: number;
  name: string;
  pos?: string;
  x: number;
  y: number;
  rating?: number;
  off?: string;
  age?: number;
  height?: string;
}

/** A bench player. `on` is the minute they came on; absent = unused. */
export interface SubPlayer {
  n: number;
  name: string;
  pos?: string;
  rating?: number;
  on?: string;
  yellow?: boolean;
  age?: number;
  height?: string;
}

/** One dot on the shot map. x/y are fractions of the full pitch, home
 *  attacking the left goal. */
export interface ShotPoint {
  x: number;
  y: number;
  side: Side;
  goal?: boolean;
  period: 1 | 2;
}

export interface ShotDetail {
  player: string;
  minute: string;
  shot_type?: string;
  situation?: string;
  result?: string;
  xg?: number;
  xgot?: number;
}

/** A group of the full stats list. Values stay strings because several rows
 *  carry their own qualifier ("618 (88%)"). */
export interface StatSection {
  title: string;
  rows: { label: string; home: string | number; away: string | number }[];
}

/** One row of the Player stats table. Which of the flat stat fields a column
 *  reads is decided by the tab config in the report UI, so the row carries
 *  the whole set and strings pass through untouched ("41 (56%)"). */
export interface PlayerStatLine {
  name: string;
  side: Side;
  pos?: string;
  stats: Record<string, string | number>;
}

/** One moment of a match. `board` is deliberately loose here — it is parsed
 *  and narrowed by the notepad's own pitch parsers, which are the authority on
 *  what a board may contain, and duplicating that shape in this file would give
 *  us two definitions to keep in step. */
export interface Play {
  id: string;
  /** A key of PLAY_TYPES in the notepad. */
  type: string;
  side: Side;
  period: 1 | 2;
  minute: number;
  /** "40:09" — what a scouting report prints beside the fixture. */
  clock: string;
  zone: string;
  outcome: string;
  /** Shirt numbers involved. */
  players: number[];
  board: unknown;
  /** Seconds into the video, so a play can be watched back. */
  t: number;
  note?: string;
  /** "demo" until the pipeline computes these. */
  source?: string;
}

export interface MatchReport {
  meta?: {
    /** Imported from a public feed and given GENERATED tracks — a motion model
     *  stood in for a camera. A feed importer (not in this repository) writes both; a real pipeline run
     *  writes neither. Declared here because the flag was already in
     *  match.json and nothing in the frontend was reading it, so every
     *  generated distance in the demo library was published as "measured". */
    demo?: boolean;
    synthetic_tracking?: boolean;
    competition?: string;
    home_name?: string;
    away_name?: string;
    score?: string;
    status?: string;
    /** "Name minute'" strings per side, human-entered like the score — the
     *  pipeline's ball rows are pixel centres, not evaluated. stats.py does not emit this yet; it
     *  exists so the scoreboard can render scorers when a match carries them. */
    scorers?: { home?: string[]; away?: string[] };
    /** A provider-style Facts feed: team numbers the camera cannot see
     *  (shots, possession) and the event timeline. Present on the demo
     *  match, absent on real ones — when it is absent the Facts tab shows
     *  the provenance cards instead. */
    facts?: {
      top_stats?: { label: string; home: number; away: number; kind?: "possession" }[];
      events?: MatchEvent[];
      /** One pressure value per minute, + toward home / - toward away, plus
       *  the goal minutes marked on the axis. Feeds the Momentum chart. */
      momentum?: { values: number[]; goals?: { minute: number; side: Side }[] };
      lineups?: Partial<
        Record<Side, { formation: string; team_rating?: number; xi: LineupPlayer[] }>
      >;
      subs?: Partial<Record<Side, SubPlayer[]>>;
      shots?: { points: ShotPoint[]; detail?: ShotDetail };
      sections?: StatSection[];
      /** Share of attacks down each channel: [left, centre, right] as the
       *  team attacks. Home attacks the left goal, away the right. */
      attacking_zones?: { home: number[]; away: number[] };
      player_stats?: PlayerStatLine[];
      /** Moments of the match, each drawn on the board the notepad already
       *  speaks. See model/engine/plays.py — a play IS a board plus metadata, so
       *  one dropped into a note is editable rather than a picture of somebody
       *  else's analysis. Synthetic today: the pipeline's ball rows are pixel
       *  centres, not ground positions, and not evaluated, so it cannot know a
       *  moment was a corner. */
      plays?: Play[];
    };
    fps?: number;
    pitch?: { length: number; width: number };
    calibration_rms_m?: number | null;
    officials_excluded?: number;
  };
  /** A side is absent when nothing was tracked for it. */
  teams?: Partial<Record<Side, TeamTotals>>;
  /** The ONLY valid way to state a between-team shape difference, and the
   *  reason `teams[side].shape` must never be subtracted in the UI.
   *
   *  Each side's shape median is taken over the instants where THAT side was
   *  measurable, and those sets differ — `stats.py::relative_shape` records
   *  that differencing them produced +0.75 / -0.75 / +4.79 m from one clip
   *  under three groupings. This is paired on the intersection instead, so a
   *  calibration error common to both teams cancels within each sample.
   *
   *  `diff` is b minus a (positive = `b` is wider/deeper). Read it against
   *  `sd`: a mean difference far below its own spread is not a finding. */
  relative_shape?: {
    /** Paired sample count. This is the number that governs, not either side's. */
    n: number;
    possible: number;
    /** n / possible — how much of the match the comparison rests on. */
    frac: number;
    a: Side;
    b: Side;
    metrics: Partial<
      Record<
        "width_m" | "depth_m" | "rear_line_m" | "hull_area_m2" | "stretch_index_m",
        { diff: number; sd: number | null } & Record<string, number | null>
      >
    >;
  };
  /** Share of the players table's position samples per third, by pitch x: the
   *  keys read LEFT / CENTRE / RIGHT (defensive_third is x < L/3 for both sides),
   *  not either team's own thirds. Not possession. */
  territory?: {
    defensive_third: number;
    middle_third: number;
    attacking_third: number;
  };
  players?: PlayerStats[];
  /** track id (as a string) -> sampled [x, y] positions in metres. */
  heatmaps?: Record<string, [number, number][]>;
  /** The provenance envelope stats.py COMPUTED for this match, keyed by the
   *  field's dotted path. `confidence` is the resolved tier: stats.metric()
   *  takes min() over the field's registered tier and every caveat it carries,
   *  so this is the authority, not the static table in report.ts.
   *
   *  Note the key space is PER SIDE where the value is — `teams.home.distance_m`,
   *  `structure_stress.away` — while the UI asks about `teams.distance_m` for a
   *  header that covers both. tierOf() resolves that. */
  fields?: Record<string, FieldEnvelope>;
}

/** One number's provenance, as stats.metric() emits it. `value` is null on a
 *  column envelope: it governs a whole table column and the values live in the
 *  rows. */
export interface FieldEnvelope {
  value: number | null;
  units: string;
  n: number | null;
  confidence: "not_evaluated" | "ok" | "rel" | "prov";
  /** `api://<id>` in model/validation/api-golden.json that justifies `confidence`,
   *  or null when nothing does (in which case confidence is not_evaluated). */
  justified_by?: string | null;
  provenance: "measured" | "derived" | "tagged";
  caveat_flags: string[];
}

/** One clip a coach cut. Times are SECONDS into the video, never frames or
 *  track ids — re-tracking renumbers those and would orphan an evening's work. */
export interface Clip {
  id: string;
  start: number;
  end: number;
  name: string;
  tags: string[];
  note: string;
}

export interface ClipsResponse {
  clips: Clip[];
  tags: string[];
}

/** Per-track digest for the identity screen (not the full detections file). */
export interface TrackSummary {
  id: number;
  first_f: number;
  last_f: number;
  mid_f: number;
  n: number;
  team: string | null;
  jersey: number | null;
  confidence: number | null;
  seconds?: number;
}

export interface TracksDigest {
  fps: number;
  tracks: TrackSummary[];
  corrections?: Record<string, unknown>;
  [k: string]: unknown;
}

/** One human, stable across the season — `model/engine/roster.py :: Player`.
 *
 *  `team` is the CLUB (the match's `team` field), not home/away. Home and away
 *  swap every other week; a season profile keyed on them would merge two clubs.
 *
 *  `id` is minted server-side by slugging the name, never here. That slug
 *  accent-folds on purpose, so "José" and "Jose" stay one player instead of two
 *  half-seasons — reproducing it in TypeScript is how they would split again.
 */
export interface RosterPlayer {
  id: string;
  name: string | null;
  number: number | null;
  team: string | null;
  /** What a coach typed, e.g. "CB". Never derived from avg_position — sides
   *  swap at half time, so a mean x reads as midfield for everybody and the
   *  pipeline has no attacking direction to correct it with. Absent until
   *  somebody says. */
  position?: string | null;
}

/** One player's season, as `jobs.season` joins it: every match where a HUMAN
 *  confirmed which track was this person. Machine guesses are excluded there,
 *  not here — see that function for why. */
export interface SeasonPlayer {
  id: string;
  name: string | null;
  number: number | null;
  position: string | null;
  team: string | null;
  matches: number;
  minutes: number;
  distance_km: number;
  /** Always null: WITHHELD (SPRINTS_WITHHELD_WHY), never summed, whatever old match.json holds. */
  sprints: number | null;
  /** The best single reading of the season, never a sum or a mean.
   *  `null` where every match that would have contributed was REFUSED - the
   *  clamp bound, so 34.2 km/h would read as an elite sprint while meaning
   *  "this track teleported". Render "refused", never 0. */
  top_speed_kmh: number | null;
  km_per90: number;
  /** Always null: withheld, as `sprints`. */
  sprints_per90: number | null;
  by_match: Record<
    string,
    {
      minutes: number;
      distance_km: number;
      /** Always null: withheld. */
      sprints: number | null;
      /** null = refused for this match; see the season-level field above. */
      top_speed_kmh: number | null;
      side: string;
      jersey: number | null;
    }
  >;
}

export interface Season {
  team: string;
  matches: {
    id: string;
    name: string;
    created: number | null;
    home_name: string | null;
    away_name: string | null;
    score: string | null;
  }[];
  players: SeasonPlayer[];
  /** `bound` of `total` tracked rows across every match could be attributed to
   *  a person. The gap is what the table is NOT standing on, and every caller
   *  has to say so rather than presenting a partial total as a whole one. */
  coverage: { bound: number; total: number };
}

/** The full per-frame detections the video overlay draws. Tens of MB — always
 *  parsed in a worker, never on the main thread. */
export interface Detection {
  f: number;
  id: number;
  x: number;
  y: number;
  team?: string;
  jersey?: number | null;
}

export interface TracksFile {
  fps: number;
  detections: Detection[];
}

/** calibration.json, as pitch.Calibration.to_dict() writes it and
 *  jobs.load_calibration reads it back. `pitch` is the Pitch dataclass's own
 *  __dict__, so the keys are the Python field names in METRES — there is no
 *  `length_m`/`width_m` and never was. */
export interface Calibration {
  projection_type?: "perspective" | "remap" | "panoramic" | "spherical";
  /** Only on a "spherical" (automatic) calibration: the pitch drawing as the
   *  found camera sees it, in frame pixels, one polyline per line family. */
  auto?: { overlay_px?: Record<string, [number, number][]> } | null;
  H: number[][];
  pitch?: {
    length: number;
    width: number;
    penalty_depth: number;
    penalty_width: number;
    goal_box_depth: number;
    goal_box_width: number;
    centre_circle_r: number;
  };
  /** Image-space clicks, keyed by landmark name. */
  points?: Record<string, [number, number]>;
  rms_error_m?: number | null;
  outliers?: string[];
  [k: string]: unknown;
}

/** A named set of stat columns, saved for the project rather than for a note.
 *  `cols` are measure ids, which are also the report table's column keys —
 *  FEED_MEASURES is built from PLAYER_STAT_COLS, so the two vocabularies are
 *  one and a preset means the same thing in a chart and in a table. */
export interface StatPreset {
  name: string;
  cols: string[];
  sort?: string;
}

// ---------------------------------------------------------------------------
// THE CLUB — `model/engine/org.py`. An academy is not one team: it is a ladder of
// squads (U9 ... U19, reserves, first team), each with its own staff, and ONE
// player who climbs it over five years. That last part is the product, which is
// why `ClubPlayer` below is keyed on the human and carries their teams, rather
// than a team carrying a list of people.

/** One squad. `team_id` is also the roster filename, so a team and its players
 *  cannot drift apart — and it therefore survives a rename, which is why the id
 *  is never re-derived from the name on the client. */
export interface Team {
  team_id: string;
  name: string;
  /** "U15", or "" for a senior side. Parsed from the name on create and then
   *  stored, because a club that calls a squad "Infantil A" still wants it on
   *  the U13 rung and only a human knows that. */
  age_band: string;
  gender: string;
  /** "A" | "B" — the second axis under one age band. */
  level: string;
  season: string;
  coach_ids: string[];
  /** Joined server-side from what is actually on disk: a squad with no players
   *  and no footage is exactly the one a coach needs to see is empty. */
  players: number;
  matches: number;
}

/** Staff. A RECORD, NOT AN ACCOUNT — there is no authentication anywhere in
 *  this product and nothing here is a security boundary. See TODOS (not published). */
export interface Coach {
  coach_id: string;
  name: string;
  role: string;
  email: string;
  team_ids: string[];
}

export interface Org {
  club: { club_id: string; name: string; season: string };
  /** Youngest first, seniors last — the ladder order, decided server-side so
   *  every surface agrees on it. Alphabetical would put U9 after U15. */
  teams: Team[];
  coaches: Coach[];
}

/** One HUMAN, across every team in the club — the career view, and the thing an
 *  academy actually buys. Distinct from `RosterPlayer`, which is one row per
 *  (player, team) because the correction screen is fixing one match.
 *
 *  A player carrying two wildly separated teams is the visible shape of the one
 *  identity limit this inherits: two different humans with the same name share
 *  an id (roster.py, BUG 1). Shown rather than hidden. */
export interface ClubPlayer {
  id: string;
  name: string | null;
  position: string | null;
  teams: { team_id: string; team: string | null; number: number | null; position: string | null }[];
  /** Matches with a HUMAN-CONFIRMED track binding. A machine guess is not a
   *  season profile — see jobs.season. */
  matches: number;
  /** Whether the photo store has a headshot for this id, so a row can draw the
   *  silhouette without firing a request that 404s. `/api/players` has always
   *  sent it; nothing read it until the registry started showing faces. */
  photo?: boolean;
}

// ---------------------------------------------------------------------------
// The store. `data/match.db` is a query layer over match.json and org.json, so
// it is rebuildable and served read-only.

export interface DbColumn {
  name: string;
  type: string;
  pk: boolean;
}

export interface DbTable {
  name: string;
  kind: "table" | "view";
  columns: DbColumn[];
  rows: number | null;
}

/** ONE APPEARANCE - a player on a team sheet for one match, with the numbers
 *  the FEED supplied for it.
 *
 *  ENTERED, not measured, and the distinction is the whole reason this is a
 *  separate type from `SeasonPlayer`. These are a provider's numbers: goals,
 *  assists, the rating somebody assigned. `SeasonPlayer` is what the pipeline
 *  computed off tracks. They disagree and are both right - the feed has Joao
 *  Pedro at 851 minutes and the measured season at 900, because on an imported
 *  fixture the tracking is synthetic and every synthetic track runs the full
 *  match. Mixing them into one row would publish the fabricated number as the
 *  real one, which is the bug `meta.demo` was added to stop.
 *
 *  Keyed on player_id, so it joins the registry without guessing at a name:
 *  db.py's `_ingest_squad` mints the same accent-folded slug roster.py does.
 */
export interface Appearance {
  player_id: string;
  match_id: string;
  /** Which side of the match the player was on. */
  side: "home" | "away";
  name: string;
  /** The feed's minutes for this appearance, NOT the tracked ones. */
  minutes: number | null;
  /** Entered by the provider. An opinion, not a measurement. */
  rating: number | null;
  age: number | null;
  shirt: number | null;
  goals: number | null;
  assists: number | null;
  /** Keeper-only, and PRESENT-BUT-ZERO is the normal case for a keeper who had
   *  a quiet game. `null` means the feed filed no such metric for this
   *  appearance, which is how an outfielder is told apart from a keeper who
   *  made no saves - so these must be checked for null, never truthiness. */
  saves: number | null;
  conceded: number | null;
  /** The match's competition, season included ("Premier League 2025/2026"). */
  competition: string | null;
  /** Cards shown to this player in this match, from the event feed. */
  yellow: number;
  red: number;
}

/** WHO A PLAYER IS, as the feed has it. Height, foot, country, contract,
 *  market value, and the positions he is actually played in.
 *
 *  ENTERED, every field, and none of it derivable from footage - which is why
 *  it is a separate type and a separate route from anything the pipeline
 *  produced. A card may print these beside a measured distance; it may never
 *  let them read as the same kind of number.
 *
 *  EVERY FIELD IS OPTIONAL, and that is measured rather than defensive. Over
 *  the 228 players cached from ten matches: age, foot and country are on all
 *  of them; market value 226, height 225, shirt 216, contract end 190. So a
 *  card that assumes any of them draws a label over an empty space.
 */
export interface PlayerBio {
  fotmob_id?: number | null;
  name?: string | null;
  team?: string | null;
  on_loan?: boolean | null;
  /** The club's own hex, for the card's header band. */
  team_color?: string | null;
  birth_date?: string | null;
  age?: number | null;
  height_cm?: number | null;
  shirt?: number | null;
  foot?: string | null;
  country?: string | null;
  /** Three-letter code, for the flag. */
  country_code?: string | null;
  market_value?: number | null;
  /** The feed's own rendering ("€91.9m"), so nothing here guesses a currency. */
  market_value_str?: string | null;
  contract_end?: string | null;
  primary_position?: string | null;
  other_positions?: string[];
  /** The cohort the percentiles below are against - "attacking midfielders/
   *  wingers". Never render a trait without it: 87% is meaningless until you
   *  know 87% of whom. */
  traits_group?: string | null;
  /** Percentile 0-1 against that cohort, in the feed's own order, which is
   *  clockwise from the top-right of the hexagon. Two shapes exist and both
   *  are six axes - one for outfielders, one for keepers. */
  traits?: { key?: string | null; label?: string | null; value?: number | null }[];
  /** Touch positions in METRES on a 105x68 pitch, as the feed sends them.
   *
   *  FOR `heatmap_season`, which is the season the library was imported for
   *  and NOT the feed's current one - playerData ships a heatmap of its own
   *  and it is whatever season is running now. It also covers the player's
   *  whole season in that competition, which is more matches than the library
   *  holds, so `heatmap_matches` travels with it and has to be shown. */
  heatmap?: { x: number; y: number }[];
  /** Every shot of `heatmap_season`, in the same 105x68 metres.
   *
   *  `on_target` is the feed's own flag and is NOT `goal`: a save is on target
   *  too, which is the entire point of the percentage. Own goals are filtered
   *  out upstream - the feed files them in this list and counting one credits
   *  a defender with a goal at the wrong end. */
  shots?: {
    x: number;
    y: number;
    xg: number;
    goal: boolean;
    on_target: boolean;
    blocked: boolean;
    /** Where the shot crossed the goal line, in the same metres: the posts
     *  stand at y 30.34 and 37.66, the bar at z 2.44. */
    goal_y?: number | null;
    goal_z?: number | null;
    /** Expected goals ON TARGET. Only meaningful for a shot that reached the
     *  frame, so it is null off target rather than 0 - a zero there would read
     *  as "a chance worth nothing" instead of "never tested the keeper". */
    xgot?: number | null;
    /** The day it was played, YYYY-MM-DD. This is what joins a season shot to
     *  a fixture in the library: the shot carries the feed's match id and the
     *  library keys on an md5 of it, so there is no shared id, and nobody
     *  plays twice in a day. */
    match_date?: string | null;
    min?: number | null;
    situation?: string | null;
    foot?: string | null;
    opponent?: string | null;
  }[];
  heatmap_season?: string | null;
  heatmap_matches?: number | null;
  /** Minutes played in `heatmap_season`, for the performance table's header. */
  season_minutes?: number | null;
  /** THE SEASON PERFORMANCE TABLE, in the feed's own groups and order -
   *  Shooting, Physical, Passing, Possession, Defending, Discipline.
   *
   *  `rank` is a percentile ALREADY ORIENTED so higher means better, which was
   *  verified and not assumed: red cards 0 ranks 100, yellow cards 5 ranks 12,
   *  goals conceded 33 ranks 17. The feed inverts it for the stats where more
   *  is worse, which is what makes a red-to-green ramp on it the feed's claim
   *  rather than one invented here. */
  stat_groups?: {
    title?: string | null;
    stats: {
      key?: string | null;
      label?: string | null;
      value?: number | null;
      per90?: number | null;
      rank?: number | null;
      rank90?: number | null;
      /** number | fraction | percent | kph | meter */
      format?: string | null;
    }[];
  }[];
  /** Season touch count. Absent for keepers - the feed files them under a
   *  different stat set - so a heatmap can exist with no number beside it. */
  touches?: number | null;
  /** One per position actually played, with how often and where on the pitch
   *  the feed puts it. `x`/`y` are 0-1 fractions and are null for a position
   *  the feed gives no coordinates for. */
  positions?: {
    label?: string | null;
    short?: string | null;
    main?: boolean;
    played?: number | null;
    x?: number | null;
    y?: number | null;
  }[];
}

/** One team stat in one match: the number, the feed's own text for it
 *  ("343 (81%)", "52%") and the percentage part where it has one. */
export interface TeamStat {
  value: number | null;
  raw: string | null;
  pct: number | null;
}

/** One match's own story, from the store. */
export interface MatchInfo {
  status: string | null;
  /** In order. `who` is the scorer, the carded player or the sub coming on;
   *  `other` the assist or the player going off. A period row is the
   *  half-time marker and carries the score at it. */
  events: {
    minute: number | null;
    side: "home" | "away" | null;
    type: string;
    who: string | null;
    other: string | null;
    score: string | null;
  }[];
  /** Positive leans home, negative away; empty where the feed sent none. */
  momentum: { minute: number; value: number }[];
}

/** One side of one match, from the store. */
export interface MatchSide {
  match_id: string;
  side: "home" | "away";
  club_id: string;
  formation: string | null;
  team_rating: number | null;
  /** Which of our squads the match belongs to; null for a match with none. */
  team_id: string | null;
}

/** One player on a team sheet. `x`/`y` are the feed's plot, fractions of the
 *  team's own end: y from its goal line up, x across. Null on the bench. */
export interface LineupRow {
  player_id: string;
  name: string;
  shirt: number | null;
  position: string | null;
  started: boolean;
  /** The minute a sub came on; null for a starter or an unused sub. */
  on_minute: number | null;
  rating: number | null;
  x: number | null;
  y: number | null;
}

export interface DbResult {
  columns: string[];
  rows: unknown[][];
  /** The answer was cut off at the row cap. Says so rather than presenting a
   *  truncated result as the whole one. */
  truncated: boolean;
}

/** One row of the note library — a note, or a folder holding notes.
 *
 *  A note has no match id and no team id on purpose: the automated report is
 *  what belongs to one match, and a note is the thing written across several.
 *  Its `path` inside the library IS its identity, so a rename is a move and the
 *  folder tree is the organisation. See app/server/jobs.py's note store. */
export interface NoteEntry {
  path: string;
  dir: boolean;
  /** Files only: epoch seconds, bytes on disk, and the opening prose with the
   *  markup taken out — what the library page prints on a card. */
  updated?: number;
  bytes?: number;
  snippet?: string;
  /** The top of the note's own markup, cut on a tag boundary — what the card
   *  renders, in a sandboxed frame where the browser refuses scripts and
   *  network fetches outright. See jobs.py's _head. */
  head?: string;
}

/** GET /api/matches/{id}/pitch/osm. `usable` is the length gate: about half of
 *  OpenStreetMap's pitches are traced round the grass, and only a length that
 *  disagrees with the painted one gives that away. */
export interface OsmPitches {
  painted_length_m: number | null;
  attribution: string;
  pitches: {
    way: number;
    length: number;
    width: number;
    distance_m: number;
    name: string | null;
    surface: string | null;
    usable: boolean;
    why: string;
  }[];
}
