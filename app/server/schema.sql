-- Fut Traffic match store.
--
-- WHY A DATABASE NOW, when roster.py says a database "buys nothing at that size
-- and costs a deployment dependency". That comment is about ROSTERS - tens of
-- thousands of rows, read whole, written whole. It is still true and roster.py
-- still owns the roster.
--
-- What changed is the assumption this file is built on: that the match came from
-- our model and we have a programmatic 2D map of the whole game. A 2D map is
-- ~25 samples/s x ~23 objects x 5400 s = ~3.1 M rows PER MATCH. The questions a
-- coach asks of it - "where was everyone at 63:12", "every time 8 received in
-- zone 14", "this player's last six matches" - are range scans over a join, not
-- a file read. match.json is already 830 KB with the map DOWNSAMPLED to 600
-- points per track, and jobs.py had to grow an lru_cache because re-parsing it
-- per request moved tens of megabytes.
--
-- The deployment-dependency objection survives intact: sqlite3 is in the Python
-- standard library. No server, no driver, no container. One file on disk, same
-- as today, except it can be queried.
--
-- ---------------------------------------------------------------------------
-- THE THREE GRAINS, and why they are not one table.
--
--   MEASURED   a number the model computed from the 2D map (a track's distance,
--              a team's hull area). Keyed on a TRACK, never on a player.
--   ENTERED    a number a human typed or a feed supplied (xG, ratings, ages,
--              the whole of meta.facts today). Keyed on a PLAYER.
--   BOUND      a measured number reached through a track -> player binding, so
--              it is only as good as the binding.
--
-- The product's central honesty rule is that MEASURED and ENTERED never share a
-- row, because they do not share a key: the feed knows names, the model knows
-- track ids, and joining them is a human act with a confidence attached. So:
-- track_metric is measured, player_metric is entered, and the only bridge is
-- track.player_id - nullable, with the binding's source recorded. If nobody has
-- bound the track, the physical numbers have no player and the view says so
-- rather than guessing.
--
-- ---------------------------------------------------------------------------
-- STRICT everywhere: SQLite's default affinity would happily store the string
-- "618 (88%)" in a REAL column as text, which is the exact bug class this
-- schema exists to kill - a qualified feed value silently surviving as a
-- number. STRICT makes it an error at insert.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ===========================================================================
-- Stable entities. These outlive any one match; everything else hangs off them.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS competition (
  competition_id TEXT PRIMARY KEY,          -- slug: 'leagues-cup-2025'
  name           TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS club (
  club_id TEXT PRIMARY KEY,                 -- slug: 'cf-america'
  name    TEXT NOT NULL,
  -- 1 for the academy this install belongs to, 0 for everyone we played. There
  -- is exactly one "us" (org.py) and the difference is not cosmetic: every
  -- season question a coach asks is "ours vs theirs", and an opponent club
  -- arrives here as a NAME off a team sheet with no squad and no footage
  -- behind it. Without this flag the two are indistinguishable rows and the
  -- club list fills up with every side we have ever faced.
  is_ours INTEGER NOT NULL DEFAULT 0
) STRICT;

-- ===========================================================================
-- THE LADDER. A club is not one team - an academy is U9, U11, U13 ... reserves,
-- first team, each with its own staff and its own squad, and ONE player who
-- climbs it over five years. That last part is the product: a career, not a
-- squad list. So the player is keyed on the CLUB (above) and the team is a
-- membership (below), never the other way round. A player_id that carried its
-- team would split one human into a new person every August.
--
-- Written by org.py, ingested here. Like every other table in this file, this
-- is a QUERY LAYER: `data/org.json` is the source of truth and the store is
-- rebuildable from it. Nothing writes a team here except ingest.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS team (
  team_id TEXT PRIMARY KEY,                 -- slug; ALSO the roster file name
  club_id TEXT NOT NULL REFERENCES club(club_id),
  name    TEXT NOT NULL,                    -- 'U15 Boys', 'First team'
  age_band TEXT,                            -- 'U15'; NULL/'' for a senior side
  -- The ladder order, computed once by org.age_key. Stored rather than derived
  -- because every naive sort of an age band is wrong - alphabetically U9 comes
  -- after U15 - and a view that gets it wrong is a coach's team list in a
  -- nonsense order with nothing to point at.
  age_key INTEGER NOT NULL DEFAULT 99,
  gender  TEXT,                             -- 'boys' | 'girls' | 'mixed' | ''
  level   TEXT,                             -- 'A' | 'B' - second axis under a band
  season  TEXT
) STRICT;

CREATE INDEX IF NOT EXISTS team_by_ladder ON team(club_id, age_key, level);

-- Staff. A RECORD, NOT AN ACCOUNT: there is no authentication anywhere in this
-- product and nothing here is a security boundary. See docs/TODOS.md, which gates
-- auth on the first deployment a second person can reach.
CREATE TABLE IF NOT EXISTS coach (
  coach_id TEXT PRIMARY KEY,
  club_id  TEXT NOT NULL REFERENCES club(club_id),
  name     TEXT NOT NULL,
  role     TEXT,                            -- 'Head coach', 'Assistant', 'GK'
  email    TEXT
) STRICT;

-- Many-to-many on purpose. One coach runs two age groups and one team has a
-- head, an assistant and a keeper coach; a column on either side would force
-- the club to lie about whichever case it met second.
CREATE TABLE IF NOT EXISTS team_coach (
  team_id  TEXT NOT NULL REFERENCES team(team_id) ON DELETE CASCADE,
  coach_id TEXT NOT NULL REFERENCES coach(coach_id) ON DELETE CASCADE,
  PRIMARY KEY (team_id, coach_id)
) STRICT;

CREATE INDEX IF NOT EXISTS team_coach_by_coach ON team_coach(coach_id);

-- One human, stable across the season. Mirrors roster.Player so a roster.json
-- can be loaded straight in; roster.py stays the write path for identity work.
CREATE TABLE IF NOT EXISTS player (
  player_id TEXT PRIMARY KEY,               -- roster.py's accent-folded slug
  club_id   TEXT NOT NULL REFERENCES club(club_id),
  name      TEXT,
  number    INTEGER,
  -- Free text a coach types once. NEVER derived from avg_position: sides swap
  -- at half time, so a mean x reads as midfield for everyone. See roster.py.
  position  TEXT
) STRICT;

CREATE INDEX IF NOT EXISTS player_by_club ON player(club_id);

-- SQUAD MEMBERSHIP - the ladder join, and the row that makes a career queryable.
--
-- One row per (player, team, season). The shirt number lives HERE and not on
-- `player` because it is a property of the membership: the same boy is 7 for
-- the U15s and 24 for the reserves in the same month, and a number on the
-- player would have to pick one and be wrong about the other. Position is the
-- same - a coach's season-long "RB" for one squad is not the other's.
--
-- Season is part of the key rather than a date range. A squad list is decided
-- once a year, so "when exactly did he move up" is a question no club can
-- answer to the day and every club can answer to the season; a from/to pair
-- would be two more columns nobody could fill honestly.
CREATE TABLE IF NOT EXISTS squad (
  team_id   TEXT NOT NULL REFERENCES team(team_id) ON DELETE CASCADE,
  player_id TEXT NOT NULL REFERENCES player(player_id) ON DELETE CASCADE,
  season    TEXT NOT NULL DEFAULT '',
  number    INTEGER,
  position  TEXT,
  PRIMARY KEY (team_id, player_id, season)
) STRICT;

CREATE INDEX IF NOT EXISTS squad_by_player ON squad(player_id);

-- ===========================================================================
-- The match, and the two sides of it.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS match (
  match_id       TEXT PRIMARY KEY,          -- the job id
  competition_id TEXT REFERENCES competition(competition_id),
  -- WHICH OF OUR TEAMS THIS IS. Not the same question as match_side.club_id:
  -- that says who played, this says whose match it is in our club - the U15s'
  -- or the reserves'. Both sides of a fixture can even be ours in a practice
  -- game, which is why the answer cannot be read off the sides.
  --
  -- Nullable, and it stays that way: a match uploaded before the ladder
  -- existed has no team, and guessing one from the club name would attribute a
  -- whole season to whichever squad shares the club's name.
  -- SET NULL, not RESTRICT: ingest_org replaces the ladder wholesale on every
  -- rebuild, and a match holding a team down would make deleting a squad from
  -- org.json fail with a foreign-key error instead of doing the obvious thing.
  -- The match is then re-attributed by set_match_team from the job, which is
  -- where the answer actually lives.
  team_id        TEXT REFERENCES team(team_id) ON DELETE SET NULL,
  season         TEXT,                      -- entered; the team's at upload time
  kickoff_utc    TEXT,                      -- ISO 8601; NULL until someone says
  status         TEXT,                      -- 'Full time', 'Half time', ...
  fps            REAL NOT NULL,             -- frame -> seconds for every position row
  pitch_length_m REAL NOT NULL,
  pitch_width_m  REAL NOT NULL,
  -- Calibration residual in metres. The one number that bounds how much any
  -- spatial claim below is worth; 0.0 means synthetic.
  calibration_rms_m REAL,
  officials_excluded INTEGER NOT NULL DEFAULT 0
) STRICT;

CREATE TABLE IF NOT EXISTS match_side (
  match_id TEXT NOT NULL REFERENCES match(match_id) ON DELETE CASCADE,
  side     TEXT NOT NULL CHECK (side IN ('home', 'away')),
  club_id  TEXT NOT NULL REFERENCES club(club_id),
  formation   TEXT,                         -- entered, e.g. '4-2-3-1'
  team_rating REAL,                         -- entered
  -- Derived by stats.infer_attacking_right. FIRST HALF ONLY - sides swap, so
  -- the second half is its negation. One column instead of a period table
  -- because the swap is a rule of the game, not data.
  attacking_right_p1 INTEGER,
  -- stats.series_coverage: how much of the shape series actually got emitted.
  -- Carried here so any shape query can say what it is averaging over.
  shape_possible    INTEGER,
  shape_emitted     INTEGER,
  shape_min_players INTEGER,
  -- stats.stress_profile
  stress_gini    REAL,
  stress_profile TEXT,                      -- 'flat' | 'peaked' | ...
  PRIMARY KEY (match_id, side)
) STRICT;

-- ===========================================================================
-- Tracks: what the model actually produced. The measured half of the world.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS track (
  match_id TEXT NOT NULL REFERENCES match(match_id) ON DELETE CASCADE,
  track_id INTEGER NOT NULL,
  -- The 2D map is of the WHOLE game, so it has objects that are not players.
  -- stats.py emits players only today; 'ball' and 'referee' are here because
  -- every ball-derived table below is meaningless without a ball track.
  kind     TEXT NOT NULL DEFAULT 'player'
           CHECK (kind IN ('player', 'goalkeeper', 'referee', 'ball')),
  side     TEXT CHECK (side IN ('home', 'away')),   -- NULL for ball/referee
  jersey   INTEGER,                                  -- read off the shirt, may be wrong
  -- THE BINDING. Nullable on purpose: an unbound track is the normal state
  -- after a run, and every physical number it carries stays anonymous until a
  -- human says whose it is. Never infer this from jersey alone.
  player_id  TEXT REFERENCES player(player_id),
  bind_confidence REAL,
  bind_source TEXT CHECK (bind_source IN ('manual', 'auto', 'carried')),
  -- Where the position rows for this track came from. 'frames' is the real 2D
  -- map; 'heatmap_sample' is the 600-point downsample match.json carries when
  -- the raw tracks file is gone. Any density or dwell query MUST check this.
  position_source TEXT NOT NULL DEFAULT 'frames'
           CHECK (position_source IN ('frames', 'heatmap_sample')),
  PRIMARY KEY (match_id, track_id)
) STRICT;

CREATE INDEX IF NOT EXISTS track_by_player ON track(player_id);

-- THE 2D MAP. One row per object per frame, in pitch metres.
--
-- Clustered by (match_id, frame, track_id) because the dominant read is a
-- FRAME SLICE - every object at time t - which is what a 2D playback, a play
-- board, a pressure calculation and a possession fit all want. The secondary
-- index serves the other shape, one object's whole trajectory.
--
-- Metres, not pixels: the pixel positions are an artefact of one camera rig and
-- die with it. The calibration is applied once, on the way in, so nothing
-- downstream needs to know a homography existed.
CREATE TABLE IF NOT EXISTS position (
  match_id TEXT NOT NULL,
  frame    INTEGER NOT NULL,
  track_id INTEGER NOT NULL,
  x_m      REAL NOT NULL,
  y_m      REAL NOT NULL,
  -- Speed at stats.DISTANCE_WINDOW_S, not an instantaneous difference. Stored
  -- rather than derived because that window is a calibrated constant with a
  -- long argument behind it (see stats.py) and recomputing it per query would
  -- let two callers disagree about the same player's top speed.
  v_ms     REAL,
  -- Set when the position had to be clamped back onto the pitch. A dwell map
  -- that counts clamped points reports a touchline the player never stood on.
  clamped  INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (match_id, frame, track_id),
  FOREIGN KEY (match_id, track_id) REFERENCES track(match_id, track_id) ON DELETE CASCADE
) STRICT, WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS position_by_track ON position(match_id, track_id, frame);

-- Seconds, for everything that thinks in clock time rather than frames.
CREATE VIEW IF NOT EXISTS position_t AS
SELECT p.*, p.frame / m.fps AS t_s
FROM position p JOIN match m USING (match_id);

-- Who had the ball, and when. EMPTY until the pipeline has a ball - possession.py
-- exists but stats.py is player-positions-only by design. Modelled now because
-- passes, pressure and every "in possession" filter are spans over this table,
-- and bolting it on later means rewriting those queries.
CREATE TABLE IF NOT EXISTS possession (
  match_id TEXT NOT NULL REFERENCES match(match_id) ON DELETE CASCADE,
  t0_s     REAL NOT NULL,
  t1_s     REAL NOT NULL,
  side     TEXT NOT NULL CHECK (side IN ('home', 'away')),
  track_id INTEGER,                          -- the carrier, when known
  PRIMARY KEY (match_id, t0_s)
) STRICT;

-- ===========================================================================
-- Appearances: a player's involvement in one match. The season join lives here.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS appearance (
  match_id  TEXT NOT NULL REFERENCES match(match_id) ON DELETE CASCADE,
  player_id TEXT NOT NULL REFERENCES player(player_id),
  side      TEXT NOT NULL CHECK (side IN ('home', 'away')),
  shirt     INTEGER,
  -- Entered by whoever typed the team sheet. Coarser than roster position and
  -- kept separate from it: a coach's season-long "RB" is not this match's role.
  position  TEXT,
  started   INTEGER NOT NULL DEFAULT 0,
  on_minute  INTEGER,                        -- NULL when started
  off_minute INTEGER,
  minutes   REAL,
  rating    REAL,                            -- entered
  age       INTEGER,
  height_cm INTEGER,                         -- normalised; the feed says 6'5"
  -- The formation-diagram slot, 0-1 across the half the side defends. This is
  -- a DRAWING, not a measurement - it is where the team sheet puts the dot, and
  -- it must never be confused with a mean position out of the 2D map.
  plot_x REAL,
  plot_y REAL,
  PRIMARY KEY (match_id, player_id)
) STRICT;

CREATE INDEX IF NOT EXISTS appearance_by_player ON appearance(player_id);

-- ===========================================================================
-- Metrics. Long, not wide, in all three tables below.
--
-- One column per metric would be ~31 columns for players and ~43 for teams
-- TODAY, and the pipeline that fills them is still being written - so it would
-- also be a migration per new metric. Long form means a new metric is a new
-- row, and "compare any players over any factors across any matches", which is
-- what the note builder asks for, is one query instead of dynamic SQL over a
-- column list.
--
-- WHY value/attempted/pct AND raw: the feed emits "618 (88%)", "6/12 (50%)" and
-- "72%" in the same column as plain numbers. Something has to decompose them or
-- every consumer re-invents the parse - and the frontend already shipped a bug
-- where "4-2-3-1" was read as the number 4. Decompose once, on the way in, and
-- keep the raw string so nothing is lost.
-- ===========================================================================

-- MEASURED. Keyed on a track, so it is honest before anyone has done identity.
CREATE TABLE IF NOT EXISTS track_metric (
  match_id TEXT NOT NULL,
  track_id INTEGER NOT NULL,
  metric   TEXT NOT NULL,                    -- 'distance_km', 'top_speed_kmh', ...
  value    REAL,
  PRIMARY KEY (match_id, track_id, metric),
  FOREIGN KEY (match_id, track_id) REFERENCES track(match_id, track_id) ON DELETE CASCADE
) STRICT;

-- ENTERED. Keyed on a player, because that is what the feed knows.
CREATE TABLE IF NOT EXISTS player_metric (
  match_id  TEXT NOT NULL REFERENCES match(match_id) ON DELETE CASCADE,
  player_id TEXT NOT NULL REFERENCES player(player_id),
  metric    TEXT NOT NULL,
  value     REAL,
  attempted REAL,                            -- the denominator of "6/12"
  pct       REAL,
  raw       TEXT,
  PRIMARY KEY (match_id, player_id, metric)
) STRICT;

CREATE INDEX IF NOT EXISTS player_metric_by_metric ON player_metric(metric, player_id);

CREATE TABLE IF NOT EXISTS team_metric (
  match_id TEXT NOT NULL REFERENCES match(match_id) ON DELETE CASCADE,
  -- 'both' is for the genuinely match-level numbers - territory shares, the
  -- home-vs-away spread in relative_shape - which are not either side's.
  side     TEXT NOT NULL CHECK (side IN ('home', 'away', 'both')),
  metric   TEXT NOT NULL,
  value    REAL,
  attempted REAL,
  pct      REAL,
  raw      TEXT,
  section  TEXT,                             -- 'Passes', 'Duels', ... display grouping
  ord      INTEGER NOT NULL DEFAULT 0,       -- order within the section
  -- The four the report puts at the top. A flag, not a second table, because
  -- top_stats is the same metrics as sections - the demo just stores them twice.
  featured INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (match_id, side, metric)
) STRICT;

CREATE INDEX IF NOT EXISTS team_metric_by_metric ON team_metric(metric);

-- Measured numbers presented as a player's, WITH the binding that got them
-- there. Any consumer that wants physicals per player goes through this, so the
-- binding's source and confidence travel with the number and an unbound track
-- simply is not in the result rather than being silently attributed.
CREATE VIEW IF NOT EXISTS player_track_metric AS
SELECT t.match_id, t.player_id, tm.metric, tm.value,
       t.track_id, t.bind_confidence, t.bind_source
FROM track_metric tm
JOIN track t USING (match_id, track_id)
WHERE t.player_id IS NOT NULL;

-- ===========================================================================
-- Events. One table, because a goal, a card, a substitution and a shot are the
-- same shape: something happened, at a time, on a side, to a player, somewhere.
-- The demo stores these four times over (facts.events, facts.shots,
-- facts.subs, momentum.goals) and they drift.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS event (
  event_id INTEGER PRIMARY KEY,
  match_id TEXT NOT NULL REFERENCES match(match_id) ON DELETE CASCADE,
  period   INTEGER,
  minute   INTEGER,
  t_s      REAL,                             -- exact clock when the map gives one
  side     TEXT CHECK (side IN ('home', 'away')),
  type     TEXT NOT NULL,                    -- goal|own_goal|shot|yellow_card|red_card|sub|pass|duel|...
  player_id TEXT REFERENCES player(player_id),
  -- Position on the pitch in METRES, always in the acting side's own frame of
  -- reference is EXACTLY WHAT THIS IS NOT. Stored in absolute pitch coordinates
  -- with the home goal at x=0, because a feed that keeps each side in its own
  -- frame needs the home rows point-reflected before any pitch join, and doing
  -- that at read time is how it gets forgotten.
  x_m REAL,
  y_m REAL,
  outcome TEXT,                              -- 'Goal', 'Saved', 'Off target', 'cleared', ...
  body    TEXT,                              -- 'Right foot', 'Header', ...
  situation TEXT,                            -- 'Regular play', 'Corner', ...
  xg    REAL,
  xgot  REAL,
  detail TEXT                                -- JSON, for kind-specific extras
) STRICT;

CREATE INDEX IF NOT EXISTS event_by_match ON event(match_id, t_s);
CREATE INDEX IF NOT EXISTS event_by_type ON event(type, match_id);
CREATE INDEX IF NOT EXISTS event_by_player ON event(player_id, type);

-- Everyone involved, in their role. An assist, the player coming off, both
-- sides of a duel - all the same relation, so none of them needs its own column
-- and a fifth participant does not need a migration.
CREATE TABLE IF NOT EXISTS event_player (
  event_id  INTEGER NOT NULL REFERENCES event(event_id) ON DELETE CASCADE,
  player_id TEXT NOT NULL REFERENCES player(player_id),
  role      TEXT NOT NULL,                   -- 'actor'|'assist'|'on'|'off'|'opponent'|'involved'
  PRIMARY KEY (event_id, player_id, role)
) STRICT;

-- ===========================================================================
-- Plays: a window of the 2D map, saved as a board a coach can draw on.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS play (
  play_id  TEXT NOT NULL,
  match_id TEXT NOT NULL REFERENCES match(match_id) ON DELETE CASCADE,
  type     TEXT NOT NULL,                    -- positional_attack|counter|set_piece|press|...
  side     TEXT CHECK (side IN ('home', 'away')),
  period   INTEGER,
  minute   INTEGER,
  clock    TEXT,                             -- '9:33', as displayed
  t0_s     REAL NOT NULL,
  t1_s     REAL,                             -- NULL while a play is a moment, not a span
  zone     TEXT,                             -- final_third|middle_third|own_third|box
  outcome  TEXT,
  note     TEXT,
  -- 'model' once the pipeline cuts these out of the map itself; 'demo' and
  -- 'manual' until then. A coach must be able to tell which is which.
  source   TEXT NOT NULL DEFAULT 'model'
           CHECK (source IN ('model', 'manual', 'demo')),
  -- The whiteboard document: magnets, shapes, view, orient, lines, grass. JSON
  -- rather than tables because it is a DRAWING and the app owns its format -
  -- nothing here ever queries inside it. The queryable facts are the columns
  -- above and the players below.
  board    TEXT NOT NULL,
  -- The event this play produced, when it produced one.
  event_id INTEGER REFERENCES event(event_id) ON DELETE SET NULL,
  PRIMARY KEY (match_id, play_id)
) STRICT;

CREATE INDEX IF NOT EXISTS play_by_type ON play(match_id, type);

-- Derived from the board's own discs, never sampled independently: filtering
-- for "number 19 involved" must not return a board with no 19 on it.
CREATE TABLE IF NOT EXISTS play_player (
  match_id TEXT NOT NULL,
  play_id  TEXT NOT NULL,
  shirt    INTEGER NOT NULL,
  player_id TEXT REFERENCES player(player_id),
  PRIMARY KEY (match_id, play_id, shirt),
  FOREIGN KEY (match_id, play_id) REFERENCES play(match_id, play_id) ON DELETE CASCADE
) STRICT;

-- ===========================================================================
-- Time series off the 2D map. All three are per-side and regularly sampled, so
-- they are rows, not JSON arrays: "average width while 2-0 up" is a WHERE.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS shape_sample (
  match_id TEXT NOT NULL,
  side     TEXT NOT NULL CHECK (side IN ('home', 'away')),
  t_s      REAL NOT NULL,
  width_m  REAL,
  depth_m  REAL,
  hull_area_m2 REAL,
  rear_line_m  REAL,
  stretch_index_m REAL,
  centroid_x_m REAL,
  centroid_y_m REAL,
  -- The three bank depths and the two gaps between them. Fixed arity (defence,
  -- middle, front) so they are columns; a 4-bank fit would still project onto
  -- these three by definition of stats.team_lines.
  line1_m REAL, line2_m REAL, line3_m REAL,
  gap1_m  REAL, gap2_m  REAL,
  -- How many players the sample was computed from, and how many after
  -- de-duplication. A width taken off 7 players is not a team's width.
  n       INTEGER,
  n_dedup INTEGER,
  PRIMARY KEY (match_id, side, t_s),
  FOREIGN KEY (match_id, side) REFERENCES match_side(match_id, side) ON DELETE CASCADE
) STRICT, WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS formation_window (
  match_id  TEXT NOT NULL,
  side      TEXT NOT NULL CHECK (side IN ('home', 'away')),
  t0_s      REAL NOT NULL,
  t1_s      REAL NOT NULL,
  formation TEXT NOT NULL,
  cost      REAL,                            -- template fit; lower is better
  PRIMARY KEY (match_id, side, t0_s),
  FOREIGN KEY (match_id, side) REFERENCES match_side(match_id, side) ON DELETE CASCADE
) STRICT;

CREATE TABLE IF NOT EXISTS stress_interval (
  match_id   TEXT NOT NULL,
  side       TEXT NOT NULL CHECK (side IN ('home', 'away')),
  t0_s       REAL NOT NULL,
  t1_s       REAL NOT NULL,
  duration_s REAL,
  severity   REAL,
  peak_z     REAL,
  formation  TEXT,
  cost       REAL,
  hull_area_m2 REAL,
  PRIMARY KEY (match_id, side, t0_s),
  FOREIGN KEY (match_id, side) REFERENCES match_side(match_id, side) ON DELETE CASCADE
) STRICT;

-- Per-minute swing, -1 (away) to +1 (home). ENTERED today - nothing in the
-- pipeline computes it, and it is signed toward home for the whole match, which
-- is why it is one series and not two side rows.
CREATE TABLE IF NOT EXISTS momentum_sample (
  match_id TEXT NOT NULL REFERENCES match(match_id) ON DELETE CASCADE,
  minute   INTEGER NOT NULL,
  value    REAL NOT NULL,
  PRIMARY KEY (match_id, minute)
) STRICT;

-- ===========================================================================
-- Provenance. A first-class table, not a comment.
--
-- Every number above is one of measured / derived / tagged / entered, at one of
-- three confidence tiers, with a list of caveats. The report already refuses to
-- print a figure without its tier; the store has to be able to answer the same
-- question, or the tier is reconstructed by whoever renders it and drifts.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS field_meta (
  match_id   TEXT NOT NULL REFERENCES match(match_id) ON DELETE CASCADE,
  -- Dotted path into the logical model: 'players.distance_km',
  -- 'teams.home.shape.width_m', 'territory'.
  field_path TEXT NOT NULL,
  units      TEXT,
  n          INTEGER,                        -- sample count the value rests on
  -- not_evaluated: nothing measures this field (stats.metric). Missing from
  -- this CHECK until 2026-09-10, so every match built after tiers started
  -- resolving failed to ingest - silently, see jobs.ingest.
  confidence TEXT CHECK (confidence IN ('ok', 'rel', 'prov', 'not_evaluated')),
  provenance TEXT CHECK (provenance IN ('measured', 'derived', 'tagged', 'entered')),
  caveat_flags TEXT,                         -- JSON array of flag strings
  justified_by TEXT,                         -- the api:// id behind the tier, or NULL
  PRIMARY KEY (match_id, field_path)
) STRICT;

-- ===========================================================================
-- Season views. The reason any of this is a database rather than a directory.
-- ===========================================================================

-- One row per player per match, entered numbers only, ready to average.
CREATE VIEW IF NOT EXISTS season_player_metric AS
SELECT a.player_id, p.club_id, a.side, a.match_id, a.minutes,
       pm.metric, pm.value, pm.attempted, pm.pct
FROM appearance a
JOIN player p USING (player_id)
JOIN player_metric pm ON pm.match_id = a.match_id AND pm.player_id = a.player_id;

-- The same for measured physicals, which only exist where a track is bound.
CREATE VIEW IF NOT EXISTS season_track_metric AS
SELECT ptm.player_id, ptm.match_id, ptm.metric, ptm.value,
       ptm.bind_source, ptm.bind_confidence, a.minutes
FROM player_track_metric ptm
LEFT JOIN appearance a ON a.match_id = ptm.match_id AND a.player_id = ptm.player_id;
