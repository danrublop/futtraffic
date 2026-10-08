import type {
  Appearance,
  LineupRow,
  MatchInfo,
  MatchSide,
  OsmPitches,
  TeamStat,
  Calibration,
  Clip,
  ClipsResponse,
  ClubPlayer,
  Coach,
  DbResult,
  DbTable,
  Match,
  MatchReport,
  PlayerBio,
  NoteEntry,
  Org,
  RosterPlayer,
  Season,
  StatPreset,
  Team,
  TracksDigest,
} from "./types";
import type { SeasonTable } from "./season";

/** The one place that knows the FastAPI contract.
 *
 *  Everything is same-origin `/api/...` because next.config.ts rewrites that to
 *  the Python service. Do not hardcode a host here — a component that reaches a
 *  different origin is the thing that breaks the moment this is deployed.
 */

/** The season fact table, cached for the life of the page. Module-level rather
 *  than a hook, because four Data sections and the notepad all want the same
 *  copy and none of them should race the others for it. */
let seasonTable: Promise<SeasonTable> | null = null;

/** Drop the cached fact table. Called after the store is rebuilt, when the
 *  answer on disk has genuinely changed under us. */
export function reloadSeason() {
  seasonTable = null;
  clearStoreCache();
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** FastAPI puts the operator-readable sentence in {"detail": "..."}. The
 *  calibration refusal in particular is a paragraph written for the person who
 *  just clicked the landmarks; showing "400 Bad Request" throws that away. */
export async function detail(res: Response): Promise<string> {
  const text = await res.text().catch(() => "");
  try {
    const body = JSON.parse(text);
    if (typeof body?.detail === "string") return body.detail;
  } catch {
    /* a static server answers a missing route with a whole HTML page */
  }
  return text.trim().startsWith("<") || !text
    ? `${res.status} ${res.statusText}`
    : text;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { cache: "no-store", ...init });
  if (!res.ok) throw new ApiError(res.status, await detail(res));
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

const m = (id: string, sub = "") =>
  `/api/matches/${encodeURIComponent(id)}${sub}`;

/** A note path is several segments, so each is encoded on its own — encoding
 *  the whole path would turn its slashes into %2F and address a note that does
 *  not exist. */
const n = (path: string) =>
  `/api/notes/${path.split("/").map(encodeURIComponent).join("/")}`;

/** A STORE READ THAT HIT THE ROW CAP IS A WRONG TOTAL, not a short list.
 *  `/api/db/query` stops at DB_ROWS (5,000) and says so in `truncated`. Every
 *  reader below sums its rows into a season total, a per-90 divisor or a
 *  best-in-squad maximum, so a silently shortened result does not look empty -
 *  it looks like a smaller club. Ten matches are nowhere near the cap; a league
 *  season of team metrics is. Fail loudly at the boundary instead. */
export function whole(r: DbResult, what: string): DbResult {
  if (r.truncated)
    throw new Error(`${what}: the store returned more rows than one query carries, so any total from them would be short. Narrow the query.`);
  return r;
}

/** Column lookup by name that REFUSES a miss. `columns.indexOf` answers -1 for
 *  a renamed column, `row[-1]` is undefined, and the field then reads as "no
 *  data" on every screen rather than as the broken query it is. */
export function column(r: DbResult): (c: string) => number {
  return (c) => {
    const k = r.columns.indexOf(c);
    if (k < 0) throw new Error(`the store returned no column "${c}"`);
    return k;
  };
}

export const api = {
  // ---- matches -----------------------------------------------------------
  /** Every match, or one team's. Matched on the team slug server-side, so the
   *  caller does not have to reproduce the club's exact capitalisation. */
  listMatches: (team?: string) =>
    req<Match[]>(
      team ? `/api/matches?team=${encodeURIComponent(team)}` : "/api/matches",
    ),
  getMatch: (id: string) => req<Match>(m(id)),

  /** Kick a parked/failed job back into the queue. */
  retry: (id: string) =>
    req<Match>(m(id, "/retry"), { method: "POST" }).catch(() => api.getMatch(id)),

  // ---- results -----------------------------------------------------------
  getReport: (id: string) => req<MatchReport>(m(id, "/match.json")),

  /** The whole library as one fact table - see lib/season.ts for the shape and
   *  model/engine/season.py for why it is atoms rather than three endpoints.
   *
   *  CACHED, because it is the biggest thing the app fetches and it grows with
   *  the library: measured at 117 KB for 10 matches, about 12 KB per match,
   *  parsed whole in the browser. Every mount of the Data surface refetched and
   *  reparsed all of it. The cache is the same module-promise idiom lib/club.ts
   *  uses for the org - one in flight at a time, and a failure does not stick.
   *  `reloadSeason()` drops it, which is what the store rebuild needs. */
  season: () => {
    if (!seasonTable) {
      seasonTable = req<SeasonTable>("/api/season");
      seasonTable.catch(() => {
        seasonTable = null; // a blip must not become permanent
      });
    }
    return seasonTable;
  },
  getTracksDigest: (id: string) => req<TracksDigest>(m(id, "/tracks")),
  getCalibration: (id: string) => req<Calibration>(m(id, "/calibration")),
  /** The venue's pitch size. Drops an automatic calibration made under the old
   *  size and re-queues the match, so the camera is found again. */
  setPitch: (id: string, width: number, length?: number, osmWay?: number) =>
    req<Match>(m(id, "/pitch"), {
      method: "POST",
      body: JSON.stringify({ width, length, ...(osmWay ? { source: "osm", osm_way: osmWay } : {}) }),
    }),
  /** Pitch outlines OpenStreetMap has near a point, each checked against this
   *  match's painted length. Sends the coordinates and nothing else. */
  osmPitches: (id: string, lat: number, lon: number) =>
    req<OsmPitches>(m(id, `/pitch/osm?lat=${lat}&lon=${lon}`)),

  /** URLs, not fetches: these are handed to <video> and to the worker, which
   *  stream them. Pulling tens of MB through fetch() first would defeat both. */
  videoUrl: (id: string) => m(id, "/video"),
  tracksUrl: (id: string) => m(id, "/tracks.json"),
  frameUrl: (id: string, frame = 0, width = 0) =>
    m(id, `/frame.jpg?f=${frame}${width ? `&w=${width}` : ""}`),
  /** A match card's still. NOT frameUrl: that one is `no-store` because
   *  calibration must see the real frame every time, so a gallery served off it
   *  costs a video decode per card per paint. This one is cached on disk under
   *  the video's mtime and held by the browser for a week. */
  thumbUrl: (id: string, width = 240) => m(id, `/thumb.jpg?w=${width}`),
  /** `frame` picks the moment the crop is taken from - the split scrubber's
   *  view of a track. Omitted, the server uses the track's middle frame. */
  cropUrl: (id: string, trackId: number, height = 0, frame?: number) =>
    m(id, `/crop/${trackId}.jpg?h=${height}${frame === undefined ? "" : `&f=${frame}`}`),

  // ---- calibration -------------------------------------------------------
  saveCalibration: (id: string, body: unknown) =>
    req<Match>(m(id, "/calibration"), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }),

  // ---- identity ----------------------------------------------------------
  saveIdentity: (id: string, body: unknown) =>
    req<unknown>(m(id, "/identity"), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }),

  // ---- presets -----------------------------------------------------------
  /** Saved stat selections, shared by every note in the project. Not stored in
   *  a document: the point of building a comparison once is that it is offered
   *  in the next note and the next match. */
  getPresets: () => req<StatPreset[]>("/api/presets"),

  /** Whole-list replace. A preset has no server-minted identity to preserve,
   *  unlike a roster player where the id IS the season join, so the contract is
   *  simply: send the list you want, get the cleaned list back. */
  savePresets: (items: StatPreset[]) =>
    req<StatPreset[]>("/api/presets", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(items),
    }),

  // ---- the club ----------------------------------------------------------
  /** The ladder: every team youngest-first, the staff, and the size of each.
   *  Seeds itself server-side from the team strings already on the jobs, so an
   *  install that predates the club model comes up with its teams on it. */
  getOrg: () => req<Org>("/api/org"),

  /** Upsert the whole club. Team ids are server-minted — the id is the join to
   *  the squad file, so a client-invented one orphans it, exactly as with a
   *  roster player id. Send names; read ids back. */
  saveOrg: (org: {
    club?: Partial<Org["club"]>;
    teams?: Partial<Team>[];
    coaches?: Partial<Coach>[];
  }) =>
    req<Org>("/api/org", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(org),
    }),

  // ---- roster ------------------------------------------------------------
  /** The season roster, every team's. Small by construction: roster.py's own
   *  worst case is one club-season at tens of thousands of rows total.
   *
   *  One row per (player, TEAM) — the correction screen's shape. For one row
   *  per HUMAN across the whole club, which is the career, use `getPlayers`. */
  getRoster: () => req<RosterPlayer[]>("/api/roster"),

  /** THE PLAYER REGISTRY: one row per human, with every team they belong to.
   *  A player climbing U13 → U15 → U17 is one profile here and three squad
   *  rows in `getRoster`. */
  getPlayers: () => req<ClubPlayer[]>("/api/players"),

  /** Every player bio, keyed on the same roster id the registry uses. One
   *  fetch for the squad rather than one per profile: they are a few hundred
   *  bytes each and the panel walks down a list. */
  getBios: () => req<Record<string, PlayerBio>>("/api/players/bios"),

  /** The ENTERED half of one profile - height, foot, country, positions.
   *
   *  A merge on the server, not a replace: the feed owns the traits, the
   *  heatmap and the shots in the same file, and this form has no business
   *  deleting a measurement because somebody fixed a shirt number. The whole
   *  stored bio comes back, `age` derived from `birth_date` on the way in, so
   *  render the REPLY rather than what you sent.
   *
   *  `getBios` is memoized (see the bottom of this file), so a caller that
   *  wants the list to agree with the panel calls `clearStoreCache()` after. */
  saveBio: (pid: string, bio: Partial<PlayerBio>) =>
    req<PlayerBio>(`/api/players/${encodeURIComponent(pid)}/bio`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(bio),
    }),

  /** One headshot, the raw file as the body - the server reads the type off
   *  the bytes, so no multipart envelope and no content-type to get wrong. */
  savePhoto: (pid: string, file: Blob) =>
    req<{ ok: boolean; type: string; bytes: number }>(
      `/api/players/${encodeURIComponent(pid)}/photo`,
      { method: "PUT", body: file },
    ),

  // ---- the store ---------------------------------------------------------
  /** Every table and view in `data/match.db`, with columns and row counts.
   *  The only honest way to write a query against a schema that grows. */
  dbSchema: () => req<DbTable[]>("/api/db/schema"),

  /** One read-only statement. The connection is opened `mode=ro` with
   *  `query_only`, and `Connection.execute` runs exactly one statement — so
   *  this is safe by the database's own rules rather than by a keyword filter
   *  on the client. */
  dbQuery: (sql: string, params: unknown[] = []) =>
    req<DbResult>("/api/db/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ sql, params }),
    }),

  /** Re-ingest every match.json and the club into a fresh store. The store is
   *  a query layer, so this is always safe and never loses anything. */
  dbRebuild: () => req<unknown>("/api/db/rebuild", { method: "POST" }),

  /** Every appearance in the store, with the FEED's numbers for it.
   *
   *  Through dbQuery rather than a route of its own, which is the pattern
   *  club-editor.tsx already uses for the same store: the shape wanted here is
   *  a join over appearance + player_metric, and `season_player_metric` is the
   *  view that exists for it. A dedicated endpoint would be a third place that
   *  has to agree about what an appearance is.
   *
   *  ONE fetch for the whole store, not one per selected player. It is 400 rows
   *  over ten matches and it powers both the season totals in the table and the
   *  per-match breakdown in the panel, so refetching on every row click would
   *  buy nothing.
   *
   *  Grouped by (player, match) BECAUSE the metric join fans out: one
   *  appearance has a row per metric, so a bare COUNT(*) over the join counts
   *  metrics and reports a player with nine appearances as having ninety.
   */
  getAppearances: async (): Promise<Appearance[]> => {
    const r = await api.dbQuery(
      `SELECT a.player_id, a.match_id, a.side,
              (SELECT p.name FROM player p WHERE p.player_id = a.player_id) AS name,
              a.minutes, a.rating, a.age, a.shirt,
              SUM(CASE WHEN pm.metric = ? THEN pm.value END) AS goals,
              SUM(CASE WHEN pm.metric = ? THEN pm.value END) AS assists,
              SUM(CASE WHEN pm.metric = ? THEN pm.value END) AS saves,
              SUM(CASE WHEN pm.metric = ? THEN pm.value END) AS conceded,
              (SELECT c.name FROM match m JOIN competition c ON c.competition_id = m.competition_id
                WHERE m.match_id = a.match_id) AS competition,
              (SELECT COUNT(*) FROM event e WHERE e.match_id = a.match_id
                  AND e.player_id = a.player_id AND e.type = 'yellow_card') AS yellow,
              (SELECT COUNT(*) FROM event e WHERE e.match_id = a.match_id
                  AND e.player_id = a.player_id AND e.type = 'red_card') AS red
         FROM appearance a
         LEFT JOIN player_metric pm
                ON pm.match_id = a.match_id AND pm.player_id = a.player_id
        GROUP BY a.player_id, a.match_id`,
      ["goals", "assists", "saves", "conceded"],
    );
    whole(r, "appearances");
    const i = column(r);
    const [p, m, sd, nm, mi, ra, ag, sh, go, as, sv, co, cp, ye, re] = [
      i("player_id"), i("match_id"), i("side"), i("name"), i("minutes"), i("rating"),
      i("age"), i("shirt"), i("goals"), i("assists"), i("saves"), i("conceded"),
      i("competition"), i("yellow"), i("red"),
    ];
    return r.rows.map((row) => ({
      player_id: String(row[p]),
      match_id: String(row[m]),
      side: row[sd] as "home" | "away",
      name: (row[nm] as string | null) ?? String(row[p]),
      minutes: row[mi] as number | null,
      rating: row[ra] as number | null,
      age: row[ag] as number | null,
      shirt: row[sh] as number | null,
      goals: row[go] as number | null,
      assists: row[as] as number | null,
      saves: row[sv] as number | null,
      conceded: row[co] as number | null,
      competition: (row[cp] as string | null) ?? null,
      yellow: Number(row[ye] ?? 0),
      red: Number(row[re] ?? 0),
    }));
  },

  /** Both sides of every match: which club, its formation and team rating,
   *  and which of our squads the match belongs to. */
  getMatchSides: async (): Promise<MatchSide[]> => {
    const r = await api.dbQuery(
      `SELECT ms.match_id, ms.side, ms.club_id, ms.formation, ms.team_rating, m.team_id
         FROM match_side ms JOIN match m ON m.match_id = ms.match_id`,
    );
    whole(r, "sides");
    return r.rows.map(([match_id, side, club_id, formation, team_rating, team_id]) => ({
      match_id: String(match_id),
      side: side as "home" | "away",
      club_id: String(club_id),
      formation: (formation as string | null) ?? null,
      team_rating: (team_rating as number | null) ?? null,
      team_id: (team_id as string | null) ?? null,
    }));
  },

  /** One club's team stats in every match it played, as the report's Teams
   *  tab lists them: match -> metric id -> value and the feed's own text. */
  getTeamStats: async (
    clubId: string,
  ): Promise<Record<string, Record<string, TeamStat>>> => {
    const r = await api.dbQuery(
      `SELECT tm.match_id, tm.metric, tm.value, tm.raw, tm.pct
         FROM team_metric tm
         JOIN match_side ms ON ms.match_id = tm.match_id AND ms.side = tm.side
        WHERE ms.club_id = ?`,
      [clubId],
    );
    whole(r, "team stats");
    const out: Record<string, Record<string, TeamStat>> = {};
    for (const [m, k, v, raw, pct] of r.rows) {
      (out[String(m)] ??= {})[String(k)] = {
        value: v == null ? null : Number(v),
        raw: (raw as string | null) ?? null,
        pct: pct == null ? null : Number(pct),
      };
    }
    return out;
  },

  /** The OTHER side's team stats in every match `clubId` played - its
   *  opponents, one per match, keyed the same way as getTeamStats. */
  getOpponentStats: async (clubId: string): Promise<Record<string, Record<string, TeamStat>>> => {
    const r = await api.dbQuery(
      `SELECT tm.match_id, tm.metric, tm.value, tm.raw, tm.pct
         FROM team_metric tm
         JOIN match_side ms ON ms.match_id = tm.match_id AND ms.side = tm.side
        WHERE ms.club_id != ?
          AND tm.match_id IN (SELECT match_id FROM match_side WHERE club_id = ?)`,
      [clubId, clubId],
    );
    whole(r, "opponent stats");
    const out: Record<string, Record<string, TeamStat>> = {};
    for (const [m, k, v, raw, pct] of r.rows) {
      (out[String(m)] ??= {})[String(k)] = {
        value: v == null ? null : Number(v),
        raw: (raw as string | null) ?? null,
        pct: pct == null ? null : Number(pct),
      };
    }
    return out;
  },

  /** What happened in one match, from the store: its status, its key events
   *  (goals with scorer and assist, cards, subs, the half-time marker) in
   *  order, and the minute-by-minute momentum where the feed sent one. */
  getMatchInfo: async (matchId: string): Promise<MatchInfo> => {
    const [st, ev, mo] = await Promise.all([
      api.dbQuery("SELECT status FROM match WHERE match_id = ?", [matchId]),
      api.dbQuery(
        `SELECT e.minute, e.side, e.type, e.detail,
                (SELECT p.name FROM event_player ep JOIN player p ON p.player_id = ep.player_id
                  WHERE ep.event_id = e.event_id AND ep.role IN ('actor', 'on') LIMIT 1) AS who,
                (SELECT p.name FROM event_player ep JOIN player p ON p.player_id = ep.player_id
                  WHERE ep.event_id = e.event_id AND ep.role IN ('assist', 'off') LIMIT 1) AS other
           FROM event e
          WHERE e.match_id = ? AND e.type IN ('goal', 'own_goal', 'yellow_card', 'red_card', 'sub', 'period')
          ORDER BY e.event_id`,
        [matchId],
      ),
      api.dbQuery("SELECT minute, value FROM momentum_sample WHERE match_id = ? ORDER BY minute", [
        matchId,
      ]),
    ]);
    return {
      status: (st.rows[0]?.[0] as string | null) ?? null,
      events: ev.rows.map(([minute, side, type, detail, who, other]) => {
        let score: string | null = null;
        try {
          score = detail ? (JSON.parse(String(detail)).score ?? null) : null;
        } catch {
          /* a detail that is not JSON carries no score */
        }
        return {
          minute: (minute as number | null) ?? null,
          side: (side as "home" | "away" | null) ?? null,
          type: String(type),
          who: (who as string | null) ?? null,
          other: (other as string | null) ?? null,
          score,
        };
      }),
      momentum: mo.rows.map(([minute, value]) => ({ minute: Number(minute), value: Number(value) })),
    };
  },

  /** One side's team sheet in one match: the XI with where the feed plots
   *  them, then the bench. */
  getLineup: async (matchId: string, side: "home" | "away"): Promise<LineupRow[]> => {
    const r = await api.dbQuery(
      `SELECT a.player_id, p.name, a.shirt, a.position, a.started, a.on_minute, a.rating,
              a.plot_x, a.plot_y
         FROM appearance a JOIN player p ON p.player_id = a.player_id
        WHERE a.match_id = ? AND a.side = ?
        ORDER BY a.started DESC, a.plot_y, a.plot_x`,
      [matchId, side],
    );
    whole(r, "lineup");
    return r.rows.map(([player_id, name, shirt, position, started, on, rating, x, y]) => ({
      player_id: String(player_id),
      name: String(name ?? player_id),
      shirt: (shirt as number | null) ?? null,
      position: (position as string | null) ?? null,
      started: !!started,
      on_minute: (on as number | null) ?? null,
      rating: (rating as number | null) ?? null,
      x: (x as number | null) ?? null,
      y: (y as number | null) ?? null,
    }));
  },

  /** Every feed stat per player, SUMMED over the matches asked for (all of
   *  them by default; the store's per-match rows exceed the query's row cap,
   *  so the sum has to happen here and not in the browser): player id -> metric
   *  -> total, keyed in the report's spelling (`acc_passes`; the store files
   *  it `acc-passes`). A sum is wrong for rating, so callers average rating
   *  from the appearances instead. */
  getSeasonStats: async (
    /** Sum over these matches only - one season's, or one match. */
    matchIds?: string[],
  ): Promise<Record<string, Record<string, number>>> => {
    if (matchIds && !matchIds.length) return {};
    const where = matchIds ? `WHERE match_id IN (${matchIds.map(() => "?").join(",")})` : "";
    const r = await api.dbQuery(
      `SELECT player_id, metric, SUM(value) AS v FROM player_metric ${where} GROUP BY player_id, metric`,
      matchIds ?? [],
    );
    whole(r, "season stats");
    const out: Record<string, Record<string, number>> = {};
    for (const [p, m, v] of r.rows) {
      if (v != null) (out[String(p)] ??= {})[String(m).replaceAll("-", "_")] = Number(v);
    }
    return out;
  },

  /** Every finished match for a team, joined into per-player season rows on the
   *  roster's human-confirmed track bindings. The join CANNOT be done here: the
   *  bindings live in the roster file and are on no other route, so a frontend
   *  aggregating match.json files could only join on (team, shirt number) —
   *  which merges two players the week a shirt changes. See jobs.season. */
  getSeason: (team?: string) =>
    req<Season>(
      team
        ? `/api/season/measured?team=${encodeURIComponent(team)}`
        : "/api/season/measured",
    ),

  /** Upsert. Returns the CANONICAL list, not an echo — the server mints
   *  `player_id` by slugging the name, so the id a new player ends up with is
   *  only knowable from the reply. Assigning with a locally-guessed id binds a
   *  track to nothing, silently. */
  saveRoster: (players: Partial<RosterPlayer>[]) =>
    req<RosterPlayer[]>("/api/roster", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(players),
    }),

  // ---- clips -------------------------------------------------------------
  listClips: (id: string) => req<ClipsResponse>(m(id, "/clips")),
  saveClip: (id: string, clip: Partial<Clip>) =>
    req<ClipsResponse>(m(id, "/clips"), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(clip),
    }),
  deleteClip: (id: string, clipId: string) =>
    req<ClipsResponse>(m(id, `/clips/${encodeURIComponent(clipId)}`), {
      method: "DELETE",
    }),

  // ---- notes -------------------------------------------------------------
  /** THE NOTE LIBRARY. One folder tree for the whole club — a note carries no
   *  match id and no team id, because the thing a coach writes usually spans
   *  several of both. Its path is its identity, so a rename is `moveNote`.
   *
   *  Every mutation answers with the whole listing, which is small and saves
   *  the tree a refetch it would otherwise have to do to stay honest. */
  notes: () => req<NoteEntry[]>("/api/notes"),
  /** 404 until the first save — the caller seeds a template rather than
   *  treating that as an error. */
  getNote: (path: string) => req<{ html: string; version: string }>(n(path)),
  /** `base` makes this a write-if-unchanged: the server answers 409 rather
   *  than overwriting a version this caller never saw. Omit it to overwrite,
   *  which is what a report script regenerating its own note wants. */
  putNote: (path: string, html: string, base?: string) =>
    req<{ ok: boolean; bytes: number; version: string }>(n(path), {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(base === undefined ? { html } : { html, base }),
    }),
  deleteNote: (path: string) => req<NoteEntry[]>(n(path), { method: "DELETE" }),
  newFolder: (path: string) =>
    req<NoteEntry[]>("/api/notes/folder", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ path }),
    }),
  moveNote: (from: string, to: string) =>
    req<NoteEntry[]>("/api/notes/move", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ from, to }),
    }),

  // ---- settings ----------------------------------------------------------
  /** The agent's API key. GET never returns the key itself — only whether one
   *  exists and where it came from, so `source: "env"` can tell someone their
   *  environment already supplies it and this box will not override it. */
  getSettings: () =>
    req<{ has_key: boolean; source: "env" | "settings" | "none" }>("/api/settings"),
  putSettings: (anthropic_api_key: string) =>
    // `/api/`, like every other route: bare "/settings" is the Next page, so
    // these two never reached the backend and the page read has_key: false.
    req<{ has_key: boolean; source: "env" | "settings" | "none" }>("/api/settings", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ anthropic_api_key }),
    }),
};

/* ------------------------------------------------------------------ cache */

/** THE STORE READS, CACHED for the life of the page - the same module-promise
 *  idiom as `season()`. The Data tables mount and remount as you move between
 *  them, and without this every visit refetched and rebuilt from nothing.
 *  One in flight per key; a failure does not stick; a store rebuild or a
 *  season reload drops the lot, which is when the answers actually change.
 *
 *  Not cached: the player registry - the club editor changes it in place. */
const memo = new Map<string, Promise<unknown>>();

function once<T>(key: string, load: () => Promise<T>): Promise<T> {
  let p = memo.get(key) as Promise<T> | undefined;
  if (!p) {
    p = load();
    memo.set(key, p);
    p.catch(() => memo.delete(key));
  }
  return p;
}

export function clearStoreCache() {
  memo.clear();
}

const uncached = {
  getAppearances: api.getAppearances,
  getSeasonStats: api.getSeasonStats,
  getMatchSides: api.getMatchSides,
  getTeamStats: api.getTeamStats,
  getOpponentStats: api.getOpponentStats,
  getMatchInfo: api.getMatchInfo,
  getLineup: api.getLineup,
  getBios: api.getBios,
  dbRebuild: api.dbRebuild,
};
api.getAppearances = () => once("appearances", uncached.getAppearances);
api.getSeasonStats = (ids) => once(`season-stats:${ids?.join(",") ?? "*"}`, () => uncached.getSeasonStats(ids));
api.getMatchSides = () => once("sides", uncached.getMatchSides);
api.getTeamStats = (club) => once(`team:${club}`, () => uncached.getTeamStats(club));
api.getOpponentStats = (club) => once(`opp:${club}`, () => uncached.getOpponentStats(club));
api.getMatchInfo = (m) => once(`info:${m}`, () => uncached.getMatchInfo(m));
api.getLineup = (m, side) => once(`lineup:${m}:${side}`, () => uncached.getLineup(m, side));
api.getBios = () => once("bios", uncached.getBios);
api.dbRebuild = () => uncached.dbRebuild().finally(clearStoreCache);

/** Resumable upload.
 *
 *  XHR, not fetch: fetch cannot report upload progress, and a 3 GB upload with
 *  no progress bar is indistinguishable from a hung page. The body is the File
 *  object itself, which the browser streams — it is never read into memory.
 *
 *  X-Upload-Total is REQUIRED by the server when resuming. It is what stops a
 *  different file being appended at the right offset, which produces a video
 *  that only fails hours later, after tracking has already run on it.
 *
 *  `venue` and `team` are their own query params rather than more text packed
 *  into `name` (which is where the date rides). The venue decides whether
 *  another match's pitch setup may be reused here, so the server compares it by
 *  machine; digging a safety condition back out of a display string is how it
 *  would quietly start matching the wrong ground.
 */
export function uploadMatch(
  file: File,
  opts: {
    name: string;
    venue?: string;
    team?: string;
    jobId?: string;
    offset?: number;
    onProgress?: (sent: number, total: number) => void;
    signal?: AbortSignal;
  },
): Promise<Match> {
  const offset = opts.offset ?? 0;
  const q = new URLSearchParams({ name: opts.name, total: String(file.size) });
  if (opts.venue) q.set("venue", opts.venue);
  if (opts.team) q.set("team", opts.team);
  const url = opts.jobId ? m(opts.jobId, "/upload") : `/api/matches?${q}`;

  return new Promise<Match>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    if (offset) {
      xhr.setRequestHeader("X-Upload-Offset", String(offset));
      xhr.setRequestHeader("X-Upload-Total", String(file.size));
    }
    xhr.upload.onprogress = (e) =>
      opts.onProgress?.(offset + e.loaded, file.size);
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          return resolve(JSON.parse(xhr.responseText) as Match);
        } catch {
          return reject(new ApiError(xhr.status, "unreadable reply from the server"));
        }
      }
      let msg = `${xhr.status} ${xhr.statusText}`;
      try {
        const d = JSON.parse(xhr.responseText)?.detail;
        if (typeof d === "string") msg = d;
      } catch {
        /* keep the status line */
      }
      reject(new ApiError(xhr.status, msg));
    };
    xhr.onerror = () => reject(new ApiError(0, "connection lost during upload"));
    xhr.onabort = () => reject(new ApiError(0, "upload cancelled"));
    opts.signal?.addEventListener("abort", () => xhr.abort());
    xhr.send(offset ? file.slice(offset) : file);
  });
}
