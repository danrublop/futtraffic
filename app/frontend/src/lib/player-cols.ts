import type { PlayerStatLine } from "@/lib/types";

/** The per-player table. Category pills pick the column set; rows sort on the
 *  rating, and the match's top rating gets the blue star pill. xG + xA is the
 *  one computed column — the two terms arrive separately and the sum is
 *  addition, not a statistic. */
export const PLAYER_STAT_COLS: Record<string, string> = {
  rating: "Rating",
  minutes: "Minutes played",
  goals: "Goals",
  assists: "Assists",
  xg: "xG",
  xa: "xA",
  xgxa: "xG + xA",
  def: "Defensive contributions",
  shots: "Shots",
  sot: "Shots on target",
  chances: "Chances created",
  dribbles: "Successful dribbles",
  passes: "Passes",
  acc_passes: "Accurate passes",
  long_balls: "Accurate long balls",
  crosses: "Accurate crosses",
  tackles: "Tackles",
  interceptions: "Interceptions",
  clearances: "Clearances",
  blocks: "Blocks",
  duels: "Duels won",
  duels_won: "Duels won",
  duels_lost: "Duels lost",
  ground_duels: "Ground duels won",
  aerial_duels: "Aerial duels won",
  fouls: "Fouls committed",
  was_fouled: "Was fouled",
  saves: "Saves",
  conceded: "Goals conceded",
  xgot_faced: "xGOT faced",
  goals_prevented: "Goals prevented",
  sweeper: "Acted as sweeper",
  high_claim: "High claim",
};

export const PLAYER_STAT_TABS: {
  id: string;
  label: string;
  cols: string[];
  sort: string;
  gkOnly?: boolean;
}[] = [
  { id: "top", label: "Top stats", sort: "rating", cols: ["rating", "minutes", "goals", "assists", "xg", "xa", "xgxa", "def"] },
  { id: "attack", label: "Attack", sort: "rating", cols: ["rating", "minutes", "goals", "xg", "shots", "sot", "chances", "dribbles"] },
  { id: "passes", label: "Passes", sort: "rating", cols: ["rating", "minutes", "assists", "xa", "passes", "acc_passes", "long_balls", "crosses"] },
  { id: "defense", label: "Defense", sort: "rating", cols: ["rating", "minutes", "def", "tackles", "interceptions", "clearances", "blocks"] },
  // Duels drops the rating and minutes entirely and leads with Duels won —
  // with Goalkeeping, one of the two tabs whose table is about a different
  // quantity.
  { id: "duels", label: "Duels", sort: "duels_won", cols: ["duels_won", "duels_lost", "ground_duels", "aerial_duels", "fouls", "was_fouled", "dribbles", "tackles"] },
  { id: "gk", label: "Goalkeeping", sort: "rating", cols: ["saves", "conceded", "xgot_faced", "goals_prevented", "sweeper", "high_claim", "long_balls", "acc_passes"], gkOnly: true },
];

export type PlayerStatTab = (typeof PLAYER_STAT_TABS)[number];

/** Sort a feed's rows for one tab. Shared so the report screen's interactive
 *  table and a frozen one in a document cannot order the same data differently
 *  and quietly disagree about who led the match. */
export function sortForTab(
  players: PlayerStatLine[],
  tab: PlayerStatTab,
  asc = false,
): PlayerStatLine[] {
  return players
    .filter((p) => !tab.gkOnly || p.pos === "Goalkeeper")
    .sort(
      (a, b) =>
        (Number(b.stats[tab.sort] ?? 0) - Number(a.stats[tab.sort] ?? 0)) * (asc ? -1 : 1),
    );
}

/** One cell, as text. `xgxa` is the one computed column — the two terms arrive
 *  separately and the sum is addition, not a statistic. */
export function statCell(p: PlayerStatLine, col: string): string {
  if (col === "xgxa") {
    const v = Number(p.stats.xg ?? 0) + Number(p.stats.xa ?? 0);
    return v ? v.toFixed(2) : "0";
  }
  return String(p.stats[col] ?? "0");
}
