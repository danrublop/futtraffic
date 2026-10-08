"use client";

import { useEffect, useState } from "react";
import { Bolt, Plus, X } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { DATA_HEAD, MeasuredCell, ZEBRA } from "@/components/data/section-shell";
import { Chevron } from "@/components/data/tables";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { api } from "@/lib/api";
import { reloadOrg, setTeam, useOrg, useTeam } from "@/lib/club";
import type { Coach, Org, Team } from "@/lib/types";
import { cn } from "@/lib/utils";

/** THE CLUB — the ladder of squads, the staff who run them, and what each squad
 *  actually has behind it.
 *
 *  WHY THIS IS A VIEW OF `DATA` AND NOT ITS OWN PAGE. It was both for a while,
 *  and that was one subject on two screens: a Club page listing teams, and a
 *  Data tab listing the same teams with their totals. Whichever one you were on
 *  was missing half the answer — you could see a squad's distance but not who
 *  coached it, or edit its name but not learn it had no footage. They are one
 *  table: the club IS the first cut of the database, so it lives with the rest
 *  of the cuts.
 *
 *  EDITING AND READING IN ONE ROW. The name, band and staff are inputs; the
 *  matches, squad size and measured totals to their right are not. A coach
 *  fixing an age band should be able to see, on the same line, that the squad
 *  it belongs to has eleven players and no confirmed identities.
 *
 *  WHY SAVE IS A BUTTON AND NOT AUTOSAVE. Removing a team's row is the one
 *  destructive act available here, and an autosaving form turns a mis-click
 *  into a silent deletion. The server keeps the squad file either way (see
 *  jobs.save_org), but the ladder is what every other surface reads.
 *
 *  WHAT IS NOT HERE: squad lists. Names get attached to people on a match's
 *  pitch-setup screen, where there is a track to attach them to. A second
 *  editor for the same fact is how two stores start disagreeing about who
 *  somebody is.
 */

/** A draft row carries the server's `team_id` when it has one and nothing when
 *  it is new — the id is minted server-side because it is the join to the
 *  squad file, so guessing one here would orphan a squad. */
type TeamDraft = Partial<Team> & { name: string; key: string };
type CoachDraft = Partial<Coach> & { name: string; key: string };

let seq = 0;
const newKey = () => `new-${++seq}`;


/** What the STORE knows about a squad, as opposed to what the club declares.
 *  `named` is the one that matters: matches say how many games exist, named
 *  says how many players in them a human has confirmed the identity of, and
 *  only the second kind can carry a per-player number. */
interface Totals {
  matches: number;
  named: number;
  km: number | null;
  top: number | null;
}

/** No fan-out: track_metric is one row per (track, metric) and the join keys
 *  include both, so nothing is double-counted. Squad size is deliberately NOT
 *  joined in here — a squad join multiplies every track row by the number of
 *  players and silently inflates the sums. It comes from the club instead. */
const TEAM_TOTALS = `
SELECT m.team_id                                                       AS team_id,
       count(DISTINCT m.match_id)                                      AS matches,
       count(DISTINCT tk.player_id)                                    AS named,
       sum(CASE WHEN tm.metric = 'distance_km'  THEN tm.value END)     AS km,
       max(CASE WHEN tm.metric = 'top_speed_kmh' THEN tm.value END)    AS top_kmh
FROM match m
LEFT JOIN track tk ON tk.match_id = m.match_id AND tk.kind = 'player'
LEFT JOIN track_metric tm ON tm.match_id = tk.match_id AND tm.track_id = tk.track_id
GROUP BY m.team_id`;

function useTotals() {
  const [stats, setStats] = useState<Map<string, Totals> | null>(null);
  useEffect(() => {
    let live = true;
    api
      .dbQuery(TEAM_TOTALS)
      .then((r) => {
        if (!live) return;
        const at = Object.fromEntries(r.columns.map((c, i) => [c, i]));
        const m = new Map<string, Totals>();
        for (const row of r.rows) {
          m.set(String(row[at.team_id] ?? ""), {
            matches: Number(row[at.matches] ?? 0),
            named: Number(row[at.named] ?? 0),
            km: row[at.km] == null ? null : Number(row[at.km]),
            top: row[at.top_kmh] == null ? null : Number(row[at.top_kmh]),
          });
        }
        setStats(m);
      })
      // No store yet is not an error here: the ladder still lists, the
      // measured columns just say nothing.
      .catch(() => live && setStats(new Map()));
    return () => {
      live = false;
    };
  }, []);
  return stats;
}

export function ClubPanel({
  /** The squad this pane has open, and how to change it. A SELECTION, passed
   *  in — not the app's global team scope. The row click used to call
   *  `setTeam`, which re-scoped the sidebar, the header and the notepad from a
   *  table nobody thought of as a control. The header switcher owns that lens
   *  now; this owns which squad the page below is about. */
  selected = "",
  onSelect,
  compact = false,
}: {
  selected?: string;
  onSelect?: (teamId: string) => void;
  /** The "..." menu alone - Add team, the club's settings, each squad's Edit
   *  and Delete - with its dialogs, for the squad panel's corner. No club
   *  line, no squad table. */
  compact?: boolean;
} = {}) {
  const { org, error } = useOrg();
  const [appScope] = useTeam();
  const scope = selected;
  const totals = useTotals();
  const [busy, setBusy] = useState(false);
  /** Which dialog is open, and on what. A single piece of state because only
   *  one of them can be open at a time and three booleans that must never both
   *  be true is three ways to be wrong. */
  const [editing, setEditing] = useState<Editing>(null);

  /** Write the whole club and reload every reader of it.
   *
   *  WHY THIS SAVES ON COMMIT, when this panel used to hold a draft and a Save
   *  button. That button existed because deleting a team was one mis-click
   *  inside an ever-editable form, and autosave would have made that silent.
   *  Both halves of the danger are gone: a row is now read-only until you open
   *  it deliberately in a dialog, and delete is behind a menu and a confirm. So
   *  pressing Save in a dialog can mean what it says.
   */
  const commit = async (next: {
    club?: { name: string; season: string };
    teams?: TeamDraft[];
    coaches?: CoachDraft[];
  }) => {
    if (!org) return;
    setBusy(true);
    try {
      const teams = next.teams ?? org.teams.map((t) => ({ ...t, key: t.team_id }));
      const coaches = next.coaches ?? org.coaches.map((c) => ({ ...c, key: c.coach_id }));
      const saved = await api.saveOrg({
        club: next.club ?? { name: org.club.name, season: org.club.season },
        // `key`, `players`, `matches` and `team_ids` are ours or the server's
        // join, never input - sending them back would invite the server to
        // start trusting a count the browser made up.
        teams: teams
          .filter((t) => t.name.trim())
          .map((t) => ({
            team_id: t.team_id,
            name: t.name,
            age_band: t.age_band ?? "",
            gender: t.gender ?? "",
            level: t.level ?? "",
            season: t.season ?? "",
            coach_ids: t.coach_ids ?? [],
          })),
        coaches: coaches
          .filter((c) => c.name.trim())
          .map((c) => ({
            coach_id: c.coach_id,
            name: c.name,
            role: c.role ?? "",
            email: c.email ?? "",
          })),
      });
      // A team the coach was scoped to and then deleted would leave a stale
      // scope. It degrades to "all teams" on its own (lib/club.ts), but saying
      // so beats a header that silently widens.
      // Two different things to clear. The pane's selection is ours; the app's
      // team lens belongs to the header and is cleared only because a deleted
      // team leaves a stale id there too.
      if (scope && !saved.teams.some((t) => t.team_id === scope)) {
        onSelect?.("");
        if (appScope === scope) setTeam("");
      }
      reloadOrg();
      setEditing(null);
    } catch (e) {
      toast.error("Could not save", {
        description: e instanceof Error ? e.message : String(e),
      });
    } finally {
      setBusy(false);
    }
  };

  if (error) {
    return <p className="text-sm text-[var(--mute)]">The backend is not answering — {error}</p>;
  }
  if (!org) {
    if (compact) return null;
    return (
      <div className="grid gap-4">
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-48 w-full" />
      </div>
    );
  }

  const coachName = new Map(org.coaches.map((c) => [c.coach_id, c.name]));
  const deleteTeam = (t: Team) => {
    if (
      !window.confirm(
        `Delete ${t.name}? Its squad file is kept, so re-adding a team with the same name gets its players back.`,
      )
    )
      return;
    void commit({
      teams: org.teams.filter((x) => x.team_id !== t.team_id).map((x) => ({ ...x, key: x.team_id })),
    });
  };
  const menu = (
    <RowMenu
      label="Club settings"
      items={[
        ...(compact
          ? [{ label: "Add team", onClick: () => setEditing({ kind: "team", team: null }) }]
          : []),
        { label: "Rename club, set season", onClick: () => setEditing({ kind: "club" }) },
        { label: "Staff", onClick: () => setEditing({ kind: "staff" }) },
        ...(compact
          ? org.teams.flatMap((t) => [
              { label: `Edit ${t.name}`, onClick: () => setEditing({ kind: "team", team: t }) },
              { label: `Delete ${t.name}`, destructive: true, onClick: () => deleteTeam(t) },
            ])
          : []),
      ]}
    />
  );
  return (
    // `contents` when compact, so the menu sits wherever the caller puts it.
    <div className={compact ? "contents" : "grid gap-6"}>
      {compact && menu}
      {!compact && (
      <>
      {/* The club itself is one line, not a form. Its two fields are set once
          and then read for years; a card with two inputs and a Save button at
          the top of the page spent the most valuable rows on the least
          frequent edit. */}
      {/* ONE row of chrome, not three. The database browser above already gives
          this table its heading and its provenance line, so a "Teams" section
          header under a "Squads" heading was the same word twice. What is left
          is the only thing the browser cannot say: WHICH club this is, and the
          two menus that change it. */}
      <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1">
        <span className="truncate text-[13px] text-[var(--ink)]">
          {org.club.name || "Your club"}
        </span>
        <span className="text-muted-foreground truncate text-xs">
          {org.club.season || "No season set"} · {org.teams.length}{" "}
          {org.teams.length === 1 ? "team" : "teams"} · {org.coaches.length}{" "}
          {org.coaches.length === 1 ? "coach" : "staff"}
        </span>
        <span className="ml-auto flex items-center gap-1">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setEditing({ kind: "team", team: null })}
          >
            <Plus className="size-3.5" />
            Add team
          </Button>
          {menu}
        </span>
      </div>

      <Bare>
        {!org.teams.length ? (
          <Empty>No teams yet. Add your first age group.</Empty>
        ) : (
          <div className="overflow-x-auto">
            <Table className={cn(ZEBRA, DATA_HEAD)}>
              <TableHeader>
                <TableRow>
                  <TableHead>Team</TableHead>
                  <TableHead>Age</TableHead>
                  <TableHead>Level</TableHead>
                  <TableHead>Season</TableHead>
                  <TableHead>Staff</TableHead>
                  <TableHead className="text-right">Matches</TableHead>
                  <TableHead className="text-right whitespace-nowrap">Named</TableHead>
                  <TableHead className="text-right">Distance</TableHead>
                  <TableHead className="text-right whitespace-nowrap">Top km/h</TableHead>
                  <TableHead className="w-8" />
                  <TableHead className="w-8" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {(scope ? org.teams.filter((t) => t.team_id === scope) : org.teams).map((t) => {
                  const m = totals?.get(t.team_id);
                  const on = t.team_id === scope;
                  return (
                    <TableRow
                      key={t.team_id}
                      onClick={() => onSelect?.(on ? "" : t.team_id)}
                      aria-selected={on}
                      className={cn("cursor-pointer", on ? "bg-muted/60" : "hover:bg-muted/40")}
                    >
                      <TableCell className="font-heading whitespace-nowrap">{t.name}</TableCell>
                      <TableCell className="text-muted-foreground">{t.age_band || "—"}</TableCell>
                      <TableCell className="text-muted-foreground">{t.level || "—"}</TableCell>
                      <TableCell className="text-muted-foreground whitespace-nowrap">
                        {t.season || "—"}
                      </TableCell>
                      <TableCell className="text-muted-foreground whitespace-nowrap">
                        {(t.coach_ids ?? []).map((id) => coachName.get(id)).filter(Boolean).join(", ") ||
                          "—"}
                      </TableCell>
                      <TableCell className="readout text-right">{m?.matches ?? 0}</TableCell>
                      {/* NAMED, not squad size: matches say how many games
                          exist, named says how many players in them a human
                          confirmed the identity of, and only the second kind
                          can carry a per-player number. Zero is worth shouting
                          about because every measured column beside it is then
                          standing on nothing. */}
                      <TableCell
                        className={cn(
                          "readout text-right",
                          m && m.matches > 0 && m.named === 0 && "text-warn",
                        )}
                        title="Players in these matches whose identity a human confirmed"
                      >
                        {m?.named ?? 0}
                      </TableCell>
                      <MeasuredCell v={m?.km} dp={1} unit=" km" />
                      <MeasuredCell v={m?.top} dp={1} />
                      <TableCell className="w-8">
                        <Chevron open={on} />
                      </TableCell>
                      <TableCell className="w-8">
                        <RowMenu
                          label={`Actions for ${t.name}`}
                          items={[
                            {
                              label: "Edit team",
                              onClick: () => setEditing({ kind: "team", team: t }),
                            },
                            {
                              label: on ? "Show every team" : "Show only this",
                              onClick: () => onSelect?.(on ? "" : t.team_id),
                            },
                            {
                              label: "Delete team",
                              destructive: true,
                              onClick: () => deleteTeam(t),
                            },
                          ]}
                        />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        )}
      </Bare>
      </>
      )}

      {editing?.kind === "club" && (
        <ClubDialog
          club={org.club}
          busy={busy}
          onClose={() => setEditing(null)}
          onSave={(club) => void commit({ club })}
        />
      )}
      {editing?.kind === "team" && (
        <TeamDialog
          team={editing.team}
          coaches={org.coaches}
          defaultSeason={org.club.season}
          busy={busy}
          onClose={() => setEditing(null)}
          onSave={(draft) =>
            void commit({
              teams: editing.team
                ? org.teams.map((x) =>
                    x.team_id === editing.team!.team_id
                      ? { ...x, ...draft, key: x.team_id }
                      : { ...x, key: x.team_id },
                  )
                : [...org.teams.map((x) => ({ ...x, key: x.team_id })), { ...draft, key: newKey() }],
            })
          }
        />
      )}
      {editing?.kind === "staff" && (
        <StaffDialog
          coaches={org.coaches}
          busy={busy}
          onClose={() => setEditing(null)}
          onSave={(next) =>
            void commit({
              coaches: next.map((c) => ({ ...c, key: c.coach_id ?? newKey() })),
              // A coach removed here must come off the teams too, or the ladder
              // keeps an id that resolves to nobody.
              teams: org.teams.map((t) => ({
                ...t,
                key: t.team_id,
                coach_ids: (t.coach_ids ?? []).filter((id) =>
                  next.some((c) => c.coach_id === id),
                ),
              })),
            })
          }
        />
      )}
    </div>
  );
}

type Editing =
  | null
  | { kind: "club" }
  | { kind: "team"; team: Team | null }
  | { kind: "staff" };


/** The row's ellipsis.
 *
 *  Everything that used to be a permanently-live input or a bin icon on the row
 *  lives behind this. A table is for READING - eleven squads and what each one
 *  has behind it - and a row of text boxes is not a table, it is a form
 *  pretending. Editing is rare and deliberate, so it costs one click and gets a
 *  dialog with room for the fields that do not deserve a column.
 */
function RowMenu({
  label,
  items,
}: {
  label: string;
  items: { label: string; onClick: () => void; destructive?: boolean }[];
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          aria-label={label}
          title={label}
          // A row click scopes the team; the menu must not also do that.
          onClick={(e) => e.stopPropagation()}
          className="text-muted-foreground hover:text-foreground size-7"
        >
          {/* The bolt, as on the player panel: one mark for "things you can do
              to this record", rather than an ellipsis here and a bolt there
              for the same kind of menu. */}
          <Bolt className="size-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" onClick={(e) => e.stopPropagation()}>
        {items.map((i) => (
          <DropdownMenuItem
            key={i.label}
            variant={i.destructive ? "destructive" : undefined}
            onSelect={() => i.onClick()}
          >
            {i.label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

/** The shell every one of these dialogs is: a title, the fields, Cancel/Save.
 *  Save is disabled until there is a name, because a nameless team is a row
 *  the server drops on write and the user never learns why. */
function EditDialog({
  title,
  detail,
  canSave,
  busy,
  onClose,
  onSave,
  children,
}: {
  title: string;
  detail: string;
  canSave: boolean;
  busy: boolean;
  onClose: () => void;
  onSave: () => void;
  children: React.ReactNode;
}) {
  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="sm:max-w-[440px]">
        {/* This app's DialogHeader is a 48px ROW, not a stack — title and
            description sit side by side, so the description has to be a phrase
            rather than a sentence or the title wraps to two lines beside it. */}
        <DialogHeader>
          <DialogTitle className="shrink-0 whitespace-nowrap">{title}</DialogTitle>
          <DialogDescription className="truncate">{detail}</DialogDescription>
        </DialogHeader>
        <form
          className="grid gap-4"
          onSubmit={(e) => {
            e.preventDefault();
            if (canSave && !busy) onSave();
          }}
        >
          {children}
          <DialogFooter>
            <Button type="button" variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={!canSave || busy}>
              {busy ? "Saving…" : "Save"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function ClubDialog({
  club,
  busy,
  onClose,
  onSave,
}: {
  club: Org["club"];
  busy: boolean;
  onClose: () => void;
  onSave: (v: { name: string; season: string }) => void;
}) {
  const [name, setName] = useState(club.name);
  const [season, setSeason] = useState(club.season);
  return (
    <EditDialog
      title="Club"
      detail="Shown beside every match"
      canSave={Boolean(name.trim())}
      busy={busy}
      onClose={onClose}
      onSave={() => onSave({ name: name.trim(), season: season.trim() })}
    >
      <Field label="Club name" hint="Shown beside the match, top left.">
        <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Westfield FC" />
      </Field>
      <Field label="Current season" hint="Squad lists are kept per season.">
        <Input value={season} onChange={(e) => setSeason(e.target.value)} placeholder="2026/27" />
      </Field>
    </EditDialog>
  );
}

/** One squad: what the club declares. The age band is its own field rather than
 *  parsed from the name on every read - a club that calls a squad "Infantil A"
 *  still wants it on the U13 rung, and only a human knows that. It is pre-filled
 *  from the name on the server when the team is created, so the common case
 *  costs nothing. */
function TeamDialog({
  team,
  coaches,
  defaultSeason,
  busy,
  onClose,
  onSave,
}: {
  team: Team | null;
  coaches: Coach[];
  defaultSeason: string;
  busy: boolean;
  onClose: () => void;
  onSave: (v: Partial<Team> & { name: string }) => void;
}) {
  const [name, setName] = useState(team?.name ?? "");
  const [age, setAge] = useState(team?.age_band ?? "");
  const [level, setLevel] = useState(team?.level ?? "");
  const [gender, setGender] = useState(team?.gender ?? "");
  const [season, setSeason] = useState(team?.season ?? defaultSeason);
  const [staff, setStaff] = useState<string[]>(team?.coach_ids ?? []);

  return (
    <EditDialog
      title={team ? team.name || "Team" : "New team"}
      detail="Keeps its squad through a rename"
      canSave={Boolean(name.trim())}
      busy={busy}
      onClose={onClose}
      onSave={() =>
        onSave({
          name: name.trim(),
          age_band: age.trim(),
          level: level.trim(),
          gender: gender.trim(),
          season: season.trim(),
          coach_ids: staff,
        })
      }
    >
      <Field label="Name">
        <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="U15 Boys" />
      </Field>
      <div className="grid grid-cols-3 gap-3">
        <Field label="Age band">
          <Input value={age} onChange={(e) => setAge(e.target.value)} placeholder="U15" />
        </Field>
        <Field label="Level">
          <Input value={level} onChange={(e) => setLevel(e.target.value)} placeholder="A" />
        </Field>
        <Field label="Season">
          <Input
            value={season}
            onChange={(e) => setSeason(e.target.value)}
            placeholder={defaultSeason || "2026/27"}
          />
        </Field>
      </div>
      <p className="text-muted-foreground -mt-2 text-xs">
        Age band is where the squad sits on the ladder, and it is its own field rather than
        parsed from the name: a club that calls a squad “Infantil A” still wants it on the U13
        rung, and only a human knows that. Level separates two squads in one age group.
      </p>
      <Field label="Gender" hint="Optional — blank for a mixed squad.">
        <Input value={gender} onChange={(e) => setGender(e.target.value)} placeholder="Boys" />
      </Field>
      <Field label="Staff" hint="One coach runs two age groups; one team has three staff.">
        <div className="flex flex-wrap gap-1.5">
          {coaches.length ? (
            coaches.map((c) => {
              const on = staff.includes(c.coach_id);
              return (
                <button
                  key={c.coach_id}
                  type="button"
                  onClick={() =>
                    setStaff((p) =>
                      on ? p.filter((id) => id !== c.coach_id) : [...p, c.coach_id],
                    )
                  }
                  className={cn(
                    "h-7 rounded-[var(--radius-pill)] border border-[var(--hairline)] px-3 text-[12px] transition-colors",
                    on
                      ? "bg-[var(--card)] text-[var(--ink)]"
                      : "text-[var(--ink-soft)] hover:bg-[var(--card)] hover:text-[var(--ink)]",
                  )}
                >
                  {c.name}
                </button>
              );
            })
          ) : (
            <span className="text-muted-foreground text-xs">
              No staff yet — add a coach and they can be assigned here.
            </span>
          )}
        </div>
      </Field>
    </EditDialog>
  );
}

/** THE STAFF LIST, in a dialog rather than on the page.
 *
 *  It was a second table under the ladder, and it was the wrong thing to spend
 *  a screen on: the Teams tab is read many times a week to compare squads, and
 *  the staff list is edited about twice a season. Assignment already happens
 *  where it means something — the chips inside a team's own dialog — so all
 *  this has to be is somewhere to add a name and correct a spelling.
 *
 *  The whole list is edited at once, which is why this one keeps inputs rather
 *  than a row menu: three coaches in a club is the normal case, and a dialog
 *  per coach on top of a dialog listing coaches is a stack nobody wants.
 */
function StaffDialog({
  coaches,
  busy,
  onClose,
  onSave,
}: {
  coaches: Coach[];
  busy: boolean;
  onClose: () => void;
  onSave: (next: (Partial<Coach> & { name: string })[]) => void;
}) {
  const [rows, setRows] = useState<(Partial<Coach> & { name: string; key: string })[]>(() =>
    coaches.map((c) => ({ ...c, key: c.coach_id })),
  );
  const set = (i: number, next: Partial<Coach>) =>
    setRows((p) => p.map((r, j) => (j === i ? { ...r, ...next } : r)));

  return (
    <EditDialog
      title="Staff"
      detail="Records, not logins"
      canSave={rows.every((r) => r.name.trim() || !r.coach_id)}
      busy={busy}
      onClose={onClose}
      onSave={() => onSave(rows.filter((r) => r.name.trim()))}
    >
      <div className="grid gap-2">
        {rows.length === 0 && (
          <p className="text-muted-foreground text-xs">
            Nobody yet. A coach is a name you can hang a squad on — there are no accounts in this
            app.
          </p>
        )}
        {rows.map((c, i) => (
          <div key={c.key} className="grid grid-cols-[1fr_1fr_auto] gap-2">
            <Input
              value={c.name}
              placeholder="Name"
              onChange={(e) => set(i, { name: e.target.value })}
            />
            <Input
              value={c.role ?? ""}
              placeholder="Head coach"
              onChange={(e) => set(i, { role: e.target.value })}
            />
            <Button
              type="button"
              variant="ghost"
              size="icon"
              aria-label={`Remove ${c.name || "coach"}`}
              onClick={() => setRows((p) => p.filter((_, j) => j !== i))}
              className="text-muted-foreground hover:text-destructive"
            >
              <X className="size-4" />
            </Button>
          </div>
        ))}
      </div>
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="justify-self-start"
        onClick={() =>
          setRows((p) => [...p, { key: `new-${p.length}-${Date.now()}`, name: "", role: "" }])
        }
      >
        <Plus className="size-3.5" />
        Add coach
      </Button>
    </EditDialog>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="grid gap-1.5">
      <Label className="text-[13px]">{label}</Label>
      {children}
      {hint && <p className="text-muted-foreground text-xs">{hint}</p>}
    </div>
  );
}

/** The card the ladder sits in. No header — the browser above owns the title
 *  and the one-line note, and a second heading here is the same word twice. */
function Bare({ children }: { children: React.ReactNode }) {
  return (
    <Card className="py-0">
      <CardContent className="divide-y p-0">{children}</CardContent>
    </Card>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="text-muted-foreground p-4 text-sm">{children}</p>;
}
