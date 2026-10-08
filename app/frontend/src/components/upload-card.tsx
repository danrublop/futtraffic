"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import { CloudUpload, X } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { api, uploadMatch } from "@/lib/api";
import { reloadOrg, useOrg, useTeam } from "@/lib/club";
import type { Match } from "@/lib/types";

const GB = 1e9;
const fmt = (b: number) => (b >= GB ? `${(b / GB).toFixed(2)} GB` : `${Math.round(b / 1e6)} MB`);

interface Partial {
  job: Match;
  offset: number;
}

/** Distinct, non-empty values of one field across the matches already on the
 *  box. What makes the ground and club fields a picker over what this club
 *  actually uses instead of free text that spawns four spellings of one pitch. */
const seen = (list: Match[], pick: (m: Match) => string) =>
  [...new Set(list.map(pick).filter(Boolean))].sort((a, b) => a.localeCompare(b));

export function UploadCard({ onDone }: { onDone: () => void }) {
  // The club's OWN ladder is the authority on what a team is now, not the
  // strings that happen to be on existing matches. Whichever squad the app is
  // scoped to is the default, because a coach uploading a match is almost
  // always uploading THEIR team's match.
  const { org } = useOrg();
  const [scope] = useTeam();
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [date, setDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [team, setTeam] = useState("");
  const [venue, setVenue] = useState("");
  const [known, setKnown] = useState<Match[]>([]);
  const [sent, setSent] = useState(0);
  const [busy, setBusy] = useState(false);
  const [partial, setPartial] = useState<Partial | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const abort = useRef<AbortController | null>(null);

  // Ladder first, with anything already on a match folded in — a team name
  // that predates the club screen must not vanish from the picker just because
  // nobody has added it as a squad yet.
  const teams = useMemo(() => {
    const ladder = (org?.teams ?? []).map((t) => t.name);
    return [...new Set([...ladder, ...seen(known, (m) => m.team ?? "")])];
  }, [org, known]);
  const venues = useMemo(() => {
    // A club owns its grounds. But a club being typed for the first time owns
    // none yet, and offering an empty list is worse than offering everyone's.
    const mine = known.filter(
      (m) => (m.team ?? "").toLowerCase() === team.trim().toLowerCase(),
    );
    return seen(mine.length ? mine : known, (m) => m.venue ?? "");
  }, [known, team]);

  /** Look for an unfinished upload of a file this exact size, so resume is
   *  offered BEFORE the button is pressed rather than discovered halfway. */
  const choose = useCallback(async (f: File) => {
    setFile(f);
    setSent(0);
    setPartial(null);
    if (!name) setName(f.name.replace(/\.[^.]+$/, ""));
    try {
      const list = await api.listMatches();
      setKnown(list);
      // The club carries over from the last upload; the GROUND deliberately
      // does not. Pre-filling the ground would default every away fixture to
      // the home pitch, which is the mistake that made this field exist.
      // The scoped squad wins; then whatever the last match used. A coach
      // working on the U15s should never have to retype "U15 Boys" per upload.
      const scoped = (org?.teams ?? []).find((t) => t.team_id === scope)?.name;
      setTeam((t) => t || scoped || list[0]?.team || "");
      const hit = list.find(
        (m) => !m.upload_complete && m.total_bytes === f.size && m.uploaded_bytes > 0,
      );
      if (hit) {
        // Resuming: the club and ground were fixed when this job was created
        // and /upload cannot change them, so show that job's, not a guess.
        setPartial({ job: hit, offset: hit.uploaded_bytes });
        setTeam(hit.team || "");
        setVenue(hit.venue || "");
      }
    } catch {
      /* no backend: the button will say so */
    }
  }, [name, org, scope]);

  const start = async () => {
    if (!file) return;
    setBusy(true);
    abort.current = new AbortController();
    try {
      await uploadMatch(file, {
        name: `${name || file.name} ${date}`.trim(),
        venue: venue.trim(),
        team: team.trim(),
        jobId: partial?.job.id,
        offset: partial?.offset,
        onProgress: setSent,
        signal: abort.current.signal,
      });
      toast.success("Uploaded", {
        description: "Processing starts on its own. You can close this page.",
      });
      // A team typed here that is not on the ladder yet gets seeded onto it
      // server-side; drop the cache so the sidebar shows it without a reload.
      reloadOrg();
      setFile(null);
      setName("");
      setVenue("");
      setPartial(null);
      setSent(0);
      onDone();
    } catch (e) {
      toast.error("Upload failed", {
        description: e instanceof Error ? e.message : String(e),
      });
    } finally {
      setBusy(false);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) void choose(f);
  };

  const pct = file && sent ? (sent / file.size) * 100 : 0;

  return (
    <Card
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={dragging ? "border-primary border-dashed" : undefined}
    >
      <CardContent className="p-4 md:p-6">
        {!file ? (
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="hover:border-primary/60 hover:bg-accent/40 flex w-full flex-col items-center gap-2 rounded-[var(--radius-md)] border border-dashed px-4 py-8 text-center transition"
          >
            <CloudUpload className="text-muted-foreground size-6" />
            <span className="text-sm font-medium">Add a match</span>
            <span className="text-muted-foreground text-xs">
              Drop a video here, or click to pick one. Processing starts by itself.
            </span>
          </button>
        ) : (
          <div className="grid gap-4">
            <div className="flex items-center gap-3">
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium">{file.name}</div>
                <div className="text-muted-foreground text-xs">{fmt(file.size)}</div>
              </div>
              {!busy && (
                <Button variant="ghost" size="icon" onClick={() => setFile(null)}>
                  <X />
                </Button>
              )}
            </div>

            <div className="grid gap-3 sm:grid-cols-[1fr_auto]">
              <div className="grid gap-1.5">
                <Label htmlFor="mname">Match</Label>
                <Input
                  id="mname"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Westfield vs Rovers"
                  disabled={busy}
                />
              </div>
              <div className="grid gap-1.5">
                <Label htmlFor="mdate">Date</Label>
                <Input
                  id="mdate"
                  type="date"
                  value={date}
                  onChange={(e) => setDate(e.target.value)}
                  disabled={busy}
                />
              </div>
            </div>

            {/* The ground is a safety field, not a label. A pitch setup is only
                reused across matches at the SAME ground: two 1080p matches at
                different grounds used to share one homography, which rescaled
                every distance on the report with nothing looking wrong.
                <datalist> rather than a select: it is both a picker over the
                grounds this club already uses AND a plain text box for a new
                one, natively, with no component and no "add new" mode. */}
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="grid gap-1.5">
                <Label htmlFor="mteam">Team</Label>
                <Input
                  id="mteam"
                  list="known-teams"
                  value={team}
                  onChange={(e) => setTeam(e.target.value)}
                  placeholder="Westfield"
                  disabled={busy || !!partial}
                />
                <datalist id="known-teams">
                  {teams.map((t) => (
                    <option key={t} value={t} />
                  ))}
                </datalist>
              </div>
              <div className="grid gap-1.5">
                <Label htmlFor="mvenue">Ground</Label>
                <Input
                  id="mvenue"
                  list="known-venues"
                  value={venue}
                  onChange={(e) => setVenue(e.target.value)}
                  placeholder="Westfield Park"
                  disabled={busy || !!partial}
                />
                <datalist id="known-venues">
                  {venues.map((v) => (
                    <option key={v} value={v} />
                  ))}
                </datalist>
                <p className="text-muted-foreground text-xs">
                  {venue.trim() && venues.includes(venue.trim())
                    ? "Same ground as before — the pitch setup is reused, no clicking corners."
                    : "Pick a ground you have used, or type a new one. Pitch setup is only ever reused within one ground."}
                </p>
              </div>
            </div>

            {partial && !busy && (
              <p className="text-warn text-xs">
                {fmt(partial.offset)} of this file is already uploaded. Continuing from there.
              </p>
            )}

            {busy && (
              <div className="grid gap-1.5">
                <Progress value={pct} className="h-2" />
                <p className="text-muted-foreground text-xs">
                  {fmt(sent)} of {fmt(file.size)} · {Math.round(pct)}%
                </p>
              </div>
            )}

            <div className="flex gap-2">
              {/* No ground, no upload: a match with no ground recorded can never
                  reuse a pitch setup, so the operator would be clicking corners
                  again for a camera that has not moved. */}
              <Button onClick={start} disabled={busy || !venue.trim()}>
                {busy ? "Uploading…" : partial ? "Resume upload" : "Upload"}
              </Button>
              {busy && (
                <Button variant="outline" onClick={() => abort.current?.abort()}>
                  Cancel
                </Button>
              )}
            </div>
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void choose(f);
          }}
        />
      </CardContent>
    </Card>
  );
}
