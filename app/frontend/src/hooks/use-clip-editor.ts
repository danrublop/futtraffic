"use client";

import { useCallback, useEffect, useState } from "react";

import { api } from "@/lib/api";
import { clock } from "@/lib/timecode";
import type { Clip } from "@/lib/types";

/** Everything about marking, naming, saving and picking clips, lifted out of
 *  the film panel.
 *
 *  WHY IT MOVED. video-panel.tsx was 1693 lines and handed ClipsPanel 28
 *  separate props, eleven of which were this. A hook that returns ONE object is
 *  the whole fix: the panel takes one prop, and the thing that owns the state
 *  is the thing that names it.
 *
 *  Deliberately NOT a context. Every consumer of this — the rail, the timeline,
 *  the keyboard map — renders inside the film panel's own tree, and ClipsPanel
 *  is its direct child. A context between a parent and its own child is
 *  indirection that buys nothing and costs traceability.
 *
 *  What is NOT in here: the reel. Playing clips back to back is transport, not
 *  editing, and it needs the video element rather than this state.
 */
export interface ClipEditor {
  clips: Clip[];
  /** The fixed tag vocabulary, as the server returned it. Never hardcoded on
   *  the client: a list that drifts from jobs.CLIP_TAGS is silently dropped on
   *  save. */
  tags: string[];

  // ---- the draft being marked or edited
  in: number | null;
  out: number | null;
  name: string;
  note: string;
  tagSel: Set<string>;
  /** The id being edited, or null when this is a new clip. */
  editing: string | null;
  error: string | null;
  saving: boolean;

  // ---- the reel selection
  picked: Set<string>;
  /** The clip whose link was just copied, for 1.4s of feedback. */
  copied: string | null;

  setName: (v: string) => void;
  setNote: (v: string) => void;
  toggleTag: (t: string) => void;
  markIn: () => void;
  markOut: () => void;
  /** Set both points at once. Only the `?t=12.5-30` deep link uses this: a
   *  pasted clip link opens with the range already marked, so whoever it was
   *  sent to can save, re-cut or extend it rather than re-marking by hand. */
  setRange: (inT: number, outT: number) => void;
  clear: () => void;
  save: () => void;
  edit: (c: Clip) => void;
  remove: (c: Clip) => void;
  copyLink: (c: Clip) => void;
  pick: (id: string, on: boolean) => void;
  pickAll: (on: boolean) => void;
}

export function useClipEditor(
  matchId: string | null,
  /** Where the playhead is, for I and O. A getter rather than a number: this
   *  is read at the moment of the keypress and must not be a render-old value. */
  getTime: () => number,
  seek: (t: number) => void,
): ClipEditor {
  const [clips, setClips] = useState<Clip[]>([]);
  const [tags, setTags] = useState<string[]>([]);
  const [io, setIo] = useState<{ in: number | null; out: number | null }>({ in: null, out: null });
  const [name, setName] = useState("");
  const [note, setNote] = useState("");
  const [tagSel, setTagSel] = useState<Set<string>>(new Set());
  const [editing, setEditing] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [copied, setCopied] = useState<string | null>(null);

  useEffect(() => {
    if (!matchId) return;
    let live = true;
    api
      .listClips(matchId)
      .then((r) => {
        if (!live) return;
        setClips(r.clips);
        setTags(r.tags);
      })
      .catch((e: unknown) => {
        if (live) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      live = false;
    };
  }, [matchId]);

  // Marking one point clears the other when they cross: an out-point before its
  // in-point is not a range, and keeping it would let a save through that the
  // server then rejects.
  const markIn = useCallback(() => {
    const t = getTime();
    setIo((p) => ({ in: t, out: p.out != null && p.out <= t ? null : p.out }));
  }, [getTime]);

  const markOut = useCallback(() => {
    const t = getTime();
    setIo((p) => ({ in: p.in != null && p.in >= t ? null : p.in, out: t }));
  }, [getTime]);

  const setRange = useCallback(
    (inT: number, outT: number) => setIo({ in: inT, out: outT }),
    [],
  );

  const clear = useCallback(() => {
    setIo({ in: null, out: null });
    setEditing(null);
    setName("");
    setNote("");
    setTagSel(new Set());
    setError(null);
  }, []);

  const save = useCallback(() => {
    if (!matchId) return;
    if (io.in == null || io.out == null || io.out <= io.in) {
      setError("Mark an in-point (I) and an out-point (O) after it.");
      return;
    }
    setSaving(true);
    setError(null);
    const start = io.in;
    void api
      // Keyed on SECONDS. Never frames or track ids — re-tracking renumbers
      // those and would orphan an evening of work.
      .saveClip(matchId, {
        ...(editing ? { id: editing } : {}),
        name: name.trim() || `Clip ${clock(start)}`,
        start,
        end: io.out,
        tags: tags.filter((t) => tagSel.has(t)),
        note: note.trim(),
      })
      .then((r) => {
        setClips(r.clips);
        setTags(r.tags);
        clear();
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setSaving(false));
  }, [matchId, io, editing, name, note, tags, tagSel, clear]);

  const edit = useCallback(
    (c: Clip) => {
      setEditing(c.id);
      setIo({ in: c.start, out: c.end });
      setName(c.name);
      setNote(c.note);
      setTagSel(new Set(c.tags));
      seek(c.start);
    },
    [seek],
  );

  const remove = useCallback(
    (c: Clip) => {
      if (!matchId || !window.confirm(`Delete "${c.name}"?`)) return;
      void api
        .deleteClip(matchId, c.id)
        .then((r) => {
          setClips(r.clips);
          setPicked((p) => {
            const n = new Set(p);
            n.delete(c.id);
            return n;
          });
          if (editing === c.id) clear();
        })
        .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
    },
    [matchId, editing, clear],
  );

  const copyLink = useCallback(
    (c: Clip) => {
      const url = `${location.origin}/review?m=${encodeURIComponent(
        matchId ?? "",
      )}&t=${c.start.toFixed(2)}-${c.end.toFixed(2)}`;
      void navigator.clipboard?.writeText(url).then(
        () => {
          setCopied(c.id);
          window.setTimeout(() => setCopied(null), 1400);
        },
        () => window.prompt("Copy this link", url),
      );
    },
    [matchId],
  );

  const toggleTag = useCallback((t: string) => {
    setTagSel((p) => {
      const n = new Set(p);
      if (n.has(t)) n.delete(t);
      else n.add(t);
      return n;
    });
  }, []);

  const pick = useCallback((cid: string, on: boolean) => {
    setPicked((p) => {
      const n = new Set(p);
      if (on) n.add(cid);
      else n.delete(cid);
      return n;
    });
  }, []);

  const pickAll = useCallback(
    (on: boolean) => setPicked(on ? new Set(clips.map((c) => c.id)) : new Set()),
    [clips],
  );

  return {
    clips,
    tags,
    in: io.in,
    out: io.out,
    name,
    note,
    tagSel,
    editing,
    error,
    saving,
    picked,
    copied,
    setName,
    setNote,
    toggleTag,
    markIn,
    markOut,
    setRange,
    clear,
    save,
    edit,
    remove,
    copyLink,
    pick,
    pickAll,
  };
}
