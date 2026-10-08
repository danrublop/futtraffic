/** WHERE A REJECTED WRITE GOES INSTEAD OF NOWHERE.
 *
 *  Switching notes unmounts the editor mid-debounce, so doc-panel flushes the
 *  last text on the way out. That flush has no UI left to complain to, and it
 *  used to end `.catch(() => {})`.
 *
 *  That was survivable while the only way to lose was a network blip. It is not
 *  survivable now that writes carry a version: a stale write comes back 409,
 *  and swallowing THAT turns a loud clobber into a quiet deletion — the coach
 *  switches notes, the last thing they typed is refused, and nothing anywhere
 *  says so. A guard whose rejections are invisible is worse than no guard.
 *
 *  sessionStorage rather than localStorage: this is a crash mat for the current
 *  sitting, not a second copy of the library to go stale on disk beside the
 *  real one. Every call is wrapped, because storage throws outright in private
 *  mode and when the quota is gone, and losing the mat must not also lose the
 *  navigation that was trying to use it.
 */
const key = (path: string) => `ft-note-unsaved:${path}`;

/** Hold text that could not be written. Silent on failure: there is nothing
 *  better to do, and throwing here would break the note switch as well. */
export function park(path: string, html: string): void {
  try {
    sessionStorage.setItem(key(path), html);
  } catch {
    /* private mode, or full */
  }
}

/** Take it back, once. Consumed on read so a recovered note that then saves
 *  normally does not keep offering the same rescue every time it is opened. */
export function unpark(path: string): string | null {
  try {
    const html = sessionStorage.getItem(key(path));
    if (html !== null) sessionStorage.removeItem(key(path));
    return html;
  } catch {
    return null;
  }
}
