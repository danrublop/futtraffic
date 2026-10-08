/* tracks.json -> frame index + derived moments, entirely off the main thread.
 *
 * The whole reason this file exists: tracks.json for a real match is ~300 MB.
 * `JSON.parse` on that string needs the 300 MB of text AND the ~1 GB object
 * graph resident at the same time, on the main thread, with no progress and no
 * way back if it OOMs. So we stream the response, pull out one detection object
 * at a time, and land it straight into typed arrays - 14 bytes per detection
 * instead of ~200. A 3M-detection match is ~42 MB of memory and the tab never
 * blocks, because none of it happens on the main thread.
 *
 * Everything expensive lives here: the scan, the frame index, the speed
 * estimate and the moment derivation. The page only draws.
 */

// `default`, NOT `no-store`. tracks.json is the biggest thing this product
// moves - 60 MB on a real match - and the server now answers 304 on
// If-None-Match. `no-store` means the browser never keeps a copy, so it never
// sends the validator, so the 304 can never fire and all 60 MB crosses the wire
// on every single visit to the review page.
//
// Safe because the file is IMMUTABLE once written: a re-track writes a new
// tracks.json, which changes its mtime and therefore its ETag, and the browser
// revalidates rather than serving blind from cache. That is exactly the case
// conditional requests exist for. The staleness worry `no-store` was guarding
// against is real for job STATUS, which is polled and mutates; it was never
// true of this file.
const NOCACHE = { cache: 'default' };

// stats.py constants. Duplicated deliberately and marked so: if these drift
// from stats.py the timeline stops agreeing with the numbers on every other
// page, which is worse than the duplication.
const MAX_SPEED = 9.5;      // stats.MAX_SPEED - the clamp, i.e. suspect tracking
const SPRINT_SPEED = 5.5;   // stats.SPRINT_SPEED
const SPRINT_MIN_S = 0.8;   // stats.SPRINT_MIN_S
const SMOOTH_S = 0.4;       // stats.SMOOTH_S, boxcar window

// ---------------------------------------------------------------- streaming

/** Growable Int32/Float32 column. Doubling, because we cannot know the count
 *  until the stream ends and Content-Length is bytes, not detections. */
function column(Type, cap = 1 << 16) {
  return { a: new Type(cap), n: 0,
    push(v) {
      if (this.n === this.a.length) {
        const b = new Type(this.a.length * 2);
        b.set(this.a); this.a = b;
      }
      this.a[this.n++] = v;
    },
    done() { return this.a.subarray(0, this.n); } };
}

/** Incremental scanner over the `detections` array.
 *
 * Not a JSON parser: it finds the array, then hands each top-level `{...}`
 * inside it to JSON.parse individually. Brace depth is tracked with string and
 * escape state so a `}` inside a team name can never end an object early.
 * State persists across chunks, so an object split down the middle by a network
 * boundary is resumed rather than dropped.
 */
function makeScanner(emit) {
  let buf = '', pos = 0, objStart = -1, depth = 0, inStr = false, esc = false;
  let mode = 'seek';
  return {
    get done() { return mode === 'done'; },
    feed(chunk) {
      buf += chunk;
      if (mode === 'seek') {
        const k = buf.indexOf('"detections"');
        if (k < 0) { buf = buf.slice(-32); pos = 0; return; }   // key split across chunks
        const b = buf.indexOf('[', k);
        if (b < 0) { buf = buf.slice(k); pos = 0; return; }
        pos = b + 1; mode = 'array';
      }
      if (mode !== 'array') return;
      const n = buf.length;
      let i = pos;
      while (i < n) {
        const c = buf[i];
        if (objStart < 0) {
          if (c === '{') { objStart = i; depth = 1; inStr = false; esc = false; }
          else if (c === ']') { mode = 'done'; i++; break; }
          i++; continue;
        }
        if (inStr) {
          if (esc) esc = false;
          else if (c === '\\') esc = true;
          else if (c === '"') inStr = false;
        } else if (c === '"') inStr = true;
        else if (c === '{') depth++;
        else if (c === '}' && --depth === 0) {
          emit(buf.slice(objStart, i + 1));
          objStart = -1;
        }
        i++;
      }
      pos = i;
      // Compact. Without this `buf` grows to the whole file and we are back to
      // holding 300 MB of text, which is the thing this file exists to avoid.
      const keep = objStart >= 0 ? objStart : pos;
      if (keep > 0) { buf = buf.slice(keep); pos -= keep; if (objStart >= 0) objStart = 0; }
    },
  };
}

async function load(url) {
  const res = await fetch(url, NOCACHE);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  const total = Number(res.headers.get('content-length')) || 0;

  const F = column(Int32Array), ID = column(Int32Array);
  const X = column(Float32Array), Y = column(Float32Array);
  const meta = new Map();           // id -> {team, jersey}
  let fps = 0, seen = 0, head = '';

  const scan = makeScanner(src => {
    let d; try { d = JSON.parse(src); } catch { return; }   // skip a mangled record, never abort the load
    if (d.f == null || d.id == null || d.x == null || d.y == null) return;
    F.push(d.f); ID.push(d.id); X.push(d.x); Y.push(d.y);
    let m = meta.get(d.id);
    if (!m) meta.set(d.id, m = { team: d.team ?? null, jersey: d.jersey ?? null });
    // Both are per-detection in the contract but constant per track in
    // practice (track.py assigns them per id). First non-null wins.
    if (m.team == null && d.team != null) m.team = d.team;
    if (m.jersey == null && d.jersey != null) m.jersey = d.jersey;
  });

  const dec = new TextDecoder();
  const reader = res.body.getReader();
  let bytes = 0, tick = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    bytes += value.byteLength;
    const text = dec.decode(value, { stream: true });
    if (head.length < 400) head += text.slice(0, 400);
    scan.feed(text);
    if (++tick % 8 === 0 || F.n - seen > 200000) {
      seen = F.n;
      postMessage({ type: 'progress', bytes, total, n: F.n });
    }
    if (scan.done) { reader.cancel().catch(() => {}); break; }
  }
  const m = /"fps"\s*:\s*([0-9.eE+-]+)/.exec(head);
  fps = m ? parseFloat(m[1]) : 0;
  postMessage({ type: 'progress', bytes, total, n: F.n });
  // The one failure this streaming reader can hide: the connection drops or the
  // file was written by an interrupted run, and we return a partial index that
  // looks exactly like a short match. The closing `]` is the only proof we read
  // all of it, so its absence is reported rather than shrugged off.
  const truncated = !scan.done
    ? (F.n === 0 ? 'no "detections" array found in this file'
                 : `stream ended before the detections array closed - the file is truncated, so this is a partial index`)
    : null;
  return { F: F.done(), ID: ID.done(), X: X.done(), Y: Y.done(), meta, fps, bytes, truncated };
}

// ------------------------------------------------------------- frame index

/** Counting sort by frame. O(n + span) and it produces the frame offsets as a
 *  by-product, which a comparator sort would not - and a comparator sort of 3M
 *  Uint32s is seconds, not milliseconds. */
function index(raw) {
  const { F, ID, X, Y } = raw, n = F.length;
  if (!n) return { n: 0 };
  let fmin = Infinity, fmax = -Infinity;
  for (let i = 0; i < n; i++) { const f = F[i]; if (f < fmin) fmin = f; if (f > fmax) fmax = f; }
  const span = fmax - fmin + 1;

  const start = new Int32Array(span + 1);
  for (let i = 0; i < n; i++) start[F[i] - fmin + 1]++;
  for (let k = 0; k < span; k++) start[k + 1] += start[k];

  const cur = start.slice(0, span);
  const sF = new Int32Array(n), sID = new Int32Array(n);
  const sX = new Float32Array(n), sY = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const j = cur[F[i] - fmin]++;
    sF[j] = F[i]; sID[j] = ID[i]; sX[j] = X[i]; sY[j] = Y[i];
  }

  // Forward-filled "last frame at or before this one that has any detection".
  // Detections are routinely sparse (a pipeline sampling every 5th frame is
  // normal), so a bare per-frame lookup would blank the overlay 4 frames in 5.
  const prev = new Int32Array(span).fill(-1);
  let last = -1;
  for (let k = 0; k < span; k++) { if (start[k + 1] > start[k]) last = k; prev[k] = last; }

  return { n, fmin, fmax, span, start, prev, F: sF, ID: sID, X: sX, Y: sY };
}

function tracks(ix, meta) {
  const idx = new Map();
  for (let i = 0; i < ix.n; i++) {
    let a = idx.get(ix.ID[i]);
    if (!a) idx.set(ix.ID[i], a = []);
    a.push(i);
  }
  return [...idx].map(([id, a]) => {
    const m = meta.get(id) || {};
    return { id, team: m.team ?? null, jersey: m.jersey ?? null,
      idx: Uint32Array.from(a), n: a.length,
      firstF: ix.F[a[0]], lastF: ix.F[a[a.length - 1]] };
  }).sort((a, b) => b.n - a.n);
}

// ------------------------------------------------------------------ speed

const applyH = (H, u, v) => {
  const w = H[2][0] * u + H[2][1] * v + H[2][2] || 1e-12;
  return [(H[0][0] * u + H[0][1] * v + H[0][2]) / w,
          (H[1][0] * u + H[1][1] * v + H[1][2]) / w];
};

/** stats.py's `method="boxcar"` path: moving-average the ground positions, then
 *  finite-difference and clamp. NOT the Kalman path that produces the headline
 *  numbers - reimplementing that filter in JS to place markers on a strip would
 *  be a lot of code to agree with a number the user can already read elsewhere.
 *  The page says which estimator it used; the two disagree and pretending
 *  otherwise is the dishonest option. */
function speeds(t, xy, fps) {
  const n = t.length;
  if (n < 2) return { v: new Float32Array(0), real: new Uint8Array(0) };
  const diffs = [];
  for (let i = 1; i < n; i++) diffs.push(t[i] - t[i - 1]);
  diffs.sort((a, b) => a - b);
  const step = diffs[diffs.length >> 1] || 0.04;
  const w = Math.max(1, Math.round(SMOOTH_S / step));

  const sx = new Float64Array(n), sy = new Float64Array(n), h = w >> 1;
  for (let i = 0; i < n; i++) {
    let ax = 0, ay = 0;
    for (let k = -h; k <= h; k++) {              // edge-clamped, same as np.pad(mode='edge')
      const j = Math.min(n - 1, Math.max(0, i + k));
      ax += xy[2 * j]; ay += xy[2 * j + 1];
    }
    sx[i] = ax / (2 * h + 1); sy[i] = ay / (2 * h + 1);
  }

  const tol = Math.max(3 * step, 2 / fps);   // stats.speed_series' gap rule, verbatim
  const v = new Float32Array(n - 1), real = new Uint8Array(n - 1);
  for (let i = 1; i < n; i++) {
    const dt = t[i] - t[i - 1];
    const d = Math.hypot(sx[i] - sx[i - 1], sy[i] - sy[i - 1]);
    v[i - 1] = dt > 0 ? Math.min(d / dt, MAX_SPEED) : 0;
    // A gap is occlusion or a dropped frame. The player did move; we did not
    // see it and must not attribute a path to it.
    real[i - 1] = dt > 0 && dt <= tol ? 1 : 0;
  }
  return { v, real };
}

// ---------------------------------------------------------------- moments

/** Every moment here is DERIVED from tracked positions. None is a detected
 *  event: there is no ball and no event model anywhere in this pipeline. */
function moments(ix, trk, cal, fps) {
  const out = [];
  const H = cal && cal.H, L = cal && cal.pitch ? cal.pitch.length : 0;
  const tOf = f => (f - ix.fmin) / fps;
  const inPitch = new Map();   // track id -> Float32Array of pitch x, for territory

  for (const tr of trk) {
    const a = tr.idx, n = a.length;
    if (tr.firstF > ix.fmin) out.push({ f: tr.firstF, id: tr.id, type: 'appear' });
    if (tr.lastF < ix.fmax) out.push({ f: tr.lastF, id: tr.id, type: 'vanish' });
    if (!H || n < 3) { tr.topSpeed = null; tr.clampedFrac = null; tr.sprints = null; continue; }

    const t = new Float64Array(n), xy = new Float64Array(2 * n), px = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      t[i] = (ix.F[a[i]] - ix.fmin) / fps;
      const p = applyH(H, ix.X[a[i]], ix.Y[a[i]]);
      xy[2 * i] = p[0]; xy[2 * i + 1] = p[1]; px[i] = p[0];
    }
    inPitch.set(tr.id, px);
    const { v, real } = speeds(t, xy, fps);

    let best = -1, bestV = 0, nClamp = 0, nReal = 0, sprints = 0;
    for (let i = 0; i < v.length; i++) {
      if (!real[i]) continue;
      nReal++;
      if (v[i] > bestV) { bestV = v[i]; best = i; }
      if (v[i] >= MAX_SPEED - 1e-6) nClamp++;
    }
    tr.topSpeed = best >= 0 ? bestV : null;
    tr.clampedFrac = nReal ? nClamp / nReal : null;
    if (best >= 0 && bestV > 0)
      out.push({ f: ix.F[a[best + 1]], id: tr.id, type: 'peak', v: bestV });

    // Sprint starts: >= SPRINT_SPEED sustained >= SPRINT_MIN_S, same rule as
    // stats._count_sprints. The marker goes on the START, which is the instant
    // worth watching; the count is the same either way.
    let run = -1;
    const close = i => {
      if (run < 0) return;
      if (t[i] - t[run] >= SPRINT_MIN_S) {
        sprints++;
        out.push({ f: ix.F[a[run + 1]], id: tr.id, type: 'sprint',
                   dur: t[i] - t[run], v: Math.max(...v.slice(run, i)) });
      }
      run = -1;
    };
    for (let i = 0; i < v.length; i++) {
      if (v[i] >= SPRINT_SPEED) { if (run < 0) run = i; }
      else close(i);
    }
    close(v.length);
    tr.sprints = sprints;

    // Clamp runs. These are NOT athletic feats - MAX_SPEED is a ceiling stats.py
    // imposes precisely because nobody runs that fast, so hitting it means the
    // track jumped. Grouped so one bad stretch is one marker, not two hundred.
    let cs = -1;
    for (let i = 0; i <= v.length; i++) {
      const hot = i < v.length && real[i] && v[i] >= MAX_SPEED - 1e-6;
      if (hot && cs < 0) cs = i;
      else if (!hot && cs >= 0) {
        out.push({ f: ix.F[a[cs + 1]], id: tr.id, type: 'clamp', dur: t[i] - t[cs] });
        cs = -1;
      }
    }
  }

  // Territory concentration. Sampled at 2 Hz - per frame would produce a
  // marker every 40 ms and say nothing a 2 Hz sample does not.
  if (H && L > 0 && inPitch.size) {
    const pos = new Map();                       // id -> {frames, px} cursor
    for (const tr of trk) if (inPitch.has(tr.id)) pos.set(tr.id, { tr, px: inPitch.get(tr.id), k: 0 });
    const step = Math.max(1, Math.round(fps / 2));
    let runStart = -1, runThird = -1;
    for (let f = ix.fmin; f <= ix.fmax + step; f += step) {
      const c = [0, 0, 0];
      let tot = 0;
      for (const p of pos.values()) {
        if (p.tr.team === 'official') continue;
        const a = p.tr.idx;
        while (p.k < a.length - 1 && ix.F[a[p.k]] < f) p.k++;
        if (Math.abs(ix.F[a[p.k]] - f) > step) continue;
        const third = Math.min(2, Math.max(0, Math.floor(p.px[p.k] / L * 3)));
        c[third]++; tot++;
      }
      const top = c.indexOf(Math.max(...c));
      const hit = tot >= 6 && c[top] / tot >= 0.75 ? top : -1;
      if (hit !== runThird) {
        if (runThird >= 0 && runStart >= 0 && f - runStart >= fps)
          out.push({ f: runStart, id: null, type: 'territory', third: runThird,
                     dur: (f - runStart) / fps });
        runThird = hit; runStart = hit >= 0 ? f : -1;
      }
    }
  }

  for (const m of out) m.t = tOf(m.f);
  out.sort((a, b) => a.t - b.t);
  return out;
}

// -------------------------------------------------------------------- main

onmessage = async e => {
  const { tracksUrl, calibUrl, fpsHint } = e.data;
  try {
    let cal = null, calErr = null;
    if (calibUrl) {
      try {
        const r = await fetch(calibUrl, NOCACHE);
        if (!r.ok) throw new Error(`${r.status}`);
        cal = await r.json();
        const p = cal.projection_type || 'perspective';
        // calibrators.py has panoramic and remap models too. They are not a
        // 3x3 and evaluating them here would mean reimplementing a weighted
        // piecewise-homography solver in the browser. Refuse loudly instead of
        // quietly using the global-fit H, which on a real panorama is metres
        // wrong and would put fabricated speeds on the timeline.
        if (p !== 'perspective') { calErr = `calibration is "${p}", not perspective`; cal = null; }
        else if (!Array.isArray(cal.H)) { calErr = 'calibration has no 3x3 H'; cal = null; }
      } catch (err) { calErr = String(err.message || err); cal = null; }
    }

    const t0 = performance.now();
    const raw = await load(tracksUrl);
    const ix = index(raw);
    if (!ix.n) { postMessage({ type: 'done', empty: true, calErr, truncated: raw.truncated }); return; }
    const fps = raw.fps || fpsHint || 25;
    const trk = tracks(ix, raw.meta);
    const mom = moments(ix, trk, cal, fps);

    postMessage({
      type: 'done',
      fps, calErr, truncated: raw.truncated, bytes: raw.bytes, ms: Math.round(performance.now() - t0),
      n: ix.n, fmin: ix.fmin, fmax: ix.fmax, span: ix.span,
      F: ix.F, ID: ix.ID, X: ix.X, Y: ix.Y, start: ix.start, prev: ix.prev,
      hasCalib: !!cal, pitch: cal ? cal.pitch : null,
      rms: cal ? cal.rms_error_m : null,
      tracks: trk.map(t => ({ id: t.id, team: t.team, jersey: t.jersey, n: t.n,
        firstF: t.firstF, lastF: t.lastF, idx: t.idx,
        topSpeed: t.topSpeed, clampedFrac: t.clampedFrac, sprints: t.sprints })),
      moments: mom,
    }, [ix.F.buffer, ix.ID.buffer, ix.X.buffer, ix.Y.buffer, ix.start.buffer, ix.prev.buffer,
        ...trk.map(t => t.idx.buffer)]);
  } catch (err) {
    postMessage({ type: 'error', message: String(err && err.message || err) });
  }
};
