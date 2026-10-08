"""EVIDENCE GENERATOR, NOT A GATE: does the VIDEO show a PASS release at the
corrected label time, or a second later?

WHY IT HAD TO BE THE VIDEO. Every other check on the LABEL OFFSET is referenced to
the FEED's ball, including the functional that chose the offset and G0's rule (b)
that then tests it (st2-p1-step6 (not published), "rule (b) is near-tautological"). A
ball-referenced check cannot tell a late feed ball from an early label. This cuts
crops from the STC clip itself, around the labelled actor's MOT box, at the
CORRECTED label frame, +0.5 s and +1.0 s, and a human looks at where the ball
leaves the foot. The clip's own pixels are the independent instrument.

It produces a contact sheet and nothing else. It computes no statistic, has no
pass rule, and the count in step 6 is a human read of its output, carried as a
citation and not as a measurement. Do not
wire it into a gate.

    .venv/bin/python model/validation/st2_offset_sheet.py OUT.png <seq> [seed] [n]

BOTH SHEETS THE STEP 6 TALLY RESTS ON ARE REPRODUCIBLE FROM HERE: sheet 1 is
`seed 0, n 8` (the defaults) and sheet 2 is `seed 1, n 16`. Rows are written eight
to an image, suffixed `_0`, `_1`, ..., because sixteen rows do not fit on a screen.
The sheet images themselves are NOT committed - they are large, and they are an
input to a human read rather than an artefact anything re-checks.

`ponytail: the clip and MOT paths are the absolute ones this was run with, left
alone so the sheet reproduces exactly. Upgrade path is st2_team_gt's roots if this
is ever run anywhere else.`
"""
import sys, random, collections
import numpy as np, cv2
sys.path[:0] = ['model/validation', 'model/engine']
import st2_team_gt as G, st2_cache as S
from st2_cache import PERIODS
import os
_ST2 = os.path.expanduser(os.environ.get("ST2_DIR", "~/Downloads/st2"))  # SoccerTrack v2 download
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
N_PER_CLIP = int(sys.argv[4]) if len(sys.argv) > 4 else 8
random.seed(SEED)
J = G.load()
T = (0, 12, 25)                    # label, +0.5 s, +1.0 s at 25 fps
rows_img = []
for seq in (__import__('sys').argv[2],):
    j, c = J[seq], S.load(seq)
    k = PERIODS.index(j['period'])
    fn = c[f'p{k}_frame_number']
    p2m = {int(v): int(q) for q, v in j['mot_to_player'].items()}
    mot = collections.defaultdict(dict)
    for line in open(_ST2 + f'/mot/{seq}.txt'):
        f, i, x, y, w, h = line.split(',')[:6]
        mot[int(f)][int(i)] = (float(x), float(y), float(w), float(h))
    cand = []
    for e in np.where((c['ev_period'] == k) & (c['ev_label'] == 'PASS') & (c['ev_row'] >= 0))[0]:
        cf = int(fn[c['ev_row'][e]]) - j['offset']
        mid = p2m.get(int(c['ev_pid'][e]))
        if 0 <= cf < j['clip_frames'] - 30 and mid in mot[cf]:
            cand.append((cf, mid, float(c['ev_t_s'][e])))
    pick = random.sample(cand, min(N_PER_CLIP, len(cand)))
    cap = cv2.VideoCapture(_ST2 + f'/mot/clips/{seq}.mp4')
    for cf, mid, ts in sorted(pick):
        x, y, w, h = mot[cf][mid]
        cx, cy = int(x + w / 2), int(y + h)
        tiles = []
        for d in T:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cf + d)
            ok, im = cap.read()
            b = mot[cf + d].get(mid)
            if b:
                cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[0] + b[2]), int(b[1] + b[3])), (0, 255, 255), 1)
            x0, y0 = max(cx - 160, 0), max(cy - 110, 0)
            t = cv2.resize(im[y0:y0 + 200, x0:x0 + 320], (480, 300), interpolation=cv2.INTER_CUBIC)
            cv2.putText(t, f'{seq} t={ts:.1f}s {"+%.1fs" % (d / 25)}', (6, 20), 0, 0.55, (255, 255, 255), 2)
            tiles.append(t)
        rows_img.append(np.hstack(tiles))
base, ext = sys.argv[1].rsplit('.', 1)
for i in range(0, len(rows_img), 8):
    cv2.imwrite(f'{base}_{i // 8}.{ext}', np.vstack(rows_img[i:i + 8]))
print('rows', len(rows_img), 'seed', SEED, 'per clip', N_PER_CLIP)
