"""Bootstrap N40's V0 vs V2 link-AUC difference (docs/experiments/n40-association.md) - pairs and exit-tracklet clusters, 2000 draws, seed 0.

    .venv/bin/python model/validation/n40_boot.py
"""
import sys, time, json
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "model", "validation"), os.path.join(ROOT, "model", "engine")]
import numpy as np
import n40_association as N
import stitch as S

t0 = time.time()
pooled = {v: [] for v in ("V0", "V1", "V2")}
labels, exits, clips = [], [], []
for q in N.DEV:
    rows, tl, px = N._split_tracklets(q)
    ident = N.gt_identity(q, rows)
    known = {tid for tid, (g, p) in ident.items() if g is not None and p >= N.PURITY}
    links = {v: N.make_link(v, N._cal(q), tl, N.FPS, px) for v in pooled}
    ids = sorted(tl)
    for i in ids:
        for j in ids:
            if i == j or tl[i].end >= tl[j].start or i not in known or j not in known:
                continue
            c0 = links["V0"](tl[i], tl[j], N.FPS, S.MAX_SPEED, S.SLACK_M, S.MAX_GAP_S)
            if c0 is None:
                continue
            labels.append(ident[i][0] == ident[j][0]); exits.append(f"{q}:{i}"); clips.append(q)
            for v in pooled:
                pooled[v].append(links[v](tl[i], tl[j], N.FPS, S.MAX_SPEED, S.SLACK_M, S.MAX_GAP_S))
    print(q, "done", round(time.time() - t0, 1), "s", flush=True)
y = np.array(labels, bool)
sc = {v: np.array(pooled[v], float) for v in pooled}
print(f"pairs {len(y)} same {int(y.sum())}  enumeration {time.time()-t0:.1f}s")
print("point AUC:", {v: round(N._auc(sc[v], y), 4) for v in sc})
stored = json.load(open(N.STC + "/n40_l1.json"))
print("stored  :", stored["AUC"], stored["pairs_pooled"], stored["same_pooled"])

rng = np.random.default_rng(0)
B = 2000
n = len(y)
d_pair = np.empty(B)
for b in range(B):
    idx = rng.integers(0, n, n)
    d_pair[b] = N._auc(sc["V0"][idx], y[idx]) - N._auc(sc["V2"][idx], y[idx])
# cluster by exit tracklet
ex = np.array(exits); uniq = np.unique(ex); groups = {u: np.flatnonzero(ex == u) for u in uniq}
d_cl = np.empty(B); a0 = np.empty(B); a2 = np.empty(B)
for b in range(B):
    pick = rng.choice(uniq, len(uniq), replace=True)
    idx = np.concatenate([groups[u] for u in pick])
    a0[b] = N._auc(sc["V0"][idx], y[idx]); a2[b] = N._auc(sc["V2"][idx], y[idx]); d_cl[b] = a0[b] - a2[b]
pct = lambda a: np.percentile(a, [2.5, 50, 97.5]).round(4).tolist()
print(f"V0-V2 AUC diff: pair bootstrap 95% {pct(d_pair)} sd {d_pair.std():.4f}; cluster(exit tracklet, {len(uniq)} clusters) 95% {pct(d_cl)} sd {d_cl.std():.4f}")
print(f"V0 AUC cluster 95% {pct(a0)}; V2 AUC cluster 95% {pct(a2)}")
print("per-clip AUC V0/V2:", {q: (round(N._auc(sc['V0'][np.array(clips)==q], y[np.array(clips)==q]),4), round(N._auc(sc['V2'][np.array(clips)==q], y[np.array(clips)==q]),4)) for q in N.DEV})
print("V0 vs V2 identical costs share:", float(np.mean(np.isclose(sc['V0'], sc['V2']))))
