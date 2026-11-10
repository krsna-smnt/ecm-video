"""R1b: robustness checks for the R1 result (PVS-mean predicts human coherence votes).

1. Held-out pairs: all OTHER human-voted pairs whose two clips are both in our 103-clip set
   (margin < 0.4, never used for selection); accuracy by margin bucket.
2. Ablations of the score on the 103 clips: scalar density mode, flow-boundaries from the
   judged pair (no neighbour trick), no legal-zone masking at all, and a crude baseline:
   the frame's residual noise level after motion compensation (median |r|).
3. Generator confound: (a) accuracy on pairs where generator-level human win rate would
   predict the WRONG answer; (b) within-generator ranking: for generators with several clips,
   correlate PVS with each clip's average coherence share across its pairs.
"""
import sys, os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, ".")
from ecm.pvs import score_video, load_video
from ecm.flow import farneback, warp

CLIPS = "/tmp/r1clips"
feats = json.load(open("results/r1_features.json"))
CACHE = "results/r1b_ablation_features.json"
abl = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
VAR = {"scalar": dict(prefilter=2.0, mode="scalar"),
       "no_neighbour": dict(prefilter=2.0, neighbour_boundaries=False),
       "no_mask": dict(prefilter=2.0, mb_thresh=1e9, edge_margin=0)}
for i, n in enumerate(sorted(os.listdir(CLIPS))):
    if n in abl and all(k in abl[n] for k in list(VAR) + ["noise"]):
        continue
    f = load_video(os.path.join(CLIPS, n), 256, 64, 2)
    d = abl.get(n, {})
    for k, kw in VAR.items():
        if k not in d:
            d[k] = float(score_video(f, **kw).frame_scores.mean())
    if "noise" not in d:
        import cv2
        fb = np.stack([cv2.GaussianBlur(x.astype(np.float32), (0, 0), 2.0) for x in f])
        rs = [np.median(np.abs(warp(fb[t + 1], farneback(fb[t], fb[t + 1])) - fb[t])) for t in range(len(fb) - 1)]
        d["noise"] = float(np.mean(rs))
    abl[n] = d
    if i % 10 == 0:
        json.dump(abl, open(CACHE, "w")); print("ablation", i, flush=True)
json.dump(abl, open(CACHE, "w"))

allp = pd.read_parquet("results/rapidata_pairs.parquet")
have = set(feats)
key = lambda s, f: f"{s}__{f}"
allp["a"] = [key(s, f) for s, f in zip(allp.src, allp.file_name1)]
allp["b"] = [key(s, f) for s, f in zip(allp.src, allp.file_name2)]
allp = allp[allp.a.isin(have) & allp.b.isin(have)].dropna(subset=["margin"])
strong = allp[allp.margin.abs() >= 0.4]; weak = allp[allp.margin.abs() < 0.4]
pvs = lambda n: feats[n]["base"]["mean"]


def acc(df, fn):
    h = [0.5 if fn(a) == fn(b) else float((fn(a) < fn(b)) == (m < 0)) for a, b, m in zip(df.a, df.b, df.margin)]  # m<0: a judged less glitchy
    return (float(np.mean(h)), len(h)) if h else (None, 0)


out = {"strong": acc(strong, pvs), "weak_heldout": acc(weak, pvs)}
for lo, hi in [(0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)]:
    out[f"margin_{lo}-{hi}"] = acc(allp[(allp.margin.abs() >= lo) & (allp.margin.abs() < hi)], pvs)
for k in list(VAR) + ["noise"]:
    out[f"ablation_{k}_strong"] = acc(strong, lambda n, k=k: abl[n][k])
    out[f"ablation_{k}_all"] = acc(allp, lambda n, k=k: abl[n][k])
out["pvs_all"] = acc(allp, pvs)

# generator confound
gen = lambda n: n.split("__")[1].split("_")[0] if not n.split("__")[1][0].isdigit() else n.split("__")[1].split("_")[1]
wins = {}
for a, b, m in zip(allp.a, allp.b, allp.margin):
    wins.setdefault(gen(a), []).append(0.5 - m / 2); wins.setdefault(gen(b), []).append(0.5 + m / 2)  # share judged LESS glitchy
grate = {g: float(np.mean(v)) for g, v in wins.items()}
contra = strong[[(grate[gen(a)] - grate[gen(b)]) * m < 0 for a, b, m in zip(strong.a, strong.b, strong.margin)]]
out["generator_rates"] = grate
out["strong_pairs_where_generator_prior_is_wrong"] = acc(contra, pvs)
out["generator_prior_accuracy_strong"] = (float(np.mean([float((grate[gen(a)] > grate[gen(b)]) == (m > 0)) for a, b, m in zip(strong.a, strong.b, strong.margin)])), len(strong))
gpvs = {}
for n in have:
    gpvs.setdefault(gen(n), []).append(pvs(n))
out["spearman_generator_pvs_vs_human_rate"] = float(spearmanr([np.mean(gpvs[g]) for g in grate], [grate[g] for g in grate])[0])
# within-generator: clip coherence share vs clip PVS
share = {}
for a, b, m in zip(allp.a, allp.b, allp.margin):
    share.setdefault(a, []).append(0.5 - m / 2); share.setdefault(b, []).append(0.5 + m / 2)  # coherence share (less glitchy)
within = []
for g in gpvs:
    cl = [n for n in share if gen(n) == g]
    if len(cl) >= 5:
        within.append((g, len(cl), float(spearmanr([pvs(n) for n in cl], [np.mean(share[n]) for n in cl])[0])))
out["within_generator_spearman (negative = agrees)"] = within
print(json.dumps(out, indent=1, default=float))
json.dump(out, open("results/r1b_checks.json", "w"), indent=1, default=float)
