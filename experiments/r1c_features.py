"""R1c: which information-theoretic video features track human glitch judgments?

Per clip (64 frames, stride 2, 256 px):
  iota_mean     mean information density (bits/px): amount of detail
  iota_trend    relative slope of mean density over the clip (information decay = melting/blur?)
  src_signed    mean SIGNED scalar source along the flow (negative = information destroyed)
  src_neg_frac  share of open-view source mass that is destruction (s < 0)
  sharp         mean Laplacian variance (edge sharpness)
  jitter        temporal second difference / first difference (high-frequency flicker)
  aot           arrow-of-time asymmetry: |forward - backward conditional surprise| / sum
  plus from R1: pvs (mean), warp error, motion

Evaluation, with the corrected sign (votes mark the MORE glitchy clip):
  * pooled pairwise accuracy on every voted pair whose two clips we have, with a direction
    fitted on the 93 strong pairs and then frozen (so "inverted" features count properly);
  * within-generator Spearman between the feature and each clip's human coherence share,
    which removes the generator-identity confound. Also reported: a sign-consistency count
    across generators.
"""
import sys, os, json
import numpy as np
import pandas as pd
import cv2
from scipy.stats import spearmanr
sys.path.insert(0, ".")
from ecm.pvs import load_video, conditional_surprise
from ecm.density import info_density
from ecm.flow import farneback, warp, motion_boundaries, out_of_frame

CLIPS = "/tmp/r1clips"
CACHE = "results/r1c_features.json"
F = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
for i, n in enumerate(sorted(os.listdir(CLIPS))):
    if n in F:
        continue
    f = load_video(os.path.join(CLIPS, n), 256, 64, 2).astype(np.float32)
    f = np.stack([cv2.GaussianBlur(x, (0, 0), 2.0) for x in f])
    io = np.stack([cv2.GaussianBlur(info_density(x), (0, 0), 1.5) for x in f])
    m = io.mean((1, 2)); tt = np.arange(len(m))
    trend = float(np.polyfit(tt, m, 1)[0] * len(m) / (m.mean() + 1e-6))
    signed, negfrac, fw, bw = [], [], [], []
    for t in range(len(f) - 1):
        fwd = farneback(f[t], f[t + 1]); bwd = farneback(f[t + 1], f[t])
        s = warp(io[t + 1], fwd) - io[t]
        legal = motion_boundaries(fwd, 1.2, 3) | out_of_frame(fwd, 4)
        so = s[~legal]
        signed.append(float(so.mean()) if so.size else 0.0)
        mag = np.abs(so).sum() + 1e-6
        negfrac.append(float(-so[so < 0].sum() / mag))
        fw.append(conditional_surprise(f[t], f[t + 1], fwd).mean())
        bw.append(conditional_surprise(f[t + 1], f[t], bwd).mean())
    d1 = np.abs(np.diff(f, axis=0)).mean(); d2 = np.abs(np.diff(f, n=2, axis=0)).mean()
    lap = np.mean([cv2.Laplacian(x, cv2.CV_32F).var() for x in f[::4]])
    fw, bw = np.array(fw), np.array(bw)
    F[n] = dict(iota_mean=float(io.mean()), iota_trend=trend, src_signed=float(np.mean(signed)),
                src_neg_frac=float(np.mean(negfrac)), sharp=float(lap), jitter=float(d2 / (d1 + 1e-6)),
                aot=float(np.abs(fw - bw).sum() / (fw.sum() + bw.sum() + 1e-6)))
    if i % 10 == 0:
        json.dump(F, open(CACHE, "w")); print(i, flush=True)
json.dump(F, open(CACHE, "w"))

R1 = json.load(open("results/r1_features.json"))
for n in F:
    F[n].update(pvs=R1[n]["base"]["mean"], warp=R1[n]["base"]["werr"], motion=R1[n]["motion"])
feats = list(next(iter(F.values())).keys())

allp = pd.read_parquet("results/rapidata_pairs.parquet")
key = lambda s, fn: f"{s}__{fn}"
allp["a"] = [key(s, x) for s, x in zip(allp.src, allp.file_name1)]
allp["b"] = [key(s, x) for s, x in zip(allp.src, allp.file_name2)]
allp = allp[allp.a.isin(F) & allp.b.isin(F)].dropna(subset=["margin"])
strong = allp[allp.margin.abs() >= 0.4]


def acc(df, k, sign):
    """sign=+1: lower feature = less glitchy; -1: higher = less glitchy. m<0 means a less glitchy."""
    h = [float(((F[a][k] - F[b][k]) * sign < 0) == (m < 0)) for a, b, m in zip(df.a, df.b, df.margin) if F[a][k] != F[b][k]]
    return float(np.mean(h)), len(h)


def gen(n):
    s = n.split("__")[1]
    return s.split("_")[1] if s[0].isdigit() else s.split("_")[0]


share = {}
for a, b, m in zip(allp.a, allp.b, allp.margin):
    share.setdefault(a, []).append(0.5 - m / 2); share.setdefault(b, []).append(0.5 + m / 2)
gens = {}
for n in share:
    gens.setdefault(gen(n), []).append(n)

out = {"n_pairs_all": len(allp), "n_strong": len(strong)}
for k in feats:
    s_fit = 1 if acc(strong, k, 1)[0] >= 0.5 else -1
    within = []
    for g, cl in gens.items():
        if len(cl) >= 6:
            rho = spearmanr([F[n][k] for n in cl], [np.mean(share[n]) for n in cl])[0]
            within.append((g, len(cl), float(rho)))
    rhos = np.array([w[2] for w in within])
    out[k] = dict(direction=("lower=less glitchy" if s_fit == 1 else "higher=less glitchy"),
                  acc_strong=acc(strong, k, s_fit)[0], acc_all=acc(allp, k, s_fit)[0],
                  within_gen_mean_rho=float(np.nanmean(rhos)),
                  within_gen_n_positive=int((rhos > 0).sum()), within_gen_n=len(rhos),
                  within=within)
    print(f"{k:13s} {out[k]['direction']:22s} strong {out[k]['acc_strong']:.2f}  all {out[k]['acc_all']:.2f}  "
          f"within-gen mean rho {out[k]['within_gen_mean_rho']:+.2f} ({out[k]['within_gen_n_positive']}/{len(rhos)} positive)")
json.dump(out, open("results/r1c_features_eval.json", "w"), indent=1)
