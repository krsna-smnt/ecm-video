"""R5b: does frame sharing adapt to camera motion? (pan-fine-tuned model)

For 200 generated clips: estimate the clip's global camera motion (phase correlation between
consecutive frames, px/frame) and the shared fraction the 2-step draft assigns at thr 0.02.
Also, against normal 20-step output from the same noise, the PSNR of re-noised sharing on
low- vs high-motion clips (does sharing hurt more where the camera moves?).
"""
import sys, json
import numpy as np
import torch
import cv2
from scipy.stats import spearmanr
sys.path.insert(0, ".")
import experiments.tiny_exps as E
import experiments.r3f_battery as B   # noqa: runs nothing heavy when ONLY selects nothing
m = B.m; g2 = m.g ** 2
rows = []
for i in range(200):
    x0 = torch.randn(1, m.N, m.P ** 2, generator=torch.Generator().manual_seed(50_000 + i))
    ref = np.clip(m.unpatchify(E.sample(m, x0)[0])[0].numpy(), -1, 1)
    shifts = [np.hypot(*cv2.phaseCorrelate(ref[t].astype(np.float64), ref[t + 1].astype(np.float64))[0]) for t in range(m.T - 1)]
    src, keep = B.shared_map(x0, 0.02)
    shared = 1 - len(keep) / m.N
    B.CUR["seed"] = 900_000 + i
    xf, _ = B.avg_sample(x0, 20, 0.02, renoise=True)
    v = np.clip(m.unpatchify(xf)[0].numpy(), -1, 1)
    rows.append((float(np.median(shifts)), shared, E.psnr(v, ref)))
r = np.array(rows)
hi = r[:, 0] > np.median(r[:, 0])
out = dict(spearman_motion_vs_shared=float(spearmanr(r[:, 0], r[:, 1])[0]),
           shared_low_motion=float(r[~hi, 1].mean()), shared_high_motion=float(r[hi, 1].mean()),
           psnr_low_motion=float(r[~hi, 2].mean()), psnr_high_motion=float(r[hi, 2].mean()),
           motion_median=float(np.median(r[:, 0])))
print(out)
json.dump(dict(summary=out, rows=rows), open("results/r5_adaptivity.json", "w"), indent=1)
