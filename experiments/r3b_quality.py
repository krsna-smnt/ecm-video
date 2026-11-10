"""R3b: graded quality comparison of normal vs shared-noise samples (tiny model).

Metrics (lower = closer to real):
  * FD-rand: Frechet distance between real and generated feature distributions, features
    from 5 fixed randomly initialised 3D-conv networks (a toy stand-in for FVD; random
    conv features are a known reasonable proxy at small scale). real-vs-real gives the floor.
  * Per-region temporal statistics vs real: fraction of static pixels; mean change of
    moving pixels; temporal 2nd difference (jerkiness) of moving pixels.
Also renders sample grids for visual inspection.
"""
import sys, json
import numpy as np
import torch
import torch.nn as nn
import cv2
from scipy import linalg
sys.path.insert(0, ".")
from experiments.r3_shared_noise_quality import generate, model   # reuses the same generator code
from experiments.tiny_data import make_dataset

torch.set_grad_enabled(False)
def feats(x, seed):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Conv3d(1, 32, 3, padding=1), nn.ReLU(), nn.AvgPool3d(2),
                        nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(), nn.AvgPool3d(2),
                        nn.Conv3d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d(1), nn.Flatten())
    return net(torch.from_numpy(x)[:, None]).numpy()


def fd(a, b):
    mu1, mu2 = a.mean(0), b.mean(0); s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cs = linalg.sqrtm(s1 @ s2).real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs))


def tstats(v):
    d = np.abs(np.diff(v, axis=1)); d2 = np.abs(np.diff(v, n=2, axis=1))
    mov = d.max(1) >= 0.03
    return dict(static_frac=float(1 - mov.mean()),
                moving_change=float(np.mean([dd[:, m].mean() for dd, m in zip(d, mov) if m.any()])),
                moving_jerk=float(np.mean([dd[:, m].mean() for dd, m in zip(d2, mov) if m.any()])))


if __name__ == "__main__":
    N = 400
    real = make_dataset(N, seed=4321); real2 = make_dataset(N, seed=8765)
    sets = {"A_normal": generate(N, "A", 30_000), "B_shared_noise": generate(N, "B", 30_000),
            "C_shared_compute": generate(N, "C", 30_000)}


    res = {"real_vs_real": {"FD": float(np.mean([fd(feats(real, s), feats(real2, s)) for s in range(5)])), **tstats(real2)},
           "real": tstats(real)}
    for k, v in sets.items():
        res[k] = {"FD": float(np.mean([fd(feats(real, s), feats(v, s)) for s in range(5)])), **tstats(v)}
        print(k, res[k], flush=True)
    print("real_vs_real", res["real_vs_real"])
    json.dump(res, open("results/r3b_quality.json", "w"), indent=1)

    rows = [np.concatenate(list(real[i]), 1) for i in range(2)]
    for k in ["A_normal", "B_shared_noise"]:
        rows += [np.concatenate(list(sets[k][i]), 1) for i in range(3)]
    g = np.concatenate(rows, 0)
    cv2.imwrite("results/fig_r3_samples.png", cv2.resize(((g + 1) * 127.5).clip(0, 255).astype(np.uint8), None, fx=3, fy=3,
                                                           interpolation=cv2.INTER_NEAREST))

