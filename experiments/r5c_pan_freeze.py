"""R5c: does frame sharing freeze camera pans? For 150 clips from the pan model: global camera
motion (phase correlation, px/frame) of normal 20-step output vs re-noised sharing (thr 0.02)
from the same noise. Also the real mixed set's static share for reference."""
import sys, json
import numpy as np, torch, cv2
sys.path.insert(0, ".")
import experiments.tiny_exps as E
import experiments.r3f_battery as B
from experiments.tiny_data import make_dataset
from experiments.r3b_quality import tstats
m = B.m
gm = lambda v: float(np.median([np.hypot(*cv2.phaseCorrelate(v[t].astype(np.float64), v[t + 1].astype(np.float64))[0]) for t in range(len(v) - 1)]))
rows = []
for i in range(150):
    x0 = torch.randn(1, m.N, m.P ** 2, generator=torch.Generator().manual_seed(60_000 + i))
    ref = np.clip(m.unpatchify(E.sample(m, x0)[0])[0].numpy(), -1, 1)
    B.CUR["seed"] = 910_000 + i
    sh = np.clip(m.unpatchify(B.avg_sample(x0, 20, 0.02, renoise=True)[0])[0].numpy(), -1, 1)
    rows.append((gm(ref), gm(sh)))
r = np.array(rows); hi = r[:, 0] > 0.5
real = make_dataset(300, seed=4321, pan_prob=0.5)
out = dict(motion_normal_mean=float(r[:, 0].mean()), motion_shared_mean=float(r[:, 1].mean()),
           n_clear_pans=int(hi.sum()), motion_normal_on_pans=float(r[hi, 0].mean()) if hi.any() else None,
           motion_shared_on_pans=float(r[hi, 1].mean()) if hi.any() else None,
           real_mixed_static_frac=tstats(real)["static_frac"],
           real_pan_motion=float(np.mean([gm(v) for v in make_dataset(60, seed=99, pan_prob=1.0)])))
print(out); json.dump(dict(summary=out, rows=rows), open("results/r5c_pan_freeze.json", "w"), indent=1)
