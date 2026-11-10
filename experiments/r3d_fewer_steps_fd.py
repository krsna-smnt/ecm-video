"""R3d: the fair comparison. Quality vs REAL data (FD-rand, temporal stats) at matched speedup:
fewer sampling steps vs averaged frame sharing (R3c). Same 400 noise seeds, same real set."""
import sys, json
import numpy as np
import torch
sys.path.insert(0, ".")
import experiments.tiny_exps as E
from experiments.r3_shared_noise_quality import model
from experiments.r3b_quality import fd, feats, tstats
from experiments.r3c_avg_sharing import avg_share_sample2
from experiments.tiny_data import make_dataset

torch.set_grad_enabled(False)
N = 400
real = make_dataset(N, seed=4321)
out = {}
for steps in [20, 10, 8, 6, 5]:
    vids = np.stack([np.clip(model.unpatchify(E.sample(model, torch.randn(1, model.N, model.P ** 2,
                     generator=torch.Generator().manual_seed(30_000 + i)), n_steps=steps)[0])[0].numpy(), -1, 1)
                     for i in range(N)]).astype(np.float32)
    out[f"steps_{steps}"] = dict(speedup=20 / steps, FD=float(np.mean([fd(feats(real, s), feats(vids, s)) for s in range(5)])), **tstats(vids))
    print(f"steps_{steps}", out[f"steps_{steps}"], flush=True)
# averaged sharing on top of fewer steps (stacking)
for steps, thr in [(10, 0.03), (10, 0.06)]:
    E.N_STEPS = steps
    import experiments.r3c_avg_sharing as R; R.NS = steps
    vids, sp = [], []
    for i in range(N):
        x0 = torch.randn(1, model.N, model.P ** 2, generator=torch.Generator().manual_seed(30_000 + i))
        xf, f, _ = avg_share_sample2(x0, thr=thr)
        vids.append(np.clip(model.unpatchify(xf)[0].numpy(), -1, 1)); sp.append(model.flops(model.N) * 20 / f)
    vids = np.stack(vids).astype(np.float32)
    out[f"steps_{steps}+avgshare_thr{thr}"] = dict(speedup=float(np.mean(sp)), FD=float(np.mean([fd(feats(real, s), feats(vids, s)) for s in range(5)])), **tstats(vids))
    print(f"steps_{steps}+avgshare_thr{thr}", out[f"steps_{steps}+avgshare_thr{thr}"], flush=True)
json.dump(out, open("results/r3d_fewer_steps_fd.json", "w"), indent=1)
