"""R3f: the full quality battery on a given checkpoint (CKPT env var).

Methods (all from the same 400 noise seeds):
  normal sampling at 20 / 10 / 8 / 6 / 5 steps           (fewer-steps baseline)
  shared-noise frame sharing (exact, B5b)                 (known to hurt)
  averaged sharing at 20 and 10 steps, thr 0.03 / 0.06    (R3c)
  averaged + re-noised sharing at 20 and 10 steps         (restores the correct noise level)
Metrics vs 400 real clips: FD-rand (5 random-feature seeds), temporal statistics, and a
classifier two-sample test (C2ST accuracy; 0.5 = indistinguishable).
"""
import sys, os, json, time
import numpy as np
import torch
sys.path.insert(0, ".")
import experiments.tiny_exps as E
import experiments.r3c_avg_sharing as R
from experiments.r3_shared_noise_quality import c2st
from experiments.r3b_quality import fd, feats, tstats
from experiments.tiny_data import make_dataset
from experiments.toy_physics import physics

torch.set_grad_enabled(False)
m = R.model; g2 = m.g ** 2
N = int(os.environ.get("NSAMP", 400))
real = make_dataset(N, seed=4321)
tag = os.path.basename(os.environ.get("CKPT", "results/tiny_ckpt.pt")).replace(".pt", "")


def shared_map(x0, thr):
    draft = m.unpatchify(E.sample(m, x0, n_steps=2)[0])[0].numpy()
    stat = E.token_activity(m, draft)[:g2] < thr
    src = np.arange(m.N)
    for tf in range(m.T):
        for p in np.where(stat)[0]:
            src[tf * g2 + p] = p
    src = torch.from_numpy(src)
    return src, torch.from_numpy(np.unique(src.numpy()))


def avg_sample(x0, steps, thr, renoise=False, seed=0):
    src, keep = shared_map(x0, thr)
    inv = E.inv_index(keep, src); mult = torch.bincount(inv, minlength=len(keep)).float(); kb = torch.log(mult)
    x = x0.clone(); g = torch.Generator().manual_seed(seed); flops = m.flops(m.N) * 2
    for k in range(steps):
        t = k / steps
        z = torch.zeros(1, len(keep), x.shape[-1]).index_add_(1, inv, x) / mult[None, :, None]
        if renoise:   # restore per-token noise std (1-t): add (1-t)*sqrt(1-1/m) fresh noise
            z = z + (1 - t) * torch.sqrt(1 - 1 / mult)[None, :, None] * torch.randn(z.shape, generator=g)
        v = R.run_kept(z, torch.full((1,), t), kb, keep); flops += m.flops(len(keep))
        x1 = (z + (1 - t) * v)[:, inv]
        x = x + (x1 - (x - t * x1) / (1 - t)) / steps
    return x, flops


def shared_noise_sample(x0, thr=0.03):
    src, keep = shared_map(x0, thr)
    xf, f, _, _ = E.sample(m, x0, keep=keep, share_src=src, prop_attn=True)
    return xf, f + m.flops(m.N) * 2


METHODS = {f"normal_{s}": (lambda x0, s=s: (E.sample(m, x0, n_steps=s)[0], m.flops(m.N) * s)) for s in [20, 10, 8, 6, 5]}
METHODS["shared_noise_20"] = lambda x0: shared_noise_sample(x0)
for s in [20, 10]:
    for thr in [0.03, 0.06]:
        METHODS[f"avg_{s}_thr{thr}"] = lambda x0, s=s, thr=thr: avg_sample(x0, s, thr)
    METHODS[f"avg_renoise_{s}_thr0.03"] = lambda x0, s=s: avg_sample(x0, s, 0.03, renoise=True)

OUTF = f"results/r3f_battery_{tag}.json"
out = json.load(open(OUTF)) if os.path.exists(OUTF) else {"ckpt": tag, "n": N}
full = m.flops(m.N) * 20
fd_floor = out.get("real_vs_real_FD") or float(np.mean([fd(feats(real, s), feats(make_dataset(N, seed=8765), s)) for s in range(5)]))
out["real_vs_real_FD"] = fd_floor
out.setdefault("real_physics", physics(real))
t0 = time.time()
for name, fn in METHODS.items():
    if name in out:
        continue
    vids, fl = [], []
    for i in range(N):
        x0 = torch.randn(1, m.N, m.P ** 2, generator=torch.Generator().manual_seed(30_000 + i))
        xf, f = fn(x0)
        vids.append(np.clip(m.unpatchify(xf)[0].numpy(), -1, 1)); fl.append(f)
    vids = np.stack(vids).astype(np.float32)
    out[name] = dict(speedup=float(full / np.mean(fl)), FD=float(np.mean([fd(feats(real, s), feats(vids, s)) for s in range(5)])),
                     c2st=float(np.mean([c2st(real, vids, s, epochs=10) for s in range(2)])), **tstats(vids),
                     **{"phys_" + k: v for k, v in physics(vids).items()})
    print(f"{name:24s} speed {out[name]['speedup']:.2f}  FD {out[name]['FD']:.6f}  C2ST {out[name]['c2st']:.3f}  "
          f"static {out[name]['static_frac']:.2f}  jerk {out[name]['moving_jerk']:.3f}  "
          f"phys: flick {out[name]['phys_bg_flicker']:.3f} area {out[name]['phys_area_drift']:.2f} "
          f"vjerk {out[name]['phys_vel_jerk']:.2f} app {out[name]['phys_app_drift']:.2f} valid {out[name]['phys_valid']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(out, open(OUTF, "w"), indent=1)
