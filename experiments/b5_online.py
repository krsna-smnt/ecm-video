"""B5: online temporal token sharing, no oracle.

Run w full steps from the original noise, read each spatial position's activity from the
model's own predicted clean video at that point, then share static positions across all
frames (non-leader tokens are set to their leader's state and removed from the sequence)
with proportional attention for the remaining steps.

Compared against the TRUE full-compute output from the same original noise, so the
switch itself (including resetting the non-leader tokens' noise) counts as error.
"""
import sys, json
import numpy as np
import torch
sys.path.insert(0, ".")
import experiments.tiny_exps as E

torch.set_grad_enabled(False)
model = E.load()
N_STEPS = E.N_STEPS
g2 = model.g ** 2
full_flops = model.flops(model.N) * N_STEPS


def run_online(x0, w, thr, k_grp=8, own_noise=True):
    x = x0.clone(); flops = 0; counts = []
    for k in range(w):                                    # full steps
        v, _ = model(x, torch.full((1,), k / N_STEPS)); flops += model.flops(model.N); counts.append(model.N)
        xhat = x + (1 - k / N_STEPS) * v
        x = x + v / N_STEPS
    vid_hat = model.unpatchify(xhat)[0].numpy()
    act = E.token_activity(model, vid_hat)                # from the model's own prediction
    stat = act[:g2] < thr
    src = np.arange(model.N)
    for tf in range(model.T):
        lead = (tf // k_grp) * k_grp
        for p in np.where(stat)[0]:
            src[tf * g2 + p] = lead * g2 + p
    src = torch.from_numpy(src); keep = torch.from_numpy(np.unique(src.numpy()))
    t_w = w / N_STEPS
    if own_noise:
        # keep every token's own noise; share only the predicted CLEAN content of its leader.
        # rectified flow: x_t = t*x1 + (1-t)*eps, so eps_i = (x_i - t*x1hat_i) / (1 - t)
        eps = (x - t_w * xhat) / (1 - t_w)
    else:
        x = x[:, src]                                      # v1: non-leaders take the leader's state
    mult = torch.bincount(E.inv_index(keep, src), minlength=len(keep)).float()
    kb = torch.log(mult); inv = E.inv_index(keep, src)
    for k in range(w, N_STEPS):
        tk = k / N_STEPS
        v, _ = model(x, torch.full((1,), tk), keep=keep, key_bias=kb)
        flops += model.flops(len(keep)); counts.append(len(keep))
        if own_noise:
            xl = x[:, keep]
            x1_lead = xl + (1 - tk) * v                        # leaders' clean prediction
            x1_all = x1_lead[:, inv]                           # shared to every copy
            t_next = (k + 1) / N_STEPS
            is_lead = torch.zeros(model.N, dtype=torch.bool); is_lead[keep] = True
            moved = x + v[:, inv] / N_STEPS                    # leaders (and unshared tokens) step normally
            synced = t_next * x1_all + (1 - t_next) * eps      # copies: shared content, own noise
            x = torch.where(is_lead[None, :, None], moved, synced)
        else:
            x = x + v[:, inv] / N_STEPS
    seq_fracs = [c / model.N for c in counts]
    # real-scale projection: per step, sequence fraction c/N and all of it queried
    def fl(n):
        d, depth = 3072, 40
        return depth * (2 * n * d * d * 3 + 4 * n * n * d + 2 * n * d * d + 16 * n * d * d)
    proj = fl(75_600) * N_STEPS / sum(fl(f * 75_600) for f in seq_fracs)
    return x, flops, float(stat.mean()), proj


res = {}
clips = []
for i in range(24):
    x0 = torch.randn(1, model.N, model.P ** 2, generator=torch.Generator().manual_seed(10_000 + i))
    ref = model.unpatchify(E.sample(model, x0)[0])[0].numpy()
    clips.append((x0, ref))
for w in [4, 6, 8]:
    for thr in [0.005, 0.01, 0.02]:
        ps, sp, fr, pj, pv = [], [], [], [], []
        for x0, ref in clips:
            xf, f, frac, proj = run_online(x0, w, thr)
            v = model.unpatchify(xf)[0].numpy()
            ps.append(E.psnr(v, ref)); sp.append(full_flops / f); fr.append(frac); pj.append(proj); pv.append(E.pvs_of(v))
        r = dict(warm_steps=w, thr=thr, psnr=float(np.mean(ps)), psnr_min=float(np.min(ps)),
                 speedup=float(np.mean(sp)), projected_speedup_real_scale=float(np.mean(pj)),
                 static_frac=float(np.mean(fr)), pvs=float(np.mean(pv)))
        res[f"w{w}_thr{thr}"] = r
        print(r, flush=True)
res["ref_pvs"] = float(np.mean([E.pvs_of(r) for _, r in clips]))
json.dump(res, open("results/b5_online.json", "w"), indent=1)
print("ref pvs", res["ref_pvs"])
