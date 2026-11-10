"""R3c: frame sharing WITHOUT shared noise ("averaged sharing").

Diagnosis from R3b: sharing noise across frames removes the temporal redundancy the model
uses to denoise static backgrounds (8 independent noisy views of the same content), so
noise is baked in. Fix:
  * every token keeps its own independent noise (inputs stay in-distribution);
  * a shared position is computed ONCE, from the MEAN of its copies' current states
    (averaging independent views is exactly the denoising the model would do);
  * the leader's clean prediction x1hat is handed to every copy, and each copy moves along
    its own trajectory: eps_i = (x_i - t x1hat) / (1 - t), x_i <- x_i + (x1hat - eps_i)/N.
    For unshared tokens this is exactly the ordinary Euler step.
Static map from a 2-step draft from the same noise, as in B5b. Compute is the same as B5b.

Compared with normal sampling (A) and shared-noise sharing (C, from R3b) on FD-rand and
temporal statistics vs real, plus fidelity (PSNR) to normal full compute from the same noise.
"""
import sys, json
import numpy as np
import torch
sys.path.insert(0, ".")
import experiments.tiny_exps as E
from experiments.r3_shared_noise_quality import generate, model
from experiments.r3b_quality import fd, feats, tstats   # noqa (module runs its own eval on import otherwise)
from experiments.tiny_data import make_dataset

torch.set_grad_enabled(False)
g2 = model.g ** 2
NS = E.N_STEPS


def run_kept(z, t, kb, keep):
    """Forward on a kept-only sequence: positions are the kept tokens' own positions."""
    temb = model.time_embed(t)
    h = model.embed(z) + model.pos()[keep]
    for blk in model.blocks:
        h, _ = blk(h, temb, None, None, kb)
    return model.out(model.nf(h))


def avg_share_sample2(x0, thr=0.03, draft_steps=2, average=True):
    draft = model.unpatchify(E.sample(model, x0, n_steps=draft_steps)[0])[0].numpy()
    stat = E.token_activity(model, draft)[:g2] < thr
    src = np.arange(model.N)
    for tf in range(model.T):
        for p in np.where(stat)[0]:
            src[tf * g2 + p] = p
    src = torch.from_numpy(src); keep = torch.from_numpy(np.unique(src.numpy()))
    inv = E.inv_index(keep, src)
    mult = torch.bincount(inv, minlength=len(keep)).float(); kb = torch.log(mult)
    x = x0.clone(); flops = model.flops(model.N) * draft_steps
    for k in range(NS):
        t = k / NS
        z = (torch.zeros(1, len(keep), x.shape[-1]).index_add_(1, inv, x) / mult[None, :, None]) if average else x[:, keep]
        v = run_kept(z, torch.full((1,), t), kb, keep); flops += model.flops(len(keep))
        x1 = (z + (1 - t) * v)[:, inv]
        eps = (x - t * x1) / (1 - t)
        x = x + (x1 - eps) / NS
    return x, flops, float(stat.mean())


if __name__ == "__main__":
    N = 400
    real = make_dataset(N, seed=4321)
    full = model.flops(model.N) * NS
    out = {}
    for name, kw in [("D_avg_sharing", dict(average=True)), ("D0_no_avg_own_noise", dict(average=False)),
                     ("D_avg_sharing_thr0.06", dict(average=True, thr=0.06))]:
        vids, sp, fr, ps = [], [], [], []
        for i in range(N):
            x0 = torch.randn(1, model.N, model.P ** 2, generator=torch.Generator().manual_seed(30_000 + i))
            xf, f, frac = avg_share_sample2(x0, **kw)
            v = np.clip(model.unpatchify(xf)[0].numpy(), -1, 1)
            vids.append(v); sp.append(full / f); fr.append(frac)
            if i < 48:
                ref = model.unpatchify(E.sample(model, x0)[0])[0].numpy()
                ps.append(E.psnr(v, ref))
        vids = np.stack(vids).astype(np.float32)
        out[name] = dict(FD=float(np.mean([fd(feats(real, s), feats(vids, s)) for s in range(5)])), **tstats(vids),
                         speedup=float(np.mean(sp)), shared_frac=float(np.mean(fr)), psnr_vs_normal_full=float(np.mean(ps)))
        print(name, out[name], flush=True)
    prev = json.load(open("results/r3b_quality.json"))
    out["reference_from_R3b"] = {k: prev[k] for k in ["real_vs_real", "A_normal", "C_shared_compute"]}
    json.dump(out, open("results/r3c_avg_sharing.json", "w"), indent=1)
