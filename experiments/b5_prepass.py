"""B5b: decide the static map BEFORE generating, from a cheap draft pass.

1. Draft: a 3-step full generation from the clip's noise (15% of a 20-step run).
2. Static map from the draft's activity.
3. Generate from scratch with those positions shared across all frames from step 0
   (shared noise + proportional attention), 20 steps.
Cost counted = draft + shared generation.

Two references:
  * same-noise full compute (does sharing reproduce what full compute gives from the
    same starting noise? isolates the compute approximation), and
  * the ORIGINAL full-compute video (how far the whole procedure moves the output).
"""
import sys, json
import numpy as np
import torch
sys.path.insert(0, ".")
import experiments.tiny_exps as E

torch.set_grad_enabled(False)
model = E.load()
g2 = model.g ** 2
full_flops = model.flops(model.N) * E.N_STEPS
res = {}
for draft_steps in [2, 3, 5]:
    for thr in [0.02, 0.03, 0.06]:
        ps_same, ps_orig, sp, fr, pj, pv, pvr = [], [], [], [], [], [], []
        for i in range(24):
            x0 = torch.randn(1, model.N, model.P ** 2, generator=torch.Generator().manual_seed(10_000 + i))
            orig = model.unpatchify(E.sample(model, x0)[0])[0].numpy()
            draft = model.unpatchify(E.sample(model, x0, n_steps=draft_steps)[0])[0].numpy()
            stat = E.token_activity(model, draft)[:g2] < thr
            src = np.arange(model.N)
            for tf in range(model.T):
                for p in np.where(stat)[0]:
                    src[tf * g2 + p] = p
            src = torch.from_numpy(src); keep = torch.from_numpy(np.unique(src.numpy()))
            x0s = x0[:, src]
            same = model.unpatchify(E.sample(model, x0s)[0])[0].numpy()
            xf, f, _, counts = E.sample(model, x0, keep=keep, share_src=src, prop_attn=True)
            v = model.unpatchify(xf)[0].numpy()
            cost = f + model.flops(model.N) * draft_steps
            ps_same.append(E.psnr(v, same)); ps_orig.append(E.psnr(v, orig))
            sp.append(full_flops / cost); fr.append(float(stat.mean()))
            seq = len(keep) / model.N
            # real-scale projection including the draft pass
            def fl(n, d=3072, depth=40):
                return depth * (2 * n * d * d * 3 + 4 * n * n * d + 2 * n * d * d + 16 * n * d * d)
            Nr = 75_600
            pj.append(fl(Nr) * E.N_STEPS / (fl(seq * Nr) * E.N_STEPS + fl(Nr) * draft_steps))
            pv.append(E.pvs_of(v)); pvr.append(E.pvs_of(orig))
        r = dict(draft_steps=draft_steps, thr=thr, static_frac=float(np.mean(fr)), speedup=float(np.mean(sp)),
                 projected_speedup_real_scale=float(np.mean(pj)),
                 psnr_vs_same_noise_full=float(np.mean(ps_same)), psnr_min_same=float(np.min(ps_same)),
                 psnr_vs_original=float(np.mean(ps_orig)), pvs=float(np.mean(pv)), pvs_original=float(np.mean(pvr)))
        res[f"d{draft_steps}_thr{thr}"] = r
        print(r, flush=True)
json.dump(res, open("results/b5_prepass.json", "w"), indent=1)
