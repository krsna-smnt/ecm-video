"""Speed and realism mechanism tests on the tiny video model.

  B1  settling: when does each token reach its final value; does it track density / motion?
  B2  oracle spatial pruning: recompute only the top-p tokens by information density
      (vs random tokens at the same budget); the rest reuse cached K/V and velocity.
  B3  oracle temporal sharing: static-background tokens computed once per k-frame group
      and copied to the other frames (removed from the sequence).
  B4  online freezing: stop recomputing a token once it settles, by two rules:
      'delta' (change of predicted clean token) and 'heat' (that change x temperature).
  H   heat conduction: how much attention flows between static and moving regions,
      and what happens if static queries are cut off from moving keys.
  C1  early abort: does the melting score of the step-k prediction predict the final one?
  A5t generated vs real: does the melting score separate tiny-model samples from data?

Quality = PSNR (dB) of the output against the full-compute output from the same noise,
plus the melting score (PVS). Cost = analytic FLOPs of every forward call.
"""
import sys, json, time
import numpy as np
import torch
import cv2
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, ".")
from ecm.tinydit import TinyVideoDiT
from ecm.density import info_density
from ecm.pvs import score_video
from experiments.tiny_data import make_dataset

torch.set_grad_enabled(False)
torch.manual_seed(0)
N_STEPS = 20


def load():
    m = TinyVideoDiT()
    import os
    s = torch.load(os.environ.get("CKPT", "results/tiny_ckpt.pt"), weights_only=False)  # our own file
    m.load_state_dict(s["ema"]); m.eval()
    return m


def to_u8(vid):           # vid: T x S x S in [-1, 1] -> 0-255
    return np.clip((vid + 1) * 127.5, 0, 255).astype(np.float32)


def pvs_of(vid, up=4):
    f = to_u8(vid)
    f = np.stack([cv2.resize(x, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) for x in f])
    return score_video(np.clip(f, 0, 255)).pvs_peak


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(4.0 / mse)     # peak-to-peak range is 2


# ---------------------------------------------------------------- samplers
def sample(model, x0, n_steps=N_STEPS, policy=None, keep=None, share_src=None, attn_mask_fn=None):
    """Euler rectified-flow sampler for ONE clip (batch 1).

    policy(k, xhat_prev, xhat_now, temp) -> LongTensor of token positions to recompute
        at step k (within the kept sequence), or None for all. Step 0 is always full.
    keep: kept token positions (temporal sharing); share_src: for every full-sequence token,
        the position of the token it copies (itself if kept).
    Returns final tokens, FLOPs, list of predicted clean tokens per step, #recomputed per step.
    """
    x = x0.clone()
    if share_src is not None:
        x = x[:, share_src]                       # shared tokens start from their leader's noise
    n = model.N if keep is None else len(keep)
    vel = torch.zeros(1, n, x.shape[-1]); caches = None
    flops = 0; hist = []; counts = []
    prev_xhat = None
    for k in range(n_steps):
        tval = k / n_steps
        t = torch.full((1,), tval)
        xs = x if keep is None else x[:, keep]
        q = None if (policy is None or caches is None) else policy(k, prev_xhat, xs + (1 - tval) * vel, 1 - tval)
        if q is not None and len(q) == n:
            q = None
        if q is not None and len(q) == 0:
            v_new = None
        else:
            v_new, caches = model(x, t, q_idx=q, caches=caches, keep=keep) if attn_mask_fn is None \
                else masked_forward(model, x, t, attn_mask_fn)
        if v_new is not None:
            if q is None:
                vel = v_new
            else:
                vel = vel.clone(); vel[:, q] = v_new
        flops += 0 if v_new is None else model.flops(n, None if q is None else len(q))
        counts.append(n if q is None else len(q))
        full_vel = vel if keep is None else vel[:, inv_index(keep, share_src)]
        xhat = x + (1 - tval) * full_vel
        hist.append(xhat[0].clone())
        prev_xhat = (xhat if keep is None else xhat[:, keep])
        x = x + full_vel / n_steps
    return x, flops, hist, counts


def inv_index(keep, share_src):
    """For every full-sequence token, its index inside the kept sequence (via its leader)."""
    pos = {int(p): i for i, p in enumerate(keep.tolist())}
    return torch.tensor([pos[int(s)] for s in share_src.tolist()])


def masked_forward(model, x, t, mask_fn):
    """Full forward with an additive attention mask (for the heat-conduction cut)."""
    import torch.nn.functional as F
    temb = model.time_embed(t)
    h = model.embed(x) + model.pos()
    mask = mask_fn()
    for blk in model.blocks:
        B, n, d = h.shape
        s1, b1, s2, b2 = blk.ada(temb).unsqueeze(1).chunk(4, -1)
        hdn = blk.n1(h) * (1 + s1) + b1
        q, k, v = blk.qkv(hdn).chunk(3, -1)
        sp = lambda z: z.view(B, n, blk.h, d // blk.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(sp(q), sp(k), sp(v), attn_mask=mask)
        h = h + blk.o(a.transpose(1, 2).reshape(B, n, d))
        h = h + blk.mlp(blk.n2(h) * (1 + s2) + b2)
    return model.out(model.nf(h)), None


def attention_maps(model, x, t):
    """Per-layer attention probabilities (averaged over heads) for a full forward."""
    temb = model.time_embed(t)
    h = model.embed(x) + model.pos()
    maps = []
    for blk in model.blocks:
        B, n, d = h.shape
        s1, b1, s2, b2 = blk.ada(temb).unsqueeze(1).chunk(4, -1)
        hdn = blk.n1(h) * (1 + s1) + b1
        q, k, v = blk.qkv(hdn).chunk(3, -1)
        sp = lambda z: z.view(B, n, blk.h, d // blk.h).transpose(1, 2)
        att = torch.softmax(sp(q) @ sp(k).transpose(-1, -2) / np.sqrt(d // blk.h), -1)
        maps.append(att.mean(1)[0])
        a = (att @ sp(v)).transpose(1, 2).reshape(B, n, d)
        h = h + blk.o(a)
        h = h + blk.mlp(blk.n2(h) * (1 + s2) + b2)
    return maps


def projected_speedup(seq_frac, q_fracs, N=75_600, d=3072, depth=40):
    """Speedup the same token fractions would give at real video-DiT scale.

    Default shape is roughly a 5 s, 720p clip in a ~13B-parameter-class video DiT
    (about 75k latent tokens, width 3072, 40 layers). This is a FLOP projection from the
    toy run's token fractions, not a measurement. Same per-layer formula as TinyVideoDiT.flops.
    """
    def fl(n_seq, n_q):
        return depth * (2 * n_q * d * d * 3 + 4 * n_q * n_seq * d + 2 * n_q * d * d + 16 * n_q * d * d)
    full = fl(N, N) * len(q_fracs)
    n_seq = seq_frac * N
    return full / sum(fl(n_seq, qf * n_seq) for qf in q_fracs if qf > 0)


# ---------------------------------------------------------------- oracle maps
def token_density(model, vid):
    """Mean information density (bits/px) of each token's patch in the finished video."""
    d = np.stack([info_density(f) for f in to_u8(vid)])
    return model.patchify(torch.from_numpy(d)[None])[0].mean(-1).numpy()


def token_activity(model, vid):
    """Max frame-to-frame change of each spatial patch (static camera => this is the source term)."""
    dv = np.abs(np.diff(vid, axis=0))
    dv = np.concatenate([dv[:1], dv], 0)                       # frame 0 inherits frame 1's change
    a = model.patchify(torch.from_numpy(dv)[None])[0].abs().mean(-1).numpy()
    g2 = model.g ** 2
    a = a.reshape(model.T, g2)
    return np.repeat(a.max(0, keepdims=True), model.T, 0).reshape(-1)    # per position, all frames


def settle_steps(hist, final, eps=0.05):
    """First step after which a token's predicted clean value stays within eps (RMS) of the final."""
    err = torch.stack([(h - final).pow(2).mean(-1).sqrt() for h in hist])      # steps x N
    ok = (err < eps).numpy()
    out = np.full(ok.shape[1], len(hist))
    for j in range(ok.shape[1]):
        bad = np.where(~ok[:, j])[0]
        out[j] = 0 if len(bad) == 0 else bad[-1] + 1
    return out


# ---------------------------------------------------------------- experiments
def run(n_clips=24):
    model = load()
    full_flops = model.flops(model.N) * N_STEPS
    res = {"n_clips": n_clips, "steps": N_STEPS, "full_gflops": full_flops / 1e9}
    clips = []
    t0 = time.time()
    for i in range(n_clips):
        g = torch.Generator().manual_seed(10_000 + i)
        x0 = torch.randn(1, model.N, model.P ** 2, generator=g)
        xf, _, hist, _ = sample(model, x0)
        vid = model.unpatchify(xf)[0].numpy()
        clips.append(dict(x0=x0, ref=xf, vid=vid, hist=hist,
                          dens=token_density(model, vid), act=token_activity(model, vid)))
    print(f"references: {time.time() - t0:.0f}s", flush=True)

    # ---- B1 settling (20 steps) + few-step regime (4 steps)
    st_all, dn_all, ac_all = [], [], []
    for c in clips:
        st = settle_steps(c["hist"], c["ref"][0])
        st_all.append(st); dn_all.append(c["dens"]); ac_all.append(c["act"])
    st, dn, ac = map(np.concatenate, (st_all, dn_all, ac_all))
    static = ac < 0.03
    few = []
    for c in clips[:12]:
        xf4, _, h4, _ = sample(model, c["x0"], n_steps=4)
        few.append(settle_steps(h4, xf4[0]))
    few = np.concatenate(few)
    res["B1"] = dict(
        frac_settled_by_25pct=float(np.mean(st <= N_STEPS // 4)),
        frac_settled_by_50pct=float(np.mean(st <= N_STEPS // 2)),
        median_settle_static=float(np.median(st[static])), median_settle_moving=float(np.median(st[~static])),
        frac_static_tokens=float(static.mean()),
        spearman_settle_vs_density=float(spearmanr(st, dn)[0]),
        spearman_settle_vs_activity=float(spearmanr(st, ac)[0]),
        few_step_settle_hist=np.bincount(few, minlength=5).tolist())
    print("B1", res["B1"], flush=True)

    def evaluate(make_policy=None, make_share=None, n_eval=n_clips, steps=N_STEPS):
        ps, fl, pv, proj = [], [], [], []
        for c in clips[:n_eval]:
            keep = share = None
            if make_share is not None:
                keep, share = make_share(c)
            pol = None if make_policy is None else make_policy(c)
            xf, f, _, counts = sample(model, c["x0"], steps, policy=pol, keep=keep, share_src=share)
            v = model.unpatchify(xf)[0].numpy()
            ps.append(psnr(v, c["vid"])); fl.append(f); pv.append(pvs_of(v))
            seq_frac = (model.N if keep is None else len(keep)) / model.N
            proj.append(projected_speedup(seq_frac, [q / (seq_frac * model.N) for q in counts]))
        return dict(psnr=float(np.mean(ps)), psnr_min=float(np.min(ps)),
                    speedup=float(full_flops / np.mean(fl)), projected_speedup_real_scale=float(np.mean(proj)),
                    pvs=float(np.mean(pv)))

    ref_pvs = float(np.mean([pvs_of(c["vid"]) for c in clips]))
    res["ref_pvs"] = ref_pvs

    # ---- B2 oracle spatial pruning vs random, warmup 25% of steps full
    warm = N_STEPS // 4
    res["B2"] = []
    for p in [0.5, 0.25, 0.1]:
        for kind in ["density", "random"]:
            def mk(c, p=p, kind=kind):
                nq = int(p * model.N)
                if kind == "density":
                    idx = torch.from_numpy(np.argsort(-c["dens"])[:nq].copy())
                else:
                    idx = torch.randperm(model.N, generator=torch.Generator().manual_seed(1))[:nq]
                return lambda k, a, b, temp: None if k < warm else idx
            r = evaluate(mk); r.update(keep_frac=p, kind=kind); res["B2"].append(r)
            print("B2", r, flush=True)
    # density + motion (what a real system would combine)
    for p in [0.25, 0.1]:
        def mk(c, p=p):
            nq = int(p * model.N)
            score = c["dens"] / c["dens"].max() + c["act"] / max(c["act"].max(), 1e-6)
            idx = torch.from_numpy(np.argsort(-score)[:nq].copy())
            return lambda k, a, b, temp: None if k < warm else idx
        r = evaluate(mk); r.update(keep_frac=p, kind="density+motion"); res["B2"].append(r)
        print("B2", r, flush=True)

    # ---- B3 oracle temporal sharing of static tokens across k-frame groups
    res["B3"] = []
    g2 = model.g ** 2
    for k_grp in [2, 4, 8]:
        for thr in [0.03, 0.06]:
            def mk_share(c, k_grp=k_grp, thr=thr):
                stat_pos = c["act"][:g2] < thr                     # per spatial position
                src = np.arange(model.N)
                for tf in range(model.T):
                    lead = (tf // k_grp) * k_grp
                    for p in np.where(stat_pos)[0]:
                        src[tf * g2 + p] = lead * g2 + p
                keep = np.unique(src)
                return torch.from_numpy(keep), torch.from_numpy(src)
            r = evaluate(make_share=mk_share)
            fr = np.mean([(clips[j]["act"][:g2] < thr).mean() for j in range(n_clips)])
            r.update(group=k_grp, static_thr=thr, static_frac=float(fr)); res["B3"].append(r)
            print("B3", r, flush=True)

    # ---- B3 + B2 combined: share static tokens, and cache the low-density remainder
    res["B3+B2"] = []
    for p in [0.5, 0.3]:
        def mk_share(c):
            stat_pos = c["act"][:g2] < 0.03
            src = np.arange(model.N)
            for tf in range(model.T):
                for q in np.where(stat_pos)[0]:
                    src[tf * g2 + q] = q
            keep = np.unique(src)
            return torch.from_numpy(keep), torch.from_numpy(src)

        def mk(c, p=p):
            keep, _ = mk_share(c)
            score = (c["dens"] / c["dens"].max() + c["act"] / max(c["act"].max(), 1e-6))[keep.numpy()]
            nq = int(p * len(keep))
            idx = torch.from_numpy(np.argsort(-score)[:nq].copy())
            return lambda k, a, b, temp: None if k < warm else idx
        r = evaluate(mk, mk_share); r.update(keep_frac_of_remaining=p); res["B3+B2"].append(r)
        print("B3+B2", r, flush=True)

    # ---- B4 online freezing (no oracle): 'delta' vs 'heat' rule
    res["B4"] = []
    for rule in ["delta", "heat"]:
        for tau in ([0.01, 0.02, 0.04, 0.08] if rule == "delta" else [0.005, 0.01, 0.02, 0.04]):
            def mk(c, rule=rule, tau=tau):
                frozen = torch.zeros(model.N, dtype=torch.bool)

                def pol(k, prev, now, temp):
                    if k < 2 or prev is None:
                        return None
                    chg = (now[0] - prev[0]).pow(2).mean(-1).sqrt()
                    flux = chg if rule == "delta" else chg * temp      # heat ~ T * dS
                    frozen[:] = frozen | (flux < tau)
                    return torch.where(~frozen)[0]
                return pol
            r = evaluate(mk); r.update(rule=rule, tau=tau); res["B4"].append(r)
            print("B4", r, flush=True)

    # ---- H heat conduction: attention mass between static and moving tokens
    cond = []
    for c in clips[:8]:
        xmid = None
        x = c["x0"].clone()
        for k in range(N_STEPS // 2):                                   # state at t = 0.5
            v, _ = model(x, torch.full((1,), k / N_STEPS)); x = x + v / N_STEPS
        maps = attention_maps(model, x, torch.full((1,), 0.5))
        st = torch.from_numpy(c["act"] < 0.03)
        if st.all() or (~st).all():
            continue
        for L, a in enumerate(maps):
            s2m = a[st][:, ~st].sum(-1).mean().item()                 # static queries -> moving keys
            cond.append(dict(layer=L, static_to_moving=s2m, uniform=float((~st).float().mean())))
    res["H_conduction"] = dict(
        mean_static_to_moving=float(np.mean([c["static_to_moving"] for c in cond])),
        mean_uniform_expectation=float(np.mean([c["uniform"] for c in cond])),
        by_layer=[float(np.mean([c["static_to_moving"] for c in cond if c["layer"] == L])) for L in range(len(model.blocks))])
    # cut: static queries may not attend to moving keys
    ps, pv = [], []
    for c in clips[:12]:
        st = torch.from_numpy(c["act"] < 0.03)
        mask = torch.zeros(model.N, model.N)
        mask[st.nonzero()[:, 0][:, None], (~st).nonzero()[:, 0][None]] = float("-inf")
        xf, _, _, _ = sample(model, c["x0"], attn_mask_fn=lambda: mask)
        v = model.unpatchify(xf)[0].numpy(); ps.append(psnr(v, c["vid"])); pv.append(pvs_of(v))
    res["H_cut"] = dict(psnr=float(np.mean(ps)), pvs=float(np.mean(pv)))
    print("H", res["H_conduction"], res["H_cut"], flush=True)
    json.dump(res, open("results/tiny_exps.json", "w"), indent=1, default=float)
    return model, clips, res


def run_c1(model, n=120):
    """C1 early abort + A5t generated vs real, on a larger batch of samples."""
    out = {}
    g = torch.Generator().manual_seed(77)
    x0 = torch.randn(n, model.N, model.P ** 2, generator=g)
    x = x0.clone(); early = {}
    for k in range(N_STEPS):
        v, _ = model(x, torch.full((n,), k / N_STEPS))
        if k in (1, 3, 5):
            early[k] = (x + (1 - k / N_STEPS) * v).clone()
        x = x + v / N_STEPS
    fin = [pvs_of(model.unpatchify(x[i:i + 1])[0].numpy()) for i in range(n)]
    fin = np.array(fin)
    bad = fin >= np.percentile(fin, 75)
    for k, xe in early.items():
        e = np.array([pvs_of(model.unpatchify(xe[i:i + 1])[0].numpy()) for i in range(n)])
        out[f"step{k + 1}"] = dict(spearman=float(spearmanr(e, fin)[0]), auc_worst25=float(roc_auc_score(bad, e)))
    # abort economics at step 4 (20% of compute): drop the 25% of clips with the worst early score
    k = 3
    e = np.array([pvs_of(model.unpatchify(early[k][i:i + 1])[0].numpy()) for i in range(n)])
    keep = e < np.percentile(e, 75)
    out["abort_at_step4"] = dict(bad_rate_before=float(bad.mean()), bad_rate_after=float(bad[keep].mean()),
                                 compute_per_kept_clip=float((keep.sum() * 1.0 + (~keep).sum() * 0.2) / keep.sum()))
    real = make_dataset(n, seed=999)
    rp = np.array([pvs_of(v) for v in real])
    out["A5t_generated_vs_real"] = dict(auc=float(roc_auc_score(np.r_[np.zeros(n), np.ones(n)], np.r_[rp, fin])),
                                        pvs_real=float(rp.mean()), pvs_generated=float(fin.mean()))
    print("C1/A5t", out, flush=True)
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "main"):
        model, clips, res = run()
    if which in ("all", "c1"):
        model = load()
        c1 = run_c1(model)
        try:
            res = json.load(open("results/tiny_exps.json"))
        except Exception:
            res = {}
        res["C1"] = c1
        json.dump(res, open("results/tiny_exps.json", "w"), indent=1, default=float)
