"""CPU checks of the Wan port against diffusers' own WanTransformer3DModel (random small weights)."""
import sys, time
import torch
from diffusers import WanTransformer3DModel

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import ecm_wan as E

torch.manual_seed(0)
cfg = dict(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=24, in_channels=16, out_channels=16,
           text_dim=32, freq_dim=32, ffn_dim=96, num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
           rope_max_seq_len=64)
m = WanTransformer3DModel(**cfg).eval()
# randomise zero-initialised parameters so the test is not trivially passed
with torch.no_grad():
    for p in m.parameters():
        p.add_(torch.randn_like(p) * 0.02)
lat = torch.randn(1, 16, 5, 8, 12)
enc = torch.randn(1, 7, 32)
ts = torch.tensor([700.0])

# 1. patchify/unpatchify roundtrip
tok = E.patchify(lat, m.config.patch_size)
assert torch.equal(E.unpatchify(tok, m.config.patch_size, lat.shape), lat)

# 2. full-token forward == diffusers forward
ref = m(lat, ts, enc, return_dict=False)[0]
N = tok.shape[1]
out = E.forward_tokens(m, lat.shape, tok, torch.arange(N), ts, enc)
err = (E.unpatchify(out, m.config.patch_size, lat.shape) - ref).abs().max().item()
print("full forward max |diff|:", err); assert err < 1e-4

# 3. log(m) key bias == m duplicated keys (attention-level identity), using one block's self-attention
blk = m.blocks[0]
h = torch.randn(1, 6, 48)
cos, sin = m.rope(lat)
rope = (cos[:, :6], sin[:, :6])
dup = torch.tensor([0, 1, 2, 3, 4, 5, 5, 5])                   # token 5 appears 3 times
o_dup = E.self_attn(blk.attn1, h[:, dup], (cos[:, :6][:, dup], sin[:, :6][:, dup]), None)[:, :6]
kb = torch.zeros(1, 6); kb[0, 5] = torch.log(torch.tensor(3.0))
o_bias = E.self_attn(blk.attn1, h, rope, kb)
err = (o_dup - o_bias).abs().max().item()
print("log(m) bias vs duplication max |diff|:", err); assert err < 1e-5

# 4. sampler: sharing with thr=0 (nothing static) reproduces plain sampling exactly
noise = torch.randn(1, 16, 5, 8, 12)
a, ia = E.sample(m, noise, enc, torch.zeros_like(enc), n_steps=6)
b, ib = E.sample(m, noise, enc, torch.zeros_like(enc), n_steps=6, share=dict(thr=0.0, warm=2))
print("thr=0 sharing vs plain max |diff|:", (a - b).abs().max().item(), ib["kept_frac"]); assert torch.allclose(a, b, atol=1e-5)

# 5. sharing everything runs, compute drops, output finite
c, ic = E.sample(m, noise, enc, torch.zeros_like(enc), n_steps=6, share=dict(thr=1e9, warm=2, first_frame=1))
print("thr=inf: kept_frac", ic["kept_frac"], "work ratio", ic["full_token_forwards"] / ic["token_forwards"])
assert torch.isfinite(c).all() and ic["kept_frac"] < 0.5
# all static tokens from frame 1 on should end identical in their clean part only if noise is equal; check frames 1.. are close to each other
print("frame spread (thr=inf):", (c[:, :, 1:] - c[:, :, 1:].mean(2, keepdim=True)).abs().mean().item(),
      "vs plain:", (a[:, :, 1:] - a[:, :, 1:].mean(2, keepdim=True)).abs().mean().item())

# 6. pan gate: a translating latent disables sharing
x = torch.randn(1, 20, 20)
seq = torch.stack([torch.roll(x, shifts=2 * f, dims=2) for f in range(6)], 1)    # 1, F, H, W (token grid)
tokmap = seq.reshape(1, 6 * 400, 1).repeat(1, 1, 4)
src, info = E.static_map(tokmap, (6, 20, 20), thr=1e9, first_frame=1, pan_gate=0.5)
print("pan estimate (tokens/frame):", info["pan"], "static_frac", info["static_frac"]); assert info["static_frac"] == 0
print("ALL CHECKS PASSED")

# 7. quantile sharing + warm_sigma path + UniPC baseline run
d, idd = E.sample(m, noise, enc, torch.zeros_like(enc), n_steps=10, share=dict(frac=0.5, warm_sigma=0.85, first_frame=1))
print("frac=0.5:", {k: v for k, v in idd.items() if k != "act"}); assert 0 < idd["static_frac"] <= 0.55
u, iu = E.sample_unipc(m, noise, enc, torch.zeros_like(enc), n_steps=8)
assert torch.isfinite(u).all(); print("unipc ok")
# sub-token pan: 0.5-token shift per frame via linear interpolation
xs = torch.nn.functional.avg_pool2d(torch.randn(1, 1, 40, 40), 2)[0, 0]
fr = [xs] + [0.5 * (torch.roll(xs, f, 1) + torch.roll(xs, f - 1, 1)) if False else torch.roll(xs, f, 1) for f in range(1, 3)]
print("pan est (1 token/frame expected ~1):", E.global_shift(fr[0], fr[2]) / 2)
print("ALL CHECKS PASSED 2")

# 8. draft-map path (toy protocol): runs, counts the draft's work, shares from step 0
e, ie = E.sample(m, noise, enc, torch.zeros_like(enc), n_steps=10, share=dict(frac=0.75, draft=3, first_frame=1))
print("draft3 frac0.75:", {k: v for k, v in ie.items() if k != "act"})
assert ie["token_forwards"] < ie["full_token_forwards"] and ie["draft_steps"] == 3 and torch.isfinite(e).all()
print("ALL CHECKS PASSED 3")

# 9. offset-preserving sharing: members end at leader + (map offset), not identical
g, ig = E.sample(m, noise, enc, torch.zeros_like(enc), n_steps=10, share=dict(frac=0.75, draft=3, first_frame=1, offset=True))
dl, _ = E.sample(m, noise, enc, torch.zeros_like(enc), n_steps=3)
src9, _ = E.static_map(E.patchify(dl, (1, 2, 2)), (5, 4, 6), 0, 1, None, 0.75)
tg, td = E.patchify(g, (1, 2, 2))[0], E.patchify(dl, (1, 2, 2))[0]
sh = src9 != torch.arange(len(src9))
err = ((tg[sh] - tg[src9[sh]]) - (td[sh] - td[src9[sh]])).abs().max().item()
print("offset sharing: member-leader diff equals draft offsets, max err", err); assert err < 1e-4 and torch.isfinite(g).all()
print("ALL CHECKS PASSED 4")
