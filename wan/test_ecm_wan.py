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
