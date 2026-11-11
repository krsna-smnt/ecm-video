"""Re-noised averaged frame sharing for Wan 2.1 (diffusers WanTransformer3DModel).

The method (same as the toy study, experiments/r3f_battery.py::avg_sample):
  1. Run the first `warm` sampling steps at full compute. From the model's clean-latent estimate
     x0_hat at that point, mark spatial token positions whose content barely changes across latent
     frames ("static"). Optionally disable sharing if the whole frame is translating (camera pan).
  2. For the remaining steps, every static position forms a group: its tokens in latent frames
     `first..F-1`. Each token keeps its OWN noisy state. The group's leader gets the average of the
     group's states plus sigma*sqrt(1-1/m) fresh noise (restoring the per-token noise level), and a
     log(m) key bias in self-attention (so the leader carries the attention weight of m copies).
     Only leaders and non-static tokens go through the transformer.
  3. The leader's clean estimate x0_hat is given to every token of the group; each token then steps
     with its own noise estimate eps_i = (x_i - (1 - sigma) x0_hat) / sigma.

Conventions (diffusers Wan / flow matching): x_sigma = (1 - sigma) x0 + sigma eps, model output
v = eps - x0, so x0_hat = x - sigma v.  Euler step: x <- x + (sigma_next - sigma) v.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- token layout

def patchify(lat, p):
    """(B,C,F,H,W) -> (B, N, pt*ph*pw*C), token order and channel layout identical to proj_out's."""
    B, C, Fr, H, W = lat.shape
    pt, ph, pw = p
    x = lat.reshape(B, C, Fr // pt, pt, H // ph, ph, W // pw, pw)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)          # B, f, h, w, pt, ph, pw, C
    return x.reshape(B, (Fr // pt) * (H // ph) * (W // pw), pt * ph * pw * C)


def unpatchify(tok, p, shape):
    B, C, Fr, H, W = shape
    pt, ph, pw = p
    x = tok.reshape(B, Fr // pt, H // ph, W // pw, pt, ph, pw, C)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return x.flatten(6, 7).flatten(4, 5).flatten(2, 3)


# ----------------------------------------------------------------------------- kept-token forward

def _rope(x, cos, sin):
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    c, s = cos[..., 0::2], sin[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out.type_as(x)


def self_attn(attn, h, rope, key_bias):
    """WanAttnProcessor self-attention with an additive per-key bias (B or 1, K)."""
    if attn.fused_projections:
        q, k, v = attn.to_qkv(h).chunk(3, dim=-1)
    else:
        q, k, v = attn.to_q(h), attn.to_k(h), attn.to_v(h)
    q, k = attn.norm_q(q), attn.norm_k(k)
    q, k, v = (t.unflatten(2, (attn.heads, -1)) for t in (q, k, v))
    q, k = _rope(q, *rope), _rope(k, *rope)
    mask = None if key_bias is None else key_bias.to(q.dtype)[:, None, None, :]
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    o = o.transpose(1, 2).flatten(2, 3).type_as(q)
    o = attn.to_out[0](o)
    return attn.to_out[1](o)


def block_forward(blk, h, enc, temb, rope, key_bias):
    sh, sc, g, csh, csc, cg = (blk.scale_shift_table + temb.float()).chunk(6, dim=1)
    n = (blk.norm1(h.float()) * (1 + sc) + sh).type_as(h)
    h = (h.float() + self_attn(blk.attn1, n, rope, key_bias) * g).type_as(h)
    n = blk.norm2(h.float()).type_as(h)
    h = h + blk.attn2(n, enc, None, None)
    n = (blk.norm3(h.float()) * (1 + csc) + csh).type_as(h)
    return (h.float() + blk.ffn(n).float() * cg).type_as(h)


@torch.no_grad()
def forward_tokens(model, lat_shape, tok, keep, timestep, enc, key_bias=None):
    """Run WanTransformer3DModel on a subset of tokens.

    tok: (B, K, pt*ph*pw*C) latent tokens at positions `keep` (LongTensor of K indices into the full
    token grid). Returns the model output for those tokens, same layout. With keep = all positions and
    key_bias = None this equals model(...) exactly (checked in test_ecm_wan.py)."""
    p = model.config.patch_size
    B = tok.shape[0]
    full_cos, full_sin = model.rope(torch.empty(lat_shape, device=tok.device))
    rope = (full_cos[:, keep], full_sin[:, keep])
    # patch embedding is a strided conv == a linear map on each token
    W = model.patch_embedding.weight                                    # D, C, pt, ph, pw
    Wl = W.permute(0, 2, 3, 4, 1).reshape(W.shape[0], -1)               # D, pt*ph*pw*C (token layout)
    h = F.linear(tok.to(W.dtype), Wl, model.patch_embedding.bias)
    temb, tproj, enc, _ = model.condition_embedder(timestep, enc, None, timestep_seq_len=None)
    tproj = tproj.unflatten(1, (6, -1))
    for blk in model.blocks:
        h = block_forward(blk, h, enc, tproj, rope, key_bias)
    sh, sc = (model.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
    h = (model.norm_out(h.float()) * (1 + sc) + sh).type_as(h)
    return model.proj_out(h)


# ----------------------------------------------------------------------------- schedule & sampling

def flow_sigmas(n_steps, shift):
    s = torch.linspace(1.0, 1.0 / 1000, n_steps, dtype=torch.float64)
    s = shift * s / (1 + (shift - 1) * s)
    return torch.cat([s, torch.zeros(1, dtype=torch.float64)])


def _guided(model, shape, tok, keep, sigma, enc_c, enc_u, gscale, key_bias):
    """CFG velocity for kept tokens; cond & uncond batched together."""
    B = tok.shape[0]
    ts = torch.full((2 * B,), float(sigma) * 1000, device=tok.device)
    kb = None if key_bias is None else key_bias.expand(2 * B, -1)
    v = forward_tokens(model, shape, torch.cat([tok, tok]), keep, ts, torch.cat([enc_c, enc_u]), kb).float()
    vc, vu = v[:B], v[B:]
    return vu + gscale * (vc - vu)


def static_map(x0_tok, grid, thr, first_frame, pan_gate=None):
    """x0_tok: (1, N, D) clean-latent estimate. grid = (Fl, Hl, Wl) token grid.
    A position is static if the max over consecutive latent frames (from first_frame on) of the RMS
    token change, divided by the RMS token magnitude, is below thr. Returns (src, info)."""
    Fl, Hl, Wl = grid
    x = x0_tok[0].float().reshape(Fl, Hl * Wl, -1)[first_frame:]
    scale = x.pow(2).mean().sqrt() + 1e-8
    act = (x[1:] - x[:-1]).pow(2).mean(-1).sqrt().amax(0) / scale       # Hl*Wl
    stat = act < thr
    pan = 0.0
    if pan_gate is not None:
        pan = global_shift(x[0].mean(-1).reshape(Hl, Wl), x[-1].mean(-1).reshape(Hl, Wl)) / max(1, x.shape[0] - 1)
        if pan > pan_gate:
            stat[:] = False
    src = torch.arange(Fl * Hl * Wl)
    pos = torch.nonzero(stat).flatten()
    leader = first_frame * Hl * Wl + pos
    for f in range(first_frame + 1, Fl):
        src[f * Hl * Wl + pos] = leader
    return src, dict(static_frac=float(stat.float().mean()), pan=float(pan), act=act)


def global_shift(a, b):
    """Phase-correlation translation (in tokens) between two 2-D maps."""
    A, Bf = torch.fft.fft2(a - a.mean()), torch.fft.fft2(b - b.mean())
    R = A * Bf.conj()
    r = torch.fft.ifft2(R / (R.abs() + 1e-8)).real
    i = int(torch.argmax(r))
    dy, dx = divmod(i, r.shape[1])
    dy = dy - r.shape[0] if dy > r.shape[0] // 2 else dy
    dx = dx - r.shape[1] if dx > r.shape[1] // 2 else dx
    return math.hypot(dy, dx)


@torch.no_grad()
def sample(model, noise, enc_c, enc_u, n_steps=30, shift=3.0, gscale=5.0, share=None, seed=0, callback=None):
    """Euler flow sampler. share=None -> plain sampling. share=dict(thr, warm, first_frame, pan_gate,
    renoise=True, average=True, prop=True) -> re-noised averaged frame sharing.
    Returns (latents, stats) where stats counts token-forwards (compute) and sharing info."""
    p = model.config.patch_size
    shape = noise.shape
    grid = (shape[2] // p[0], shape[3] // p[1], shape[4] // p[2])
    N = grid[0] * grid[1] * grid[2]
    x = patchify(noise.float(), p)
    sig = flow_sigmas(n_steps, shift)
    allk = torch.arange(N, device=x.device)
    work, info = 0, {}
    gen = torch.Generator(device=x.device).manual_seed(seed)
    keep = inv = mult = kb = None
    for k in range(n_steps):
        s, s2 = float(sig[k]), float(sig[k + 1])
        if share is None or k < share.get("warm", 2) or keep is None:
            v = _guided(model, shape, x.to(model.dtype), allk, s, enc_c, enc_u, gscale, None)
            work += N
            x0 = x - s * v
            x = x + (s2 - s) * v
            if share is not None and k == share.get("warm", 2) - 1:
                src, info = static_map(x0, grid, share["thr"], share.get("first_frame", 1), share.get("pan_gate"))
                keep = torch.unique(src).to(x.device)
                inv = torch.searchsorted(keep, src.to(x.device))
                mult = torch.bincount(inv, minlength=len(keep)).float()
                kb = torch.log(mult)[None] if share.get("prop", True) else None
                info["kept_frac"] = len(keep) / N
        else:
            if share.get("average", True):
                z = torch.zeros(x.shape[0], len(keep), x.shape[-1], device=x.device).index_add_(1, inv, x) / mult[None, :, None]
                if share.get("renoise", True):
                    z = z + s * torch.sqrt(1 - 1 / mult)[None, :, None] * torch.randn(z.shape, generator=gen, device=x.device)
            else:
                z = x[:, keep]
            v = _guided(model, shape, z.to(model.dtype), keep, s, enc_c, enc_u, gscale, kb)
            work += len(keep)
            x0 = (z - s * v)[:, inv]                       # leader's clean estimate for every member
            eps = (x - (1 - s) * x0) / s                    # each token keeps its own noise estimate
            x = (1 - s2) * x0 + s2 * eps
        if callback is not None:
            callback(k, x, x0)
    info["token_forwards"] = work
    info["full_token_forwards"] = N * n_steps
    return unpatchify(x, p, shape), info
