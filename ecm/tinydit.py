"""A tiny video diffusion transformer (rectified flow) for mechanism tests on CPU.

Video: T frames of S x S gray pixels, patchified into P x P patches, so
N = T * (S/P)^2 tokens with full spatio-temporal attention (like video DiTs).

The forward pass supports the compute-saving modes under test:
  * query subset  - only some tokens are recomputed this step; the rest reuse their
                    cached per-layer keys/values and cached velocity (token caching).
  * removed tokens - tokens dropped from the sequence entirely (temporal sharing:
                    a shared token is computed once and copied to other frames).
A FLOP counter reports the cost of every call so speedups are measured, not guessed.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ada = nn.Linear(d, 4 * d)   # time-conditioned scale/shift for both norms

    def forward(self, x, temb, q_idx=None, kv_cache=None):
        """x: B x n x d hidden states of the tokens being computed (all, or the query subset).
        q_idx: LongTensor of positions (in the full sequence) of x's tokens, or None = all.
        kv_cache: (K, V) for the FULL sequence from the last time each token was computed;
                  fresh K/V of the computed tokens are written into it.
        Returns new x and the updated full (K, V)."""
        B, n, d = x.shape
        s1, b1, s2, b2 = self.ada(temb).unsqueeze(1).chunk(4, -1)
        hdn = self.n1(x) * (1 + s1) + b1
        q, k, v = self.qkv(hdn).chunk(3, -1)
        if q_idx is not None:
            K, V = kv_cache
            K = K.clone(); V = V.clone()
            K[:, q_idx] = k; V[:, q_idx] = v
        else:
            K, V = k, v
        split = lambda t: t.view(B, -1, self.h, d // self.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(split(q), split(K), split(V))
        a = a.transpose(1, 2).reshape(B, n, d)
        x = x + self.o(a)
        x = x + self.mlp(self.n2(x) * (1 + s2) + b2)
        return x, (K, V)


class TinyVideoDiT(nn.Module):
    def __init__(self, T=8, S=32, P=4, d=96, depth=4, heads=4):
        super().__init__()
        self.T, self.S, self.P, self.d = T, S, P, d
        self.g = S // P
        self.N = T * self.g * self.g
        self.embed = nn.Linear(P * P, d)
        self.pos_t = nn.Parameter(torch.randn(T, 1, d) * 0.02)
        self.pos_s = nn.Parameter(torch.randn(1, self.g * self.g, d) * 0.02)
        self.temb = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.nf = nn.LayerNorm(d)
        self.out = nn.Linear(d, P * P)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    # --- patch helpers: video B x T x S x S  <->  tokens B x N x P*P ---
    def patchify(self, v):
        B = v.shape[0]; g, P = self.g, self.P
        return v.view(B, self.T, g, P, g, P).permute(0, 1, 2, 4, 3, 5).reshape(B, self.N, P * P)

    def unpatchify(self, x):
        B = x.shape[0]; g, P = self.g, self.P
        return x.view(B, self.T, g, g, P, P).permute(0, 1, 2, 4, 3, 5).reshape(B, self.T, self.S, self.S)

    def time_embed(self, t):
        f = torch.exp(torch.linspace(0, math.log(1000), 32, device=t.device))
        e = t[:, None] * f[None]
        return self.temb(torch.cat([e.sin(), e.cos()], -1))

    def pos(self):
        return (self.pos_t + self.pos_s).reshape(self.N, self.d)

    def forward(self, xt_tok, t, q_idx=None, caches=None, keep=None):
        """xt_tok: B x N x P*P noisy tokens. Returns velocity for computed tokens.

        keep:  LongTensor of token positions kept in the sequence (others removed); None = all.
        q_idx: positions (within the kept sequence) recomputed now; None = all kept tokens.
        caches: per-layer (K, V) over the kept sequence, required when q_idx is given.
        Returns (velocity of computed tokens, new caches)."""
        temb = self.time_embed(t)
        pos = self.pos()
        if keep is not None:
            xt_tok = xt_tok[:, keep]; pos = pos[keep]
        h = self.embed(xt_tok) + pos
        if q_idx is not None:
            h = h[:, q_idx]
        new = []
        for i, blk in enumerate(self.blocks):
            h, kv = blk(h, temb, q_idx, None if caches is None else caches[i])
            new.append(kv)
        return self.out(self.nf(h)), new

    # --- analytic FLOP count (multiply-adds x2) of one forward call ---
    def flops(self, n_seq, n_q=None):
        n_q = n_seq if n_q is None else n_q
        d = self.d
        per_layer = (2 * n_q * d * d * 1          # Q projection (computed tokens)
                     + 2 * n_q * d * d * 2        # K, V projections (computed tokens; others cached)
                     + 2 * 2 * n_q * n_seq * d    # QK^T and AV
                     + 2 * n_q * d * d            # output projection
                     + 2 * n_q * 8 * d * d)       # MLP (4x hidden, two matmuls)
        return len(self.blocks) * per_layer + 2 * n_q * (self.P ** 2) * d * 2


def flow_matching_loss(model, x1_tok):
    """Rectified flow: x_t = t*x1 + (1-t)*x0, target velocity x1 - x0."""
    B = x1_tok.shape[0]
    x0 = torch.randn_like(x1_tok)
    t = torch.rand(B)
    xt = t[:, None, None] * x1_tok + (1 - t[:, None, None]) * x0
    v, _ = model(xt, t)
    return F.mse_loss(v, x1_tok - x0)
