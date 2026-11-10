"""R3: does sharing noise across frames (needed for exact frame sharing) hurt sample quality?

Classifier two-sample test (C2ST). A small 3D-conv classifier is trained to tell REAL
training-distribution clips from GENERATED clips; held-out accuracy near 50% means
the generated set is indistinguishable from real. We compare:

  A  normal sampling (independent noise per frame)
  B  draft-pass shared noise in static regions (what frame sharing requires)
  C  B, but generated with frame-sharing compute + proportional attention

Both A and B are scored against the same real set, with the same classifier recipe and
several seeds. If B's detectability is clearly above A's, shared noise hurts quality.
A cheaper, fully independent check is also reported: the per-clip temporal statistics
(mean |frame difference| in static vs moving regions) compared with real clips.
"""
import sys, json, time
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, ".")
import experiments.tiny_exps as E
from experiments.tiny_data import make_dataset

torch.set_grad_enabled(False)
model = E.load()
g2 = model.g ** 2
N = int(sys.argv[1]) if len(sys.argv) > 1 else 600


def generate(n, mode, seed0, thr=0.03, draft_steps=2):
    out = []
    for i in range(n):
        x0 = torch.randn(1, model.N, model.P ** 2, generator=torch.Generator().manual_seed(seed0 + i))
        if mode == "A":
            xf = E.sample(model, x0)[0]
        else:
            draft = model.unpatchify(E.sample(model, x0, n_steps=draft_steps)[0])[0].numpy()
            stat = E.token_activity(model, draft)[:g2] < thr
            src = np.arange(model.N)
            for tf in range(model.T):
                for p in np.where(stat)[0]:
                    src[tf * g2 + p] = p
            src = torch.from_numpy(src)
            if mode == "B":
                xf = E.sample(model, x0[:, src])[0]
            else:
                keep = torch.from_numpy(np.unique(src.numpy()))
                xf = E.sample(model, x0, keep=keep, share_src=src, prop_attn=True)[0]
        out.append(model.unpatchify(xf)[0].numpy())
    return np.clip(np.stack(out), -1, 1).astype(np.float32)


class Clf(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool3d((1, 2, 2)),
                               nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
                               nn.Conv3d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d(1),
                               nn.Flatten(), nn.Linear(32, 1))

    def forward(self, x):
        return self.f(x[:, None])


def c2st(real, fake, seed, epochs=15):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    X = np.concatenate([real, fake]); y = np.r_[np.zeros(len(real)), np.ones(len(fake))].astype(np.float32)
    idx = rng.permutation(len(X)); cut = int(0.7 * len(X)); tr, te = idx[:cut], idx[cut:]
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    clf = Clf(); opt = torch.optim.Adam(clf.parameters(), 1e-3)
    with torch.enable_grad():
        for ep in range(epochs):
            for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 32)):
                loss = nn.functional.binary_cross_entropy_with_logits(clf(Xt[b])[:, 0], yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
    p = (clf(Xt[te])[:, 0] > 0).float().numpy()
    return float((p == y[te]).mean())


def temporal_stats(v):
    """Per clip: fraction of near-static pixels and mean change of moving pixels."""
    d = np.abs(np.diff(v, axis=1))
    stat = d.max(1) < 0.03
    return np.c_[stat.mean((1, 2)), np.array([dd[:, ~s].mean() if (~s).any() else 0 for dd, s in zip(d, stat)])]


if __name__ == "__main__":
    t0 = time.time()
    real = make_dataset(N, seed=4321)
    sets = {m: generate(N, m, 20_000 + {"A": 0, "B": 0, "C": 0}[m]) for m in ["A", "B", "C"]}
    print(f"generated in {time.time() - t0:.0f}s", flush=True)
    res = {"n": N}
    for m, fake in sets.items():
        accs = [c2st(real, fake, s) for s in range(3)]
        ts_f, ts_r = temporal_stats(fake), temporal_stats(real)
        res[m] = dict(c2st_acc_mean=float(np.mean(accs)), c2st_accs=accs,
                      static_frac=float(ts_f[:, 0].mean()), moving_change=float(ts_f[:, 1].mean()))
        print(m, res[m], flush=True)
    ts_r = temporal_stats(real)
    res["real"] = dict(static_frac=float(ts_r[:, 0].mean()), moving_change=float(ts_r[:, 1].mean()))
    # B vs A directly: can a classifier tell shared-noise samples from normal ones?
    res["A_vs_B_c2st"] = float(np.mean([c2st(sets["A"], sets["B"], s) for s in range(3)]))
    print(res, flush=True)
    json.dump(res, open("results/r3_shared_noise_quality.json", "w"), indent=1)

