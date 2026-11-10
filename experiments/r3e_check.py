import sys, json; sys.path.insert(0, ".")
import numpy as np, torch, cv2
import experiments.tiny_exps as E
import experiments.r3c_avg_sharing as R
from experiments.r3b_quality import fd, feats, tstats
from experiments.tiny_data import make_dataset
torch.set_grad_enabled(False)
m = R.model; N = 400; real = make_dataset(N, seed=4321)
def run(steps, share, thr=0.03):
    R.NS = steps; vids = []
    for i in range(N):
        x0 = torch.randn(1, m.N, m.P ** 2, generator=torch.Generator().manual_seed(30_000 + i))
        xf = R.avg_share_sample2(x0, thr=thr)[0] if share else E.sample(m, x0, n_steps=steps)[0]
        vids.append(np.clip(m.unpatchify(xf)[0].numpy(), -1, 1))
    return np.stack(vids).astype(np.float32)
out = {}; keep = {}
for name, steps, share in [("full20", 20, False), ("avg20", 20, True), ("avg10", 10, True), ("full10", 10, False)]:
    v = run(steps, share); keep[name] = v
    fds = [fd(feats(real, s), feats(v, s)) for s in range(5)]
    out[name] = dict(FD=float(np.mean(fds)), FD_per_seed=[float(x) for x in fds], **tstats(v)); print(name, out[name], flush=True)
json.dump(out, open("results/r3e_check.json", "w"), indent=1)
rows = [np.concatenate(list(real[i]), 1) for i in range(2)]
for k in ["full20", "avg20", "avg10"]:
    rows += [np.concatenate(list(keep[k][i]), 1) for i in range(2)]
g = np.concatenate(rows, 0)
cv2.imwrite("results/fig_r3e_samples.png", cv2.resize(((g + 1) * 127.5).clip(0, 255).astype(np.uint8), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
