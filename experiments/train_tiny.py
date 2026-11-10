"""Train the tiny video DiT with rectified flow. Resumable; runs for a time budget per call.

usage: python3 experiments/train_tiny.py <minutes> [total_steps]
"""
import sys, os, time, json
import numpy as np
import torch
sys.path.insert(0, ".")
from ecm.tinydit import TinyVideoDiT, flow_matching_loss
from experiments.tiny_data import make_dataset

torch.manual_seed(0)
budget = float(sys.argv[1]) * 60 if len(sys.argv) > 1 else 540
total = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
ck = os.environ.get("CKPT", "results/tiny_ckpt.pt")
data_path = os.environ.get("DATA", "results/tiny_data.npz")
PAN = float(os.environ.get("PAN", 0.0))

if not os.path.exists(data_path):
    np.savez_compressed(data_path, x=make_dataset(12000, seed=0 if PAN == 0 else 11, pan_prob=PAN))
X = torch.from_numpy(np.load(data_path)["x"])

import os as _os
CFG = dict(P=int(_os.environ.get("P", 4)), d=int(_os.environ.get("D", 96)))
model = TinyVideoDiT(**CFG)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
ema = TinyVideoDiT(**CFG); ema.load_state_dict(model.state_dict())
step, log = 0, []
if os.path.exists(ck):
    s = torch.load(ck, weights_only=False)  # our own checkpoint
    model.load_state_dict(s["model"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"])
    step, log = s["step"], s["log"]

sched = lambda k: 1e-3 * min(1.0, k / 200) * (0.5 * (1 + np.cos(np.pi * min(k, total) / total)) * 0.9 + 0.1)
t0 = time.time(); B = 32
model.train()
while step < total and time.time() - t0 < budget:
    for g in opt.param_groups:
        g["lr"] = sched(step)
    idx = torch.randint(0, len(X), (B,))
    x1 = model.patchify(X[idx])
    loss = flow_matching_loss(model, x1)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.mul_(0.999).add_(pm, alpha=0.001)
    step += 1
    if step % 1000 == 0:
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step, log=log, cfg=CFG), ck)
    if step % 100 == 0:
        log.append((step, float(loss)))
        print(step, round(float(loss), 4), f"{(time.time() - t0) / 60:.1f}min", flush=True)
torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step, log=log, cfg=CFG), ck)
print("saved at step", step)
