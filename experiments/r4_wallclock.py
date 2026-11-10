"""R4: is the token saving real wall-clock time? (CPU; random weights, speed only)

Times one forward pass of video DiTs of increasing token count with (a) all tokens and
(b) only the kept tokens of frame sharing (30% kept, as in re-noised sharing at thr 0.02),
including the gather / mean / scatter overhead of the sharing step. Reports the measured
speed ratio next to the FLOP ratio from the same formula used for projections.
"""
import sys, time, json
import torch
sys.path.insert(0, ".")
from ecm.tinydit import TinyVideoDiT

torch.set_grad_enabled(False)
torch.manual_seed(0)
res = []
for T, S, P, d, depth in [(8, 32, 8, 128, 4), (8, 64, 8, 256, 6), (16, 64, 8, 256, 6), (16, 128, 8, 256, 4)]:
    m = TinyVideoDiT(T=T, S=S, P=P, d=d, depth=depth).eval()
    N = m.N; keep_n = int(0.30 * N)
    x = torch.randn(1, N, P * P); t = torch.full((1,), 0.5)
    keep = torch.sort(torch.randperm(N)[:keep_n]).values
    inv = torch.randint(0, keep_n, (N,)); mult = torch.bincount(inv, minlength=keep_n).float().clamp(min=1)
    kb = torch.log(mult)

    def full():
        return m(x, t)[0]

    def shared():
        z = torch.zeros(1, keep_n, x.shape[-1]).index_add_(1, inv, x) / mult[None, :, None]   # averaging
        temb = m.time_embed(t); h = m.embed(z) + m.pos()[keep]
        for blk in m.blocks:
            h, _ = blk(h, temb, None, None, kb)
        return m.out(m.nf(h))[:, inv]                                                        # scatter back

    def bench(fn, reps):
        fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps

    reps = max(2, int(2000 / N))
    tf, ts = bench(full, reps), bench(shared, reps)
    fl = m.flops(N) / m.flops(keep_n)
    r = dict(tokens=N, width=d, depth=depth, full_ms=tf * 1e3, shared_ms=ts * 1e3, measured_speedup=tf / ts, flop_speedup=fl)
    res.append(r)
    print(f"tokens {N:5d}  d {d}  full {tf*1e3:8.1f} ms  shared(30%) {ts*1e3:8.1f} ms  measured {tf/ts:4.2f}x  FLOPs {fl:4.2f}x", flush=True)
json.dump(res, open("results/r4_wallclock.json", "w"), indent=1)
