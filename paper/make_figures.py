"""Builds every figure in the technical report from results/*.json and the clips.

usage (from the repo root): python3 paper/make_figures.py
"""
import sys, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from ecm.pvs import score_video, load_video
from experiments.synth import make_clip

OUT = "paper/figs"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
                     "legend.fontsize": 6.2, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
COL, FULL = 3.3, 6.9   # inches: one column / full width


def overlay(ax, frame, heat, thr=0.5, vmax=4):
    ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
    ax.imshow(np.ma.masked_where(heat < thr, heat), cmap="autumn", alpha=0.85, vmin=thr, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])


# --- Fig 1: synthetic sanity cases ---------------------------------------------------
fig, axs = plt.subplots(3, 4, figsize=(FULL, 4.3))
for r, (ev, lab) in enumerate([("none", "Legal:\nmoves"), ("occlude", "Legal:\noccluded"),
                               ("vanish", "Violation:\nvanishes")]):
    f, _, _ = make_clip(ev, np.random.default_rng(1002)); s = score_video(f)
    for c, t in enumerate([5, 6, 7, 8]):
        overlay(axs[r, c], f[t + 1], s.heatmaps[t])
        axs[r, c].set_title(f"{t}$\\to${t+1}   PVS$_t$ = {s.frame_scores[t]:.3f}")
    axs[r, 0].set_ylabel(lab)
plt.tight_layout(pad=0.3); plt.savefig(f"{OUT}/fig_sanity.pdf"); plt.close()

# --- Fig 2: detection AUCs -----------------------------------------------------------
a1 = json.load(open("results/a1_benchmark.json"))["summary"]
b2 = json.load(open("results/a1b_vector_pf2.json"))["summary"]
fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.15), gridspec_kw=dict(width_ratios=[1.25, 1]))
types = ["vanish", "appear", "melt", "swap", "boil"]
x = np.arange(len(types) + 1); w = 0.27
for i, (k, lab, c) in enumerate([("pvs_peak", "PVS (ours)", "C0"), ("werr_peak", "Warping error", "C1"),
                                  ("scalar_peak", "Scalar density", "C2")]):
    axs[0].bar(x + (i - 1) * w, [a1[k]["auc_all"]] + [a1[k]["auc_" + t] for t in types], w, label=lab, color=c)
axs[0].set_xticks(x); axs[0].set_xticklabels(["all"] + types); axs[0].set_ylim(0.4, 1.0); axs[0].tick_params(axis="x", pad=1)
axs[0].axhline(0.5, color="gray", lw=0.6, ls="--"); axs[0].set_ylabel("ROC AUC"); axs[0].set_title("(a) A1: 360 clean held-out clips")
axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
cols = [("all", "all"), ("jpeg", "JPEG"), ("vs_lighting_only", "lighting")]
modes = [("conditional", "PVS content", "C0"), ("combined", "PVS combined", "C4"), ("vector", "PVS vector", "C5"),
         ("scalar", "Scalar density", "C2"), ("warp_error", "Warping error", "C1")]
x = np.arange(len(cols)); w = 0.16
for i, (k, lab, c) in enumerate(modes):
    axs[1].bar(x + (i - 2) * w, [b2[k][cc] for cc, _ in cols], w, label=lab, color=c)
axs[1].set_xticks(x); axs[1].set_xticklabels([l for _, l in cols]); axs[1].set_ylim(0.0, 1.0)
axs[1].axhline(0.5, color="gray", lw=0.6, ls="--"); axs[1].set_title("(b) A1b: JPEG + lighting stress")
axs[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=5.6)
plt.tight_layout(pad=0.3); plt.savefig(f"{OUT}/fig_detection.pdf"); plt.close()

# --- Fig 3: speed vs fidelity ----------------------------------------------------------
r = json.load(open("results/tiny_exps.json")); b5 = json.load(open("results/b5_prepass.json"))
fig, ax = plt.subplots(figsize=(COL, 2.45))
fs = r["fewer_steps_baseline"]
pts = sorted([(v["speedup"], v["psnr"]) for k, v in fs.items() if k.isdigit()])
ax.plot(*zip(*pts), "k-o", ms=3, lw=1.4, label="Fewer steps (baseline)")
q = sorted([(e["speedup"], e["psnr"]) for e in r["B4"] if e["rule"] == "heat"]); ax.plot(*zip(*q), "s--", color="C3", ms=3, lw=0.9, label="Freezing (heat rule)")
q = sorted([(e["speedup"], e["psnr"]) for e in r["B2"] if e["kind"] == "density+motion"]); ax.plot(*zip(*q), "P:", color="C2", ms=4, lw=0.9, label="Oracle token caching")
q = sorted([(v["speedup"], v["psnr"]) for k, v in fs.items() if "heat" in k]); ax.plot(*zip(*q), "*-", color="C3", ms=6, lw=0.9, label="10 steps + freezing")
q = sorted([(e["speedup"], e["psnr"]) for e in r["B3"] if e["static_thr"] == 0.06]); ax.plot(*zip(*q), "v:", color="C4", ms=3, lw=0.9, alpha=0.6, label="Frame sharing, plain attn.$^\\dagger$")
q = sorted([(e["speedup"], e["prop_attn"]["psnr"]) for e in r["B3"] if e["static_thr"] == 0.06]); ax.plot(*zip(*q), "v-", color="C4", ms=3.5, lw=1.4, label="Frame sharing + prop. attn. (oracle)$^\\dagger$")
q = sorted([(v["speedup"], v["psnr_vs_same_noise_full"]) for v in b5.values() if v["draft_steps"] == 2]); ax.plot(*zip(*q), "D-", color="C1", ms=3, lw=1.4, label="Frame sharing + prop. attn. (draft)$^\\dagger$")
ax.set_xlabel("Speedup (measured FLOPs, tiny model)"); ax.set_ylabel("PSNR vs. full compute (dB)")
ax.set_ylim(15, 66); ax.grid(alpha=0.25, lw=0.4); ax.legend(frameon=False, loc="upper right", fontsize=5.4)
plt.tight_layout(pad=0.3); plt.savefig(f"{OUT}/fig_speed.pdf"); plt.close()

# --- Fig 4: real AI clips --------------------------------------------------------------
clips = [("ecm_0002_veo3_0", "Veo 3: stage lights / pyrotechnics"), ("ecm_0002_veo2_0", "Veo 2: cross-dissolve"),
         ("ecm_0005_pika2.2_0", "Pika 2.2: fast skateboard jump")]
if all(os.path.exists(f"/mnt/user-data/uploads/ecm/{n}.mp4") for n, _ in clips):
    fig, axs = plt.subplots(2, 3, figsize=(FULL, 2.75), gridspec_kw=dict(height_ratios=[1.35, 0.8]))
    for c, (n, lab) in enumerate(clips):
        f = load_video(f"/mnt/user-data/uploads/ecm/{n}.mp4", 256, 64, 2); s = score_video(f, prefilter=2.0)
        t = int(np.argmax(s.frame_scores))
        overlay(axs[0, c], f[t + 1], s.heatmaps[t]); axs[0, c].set_title(lab)
        axs[1, c].plot(s.frame_scores, "k-", lw=0.8); axs[1, c].axvline(t, color="C3", ls="--", lw=0.8)
        axs[1, c].set_xlabel("frame (stride 2)"); axs[1, c].set_title(f"per-frame PVS, peak {s.pvs_peak:.2f}")
    plt.tight_layout(pad=0.3); plt.savefig(f"{OUT}/fig_real_ai.pdf"); plt.close()
print("figures written to", OUT)
