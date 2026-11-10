"""Toy A1: does the Permanence Violation Score separate violations from legal events?

Held-out seeds (the sanity seeds 1000-1004 were used for tuning; these start at 50000).
Each clip: random natural texture background (50% with camera pan), one textured sprite,
one event. Legal: none, occlude, exit, enter. Violations: vanish, appear (abrupt or
4-frame fade), melt, swap, boil.

Reports ROC AUC (violation vs legal) for:
  PVS (conditional, default), PVS (scalar density, the original formulation),
  and the standard warping-error baseline.
Plus per-type AUCs, frame localisation and heatmap localisation.
"""
import sys, json, time
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, ".")
from ecm.pvs import score_video
from experiments.synth import make_clip, LEGAL, VIOLATIONS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
rows = []
t0 = time.time()
seed = 50000
for ev in LEGAL + VIOLATIONS:
    for i in range(N):
        seed += 1
        rng = np.random.default_rng(seed)
        pan = bool(rng.random() < 0.5)
        fade = int(rng.choice([1, 4])) if ev in ("vanish", "appear") else 1
        f, label, em = make_clip(ev, rng, pan=pan, fade=fade)
        rc = score_video(f)
        rs = score_video(f, mode="scalar")
        row = dict(event=ev, label=label, pan=pan, fade=fade, seed=seed,
                   pvs_peak=rc.pvs_peak, pvs_mean=rc.pvs_mean,
                   scalar_peak=rs.pvs_peak, scalar_mean=rs.pvs_mean,
                   werr_peak=rc.warp_err_peak, werr_mean=rc.warp_err_mean)
        if label:
            ev_frames = np.where(em[:-1].any((1, 2)) | em[1:].any((1, 2)))[0]
            am = int(np.argmax(rc.frame_scores))
            row["frame_hit"] = bool(len(ev_frames) and ev_frames.min() - 1 <= am <= ev_frames.max())
            # heatmap localisation: share of violation mass inside the (dilated) event region
            import cv2
            reg = (em[am] | em[am + 1]).astype(np.uint8)
            reg = cv2.dilate(reg, np.ones((9, 9), np.uint8)) > 0
            hm = rc.heatmaps[am]
            row["loc_frac"] = float(hm[reg].sum() / (hm.sum() + 1e-9))
            row["region_frac"] = float(reg.mean())
        rows.append(row)
print(f"{len(rows)} clips in {time.time() - t0:.0f}s")

y = np.array([r["label"] for r in rows])
summary = {}
for key in ["pvs_peak", "pvs_mean", "scalar_peak", "scalar_mean", "werr_peak", "werr_mean"]:
    x = np.array([r[key] for r in rows])
    summary[key] = dict(auc_all=roc_auc_score(y, x))
    for v in VIOLATIONS:
        m = np.array([(r["label"] == 0) or (r["event"] == v) for r in rows])
        summary[key][f"auc_{v}"] = roc_auc_score(y[m], x[m])
    for pan in (False, True):
        m = np.array([r["pan"] == pan for r in rows])
        summary[key][f"auc_pan{int(pan)}"] = roc_auc_score(y[m], x[m])

# false positives: at the threshold that catches 80% of violations, how often is each legal type flagged?
x = np.array([r["pvs_peak"] for r in rows])
thr = np.percentile(x[y == 1], 20)
summary["fpr_at_tpr80"] = {e: float(np.mean([r["pvs_peak"] >= thr for r in rows if r["event"] == e])) for e in LEGAL}
vrows = [r for r in rows if r["label"]]
summary["frame_hit_rate"] = float(np.mean([r["frame_hit"] for r in vrows]))
# boil's event window covers most of the clip, so its hits are nearly automatic (review note)
summary["frame_hit_rate_excl_boil"] = float(np.mean([r["frame_hit"] for r in vrows if r["event"] != "boil"]))
summary["loc_frac_mean"] = float(np.mean([r["loc_frac"] for r in vrows]))
summary["region_frac_mean"] = float(np.mean([r["region_frac"] for r in vrows]))
summary["loc_by_type"] = {v: float(np.mean([r["loc_frac"] for r in vrows if r["event"] == v])) for v in VIOLATIONS}
summary["n_clips"] = len(rows)

print(json.dumps(summary, indent=1, default=float))
json.dump(dict(summary=summary, rows=rows), open("results/a1_benchmark.json", "w"), indent=1, default=float)
