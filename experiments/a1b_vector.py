"""A1b: scalar vs vector vs full-content vs combined source terms, with stressors.

Same generator as A1, new held-out seeds (from 70000). Legal events now include a
lighting change (brightness drift + moving soft shadow), and half of all clips are
JPEG-compressed (quality 35) to mimic real footage. Reports ROC AUC overall, per
violation type, and on the stressed subsets.
"""
import sys, json, time
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, ".")
from ecm.pvs import score_video
from experiments.synth import make_clip, LEGAL_STRESS, VIOLATIONS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
PF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0   # prefilter sigma (chosen on dev seeds 90000+)
MODES = ["scalar", "vector", "conditional", "combined"]
rows, seed, t0 = [], 70000, time.time()
for ev in LEGAL_STRESS + VIOLATIONS:
    for i in range(N):
        seed += 1
        rng = np.random.default_rng(seed)
        pan = bool(rng.random() < 0.5)
        jpeg = 35 if rng.random() < 0.5 else 0
        fade = int(rng.choice([1, 4])) if ev in ("vanish", "appear") else 1
        f, label, _ = make_clip(ev, rng, pan=pan, fade=fade, jpeg=jpeg)
        row = dict(event=ev, label=label, pan=pan, jpeg=jpeg)
        for m in MODES:
            r = score_video(f, mode=m, prefilter=PF)
            row[m] = r.pvs_peak
            if m == "conditional":
                row["warp_error"] = r.warp_err_peak
        rows.append(row)
    print(ev, f"{time.time() - t0:.0f}s", flush=True)

y = np.array([r["label"] for r in rows])
keys = MODES + ["warp_error"]
summ = {}
for k in keys:
    x = np.array([r[k] for r in rows])
    d = dict(all=roc_auc_score(y, x))
    for v in VIOLATIONS:
        m = np.array([(r["label"] == 0) or (r["event"] == v) for r in rows]); d[v] = roc_auc_score(y[m], x[m])
    for name, sel in [("jpeg", lambda r: r["jpeg"] > 0), ("clean_codec", lambda r: r["jpeg"] == 0),
                      ("pan", lambda r: r["pan"]), ("static_cam", lambda r: not r["pan"])]:
        m = np.array([sel(r) for r in rows]); d[name] = roc_auc_score(y[m], x[m])
    # lighting specifically: lighting clips vs violations
    m = np.array([(r["event"] == "light") or r["label"] == 1 for r in rows]); d["vs_lighting_only"] = roc_auc_score(y[m], x[m])
    # false-alarm rate per legal type at the threshold catching 80% of violations
    thr = np.percentile(x[y == 1], 20)
    d["fpr_at_tpr80"] = {e: float(np.mean([r[k] >= thr for r in rows if r["event"] == e])) for e in LEGAL_STRESS}
    summ[k] = d
print(json.dumps(summ, indent=1, default=float))
json.dump(dict(summary=summ, rows=rows), open(f"results/a1b_vector_pf{PF:g}.json", "w"), indent=1, default=float)
