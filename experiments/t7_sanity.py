"""T7 sanity tests: three clips with known answers.

1. clean   - sprite moves over texture            -> PVS near zero everywhere
2. vanish  - sprite disappears in open view at f8 -> spike at the 7->8 transition
3. occlude - sprite passes behind an occluder     -> no spike

Pass rules (fixed before running):
  * vanish peak >= 5x clean peak, and the vanish peak sits at transition 7 (frames 7->8)
  * occlude peak <= 2x clean peak
Run over several seeds; all must pass.
"""
import sys, json
import numpy as np
sys.path.insert(0, ".")
from ecm.pvs import score_video
from experiments.synth import make_clip

results = []
for seed in range(5):
    out = {}
    for ev in ["none", "vanish", "occlude"]:
        rng = np.random.default_rng(1000 + seed)   # same scene across the three
        f, _, _ = make_clip(ev, rng)
        r = score_video(f)
        out[ev] = dict(peak=r.pvs_peak, argmax=int(np.argmax(r.frame_scores)),
                       frames=np.round(r.frame_scores, 4).tolist(),
                       werr_peak=r.warp_err_peak)
    c, v, o = out["none"]["peak"], out["vanish"]["peak"], out["occlude"]["peak"]
    ok = (v >= 5 * c) and (out["vanish"]["argmax"] == 7) and (o <= 2 * c)
    results.append(dict(seed=seed, clean=c, vanish=v, vanish_at=out["vanish"]["argmax"],
                        occlude=o, ratio_vanish=v / max(c, 1e-9), ratio_occl=o / max(c, 1e-9), passed=bool(ok),
                        werr=dict(clean=out["none"]["werr_peak"], vanish=out["vanish"]["werr_peak"],
                                  occlude=out["occlude"]["werr_peak"])))
    print(json.dumps(results[-1], default=float))
print("ALL PASS" if all(r["passed"] for r in results) else "SOME FAIL")
json.dump(results, open("results/t7_sanity.json", "w"), indent=1, default=float)
