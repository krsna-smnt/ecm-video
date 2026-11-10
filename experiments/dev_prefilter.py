"""Dev-set tuning of the compression prefilter (seeds 90000+, all JPEG q35). Used ONLY to pick prefilter=2.0;
the held-out A1b set (seeds 70001+) was then scored once with it. Originally run with pf in [0,1,1.5] for all
three modes, then [2.0, 2.5] for combined: results 0.612/0.63/0.64 (pf0), 0.693/0.697/0.726 (pf1),
0.742/0.756/0.748 (pf1.5) for conditional/combined/vector; combined 0.783 (pf2), 0.798 (pf2.5)."""
import sys; sys.path.insert(0,'.')
import numpy as np
from sklearn.metrics import roc_auc_score
from ecm.pvs import score_video
from experiments.synth import make_clip, LEGAL_STRESS, VIOLATIONS
# DEV set (seeds 90000+), all JPEG q35, used only to choose the prefilter
clips=[]; seed=90000
for ev in LEGAL_STRESS+VIOLATIONS:
    for i in range(8):
        seed+=1; rng=np.random.default_rng(seed)
        f,l,_=make_clip(ev,rng,pan=bool(rng.random()<0.5),jpeg=35); clips.append((f,l))
y=np.array([l for _,l in clips])
for pf in [0.0, 1.0, 1.5, 2.0, 2.5]:
    for mode in ['conditional', 'combined', 'vector']:
        x=[score_video(f,mode=mode,prefilter=pf).pvs_peak for f,_ in clips]
        print(pf,mode,round(roc_auc_score(y,x),3),flush=True)
