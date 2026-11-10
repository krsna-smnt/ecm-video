"""R1: does the Permanence Violation Score agree with human coherence judgments?

NOTE: the Rapidata "Coherence" question is "Which video has more glitches and is more likely
to be AI generated?", so a vote marks the WORSE clip. (A first version of this script had
the sign reversed and reported 86%; the corrected value is 1 - 0.86.)

Data: 93 pairs of AI-generated clips (same prompt, different generators) from the Rapidata
text-to-video preference datasets (Apache-2.0), each with a weighted human vote on
"Coherence" and a margin >= 0.4. For every clip we compute several scores; for a pair,
a score "predicts" the human choice when the clip humans judged more coherent has the
LOWER score (fewer violations). Pairwise accuracy 0.5 = chance.

Controls: warping error, mean optical-flow magnitude (does the score just measure
motion?), and file bitrate. 95% bootstrap CIs over pairs.

usage: python3 experiments/r1_human_votes.py  (features cached in results/r1_features.json)
"""
import sys, os, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from ecm.pvs import score_video, load_video
from ecm.flow import farneback

CLIPS = "/tmp/r1clips"
CACHE = "results/r1_features.json"
VARIANTS = {
    "base": dict(prefilter=2.0),
    "light": dict(prefilter=2.0, light_norm=True),
    "light_cuts": dict(prefilter=2.0, light_norm=True, skip_cuts=True),
}

feats = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
names = sorted(os.listdir(CLIPS))
t0 = time.time()
for i, n in enumerate(names):
    if n in feats and all(k in feats[n] for k in VARIANTS):
        continue
    f = load_video(os.path.join(CLIPS, n), max_side=256, max_frames=64, stride=2)
    d = feats.get(n, {})
    for k, kw in VARIANTS.items():
        if k in d:
            continue
        r = score_video(f, **kw)
        fs = r.frame_scores
        d[k] = dict(peak=float(fs.max()), mean=float(fs.mean()), p90=float(np.percentile(fs, 90)),
                    werr=float(r.warp_err_mean))
    if "motion" not in d:
        mags = [np.sqrt((farneback(f[t], f[t + 1]) ** 2).sum(-1)).mean() for t in range(0, len(f) - 1, 4)]
        d["motion"] = float(np.mean(mags))
        d["bitrate"] = float(os.path.getsize(os.path.join(CLIPS, n)) / max(len(f), 1))
    feats[n] = d
    if i % 10 == 0:
        json.dump(feats, open(CACHE, "w"))
        print(i, n, f"{time.time() - t0:.0f}s", flush=True)
json.dump(feats, open(CACHE, "w"))

pairs = pd.read_parquet("results/r1_pairs.parquet")
key = lambda src, fn: f"{src}__{fn}"
rows = []
for _, p in pairs.iterrows():
    a, b = key(p.src, p.file_name1), key(p.src, p.file_name2)
    if a in feats and b in feats:
        rows.append((a, b, float(p.margin)))       # margin > 0: MORE humans said clip a has more glitches (worse)
print("pairs usable:", len(rows))


def acc_of(fn):
    """Pairwise accuracy: humans' more-coherent clip has the LOWER value of fn."""
    hits = []
    for a, b, m in rows:
        va, vb = fn(feats[a]), fn(feats[b])
        if va == vb:
            hits.append(0.5); continue
        better_is_a = m < 0          # the vote question is "which has MORE glitches"
        hits.append(float((va < vb) == better_is_a))
    hits = np.array(hits)
    rng = np.random.default_rng(0)
    boots = [hits[rng.integers(0, len(hits), len(hits))].mean() for _ in range(2000)]
    return float(hits.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


res = {"n_pairs": len(rows)}
for k in VARIANTS:
    for agg in ["peak", "mean", "p90"]:
        res[f"pvs_{k}_{agg}"] = acc_of(lambda d, k=k, agg=agg: d[k][agg])
res["warp_error"] = acc_of(lambda d: d["base"]["werr"])
res["motion (less = better)"] = acc_of(lambda d: d["motion"])
res["bitrate (less = better)"] = acc_of(lambda d: d["bitrate"])
# motion-normalised PVS: violations per unit of motion
res["pvs_light_cuts_mean_per_motion"] = acc_of(lambda d: d["light_cuts"]["mean"] / (d["motion"] + 1e-3))
for k, v in res.items():
    print(f"{k:40s} {v}")
json.dump(res, open("results/r1_human_votes.json", "w"), indent=1)
