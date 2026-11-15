# ECM-Video

Entropy-Continuum Mechanics for video: treat information as a material that moves with
the scene. Two uses were tested:

1. **Detection.** Flag places where information is created or destroyed without a physical reason
   (objects melting, vanishing, boiling, swapping identity).
2. **Speed.** Skip recomputing content that is conserved across frames when generating video.

Status: research code, with results on toy models (CPU) and on a real text-to-video model
(Wan 2.1 1.3B, on free Kaggle T4 GPUs). The headline results are mixed and summarised
honestly below. The full chronological record is in `RESEARCH_LOG.md`.

## Results at a glance

| Question | Toy scale | Real data / real model |
|---|---|---|
| Does the Permanence Violation Score (PVS) catch physics violations? | Yes: AUC 0.86 vs 0.63 for warping error on synthetic clips (0.80 vs 0.51 under legal stress) | **No.** On real AI clips it does not separate AI from real footage. On 93 human-judged pairs it disagrees with human "glitch" votes (14% agreement). |
| Does frame sharing ("share content, not noise") beat simply using fewer sampling steps? | **Yes.** On 8- and 16-frame toy video DiTs, at matched compute (2.4-4.2x), on two independent seed sets. | **No.** On Wan 2.1, Wan's own UniPC scheduler with fewer steps beats every sharing variant at every matched speed (3 runs, about 300 videos). |
| Why does sharing fail on Wan? | n/a | Partly measured: Wan's causal VAE does not encode static content as identical latent frames (18% frame-to-frame change). Forcing equality alone adds 5.6x flicker. The rest of the damage happens during sampling. |

## The idea in one equation

```
d(iota)/dt + div(iota * v) = s
```

`iota` is local information (bits) and `v` is motion. `s` is where information is created or
destroyed. In real footage `s` should be non-zero only at occlusion boundaries, frame edges and
genuine new events. The **Permanence Violation Score (PVS)** sums `|s|` everywhere else.

## Frame sharing (the speed idea)

Static content is computed once per group of frames instead of once per frame:
- each token keeps its own noise;
- the group's leader sees the average of its copies, re-noised back to the right noise level;
- a log(m) attention bias stands in for the m copies.

Sharing the noise as well hurts quality. Averaging is the part every ablation confirms is
necessary.

- Toy implementation: `ecm/tinydit.py`, `experiments/r3f_battery.py`
- Wan 2.1 port (diffusers, verified against the original forward pass on CPU): `wan/ecm_wan.py`
- Kaggle runner and notebooks: `wan/kaggle_run.py`, `wan/ecm_wan_kaggle*.ipynb`
- Results: `results/r3f_battery_*.json` (toy), `results/wan/` (Wan runs 2 and 3, including the VAE
  diagnostic)

## Score your own clips

Runs on any CPU (opencv + numpy, no model weights):

```bash
pip install numpy scipy opencv-python-headless scikit-learn scikit-image
python -m ecm.score real_clip.mp4 ai_clip.mp4 --heatmap out/
```

Prints the worst-frame score, mean score and worst frame per clip, and writes
`out/<name>_pvs.mp4` with violations overlaid in red. Note the real-clip results above: treat the
score as an experimental signal, not a detector.

## Layout

| Path | What |
| --- | --- |
| `ecm/density.py` | information density (LOCO-I/JPEG-LS MED predictor + adaptive Laplacian, bits/px) |
| `ecm/flow.py` | optical flow, warping, occlusion and motion-boundary masks |
| `ecm/pvs.py` | source term (scalar, vector, content, combined) and PVS |
| `ecm/vector.py` | 9-channel entropy fingerprint (tone + oriented texture bands) |
| `ecm/score.py` | command-line scorer for real video files |
| `ecm/tinydit.py` | tiny video diffusion transformer with token caching, removal and FLOP counting |
| `wan/` | frame sharing for Wan 2.1 (diffusers), CPU tests, Kaggle runner and notebooks |
| `experiments/` | benchmarks, tiny-model training, sharing batteries, real-clip and human-vote studies |
| `results/` | JSON results, logs, figures and sample GIFs |
| `paper/` | technical report (written before the Wan results; being revised) |

## Further reading

- `RESEARCH_LOG.md`: everything in order, including mistakes and corrections
- `FINDINGS.md`, `MORNING_SUMMARY.md`: earlier summaries (pre-Wan)
