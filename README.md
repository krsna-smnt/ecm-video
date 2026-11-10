# ECM-Video

Entropy-Continuum Mechanics for video: treat information as a material that moves with
the scene, and flag places where it is created or destroyed without a physical reason
(objects melting, vanishing, boiling, swapping identity). Plus toy tests of the
compute-saving ideas the framework suggests for video generation.

Status: **toy-scale research code**. Everything here was run on 2 CPU cores with
synthetic video and a 0.6M-parameter video model. Mechanisms are tested; real-world
numbers still need real footage, real generators and a GPU (see `GPU_PLAN.md`).

## The idea in one equation

```
d(iota)/dt + div(iota * v) = s
```

`iota` is local information (bits), `v` is motion, `s` is where information is created
or destroyed. In real footage `s` is non-zero only at occlusion boundaries, frame edges
and genuine new events. The **Permanence Violation Score (PVS)** sums `|s|` everywhere
else.

## Score your own clips

Runs on any CPU (opencv + numpy, no model weights):

```bash
pip install numpy scipy opencv-python-headless scikit-learn scikit-image
python -m ecm.score real_clip.mp4 ai_clip.mp4 --heatmap out/
```

Prints the worst-frame score, mean score and worst frame per clip, and writes
`out/<name>_pvs.mp4` with violations overlaid in red. Compare clips from the same source
(resolution, frame rate, compression all shift the absolute value).

## Layout

| Path | What |
| --- | --- |
| `ecm/density.py` | T1 information density (LOCO-I/JPEG-LS MED predictor + adaptive Laplacian, bits/px) |
| `ecm/flow.py` | T2 optical flow, warping, occlusion and motion-boundary masks |
| `ecm/pvs.py` | T3/T4 source term (scalar, vector, content, combined) and PVS |
| `ecm/vector.py` | 9-channel entropy fingerprint (tone + oriented texture bands) |
| `ecm/score.py` | command-line scorer for real video files |
| `ecm/tinydit.py` | tiny video diffusion transformer with token caching / removal and FLOP counting |
| `experiments/` | sanity tests (T7), toy A1/A1b benchmarks, tiny-model training and speed experiments |
| `results/` | JSON results and figures |

## Results

See `FINDINGS.md`.
