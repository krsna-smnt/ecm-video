# Findings (toy scale, 10 Nov 2026)

All results come from synthetic video and a 1.1M-parameter video diffusion model trained on 2 CPU
cores. They test **mechanisms**. They do not give real-world numbers. Each result names the script
that produced it and its JSON file in `results/`. An independent code review found 4 bugs (B3
oracle map, B3 reference, unsaved baseline code, weak exit/enter clips), and all of the numbers
below were recomputed after those fixes.

## 1. Melting detection works, and beats the standard metric

| Benchmark | Ours (PVS) | Standard warping error | Scalar "entropy as mass" |
| --- | --- | --- | --- |
| A1, 360 clean held-out clips | **0.86** AUC (0.94 static camera, 0.79 panning) | 0.63 | 0.57 |
| A1b, 300 clips, half JPEG, legal lighting changes (blur 2) | **0.80** | 0.51 | 0.66 |
| A1b, JPEG clips only | **0.77** | 0.53 | 0.63 |
| A1b, lighting changes vs violations | **0.70** | 0.02 | 0.42 |

- **Localization:** 72% of the flagged mass lands inside the true event region, which covers 5%
  of the frame. The worst frame is the event frame in 95% of violations (boiling excluded).
- **False alarms:** at 80% detection, 22–28% of legal clips are flagged. This is still the
  main weakness, worst when the camera pans.
- Scripts: `experiments/t7_sanity.py`, `a1_benchmark.py`, `a1b_vector.py`, `dev_prefilter.py`.

**What made it work (each change was found by a failing test):**

1. **Conserve content, not amount.** The scalar version fails because a vanished object is
   replaced by background with about the same number of bits. The source term has to be the
   *new* information, i.e. the bits needed to explain frame t+1 given frame t moved along the
   flow.
2. **Find occlusion boundaries from the neighbouring frame pairs.** The motion estimator invents
   motion over a violation, which marks the violation itself as a legal occlusion. Without this
   change the vanish test was detected in 0 of 5 clips; with it, in 5 of 5.
3. **Blur before scoring (σ = 2).** JPEG blocks look like new information. On compressed clips
   the blur raises AUC from 0.63 to 0.77. The value was chosen on a separate dev set.
4. **Vector entropy** (a 9-channel fingerprint: tone plus oriented texture bands) beats scalar
   entropy (0.75 vs 0.66). It does *not* beat full content (0.80), and combining the two adds
   nothing. Learned features (level 3) are still untested.

## 2. Speed: the only big win is sharing static regions across frames

Speedup is in measured FLOPs vs 20-step full compute. Quality is PSNR against full compute.
For calibration, halving the steps (2x) gives 34.3 dB.

| Method | Speedup (toy) | Projected at real scale* | Quality |
| --- | --- | --- | --- |
| Fewer steps only (baseline) | 2.9x / 4x | same | 30.0 / 27.0 dB |
| Freezing settled regions, heat rule (B4) | 2.9x / 4.1x | same | 25.5 / 24.2 dB (worse than baseline) |
| Oracle caching by density + motion (B2) | 2.3x / 3.1x | same | 30.4 / 28.8 dB (worse or equal) |
| 10 steps + heat freezing | 3.1x / 4.0x | same | 30.5 / 28.5 dB (+1.2–1.5 dB over baseline) |
| Oracle frame sharing, plain attention (B3) | 2.7x | 5.3x | 20.2 dB |
| **Oracle frame sharing + proportional attention** | 2.7x | 5.3x | **43.6 dB** |
| **Draft-pass frame sharing, no oracle (B5b)** | 2.4x / 3.6x | **3.7x / 5.7x** | **43.8 / 41.5 dB** |
| Online sharing, switched on mid-generation (B5) | 1.2–1.4x | 1.4–1.6x | 24–30 dB (fails) |

\* Projection: the same token fractions applied to a ~75k-token, 40-layer, width-3072 video DiT,
where attention dominates. This is a FLOP calculation, not a measurement.

- **Frame sharing:** static background tokens are computed once and copied to the other frames.
  Two things make it nearly exact:
  1. **Proportional attention:** each shared token gets a log(number of copies) attention bias,
     so other tokens still give it the weight of all its copies.
  2. **Deciding the static map before generating,** from a 2–3 step draft pass, then sharing
     from step 0 with shared noise.
- **Caveat (the most important open question):** sharing noise produces a different sample
  than the original noise would. The melting score of those samples is higher (1.1–1.3 vs
  0.70). That could be real artifacts, for example an object cannot move into a region marked
  static. It could also be the score's unreliability at 32 px. This must be checked on a real
  model.
- **Settling (B1) is real:** boring regions finish early. The median settling step is 4 for
  static tokens and 7 for moving ones, and settling time tracks information density (Spearman
  0.70). But exploiting it by freezing loses to just using fewer steps.
- **Heat (B4, H):** the heat-rule freezer beats the plain-change rule at low speedups (28.6 vs
  26.8 dB at about 2.2x). Static tokens send less attention to moving tokens than uniform (0.27
  vs 0.38; 0.16 in the last layer), but cutting those links costs quality (27.9 dB).
- Scripts: `experiments/tiny_exps.py`, `b5_online.py`, `b5_prepass.py`. Figure:
  `results/fig_speed_quality.png`.

## 3. Negative or inconclusive

- **Early abort (C1):** the early prediction's melting score does not predict the final one
  (Spearman about 0).
- **Tiny generated vs real (A5t):** AUC 0.53. At 32 px the score has a false-alarm floor: small
  moving textures change appearance between sub-pixel positions (aliasing). Score real video at
  128 px or more.
- **The tiny model is weak:** samples show patch seams. The speed tests measure fidelity to full
  compute, which is valid regardless, but content-dependent results such as B1 are weaker
  evidence.

## What this means for the 10x goal

On this toy, the framework's own speed contribution is the frame-sharing result: about 3.7–5.7x
projected at real scale on static-camera content, nearly lossless relative to same-noise full
compute. Stacked with step distillation (about 10x, existing work), that is the path to the 10x+
goal. Whether shared-noise samples look as good as unshared ones is unresolved, and it is the
first thing to test on a GPU (see `GPU_PLAN.md`). Nothing else here showed a large speedup.
