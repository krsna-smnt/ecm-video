# Research log (overnight run, from 10 Nov 2026 20:50 IST)

Each entry records the question, what was run, the result, and the decision it led to.
Negative results and mistakes stay in the log on purpose.

---

## R1. Does PVS agree with human glitch judgments on real AI video?

**Data.** Rapidata text-to-video human-preference datasets (Apache-2.0), via vote tables from 6
datasets: 6,243 human-voted pairs over 2,588 clips. I selected 10 prompt groups that maximise
pairs with a strong consensus (vote margin at least 0.4), giving 93 pairs over 103 clips. The
clips were downloaded in one zip through the built-in browser. Generators: Veo 3, Seedance 1
Pro, Sora, Pika 1 and 2.2, Ray 2, HunyuanVideo, Wan 2.1, Veo 2, Runway Alpha.

**The vote question matters.** The Rapidata "Coherence" question is *"Which video has more
glitches and is more likely to be AI generated?"*, so a vote marks the WORSE clip. My first
analysis had this sign reversed and reported 86% agreement. I caught the error when Veo 3
appeared to "lose" on coherence 77% of the time, and confirmed the question wording in the
dataset card. All numbers below use the corrected sign.

**Result** (pairwise accuracy at picking the clip humans judged *less* glitchy, 93 strong
pairs, 95% bootstrap CI):

| Score | Accuracy |
| --- | --- |
| PVS, frame average | 0.14 [0.08, 0.22] |
| PVS + local lighting normalisation | 0.19 [0.12, 0.27] |
| PVS + lighting + cut skipping | 0.19 |
| PVS, worst frame | 0.28 |
| Warping error (lower = better) | **0.75** [0.67, 0.84] |
| Mean optical-flow magnitude (lower = better) | **0.78** [0.70, 0.86] |
| Bitrate (lower = better) | 0.29 (so higher bitrate = less glitchy, 0.71) |

**Confound.** Generator identity alone, i.e. each generator's average human rate, predicts
**97%** of these strong pairs. Almost every strong pair is a top model (Veo 3, Seedance) against
an older one. So every score above is, to first order, a test of whether it tells generators
apart. Only 3 strong pairs go against the generator prior.

**Observations.**
1. PVS is strongly *anti*-correlated with human glitch judgments. The clips humans find
   cleaner (mostly Veo 3 and Seedance) have *more* information created in open view.
   Plausible reason: modern generators produce rich dynamic content (lighting changes, camera
   moves, fine detail appearing), while older generators produce blurry, slowly smearing
   content. Blur destroys information *smoothly*, so per frame it creates small residuals and
   therefore a low PVS.
2. Humans flag high-motion clips as glitchy (motion 0.78). The "less motion = fewer glitches"
   prior is strong in this data.
3. Warping error works in the human direction (0.75), but it is also correlated with motion.

**Decision.** PVS as designed measures "unexplained new content", which on modern generators
is mostly legitimate richness, not glitches. Melting by blur, the dominant failure in older
models, is a *slow information loss*, not a burst of new content. The next experiment tests
whether information *decay* (the sign of the source term, and the trend of density over time)
tracks human glitch judgments better than |s|, and evaluates it **within generator** to avoid
the generator confound.

## R1c. Which information features track human glitch judgments, *within* a generator?

**Question.** If older generators' glitches are a slow information *loss* (blur, smear), do
decay-type features (density trend, signed source, share of destruction) beat PVS?

**Run.** 10 features per clip (`experiments/r1c_features.py`). Pooled pairwise accuracy, with
the direction fitted on the strong pairs, plus within-generator Spearman against each clip's
human coherence share (7 generators with at least 6 clips).

| Feature | Pooled accuracy | Within-generator mean rho |
| --- | --- | --- |
| PVS (higher = less glitchy) | 0.86 | +0.10 (6 of 7 positive) |
| motion (lower = less glitchy) | 0.78 | +0.03 |
| warp error (lower) | 0.75 | -0.10 |
| jitter (temporal 2nd/1st difference, lower) | 0.69 | -0.02 |
| signed source (lower) | 0.66 | -0.07 |
| density trend, sharpness, density, arrow-of-time, destruction share | 0.54-0.59 | -0.16 to +0.03 |

**Observation.** Pooled accuracies reflect generator identity (the generator prior alone is
0.97). Within generators every correlation is |rho| <= 0.16, which is indistinguishable from
zero with 6-19 clips each. PVS is the only feature with a consistent sign across generators
(6 of 7), but the effect is tiny. Decay features do not help.

**Decision.** This data cannot validate any per-clip glitch detector. Doing so needs
same-generator pairs (several samples per prompt per model) with glitch votes, or human
labels on our own clips. Both need downloads or people, so neither is possible overnight.
Park detection; switch to the speed direction (R3), which can be tested offline.

## R3. Does shared noise, which exact frame sharing requires, hurt sample quality?

**Run.** 400 samples per method from the tiny model (same noise seeds), compared with 400
real clips. Metrics: FD-rand (Frechet distance on features from 5 fixed random 3D-conv
nets; a toy stand-in for FVD; real-vs-real floor 5e-6), the share of static pixels, the mean
change of moving pixels, and jerk (temporal 2nd difference) of moving pixels. A classifier
two-sample test saturated (normal samples already 97% detectable, because the model is weak),
so FD-rand and the temporal statistics are the main metrics.

| Method | Speedup | FD-rand | Static share | Moving jerk |
| --- | --- | --- | --- | --- |
| real (vs real) | - | 0.000005 | 0.81 | 0.069 |
| normal 20 steps | 1x | 0.000056 | 0.55 | 0.112 |
| **shared noise** (B5b / frame sharing) | 2.4x | **0.00151** | 0.74 | **0.242** |
| shared noise + shared compute | 2.4x | 0.00157 | 0.74 | 0.245 |
| own noise, leader input = own state | 2.4x | 0.00157 | 0.74 | 0.245 |
| **averaged sharing** (own noise, leader input = mean of copies) | 2.4x | **0.00035** | 0.77 | 0.158 |
| averaged sharing, thr 0.06 | 3.7x | 0.00024 | 0.91 | 0.182 |
| normal 10 / 8 / 6 / 5 steps | 2 / 2.5 / 3.3 / 4x | 0.00019 / 0.00026 / 0.00035 / 0.00043 | ~0.59 | 0.10-0.07 |
| 10 steps + averaged sharing | 3.9x | 0.000044 | 0.77 | 0.127 |
| 10 steps + averaged sharing, thr 0.06 | 5.3x | 0.000038 | 0.91 | 0.153 |

**Observations.**
1. **Shared noise hurts badly:** 27x worse FD, 2x jerkier moving regions, and visible
   saturated blocks. The frame-sharing speed result (B5b) was exact *relative to same-noise
   full compute*, but same-noise full compute is itself a poor sample.
2. **Mechanism.** The model denoises static backgrounds by pooling independent noisy views
   across frames. Shared noise removes that redundancy, so noise is "agreed on" by every frame
   and baked in as content. Evidence: giving copies their own noise while computing the leader
   from its own state alone changes nothing (identical FD 0.00157); averaging the copies'
   states into the leader's input recovers most of the loss (0.00035).
3. **Averaged sharing** keeps the compute saving of frame sharing without the shared-noise
   failure. Fidelity to normal full compute is about 22-23 dB, i.e. a nearby sample, not an
   exact reproduction.
4. **Anomaly, not yet trusted.** 10 steps + averaged sharing scores *better* than 20-step full
   compute (FD 0.000044 vs 0.000056), consistently on all 5 feature seeds, while 20-step
   averaged sharing scores worse (0.00035). The likely explanation is that FD-rand rewards
   removing the weak model's background flicker (static share 0.55 vs 0.81 real). The
   non-monotonic step dependence is unexplained. Visually the three sets look alike.
   **Not a claim yet.**

**Decision.** The toy model is too weak for quality claims (its samples are 97% detectable).
Train it much longer, about 3.5x more steps, then re-run the quality battery (normal, fewer
steps, shared noise, averaged sharing) on the better model. If averaged sharing still beats
fewer steps at matched speedup on a model whose samples are closer to real, that is a real
result.

## R3f (12k-step model). Full quality battery: two metrics, and they disagree

300 samples per method, same seeds; FD-rand floor (real vs real) 5.6e-6; C2ST = held-out
accuracy of a small 3D-conv real-vs-generated classifier (0.5 = indistinguishable).

| Method | Speedup | FD-rand | C2ST | Static share | Jerk |
| --- | --- | --- | --- | --- | --- |
| normal 20 steps | 1.0x | 0.000062 | 0.94 | 0.55 | 0.111 |
| normal 10 / 8 / 6 / 5 | 2 / 2.5 / 3.3 / 4x | 0.00019 / 0.00027 / 0.00036 / 0.00043 | 0.82 / 0.74 / 0.75 / 0.81 | ~0.59 | 0.10-0.08 |
| shared noise (exact sharing) | 2.4x | 0.00153 | **0.99** | 0.74 | 0.244 |
| averaged sharing 20 steps, thr 0.03 / 0.06 | 2.4 / 3.5x | 0.00036 / 0.00023 | 0.89 / 0.89 | 0.76 / 0.91 | 0.16 / 0.18 |
| averaged + re-noised 20 steps | 2.4x | 0.00033 | 0.94 | 0.75 | 0.169 |
| averaged sharing 10 steps, thr 0.03 / 0.06 | 3.8 / 5.2x | 0.000053 / 0.000045 | 0.84 / 0.79 | 0.76 / 0.91 | 0.13 / 0.15 |
| averaged + re-noised 10 steps | 3.8x | 0.00030 | 0.90 | 0.75 | 0.144 |

**Observations.**
1. FD-rand and C2ST *disagree*. FD ranks 20-step full compute near the best; C2ST finds it
   the most detectable and fewer-step (blurrier) samples less detectable. The real training
   clips are smooth and the model's samples are blocky, so on a model this weak each metric
   largely measures how much of the model's own artifact survives.
2. The only conclusion both metrics support: **shared noise is the worst method** (FD 25x
   worse than normal, C2ST 0.99).
3. Re-noising the averaged leader input, to restore the per-token noise level, does not help
   and is worse at 10 steps.
4. 10 steps + averaged sharing is good on both metrics (FD equal to 20-step normal; C2ST 0.79-0.84,
   better than 20-step normal), at 3.8-5.2x fewer FLOPs. Still not trusted until the
   better-trained model reproduces it.

## Toy-world physics checker, and R3f rerun with it (12k model)

**Why.** FD-rand and C2ST disagreed, and both mostly measure the weak model's appearance
artifacts. The toy world has known laws (static background; rigid sprites with constant size,
texture and velocity), so violations can be measured directly (`experiments/toy_physics.py`):
background flicker, object-area drift (melting or vanishing), velocity jerk, appearance drift,
and the share of clips whose object is tracked in every frame. Real clips are not exactly zero
(overlapping or low-contrast sprites, edge entries), so the checker is read relative to real.

| Method | Speedup | Flicker | Area drift | Velocity jerk (px) | Valid | FD | C2ST |
| --- | --- | --- | --- | --- | --- | --- | --- |
| real | - | 0.002 | 0.55 | 0.70 | 0.66 | - | - |
| normal 20 | 1x | 0.012 | 0.97 | 3.01 | 0.19 | 0.000062 | 0.94 |
| normal 10 | 2x | 0.012 | 1.14 | 2.80 | 0.10 | 0.00019 | 0.82 |
| normal 5 | 4x | 0.012 | 1.49 | 2.48 | 0.03 | 0.00043 | 0.81 |
| shared noise 20 | 2.4x | 0.006 | **0.75** | 2.80 | **0.44** | 0.00153 | 0.99 |
| averaged sharing 20, thr 0.03 | 2.4x | 0.006 | 1.05 | 2.63 | 0.18 | 0.00036 | 0.89 |
| averaged sharing 20, thr 0.06 | 3.5x | 0.002 | 1.39 | 1.87 | 0.07 | 0.00023 | 0.89 |
| **averaged sharing 10, thr 0.03** | **3.8x** | 0.006 | **1.21** | 2.59 | **0.09** | **0.000053** | 0.84 |
| averaged sharing 10, thr 0.06 | 5.2x | 0.002 | 1.52 | 1.88 | 0.04 | 0.000045 | 0.79 |

**Observations.**
1. The physics checker is monotone in sampling steps (valid 0.19, 0.10, 0.07, 0.05, 0.03 from
   20 down to 5 steps), so it behaves like a quality measure, unlike C2ST.
2. **Shared noise is the most physically consistent method** (flicker halved, area drift lowest,
   objects tracked 2.3x more often) while being the worst on appearance (FD, C2ST).
   "Does shared noise hurt quality?" has a split answer: it helps physics and hurts appearance
   (saturated, high-contrast blocks).
3. At **matched speed (about 4x)**, 10 steps + averaged sharing beats 5-step sampling on physics
   (valid 0.09 vs 0.03, area drift 1.21 vs 1.49) and FD (8x lower), and is about level on C2ST.
   This is the first consistent sign of a real speed-quality gain from the framework.
4. Sharing too much (thr 0.06) freezes regions that objects later move into: lower flicker,
   but worse area drift and validity. The static-map threshold trades flicker against object
   permanence.

**Next.** Confirm or refute observation 3 on the 42k-step model.

## R3f on the 42k-step model: re-noised averaged frame sharing beats fewer steps

**Model.** The same architecture trained 42k steps instead of 12k. Its samples are much harder
to detect (C2ST 0.66 vs 0.94 at 20 steps), and the metrics now mostly agree: fewer steps get
monotonically worse on FD, C2ST and physics.

**Two changes compared with the 12k results.**
1. The better model's 2-step draft is cleaner, so at thr 0.03 it marks 90-92% of positions
   static (real data: 81%). That over-shares and freezes regions objects move into. A
   threshold sweep (0.01-0.03) shows thr ~0.02 gives a shared share of 0.80, matching real.
2. **Re-noising the averaged leader input now helps** (FD 0.00021 vs 0.00080 for plain
   averaging at thr 0.03). Re-noising restores the per-token noise level the model was trained
   on. It did not help on the 12k model; plausibly it matters once the model is accurate
   enough for the noise-level mismatch to dominate.

**Key comparison** (300 samples each; two independent seed sets: noise 30000+ / real 4321, and
noise 70000+ / real 2468):

| Method | Speedup | FD (set 1 / 2) | C2ST (1 / 2) | Valid (1 / 2) | Area drift (1 / 2) |
| --- | --- | --- | --- | --- | --- |
| normal 20 steps | 1.0x | 0.000077 / 0.000152 | 0.66 / 0.66 | 0.19 / 0.20 | 0.97 / 0.99 |
| **re-noised sharing, 20 steps, thr 0.02** | **2.4x** | 0.000123 / 0.000181 | **0.67 / 0.69** | **0.18 / 0.15** | 1.03 / 1.03 |
| normal 8 steps | 2.5x | 0.000343 / 0.000475 | 0.78 / 0.83 | 0.07 / 0.10 | 1.15 / 1.24 |
| **re-noised sharing, 10 steps, thr 0.015** | **3.0x** | 0.000198 / 0.000358 | **0.64 / 0.74** | 0.12 / 0.11 | 1.06 / 1.14 |
| normal 6 steps | 3.3x | 0.000467 / 0.000613 | 0.87 / 0.88 | 0.04 / 0.07 | 1.29 / 1.33 |
| **re-noised sharing, 10 steps, thr 0.02** | **3.9x** | 0.000196 / 0.000292 | 0.67 / 0.77 | 0.08 / 0.11 | 1.14 / 1.17 |
| normal 5 steps | 4.0x | 0.000554 / 0.000708 | 0.89 / 0.91 | 0.02 / 0.04 | 1.39 / 1.40 |
| shared noise (exact sharing) | 3.4x | 0.00227 / 0.00237 | 0.98 / 0.98 | 0.19 / 0.22 | 0.98 / 1.01 |

**Result.** At every matched speedup, re-noised averaged frame sharing beats taking fewer
steps on all four metrics, on both seed sets. At 2.4x it is close to full compute (C2ST within
0.01-0.04, validity within 0.01-0.05). Figure: `results/fig_r3_renoise_vs_steps.png`.

**Projected to a 75.6k-token, 40-layer video DiT** (same token fractions; FLOP calculation,
not a measurement): 4.3x at near-full quality (20 steps, thr 0.02), and 5.5-6.0x with modest
loss (10 steps). These stack with step distillation in principle, not yet tested.

**The method, as it now stands** ("re-noised averaged frame sharing"):
1. Run a 2-step draft; positions whose maximum frame-to-frame change is below thr are static.
2. Every token keeps its own independent noise.
3. Each static position is computed once per step, from the mean of its 8 copies' states,
   re-noised with fresh noise of std (1-t)*sqrt(1-1/m) so the input has the noise level its
   timestep implies. Proportional attention: a log(m) bias on that key.
4. The leader's clean prediction is shared, and each copy steps along its own trajectory:
   eps_i = (x_i - t*x1hat)/(1 - t), x_i <- x_i + (x1hat - eps_i)/N. Unshared tokens take
   exactly the ordinary Euler step (verified to 1e-6).

**Why it works (ECM reading).** A static region's information is conserved across frames, so
it only needs to be computed once. But its *noise* is not information, and the model needs
independent noise per frame in order to denoise. Averaging the copies' states pools that
redundancy (what the model does internally), and re-noising keeps the input in-distribution.
"Share the conserved content, not the noise" is the lesson.

**Caveats.** Tiny model (1.1M parameters), synthetic static-camera toy world, 32x32, 8 frames;
FLOP speedups measured, real-scale speedups projected; thresholds swept on seed set 1 and
then confirmed on seed set 2 (the confirmation used the thresholds picked on set 1).
