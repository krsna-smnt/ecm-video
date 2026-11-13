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

## R3g. Ablations of re-noised averaged frame sharing (42k model, 2.4x, thr 0.02, seed set 1)

| Variant | FD | C2ST | Valid | Area drift |
| --- | --- | --- | --- | --- |
| full method | 0.000123 | 0.67 | 0.18 | 1.03 |
| no re-noising (plain averaging) | 0.000841 | 0.69 | 0.15 | 1.19 |
| no proportional attention (no log m bias) | 0.000070 | 0.79 | 0.06 | 1.16 |
| no averaging (re-noise the leader's own state) | 0.00153 | 0.96 | 0.27 | 0.91 |

**Observations.**
- Each component protects a different aspect of quality, and only the full method is good on
  all four metrics.
- Without averaging, the result is the shared-noise failure (C2ST 0.96). The frozen
  background still makes objects easier to track (valid 0.27): the appearance/physics split
  again.
- Proportional attention is what keeps object permanence (valid 0.06 without it): with one
  token standing for 8 copies, moving tokens otherwise under-attend to the background.
- Re-noising mostly fixes FD and area drift.
- FD alone would prefer "no proportional attention" (0.000070, even below normal 20-step).
  FD-rand is not trustworthy alone; decisions should rest on agreement across metrics.

An independent automated review of the R3 code has been launched; results go below.

## R4. Does the token saving show up in wall-clock time? (CPU, random weights)

One forward pass, all tokens vs the 30% kept by re-noised sharing (thr 0.02), including the
averaging gather and the scatter back (`experiments/r4_wallclock.py`):

| Tokens | Width x depth | Full | Shared (30% kept) | Measured | FLOP ratio |
| --- | --- | --- | --- | --- | --- |
| 128 | 128 x 4 | 3.1 ms | 2.5 ms | 1.26x | 3.74x |
| 512 | 256 x 6 | 52 ms | 15 ms | 3.58x | 4.05x |
| 1,024 | 256 x 6 | 81 ms | 26 ms | 3.10x | 4.63x |
| 4,096 | 256 x 4 | 471 ms | 75 ms | **6.30x** | 6.78x |

**Observation.** At the toy model's size (128 tokens), overhead dominates and little of the
saving appears. From about 500 tokens up, the measured speedup tracks the FLOP ratio (93% of
it at 4k tokens). This is CPU only; GPU attention kernels have different overheads, but the
direction supports the FLOP projection.

## Independent review of the R3 code, and the corrected numbers

An automated reviewer re-derived the method and checked it numerically:
- run_kept and the kept-sequence forward match the full forward exactly.
- A log(m) key bias equals duplicating the key m times (difference 4e-6).
- avg_sample equals the ordinary sampler when nothing is shared (to within 1e-6).
- The re-noise variance is correct, and there is no t = 1 division.
- The draft cost is counted, and every method is costed the same way.
- C2ST has a clean train/test split, and FD's sqrtm is real.
- One reported number was reproduced exactly.

**One real issue, now fixed.** The re-noising draws used seed 0 for every clip, so all clips
shared the same fresh noise at each static position. They now use a per-clip seed. The old
numbers are kept in the JSONs under `superseded_constant_renoise_seed`.

**Corrected key comparison** (per-clip seeds; all seven metrics; set 1 / set 2):

| Method | Speedup | FD | C2ST | Valid | Area drift | Flicker | Velocity jerk | Appearance drift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| normal 20 | 1x | 0.000077 / 0.000152 | 0.66 / 0.66 | 0.19 / 0.20 | 0.97 / 0.99 | 0.010 / 0.009 | 1.74 / 1.68 | 0.40 / 0.40 |
| **re-noised sharing 20, thr 0.02** | **2.4x** | 0.000084 / 0.000040 | 0.68 / 0.65 | 0.16 / 0.17 | 1.02 / 1.05 | 0.006 / 0.006 | 1.56 / 1.50 | 0.40 / 0.38 |
| normal 8 | 2.5x | 0.000343 / 0.000475 | 0.78 / 0.83 | 0.07 / 0.10 | 1.15 / 1.24 | 0.010 / 0.010 | 1.38 / 1.32 | 0.38 / 0.35 |
| **re-noised sharing 10, thr 0.015** | **3.0x** | 0.000046 / 0.000086 | 0.66 / 0.58 | 0.12 / 0.10 | 1.11 / 1.18 | 0.007 / 0.007 | 1.61 / 1.51 | 0.38 / 0.37 |
| normal 6 | 3.3x | 0.000467 / 0.000613 | 0.87 / 0.88 | 0.04 / 0.07 | 1.29 / 1.33 | 0.010 / 0.010 | 1.22 / 1.18 | 0.36 / 0.35 |
| **re-noised sharing 10, thr 0.02** | **3.9x** | 0.000048 / 0.000041 | 0.68 / 0.67 | 0.08 / 0.10 | 1.16 / 1.18 | 0.006 / 0.006 | 1.42 / 1.37 | 0.38 / 0.38 |
| normal 5 | 4x | 0.000554 / 0.000708 | 0.89 / 0.91 | 0.02 / 0.04 | 1.39 / 1.40 | 0.010 / 0.010 | 1.06 / 1.11 | 0.33 / 0.35 |

**Honest reading, with the reviewer's caveats.**
1. Against the fewer-steps baseline at matched speed, re-noised sharing wins on FD, C2ST,
   validity and area drift on both seed sets, by margins larger than the metric noise.
2. It *loses* on velocity jerk and appearance drift. Both of those also improve as steps
   decrease (normal 5 is best), because blurrier samples change less, so they do not measure
   quality here. They are reported rather than hidden.
3. The physics checker partly favours frozen backgrounds, i.e. the family under test: flicker
   rewards freezing by construction, and shared noise also scores well on validity. That is why
   FD and C2ST are needed as appearance checks. The full method is the only one good on both.
4. Metric noise: FD for normal 20 differs 2x between seed sets; C2ST is about ±0.035.
   "Matches full compute at 2.4x" is within that noise. The gains over fewer steps are not.
5. FD-rand is close to one-dimensional (top covariance eigenvalue about 80% of the trace),
   which is why it can prefer clearly worse variants (no proportional attention: FD 0.000012,
   C2ST 0.93). Decisions rest on agreement across metrics, never on FD alone.
6. Thresholds were chosen on set 1; set 2 (different noise and a different real set) is a
   genuine holdout.

## R5. Camera motion: frame sharing freezes pans unless it is gated

**Setup.** The 42k model was fine-tuned for 8k steps on a 50/50 mix of static-camera and
panning clips (0.4-1.2 px/frame over smooth backgrounds). The real set has the same mix. The
toy-physics checker assumes a static background, so R5 uses FD, C2ST and global camera motion
(phase correlation, px/frame).

**Failure found.** Ungated re-noised sharing still beats fewer steps on FD and C2ST (2.8x: FD
0.00011, C2ST 0.71, vs normal 8 at 2.5x: 0.00044, 0.79), but it **halves camera motion** on
clear pans (0.60 to 0.31 px/frame). Slow pans over smooth backgrounds pass the per-patch static
test, so the moving background is frozen. Adaptivity is weak: the Spearman correlation of
motion vs shared fraction is -0.21.

**Fix: a camera-motion gate.** Estimate global motion in the 2-step draft and disable sharing
for the clip if the camera moves.
- A consecutive-frame gate fires on almost every clip, because 2-step drafts are too noisy
  for frame-to-frame phase correlation (20 steps fall to 1.03x).
- A long-baseline gate (frame 0 to frame 7 displacement / 7, threshold 0.15 px/frame) preserves
  pans (0.57 vs 0.60) and keeps some sharing:

| Pan model, mixed real set | Speedup | FD | C2ST | Pans preserved |
| --- | --- | --- | --- | --- |
| normal 20 | 1.0x | 0.000131 | 0.66 | yes |
| normal 10 | 2.0x | 0.000340 | 0.75 | yes |
| normal 8 | 2.5x | 0.000442 | 0.79 | yes |
| 10 steps + gated sharing, thr 0.02 | 2.1x | 0.000136 | 0.61 | yes (0.57/0.60) |
| 10 steps + ungated sharing, thr 0.02 | 4.3x | 0.000082 | 0.74 | **no** (motion halved) |
| 20 steps + gated sharing | 1.2x | 0.000034 | 0.57 | yes |

**Observations.**
1. The method's speedup is content-dependent, as it should be. On static-camera content it is
   2.4-3.9x measured (4.3-6x projected); with a moving camera, honest gating leaves about 2x
   (mostly from the fewer steps it stacks with).
2. Even gated, sharing plus 10 steps beats plain 10 steps at the same speed (FD 2.5x better,
   C2ST 0.61 vs 0.75).
3. The proper fix is Lagrangian sharing: copy tokens *along* the estimated camera motion
   instead of freezing them. That is the ECM idea of advection made literal. Not attempted
   tonight, because patch-grid shifts need either integer-patch motion or interpolation in
   token space.

---
## 11 Nov: the real-model test (Wan 2.1 T2V 1.3B on free Kaggle GPUs)

No GPU budget, so the decisive test runs on Kaggle's free tier (2x T4, 30 h/week).

**Port** (`wan/ecm_wan.py`): a kept-token forward through diffusers' `WanTransformer3DModel`
(patch embedding as a linear map on tokens, gathered RoPE, per-key log(m) bias), plus the
re-noised averaged sharing sampler in the flow-matching convention (x = (1-s) x0 + s eps,
v = eps - x0). CPU checks against diffusers with random weights (`wan/test_ecm_wan.py`):
- the full-token forward matches `model(...)` to 7e-7;
- the log(m) key bias equals duplicating the key m times, to 1e-7;
- sharing with nothing static reproduces plain sampling exactly.

**Design decision:** the log(m) bias is folded into an extra q/k channel instead of a float
attention mask. A mask can push SDPA onto a kernel that materialises the L x L matrix (about 9 GB
at 14k tokens on a T4). The fold is exact.

**An independent review agent** caught five problems before the run, all fixed:
1. The static map was built from x0_hat at sigma ~ 0.98, which is near-pure noise. It is now
   built after warm-up to sigma 0.85, and a sigma 0.95 variant is kept as an ablation.
2. The absolute thresholds were uncalibrated for Wan latents. Sharing now uses a quantile
   fraction of positions (0.25 / 0.5 / 0.75), so the sweep is guaranteed to span the range.
3. Gated or nothing-static runs still went through the sharing path. They now fall back to
   plain sampling exactly.
4. fp16 on T4 could overflow silently. The residual stream is now kept in fp32, with a
   finiteness assert.
5. Flicker was measured on each video's own still pixels, which favours freezing. It now uses
   the reference's 10% stillest pixels.

Also added: a UniPC fewer-steps baseline (Wan's own scheduler), saving latents before decoding,
and sub-token pan estimation.

**Protocol** (`wan/kaggle_run.py`): 24 prompts (16 static camera, 8 moving camera), 33 frames at
480x832, CFG 5, shift 3. Every method uses the same initial noise as the full-compute Euler-30
reference. Metrics: PSNR, SSIM and LPIPS to the reference, measured wall-clock, compute ratio,
reference-masked flicker, and motion ratio.

Methods:
- Euler 15 / 10 / 7 steps and UniPC 15 / 10 steps;
- sharing at 30 steps with frac 0.25 / 0.5 / 0.75, plus frac 0.5 with an early map;
- sharing at 15 steps (stacking);
- ablations: no re-noise, no log(m) bias, no pan gate.

Started as Kaggle Version #1 (headless Save & Run All), 11 Nov ~11:40 IST.

---
## 11 Nov: 16-frame clips (task 14)

**Model.** The 42k checkpoint extended to T=16: temporal embeddings interpolated linearly, then
fine-tuned 8k steps on fresh 16-frame data, giving 256 tokens. It is a weaker model than the
8-frame one: full-compute C2ST is 0.77 / 0.70 on the two seed sets, vs 0.66 for T=8.

**Full battery, 400 clips** (`results/r3f_battery_tiny_ckpt_t16*.json`). C2ST accuracy, where
0.5 means indistinguishable from real. Format is set 1 / set 2 (set 2: fresh noise and real
seeds).

| method | compute x | C2ST | area drift | valid |
|---|---|---|---|---|
| full, 20 steps | 1.0 | 0.769 / 0.696 | 0.97 / 0.96 | 0.23 / 0.26 |
| normal 10 steps | 2.0 | 0.877 / 0.817 | 1.13 / 1.16 | 0.12 / 0.14 |
| normal 8 steps | 2.5 | 0.906 / 0.863 | 1.25 / 1.25 | 0.08 / 0.10 |
| **share 20 steps, thr 0.02** | 2.44 | **0.777 / 0.817** | 0.77 / 0.74 | 0.35 / 0.37 |
| **share 10 steps, thr 0.015** | 2.6 | **0.758 / 0.812** | 0.87 / 0.85 | 0.29 / 0.30 |
| normal 6 steps | 3.3 | 0.935 / 0.896 | 1.37 / 1.38 | 0.03 / 0.04 |
| normal 5 steps | 4.0 | 0.958 / 0.915 | 1.50 / 1.54 | 0.01 / 0.02 |
| **share 10 steps, thr 0.02** | 3.9 | **0.913 / 0.887** | 0.80 / 0.80 | 0.26 / 0.28 |
| **share 20 steps, thr 0.03** | 4.2 | **0.829 / 0.871** | 1.05 / 1.04 | 0.16 / 0.16 |

**Observations**
1. With twice the frames, the same thresholds share more, so speedups rise: 4.2x at thr 0.03,
   vs 3.0-3.9x at T=8. This is the expected scaling, since a static position collapses T copies
   into one.
2. At matched compute, sharing beats fewer steps on C2ST in both seed sets at both speeds.
   - At about 2.5x, sharing is within noise of full compute on set 1 (0.76-0.78 vs 0.77). On set
     2 it is 0.81 against full 0.70, so a real but smaller loss.
   - At about 4x the margin over normal 5 steps is 0.04-0.13.
   - Area drift and validity favour sharing everywhere. Validity is biased toward frozen
     backgrounds, so it is supporting evidence only.
3. Ablations (set 1 only):
   - Averaging is clearly necessary: without it, C2ST is 0.977 / 0.965.
   - Re-noising and the log(m) bias are not consistently necessary at T=16. No re-noise: 0.846
     at 20 steps (worse than 0.777) but 0.727 at 10 steps (better than 0.913). No bias: 0.865 at
     20 steps (worse) but 0.794 at 10 steps (better).
   - The C2ST standard error is about 0.03 (240 held-out clips), so these swings are partly
     noise. **Claim only averaging as universally necessary**; report the others as helping at
     T=8 and mixed at T=16.

---
## 13 Nov: Wan 2.1 results (Kaggle Version 2). The toy result does NOT transfer as-is.

Full table: `results/wan/v2_results_summary.md` (16 of 24 prompts finished).

1. **Fewer steps is a strong baseline on Wan.** UniPC-20 at 1.5x gives LPIPS 0.225 and keeps motion
   at 1.05. UniPC-15 at 2x gives LPIPS 0.31, CLIP 30.6 (full: 30.0), flicker 1.0.
2. **Draft-map sharing (the toy protocol) fails badly.** PSNR ~12, LPIPS 0.66-0.78, flicker 8-10x
   the reference, and 2.6-4.3x the reference's frame-to-frame change. CLIP stays near full
   (28.6-29.0), so the content is right but the video churns over time.
3. **Late sharing (map at sigma 0.85, the last 20 steps shared)** is much milder but still no
   better than fewer steps at matched speed (1.4x: LPIPS ~0.36 ungated vs steps_20's 0.25), and
   it flickers more.
4. **Ablations.** No re-noise breaks content (CLIP 20). No log(m) bias makes the flicker worst
   (10.5) but is 28% faster in wall-clock, which measures the cost of the 8-channel bias fold.
5. **The pan gate is unreliable.** It never fired on drafts (0 of 5 moving prompts gated) and
   false-alarmed on 3 of 11 static prompts from the mid-trajectory estimate.
6. **Code check (CPU):** shared latent tokens end exactly identical across latent frames 1-8, as
   designed (max diff 0.0). So the extra pixel-space motion cannot come from the latents of the
   shared region changing; it must come from how Wan's VAE decodes them.

**Leading hypothesis:** Wan's causal 3D VAE does not represent a static scene with identical
latent frames. Real static content has a per-latent-frame structure (each latent frame packs 4
pixel frames with a causal temporal cache). Forcing exact equality produces latents off the data
manifold, which the decoder turns into temporal churn. In the toy model, pixels were the latents,
so "static = identical tokens" held exactly.

**Next (needs one cheap Kaggle run, VAE-only diagnostic plus a fix):**
- (a) Encode real static and near-static clips with Wan-VAE and measure latent frame-to-frame
  differences.
- (b) Decode a reference latent with its static positions forced equal across frames 1-8 and
  measure the flicker. That alone would confirm the hypothesis without the transformer.
- (c) Fix by sharing in the right space: keep each frame's latent offset from the draft and share
  only the leader's update, i.e. x0_i = x0_leader + (d_i - d_leader), where d is the draft's clean
  latent.
