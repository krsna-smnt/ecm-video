# Overnight research summary (10-11 Nov 2026)

Full details, every table and every mistake are in `RESEARCH_LOG.md`.

## The one result worth pursuing

**Re-noised averaged frame sharing.** Static regions of a video are computed once per
denoising step instead of once per frame, without the quality loss that the earlier
frame-sharing method had.

- On the best toy model (42k training steps), at every matched speedup it beats simply taking
  fewer sampling steps on FD, C2ST (a real-vs-generated classifier), object validity and object
  area drift. This holds on two independent seed sets.
- At 2.4x fewer FLOPs, its quality is within metric noise of full compute.
- Measured speedups are 2.4-3.9x on the toy model. At the scale of a real video transformer
  the same token fractions project to 4.3-6x (a FLOP calculation), and a CPU wall-clock test
  showed 6.3x at 4k tokens.
- Each ingredient is necessary (ablation): averaging, re-noising and proportional attention
  each protect a different aspect of quality.

**Mechanism found along the way.** The original sharing method needed *shared noise*, which
turns out to hurt quality badly (FD 27x worse, saturated blocks). Models denoise static
backgrounds by pooling independent noisy views across frames, and shared noise destroys that
redundancy. The fix gives every frame its own noise, computes a shared region from the *mean*
of its copies, re-noises that mean back to the correct noise level, and shares only the clean
prediction. In ECM terms: **share the conserved content, not the noise.**

**Limitation found and partly fixed.** With a moving camera, ungated sharing freezes pans
(camera motion halved). A camera-motion gate on the draft preserves pans, but it leaves only
about 2x; even then it beats plain fewer steps at the same speed. The proper fix,
"Lagrangian" sharing along the camera motion, is the natural next experiment.

## What failed (equally important)

- **PVS (the melting score) does not match human glitch judgments on real AI video.** On 93
  human-voted pairs it was strongly *anti*-correlated (14% agreement). I first reported 86%
  because of a sign error (the vote question asks which clip has MORE glitches), caught it
  while checking a suspicious detail, and corrected it.
- **Nothing per-clip could be validated within a generator.** Generator identity alone
  explains 97% of those human votes. Every information feature tried (density, decay, signed
  source, sharpness, arrow of time) correlates |rho| <= 0.16 within a generator.
- **Toy metrics disagree with each other.** FD-rand is nearly one-dimensional and can prefer
  clearly worse variants; the physics checker favours frozen backgrounds. Conclusions above
  rest only on agreement across metrics.

## Bugs caught overnight

1. The vote sign (above).
2. The re-noise seed was constant across clips (found by an independent code-review agent);
   fixed and all affected numbers rerun.
3. Two jobs hit time limits; both resumed from checkpoints with no data loss.

## Is it publishable?

Not yet. Everything is a 1.1M-parameter model on a synthetic 32x32 toy world. But the
frame-sharing result is now a clean, falsifiable claim with a mechanism and ablations. The test
that would make it a paper: port re-noised averaged sharing to an open video model (e.g.
Wan 2.1 1.3B) on a rented GPU, and compare against fewer steps at matched wall-clock time with
human preference or VBench. It costs roughly 1-2 GPU-days. `GPU_PLAN.md` step 3 describes the
port; the method is in `experiments/r3f_battery.py` (`avg_sample`, renoise=True).

## Files

- `RESEARCH_LOG.md`: every experiment, observation and decision
- `results/fig_r3_renoise_vs_steps.png`: the main figure
- `experiments/r1*`: human-vote experiments; `experiments/r3*`: quality and sharing;
  `experiments/r4_wallclock.py`; `experiments/r5*`: camera motion; `experiments/toy_physics.py`
