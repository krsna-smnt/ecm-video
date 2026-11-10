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
