# GPU plan: from toy results to real numbers

Everything in this repo so far ran on CPU with synthetic video and a tiny model. These are
the steps that need a real GPU and real models, in order. Each step has a go/no-go check,
so you only pay for the next step if the previous one worked.

## Hardware and setup

- One NVIDIA GPU with 24 GB+ (RTX 4090 / A5000 / L4 class) is enough for a ~1-2B
  parameter open text-to-video model. An 80 GB card (A100/H100) is needed for the large
  (~13B-class) models. Rough rental price: $0.50-$2 per hour (RunPod, Lambda, Vast).
- Python 3.11, PyTorch with CUDA, `diffusers`, `transformers`, `accelerate`, plus this
  repo's requirements. Pin versions in a `requirements.lock` on day one; video-model
  repos change often.
- Pick ONE open text-to-video model available in `diffusers` with a small variant, and
  if possible a step-distilled variant as well. Write the exact model id and version in
  `results/MODEL.md` so every number is traceable.

## Step 1: score real vs generated clips (A1/A5 on real data), about 1 GPU-day

1. Collect ~200 real clips (e.g. DAVIS, which has occlusions, plus your own phone videos)
   and generate ~200 clips with the model from matching prompts.
2. Run `python -m ecm.score --heatmap out/ <clips>` with `--max-side 256`.
   Use `prefilter=2.0` (all real video is compressed; see FINDINGS).
3. Have 2-3 people rate 200 generated clips 1-5 for melting (T6).
4. **Go/no-go:** PVS vs human ratings, Spearman >= 0.6. Real vs generated AUC >= 0.7.
5. Upgrade the flow from Farneback to RAFT (`torchvision.models.optical_flow.raft_large`)
   and re-run; it should only get better.

## Step 1b: does shared noise hurt quality? (the key open question), a few GPU-hours

Frame sharing (FINDINGS section 2) needs the static background to start from the same noise in
every frame. Generate the same prompts (static-camera scenes) twice, once with normal noise and
once with noise shared in static regions (static map from a 2-3 step draft pass). Compare them by
human preference, FVD and PVS. **Go/no-go:** no visible loss. If quality drops, try sharing only
from the middle of sampling, or sharing the low-frequency part of the noise only.

## Step 2: settling on the real model (B1), a few GPU-hours

Use the pipeline's per-step callback (`callback_on_step_end`) to save the latents at every
step. For flow-matching schedulers the predicted clean latent is
`x0_hat = latents - sigma_t * model_output`; save the model output too (wrap the
transformer's forward). Per latent token, record the first step after which `x0_hat`
stays within eps of its final value.

**Go/no-go:** in the distilled model, a real spread in settling time across tokens
(the tiny model shows what to look for; see FINDINGS).

## Step 3: oracle token sharing and pruning (B2/B3), 1-3 GPU-days

Needs a patched attention: wrap each transformer block so it can (a) take a subset of
query tokens and reuse cached K/V for the rest, and (b) drop tokens from the sequence
entirely and copy their outputs from a leader token. `ecm/tinydit.py` is the reference
implementation of both, with a FLOP counter; port the same logic. Measure wall-clock
with `torch.cuda.synchronize()` and peak memory with `torch.cuda.max_memory_allocated()`.

Port proportional attention too (log(copies) bias on each shared key); without it, sharing
fails on the toy (20 dB vs 43.6 dB).

**Go/no-go (the main gate):** draft-pass temporal sharing on static-camera prompts reaches
3x or more wall-clock at a quality loss you would accept (FVD and PVS, plus your eyes).

## Step 4: online methods (B4, C1)

Freezing by the heat rule, early abort by PVS of the early prediction. Only after Step 3.

## What not to do yet

Do not train a tokenizer (B7) or run reward fine-tuning (C3) until Steps 1-3 pass.
They cost 10-100x more GPU time than everything above.
