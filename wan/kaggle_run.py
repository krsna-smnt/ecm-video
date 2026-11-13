"""Decisive real-model experiment: re-noised averaged frame sharing on Wan 2.1 T2V 1.3B.

Designed for a free Kaggle notebook (2x T4 16GB). Steps:
  python kaggle_run.py encode                 # T5 prompt embeddings on CPU (bf16) -> embeds.pt
  python kaggle_run.py gen --gpu 0 --shard 0/2 &  python kaggle_run.py gen --gpu 1 --shard 1/2
  python kaggle_run.py report                 # metrics vs the full-compute reference -> results.md

Every method for a prompt uses the same initial noise, so outputs are directly comparable with the
full-compute reference (the standard protocol of training-free acceleration papers: PSNR/SSIM/LPIPS
to the original model's output, plus wall-clock). Resumable: finished videos are skipped.
Use --dry to run the whole pipeline on CPU with a tiny random-weight transformer (no downloads).
"""
import argparse, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ecm_wan as E

MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
NEG = ("Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall "
       "gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
       "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three "
       "legs, many people in the background, walking backwards")

# 16 static/locked-off camera scenes (where sharing should help) + 8 moving-camera scenes (gate test)
PROMPTS_STATIC = [
    "A cat sitting on a windowsill, its tail slowly swishing, soft afternoon light, locked-off camera",
    "A candle flame flickering on a wooden table in a dark room, static shot",
    "A barista pouring milk into a cup of coffee, close-up, tripod shot",
    "A goldfish swimming in a round glass bowl on a kitchen counter, static camera",
    "A man reading a newspaper on a park bench, pigeons walking nearby, fixed camera",
    "Rain drops falling into a puddle on a street at night, neon reflections, static shot",
    "A woman typing on a laptop in a quiet library, medium shot, static camera",
    "A red ball bouncing on a tiled floor in an empty room, static shot",
    "A dog wagging its tail on a front porch, fixed camera",
    "Steam rising from a bowl of ramen on a restaurant table, static close-up",
    "A clock with a moving second hand on a white wall, static shot",
    "Leaves of a potted plant moving gently in the breeze near an open window, static camera",
    "A chef chopping vegetables on a cutting board, overhead static shot",
    "A child blowing soap bubbles in a backyard, fixed camera",
    "Waves gently lapping at a rocky shore, static wide shot",
    "A person walking across a crosswalk in front of a stationary camera, city street",
]
PROMPTS_MOVING = [
    "Camera slowly pans right across a mountain landscape at sunrise",
    "Drone shot flying forward over a forest river",
    "Tracking shot following a cyclist on a coastal road",
    "Camera dolly moving through a busy market street",
    "Slow pan across a bookshelf full of old books",
    "Handheld camera walking through a subway station",
    "Aerial shot circling a lighthouse on a cliff",
    "Camera tilting up a tall glass skyscraper",
]
# interleave 2 static : 1 moving so a run cut short by the time budget still covers both kinds
PROMPTS = []
for j in range(8):
    PROMPTS += [("static", PROMPTS_STATIC[2 * j]), ("static", PROMPTS_STATIC[2 * j + 1]), ("moving", PROMPTS_MOVING[j])]


def methods_v3(steps):
    """Run 3: offset-preserving sharing (fix for the VAE latent-equality churn seen in run 2)."""
    d = dict(first_frame=1, pan_gate=None, draft=3)
    return {f"full_{steps}": (steps, None),
            f"steps_{steps * 2 // 3}": (steps * 2 // 3, None),
            f"unipc_{steps * 2 // 3}": (steps * 2 // 3, "unipc"),
            f"steps_{steps // 2}": (steps // 2, None),
            f"unipc_{steps // 2}": (steps // 2, "unipc"),
            f"share_{steps}_f0.75_draft3": (steps, dict(d, frac=0.75)),                      # run-2 control
            f"offset_{steps}_f0.75_draft3": (steps, dict(d, frac=0.75, offset=True)),
            f"offset_{steps}_f0.9_draft3": (steps, dict(d, frac=0.9, offset=True)),
            f"offset_{steps // 2}_f0.75_draft3": (steps // 2, dict(d, frac=0.75, offset=True)),
            f"offset_{steps}_f0.75_warm": (steps, dict(first_frame=1, pan_gate=None, warm_sigma=0.85, frac=0.75, offset=True))}


def methods(steps):
    """name -> (n_steps, share-config | None | "unipc").
    Sharing picks the `frac` least-active token positions (quantile, so the sweep spans the range).
    The map comes either from a cheap separate draft from the same noise (draft=3: the toy-study
    protocol, sharing from step 0) or from the model's own estimate after the steps with sigma > 0.85
    (warm: no extra pass, but sharing starts late, which caps the speedup)."""
    M = {f"full_{steps}": (steps, None)}
    for n in [steps * 2 // 3, steps // 2, steps // 3]:          # 20 / 15 / 10 at steps=30
        M[f"steps_{n}"] = (n, None)
    for n in [steps * 2 // 3, steps // 2]:
        M[f"unipc_{n}"] = (n, "unipc")
    d = dict(first_frame=1, pan_gate=0.25, draft=3)
    for frac in [0.5, 0.75, 0.9]:
        M[f"share_{steps}_f{frac}_draft3"] = (steps, dict(d, frac=frac))
    M[f"share_{steps}_f0.75_warm"] = (steps, dict(first_frame=1, pan_gate=0.25, warm_sigma=0.85, frac=0.75))
    M[f"share_{steps // 2}_f0.75_draft3"] = (steps // 2, dict(d, frac=0.75))         # stacking on fewer steps
    M[f"abl_noRenoise_{steps}_f0.75_draft3"] = (steps, dict(d, frac=0.75, renoise=False))
    M[f"abl_noBias_{steps}_f0.75_draft3"] = (steps, dict(d, frac=0.75, prop=False))
    M[f"abl_noGate_{steps}_f0.75_draft3"] = (steps, dict(d, frac=0.75, pan_gate=None))   # also a guard if the gate misfires
    return M


# ----------------------------------------------------------------------------- model loading

def load_transformer(args, device, dtype):
    if args.dry:
        from diffusers import WanTransformer3DModel
        torch.manual_seed(0)
        m = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=24, in_channels=16,
                                  out_channels=16, text_dim=64, freq_dim=32, ffn_dim=96, num_layers=2,
                                  cross_attn_norm=True, rope_max_seq_len=1024)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.02)
        return m.to(device, dtype).eval()
    from diffusers import WanTransformer3DModel
    return WanTransformer3DModel.from_pretrained(MODEL, subfolder="transformer", torch_dtype=dtype).to(device).eval()


def load_vae(args, device):
    if args.dry:
        return None
    from diffusers import AutoencoderKLWan
    return AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32).to(device).eval()


@torch.no_grad()
def decode(vae, lat, args):
    """latents -> uint8 video (F,H,W,3)."""
    if vae is None:   # dry run: fake 'decode' by nearest upsampling of 3 latent channels
        v = lat[0, :3].float().clamp(-2, 2) / 4 + 0.5
        v = v.repeat_interleave(4, 1)[:, :args.frames].repeat_interleave(8, 2).repeat_interleave(8, 3)
        return (v.permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(lat)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(lat)
    try:
        v = vae.decode(lat.float() * std + mean, return_dict=False)[0]      # B,3,F,H,W in [-1,1]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); vae.enable_tiling()
        v = vae.decode(lat.float() * std + mean, return_dict=False)[0]
    v = ((v[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
    return v


# ----------------------------------------------------------------------------- stages

def stage_encode(args):
    out = os.path.join(args.out, "embeds.pt")
    if os.path.exists(out):
        print("embeds exist"); return
    texts = [p for _, p in PROMPTS] + [NEG]
    if args.dry:
        g = torch.Generator().manual_seed(1)
        embs = [torch.randn(1, 12, 64, generator=g) for _ in texts]
    else:
        from transformers import AutoTokenizer, UMT5EncoderModel
        tok = AutoTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
        dev = "cuda" if torch.cuda.is_available() and args.encode_on_gpu else "cpu"
        enc = UMT5EncoderModel.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(dev).eval()
        embs = []
        for i, t in enumerate(texts):
            ids = tok([t], max_length=512, truncation=True, add_special_tokens=True, return_tensors="pt")
            n = int(ids.attention_mask.sum())      # encode only the real tokens: identical to masked padding, 10x cheaper
            with torch.no_grad():
                e = enc(ids.input_ids[:, :n].to(dev), ids.attention_mask[:, :n].to(dev)).last_hidden_state[0].float().cpu()
            # pipeline convention: zero-pad to 512 tokens
            embs.append(torch.cat([e, e.new_zeros(512 - n, e.shape[1])])[None])
            print(f"encoded {i + 1}/{len(texts)}", flush=True)
    torch.save(embs, out)


def stage_gen(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    model = load_transformer(args, device, dtype)
    vae = load_vae(args, device)
    embs = torch.load(os.path.join(args.out, "embeds.pt"))
    neg = embs[-1].to(device, dtype)
    k, n = map(int, args.shard.split("/"))
    M = methods_v3(args.steps) if args.set == "v3" else methods(args.steps)
    if args.only:
        M = {a: b for a, b in M.items() if a in args.only.split(",")}
    lat_shape = (1, 16, (args.frames - 1) // 4 + 1, args.height // 8, args.width // 8)
    os.makedirs(os.path.join(args.out, "vids"), exist_ok=True)
    meta_f = os.path.join(args.out, f"meta_{k}.json")
    meta = json.load(open(meta_f)) if os.path.exists(meta_f) else {}
    idx = [i for i in range(len(PROMPTS)) if i % n == k][: args.max_prompts]
    # warm-up (CUDA kernels / allocator) so the first timed method is not penalised
    E.sample(model, torch.randn(lat_shape).to(device), embs[0].to(device, dtype), neg, n_steps=1, shift=args.shift, gscale=args.cfg)
    t_start = time.time()
    for i in idx:
        if (time.time() - t_start) / 3600 > args.hours:
            print(f"[gpu{args.gpu}] time budget reached, stopping before prompt {i}", flush=True)
            break
        cat, prompt = PROMPTS[i]
        enc = embs[i].to(device, dtype)
        noise = torch.randn(lat_shape, generator=torch.Generator().manual_seed(1000 + i)).to(device)
        for name, (steps, share) in M.items():
            key = f"{i:02d}_{name}"
            fn = os.path.join(args.out, "vids", key + ".npy")
            if key in meta and os.path.exists(fn):
                continue
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            t0 = time.time()
            if share == "unipc":
                lat, info = E.sample_unipc(model, noise, enc, neg, n_steps=steps, shift=args.shift, gscale=args.cfg)
            else:
                lat, info = E.sample(model, noise, enc, neg, n_steps=steps, shift=args.shift, gscale=args.cfg,
                                     share=share, seed=50_000 + i)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            dt = time.time() - t0
            torch.save(lat.half().cpu(), fn[:-4] + ".lat.pt")      # keep the sampling work even if decode fails
            vid = decode(vae, lat, args)
            np.save(fn, vid)
            if "act" in info:   # activity distribution, for threshold calibration
                info["act_q"] = [float(q) for q in torch.quantile(info["act"].float().cpu(), torch.tensor([.1, .25, .5, .75, .9]))]
            info = {a: b for a, b in info.items() if a != "act"}
            meta[key] = dict(prompt_idx=i, category=cat, method=name, seconds=dt, **info)
            json.dump(meta, open(meta_f, "w"), indent=1)
            print(f"[gpu{args.gpu}] {key:32s} {dt:7.1f}s  compute x{info['full_token_forwards'] / info['token_forwards'] * args.steps / steps:.2f}"
                  f"  static {info.get('static_frac', 0):.2f} pan {info.get('pan', 0):.2f}", flush=True)


# ----------------------------------------------------------------------------- metrics

def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim_video(a, b):
    from skimage.metrics import structural_similarity
    return float(np.mean([structural_similarity(x, y, channel_axis=2, data_range=255) for x, y in zip(a, b)]))


def flicker(v, ref):
    """Mean |frame diff| over the 10% of pixels that change least IN THE REFERENCE (fixed mask, so a
    method cannot lower its score by freezing a different set of pixels)."""
    dr = np.abs(np.diff(ref.astype(np.float32), axis=0)).mean(-1).mean(0)
    mask = dr <= np.quantile(dr, 0.1)
    d = np.abs(np.diff(v.astype(np.float32), axis=0)).mean(-1)
    return float(d[:, mask].mean())


def motion(v):
    return float(np.abs(np.diff(v.astype(np.float32), axis=0)).mean())


def stage_report(args):
    meta = {}
    for f in os.listdir(args.out):
        if f.startswith("meta_"):
            meta.update(json.load(open(os.path.join(args.out, f))))
    lp = None
    try:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False)
    except Exception as e:  # noqa
        print("LPIPS unavailable:", e)
    clip = None
    try:
        from transformers import CLIPModel, CLIPProcessor
        clip = (CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval(),
                CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"))
    except Exception as e:  # noqa
        print("CLIP unavailable:", e)

    def clip_score(v, prompt):
        with torch.no_grad():
            fr = [v[j] for j in np.linspace(0, len(v) - 1, 8).astype(int)]
            inp = clip[1](text=[prompt], images=fr, return_tensors="pt", padding=True)
            out = clip[0](**inp)
            im = torch.nn.functional.normalize(out.image_embeds, dim=-1)
            tx = torch.nn.functional.normalize(out.text_embeds, dim=-1)
            return float((im @ tx.T).mean() * 100)
    ref_name = f"full_{args.steps}"
    rows = {}
    for key, m in sorted(meta.items()):
        i, name = m["prompt_idx"], m["method"]
        ref_key = f"{i:02d}_{ref_name}"
        if ref_key not in meta:
            continue
        v = np.load(os.path.join(args.out, "vids", key + ".npy"))
        r = np.load(os.path.join(args.out, "vids", ref_key + ".npy"))
        rec = dict(category=m["category"], seconds=m["seconds"], ref_seconds=meta[ref_key]["seconds"],
                   work=m["token_forwards"], ref_work=meta[ref_key]["token_forwards"],
                   psnr=psnr(v, r), ssim=ssim_video(v, r), flicker=flicker(v, r), ref_flicker=flicker(r, r), motion=motion(v),
                   ref_motion=motion(r), static_frac=m.get("static_frac", 0.0), gated=m.get("pan", 0) > 0.25)
        if clip is not None:
            rec["clip"] = clip_score(v, PROMPTS[i][1])
        if lp is not None:
            with torch.no_grad():
                tv = torch.from_numpy(v).permute(0, 3, 1, 2).float() / 127.5 - 1
                tr = torch.from_numpy(r).permute(0, 3, 1, 2).float() / 127.5 - 1
                rec["lpips"] = float(np.mean([lp(tv[j:j + 8], tr[j:j + 8]).mean().item() for j in range(0, len(tv), 8)]))
        rows.setdefault(name, []).append(rec)
    json.dump(rows, open(os.path.join(args.out, "per_video.json"), "w"), indent=1)

    def agg(recs, cat):
        rs = [r for r in recs if cat == "all" or r["category"] == cat]
        if not rs:
            return None
        g = lambda k: float(np.mean([r[k] for r in rs])) if k in rs[0] else float("nan")
        return dict(n=len(rs), wall=float(np.sum([r["ref_seconds"] for r in rs]) / np.sum([r["seconds"] for r in rs])),
                    flops=float(np.sum([r["ref_work"] for r in rs]) / np.sum([r["work"] for r in rs])),
                    psnr=g("psnr"), ssim=g("ssim"), lpips=g("lpips"), flicker=g("flicker"), clip=g("clip"),
                    motion_ratio=float(np.mean([r["motion"] / max(r["ref_motion"], 1e-6) for r in rs])),
                    static=g("static_frac"), gated=g("gated"))
    lines = ["# Wan 2.1 1.3B: re-noised averaged frame sharing vs fewer steps", "",
             f"{args.frames} frames, {args.height}x{args.width}, Euler flow sampler, shift {args.shift}, CFG {args.cfg}. "
             f"Reference = {ref_name} with identical noise. Speedups: wall-clock (measured, sampling only) and "
             "token-forward ratio (compute; conservative: attention cost scales with tokens^2). Baselines steps_* use Euler, unipc_* use Wan's own UniPC scheduler. "
             "flicker uses the reference's 10% least-changing pixels. 'gated' = pan gate fired (pan > 0.25 tokens/frame). motion = mean |frame diff| relative to the reference (1.0 = same amount of motion).", ""]
    summary = {}
    for cat in ["all", "static", "moving"]:
        lines += [f"## {cat} prompts", "",
                  "| method | n | wall-clock x | compute x | PSNR | SSIM | LPIPS | CLIP | flicker | motion | static frac | gated |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, recs in rows.items():
            a = agg(recs, cat)
            if a is None:
                continue
            summary.setdefault(cat, {})[name] = a
            lines.append(f"| {name} | {a['n']} | {a['wall']:.2f} | {a['flops']:.2f} | {a['psnr']:.2f} | {a['ssim']:.3f} | "
                         f"{a['lpips']:.3f} | {a['clip']:.2f} | {a['flicker']:.2f} | {a['motion_ratio']:.2f} | {a['static']:.2f} | {a['gated']:.2f} |")
        lines.append("")
    open(os.path.join(args.out, "results.md"), "w").write("\n".join(lines))
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print("\n".join(lines))


def stage_diag(args):
    """VAE-only test of the run-2 hypothesis: does forcing static latent positions to be identical
    across latent frames (what plain sharing does) by itself make Wan's decoder churn?"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vae = load_vae(args, device)
    lp = None
    try:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False)
    except Exception as e:  # noqa
        print("LPIPS unavailable:", e)
    vd = os.path.join(args.out, "vids")
    lines = ["# VAE diagnostic: forcing latent equality on static positions (no transformer involved)", "",
             "Each row decodes the full_30 reference latent after modifying only the 75% least-active "
             "positions in latent frames 2..8. Metrics vs the unmodified reference decode.", "",
             "| prompt | variant | PSNR | LPIPS | flicker (ref mask) | motion | latent rel. change, static pos. |",
             "|---|---|---|---|---|---|---|"]
    rows = []
    for f in sorted(os.listdir(vd)):
        if not f.endswith(f"_full_{args.steps}.lat.pt"):
            continue
        lat = torch.load(os.path.join(vd, f)).float().to(device)            # 1,16,F,H,W
        ref = np.load(os.path.join(vd, f.replace(".lat.pt", ".npy")))
        p = (1, 2, 2)
        Fl, Hl, Wl = lat.shape[2], lat.shape[3] // 2, lat.shape[4] // 2
        tok = E.patchify(lat, p)
        src, info = E.static_map(tok, (Fl, Hl, Wl), 0.0, 1, None, 0.75)
        T = tok[0].reshape(Fl, Hl * Wl, -1)
        stat = (src.reshape(Fl, Hl * Wl)[2] != torch.arange(Hl * Wl) + 2 * Hl * Wl).to(device)
        rel = float(((T[2:, stat] - T[1:-1, stat]).pow(2).mean().sqrt() / T[1:, stat].pow(2).mean().sqrt()))
        variants = {"copy_frame1": T.clone(), "temporal_mean": T.clone(), "untouched": T.clone()}
        variants["copy_frame1"][2:, stat] = T[1:2, stat]
        variants["temporal_mean"][1:, stat] = T[1:, stat].mean(0, keepdim=True)
        for name, Tv in variants.items():
            v = decode(vae, E.unpatchify(Tv.reshape(1, Fl * Hl * Wl, -1), p, lat.shape), args)
            rec = dict(prompt=f[:2], variant=name, psnr=psnr(v, ref), flicker=flicker(v, ref), ref_flicker=flicker(ref, ref),
                       motion=motion(v) / max(motion(ref), 1e-6), latent_rel_change=rel)
            if lp is not None:
                with torch.no_grad():
                    tv = torch.from_numpy(v).permute(0, 3, 1, 2).float() / 127.5 - 1
                    tr = torch.from_numpy(ref).permute(0, 3, 1, 2).float() / 127.5 - 1
                    rec["lpips"] = float(np.mean([lp(tv[j:j + 8], tr[j:j + 8]).mean().item() for j in range(0, len(tv), 8)]))
            rows.append(rec)
            lines.append(f"| {rec['prompt']} | {name} | {rec['psnr']:.2f} | {rec.get('lpips', float('nan')):.3f} | "
                         f"{rec['flicker']:.2f} (ref {rec['ref_flicker']:.2f}) | {rec['motion']:.2f} | {rel:.3f} |")
            print(lines[-1], flush=True)
    for name in ["copy_frame1", "temporal_mean", "untouched"]:
        rs = [r for r in rows if r["variant"] == name]
        if rs:
            lines.append(f"| **mean** | {name} | {np.mean([r['psnr'] for r in rs]):.2f} | "
                         f"{np.mean([r.get('lpips', np.nan) for r in rs]):.3f} | {np.mean([r['flicker'] for r in rs]):.2f} | "
                         f"{np.mean([r['motion'] for r in rs]):.2f} | {np.mean([r['latent_rel_change'] for r in rs]):.3f} |")
    open(os.path.join(args.out, "diag.md"), "w").write("\n".join(lines))
    json.dump(rows, open(os.path.join(args.out, "diag.json"), "w"), indent=1)
    print("\n".join(lines[-3:]))


def stage_mp4(args):
    import imageio
    os.makedirs(os.path.join(args.out, "mp4"), exist_ok=True)
    for f in sorted(os.listdir(os.path.join(args.out, "vids"))):
        v = np.load(os.path.join(args.out, "vids", f))
        imageio.mimwrite(os.path.join(args.out, "mp4", f[:-4] + ".mp4"), list(v), fps=16, quality=8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["encode", "gen", "report", "mp4", "diag"])
    ap.add_argument("--set", default="v2", choices=["v2", "v3"])
    ap.add_argument("--out", default="/kaggle/working/ecm_wan")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--max_prompts", type=int, default=999)
    ap.add_argument("--only", default="")
    ap.add_argument("--hours", type=float, default=8.0, help="stop starting new prompts after this many hours")
    ap.add_argument("--encode_on_gpu", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"encode": stage_encode, "gen": stage_gen, "report": stage_report, "mp4": stage_mp4, "diag": stage_diag}[a.stage](a)
