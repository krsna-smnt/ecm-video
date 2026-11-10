"""Score real video files for permanence violations (melting / vanishing / boiling).

usage:
    python -m ecm.score clip1.mp4 [clip2.mp4 ...] [--max-side 256] [--frames 64] [--heatmap out_dir]

Prints one line per file: PVS peak (worst frame), PVS mean, worst frame index, and the
warping-error baseline. With --heatmap, writes <name>_pvs.mp4 showing the frames with the
violation map overlaid in red, so you can see *where* physics broke.

Higher PVS = more information created or destroyed in open view (away from occlusion
boundaries and frame edges). Compare clips at the same resolution and frame rate;
the absolute value depends on both.
"""
import argparse, os, sys
import numpy as np
import cv2
from .pvs import score_video, load_video


def overlay(frames, heat, path, fps=8):
    h, w = frames.shape[1:]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    scale = np.percentile(heat[heat > 0], 99) if (heat > 0).any() else 1.0
    for t in range(len(heat)):
        g = cv2.cvtColor(frames[t].astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)
        a = np.clip(heat[t] / scale, 0, 1)[..., None]
        red = np.zeros_like(g); red[..., 2] = 255
        vw.write((g * (1 - 0.7 * a) + red * 0.7 * a).astype(np.uint8))
    vw.release()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--max-side", type=int, default=256)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--stride", type=int, default=1, help="use every n-th frame (for high-fps footage)")
    ap.add_argument("--heatmap", default=None, help="directory for overlay videos")
    ap.add_argument("--prefilter", type=float, default=2.0,
                    help="blur sigma applied first to wash out compression blocks (A1b: 0.64 -> 0.76 AUC on JPEG clips)")
    ap.add_argument("--mode", default="conditional", choices=["conditional", "vector", "combined", "scalar"])
    a = ap.parse_args(argv)
    print(f"{'file':40s} {'pvs_peak':>9s} {'pvs_mean':>9s} {'worst_fr':>8s} {'warp_err':>9s}")
    for p in a.videos:
        try:
            f = load_video(p, a.max_side, a.frames, a.stride)
        except Exception as e:
            print(f"{os.path.basename(p):40s} ERROR {e}"); continue
        r = score_video(f, mode=a.mode, prefilter=a.prefilter)
        print(f"{os.path.basename(p)[:40]:40s} {r.pvs_peak:9.4f} {r.pvs_mean:9.4f} "
              f"{int(np.argmax(r.frame_scores)):8d} {r.warp_err_mean:9.3f}")
        if a.heatmap:
            os.makedirs(a.heatmap, exist_ok=True)
            overlay(f, r.heatmaps, os.path.join(a.heatmap, os.path.splitext(os.path.basename(p))[0] + "_pvs.mp4"))


if __name__ == "__main__":
    sys.exit(main())
